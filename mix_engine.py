"""mix_engine.py

Autonomous mix and mastering engine for vocal stems and beat. The script loads all WAV
stems from an input directory, classifies each track role based on its filename, applies
role-aware processing chains (EQ, compression, creative FX), mixes the result through
logical buses, applies a mastering chain, and renders final masters plus analysis
reports. Designed for offline console usage with configurable defaults.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import soundfile as sf
from scipy.signal import butter, lfilter

try:
    import librosa
except Exception as exc:  # pragma: no cover - librosa should be installed
    raise RuntimeError("librosa is required for this script") from exc

try:  # Optional loudness metering
    import pyloudnorm as pyln
except ImportError:  # pragma: no cover - optional dependency
    pyln = None

try:  # Optional YAML config support
    import yaml
except ImportError:  # pragma: no cover - optional dependency
    yaml = None


DEFAULT_SR = 48_000
MASTER_SR = 44_100


@dataclass
class Track:
    """Container describing a single audio stem."""

    name: str
    role: str
    data: np.ndarray
    sr: int = DEFAULT_SR
    channels: int = 1
    report: Dict[str, float] = field(default_factory=dict)


ROLE_KEYWORDS = {
    "beat": ["beat", "бит", "untitled", "минус", "instrumental"],
    "lead_vocal": ["lead", "лид", "vocal", "вокал"],
    "double_vocal": ["double", "дабл", "dbl"],
    "pad_vocal": [
        "pad",
        "уу-уу",
        "ууу",
        "оо-оо",
        "гаммы",
        "back",
        "хор",
        "oo",
    ],
    "adlib_vocal": ["adlib", "adlibs", "fx", "шу_шу", "эй", "ха", "шуу", "эйй"],
    "support_vocal": ["support", "поддерж", "фраз", "backing", "важно", "фраза"],
}


def infer_role_from_filename(name: str) -> str:
    """Infer track role from filename keywords and heuristics."""

    lowered = name.lower()

    def contains_any(words: List[str]) -> bool:
        return any(keyword in lowered for keyword in words)

    if contains_any(ROLE_KEYWORDS["beat"]):
        return "beat"
    if contains_any(ROLE_KEYWORDS["double_vocal"]):
        return "double_vocal"
    if contains_any(ROLE_KEYWORDS["pad_vocal"]):
        return "pad_vocal"
    if contains_any(ROLE_KEYWORDS["adlib_vocal"]):
        return "adlib_vocal"
    if contains_any(ROLE_KEYWORDS["support_vocal"]):
        return "support_vocal"
    if contains_any(ROLE_KEYWORDS["lead_vocal"]):
        return "lead_vocal"

    numeric_map = {
        "1": "lead_vocal",
        "2": "double_vocal",
        "3": "pad_vocal",
        "4": "adlib_vocal",
        "5": "support_vocal",
    }
    for digit, role in numeric_map.items():
        if lowered.strip().startswith(digit):
            return role
    return "unknown"


def ensure_stereo(arr: np.ndarray) -> np.ndarray:
    if arr.ndim == 1:
        arr = arr[:, None]
    if arr.shape[1] == 1:
        return np.repeat(arr, 2, axis=1)
    return arr[:, :2]


def butter_highpass(cutoff: float, sr: int, order: int = 2) -> Tuple[np.ndarray, np.ndarray]:
    nyq = 0.5 * sr
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype="high", analog=False)
    return b, a


def apply_highpass(data: np.ndarray, cutoff: float, sr: int, order: int = 2) -> np.ndarray:
    b, a = butter_highpass(cutoff, sr, order)
    return lfilter(b, a, data, axis=0)


def db_to_amp(db: float) -> float:
    return 10 ** (db / 20.0)


def apply_peaking_eq(data: np.ndarray, bands: List[Dict[str, float]], sr: int) -> np.ndarray:
    # Simple biquad implementation per band
    out = data.copy()
    for band in bands:
        f0 = band.get("f0", 1000.0)
        q = band.get("q", 1.0)
        gain_db = band.get("gain_db", 0.0)
        if math.isclose(gain_db, 0.0, abs_tol=1e-3):
            continue
        a0 = 1 + (np.sin(2 * np.pi * f0 / sr) / (2 * q))
        A = db_to_amp(gain_db)
        omega = 2 * np.pi * f0 / sr
        alpha = np.sin(omega) / (2 * q)
        b0 = 1 + alpha * A
        b1 = -2 * np.cos(omega)
        b2 = 1 - alpha * A
        a0 = 1 + alpha / A
        a1 = -2 * np.cos(omega)
        a2 = 1 - alpha / A
        out = lfilter([b0 / a0, b1 / a0, b2 / a0], [1, a1 / a0, a2 / a0], out, axis=0)
    return out


def apply_compressor(
    data: np.ndarray,
    threshold_db: float,
    ratio: float,
    attack_ms: float,
    release_ms: float,
    sr: int,
) -> np.ndarray:
    attack_coeff = math.exp(-1.0 / (sr * (attack_ms / 1000.0)))
    release_coeff = math.exp(-1.0 / (sr * (release_ms / 1000.0)))
    envelope = np.zeros(data.shape[0])
    gain = np.ones_like(envelope)
    for i in range(data.shape[0]):
        sample = np.max(np.abs(data[i]))
        if sample > envelope[i - 1] if i > 0 else 0:
            envelope[i] = attack_coeff * (envelope[i - 1] if i > 0 else 0) + (1 - attack_coeff) * sample
        else:
            envelope[i] = release_coeff * (envelope[i - 1] if i > 0 else 0) + (1 - release_coeff) * sample
        envelope_db = 20 * np.log10(envelope[i] + 1e-8)
        if envelope_db > threshold_db:
            gain_db = threshold_db + (envelope_db - threshold_db) / ratio - envelope_db
            gain[i] = db_to_amp(gain_db)
        else:
            gain[i] = 1.0
    return data * gain[:, None]


def normalize_to_rms(data: np.ndarray, target_rms_db: float) -> np.ndarray:
    rms = np.sqrt(np.mean(np.square(data))) + 1e-8
    current_db = 20 * np.log10(rms)
    gain = db_to_amp(target_rms_db - current_db)
    out = data * gain
    peak = np.max(np.abs(out))
    if peak > db_to_amp(-1.0):
        out = out / (peak / db_to_amp(-1.0))
    return out


def apply_pan(data: np.ndarray, pan: float) -> np.ndarray:
    stereo = ensure_stereo(data)
    left_gain = math.cos((pan + 1) * math.pi / 4)
    right_gain = math.sin((pan + 1) * math.pi / 4)
    stereo[:, 0] *= left_gain
    stereo[:, 1] *= right_gain
    return stereo


def multi_tap_delay(data: np.ndarray, sr: int, taps: List[Tuple[float, float]]) -> np.ndarray:
    length = data.shape[0]
    out = np.zeros_like(data)
    for delay_ms, gain_db in taps:
        delay_samples = int(sr * delay_ms / 1000.0)
        if delay_samples >= length:
            continue
        gain = db_to_amp(gain_db)
        delayed = np.pad(data[:-delay_samples], ((delay_samples, 0), (0, 0)), mode="constant")
        out += delayed * gain
    return out


def gentle_lpf(data: np.ndarray, sr: int, cutoff: float = 8000.0) -> np.ndarray:
    nyq = 0.5 * sr
    normal_cutoff = cutoff / nyq
    b, a = butter(2, normal_cutoff, btype="low", analog=False)
    return lfilter(b, a, data, axis=0)


def slapback_delay(track: np.ndarray, sr: int, base_delay_ms: float = 150.0) -> np.ndarray:
    taps = [(base_delay_ms, -9.0), (base_delay_ms * 1.5, -12.0)]
    return gentle_lpf(multi_tap_delay(track, sr, taps), sr)


def pad_gain_envelope(length: int, sr: int, lift_db: float = 5.0, tail_seconds: float = 3.5) -> np.ndarray:
    envelope = np.ones(length)
    tail_samples = int(sr * tail_seconds)
    if tail_samples >= length:
        return envelope
    ramp = np.linspace(0, 1, tail_samples)
    envelope[-tail_samples:] *= db_to_amp(ramp * lift_db)
    return envelope


def pitch_shift_track(data: np.ndarray, sr: int, semitones: float) -> np.ndarray:
    mono = np.mean(data, axis=1)
    shifted = librosa.effects.pitch_shift(mono, sr=sr, n_steps=semitones)
    return ensure_stereo(shifted[:, None])


def autopan(data: np.ndarray, sr: int, period_s: float = 8.0, depth: float = 0.4) -> np.ndarray:
    stereo = ensure_stereo(data)
    t = np.linspace(0, data.shape[0] / sr, data.shape[0])
    lfo = np.sin(2 * np.pi * t / period_s) * depth
    stereo[:, 0] *= 1 - lfo
    stereo[:, 1] *= 1 + lfo
    return stereo


def saturate(data: np.ndarray, drive: float = 1.2) -> np.ndarray:
    return np.tanh(data * drive)


def soft_limiter(data: np.ndarray, threshold_db: float = -1.0) -> np.ndarray:
    threshold = db_to_amp(threshold_db)
    over = np.abs(data) - threshold
    over[over < 0] = 0
    clipped = threshold + np.tanh(over) * (1 - threshold)
    return np.sign(data) * clipped


def compute_loudness(data: np.ndarray, sr: int) -> Dict[str, float]:
    rms = np.sqrt(np.mean(np.square(data)))
    rms_db = 20 * np.log10(rms + 1e-8)
    peak = np.max(np.abs(data))
    peak_db = 20 * np.log10(peak + 1e-8)
    result = {"rms_db": float(rms_db), "peak_db": float(peak_db)}
    if pyln is not None:
        meter = pyln.Meter(sr)
        lufs = meter.integrated_loudness(data)
        result["lufs"] = float(lufs)
    return result


ROLE_SETTINGS = {
    "lead_vocal": {
        "hpf": 90.0,
        "eq": [
            {"f0": 250.0, "q": 1.0, "gain_db": -2.0},
            {"f0": 3800.0, "q": 1.2, "gain_db": 3.0},
            {"f0": 9500.0, "q": 1.5, "gain_db": 2.0},
        ],
        "compressor": {"threshold_db": -20.0, "ratio": 3.0, "attack_ms": 5.0, "release_ms": 60.0},
        "target_rms": -18.0,
        "pan": 0.0,
        "delay_ms": 160.0,
    },
    "double_vocal": {
        "hpf": 105.0,
        "eq": [
            {"f0": 300.0, "q": 1.0, "gain_db": -1.0},
            {"f0": 3200.0, "q": 1.2, "gain_db": 2.0},
        ],
        "compressor": {"threshold_db": -22.0, "ratio": 2.5, "attack_ms": 8.0, "release_ms": 80.0},
        "target_rms": -20.0,
        "pan": -0.3,
        "delay_ms": 200.0,
    },
    "support_vocal": {
        "hpf": 115.0,
        "eq": [
            {"f0": 400.0, "q": 1.0, "gain_db": -1.5},
            {"f0": 4500.0, "q": 1.3, "gain_db": 2.0},
        ],
        "compressor": {"threshold_db": -21.0, "ratio": 2.3, "attack_ms": 8.0, "release_ms": 90.0},
        "target_rms": -20.0,
        "pan": 0.3,
        "delay_ms": 220.0,
    },
    "pad_vocal": {
        "hpf": 130.0,
        "eq": [
            {"f0": 400.0, "q": 1.0, "gain_db": -2.0},
            {"f0": 5500.0, "q": 1.2, "gain_db": 2.5},
        ],
        "compressor": {"threshold_db": -24.0, "ratio": 2.0, "attack_ms": 10.0, "release_ms": 120.0},
        "target_rms": -24.0,
        "pan": 0.0,
        "delay_ms": 350.0,
        "harmony_shifts": [4, -3],
    },
    "adlib_vocal": {
        "hpf": 150.0,
        "eq": [
            {"f0": 500.0, "q": 1.0, "gain_db": -2.5},
            {"f0": 4500.0, "q": 1.2, "gain_db": 2.5},
        ],
        "compressor": {"threshold_db": -18.0, "ratio": 3.2, "attack_ms": 6.0, "release_ms": 70.0},
        "target_rms": -19.0,
        "pan": 0.6,
        "delay_ms": 280.0,
    },
    "beat": {
        "hpf": 40.0,
        "eq": [],
        "compressor": {"threshold_db": -10.0, "ratio": 1.5, "attack_ms": 10.0, "release_ms": 120.0},
        "target_rms": -12.0,
        "pan": 0.0,
        "delay_ms": 0.0,
    },
}


MASTERING_PROFILES = {
    "soft": {
        "target_lufs": -18.0,
        "enable_saturation": False,
        "max_master_gain_db": 3.0,
        "bus_compression": {"enabled": False, "threshold_db": -24.0, "ratio": 1.15},
    },
    "normal": {
        "target_lufs": -16.0,
        "enable_saturation": False,
        "max_master_gain_db": 4.0,
        "bus_compression": {"enabled": True, "threshold_db": -18.0, "ratio": 1.25},
    },
    "loud": {
        "target_lufs": -14.0,
        "enable_saturation": True,
        "max_master_gain_db": 4.0,
        "bus_compression": {"enabled": True, "threshold_db": -16.0, "ratio": 1.35},
    },
}


class TrackProcessor:
    """Role aware processing of individual tracks."""

    def __init__(self, sr: int = DEFAULT_SR):
        self.sr = sr

    def process(self, track: Track) -> np.ndarray:
        settings = ROLE_SETTINGS.get(track.role)
        data = track.data.copy()
        if data.ndim == 1:
            data = data[:, None]
        if data.shape[1] == 1:
            data = np.repeat(data, 2, axis=1)
        # High-pass filtering
        cutoff = settings.get("hpf", 90.0) if settings else 100.0
        data = apply_highpass(data, cutoff, self.sr, order=3)
        # EQ
        if settings:
            data = apply_peaking_eq(data, settings.get("eq", []), self.sr)
        # Compression
        if settings:
            comp = settings.get("compressor", {})
            data = apply_compressor(
                data,
                threshold_db=comp.get("threshold_db", -20.0),
                ratio=comp.get("ratio", 3.0),
                attack_ms=comp.get("attack_ms", 5.0),
                release_ms=comp.get("release_ms", 80.0),
                sr=self.sr,
            )
        # Normalization
        target_rms = settings.get("target_rms", -20.0) if settings else -20.0
        data = normalize_to_rms(data, target_rms)
        # FX
        delay_ms = settings.get("delay_ms", 0.0) if settings else 0.0
        fx = np.zeros_like(data)
        if delay_ms > 0:
            fx += slapback_delay(data, self.sr, delay_ms)
        if track.role == "pad_vocal":
            fx += self.create_pad_layers(data, settings)
        if track.role == "lead_vocal":
            fx += slapback_delay(data, self.sr, settings.get("delay_ms", 150.0)) * 0.35
        if track.role == "adlib_vocal":
            fx += multi_tap_delay(data, self.sr, [(settings.get("delay_ms", 280.0), -8.0)])
        data = data + fx * 0.5
        # Creative envelope for pads
        if track.role == "pad_vocal":
            env = pad_gain_envelope(data.shape[0], self.sr)
            data = data * env[:, None]
            data = autopan(data, self.sr, period_s=8.0, depth=0.35)
        # Panning
        pan = settings.get("pan", 0.0) if settings else 0.0
        data = apply_pan(data, pan)
        return data

    def create_pad_layers(self, data: np.ndarray, settings: Dict[str, float]) -> np.ndarray:
        layers = np.zeros_like(data)
        for idx, shift in enumerate(settings.get("harmony_shifts", [])):
            shifted = pitch_shift_track(data, self.sr, shift)
            shifted = autopan(shifted, self.sr, period_s=10.0, depth=0.45)
            gain = 0.5 if idx == 0 else 0.35
            layers += shifted * gain
        return layers


def sidechain_ducking(beat: np.ndarray, vocal_bus: np.ndarray, sr: int, depth_db: float = 3.0) -> np.ndarray:
    vocal_env = np.abs(vocal_bus).mean(axis=1)
    attack = math.exp(-1.0 / (sr * 0.003))
    release = math.exp(-1.0 / (sr * 0.08))
    smooth = np.zeros_like(vocal_env)
    for i in range(len(vocal_env)):
        prev = smooth[i - 1] if i > 0 else vocal_env[0]
        if vocal_env[i] > prev:
            smooth[i] = attack * prev + (1 - attack) * vocal_env[i]
        else:
            smooth[i] = release * prev + (1 - release) * vocal_env[i]
    smooth = smooth / (np.max(smooth) + 1e-6)
    gain = 1 - smooth * (1 - db_to_amp(-depth_db))
    return beat * gain[:, None]


def apply_bus_compression(data: np.ndarray, threshold_db: float, ratio: float, sr: int) -> np.ndarray:
    return apply_compressor(data, threshold_db, ratio, 10.0, 120.0, sr)


def mix_tracks(
    tracks: List[Tuple[Track, np.ndarray]],
    sr: int,
    output_dir: Path,
    export_buses: bool = True,
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    buses = {"drums": np.zeros((0, 2)), "vocals": np.zeros((0, 2)), "music": np.zeros((0, 2)), "fx": np.zeros((0, 2))}
    max_len = max(processed.shape[0] for _, processed in tracks)

    def ensure_len(arr: np.ndarray) -> np.ndarray:
        if arr.shape[0] < max_len:
            pad_width = max_len - arr.shape[0]
            arr = np.pad(arr, ((0, pad_width), (0, 0)), mode="constant")
        elif arr.shape[0] > max_len:
            arr = arr[:max_len]
        return arr

    for track, processed in tracks:
        stereo = ensure_stereo(processed)
        stereo = ensure_len(stereo)
        if track.role == "beat":
            buses["drums"] = buses["drums"] + stereo if buses["drums"].size else stereo
        elif track.role == "unknown":
            buses["fx"] = buses["fx"] + stereo if buses["fx"].size else stereo
        else:
            buses["vocals"] = buses["vocals"] + stereo if buses["vocals"].size else stereo
    for key, value in buses.items():
        if value.size == 0:
            buses[key] = np.zeros((max_len, 2))
        else:
            buses[key] = ensure_len(value)
    if export_buses:
        for name, data in buses.items():
            bus_path = output_dir / f"bus_{name}.wav"
            sf.write(str(bus_path), data, sr)
            print(f"[info] Saved {name} bus to {bus_path}")
    if np.max(np.abs(buses["vocals"])) > 0 and np.max(np.abs(buses["drums"])) > 0:
        buses["drums"] = sidechain_ducking(buses["drums"], buses["vocals"], sr)
    buses["vocals"] = apply_bus_compression(buses["vocals"], -18.0, 1.5, sr)
    mix = buses["drums"] + buses["vocals"] + buses["music"] + buses["fx"]
    return mix, buses


def master_chain(mix: np.ndarray, sr: int, user_config: Optional[Dict[str, Any]] = None) -> np.ndarray:
    """Apply a conservative mastering chain with configurable behavior."""

    config = user_config or {}
    profile_name = config.get("master_profile", "normal")
    profile = MASTERING_PROFILES.get(profile_name, MASTERING_PROFILES["normal"])

    target_lufs = config.get("target_lufs", profile["target_lufs"])
    max_gain_db = config.get("max_master_gain_db", profile["max_master_gain_db"])
    enable_saturation = config.get("enable_saturation", profile["enable_saturation"])

    bus_threshold = config.get(
        "master_bus_comp_threshold_db", profile["bus_compression"]["threshold_db"]
    )
    bus_ratio = config.get("master_bus_comp_ratio", profile["bus_compression"]["ratio"])
    bus_enabled = config.get(
        "enable_master_bus_compression", profile["bus_compression"]["enabled"]
    )

    # Optional gentle saturation to add harmonics if enabled.
    if enable_saturation:
        mix = saturate(mix, config.get("saturation_drive", 1.05))

    # Very light master bus compression keeps transients intact.
    if bus_enabled:
        mix = apply_bus_compression(mix, bus_threshold, bus_ratio, sr)

    # Loudness normalization with limited upward gain so hot mixes stay musical.
    loudness = compute_loudness(mix, sr)
    current = loudness.get("lufs", loudness["rms_db"])
    gain_db = target_lufs - current
    if gain_db > max_gain_db:
        gain_db = max_gain_db
    mix = mix * db_to_amp(gain_db)

    # Soft limiting around -1.5 dBFS gently reins in peaks.
    limiter_threshold = config.get("master_limiter_threshold_db", -1.5)
    mix = soft_limiter(mix, limiter_threshold)

    # Final peak normalization ensures headroom of ~1 dBFS.
    peak = np.max(np.abs(mix))
    target_peak_db = config.get("master_peak_db", -1.0)
    target_peak = db_to_amp(target_peak_db)
    if peak > 0 and peak > target_peak:
        mix *= target_peak / peak

    return mix


def load_audio(path: Path, target_sr: int) -> np.ndarray:
    data, sr = sf.read(str(path))
    if sr != target_sr:
        data = librosa.resample(data.T, orig_sr=sr, target_sr=target_sr).T
    if data.ndim == 1:
        return data[:, None]
    return data[:, :2]


def align_tracks(tracks: List[Track], target_len: int) -> None:
    for track in tracks:
        data = track.data
        if data.shape[0] < target_len:
            pad = target_len - data.shape[0]
            track.data = np.pad(data, ((0, pad), (0, 0)), mode="constant")
        elif data.shape[0] > target_len:
            track.data = data[:target_len]


def load_tracks(input_dir: Path, sr: int) -> List[Track]:
    wav_files = sorted(input_dir.glob("*.wav"))
    tracks = []
    for path in wav_files:
        data = load_audio(path, sr)
        role = infer_role_from_filename(path.name)
        if role == "unknown":
            print(f"[warn] Could not infer role for {path.name}, assigning support_vocal")
            role = "support_vocal"
        track = Track(name=path.name, role=role, data=data, sr=sr, channels=data.shape[1])
        tracks.append(track)
    if not tracks:
        raise RuntimeError("No WAV files found in input directory")
    max_len = max(track.data.shape[0] for track in tracks)
    align_tracks(tracks, max_len)
    return tracks


def save_report(report: Dict, output_path: Path) -> None:
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)


def parse_config(path: Optional[Path]) -> Dict:
    if path is None:
        return {}
    if not path.exists():
        raise FileNotFoundError(f"Config file {path} does not exist")
    if path.suffix.lower() in {".yaml", ".yml"}:
        if yaml is None:
            raise RuntimeError("pyyaml is not installed")
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def export_master_versions(mix_48k: np.ndarray, output_dir: Path) -> Dict[str, str]:
    destinations = {
        "master_48k": output_dir / "master_48k.wav",
        "master_44k1": output_dir / "master_44k1.wav",
    }
    sf.write(str(destinations["master_48k"]), mix_48k, DEFAULT_SR)
    print(f"[info] Saved 48 kHz master to {destinations['master_48k']}")
    mix_44k1 = librosa.resample(mix_48k.T, orig_sr=DEFAULT_SR, target_sr=MASTER_SR).T
    sf.write(str(destinations["master_44k1"]), mix_44k1, MASTER_SR)
    print(f"[info] Saved 44.1 kHz master to {destinations['master_44k1']}")
    return {key: str(path) for key, path in destinations.items()}


def build_report(
    tracks: List[Track],
    processed: List[Tuple[Track, np.ndarray]],
    mix: np.ndarray,
    sr: int,
    artifacts: Optional[Dict[str, str]] = None,
) -> Dict:
    report = {"tracks": {}, "master": {}, "artifacts": artifacts or {}}
    for track, audio in processed:
        stats = compute_loudness(audio, sr)
        report["tracks"][track.name] = {"role": track.role, **stats}
    report["master"] = compute_loudness(mix, sr)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Autonomous vocal mix and mastering engine")
    parser.add_argument("--input", "-i", required=True, help="Input directory with stems")
    parser.add_argument("--output", "-o", required=True, help="Output directory")
    parser.add_argument("--config", "-c", help="Optional path to JSON/YAML config")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input)
    output_dir = Path(args.output)
    config_path = Path(args.config) if args.config else None

    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory {input_dir} does not exist")
    output_dir.mkdir(parents=True, exist_ok=True)

    user_config = parse_config(config_path)
    sr = user_config.get("sample_rate", DEFAULT_SR)

    print("[info] Loading tracks...")
    tracks = load_tracks(input_dir, sr)
    processor = TrackProcessor(sr=sr)
    processed_tracks = []
    for track in tracks:
        print(f"[info] Processing {track.name} ({track.role})")
        processed = processor.process(track)
        processed_tracks.append((track, processed))

    print("[info] Mixing buses...")
    mix, buses = mix_tracks(processed_tracks, sr, output_dir, export_buses=True)

    print("[info] Mastering...")
    mastered = master_chain(mix, sr, user_config=user_config)

    print("[info] Exporting masters...")
    saved_masters = export_master_versions(mastered, output_dir)

    print("[info] Building report...")
    report = build_report(tracks, processed_tracks, mastered, sr, artifacts=saved_masters)
    save_report(report, output_dir / "mix_report.json")

    print("[done] Mix and mastering complete")


if __name__ == "__main__":
    main()
