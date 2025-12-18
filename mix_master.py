#!/usr/bin/env python3
"""
Professional Audio Mixing & Mastering Engine
Автоматическое сведение и мастеринг трека
"""

import wave
import numpy as np
from scipy import signal
from scipy.ndimage import uniform_filter1d
import os
import struct

# ============================================
# CORE DSP FUNCTIONS
# ============================================

def read_wav(filepath):
    """Читает WAV файл и возвращает нормализованные сэмплы"""
    with wave.open(filepath, 'rb') as w:
        channels = w.getnchannels()
        sampwidth = w.getsampwidth()
        sample_rate = w.getframerate()
        frames = w.readframes(w.getnframes())

        if sampwidth == 2:
            samples = np.frombuffer(frames, dtype=np.int16).astype(np.float64) / 32768.0
        elif sampwidth == 3:
            n_samples = len(frames) // 3
            raw = np.frombuffer(frames, dtype=np.uint8).reshape(n_samples, 3)
            samples = (raw[:, 2].astype(np.int32) << 16) | (raw[:, 1].astype(np.int32) << 8) | raw[:, 0]
            samples = np.where(samples >= 0x800000, samples - 0x1000000, samples).astype(np.float64) / 8388608.0
        else:
            raise ValueError(f"Unsupported sample width: {sampwidth}")

        if channels == 2:
            samples = samples.reshape(-1, 2)

        return samples, sample_rate, channels

def write_wav(filepath, samples, sample_rate, channels=2):
    """Записывает WAV файл в 24-bit"""
    samples = np.clip(samples, -1.0, 1.0)

    if channels == 2 and samples.ndim == 1:
        samples = np.column_stack([samples, samples])
    elif channels == 1 and samples.ndim == 2:
        samples = samples.mean(axis=1)

    # Convert to 24-bit
    samples_int = (samples * 8388607).astype(np.int32)

    with wave.open(filepath, 'wb') as w:
        w.setnchannels(channels)
        w.setsampwidth(3)  # 24-bit
        w.setframerate(sample_rate)

        if channels == 2:
            for i in range(len(samples_int)):
                for ch in range(2):
                    val = int(samples_int[i, ch])
                    if val < 0:
                        val += 0x1000000
                    w.writeframes(struct.pack('<I', val)[:3])
        else:
            for val in samples_int:
                if val < 0:
                    val += 0x1000000
                w.writeframes(struct.pack('<I', val)[:3])

def resample(samples, orig_sr, target_sr):
    """Ресемплинг аудио"""
    if orig_sr == target_sr:
        return samples

    ratio = target_sr / orig_sr
    if samples.ndim == 1:
        new_length = int(len(samples) * ratio)
        return signal.resample(samples, new_length)
    else:
        new_length = int(len(samples) * ratio)
        return np.column_stack([
            signal.resample(samples[:, 0], new_length),
            signal.resample(samples[:, 1], new_length)
        ])

# ============================================
# FILTERS & EQ
# ============================================

def highpass_filter(samples, cutoff, sample_rate, order=4):
    """High-pass фильтр"""
    nyquist = sample_rate / 2
    normalized_cutoff = cutoff / nyquist
    b, a = signal.butter(order, normalized_cutoff, btype='high')

    if samples.ndim == 1:
        return signal.filtfilt(b, a, samples)
    else:
        return np.column_stack([
            signal.filtfilt(b, a, samples[:, 0]),
            signal.filtfilt(b, a, samples[:, 1])
        ])

def lowpass_filter(samples, cutoff, sample_rate, order=4):
    """Low-pass фильтр"""
    nyquist = sample_rate / 2
    normalized_cutoff = min(cutoff / nyquist, 0.99)
    b, a = signal.butter(order, normalized_cutoff, btype='low')

    if samples.ndim == 1:
        return signal.filtfilt(b, a, samples)
    else:
        return np.column_stack([
            signal.filtfilt(b, a, samples[:, 0]),
            signal.filtfilt(b, a, samples[:, 1])
        ])

def parametric_eq(samples, sample_rate, freq, gain_db, q=1.0):
    """Параметрический эквалайзер (peaking filter)"""
    A = 10 ** (gain_db / 40)
    w0 = 2 * np.pi * freq / sample_rate
    alpha = np.sin(w0) / (2 * q)

    b0 = 1 + alpha * A
    b1 = -2 * np.cos(w0)
    b2 = 1 - alpha * A
    a0 = 1 + alpha / A
    a1 = -2 * np.cos(w0)
    a2 = 1 - alpha / A

    b = np.array([b0/a0, b1/a0, b2/a0])
    a = np.array([1, a1/a0, a2/a0])

    if samples.ndim == 1:
        return signal.filtfilt(b, a, samples)
    else:
        return np.column_stack([
            signal.filtfilt(b, a, samples[:, 0]),
            signal.filtfilt(b, a, samples[:, 1])
        ])

def high_shelf(samples, sample_rate, freq, gain_db):
    """High shelf EQ"""
    A = 10 ** (gain_db / 40)
    w0 = 2 * np.pi * freq / sample_rate
    alpha = np.sin(w0) / 2 * np.sqrt(2)

    cos_w0 = np.cos(w0)
    sqrt_A = np.sqrt(A)

    b0 = A * ((A + 1) + (A - 1) * cos_w0 + 2 * sqrt_A * alpha)
    b1 = -2 * A * ((A - 1) + (A + 1) * cos_w0)
    b2 = A * ((A + 1) + (A - 1) * cos_w0 - 2 * sqrt_A * alpha)
    a0 = (A + 1) - (A - 1) * cos_w0 + 2 * sqrt_A * alpha
    a1 = 2 * ((A - 1) - (A + 1) * cos_w0)
    a2 = (A + 1) - (A - 1) * cos_w0 - 2 * sqrt_A * alpha

    b = np.array([b0/a0, b1/a0, b2/a0])
    a = np.array([1, a1/a0, a2/a0])

    if samples.ndim == 1:
        return signal.filtfilt(b, a, samples)
    else:
        return np.column_stack([
            signal.filtfilt(b, a, samples[:, 0]),
            signal.filtfilt(b, a, samples[:, 1])
        ])

def low_shelf(samples, sample_rate, freq, gain_db):
    """Low shelf EQ"""
    A = 10 ** (gain_db / 40)
    w0 = 2 * np.pi * freq / sample_rate
    alpha = np.sin(w0) / 2 * np.sqrt(2)

    cos_w0 = np.cos(w0)
    sqrt_A = np.sqrt(A)

    b0 = A * ((A + 1) - (A - 1) * cos_w0 + 2 * sqrt_A * alpha)
    b1 = 2 * A * ((A - 1) - (A + 1) * cos_w0)
    b2 = A * ((A + 1) - (A - 1) * cos_w0 - 2 * sqrt_A * alpha)
    a0 = (A + 1) + (A - 1) * cos_w0 + 2 * sqrt_A * alpha
    a1 = -2 * ((A - 1) + (A + 1) * cos_w0)
    a2 = (A + 1) + (A - 1) * cos_w0 - 2 * sqrt_A * alpha

    b = np.array([b0/a0, b1/a0, b2/a0])
    a = np.array([1, a1/a0, a2/a0])

    if samples.ndim == 1:
        return signal.filtfilt(b, a, samples)
    else:
        return np.column_stack([
            signal.filtfilt(b, a, samples[:, 0]),
            signal.filtfilt(b, a, samples[:, 1])
        ])

# ============================================
# DYNAMICS
# ============================================

def compressor(samples, threshold_db, ratio, attack_ms, release_ms, sample_rate, makeup_db=0):
    """Компрессор с автоматическим makeup gain"""
    threshold = 10 ** (threshold_db / 20)
    makeup = 10 ** (makeup_db / 20)

    attack_samples = int(attack_ms * sample_rate / 1000)
    release_samples = int(release_ms * sample_rate / 1000)

    attack_coef = np.exp(-1 / max(attack_samples, 1))
    release_coef = np.exp(-1 / max(release_samples, 1))

    if samples.ndim == 2:
        mono = np.abs(samples).max(axis=1)
    else:
        mono = np.abs(samples)

    envelope = np.zeros_like(mono)
    env = 0

    for i in range(len(mono)):
        if mono[i] > env:
            env = attack_coef * env + (1 - attack_coef) * mono[i]
        else:
            env = release_coef * env + (1 - release_coef) * mono[i]
        envelope[i] = env

    # Calculate gain reduction
    gain = np.ones_like(envelope)
    above_threshold = envelope > threshold
    gain[above_threshold] = threshold * (envelope[above_threshold] / threshold) ** (1/ratio - 1)
    gain = gain / (envelope + 1e-10)
    gain = np.clip(gain, 0.01, 1.0)

    # Smooth gain
    gain = uniform_filter1d(gain, size=int(sample_rate * 0.005))

    if samples.ndim == 2:
        return samples * gain[:, np.newaxis] * makeup
    else:
        return samples * gain * makeup

def limiter(samples, threshold_db=-1.0, release_ms=50, sample_rate=48000):
    """Lookahead лимитер"""
    threshold = 10 ** (threshold_db / 20)
    lookahead = int(sample_rate * 0.005)  # 5ms lookahead
    release_samples = int(release_ms * sample_rate / 1000)

    if samples.ndim == 2:
        peak = np.abs(samples).max(axis=1)
    else:
        peak = np.abs(samples)

    # Lookahead - find max in future window
    gain_reduction = np.ones_like(peak)
    for i in range(len(peak)):
        window_end = min(i + lookahead, len(peak))
        window_max = np.max(peak[i:window_end])
        if window_max > threshold:
            gain_reduction[i] = threshold / window_max

    # Smooth gain changes
    release_coef = np.exp(-1 / max(release_samples, 1))
    smoothed = np.zeros_like(gain_reduction)
    current = 1.0

    for i in range(len(gain_reduction)):
        if gain_reduction[i] < current:
            current = gain_reduction[i]
        else:
            current = release_coef * current + (1 - release_coef) * gain_reduction[i]
        smoothed[i] = current

    if samples.ndim == 2:
        return samples * smoothed[:, np.newaxis]
    else:
        return samples * smoothed

def de_esser(samples, sample_rate, freq=6000, threshold_db=-20, ratio=4):
    """Де-эссер для убирания сибилянтов"""
    # Создаём bandpass для детекции
    nyquist = sample_rate / 2
    low = (freq - 1500) / nyquist
    high = min((freq + 3000) / nyquist, 0.99)
    b, a = signal.butter(2, [low, high], btype='band')

    if samples.ndim == 1:
        sibilance = signal.filtfilt(b, a, samples)
    else:
        sibilance = signal.filtfilt(b, a, samples.mean(axis=1))

    # Компрессируем только sibilance
    compressed = compressor(sibilance, threshold_db, ratio, 0.5, 30, sample_rate)

    # Вычитаем разницу
    reduction = sibilance - compressed

    if samples.ndim == 2:
        return samples - np.column_stack([reduction, reduction])
    else:
        return samples - reduction

# ============================================
# EFFECTS
# ============================================

def reverb(samples, sample_rate, decay=2.0, wet=0.3, predelay_ms=40):
    """Алгоритмический ревербератор (Schroeder design)"""
    predelay_samples = int(predelay_ms * sample_rate / 1000)

    if samples.ndim == 2:
        mono = samples.mean(axis=1)
    else:
        mono = samples

    # Predelay
    mono_delayed = np.pad(mono, (predelay_samples, 0))[:len(mono)]

    # Comb filters (parallel)
    comb_delays = [int(d * sample_rate / 1000) for d in [29.7, 37.1, 41.1, 43.7]]
    comb_outputs = []

    for delay in comb_delays:
        feedback = 0.84 ** (delay / (decay * sample_rate))
        output = np.zeros(len(mono_delayed) + delay)

        for i in range(len(mono_delayed)):
            output[i + delay] = mono_delayed[i] + feedback * output[i]

        comb_outputs.append(output[:len(mono)])

    # Sum comb filters
    reverb_signal = sum(comb_outputs) / len(comb_outputs)

    # All-pass filters (series) for diffusion
    allpass_delays = [int(d * sample_rate / 1000) for d in [5.0, 1.7]]

    for delay in allpass_delays:
        g = 0.7
        output = np.zeros(len(reverb_signal) + delay)

        for i in range(len(reverb_signal)):
            output[i + delay] = -g * output[i] + reverb_signal[i] + g * (output[i] if i < len(output) else 0)

        reverb_signal = output[:len(mono)]

    # LPF on reverb
    reverb_signal = lowpass_filter(reverb_signal, 6000, sample_rate, 2)

    # HPF on reverb
    reverb_signal = highpass_filter(reverb_signal, 300, sample_rate, 2)

    # Mix
    if samples.ndim == 2:
        dry = samples * (1 - wet)
        wet_signal = np.column_stack([reverb_signal * wet, reverb_signal * wet])
        return dry + wet_signal
    else:
        return samples * (1 - wet) + reverb_signal * wet

def delay(samples, sample_rate, time_ms=120, feedback=0.3, wet=0.2, ping_pong=False):
    """Delay эффект с ping-pong опцией"""
    delay_samples = int(time_ms * sample_rate / 1000)

    if samples.ndim == 1:
        samples = np.column_stack([samples, samples])

    output_l = np.zeros(len(samples) + delay_samples * 10)
    output_r = np.zeros(len(samples) + delay_samples * 10)

    output_l[:len(samples)] = samples[:, 0]
    output_r[:len(samples)] = samples[:, 1]

    if ping_pong:
        for i in range(len(samples)):
            idx = i + delay_samples
            if idx < len(output_l):
                output_l[idx] += output_r[i] * feedback
            idx = i + delay_samples * 2
            if idx < len(output_r):
                output_r[idx] += output_l[i + delay_samples] * feedback * 0.7 if i + delay_samples < len(output_l) else 0
    else:
        for i in range(len(samples)):
            idx = i + delay_samples
            if idx < len(output_l):
                output_l[idx] += samples[i, 0] * feedback
                output_r[idx] += samples[i, 1] * feedback

    output_l = output_l[:len(samples)]
    output_r = output_r[:len(samples)]

    # LPF on delay (dark delay)
    output_l = lowpass_filter(output_l, 4000, sample_rate, 2)
    output_r = lowpass_filter(output_r, 4000, sample_rate, 2)

    dry = samples * (1 - wet)
    wet_signal = np.column_stack([output_l * wet, output_r * wet])

    return dry + wet_signal

def stereo_widener(samples, width=1.5):
    """Расширение стерео базы с помощью mid-side"""
    if samples.ndim == 1:
        return samples

    mid = (samples[:, 0] + samples[:, 1]) / 2
    side = (samples[:, 0] - samples[:, 1]) / 2

    # Увеличиваем side
    side = side * width

    left = mid + side
    right = mid - side

    return np.column_stack([left, right])

def mono_to_stereo_width(samples, width_cents=7, delay_ms=15):
    """Создаёт стерео из моно с micro-pitch shift эффектом"""
    if samples.ndim == 2:
        samples = samples.mean(axis=1)

    # Простой pitch shift через resampling (приблизительный)
    shift_ratio = 2 ** (width_cents / 1200)

    # Left: pitch up slightly
    left = signal.resample(samples, int(len(samples) / shift_ratio))
    left = signal.resample(left, len(samples))

    # Right: pitch down + delay
    right = signal.resample(samples, int(len(samples) * shift_ratio))
    right = signal.resample(right, len(samples))

    # Add delay to right
    delay_samples = int(delay_ms * 48000 / 1000)
    right = np.pad(right, (delay_samples, 0))[:len(samples)]

    return np.column_stack([left, right])

def sidechain_compression(samples, sidechain_signal, threshold_db=-20, ratio=3, attack_ms=10, release_ms=100, sample_rate=48000):
    """Sidechain компрессия - сигнал утихает когда sidechain громкий"""
    if sidechain_signal.ndim == 2:
        sc_mono = np.abs(sidechain_signal).max(axis=1)
    else:
        sc_mono = np.abs(sidechain_signal)

    # Extend or trim sidechain to match samples length
    if len(sc_mono) < len(samples):
        sc_mono = np.pad(sc_mono, (0, len(samples) - len(sc_mono)))
    else:
        sc_mono = sc_mono[:len(samples) if samples.ndim == 1 else len(samples)]

    threshold = 10 ** (threshold_db / 20)
    attack_coef = np.exp(-1 / max(int(attack_ms * sample_rate / 1000), 1))
    release_coef = np.exp(-1 / max(int(release_ms * sample_rate / 1000), 1))

    envelope = np.zeros(len(sc_mono))
    env = 0

    for i in range(len(sc_mono)):
        if sc_mono[i] > env:
            env = attack_coef * env + (1 - attack_coef) * sc_mono[i]
        else:
            env = release_coef * env + (1 - release_coef) * sc_mono[i]
        envelope[i] = env

    # Gain reduction
    gain = np.ones_like(envelope)
    above = envelope > threshold
    gain[above] = threshold * (envelope[above] / threshold) ** (1/ratio - 1)
    gain = gain / (envelope + 1e-10)
    gain = np.clip(gain, 0.1, 1.0)

    # Smooth
    gain = uniform_filter1d(gain, size=int(sample_rate * 0.01))

    if samples.ndim == 2:
        return samples * gain[:len(samples), np.newaxis]
    else:
        return samples * gain[:len(samples)]

def soft_saturation(samples, drive_db=3):
    """Мягкая сатурация (tape-style)"""
    drive = 10 ** (drive_db / 20)
    samples = samples * drive

    # Soft clipping
    samples = np.tanh(samples) / np.tanh(drive)

    return samples

# ============================================
# MAIN PROCESSING CHAIN
# ============================================

def process_lead_vocal(samples, sr):
    """Обработка лид вокала"""
    print("  [Lead] HPF 80 Hz...")
    samples = highpass_filter(samples, 80, sr, 4)

    print("  [Lead] Cut 250 Hz (-3 dB)...")
    samples = parametric_eq(samples, sr, 250, -3, q=2)

    print("  [Lead] Cut 450 Hz (-2 dB)...")
    samples = parametric_eq(samples, sr, 450, -2, q=3)

    print("  [Lead] Boost 3 kHz (+3 dB)...")
    samples = parametric_eq(samples, sr, 3000, 3, q=1.5)

    print("  [Lead] High shelf 10 kHz (+4 dB)...")
    samples = high_shelf(samples, sr, 10000, 4)

    print("  [Lead] De-esser...")
    samples = de_esser(samples, sr, 6000, -25, 4)

    print("  [Lead] Compression (4:1)...")
    samples = compressor(samples, -18, 4, 10, 100, sr, makeup_db=3)

    print("  [Lead] Second compression (2:1)...")
    samples = compressor(samples, -12, 2, 30, 150, sr, makeup_db=1)

    print("  [Lead] Soft saturation...")
    samples = soft_saturation(samples, 1.5)

    return samples

def process_double(samples, sr):
    """Обработка дабла"""
    print("  [Double] HPF 120 Hz...")
    samples = highpass_filter(samples, 120, sr, 3)

    print("  [Double] Cut 300 Hz (-4 dB)...")
    samples = parametric_eq(samples, sr, 300, -4, q=1.5)

    print("  [Double] Cut 2.5 kHz (-2 dB)...")
    samples = parametric_eq(samples, sr, 2500, -2, q=2)

    print("  [Double] Boost 5 kHz (+2 dB)...")
    samples = parametric_eq(samples, sr, 5000, 2, q=1.5)

    print("  [Double] Compression (3:1)...")
    samples = compressor(samples, -15, 3, 20, 100, sr, makeup_db=2)

    print("  [Double] Creating stereo width...")
    samples = mono_to_stereo_width(samples, width_cents=8, delay_ms=18)

    return samples

def process_backs_uuu(samples, sr):
    """Обработка бэков у-у-у"""
    print("  [Backs UUU] HPF 150 Hz...")
    samples = highpass_filter(samples, 150, sr, 3)

    print("  [Backs UUU] Cut 400 Hz (-3 dB)...")
    samples = parametric_eq(samples, sr, 400, -3, q=2)

    print("  [Backs UUU] Cut 1 kHz (-2 dB)...")
    samples = parametric_eq(samples, sr, 1000, -2, q=2)

    print("  [Backs UUU] Boost 4 kHz (+4 dB)...")
    samples = parametric_eq(samples, sr, 4000, 4, q=1.5)

    print("  [Backs UUU] High shelf 10 kHz (+3 dB)...")
    samples = high_shelf(samples, sr, 10000, 3)

    print("  [Backs UUU] Heavy compression (5:1)...")
    samples = compressor(samples, -20, 5, 5, 60, sr, makeup_db=4)

    print("  [Backs UUU] Creating stereo width...")
    samples = mono_to_stereo_width(samples, width_cents=10, delay_ms=22)

    return samples

def process_backs_shu(samples, sr):
    """Обработка бэков шу-шу"""
    print("  [Backs SHU] HPF 200 Hz...")
    samples = highpass_filter(samples, 200, sr, 4)

    print("  [Backs SHU] De-esser (heavy)...")
    samples = de_esser(samples, sr, 7000, -22, 6)

    print("  [Backs SHU] Cut 3 kHz (-4 dB)...")
    samples = parametric_eq(samples, sr, 3000, -4, q=2)

    print("  [Backs SHU] Boost 7 kHz (+2 dB)...")
    samples = parametric_eq(samples, sr, 7000, 2, q=1.5)

    print("  [Backs SHU] Very heavy compression (6:1)...")
    samples = compressor(samples, -22, 6, 2, 50, sr, makeup_db=5)

    print("  [Backs SHU] Creating wide stereo...")
    samples = mono_to_stereo_width(samples, width_cents=12, delay_ms=25)

    return samples

def process_backs_vse(samples, sr):
    """Обработка 'всё что было важно'"""
    print("  [Backs VSE] HPF 100 Hz...")
    samples = highpass_filter(samples, 100, sr, 3)

    print("  [Backs VSE] Cut 300 Hz (-3 dB)...")
    samples = parametric_eq(samples, sr, 300, -3, q=2)

    print("  [Backs VSE] Cut 1.5 kHz (-2 dB)...")
    samples = parametric_eq(samples, sr, 1500, -2, q=2)

    print("  [Backs VSE] Boost 4.5 kHz (+3 dB)...")
    samples = parametric_eq(samples, sr, 4500, 3, q=1.5)

    print("  [Backs VSE] High shelf 10 kHz (+2 dB)...")
    samples = high_shelf(samples, sr, 10000, 2)

    print("  [Backs VSE] Compression (3:1)...")
    samples = compressor(samples, -18, 3, 15, 100, sr, makeup_db=3)

    print("  [Backs VSE] Creating stereo width...")
    samples = mono_to_stereo_width(samples, width_cents=6, delay_ms=12)

    return samples

def process_instrumental(samples, sr, vocal_sidechain=None):
    """Обработка инструментала"""
    print("  [Inst] Cut 300 Hz (-2 dB)...")
    samples = parametric_eq(samples, sr, 300, -2, q=0.8)

    print("  [Inst] Cut 2.5 kHz (-3 dB) - making room for vocals...")
    samples = parametric_eq(samples, sr, 2500, -3, q=1.5)

    print("  [Inst] Boost 60 Hz (+1 dB)...")
    samples = parametric_eq(samples, sr, 60, 1, q=1)

    print("  [Inst] High shelf 10 kHz (+1.5 dB)...")
    samples = high_shelf(samples, sr, 10000, 1.5)

    print("  [Inst] Stereo widening...")
    samples = stereo_widener(samples, 1.2)

    if vocal_sidechain is not None:
        print("  [Inst] Sidechain compression from vocals...")
        samples = sidechain_compression(samples, vocal_sidechain, -18, 2.5, 10, 120, sr)

    return samples

def master(samples, sr):
    """Мастеринг цепочка"""
    print("  [Master] Soft saturation (analog warmth)...")
    samples = soft_saturation(samples, 1)

    print("  [Master] EQ sweetening...")
    samples = parametric_eq(samples, sr, 100, 0.5, q=1)
    samples = parametric_eq(samples, sr, 3000, 0.5, q=1)
    samples = high_shelf(samples, sr, 12000, 1)

    print("  [Master] Stereo enhancement...")
    samples = stereo_widener(samples, 1.1)

    print("  [Master] Final compression (glue)...")
    samples = compressor(samples, -10, 2, 30, 100, sr, makeup_db=1)

    print("  [Master] Limiting to -1 dBTP...")
    samples = limiter(samples, -1.0, 50, sr)

    return samples

# ============================================
# MAIN
# ============================================

def main():
    print("=" * 60)
    print("   PROFESSIONAL MIXING & MASTERING ENGINE")
    print("   Creating your HIT!")
    print("=" * 60)

    TARGET_SR = 48000
    PROJECT_DIR = "/home/user/noty"

    # File mapping
    files = {
        'lead': "1 дорожка.wav",
        'double': "2 дорожка_дабл_лида.wav",
        'backs_uuu': "3 дороожка_звуки_у-у-у_саймур_с_вами.wav",
        'backs_shu': "4 дорожка_шу_шу_еа.wav",
        'backs_vse': "5 дорожка_все_что_было_важно.wav",
        'instrumental': "Untitled (3).wav"
    }

    processed = {}

    # Load and process each track
    print("\n[STEP 1] Loading and processing tracks...\n")

    for key, filename in files.items():
        filepath = os.path.join(PROJECT_DIR, filename)
        print(f"Loading: {filename}")
        samples, sr, channels = read_wav(filepath)

        # Resample if needed
        if sr != TARGET_SR:
            print(f"  Resampling {sr} -> {TARGET_SR} Hz...")
            samples = resample(samples, sr, TARGET_SR)

        # Process based on track type
        if key == 'lead':
            print("Processing LEAD VOCAL...")
            samples = process_lead_vocal(samples, TARGET_SR)
        elif key == 'double':
            print("Processing DOUBLE...")
            samples = process_double(samples, TARGET_SR)
        elif key == 'backs_uuu':
            print("Processing BACKS (U-U-U)...")
            samples = process_backs_uuu(samples, TARGET_SR)
        elif key == 'backs_shu':
            print("Processing BACKS (SHU-SHU)...")
            samples = process_backs_shu(samples, TARGET_SR)
        elif key == 'backs_vse':
            print("Processing BACKS (VSE)...")
            samples = process_backs_vse(samples, TARGET_SR)
        elif key == 'instrumental':
            print("Processing INSTRUMENTAL (will add sidechain later)...")
            # Store for now, process after vocals

        processed[key] = samples
        print()

    # Create vocal bus for sidechain
    print("[STEP 2] Creating vocal bus for sidechain...\n")

    # Get the length of instrumental (longest track)
    inst_len = len(processed['instrumental'])

    # Pad all vocals to match instrumental length
    for key in ['lead', 'double', 'backs_uuu', 'backs_shu', 'backs_vse']:
        if processed[key].ndim == 1:
            processed[key] = np.column_stack([processed[key], processed[key]])

        if len(processed[key]) < inst_len:
            pad_len = inst_len - len(processed[key])
            processed[key] = np.pad(processed[key], ((0, pad_len), (0, 0)))

    # Create vocal sum for sidechain
    vocal_sum = (
        processed['lead'] * 1.0 +
        processed['double'] * 0.5 +
        processed['backs_uuu'] * 0.3 +
        processed['backs_shu'] * 0.2 +
        processed['backs_vse'] * 0.4
    )

    # Process instrumental with sidechain
    print("Processing INSTRUMENTAL with sidechain from vocals...")
    processed['instrumental'] = process_instrumental(processed['instrumental'], TARGET_SR, vocal_sum)
    print()

    # Add effects to vocals
    print("[STEP 3] Adding effects (reverb, delay)...\n")

    print("Adding reverb to lead vocal...")
    processed['lead'] = reverb(processed['lead'], TARGET_SR, decay=1.8, wet=0.15, predelay_ms=50)

    print("Adding reverb to double...")
    processed['double'] = reverb(processed['double'], TARGET_SR, decay=2.0, wet=0.2, predelay_ms=40)

    print("Adding reverb + delay to backs...")
    processed['backs_uuu'] = reverb(processed['backs_uuu'], TARGET_SR, decay=2.2, wet=0.25, predelay_ms=30)
    processed['backs_uuu'] = delay(processed['backs_uuu'], TARGET_SR, time_ms=180, feedback=0.25, wet=0.15, ping_pong=True)

    processed['backs_shu'] = reverb(processed['backs_shu'], TARGET_SR, decay=1.5, wet=0.2, predelay_ms=20)

    processed['backs_vse'] = reverb(processed['backs_vse'], TARGET_SR, decay=2.0, wet=0.22, predelay_ms=35)
    print()

    # Mixing
    print("[STEP 4] Mixing all tracks...\n")

    # Mix levels (relative to lead)
    levels = {
        'instrumental': 0.85,   # Slightly below vocals
        'lead': 1.0,           # Reference
        'double': 0.45,         # Supporting
        'backs_uuu': 0.35,      # Background
        'backs_shu': 0.25,      # Background
        'backs_vse': 0.40       # Background
    }

    # Pan positions (L=-1, C=0, R=1)
    pans = {
        'instrumental': 0,
        'lead': 0,
        'double': 0,  # Already stereo widened
        'backs_uuu': 0,  # Already stereo widened
        'backs_shu': 0,  # Already stereo widened
        'backs_vse': 0   # Already stereo widened
    }

    # Mix
    mix = np.zeros((inst_len, 2))

    for key, samples in processed.items():
        level = levels[key]

        if samples.ndim == 1:
            samples = np.column_stack([samples, samples])

        # Ensure same length
        if len(samples) > inst_len:
            samples = samples[:inst_len]
        elif len(samples) < inst_len:
            samples = np.pad(samples, ((0, inst_len - len(samples)), (0, 0)))

        mix += samples * level
        print(f"  Mixed: {key} at {level*100:.0f}%")

    print()

    # Normalize before mastering
    peak = np.max(np.abs(mix))
    if peak > 0.9:
        mix = mix * (0.9 / peak)
        print(f"Normalized mix (peak was {20*np.log10(peak):.1f} dBFS)")

    # Mastering
    print("\n[STEP 5] Mastering...\n")
    final = master(mix, TARGET_SR)

    # Export
    print("\n[STEP 6] Exporting...\n")

    output_path = os.path.join(PROJECT_DIR, "FINAL_MIX_MASTER.wav")
    write_wav(output_path, final, TARGET_SR, channels=2)

    # Calculate final stats
    peak_db = 20 * np.log10(np.max(np.abs(final)) + 1e-10)
    rms = np.sqrt(np.mean(final**2))
    rms_db = 20 * np.log10(rms + 1e-10)

    print("=" * 60)
    print("   DONE! Your HIT is ready!")
    print("=" * 60)
    print(f"\nOutput: {output_path}")
    print(f"Format: 24-bit / 48 kHz / Stereo")
    print(f"Duration: {len(final)/TARGET_SR:.1f} seconds")
    print(f"Peak: {peak_db:.1f} dBFS")
    print(f"RMS: {rms_db:.1f} dBFS")
    print(f"Estimated LUFS: ~{rms_db + 3:.0f} LUFS")
    print("\nGO DROP THAT FIRE!")

if __name__ == "__main__":
    main()
