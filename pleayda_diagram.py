"""Generate a clean SVG constellation diagram with styled correlations."""

import math
from pathlib import Path

WIDTH, HEIGHT = 1200, 900
MARGIN = 80

nodes = [
    {"name": "ПОН\nIST", "pos": (0.0, 0.9), "color": "#1976d2", "text_color": "white"},
    {"name": "П\n10РП", "pos": (-1.1, 0.9), "color": "#f57c00"},
    {"name": "В\n10РП", "pos": (-1.1, 0.55), "color": "#f57c00"},
    {"name": "G\n10РП", "pos": (-1.1, 0.2), "color": "#f57c00"},
    {"name": "Н\n10РП", "pos": (-1.1, -0.15), "color": "#f57c00"},
    {"name": "О\n10РП", "pos": (-1.1, -0.5), "color": "#f57c00"},
    {"name": "Пр\nСОП", "pos": (1.1, 0.9), "color": "#4caf50"},
    {"name": "ОУС\nСОП", "pos": (1.1, 0.65), "color": "#4caf50"},
    {"name": "Пп\nСОП", "pos": (1.1, 0.4), "color": "#4caf50"},
    {"name": "Ппр\nСОП", "pos": (1.1, 0.15), "color": "#4caf50"},
    {"name": "Ор\nСОП", "pos": (1.1, -0.1), "color": "#4caf50"},
    {"name": "См\nСАМОП", "pos": (-0.6, -0.8), "color": "#ef5350"},
    {"name": "ПП\nСАМОП", "pos": (-0.3, -0.8), "color": "#ef5350"},
    {"name": "Кр\nСАМОП", "pos": (0.0, -0.8), "color": "#ef5350"},
    {"name": "Ав\nСАМОП", "pos": (0.3, -0.8), "color": "#ef5350"},
    {"name": "Фр\nСАМОП", "pos": (0.6, -0.8), "color": "#ef5350"},
]

pos = {node["name"]: node["pos"] for node in nodes}

edges = [
    ("ПОН\nIST", "П\n10РП", 0.77, 0.009, "positive"),
    ("ПОН\nIST", "В\n10РП", 0.64, 0.015, "positive"),
    ("ПОН\nIST", "G\n10РП", 0.73, 0.004, "positive"),
    ("ПОН\nIST", "Н\n10РП", 0.64, 0.021, "positive"),
    ("ПОН\nIST", "О\n10РП", 0.56, 0.028, "positive"),
    ("ПОН\nIST", "Пр\nСОП", 0.77, 0.007, "positive"),
    ("ПОН\nIST", "ОУС\nСОП", 0.76, 0.012, "positive"),
    ("ПОН\nIST", "Пп\nСОП", 0.71, 0.002, "positive"),
    ("ПОН\nIST", "Ппр\nСОП", 0.69, 0.016, "positive"),
    ("ПОН\nIST", "Ор\nСОП", 0.68, 0.023, "positive"),
    ("ПОН\nIST", "См\nСАМОП", 0.64, 0.011, "positive"),
    ("ПОН\nIST", "ПП\nСАМОП", 0.60, 0.034, "positive"),
    ("ПОН\nIST", "Кр\nСАМОП", 0.59, 0.030, "positive"),
    ("ПОН\nIST", "Ав\nСАМОП", 0.56, 0.026, "positive"),
    ("ПОН\nIST", "Фр\nСАМОП", 0.52, 0.037, "positive"),
    ("G\n10РП", "ПП\nСАМОП", -0.59, 0.018, "negative"),
    ("Н\n10РП", "Кр\nСАМОП", -0.52, 0.022, "negative"),
    ("О\n10РП", "Ав\nСАМОП", -0.50, 0.031, "negative"),
]


def to_px(x: float, y: float) -> tuple[float, float]:
    # Map diagram space (-1.4..1.4, -1.05..1.15) to svg coordinates
    x_min, x_max = -1.4, 1.4
    y_min, y_max = -1.05, 1.15
    px = MARGIN + (x - x_min) / (x_max - x_min) * (WIDTH - 2 * MARGIN)
    py = HEIGHT - (MARGIN + (y - y_min) / (y_max - y_min) * (HEIGHT - 2 * MARGIN))
    return px, py


def line_svg(x1, y1, x2, y2, color, width, dashed=False):
    dash = ' stroke-dasharray="10 8"' if dashed else ''
    return f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="{color}" stroke-width="{width}" stroke-linecap="round"{dash}/>'


def label_svg(x, y, text, size=16, color="#212121", weight="600"):
    lines = text.split("\n")
    dy = -(len(lines)-1)*size*0.6/2
    parts = []
    for i, line in enumerate(lines):
        parts.append(
            f'<text x="{x:.1f}" y="{y + dy + i*size*0.6:.1f}" fill="{color}" font-family="Arial" font-size="{size}" font-weight="{weight}" text-anchor="middle" dominant-baseline="middle">{line}</text>'
        )
    return "\n".join(parts)


def draw_nodes():
    pieces = []
    for node in nodes:
        x, y = to_px(*node["pos"])
        pieces.append(
            f'<circle cx="{x:.1f}" cy="{y:.1f}" r="42" fill="{node["color"]}" />'
        )
        pieces.append(label_svg(x, y, node["name"], color=node.get("text_color", "white")))
    return "\n".join(pieces)


def draw_edges():
    pieces = []
    for start, end, value, p_val, sign in edges:
        x1, y1 = to_px(*pos[start])
        x2, y2 = to_px(*pos[end])
        color = "#2e7d32" if sign == "positive" else "#d32f2f"
        dashed = sign == "negative"

        # Offsets for double line (p < 0.01)
        if p_val < 0.01:
            dx, dy = x2 - x1, y2 - y1
            length = math.hypot(dx, dy) or 1
            ox, oy = -dy / length * 6, dx / length * 6
            offsets = [(-ox, -oy), (ox, oy)]
        else:
            offsets = [(0, 0)]

        for ox, oy in offsets:
            pieces.append(line_svg(x1 + ox, y1 + oy, x2 + ox, y2 + oy, color, width=4, dashed=dashed))

        # Label in the middle
        mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2
        pieces.append(
            f'<rect x="{mid_x-20:.1f}" y="{mid_y-12:.1f}" width="40" height="24" rx="6" ry="6" fill="#ffffff" opacity="0.9" />'
        )
        pieces.append(label_svg(mid_x, mid_y, f"{abs(value):.2f}", size=14, color="#424242", weight="700"))

    return "\n".join(pieces)


def draw_legend():
    start_y = HEIGHT - 50
    spacing = 250
    items = [
        ("#2e7d32", False, "положительная p < 0,05"),
        ("#2e7d32", True, "положительная p < 0,01"),
        ("#d32f2f", False, "отрицательная p < 0,05"),
        ("#d32f2f", True, "отрицательная p < 0,01"),
    ]
    pieces = []
    for idx, (color, double, text) in enumerate(items):
        x = MARGIN + idx * spacing
        y = start_y
        if double:
            pieces.append(line_svg(x, y - 6, x + 80, y - 6, color, 4, dashed="отриц" in text))
            pieces.append(line_svg(x, y + 6, x + 80, y + 6, color, 4, dashed="отриц" in text))
        else:
            pieces.append(line_svg(x, y, x + 80, y, color, 4, dashed="отриц" in text))
        pieces.append(label_svg(x + 40, y + 22, text, size=14, color="#424242", weight="600"))
    return "\n".join(pieces)


def build_svg():
    content = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        f'<svg width="{WIDTH}" height="{HEIGHT}" viewBox="0 0 {WIDTH} {HEIGHT}" xmlns="http://www.w3.org/2000/svg">',
        '<rect width="100%" height="100%" fill="white" />',
        draw_edges(),
        draw_nodes(),
        draw_legend(),
        '</svg>',
    ]
    return "\n".join(content)


def main():
    svg = build_svg()
    Path("pleyada.svg").write_text(svg, encoding="utf-8")


if __name__ == "__main__":
    main()
