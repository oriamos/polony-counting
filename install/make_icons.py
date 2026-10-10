"""Generate the Polony Review icons (needs Pillow). Run: python install/make_icons.py

Writes install/assets/PolonyReview.png (512 px), PolonyReview.ico (multi-size) and PolonyReview.icns.
The design is original: a dark rounded square with clusters of red, green and yellow dots.
"""
from __future__ import annotations

import random
from pathlib import Path

from PIL import Image, ImageDraw

OUT = Path(__file__).resolve().parent / "assets"
BG = (28, 33, 43, 255)
RED, GREEN, YELLOW = (229, 72, 77, 255), (48, 164, 108, 255), (245, 197, 24, 255)
S = 1024  # master size (supersampled, then downscaled)


def master() -> Image.Image:
    im = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    d.rounded_rectangle((32, 32, S - 32, S - 32), radius=220, fill=BG)
    d.rounded_rectangle((32, 32, S - 32, S - 32), radius=220, outline=(70, 80, 98, 255), width=10)
    rnd = random.Random(7)  # deterministic scatter
    clusters = [((370, 400), RED, 150), ((660, 400), GREEN, 150), ((515, 650), YELLOW, 150)]
    for (cx, cy), col, spread in clusters:
        pts: list[tuple[float, float]] = []
        while len(pts) < 7:
            x, y = cx + rnd.uniform(-spread, spread), cy + rnd.uniform(-spread, spread)
            if (x - cx) ** 2 + (y - cy) ** 2 > spread**2:
                continue
            if all((x - a) ** 2 + (y - b) ** 2 > 78**2 for a, b in pts):
                pts.append((x, y))
        for x, y in pts:
            r = rnd.choice((34, 38, 42))
            d.ellipse((x - r, y - r, x + r, y + r), fill=col)
    return im


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    m = master()
    m.resize((512, 512), Image.LANCZOS).save(OUT / "PolonyReview.png")
    sizes = [16, 24, 32, 48, 64, 128, 256]
    m.save(OUT / "PolonyReview.ico", format="ICO", sizes=[(s, s) for s in sizes])
    m.save(OUT / "PolonyReview.icns", format="ICNS")
    for f in sorted(OUT.iterdir()):
        print(f"{f.name}: {f.stat().st_size} bytes")


if __name__ == "__main__":
    main()
