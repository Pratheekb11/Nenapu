"""Render the pet and bake it down to braille.

Why this exists, and why it is a build-time tool rather than runtime code:

Four hand-drawn versions came before this — three braille bitmaps drawn dot by
dot, then line art set in ordinary characters — and every one of them looked
homemade. The lesson is that art which reads as a logo is not drawn at the
resolution it will be displayed at. It is drawn large, by someone who can draw,
and then reduced: the reducing step averages and thresholds, and that is what
makes edges come out smooth instead of as staircases.

So the shape comes from the Noto Emoji dog face (SIL Open Font License 1.1,
© Google), rendered at 400px, adjusted per mood, then reduced to a braille dot
grid. The result is baked into `pet_art.py` as text, so Pillow and the font are
needed only to regenerate the art, never to run Nenapu.

    uv run --with pillow python tools/render_pet.py --font <NotoEmoji.ttf> --write
"""

from __future__ import annotations

import argparse
import sys

from PIL import Image, ImageDraw, ImageFont

GLYPH = "\U0001F436"        # 🐶 dog face
RENDER_PX = 400

_DOTS = {(0, 0): 0x01, (0, 1): 0x02, (0, 2): 0x04, (0, 3): 0x40,
         (1, 0): 0x08, (1, 1): 0x10, (1, 2): 0x20, (1, 3): 0x80}
BLANK = chr(0x2800)

# Where the features sit on the glyph, as fractions of its bounding box. Found
# by rendering it and reading off the grid rather than guessed.
EYE_Y, EYE_DX, EYE_R = 0.52, 0.155, 0.075
MOUTH = (0.5, 0.70)

# Each mood is the same face with a different expression. The head, ears and
# snout are never touched — that is the part worth having from a font.
MOODS = {
    "content":   dict(eye="round",  mouth="smile"),
    "delighted": dict(eye="happy",  mouth="open"),
    "new":       dict(eye="round",  mouth="smile"),
    "drowsy":    dict(eye="closed", mouth="flat"),
    "stuffed":   dict(eye="closed", mouth="flat"),
    "restless":  dict(eye="small",  mouth="flat"),
    "hungry":    dict(eye="round",  mouth="open"),
    "spooked":   dict(eye="wide",   mouth="open"),
    "sick":      dict(eye="cross",  mouth="wavy"),
}


def base(font_path: str) -> Image.Image:
    font = ImageFont.truetype(font_path, RENDER_PX)
    img = Image.new("L", (RENDER_PX * 2, RENDER_PX * 2), 0)
    ImageDraw.Draw(img).text((RENDER_PX // 4, RENDER_PX // 8), GLYPH, font=font, fill=255)
    return img.crop(img.getbbox())


def expression(img: Image.Image, eye: str, mouth: str) -> Image.Image:
    img = img.copy()
    d = ImageDraw.Draw(img)
    w, h = img.size

    def blob(cx, cy, rx, ry, fill):
        d.ellipse([cx - rx, cy - ry, cx + rx, cy + ry], fill=fill)

    ex_r, ey = EYE_R * w, EYE_Y * h
    for side in (-1, 1):
        ex = w / 2 + side * EYE_DX * w
        blob(ex, ey, ex_r * 1.9, ex_r * 2.1, 0)          # wipe the eye
        if eye == "round":
            blob(ex, ey, ex_r, ex_r * 1.15, 255)
        elif eye == "wide":
            blob(ex, ey, ex_r * 1.35, ex_r * 1.5, 255)
            blob(ex, ey, ex_r * 0.5, ex_r * 0.6, 0)
        elif eye == "small":
            blob(ex, ey, ex_r * 0.55, ex_r * 0.65, 255)
        elif eye == "happy":
            d.arc([ex - ex_r * 1.3, ey - ex_r * 1.4, ex + ex_r * 1.3, ey + ex_r],
                  200, 340, fill=255, width=int(ex_r * 0.55))
        elif eye == "closed":
            d.arc([ex - ex_r * 1.3, ey - ex_r, ex + ex_r * 1.3, ey + ex_r * 1.4],
                  20, 160, fill=255, width=int(ex_r * 0.55))
        elif eye == "cross":
            for sign in (-1, 1):
                d.line([ex - ex_r, ey + sign * ex_r, ex + ex_r, ey - sign * ex_r],
                       fill=255, width=int(ex_r * 0.5))

    mx, my = MOUTH[0] * w, MOUTH[1] * h
    span = ex_r * 1.5
    if mouth == "open":
        blob(mx, my + span * 0.5, span * 0.7, span * 0.8, 255)
    elif mouth == "flat":
        d.line([mx - span * 0.7, my + span * 0.6, mx + span * 0.7, my + span * 0.6],
               fill=255, width=int(ex_r * 0.4))
    elif mouth == "wavy":
        step = span / 3
        for i in range(3):
            x0 = mx - span * 0.8 + i * step * 1.6
            d.arc([x0, my + span * 0.2, x0 + step * 1.6, my + span * 1.0],
                  0 if i % 2 else 180, 180 if i % 2 else 360,
                  fill=255, width=int(ex_r * 0.35))
    return img


def braille(img: Image.Image, cols: int, threshold: int = 110) -> list[str]:
    """Reduce to a dot grid and pack 2x4 dots into each cell."""
    w = cols * 2
    h = max(4, round(w * img.size[1] / img.size[0] / 4) * 4)
    px = img.resize((w, h), Image.LANCZOS).load()
    rows = []
    for cy in range(0, h, 4):
        row = []
        for cx in range(0, w, 2):
            value = 0
            for (dx, dy), bit in _DOTS.items():
                if px[cx + dx, cy + dy] > threshold:
                    value |= bit
            row.append(chr(0x2800 + value))
        rows.append("".join(row))
    rows = [r for r in rows if set(r) != {BLANK}]
    left = min(len(r) - len(r.lstrip(BLANK)) for r in rows)
    right = min(len(r) - len(r.rstrip(BLANK)) for r in rows)
    return [r[left:len(r) - right] for r in rows]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--font", required=True, help="NotoEmoji-Regular.ttf")
    ap.add_argument("--cols", type=int, nargs="+", default=[26, 34, 44])
    ap.add_argument("--mood", default=None, help="preview one mood and exit")
    args = ap.parse_args()

    source = base(args.font)
    if args.mood:
        face = MOODS[args.mood]
        print("\n".join(braille(expression(source, **face), args.cols[0])))
        return 0

    print("# Generated by tools/render_pet.py — do not edit by hand.")
    print("ART: dict[int, dict[str, list[str]]] = {")
    for cols in args.cols:
        print(f"    {cols}: {{")
        for mood, face in MOODS.items():
            rows = braille(expression(source, **face), cols)
            print(f'        "{mood}": [')
            for row in rows:
                print(f'            "{row}",')
            print("        ],")
        print("    },")
    print("}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
