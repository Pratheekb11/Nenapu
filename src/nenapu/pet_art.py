"""The creature, drawn.

Braille cells carry 2x4 dots each, so a 40-column block of text is a 80x64
bitmap — enough resolution for something that reads as an animal rather than as
punctuation. The alternative was hand-typing fifteen lines of ⣿⣦⠿ per mood,
which is unmaintainable the moment an eye needs to move one dot to the left.

So the bear is drawn instead: a silhouette assembled from ellipses, with the
face carved back out of it. A mood is then a handful of pixel edits — eyes
shut, mouth open, brows down — rather than a separate picture, which is what
keeps nine moods from becoming nine drawings that slowly stop matching.
"""

from __future__ import annotations

from dataclasses import dataclass

# Braille dots are numbered down the left column then down the right, with the
# fourth row tacked on the end by the 8-dot extension — hence the ordering.
_DOTS = {(0, 0): 0x01, (0, 1): 0x02, (0, 2): 0x04, (0, 3): 0x40,
         (1, 0): 0x08, (1, 1): 0x10, (1, 2): 0x20, (1, 3): 0x80}

CELL_W, CELL_H = 2, 4


@dataclass
class Canvas:
    width: int
    height: int

    def __post_init__(self) -> None:
        self.px = [[0] * self.width for _ in range(self.height)]

    def set(self, x: int, y: int, on: int = 1) -> None:
        if 0 <= x < self.width and 0 <= y < self.height:
            self.px[y][x] = on

    def ellipse(self, cx: float, cy: float, rx: float, ry: float, on: int = 1) -> None:
        for y in range(int(cy - ry) - 1, int(cy + ry) + 2):
            for x in range(int(cx - rx) - 1, int(cx + rx) + 2):
                if rx <= 0 or ry <= 0:
                    continue
                if ((x - cx) / rx) ** 2 + ((y - cy) / ry) ** 2 <= 1.0:
                    self.set(x, y, on)

    def ring(self, cx: float, cy: float, rx: float, ry: float,
             thickness: float = 1.6, on: int = 1) -> None:
        self.ellipse(cx, cy, rx, ry, on)
        self.ellipse(cx, cy, rx - thickness, ry - thickness, 0 if on else 1)

    def bar(self, x0: int, y0: int, x1: int, y1: int, on: int = 1) -> None:
        for y in range(min(y0, y1), max(y0, y1) + 1):
            for x in range(min(x0, x1), max(x0, x1) + 1):
                self.set(x, y, on)

    def to_braille(self) -> list[str]:
        rows = []
        for cy in range(0, self.height, CELL_H):
            row = []
            for cx in range(0, self.width, CELL_W):
                bits = 0
                for (dx, dy), value in _DOTS.items():
                    y, x = cy + dy, cx + dx
                    if y < self.height and x < self.width and self.px[y][x]:
                        bits |= value
                row.append(chr(0x2800 + bits))
            rows.append("".join(row))
        return _crop(rows)


# 88x72 dots is 44 columns by 18 text rows. Bigger than it needs to be for a
# silhouette and not bigger than it needs to be for a face: at 38 columns the
# eyes were two dots wide and every mood looked the same.
BLANK = chr(0x2800)  # not a space: `strip()` will not touch it


def _crop(rows: list[str]) -> list[str]:
    """Trim the empty margin around the drawing.

    The canvas is sized for the shapes rather than for the result, so a bear
    that does not fill it arrives inside a wide frame of blank braille. That is
    not whitespace — `⠀` is U+2800 and `strip()` leaves it alone — so it
    silently eats terminal width and pushes the status column off the screen.
    """
    inked = [i for i, row in enumerate(rows) if set(row) != {BLANK}]
    if not inked:
        return [""]
    rows = rows[inked[0]:inked[-1] + 1]
    left = min(len(row) - len(row.lstrip(BLANK)) for row in rows)
    right = min(len(row) - len(row.rstrip(BLANK)) for row in rows)
    return [row[left:len(row) - right] for row in rows]


W, H = 88, 72
HEAD_CX, HEAD_CY = 44, 28
HEAD_RX, HEAD_RY = 19, 16


def _body(c: Canvas) -> None:
    """Ears, head, body, feet — everything that never changes with mood.

    Order is the whole trick. A silhouette drawn as one union of ellipses is a
    blob: the head merges into the body, the ears melt into the head, and the
    result reads as a lump with two holes in it. So whatever sits in front
    carves a slightly larger hole before filling its own shape. The gap is what
    makes the parts legible, exactly as it is in a paper cut-out.
    """
    c.ellipse(HEAD_CX - 17, 56, 5, 7)                # arms, behind the body
    c.ellipse(HEAD_CX + 17, 56, 5, 7)
    c.ellipse(HEAD_CX - 10, 68, 7, 4)                # feet
    c.ellipse(HEAD_CX + 10, 68, 7, 4)

    c.ellipse(HEAD_CX, 56, 15, 12, 0)                # body clears its gap
    c.ellipse(HEAD_CX, 56, 13.5, 10.5)
    c.ellipse(HEAD_CX, 59, 5.5, 4.5, 0)              # a paler belly

    c.ellipse(HEAD_CX - 14, HEAD_CY - 18, 7.5, 7)    # ears, behind the head
    c.ellipse(HEAD_CX + 14, HEAD_CY - 18, 7.5, 7)
    c.ellipse(HEAD_CX - 14, HEAD_CY - 20, 3.2, 3, 0)   # inner ear
    c.ellipse(HEAD_CX + 14, HEAD_CY - 20, 3.2, 3, 0)

    c.ellipse(HEAD_CX, HEAD_CY, HEAD_RX + 1.2, HEAD_RY + 1.2, 0)
    c.ellipse(HEAD_CX, HEAD_CY, HEAD_RX, HEAD_RY)


def _muzzle(c: Canvas, *, open_mouth: bool = False, smile: bool = True) -> None:
    """Snout patch carved out of the head, with the nose filled back in."""
    c.ellipse(HEAD_CX, HEAD_CY + 7, 8.5, 5.5, 0)
    c.ellipse(HEAD_CX, HEAD_CY + 4, 3.2, 2.2)
    if open_mouth:
        c.ellipse(HEAD_CX, HEAD_CY + 9, 3.0, 2.6)
        return
    for dx in range(-4, 5):
        curve = abs(dx) // 2
        y = HEAD_CY + 9 + (curve if smile else -curve)
        c.set(HEAD_CX + dx, y)
        c.set(HEAD_CX + dx, y + 1)


def _eyes(c: Canvas, style: str) -> None:
    """Whites carved out of the head, pupils filled back in.

    Two levels rather than one: a plain hole reads as a hole, and a bear with
    two holes in its face is not looking at anything.
    """
    left, right, y = HEAD_CX - 8, HEAD_CX + 8, HEAD_CY - 4
    if style == "closed":
        for cx in (left, right):
            for dx in range(-4, 5):
                yy = y + (1 if abs(dx) > 2 else 0)
                c.set(cx + dx, yy, 0)
                c.set(cx + dx, yy + 1, 0)
        return
    if style == "cross":
        for cx in (left, right):
            for d in range(-4, 5):
                for t in (0, 1):
                    c.set(cx + d + t, y + d, 0)
                    c.set(cx + d + t, y - d, 0)
        return
    white = {"wide": 4.2, "normal": 3.4, "narrow": 3.4}[style]
    for cx in (left, right):
        c.ellipse(cx, y, white, white * (0.4 if style == "narrow" else 1.0), 0)
        c.ellipse(cx, y, 1.7, 1.7 if style != "narrow" else 1.0)


def _brows(c: Canvas, angle: str) -> None:
    """Worry is mostly eyebrows. Without them every mood is mildly surprised."""
    if angle == "none":
        return
    for side, cx in ((-1, HEAD_CX - 8), (1, HEAD_CX + 8)):
        for dx in range(-4, 5):
            drop = int(dx * side * 0.7) if angle == "down" else int(-dx * side * 0.7)
            c.set(cx + dx, HEAD_CY - 11 + drop, 0)
            c.set(cx + dx, HEAD_CY - 10 + drop, 0)


# Each mood is the same bear with a different face. Nothing here draws a new
# animal, which is what stops the nine of them drifting apart.
FACE_BY_MOOD = {
    "sick":      dict(eyes="cross",  brows="down", mouth="open",  smile=False),
    "spooked":   dict(eyes="wide",   brows="up",   mouth="open",  smile=False),
    "hungry":    dict(eyes="normal", brows="up",   mouth="open",  smile=False),
    "drowsy":    dict(eyes="closed", brows="none", mouth="line",  smile=True),
    "restless":  dict(eyes="narrow", brows="up",   mouth="line",  smile=False),
    "stuffed":   dict(eyes="closed", brows="none", mouth="line",  smile=True),
    "content":   dict(eyes="normal", brows="none", mouth="line",  smile=True),
    "delighted": dict(eyes="closed", brows="up",   mouth="line",  smile=True),
    "new":       dict(eyes="normal", brows="none", mouth="line",  smile=True),
}


def draw(mood: str, *, blink: bool = False) -> list[str]:
    """The bear, as braille rows. `blink` shuts whatever eyes are open."""
    face = FACE_BY_MOOD.get(mood, FACE_BY_MOOD["content"])
    canvas = Canvas(W, H)
    _body(canvas)
    _muzzle(canvas, open_mouth=face["mouth"] == "open", smile=face["smile"])
    _eyes(canvas, "closed" if (blink and face["eyes"] not in ("closed", "cross"))
          else face["eyes"])
    _brows(canvas, face["brows"])
    return canvas.to_braille()


# Moods with something wrong get their own colour rather than the theme's: the
# whole point of the creature is that a bad store cannot look like a good one,
# and a teal bear with its eyes crossed still reads as fine at a glance.
MOOD_SHADES = {
    "sick": ["#FCA5A5", "#F87171", "#EF4444", "#DC2626", "#B91C1C", "#991B1B"],
    "spooked": ["#FDE68A", "#FCD34D", "#FBBF24", "#F59E0B", "#D97706", "#B45309"],
    "hungry": ["#FDE68A", "#FCD34D", "#FBBF24", "#F59E0B", "#D97706", "#B45309"],
}

ACCENTS = {
    "drowsy": [(2, "z"), (4, "z z"), (6, "z z z")],
    "stuffed": [(4, "z z")],
    "hungry": [(6, "...")],
    "spooked": [(1, "!")],
    "sick": [(1, "?")],
}


def coloured(mood: str, shades: list[str], *, blink: bool = False) -> list[str]:
    """Rows as Rich markup, with the gradient running down the body.

    Two rows per shade rather than a per-row interpolation: the bear is
    sixteen rows tall and a gradient that fine just looks like noise on a
    terminal that quantises colour.
    """
    palette = MOOD_SHADES.get(mood, shades)
    rows = draw(mood, blink=blink)
    accents = dict((row, text) for row, text in ACCENTS.get(mood, []))

    out = []
    for i, row in enumerate(rows):
        colour = palette[min(i * len(palette) // max(len(rows), 1), len(palette) - 1)]
        suffix = f"  [dim]{accents[i]}[/]" if i in accents else ""
        out.append(f"[{colour}]{row}[/]{suffix}")
    return out
