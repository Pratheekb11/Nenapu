"""The creature, drawn.

Braille cells carry 2x4 dots each, so a 40-column block of text is a 80x64
bitmap — enough resolution for something that reads as an animal rather than as
punctuation. The alternative was hand-typing fifteen lines of ⣿⣦⠿ per mood,
which is unmaintainable the moment an eye needs to move one dot to the left.

So the dog is drawn instead: outlines and arcs, with the eyes as the only
large filled shapes on it. A mood is then a handful of pixel edits — eyes shut,
mouth open, brows down, tongue out — rather than a separate picture, which is
what keeps nine moods from becoming nine drawings that slowly stop matching.
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

    def arc(self, cx: float, cy: float, rx: float, ry: float,
            start: float, end: float, thickness: float = 1.2, on: int = 1) -> None:
        """A stroked piece of an ellipse, in radians clockwise from 3 o'clock.

        Line art needs open curves — a smile, a tail, the top of a paw — and an
        outline is the wrong tool for those because it always closes.
        """
        import math

        steps = max(12, int((end - start) * max(rx, ry)))
        for i in range(steps + 1):
            angle = start + (end - start) * i / steps
            self.ellipse(cx + rx * math.cos(angle), cy + ry * math.sin(angle),
                         thickness, thickness, on)

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


BLANK = chr(0x2800)  # not a space: `strip()` will not touch it


def _crop(rows: list[str]) -> list[str]:
    """Trim the empty margin around the drawing.

    The canvas is sized for the shapes rather than for the result, so a dog
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


# 84x58 dots is 42 columns by 15 text rows at scale 1.0, and the drawing grows
# from there. The composition went through a filled silhouette and a head on a
# body before this one; both failed the same way, with parts competing for a
# small canvas until every feature turned to mush. So there is no body. The
# face fills the frame, which is the pose every cute animal drawing uses and
# the reason it works: the eyes get to be enormous and nothing else is asking
# for room.
W, H = 84, 58
STROKE = 1.1


class Geom:
    """Every measurement in the drawing, at a given scale.

    The dog is drawn from numbers rather than typed, so a bigger one is a
    multiplication rather than a second picture. A terminal with forty rows to
    spare should not be looking at the same postage stamp as one with twenty.
    """

    def __init__(self, scale: float = 1.0) -> None:
        self.scale = scale
        self.width, self.height = round(W * scale), round(H * scale)
        self.cx, self.cy = self.width / 2, 27 * scale
        self.rx, self.ry = 25 * scale, 21 * scale
        # Strokes thicken with the drawing but not proportionally: a hairline
        # disappears when everything around it grows, and a scaled-up outline
        # turns into a tube.
        self.stroke = STROKE * max(1.0, scale * 0.8)

    def s(self, value: float) -> float:
        return value * self.scale


def _body(c: Canvas, g: Geom) -> None:
    """Ears and head. Everything that never changes with mood.

    Ears hang off the top corners, filled, and they are the only heavy shapes
    besides the eyes: outlined ears made the whole thing read as a balloon.
    """
    for side in (-1, 1):
        c.ellipse(g.cx + side * g.s(23), g.cy - g.s(2), g.s(9), g.s(15), 0)
        c.ellipse(g.cx + side * g.s(23), g.cy - g.s(2), g.s(8), g.s(14))

    c.ellipse(g.cx, g.cy, g.rx + g.s(1.6), g.ry + g.s(1.6), 0)
    c.ring(g.cx, g.cy, g.rx, g.ry, g.stroke + g.s(0.5))


def _muzzle(c: Canvas, g: Geom, *, open_mouth: bool = False,
            smile: bool = True) -> None:
    """A tiny nose set low, and a mouth of two small arcs.

    Small is the whole point. A long muzzle is what makes a drawn dog look like
    an animal rather than like a puppy.
    """
    import math

    c.ellipse(g.cx, g.cy + g.s(9), g.s(3.2), g.s(2.4))              # nose
    c.bar(round(g.cx), round(g.cy + g.s(11)),
          round(g.cx), round(g.cy + g.s(13)))                       # philtrum

    if open_mouth:
        c.ellipse(g.cx, g.cy + g.s(15), g.s(3.6), g.s(3.2), 0)
        c.ring(g.cx, g.cy + g.s(15), g.s(3.6), g.s(3.2), g.stroke)
        return
    if smile:
        for side in (-1, 1):
            c.arc(g.cx + side * g.s(4), g.cy + g.s(11), g.s(4), g.s(3.2),
                  0.15, math.pi - 0.15, g.stroke)
        c.ellipse(g.cx, g.cy + g.s(17), g.s(2.6), g.s(2.8))         # tongue
    else:
        for side in (-1, 1):
            c.arc(g.cx + side * g.s(4), g.cy + g.s(16), g.s(4), g.s(3.2),
                  math.pi + 0.15, 2 * math.pi - 0.15, g.stroke)


def _eyes(c: Canvas, g: Geom, style: str) -> None:
    """Big, filled, low on the face, with a catchlight beside the pupil.

    They are the only large dark shapes on an empty face, which is what makes
    it read as looking back rather than as a circle with marks in it.
    """
    import math

    left, right, y = g.cx - g.s(11), g.cx + g.s(11), g.cy + g.s(2)
    if style == "closed":                                           # happy arcs
        for cx in (left, right):
            c.arc(cx, y - g.s(2), g.s(5.5), g.s(4.5), 0.2, math.pi - 0.2,
                  g.stroke + g.s(0.2))
        return
    if style == "cross":
        span = round(g.s(5))
        for cx in (left, right):
            for d in range(-span, span + 1):
                for t in range(max(1, round(g.scale))):
                    c.set(round(cx) + d + t, round(y + d))
                    c.set(round(cx) + d + t, round(y - d))
        return
    rx = g.s({"wide": 7.0, "normal": 6.0, "narrow": 6.0}[style])
    ry = rx * (0.28 if style == "narrow" else 1.2)
    for cx in (left, right):
        c.ellipse(cx, y, rx, ry)
    # The catchlight goes beside the eye, not inside it. Punched out of the
    # fill it reads as a crack: braille has no grey to soften a hole with, so
    # a bite out of the eye is simply a chunk missing.
    if style != "narrow":
        for cx in (left, right):
            c.ellipse(cx - rx * 0.55, y - ry * 0.75, g.s(1.4), g.s(1.4), 0)


def _brows(c: Canvas, g: Geom, angle: str) -> None:
    """Worry is mostly eyebrows. Without them every mood is mildly surprised."""
    if angle == "none":
        return
    span = round(g.s(5))
    for side, cx in ((-1, g.cx - g.s(11)), (1, g.cx + g.s(11))):
        for dx in range(-span, span + 1):
            drop = round(dx * side * 0.7) * (1 if angle == "down" else -1)
            for t in range(max(2, round(g.s(2)))):
                c.set(round(cx) + dx, round(g.cy - g.s(11)) + drop + t)


# Each mood is the same dog with a different face. Nothing here draws a new
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


def draw(mood: str, *, blink: bool = False, scale: float = 1.0) -> list[str]:
    """The dog, as braille rows.

    `blink` shuts whatever eyes are open. `scale` grows the whole drawing —
    every measurement is a multiplication of one, which is the point of having
    drawn it rather than typed it.
    """
    face = FACE_BY_MOOD.get(mood, FACE_BY_MOOD["content"])
    g = Geom(scale)
    canvas = Canvas(g.width, g.height)
    _body(canvas, g)
    _muzzle(canvas, g, open_mouth=face["mouth"] == "open", smile=face["smile"])
    _eyes(canvas, g, "closed" if (blink and face["eyes"] not in ("closed", "cross"))
          else face["eyes"])
    _brows(canvas, g, face["brows"])
    return canvas.to_braille()


# Moods with something wrong get their own colour rather than the theme's: the
# whole point of the creature is that a bad store cannot look like a good one,
# and a teal dog with its eyes crossed still reads as fine at a glance.
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


def coloured(mood: str, shades: list[str], *, blink: bool = False,
             scale: float = 1.0) -> list[str]:
    """Rows as Rich markup, with the gradient running down the body.

    Two rows per shade rather than a per-row interpolation: the dog is
    eighteen rows tall and a gradient that fine just looks like noise on a
    terminal that quantises colour.
    """
    palette = MOOD_SHADES.get(mood, shades)
    rows = draw(mood, blink=blink, scale=scale)
    accents = dict((row, text) for row, text in ACCENTS.get(mood, []))

    out = []
    for i, row in enumerate(rows):
        colour = palette[min(i * len(palette) // max(len(rows), 1), len(palette) - 1)]
        suffix = f"  [dim]{accents[i]}[/]" if i in accents else ""
        out.append(f"[{colour}]{row}[/]{suffix}")
    return out
