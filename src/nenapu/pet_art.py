"""The creature, drawn.

This is the fourth composition and the second technique. The first three were
braille bitmaps — a filled silhouette, then chibi proportions, then outlines
with the eyes as the only filled shapes. Braille packs 2x4 dots per cell, which
sounds like plenty of resolution and is not: every curve lands on a different
dot pattern, so a smooth outline arrives as a row of unrelated glyphs and the
whole thing reads as something photocopied badly. The verdicts were "looks
pirated" and "still ugly", and both were right.

Line art in ordinary characters has far less resolution and looks better,
because the characters were drawn by a type designer. A `/` is a clean diagonal
at any size; the braille approximation of the same diagonal is a staircase of
`⣠⣴⣾`. So the dog is set out of pieces that are already well drawn, and the
only thing generated is the arithmetic that keeps the frame aligned.

Every row is built by placing characters at computed columns rather than typed
as a literal, because a hand-typed frame drifts the moment one mood needs a
wider mouth than another — and a drawing whose right edge wobbles by a column
is exactly what looked cheap about the earlier attempts.
"""

from __future__ import annotations

WIDTH = 34                 # the frame at its smallest
LEFT, RIGHT = 2, 31        # the columns the head's sides sit in
EAR = 4                    # inset of the ears from the frame edge


def _blank(width: int) -> list[str]:
    return [" "] * width


def _place(cells: list[str], at: int, text: str) -> list[str]:
    for i, char in enumerate(text):
        if 0 <= at + i < len(cells):
            cells[at + i] = char
    return cells


class Frame:
    """The head's geometry at a given width.

    Widening is the only way this drawing grows: it is seven rows tall by
    construction, and a taller one would mean a second set of hand-set pieces
    that slowly stopped matching the first. Rows the dog cannot use are better
    spent on what the store has learned anyway.
    """

    def __init__(self, width: int = WIDTH) -> None:
        self.width = max(WIDTH, width)
        self.left = LEFT
        self.right = self.width - (WIDTH - RIGHT)
        self.span = self.right - self.left - 1

    def blank(self) -> list[str]:
        return _blank(self.width)

    def walls(self, cells: list[str]) -> list[str]:
        return _place(_place(cells, self.left, "|"), self.right, "|")

    def centred(self, text: str) -> list[str]:
        start = self.left + 1 + (self.span - len(text)) // 2
        return self.walls(_place(self.blank(), start, text))


# Each mood is the same dog with a different face: two eyes, a mouth, and
# eyebrows when something is wrong. Nothing here draws a second animal, which
# is what stops nine moods from drifting apart.
FACE_BY_MOOD = {
    "sick":      dict(eye="✕", mouth="(··)", brow="__"),
    "spooked":   dict(eye="◉", mouth=" OO ", brow="''"),
    "hungry":    dict(eye="●", mouth=" ᗢ  ", brow=None),
    "drowsy":    dict(eye="˘", mouth=" ‿‿ ", brow=None),
    "restless":  dict(eye="˙", mouth=" ~~ ", brow=None),
    "stuffed":   dict(eye="─", mouth=" ‿‿ ", brow=None),
    "content":   dict(eye="●", mouth="\\__/", brow=None),
    "delighted": dict(eye="^", mouth="\\ᵕ/ ", brow=None),
    "new":       dict(eye="·", mouth=" ·· ", brow=None),
}

# Moods that are already squinting or crossed keep their face when blinking: a
# sleeping dog that flickers reads as a glitch rather than as a dog.
NO_BLINK = {"drowsy", "stuffed", "sick", "delighted"}


def draw(mood: str, *, blink: bool = False, scale: float = 1.0) -> list[str]:
    """The dog, as rows of text.

    `scale` widens the frame; the height is fixed at seven rows. `blink` shuts
    whatever eyes were open.
    """
    face = dict(FACE_BY_MOOD.get(mood, FACE_BY_MOOD["content"]))
    if blink and mood not in NO_BLINK:
        face["eye"] = "─"
    f = Frame(round(WIDTH * scale))
    eye, mouth, brow = face["eye"], face["mouth"], face["brow"]
    gap = " " * max(6, f.span // 4)

    rows = [
        _place(_place(f.blank(), EAR, ",__,"), f.width - EAR - 4, ",__,"),
        _place(f.blank(), EAR - 1,
               "/    \\" + "_" * (f.width - 2 * EAR - 10) + "/    \\"),
        f.centred(f"{brow}{gap}{brow}") if brow else f.walls(f.blank()),
        f.centred(f"{eye}{gap}{eye}"),
        f.centred("▾"),
        _place(_place(_place(f.blank(), f.left + 1, "\\"), f.right - 1, "/"),
               f.left + 1 + (f.span - len(mouth)) // 2, mouth),
        _place(f.blank(), f.left + 1, "'-." + "_" * (f.span - 6) + ".-'"),
    ]
    return ["".join(row).rstrip() for row in rows]


# Moods with something wrong get their own colour rather than the theme's: the
# point of the creature is that a bad store cannot look like a good one, and a
# calm teal dog with its eyes crossed still reads as fine at a glance.
MOOD_SHADES = {
    "sick": ["#FCA5A5", "#F87171", "#EF4444", "#DC2626", "#B91C1C", "#991B1B"],
    "spooked": ["#FDE68A", "#FCD34D", "#FBBF24", "#F59E0B", "#D97706", "#B45309"],
    "hungry": ["#FDE68A", "#FCD34D", "#FBBF24", "#F59E0B", "#D97706", "#B45309"],
}

ACCENTS = {
    "drowsy": {0: "z", 1: "z z"},
    "stuffed": {1: "z z"},
    "hungry": {5: "..."},
    "spooked": {0: "!"},
    "sick": {0: "?"},
}


def coloured(mood: str, shades: list[str], *, blink: bool = False,
             scale: float = 1.0) -> list[str]:
    """Rows as Rich markup, with the gradient running down the head.

    The rows are escaped, and that is not a precaution. The ear row ends in a
    backslash, and in Rich markup a trailing backslash escapes whatever comes
    next — so the closing tag was being swallowed and a literal `[/]` printed
    itself at the start of the following line, shoving the whole drawing one
    column sideways.
    """
    from rich.markup import escape

    palette = MOOD_SHADES.get(mood, shades)
    rows = draw(mood, blink=blink, scale=scale)
    accents = ACCENTS.get(mood, {})

    out = []
    for i, row in enumerate(rows):
        colour = palette[min(i * len(palette) // max(len(rows), 1), len(palette) - 1)]
        suffix = f"  [dim]{accents[i]}[/]" if i in accents else ""
        out.append(f"[{colour}]{escape(row)}[/]{suffix}")
    return out
