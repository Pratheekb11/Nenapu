"""The mark, and the neofetch-style panel it sits in.

Two sizes, because the banner shows on every invocation:

* the **panel** — dog plus a readout of the store — for a bare `nenapu`,
  `nenapu version`, and the first run. Someone typing the name alone is
  looking around, and this is the view worth looking at.
* the **stamp** — one dim line — everywhere else. Art that repeats before
  every `search` stops being charming by the third command.

The hero wordmark is the identity: big, warm, unmistakable at a glance, in the
shape every serious CLI uses for its opening screen. Earlier attempts at a
figurative mark — a knot, then a dog — read as a bent pipe and a jolly roger
respectively. Type at scale survives a terminal; small illustrations do not.
"""

from __future__ import annotations

import sqlite3

# 30 columns wide: with the info column and padding this fits an 80-column
# terminal, which is still the width to design for.
# Hero wordmark. Big enough to be the first thing you see, 54 columns so it
# fits an 80-column terminal without wrapping.
HERO = [
    " ███╗   ██╗███████╗███╗   ██╗ █████╗ ██████╗ ██╗   ██╗",
    " ████╗  ██║██╔════╝████╗  ██║██╔══██╗██╔══██╗██║   ██║",
    " ██╔██╗ ██║█████╗  ██╔██╗ ██║███████║██████╔╝██║   ██║",
    " ██║╚██╗██║██╔══╝  ██║╚██╗██║██╔══██║██╔═══╝ ██║   ██║",
    " ██║ ╚████║███████╗██║ ╚████║██║  ██║██║     ╚██████╔╝",
    " ╚═╝  ╚═══╝╚══════╝╚═╝  ╚═══╝╚═╝  ╚═╝╚═╝      ╚═════╝ ",
]

# Top-to-bottom gradient, so the mark has depth rather than reading as a flat
# wall of blocks. Teal by default: cool tones suit a tool about verification,
# and it stays clear of the amber every other agent CLI has settled on.
THEMES: dict[str, list[str]] = {
    "teal": ["#5EEAD4", "#2DD4BF", "#14B8A6", "#0D9488", "#0F766E", "#115E59"],
    "violet": ["#8B5CF6", "#9B5DE5", "#A855F7", "#C026D3", "#D946A6", "#E0559B"],
    "indigo": ["#818CF8", "#6366F1", "#4F73E8", "#3B82F6", "#38BDF8", "#22D3EE"],
    "jade": ["#A7F3D0", "#6EE7B7", "#34D399", "#10B981", "#84CC16", "#A3E635"],
    "mono": ["white", "white", "bright_black", "bright_black", "bright_black",
             "bright_black"],
}
DEFAULT_THEME = "teal"


def hero_shades() -> list[str]:
    import os

    return THEMES.get(os.environ.get("NENAPU_THEME", DEFAULT_THEME).lower(),
                      THEMES[DEFAULT_THEME])


HERO_SHADES = THEMES[DEFAULT_THEME]

# Same wordmark for terminals without box drawing.
HERO_ASCII = [
    " _  _ ___ _  _   _   ___ _   _ ",
    "| \\| | __| \\| | /_\\ | _ \\ | | |",
    "| .` | _|| .` |/ _ \\|  _/ |_| |",
    "|_|\\_|___|_|\\_/_/ \\_\\_|  \\___/ ",
]

# Below this the wordmark wraps and looks broken, so it is dropped for the
# one-line stamp instead.
MIN_PANEL_WIDTH = 58

STAMP = "ᐡ•ᴥ•ᐡ"

# Three lines, because a tool that opens with a logo and no explanation makes
# the reader go looking for docs. Says what it stores, what makes it different,
# and what it will not do behind your back.
EXPLAINER = [
    "Facts carry provenance and decay on a clock; a shell check can prove one.",
    "Falsify a fact and everything derived from it is flagged, not silently kept.",
    "Recalls are graded by what happened next. Checks never run unapproved.",
]
TAGLINE = "memory that knows what it rests on"


def _unicode(console) -> bool:
    encoding = (getattr(console.file, "encoding", "") or "").lower()
    return "utf" in encoding


def stamp(version: str = "") -> str:
    """One line, for routine commands."""
    return f"[dim]{STAMP}  nenapu{(' ' + version) if version else ''}[/]"


def _store_facts(conn: sqlite3.Connection | None) -> list[tuple[str, str]]:
    """Read the store cheaply. A banner must never be the slow part of a run."""
    if conn is None:
        return []
    try:
        rows = dict(conn.execute(
            "SELECT status, COUNT(*) c FROM facts GROUP BY status").fetchall())
        edges = conn.execute("SELECT COUNT(*) c FROM fact_edges").fetchone()["c"]
        inferred = conn.execute(
            "SELECT COUNT(*) c FROM fact_edges WHERE source='inferred'").fetchone()["c"]
        pending = conn.execute(
            "SELECT COUNT(*) c FROM recalls WHERE outcome='pending'").fetchone()["c"]
        graded = conn.execute(
            "SELECT COUNT(*) c FROM recalls WHERE outcome IN ('good','bad')").fetchone()["c"]
        awaiting = conn.execute(
            "SELECT COUNT(*) c FROM facts WHERE verify_cmd IS NOT NULL"
            " AND verify_status='blocked'").fetchone()["c"]
    except sqlite3.Error:
        return []

    def n(key: str) -> int:
        return rows.get(key, 0)

    lines = [("facts", f"{n('active')} active"
                       + (f" · {n('suspect')} suspect" if n("suspect") else "")
                       + (f" · {n('disputed')} disputed" if n("disputed") else ""))]
    if edges:
        lines.append(("graph", f"{edges} edges"
                               + (f" · {inferred} inferred" if inferred else "")))
    if graded or pending:
        lines.append(("recalls", f"{graded} graded"
                                 + (f" · {pending} pending" if pending else "")))
    if awaiting:
        lines.append(("checks", f"[yellow]{awaiting} awaiting approval[/]"))

    try:
        skills = conn.execute(
            "SELECT status, COUNT(*) c FROM skills GROUP BY status").fetchall()
    except sqlite3.Error:
        skills = []
    if skills:
        counts = {r["status"]: r["c"] for r in skills}
        lines.append(("skills", f"{counts.get('active', 0)} active"
                                + (f" · {counts['quarantined']} quarantined"
                                   if counts.get("quarantined") else "")))
    return lines


def _tool_surface() -> tuple[str, str]:
    """What an MCP client would see, without importing the server for it.

    The banner should answer "what can this do for my agent right now", which
    is the profile and the tool count — not a list nobody reads.
    """
    import os

    from .mcp_server import TOOL_PROFILES

    profile = os.environ.get("NENAPU_TOOLS", "full").lower()
    names = TOOL_PROFILES.get(profile, TOOL_PROFILES["full"])
    return "tools", f"{len(names)} over MCP [dim]({profile})[/]"


def panel(console, *, version: str = "", conn: sqlite3.Connection | None = None,
          path: str = "", backend: str = ""):
    """The hero view: wordmark, then a two-column readout of the store."""
    from rich.table import Table
    from rich.text import Text

    unicode = _unicode(console)
    shades = hero_shades()
    art = HERO if unicode else HERO_ASCII

    body = Table.grid()
    body.add_column(no_wrap=True)

    if console.width >= MIN_PANEL_WIDTH:
        for i, line in enumerate(art):
            body.add_row(Text(line, style=shades[i % len(shades)]))
        body.add_row("")

    tagline = ("[bold]ನೆನಪು[/]  [dim]·[/]  " if unicode else "") + f"[dim]{TAGLINE}[/]"
    body.add_row(f"  {tagline}")
    body.add_row("")
    for line in EXPLAINER:
        body.add_row(f"  [dim]{line}[/]")
    body.add_row("")

    # Pairs laid out two per row: at 80 columns a single flat line wraps and
    # the alignment falls apart.
    entries: list[tuple[str, str]] = []
    if version:
        entries.append(("version", version))
    if path:
        entries.append(("store", _shorten(path)))
    entries.extend(_store_facts(conn))
    try:
        entries.append(_tool_surface())
    except Exception:  # noqa: BLE001 — never let the banner break a command
        pass
    if backend:
        entries.append(("backend", backend))

    stats = Table.grid(padding=(0, 4))
    stats.add_column(no_wrap=True)
    stats.add_column(no_wrap=True)
    for i in range(0, len(entries), 2):
        cells = []
        for key, value in entries[i:i + 2]:
            cells.append(f"[bold cyan]{key:>7}[/]  {value}")
        while len(cells) < 2:
            cells.append("")
        stats.add_row(*cells)
    body.add_row(stats)
    return body


def _shorten(path: str, limit: int = 30) -> str:
    """Keep the store path readable without pushing the second column off screen."""
    import os

    home = os.path.expanduser("~")
    if path.startswith(home):
        path = "~" + path[len(home):]
    return path if len(path) <= limit else "…" + path[-(limit - 1):]


GREETED_KEY = "greeted"


def should_greet(conn: sqlite3.Connection) -> bool:
    """True exactly once per store, on the run that created it."""
    row = conn.execute("SELECT value FROM meta WHERE key = ?", (GREETED_KEY,)).fetchone()
    if row:
        return False
    conn.execute("INSERT OR REPLACE INTO meta(key, value) VALUES (?, '1')", (GREETED_KEY,))
    conn.commit()
    return True


FIRST_RUN_HELP = """   Your store lives at {path}

   [bold]Getting started[/]
     nenapu write "..." --kind project      remember something
     nenapu search "..."                    recall it, ranked by belief
     nenapu why <id>                        what a memory rests on
     nenapu loops                           what the store no longer trusts
     nenapu doctor --calibrate              check a model before it audits

   Facts carrying a --verify-cmd never run until you approve them:
     nenapu approve
"""
