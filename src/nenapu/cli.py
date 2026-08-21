"""Command line interface."""

from __future__ import annotations

import json
import os
import re
import sys
import time
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from . import __version__, open_store
from .audit import LLMUnavailable
from .audit import audit as run_audit
from .distill import distill as run_distill
from .export import render, write_file
from .models import Fact, Kind, Skill, Status
from .retrieval_report import MIN_DAYS_OF_DATA
from .store import DAY, effective_confidence, now
from .verify import run_check, apply_result, verify_scope

# The knot lives in `nenapu version` and the first-run greeting, not here:
# Rich reflows help text and breaks the alignment, and art that has to survive
# a word-wrapper is art in the wrong place.
HELP = """\
[bold]ನೆನಪು  ·  n e n a p u[/] — memory that knows what it rests on.

A store, not an agent. Facts carry provenance, decay on a clock, prove
themselves with a command, and lose standing when what they rest on falls.

Start with [cyan]nenapu remember[/], then [cyan]nenapu recall[/]. See [cyan]nenapu doubts[/] for anything
the store no longer stands behind.
"""

# Panels, because twenty-two commands in one list is a wall. Grouped by the
# question the user is answering, not by the module the command lives in.
REMEMBER = "Remember and recall"
NETWORK = "Belief network"
UPKEEP = "Trust and upkeep"
OUTCOMES = "Did it help?"
DIAGNOSE = "Setup and diagnostics"
ACTIVITY = "Activity ledger"

app = typer.Typer(help=HELP, rich_markup_mode="rich")
skill_app = typer.Typer(help="Skill library with an outcome loop", no_args_is_help=True)
app.add_typer(skill_app, name="skill", rich_help_panel=UPKEEP)

def alias(name: str, panel: str):
    """Register a command under an older name, hidden from the listing.

    Renaming a command in a tool people have already wired into hooks and
    scripts is a breaking change unless the old word keeps working. The old
    names stay, they just stop being advertised — a hidden alias costs a line
    and an unhidden one would put every command on the screen twice.
    """
    def register(fn):
        app.command(name, rich_help_panel=panel, hidden=True)(fn)
        return fn
    return register


console = Console()
# The banner goes to stderr so `nenapu recall --json | jq` is never corrupted
# by a one-time greeting.
err_console = Console(stderr=True)


def _height(console_out, renderable) -> int:
    """How many lines a renderable will take at this console's width."""
    return len(console_out.render_lines(renderable, pad=False))


def _big_panel(console_out, db: str | None = None, store=None,
               rows: int | None = None) -> int:
    """Wordmark plus a readout of the store, sized to the room available.

    Returns the number of lines printed, so the caller can spend what is left
    on the command list rather than pushing the wordmark off the top of the
    screen — which is what it did before anything here counted rows."""
    from .banner import panel
    from .llm import available

    conn = path = None
    try:
        if store is None:
            store, _ = open_store(db or os.environ.get("NENAPU_DB"))
        conn = store.conn
        path = store.conn.execute("PRAGMA database_list").fetchone()[2]
    except Exception:  # noqa: BLE001 — a banner must never break the command
        pass

    ok, detail = available()
    # The URL is noise in a status line; the backend and model are the answer.
    backend = detail.split(" @ ")[0] if ok else ""
    hero = panel(console_out, version=__version__, conn=conn,
                 path=path or "", backend=backend, rows=rows)
    console_out.print()
    console_out.print(hero)
    console_out.print()
    return _height(console_out, hero) + 2


DB_OPT = typer.Option(None, "--db", help="Path to the store (default ~/.nenapu/nenapu.db)")


@app.callback(invoke_without_command=True)
def main(ctx: typer.Context, db: str = DB_OPT) -> None:
    """Show the mark, sized to the occasion.

    The full panel is for someone looking around; every other command gets one
    dim line. Art that repeats before every `search` stops being charming by
    the third command. Both go to stderr, so `--json | jq` still pipes clean
    data. `NENAPU_NO_BANNER=1` silences it for cron and CI.
    """
    from .banner import should_walk, stamp

    quiet = bool(os.environ.get("NENAPU_NO_BANNER"))

    if ctx.invoked_subcommand is None:
        store = None
        try:
            # Named here as well as on every subcommand, so `nenapu --db other.db`
            # shows that store rather than silently reading the default one.
            store, _ = open_store(db or os.environ.get("NENAPU_DB"))
        except Exception:  # noqa: BLE001 — never fail on the greeting path
            pass
        if not quiet:
            console.print()
            console.print(_landing(store))
            console.print()

        # The very first bare `nenapu` is someone who just installed it. Wire
        # it up and explain it once; after that a bare invocation is someone
        # looking for a command name, and a wizard would be in the way.
        if store is not None and not quiet and should_walk(store.conn):
            _setup_walkthrough(console)
            _usage_guide(console)
            raise typer.Exit(0)

        # A bare `nenapu` is someone looking around, not a usage error — so
        # exit clean rather than with Typer's exit code 2. The command names
        # are already on the screen above; `--help` has the descriptions.
        if quiet:
            console.print(ctx.get_help())
        raise typer.Exit(0)

    # Hooks are machine-to-machine. `recall-hook` writes into a session's
    # context and `observe` runs headless after one ends; a mark on either is
    # noise in a log at best and content in a prompt at worst.
    if not quiet and ctx.invoked_subcommand not in ("version", "recall-hook", "observe", "learn"):
        err_console.print(stamp(__version__))


# The landing view is two columns when there is room for them. Stacked, the
# wordmark, the dog and fifty lines of grouped help come to seventy-three rows
# — on a twenty-four row terminal the mark someone ran the command to look at
# has scrolled off the top before they can read it. Side by side it is under
# twenty, and nothing scrolls at all.
SIDE_BY_SIDE_WIDTH = 96
# Columns the text beside the dog needs before it starts losing words. Facts
# are sentences; under this they arrive as ellipses and say nothing.
MIN_TEXT_COLUMN = 62


def _first_that_fits(candidates, rows: int):
    """The first candidate that fits the screen, or the smallest one.

    Measured rather than calculated. How tall any of these is depends on how
    many commands exist and where the terminal wraps them, and a landing view
    that is one row too tall scrolls the wordmark off the top — which is the
    entire bug this is here to prevent.
    """
    for candidate in candidates:
        if _height(console, candidate) <= rows - 1:
            return candidate
    return candidates[-1]


def _grown(build, frame, budget: int):
    """Add facts until the screen is full, keeping the last version that fits.

    Grown one at a time and measured rather than calculated from the leftover
    rows: the drawing can be taller than the column beside it, in which case
    the first few facts cost no height at all, and arithmetic that assumed
    otherwise left a quarter of the screen empty.
    """
    best = frame
    for count in range(1, 41):
        candidate = build(count)
        height = _height(console, candidate)
        if height > budget:
            break
        if height == _height(console, best) and count > 12:
            break          # the store has run out of facts to show
        best = candidate
    return best


def _command_groups(with_help: bool = False) -> dict[str, list[tuple[str, str]]]:
    """Commands by the panel they belong to, straight from the app's registry.

    Read off the registry rather than written out again here: a list that has
    to be maintained twice is a list that goes stale the first time someone
    adds a command, and nothing fails to say so.
    """
    def summary(callback) -> str:
        doc = (callback.__doc__ or "").strip() if callback else ""
        return doc.split("\n")[0] if with_help else ""

    groups: dict[str, list[tuple[str, str]]] = {}
    for command in app.registered_commands:
        name = command.name or (command.callback.__name__.replace("_", "-")
                                if command.callback else "")
        if name and not command.hidden:
            groups.setdefault(command.rich_help_panel or "Commands", []).append(
                (name, command.help or summary(command.callback)))
    for group in app.registered_groups:
        if group.name:
            groups.setdefault(group.rich_help_panel or "Commands", []).append(
                (group.name, group.help or "" if with_help else ""))
    return {panel: sorted(rows) for panel, rows in groups.items()}


def _commands_table(with_help: bool = False):
    """What you can type. With descriptions when the screen has room for them,
    and names alone when it does not — the full text is always in `--help`."""
    from rich.table import Table

    table = Table.grid(padding=(0, 2))
    if not with_help:
        table.add_column(no_wrap=True, style="bold cyan")
        table.add_column()
        for panel_name, rows in _command_groups().items():
            table.add_row(panel_name, "[dim]" + "  ".join(n for n, _ in rows) + "[/]")
        return table

    table.add_column(no_wrap=True, style="cyan")
    table.add_column(no_wrap=True)
    for i, (panel_name, rows) in enumerate(_command_groups(True).items()):
        if i:
            table.add_row("", "")
        table.add_row("", f"[bold]{panel_name}[/]")
        for name, help_text in rows:
            table.add_row(f"  {name}", f"[dim]{help_text}[/]")
    return table


def _recent(store, limit: int, width: int = 64):
    """The last few things it learned, for a screen with room to show them.

    Better filler than more art: someone landing here wants to know whether the
    thing is working, and five sentences it picked up on its own answer that
    faster than any number can.
    """
    from rich.table import Table

    table = Table.grid(padding=(0, 2))
    table.add_column(no_wrap=True, style="dim")
    table.add_column(no_wrap=True)
    rows = store.conn.execute(
        "SELECT text, origin FROM facts WHERE status = 'active'"
        " ORDER BY created_at DESC LIMIT ?", (limit,)).fetchall()
    for row in rows:
        mark = "·" if row["origin"] == "user_stated" else "◆"
        text = row["text"]
        clipped = text[:width] + ("…" if len(text) > width else "")
        table.add_row(mark, f"[dim]{clipped}[/]")
    return table if rows else None


def _landing(store):
    """Wordmark, store readout, the dog, and what you can type — on one screen.

    Two failures to avoid, in this order. Stacked and unmeasured this came to
    seventy-three rows, so on a twenty-four row terminal the wordmark someone
    ran the command to look at had scrolled off before they could read it. Then
    the fix over-corrected: twenty-three rows on a forty-row screen, most of it
    empty, which looks like the program has nothing to say.

    So the view is built at several sizes and the largest one that fits is
    printed. It grows into the room it has — a bigger dog, the three-line
    pitch, a description against every command — rather than only shrinking out
    of trouble.
    """
    from rich.console import Group
    from rich.table import Table
    from rich.text import Text

    from .banner import hero_shades, panel, wordmark
    from .llm import available

    conn = path = None
    try:
        if store is not None:
            conn = store.conn
            path = store.conn.execute("PRAGMA database_list").fetchone()[2]
    except Exception:  # noqa: BLE001 — a banner must never break the command
        pass

    ok, detail = available()
    backend = detail.split(" @ ")[0] if ok else ""
    rows, cols = console.size.height, console.width
    hint = Text.from_markup("[dim]nenapu --help for the full text of any of these[/]")

    mood = "content"
    try:
        from .pet import assess

        if store is not None:
            mood = assess(store).mood
    except Exception:  # noqa: BLE001 — the dog is never the reason a run fails
        pass

    def dog(scale: float) -> Text:
        try:
            from .pet_art import coloured

            return Text.from_markup("\n".join(coloured(mood, hero_shades(),
                                                       scale=scale)))
        except Exception:  # noqa: BLE001
            return Text("")

    def readout(explainer: bool):
        return panel(console, version=__version__, conn=conn, path=path or "",
                     backend=backend, art=False, explainer=explainer, mark=False)

    def art_width(scale: float) -> int:
        try:
            from .pet_art import draw

            return max(len(row) for row in draw(mood, scale=scale))
        except Exception:  # noqa: BLE001
            return 0

    def side_by_side(scale: float, explainer: bool, recent: int, header: bool):
        text_width = cols - art_width(scale) - 4
        right = Table.grid()
        right.add_column(no_wrap=True)
        right.add_row(readout(explainer))
        right.add_row("")
        if recent and store is not None:
            learned = _recent(store, recent, width=text_width - 6)
            if learned is not None:
                right.add_row("[bold]Lately[/]")
                right.add_row(learned)
                right.add_row("")
        right.add_row(_commands_table())
        right.add_row("")
        right.add_row(hint)

        columns = Table.grid(padding=(0, 4))
        columns.add_column(vertical="middle")
        columns.add_column(vertical="middle")
        columns.add_row(dog(scale), right)
        return Group(*wordmark(console), Text(""), columns) if header else columns

    def stacked(art: bool, commands: bool, recent: int = 0, hint_line: bool = True):
        parts = [panel(console, version=__version__, conn=conn, path=path or "",
                       backend=backend, rows=rows, art=art), Text("")]
        if recent and store is not None:
            learned = _recent(store, recent)
            if learned is not None:
                parts += [Text.from_markup("[bold]Lately[/]"), learned, Text("")]
        if commands:
            parts += [_commands_table(), Text("")]
        return Group(*parts, hint) if hint_line else Group(*parts[:-1])

    # The drawing is seven rows tall whatever its width, so a ladder of sizes
    # cannot fill a tall screen on its own — it lands on whichever rung is
    # closest and leaves the rest blank, which is what "there is space below
    # it" meant. So the frame is measured once and the leftover rows are spent
    # on what the store has learned. That fills exactly, and fills it with the
    # most useful thing on the screen.
    if cols >= SIDE_BY_SIDE_WIDTH:
        scale = 1.0
        for candidate in (2.0, 1.7, 1.4, 1.2):
            if art_width(candidate) <= cols - MIN_TEXT_COLUMN:
                scale = candidate
                break
        for header in (True, False):
            frame = side_by_side(scale, False, 0, header)
            if _height(console, frame) <= rows - 2:
                return _grown(lambda n: side_by_side(scale, False, n, header),
                              frame, rows - 2)

    # Stacked, for terminals too narrow to put anything beside anything else.
    # Same trick: measure the frame, then fill what is left with facts.
    # The one-line pointer at `--help` is shed before the block letters are.
    # Every command it refers to is already on the screen above it, and the
    # wordmark is the part being protected — one more command registered must
    # not be what costs a twenty-five row terminal the thing it came to see.
    for art, hint_line in ((True, True), (True, False), (False, True)):
        frame = stacked(art, True, hint_line=hint_line)
        if _height(console, frame) <= rows - 2:
            return _grown(lambda n: stacked(art, True, n, hint_line=hint_line),
                          frame, rows - 2)
    return _first_that_fits([stacked(False, True), stacked(False, False)], rows - 2)


def _greet(store) -> None:
    """First-run orientation, under the full panel."""
    from .banner import FIRST_RUN_HELP, should_greet

    if not should_greet(store.conn) or os.environ.get("NENAPU_NO_BANNER"):
        return
    path = store.conn.execute("PRAGMA database_list").fetchone()[2] or ":memory:"
    err_console.print()
    err_console.print(FIRST_RUN_HELP.format(path=path))


def _stores(db: str | None):
    store, skills = open_store(db or os.environ.get("NENAPU_DB"))
    _greet(store)
    return store, skills




@app.command(rich_help_panel=DIAGNOSE)
def version(plain: bool = typer.Option(False, "--plain", help="Version string only"),
            db: str = DB_OPT) -> None:
    """Print the version, with the mark."""
    if plain:
        console.print(__version__)
        return
    _big_panel(console, db)


@app.command("remember", rich_help_panel=REMEMBER)
@alias("write", REMEMBER)
def remember(
    text: str,
    kind: str = typer.Option("project", help="user|project|environment|feedback|reference"),
    scope: str = typer.Option(
        "", help="Scope; omit to infer from kind (user/feedback: global, else: this project)"
    ),
    key: str = typer.Option("", help="Contradiction join key, e.g. db.port"),
    origin: str = typer.Option("user_stated", help="user_stated|tool_observed|file_derived|agent_inferred"),
    confidence: float = 0.8,
    decay: str = typer.Option("", help="immutable|slow|medium|volatile"),
    verify_cmd: str = typer.Option("", help="Shell command that proves this fact"),
    verify_expect: str = typer.Option("", help="Substring the output must contain"),
    db: str = DB_OPT,
) -> None:
    """Store a fact."""
    from .store import scope_for

    store, _ = _stores(db)
    resolved_scope = scope or scope_for(kind)
    fact, conflicts = store.write(
        Fact(
            text=text, kind=kind, scope=resolved_scope, key=key or None, origin=origin,
            confidence=confidence, decay_class=decay or None,
            verify_cmd=verify_cmd or None, verify_expect=verify_expect or None,
        ),
        actor="cli",
    )
    console.print(f"[green]stored[/] #{fact.id}  belief={effective_confidence(fact):.2f}")
    for c in conflicts:
        colour = "yellow" if c.resolution == "superseded" else "red"
        console.print(f"[{colour}]conflict[/] with #{c.other_id}: {c.detail} -> {c.resolution}")


@app.command("recall", rich_help_panel=REMEMBER)
@alias("search", REMEMBER)
def recall(
    query: str,
    scope: str = typer.Option("", help="Limit to a scope"),
    limit: int = 10,
    min_confidence: float = 0.0,
    json_out: bool = typer.Option(False, "--json"),
    db: str = DB_OPT,
) -> None:
    """Recall facts, ranked by match and current believability."""
    store, _ = _stores(db)
    hits = store.search(query, scope=scope or None, limit=limit, min_confidence=min_confidence)

    if json_out:
        console.print_json(
            json.dumps(
                [
                    {"id": f.id, "text": f.text, "score": round(s, 3), **why}
                    for f, s, why in hits
                ]
            )
        )
        return

    if not hits:
        console.print("[dim]nothing above threshold[/]")
        return

    table = Table(show_lines=False)
    for col in ("id", "score", "belief", "age", "check", "fact"):
        table.add_column(col)
    for fact, score, why in hits:
        check = {"pass": "[green]pass[/]", "fail": "[red]FAIL[/]", "none": "[dim]-[/]"}.get(
            fact.verify_status, fact.verify_status
        )
        table.add_row(
            str(fact.id), f"{score:.2f}", f"{why['confidence']:.2f}",
            f"{why['age_days']:.0f}d", check, fact.text[:80],
        )
    console.print(table)


@app.command("list", rich_help_panel=REMEMBER)
def list_facts(
    scope: str = "",
    status: str = Status.ACTIVE,
    limit: int = 50,
    db: str = DB_OPT,
) -> None:
    """List stored facts."""
    store, _ = _stores(db)
    facts = store.list_facts(scope=scope or None, status=status, limit=limit)
    table = Table()
    for col in ("id", "kind", "scope", "key", "belief", "fact"):
        table.add_column(col)
    for f in facts:
        table.add_row(
            str(f.id), f.kind, f.scope, f.key or "-",
            f"{effective_confidence(f):.2f}", f.text[:70],
        )
    console.print(table)


@app.command(rich_help_panel=REMEMBER)
def clear(
    scope: str = typer.Option("", help="Only this scope; omit for everything"),
    kind: str = typer.Option("", help="Only this kind: user|project|environment|feedback"),
    purge: bool = typer.Option(False, "--purge",
                               help="Delete the rows outright instead of retiring them"),
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip the confirmation"),
    db: str = DB_OPT,
) -> None:
    """Forget everything at once. Retires by default; `--purge` deletes.

    Retiring is the honest default: the rows stay, the journal says who cleared
    them and when, and a store that has been emptied can still explain itself.
    `--purge` is the one that destroys history, which is why it is a separate
    word rather than a flag on the same meaning.
    """
    store, _ = _stores(db)
    active = store.stats(scope=scope or None).get("active", 0)
    total = store.conn.execute(
        "SELECT COUNT(*) c FROM facts" + (" WHERE scope = ?" if scope else ""),
        (scope,) if scope else ()).fetchone()["c"]
    doomed = total if purge else active

    if not doomed:
        console.print("[dim]nothing to clear[/]")
        return

    where = f" in scope [cyan]{scope}[/]" if scope else ""
    where += f" of kind [cyan]{kind}[/]" if kind else ""
    if purge:
        console.print(f"[red]purge[/] deletes [bold]{doomed}[/] fact(s){where} "
                      "and every edge, recall and conflict attached to them.")
        console.print("[dim]this cannot be undone; without --purge they are retired "
                      "and stay readable[/]")
    else:
        console.print(f"retire [bold]{doomed}[/] active fact(s){where}, "
                      "keeping them for audit")

    if not yes:
        # A pipe is not a person, and this is the one command here that can
        # empty a store. Non-interactive runs refuse rather than assuming.
        if not sys.stdin.isatty():
            console.print("[red]refusing[/]: not a terminal. Pass --yes to mean it.")
            raise typer.Exit(1)
        if not typer.confirm("  go ahead?", default=False):
            console.print("[dim]left alone[/]")
            raise typer.Exit(0)

    if purge:
        gone = store.purge(scope=scope or None)
        store.conn.execute("VACUUM")
        console.print(f"[red]purged[/] {gone} fact(s)")
    else:
        gone = store.forget_all(scope=scope or None, kind=kind or None)
        console.print(f"retired {gone} fact(s) — [dim]still readable with "
                      "`nenapu list --status retired`[/]")


@app.command(rich_help_panel=REMEMBER)
def forget(fact_id: str,
           yes: bool = typer.Option(False, "--yes", "-y",
                                    help="Skip the confirmation on `forget all`"),
           db: str = DB_OPT) -> None:
    """Retire a fact. `nenapu forget all` clears the store.

    `all` is accepted because it is what people type. It is the same thing as
    `nenapu clear`, confirmation and all.
    """
    if fact_id.strip().lower() == "all":
        clear(scope="", kind="", purge=False, yes=yes, db=db)
        return
    try:
        fact_id = int(fact_id)
    except ValueError:
        raise typer.BadParameter("give a fact id, or `all` to clear the store")

    store, _ = _stores(db)
    if not store.get(fact_id):
        raise typer.BadParameter(f"no fact {fact_id}")
    store.forget(fact_id, actor="cli")
    console.print(f"[yellow]retired[/] #{fact_id}")


@app.command("check", rich_help_panel=UPKEEP)
@alias("verify", UPKEEP)
def check(
    fact_id: int = typer.Option(0, help="Verify one fact instead of a whole scope"),
    scope: str = "",
    stale_after_days: float = typer.Option(0.0, help="Skip checks that ran recently"),
    db: str = DB_OPT,
) -> None:
    """Re-run executable checks."""
    store, _ = _stores(db)

    if fact_id:
        fact = store.get(fact_id)
        if not fact:
            raise typer.BadParameter(f"no fact {fact_id}")
        results = [run_check(fact, conn=store.conn)]
        apply_result(store, results[0])
    else:
        results = verify_scope(
            store, scope=scope or None, only_stale_after_days=stale_after_days or None
        )

    if not results:
        console.print("[dim]no facts carry a check[/]")
        return
    for r in results:
        colour = {"pass": "green", "fail": "red", "error": "yellow",
                  "blocked": "yellow"}.get(r.status, "white")
        console.print(f"[{colour}]{r.status:7}[/] #{r.fact_id}  {r.detail[:100]}")

    failing = sum(1 for r in results if r.status == "fail")
    blocked = [r.fact_id for r in results if r.status == "blocked"]
    ran = len(results) - len(blocked)
    console.print(f"\n{ran} checked, [red]{failing} failing[/]")
    if blocked:
        console.print(
            f"[yellow]{len(blocked)} check(s) never ran[/] — unapproved shell. "
            "Review them with `nenapu approve`."
        )


@app.command(rich_help_panel=UPKEEP)
def approve(
    fact_ids: list[int] = typer.Argument(None, help="Fact ids to approve; omit to review all"),
    all_pending: bool = typer.Option(False, "--all", help="Approve every pending check"),
    revoke_id: int = typer.Option(0, "--revoke", help="Withdraw approval for a fact's check"),
    db: str = DB_OPT,
) -> None:
    """Review and approve the shell commands attached to facts.

    A `verify_cmd` is shell, and facts are written by agents that read
    untrusted input. Nothing runs until a human has read the exact command, so
    a prompt injection that plants a fact cannot turn `nenapu check` into
    scheduled code execution.
    """
    from .approval import approve as record_approval
    from .approval import approved_list, concerns, pending, revoke

    store, _ = _stores(db)

    if revoke_id:
        fact = store.get(revoke_id)
        if not fact or not fact.verify_cmd:
            raise typer.BadParameter(f"fact {revoke_id} has no check")
        console.print("[yellow]revoked[/]" if revoke(store.conn, fact.verify_cmd)
                      else "[dim]was not approved[/]")
        return

    waiting = pending(store.conn)
    if fact_ids:
        waiting = [w for w in waiting if w[0] in set(fact_ids)]

    if not waiting:
        approved = approved_list(store.conn)
        console.print("[green]no checks awaiting approval[/]")
        if approved:
            console.print(f"[dim]{len(approved)} command(s) already approved:[/]")
            for command, by, _at in approved:
                console.print(f"  [dim]{by}[/] {command[:80]}")
        return

    for fact_id, origin, command in waiting:
        fact = store.get(fact_id)
        console.print(f"\n[bold]#{fact_id}[/] {fact.text[:70]}")
        console.print(f"  origin  : {origin}"
                      + ("  [red](written by an agent, not by you)[/]"
                         if origin == "agent_inferred" else ""))
        console.print(f"  command : [cyan]{command}[/]")
        for concern in concerns(command):
            console.print(f"  [red]!![/] {concern}")

        if all_pending:
            ok = True
        elif sys.stdin.isatty():
            ok = typer.confirm("  approve this command to run on every verify?", default=False)
        else:
            # Never auto-approve in a pipe, a cron job, or an agent's shell.
            console.print("  [yellow]skipped[/] — approval requires a terminal, or --all")
            continue

        if ok:
            record_approval(store.conn, command, fact_id=fact_id)
            console.print("  [green]approved[/]")
        else:
            console.print("  [dim]left blocked[/]")


@app.command(rich_help_panel=UPKEEP)
def audit(
    evidence_file: str = typer.Option("", help="File of current evidence to audit against"),
    scope: str = "",
    older_than_days: float = 30.0,
    dry_run: bool = typer.Option(False, "--dry-run", help="Report findings only"),
    apply_findings: bool = typer.Option(
        False, "--apply", help="Act on findings from a local model (cloud backends act by default)"
    ),
    db: str = DB_OPT,
) -> None:
    """LLM re-check of facts that decay and shell checks cannot cover.

    Local backends report without acting unless you pass --apply.
    """
    store, _ = _stores(db)
    evidence = Path(evidence_file).read_text() if evidence_file else ""
    try:
        report = run_audit(
            store, evidence=evidence, scope=scope or None,
            older_than_days=older_than_days,
            apply=False if dry_run else (True if apply_findings else None),
        )
    except LLMUnavailable as exc:
        console.print(f"[red]audit unavailable:[/] {exc}")
        raise typer.Exit(1)

    if not report.requested:
        console.print("[dim]nothing old or contested enough to audit[/]")
        return
    for f in report.findings:
        colour = {"holds": "green", "stale": "yellow", "wrong": "red"}.get(f.verdict, "dim")
        console.print(f"[{colour}]{f.verdict:8}[/] #{f.fact_id}  {f.reason}")

    console.print(
        f"\n{report.covered}/{report.requested} facts covered in {report.batches} batch(es)"
        f" via {report.backend}"
    )
    if not report.applied and not dry_run:
        console.print(
            "[yellow]report only[/] — nothing changed. A local model can be confidently "
            "wrong about a fact the evidence supports; re-run with --apply to act on these."
        )
    if report.missing:
        # Never let a partial audit read as a clean one.
        console.print(
            f"[yellow]not audited:[/] {', '.join('#' + str(i) for i in report.missing)}"
            " — these keep their current confidence"
        )
    if report.invented:
        console.print(f"[red]model invented ids:[/] {report.invented} (ignored)")


@app.command("tidy", rich_help_panel=UPKEEP)
@alias("distill", UPKEEP)
def tidy(
    scope: str = "",
    token_budget: int = 1500,
    no_llm: bool = typer.Option(False, help="Dedupe only, no model call"),
    db: str = DB_OPT,
) -> None:
    """Compress a scope: drop duplicates, then merge related facts."""
    store, _ = _stores(db)
    try:
        report = run_distill(
            store, scope=scope or None, token_budget=token_budget, use_llm=not no_llm
        )
    except LLMUnavailable as exc:
        console.print(f"[red]distill unavailable:[/] {exc}")
        raise typer.Exit(1)
    console.print(
        f"{report.tokens_before} -> {report.tokens_after} tokens "
        f"([green]{report.saved_pct:.0f}% saved[/]); "
        f"{report.deduped} deduped, {report.merged} merged"
    )


@app.command(rich_help_panel=UPKEEP)
def export(
    path: str = typer.Argument("", help="File to write; omit to print to stdout"),
    scope: str = "",
    min_confidence: float = 0.35,
    db: str = DB_OPT,
) -> None:
    """Materialize verified memory into a CLAUDE.md / AGENTS.md block."""
    store, _ = _stores(db)
    if not path:
        console.print(render(store, scope=scope or None, min_confidence=min_confidence))
        return
    written = write_file(path, store, scope=scope or None, min_confidence=min_confidence)
    console.print(f"[green]wrote[/] {written}")


@app.command(rich_help_panel=NETWORK)
def why(fact_id: int, depth: int = 3, db: str = DB_OPT) -> None:
    """Show what a fact rests on, and what rests on it."""
    store, _ = _stores(db)
    fact = store.get(fact_id)
    if not fact:
        raise typer.BadParameter(f"no fact {fact_id}")

    def show(node: dict, indent: int = 0) -> None:
        mark = {"pass": "[green]v[/]", "fail": "[red]x[/]"}.get(node.get("verify_status"), " ")
        status = node.get("status", "")
        colour = {"suspect": "yellow", "disputed": "yellow", "retired": "dim"}.get(status, "white")
        via = f" [dim]({node['via']})[/]" if node.get("via") else ""
        console.print(f"{'  ' * indent}{mark} [{colour}]#{node['id']}[/] {node['text'][:70]}{via}")
        for parent in node.get("rests_on", []):
            show(parent, indent + 1)

    chain = store.graph.why(fact_id, depth=depth)
    console.print(f"belief [bold]{effective_confidence(fact):.2f}[/]  "
                  f"track record {fact.good_recalls} good / {fact.bad_recalls} bad")
    if chain.get("suspect_reason"):
        console.print(f"[yellow]suspect:[/] {chain['suspect_reason']}")
    show(chain)
    if chain.get("supports"):
        console.print(f"\n[dim]depended on by:[/] {', '.join('#' + str(i) for i in chain['supports'])}")


@app.command(rich_help_panel=NETWORK)
def link(parent_id: int, child_id: int, db: str = DB_OPT) -> None:
    """Declare that one fact rests on another."""
    store, _ = _stores(db)
    if not store.get(parent_id) or not store.get(child_id):
        raise typer.BadParameter("both facts must exist")
    edge = store.graph.link(parent_id, child_id)
    console.print(f"[green]linked[/] #{child_id} rests on #{parent_id}" if edge else "already linked")


@app.command("doubts", rich_help_panel=NETWORK)
@alias("loops", NETWORK)
def doubts(limit: int = 20, db: str = DB_OPT) -> None:
    """Unresolved memory debt: contradicted, unsupported, or failing facts."""
    store, _ = _stores(db)
    rows = store.conn.execute(
        "SELECT id, text, status, suspect_reason, verify_detail FROM facts"
        " WHERE status IN ('disputed','suspect') OR verify_status = 'fail'"
        " ORDER BY updated_at DESC LIMIT ?",
        (limit,),
    ).fetchall()
    if not rows:
        console.print("[green]no open loops[/]")
    else:
        table = Table()
        for col in ("id", "state", "fact", "why"):
            table.add_column(col)
        for r in rows:
            table.add_row(str(r["id"]), r["status"], r["text"][:50],
                          (r["suspect_reason"] or r["verify_detail"] or "")[:50])
        console.print(table)
    console.print(f"[dim]{len(store.ledger.pending(limit=500))} recalls awaiting a grade[/]")


@app.command("helped", rich_help_panel=OUTCOMES)
@alias("good", OUTCOMES)
def helped(recall_id: int, note: str = "", db: str = DB_OPT) -> None:
    """Mark a recall as having helped."""
    store, _ = _stores(db)
    ok = store.ledger.grade(recall_id, "good", source="human", note=note or None)
    console.print("[green]graded good[/]" if ok else "[dim]already graded, or no such recall[/]")


@app.command("misled", rich_help_panel=OUTCOMES)
@alias("bad", OUTCOMES)
def misled(recall_id: int, note: str = "", db: str = DB_OPT) -> None:
    """Mark a recall as having misled you. The fact loses standing."""
    store, _ = _stores(db)
    recall = store.ledger.get(recall_id)
    if not recall:
        raise typer.BadParameter(f"no recall {recall_id}")
    if store.ledger.grade(recall_id, "bad", source="human", note=note or None):
        fact = store.get(recall.fact_id)
        console.print(f"[yellow]graded bad[/] — #{fact.id} now at "
                      f"{effective_confidence(fact):.2f}: {fact.text[:60]}")
    else:
        console.print("[dim]already graded[/]")


@app.command("grade", rich_help_panel=OUTCOMES)
@alias("outcome", OUTCOMES)
def grade(
    session_id: str,
    success: bool = typer.Option(..., "--success/--failure"),
    note: str = "",
    db: str = DB_OPT,
) -> None:
    """Grade every memory a task used, in one call."""
    store, _ = _stores(db)
    graded = store.ledger.grade_session(
        session_id, "good" if success else "bad", source="agent", note=note or None
    )
    console.print(f"graded {graded} recall(s) as {'good' if success else 'bad'}")


@app.command("ungraded", rich_help_panel=OUTCOMES)
def ungraded(limit: int = 20, db: str = DB_OPT) -> None:
    """Recalls waiting on a grade, with the ids `misled` and `helped` accept.

    `nenapu misled <id>` and `nenapu helped <id>` need a recall id no other
    command prints, which makes the human grading path unreachable to anyone
    who has not read the schema. This lists it: reuses `Ledger.pending`
    rather than a second query with its own idea of what pending means.
    """
    store, _ = _stores(db)
    recalls = store.ledger.pending(limit=limit)
    if not recalls:
        console.print("[green]no recalls awaiting a grade[/]")
        return

    table = Table()
    for col in ("id", "fact", "session", "age"):
        table.add_column(col)
    for r in recalls:
        fact = store.get(r.fact_id)
        age_hours = (now() - r.created_at) / 3600.0
        table.add_row(str(r.id), (fact.text[:60] if fact else "?"),
                      r.session_id or "-", f"{age_hours:.1f}h ago")
    console.print(table)


@app.command(rich_help_panel=DIAGNOSE)
def stats(scope: str = "", db: str = DB_OPT) -> None:
    """Health of the store."""
    store, _ = _stores(db)
    s = store.stats(scope=scope or None)
    table = Table(show_header=False)
    for key, value in s.items():
        table.add_row(key, json.dumps(value) if isinstance(value, dict) else str(value))
    console.print(table)


_DURATION_UNITS = {"s": 1, "m": 60, "h": 3600, "d": DAY, "w": 7 * DAY}


def _parse_since(spec: str) -> float:
    """A duration like `1w`, `3d`, `12h` into seconds. Raises for anything else
    rather than guessing, since a silently-wrong window is worse than a
    rejected flag."""
    match = re.fullmatch(r"(\d+)([smhdw])", spec.strip().lower())
    if not match:
        raise typer.BadParameter(f"invalid duration {spec!r} — expected e.g. 1w, 3d, 12h")
    n, unit = match.groups()
    return float(n) * _DURATION_UNITS[unit]


def _ledger(db: str | None):
    from .activity import ActivityLedger

    store, _ = _stores(db)
    return ActivityLedger(store.conn)


def _session_line(session: dict) -> str:
    ended = " (in progress)" if session["ended_at"] is None else ""
    return f"[cyan]{session['project_scope']}[/] via {session['agent']}{ended}"


@app.command(rich_help_panel=ACTIVITY)
def standup(db: str = DB_OPT) -> None:
    """What happened recently, across every project."""
    ledger = _ledger(db)
    sessions = ledger.recent_sessions(since_at=now() - 2 * DAY)
    if not sessions:
        console.print("Nothing in the last two days.")
        return
    for session in sessions:
        console.print(_session_line(session))
        for event in ledger.file_events_for_session(session["id"]):
            console.print(f"  {event['op']} [magenta]{event['path']}[/]")
        for c in ledger.commits_for_session(session["id"]):
            console.print(f"  commit {c['sha'][:8]} {c['subject'] or ''}")


@app.command(rich_help_panel=ACTIVITY)
def activity(
    since: str = typer.Option("1w", help="Duration back to look, e.g. 1w, 3d, 12h"),
    db: str = DB_OPT,
) -> None:
    """Timeline of work, grouped by project and agent."""
    ledger = _ledger(db)
    sessions = ledger.recent_sessions(since_at=now() - _parse_since(since))
    if not sessions:
        console.print(f"Nothing in the last {since}.")
        return
    for session in sessions:
        console.print(_session_line(session))
        for event in ledger.file_events_for_session(session["id"]):
            console.print(f"  {event['op']} [magenta]{event['path']}[/]")
        for c in ledger.commits_for_session(session["id"]):
            console.print(f"  commit {c['sha'][:8]} {c['subject'] or ''}")


@app.command(rich_help_panel=ACTIVITY)
def where(path: str, db: str = DB_OPT) -> None:
    """Every session and agent that touched a file."""
    ledger = _ledger(db)
    touches = ledger.file_events_for_path(path)
    if not touches:
        console.print(f"No recorded activity on [magenta]{path}[/].")
        return
    table = Table()
    for col in ("when", "agent", "project", "op", "tool"):
        table.add_column(col)
    for t in touches:
        table.add_row(
            time.strftime("%Y-%m-%d %H:%M", time.localtime(t["at"])),
            t["agent"], t["project_scope"], t["op"], t["tool"] or "-",
        )
    console.print(table)


DEFAULT_TRANSCRIPT_GLOB = "~/.claude/projects/**/*.jsonl"


@app.command(rich_help_panel=ACTIVITY)
def backfill(
    pattern: str = typer.Option(DEFAULT_TRANSCRIPT_GLOB, "--glob",
                                help="Transcripts to replay"),
    agent: str = typer.Option("claude-code", "--agent", help="Which agent wrote them"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Show what would be ingested"),
    db: str = DB_OPT,
) -> None:
    """Replay transcripts already on disk into the activity ledger.

    A parse, not an extraction: no model call, no queued job, no tokens. It is
    what makes "where did I leave off" answerable on a machine whose history
    predates the ledger, and it is safe to run again — a session already
    recorded is left alone, while transcripts written since are picked up.
    """
    from .backfill import backfill_directory, would_backfill

    ledger = _ledger(db)
    if dry_run:
        console.print(f"{would_backfill(ledger, pattern)} session(s) would be ingested")
        return
    console.print(f"{backfill_directory(ledger, pattern, agent=agent)} session(s) ingested")


@app.command(rich_help_panel=ACTIVITY)
def pending(
    project: str = typer.Option("", help="Limit to one project by name"),
    show_all: bool = typer.Option(False, "--all", help="Include loops that have gone quiet"),
    db: str = DB_OPT,
) -> None:
    """Open loops — things mentioned but not yet done. Cross-project."""
    from .loops import LoopBook

    store, _ = _stores(db)
    loops = LoopBook(store.conn).all_open(include_quiet=show_all)
    if project:
        loops = [loop for loop in loops if project.lower() in loop["scope"].lower()]
    if not loops:
        console.print(f"No open loops tracked yet{f' for {project}' if project else ''}.")
        return

    table = Table()
    table.add_column("project", style="cyan")
    table.add_column("open loop")
    table.add_column("closed by", style="dim")
    for loop in loops:
        table.add_row(loop["scope"], loop["text"], loop["resolution_hint"] or "-")
    console.print(table)


def _match_project(ledger, name: str) -> str | None:
    matches = sorted(s for s in ledger.known_scopes() if name.lower() in s.lower())
    return matches[0] if matches else None


@app.command(rich_help_panel=ACTIVITY)
def retrieval(
    window_days: int = typer.Option(MIN_DAYS_OF_DATA, "--window-days",
                                    help="How far back to read the recall ledger"),
    db: str = DB_OPT,
) -> None:
    """Read the recall ledger and say whether retrieval is what fails.

    Shows the counts it decided from, so the verdict can be disagreed with on
    the numbers rather than on faith. Reads only: nothing here grades,
    expires or dedupes anything on the way past.
    """
    from .retrieval_report import VERDICT_MEANING, retrieval_evidence

    store, _ = _stores(db)
    evidence = retrieval_evidence(store, window_days=window_days)

    table = Table()
    table.add_column("measure")
    table.add_column("value", justify="right")
    for label, value in (
        ("graded recalls", evidence["graded"]),
        ("good", evidence["good"]),
        ("bad", evidence["bad"]),
        ("neutral", evidence["neutral"]),
        ("pending (not evidence)", evidence["pending"]),
        ("bad rate", f"{evidence['bad_rate']:.0%}"),
        ("from another project", evidence["wrong_project"]),
        ("sessions given memory", evidence["sessions_with_recalls"]),
        ("sessions given nothing", evidence["sessions_without_recalls"]),
        ("days of data", f"{evidence['days_of_data']:.1f}"),
    ):
        table.add_row(label, str(value))
    console.print(table)

    # Two populations, not one pooled number: injection recalls measure
    # selection (was the right fact chosen), query recalls measure ranking
    # (was the right fact found). Printed separately so the split can be
    # disagreed with on the numbers, not just the total.
    pop_table = Table()
    pop_table.add_column("population")
    pop_table.add_column("graded", justify="right")
    pop_table.add_column("good", justify="right")
    pop_table.add_column("bad", justify="right")
    pop_table.add_column("neutral", justify="right")
    pop_table.add_column("rate", justify="right")
    injection, query = evidence["injection"], evidence["query"]
    pop_table.add_row("injection", str(injection["graded"]), str(injection["good"]),
                      str(injection["bad"]), str(injection["neutral"]),
                      f"unused {injection['unused_rate']:.0%}")
    pop_table.add_row("query", str(query["graded"]), str(query["good"]),
                      str(query["bad"]), str(query["neutral"]),
                      f"bad {query['bad_rate']:.0%}")
    console.print(pop_table)

    console.print(f"[bold]{evidence['verdict']}[/] — {VERDICT_MEANING[evidence['verdict']]}")


@app.command(rich_help_panel=ACTIVITY)
def entities(
    rebuild: bool = typer.Option(False, "--rebuild", help="Rebuild from the activity ledger"),
    scope: str = typer.Option("", help="Limit the rebuild to one project scope"),
    db: str = DB_OPT,
) -> None:
    """Bootstrap the entity graph from sessions, file events and commits.

    Zero model calls — safe to run again, since a rebuild is idempotent.
    """
    from .entities import build_from_activity

    store, _ = _stores(db)
    if not rebuild:
        console.print("Pass --rebuild to build the entity graph from the activity ledger.")
        return
    result = build_from_activity(store, scope=scope or None)
    console.print(f"rebuilt from {result['sessions']} session(s) across "
                  f"{result['scopes']} scope(s)")


@app.command("project", rich_help_panel=ACTIVITY)
def project_cmd(name: str, db: str = DB_OPT) -> None:
    """One repo: recent work, files touched, commits, pending."""
    ledger = _ledger(db)
    scope = _match_project(ledger, name)
    if scope is None:
        console.print(f"No activity found for project [magenta]{name}[/].")
        return

    console.print(f"[cyan]{scope}[/]")
    for session in ledger.sessions_for_scope(scope, limit=10):
        console.print(f"  session by {session['agent']} — {session['git_branch'] or '?'}")
    for event in ledger.file_events_for_scope(scope, limit=20):
        console.print(f"  {event['op']} [magenta]{event['path']}[/]")
    for c in ledger.commits_for_scope(scope, limit=10):
        console.print(f"  commit {c['sha'][:8]} {c['subject'] or ''}")


@app.command(rich_help_panel=UPKEEP, hidden=True)
def recall_hook(db: str = DB_OPT) -> None:
    """Emit memory for a starting session. Wired to Claude Code's SessionStart.

    Prints to stdout, which the hook feeds into the model's context — so the
    agent reads it without having to ask for it.
    """
    from .observer import hook_payload, recall_context
    from .store import project_scope

    try:
        payload = hook_payload(sys.stdin.read()) if not sys.stdin.isatty() else {}
        session_id = payload.get("session_id")
        # Deliberately not `_stores`: that fires the one-time greeting, and a
        # hook running before the user has typed anything would spend it on
        # nobody.
        store, _ = open_store(db or os.environ.get("NENAPU_DB"))
        # The line the whole project-scoping task exists for. Called without a
        # scope, this injected the top facts across every repo on the machine,
        # so a session in one backend was told about another project's Ollama
        # context window.
        cwd = payload.get("cwd") or os.getcwd()
        text = recall_context(
            store, scope=project_scope(cwd), cwd=cwd, session_id=session_id,
        )
        _open_activity_session(store, session_id, cwd)
    except Exception:  # noqa: BLE001 — a hook must never break the session
        raise typer.Exit(0)
    if text:
        print(text)


def _agent_name() -> str:
    """Which agent is running. Everything wired through hooks today is Claude
    Code; the watcher will set this per adapter."""
    return os.environ.get("NENAPU_AGENT") or "claude-code"


def _open_activity_session(store, session_id: str | None, cwd: str | None) -> None:
    """Record the starting commit while it is still knowable.

    Session start is the only moment `git_head_before` can be read honestly,
    and it is what "changed since you were last here" later diffs from.
    """
    from .activity import ActivityLedger
    from .capture import open_session

    open_session(
        ActivityLedger(store.conn),
        agent=_agent_name(),
        external_id=session_id,
        cwd=cwd or os.getcwd(),
    )


def _capture_activity(path: str, session_id: str | None, db: str | None) -> None:
    """Fold a finished session's tool calls and git diff into the ledger.

    Deterministic and free, so it runs on its own rather than behind the
    extraction: a session whose model backend is unavailable — or that was
    read with `--no-infer` — still leaves a record of what it touched.
    """
    from .activity import ActivityLedger
    from .capture import capture_session
    from .loops import LoopBook

    try:
        store, _ = open_store(db or os.environ.get("NENAPU_DB"))
        ledger = ActivityLedger(store.conn)
        row_id = capture_session(ledger, Path(path), agent=_agent_name())
        if row_id is not None:
            # Files changed and nothing committed is work left in flight, and
            # it is knowable without asking a model anything.
            LoopBook(store.conn).detect_interrupted(ledger, row_id)
    except Exception:  # noqa: BLE001 — the ledger must never break a session
        return


# The log holds the text of every fact the extraction wrote, which is the same
# private content the store itself is kept at 0600 for. It also grows once per
# session forever.
MAX_LOG_BYTES = 512_000


def _open_observe_log(path: Path):
    """Append to the observe log, owner-only and bounded.

    Two things were wrong with `open(path, "a")`. It creates with the process
    umask — 0664 here — so the extracted facts sat world-readable beside a
    store that had just been locked to 0600; the directory was covering for it.
    And nothing ever bounded the file: a hook that appends after every session
    and never rotates only goes one way.
    """
    if path.exists() and path.stat().st_size > MAX_LOG_BYTES:
        # One generation back is enough to debug the failure that just
        # happened, which is all this log is for.
        os.replace(path, path.with_suffix(".log.1"))
    fd = os.open(path, os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o600)
    try:
        os.chmod(path, 0o600)  # a log from an older version is tightened too
    except OSError:
        pass
    return os.fdopen(fd, "a")


def _queue_and_detach(path: str, session_id: str | None, db: str | None) -> None:
    """Queue the transcript and hand it to a detached worker.

    The hook's whole job: one `ingest_queue` row, one background `nenapu
    drain`, and return. It deliberately does no extraction and no capture of
    its own — both belong to the worker, so that everything downstream of the
    queue (`run_maintenance_tick`, loop closure, the ledger) also runs on a
    machine that only has hooks and never starts the watcher.

    Serialising through the queue is what ends the fan-out: two sessions
    ending together now queue two jobs that one worker reads in turn, rather
    than starting two concurrent 83-second model calls against one store.
    """
    from .ingest_queue import enqueue_once

    try:
        store, _ = open_store(db or os.environ.get("NENAPU_DB"))
        enqueue_once(store.conn, path=path, agent="claude-code", session_id=session_id)
    except Exception:  # noqa: BLE001 — an unwritable store must not fail the session
        return
    _spawn_worker(db)


def _spawn_worker(db: str | None) -> None:
    """Start a detached `nenapu drain` and do not wait for it.

    `start_new_session` puts the child in its own process group, so the
    harness tearing down the session's process tree does not take the
    extraction with it. Output goes to a log rather than to the pipe the hook
    is being read on, because a hook that prints after it has returned
    corrupts whatever is reading it.

    A second worker started while one is draining finds `WorkerLock` taken and
    returns, leaving the jobs where they are, so spawning unconditionally is
    cheap rather than duplicated work.
    """
    import subprocess

    log_dir = Path(db).expanduser().parent if db else Path("~/.nenapu").expanduser()
    try:
        log_dir.mkdir(parents=True, exist_ok=True)
        log = _open_observe_log(log_dir / "observe.log")
    except OSError:
        log = subprocess.DEVNULL

    # `sys.argv[0]` is the installed entry point when a hook invoked us, but it
    # is a source path under `python -m nenapu.cli`, which the child cannot
    # exec. Resolve the real console script, and fall back to the interpreter
    # we are already running under.
    import shutil

    entry = shutil.which("nenapu")
    argv = ([entry] if entry else [sys.executable, "-m", "nenapu.cli"]) + ["drain"]
    if db:
        argv += ["--db", db]
    # The worker calls the model, and when the backend is an agent CLI that CLI
    # fires its own Stop hook. The marker keeps that chain one level deep.
    env = dict(os.environ, NENAPU_NO_BANNER="1", NENAPU_OBSERVING="1")
    try:
        subprocess.Popen(argv, stdin=subprocess.DEVNULL, stdout=log, stderr=log,
                         start_new_session=True, env=env)
    except OSError:
        pass  # a hook must never break the session it is attached to


@app.command("learn", rich_help_panel=UPKEEP)
@alias("observe", UPKEEP)
def learn(
    transcript: str = typer.Argument("", help="Transcript to read; omit with --stdin"),
    from_stdin: bool = typer.Option(False, "--stdin", help="Read the hook payload on stdin"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Show what would be stored"),
    detach: bool = typer.Option(False, "--detach",
                                help="Hand the work to a background process and return"),
    no_infer: bool = typer.Option(
        False, "--no-infer",
        help="Store turns verbatim and skip the model (needs NENAPU_STORE_MESSAGES=1)",
    ),
    db: str = DB_OPT,
) -> None:
    """Learn from a finished session without being asked.

    Reads the transcript, extracts corrections, decisions and environment facts,
    and stores them. Wired to Claude Code's Stop hook, this is what makes the
    layer passive: the agent never has to decide to record anything.
    """
    from .observer import hook_payload, messages_from_transcript, observe_transcript, store_messages

    session_id = None
    path = transcript
    cwd = os.getcwd()
    if from_stdin:
        if os.environ.get("NENAPU_OBSERVING"):
            # A hook is firing *inside* an extraction. When the model backend
            # is an agent CLI, that CLI is a harness in its own right and fires
            # its own Stop hook when it finishes — which would start another
            # extraction, which would start another CLI. Measured: `claude -p`
            # does fire Stop. The extraction carries this marker in its
            # environment, so the inner hook stops here and the chain is one
            # level deep instead of unbounded. Only the hook path is guarded:
            # the extraction itself is invoked with a plain path argument, and
            # refusing that would mean never observing anything at all.
            raise typer.Exit(0)
        payload = hook_payload(sys.stdin.read())
        path = payload.get("transcript_path", "")
        session_id = payload.get("session_id")
        cwd = payload.get("cwd") or cwd
    if not path:
        if from_stdin:
            raise typer.Exit(0)  # a hook with no transcript is not an error
        raise typer.BadParameter("no transcript path (pass one, or --stdin from a hook)")

    if detach:
        # Extraction is a model call over an entire session — 83s against real
        # transcripts, and hooks are killed at their timeout. The hook queues
        # the work and a detached worker does all of it: capture included, so
        # the transcript is read once, by one process, holding one lock.
        _queue_and_detach(path, session_id, db)
        raise typer.Exit(0)

    # Below here the work is inline, so the ledger half runs first: the model
    # half can be skipped or unavailable, and none of that changes what the
    # session did to the files.
    _capture_activity(path, session_id, db)

    if no_infer:
        # Never calls the model — the whole point is a cheap way to inspect
        # what a transcript contains without spending an extraction on it.
        store, _ = (open_store(db or os.environ.get("NENAPU_DB")) if from_stdin
                    else _stores(db))
        pairs = messages_from_transcript(Path(path))
        stored = store_messages(store.conn, session_id, pairs)
        if from_stdin:
            raise typer.Exit(0)
        if stored:
            console.print(f"{stored} message(s) stored verbatim")
        else:
            console.print(
                "[dim]nothing stored — set NENAPU_STORE_MESSAGES=1 to enable[/]"
            )
        return

    try:
        # The detached child has no terminal to greet; opening plainly keeps
        # the first-run orientation for the run where a person is watching.
        store, _ = (open_store(db or os.environ.get("NENAPU_DB")) if from_stdin
                    else _stores(db))
        learned = observe_transcript(
            store, Path(path),
            session_id=session_id or os.environ.get("NENAPU_SESSION_ID"),
            # Which repo the session was in, so what it taught is retrieved,
            # scoped and written per project rather than into one global pile.
            cwd=cwd,
            apply=not dry_run,
        )
    except LLMUnavailable as exc:
        if from_stdin:
            raise typer.Exit(0)  # never break a session over a missing model
        console.print(f"[red]cannot observe:[/] {exc}")
        raise typer.Exit(1)
    except Exception as exc:  # noqa: BLE001
        if from_stdin:
            raise typer.Exit(0)
        console.print(f"[red]observe failed:[/] {exc}")
        raise typer.Exit(1)

    if from_stdin:
        raise typer.Exit(0)  # quiet in a hook
    if not learned:
        console.print("[dim]nothing durable to learn from that session[/]")
        return
    for fact in learned:
        marker = "[yellow]correction[/]" if fact.kind == "feedback" else fact.kind
        console.print(f"  {marker:<22} {fact.text[:70]}")
    console.print(f"\n{len(learned)} fact(s) {'would be ' if dry_run else ''}remembered")


def _setup_walkthrough(console_out, *, yes: bool = False) -> None:
    """Wire Nenapu into whatever agent is actually on this machine.

    Memory an agent has to *ask* for is memory it will forget to ask for. So
    the wiring installed here is not a tool the agent may call — it is a pair
    of hooks: one that puts what you have learned into the session before the
    agent starts, and one that reads the finished session and records what it
    taught.
    """
    from .setup_wizard import (
        CLIENT_CONFIGS, detected, install_hooks, wire_claude_code, wire_json_client,
    )

    found = detected()
    if not found:
        console_out.print("  [yellow]No supported agent found on this machine.[/]")
        console_out.print("  [dim]Nenapu still works standalone: nenapu remember / search.[/]\n")
        return

    console_out.print("  [bold]Found on this machine[/]\n")
    for target in found:
        tag = "[green]agent[/]" if target.kind == "mcp" else "[dim]model host[/]"
        console_out.print(f"    {target.label:<14} {tag}  [dim]{target.note}[/]")
    console_out.print()

    hosts = [t for t in found if t.kind == "mcp"]
    if not hosts:
        console_out.print("  [yellow]Only model hosts found — nothing to attach memory to.[/]")
        console_out.print("  [dim]Ollama and LM Studio serve models; they are not plugin hosts."
                          " Nenapu will use one for its audit pass.[/]\n")
        return

    claude = next((t for t in hosts if t.key == "claude"), None)
    if claude:
        console_out.print("  [bold]Claude Code — the observing layer[/]\n")
        # Never edit someone's settings.json without being asked. A bare
        # `nenapu` in a pipe or a CI job is not consent, and `--yes` exists so
        # a script can say so explicitly.
        agreed = yes or (sys.stdin.isatty() and typer.confirm(
            "    Install hooks so memory is injected and learned automatically?",
            default=True,
        ))
        if agreed:
            ok, detail = install_hooks()
            console_out.print(f"    {'[green]✓[/]' if ok else '[red]✗[/]'} hooks — {detail}")
            ok, detail = wire_claude_code()
            console_out.print(f"    {'[green]✓[/]' if ok else '[dim]·[/]'} tools — {detail}"
                              "  [dim](optional; the hooks do the work)[/]")
        else:
            console_out.print("    [dim]not wired. `nenapu init --yes` when you want"
                              " it.[/]")
        console_out.print()

    others = [t for t in hosts if t.key != "claude"]
    if others:
        console_out.print("  [bold]Other editors[/]  [dim]no hook API — tools + a rules"
                          " block[/]\n")
        for target in others:
            path = CLIENT_CONFIGS.get(target.key)
            if not path:
                continue
            if not (yes or sys.stdin.isatty()):
                console_out.print(f"    [dim]·[/] {target.label} — would add to {path}")
                continue
            ok, detail = wire_json_client(path)
            console_out.print(
                f"    {'[green]✓[/]' if ok else '[red]✗[/]'} {target.label} — {detail}")
        console_out.print("\n    [dim]Paste into your agent's rules so it writes memory"
                          " too:[/]")
        console_out.print("    [dim]./AGENTS.md — see `nenapu rules`[/]")
        console_out.print()


# The first-run guide. Four beats, in the order someone actually meets them:
# what happens on its own, what to type, what to do when it is wrong, and how
# to see inside. Shown once and then never again — `nenapu guide` brings it
# back, which is cheaper than making someone regret pressing enter.
GUIDE = [
    ("What happens without you", [
        ("a session starts", "memory goes into its context"),
        ("you correct it", "the correction is recorded at the end"),
        ("next session", "it knows, and does not repeat it"),
    ]),
    ("What you type", [
        ("nenapu list", "everything it has learned"),
        ("nenapu recall \"port\"", "recall, by match and by belief"),
        ("nenapu remember \"...\"", "tell it something yourself"),
    ]),
    ("When a memory is wrong", [
        ("nenapu forget <id>", "retire one; nothing is deleted"),
        ("nenapu doubts", "what it no longer stands behind"),
        ("nenapu why <id>", "what it rests on, and rests on it"),
    ]),
    ("Checking on it", [
        ("nenapu pet", "how the store is doing, with a face on it"),
        ("nenapu learn <file> --dry-run", "what it would learn"),
        ("nenapu doctor --calibrate", "prove the model can audit"),
        ("nenapu approve", "no check runs until you say so"),
    ]),
]

# Widest left-hand cell, so the two columns line up across all four sections
# without any of them having to be padded by hand.
_GUIDE_WIDTH = max(len(left) for _, rows in GUIDE for left, _ in rows)


def _usage_guide(console_out) -> None:
    """The how-to-use half of the first run.

    Laid out as a grid rather than padded f-strings: at 80 columns a fixed
    pad puts the right-hand column past the edge and Rich wraps it under the
    left one, which looks like a bug.
    """
    from rich.table import Table

    for heading, rows in GUIDE:
        console_out.print(f"  [bold]{heading}[/]\n")
        grid = Table.grid(padding=(0, 2))
        grid.add_column(no_wrap=True, width=_GUIDE_WIDTH + 2)
        grid.add_column(overflow="fold")
        for left, right in rows:
            grid.add_row(f"  [cyan]{left}[/]", f"[dim]{right}[/]")
        console_out.print(grid)
        console_out.print()
    console_out.print("  [dim]Shown once. `nenapu guide` brings it back,"
                      " `nenapu init` re-runs setup.[/]\n")


@app.command(rich_help_panel=DIAGNOSE)
def init(
    yes: bool = typer.Option(False, "--yes", help="Accept the detected defaults"),
    watch: bool = typer.Option(False, "--watch",
                               help="Also install the background watcher unit"),
    db: str = DB_OPT,
) -> None:
    """Set Nenapu up as a layer over the agent you already use."""
    from .banner import mark_walked

    store = None
    try:
        store, _ = open_store(db or os.environ.get("NENAPU_DB"))
    except Exception:  # noqa: BLE001
        pass

    _big_panel(console, db, store=store)
    _setup_walkthrough(console, yes=yes)
    if watch:
        _install_watch_unit(console, yes=yes)
    _usage_guide(console)
    if store is not None:
        # Someone who ran setup by hand should not be walked through it again
        # the next time they type a bare `nenapu`.
        mark_walked(store.conn)


WATCH_UNIT_PATH = Path("~/.config/systemd/user/nenapu-watch.service")

WATCH_UNIT = """\
[Unit]
Description=Nenapu transcript watcher

[Service]
Type=simple
ExecStart=%s watch
Restart=on-failure
RestartSec=30

[Install]
WantedBy=default.target
"""


def _install_watch_unit(console_out, *, yes: bool = False) -> None:
    """Install the user unit that keeps the watcher running.

    Under the same consent rule as the hooks: this writes into a directory the
    user owns, so a pipe or a CI job is not permission. Without a terminal it
    says what it would write and writes nothing.
    """
    import shutil

    path = WATCH_UNIT_PATH.expanduser()
    if not (yes or sys.stdin.isatty()):
        console_out.print(f"  [dim]·[/] would install {path}")
        return

    entry = shutil.which("nenapu") or f"{sys.executable} -m nenapu.cli"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        shutil.copy(path, path.with_suffix(".service.nenapu-backup"))
    path.write_text(WATCH_UNIT % entry)
    console_out.print(f"  [green]✓[/] watcher unit written to {path}")
    console_out.print("  [dim]systemctl --user enable --now nenapu-watch[/]")


@app.command(rich_help_panel=UPKEEP)
def drain(
    limit: int = typer.Option(20, "--limit", help="Most jobs to process in one pass"),
    db: str = DB_OPT,
) -> None:
    """Extract from whatever the queue is holding, one worker at a time.

    What the Stop hook spawns, and what `watch` calls after a tick. A second
    worker finds the lock taken and returns, which is what keeps sessions
    ending together from becoming concurrent model calls against one store.
    """
    from .worker import drain as drain_queue

    store, _ = open_store(db or os.environ.get("NENAPU_DB"))
    done = drain_queue(store, limit=limit)
    console.print(f"extracted {done} session(s)")


def _probe_adapters() -> None:
    """Print what every registered glob matches on this machine.

    An adapter may only be registered for an agent someone has real
    transcripts from, so this is the first thing to run on a machine that has
    Codex, Gemini, OpenCode or Cursor installed: it says whether the glob you
    are about to write down matches anything there. It reads the filesystem
    and nothing else — no store is opened and nothing is queued, because a
    probe that ingested would spend an extraction per matched file answering
    the question.
    """
    from .watch import probe

    table = Table()
    table.add_column("agent", style="cyan")
    table.add_column("glob")
    table.add_column("matches here", justify="right")
    table.add_column("newest", style="dim")
    for found in probe():
        newest = Path(found["newest"]).name if found["newest"] else "-"
        table.add_row(found["agent"], found["glob"], str(found["matched"]), newest)
    console.print(table)
    console.print("[dim]nothing was queued: a probe reads the filesystem only[/]")


@app.command(rich_help_panel=UPKEEP)
def watch(
    once: bool = typer.Option(False, "--once", help="Run a single pass and exit"),
    batch: bool = typer.Option(False, "--batch",
                               help="Enqueue the whole backlog rather than one session"),
    probe_only: bool = typer.Option(False, "--probe",
                                    help="Report what each glob matches here, and stop"),
    interval: float = typer.Option(60.0, help="Seconds between passes"),
    db: str = DB_OPT,
) -> None:
    """Watch for finished sessions from agents that have no hook API."""
    import time as _time

    from .watch import tick
    from .worker import drain

    if probe_only:
        _probe_adapters()
        return

    store, _ = open_store(db or os.environ.get("NENAPU_DB"))
    while True:
        queued = tick(store.conn, batch=batch)
        if queued:
            console.print(f"queued {len(queued)} session(s) for extraction")
        # Enqueueing without draining would be a watcher that reports work and
        # learns nothing. One worker, holding a lock, does both here.
        done = drain(store)
        if done:
            console.print(f"extracted {done} session(s)")
        if once:
            return
        _time.sleep(interval)


@app.command(rich_help_panel=DIAGNOSE)
def guide() -> None:
    """Show the first-run how-to guide again."""
    console.print()
    _usage_guide(console)


@app.command(rich_help_panel=DIAGNOSE)
def rules() -> None:
    """Print the rules block for agents that have no hook API."""
    from .setup_wizard import AGENT_RULES

    print(AGENT_RULES)


@app.command(rich_help_panel=DIAGNOSE)
def theme(
    name: str = typer.Argument("", help="Theme to switch to; omit to see them all"),
) -> None:
    """Switch the banner colour, and remember the choice.

    Saved to ~/.nenapu/config.json. `NENAPU_THEME` overrides it for a single
    run, so a script can pin `mono` without disturbing your setting.
    """
    from .banner import HERO, THEMES, resolve_theme, write_config

    if not name:
        current = resolve_theme()
        console.print()
        for theme_name, shades in THEMES.items():
            marker = "[bold]●[/]" if theme_name == current else " "
            swatch = "".join(f"[{shade}]███[/]" for shade in shades)
            console.print(f"  {marker} {theme_name:<8} {swatch}")
        suffix = "  (from NENAPU_THEME)" if os.environ.get("NENAPU_THEME") else ""
        console.print(f"\n  [dim]active: {current}{suffix}[/]")
        console.print("  [dim]nenapu theme <name> to switch[/]\n")
        return

    key = name.lower()
    if key not in THEMES:
        raise typer.BadParameter(f"unknown theme {name!r}; try: {', '.join(THEMES)}")

    write_config(theme=key)
    shades = THEMES[key]
    console.print()
    for i, line in enumerate(HERO):
        console.print(f"[{shades[i % len(shades)]}]{line}[/]")
    console.print(f"\n  [green]theme set to {key}[/]")
    if os.environ.get("NENAPU_THEME"):
        console.print("  [yellow]NENAPU_THEME is set and overrides this[/]")
    console.print()


@app.command(rich_help_panel=DIAGNOSE)
def pet(
    watch: bool = typer.Option(False, "--watch", help="Stay open and keep looking"),
    line_only: bool = typer.Option(False, "--line", help="One line, for a status bar"),
    compact: bool = typer.Option(False, "--compact", help="Just the face, no drawing"),
    as_json: bool = typer.Option(False, "--json", help="The numbers behind the face"),
    db: str = DB_OPT,
) -> None:
    """How the store is doing, with a face on it.

    Everything here is already in `nenapu stats`. Nobody reads `nenapu stats`:
    eleven numbers have no opinion about whether anything is wrong, so the
    memory debt sits there for a week. The pet cannot look happy while a check
    is failing, which is the only reason it is worth having.
    """
    from .banner import hero_shades
    from .pet import FULL_MIN_WIDTH, assess, line, render, render_full

    store, _ = _stores(db)
    creature = assess(store)
    full = console.width >= FULL_MIN_WIDTH and not compact

    if as_json:
        print(json.dumps({"mood": creature.mood, "face": creature.face,
                          "blurb": creature.blurb, "stats": creature.stats}, default=str))
        return
    if line_only:
        print(line(creature))
        return
    def frame(blink: bool = False):
        return (render_full(creature, hero_shades(), blink=blink) if full
                else _render_text(render(creature, blink=blink)))

    if not watch:
        console.print()
        console.print(frame())
        console.print()
        return

    # Watch mode re-reads the store rather than the snapshot, so a fact written
    # in another terminal shows up here — which is the only reason to leave it
    # open at all.
    from rich.live import Live

    blink = False
    try:
        with Live(console=console, refresh_per_second=4, transient=False) as live:
            while True:
                creature = assess(store)
                live.update(frame(blink))
                blink = not blink
                # A blink is short and the gap between them is long, which is
                # the difference between a creature and a flashing cursor.
                time.sleep(0.18 if blink else 3.5)
    except KeyboardInterrupt:
        console.print("[dim]bye[/]")


def _render_text(markup: str):
    from rich.text import Text

    return Text.from_markup("\n" + markup + "\n")


@app.command(rich_help_panel=DIAGNOSE)
def doctor(
    calibrate_backend: bool = typer.Option(
        False, "--calibrate", help="Probe whether the model actually reads evidence"
    ),
    repeats: int = typer.Option(3, help="Runs per probe; one sample is not a measurement"),
    db: str = DB_OPT,
) -> None:
    """Show which model backend the scheduled passes will use.

    With --calibrate, run three probes against it. Accuracy alone cannot tell a
    careful auditor from one that answers 'stale' unconditionally; the probes
    can, because they change the evidence and check whether the verdicts change
    with it.
    """
    from .calibrate import calibrate as run_calibration
    from .llm import LLMUnavailable, available

    store, _ = _stores(db)
    ok, detail = available()
    path = store.conn.execute("PRAGMA database_list").fetchone()[2] or ":memory:"
    console.print(f"store    : {path}")
    console.print(f"backend  : {'[green]' + detail + '[/]' if ok else '[yellow]' + detail + '[/]'}")
    console.print("[dim]core loop (decay, checks, cascade, grading) needs no model at all[/]")

    if not calibrate_backend:
        if ok:
            console.print("[dim]run `nenapu doctor --calibrate` before trusting it to audit[/]")
        return
    if not ok:
        raise typer.Exit(1)

    console.print(
        f"\ncalibrating — three probes x {repeats} runs, no data of yours involved..."
    )
    try:
        result = run_calibration(repeats=repeats)
    except LLMUnavailable as exc:
        console.print(f"[red]calibration failed:[/] {exc}")
        raise typer.Exit(1)

    table = Table()
    for col in ("evidence", "expected", "agreement", "spread", "covered", "verdicts", "time"):
        table.add_column(col)
    for r in result.results:
        low, high = r.spread
        table.add_row(
            r.condition, "/".join(sorted(r.expected)),
            "-" if r.error else f"{r.agreement:.0%}",
            "-" if r.error else f"{low:.0%}-{high:.0%}",
            f"{r.covered}/{r.requested}",
            ",".join(sorted(set(r.verdicts))) or (r.error or "-")[:30],
            f"{r.seconds:.0f}s",
        )
    console.print(table)

    store.conn.execute(
        "INSERT OR REPLACE INTO meta(key, value) VALUES (?, ?)",
        (f"calibration:{result.backend}", "pass" if result.passed else "fail"),
    )
    store.conn.commit()

    if result.passed:
        console.print(f"[green]passed[/] — verdicts track the evidence "
                      f"({result.accuracy:.0%} agreement). Safe to run audits.")
    else:
        console.print("[red]failed[/] — do not let this model audit your memory:")
        for problem in result.failures:
            console.print(f"  - {problem}")
        console.print(
            "[dim]local models under ~7B tend to fail the confirming probe: they read "
            "any mention of a fact as evidence it changed. Use a cloud backend, or "
            "run audits with --dry-run and read the findings yourself.[/]"
        )
        raise typer.Exit(1)


@app.command(rich_help_panel=DIAGNOSE)
def cost(db: str = DB_OPT) -> None:
    """Token footprint of the memory layer.

    The number that matters is the MCP tool surface: it sits in the context
    window on every request whether or not memory is used.
    """
    import asyncio
    import json as _json

    from .export import render
    from .mcp_server import _fact_view, mcp

    store, _ = _stores(db)

    def tokens(text: str) -> int:
        return len(text) // 4  # chars/4; exact counts need a provider tokenizer

    tools = asyncio.run(mcp.list_tools())
    surface = sum(
        tokens(_json.dumps({"name": t.name, "description": t.description or "",
                            "input_schema": t.input_schema}))
        for t in tools
    )
    hits = store.search("", limit=8, log_recall=False)
    recall = tokens(_json.dumps([_fact_view(f, s, w) for f, s, w in hits]))
    block = tokens(render(store))

    table = Table()
    for col in ("surface", "est. tokens", "paid"):
        table.add_column(col)
    table.add_row(f"MCP tools ({len(tools)})", str(surface), "every request")
    table.add_row("recall of 8 facts", str(recall), "per search")
    table.add_row("CLAUDE.md block", str(block), "per session, if exported")
    console.print(table)
    console.print("[dim]estimates at 4 chars/token[/]")


@app.command(rich_help_panel=DIAGNOSE)
def serve(host: str = "127.0.0.1", port: int = 8787, db: str = DB_OPT) -> None:
    """Run the local HTTP API."""
    import uvicorn

    from .api import create_app

    uvicorn.run(create_app(db or os.environ.get("NENAPU_DB")), host=host, port=port)


# ---------- skills ----------


@skill_app.command("add")
def skill_add(
    name: str,
    body_file: str = typer.Argument(..., help="File containing the skill document"),
    description: str = "",
    scope: str = "global",
    db: str = DB_OPT,
) -> None:
    """Save a skill from a file."""
    _, skills = _stores(db)
    saved = skills.upsert(
        Skill(name=name, body=Path(body_file).read_text(), description=description, scope=scope)
    )
    console.print(f"[green]saved[/] {saved.name} (#{saved.id})")


@skill_app.command("list")
def skill_list(status: str = "active", db: str = DB_OPT) -> None:
    """List skills with their track record."""
    _, skills = _stores(db)
    table = Table()
    for col in ("name", "status", "runs", "success", "last used", "reason"):
        table.add_column(col)
    for s in skills.list_skills(status=status or None):
        rate = "-" if s.success_rate is None else f"{s.success_rate:.0%}"
        last = "never" if not s.last_used_at else f"{(now() - s.last_used_at) / DAY:.0f}d ago"
        table.add_row(s.name, s.status, str(s.invocations), rate, last, s.quarantine_reason or "")
    console.print(table)


@skill_app.command("outcome")
def skill_outcome(
    name: str,
    outcome: str = typer.Argument(..., help="success | failure | used"),
    note: str = "",
    db: str = DB_OPT,
) -> None:
    """Record how a skill actually went."""
    _, skills = _stores(db)
    updated = skills.record_outcome(name, outcome, note=note or None)
    if not updated:
        raise typer.BadParameter(f"no skill named {name!r}")
    msg = f"{updated.name}: {updated.invocations} runs"
    if updated.success_rate is not None:
        msg += f", {updated.success_rate:.0%} success"
    console.print(msg)
    if updated.status == "quarantined":
        console.print(f"[yellow]quarantined[/] — {updated.quarantine_reason}")


@skill_app.command("sweep")
def skill_sweep(db: str = DB_OPT) -> None:
    """Quarantine skills that no longer earn their place."""
    _, skills = _stores(db)
    culled = skills.sweep()
    if not culled:
        console.print("[dim]nothing to quarantine[/]")
    for s in culled:
        console.print(f"[yellow]quarantined[/] {s.name} — {s.quarantine_reason}")


@skill_app.command("revive")
def skill_revive(name: str, db: str = DB_OPT) -> None:
    """Bring a quarantined skill back."""
    _, skills = _stores(db)
    revived = skills.revive(name)
    if not revived:
        raise typer.BadParameter(f"no skill named {name!r}")
    console.print(f"[green]active[/] {revived.name}")


if __name__ == "__main__":
    app()
