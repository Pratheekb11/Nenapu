"""Command line interface."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from . import __version__, open_store
from .audit import LLMUnavailable
from .audit import audit as run_audit
from .distill import distill as run_distill
from .export import render, write_file
from .models import Fact, Skill, Status
from .store import DAY, effective_confidence, now
from .verify import run_check, apply_result, verify_scope

# The knot lives in `nenapu version` and the first-run greeting, not here:
# Rich reflows help text and breaks the alignment, and art that has to survive
# a word-wrapper is art in the wrong place.
HELP = """\
[bold]ನೆನಪು  ·  n e n a p u[/] — memory that knows what it rests on.

A store, not an agent. Facts carry provenance, decay on a clock, prove
themselves with a command, and lose standing when what they rest on falls.

Start with [cyan]nenapu write[/], then [cyan]nenapu search[/]. See [cyan]nenapu loops[/] for anything
the store no longer stands behind.
"""

# Panels, because twenty-two commands in one list is a wall. Grouped by the
# question the user is answering, not by the module the command lives in.
REMEMBER = "Remember and recall"
NETWORK = "Belief network"
UPKEEP = "Trust and upkeep"
OUTCOMES = "Did it help?"
DIAGNOSE = "Setup and diagnostics"

app = typer.Typer(help=HELP, rich_markup_mode="rich")
skill_app = typer.Typer(help="Skill library with an outcome loop", no_args_is_help=True)
app.add_typer(skill_app, name="skill", rich_help_panel=UPKEEP)

console = Console()
# The banner goes to stderr so `nenapu search --json | jq` is never corrupted
# by a one-time greeting.
err_console = Console(stderr=True)


def _big_panel(console_out, db: str | None = None) -> None:
    """Dog plus a readout of the store. Opening the store is worth it here —
    someone asking for this view wants to see the state of their memory."""
    from .banner import panel
    from .llm import available

    conn = path = None
    try:
        store, _ = open_store(db or os.environ.get("NENAPU_DB"))
        conn = store.conn
        path = store.conn.execute("PRAGMA database_list").fetchone()[2]
    except Exception:  # noqa: BLE001 — a banner must never break the command
        pass

    ok, detail = available()
    # The URL is noise in a status line; the backend and model are the answer.
    backend = detail.split(" @ ")[0] if ok else ""
    console_out.print()
    console_out.print(panel(console_out, version=__version__, conn=conn,
                            path=path or "", backend=backend))
    console_out.print()


@app.callback(invoke_without_command=True)
def main(ctx: typer.Context) -> None:
    """Show the mark, sized to the occasion.

    The full panel is for someone looking around; every other command gets one
    dim line. Art that repeats before every `search` stops being charming by
    the third command. Both go to stderr, so `--json | jq` still pipes clean
    data. `NENAPU_NO_BANNER=1` silences it for cron and CI.
    """
    from .banner import stamp

    quiet = bool(os.environ.get("NENAPU_NO_BANNER"))

    if ctx.invoked_subcommand is None:
        if not quiet:
            _big_panel(console)
        # A bare `nenapu` is someone looking around, not a usage error — so
        # show the commands and exit clean, rather than Typer's exit code 2.
        console.print(ctx.get_help())
        raise typer.Exit(0)

    if not quiet and ctx.invoked_subcommand != "version":
        err_console.print(stamp(__version__))


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


DB_OPT = typer.Option(None, "--db", help="Path to the store (default ~/.nenapu/nenapu.db)")


@app.command(rich_help_panel=DIAGNOSE)
def version(plain: bool = typer.Option(False, "--plain", help="Version string only"),
            db: str = DB_OPT) -> None:
    """Print the version, with the mark."""
    if plain:
        console.print(__version__)
        return
    _big_panel(console, db)


@app.command(rich_help_panel=REMEMBER)
def write(
    text: str,
    kind: str = typer.Option("project", help="user|project|environment|feedback|reference"),
    scope: str = "global",
    key: str = typer.Option("", help="Contradiction join key, e.g. db.port"),
    origin: str = typer.Option("user_stated", help="user_stated|tool_observed|file_derived|agent_inferred"),
    confidence: float = 0.8,
    decay: str = typer.Option("", help="immutable|slow|medium|volatile"),
    verify_cmd: str = typer.Option("", help="Shell command that proves this fact"),
    verify_expect: str = typer.Option("", help="Substring the output must contain"),
    db: str = DB_OPT,
) -> None:
    """Store a fact."""
    store, _ = _stores(db)
    fact, conflicts = store.write(
        Fact(
            text=text, kind=kind, scope=scope, key=key or None, origin=origin,
            confidence=confidence, decay_class=decay or None,
            verify_cmd=verify_cmd or None, verify_expect=verify_expect or None,
        ),
        actor="cli",
    )
    console.print(f"[green]stored[/] #{fact.id}  belief={effective_confidence(fact):.2f}")
    for c in conflicts:
        colour = "yellow" if c.resolution == "superseded" else "red"
        console.print(f"[{colour}]conflict[/] with #{c.other_id}: {c.detail} -> {c.resolution}")


@app.command(rich_help_panel=REMEMBER)
def search(
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
def forget(fact_id: int, db: str = DB_OPT) -> None:
    """Retire a fact."""
    store, _ = _stores(db)
    if not store.get(fact_id):
        raise typer.BadParameter(f"no fact {fact_id}")
    store.forget(fact_id, actor="cli")
    console.print(f"[yellow]retired[/] #{fact_id}")


@app.command(rich_help_panel=UPKEEP)
def verify(
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
    a prompt injection that plants a fact cannot turn `nenapu verify` into
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


@app.command(rich_help_panel=UPKEEP)
def distill(
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


@app.command(rich_help_panel=NETWORK)
def loops(limit: int = 20, db: str = DB_OPT) -> None:
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


@app.command(rich_help_panel=OUTCOMES)
def good(recall_id: int, note: str = "", db: str = DB_OPT) -> None:
    """Mark a recall as having helped."""
    store, _ = _stores(db)
    ok = store.ledger.grade(recall_id, "good", source="human", note=note or None)
    console.print("[green]graded good[/]" if ok else "[dim]already graded, or no such recall[/]")


@app.command(rich_help_panel=OUTCOMES)
def bad(recall_id: int, note: str = "", db: str = DB_OPT) -> None:
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


@app.command(rich_help_panel=OUTCOMES)
def outcome(
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


@app.command(rich_help_panel=DIAGNOSE)
def stats(scope: str = "", db: str = DB_OPT) -> None:
    """Health of the store."""
    store, _ = _stores(db)
    s = store.stats(scope=scope or None)
    table = Table(show_header=False)
    for key, value in s.items():
        table.add_row(key, json.dumps(value) if isinstance(value, dict) else str(value))
    console.print(table)


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
