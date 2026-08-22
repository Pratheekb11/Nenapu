"""The command names, and the promise the old ones still keep.

Nine commands were renamed to plain words — `write` became `remember`, `loops`
became `doubts`, `observe` became `learn`. Renaming commands in a tool people
have already wired into hooks and shell scripts is a breaking change unless the
old word keeps working, so every old name survives as a hidden alias.
"""

import os
import subprocess
import sys

import pytest

import nenapu.cli as cli

RENAMED = {
    "write": "remember",
    "search": "recall",
    "verify": "check",
    "loops": "doubts",
    "distill": "tidy",
    "observe": "learn",
    "good": "helped",
    "bad": "misled",
    "outcome": "grade",
}


def registered() -> dict[str, bool]:
    """Every command name, mapped to whether it is hidden from the listing."""
    names = {}
    for command in cli.app.registered_commands:
        name = command.name or (command.callback.__name__.replace("_", "-")
                                if command.callback else "")
        names[name] = bool(command.hidden)
    return names


@pytest.mark.parametrize("old, new", RENAMED.items())
def test_the_new_name_is_the_one_advertised(old, new):
    names = registered()

    assert names.get(new) is False, f"{new} should be listed"
    assert names.get(old) is True, f"{old} should survive, hidden"


def test_the_landing_view_lists_each_command_once():
    """An unhidden alias would put every renamed command on the screen twice,
    which is worse than the jargon it replaced."""
    listed = [name for rows in cli._command_groups().values() for name, _ in rows]

    assert len(listed) == len(set(listed))
    for old in RENAMED:
        assert old not in listed


def test_the_stop_hook_uses_the_new_name():
    """The hook is written into someone's settings.json and read back by the
    upgrade check. If it kept saying `observe`, every install would look stale
    forever and be rewritten on every run."""
    from nenapu.setup_wizard import hook_config

    stop = str(hook_config()["Stop"])

    assert "nenapu learn --stdin --detach" in stop
    assert "observe" not in stop


@pytest.mark.parametrize("old", ["write", "search", "loops"])
def test_an_old_name_still_runs(old, tmp_path):
    """Someone's shell history, someone's script, someone's blog post."""
    db = str(tmp_path / "m.db")
    env = {**os.environ, "PYTHONPATH": "src", "NENAPU_NO_BANNER": "1"}
    args = {"write": ["write", "a fact"], "search": ["search", "fact"],
            "loops": ["loops"]}[old]

    result = subprocess.run([sys.executable, "-m", "nenapu.cli", *args, "--db", db],
                            capture_output=True, text=True, env=env)

    assert result.returncode == 0, result.stdout + result.stderr


def test_the_recall_hook_keeps_its_name():
    """This one is wired into ~/.claude/settings.json on every machine that has
    run `nenapu init`. Renaming it would break memory injection silently — the
    session would simply start knowing nothing."""
    assert registered().get("recall-hook") is True


# ==========================================================================
# Pre-written for G8 · make the human path reachable.
#
# `nenapu helped <recall_id>` and `nenapu misled <recall_id>` require a recall
# id that no command prints, so the human grading path is unreachable by
# anyone who has not read the schema. G8 adds a listing under the OUTCOMES
# panel — recall id, fact text, session, age — reusing `Ledger.pending`, plus
# one line in the MCP `task_outcome` description telling an agent to call it
# at task end (pinned in tests/test_mcp.py).
#
# Naming, and a correction to the plan: the plan calls the new command
# `nenapu pending`, but that name is taken — `cli.py:1036` registers `pending`
# under the ACTIVITY panel for open loops. Renaming that one is exactly the
# breaking change this file exists to prevent, and two commands cannot share a
# name. So the new listing is `nenapu ungraded`, which says what it lists, and
# the existing `pending` keeps its meaning. Change the constant below if the
# name is decided differently; the requirement is a reachable listing under
# OUTCOMES whose ids `misled` accepts, not the word.
# ==========================================================================


UNGRADED = "ungraded"


def _panels() -> dict[str, str]:
    panels = {}
    for command in cli.app.registered_commands:
        name = command.name or (command.callback.__name__.replace("_", "-")
                                if command.callback else "")
        panels[name] = command.rich_help_panel
    return panels


def _seeded_store(db):
    """A store with one pending recall, the way a session start leaves it."""
    from nenapu import connect
    from nenapu.models import Fact
    from nenapu.store import Store

    store = Store(connect(str(db)))
    fact, _ = store.write(Fact(text="the staging host is box-7"))
    store.ledger.log(fact.id, session_id="s-visible", query="")
    return store


def _run_cli(args, db):
    env = {**os.environ, "PYTHONPATH": "src", "NENAPU_NO_BANNER": "1"}
    return subprocess.run([sys.executable, "-m", "nenapu.cli", *args, "--db", str(db)],
                          capture_output=True, text=True, env=env)


def test_the_open_loops_listing_keeps_the_name_it_has():
    """`nenapu pending` means open loops today and is in the landing view.
    The new outcomes listing must not take the word off it."""
    assert _panels().get("pending") == cli.ACTIVITY


def test_the_ungraded_listing_is_under_the_outcomes_panel():
    """The panel is the question the user is answering. "Did it help?" is
    where a list of recalls awaiting a grade belongs."""
    assert _panels().get(UNGRADED) == cli.OUTCOMES
    assert registered().get(UNGRADED) is False


def test_the_listing_prints_ids_that_misled_accepts(tmp_path):
    """The whole point of the command: an id a person can act on. If the
    printed id cannot be handed straight to `nenapu misled`, the human path is
    still unreachable."""
    from nenapu import connect
    from nenapu.models import Outcome

    db = tmp_path / "g8.db"
    _seeded_store(db)

    listing = _run_cli([UNGRADED], db)
    ids = [int(word) for word in listing.stdout.split() if word.isdigit()]
    graded = _run_cli(["misled", str(ids[0])], db)

    assert listing.returncode == 0, listing.stdout + listing.stderr
    assert graded.returncode == 0, graded.stdout + graded.stderr
    row = connect(str(db)).execute("SELECT outcome FROM recalls").fetchone()
    assert row["outcome"] == Outcome.BAD


def test_the_listing_shows_the_fact_and_the_session_and_the_age(tmp_path):
    """An id with no text beside it is a number nobody can grade honestly."""
    db = tmp_path / "g8.db"
    _seeded_store(db)

    result = _run_cli([UNGRADED], db)

    assert "box-7" in result.stdout
    assert "s-visible" in result.stdout
    assert "ago" in result.stdout.lower() or "age" in result.stdout.lower()


def test_the_listing_says_so_when_nothing_is_waiting(tmp_path):
    db = tmp_path / "empty.db"

    result = _run_cli([UNGRADED], db)

    assert result.returncode == 0, result.stdout + result.stderr
    assert "no" in result.stdout.lower()


def test_the_listing_reuses_the_ledger_query():
    """`Ledger.pending` already answers this. A second query with its own idea
    of what pending means is how two answers to one question start
    disagreeing."""
    import inspect

    source = inspect.getsource(getattr(cli, UNGRADED))

    assert "ledger.pending" in source
