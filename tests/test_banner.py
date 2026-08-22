"""The greeting must be a greeting, not a recurring interruption."""

import pytest

from nenapu import connect
from nenapu.banner import FIRST_RUN_HELP, HERO, HERO_ASCII, STAMP, should_greet


def test_greets_once_per_store(tmp_path):
    conn = connect(str(tmp_path / "s.db"))
    assert should_greet(conn) is True
    assert should_greet(conn) is False
    conn.close()

    reopened = connect(str(tmp_path / "s.db"))
    assert should_greet(reopened) is False, "a greeting on every run is spam"


def test_ascii_fallback_exists_for_terminals_without_block_drawing():
    assert any("█" in line for line in HERO)
    assert not any("█" in line for line in HERO_ASCII)


def test_the_wordmark_fits_an_eighty_column_terminal():
    """Wider than this and the hero wraps, which looks broken rather than big."""
    assert max(len(line) for line in HERO) <= 78
    assert max(len(line) for line in HERO_ASCII) <= 78


def test_the_wordmark_spells_the_name():
    """Type at scale survives a terminal; small illustrations do not — earlier
    figurative marks read as a bent pipe and a jolly roger."""
    assert len(HERO) >= 5, "too short to read as a hero banner"


def test_first_run_help_points_at_the_approval_gate():
    """A new user should learn that checks need approval before they write one
    and wonder why it never ran."""
    text = FIRST_RUN_HELP.format(path="/tmp/x.db")
    assert "nenapu approve" in text
    assert "/tmp/x.db" in text


def test_help_is_grouped_not_a_wall_of_commands():
    """Twenty-two commands in one flat list is unusable. Panels group them by
    the question a user is answering."""
    from typer.testing import CliRunner

    from nenapu.cli import app

    result = CliRunner().invoke(app, ["--help"])
    assert result.exit_code == 0
    for panel in ("Remember and recall", "Belief network", "Trust and upkeep",
                  "Did it help?", "Setup and diagnostics"):
        assert panel in result.output, f"missing panel: {panel}"


def test_help_text_does_not_carry_the_art():
    """Rich reflows help text and breaks the knot's alignment. The mark belongs
    where whitespace survives — `version` and the first-run greeting."""
    from typer.testing import CliRunner

    from nenapu.cli import app

    help_output = CliRunner().invoke(app, ["--help"]).output
    version_output = CliRunner().invoke(app, ["version"]).output
    assert "█████" not in help_output
    assert "█████" in version_output


# These run the real executable. Rich binds `err_console` to sys.stderr at
# import time, so pytest's capture never sees it — an in-process test would
# assert on the wrong stream and pass for the wrong reason.

import os
import subprocess
import sys

CLI = [sys.executable, "-c",
       "import sys; sys.argv[0]='nenapu'; from nenapu.cli import app; app()"]


def _run(args, tmp_path, **env):
    return subprocess.run(
        CLI + args,
        capture_output=True, text=True,
        env={**os.environ, "NENAPU_DB": str(tmp_path / "s.db"),
             "PYTHONPATH": "src", **env},
    )


def test_banner_shows_on_every_invocation(tmp_path):
    """Asked for explicitly: the mark should be there whenever the tool runs,
    not only on first use."""
    first = _run(["stats"], tmp_path, NENAPU_NO_BANNER="")
    second = _run(["stats"], tmp_path, NENAPU_NO_BANNER="")

    assert STAMP in first.stderr
    assert STAMP in second.stderr, "the mark vanished after the first run"


def test_banner_goes_to_stderr_so_piped_output_stays_clean(tmp_path):
    """`nenapu recall --json | jq` must receive data, not ASCII art."""
    result = _run(["search", "anything", "--json"], tmp_path, NENAPU_NO_BANNER="")
    assert STAMP not in result.stdout
    assert STAMP in result.stderr


def test_no_banner_env_silences_it(tmp_path):
    """Cron and CI logs should not accumulate the mark."""
    result = _run(["stats"], tmp_path, NENAPU_NO_BANNER="1")
    assert STAMP not in result.stderr


def test_bare_invocation_is_not_an_error(tmp_path):
    """Someone typing `nenapu` to look around is not making a usage error, and
    a non-zero exit breaks any wrapper script."""
    assert _run([], tmp_path).returncode == 0


def test_routine_commands_get_one_line_not_the_whole_dog(tmp_path):
    """Art that repeats before every `search` stops being charming fast."""
    result = _run(["stats"], tmp_path, NENAPU_NO_BANNER="")
    assert STAMP in result.stderr
    assert "█████" not in result.stderr


def test_looking_around_gets_the_full_panel(tmp_path):
    """A bare invocation is someone asking what this is."""
    result = _run([], tmp_path, NENAPU_NO_BANNER="")
    assert "█████" in result.stdout
    assert "facts" in result.stdout


def test_the_stop_hook_command_does_not_leak_the_banner(tmp_path):
    """Requirement (Task 0b, priority-ordered task list): `learn` must be in
    the quiet-suppression tuple alongside its hidden alias `observe`.

    `cli.py:146` suppresses the version stamp for machine-invoked commands by
    name: `("version", "recall-hook", "observe")`. Nine commands were renamed
    to plain words (commit `8a33995`) and the Stop hook now runs `nenapu learn
    --stdin --detach`, not `nenapu observe --stdin --detach` — but the
    suppression tuple was never updated to match. The old alias still works
    (it stays hidden, see test_command_names.py), which is exactly what let
    this drift ship unnoticed: every session ending would print the version
    stamp to stderr, exactly the "hook error in chat" symptom the P0 bug
    report describes, just from a different cause.
    """
    result = _run(["learn", "--stdin"], tmp_path, NENAPU_NO_BANNER="")
    assert STAMP not in result.stderr, (
        "the Stop hook now leaks the version stamp on every session end"
    )


def test_the_old_alias_still_suppresses_the_banner(tmp_path):
    """Locks in the half of the tuple that already works, so a future edit
    cannot fix `learn` by accidentally dropping `observe`."""
    result = _run(["observe", "--stdin"], tmp_path, NENAPU_NO_BANNER="")
    assert STAMP not in result.stderr


def test_the_panel_reports_the_store(tmp_path):
    _run(["write", "a remembered thing"], tmp_path, NENAPU_NO_BANNER="1")
    result = _run(["version"], tmp_path, NENAPU_NO_BANNER="")
    assert "1 active" in result.stdout


def test_theme_persists_across_runs(tmp_path):
    """Switching a theme should stick, the way Hermes' skins do."""
    home = tmp_path / "home"
    env = {"NENAPU_HOME": str(home), "NENAPU_THEME": ""}

    listed = _run(["theme"], tmp_path, **env)
    assert "teal" in listed.stdout and "violet" in listed.stdout

    assert _run(["theme", "violet"], tmp_path, **env).returncode == 0
    assert "violet" in (home / "config.json").read_text()

    again = _run(["theme"], tmp_path, **env)
    assert "active: violet" in again.stdout


def test_env_overrides_the_saved_theme_without_changing_it(tmp_path):
    """A script pinning `mono` must not rewrite the user's preference."""
    home = tmp_path / "home"
    _run(["theme", "jade"], tmp_path, NENAPU_HOME=str(home), NENAPU_THEME="")

    overridden = _run(["theme"], tmp_path, NENAPU_HOME=str(home), NENAPU_THEME="mono")
    assert "active: mono" in overridden.stdout
    assert "jade" in (home / "config.json").read_text(), "override rewrote the config"


def test_unknown_theme_is_rejected(tmp_path):
    result = _run(["theme", "chartreuse"], tmp_path,
                  NENAPU_HOME=str(tmp_path / "home"), NENAPU_THEME="")
    assert result.returncode != 0
    assert "unknown theme" in (result.stdout + result.stderr)


# ---------- the silenced greeting claims nothing ----------
#
# Plan "Harden the four incidents into guarantees", Phase A task A4.
# `_greet` evaluates `should_greet(store.conn)` before the environment check in
# the same `or` expression, and `should_greet` claims: it INSERTs into `meta`
# and commits. So a run that was told to print no banner writes a row anyway,
# and it is the one write every `--dry-run` command performs before it reaches
# its own guard.


def _greeted(tmp_path) -> bool:
    conn = connect(str(tmp_path / "s.db"))
    try:
        return conn.execute(
            "SELECT 1 FROM meta WHERE key = 'greeted'"
        ).fetchone() is not None
    finally:
        conn.close()


def test_a_silenced_run_claims_no_greeting(tmp_path):
    """Nothing was shown, so nothing was greeted, so nothing should be recorded."""
    _run(["search", "anything"], tmp_path, NENAPU_NO_BANNER="1")

    assert not _greeted(tmp_path), "a silenced run spent the one-time greeting"


def test_the_greeting_still_shows_after_a_silenced_run(tmp_path):
    """The claim exists so the orientation appears exactly once. A silenced run
    must not consume it, or the first watching person never sees it."""
    _run(["search", "anything"], tmp_path, NENAPU_NO_BANNER="1")
    result = _run(["search", "anything"], tmp_path, NENAPU_NO_BANNER="")

    assert "Your store lives at" in result.stderr


def test_an_unsilenced_run_still_claims_the_greeting(tmp_path):
    """The guard must not turn the one-time orientation into a recurring one."""
    _run(["search", "anything"], tmp_path, NENAPU_NO_BANNER="")

    assert _greeted(tmp_path)
