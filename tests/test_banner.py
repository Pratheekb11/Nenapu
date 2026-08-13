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
    """`nenapu search --json | jq` must receive data, not ASCII art."""
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
