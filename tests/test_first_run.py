"""The first run: set it up, explain it, then get out of the way.

Run as subprocesses because the thing under test is what a person sees in a
terminal, and because the walkthrough writes to real config paths — a test
that got that wrong would edit the machine it runs on.
"""

import os
import subprocess
import sys

import pytest

from nenapu import connect
from nenapu.banner import mark_walked, should_walk

CLI = [sys.executable, "-c",
       "import sys; sys.argv[0]='nenapu'; from nenapu.cli import app; app()"]


def _run(args, tmp_path, **env):
    """Run the CLI with HOME redirected, so nothing on this machine is touched."""
    home = tmp_path / "home"
    home.mkdir(exist_ok=True)
    return subprocess.run(
        CLI + args,
        capture_output=True, text=True,
        env={**os.environ, "NENAPU_DB": str(tmp_path / "s.db"),
             "PYTHONPATH": "src", "HOME": str(home),
             "NENAPU_HOME": str(home / ".nenapu"), **env},
    )


# ---------- the once-only marker ----------


def test_the_walkthrough_is_claimed_once_per_store(tmp_path):
    conn = connect(str(tmp_path / "s.db"))
    assert should_walk(conn) is True
    assert should_walk(conn) is False


def test_running_init_by_hand_suppresses_the_automatic_one(tmp_path):
    """Someone who already ran setup should not be walked through it again."""
    conn = connect(str(tmp_path / "s.db"))
    mark_walked(conn)
    assert should_walk(conn) is False


def test_the_marker_survives_reopening(tmp_path):
    conn = connect(str(tmp_path / "s.db"))
    should_walk(conn)
    conn.close()
    assert should_walk(connect(str(tmp_path / "s.db"))) is False


# ---------- what the first run looks like ----------


def test_first_bare_run_sets_up_and_explains(tmp_path):
    result = _run([], tmp_path)
    assert result.returncode == 0
    assert "Found on this machine" in result.stdout or \
           "No supported agent" in result.stdout
    assert "What you type" in result.stdout
    assert "nenapu recall" in result.stdout


def test_the_second_run_shows_commands_instead(tmp_path):
    """A wizard that reappears is a wizard you learn to close."""
    _run([], tmp_path)
    second = _run([], tmp_path)
    assert "What you type" not in second.stdout
    assert "Setup and diagnostics" in second.stdout


def test_the_guide_can_be_summoned_back(tmp_path):
    _run([], tmp_path)
    result = _run(["guide"], tmp_path)
    assert result.returncode == 0
    assert "What you type" in result.stdout


def test_the_guide_fits_eighty_columns(tmp_path):
    """Wrapped rows fold the right-hand column under the left and read as a bug."""
    result = _run(["guide"], tmp_path, COLUMNS="80")
    over = [line for line in result.stdout.splitlines() if len(line) > 80]
    assert not over, f"lines past the edge: {over}"


def test_a_non_interactive_first_run_changes_no_configuration(tmp_path):
    """A bare `nenapu` in a pipe or a CI job is not consent to edit settings.json."""
    result = _run([], tmp_path)
    home = tmp_path / "home"
    assert not (home / ".claude" / "settings.json").exists()
    assert not (home / ".cursor" / "mcp.json").exists()
    if "Claude Code" in result.stdout:
        assert "nenapu init --yes" in result.stdout


def test_init_yes_is_how_a_script_consents(tmp_path):
    """`--yes` exists so automation can opt in explicitly rather than by accident."""
    result = _run(["init", "--yes"], tmp_path)
    assert result.returncode == 0
    settings = tmp_path / "home" / ".claude" / "settings.json"
    if "Claude Code" in result.stdout:
        assert settings.exists()
        assert "--detach" in settings.read_text()


def test_no_banner_keeps_the_walkthrough_out_of_ci_logs(tmp_path):
    result = _run([], tmp_path, NENAPU_NO_BANNER="1")
    assert "What you type" not in result.stdout
