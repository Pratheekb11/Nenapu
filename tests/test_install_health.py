"""The installed entry point must actually import nenapu.

Requirement (Task 0, priority-ordered task list): repair the editable install
so both Claude Code hooks stop crashing on every session.

Both hooks invoke the bare `nenapu` command found on `$PATH` — never the
project's dev virtualenv. When the repo is renamed (as it was,
`custom_agent` -> `Nenapu`), `uv tool install --editable` leaves a stale
`.pth` file pointing at the old, now-missing, source directory, and every
hook invocation dies with `ModuleNotFoundError: No module named 'nenapu'`
before it prints anything. `SessionStart` then injects no memory and `Stop`
learns nothing, silently, for as long as the install stays broken (measured:
five days, 2026-08-16 to 2026-08-21, on the live store).

These tests exercise the *installed* console script exactly the way a Claude
Code hook does — through `shutil.which("nenapu")`, not `python -m
nenapu.cli` — because that is the one invocation path a `uv run pytest`
inside the project's own venv does not otherwise cover, and it is the path
that was actually broken.
"""

import json
import shutil
import subprocess
import sys

import pytest

INSTALLED_NENAPU = shutil.which("nenapu")

pytestmark = pytest.mark.skipif(
    INSTALLED_NENAPU is None,
    reason="no globally-installed `nenapu` console script on PATH to check",
)


def test_the_installed_entry_point_imports_the_package():
    """The regression exactly as it presented: `nenapu --version` must not
    traceback with ModuleNotFoundError before printing anything."""
    result = subprocess.run(
        [INSTALLED_NENAPU, "version", "--plain"],
        capture_output=True, text=True, timeout=15,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "ModuleNotFoundError" not in result.stderr
    assert "Traceback" not in result.stderr


def test_the_installed_entry_point_resolves_next_to_the_real_source():
    """A `python3 -c` probe using the same interpreter the entry point's
    shebang names, so this checks the same import the hook actually performs
    rather than re-deriving it from `sys.path` in this test's own process."""
    result = subprocess.run(
        [sys.executable, "-c", "import nenapu; print(nenapu.__file__)"],
        capture_output=True, text=True, timeout=15,
    )
    assert result.returncode == 0, result.stderr
    assert "custom_agent" not in result.stdout, (
        "editable install still points at the pre-rename source tree"
    )


def test_the_session_start_hook_payload_does_not_crash():
    """Reproduces the exact failing invocation from the plan: a SessionStart
    hook payload piped to `recall-hook` must exit 0, whether or not it has
    anything to inject."""
    payload = json.dumps({"session_id": "install-health-check", "cwd": "/tmp"})
    result = subprocess.run(
        [INSTALLED_NENAPU, "recall-hook"],
        input=payload, capture_output=True, text=True, timeout=15,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "ModuleNotFoundError" not in result.stderr


def test_the_stop_hook_payload_does_not_crash_on_a_missing_transcript():
    """Same shape as Claude Code's real Stop payload; a nonexistent transcript
    path must be a clean no-op, not an import failure."""
    payload = json.dumps({
        "session_id": "install-health-check",
        "transcript_path": "/nonexistent/transcript.jsonl",
    })
    result = subprocess.run(
        [INSTALLED_NENAPU, "learn", "--stdin"],
        input=payload, capture_output=True, text=True, timeout=15,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "ModuleNotFoundError" not in result.stderr
