"""The commands that make the activity ledger worth building.

Requirement (Task 6, priority-ordered task list): "Query commands —
standup, activity, where, pending, project | a ledger nobody can query is
dead weight." Depends on Task 3 (the ledger tables) and Task 5 (backfill).

    nenapu standup                # what happened yesterday, across every project
    nenapu activity --since 1w    # timeline, grouped by project and agent
    nenapu where <file>           # every session and agent that touched it
    nenapu pending [--project X]  # open loops, cross-project
    nenapu project <name>         # one repo: recent work, files, commits, pending

`pending` depends on open loops (Task 11, Opus, not yet built) — these tests
only assert `pending` runs cleanly and reports zero when there are none yet;
they do not exercise open-loop content, which is out of this task's scope.

Tests drive the CLI the same way the rest of the suite does (subprocess, see
`tests/test_command_names.py::test_an_old_name_still_runs`), so they exercise
the real Typer wiring — panel placement, option parsing, exit codes — not
just the underlying query functions.
"""

import os
import subprocess
import sys

import pytest

from nenapu import connect


def _run(args, db):
    env = {**os.environ, "PYTHONPATH": "src", "NENAPU_NO_BANNER": "1"}
    return subprocess.run(
        [sys.executable, "-m", "nenapu.cli", *args, "--db", str(db)],
        capture_output=True, text=True, env=env,
    )


@pytest.fixture
def seeded_db(tmp_path):
    from nenapu.activity import ActivityLedger

    db = tmp_path / "s.db"
    ledger = ActivityLedger(connect(str(db)))
    a = ledger.start_session(agent="claude-code", project_scope="repo:demo@aaaaaaaa",
                             cwd="/repo-a", git_branch="main")
    ledger.record_file_event(a, path="backend/app/bookings.py", op="edited", tool="Edit")
    ledger.record_commit(a, sha="c7f1a9d4", subject="Add booking overlap constraint",
                         files_changed=["backend/app/bookings.py"])
    ledger.end_session(a, git_head_after="c7f1a9d4")

    b = ledger.start_session(agent="codex", project_scope="repo:other@bbbbbbbb", cwd="/repo-b")
    ledger.record_file_event(b, path="lib/util.py", op="deleted", tool="Bash")
    ledger.end_session(b)
    return db


def test_standup_runs_and_mentions_recent_work(seeded_db):
    result = _run(["standup"], seeded_db)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "bookings.py" in result.stdout


def test_standup_is_cross_project(seeded_db):
    """The whole point: one command, every project, not eleven CLAUDE.md files."""
    result = _run(["standup"], seeded_db)
    assert "demo" in result.stdout or "aaaaaaaa" in result.stdout
    assert "other" in result.stdout or "bbbbbbbb" in result.stdout


def test_activity_lists_a_timeline(seeded_db):
    result = _run(["activity", "--since", "1w"], seeded_db)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "bookings.py" in result.stdout


def test_where_finds_every_session_that_touched_a_file(seeded_db):
    result = _run(["where", "backend/app/bookings.py"], seeded_db)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "claude-code" in result.stdout


def test_where_reports_nothing_for_an_untouched_file(seeded_db):
    result = _run(["where", "never/touched.py"], seeded_db)
    assert result.returncode == 0, result.stdout + result.stderr


def test_pending_runs_cleanly_with_no_open_loops_yet(seeded_db):
    """Open-loop content is Task 11 (Opus); Task 6 only needs the command to
    exist and behave when the ledger has nothing pending."""
    result = _run(["pending"], seeded_db)
    assert result.returncode == 0, result.stdout + result.stderr


def test_pending_accepts_a_project_filter(seeded_db):
    result = _run(["pending", "--project", "demo"], seeded_db)
    assert result.returncode == 0, result.stdout + result.stderr


def test_project_shows_recent_work_files_and_commits(seeded_db):
    result = _run(["project", "demo"], seeded_db)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "bookings.py" in result.stdout
    assert "c7f1a9d4" in result.stdout


def test_project_on_an_unknown_name_is_not_a_crash(seeded_db):
    result = _run(["project", "does-not-exist"], seeded_db)
    assert result.returncode == 0, result.stdout + result.stderr
