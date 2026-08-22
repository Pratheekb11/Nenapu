"""`nenapu backfill` — expose the parser that has existed since task 5.

Requirement (Task 20, "Next up" list, 2026-08-21, marked **Sonnet 5**,
depends on 4 and 5):

    "`nenapu backfill` command, then run it | 198 transcripts, 326MB, sit
    unread on this machine. `backfill.py` and its tests have existed since
    task 5 and nothing exposes them, so the ledger starts empty and every
    'where did I leave off' answers nothing | S | Sonnet 5 | 4, 5"

`nenapu.backfill.backfill_transcript` / `backfill_directory` are implemented
and tested (`tests/test_backfill.py`, task 5). Nothing calls them: `grep -rn
backfill src/nenapu/cli.py` is empty. This is the same failure class as
`expire_pending` before task 12 — written, tested, unreachable.

Scope boundary with `tests/test_backfill.py`
--------------------------------------------
That file pins the *parse*: idempotence by `external_id`, no model call,
paths recorded as the transcript spelled them. This file pins the *command*:
that it exists, where it defaults, what it prints, what it refuses to do, and
that the ledger queries from task 6 answer afterwards. Nothing here re-tests
the parser.

Assumed surface::

    nenapu backfill [--glob PATTERN] [--agent NAME] [--dry-run] [--db PATH]

`--dry-run` is not decoration: the real run on this machine is 198 files and
326MB, and being able to see what a run would ingest before it happens is the
difference between a command someone runs and a command someone reads about.
"""

import json
import os
import subprocess
import sys
from unittest.mock import patch

import pytest

from nenapu import connect


def _turn(role: str, text: str, session: str, cwd: str) -> str:
    return json.dumps({
        "type": role, "sessionId": session, "cwd": cwd,
        "message": {"role": role, "content": [{"type": "text", "text": text}]},
    })


def _tool_turn(session: str, cwd: str, path: str) -> str:
    return json.dumps({
        "type": "assistant", "sessionId": session, "cwd": cwd,
        "message": {"role": "assistant", "content": [
            {"type": "tool_use", "name": "Edit", "input": {"file_path": path}},
        ]},
    })


@pytest.fixture
def projects(tmp_path):
    """Two transcripts in the shape `~/.claude/projects` actually has."""
    root = tmp_path / ".claude" / "projects"
    (root / "repo-a").mkdir(parents=True)
    (root / "repo-b").mkdir(parents=True)
    (root / "repo-a" / "s-a.jsonl").write_text("\n".join([
        _turn("user", "add the overlap constraint", "s-a", "/repo-a"),
        _tool_turn("s-a", "/repo-a", "backend/app/bookings.py"),
    ]))
    (root / "repo-b" / "s-b.jsonl").write_text("\n".join([
        _turn("user", "drop the dead helper", "s-b", "/repo-b"),
        _tool_turn("s-b", "/repo-b", "lib/util.py"),
    ]))
    return root


@pytest.fixture
def db(tmp_path):
    return tmp_path / "s.db"


def _run(args, db, **env):
    return subprocess.run(
        [sys.executable, "-m", "nenapu.cli", *args, "--db", str(db)],
        capture_output=True, text=True,
        env={**os.environ, "PYTHONPATH": "src", "NENAPU_NO_BANNER": "1", **env},
    )


def _sessions(db):
    return [dict(r) for r in connect(str(db)).execute(
        "SELECT * FROM sessions ORDER BY id"
    )]


# ---------- the command exists ----------


def test_backfill_is_a_registered_command(db):
    result = _run(["backfill", "--help"], db)
    assert result.returncode == 0, result.stdout + result.stderr


def test_backfill_is_listed_under_the_activity_commands(db):
    """It answers the same question `standup` and `where` answer, and someone
    looking for "why does my ledger start empty" will be reading that panel."""
    result = _run(["--help"], db)
    assert "backfill" in result.stdout


# ---------- what it ingests ----------


def test_backfill_records_every_transcript_it_finds(projects, db):
    result = _run(["backfill", "--glob", f"{projects}/**/*.jsonl"], db)

    assert result.returncode == 0, result.stdout + result.stderr
    assert {s["external_id"] for s in _sessions(db)} == {"s-a", "s-b"}


def test_backfill_reports_how_many_sessions_it_ingested(projects, db):
    result = _run(["backfill", "--glob", f"{projects}/**/*.jsonl"], db)

    assert "2" in result.stdout


def test_backfill_records_the_files_each_session_touched(projects, db):
    """The point of the exercise: `where did I leave off` is a file question,
    and a session row with no file events answers nothing."""
    _run(["backfill", "--glob", f"{projects}/**/*.jsonl"], db)

    paths = {r["path"] for r in connect(str(db)).execute("SELECT path FROM file_events")}
    assert "backend/app/bookings.py" in paths
    assert "lib/util.py" in paths


def test_each_session_is_scoped_to_its_own_project(projects, db):
    """232 transcripts folded into one `global` pile would recreate exactly
    the bug task 1 was written to fix."""
    _run(["backfill", "--glob", f"{projects}/**/*.jsonl"], db)

    scopes = {s["project_scope"] for s in _sessions(db)}
    assert len(scopes) == 2


def test_the_agent_defaults_to_claude_code(projects, db):
    _run(["backfill", "--glob", f"{projects}/**/*.jsonl"], db)

    assert {s["agent"] for s in _sessions(db)} == {"claude-code"}


def test_the_agent_can_be_named_for_another_tool(projects, db):
    """The same command has to serve the probing session task 22 needs — a
    directory of Codex transcripts is a backfill with a different label."""
    _run(["backfill", "--glob", f"{projects}/**/*.jsonl", "--agent", "codex"], db)

    assert {s["agent"] for s in _sessions(db)} == {"codex"}


def test_running_it_twice_ingests_nothing_new(projects, db):
    _run(["backfill", "--glob", f"{projects}/**/*.jsonl"], db)
    second = _run(["backfill", "--glob", f"{projects}/**/*.jsonl"], db)

    assert second.returncode == 0
    assert len(_sessions(db)) == 2
    assert "0" in second.stdout


def test_new_transcripts_are_picked_up_by_a_later_run(projects, db):
    """Idempotence must not become inertness: a backfill run monthly has to
    catch up rather than decide it already ran."""
    _run(["backfill", "--glob", f"{projects}/**/*.jsonl"], db)
    (projects / "repo-a" / "s-c.jsonl").write_text(
        _turn("user", "later work", "s-c", "/repo-a")
    )

    _run(["backfill", "--glob", f"{projects}/**/*.jsonl"], db)

    assert {s["external_id"] for s in _sessions(db)} == {"s-a", "s-b", "s-c"}


# ---------- what it refuses to do ----------


def test_backfill_never_calls_the_model(projects, db):
    """"A parse, not an extraction" — 198 transcripts through a model at 83
    seconds each is 4.5 hours and a quota wall. Driven in-process because a
    subprocess would not see this patch."""
    from typer.testing import CliRunner

    from nenapu.cli import app

    with patch("nenapu.llm.structured") as structured:
        result = CliRunner().invoke(
            app, ["backfill", "--glob", f"{projects}/**/*.jsonl", "--db", str(db)]
        )

    assert result.exit_code == 0, result.output
    structured.assert_not_called()


def test_backfill_does_not_enqueue_extraction_jobs(projects, db):
    """The queue is where model calls come from. A backfill that fills it has
    turned a free afternoon into 198 of them."""
    _run(["backfill", "--glob", f"{projects}/**/*.jsonl"], db)

    queued = connect(str(db)).execute("SELECT COUNT(*) c FROM ingest_queue").fetchone()["c"]
    assert queued == 0


def test_a_dry_run_writes_nothing(projects, db):
    result = _run(["backfill", "--glob", f"{projects}/**/*.jsonl", "--dry-run"], db)

    assert result.returncode == 0, result.stdout + result.stderr
    assert _sessions(db) == []


def test_a_dry_run_still_says_how_much_there_is(projects, db):
    result = _run(["backfill", "--glob", f"{projects}/**/*.jsonl", "--dry-run"], db)

    assert "2" in result.stdout


# ---------- the failures a real directory contains ----------


def test_one_unreadable_transcript_does_not_abandon_the_rest(projects, db):
    """326MB of real transcripts contains truncated files, half-written
    lines and at least one thing that is not JSON. Stopping on the first one
    means the command works only on a machine that does not need it."""
    (projects / "repo-a" / "broken.jsonl").write_text("{not json at all\n")

    result = _run(["backfill", "--glob", f"{projects}/**/*.jsonl"], db)

    assert result.returncode == 0, result.stdout + result.stderr
    assert {s["external_id"] for s in _sessions(db)} == {"s-a", "s-b"}


def test_a_transcript_with_no_session_id_is_skipped_quietly(projects, db):
    (projects / "repo-a" / "empty.jsonl").write_text("")

    result = _run(["backfill", "--glob", f"{projects}/**/*.jsonl"], db)

    assert result.returncode == 0
    assert len(_sessions(db)) == 2


def test_a_glob_that_matches_nothing_exits_cleanly(tmp_path, db):
    result = _run(["backfill", "--glob", f"{tmp_path}/nowhere/**/*.jsonl"], db)

    assert result.returncode == 0, result.stdout + result.stderr
    assert "0" in result.stdout


# ---------- the default, and the payoff ----------


def test_the_default_glob_is_the_claude_projects_directory(projects, db, tmp_path):
    """Running it with no arguments has to do the useful thing — the whole
    complaint is that the ledger is empty on a machine whose transcripts are
    all sitting in one well-known place."""
    result = _run(["backfill"], db, HOME=str(tmp_path))

    assert result.returncode == 0, result.stdout + result.stderr
    assert {s["external_id"] for s in _sessions(db)} == {"s-a", "s-b"}


def test_after_a_backfill_the_ledger_queries_answer(projects, db):
    """End to end, and the actual requirement: `where` is what a person
    types, and before this command it answered nothing on every machine."""
    _run(["backfill", "--glob", f"{projects}/**/*.jsonl"], db)

    result = _run(["where", "backend/app/bookings.py"], db)

    assert result.returncode == 0, result.stdout + result.stderr
    assert "s-a" in result.stdout or "repo-a" in result.stdout or "claude-code" in result.stdout


# ==========================================================================
# `--redate` · repairing sessions an earlier backfill mis-dated
#
# Found on 2026-08-22. Until this pass `backfill_transcript` stamped the
# session row with the moment the backfill ran. The parser is fixed, but the
# rows an earlier run wrote still claim history happened this week, and three
# things read `sessions.started_at` believing it: the retrieval gate's
# coverage measure, "Where you left off", and the rollups. The transcripts
# are still on disk, so the true times are recoverable rather than lost.
#
# Deliberately a separate flag rather than something a plain backfill does:
# re-dating rewrites rows that already exist, and a command that quietly
# rewrote the ledger every time it was run would be a worse thing to own than
# the bug it fixes.
# ==========================================================================


def _epoch(ts: str) -> float:
    from datetime import datetime, timezone

    return datetime.fromisoformat(ts.replace("Z", "+00:00")).replace(
        tzinfo=timezone.utc).timestamp()


def _dated(session, cwd, path, *, ts):
    return json.dumps({
        "type": "assistant", "sessionId": session, "cwd": cwd, "timestamp": ts,
        "message": {"role": "assistant", "content": [
            {"type": "tool_use", "name": "Edit", "input": {"file_path": path}},
        ]},
    })


@pytest.fixture
def dated_projects(tmp_path):
    root = tmp_path / ".claude" / "projects"
    (root / "repo-a").mkdir(parents=True)
    (root / "repo-a" / "s-old.jsonl").write_text("\n".join([
        _dated("s-old", "/repo-a", "app/a.py", ts="2026-06-01T09:00:00Z"),
        _dated("s-old", "/repo-a", "app/b.py", ts="2026-06-01T10:30:00Z"),
    ]))
    return root


def _mis_dated_row(db, projects, session_id="s-old"):
    """A row in the state an earlier backfill left behind: right session,
    wrong clock."""
    from nenapu.activity import ActivityLedger
    from nenapu.models import now

    ledger = ActivityLedger(connect(str(db)))
    ledger.start_session(agent="claude-code", project_scope="repo:repo-a@aaaaaaaa",
                         cwd="/repo-a", external_id=session_id, started_at=now())
    return ledger


def test_redate_is_a_registered_flag(db):
    result = _run(["backfill", "--help"], db)

    assert result.returncode == 0, result.stdout + result.stderr
    assert "--redate" in result.stdout


def test_redate_moves_a_session_to_the_time_its_transcript_says(dated_projects, db):
    _mis_dated_row(db, dated_projects)

    result = _run(["backfill", "--redate", "--glob", f"{dated_projects}/**/*.jsonl"], db)

    assert result.returncode == 0, result.stdout + result.stderr
    row = next(s for s in _sessions(db) if s["external_id"] == "s-old")
    assert row["started_at"] == pytest.approx(_epoch("2026-06-01T09:00:00Z"), abs=1)


def test_redate_reports_how_many_rows_it_moved(dated_projects, db):
    _mis_dated_row(db, dated_projects)

    result = _run(["backfill", "--redate", "--glob", f"{dated_projects}/**/*.jsonl"], db)

    assert "1" in result.stdout


def test_redate_is_idempotent(dated_projects, db):
    _mis_dated_row(db, dated_projects)
    _run(["backfill", "--redate", "--glob", f"{dated_projects}/**/*.jsonl"], db)
    first = _sessions(db)

    _run(["backfill", "--redate", "--glob", f"{dated_projects}/**/*.jsonl"], db)

    assert _sessions(db) == first


def test_redate_leaves_a_session_the_hook_recorded_alone(dated_projects, db):
    """A live session's `started_at` was written by the SessionStart hook at
    the moment it began, which is a better answer than anything a transcript
    can be read to say. Only a row with no start of its own is repaired."""
    from nenapu.activity import ActivityLedger
    from nenapu.models import now

    ledger = ActivityLedger(connect(str(db)))
    hooked = now() - 120
    ledger.start_session(agent="claude-code", project_scope="repo:repo-a@aaaaaaaa",
                         cwd="/repo-a", external_id="s-old", started_at=hooked,
                         git_head_before="a" * 40)

    _run(["backfill", "--redate", "--glob", f"{dated_projects}/**/*.jsonl"], db)

    row = next(s for s in _sessions(db) if s["external_id"] == "s-old")
    assert row["started_at"] == pytest.approx(hooked, abs=1)


def test_redate_ingests_nothing_new(dated_projects, db):
    """It repairs what is there. A transcript with no row yet is a job for a
    plain backfill, and doing both in one flag would hide which one ran."""
    result = _run(["backfill", "--redate", "--glob", f"{dated_projects}/**/*.jsonl"], db)

    assert result.returncode == 0, result.stdout + result.stderr
    assert _sessions(db) == []


def test_a_plain_backfill_does_not_redate(dated_projects, db):
    """The rows a backfill already wrote are left where they are unless the
    flag asks. A command that quietly rewrote the ledger on every run would be
    worse to own than the bug."""
    from nenapu.models import now

    _mis_dated_row(db, dated_projects)
    before = next(s for s in _sessions(db) if s["external_id"] == "s-old")["started_at"]

    _run(["backfill", "--glob", f"{dated_projects}/**/*.jsonl"], db)

    row = next(s for s in _sessions(db) if s["external_id"] == "s-old")
    assert row["started_at"] == pytest.approx(before, abs=1)
    assert row["started_at"] == pytest.approx(now(), abs=60)


def test_a_dry_run_redate_writes_nothing(dated_projects, db):
    """`--dry-run` is the flag that promises nothing happens. A combination
    that ignores it and rewrites the ledger anyway is worse than not offering
    it: found by running exactly that against a real store."""
    _mis_dated_row(db, dated_projects)
    before = _sessions(db)

    result = _run(["backfill", "--redate", "--dry-run",
                   "--glob", f"{dated_projects}/**/*.jsonl"], db)

    assert result.returncode == 0, result.stdout + result.stderr
    assert _sessions(db) == before


def test_a_dry_run_redate_reports_what_it_would_move(dated_projects, db):
    _mis_dated_row(db, dated_projects)

    result = _run(["backfill", "--redate", "--dry-run",
                   "--glob", f"{dated_projects}/**/*.jsonl"], db)

    assert "1" in result.stdout
