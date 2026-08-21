"""Backfilling the activity ledger from transcripts already on disk.

Requirement (Task 5, priority-ordered task list): "Backfill 232 transcripts —
recovers months of lost history in an afternoon; no tokens; proves the
design." Depends on Task 3 (the ledger tables) and Task 4 (tool-call capture
+ git diffing), so these tests build ledger rows from synthetic transcripts
shaped like the real ones (see `tests/test_observer.py`'s docstring on why
that shape matters — mostly tool traffic, conversation scattered through it)
and check the two things the plan is explicit about:

* it is a **parse, not an extraction** — no model call, ever, for any input;
* it is **idempotent** — re-running a backfill after new transcripts arrive
  must not duplicate sessions already ingested, since the eventual watcher
  (Task 15) and the Stop hook will both feed the same ledger going forward.

Event shape below matches a real Claude Code JSONL line (verified against
`~/.claude/projects/*/*.jsonl` on this machine): top-level `sessionId`, `cwd`,
`gitBranch`, `timestamp`, and either `message.role`/`content` for turns or a
`tool_use` content block carrying `input.file_path` for Edit/Write calls.

This test file assumes the entry point lives in a new `nenapu.backfill`
module, proposed here as the contract for Task 5's implementation.
"""

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from nenapu import connect


def _line(session_id, role, text=None, tool=None, file_path=None, ts="2026-08-15T10:00:00Z",
          cwd="/home/user/proj", branch="main"):
    event = {
        "sessionId": session_id, "cwd": cwd, "gitBranch": branch, "timestamp": ts,
        "type": role,
    }
    if tool:
        event["message"] = {"role": "assistant", "content": [
            {"type": "tool_use", "name": tool, "input": {"file_path": file_path}},
        ]}
    else:
        event["message"] = {"role": role, "content": [{"type": "text", "text": text}]}
    return json.dumps(event)


@pytest.fixture
def transcript(tmp_path):
    session_id = "s1"
    lines = [
        _line(session_id, "user", text="fix the booking overlap bug"),
        _line(session_id, "assistant", tool="Edit", file_path="backend/app/bookings.py"),
        _line(session_id, "assistant", tool="Write", file_path="backend/app/new_check.py"),
        _line(session_id, "assistant", text="done, added the constraint"),
    ]
    path = tmp_path / f"{session_id}.jsonl"
    path.write_text("\n".join(lines) + "\n")
    return path


@pytest.fixture
def ledger():
    from nenapu.activity import ActivityLedger

    return ActivityLedger(connect(":memory:"))


def test_backfill_never_calls_the_model(transcript, ledger):
    """The load-bearing property: a parse, not an extraction. If this calls
    `llm.structured`, backfilling 232 transcripts costs 232 model calls and
    ~5 hours (83s/session, measured in `IMPLEMENTATION_NOTES.md`) instead of
    an afternoon with zero tokens."""
    from nenapu.backfill import backfill_transcript

    with patch("nenapu.llm.structured") as mock_structured:
        backfill_transcript(ledger, transcript, agent="claude-code")
        mock_structured.assert_not_called()


def test_backfill_creates_a_session_row(transcript, ledger):
    from nenapu.backfill import backfill_transcript

    backfill_transcript(ledger, transcript, agent="claude-code")

    session = ledger.get_session("s1")
    assert session is not None
    assert session["agent"] == "claude-code"
    assert session["cwd"] == "/home/user/proj"
    assert session["git_branch"] == "main"


def test_backfill_records_file_events_from_tool_calls(transcript, ledger):
    from nenapu.backfill import backfill_transcript

    backfill_transcript(ledger, transcript, agent="claude-code")

    events = ledger.file_events_for_session("s1")
    paths_ops = {(e["path"], e["op"]) for e in events}
    assert ("backend/app/bookings.py", "edited") in paths_ops
    assert ("backend/app/new_check.py", "created") in paths_ops


def test_backfill_is_idempotent(transcript, ledger):
    """Re-running must not duplicate a session already ingested — the watcher
    (Task 15) will discover the same files on every restart."""
    from nenapu.backfill import backfill_transcript

    backfill_transcript(ledger, transcript, agent="claude-code")
    backfill_transcript(ledger, transcript, agent="claude-code")

    events = ledger.file_events_for_session("s1")
    assert len(events) == 2, "re-running the backfill duplicated file events"


def test_backfill_directory_processes_every_transcript(tmp_path, ledger):
    from nenapu.backfill import backfill_directory

    for i in range(3):
        sid = f"s{i}"
        lines = [_line(sid, "user", text=f"session {i}"),
                 _line(sid, "assistant", tool="Edit", file_path=f"file{i}.py")]
        (tmp_path / f"{sid}.jsonl").write_text("\n".join(lines) + "\n")

    count = backfill_directory(ledger, str(tmp_path / "*.jsonl"), agent="claude-code")

    assert count == 3
    assert ledger.get_session("s0") is not None
    assert ledger.get_session("s2") is not None


# ==========================================================================
# When the backfilled session says it happened
#
# Found in use on 2026-08-22, diagnosing why `nenapu retrieval` reported a
# coverage problem. A backfilled session row was stamped with the moment the
# backfill ran, not with the moment the session happened, so 193 sessions
# from weeks of history looked like they had all started that afternoon.
#
# Three things read `sessions.started_at` and all three were wrong because of
# it: the retrieval gate's coverage measure, which excludes sessions that
# predate the hook era and so counted every one of them as a live session
# given nothing; `sessions_for_scope`, which orders by it, so "Where you left
# off" could point at a session from three weeks ago; and the rollups, which
# age the ledger by it.
#
# The transcript's own clock is already parsed — `capture._timestamp` reads
# it for every file event, precisely so that "a backfill of months of history
# does not look like one busy afternoon".
# ==========================================================================


def _at(ts: str) -> float:
    from datetime import datetime, timezone

    return datetime.fromisoformat(ts.replace("Z", "+00:00")).replace(
        tzinfo=timezone.utc).timestamp()


def test_a_backfilled_session_starts_when_the_transcript_says_it_did(tmp_path, ledger):
    from nenapu.backfill import backfill_transcript

    path = tmp_path / "s-old.jsonl"
    path.write_text("\n".join([
        _line("s-old", "user", text="what broke", ts="2026-07-01T09:00:00Z"),
        _line("s-old", "assistant", tool="Edit", file_path="app/a.py",
              ts="2026-07-01T09:05:00Z"),
    ]) + "\n")

    row_id = backfill_transcript(ledger, path, agent="claude-code")

    assert ledger.get_session("s-old")["started_at"] == pytest.approx(
        _at("2026-07-01T09:00:00Z"), abs=1)
    assert row_id is not None


def test_a_backfill_of_months_does_not_look_like_one_afternoon(tmp_path, ledger):
    """The property the ingestion-time stamp destroyed: two sessions six
    weeks apart have to stay six weeks apart in the ledger."""
    from nenapu.backfill import backfill_transcript

    for name, ts in (("s-june", "2026-06-01T12:00:00Z"), ("s-july", "2026-07-15T12:00:00Z")):
        path = tmp_path / f"{name}.jsonl"
        path.write_text(_line(name, "user", text="work", ts=ts) + "\n")
        backfill_transcript(ledger, path, agent="claude-code")

    june = ledger.get_session("s-june")["started_at"]
    july = ledger.get_session("s-july")["started_at"]

    assert july - june == pytest.approx(44 * 86400, abs=86400)


def test_a_transcript_with_no_usable_clock_still_backfills(tmp_path, ledger):
    """A transcript that carries no timestamp is not an error. It has no
    better answer than "now", which is what it got before."""
    import json

    from nenapu.backfill import backfill_transcript
    from nenapu.models import now

    path = tmp_path / "s-blank.jsonl"
    path.write_text(json.dumps({
        "sessionId": "s-blank", "cwd": "/repo", "type": "user",
        "message": {"role": "user", "content": [{"type": "text", "text": "hello"}]},
    }) + "\n")

    backfill_transcript(ledger, path, agent="claude-code")

    assert ledger.get_session("s-blank")["started_at"] == pytest.approx(now(), abs=60)


def test_the_backfilled_session_ends_when_the_transcript_ends(tmp_path, ledger):
    """`ended_at` is what "3 days ago" in the injected block is measured
    from, and a backfilled session that never ended reads as one still
    running."""
    from nenapu.backfill import backfill_transcript

    path = tmp_path / "s-span.jsonl"
    path.write_text("\n".join([
        _line("s-span", "user", text="start", ts="2026-07-01T09:00:00Z"),
        _line("s-span", "assistant", text="end", ts="2026-07-01T11:30:00Z"),
    ]) + "\n")

    backfill_transcript(ledger, path, agent="claude-code")

    assert ledger.get_session("s-span")["ended_at"] == pytest.approx(
        _at("2026-07-01T11:30:00Z"), abs=1)
