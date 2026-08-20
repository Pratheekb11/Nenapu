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
