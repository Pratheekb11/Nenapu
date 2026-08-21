"""Backfill the activity ledger from transcripts already on disk.

A parse, not an extraction: session metadata and tool-use blocks are read
straight out of each JSONL line by `capture`, so recovering months of history
costs no tokens and no model calls.

The parse itself lives in `capture` rather than here. Two parsers would mean
the backfilled sessions and every future session disagree about what a
session contains — the backfill would keep missing `Read`, and "which agent
looked at this file" would be answerable only for sessions recorded after
the split was noticed.
"""

from __future__ import annotations

import glob
from pathlib import Path

from .activity import ActivityLedger
from .capture import file_events_from, read_lines, session_meta_from
from .store import project_scope


def backfill_transcript(ledger: ActivityLedger, path: str | Path, *, agent: str) -> int | None:
    """Replay one transcript into the ledger.

    Returns the session's ledger row id, or `None` if the transcript had no
    session id or was already backfilled — safe to call again once new
    transcripts have arrived, since a session already present by
    `external_id` is left untouched rather than re-ingested.

    Paths are stored exactly as the transcript spelled them. Unlike a live
    capture, a backfill is reading history: the working directory it names
    may since have moved or been deleted, so resolving relative paths against
    it would invent locations rather than record them.
    """
    lines = read_lines(path)
    meta = session_meta_from(lines)
    session_id = meta["session_id"]
    if session_id is None or ledger.get_session(session_id) is not None:
        return None

    row_id = ledger.start_session(
        agent=agent,
        project_scope=project_scope(meta["cwd"]) if meta["cwd"] else "global",
        cwd=meta["cwd"],
        git_branch=meta["git_branch"],
        external_id=session_id,
    )
    for event in file_events_from(lines):
        ledger.record_file_event(
            row_id, path=event["path"], op=event["op"], tool=event["tool"], at=event["at"],
        )
    return row_id


def backfill_directory(ledger: ActivityLedger, glob_pattern: str, *, agent: str) -> int:
    """Backfill every transcript matching `glob_pattern`. Returns the count of
    sessions newly ingested (already-backfilled transcripts are not counted)."""
    count = 0
    for path in sorted(glob.glob(glob_pattern)):
        if backfill_transcript(ledger, path, agent=agent) is not None:
            count += 1
    return count
