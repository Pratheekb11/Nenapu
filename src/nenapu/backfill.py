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
import os
from pathlib import Path

from .activity import ActivityLedger
from .capture import (
    file_events_from,
    read_lines,
    session_meta_from,
    session_span_from,
)
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
    if _unseen_session_id(ledger, meta) is None:
        return None
    session_id = meta["session_id"]

    # The transcript's own clock, not this afternoon's. A row stamped at
    # ingestion time claims a session from six weeks ago happened during the
    # hook era, which is read as a live session that was given no memory.
    started_at, ended_at = session_span_from(lines)

    row_id = ledger.start_session(
        agent=agent,
        project_scope=project_scope(meta["cwd"]) if meta["cwd"] else "global",
        cwd=meta["cwd"],
        git_branch=meta["git_branch"],
        started_at=started_at,
        external_id=session_id,
        source="backfill",
    )
    for event in file_events_from(lines):
        ledger.record_file_event(
            row_id, path=event["path"], op=event["op"], tool=event["tool"], at=event["at"],
        )
    # A backfilled session is over: it is history being read back. Leaving
    # `ended_at` null reads as one still running, and "3 days ago" in the
    # injected block is measured from it.
    if ended_at is not None:
        ledger.end_session(row_id, ended_at=ended_at)
    return row_id


def redate_backfilled_sessions(ledger: ActivityLedger, glob_pattern: str, *,
                               apply: bool = True) -> int:
    """Move sessions an earlier backfill mis-dated onto their own clock.

    Until this was fixed, `backfill_transcript` stamped the row with the
    moment the backfill ran, so weeks of history read as one afternoon. The
    parser is fixed; these are the rows it already wrote, and the transcripts
    are still on disk, so the true times are recoverable rather than lost.

    Only reconstructed rows are touched. A session the SessionStart hook
    recorded began at a moment something watched, which is a better answer
    than anything a transcript can be read to say. `source` records which is
    which; rows written before that column existed have no answer to give and
    fall back to the rule this used to use on its own — `git_head_before` set
    means the hook recorded it — which is right except for a live session that
    ran outside a git repo, where there was no head to write down. Repairs
    only; a transcript with no row yet is a job for a plain backfill.

    With `apply=False` it counts what it would move and writes nothing, which
    is what `--dry-run` promises.
    """
    moved = 0
    for path in transcripts_matching(glob_pattern):
        lines = read_lines(path)
        session_id = session_meta_from(lines)["session_id"]
        if not session_id:
            continue
        row = ledger.get_session(session_id)
        if row is None or not _was_reconstructed(row):
            continue
        started_at, ended_at = session_span_from(lines)
        if started_at is None:
            continue
        if _already_dated(row, started_at, ended_at):
            continue
        if apply:
            ledger.redate_session(row["id"], started_at=started_at, ended_at=ended_at)
        moved += 1
    return moved


def _was_reconstructed(row: dict) -> bool:
    """May this row be moved onto the clock its transcript carries?

    Rows recorded since `sessions.source` exists say so themselves. Older rows
    fall back to the heuristic that came first: a session the hook watched has
    a `git_head_before` written at the moment it began.
    """
    if row.get("source"):
        return row["source"] == "backfill"
    return row["git_head_before"] is None


def _already_dated(row: dict, started_at: float, ended_at: float | None) -> bool:
    """Has this row already been moved? Re-running must change nothing."""
    return (abs(row["started_at"] - started_at) < 1.0
            and (ended_at is None
                 or (row["ended_at"] is not None and abs(row["ended_at"] - ended_at) < 1.0)))


def transcripts_matching(glob_pattern: str) -> list[str]:
    """Every file the pattern names, oldest path first.

    `recursive=True` because the directory this runs against in practice is
    `~/.claude/projects/**/*.jsonl`, where the transcripts sit one directory
    per project — without it `**` collapses to a single level and the pattern
    silently answers for only part of the tree.
    """
    return sorted(glob.glob(os.path.expanduser(glob_pattern), recursive=True))


def would_backfill(ledger: ActivityLedger, glob_pattern: str) -> int:
    """How many sessions a run would newly ingest, writing nothing.

    A real run here is 198 files and 326MB; being able to see what it would
    do before it does it is what makes it a command someone runs.
    """
    return sum(
        1 for path in transcripts_matching(glob_pattern)
        if _unseen_session_id(ledger, session_meta_from(read_lines(path))) is not None
    )


def _unseen_session_id(ledger: ActivityLedger, meta: dict) -> str | None:
    """The session id this transcript would add, or `None` if it adds nothing.

    One rule, read by both the run and the preview, so what `--dry-run`
    counts and what a run ingests cannot drift apart.
    """
    session_id = meta["session_id"]
    if session_id is None or ledger.get_session(session_id) is not None:
        return None
    return session_id


def backfill_directory(ledger: ActivityLedger, glob_pattern: str, *, agent: str) -> int:
    """Backfill every transcript matching `glob_pattern`. Returns the count of
    sessions newly ingested (already-backfilled transcripts are not counted).

    A directory of real transcripts contains truncated files, half-written
    lines and at least one thing that is not JSON, so a transcript that cannot
    be replayed ends its own file and nothing else: stopping on the first one
    would make this work only on a machine that does not need it.
    """
    count = 0
    for path in transcripts_matching(glob_pattern):
        try:
            if backfill_transcript(ledger, path, agent=agent) is not None:
                count += 1
        except Exception:  # noqa: BLE001 — one bad transcript, not a lost run
            continue
    return count
