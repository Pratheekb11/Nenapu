"""Backfill the activity ledger from transcripts already on disk.

A parse, not an extraction: only session metadata and tool-use `file_path`
blocks are read from each JSONL line, so recovering months of history costs
no tokens and no model calls.
"""

from __future__ import annotations

import glob
import json
from pathlib import Path

from .activity import ActivityLedger
from .store import project_scope

# Tools that touch a file, and what that touch means for the ledger.
_TOOL_OP = {
    "Write": "created",
    "Edit": "edited",
    "NotebookEdit": "edited",
}


def backfill_transcript(ledger: ActivityLedger, path: str | Path, *, agent: str) -> int | None:
    """Replay one transcript into the ledger.

    Returns the session's ledger row id, or `None` if the transcript had no
    session id or was already backfilled — safe to call again once new
    transcripts have arrived, since a session already present by
    `external_id` is left untouched rather than re-ingested.
    """
    text = Path(path).read_text()

    session_id: str | None = None
    cwd: str | None = None
    git_branch: str | None = None
    events: list[tuple[str, str, str | None]] = []

    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue

        session_id = event.get("sessionId") or session_id
        cwd = event.get("cwd") or cwd
        git_branch = event.get("gitBranch") or git_branch

        message = event.get("message") or {}
        for block in message.get("content") or []:
            if not isinstance(block, dict) or block.get("type") != "tool_use":
                continue
            op = _TOOL_OP.get(block.get("name"))
            if op is None:
                continue
            file_path = (block.get("input") or {}).get("file_path")
            if file_path:
                events.append((file_path, op, block.get("name")))

    if session_id is None:
        return None
    if ledger.get_session(session_id) is not None:
        return None

    scope = project_scope(cwd) if cwd else "global"
    row_id = ledger.start_session(
        agent=agent, project_scope=scope, cwd=cwd, git_branch=git_branch,
        external_id=session_id,
    )
    for file_path, op, tool in events:
        ledger.record_file_event(row_id, path=file_path, op=op, tool=tool)
    return row_id


def backfill_directory(ledger: ActivityLedger, glob_pattern: str, *, agent: str) -> int:
    """Backfill every transcript matching `glob_pattern`. Returns the count of
    sessions newly ingested (already-backfilled transcripts are not counted)."""
    count = 0
    for path in sorted(glob.glob(glob_pattern)):
        if backfill_transcript(ledger, path, agent=agent) is not None:
            count += 1
    return count
