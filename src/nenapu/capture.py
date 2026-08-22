"""What a session actually did: tool calls from the transcript, deletes from git.

Two sources, ranked, because neither is complete on its own:

1. **Git** is authoritative for the net effect, and is the only source that
   can report a *deletion*. Files die via `Bash rm` or `git rm`, and parsing
   shell strings for that is fragile enough to be worse than not trying.
2. **Tool calls** carry per-action attribution and ordering — which tool
   touched a file, in what sequence, including edits later reverted and files
   touched outside any repository. Git knows the net effect; only the
   transcript knows the sequence.

Everything here is a parse or a `git` read. No model call is made on this
path, which is what makes capturing every session on the machine free.
"""

from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from .activity import ActivityLedger
from .models import now
from .store import project_scope

# Tools that touch a file, and what that touch means for the ledger. The `op`
# column is constrained to this vocabulary by every query built on it, so a
# tool mapping to a fifth value would write rows nothing understands.
TOOL_OPS: dict[str, str] = {
    "Write": "created",
    "Edit": "edited",
    "MultiEdit": "edited",
    "NotebookEdit": "edited",
    "Read": "read",
}

# `git diff --name-status` letters. A rename has no `op` of its own — it is
# recorded as the death of one path and the birth of another, so that "where
# did models.py go" is answerable from either end.
_GIT_STATUS_OPS = {
    "A": "created",
    "M": "edited",
    "T": "edited",  # type change, e.g. file to symlink: the path still changed
    "D": "deleted",
    "C": "created",  # a copy is a new path; the source is untouched
}


# ---------- source 2: the transcript ----------


def file_events_from(lines: list[str], *, cwd: str | None = None) -> list[dict]:
    """Every file-touching tool call in `lines`, in the order it happened.

    `cwd` is optional on purpose. Passing it resolves relative `file_path`
    arguments so that two sessions in the same repo agree on how a file is
    spelled; omitting it stores paths exactly as the transcript recorded
    them, which is what the backfill of already-ingested history relies on.
    """
    events: list[dict] = []
    for line in lines:
        event = _loads(line)
        if event is None:
            continue
        at = _timestamp(event)
        for block in _tool_use_blocks(event):
            op = TOOL_OPS.get(block.get("name"))
            if op is None:
                continue  # Bash, Grep, Task: real work, but not a file event
            path = (block.get("input") or {}).get("file_path")
            if not isinstance(path, str) or not path:
                continue
            events.append({
                "path": _resolve(path, cwd), "op": op, "tool": block.get("name"), "at": at,
            })
    return events


def session_meta_from(lines: list[str]) -> dict:
    """The session id, cwd and branch the transcript reports for itself.

    Later lines win: a session can change directory, and the last thing it
    said about where it was is the most useful answer.

    Two spellings, because two agents write these files. Claude Code puts
    `sessionId` and `cwd` on every line; a Codex rollout wraps everything in
    `payload`, where `session_meta` carries `session_id` and both it and
    `turn_context` carry `cwd`. Probed against real transcripts from both.
    Without the second spelling a Codex session lands with no ledger row and
    its facts fall back to the `global` scope, which is the bug project
    scoping was built to end.
    """
    meta = {"session_id": None, "cwd": None, "git_branch": None}
    for line in lines:
        event = _loads(line)
        if event is None:
            continue
        payload = event.get("payload")
        if not isinstance(payload, dict):
            payload = {}
        meta["session_id"] = (event.get("sessionId") or payload.get("session_id")
                              or meta["session_id"])
        meta["cwd"] = event.get("cwd") or payload.get("cwd") or meta["cwd"]
        meta["git_branch"] = event.get("gitBranch") or meta["git_branch"]
    return meta


def read_lines(path: str | Path) -> list[str]:
    try:
        return Path(path).read_text(errors="ignore").splitlines()
    except OSError:
        return []


def _loads(line: str) -> dict | None:
    """A tail read starts mid-line by construction and a killed session leaves
    a truncated one at the end, so a bad line is ordinary, not exceptional."""
    line = line.strip()
    if not line:
        return None
    try:
        event = json.loads(line)
    except ValueError:
        return None
    return event if isinstance(event, dict) else None


def _tool_use_blocks(event: dict) -> list[dict]:
    message = event.get("message")
    if not isinstance(message, dict):
        return []
    content = message.get("content")
    if not isinstance(content, list):
        return []  # a plain-string message carries no tool calls
    return [b for b in content
            if isinstance(b, dict) and b.get("type") == "tool_use"]


def _timestamp(event: dict) -> float:
    """The transcript's own clock when it has one. Stamping ingestion time
    instead would make a backfill of months of history look like one busy
    afternoon."""
    raw = event.get("timestamp")
    if isinstance(raw, (int, float)):
        return float(raw)
    if isinstance(raw, str) and raw:
        try:
            parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
        except ValueError:
            return now()
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed.timestamp()
    return now()


def session_span_from(lines: list[str]) -> tuple[float | None, float | None]:
    """When the transcript says the session began and ended.

    Read from the same clock `_timestamp` already reads for every file event,
    and for the same reason: a backfill of months of history that stamps
    ingestion time looks like one busy afternoon, and three things read
    `sessions.started_at` believing it — the retrieval gate's coverage
    measure, "Where you left off", and the rollups.

    Returns `(None, None)` for a transcript that carries no clock at all,
    which is not an error: it has no better answer than "now", and the caller
    supplies it.
    """
    stamps = []
    for line in lines:
        event = _loads(line)
        if event is None:
            continue
        raw = event.get("timestamp")
        if isinstance(raw, (int, float)) or (isinstance(raw, str) and raw):
            stamps.append(_timestamp(event))
    if not stamps:
        return None, None
    return min(stamps), max(stamps)


def _resolve(path: str, cwd: str | None) -> str:
    if not cwd or path.startswith("/"):
        return path
    return str(Path(cwd) / path)


# ---------- source 1: git ----------


def git_head(cwd: str) -> str | None:
    """`None` outside a repository rather than an exception: sessions run in
    `/tmp` and in `~/Downloads`, and losing their tool events because git had
    nothing to say would be the ledger under-reporting silently."""
    return _git(cwd, "rev-parse", "HEAD")


def git_branch(cwd: str) -> str | None:
    """`None` when HEAD is detached.

    `rev-parse --abbrev-ref HEAD` answers the literal string "HEAD" there,
    which would land in the ledger as a branch by that name and group
    unrelated sessions together. `symbolic-ref` fails instead, which is the
    honest answer. Read from `cwd`, so a linked worktree reports its own
    branch rather than the main checkout's.
    """
    return _git(cwd, "symbolic-ref", "--quiet", "--short", "HEAD")


def changed_paths(cwd: str, before: str | None, after: str | None) -> list[tuple[str, str]]:
    """`(op, path)` for everything that changed between two commits.

    Two-dot `before..after` on purpose: diffing a merge commit against its
    first parent drops everything the side branch did, and the ledger cares
    about the net change the session produced, not about how it was reached.

    A recorded commit can vanish — a rebase, a reset, a pruned branch — so an
    unknown revision degrades to "no git evidence" instead of taking the
    session's tool events down with it.
    """
    if not before or not after or before == after:
        return []
    out = _git(cwd, "diff", "--name-status", "-M", f"{before}..{after}")
    if out is None:
        return []

    changed: list[tuple[str, str]] = []
    for line in out.splitlines():
        fields = line.split("\t")
        if len(fields) < 2:
            continue
        status = fields[0][:1]
        if status == "R":
            changed.append(("deleted", fields[1]))
            changed.append(("created", fields[2]))
            continue
        op = _GIT_STATUS_OPS.get(status)
        if op:
            changed.append((op, fields[-1]))
    return changed


def git_root(cwd: str) -> str | None:
    """The top of the checkout `cwd` belongs to. Read from `cwd` so a linked
    worktree resolves to its own root rather than the main checkout's."""
    return _git(cwd, "rev-parse", "--show-toplevel")


def commits_between(cwd: str, before: str | None, after: str | None) -> list[dict]:
    """The commits a session made, oldest first, with the files each touched."""
    if not before or not after or before == after:
        return []
    out = _git(cwd, "log", "--reverse", "--format=%H\x1f%s\x1f%ct", f"{before}..{after}")
    if out is None:
        return []

    commits: list[dict] = []
    for line in out.splitlines():
        parts = line.split("\x1f")
        if len(parts) != 3:
            continue
        sha, subject, at = parts
        files = _git(cwd, "diff-tree", "--no-commit-id", "--name-only", "-r", sha) or ""
        commits.append({
            "sha": sha,
            "subject": subject,
            "files_changed": files.split(),
            "at": float(at),
        })
    return commits


def _git(cwd: str, *args: str) -> str | None:
    try:
        result = subprocess.run(
            ["git", "-C", cwd, *args], capture_output=True, text=True, check=False,
        )
    except OSError:
        return None  # no git on this machine, or cwd is gone
    if result.returncode != 0:
        return None
    out = result.stdout.strip()
    return out or None


# ---------- the two sources, combined ----------


def capture_session(
    ledger: ActivityLedger,
    transcript: str | Path,
    *,
    agent: str,
    cwd: str | None = None,
    git_head_before: str | None = None,
) -> int | None:
    """Record one finished session in the ledger.

    Returns the session's ledger row id, or `None` when the transcript names
    no session or that session is already recorded. The Stop hook, the
    watcher and a backfill can all reach the same transcript, so ingestion
    has to be safe to repeat; `external_id` is what makes it so.
    """
    lines = read_lines(transcript)
    meta = session_meta_from(lines)
    external_id = meta["session_id"]
    if external_id is None:
        return None

    # A session may already have a row: `open_session` records one at
    # SessionStart, which is the only moment `git_head_before` can be read
    # honestly. That row is finished here rather than duplicated. A row that
    # already has `ended_at` has been captured, so this is the repeat call
    # the Stop hook, the watcher and a backfill can all make.
    existing = ledger.get_session(external_id)
    if existing is not None and existing["ended_at"] is not None:
        return None

    if existing is not None:
        cwd = cwd or existing["cwd"]
        git_head_before = git_head_before or existing["git_head_before"]
    cwd = cwd or meta["cwd"]
    events = file_events_from(lines, cwd=cwd)
    head_after = git_head(cwd) if cwd else None

    pscope = project_scope(cwd) if cwd else "global"
    row_id = existing["id"] if existing is not None else ledger.start_session(
        agent=agent,
        project_scope=pscope,
        cwd=cwd,
        git_branch=(git_branch(cwd) if cwd else None) or meta["git_branch"],
        git_head_before=git_head_before,
        external_id=external_id,
    )
    for event in events:
        ledger.record_file_event(
            row_id, path=event["path"], op=event["op"], tool=event["tool"], at=event["at"],
        )
    if cwd:
        _record_git_evidence(ledger, row_id, cwd, git_head_before, head_after, events)
    ledger.end_session(row_id, git_head_after=head_after)

    # Live sessions build entities as they land rather than waiting for an
    # offline `nenapu entities --rebuild`. `build_from_activity` only reads
    # `.conn`, so the ledger's connection is enough — no need for a Store
    # here. Scoped to this session's project so a busy machine does not
    # re-walk every repo's history on every session that ends.
    from .entities import build_from_activity

    build_from_activity(ledger, scope=pscope)
    return row_id


def _sync_entity_life(conn, path: str, op: str, scope: str) -> None:
    """Let a file's life and death reach the facts that are about it.

    Only ever acts on an entity that already exists: a deletion is not a
    reason to invent a node for a file nothing was ever recorded about.
    """
    from .entities import EntityGraph
    from .models import EntityKind, EntityStatus

    graph = EntityGraph(conn)
    entity = graph.find(kind=EntityKind.FILE, name=path, scope=scope)
    if entity is None:
        return
    if op == "deleted" and entity.status != EntityStatus.GONE:
        graph.mark_gone(entity.id, reason="deleted in this session")
    elif op != "deleted" and entity.status == EntityStatus.GONE:
        graph.mark_alive(entity.id)


def _record_git_evidence(
    ledger: ActivityLedger,
    row_id: int,
    cwd: str,
    before: str | None,
    after: str | None,
    tool_events: list[dict],
) -> None:
    """Add what only git could see, and nothing the transcript already said.

    A path the transcript named is left to the tool event, which knows the
    tool and the moment; recording git's view of it as well would double
    every `files_touched` count the rollups and `standup` are built on.
    Deletions are the exception — a file can be edited and then removed in
    one session, and both are true.
    """
    seen = {event["path"] for event in tool_events}
    # `git diff` names files relative to the top of the checkout; tool events
    # are absolute. Both must be spelled the same way or the same file lands
    # twice, once per source.
    root = git_root(cwd) or cwd
    for op, path in changed_paths(cwd, before, after):
        absolute = str(Path(root) / path)
        if op != "deleted" and absolute in seen:
            continue
        ledger.record_file_event(row_id, path=absolute, op=op, tool="git")
        # E6: git is the only place a deletion is visible without asking
        # anyone, and a fact about a file that no longer exists is
        # unsupported from that moment. Restoration reads the same way round.
        _sync_entity_life(ledger.conn, absolute, op, project_scope(cwd))
    for entry in commits_between(cwd, before, after):
        ledger.record_commit(
            row_id, sha=entry["sha"], subject=entry["subject"],
            files_changed=entry["files_changed"], at=entry["at"],
        )


def open_session(
    ledger: ActivityLedger,
    *,
    agent: str,
    external_id: str | None,
    cwd: str | None = None,
) -> int | None:
    """Record a session as it begins, and the commit it began on.

    `git_head_before` is the one field that cannot be recovered afterwards:
    by the time the session ends, the commit it started from is only
    reachable if something wrote it down. Returns the row id, or `None` when
    there is no session id to key on or the session is already known.
    """
    if not external_id or ledger.get_session(external_id) is not None:
        return None
    return ledger.start_session(
        agent=agent,
        project_scope=project_scope(cwd) if cwd else "global",
        cwd=cwd,
        git_branch=git_branch(cwd) if cwd else None,
        git_head_before=git_head(cwd) if cwd else None,
        external_id=external_id,
        # Watched as it ran. `git_head_before` used to stand in for this, and
        # it is NULL for a session outside a git repo, so a repair meant for
        # reconstructed history moved live rows too.
        source="hook",
    )
