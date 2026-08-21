"""The watcher: capture from agents that have no hook API.

Only Claude Code can tell us a session ended. Everything else writes a
transcript to disk and says nothing, so the only way to learn from those
sessions is to watch the files. Four rules shape this:

**An adapter is data, not a branch.** Adding an agent is registering a glob
and a parser, not editing the observer. Nothing is registered that has not
been seen to match a real file on a real machine — a glob nobody has watched
work is a feature that reports success and captures nothing.

**"Finished" is measured, not guessed.** A session in progress is written
continuously, and ingesting one spends an 83-second extraction on half a
conversation. A transcript counts as finished when its size has been stable
across ticks for the quiet window.

**The queue does the work.** A tick enqueues; the single-flight worker
extracts. That is what keeps a discovered backlog from fanning out into
concurrent model calls against one store.

**Never do the Stop hook's job twice.** An agent whose hook is installed is
skipped. The unique index would de-duplicate the resulting facts, but not the
83 seconds spent producing them.
"""

from __future__ import annotations

import glob as globlib
import json
import os
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

from .db import commit, transaction
from .ingest_queue import enqueue
from .models import now
from .observer import _turns_from

# How long a transcript's size must hold still before it counts as finished.
QUIET_SECONDS = 120.0
# One extraction per tick by default. Discovering a backlog of a hundred
# transcripts at startup must not queue a hundred model calls at once; that is
# what `--batch` is for, on a schedule someone chose.
MIN_SECONDS_BETWEEN_EXTRACTIONS = QUIET_SECONDS

DEFAULT_SETTINGS_PATH = "~/.claude/settings.json"


@dataclass(frozen=True)
class TranscriptFormat:
    """One agent's transcripts: where they live, and how to read them."""

    agent: str
    glob: str
    parse: Callable[[list[str]], list[str]]


# Claude Code only, deliberately. Its format is already parsed by the observer
# and its transcripts are on this machine; Codex, Gemini, OpenCode and Cursor
# belong here only once someone has probed a real file each writes.
ADAPTERS: list[TranscriptFormat] = [
    TranscriptFormat(
        agent="claude-code",
        glob="~/.claude/projects/**/*.jsonl",
        parse=_turns_from,
    ),
]


# ---------- discovery ----------


def discover(adapters: Sequence[TranscriptFormat] = ADAPTERS) -> list[tuple[TranscriptFormat, Path]]:
    """Every transcript on disk that some adapter claims.

    A missing directory is the normal case — most machines have one or two of
    these agents installed — so an adapter that matches nothing contributes
    nothing rather than failing the tick.
    """
    found: list[tuple[TranscriptFormat, Path]] = []
    for adapter in adapters:
        pattern = os.path.expanduser(adapter.glob)
        for path in sorted(globlib.glob(pattern, recursive=True)):
            if os.path.isfile(path):
                found.append((adapter, Path(path)))
    return found


# ---------- state ----------


def get_state(conn: sqlite3.Connection, path: str) -> dict | None:
    row = conn.execute("SELECT * FROM watch_state WHERE path = ?", (path,)).fetchone()
    return dict(row) if row else None


def record_state(
    conn: sqlite3.Connection,
    *,
    path: str,
    agent: str,
    last_size: int,
    last_mtime: float | None = None,
    seen_at: float | None = None,
    ingested_at: float | None = None,
    ingested_size: int | None = None,
    session_id: str | None = None,
) -> None:
    """Remember what this file looked like at this moment.

    Ingestion fields are only overwritten when given, so recording that a file
    grew does not erase the fact that an earlier version of it was ingested.
    """
    with transaction(conn):
        existing = get_state(conn, path) or {}
        conn.execute(
            "INSERT INTO watch_state(path, agent, last_size, last_mtime, seen_at,"
            " ingested_at, ingested_size, session_id) VALUES (?,?,?,?,?,?,?,?)"
            " ON CONFLICT(path) DO UPDATE SET agent = excluded.agent,"
            " last_size = excluded.last_size, last_mtime = excluded.last_mtime,"
            " seen_at = excluded.seen_at, ingested_at = excluded.ingested_at,"
            " ingested_size = excluded.ingested_size, session_id = excluded.session_id",
            (path, agent, last_size, last_mtime, seen_at or now(),
             ingested_at if ingested_at is not None else existing.get("ingested_at"),
             ingested_size if ingested_size is not None else existing.get("ingested_size"),
             session_id if session_id is not None else existing.get("session_id")),
        )
        commit(conn)


def is_finished(path: str | Path, conn: sqlite3.Connection, *, at: float | None = None) -> bool:
    """Has this transcript held still long enough to be worth reading?

    Compared against what the watcher last observed rather than against the
    file's own mtime: a session that ended two minutes ago and one still being
    written have the same mtime resolution, and only the tick history can tell
    them apart.
    """
    state = get_state(conn, str(path))
    if state is None:
        return False  # first sight starts the clock; there is nothing to compare
    try:
        size = Path(path).stat().st_size
    except OSError:
        return False
    if size != state["last_size"]:
        return False
    return (at or now()) - (state["seen_at"] or 0) >= QUIET_SECONDS


# ---------- not doing the Stop hook's job twice ----------


def agents_with_hooks(settings_path: str | Path | None) -> set[str]:
    """Which agents already report their own sessions.

    Reads the same file the installer writes. An absent or unreadable file
    means no hooks, which is the safe direction: the watcher covers the agent
    and the unique index absorbs anything that arrives twice.
    """
    if not settings_path:
        return set()
    path = Path(settings_path).expanduser()
    try:
        settings = json.loads(path.read_text())
    except (OSError, ValueError):
        return set()
    hooks = settings.get("hooks") or {}
    if "nenapu" in json.dumps(hooks.get("Stop") or []):
        return {"claude-code"}
    return set()


# ---------- the tick ----------


def tick(
    conn: sqlite3.Connection,
    *,
    adapters: Sequence[TranscriptFormat] = ADAPTERS,
    settings_path: str | Path | None = DEFAULT_SETTINGS_PATH,
    at: float | None = None,
    batch: bool = False,
) -> list[int]:
    """One pass over every transcript on disk. Returns the jobs it enqueued.

    Enqueues rather than extracts: the watcher and the Stop hook feed one
    serialized worker, so two agents finishing together cost two queued jobs
    rather than two concurrent 83-second model calls.
    """
    at = at or now()
    skip = agents_with_hooks(settings_path)
    queued: list[int] = []

    for adapter, path in discover(adapters):
        if adapter.agent in skip:
            continue
        try:
            stat = path.stat()
        except OSError:
            continue  # a transcript deleted between discovery and here

        state = get_state(conn, str(path))
        if state is None or stat.st_size != state["last_size"]:
            # Either the first sight of this file or a session still being
            # written. Both mean: start the clock again and come back later.
            record_state(conn, path=str(path), agent=adapter.agent,
                         last_size=stat.st_size, last_mtime=stat.st_mtime, seen_at=at)
            continue

        if not is_finished(path, conn, at=at):
            continue
        if state["ingested_size"] == stat.st_size:
            continue  # already ingested at exactly this length

        queued.append(enqueue(conn, path=str(path), agent=adapter.agent))
        record_state(conn, path=str(path), agent=adapter.agent, last_size=stat.st_size,
                     last_mtime=stat.st_mtime, seen_at=at, ingested_at=at,
                     ingested_size=stat.st_size)
        if not batch:
            break  # the floor: one extraction per tick unless asked otherwise
    return queued
