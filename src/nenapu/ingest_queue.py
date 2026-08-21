"""A durable, single-flight ingestion queue.

Replaces the fire-and-forget `Popen` the Stop hook used to spawn: the hook's
job shrinks to enqueue-and-return, and one worker holding `WorkerLock` drains
the queue strictly serially. Without this, sessions ending together — or the
watcher discovering several stale transcripts at startup — fan out into
concurrent 83-second model calls against the one store and log file.
"""

from __future__ import annotations

import contextlib
import fcntl
import sqlite3
import threading
from pathlib import Path

from .db import commit, transaction
from .models import now

STATE_PENDING = "pending"
STATE_CLAIMED = "claimed"
STATE_DONE = "done"
STATE_FAILED = "failed"


def enqueue(
    conn: sqlite3.Connection, *, path: str, agent: str, session_id: str | None = None,
) -> int:
    with transaction(conn):
        cur = conn.execute(
            "INSERT INTO ingest_queue(path, agent, session_id, enqueued_at, state)"
            " VALUES (?,?,?,?,?)",
            (path, agent, session_id, now(), STATE_PENDING),
        )
        commit(conn)
        return cur.lastrowid


def enqueue_once(
    conn: sqlite3.Connection, *, path: str, agent: str, session_id: str | None = None,
) -> int | None:
    """Enqueue unless this transcript is already waiting. Returns the new id.

    Two hooks for one session — a retry, or the watcher and the hook reaching
    the same file — must not buy two 83-second extractions of identical
    content. Dedupe covers *unfinished* work only: a session that was resumed
    and appended to is new material, and a finished job is no reason to refuse
    to read it again. The check and the insert share one transaction, so two
    hooks firing together cannot both find nothing and both insert.
    """
    with transaction(conn):
        waiting = conn.execute(
            "SELECT id FROM ingest_queue WHERE path = ? AND state IN (?, ?) LIMIT 1",
            (path, STATE_PENDING, STATE_CLAIMED),
        ).fetchone()
        if waiting is not None:
            return None
        return enqueue(conn, path=path, agent=agent, session_id=session_id)


def has_pending(conn: sqlite3.Connection) -> bool:
    """Is there work waiting? A look, not a claim.

    The draining worker uses this to decide whether to go round again before
    releasing `WorkerLock`; claiming a job just to find out would leave it
    claimed by a worker that has decided to stop.
    """
    row = conn.execute(
        "SELECT 1 FROM ingest_queue WHERE state = ? LIMIT 1", (STATE_PENDING,)
    ).fetchone()
    return row is not None


def claim_next(conn: sqlite3.Connection) -> dict | None:
    """Atomically claim the oldest pending job, so two workers racing on the
    same store cannot both come away with it."""
    with transaction(conn):
        row = conn.execute(
            "SELECT * FROM ingest_queue WHERE state = ? ORDER BY enqueued_at LIMIT 1",
            (STATE_PENDING,),
        ).fetchone()
        if row is None:
            return None
        job = dict(row)
        conn.execute(
            "UPDATE ingest_queue SET state = ?, claimed_at = ? WHERE id = ?",
            (STATE_CLAIMED, now(), job["id"]),
        )
        commit(conn)
        job["state"] = STATE_CLAIMED
        return job


def mark_done(conn: sqlite3.Connection, job_id: int) -> None:
    with transaction(conn):
        conn.execute(
            "UPDATE ingest_queue SET state = ?, finished_at = ? WHERE id = ?",
            (STATE_DONE, now(), job_id),
        )
        commit(conn)


def mark_failed(conn: sqlite3.Connection, job_id: int, *, detail: str | None = None) -> None:
    """A bad transcript ends its own job, not the queue — the next `claim_next`
    moves straight to whatever is behind it."""
    with transaction(conn):
        conn.execute(
            "UPDATE ingest_queue SET state = ?, finished_at = ?, detail = ? WHERE id = ?",
            (STATE_FAILED, now(), detail, job_id),
        )
        commit(conn)


class WorkerLock:
    """Exclusive, non-blocking lock so only one worker drains the queue.

    `fcntl.flock` locks are per open-file-description, not per process — a
    second `open()` in the *same* process can flock the same path
    successfully even while the first is held. A `threading.Lock` keyed by
    path closes that gap so the lock is exclusive across processes (flock)
    and within one process (the dict of thread locks).
    """

    _process_locks: dict[str, threading.Lock] = {}

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._fh = None
        self._thread_lock: threading.Lock | None = None

    def try_acquire(self) -> bool:
        key = str(self.path)
        thread_lock = WorkerLock._process_locks.setdefault(key, threading.Lock())
        if not thread_lock.acquire(blocking=False):
            return False

        self.path.parent.mkdir(parents=True, exist_ok=True)
        fh = open(self.path, "a+")
        try:
            fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            fh.close()
            thread_lock.release()
            return False

        self._fh = fh
        self._thread_lock = thread_lock
        return True

    def release(self) -> None:
        if self._fh is not None:
            with contextlib.suppress(OSError):
                fcntl.flock(self._fh.fileno(), fcntl.LOCK_UN)
            self._fh.close()
            self._fh = None
        if self._thread_lock is not None:
            self._thread_lock.release()
            self._thread_lock = None

    def __enter__(self) -> bool:
        return self.try_acquire()

    def __exit__(self, *exc_info) -> None:
        self.release()
