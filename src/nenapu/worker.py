"""The single worker that turns queued transcripts into memory.

The queue exists so that sessions ending together do not fan out into
concurrent 83-second model calls against one store. That only holds if
exactly one process drains it, which is what `WorkerLock` is for: a second
worker finds the lock taken and returns rather than doubling the load.

This module is the composition point — queue, extraction, activity capture
and upkeep — so none of those has to know about the others.
"""

from __future__ import annotations

from pathlib import Path

from .activity import ActivityLedger
from .capture import capture_session
from .ingest_queue import WorkerLock, claim_next, mark_done, mark_failed
from .loops import LoopBook
from .maintenance import run_maintenance_tick
from .observer import observe_transcript
from .store import Store, project_scope


DEFAULT_LOCK_PATH = "~/.nenapu/worker.lock"


def lock_for(store: Store) -> Path:
    """The worker lock that belongs to this store.

    Beside the store file rather than at one fixed path, so two stores on one
    machine do not block each other, and so the lock is findable from `--db`
    alone — the Stop hook passes the worker nothing else.
    """
    row = store.conn.execute("PRAGMA database_list").fetchone()
    path = row[2] if row else ""
    if not path:  # an in-memory store has no directory to sit beside
        return Path(DEFAULT_LOCK_PATH).expanduser()
    return Path(path).parent / "worker.lock"


def drain(store: Store, *, lock_path: str | Path | None = None, limit: int = 20) -> int:
    """Process queued transcripts until the queue is empty. Returns the count.

    A job that fails is marked failed and the queue moves on: one unreadable
    transcript, or one model backend outage, must not stop everything behind
    it from ever being learned from.
    """
    lock = WorkerLock(lock_path or lock_for(store))
    if not lock.try_acquire():
        return 0  # another worker already has it; the jobs are not going anywhere

    processed = 0
    touched: set[str] = set()
    try:
        while processed < limit:
            job = claim_next(store.conn)
            if job is None:
                break
            try:
                touched.add(_ingest(store, job))
                mark_done(store.conn, job["id"])
            except Exception as exc:  # noqa: BLE001 — one bad job, not a stuck queue
                mark_failed(store.conn, job["id"], detail=str(exc)[:200])
            processed += 1

        run_maintenance_tick(store, touched_scopes=sorted(s for s in touched if s))
    finally:
        lock.release()
    return processed


def _ingest(store: Store, job: dict) -> str:
    """One transcript: what it did, then what it taught. Returns its scope.

    The ledger half runs first and unconditionally, because it is
    deterministic and free — a model backend that is down should cost the
    facts, not the record of what was touched.
    """
    transcript = Path(job["path"])
    ledger = ActivityLedger(store.conn)
    row_id = capture_session(ledger, transcript, agent=job["agent"])
    if row_id is not None:
        LoopBook(store.conn).detect_interrupted(ledger, row_id)

    session = ledger.get_session(row_id) if row_id is not None else None
    cwd = session["cwd"] if session else None
    scope = session["project_scope"] if session else (project_scope(cwd) if cwd else "global")

    observe_transcript(
        store, transcript, session_id=job["session_id"], scope=scope, cwd=cwd,
    )
    return scope
