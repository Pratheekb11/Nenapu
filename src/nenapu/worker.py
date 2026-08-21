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
from .capture import capture_session, read_lines, session_meta_from
from .ingest_queue import (
    WorkerLock,
    claim_next,
    has_pending,
    mark_done,
    mark_failed,
    release_stale_claims,
)
from .loops import LoopBook
from .maintenance import run_maintenance_tick
from .observer import observe_transcript
from .store import Store, project_scope
from .watch import parser_for


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

    # Holding the lock means no other worker is draining, so any claim still
    # open long past a model call belongs to a worker that died. Released here
    # rather than on a timer: this is the one moment it is safe to decide.
    release_stale_claims(store.conn)

    processed = 0
    touched: set[str] = set()
    try:
        while True:
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
            # Upkeep runs inside the lock and can take minutes when an audit
            # or a check is due. A hook firing in that window enqueues fine
            # and spawns a worker that finds the lock taken and returns, so
            # this worker has to look at the queue once more before letting
            # go — otherwise that job waits for an unrelated later trigger.
            if processed >= limit or not has_pending(store.conn):
                break
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
    if session is None:
        # `capture_session` returns None for a session it has already
        # recorded — a resume, or a retry after the extraction failed, since
        # the session row is ended before the model is called. The row still
        # holds the cwd and the scope, so look it up by the id the job
        # carries, or by the one the transcript names when the watcher
        # queued this without an id. Falling through to `global` here would
        # store a project's facts where every other project recalls them.
        external_id = (job["session_id"]
                       or session_meta_from(read_lines(transcript))["session_id"])
        session = ledger.get_session(external_id) if external_id else None

    cwd = session["cwd"] if session else None
    scope = session["project_scope"] if session else (project_scope(cwd) if cwd else "global")
    # The watcher enqueues with no session id, and a fact stored without one
    # gets no inferred edges — `Graph.infer_edges_for` returns nothing on a
    # null session — so the whole watcher path would build no graph at all.
    session_id = job["session_id"] or (session["external_id"] if session else None)

    # Read the transcript with the parser belonging to the agent that wrote
    # it: a Codex rollout read by Claude Code's parser yields an empty
    # conversation, which looks exactly like a session that taught nothing.
    # A replayed job grades under its own source, so an audit can tell
    # backfilled evidence from evidence that arrived as the sessions ran.
    grade_source = job.get("grade_source")
    observe_transcript(
        store, transcript, session_id=session_id, scope=scope, cwd=cwd,
        parse=parser_for(job["agent"]),
        **({"grade_source": grade_source} if grade_source else {}),
    )
    return scope
