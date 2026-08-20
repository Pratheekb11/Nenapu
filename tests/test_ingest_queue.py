"""A durable, single-flight ingestion queue.

Requirement (Task 8, priority-ordered task list): "Single-flight ingestion
queue + worker | fixes unbounded fan-out; prerequisite for the watcher and
for self-maintenance."

Today `_detach_observe` (`cli.py:959-994`) is fire-and-forget: every Stop
hook spawns a detached `Popen` with no lock, no pidfile, no queue, no wait.
Two sessions ending together spawn two concurrent 83-second model calls
against one store and one log file — and Task 15's watcher would make this
sharply worse, discovering N stale session files at startup and fanning out
N parallel extractions.

The fix, per the plan: an `ingest_queue(path, agent, session_id,
enqueued_at, state)` table in the same SQLite file, plus one worker holding
an exclusive lock (`~/.nenapu/ingest.lock` via `fcntl.flock`, or a `meta`
row with a lease). The hook's job shrinks to enqueue-and-return; the worker
drains strictly serially.

These tests exercise the queue's data-layer contract (enqueue / claim /
complete) and the single-flight lock in isolation — they do not yet wire
`cli.py`'s Stop hook to enqueue instead of spawning, which is a CLI-layer
change riding on this same module.

Assumes a new `nenapu.ingest_queue` module: `enqueue()`, `claim_next()`,
`mark_done()`, `mark_failed()`, and a `WorkerLock` context manager. Proposed
contract, not yet in the codebase.
"""

import threading
import time

import pytest

from nenapu import connect


@pytest.fixture
def conn():
    return connect(":memory:")


# ---------- schema ----------


def test_ingest_queue_table_exists():
    conn = connect(":memory:")
    columns = {r["name"] for r in conn.execute("PRAGMA table_info(ingest_queue)")}
    assert columns >= {"id", "path", "agent", "session_id", "enqueued_at", "state"}


def test_an_existing_store_gains_the_queue_table_on_reconnect(tmp_path):
    path = tmp_path / "old.db"
    connect(str(path))
    reopened = connect(str(path))
    tables = {r["name"] for r in reopened.execute(
        "SELECT name FROM sqlite_master WHERE type = 'table'"
    )}
    assert "ingest_queue" in tables


# ---------- enqueue / claim / complete ----------


def test_enqueue_and_claim_round_trip(conn):
    from nenapu.ingest_queue import claim_next, enqueue

    enqueue(conn, path="/tmp/session.jsonl", agent="claude-code", session_id="s1")

    job = claim_next(conn)
    assert job is not None
    assert job["path"] == "/tmp/session.jsonl"
    assert job["session_id"] == "s1"


def test_a_claimed_job_is_not_claimed_again(conn):
    """The single-flight property at the data level: two workers racing to
    claim must not both get the same job."""
    from nenapu.ingest_queue import claim_next, enqueue

    enqueue(conn, path="/tmp/session.jsonl", agent="claude-code", session_id="s1")
    first = claim_next(conn)
    second = claim_next(conn)

    assert first is not None
    assert second is None


def test_claim_returns_jobs_in_enqueue_order(conn):
    from nenapu.ingest_queue import claim_next, enqueue

    enqueue(conn, path="/tmp/a.jsonl", agent="claude-code", session_id="a")
    enqueue(conn, path="/tmp/b.jsonl", agent="claude-code", session_id="b")

    assert claim_next(conn)["session_id"] == "a"
    assert claim_next(conn)["session_id"] == "b"


def test_mark_done_and_mark_failed_are_the_terminal_states(conn):
    from nenapu.ingest_queue import claim_next, enqueue, mark_done, mark_failed

    enqueue(conn, path="/tmp/a.jsonl", agent="claude-code", session_id="a")
    enqueue(conn, path="/tmp/b.jsonl", agent="claude-code", session_id="b")
    a = claim_next(conn)
    b = claim_next(conn)

    mark_done(conn, a["id"])
    mark_failed(conn, b["id"], detail="model unavailable")

    row_a = conn.execute("SELECT state FROM ingest_queue WHERE id = ?", (a["id"],)).fetchone()
    row_b = conn.execute("SELECT state FROM ingest_queue WHERE id = ?", (b["id"],)).fetchone()
    assert row_a["state"] == "done"
    assert row_b["state"] == "failed"


def test_a_failed_job_does_not_block_the_queue(conn):
    """One bad transcript must not wedge every session behind it."""
    from nenapu.ingest_queue import claim_next, enqueue, mark_failed

    enqueue(conn, path="/tmp/bad.jsonl", agent="claude-code", session_id="bad")
    enqueue(conn, path="/tmp/good.jsonl", agent="claude-code", session_id="good")
    bad = claim_next(conn)
    mark_failed(conn, bad["id"], detail="parse error")

    nxt = claim_next(conn)
    assert nxt["session_id"] == "good"


# ---------- single-flight worker lock ----------


def test_the_worker_lock_is_exclusive(tmp_path):
    """Two worker processes must not drain the queue concurrently — the
    property that stops N discovered sessions from fanning out into N
    parallel 83-second model calls."""
    from nenapu.ingest_queue import WorkerLock

    lock_path = tmp_path / "ingest.lock"
    with WorkerLock(lock_path) as held:
        assert held is True
        second = WorkerLock(lock_path)
        assert second.try_acquire() is False


def test_the_worker_lock_releases_on_exit(tmp_path):
    from nenapu.ingest_queue import WorkerLock

    lock_path = tmp_path / "ingest.lock"
    with WorkerLock(lock_path):
        pass

    with WorkerLock(lock_path) as held:
        assert held is True


def test_two_threads_cannot_both_hold_the_lock(tmp_path):
    from nenapu.ingest_queue import WorkerLock

    lock_path = tmp_path / "ingest.lock"
    results = []

    def contend():
        lock = WorkerLock(lock_path)
        results.append(lock.try_acquire())
        if results[-1]:
            time.sleep(0.2)
            lock.release()

    with WorkerLock(lock_path):
        t = threading.Thread(target=contend)
        t.start()
        t.join()

    assert results == [False]
