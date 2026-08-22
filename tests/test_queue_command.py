"""The stranded claim, and the command that can finally reach it.

Requirement (plan "Harden the four incidents into guarantees", Phase C,
tasks C1 and C2, marked **Sonnet 5**):

    "A claim outlived its worker: job stranded permanently, unreachable to any
    command."

Half of that was fixed in `2bfb51a`: `release_stale_claims` returns a claim
older than an hour, and `worker.drain` calls it when it takes the lock. The
other half is still true. `release_stale_claims` has exactly one caller,
nothing shows the queue at all, and `failed` is terminal and never surfaced.
Recovery today requires knowing that spawning a drain is what reaps, which is
not something the tool tells anyone.

So: a command that shows what the queue holds, and a door for releasing a
claim whose worker is known to be dead without waiting out the hour.

C2 is the other end of the same gap. `release_stale_claims` filters on
`claimed_at IS NOT NULL AND claimed_at < ?`, so a row that reaches `claimed`
with no timestamp is released by nothing, ever. `claim_next` always sets one,
which is exactly why the case should be handled rather than trusted: it can
only arise from something already wrong, and the state it produces is the
permanent strand this whole task exists to end.

Scope boundary with `tests/test_worker_ingest_recovery.py`
----------------------------------------------------------
That file pins the reaping policy: the hour, that a fresh claim is left alone,
that a released job keeps what it carried. Nothing here re-tests the policy.
This file pins the command over it, and the one row shape that policy misses.
"""

import os
import subprocess
import sys

import pytest

import nenapu.cli as cli
from nenapu import connect
from nenapu.ingest_queue import (
    STATE_CLAIMED,
    claim_next,
    enqueue,
    mark_failed,
    release_stale_claims,
)
from nenapu.models import now

HOUR = 3600.0


@pytest.fixture
def db(tmp_path):
    return tmp_path / "s.db"


def _run(args, db, **env):
    return subprocess.run(
        [sys.executable, "-m", "nenapu.cli", *args, "--db", str(db)],
        capture_output=True, text=True,
        env={**os.environ, "PYTHONPATH": "src", "NENAPU_NO_BANNER": "1", **env},
    )


def _conn(db):
    return connect(str(db))


def _stranded(conn, path: str, *, age: float) -> int:
    """A job whose worker took it and never came back."""
    job_id = enqueue(conn, path=path, agent="claude-code", session_id="s-1")
    claim_next(conn)
    conn.execute("UPDATE ingest_queue SET claimed_at = ? WHERE id = ?",
                 (now() - age, job_id))
    conn.commit()
    return job_id


def _state(conn, job_id: int) -> str:
    return conn.execute(
        "SELECT state FROM ingest_queue WHERE id = ?", (job_id,)
    ).fetchone()[0]


# ---------- C1: the queue is visible ----------


def test_queue_is_a_registered_command(db):
    result = _run(["queue", "--help"], db)

    assert result.returncode == 0


def test_queue_is_listed_under_upkeep():
    """Recovery is upkeep. A command nobody can find is the state this task is
    fixing, one level up."""
    panels = {
        (command.name or command.callback.__name__): command.rich_help_panel
        for command in cli.app.registered_commands
    }

    assert panels.get("queue") == cli.UPKEEP


def test_queue_reports_what_is_waiting(db):
    conn = _conn(db)
    enqueue(conn, path="/t/a.jsonl", agent="claude-code")
    enqueue(conn, path="/t/b.jsonl", agent="claude-code")
    conn.close()

    result = _run(["queue"], db)

    assert "pending" in result.stdout
    assert "2" in result.stdout


def test_queue_shows_a_claim_and_how_old_it_is(db):
    """The age is the whole diagnosis: a claim minutes old is a worker still
    inside its model call, and one hours old is a worker that died."""
    conn = _conn(db)
    _stranded(conn, "/t/stranded.jsonl", age=4 * HOUR)
    conn.close()

    result = _run(["queue"], db)

    assert "stranded.jsonl" in result.stdout
    assert "4" in result.stdout


def test_queue_surfaces_a_failed_job_and_why(db):
    """`failed` is terminal and nothing else ever shows it, so a transcript
    that cannot be read fails silently and forever."""
    conn = _conn(db)
    job_id = enqueue(conn, path="/t/bad.jsonl", agent="claude-code")
    mark_failed(conn, job_id, detail="transcript is not JSON")
    conn.close()

    result = _run(["queue"], db)

    assert "bad.jsonl" in result.stdout
    assert "not JSON" in result.stdout


def test_an_empty_queue_says_so_rather_than_printing_nothing(db):
    result = _run(["queue"], db)

    assert result.returncode == 0
    assert result.stdout.strip()


# ---------- C1: and reachable ----------


def test_release_returns_a_stale_claim_to_pending(db):
    conn = _conn(db)
    job_id = _stranded(conn, "/t/stranded.jsonl", age=4 * HOUR)
    conn.close()

    _run(["queue", "--release"], db)

    conn = _conn(db)
    assert _state(conn, job_id) == "pending"
    conn.close()


def test_release_reports_how_many_it_freed(db):
    conn = _conn(db)
    _stranded(conn, "/t/stranded.jsonl", age=4 * HOUR)
    conn.close()

    result = _run(["queue", "--release"], db)

    assert "1" in result.stdout


def test_release_leaves_a_live_worker_alone(db):
    """A claim taken moments ago belongs to a worker still inside an 83-second
    model call. Stealing it buys two extractions of one transcript."""
    conn = _conn(db)
    job_id = _stranded(conn, "/t/running.jsonl", age=30.0)
    conn.close()

    _run(["queue", "--release"], db)

    conn = _conn(db)
    assert _state(conn, job_id) == STATE_CLAIMED
    conn.close()


def test_older_than_releases_a_claim_the_hour_would_have_kept(db):
    """Someone who knows the worker is dead should not have to wait out an
    hour of a policy meant for the case where nobody knows."""
    conn = _conn(db)
    job_id = _stranded(conn, "/t/running.jsonl", age=30.0)
    conn.close()

    _run(["queue", "--release", "--older-than", "0"], db)

    conn = _conn(db)
    assert _state(conn, job_id) == "pending"
    conn.close()


def test_a_released_job_can_be_queued_again(db):
    """The strand was never the claim itself: `enqueue_once` refuses to
    re-queue a path that is pending or claimed, so a dead worker's claim also
    blocked every later attempt to get the job done."""
    from nenapu.ingest_queue import enqueue_once

    conn = _conn(db)
    _stranded(conn, "/t/stranded.jsonl", age=4 * HOUR)
    conn.close()

    _run(["queue", "--release", "--older-than", "0"], db)

    conn = _conn(db)
    # Still refused, because releasing returns the job to pending rather than
    # dropping it: the same job is retried, not duplicated.
    assert enqueue_once(conn, path="/t/stranded.jsonl", agent="claude-code") is None
    assert _state(conn, 1) == "pending"
    conn.close()


# ---------- C2: a claim with no timestamp is still a claim ----------


def test_a_claim_with_no_timestamp_is_released(db):
    """Nothing produces this row on purpose, which is the point: it can only
    come from something already wrong, and the state it leaves behind is the
    permanent strand. `claimed_at IS NOT NULL` in the filter meant no age was
    ever old enough."""
    conn = _conn(db)
    job_id = enqueue(conn, path="/t/timeless.jsonl", agent="claude-code")
    claim_next(conn)
    conn.execute("UPDATE ingest_queue SET claimed_at = NULL WHERE id = ?", (job_id,))
    conn.commit()

    freed = release_stale_claims(conn)

    assert freed == 1
    assert _state(conn, job_id) == "pending"
    conn.close()


def test_a_timeless_claim_is_released_however_long_the_window(db):
    """It has no age to compare, so no window can protect it."""
    conn = _conn(db)
    job_id = enqueue(conn, path="/t/timeless.jsonl", agent="claude-code")
    claim_next(conn)
    conn.execute("UPDATE ingest_queue SET claimed_at = NULL WHERE id = ?", (job_id,))
    conn.commit()

    release_stale_claims(conn, older_than=10 * 24 * HOUR)

    assert _state(conn, job_id) == "pending"
    conn.close()
