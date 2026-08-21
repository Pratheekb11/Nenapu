"""Schedule `rollup_activity` on the maintenance tick.

Requirement (Task 21, "Next up" list, 2026-08-21, marked **Sonnet 5**,
depends on 12 and 13):

    "Schedule `rollup_activity` on the maintenance tick | dead code in the
    precise sense `expire_pending` was: written, tested, never called.
    Without it `standup` degrades as the ledger grows, which is what the
    downsampling was for | S | Sonnet 5 | 12, 13"

`nenapu.rollup.rollup_activity` is implemented and covered by
`tests/test_activity_rollup.py`. `grep -rn rollup_activity src/` finds it
defined and never invoked, so on every real machine the ledger keeps every
session at full detail forever and the 14d/weekly/monthly policy the user
asked for ("more than deleted, I would like a compressed cache") exists only
in a module nothing imports.

Scope boundary with `tests/test_activity_rollup.py`
--------------------------------------------------
That file pins the *fold*: week and month boundaries, per-project bucketing,
idempotence, what survives. This file pins the *schedule*: that the tick
calls it, on its own cadence, unscoped, and that a failure there cannot take
the tick down with it. Nothing here re-tests the fold arithmetic.

Assumed surface: `nenapu.maintenance.ROLLUP_CADENCE_SECONDS`, and a
`meta` key `maintenance:last_run:rollup` following the convention already
established by `audit:<scope>` and `check:<scope>`.
"""

import time
from unittest.mock import patch

import pytest

from nenapu import connect
from nenapu.store import Store

DAY = 86400.0


@pytest.fixture
def store():
    return Store(connect(":memory:"))


@pytest.fixture
def ledger(store):
    from nenapu.activity import ActivityLedger

    return ActivityLedger(store.conn)


def _session_days_ago(ledger, days_ago, *, scope="repo:demo@aaaaaaaa", paths=("app.py",)):
    at = time.time() - days_ago * DAY
    row_id = ledger.start_session(agent="claude-code", project_scope=scope,
                                  cwd="/repo", started_at=at)
    for path in paths:
        ledger.record_file_event(row_id, path=path, op="edited", tool="Edit", at=at)
    ledger.end_session(row_id, ended_at=at + 60)
    return row_id


def _marks(store):
    return {r["key"] for r in store.conn.execute(
        "SELECT key FROM meta WHERE key LIKE 'maintenance:last_run:%'"
    )}


# ---------- the cadence constant ----------


def test_the_rollup_has_its_own_cadence():
    """Not every tick: the fold is a full scan of everything older than 14
    days, and the worker ticks once per ingested session. Not weekly either —
    a month of daily use between folds is the readability problem this was
    built to prevent."""
    from nenapu.maintenance import ROLLUP_CADENCE_SECONDS

    assert DAY <= ROLLUP_CADENCE_SECONDS <= 7 * DAY


# ---------- it actually runs ----------


def test_the_tick_folds_an_old_session_into_a_rollup(store, ledger):
    """The dead-code fix, end to end rather than by asserting a call: a
    30-day-old session must be gone from `sessions` and present as a weekly
    rollup after a tick that nobody told about rollups."""
    from nenapu.maintenance import run_maintenance_tick

    old = _session_days_ago(ledger, 30)

    run_maintenance_tick(store)

    assert ledger.get_session(old) is None
    rollups = ledger.rollups_for_scope("repo:demo@aaaaaaaa")
    assert [r["period"] for r in rollups] == ["week"]
    assert rollups[0]["session_count"] == 1


def test_the_rollup_runs_even_when_no_scope_was_touched(store, ledger):
    """A machine whose worker drains a session in project A still has to
    fold project B's year-old sessions. Ageing is not scoped to whatever was
    just ingested — the same reason loop closure runs unscoped."""
    from nenapu.maintenance import run_maintenance_tick

    _session_days_ago(ledger, 200, scope="repo:other@bbbbbbbb")

    run_maintenance_tick(store, touched_scopes=["repo:demo@aaaaaaaa"])

    assert ledger.rollups_for_scope("repo:other@bbbbbbbb")


def test_recent_work_survives_the_tick(store, ledger):
    """`standup` reads the last day or two at full detail. A tick that folded
    those would break the command it was added to protect."""
    from nenapu.maintenance import run_maintenance_tick

    recent = _session_days_ago(ledger, 2)

    run_maintenance_tick(store)

    assert ledger.get_session(recent) is not None


# ---------- ...on a cadence, tracked where the others are ----------


def test_the_first_tick_records_that_the_rollup_ran(store):
    from nenapu.maintenance import run_maintenance_tick

    run_maintenance_tick(store)

    assert "maintenance:last_run:rollup" in _marks(store)


def test_a_second_tick_straight_away_does_not_fold_again(store, ledger):
    """The worker ticks once per drained session. Folding on every one of
    them is a full-table scan per ingestion for no new answer."""
    from nenapu import maintenance

    maintenance.run_maintenance_tick(store)
    with patch.object(maintenance, "rollup_activity") as rollup:
        maintenance.run_maintenance_tick(store)

    rollup.assert_not_called()


def test_once_the_cadence_has_elapsed_it_folds_again(store, ledger):
    """The other half: a machine that ingests one session a week must not
    stop ageing its ledger because a mark was written once."""
    from nenapu import maintenance

    maintenance.run_maintenance_tick(store)
    stale = time.time() - maintenance.ROLLUP_CADENCE_SECONDS - 60
    store.conn.execute(
        "UPDATE meta SET value = ? WHERE key = 'maintenance:last_run:rollup'", (str(stale),)
    )
    store.conn.commit()

    with patch.object(maintenance, "rollup_activity") as rollup:
        maintenance.run_maintenance_tick(store)

    rollup.assert_called_once()


def test_the_rollup_is_given_the_ledger_it_should_fold(store):
    """Guards the shape of the call rather than only that something ran: the
    fold takes an `ActivityLedger` over this store's connection, not a new
    one over the default database path."""
    from nenapu import maintenance
    from nenapu.activity import ActivityLedger

    with patch.object(maintenance, "rollup_activity") as rollup:
        maintenance.run_maintenance_tick(store)

    passed = rollup.call_args.args[0]
    assert isinstance(passed, ActivityLedger)
    assert passed.conn is store.conn


# ---------- and cannot take the tick down ----------


def test_a_failing_rollup_does_not_break_the_tick(store):
    """After task 19 the tick runs at the end of every hook-driven drain. An
    exception escaping here is an exception in a Stop hook — the thing every
    other path in this file is written to avoid."""
    from nenapu import maintenance

    with patch.object(maintenance, "rollup_activity", side_effect=RuntimeError("boom")):
        maintenance.run_maintenance_tick(store)  # must not raise

    assert "maintenance:last_run:expire_pending" in _marks(store)


def test_a_failing_rollup_is_retried_on_the_next_tick(store):
    """A cadence mark written before the work succeeded would silence the
    fold for a whole day on the strength of a run that did nothing."""
    from nenapu import maintenance

    with patch.object(maintenance, "rollup_activity", side_effect=RuntimeError("boom")):
        maintenance.run_maintenance_tick(store)

    with patch.object(maintenance, "rollup_activity") as rollup:
        maintenance.run_maintenance_tick(store)

    rollup.assert_called_once()


def test_expire_pending_still_runs_every_tick(store):
    """Green pin (task 12). Adding a cadence-gated job must not accidentally
    put the cheap unconditional one behind the same gate."""
    from nenapu.maintenance import run_maintenance_tick

    with patch.object(store.ledger, "expire_pending") as expire:
        run_maintenance_tick(store)
        run_maintenance_tick(store)

    assert expire.call_count == 2


# ---------- through the worker, which is what calls the tick ----------


def test_draining_the_queue_ages_the_ledger(tmp_path):
    """The integration the task is actually asking for: nobody runs
    `run_maintenance_tick` by hand, so this has to happen off the back of an
    ordinary ingestion."""
    import json

    from nenapu.activity import ActivityLedger
    from nenapu.ingest_queue import enqueue
    from nenapu.worker import drain

    db = tmp_path / "s.db"
    store = Store(connect(str(db)))
    ledger = ActivityLedger(store.conn)
    _session_days_ago(ledger, 120, scope="repo:old@cccccccc")

    transcript = tmp_path / "s-1.jsonl"
    transcript.write_text(json.dumps({
        "type": "user", "sessionId": "s-1", "cwd": str(tmp_path),
        "message": {"role": "user", "content": [{"type": "text", "text": "hi"}]},
    }))
    enqueue(store.conn, path=str(transcript), agent="claude-code")

    with patch("nenapu.worker.observe_transcript", return_value=[]):
        drain(store, lock_path=tmp_path / "worker.lock")

    assert [r["period"] for r in ledger.rollups_for_scope("repo:old@cccccccc")] == ["month"]
