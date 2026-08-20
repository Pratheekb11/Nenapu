"""The self-maintenance tick: the ingestion worker becomes its own garbage collector.

Requirement (Task 12, priority-ordered task list): "Self-maintenance tick on
the worker | ends the manual-GC role; `expire_pending` is currently dead
code." Depends on Task 8 (the ingest queue worker).

The user's own words, and the constraint this task exists to satisfy:

    "I don't want to be the garbage collector who cleans up whenever he sees
    an overspill."

The plan's audit found every lifecycle mechanism already implemented and
already tested, but nothing schedules any of them:

    | mechanism                    | exists? | automatic? |
    |-------------------------------|---------|------------|
    | belief decay by half-life     | yes     | yes (continuous) |
    | dedupe (0.85 similarity)      | yes     | no — manual `nenapu tidy` |
    | LLM merge into summary fact   | yes     | no — manual |
    | LLM staleness audit           | yes     | no — manual `nenapu audit` |
    | re-run checks                 | yes     | no — manual `nenapu check` |
    | recall expiry (7d)            | yes     | **no — dead code, never called** |

Task 12's job is one function that the ingest-queue worker (Task 8) calls
after draining: expire pending recalls every tick, run `distill.dedupe` on
scopes touched this tick, and track `last_run` timestamps in the existing
`meta` table so heavier jobs (`distill`, `audit`, `check`) run on longer
cadences instead of every tick.

Assumes a new `nenapu.maintenance` module: `run_maintenance_tick(store, *,
touched_scopes=())`.
"""

import time
from unittest.mock import patch

import pytest

from nenapu import connect
from nenapu.models import Fact, Outcome
from nenapu.outcomes import PENDING_EXPIRY_SECONDS
from nenapu.store import Store


@pytest.fixture
def store():
    return Store(connect(":memory:"))


def test_a_maintenance_tick_expires_stale_pending_recalls(store):
    """This is the dead-code fix, verified end to end rather than just
    calling `expire_pending` directly: nothing in the codebase invokes it
    today, so this proves *something* now does."""
    from nenapu.maintenance import run_maintenance_tick

    fact, _ = store.write(Fact(text="a fact that got recalled once"))
    recall_id = store.ledger.log(fact.id, session_id="s1")
    old = time.time() - PENDING_EXPIRY_SECONDS - 3600
    store.conn.execute("UPDATE recalls SET created_at = ? WHERE id = ?", (old, recall_id))
    store.conn.commit()

    run_maintenance_tick(store)

    recall = store.ledger.get(recall_id)
    assert recall.outcome == Outcome.NEUTRAL


def test_a_fresh_pending_recall_is_left_alone(store):
    from nenapu.maintenance import run_maintenance_tick

    fact, _ = store.write(Fact(text="a fact recalled just now"))
    recall_id = store.ledger.log(fact.id, session_id="s1")

    run_maintenance_tick(store)

    assert store.ledger.get(recall_id).outcome == Outcome.PENDING


def test_a_maintenance_tick_runs_dedupe_on_touched_scopes(store):
    """Task 8's worker knows which scope(s) it just ingested into; this is
    the "on touched scopes" half of the tick, not a full-store sweep on
    every session end."""
    from nenapu.maintenance import run_maintenance_tick

    with patch("nenapu.maintenance.dedupe") as mock_dedupe:
        mock_dedupe.return_value = 0
        run_maintenance_tick(store, touched_scopes=["repo:demo@aaaaaaaa"])

    mock_dedupe.assert_called_once()
    _args, kwargs = mock_dedupe.call_args
    assert kwargs.get("scope") == "repo:demo@aaaaaaaa" or "repo:demo@aaaaaaaa" in _args


def test_a_tick_with_no_touched_scopes_does_not_run_dedupe(store):
    """Nothing was just ingested — there is nothing to have gone duplicate,
    so the tick should not pay for a full-store dedupe pass every time the
    worker merely wakes up to check the queue."""
    from nenapu.maintenance import run_maintenance_tick

    with patch("nenapu.maintenance.dedupe") as mock_dedupe:
        run_maintenance_tick(store, touched_scopes=[])

    mock_dedupe.assert_not_called()


def test_last_run_timestamps_are_recorded_in_meta(store):
    """The mechanism that lets heavier jobs (distill, audit, check) run on
    longer cadences instead of every tick."""
    from nenapu.maintenance import run_maintenance_tick

    run_maintenance_tick(store)

    row = store.conn.execute(
        "SELECT value FROM meta WHERE key = 'maintenance:last_run:expire_pending'"
    ).fetchone()
    assert row is not None
    assert float(row["value"]) <= time.time()


def test_a_heavier_job_does_not_rerun_before_its_cadence_elapses(store):
    """`nenapu audit` is a model call; running it on every single tick would
    turn "the worker cleans up after itself" back into the manual-GC problem
    the user explicitly rejected, just automated into a cost sink instead."""
    from nenapu.maintenance import run_maintenance_tick

    with patch("nenapu.maintenance.run_audit") as mock_audit:
        run_maintenance_tick(store, touched_scopes=["repo:demo@aaaaaaaa"])
        first_calls = mock_audit.call_count
        run_maintenance_tick(store, touched_scopes=["repo:demo@aaaaaaaa"])
        second_calls = mock_audit.call_count

    assert second_calls == first_calls, "audit re-ran before its cadence elapsed"
