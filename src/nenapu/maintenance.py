"""The self-maintenance tick: the worker becomes its own garbage collector.

Every lifecycle mechanism the store needed already existed and was already
tested — decay, dedupe, distill, audit, checks, recall expiry. Nothing
scheduled any of them. `expire_pending` in particular was dead code: defined,
tested in isolation, never called from anywhere real. This is the one
function the ingest-queue worker calls after draining, so cleaning up the
store stops being a job a human has to remember to do.

Loop closure runs here too. A loop nobody ever closes is the failure mode
the feature cannot survive — being told about work that shipped months ago
destroys trust in the whole block — so the evidence is re-checked on every
tick rather than only when someone runs a command.

Distillation (merging related facts once a scope exceeds its token budget)
is deliberately not wired in here yet — it needs a budget heuristic this
task does not have a tested contract for, and guessing one would be exactly
the kind of half-finished mechanism this project avoids.
"""

from __future__ import annotations

from typing import Sequence

from .activity import ActivityLedger
from .audit import audit as run_audit
from .db import commit
from .distill import dedupe
from .loops import LoopBook
from .models import now
from .rollup import rollup_activity
from .store import Store
from .verify import verify_scope as run_check

# Cheap and safe to run on every tick. Expensive ones (a model call, a shell
# command) run per touched scope on their own longer cadence instead.
AUDIT_CADENCE_SECONDS = 7 * 86400.0
CHECK_CADENCE_SECONDS = 1 * 86400.0
# The fold is a full scan of everything older than fourteen days, and the
# worker ticks once per ingested session, so it cannot ride every tick. A week
# is too slow the other way: a month of daily use between folds is the
# readability problem the downsampling exists to prevent.
ROLLUP_CADENCE_SECONDS = 1 * 86400.0


def _last_run(store: Store, key: str) -> float | None:
    row = store.conn.execute(
        "SELECT value FROM meta WHERE key = ?", (f"maintenance:last_run:{key}",)
    ).fetchone()
    return float(row["value"]) if row else None


def _mark_run(store: Store, key: str) -> None:
    store.conn.execute(
        "INSERT OR REPLACE INTO meta(key, value) VALUES (?, ?)",
        (f"maintenance:last_run:{key}", str(now())),
    )
    commit(store.conn)


def _due(store: Store, key: str, cadence_seconds: float) -> bool:
    last = _last_run(store, key)
    return last is None or (now() - last) >= cadence_seconds


def run_maintenance_tick(store: Store, *, touched_scopes: Sequence[str] = ()) -> None:
    """Run whatever upkeep is due.

    `expire_pending` runs every tick — nearly free, and otherwise never runs
    at all. `rollup_activity` folds the ledger by age on its own daily
    cadence. `dedupe` runs once per scope the worker just wrote to, since
    duplicates can only appear where something was just ingested. `audit`
    and `check` cost real money or time, so they run per touched scope on
    their own longer cadence, tracked in `meta`.
    """
    store.ledger.expire_pending()
    _mark_run(store, "expire_pending")

    # Cheap: two indexed reads per open loop, no model call. Run unscoped
    # because evidence for a loop can land in a session the worker did not
    # just ingest.
    LoopBook(store.conn).close_satisfied(ActivityLedger(store.conn))

    # Ageing the work log, on its own cadence. Unscoped for the same reason
    # loop closure is: a worker that just drained a session in one project
    # still has to fold another project's year-old sessions. The mark is
    # written only after the fold returns, so a run that failed is retried on
    # the next tick rather than silenced for a day, and the failure itself
    # stays inside the tick — after task 19 this runs inside a Stop hook.
    if _due(store, "rollup", ROLLUP_CADENCE_SECONDS):
        try:
            rollup_activity(ActivityLedger(store.conn))
        except Exception:  # noqa: BLE001 — upkeep must never break a session
            pass
        else:
            _mark_run(store, "rollup")

    for scope in touched_scopes:
        dedupe(store, scope=scope)

        if _due(store, f"audit:{scope}", AUDIT_CADENCE_SECONDS):
            run_audit(store, scope=scope)
            _mark_run(store, f"audit:{scope}")

        if _due(store, f"check:{scope}", CHECK_CADENCE_SECONDS):
            run_check(store, scope=scope)
            _mark_run(store, f"check:{scope}")
