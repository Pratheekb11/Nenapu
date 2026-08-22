"""Executable verification.

A fact can carry a shell command that proves it. `verify_expect` is a substring
the output must contain; if it is absent, exit code 0 is the assertion.

This is the cheapest anti-staleness signal in the system — no model call, no
judgement, just a fact that either still holds or does not.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass, field

from .approval import is_approved
from .models import Fact, VerifyStatus, now
from .store import Store
from .db import commit

DEFAULT_TIMEOUT = 20


@dataclass
class VerifyResult:
    fact_id: int
    status: str
    detail: str
    # What applying this result changed downstream: recalls graded, dependents
    # marked suspect, dependents reinstated.
    fallout: dict = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.status == VerifyStatus.PASS


def run_check(
    fact: Fact,
    *,
    timeout: int = DEFAULT_TIMEOUT,
    cwd: str | None = None,
    conn=None,
) -> VerifyResult:
    """Run a fact's check — but only if a human approved this exact command.

    `conn` is required to consult the approval ledger. It is keyword-optional
    purely so existing call sites fail loudly rather than silently running
    unapproved shell: without it, nothing executes.
    """
    if not fact.verify_cmd:
        return VerifyResult(fact.id, VerifyStatus.NONE, "no check defined")

    if conn is None or not is_approved(conn, fact.verify_cmd):
        return VerifyResult(
            fact.id,
            VerifyStatus.BLOCKED,
            f"not approved to run: {fact.verify_cmd[:120]}"
            + (" (no store connection supplied)" if conn is None else ""),
        )

    try:
        proc = subprocess.run(
            fact.verify_cmd,
            shell=True,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=cwd,
        )
    except subprocess.TimeoutExpired:
        return VerifyResult(fact.id, VerifyStatus.ERROR, f"timed out after {timeout}s")
    except Exception as exc:  # the check itself is broken; says nothing about the fact
        return VerifyResult(fact.id, VerifyStatus.ERROR, f"{type(exc).__name__}: {exc}")

    output = (proc.stdout + proc.stderr).strip()
    tail = output[-400:]

    if fact.verify_expect:
        if fact.verify_expect in output:
            return VerifyResult(fact.id, VerifyStatus.PASS, f"matched {fact.verify_expect!r}")
        return VerifyResult(
            fact.id, VerifyStatus.FAIL, f"expected {fact.verify_expect!r}; got: {tail}"
        )

    if proc.returncode == 0:
        return VerifyResult(fact.id, VerifyStatus.PASS, f"exit 0: {tail[:200]}")
    return VerifyResult(fact.id, VerifyStatus.FAIL, f"exit {proc.returncode}: {tail}")


def apply_result(store: Store, result: VerifyResult) -> dict:
    """Record a check result and let it propagate.

    A check outcome is not just a flag on one row. It is a grading signal for
    everyone who recently acted on the fact, and a truth signal for everything
    derived from it — which is why this returns what it touched.
    """
    if result.status == VerifyStatus.BLOCKED:
        store.conn.execute(
            "UPDATE facts SET verify_status=?, verify_detail=?, updated_at=? WHERE id=?",
            (result.status, result.detail, now(), result.fact_id),
        )
        commit(store.conn)
        return {"graded_recalls": 0, "cascaded": [], "restored": []}

    ts = now()
    if result.status == VerifyStatus.PASS:
        # A pass is fresh evidence: reset the decay clock as well as the flag.
        store.conn.execute(
            "UPDATE facts SET verify_status=?, verify_last_run=?, verify_detail=?,"
            " last_verified_at=?, updated_at=? WHERE id=?",
            (result.status, ts, result.detail, ts, ts, result.fact_id),
        )
    else:
        store.conn.execute(
            "UPDATE facts SET verify_status=?, verify_last_run=?, verify_detail=?,"
            " updated_at=? WHERE id=?",
            (result.status, ts, result.detail, ts, result.fact_id),
        )
    store._journal("verify", fact_id=result.fact_id, actor="verifier", detail=result.status)
    commit(store.conn)

    fallout: dict = {"graded_recalls": 0, "cascaded": [], "restored": []}

    if result.status == VerifyStatus.BLOCKED:
        # An unapproved check is not evidence either way; it must not be
        # allowed to look like a failure and demote the fact.
        return fallout

    if result.status == VerifyStatus.FAIL:
        fallout["graded_recalls"] = store.ledger.blame_recent_recalls(
            result.fact_id, source="verification", note=result.detail[:200]
        )
        fallout["cascaded"] = store.graph.cascade_falsification(
            result.fact_id, "check failed"
        )
    elif result.status == VerifyStatus.PASS:
        fallout["graded_recalls"] = store.ledger.credit_recent_recalls(
            result.fact_id, source="verification", note="check passed"
        )
        fallout["restored"] = store.graph.clear_suspicion(result.fact_id)

    return fallout


def verify_scope(
    store: Store,
    scope: str | None = None,
    *,
    only_stale_after_days: float | None = None,
    timeout: int = DEFAULT_TIMEOUT,
    cwd: str | None = None,
) -> list[VerifyResult]:
    """Re-run every check in a scope. Safe to wire to cron or a session hook."""
    results: list[VerifyResult] = []
    cutoff = None if only_stale_after_days is None else now() - only_stale_after_days * 86400

    for fact in store.list_facts(scope=scope, limit=10_000):
        if not fact.verify_cmd:
            continue
        if cutoff is not None and fact.verify_last_run and fact.verify_last_run > cutoff:
            continue
        result = run_check(fact, timeout=timeout, cwd=cwd, conn=store.conn)
        result.fallout = apply_result(store, result)
        results.append(result)
    return results
