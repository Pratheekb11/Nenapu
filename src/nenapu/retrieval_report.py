"""Read the recall ledger and answer one question: is retrieval what fails?

Vectors and the entity tier were left unbuilt on purpose — "building them
first would invent the design that evidence is supposed to choose." This
module is that evidence, and the rule that reads it, written down as code so
the answer is counted rather than argued for.

The rule itself is the plan's, unchanged:

    recalled facts graded `bad`        -> wrong facts are surfacing, so
                                          semantic retrieval is the fix
    few bad grades, sessions given
    nothing                            -> a coverage problem, which is a
                                          scope or budget question instead
    the bad ones came from another
    project                            -> scoping already fixed this, and
                                          vectors would have been wasted

Nothing here calls a model and nothing here writes: a report that graded or
expired anything on its way past would be changing the measurement it is
reporting. And a store with nine graded recalls says so rather than
producing a number-shaped opinion about a large piece of work.
"""

from __future__ import annotations

import sqlite3

from .models import Outcome, now as _now
from .store import Store

DAY = 86400.0

# The plan asks the question over two weeks of hook-path recalls.
MIN_DAYS_OF_DATA = 14
# Below this the ledger is one person's week, not a measurement. Thirty is
# the floor at which a bad rate stops moving by ten points per grade.
MIN_GRADED_RECALLS = 30
# ...and those grades have to come from more than one sitting: thirty grades
# that all landed on one afternoon are one session's opinion. Half the window
# is the point at which the data covers days rather than hours.
MIN_SPAN_FRACTION = 0.5
# One recall in three going wrong is a retrieval failure rather than noise;
# below it the misses are individual facts, which is a different repair.
BAD_RATE_THRESHOLD = 0.3
# Sessions that were handed nothing while the store held facts for their
# scope. Under half and the problem is that memory is not arriving at all,
# which no amount of ranking inside it would fix.
COVERAGE_FLOOR = 0.5

INSUFFICIENT = "insufficient-evidence"
BUILD_VECTORS = "build-vectors"
COVERAGE_PROBLEM = "coverage-problem"
ALREADY_FIXED = "already-fixed-by-scoping"
NOT_THE_PROBLEM = "retrieval-is-not-the-problem"


def retrieval_evidence(
    store: Store, *, window_days: int = MIN_DAYS_OF_DATA, now: float | None = None,
) -> dict:
    """Count what the recall ledger holds inside the window.

    Everything here is a count over `recalls`, `facts` and `sessions`. The
    verdict is carried on the result so a caller cannot report the numbers
    and the decision separately and let them drift apart.
    """
    at = now if now is not None else _now()
    since = at - window_days * DAY
    conn = store.conn

    evidence = {
        "window_days": window_days,
        **_recall_counts(conn, since),
        "wrong_project": _wrong_project_recalls(conn, since),
        **_coverage(conn, since),
    }
    evidence["verdict"] = decide(evidence)
    return evidence


def _recall_counts(conn: sqlite3.Connection, since: float) -> dict:
    """Outcomes, the bad rate over graded recalls, and the span of the data.

    Pending recalls are reported and excluded from the rate: a recall nobody
    graded is not evidence of anything, and counting one as a success would
    make every store look healthy.
    """
    row = conn.execute(
        "SELECT COUNT(*) AS total,"
        " SUM(outcome = ?) AS good, SUM(outcome = ?) AS bad,"
        " SUM(outcome = ?) AS neutral, SUM(outcome = ?) AS pending,"
        " MIN(created_at) AS first_at, MAX(created_at) AS last_at"
        " FROM recalls WHERE created_at >= ?",
        (Outcome.GOOD, Outcome.BAD, Outcome.NEUTRAL, Outcome.PENDING, since),
    ).fetchone()

    good, bad, neutral = (row["good"] or 0), (row["bad"] or 0), (row["neutral"] or 0)
    graded = good + bad + neutral
    span = (row["last_at"] - row["first_at"]) if row["total"] else 0.0
    return {
        "good": good,
        "bad": bad,
        "neutral": neutral,
        "pending": row["pending"] or 0,
        "graded": graded,
        "bad_rate": (bad / graded) if graded else 0.0,
        "days_of_data": span / DAY,
    }


def _wrong_project_recalls(conn: sqlite3.Connection, since: float) -> int:
    """Bad recalls that surfaced another project's fact.

    The branch that decides whether this work is already done: a fact from
    elsewhere arriving here is a scope failure, not a similarity failure.
    Global facts are excluded because they are meant to surface everywhere —
    counting them would manufacture the verdict that says vectors are
    unnecessary.
    """
    row = conn.execute(
        "SELECT COUNT(*) AS n FROM recalls r"
        " JOIN facts f ON f.id = r.fact_id"
        " WHERE r.created_at >= ? AND r.outcome = ? AND f.scope <> 'global'"
        " AND f.scope <> (SELECT s.project_scope FROM sessions s"
        "                 WHERE s.external_id = r.session_id LIMIT 1)",
        (since, Outcome.BAD),
    ).fetchone()
    return row["n"] or 0


def _coverage(conn: sqlite3.Connection, since: float) -> dict:
    """How many sessions were given any memory at all.

    "Obviously missing facts" cannot be observed directly, but a session that
    ran to its end with no recall logged, while the store held facts for its
    scope, is the measurable shadow of it. Sessions in a scope the store
    knows nothing about are left out: they were given nothing because there
    was nothing to give.
    """
    row = conn.execute(
        "SELECT COUNT(*) AS total, SUM(EXISTS("
        "   SELECT 1 FROM recalls r"
        "   WHERE r.session_id = s.external_id AND r.created_at >= :since"
        " )) AS given"
        " FROM sessions s"
        " WHERE s.started_at >= :since AND s.external_id IS NOT NULL"
        " AND EXISTS(SELECT 1 FROM facts f WHERE f.status = 'active'"
        "            AND (f.scope = s.project_scope OR f.scope = 'global'))",
        {"since": since},
    ).fetchone()

    total = row["total"] or 0
    given = row["given"] or 0
    return {
        "sessions_with_recalls": given,
        "sessions_without_recalls": total - given,
        # No sessions to judge is not a coverage failure; saying 0.0 here
        # would report a problem on a store that has simply not been used.
        "coverage_rate": (given / total) if total else 1.0,
    }


def decide(evidence: dict) -> str:
    """The plan's rule, executed against the counts.

    Order matters. Sufficiency comes first, because every branch below it is
    a claim about a large piece of work. Wrong-project bad recalls are
    checked before vectors, since that is the case where the repair already
    shipped. Coverage is last of the failures: it only means anything once
    the facts that did arrive were mostly right.
    """
    covered_enough = evidence["days_of_data"] >= MIN_DAYS_OF_DATA * MIN_SPAN_FRACTION
    if evidence["graded"] < MIN_GRADED_RECALLS or not covered_enough:
        return INSUFFICIENT

    if evidence["bad_rate"] >= BAD_RATE_THRESHOLD:
        if evidence["wrong_project"] >= evidence["bad"] / 2:
            return ALREADY_FIXED
        return BUILD_VECTORS

    if evidence["coverage_rate"] < COVERAGE_FLOOR:
        return COVERAGE_PROBLEM

    return NOT_THE_PROBLEM


VERDICT_MEANING = {
    INSUFFICIENT: "not enough graded recalls yet — insufficient evidence to decide",
    BUILD_VECTORS: "wrong facts are surfacing — semantic retrieval is the fix",
    COVERAGE_PROBLEM: "sessions are being given nothing — a scope or budget problem",
    ALREADY_FIXED: "the bad recalls came from other projects — scoping already fixed this",
    NOT_THE_PROBLEM: "retrieval is not what fails here — leave it alone",
}
