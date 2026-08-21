"""Re-read the recall ledger and decide task 17.

Requirement (Task 23, "Next up" list, 2026-08-21, marked **Sonnet 5**,
depends on 14 and 19):

    "Re-read the recall ledger and decide 17 | two weeks of hook-path recalls
    is the evidence 17 and 18 are gated on. The decision is 'is retrieval
    what fails', and it is answerable from `recalls` once there is data |
    S | Sonnet 5 | 14, 19"

The decision rule is already written down, in Stage D of the plan:

    - "recalled facts graded `bad` → wrong facts are surfacing → semantic
      retrieval is the fix, proceed;
    - few bad grades but obviously missing facts → a recall/coverage problem,
      which is different and may be a scope or budget issue;
    - problem was 'right fact, wrong project' → Stage A1 already fixed it,
      and vectors would have been wasted effort."

So this task is not "look at the numbers and form an opinion". It is: make
the rule executable, so the answer is read off the ledger rather than argued
for. That is the same reason `calibrate.py` exists — an audit backend earns
trust by probe, not by name — and the same reason 17 and 18 were left
unbuilt: "building them first would invent the design that evidence is
supposed to choose."

Assumed surface: a new module `nenapu.retrieval_report` ::

    retrieval_evidence(store, *, window_days=14, now=None) -> dict
    decide(evidence) -> str
    MIN_GRADED_RECALLS, MIN_DAYS_OF_DATA, BAD_RATE_THRESHOLD, COVERAGE_FLOOR

    Verdicts: "insufficient-evidence" | "build-vectors"
              | "coverage-problem" | "already-fixed-by-scoping"
              | "retrieval-is-not-the-problem"

plus `nenapu retrieval` printing the evidence and the verdict.

The verdict this is *most* likely to return on a young store is
`insufficient-evidence`, and that is the feature: a store with nine graded
recalls must say so rather than produce a number-shaped opinion about a
large piece of work. Tasks 17 and 18 stay unbuilt until this command says
otherwise — no tests are written here for either, deliberately.
"""

import os
import subprocess
import sys
import time
from unittest.mock import patch

import pytest

from nenapu import connect
from nenapu.models import Fact, Outcome
from nenapu.store import Store

DAY = 86400.0


@pytest.fixture
def store():
    return Store(connect(":memory:"))


def _fact(store, text, *, scope="global"):
    fact, _ = store.write(Fact(text=text, scope=scope))
    return fact


def _recall(store, fact, *, session_id="s-1", outcome=Outcome.PENDING, days_ago=1.0,
            source="test"):
    recall_id = store.ledger.log(fact.id, session_id=session_id)
    at = time.time() - days_ago * DAY
    store.conn.execute("UPDATE recalls SET created_at = ? WHERE id = ?", (at, recall_id))
    if outcome != Outcome.PENDING:
        store.conn.execute(
            "UPDATE recalls SET outcome = ?, outcome_source = ?, outcome_at = ? WHERE id = ?",
            (outcome, source, at, recall_id),
        )
    store.conn.commit()
    return recall_id


def _session(store, *, external_id, scope, days_ago=1.0):
    from nenapu.activity import ActivityLedger

    return ActivityLedger(store.conn).start_session(
        agent="claude-code", project_scope=scope, cwd="/repo",
        started_at=time.time() - days_ago * DAY, external_id=external_id,
    )


def _spread(store, *, good=0, bad=0, neutral=0, scope="global", session="s-1"):
    """Graded recalls spread across the window, so the data-span check is
    satisfied by anything that means to exercise a verdict past it."""
    n = good + bad + neutral
    outcomes = [Outcome.GOOD] * good + [Outcome.BAD] * bad + [Outcome.NEUTRAL] * neutral
    for i, outcome in enumerate(outcomes):
        fact = _fact(store, f"fact number {i} about the {outcome} path", scope=scope)
        _recall(store, fact, session_id=session, outcome=outcome,
                days_ago=0.5 + (13.0 * i / max(n - 1, 1)))


# ---------- the evidence is counted, not judged ----------


def test_the_report_counts_the_graded_recalls_in_the_window(store):
    from nenapu.retrieval_report import retrieval_evidence

    _spread(store, good=4, bad=2, neutral=1)

    evidence = retrieval_evidence(store)

    assert evidence["good"] == 4
    assert evidence["bad"] == 2
    assert evidence["neutral"] == 1
    assert evidence["graded"] == 7


def test_pending_recalls_are_reported_but_not_counted_as_evidence(store):
    """A recall nobody graded is not evidence of anything — the same
    reasoning `expire_pending` encodes. Counting them as successes would
    make every store look healthy."""
    from nenapu.retrieval_report import retrieval_evidence

    _spread(store, good=2)
    _recall(store, _fact(store, "an ungraded one"), outcome=Outcome.PENDING)

    evidence = retrieval_evidence(store)

    assert evidence["pending"] == 1
    assert evidence["graded"] == 2


def test_recalls_older_than_the_window_are_excluded(store):
    """"Two weeks of hook-path recalls" — a year of pre-task-14 rows would
    answer a question about a mechanism that was not running yet."""
    from nenapu.retrieval_report import retrieval_evidence

    _recall(store, _fact(store, "recent"), outcome=Outcome.BAD, days_ago=3)
    _recall(store, _fact(store, "ancient"), outcome=Outcome.BAD, days_ago=400)

    assert retrieval_evidence(store, window_days=14)["bad"] == 1


def test_the_window_is_adjustable(store):
    from nenapu.retrieval_report import retrieval_evidence

    _recall(store, _fact(store, "a month back"), outcome=Outcome.BAD, days_ago=30)

    assert retrieval_evidence(store, window_days=14)["bad"] == 0
    assert retrieval_evidence(store, window_days=60)["bad"] == 1


def test_the_bad_rate_is_over_graded_recalls_only(store):
    from nenapu.retrieval_report import retrieval_evidence

    _spread(store, good=3, bad=1)
    _recall(store, _fact(store, "ungraded"), outcome=Outcome.PENDING)

    assert retrieval_evidence(store)["bad_rate"] == pytest.approx(0.25)


def test_an_empty_ledger_reports_zeroes_rather_than_dividing_by_them(store):
    from nenapu.retrieval_report import retrieval_evidence

    evidence = retrieval_evidence(store)

    assert evidence["graded"] == 0
    assert evidence["bad_rate"] == 0.0


def test_the_report_says_how_long_the_data_spans(store):
    """Thirty grades that all landed this afternoon are one session's
    opinion, not two weeks of measurement."""
    from nenapu.retrieval_report import retrieval_evidence

    _recall(store, _fact(store, "early"), outcome=Outcome.GOOD, days_ago=13)
    _recall(store, _fact(store, "late"), outcome=Outcome.GOOD, days_ago=1)

    assert retrieval_evidence(store)["days_of_data"] == pytest.approx(12, abs=0.1)


def test_the_report_separates_wrong_project_recalls(store):
    """The third branch of the rule, and the one that decides whether this
    work was already done: a fact from another project surfacing here is a
    scope failure, not a similarity failure."""
    from nenapu.retrieval_report import retrieval_evidence

    _session(store, external_id="s-here", scope="repo:here@aaaaaaaa")
    off_project = _fact(store, "the other repo runs on port 5544",
                        scope="repo:elsewhere@bbbbbbbb")
    on_project = _fact(store, "this repo runs on port 8080", scope="repo:here@aaaaaaaa")
    _recall(store, off_project, session_id="s-here", outcome=Outcome.BAD)
    _recall(store, on_project, session_id="s-here", outcome=Outcome.BAD)

    assert retrieval_evidence(store)["wrong_project"] == 1


def test_a_global_fact_recalled_in_a_project_is_not_wrong_project(store):
    """Global facts are supposed to surface everywhere — counting them as
    scope failures would manufacture the verdict that says the work is
    already done."""
    from nenapu.retrieval_report import retrieval_evidence

    _session(store, external_id="s-here", scope="repo:here@aaaaaaaa")
    _recall(store, _fact(store, "always use uv, never pip", scope="global"),
            session_id="s-here", outcome=Outcome.BAD)

    assert retrieval_evidence(store)["wrong_project"] == 0


def test_sessions_that_were_given_no_memory_at_all_are_counted(store):
    """The coverage branch. "Obviously missing facts" is not directly
    observable, but a session that ended with zero recalls logged while the
    store held facts for its scope is the measurable shadow of it."""
    from nenapu.retrieval_report import retrieval_evidence

    _fact(store, "there are facts in this scope", scope="repo:here@aaaaaaaa")
    _session(store, external_id="s-with", scope="repo:here@aaaaaaaa")
    _session(store, external_id="s-without", scope="repo:here@aaaaaaaa")
    _recall(store, _fact(store, "a recalled one", scope="repo:here@aaaaaaaa"),
            session_id="s-with", outcome=Outcome.GOOD)

    evidence = retrieval_evidence(store)

    assert evidence["sessions_with_recalls"] == 1
    assert evidence["sessions_without_recalls"] == 1
    assert evidence["coverage_rate"] == pytest.approx(0.5)


def test_the_report_never_calls_a_model(store):
    """It is a count over one table. A model asked "is retrieval failing"
    would answer, which is exactly the invented answer this task exists to
    avoid."""
    from nenapu.retrieval_report import retrieval_evidence

    _spread(store, good=2, bad=1)
    with patch("nenapu.llm.structured") as structured:
        retrieval_evidence(store)

    structured.assert_not_called()


# ---------- the rule, executed ----------


def test_a_young_store_refuses_to_decide(store):
    """The likely answer today, and the one that matters most: 3 recalls in
    the ledger is not a mandate to build a vector index."""
    from nenapu.retrieval_report import decide, retrieval_evidence

    _spread(store, bad=3)

    assert decide(retrieval_evidence(store)) == "insufficient-evidence"


def test_a_pile_of_grades_from_one_afternoon_is_not_two_weeks(store):
    from nenapu.retrieval_report import MIN_GRADED_RECALLS, decide, retrieval_evidence

    for i in range(MIN_GRADED_RECALLS + 5):
        _recall(store, _fact(store, f"same day fact {i}"), outcome=Outcome.BAD, days_ago=1)

    assert decide(retrieval_evidence(store)) == "insufficient-evidence"


def test_mostly_bad_recalls_say_build_vectors(store):
    from nenapu.retrieval_report import MIN_GRADED_RECALLS, decide, retrieval_evidence

    n = MIN_GRADED_RECALLS + 4
    _spread(store, bad=int(n * 0.6), good=n - int(n * 0.6))

    assert decide(retrieval_evidence(store)) == "build-vectors"


def test_bad_recalls_that_are_mostly_the_wrong_project_say_it_is_already_fixed(store):
    """"Vectors would have been wasted effort" — pinned as an outcome the
    rule can actually reach, because it is the branch a reader is most
    tempted to skip."""
    from nenapu.retrieval_report import MIN_GRADED_RECALLS, decide, retrieval_evidence

    _session(store, external_id="s-here", scope="repo:here@aaaaaaaa")
    n = MIN_GRADED_RECALLS + 4
    for i in range(n):
        elsewhere = _fact(store, f"other repo fact {i}", scope="repo:elsewhere@bbbbbbbb")
        _recall(store, elsewhere, session_id="s-here", outcome=Outcome.BAD,
                days_ago=0.5 + 13.0 * i / n)

    assert decide(retrieval_evidence(store)) == "already-fixed-by-scoping"


def test_few_bad_grades_but_sessions_getting_nothing_is_a_coverage_problem(store):
    from nenapu.retrieval_report import (
        COVERAGE_FLOOR, MIN_GRADED_RECALLS, decide, retrieval_evidence,
    )

    scope = "repo:here@aaaaaaaa"
    _spread(store, good=MIN_GRADED_RECALLS + 4, scope=scope, session="s-with")
    _session(store, external_id="s-with", scope=scope)
    for i in range(20):
        _session(store, external_id=f"s-empty-{i}", scope=scope, days_ago=1 + i * 0.5)

    evidence = retrieval_evidence(store)
    assert evidence["coverage_rate"] < COVERAGE_FLOOR
    assert decide(evidence) == "coverage-problem"


def test_a_healthy_ledger_says_retrieval_is_not_the_problem(store):
    """The verdict that keeps 17 and 18 unbuilt. Without it the rule has no
    way to say "do nothing", and every measurement ends in a project."""
    from nenapu.retrieval_report import MIN_GRADED_RECALLS, decide, retrieval_evidence

    n = MIN_GRADED_RECALLS + 4
    scope = "repo:here@aaaaaaaa"
    _spread(store, good=n - 1, bad=1, scope=scope, session="s-with")
    _session(store, external_id="s-with", scope=scope)

    assert decide(retrieval_evidence(store)) == "retrieval-is-not-the-problem"


def test_the_verdict_is_carried_on_the_evidence_itself(store):
    from nenapu.retrieval_report import decide, retrieval_evidence

    evidence = retrieval_evidence(store)

    assert evidence["verdict"] == decide(evidence)


def test_the_thresholds_are_the_ones_the_plan_named(store):
    """A rule whose thresholds can be tuned to taste after seeing the data is
    not evidence — it is the opinion it was meant to replace. Two weeks is
    the plan's own number."""
    from nenapu.retrieval_report import (
        BAD_RATE_THRESHOLD, COVERAGE_FLOOR, MIN_DAYS_OF_DATA, MIN_GRADED_RECALLS,
    )

    assert MIN_DAYS_OF_DATA == 14
    assert MIN_GRADED_RECALLS >= 30
    assert 0.0 < BAD_RATE_THRESHOLD < 1.0
    assert 0.0 < COVERAGE_FLOOR < 1.0


# ---------- the command ----------


def _run(args, db, **env):
    return subprocess.run(
        [sys.executable, "-m", "nenapu.cli", *args, "--db", str(db)],
        capture_output=True, text=True,
        env={**os.environ, "PYTHONPATH": "src", "NENAPU_NO_BANNER": "1", **env},
    )


def test_retrieval_is_a_registered_command(tmp_path):
    result = _run(["retrieval"], tmp_path / "s.db")

    assert result.returncode == 0, result.stdout + result.stderr


def test_the_command_prints_the_counts_it_decided_from(tmp_path):
    """Showing the working is the point: someone reading this has to be able
    to disagree with the verdict on the numbers rather than on faith."""
    db = tmp_path / "s.db"
    store = Store(connect(str(db)))
    _spread(store, good=3, bad=2)

    result = _run(["retrieval"], db)

    assert "3" in result.stdout and "2" in result.stdout


def test_the_command_says_when_there_is_not_enough_data(tmp_path):
    db = tmp_path / "s.db"
    store = Store(connect(str(db)))
    _spread(store, good=2, bad=1)

    result = _run(["retrieval"], db)

    assert "insufficient" in result.stdout.lower() or "not enough" in result.stdout.lower()


def test_the_command_accepts_a_window(tmp_path):
    db = tmp_path / "s.db"
    store = Store(connect(str(db)))
    _recall(store, _fact(store, "a month back"), outcome=Outcome.BAD, days_ago=30)

    result = _run(["retrieval", "--window-days", "60"], db)

    assert result.returncode == 0, result.stdout + result.stderr
    assert "1" in result.stdout


def test_the_command_reads_and_writes_nothing_back(tmp_path):
    """A report that grades, expires or dedupes anything on the way past
    would be changing the measurement it is reporting."""
    db = tmp_path / "s.db"
    store = Store(connect(str(db)))
    _spread(store, good=2, bad=2)
    before = [dict(r) for r in store.conn.execute("SELECT * FROM recalls ORDER BY id")]

    _run(["retrieval"], db)

    after = [dict(r) for r in connect(str(db)).execute("SELECT * FROM recalls ORDER BY id")]
    assert after == before
