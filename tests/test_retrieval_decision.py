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


# ==========================================================================
# Pre-written for the grading plan: G1, G2, G7, G9.
#
# Written before the implementation, per the plan's TDD rule. Every test
# below that describes behaviour the code does not have yet carries a strict
# xfail, so the suite stays honest today and turns red the moment a marker
# outlives its implementation — remove the marker when the task lands.
#
# The helpers above are reused unchanged. `_recall` already takes `source`,
# which is what G1 turns into a load-bearing distinction.
# ==========================================================================

g1 = pytest.mark.xfail(strict=True, reason="G1 not implemented yet: remove when it lands")
g2 = pytest.mark.xfail(strict=True, reason="G2 not implemented yet: remove when it lands")
g7 = pytest.mark.xfail(strict=True, reason="G7 not implemented yet: remove when it lands")
g9 = pytest.mark.xfail(strict=True, reason="G9 not recorded yet: remove when it lands")


def _expired(store, fact, **kw):
    """A recall closed by `expire_pending`, spelled the way maintenance does."""
    return _recall(store, fact, outcome=Outcome.NEUTRAL, source="expiry", **kw)


def _query_recall(store, fact, *, query, session_id="s-1", outcome=Outcome.PENDING,
                  days_ago=1.0, source="test"):
    """A recall that came from a search rather than from a bulk injection.

    G7 keys the two populations on `query = ''`, so a recall with a query on
    it is the ranking population and one without is the selection population.
    """
    recall_id = _recall(store, fact, session_id=session_id, outcome=outcome,
                        days_ago=days_ago, source=source)
    store.conn.execute("UPDATE recalls SET query = ? WHERE id = ?", (query, recall_id))
    store.conn.commit()
    return recall_id


def _spread_injections(store, *, n, outcome=Outcome.NEUTRAL, source="observer-unused",
                       session="s-inject"):
    for i in range(n):
        fact = _fact(store, f"injected fact number {i} nobody used")
        _recall(store, fact, session_id=session, outcome=outcome, source=source,
                days_ago=0.5 + 13.0 * i / max(n - 1, 1))


# ---------- G1 · expiry is not evidence ----------


def test_a_store_of_only_expiry_neutrals_has_graded_nothing(store):
    """`expire_pending` runs every maintenance tick and closes pending recalls
    as `neutral`. That is the absence of evidence, not evidence: nobody looked
    at those recalls, a timer did."""
    from nenapu.retrieval_report import retrieval_evidence

    for i in range(40):
        _expired(store, _fact(store, f"nobody ever graded fact {i}"),
                 days_ago=0.5 + 13.0 * i / 39)

    evidence = retrieval_evidence(store)

    assert evidence["graded"] == 0


def test_expired_recalls_are_reported_under_their_own_key(store):
    """Excluded from the evidence, still visible: a store whose whole ledger
    timed out should be able to say so on the numbers."""
    from nenapu.retrieval_report import retrieval_evidence

    _expired(store, _fact(store, "one"))
    _expired(store, _fact(store, "two"))
    _expired(store, _fact(store, "three"))

    assert retrieval_evidence(store)["expired"] == 3


def test_a_neutral_from_a_grader_still_counts_as_evidence(store):
    """A grader that read the transcript and found the fact unused *did*
    produce evidence. Only the timeout is excluded."""
    from nenapu.retrieval_report import retrieval_evidence

    _recall(store, _fact(store, "looked at and found unused"), outcome=Outcome.NEUTRAL,
            source="observer-unused")
    _expired(store, _fact(store, "closed by the clock"))

    evidence = retrieval_evidence(store)

    assert evidence["graded"] == 1
    assert evidence["expired"] == 1
    assert evidence["neutral"] == 1


def test_expiry_neutrals_are_out_of_the_bad_rate_denominator(store):
    """Three bad out of four looked-at recalls is a 75% failure rate. Diluting
    it with a hundred timeouts would report 3%."""
    from nenapu.retrieval_report import retrieval_evidence

    _spread(store, bad=3, good=1)
    for i in range(100):
        _expired(store, _fact(store, f"timed out {i}"), days_ago=0.5 + 13.0 * i / 99)

    assert retrieval_evidence(store)["bad_rate"] == pytest.approx(0.75)


def test_a_ledger_closed_by_the_clock_cannot_clear_the_gate(store):
    """The latent fault this task exists for. In seven days `expire_pending`
    would have neutralised 480 pending recalls, `graded` would read 480,
    `bad_rate` 0, and the gate would clear two large pieces of work on the
    strength of a timeout."""
    from nenapu.retrieval_report import decide, retrieval_evidence

    for i in range(480):
        _expired(store, _fact(store, f"bulk injected fact {i}"),
                 days_ago=0.5 + 13.0 * i / 479)

    assert decide(retrieval_evidence(store)) == "insufficient-evidence"


# ---------- G2 · coverage only over the hook era ----------


def test_sessions_that_predate_the_first_recall_are_not_counted(store):
    """193 of the store's 213 sessions were backfilled from transcripts that
    ran before the recall hook existed. They were not "given nothing" — there
    was no mechanism to give them anything, and counting them drags coverage
    toward a coverage-problem verdict that describes history rather than the
    system."""
    from nenapu.retrieval_report import retrieval_evidence

    scope = "repo:here@aaaaaaaa"
    _fact(store, "there are facts in this scope", scope=scope)
    _session(store, external_id="s-modern", scope=scope, days_ago=1)
    _recall(store, _fact(store, "a recalled one", scope=scope), session_id="s-modern",
            outcome=Outcome.GOOD, days_ago=1)
    for i in range(10):
        _session(store, external_id=f"s-ancient-{i}", scope=scope, days_ago=5 + i)

    evidence = retrieval_evidence(store)

    assert evidence["sessions_without_recalls"] == 0
    assert evidence["coverage_rate"] == pytest.approx(1.0)


def test_the_report_says_when_coverage_starts(store):
    """The number is only readable next to the moment it starts from."""
    from nenapu.retrieval_report import retrieval_evidence

    _recall(store, _fact(store, "the first recall ever logged"), outcome=Outcome.GOOD,
            days_ago=3)

    first = store.conn.execute("SELECT MIN(created_at) AS m FROM recalls").fetchone()["m"]
    assert retrieval_evidence(store)["coverage_since"] == pytest.approx(first)


def test_an_empty_recall_ledger_is_not_a_coverage_failure(store):
    """No recalls at all means there is no hook era to measure inside. The
    existing answer — 1.0, because no sessions to judge is not a failure —
    is the one to keep."""
    from nenapu.retrieval_report import retrieval_evidence

    scope = "repo:here@aaaaaaaa"
    _fact(store, "a fact in scope", scope=scope)
    _session(store, external_id="s-1", scope=scope)

    evidence = retrieval_evidence(store)

    assert evidence["coverage_rate"] == 1.0
    assert evidence["coverage_since"] is None


def test_a_session_inside_the_hook_era_that_got_nothing_still_counts(store):
    """The restriction must not silence the signal it is narrowing: a session
    that ran after recalls started and was handed none is exactly what the
    coverage branch is for."""
    from nenapu.retrieval_report import retrieval_evidence

    scope = "repo:here@aaaaaaaa"
    _fact(store, "a fact in scope", scope=scope)
    _session(store, external_id="s-with", scope=scope, days_ago=5)
    _recall(store, _fact(store, "recalled", scope=scope), session_id="s-with",
            outcome=Outcome.GOOD, days_ago=5)
    _session(store, external_id="s-without", scope=scope, days_ago=2)

    evidence = retrieval_evidence(store)

    assert evidence["sessions_with_recalls"] == 1
    assert evidence["sessions_without_recalls"] == 1
    assert evidence["coverage_rate"] == pytest.approx(0.5)


# ---------- G7 · injection and query are two populations ----------


def test_the_two_populations_are_reported_with_their_own_counts(store):
    """480 of 483 recalls in the live store are SessionStart bulk injections
    with `query = ''`. They measure *selection*; the three that came from a
    search measure *ranking*. One pooled rate is a number about neither."""
    from nenapu.retrieval_report import retrieval_evidence

    _spread_injections(store, n=20)
    _query_recall(store, _fact(store, "a searched fact"), query="deploy",
                  outcome=Outcome.GOOD)
    _query_recall(store, _fact(store, "another searched fact"), query="deploy",
                  outcome=Outcome.BAD)

    evidence = retrieval_evidence(store)

    assert evidence["injection"]["graded"] == 20
    assert evidence["injection"]["neutral"] == 20
    assert evidence["query"]["graded"] == 2
    assert evidence["query"]["good"] == 1
    assert evidence["query"]["bad"] == 1


def test_each_population_carries_its_own_rate(store):
    """A high unused-rate on injections is a selection failure, which R2, R3,
    R4 and E7 address. A high bad-rate on query hits is a ranking failure,
    which R1 addresses first."""
    from nenapu.retrieval_report import retrieval_evidence

    _spread_injections(store, n=10)
    for i in range(4):
        _query_recall(store, _fact(store, f"searched {i}"), query="cache",
                      outcome=Outcome.BAD if i < 3 else Outcome.GOOD,
                      days_ago=1 + i)

    evidence = retrieval_evidence(store)

    assert evidence["injection"]["unused_rate"] == pytest.approx(1.0)
    assert evidence["query"]["bad_rate"] == pytest.approx(0.75)


def test_the_populations_add_up_to_the_pooled_counts(store):
    """Two views of one ledger, not two ledgers. If they stop summing, one of
    them is quietly dropping recalls."""
    from nenapu.retrieval_report import retrieval_evidence

    _spread_injections(store, n=7)
    _query_recall(store, _fact(store, "searched one"), query="port", outcome=Outcome.GOOD)
    _query_recall(store, _fact(store, "searched two"), query="port", outcome=Outcome.BAD)

    evidence = retrieval_evidence(store)

    assert evidence["injection"]["graded"] + evidence["query"]["graded"] == evidence["graded"]


def test_a_pile_of_unused_injections_cannot_clear_the_gate(store):
    """The ordering constraint the plan states: with G5 in, ~460 neutrals from
    bulk injection would pool with 3 query recalls and the verdict would be
    computed off a population that is 95% "nobody used this"."""
    from nenapu.retrieval_report import decide, retrieval_evidence

    _spread_injections(store, n=460)
    for i in range(3):
        _query_recall(store, _fact(store, f"searched and wrong {i}"), query="auth",
                      outcome=Outcome.BAD, days_ago=1 + i * 4)

    assert decide(retrieval_evidence(store)) != "retrieval-is-not-the-problem"


def test_a_healthy_query_population_is_not_drowned_by_unused_injections(store):
    """The mirror of the test above: injections that nobody used must not
    manufacture a `build-vectors` verdict either. The unused-rate is a
    selection number and belongs to R2/R3/R4, not to the ranking branch."""
    from nenapu.retrieval_report import MIN_GRADED_RECALLS, decide, retrieval_evidence

    _spread_injections(store, n=200)
    scope = "repo:here@aaaaaaaa"
    _session(store, external_id="s-with", scope=scope)
    for i in range(MIN_GRADED_RECALLS + 4):
        fact = _fact(store, f"searched and useful {i}", scope=scope)
        _query_recall(store, fact, query="deploy", session_id="s-with",
                      outcome=Outcome.GOOD, days_ago=0.5 + 13.0 * i / 33)

    assert decide(retrieval_evidence(store)) != "build-vectors"


def test_the_command_prints_both_populations(tmp_path):
    """Showing the working, in the shape the verdict was computed in: a reader
    has to be able to disagree with the split, not only with the total."""
    db = tmp_path / "s.db"
    store = Store(connect(str(db)))
    _spread_injections(store, n=6)
    _query_recall(store, _fact(store, "a searched fact"), query="deploy",
                  outcome=Outcome.GOOD)

    result = _run(["retrieval"], db)

    assert result.returncode == 0, result.stdout + result.stderr
    assert "injection" in result.stdout.lower()
    assert "query" in result.stdout.lower()


def test_the_thresholds_survive_the_split(store):
    """G7 changes how the populations are counted, not what they are judged
    against. The constants are the guard that the rule did not get tuned to
    taste after seeing the data."""
    from nenapu.retrieval_report import (
        BAD_RATE_THRESHOLD, COVERAGE_FLOOR, MIN_DAYS_OF_DATA, MIN_GRADED_RECALLS,
        MIN_SPAN_FRACTION,
    )

    assert (MIN_DAYS_OF_DATA, MIN_GRADED_RECALLS, MIN_SPAN_FRACTION) == (14, 30, 0.5)
    assert (BAD_RATE_THRESHOLD, COVERAGE_FLOOR) == (0.3, 0.5)


def test_the_five_verdicts_are_still_the_five(store):
    """R2, R3, R4 and E7 are all gated on one of these strings. Adding a sixth
    would quietly change what the gate can say."""
    from nenapu.retrieval_report import VERDICT_MEANING

    assert set(VERDICT_MEANING) == {
        "insufficient-evidence", "build-vectors", "coverage-problem",
        "already-fixed-by-scoping", "retrieval-is-not-the-problem",
    }


# ---------- G9 · the baseline is written down ----------


def _notes() -> str:
    from pathlib import Path

    return (Path(__file__).resolve().parent.parent / "IMPLEMENTATION_NOTES.md").read_text()


def _baseline_block() -> str:
    """The recorded gate baseline, as its own block.

    Pinned to a heading rather than searched for across the whole file: the
    words "injection", "query" and "insufficient-evidence" all appear in the
    notes already, describing the problem. What G9 adds is the measurement,
    and a test that cannot tell those apart would pass before the work.
    """
    notes = _notes()
    lower = notes.lower()
    if "gate baseline" not in lower:
        return ""
    return notes[lower.index("gate baseline"):]


@g9
def test_the_notes_no_longer_claim_the_gate_was_never_answered():
    """The sentence being replaced, quoted so its removal is deliberate:
    "today it answers `insufficient-evidence` on 0 graded recalls out of 327
    logged"."""
    assert "0 graded recalls out of 327" not in _notes()


@g9
def test_the_notes_record_a_verdict_from_the_gate_itself():
    """Tasks 17 and 18 proceed or do not on that line, so it has to be one of
    the strings the gate can actually return rather than a summary of one."""
    from nenapu.retrieval_report import VERDICT_MEANING

    block = _baseline_block()

    assert block, "no `gate baseline` block in IMPLEMENTATION_NOTES.md"
    assert any(verdict in block for verdict in VERDICT_MEANING)


@g9
def test_the_baseline_records_both_populations_not_a_pooled_number():
    """"Record both populations, not a pooled number" — the baseline R2, R3,
    R4 and E7 are measured against is a pair of rates, and a single pooled
    figure cannot be compared against either of them afterwards."""
    block = _baseline_block().lower()

    assert "injection" in block and "query" in block
    assert "unused" in block


@g9
def test_the_baseline_records_the_counts_it_came_from():
    """A verdict with no numbers beside it is the opinion the gate exists to
    replace. If the verdict is still `insufficient-evidence`, the reason has
    to be stated in numbers: which threshold was not met."""
    import re

    block = _baseline_block()

    assert len(re.findall(r"\d+", block)) >= 4
