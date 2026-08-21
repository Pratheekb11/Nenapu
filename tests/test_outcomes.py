"""Recall relevance is not recall usefulness. These tests are the other half."""

import pytest

from nenapu import connect
from nenapu.models import Fact, Origin, Outcome, Status
from nenapu.outcomes import outcome_signal
from nenapu.store import Store, effective_confidence
from nenapu.verify import apply_result, run_check


@pytest.fixture
def store():
    return Store(connect(":memory:"))


def test_recall_is_logged_and_gradeable(store, approve_all):
    store.write(Fact(text="the staging host is box-7"))
    hits = store.search("staging host", session_id="task-1")
    recall_id = hits[0][2]["recall_id"]
    assert store.ledger.get(recall_id).outcome == Outcome.PENDING
    assert store.ledger.grade(recall_id, Outcome.BAD, source="human")


def test_bad_recalls_cost_the_fact_confidence(store, approve_all):
    fact, _ = store.write(Fact(text="run migrations with make db-migrate",
                               origin=Origin.USER_STATED, confidence=0.9))
    before = effective_confidence(store.get(fact.id))
    for _ in range(4):
        rid = store.search("migrations", session_id="s")[0][2]["recall_id"]
        store.ledger.grade(rid, Outcome.BAD, source="human")
    assert effective_confidence(store.get(fact.id)) < before


def test_good_recalls_do_not_punish(store, approve_all):
    fact, _ = store.write(Fact(text="tests live under tests/", confidence=0.7))
    for _ in range(3):
        rid = store.search("tests", session_id="s")[0][2]["recall_id"]
        store.ledger.grade(rid, Outcome.GOOD, source="human")
    assert effective_confidence(store.get(fact.id)) >= 0.4


def test_ungraded_facts_are_neither_rewarded_nor_punished(store, approve_all):
    assert outcome_signal(0, 0) == 1.0
    assert outcome_signal(0, 5) < 1.0
    assert outcome_signal(5, 0) > 1.0


def test_first_grade_wins(store, approve_all):
    store.write(Fact(text="a fact"))
    rid = store.search("fact", session_id="s")[0][2]["recall_id"]
    assert store.ledger.grade(rid, Outcome.GOOD, source="human")
    assert not store.ledger.grade(rid, Outcome.BAD, source="verification")
    assert store.ledger.get(rid).outcome == Outcome.GOOD


def test_session_grading_covers_every_fact_the_task_used(store, approve_all):
    store.write(Fact(text="deploy with make ship"))
    store.write(Fact(text="deploy needs the VPN"))
    store.search("deploy", session_id="task-9", limit=5)
    assert store.ledger.grade_session("task-9", Outcome.BAD) == 2
    assert store.ledger.pending(session_id="task-9") == []


def test_verification_failure_retroactively_blames_recalls(store, approve_all):
    # Signal that needs no cooperation from the harness at all.
    fact, _ = store.write(Fact(text="the binary is on PATH",
                               verify_cmd="command -v not-a-real-binary"))
    rid = store.search("binary PATH", session_id="s")[0][2]["recall_id"]
    approve_all(store)
    apply_result(store, run_check(store.get(fact.id), conn=store.conn))
    assert store.ledger.get(rid).outcome == Outcome.BAD
    assert store.ledger.get(rid).outcome_source == "verification"


def test_verification_pass_credits_recalls(store, approve_all):
    fact, _ = store.write(Fact(text="echo works", verify_cmd="echo hi", verify_expect="hi"))
    rid = store.search("echo works", session_id="s")[0][2]["recall_id"]
    approve_all(store)
    apply_result(store, run_check(store.get(fact.id), conn=store.conn))
    assert store.ledger.get(rid).outcome == Outcome.GOOD


def test_correction_retroactively_blames_recalls(store, approve_all):
    old, _ = store.write(Fact(text="the retry limit is 3", key="retry.limit",
                              origin=Origin.USER_STATED, confidence=0.8))
    rid = store.search("retry limit", session_id="s")[0][2]["recall_id"]
    store.write(Fact(text="the retry limit is 10", key="retry.limit",
                     origin=Origin.USER_STATED, confidence=0.95))
    graded = store.ledger.get(rid)
    assert graded.outcome == Outcome.BAD
    assert graded.outcome_source == "correction"


def test_retiring_a_fact_blames_its_recent_recalls(store, approve_all):
    fact, _ = store.write(Fact(text="a mistaken belief"))
    rid = store.search("mistaken belief", session_id="s")[0][2]["recall_id"]
    store.forget(fact.id)
    assert store.ledger.get(rid).outcome == Outcome.BAD


def test_pending_recalls_expire_to_neutral(store, approve_all):
    import time
    store.write(Fact(text="something nobody grades"))
    rid = store.search("something", session_id="s")[0][2]["recall_id"]
    store.conn.execute("UPDATE recalls SET created_at = ? WHERE id = ?",
                       (time.time() - 30 * 86400, rid))
    store.conn.commit()
    assert store.ledger.expire_pending() == 1
    assert store.ledger.get(rid).outcome == Outcome.NEUTRAL


def test_search_can_opt_out_of_logging(store, approve_all):
    store.write(Fact(text="quiet lookup"))
    store.search("quiet", log_recall=False)
    assert store.ledger.pending() == []


def test_a_confirmed_fact_does_not_rot_to_the_floor(store, approve_all):
    """Found end to end: the audit confirmed 'the test runner is pytest',
    citing pytest.ini, and the store still believed it at 0.05 — because
    `holds` was inert and the fact had aged past its half-life. An audit that
    can only ever subtract confidence is a ratchet, not an audit."""
    import time

    from nenapu.models import Decay

    old = time.time() - 120 * 86400
    fact, _ = store.write(Fact(text="the test runner is pytest", kind="environment",
                               decay_class=Decay.VOLATILE, origin=Origin.USER_STATED,
                               confidence=0.85, created_at=old, last_verified_at=old))
    assert effective_confidence(store.get(fact.id)) < 0.1   # rotted

    store.soft_verify(fact.id)
    revived = effective_confidence(store.get(fact.id))
    assert revived > 0.5                                     # believable again


def test_soft_verification_is_weaker_than_a_passing_check(store, approve_all):
    """Model confirmation must not be worth as much as running the command."""
    hard, _ = store.write(Fact(text="a", origin=Origin.USER_STATED, confidence=0.9))
    soft, _ = store.write(Fact(text="b", origin=Origin.USER_STATED, confidence=0.9))

    store.touch_verified(hard.id)
    store.soft_verify(soft.id)
    assert effective_confidence(store.get(soft.id)) < effective_confidence(store.get(hard.id))


def test_repeated_soft_confirmation_still_fades(store, approve_all):
    """A fact kept alive only by a model agreeing with it should decline."""
    fact, _ = store.write(Fact(text="unchecked claim", origin=Origin.USER_STATED,
                               confidence=0.9))
    scores = []
    for _ in range(5):
        store.soft_verify(fact.id)
        scores.append(store.get(fact.id).confidence)
    assert scores == sorted(scores, reverse=True)   # monotonically down
    assert scores[-1] < scores[0]


# ==========================================================================
# Pre-written for G3 · session-scoped blame.
#
# Requirement: `blame_recent_recalls` reaches back `IMPLICIT_WINDOW_SECONDS`
# (6h). A SessionStart injection happens at minute zero of a session that may
# run far longer than that, so the correction signal misses its own session.
#
#     Ledger.blame_session_recalls(fact_id, session_id, *, source, note)
#
# grades that session's pending recalls of that fact `bad` with no window, and
# is called alongside the existing `blame_recent_recalls` from `_resolve`
# (supersede) and `set_status` (retire).
#
# Assumed seam: `set_status` grows an optional `session_id`, since a retire
# has no superseding fact to read a session off. `_resolve` reads it from the
# fact doing the superseding.
# ==========================================================================

g3 = pytest.mark.xfail(strict=True, reason="G3 not implemented yet: remove when it lands")

EIGHT_HOURS = 8 * 3600.0


def _aged_recall(store, fact_id, *, session_id, age_seconds):
    """A recall logged `age_seconds` ago — a session-start injection into a
    session that then ran for the rest of the working day."""
    import time

    recall_id = store.ledger.log(fact_id, session_id=session_id)
    store.conn.execute(
        "UPDATE recalls SET created_at = ? WHERE id = ?",
        (time.time() - age_seconds, recall_id),
    )
    store.conn.commit()
    return recall_id


def test_the_six_hour_window_really_does_miss_its_own_session(store):
    """The fault, pinned before the fix so the fix has something to be
    measured against. Not marked pending: this is today's behaviour."""
    from nenapu.models import Outcome

    fact, _ = store.write(Fact(text="the staging host is box-7", key="staging.host"))
    recall_id = _aged_recall(store, fact.id, session_id="s-long", age_seconds=EIGHT_HOURS)

    graded = store.ledger.blame_recent_recalls(fact.id, source="correction", note="x")

    assert graded == 0
    assert store.ledger.get(recall_id).outcome == Outcome.PENDING


def test_a_session_start_injection_is_blamed_eight_hours_later(store):
    """The whole point: the fact was injected at minute zero, acted on all
    day, and corrected at hour eight. That correction is about that recall."""
    from nenapu.models import Outcome

    fact, _ = store.write(Fact(text="the staging host is box-7", key="staging.host"))
    recall_id = _aged_recall(store, fact.id, session_id="s-long", age_seconds=EIGHT_HOURS)

    graded = store.ledger.blame_session_recalls(
        fact.id, "s-long", source="correction", note="superseded"
    )

    assert graded == 1
    assert store.ledger.get(recall_id).outcome == Outcome.BAD


def test_blame_is_scoped_to_the_session_that_did_the_correcting(store):
    """No window means no other guard, so the session id is the guard. A
    recall in somebody else's session is not evidence about this correction."""
    from nenapu.models import Outcome

    fact, _ = store.write(Fact(text="the staging host is box-7", key="staging.host"))
    mine = _aged_recall(store, fact.id, session_id="s-mine", age_seconds=EIGHT_HOURS)
    theirs = _aged_recall(store, fact.id, session_id="s-theirs", age_seconds=EIGHT_HOURS)

    store.ledger.blame_session_recalls(fact.id, "s-mine", source="correction", note="x")

    assert store.ledger.get(mine).outcome == Outcome.BAD
    assert store.ledger.get(theirs).outcome == Outcome.PENDING


def test_only_recalls_of_that_fact_are_blamed(store):
    """A session recalls a dozen facts. One of them turned out wrong; the
    other eleven are not implicated by it."""
    from nenapu.models import Outcome

    wrong, _ = store.write(Fact(text="the staging host is box-7", key="staging.host"))
    other, _ = store.write(Fact(text="deploys run from main", key="deploy.branch"))
    wrong_recall = _aged_recall(store, wrong.id, session_id="s", age_seconds=EIGHT_HOURS)
    other_recall = _aged_recall(store, other.id, session_id="s", age_seconds=EIGHT_HOURS)

    store.ledger.blame_session_recalls(wrong.id, "s", source="correction", note="x")

    assert store.ledger.get(wrong_recall).outcome == Outcome.BAD
    assert store.ledger.get(other_recall).outcome == Outcome.PENDING


def test_the_first_grade_still_wins(store):
    """`Ledger.grade`'s single-statement pending check is reused rather than
    re-implemented, so an explicit human verdict is not overwritten by the
    implicit signal that arrives later."""
    from nenapu.models import Outcome

    fact, _ = store.write(Fact(text="the staging host is box-7", key="staging.host"))
    recall_id = _aged_recall(store, fact.id, session_id="s", age_seconds=EIGHT_HOURS)
    store.ledger.grade(recall_id, Outcome.GOOD, source="human")

    graded = store.ledger.blame_session_recalls(fact.id, "s", source="correction", note="x")

    assert graded == 0
    assert store.ledger.get(recall_id).outcome == Outcome.GOOD
    assert store.ledger.get(recall_id).outcome_source == "human"


def test_a_missing_session_id_grades_nothing(store):
    """Every call site has a session id that can be absent — a fact written
    outside a session, a retire from the CLI. Absent must mean "no implicit
    signal", never "every pending recall of this fact"."""
    from nenapu.models import Outcome

    fact, _ = store.write(Fact(text="the staging host is box-7", key="staging.host"))
    recall_id = _aged_recall(store, fact.id, session_id="s", age_seconds=EIGHT_HOURS)

    assert store.ledger.blame_session_recalls(fact.id, None, source="correction",
                                              note="x") == 0
    assert store.ledger.get(recall_id).outcome == Outcome.PENDING


def test_superseding_a_fact_blames_its_own_session(store):
    """Call site one: `Store._resolve`. The fact recalled at session start and
    contradicted eight hours later is graded from the same write that
    supersedes it, with no hook and no harness cooperation."""
    from nenapu.models import Outcome

    old, _ = store.write(Fact(text="the staging host is box-7", key="staging.host",
                              origin=Origin.AGENT_INFERRED, confidence=0.6))
    recall_id = _aged_recall(store, old.id, session_id="s-long", age_seconds=EIGHT_HOURS)

    store.write(Fact(text="the staging host is box-9", key="staging.host",
                     origin=Origin.USER_STATED, confidence=0.95, session_id="s-long"))

    assert store.get(old.id).status == Status.SUPERSEDED
    assert store.ledger.get(recall_id).outcome == Outcome.BAD


def test_retiring_a_fact_blames_its_own_session(store):
    """Call site two: `Store.set_status` on a retire. A human saying "forget
    that" is the same evidence about the same recall, arriving by hand."""
    from nenapu.models import Outcome

    fact, _ = store.write(Fact(text="the staging host is box-7", key="staging.host"))
    recall_id = _aged_recall(store, fact.id, session_id="s-long", age_seconds=EIGHT_HOURS)

    store.set_status(fact.id, Status.RETIRED, session_id="s-long")

    assert store.ledger.get(recall_id).outcome == Outcome.BAD


def test_the_recent_window_signal_is_not_removed(store):
    """Added alongside, not instead of: a correction with no session id
    attached still reaches the recalls the 6h window covers."""
    from nenapu.models import Outcome

    old, _ = store.write(Fact(text="the port is 8080", key="app.port",
                              origin=Origin.AGENT_INFERRED, confidence=0.6))
    recent = store.ledger.log(old.id, session_id="s-other")

    store.write(Fact(text="the port is 9090", key="app.port",
                     origin=Origin.USER_STATED, confidence=0.95))

    assert store.ledger.get(recent).outcome == Outcome.BAD
