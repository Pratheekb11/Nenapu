"""Recall relevance is not recall usefulness. These tests are the other half."""

import pytest

from nenapu import connect
from nenapu.models import Fact, Origin, Outcome
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
