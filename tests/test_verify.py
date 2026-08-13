import pytest

from nenapu import connect
from nenapu.models import Fact, VerifyStatus
from nenapu.store import Store, effective_confidence
from nenapu.verify import apply_result, run_check, verify_scope


@pytest.fixture
def store():
    return Store(connect(":memory:"))


def test_passing_check_marks_verified(store, approve_all):
    fact, _ = store.write(Fact(text="python is on PATH", verify_cmd="python3 --version",
                               verify_expect="Python 3"))
    approve_all(store)
    result = run_check(store.get(fact.id), conn=store.conn)
    assert result.ok
    apply_result(store, result)
    assert store.get(fact.id).verify_status == VerifyStatus.PASS


def test_failing_check_collapses_confidence(store, approve_all):
    fact, _ = store.write(Fact(text="the frobnicator is installed",
                               verify_cmd="command -v definitely-not-a-real-binary"))
    before = effective_confidence(store.get(fact.id))
    approve_all(store)
    apply_result(store, run_check(store.get(fact.id), conn=store.conn))
    after = effective_confidence(store.get(fact.id))
    assert store.get(fact.id).verify_status == VerifyStatus.FAIL
    assert after < before / 2


def test_expect_mismatch_is_failure(store, approve_all):
    fact, _ = store.write(Fact(text="prints hello", verify_cmd="echo goodbye",
                               verify_expect="hello"))
    approve_all(store)
    assert run_check(store.get(fact.id), conn=store.conn).status == VerifyStatus.FAIL


def test_passing_check_resets_the_decay_clock(store, approve_all):
    import time
    old = time.time() - 200 * 86400
    fact, _ = store.write(Fact(text="echo works", verify_cmd="echo ok", decay_class="volatile",
                               created_at=old, last_verified_at=old))
    stale = effective_confidence(store.get(fact.id))
    approve_all(store)
    apply_result(store, run_check(store.get(fact.id), conn=store.conn))
    assert effective_confidence(store.get(fact.id)) > stale * 5


def test_verify_scope_skips_facts_without_checks(store, approve_all):
    store.write(Fact(text="no check here"))
    store.write(Fact(text="has a check", verify_cmd="true"))
    approve_all(store)
    assert len(verify_scope(store)) == 1


def test_broken_check_is_error_not_failure(store, approve_all):
    fact, _ = store.write(Fact(text="times out", verify_cmd="sleep 5"))
    approve_all(store)
    assert run_check(store.get(fact.id), conn=store.conn, timeout=1).status == VerifyStatus.ERROR
