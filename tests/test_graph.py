"""Falsification has to travel. These tests are the thesis."""

import pytest

from nenapu import connect
from nenapu.models import EdgeSource, Fact, Origin, Status
from nenapu.store import Store, effective_confidence
from nenapu.verify import apply_result, run_check


@pytest.fixture
def store():
    return Store(connect(":memory:"))


def test_failing_root_makes_dependents_suspect(store, approve_all):
    root, _ = store.write(Fact(text="auth lives in services/auth",
                               verify_cmd="test -d /definitely/not/here"))
    derived, _ = store.write(Fact(text="new endpoints go in services/auth/routes.py"),
                             derived_from=[root.id])
    assert store.get(derived.id).status == Status.ACTIVE

    approve_all(store)
    apply_result(store, run_check(store.get(root.id), conn=store.conn))

    after = store.get(derived.id)
    assert after.status == Status.SUSPECT
    assert f"#{root.id}" in after.suspect_reason


def test_suspicion_costs_confidence(store, approve_all):
    root, _ = store.write(Fact(text="root", verify_cmd="false"))
    derived, _ = store.write(Fact(text="a conclusion built on the root",
                                  origin=Origin.USER_STATED, confidence=0.9),
                             derived_from=[root.id])
    before = effective_confidence(store.get(derived.id))
    approve_all(store)
    apply_result(store, run_check(store.get(root.id), conn=store.conn))
    assert effective_confidence(store.get(derived.id)) < before / 2


def test_cascade_is_transitive(store, approve_all):
    a, _ = store.write(Fact(text="a", verify_cmd="false"))
    b, _ = store.write(Fact(text="b"), derived_from=[a.id])
    c, _ = store.write(Fact(text="c"), derived_from=[b.id])
    approve_all(store)
    apply_result(store, run_check(store.get(a.id), conn=store.conn))
    assert store.get(b.id).status == Status.SUSPECT
    assert store.get(c.id).status == Status.SUSPECT


def test_cascade_survives_a_cycle(store, approve_all):
    a, _ = store.write(Fact(text="a", verify_cmd="false"))
    b, _ = store.write(Fact(text="b"), derived_from=[a.id])
    store.graph.link(b.id, a.id)  # cycle
    approve_all(store)
    apply_result(store, run_check(store.get(a.id), conn=store.conn))
    assert store.get(b.id).status == Status.SUSPECT


def test_recovery_reinstates_dependents(store, approve_all):
    root, _ = store.write(Fact(text="root", verify_cmd="false"))
    derived, _ = store.write(Fact(text="derived"), derived_from=[root.id])
    approve_all(store)
    apply_result(store, run_check(store.get(root.id), conn=store.conn))
    assert store.get(derived.id).status == Status.SUSPECT

    store.conn.execute("UPDATE facts SET verify_cmd='true' WHERE id=?", (root.id,))
    store.conn.commit()
    approve_all(store)
    apply_result(store, run_check(store.get(root.id), conn=store.conn))
    assert store.get(derived.id).status == Status.ACTIVE


def test_recovery_leaves_dependents_of_another_broken_parent_alone(store, approve_all):
    good_root, _ = store.write(Fact(text="good root", verify_cmd="false"))
    bad_root, _ = store.write(Fact(text="bad root", verify_cmd="false"))
    derived, _ = store.write(Fact(text="rests on both"),
                             derived_from=[good_root.id, bad_root.id])
    approve_all(store)
    apply_result(store, run_check(store.get(good_root.id), conn=store.conn))
    apply_result(store, run_check(store.get(bad_root.id), conn=store.conn))

    store.conn.execute("UPDATE facts SET verify_cmd='true' WHERE id=?", (good_root.id,))
    store.conn.commit()
    approve_all(store)
    apply_result(store, run_check(store.get(good_root.id), conn=store.conn))
    assert store.get(derived.id).status == Status.SUSPECT  # bad_root still broken


def test_supersession_cascades(store, approve_all):
    root, _ = store.write(Fact(text="cache backend is redis", key="cache.backend",
                               origin=Origin.USER_STATED, confidence=0.8))
    derived, _ = store.write(Fact(text="use redis-cli to inspect the cache"),
                             derived_from=[root.id])
    store.write(Fact(text="cache backend is memcached", key="cache.backend",
                     origin=Origin.USER_STATED, confidence=0.95))
    assert store.get(derived.id).status == Status.SUSPECT


def test_retiring_a_fact_cascades(store, approve_all):
    root, _ = store.write(Fact(text="root"))
    derived, _ = store.write(Fact(text="derived"), derived_from=[root.id])
    store.forget(root.id)
    assert store.get(derived.id).status == Status.SUSPECT


def test_edges_are_inferred_from_recall_then_write(store, approve_all):
    # Nobody maintains a dependency graph by hand; it has to build itself.
    root, _ = store.write(Fact(text="the deploy script is scripts/deploy.sh"))
    store.search("deploy script", session_id="task-1")
    derived, _ = store.write(Fact(text="rollback is deploy.sh --rollback",
                                  session_id="task-1"))

    parents = store.graph.parents(derived.id)
    assert (root.id, EdgeSource.INFERRED, pytest.approx(0.6)) in [
        (p, s, w) for p, s, w in parents
    ]


def test_inferred_edges_do_not_cross_sessions(store, approve_all):
    store.write(Fact(text="unrelated fact about billing"))
    store.search("billing", session_id="task-a")
    derived, _ = store.write(Fact(text="conclusion in another task", session_id="task-b"))
    assert store.graph.parents(derived.id) == []


def test_self_links_and_duplicates_are_ignored(store, approve_all):
    a, _ = store.write(Fact(text="a"))
    b, _ = store.write(Fact(text="b"))
    assert store.graph.link(a.id, a.id) is None
    assert store.graph.link(a.id, b.id) is not None
    assert store.graph.link(a.id, b.id) is None


def test_why_explains_the_chain(store, approve_all):
    root, _ = store.write(Fact(text="postgres is the primary store"))
    derived, _ = store.write(Fact(text="migrations use alembic"), derived_from=[root.id])
    chain = store.graph.why(derived.id)
    assert chain["rests_on"][0]["id"] == root.id
    assert chain["rests_on"][0]["text"] == "postgres is the primary store"


def test_reasserting_a_fact_still_picks_up_this_session_dependencies(store, approve_all):
    conclusion, _ = store.write(Fact(text="new endpoints go in services/auth/routes.py"))
    root, _ = store.write(Fact(text="auth code lives in services/auth"))

    store.search("where does auth live", session_id="task-42")
    again, _ = store.write(Fact(text="new endpoints go in services/auth/routes.py",
                                session_id="task-42"))

    assert again.id == conclusion.id  # deduped, as before
    assert root.id in [p for p, _s, _w in store.graph.parents(conclusion.id)]
