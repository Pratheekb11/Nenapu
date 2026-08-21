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


# ==========================================================================
# Pre-written for G10, E6 (belief half) and E10.
#
# G10 · infer edges from *used* recalls. `infer_edges_for` links a newly
# written fact to whatever the session recalled in the last hour, capped at
# MAX_INFERRED_PARENTS. On the live store "whatever the session recalled" is
# the 12-17 fact context dump nobody chose, so all 66 edges there are
# co-occurrence with a dump rather than causality. Prefer parents whose recall
# in that session was graded `good`; fall back to today's behaviour when a
# session has no grades, so the feature does not disappear on ungraded stores.
#
# E6 · entity death cascades into belief. `entity.status = 'gone'` ->
# `fact_entities WHERE role='subject'` -> Status.SUSPECT -> the *existing*
# cascade takes the descendants. The entity-side and capture-side halves are
# pinned in tests/test_entities.py; what is pinned here is that the belief
# layer is reused rather than duplicated, and that recovery works both ways.
#
# E10 · `nenapu why` shows both layers: the belief ancestry it prints today
# plus the fact's subject entity and its neighbourhood, so a human sees why a
# fact *surfaced*, not only why it is *believed*.
#
# Strict xfail on everything not yet implemented; remove the marker as each
# task lands.
# ==========================================================================

g10 = pytest.mark.xfail(strict=True, reason="G10 not implemented yet: remove when it lands")
e6 = pytest.mark.xfail(strict=True, reason="E6 not implemented yet: remove when it lands")
e10 = pytest.mark.xfail(strict=True, reason="E10 not implemented yet: remove when it lands")


def _graded_recall(store, fact_id, outcome, *, session_id):
    from nenapu.models import Outcome

    recall_id = store.ledger.log(fact_id, session_id=session_id)
    store.ledger.grade(recall_id, getattr(Outcome, outcome.upper()), source="observer")
    return recall_id


def _parent_ids(store, fact_id):
    return {p for p, _s, _w in store.graph.parents(fact_id)}


# ---------- G10 · edges from what was actually used ----------


def test_only_good_graded_recalls_become_parents(store, approve_all):
    """An agent that recalled A and B and then concluded C used A and B to get
    there — but only if it used them at all. On a store where every session
    starts with a twelve-fact dump, co-occurrence alone links the conclusion
    to whatever the dump happened to contain."""
    used, _ = store.write(Fact(text="the deploy script is scripts/deploy.sh"))
    ignored, _ = store.write(Fact(text="an unrelated fact about billing"))
    _graded_recall(store, used.id, "good", session_id="task-1")
    _graded_recall(store, ignored.id, "neutral", session_id="task-1")

    derived, _ = store.write(Fact(text="rollback is deploy.sh --rollback",
                                  session_id="task-1"))

    assert _parent_ids(store, derived.id) == {used.id}


def test_a_misleading_recall_is_never_made_a_parent(store, approve_all):
    """A fact the session was misled by is the opposite of what the conclusion
    rests on."""
    bad, _ = store.write(Fact(text="the deploy script is scripts/old.sh"))
    _graded_recall(store, bad.id, "bad", session_id="task-1")

    derived, _ = store.write(Fact(text="rollback is deploy.sh --rollback",
                                  session_id="task-1"))

    assert _parent_ids(store, derived.id) == set()


def test_the_parent_cap_still_holds_when_grades_exist(store, approve_all):
    """MAX_INFERRED_PARENTS is a statement about how diffuse a co-occurrence
    signal may get before it means nothing, and grading does not repeal it."""
    from nenapu.graph import MAX_INFERRED_PARENTS

    for i in range(MAX_INFERRED_PARENTS + 3):
        fact, _ = store.write(Fact(text=f"a fact the session actually used, number {i}"))
        _graded_recall(store, fact.id, "good", session_id="task-1")

    derived, _ = store.write(Fact(text="the conclusion", session_id="task-1"))

    assert len(store.graph.parents(derived.id)) == MAX_INFERRED_PARENTS


def test_a_session_with_no_grades_keeps_todays_behaviour(store, approve_all):
    """The fallback is what keeps this from deleting the feature on every
    store that has never graded anything — which is every store today."""
    root, _ = store.write(Fact(text="the deploy script is scripts/deploy.sh"))
    store.search("deploy script", session_id="task-1")

    derived, _ = store.write(Fact(text="rollback is deploy.sh --rollback",
                                  session_id="task-1"))

    assert root.id in _parent_ids(store, derived.id)


def test_one_good_grade_switches_the_session_off_co_occurrence(store, approve_all):
    """The fallback is per session, not per recall: once a session has told us
    which of its recalls were used, the ungraded rest are not evidence."""
    used, _ = store.write(Fact(text="the deploy script is scripts/deploy.sh"))
    ungraded, _ = store.write(Fact(text="an unrelated fact about billing"))
    _graded_recall(store, used.id, "good", session_id="task-1")
    store.ledger.log(ungraded.id, session_id="task-1")

    derived, _ = store.write(Fact(text="rollback is deploy.sh --rollback",
                                  session_id="task-1"))

    assert _parent_ids(store, derived.id) == {used.id}


def test_grades_from_another_session_do_not_leak_in(store, approve_all):
    """Session scoping is the one guard `infer_edges_for` already had, and
    reading grades must not step around it."""
    elsewhere, _ = store.write(Fact(text="a fact used well in another task"))
    _graded_recall(store, elsewhere.id, "good", session_id="task-a")

    derived, _ = store.write(Fact(text="a conclusion in another task",
                                  session_id="task-b"))

    assert store.graph.parents(derived.id) == []


# ---------- E6 · an entity stops existing ----------


def _entity_backed_fact(store, *, text, entity_name, role, scope="global"):
    """A fact joined to an entity through `fact_entities`.

    `role='subject'` is load-bearing: a fact *about* a deleted file dies with
    it, a fact that merely *mentions* it does not.
    """
    from nenapu.entities import EntityGraph

    fact, _ = store.write(Fact(text=text, scope=scope))
    graph = EntityGraph(store.conn)
    entity = graph.upsert(kind="file", name=entity_name, scope=scope)
    graph.attach(fact.id, entity.id, role=role, source="key")
    return fact, entity


@e6
def test_killing_an_entity_makes_its_subject_facts_suspect(store, approve_all):
    from nenapu.entities import EntityGraph

    fact, entity = _entity_backed_fact(
        store, text="services/auth/routes.py owns the login handler",
        entity_name="services/auth/routes.py", role="subject",
    )

    EntityGraph(store.conn).mark_gone(entity.id, reason="deleted in commit abc123")

    after = store.get(fact.id)
    assert after.status == Status.SUSPECT
    assert "auth" in (after.suspect_reason or "") or "gone" in (after.suspect_reason or "")


@e6
def test_a_fact_that_only_mentions_the_entity_survives(store, approve_all):
    """The distinction the bridge table exists for. Deleting a file does not
    falsify every sentence that ever named it."""
    from nenapu.entities import EntityGraph

    fact, entity = _entity_backed_fact(
        store, text="the login handler used to live in services/auth/routes.py",
        entity_name="services/auth/routes.py", role="mentions",
    )

    EntityGraph(store.conn).mark_gone(entity.id, reason="deleted")

    assert store.get(fact.id).status == Status.ACTIVE


@e6
def test_descendants_go_suspect_through_the_existing_cascade(store, approve_all):
    """Reuse, not a second walk: `cascade_falsification` is already
    depth-capped and cycle-safe, and a parallel implementation would be one
    more place for the two to disagree."""
    from nenapu.entities import EntityGraph

    subject, entity = _entity_backed_fact(
        store, text="services/auth/routes.py owns the login handler",
        entity_name="services/auth/routes.py", role="subject",
    )
    derived, _ = store.write(Fact(text="new endpoints go beside the login handler"),
                             derived_from=[subject.id])

    EntityGraph(store.conn).mark_gone(entity.id, reason="deleted")

    assert store.get(derived.id).status == Status.SUSPECT
    assert f"#{subject.id}" in store.get(derived.id).suspect_reason


@e6
def test_the_entity_coming_back_reinstates_what_it_falsified(store, approve_all):
    """Recovery is tested both ways, the way a failed check already is: a file
    restored in the next commit must not leave a permanent scar."""
    from nenapu.entities import EntityGraph

    fact, entity = _entity_backed_fact(
        store, text="services/auth/routes.py owns the login handler",
        entity_name="services/auth/routes.py", role="subject",
    )
    graph = EntityGraph(store.conn)
    graph.mark_gone(entity.id, reason="deleted")

    graph.mark_alive(entity.id)

    assert store.get(fact.id).status == Status.ACTIVE


@e6
def test_a_fact_with_another_broken_parent_stays_suspect(store, approve_all):
    """`clear_suspicion` already refuses to reinstate anything still propped
    up by a different broken parent, and the entity trigger inherits that."""
    from nenapu.entities import EntityGraph

    subject, entity = _entity_backed_fact(
        store, text="services/auth/routes.py owns the login handler",
        entity_name="services/auth/routes.py", role="subject",
    )
    other_root, _ = store.write(Fact(text="a second root", verify_cmd="false"))
    derived, _ = store.write(Fact(text="a conclusion resting on both"),
                             derived_from=[subject.id, other_root.id])
    graph = EntityGraph(store.conn)
    graph.mark_gone(entity.id, reason="deleted")
    approve_all(store)
    apply_result(store, run_check(store.get(other_root.id), conn=store.conn))

    graph.mark_alive(entity.id)

    assert store.get(derived.id).status == Status.SUSPECT


@e6
def test_an_entity_with_no_facts_cascades_into_nothing(store, approve_all):
    from nenapu.entities import EntityGraph

    graph = EntityGraph(store.conn)
    entity = graph.upsert(kind="file", name="tools/unreferenced.py", scope="global")

    assert graph.mark_gone(entity.id, reason="deleted") == []


# ---------- E10 · why, across both layers ----------


def test_why_shows_the_subject_entity_and_its_neighbourhood(store, approve_all):
    """Belief ancestry answers why a fact is *believed*. The entity layer
    answers why it *surfaced*, and a human debugging a bad recall needs the
    second question as much as the first."""
    fact, entity = _entity_backed_fact(
        store, text="services/auth/routes.py owns the login handler",
        entity_name="services/auth/routes.py", role="subject",
    )

    chain = store.graph.why(fact.id)

    assert chain["subject_entity"]["name"] == "services/auth/routes.py"
    assert "neighbourhood" in chain["subject_entity"]


def test_why_on_a_fact_with_no_entity_is_unchanged(store, approve_all):
    """A store that has never built the entity tier gets today's output, so
    every existing test in this file stays true unmodified."""
    root, _ = store.write(Fact(text="the deploy script is scripts/deploy.sh"))
    derived, _ = store.write(Fact(text="rollback is deploy.sh --rollback"),
                             derived_from=[root.id])

    chain = store.graph.why(derived.id)

    assert chain["rests_on"][0]["id"] == root.id
    assert chain.get("subject_entity") in (None, {})
