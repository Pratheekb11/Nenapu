"""Entities named in the query, and what a link to one is worth.

Requirement (Task 5, query-driven hybrid retrieval plan):

The entity tier already answers "what is this session near?" -- `near=` carries
the paths being edited, and `proximity_scores` walks out from them. It has
never answered "what did the user just ask about?", because until now nothing
passed a query into it. The `proximity` weight has therefore been zero on every
query surface since it was added.

This task supplies the second anchor. The traversal is the same one; only the
starting names change, from files the session touched to entities the query
named. Reusing `proximity_scores` rather than writing a second walk is
deliberate: it is already depth-capped, cycle-safe, alias-resolved and
scope-guarded, and it has its own regression file.

Two weights decide what a link is worth.

**Promiscuity.** An entity attached to half the store distinguishes nothing.
`link_weight` falls off with how many facts an entity touches, with the knee at
a *fraction* of the store rather than a fixed count, so the term does not go
flat as the store grows. It floors the corpus size, because on a five-fact
store every entity looks promiscuous and the term would zero itself out.

**Track record.** `entity_edges.weight` already learns from graded recalls.
`entity_weight` reads it, and clamps so that learning can only ever *demote*.
The asymmetry is on purpose and it is the same one alias resolution documents:
a missed boost leaves ranking where it already is, while an inflated one
compounds into a loop that promotes whatever the ranker already likes.

Scope boundary
--------------
`tests/test_proximity_scope_guard.py` pins the traversal's scope guarantee and
`tests/test_store.py` pins the `near=` anchor's ordering. Nothing here retests
either. This file pins the query anchor, the two weights, and the promise that
a query naming nothing costs nothing.

Assumed seam, proposed by the plan and not yet in the codebase::

    entities.query_entities(conn, terms, scopes) -> list[str]
    entities.link_weight(n_linked, n_active) -> float
    entities.entity_weight(conn, entity_id) -> float
    entities.LINK_WEIGHT_FRACTION, MIN_ACTIVE_FACTS, MAX_ENTITY_BOOSTED
    explain["entity"]
"""

import pytest

from nenapu import connect
from nenapu.entities import (
    LINK_WEIGHT_FRACTION,
    MAX_EDGE_WEIGHT,
    MAX_ENTITY_BOOSTED,
    MIN_ACTIVE_FACTS,
    MIN_EDGE_WEIGHT,
    EntityGraph,
    entity_weight,
    link_weight,
    query_entities,
)
from nenapu.models import Fact
from nenapu.store import Store

PATH = "services/auth/routes.py"


@pytest.fixture
def store():
    return Store(connect(":memory:"))


@pytest.fixture
def graph(store):
    return EntityGraph(store.conn)


def _attached(store, graph, text, path=PATH, *, scope="global", role="subject"):
    fact, _ = store.write(Fact(text=text, scope=scope))
    entity = graph.upsert(kind="file", name=path, scope=scope)
    graph.attach(fact.id, entity.id, role=role, source="path")
    return fact, entity


def _ids(results):
    return [f.id for f, _score, _why in results]


class _Trace:
    """Records every statement the connection prepares.

    A wall-clock assertion cannot tell "did no work" from "did fast work" on a
    small fixture, and the budget this protects lives in tests/test_scale.py
    with 2000 facts. Counting statements says the thing directly.
    """

    def __init__(self, conn):
        self.statements: list[str] = []
        conn.set_trace_callback(self.statements.append)

    def touched(self, table: str) -> bool:
        return any(table in s for s in self.statements)


# --- the query anchor --------------------------------------------------------


def test_a_query_naming_an_entity_surfaces_its_facts(store, graph):
    """The point of the task. The fact shares no term with the query at all,
    so the lexical pool cannot contain it; only the entity link puts it there.

    Presence, not position: what a retrieved fact then scores is the fusion's
    business and is pinned in tests/test_hybrid_fusion.py."""
    wanted, _ = store.write(Fact(text="the login endpoint rejects empty tokens"))
    for i in range(8):
        store.write(Fact(text=f"routes and handlers note number {i}"))
    query = f"what does {PATH} do"

    before = store.search(query, limit=10, semantic=False,
                          log_recall=False, mark_used=False)

    entity = graph.upsert(kind="file", name=PATH, scope="global")
    graph.attach(wanted.id, entity.id, role="subject", source="path")
    after = store.search(query, limit=10, semantic=False,
                         log_recall=False, mark_used=False)

    assert wanted.id not in _ids(before)
    assert wanted.id in _ids(after)


def test_the_entity_score_is_reported(store, graph):
    wanted, _entity = _attached(store, graph, "the login handler rejects empty tokens")

    results = store.search(f"what does {PATH} do", semantic=False,
                           log_recall=False, mark_used=False)

    why = next(w for f, _s, w in results if f.id == wanted.id)
    assert 0.0 < why["entity"] <= 1.0


def test_a_query_naming_nothing_pays_nothing(store, graph):
    """The guard for tests/test_scale.py's unanchored budget. The cheap
    indexed probe is allowed; the traversal it would start is not."""
    _attached(store, graph, "the login handler rejects empty tokens")
    trace = _Trace(store.conn)

    store.search("cache queue", semantic=False, log_recall=False, mark_used=False)

    assert not trace.touched("fact_entities")
    assert not trace.touched("entity_edges")


def test_extraction_never_invents_an_entity(store, graph):
    """The rule `mentions_from_text` is built on. This step may connect a
    query to something observed to exist, never conjure a node."""
    graph.upsert(kind="file", name=PATH, scope="global")

    found = query_entities(store.conn, ["services/other/missing.py", "routes.py"], ["global"])

    assert found == []
    assert store.conn.execute("SELECT COUNT(*) c FROM entities").fetchone()["c"] == 1


def test_extraction_finds_a_path_shaped_token(store, graph):
    graph.upsert(kind="file", name=PATH, scope="global")

    found = query_entities(store.conn, [PATH, "unrelated"], ["global"])

    assert found == [PATH]


def test_another_scopes_entity_contributes_nothing(store, graph):
    """A new caller of the traversal is a new way to lose the scope guard, so
    it is asserted here as well as in its own file."""
    theirs, _ = _attached(store, graph, "their login handler", scope="repo:theirs@2222")

    results = store.search(f"what does {PATH} do", scope=["global", "repo:mine@1111"],
                           semantic=False, log_recall=False, mark_used=False)

    assert theirs.id not in _ids(results)


def test_at_most_five_facts_ride_one_entity(store, graph):
    """An entity on twenty facts would otherwise fill every slot the query
    had, which is the failure the promiscuity weight exists to soften and the
    cap exists to bound."""
    entity = graph.upsert(kind="file", name=PATH, scope="global")
    ids = []
    for i in range(20):
        fact, _ = store.write(Fact(text=f"a note about the handler number {i}"))
        graph.attach(fact.id, entity.id, role="subject", source="path")
        ids.append(fact.id)

    results = store.search(f"what does {PATH} do", limit=25, semantic=False,
                           log_recall=False, mark_used=False)

    boosted = [f.id for f, _s, why in results if why["entity"] > 0.0]
    assert len(boosted) <= MAX_ENTITY_BOOSTED


# --- the promiscuity weight --------------------------------------------------


def test_a_unique_entity_is_worth_full_weight():
    assert link_weight(1, 100) == pytest.approx(1.0)


def test_the_knee_sits_at_a_fraction_of_the_store():
    """Half weight where an entity touches `LINK_WEIGHT_FRACTION` of the
    corpus, so the term keeps its shape as the store grows instead of going
    flat at a count that made sense when it was written."""
    for total in (100, 1000, 10000):
        knee = int(LINK_WEIGHT_FRACTION * total) + 1

        assert link_weight(knee, total) == pytest.approx(0.5, abs=0.02)


def test_the_weight_falls_off_monotonically():
    weights = [link_weight(n, 400) for n in range(1, 200, 7)]

    assert weights == sorted(weights, reverse=True)
    assert weights[-1] < weights[0]


def test_a_tiny_store_does_not_zero_every_entity():
    """On a five-fact store every entity looks promiscuous. Flooring the
    corpus size keeps the term meaningful until there is enough of one."""
    assert link_weight(2, 1) == pytest.approx(link_weight(2, MIN_ACTIVE_FACTS))
    assert link_weight(2, 1) > 0.1


def test_an_unlinked_entity_does_not_divide_by_zero():
    assert 0.0 <= link_weight(0, 100) <= 1.0


# --- the learned weight ------------------------------------------------------


def test_an_untrained_entity_is_neutral(store, graph):
    entity = graph.upsert(kind="file", name=PATH, scope="global")

    assert entity_weight(store.conn, entity.id) == pytest.approx(1.0)


def test_a_rewarded_entity_cannot_climb_above_neutral(store, graph):
    """Learning demotes only. A missed boost leaves ranking where it is; an
    inflated one compounds into a loop that promotes what already ranks."""
    entity = graph.upsert(kind="file", name=PATH, scope="global")
    other = graph.upsert(kind="file", name="services/auth/models.py", scope="global")
    graph.link(entity.id, other.id, kind="touched_with")
    store.conn.execute("UPDATE entity_edges SET weight = ?", (MAX_EDGE_WEIGHT,))
    store.conn.commit()

    assert entity_weight(store.conn, entity.id) == pytest.approx(1.0)


def test_a_penalised_entity_stops_steering(store, graph):
    entity = graph.upsert(kind="file", name=PATH, scope="global")
    other = graph.upsert(kind="file", name="services/auth/models.py", scope="global")
    graph.link(entity.id, other.id, kind="touched_with")
    store.conn.execute("UPDATE entity_edges SET weight = ?", (MIN_EDGE_WEIGHT,))
    store.conn.commit()

    assert entity_weight(store.conn, entity.id) == pytest.approx(MIN_EDGE_WEIGHT)


# --- regression --------------------------------------------------------------


def test_the_session_anchor_is_unchanged_by_the_query_anchor(store, graph):
    """`near=` and its `proximity` reading are what four tests in
    tests/test_store.py assert on. A query naming no entity must leave them
    numerically where they were."""
    fact, _entity = _attached(store, graph, "the login handler rejects empty tokens")

    results = store.search("handler", near=[PATH], semantic=False,
                           log_recall=False, mark_used=False)

    why = next(w for f, _s, w in results if f.id == fact.id)
    assert why["proximity"] == pytest.approx(1.0)
