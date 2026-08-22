"""The scope guard on the graph walk, tested where it can actually fail.

Requirement (plan "Harden the four incidents into guarantees", Phase F task
F2, marked **Opus**):

    A green run on a test that cannot fail is worse than no test.

Two tests claim to hold the E7 traversal inside its project —
`test_traversal_does_not_cross_scope` in `tests/test_store.py` and
`test_the_anchor_does_not_traverse_out_of_the_project` in
`tests/test_project_injection.py`. Both go through `Store.search`, and both
still pass with the guard deleted. Verified by removing the `_in_scope` filter
from `proximity_scores` and running all three files: 152 passed.

They cannot fail because `Store.search` already restricts its candidate pool
by scope, so a proximity score computed for an out-of-scope fact never reaches
the result they assert about. The guard they are named for is invisible to
them. They are not wrong, and they are not deleted here — as end-to-end
statements they still say something true about the block. They just do not
hold the guard up, and something has to.

So this file asks `proximity_scores` directly, which is the one place the
guard operates. Removing `_in_scope` turns these red.
"""

import pytest

from nenapu import connect
from nenapu.entities import EntityGraph, proximity_scores
from nenapu.models import Fact
from nenapu.store import Store

HERE = "repo:here@aaaaaaaa"
THERE = "repo:there@bbbbbbbb"


@pytest.fixture
def store():
    return Store(connect(":memory:"))


def _fact_on(store, graph, *, text, path, scope):
    """A fact attached to a file entity, the shape the anchor walks."""
    fact, _ = store.write(Fact(text=text, scope=scope))
    entity = graph.upsert(kind="file", name=path, scope=scope)
    graph.attach(fact.id, entity.id, role="subject", source="path")
    return fact, entity


def test_the_walk_does_not_score_a_fact_from_another_project(store):
    """"Right fact, wrong project" is the failure scoping already fixed once.
    A graph walk that ignores scope recreates it one layer down: the edge is
    real, the neighbour is real, and the fact belongs to someone else."""
    graph = EntityGraph(store.conn)
    _, here = _fact_on(store, graph, text="the handler validates the token",
                       path="app/routes.py", scope=HERE)
    elsewhere, there = _fact_on(store, graph, text="the portfolio deploys from netlify",
                                path="portfolio/src/index.astro", scope=THERE)
    graph.link(here.id, there.id, kind="touched_with", source="observed")

    scores = proximity_scores(store.conn, ["app/routes.py"], ["global", HERE])

    assert elsewhere.id not in scores


def test_the_walk_still_scores_the_fact_in_this_project(store):
    """The other half of the same claim: a guard that returned nothing would
    pass the test above and be useless."""
    graph = EntityGraph(store.conn)
    mine, _ = _fact_on(store, graph, text="the handler validates the token",
                       path="app/routes.py", scope=HERE)

    scores = proximity_scores(store.conn, ["app/routes.py"], ["global", HERE])

    assert scores.get(mine.id) == 1.0


def test_a_global_fact_is_still_reached_from_a_project(store):
    """The one crossing that is allowed. Global facts are meant to surface
    everywhere, so the guard must be a scope filter and not a same-scope
    equality check."""
    graph = EntityGraph(store.conn)
    _, here = _fact_on(store, graph, text="the handler validates the token",
                       path="app/routes.py", scope=HERE)
    everywhere, shared = _fact_on(store, graph,
                                  text="commits never carry a co-author trailer",
                                  path="tools/commit-check.sh", scope="global")
    graph.link(here.id, shared.id, kind="touched_with", source="observed")

    scores = proximity_scores(store.conn, ["app/routes.py"], ["global", HERE])

    assert everywhere.id in scores


def test_a_neighbour_two_hops_out_of_scope_is_not_reached(store):
    """Depth is not a way around the guard: the walk reaches two hops, and the
    second hop must be filtered the same as the first."""
    graph = EntityGraph(store.conn)
    _, here = _fact_on(store, graph, text="the handler validates the token",
                       path="app/routes.py", scope=HERE)
    _, middle = _fact_on(store, graph, text="the middle file", path="app/middle.py",
                         scope=HERE)
    far, there = _fact_on(store, graph, text="the portfolio deploys from netlify",
                          path="portfolio/src/index.astro", scope=THERE)
    graph.link(here.id, middle.id, kind="touched_with", source="observed")
    graph.link(middle.id, there.id, kind="touched_with", source="observed")

    scores = proximity_scores(store.conn, ["app/routes.py"], ["global", HERE])

    assert far.id not in scores
