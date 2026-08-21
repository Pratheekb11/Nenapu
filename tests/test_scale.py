"""Performance regressions, kept honest by measurement.

Every budget here is generous relative to the fixed timing but far under the
pre-fix number, which is recorded beside it. These are the operations that
looked fine on twenty facts and did not return on a few thousand.
"""

import random
import time

import pytest

from nenapu import connect
from nenapu.distill import _similarity, dedupe
from nenapu.models import Fact, Status
from nenapu.store import Store

WORDS = "deploy cache queue auth billing render index worker shard token".split()


def _seeded(tmp_path, n, name="s"):
    random.seed(7)
    store = Store(connect(str(tmp_path / f"{name}.db")))
    with store.transaction():
        for i in range(n):
            # Distinct text per row: `_insert` bypasses the duplicate check, and
            # the partial unique index on active facts rightly rejects repeats.
            store._insert(Fact(
                text=f"fact {i}: the {random.choice(WORDS)} service uses "
                     f"{random.choice(WORDS)} on port {8000 + i % 400}",
                kind="project",
            ))
    return store


def _brute_force(store, threshold=0.85):
    """The original O(n^2) implementation, as the reference answer."""
    from nenapu.store import effective_confidence

    kept, archived = [], []
    for fact in sorted(store.list_facts(limit=10_000), key=effective_confidence,
                       reverse=True):
        twin = next((k for k in kept if _similarity(k.text, fact.text) >= threshold), None)
        if twin is None:
            kept.append(fact)
        else:
            archived.append(fact.id)
    return set(archived)


def test_blocking_agrees_with_brute_force(tmp_path):
    """The prefix filter exists to skip comparisons that cannot match. If it
    skips one that could, duplicates survive silently — so the fast path must
    return exactly what comparing every pair returns."""
    store = _seeded(tmp_path, 300, "agree")
    expected = _brute_force(store)

    dedupe(store)
    actual = {f.id for f in store.list_facts(status=Status.ARCHIVED, limit=10_000)}

    assert actual == expected


def test_dedupe_scales(tmp_path):
    """Was ~103s at 3,000 facts and did not finish the benchmark at 5,000."""
    store = _seeded(tmp_path, 1500, "scale")
    started = time.time()
    dedupe(store)
    assert time.time() - started < 10


def test_recall_is_fast_on_a_large_store(tmp_path):
    """The hottest path in the system. Was 5.7s at 3,000 facts, because the
    write-back cost one fsync per row and every bump re-indexed the row."""
    store = _seeded(tmp_path, 2000, "recall")
    store.search("cache queue", limit=8)          # warm

    started = time.time()
    for _ in range(5):
        store.search("cache queue", limit=8, session_id="bench")
    assert (time.time() - started) / 5 < 0.5


def test_cascade_is_fast_on_a_wide_graph(tmp_path):
    """Was 346s over ~900 dependents: the walk is cheap, but each affected node
    paid its own durable write."""
    store = _seeded(tmp_path, 1000, "cascade")
    ids = [f.id for f in store.list_facts(limit=1000)]
    root = ids[0]
    with store.transaction():
        for child in ids[1:900]:
            store.graph.link(root, child)

    started = time.time()
    affected = store.graph.cascade_falsification(root, "bench")
    elapsed = time.time() - started

    assert len(affected) > 800
    assert elapsed < 10


def test_writing_many_facts_stays_linear(tmp_path):
    store = Store(connect(str(tmp_path / "w.db")))
    started = time.time()
    for i in range(200):
        store.write(Fact(text=f"distinct claim number {i}"))
    assert time.time() - started < 20


# ==========================================================================
# Pre-written budgets for R1 · candidate generation and E7 · entity-anchored
# retrieval. Both touch `Store.search`, which is the hottest path in the
# system, and both make it do more work: R1 unions a confidence-ordered pool
# with the lexical one, E7 adds a depth-2 graph traversal per query.
#
# The budgets below are generous relative to the fixed timing and far under
# anything a user would notice, which is the same shape every budget in this
# file already has. A regression here is paid on every recall of every
# session.
# ==========================================================================



def test_the_pool_union_does_not_slow_recall_down(tmp_path):
    """R1's fix is a second, confidence-ordered pool unioned with the lexical
    one before scoring. Two pools is more rows to score, on the path that was
    5.7s at 3,000 facts before the last round of work."""
    store = _seeded(tmp_path, 2000, "pool")
    store.search("cache queue", limit=8)          # warm

    started = time.time()
    for _ in range(5):
        store.search("cache queue", limit=8, session_id="bench")

    assert (time.time() - started) / 5 < 0.5


def test_a_multi_term_query_stays_cheap(tmp_path):
    """`relevant_memory` feeds `search` up to 40 salient terms from a whole
    session. Whatever R1 does about required versus optional terms, forty of
    them must not turn one query into forty."""
    store = _seeded(tmp_path, 2000, "terms")
    query = " ".join(WORDS * 4)
    store.search(query, limit=8)                  # warm

    started = time.time()
    store.search(query, limit=8, session_id="bench")

    assert time.time() - started < 1.0


def test_entity_anchored_recall_stays_within_budget(tmp_path):
    """Depth-2 traversal with per-hop decay, per query, on a store with an
    entity per touched path — 647 of them in the live store today."""
    from nenapu.entities import EntityGraph

    store = _seeded(tmp_path, 2000, "anchor")
    graph = EntityGraph(store.conn)
    with store.transaction():
        previous = None
        for i, fact in enumerate(store.list_facts(limit=400)):
            entity = graph.upsert(kind="file", name=f"app/module_{i}.py", scope="global")
            graph.attach(fact.id, entity.id, role="subject", source="path")
            if previous is not None:
                graph.link(previous, entity.id, kind="touched_with")
            previous = entity.id
    store.search("cache queue", limit=8, near=["app/module_0.py"])   # warm

    started = time.time()
    for _ in range(5):
        store.search("cache queue", limit=8, near=["app/module_0.py"], session_id="bench")

    assert (time.time() - started) / 5 < 0.5


def test_an_unanchored_query_pays_nothing_for_the_entity_tier(tmp_path):
    """A store with entities in it must not make every unanchored recall walk
    a graph it was not asked about."""
    from nenapu.entities import EntityGraph

    store = _seeded(tmp_path, 2000, "unanchored")
    graph = EntityGraph(store.conn)
    with store.transaction():
        for i, fact in enumerate(store.list_facts(limit=400)):
            entity = graph.upsert(kind="file", name=f"app/module_{i}.py", scope="global")
            graph.attach(fact.id, entity.id, role="subject", source="path")
    store.search("cache queue", limit=8)          # warm

    started = time.time()
    for _ in range(5):
        store.search("cache queue", limit=8, session_id="bench")

    assert (time.time() - started) / 5 < 0.5
