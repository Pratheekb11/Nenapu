"""Semantic hits enter the pool, they do not merely reorder it.

Requirement (Task 4, query-driven hybrid retrieval plan):

The whole reason to embed anything is the query that shares no word with the
fact that answers it. "Which database do we use" and "the datastore is
postgres" have no term in common, and no amount of stemming will join them.
FTS5 returns nothing for that pair, so a semantic layer that only re-ranks
what FTS already found would be sold as retrieval and deliver none: the fact
would never be in the pool to reorder.

So the assertion that matters here is not "the right fact moved up". It is
"the right fact is present at all, in a query where the lexical index returned
nothing". Everything else in this file protects the guarantees that pool has
to keep on the way in.

* **Scope.** A perfect semantic match in another repo is still another repo's
  business. Cosine similarity has no opinion about project boundaries, so the
  SQL has to.
* **Status.** Retired facts never surface, disputed ones follow the caller's
  `include_disputed`, exactly as the lexical pool already behaves.
* **Absence.** A store with no vectors, or no embedder, searches precisely as
  it did before this task. That is the degradation contract at the read path
  and it is what lets CI run the whole suite without the optional dependency.
* **Cost.** The pool is bounded like the lexical one and the budget from
  tests/test_scale.py still holds with every fact indexed.

Scope boundary with tests/test_hybrid_fusion.py
------------------------------------------------
This file pins *retrieval*: what is in the pool and what `explain` reports.
The weights that turn a semantic score into a rank are Task 6's, and nothing
here asserts on the blend. `SEARCH_WEIGHTS` is untouched by this task.

Assumed seam, proposed by the plan and not yet in the codebase::

    Store._semantic_pool(query_vec, *, statuses, scope, pool) -> dict[int, float]
    Store.search(..., semantic: bool = True)
    explain["semantic"]  in [0, 1]
"""

import hashlib
import time

import pytest

from nenapu import connect, embeddings
from nenapu.models import Fact, Status
from nenapu.store import Store

# A query and a fact that answer each other and share no word. Every claim in
# this file rests on that pair.
QUERY = "which database do we use"
ANSWER = "the datastore is postgres"


class _ScriptedEmbedder:
    """An embedder whose geometry the test decides.

    Three dimensions, one axis per meaning. Texts listed in `near` sit on the
    query's axis; everything else sits orthogonal to it, so cosine is exactly
    1.0 or exactly 0.0 and an assertion about retrieval cannot accidentally be
    an assertion about a threshold.
    """

    dim = 3

    def __init__(self, near=(), *, graded=None):
        self._near = set(near)
        self._graded = graded or {}
        self.calls = 0

    def embed(self, texts):
        self.calls += 1
        return [self._vector(t) for t in texts]

    def _vector(self, text):
        if text in self._graded:
            weight = self._graded[text]
            return [weight, 1.0 - weight, 0.0]
        if text == QUERY or text in self._near:
            return [1.0, 0.0, 0.0]
        # Hash into the third axis so unrelated facts are stably orthogonal to
        # the query rather than accidentally similar to each other.
        seed = hashlib.sha256(text.encode()).digest()[0] / 255.0
        return [0.0, 1.0 - seed, seed]


@pytest.fixture
def store():
    return Store(connect(":memory:"))


def _with_embedder(monkeypatch, embedder):
    monkeypatch.setattr(embeddings, "get_embedder", lambda: embedder)
    return embedder


def _ids(results):
    return [f.id for f, _score, _why in results]


# --- the point of the whole task ---------------------------------------------


def test_a_fact_with_no_shared_term_is_retrieved(store, monkeypatch):
    """The requirement. Not "ranked higher" -- present at all, from a query the
    lexical index cannot answer."""
    _with_embedder(monkeypatch, _ScriptedEmbedder(near=[ANSWER]))
    answer, _ = store.write(Fact(text=ANSWER))
    store.write(Fact(text="the deploy script lives in bin/release"))

    lexical_only = store.search(QUERY, semantic=False, log_recall=False, mark_used=False)
    hybrid = store.search(QUERY, log_recall=False, mark_used=False)

    assert answer.id not in _ids(lexical_only)
    assert answer.id in _ids(hybrid)


def test_the_semantic_score_is_reported(store, monkeypatch):
    """A term that moves retrieval and is invisible in `explain` is a silent
    re-ranker, which this store already refuses elsewhere."""
    _with_embedder(monkeypatch, _ScriptedEmbedder(near=[ANSWER]))
    answer, _ = store.write(Fact(text=ANSWER))

    results = store.search(QUERY, log_recall=False, mark_used=False)

    why = next(w for f, _s, w in results if f.id == answer.id)
    assert 0.0 <= why["semantic"] <= 1.0
    assert why["semantic"] == pytest.approx(1.0, abs=1e-6)


def test_an_unrelated_fact_scores_zero_semantically(store, monkeypatch):
    _with_embedder(monkeypatch, _ScriptedEmbedder(near=[ANSWER]))
    store.write(Fact(text=ANSWER))
    other, _ = store.write(Fact(text="database migrations run on deploy"))

    results = store.search("database", log_recall=False, mark_used=False)

    why = next((w for f, _s, w in results if f.id == other.id), None)
    assert why is not None
    assert why["semantic"] == pytest.approx(0.0, abs=1e-6)


# --- guarantees the pool must keep -------------------------------------------


def test_another_repos_fact_never_surfaces(store, monkeypatch):
    """Cosine has no opinion about project boundaries, so the query does."""
    _with_embedder(monkeypatch, _ScriptedEmbedder(near=[ANSWER]))
    mine, _ = store.write(Fact(text=ANSWER, scope="repo:mine@1111"))
    theirs, _ = store.write(Fact(text=ANSWER + " over there", scope="repo:theirs@2222"))
    store.conn.execute(
        "UPDATE fact_vectors SET vec = (SELECT vec FROM fact_vectors WHERE fact_id = ?)"
        " WHERE fact_id = ?", (mine.id, theirs.id))
    store.conn.commit()

    results = store.search(QUERY, scope=["global", "repo:mine@1111"],
                           log_recall=False, mark_used=False)

    assert theirs.id not in _ids(results)
    assert mine.id in _ids(results)


def test_a_retired_fact_never_surfaces(store, monkeypatch):
    _with_embedder(monkeypatch, _ScriptedEmbedder(near=[ANSWER]))
    answer, _ = store.write(Fact(text=ANSWER))
    store.forget(answer.id)

    results = store.search(QUERY, log_recall=False, mark_used=False)

    assert answer.id not in _ids(results)


def test_disputed_facts_follow_the_callers_choice(store, monkeypatch):
    _with_embedder(monkeypatch, _ScriptedEmbedder(near=[ANSWER]))
    answer, _ = store.write(Fact(text=ANSWER))
    store.set_status(answer.id, Status.DISPUTED)

    included = store.search(QUERY, include_disputed=True,
                            log_recall=False, mark_used=False)
    excluded = store.search(QUERY, include_disputed=False,
                            log_recall=False, mark_used=False)

    assert answer.id in _ids(included)
    assert answer.id not in _ids(excluded)


def test_an_unindexed_fact_is_simply_absent(store, monkeypatch):
    """No vector is not an error and not a zero-scored hit. It is a fact this
    leg cannot speak about."""
    _with_embedder(monkeypatch, _ScriptedEmbedder(near=[ANSWER]))
    answer, _ = store.write(Fact(text=ANSWER))
    store.conn.execute("DELETE FROM fact_vectors")
    store.conn.commit()

    results = store.search(QUERY, log_recall=False, mark_used=False)

    assert answer.id not in _ids(results)


def test_the_pool_is_capped_by_score_and_not_by_rowid(store, monkeypatch):
    """The best match is written last, so a pool that truncated by insertion
    order would drop exactly the fact the query was for."""
    graded = {f"candidate fact number {i}": i / 40.0 for i in range(40)}
    _with_embedder(monkeypatch, _ScriptedEmbedder(graded=graded))
    written = [store.write(Fact(text=t))[0] for t in graded]
    best = written[-1]

    results = store.search(QUERY, limit=2, log_recall=False, mark_used=False)

    assert best.id in _ids(results)


# --- the degradation contract ------------------------------------------------


def test_search_without_an_embedder_is_unchanged(store, monkeypatch):
    """The property that lets CI run the whole suite without the optional
    dependency, and that keeps a copied store readable anywhere."""
    _with_embedder(monkeypatch, _ScriptedEmbedder(near=[ANSWER]))
    store.write(Fact(text="the deploy script lives in bin/release"))
    store.write(Fact(text="deploy runs from the release branch"))
    with_leg = store.search("deploy", log_recall=False, mark_used=False)

    monkeypatch.setattr(embeddings, "get_embedder", lambda: None)
    without = store.search("deploy", log_recall=False, mark_used=False)

    assert _ids(with_leg) == _ids(without)
    for _f, _s, why in without:
        assert why.get("semantic", 0.0) == 0.0


def test_an_unindexed_store_returns_what_it_always_did(store, monkeypatch):
    store.write(Fact(text="the deploy script lives in bin/release"))
    baseline = store.search("deploy", log_recall=False, mark_used=False)

    _with_embedder(monkeypatch, _ScriptedEmbedder(near=[ANSWER]))
    store.conn.execute("DELETE FROM fact_vectors")
    store.conn.commit()
    after = store.search("deploy", log_recall=False, mark_used=False)

    assert _ids(baseline) == _ids(after)


def test_the_caller_can_turn_the_leg_off(store, monkeypatch):
    _with_embedder(monkeypatch, _ScriptedEmbedder(near=[ANSWER]))
    answer, _ = store.write(Fact(text=ANSWER))

    results = store.search(QUERY, semantic=False, log_recall=False, mark_used=False)

    assert answer.id not in _ids(results)


# --- cost --------------------------------------------------------------------


def test_a_fully_indexed_store_stays_inside_the_search_budget(tmp_path, monkeypatch):
    """The same budget shape tests/test_scale.py uses, so a regression here
    fails the way the existing ones do."""
    store = Store(connect(str(tmp_path / "s.db")))
    _with_embedder(monkeypatch, _ScriptedEmbedder(near=[ANSWER]))
    with store.transaction():
        for i in range(2000):
            store.write(Fact(text=f"cache queue fact number {i} about module_{i % 50}"))
    assert store.conn.execute(
        "SELECT COUNT(*) c FROM fact_vectors").fetchone()["c"] == 2000
    store.search("cache queue", limit=8, log_recall=False, mark_used=False)  # warm

    started = time.time()
    for _ in range(5):
        store.search("cache queue", limit=8, log_recall=False, mark_used=False)

    assert (time.time() - started) / 5 < 0.5
