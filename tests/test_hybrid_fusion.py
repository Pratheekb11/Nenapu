"""Two weight profiles, and which one a query gets.

Requirement (Task 6, query-driven hybrid retrieval plan):

The legs are built. This is where they become a number.

    hybrid   = 0.30*lexical + 0.25*semantic + 0.30*confidence
             + 0.10*anchor  + 0.05*usage
    degraded = 0.45*lexical + 0.35*confidence + 0.10*usage + 0.10*anchor

`SEARCH_WEIGHTS` is not replaced. It becomes the profile a query gets when
there is no semantic leg to weigh, which is what `tests/test_store.py` has
always pinned and is still true: those are the weights when there is nothing to
embed with. `HYBRID_WEIGHTS` is the profile for when there is. One table per
retrieval mode, both in one place, chosen once.

Since fastembed is not installed in CI, the degraded profile is what the whole
suite runs on, and the hybrid path is exercised only through a fake embedder.
That split is deliberate: a CI job that downloaded a model on every matrix
entry is a CI job people start skipping.

**Confidence stays additive.** A multiplicative gate would suppress a decayed
or disputed fact harder, and might well be the better design, but it is a
behaviour change this store has no evidence for yet. Additive is what the
existing blend does and what the belief layer's own filters already assume: it
means a fact can still surface on an overwhelming match despite weak standing,
and then arrive carrying its warning rather than not arriving at all.

Scope boundary
--------------
What is *in* the pool is Tasks 4 and 5, pinned in their own files. Nothing here
retests retrieval. This file pins the arithmetic and the mode choice.

Assumed seam, proposed by the plan and not yet in the codebase::

    store.HYBRID_WEIGHTS
    explain["mode"] in {"hybrid", "lexical"}
"""

import hashlib

import pytest

from nenapu import connect, embeddings
from nenapu.models import Fact, Origin, Status
from nenapu.store import HYBRID_WEIGHTS, SEARCH_WEIGHTS, Store

QUERY = "which database do we use"
ANSWER = "the datastore is postgres"


class _ScriptedEmbedder:
    dim = 3

    def __init__(self, near=()):
        self._near = set(near)

    def embed(self, texts):
        return [self._vector(t) for t in texts]

    def _vector(self, text):
        if text == QUERY or text in self._near:
            return [1.0, 0.0, 0.0]
        seed = hashlib.sha256(text.encode()).digest()[0] / 255.0
        return [0.0, 1.0 - seed, seed]


@pytest.fixture
def store():
    return Store(connect(":memory:"))


def _hybrid(monkeypatch, near=(ANSWER,)):
    monkeypatch.setattr(embeddings, "get_embedder", lambda: _ScriptedEmbedder(near))


def _degraded(monkeypatch):
    monkeypatch.setattr(embeddings, "get_embedder", lambda: None)


def _ids(results):
    return [f.id for f, _score, _why in results]


# --- the two profiles --------------------------------------------------------


def test_the_hybrid_weights_are_the_ones_the_plan_named():
    """On the hottest path in the system, written down in one place so a
    change to them is a change someone has to make on purpose."""
    assert HYBRID_WEIGHTS == {
        "lexical": 0.30, "semantic": 0.25, "confidence": 0.30,
        "entity": 0.10, "usage": 0.05,
    }
    assert sum(HYBRID_WEIGHTS.values()) == pytest.approx(1.0)


def test_the_old_weights_survive_as_the_degraded_profile():
    """Not a dodge around an existing assertion: these really are the weights
    that apply when there is nothing to embed with, which is what they always
    described and what CI runs on every push."""
    assert SEARCH_WEIGHTS == {
        "lexical": 0.45, "confidence": 0.35, "usage": 0.1, "proximity": 0.10,
    }
    assert sum(SEARCH_WEIGHTS.values()) == pytest.approx(1.0)
    assert "semantic" not in SEARCH_WEIGHTS


# --- mode selection ----------------------------------------------------------


def test_a_query_with_an_embedder_is_scored_hybrid(store, monkeypatch):
    _hybrid(monkeypatch)
    store.write(Fact(text=ANSWER))

    results = store.search(QUERY, log_recall=False, mark_used=False)

    assert all(why["mode"] == "hybrid" for _f, _s, why in results)


def test_a_query_without_an_embedder_is_scored_lexically(store, monkeypatch):
    _degraded(monkeypatch)
    store.write(Fact(text="the deploy script lives in bin/release"))

    results = store.search("deploy", log_recall=False, mark_used=False)

    assert results
    assert all(why["mode"] == "lexical" for _f, _s, why in results)


def test_turning_the_leg_off_returns_to_the_degraded_profile(store, monkeypatch):
    """An embedder being present is not the same as the caller wanting one."""
    _hybrid(monkeypatch)
    store.write(Fact(text="the deploy script lives in bin/release"))

    results = store.search("deploy", semantic=False, log_recall=False, mark_used=False)

    assert all(why["mode"] == "lexical" for _f, _s, why in results)


def test_the_degraded_score_is_arithmetically_unchanged(store, monkeypatch):
    """The property that keeps 1000-odd existing assertions true."""
    _degraded(monkeypatch)
    fact, _ = store.write(Fact(text="the deploy script lives in bin/release"))
    store.mark_used([fact.id])

    (_f, score, why), = store.search("deploy", log_recall=False, mark_used=False)

    expected = (
        SEARCH_WEIGHTS["lexical"] * why["lexical"]
        + SEARCH_WEIGHTS["confidence"] * why["confidence"]
        + SEARCH_WEIGHTS["usage"] * why["usage"]
        + SEARCH_WEIGHTS["proximity"] * max(why["proximity"], why["entity"])
    )
    assert score == pytest.approx(expected, abs=0.002)


def test_the_hybrid_score_is_the_blend_it_claims(store, monkeypatch):
    _hybrid(monkeypatch)
    fact, _ = store.write(Fact(text=ANSWER))
    store.mark_used([fact.id])

    (_f, score, why), = store.search(QUERY, log_recall=False, mark_used=False)

    expected = (
        HYBRID_WEIGHTS["lexical"] * why["lexical"]
        + HYBRID_WEIGHTS["semantic"] * why["semantic"]
        + HYBRID_WEIGHTS["confidence"] * why["confidence"]
        + HYBRID_WEIGHTS["entity"] * max(why["proximity"], why["entity"])
        + HYBRID_WEIGHTS["usage"] * why["usage"]
    )
    assert score == pytest.approx(expected, abs=0.002)


# --- confidence is a term, not a gate ----------------------------------------


def test_a_weakly_believed_fact_can_still_surface_on_an_overwhelming_match(
    store, monkeypatch
):
    """The whole difference between additive and multiplicative. Under a gate
    this fact would be suppressed to near nothing; under a term it arrives,
    carrying its standing for the caller to see."""
    _hybrid(monkeypatch)
    weak, _ = store.write(Fact(text=ANSWER, origin=Origin.AGENT_INFERRED,
                               confidence=0.1))

    results = store.search(QUERY, log_recall=False, mark_used=False)

    (_f, score, why), = results
    assert weak.id in _ids(results)
    assert why["confidence"] < 0.35
    assert score > HYBRID_WEIGHTS["semantic"] * 0.9


def test_confidence_still_filters_after_scoring(store, monkeypatch):
    """`min_confidence` is a belief filter, not a ranking term. It applies to
    what a fact is believed to be worth, whatever it scored."""
    _hybrid(monkeypatch)
    weak, _ = store.write(Fact(text=ANSWER, confidence=0.1))

    kept = store.search(QUERY, min_confidence=0.0, log_recall=False, mark_used=False)
    dropped = store.search(QUERY, min_confidence=0.6, log_recall=False, mark_used=False)

    assert weak.id in _ids(kept)
    assert weak.id not in _ids(dropped)


# --- invariants that must survive the new profile ----------------------------


def test_every_score_stays_inside_the_unit_interval(store, monkeypatch):
    _hybrid(monkeypatch)
    for i in range(6):
        store.write(Fact(text=f"{ANSWER} note {i}"))
    store.write(Fact(text=ANSWER))

    for _f, score, _why in store.search(QUERY, log_recall=False, mark_used=False):
        assert 0.0 <= score <= 1.0


def test_explain_still_carries_everything_it_used_to(store, monkeypatch):
    """A caller that could see why a memory surfaced must not lose that
    because the blend gained terms."""
    _hybrid(monkeypatch)
    store.write(Fact(text=ANSWER))

    (_f, _score, why), = store.search(QUERY, session_id="s-1", mark_used=False)

    assert set(why) >= {
        "lexical", "semantic", "confidence", "entity", "proximity", "usage",
        "key_match", "tag_match", "fallback", "age_days", "verify_status",
        "track_record", "suspect_reason", "recall_id", "mode",
    }


def test_a_suspect_fact_still_arrives_with_its_warning(store, monkeypatch):
    """Ranking never silences the belief layer. However well a falsified fact
    matches, it surfaces saying so."""
    _hybrid(monkeypatch)
    fact, _ = store.write(Fact(text=ANSWER))
    store.conn.execute(
        "UPDATE facts SET status = ?, suspect_reason = ? WHERE id = ?",
        (Status.SUSPECT, "its premise was falsified", fact.id))
    store.conn.commit()

    (_f, _score, why), = store.search(QUERY, log_recall=False, mark_used=False)

    assert why["suspect_reason"] == "its premise was falsified"


def test_the_no_query_fallback_is_untouched(store, monkeypatch):
    """Recency answers "no query at all". It is not a ranked result and must
    not start claiming to be one because a new profile exists."""
    _hybrid(monkeypatch)
    store.write(Fact(text=ANSWER))

    results = store.search("   ", log_recall=False, mark_used=False)

    assert results
    for _f, _score, why in results:
        assert why["fallback"] is True
        assert why["lexical"] == 0.0
        assert why["semantic"] == 0.0


def test_limit_still_truncates_after_sorting(store, monkeypatch):
    _hybrid(monkeypatch)
    for i in range(10):
        store.write(Fact(text=f"{ANSWER} variant {i}"))

    top = store.search(QUERY, limit=3, log_recall=False, mark_used=False)
    wide = store.search(QUERY, limit=10, log_recall=False, mark_used=False)

    assert len(top) == 3
    assert _ids(top) == _ids(wide)[:3]
