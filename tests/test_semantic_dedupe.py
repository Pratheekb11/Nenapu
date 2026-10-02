"""Dedupe also merges reworded facts, judged by their stored vectors.

Lexical dedupe (token similarity at 0.85) merged 0 of the 23 true paraphrases
in the labelled pair set at ~/.nenapu/eval/dedup_pairs.jsonl; reworded rules
score about 0.26 to 0.69 on it. Cosine over the stored bge-small vectors at
0.875 caught 20 of those 23 with no wrong merge, and over all 240 labelled
pairs took recall from 0.51 to 0.92 at the same 0.97 precision.

What the cosine cannot tell apart is an update: the same subject with a newer
value ("238 tests passing" then "243"). Three of the four wrong merges at
0.875 were updates. So the newer fact survives a semantic merge, which turns
those into the right outcome, and the survivor keeps the higher confidence
and the summed occurrence count, so merging never weakens what is believed.

Guards were measured and rejected: requiring equal numbers dropped recall to
0.71, and blocking pairs that name different identifiers dropped it to 0.77,
each to remove at most one wrong merge.

Only stored vectors are read. A fact with no vector, a vector from another
model, or a vector for text that has since been revised is skipped rather
than embedded here, so dedupe stays as cheap as before when nothing is
indexed. The maintenance tick fills missing vectors first, in a bounded batch.
"""

from __future__ import annotations

import math

import pytest

from nenapu import connect, embeddings
from nenapu.distill import SEMANTIC_DUPLICATE, dedupe
from nenapu.models import Fact, Status
from nenapu.store import Store

PUSH = "Nenapu is pushed to GitHub from the main branch."
PUSH_REWORDED = "Push the Nenapu repo to GitHub using main."
FRONTEND = "The marketplace frontend builds with vite."


@pytest.fixture
def store(monkeypatch):
    # No embedder: writes store no vectors, so each test decides exactly
    # which vectors exist.
    monkeypatch.setattr(embeddings, "get_embedder", lambda: None)
    return Store(connect(":memory:"))


def _unit(cos: float) -> list[float]:
    """A vector at cosine `cos` from [1, 0, 0]."""
    return [cos, math.sqrt(1 - cos * cos), 0.0]


BASE = [1.0, 0.0, 0.0]


def _vec(store, fact_id, vec, *, text=None, model=embeddings.MODEL_NAME):
    if text is None:
        text = store.get(fact_id).text
    store.conn.execute(
        "INSERT OR REPLACE INTO fact_vectors(fact_id, model, dim, text_sha, vec, created_at)"
        " VALUES (?,?,?,?,?,0)",
        (fact_id, model, len(vec), embeddings.text_sha(text), embeddings.pack(vec)),
    )


def _write(store, text, *, created_at, scope="global", confidence=0.8, occurrences=1):
    fact, _ = store.write(Fact(text=text, kind="feedback", scope=scope,
                               confidence=confidence))
    store.conn.execute("UPDATE facts SET created_at = ?, occurrences = ? WHERE id = ?",
                       (created_at, occurrences, fact.id))
    return fact.id


def test_the_cutoff_is_the_measured_one():
    assert SEMANTIC_DUPLICATE == 0.875


def test_a_reworded_fact_merges_into_the_newer_one(store):
    old = _write(store, PUSH, created_at=1000.0)
    new = _write(store, PUSH_REWORDED, created_at=2000.0)
    _vec(store, old, BASE)
    _vec(store, new, _unit(0.9))

    assert dedupe(store, scope="global") == 1

    archived = store.get(old)
    assert archived.status == Status.ARCHIVED
    assert archived.distilled_into_id == new
    assert store.get(new).status == Status.ACTIVE


def test_the_survivor_keeps_the_stronger_belief_and_the_counts(store):
    old = _write(store, PUSH, created_at=1000.0, confidence=0.95, occurrences=3)
    new = _write(store, PUSH_REWORDED, created_at=2000.0, confidence=0.6, occurrences=2)
    _vec(store, old, BASE)
    _vec(store, new, _unit(0.95))

    dedupe(store, scope="global")

    survivor = store.get(new)
    assert survivor.confidence == pytest.approx(0.95)
    assert survivor.occurrences == 5


def test_pairs_below_the_cutoff_stay_apart(store):
    a = _write(store, PUSH, created_at=1000.0)
    b = _write(store, FRONTEND, created_at=2000.0)
    _vec(store, a, BASE)
    _vec(store, b, _unit(0.85))

    assert dedupe(store, scope="global") == 0
    assert store.get(a).status == Status.ACTIVE


def test_without_vectors_dedupe_is_lexical_as_before(store):
    a = _write(store, PUSH, created_at=1000.0)
    b = _write(store, PUSH_REWORDED, created_at=2000.0)

    assert dedupe(store, scope="global") == 0
    assert {store.get(a).status, store.get(b).status} == {Status.ACTIVE}


def test_a_vector_for_text_since_revised_is_not_trusted(store):
    old = _write(store, PUSH, created_at=1000.0)
    new = _write(store, PUSH_REWORDED, created_at=2000.0)
    _vec(store, old, BASE, text="something the fact used to say")
    _vec(store, new, _unit(0.95))

    assert dedupe(store, scope="global") == 0


def test_a_vector_from_another_model_is_not_trusted(store):
    old = _write(store, PUSH, created_at=1000.0)
    new = _write(store, PUSH_REWORDED, created_at=2000.0)
    _vec(store, old, BASE, model="some/other-model")
    _vec(store, new, _unit(0.95))

    assert dedupe(store, scope="global") == 0


def test_facts_in_other_scopes_are_not_merged(store):
    a = _write(store, PUSH, created_at=1000.0, scope="repo:a")
    b = _write(store, PUSH_REWORDED, created_at=2000.0, scope="repo:b")
    _vec(store, a, BASE)
    _vec(store, b, _unit(0.95))

    assert dedupe(store, scope="repo:a") == 0
    assert dedupe(store, scope="repo:b") == 0


def test_three_rewordings_collapse_into_the_newest(store):
    ids = [_write(store, text, created_at=t) for text, t in (
        (PUSH, 1000.0),
        (PUSH_REWORDED, 2000.0),
        ("Nenapu goes to GitHub from main.", 3000.0),
    )]
    for fact_id, cos in zip(ids, (1.0, 0.97, 0.95)):
        _vec(store, fact_id, _unit(cos))

    assert dedupe(store, scope="global") == 2

    newest = ids[-1]
    assert store.get(newest).status == Status.ACTIVE
    assert all(store.get(i).distilled_into_id == newest for i in ids[:-1])
    assert store.get(newest).occurrences == 3


def test_a_lexical_duplicate_is_not_merged_twice(store):
    a = _write(store, PUSH, created_at=1000.0)
    b = _write(store, PUSH + " ", created_at=2000.0)
    _vec(store, a, BASE)
    _vec(store, b, BASE, text=PUSH + " ")

    assert dedupe(store, scope="global") == 1
    statuses = sorted(store.get(i).status for i in (a, b))
    assert statuses == sorted([Status.ACTIVE, Status.ARCHIVED])


# --- the maintenance tick fills vectors first ---------------------------------


class _Embedder:
    def embed(self, texts):
        return [BASE if "GitHub" in t else [0.0, 1.0, 0.0] for t in texts]


def test_the_tick_indexes_missing_vectors_then_merges(store, monkeypatch):
    from nenapu.maintenance import run_maintenance_tick

    old = _write(store, PUSH, created_at=1000.0)
    new = _write(store, PUSH_REWORDED, created_at=2000.0)
    other = _write(store, FRONTEND, created_at=3000.0)
    monkeypatch.setattr(embeddings, "get_embedder", lambda: _Embedder())

    run_maintenance_tick(store, touched_scopes=["global"])

    assert store.get(old).distilled_into_id == new
    assert store.get(other).status == Status.ACTIVE
