"""A labelled set of fact pairs, for deciding when two facts are one rule.

The open near-duplicate work needs a cosine cutoff "calibrated against labelled
real pairs", and any future decider (another model, another library) needs the
same yardstick. This module builds that yardstick from the live store and
scores deciders against it.

Mining draws from four places, each answering a different question:

  * `distilled`  - pairs the current dedupe merged. Did it merge correctly?
  * `superseded` - same subject, changed value. These must never be merged.
  * `near_miss`  - semantically close pairs nothing ever linked. The paraphrases
                   the current dedupe misses live here.
  * `random`     - unrelated pairs of the same kind. Easy negatives.

Exact textual repeats are left out: every decider gets them right, and they
would only inflate the scores.

Labels are `same`, `update` or `different`. Only `same` should merge; an
`update` merged as a duplicate silently loses the newer value, so scoring
counts those separately.

The labelled file holds real user facts and lives under NENAPU_HOME, never in
the repository.
"""

from __future__ import annotations

import json

import pytest

from nenapu import connect
from nenapu.dedup_eval import (
    default_path,
    evaluate,
    load_pairs,
    merge_labels,
    mine_pairs,
    pick_threshold,
    save_pairs,
)
from nenapu.models import Fact
from nenapu.store import Store


class _Embedder:
    """Vectors by hand, so a test decides which facts are near each other."""

    def __init__(self, vectors: dict[str, list[float]]):
        self.vectors = vectors

    def embed(self, texts):
        return [self.vectors.get(t, [0.0, 0.0, 1.0]) for t in texts]


@pytest.fixture
def store():
    return Store(connect(":memory:"))


def _add(store, text, kind="project"):
    fact, _ = store.write(Fact(text=text, kind=kind))
    return fact.id


def _link(store, column, child, parent):
    store.conn.execute(f"UPDATE facts SET {column} = ? WHERE id = ?", (parent, child))


def _pairs_by_source(pairs):
    out: dict[str, list[dict]] = {}
    for p in pairs:
        out.setdefault(p["source"], []).append(p)
    return out


# --- mining -------------------------------------------------------------------


def test_merged_and_superseded_pairs_are_mined_with_their_source(store):
    a = _add(store, "run the tests before every commit")
    b = _add(store, "always run tests prior to committing")
    c = _add(store, "the api listens on port 8000")
    d = _add(store, "the api listens on port 9000")
    _link(store, "distilled_into_id", b, a)
    _link(store, "superseded_by_id", c, d)

    pairs = _pairs_by_source(mine_pairs(store.conn, embedder=_Embedder({})))

    assert {(p["a_id"], p["b_id"]) for p in pairs["distilled"]} == {(min(a, b), max(a, b))}
    assert {(p["a_id"], p["b_id"]) for p in pairs["superseded"]} == {(min(c, d), max(c, d))}


def test_exact_repeats_are_left_out(store):
    a = _add(store, "Use uv for installs.")
    b = _add(store, "use uv for installs")
    _link(store, "distilled_into_id", b, a)

    assert mine_pairs(store.conn, embedder=_Embedder({})) == []


def test_near_misses_are_close_pairs_nothing_linked(store):
    texts = {
        "push to github from main": [1.0, 0.0, 0.0],
        "nenapu is pushed from the main branch": [0.9, 0.43, 0.0],
        "the frontend builds with vite": [0.0, 1.0, 0.0],
    }
    ids = {t: _add(store, t) for t in texts}

    pairs = _pairs_by_source(mine_pairs(store.conn, embedder=_Embedder(texts)))

    near = pairs["near_miss"]
    assert len(near) == 1
    got = near[0]
    assert {got["a"], got["b"]} == {"push to github from main",
                                    "nenapu is pushed from the main branch"}
    assert got["cosine"] == pytest.approx(0.9, abs=0.01)
    assert ids["the frontend builds with vite"] not in (got["a_id"], got["b_id"])


def test_a_linked_pair_is_not_also_a_near_miss(store):
    texts = {"run tests first": [1.0, 0.0, 0.0], "tests run first": [0.99, 0.14, 0.0]}
    a, b = (_add(store, t) for t in texts)
    _link(store, "distilled_into_id", b, a)

    pairs = mine_pairs(store.conn, embedder=_Embedder(texts))

    assert [p["source"] for p in pairs] == ["distilled"]


def test_random_negatives_pair_facts_of_the_same_kind(store):
    for i in range(6):
        _add(store, f"project fact number {i} about topic {i * 7}", kind="project")
    for i in range(6):
        _add(store, f"user prefers option {i} for case {i * 3}", kind="user")

    pairs = _pairs_by_source(mine_pairs(store.conn, embedder=_Embedder({}), per_source=4))

    rand = pairs["random"]
    assert 0 < len(rand) <= 4
    kinds = dict(store.conn.execute("SELECT id, kind FROM facts").fetchall())
    assert all(kinds[p["a_id"]] == kinds[p["b_id"]] for p in rand)


def test_mining_is_deterministic_and_capped(store):
    for i in range(30):
        _add(store, f"fact {i} says value {i * 11} for module m{i}")

    first = mine_pairs(store.conn, embedder=_Embedder({}), per_source=5, seed=3)
    again = mine_pairs(store.conn, embedder=_Embedder({}), per_source=5, seed=3)

    assert first == again
    assert all(len(v) <= 5 for v in _pairs_by_source(first).values())
    keys = [(p["a_id"], p["b_id"]) for p in first]
    assert len(keys) == len(set(keys))


def test_mining_works_without_an_embedder(store):
    a = _add(store, "run the tests before every commit")
    b = _add(store, "always run tests prior to committing")
    _link(store, "distilled_into_id", b, a)

    pairs = mine_pairs(store.conn, embedder=None)

    assert [p["source"] for p in pairs] == ["distilled"]
    assert pairs[0]["cosine"] is None


def test_every_pair_carries_the_lexical_score_dedupe_uses(store):
    from nenapu.distill import _similarity

    a = _add(store, "run the tests before every commit")
    b = _add(store, "always run tests prior to committing")
    _link(store, "distilled_into_id", b, a)

    (pair,) = mine_pairs(store.conn, embedder=None)

    assert pair["jaccard"] == pytest.approx(_similarity(pair["a"], pair["b"]))


# --- storage ------------------------------------------------------------------


def test_the_labelled_file_lives_under_nenapu_home(tmp_path, monkeypatch):
    monkeypatch.setenv("NENAPU_HOME", str(tmp_path))
    assert default_path() == tmp_path / "eval" / "dedup_pairs.jsonl"


def test_pairs_round_trip_through_jsonl(tmp_path):
    pairs = [{"a_id": 1, "b_id": 2, "a": "x", "b": "y", "source": "random",
              "cosine": 0.1, "jaccard": 0.0, "label": "different", "note": None}]
    path = tmp_path / "eval" / "pairs.jsonl"

    save_pairs(pairs, path)

    assert load_pairs(path) == pairs
    assert all(json.loads(line) for line in path.read_text().splitlines())


def test_remining_keeps_labels_already_given():
    old = [{"a_id": 1, "b_id": 2, "label": "same", "note": "checked"}]
    new = [{"a_id": 1, "b_id": 2, "label": None, "note": None, "cosine": 0.8},
           {"a_id": 3, "b_id": 4, "label": None, "note": None, "cosine": 0.7}]

    merged = merge_labels(new, old)

    assert merged[0]["label"] == "same" and merged[0]["note"] == "checked"
    assert merged[0]["cosine"] == 0.8
    assert merged[1]["label"] is None


# --- scoring ------------------------------------------------------------------


def _labelled(*rows):
    return [{"a_id": i, "b_id": i + 1000, "cosine": c, "jaccard": c, "label": lab}
            for i, (c, lab) in enumerate(rows)]


def test_evaluate_counts_merges_against_labels():
    pairs = _labelled((0.95, "same"), (0.90, "same"), (0.85, "update"),
                      (0.70, "same"), (0.50, "different"))

    (row,) = evaluate(pairs, "cosine", [0.8])

    assert row["tp"] == 2 and row["fp"] == 1 and row["fn"] == 1 and row["tn"] == 1
    assert row["precision"] == pytest.approx(2 / 3)
    assert row["recall"] == pytest.approx(2 / 3)
    assert row["updates_merged"] == 1


def test_unlabelled_and_unscored_pairs_are_ignored():
    pairs = _labelled((0.9, "same"), (0.9, None)) + [
        {"a_id": 9, "b_id": 10, "cosine": None, "jaccard": 0.9, "label": "same"}]

    (row,) = evaluate(pairs, "cosine", [0.5])

    assert row["n"] == 1


def test_pick_threshold_takes_the_best_recall_that_keeps_precision():
    """A wrong merge destroys a fact, a missed one only costs a duplicate line,
    so precision is the constraint and recall is what gets maximised."""
    pairs = _labelled((0.95, "same"), (0.90, "same"), (0.85, "different"),
                      (0.80, "same"), (0.60, "different"))
    rows = evaluate(pairs, "cosine", [0.6, 0.8, 0.85, 0.9])

    best = pick_threshold(rows, min_precision=0.95)

    assert best["threshold"] == 0.9
    assert best["recall"] == pytest.approx(2 / 3)


def test_pick_threshold_returns_none_when_nothing_is_precise_enough():
    pairs = _labelled((0.9, "different"), (0.8, "same"))
    rows = evaluate(pairs, "cosine", [0.5, 0.85])

    assert pick_threshold(rows, min_precision=0.95) is None
