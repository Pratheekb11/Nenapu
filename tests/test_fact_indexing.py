"""Keeping the vector index in step with the facts, and backfilling what exists.

Requirement (Task 3, query-driven hybrid retrieval plan):

* A fact gets a vector when it is written, and a new one when it is reworded.
  The trigger added in Task 1 only *invalidates*; something has to put the row
  back, and the write path is the only place that knows the new text.
* Indexing is best-effort. A missing embedder, a slow model, a refused write --
  none of them may turn `remember` into an error. The fact is the thing that
  matters; the vector is an accelerator.
* `index_missing` fills the gap for facts written before this existed, and for
  facts whose vector the trigger threw away. It is idempotent, bounded, and
  re-embeds anything stored under a different model, because two embedding
  spaces mixed in one dot product produce numbers that look fine and mean
  nothing.
* `nenapu index` is the command over it. `--status` reads, `--backfill` fills,
  `--warm` is the one entry point in the system allowed to download a model.

Scope boundary with tests/test_embeddings_schema.py
---------------------------------------------------
That file pins the table and the triggers: what happens to a vector when the
fact under it changes. Nothing here re-tests a trigger. This file pins the
half that puts rows *in*.

Assumed seam, proposed by the plan and not yet in the codebase::

    Store._index_fact(fact)                      called by write and revise
    embeddings.index_missing(store, *, limit=None, batch=32) -> (indexed, skipped)
    nenapu index [--status | --backfill | --warm]
"""

import hashlib
import os
import subprocess
import sys

import pytest

from nenapu import connect, embeddings
from nenapu.db import readonly
from nenapu.models import Fact
from nenapu.store import Store


class _FakeEmbedder:
    dim = 8

    def __init__(self):
        self.texts = []

    def embed(self, texts):
        self.texts.extend(texts)
        return [self._vector(t) for t in texts]

    def _vector(self, text):
        digest = hashlib.sha256(text.encode()).digest()
        return [digest[i] / 255.0 for i in range(self.dim)]


@pytest.fixture
def store():
    return Store(connect(":memory:"))


@pytest.fixture
def embedder(monkeypatch):
    fake = _FakeEmbedder()
    monkeypatch.setattr(embeddings, "get_embedder", lambda: fake)
    return fake


@pytest.fixture
def no_embedder(monkeypatch):
    monkeypatch.setattr(embeddings, "get_embedder", lambda: None)


def _vector_row(store, fact_id):
    return store.conn.execute(
        "SELECT * FROM fact_vectors WHERE fact_id = ?", (fact_id,)
    ).fetchone()


def _count(store):
    return store.conn.execute("SELECT COUNT(*) AS n FROM fact_vectors").fetchone()["n"]


# --- the write path ----------------------------------------------------------


def test_a_written_fact_is_indexed(store, embedder):
    fact, _ = store.write(Fact(text="the datastore is postgres"))

    row = _vector_row(store, fact.id)

    assert row is not None
    assert row["text_sha"] == embeddings.text_sha("the datastore is postgres")
    assert row["model"] == embeddings.MODEL_NAME
    assert row["dim"] == _FakeEmbedder.dim
    assert len(embeddings.unpack(row["vec"])) == _FakeEmbedder.dim


def test_writing_without_an_embedder_stores_the_fact_anyway(store, no_embedder):
    """The degradation contract at the write path. `remember` must not become
    an error because an optional dependency is absent."""
    fact, _ = store.write(Fact(text="a fact written on a machine with no embedder"))

    assert store.get(fact.id) is not None
    assert _vector_row(store, fact.id) is None


def test_a_throwing_embedder_does_not_break_the_write(store, monkeypatch):
    class _Broken:
        def embed(self, texts):
            raise RuntimeError("onnx fell over")

    monkeypatch.setattr(embeddings, "get_embedder", lambda: _Broken())

    fact, _ = store.write(Fact(text="a fact written while the model was broken"))

    assert store.get(fact.id) is not None
    assert _vector_row(store, fact.id) is None


def test_rewording_a_fact_reindexes_it(store, embedder):
    """The trigger deletes the stale vector; `revise` is what puts the new one
    back, because it is the only place that knows the new text."""
    fact, _ = store.write(Fact(text="the datastore is postgres"))
    before = bytes(_vector_row(store, fact.id)["vec"])

    store.revise(fact.id, text="the datastore is sqlite")

    row = _vector_row(store, fact.id)
    assert row is not None
    assert row["text_sha"] == embeddings.text_sha("the datastore is sqlite")
    assert bytes(row["vec"]) != before


def test_recurrence_and_recall_leave_the_vector_alone(store, embedder):
    """Neither changes the text, so neither should cost an embedding. Recall
    is the most common write in the system."""
    fact, _ = store.write(Fact(text="a fact that recurs and gets recalled"))
    before = bytes(_vector_row(store, fact.id)["vec"])
    calls = len(embedder.texts)

    store.note_recurrence(fact.id)
    store.mark_used([fact.id])

    assert bytes(_vector_row(store, fact.id)["vec"]) == before
    assert len(embedder.texts) == calls


# --- backfill ----------------------------------------------------------------


def test_index_missing_fills_only_what_is_missing(store, embedder, monkeypatch):
    with_vectors, _ = store.write(Fact(text="a fact indexed when it was written"))
    monkeypatch.setattr(embeddings, "get_embedder", lambda: None)
    without, _ = store.write(Fact(text="a fact written before the index existed"))
    monkeypatch.setattr(embeddings, "get_embedder", lambda: embedder)
    embedder.texts.clear()

    indexed, _skipped = embeddings.index_missing(store)

    assert indexed == 1
    assert embedder.texts == ["a fact written before the index existed"]
    assert _vector_row(store, without.id) is not None
    assert _vector_row(store, with_vectors.id) is not None


def test_index_missing_is_idempotent(store, embedder):
    store.write(Fact(text="one"))
    store.write(Fact(text="two"))
    embeddings.index_missing(store)
    embedder.texts.clear()

    indexed, _skipped = embeddings.index_missing(store)

    assert indexed == 0
    assert embedder.texts == []


def test_index_missing_respects_a_limit(store, monkeypatch):
    """A backfill over a large store has to be resumable in bounded chunks."""
    monkeypatch.setattr(embeddings, "get_embedder", lambda: None)
    for i in range(5):
        store.write(Fact(text=f"fact number {i}"))
    monkeypatch.setattr(embeddings, "get_embedder", lambda: _FakeEmbedder())

    indexed, _skipped = embeddings.index_missing(store, limit=2)

    assert indexed == 2
    assert _count(store) == 2


def test_index_missing_reembeds_a_different_model(store, embedder):
    """A model switch has to force a re-index. Cosine between vectors from two
    different models is a number with no meaning."""
    fact, _ = store.write(Fact(text="a fact embedded by the previous model"))
    store.conn.execute(
        "UPDATE fact_vectors SET model = 'some-older-model' WHERE fact_id = ?",
        (fact.id,),
    )
    store.conn.commit()
    embedder.texts.clear()

    indexed, _skipped = embeddings.index_missing(store)

    assert indexed == 1
    assert embedder.texts == ["a fact embedded by the previous model"]
    assert _vector_row(store, fact.id)["model"] == embeddings.MODEL_NAME


def test_index_missing_without_an_embedder_does_nothing_quietly(store, no_embedder):
    store.write(Fact(text="a fact on a machine with no embedder"))

    indexed, skipped = embeddings.index_missing(store)

    assert indexed == 0
    assert skipped >= 1
    assert _count(store) == 0


def test_index_missing_under_a_denied_connection_does_not_raise(store, embedder):
    """A dry run installs an authorizer that refuses writes. Indexing is an
    accelerator, so a refused write is a skipped vector, not a crash."""
    store.write(Fact(text="a fact indexed during a dry run"))
    store.conn.execute("DELETE FROM fact_vectors")
    store.conn.commit()

    with readonly(store.conn):
        indexed, _skipped = embeddings.index_missing(store)

    assert indexed == 0
    assert _count(store) == 0


def test_retired_facts_are_not_backfilled(store, embedder):
    """Retired facts are excluded from search, so embedding them buys nothing
    and a backfill over a long-lived store would pay for all 170 of them."""
    fact, _ = store.write(Fact(text="a fact that was retired"))
    store.conn.execute("DELETE FROM fact_vectors")
    store.conn.commit()
    store.forget(fact.id)
    embedder.texts.clear()

    indexed, _skipped = embeddings.index_missing(store)

    assert indexed == 0
    assert embedder.texts == []


# --- the command -------------------------------------------------------------


def _run(args, db, **env):
    return subprocess.run(
        [sys.executable, "-m", "nenapu.cli", *args, "--db", str(db)],
        capture_output=True, text=True,
        env={**os.environ, "PYTHONPATH": "src", "NENAPU_NO_BANNER": "1", **env},
    )


def test_index_status_reports_coverage_without_writing(tmp_path, no_embedder):
    """The fixture writes with the embedder off deliberately. Without that the
    in-process write indexes the fact and the test reports 1/1 on a machine
    that installed the optional extra and 0/1 on one that did not, which makes
    it a test of the environment rather than of the command."""
    db = tmp_path / "s.db"
    store = Store(connect(str(db)))
    store.write(Fact(text="a fact with no vector"))
    store.conn.close()

    out = _run(["index", "--status"], db, NENAPU_EMBEDDINGS="off")

    assert out.returncode == 0
    assert "0/1" in out.stdout
    check = Store(connect(str(db)))
    assert _count(check) == 0


def test_index_backfill_without_an_embedder_exits_clean(tmp_path):
    """Exit 0 and say so. A backfill that cannot run is a state to report, not
    a failure: the store still works, it just has no semantic leg."""
    db = tmp_path / "s.db"
    store = Store(connect(str(db)))
    store.write(Fact(text="a fact that cannot be embedded here"))
    store.conn.close()

    out = _run(["index", "--backfill"], db, NENAPU_EMBEDDINGS="off")

    assert out.returncode == 0
    assert out.stdout.strip()


def test_index_is_hidden_from_the_listing(tmp_path):
    """Machine-to-machine upkeep, not something to put in front of someone
    reading the command list."""
    db = tmp_path / "s.db"

    listed = _run(["--help"], db)
    reachable = _run(["index", "--status"], db, NENAPU_EMBEDDINGS="off")

    assert reachable.returncode == 0
    assert listed.returncode == 0
    rows = [ln for ln in listed.stdout.splitlines() if ln.strip().startswith("index")]
    assert rows == []
