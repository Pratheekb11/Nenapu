"""Storage for fact vectors, and the invalidation discipline around them.

Requirement (Task 1, query-driven hybrid retrieval plan):

* A `fact_vectors` table sits beside `facts`, one row per embedded fact,
  keyed by `fact_id` so a fact cannot carry two vectors at once. It records
  which model produced the vector and a hash of the text that was embedded,
  because a model switch has to be detectable rather than silently mixing
  two embedding spaces in one dot product.
* The triggers on it **invalidate, they never embed**. SQLite cannot call an
  ONNX model, and a trigger that tried would put inference on the write path.
  Rewording a fact deletes its vector; a missing row is the signal the
  indexer looks for.
* Invalidation is scoped to `text` for the same reason `facts_au` was scoped
  to the indexed columns (`db.py:78-81`): recall bumps `use_count` on every
  surfaced fact, and an unscoped `AFTER UPDATE` would throw away a vector on
  the single most common write in the system, then pay to recompute it.
* `entities` gains an index on `name` alone. `UNIQUE(kind, name, scope)`
  leads with `kind`, so the query-entity lookup this plan adds would be a
  full table scan, which `tests/test_scale.py:174` forbids paying for on a
  query that named no entity.

Assumed seam, proposed by the plan and not yet in the codebase::

    fact_vectors(fact_id PK, model, dim, text_sha, vec BLOB, created_at)
    trigger facts_vec_au  AFTER UPDATE OF text ON facts -> DELETE the vector
    trigger facts_vec_ad  AFTER DELETE ON facts         -> DELETE the vector
    index   idx_entities_name ON entities(name)
"""

import sqlite3

import pytest

from nenapu import connect
from nenapu.models import Fact
from nenapu.store import Store


@pytest.fixture(autouse=True)
def _no_embedder(monkeypatch):
    """This file is about the table and its triggers, and it stubs vectors by
    hand so it can prove them without an embedder. With the optional extra
    installed the write path would index for real, and the assertions would
    then measure whether fastembed happens to be present."""
    from nenapu import embeddings

    monkeypatch.setattr(embeddings, "get_embedder", lambda: None)


@pytest.fixture
def store():
    return Store(connect(":memory:"))


def _stub_vector(conn, fact_id, *, model="bge-small-en-v1.5", dim=4, blob=b"\x00\x01\x02\x03"):
    """Write a vector row without an embedder.

    Task 1 is the schema, and it has to be provable on its own: an assertion
    about a trigger must not fail because fastembed is absent.
    """
    conn.execute(
        "INSERT OR REPLACE INTO fact_vectors(fact_id, model, dim, text_sha, vec, created_at)"
        " VALUES (?, ?, ?, ?, ?, ?)",
        (fact_id, model, dim, "sha-of-the-text", sqlite3.Binary(blob), 1000.0),
    )
    conn.commit()


def _vector_rows(conn, fact_id):
    return conn.execute(
        "SELECT * FROM fact_vectors WHERE fact_id = ?", (fact_id,)
    ).fetchall()


def test_the_schema_version_is_bumped():
    from nenapu.db import SCHEMA_VERSION

    assert SCHEMA_VERSION > 10


def test_an_existing_store_gains_the_vector_table_on_reconnect(tmp_path):
    """Every table is `CREATE TABLE IF NOT EXISTS` and `connect()` reruns the
    script, so a store written before this task migrates by being opened."""
    path = tmp_path / "old.db"
    old = Store(connect(str(path)))
    fact, _ = old.write(Fact(text="a fact written before the vector table"))
    old.conn.close()

    reopened = Store(connect(str(path)))
    tables = {r["name"] for r in reopened.conn.execute(
        "SELECT name FROM sqlite_master WHERE type = 'table'"
    )}

    assert "fact_vectors" in tables
    assert reopened.get(fact.id).text == "a fact written before the vector table"


def test_the_vector_table_carries_what_the_indexer_needs(store):
    columns = {r["name"] for r in store.conn.execute("PRAGMA table_info(fact_vectors)")}

    assert columns >= {"fact_id", "model", "dim", "text_sha", "vec", "created_at"}


def test_a_fact_can_only_carry_one_vector(store):
    """`fact_id` is the primary key. Two rows for one fact would make the
    semantic pool return the same fact twice at two different scores."""
    keys = [r["name"] for r in store.conn.execute("PRAGMA table_info(fact_vectors)")
            if r["pk"]]

    assert keys == ["fact_id"]


def test_a_vector_blob_round_trips_byte_identical(store):
    fact, _ = store.write(Fact(text="a fact with a vector"))
    payload = bytes(range(256))
    _stub_vector(store.conn, fact.id, blob=payload)

    stored = _vector_rows(store.conn, fact.id)[0]

    assert bytes(stored["vec"]) == payload
    assert stored["dim"] == 4
    assert stored["model"] == "bge-small-en-v1.5"


def test_rewording_a_fact_invalidates_its_vector(store):
    """The vector describes text that no longer exists. Keeping it would rank
    the fact by what it used to say."""
    fact, _ = store.write(Fact(text="the datastore is postgres"))
    _stub_vector(store.conn, fact.id)

    store.revise(fact.id, text="the datastore is sqlite")

    assert _vector_rows(store.conn, fact.id) == []


def test_recall_bumping_use_count_leaves_the_vector_alone(store):
    """The reason the trigger is scoped to `text`. `mark_used` fires on every
    surfaced fact of every recall; an unscoped AFTER UPDATE would discard a
    vector on the most common write in the system and pay to recompute it."""
    fact, _ = store.write(Fact(text="a fact that gets recalled a lot"))
    _stub_vector(store.conn, fact.id)
    before = bytes(_vector_rows(store.conn, fact.id)[0]["vec"])

    store.mark_used([fact.id])
    store.mark_used([fact.id])

    rows = _vector_rows(store.conn, fact.id)
    assert len(rows) == 1
    assert bytes(rows[0]["vec"]) == before
    assert store.get(fact.id).use_count == 2


def test_changing_key_or_tags_alone_leaves_the_vector_alone(store):
    """Only `text` is embedded. `facts_au` re-indexes on key and tags because
    FTS5 indexes those columns; the vector does not."""
    fact, _ = store.write(Fact(text="a fact whose metadata changes"))
    _stub_vector(store.conn, fact.id)

    store.conn.execute("UPDATE facts SET key = 'db.engine' WHERE id = ?", (fact.id,))
    store.conn.execute("UPDATE facts SET tags_csv = 'infra,db' WHERE id = ?", (fact.id,))
    store.conn.commit()

    assert len(_vector_rows(store.conn, fact.id)) == 1


def test_retiring_a_fact_keeps_its_vector(store):
    """`forget` is a status change, and it is reversible. Re-embedding is the
    expensive half, so a retired fact keeps the vector it already paid for."""
    fact, _ = store.write(Fact(text="a fact that gets retired"))
    _stub_vector(store.conn, fact.id)

    store.forget(fact.id)

    assert len(_vector_rows(store.conn, fact.id)) == 1


def test_deleting_a_fact_deletes_its_vector(store):
    fact, _ = store.write(Fact(text="a fact about to be deleted"))
    _stub_vector(store.conn, fact.id)

    store.conn.execute("DELETE FROM facts WHERE id = ?", (fact.id,))
    store.conn.commit()

    assert _vector_rows(store.conn, fact.id) == []


def test_purge_leaves_no_orphan_vectors(store):
    """`purge` is a plain `DELETE FROM facts` after it clears the tables it
    knows about. The trigger is what keeps it correct without purge having to
    learn a new table name."""
    keep, _ = store.write(Fact(text="a fact in another project", scope="repo:other@1234"))
    doomed, _ = store.write(Fact(text="a fact in the purged project", scope="repo:gone@abcd"))
    _stub_vector(store.conn, keep.id)
    _stub_vector(store.conn, doomed.id)

    store.purge(scope="repo:gone@abcd")

    orphans = store.conn.execute(
        "SELECT v.fact_id FROM fact_vectors v"
        " LEFT JOIN facts f ON f.id = v.fact_id WHERE f.id IS NULL"
    ).fetchall()
    assert orphans == []
    assert len(_vector_rows(store.conn, keep.id)) == 1


def test_entities_are_indexed_by_name_alone(store):
    """`UNIQUE(kind, name, scope)` leads with `kind`, so it cannot serve a
    lookup that knows only the name. Query-entity extraction runs on every
    prompt and must not scan the table to conclude a query named nothing."""
    indexes = {r["name"] for r in store.conn.execute(
        "SELECT name FROM sqlite_master WHERE type = 'index' AND tbl_name = 'entities'"
    )}
    assert "idx_entities_name" in indexes

    plan = " ".join(
        str(row["detail"]) for row in store.conn.execute(
            "EXPLAIN QUERY PLAN SELECT id FROM entities WHERE name IN ('app/routes.py')"
        )
    )
    assert "idx_entities_name" in plan
    assert "SCAN entities" not in plan
