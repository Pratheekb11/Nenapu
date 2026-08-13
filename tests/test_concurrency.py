"""Races that actually reproduced, kept as tests.

Every case here was observed failing before the fix — 8 duplicate facts from 8
writers, 8 graders each scoring one recall. Concurrency bugs do not stay fixed
by accident, and this is the one area where "works on my machine" is worthless:
the MCP server, the CLI and a cron job all write to the same file.
"""

import threading

import pytest

from nenapu import connect
from nenapu.models import Fact, Origin, Status
from nenapu.store import Store


@pytest.fixture
def path(tmp_path):
    target = str(tmp_path / "concurrent.db")
    connect(target).close()
    return target


def _hammer(path, work, n=10):
    """Run `work(i)` in n threads released simultaneously."""
    errors: list[str] = []
    barrier = threading.Barrier(n)

    def run(i):
        try:
            store = Store(connect(path))
            barrier.wait()
            work(store, i)
        except Exception as exc:  # noqa: BLE001 - the point is to see them
            errors.append(f"{type(exc).__name__}: {exc}")

    threads = [threading.Thread(target=run, args=(i,)) for i in range(n)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    return errors


def test_concurrent_writers_do_not_duplicate_a_fact(path):
    """Observed: 8 writers produced 8 identical active facts."""
    errors = _hammer(path, lambda s, i: s.write(Fact(text="the API listens on port 8080")))
    assert errors == []

    active = [f for f in Store(connect(path)).list_facts() if "8080" in f.text]
    assert len(active) == 1


def test_the_unique_index_makes_duplicates_unrepresentable(path):
    """Belt and braces: even a caller writing outside a transaction cannot
    leave two active copies behind."""
    import sqlite3

    store = Store(connect(path))
    store.write(Fact(text="only one of me"))
    with pytest.raises(sqlite3.IntegrityError):
        store.conn.execute(
            "INSERT INTO facts(text, kind, scope, origin, confidence, decay_class,"
            " tags_csv, status, created_at, updated_at, verify_status)"
            " VALUES ('only one of me','project','global','user_stated',0.8,'medium',"
            "'','active',1,1,'none')"
        )


def test_concurrent_conflicting_writes_leave_one_winner(path):
    """Ten processes assert ten different values for one key at once."""
    errors = _hammer(
        path,
        lambda s, i: s.write(Fact(text=f"db port is {5000 + i}", key="db.port",
                                  origin=Origin.USER_STATED, confidence=0.9)),
    )
    assert errors == []

    store = Store(connect(path))
    active = [f for f in store.list_facts(status=Status.ACTIVE) if f.key == "db.port"]
    assert len(active) == 1, "two contradictory values both survived as active"
    # Nothing is lost: the losers are superseded history, not deleted rows.
    assert len(store.list_facts(status=None, limit=100)) == 10


def test_a_recall_is_graded_exactly_once(path):
    """Observed: 8 concurrent graders each incremented the counter, so one
    recall cost a fact eight times its due."""
    store = Store(connect(path))
    fact, _ = store.write(Fact(text="a fact worth recalling"))
    recall_id = store.search("recalling", session_id="s")[0][2]["recall_id"]

    won: list[bool] = []
    errors = _hammer(
        path, lambda s, i: won.append(s.ledger.grade(recall_id, "bad", source="human")), n=8
    )

    assert errors == []
    assert sum(won) == 1, "more than one grader claimed the same recall"
    assert Store(connect(path)).get(fact.id).bad_recalls == 1


def test_writes_survive_sustained_contention(path):
    """Twenty writers, distinct facts. None may fail with a lock error."""
    errors = _hammer(path, lambda s, i: s.write(Fact(text=f"distinct fact number {i}")), n=20)
    assert errors == []
    assert len(Store(connect(path)).list_facts(limit=100)) == 20


def test_a_failed_write_leaves_no_partial_state(path):
    """A crash mid-write must not leave a fact without its conflict record."""
    store = Store(connect(path))
    store.write(Fact(text="port is 100", key="p", origin=Origin.USER_STATED, confidence=0.9))

    with pytest.raises(RuntimeError):
        with store.transaction():
            store.conn.execute(
                "INSERT INTO facts(text, kind, scope, origin, confidence, decay_class,"
                " tags_csv, status, created_at, updated_at, verify_status)"
                " VALUES ('half written','project','global','user_stated',0.8,'medium',"
                "'','active',1,1,'none')"
            )
            raise RuntimeError("boom")

    assert not [f for f in Store(connect(path)).list_facts() if f.text == "half written"]
