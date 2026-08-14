import os
import stat
import time

import pytest

from nenapu import connect
from nenapu.models import Decay, Fact, Kind, Origin, Status, VerifyStatus
from nenapu.store import DAY, Store, effective_confidence, looks_contradictory


@pytest.fixture
def store():
    return Store(connect(":memory:"))


def test_write_and_recall(store):
    store.write(Fact(text="The API runs on port 8080", kind=Kind.ENVIRONMENT, key="api.port"))
    hits = store.search("port")
    assert hits and "8080" in hits[0][0].text
    assert "confidence" in hits[0][2]


def test_volatile_facts_decay_faster_than_slow(store):
    old = time.time() - 60 * DAY
    volatile = Fact(text="branch is release-3", decay_class=Decay.VOLATILE, created_at=old,
                    last_verified_at=old)
    slow = Fact(text="we use hexagonal architecture", decay_class=Decay.SLOW, created_at=old,
                last_verified_at=old)
    assert effective_confidence(volatile) < effective_confidence(slow)


def test_immutable_never_decays(store):
    old = time.time() - 3000 * DAY
    f = Fact(text="user prefers tabs", decay_class=Decay.IMMUTABLE, origin=Origin.USER_STATED,
             confidence=0.9, created_at=old, last_verified_at=old)
    assert effective_confidence(f) == pytest.approx(0.9, abs=0.01)


def test_user_stated_outranks_agent_inferred(store):
    stated = Fact(text="x", origin=Origin.USER_STATED, confidence=0.7)
    guessed = Fact(text="x", origin=Origin.AGENT_INFERRED, confidence=0.7)
    assert effective_confidence(stated) > effective_confidence(guessed)


def test_contradiction_detection():
    assert looks_contradictory("port is 8080", "port is 9090")[0]
    assert looks_contradictory("cache enabled", "cache is disabled")[0]
    assert not looks_contradictory("port is 8080", "the port is 8080")[0]


def test_stronger_new_fact_supersedes_old(store):
    old, _ = store.write(Fact(text="db port is 5432", key="db.port", origin=Origin.AGENT_INFERRED,
                              confidence=0.6))
    new, conflicts = store.write(Fact(text="db port is 6543", key="db.port",
                                      origin=Origin.USER_STATED, confidence=0.9))
    assert conflicts and conflicts[0].resolution == "superseded"
    assert store.get(old.id).status == Status.SUPERSEDED
    assert store.get(old.id).superseded_by_id == new.id
    assert store.get(new.id).status == Status.ACTIVE


def test_weaker_new_fact_is_disputed_not_applied(store):
    strong, _ = store.write(Fact(text="db port is 5432", key="db.port",
                                 origin=Origin.USER_STATED, confidence=0.95))
    weak, conflicts = store.write(Fact(text="db port is 9999", key="db.port",
                                       origin=Origin.AGENT_INFERRED, confidence=0.4))
    assert conflicts[0].resolution == "disputed"
    assert store.get(weak.id).status == Status.DISPUTED
    assert store.get(strong.id).status == Status.ACTIVE


def test_reassertion_refreshes_instead_of_duplicating(store):
    old = time.time() - 200 * DAY
    first, _ = store.write(Fact(text="we deploy on Fridays", created_at=old, last_verified_at=old))
    again, conflicts = store.write(Fact(text="we deploy on Fridays"))
    assert again.id == first.id
    assert not conflicts
    assert again.last_verified_at > old
    assert len(store.list_facts()) == 1


def test_recall_prefers_fresh_over_stale(store):
    old = time.time() - 300 * DAY
    store.write(Fact(text="deploy target is staging cluster", decay_class=Decay.VOLATILE,
                     created_at=old, last_verified_at=old))
    store.write(Fact(text="deploy target is prod cluster", decay_class=Decay.VOLATILE))
    hits = store.search("deploy target cluster", limit=2)
    assert "prod" in hits[0][0].text


def test_forget_removes_from_recall(store):
    fact, _ = store.write(Fact(text="temporary workaround for the parser"))
    store.forget(fact.id)
    assert not [h for h in store.search("parser workaround") if h[0].id == fact.id]


def test_min_confidence_filters_stale(store):
    old = time.time() - 400 * DAY
    store.write(Fact(text="redis on 6379", decay_class=Decay.VOLATILE, created_at=old,
                     last_verified_at=old))
    assert store.search("redis") != []
    assert store.search("redis", min_confidence=0.5) == []


def test_stats(store):
    store.write(Fact(text="queue backend is sqs", key="queue.backend", confidence=0.9,
                     origin=Origin.USER_STATED))
    store.write(Fact(text="queue backend is rabbitmq", key="queue.backend", confidence=0.2))
    s = store.stats()
    assert s["active"] >= 1 and s["conflicts"] == 1


def test_contentless_facts_are_not_forced_into_a_conflict(store):
    # "a" carries no claim; comparing it to anything is meaningless.
    assert not looks_contradictory("a", "b totally different words here")[0]


def test_disputed_facts_are_downweighted_but_still_recallable(store):
    store.write(Fact(text="cache ttl is 60 seconds", key="cache.ttl",
                     origin=Origin.USER_STATED, confidence=0.95))
    weak, _ = store.write(Fact(text="cache ttl is 3600 seconds", key="cache.ttl",
                               origin=Origin.AGENT_INFERRED, confidence=0.4))
    hits = {f.id: (s, why) for f, s, why in store.search("cache ttl", limit=5)}
    assert weak.id in hits
    assert store.get(weak.id).status == Status.DISPUTED
    assert hits[weak.id][1]["confidence"] < 0.4


def test_journal_records_every_action(store):
    fact, _ = store.write(Fact(text="something worth auditing"))
    store.forget(fact.id)
    actions = [r["action"] for r in store.conn.execute("SELECT action FROM journal")]
    assert "write" in actions and "status" in actions


def test_enums_render_as_their_values(store):
    from nenapu.models import Kind, VerifyStatus
    assert f"{VerifyStatus.FAIL:>6}" == "  fail"
    assert str(Kind.USER) == "user"
    fact, _ = store.write(Fact(text="x", kind=Kind.USER))
    row = store.conn.execute("SELECT kind FROM facts WHERE id=?", (fact.id,)).fetchone()
    assert row["kind"] == "user"


def test_shared_key_makes_different_values_conflict_by_default(store):
    # High word overlap must not excuse a different value under the same key.
    assert looks_contradictory("cache backend is redis", "cache backend is memcached")[0]
    assert looks_contradictory("the region is us-east-1", "the region is eu-west-2")[0]


def test_rephrasing_and_elaboration_are_not_conflicts(store):
    assert not looks_contradictory("port is 8080", "the port is 8080")[0]
    assert not looks_contradictory("we deploy on Tuesday",
                                   "we deploy on Tuesday, after standup")[0]


# ---------- who else can read your memory ----------


def test_a_new_store_is_owner_only(tmp_path):
    """The file holds facts extracted from private sessions. The process umask
    left it 0644 inside a 0755 directory, so on a shared machine every account
    on the box could read them."""
    db = tmp_path / "nest" / "nenapu.db"
    connect(str(db)).close()

    assert stat.S_IMODE(db.stat().st_mode) == 0o600
    assert stat.S_IMODE(db.parent.stat().st_mode) == 0o700


def test_an_existing_loose_store_is_tightened_on_open(tmp_path):
    """Anyone already running this has a 0644 store. Fixing only new installs
    would leave exactly the people who trusted it earliest exposed."""
    db = tmp_path / "nenapu.db"
    connect(str(db)).close()
    os.chmod(db, 0o644)
    os.chmod(tmp_path, 0o755)

    connect(str(db)).close()

    assert stat.S_IMODE(db.stat().st_mode) == 0o600
    assert stat.S_IMODE(tmp_path.stat().st_mode) == 0o700


def test_the_wal_sidecars_are_owner_only_too(tmp_path):
    """`-wal` carries the same content and is created by the driver, not us."""
    db = tmp_path / "nenapu.db"
    conn = connect(str(db))
    Store(conn).write(Fact(text="something private", kind=Kind.PROJECT,
                           origin=Origin.USER_STATED))

    wal = db.with_name(db.name + "-wal")
    assert wal.exists(), "WAL is on; this test is checking the wrong thing"
    assert stat.S_IMODE(wal.stat().st_mode) == 0o600


def test_a_filesystem_that_cannot_chmod_still_opens(tmp_path, monkeypatch):
    """A share or a Windows volume must not cost someone their memory."""
    def _refuse(*args, **kwargs):
        raise OSError("this filesystem does not do modes")

    monkeypatch.setattr(os, "chmod", _refuse)
    conn = connect(str(tmp_path / "nenapu.db"))

    assert Store(conn).search("anything") == []
