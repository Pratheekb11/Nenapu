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


# ==========================================================================
# Pre-written for R1 · candidate generation, and E7 · entity-anchored recall.
#
# R1 fixes three faults in `Store.search`, all deterministic and none of them
# waiting on vectors:
#
#   1. the query is pure OR — `_fts_query` joins every term with OR, so a
#      twelve-term query matches any fact sharing one word, and
#      `relevant_memory` feeds it exactly that;
#   2. `key` and `tags_csv` are indexed but never weighted — bm25 treats a
#      match on a fact's key the same as a match on a word in its prose;
#   3. the candidate pool is bm25-ordered only — `ORDER BY rank LIMIT limit*5`
#      then rescored, so confidence can re-rank what lexical search already
#      liked but can never retrieve what it missed.
#
# Plus the silent failure at the recency fallback: when FTS finds nothing the
# code returns arbitrary recent facts scored a flat 0.3 *presented as hits*,
# and in the MCP path those get logged as recalls with a query attached —
# poisoning the exact population G7 uses to measure ranking.
#
# E7 adds a proximity term over the entity graph, held until G9's verdict is
# recorded:
#
#     0.45·lexical + 0.35·confidence + 0.1·usage + 0.10·proximity
#
# with two constraints: the belief layer stays *after* ranking (proximity must
# never promote a falsified fact), and traversal must not cross scope except
# through `global`.
#
# Assumed seams: `store.search(..., near=[...])` carries the anchor E7 scores
# proximity from, `explain` grows `proximity`, and the weights live in one
# named place so a reader can check them against the plan.
# ==========================================================================

e7 = pytest.mark.xfail(strict=True, reason="E7 not implemented yet: remove when it lands")


def _texts(hits):
    return [fact.text for fact, _score, _why in hits]


# ---------- R1 · what the query is allowed to match ----------


def test_a_multi_term_query_does_not_match_on_one_word(store):
    """`" OR ".join(terms)` is why a session about deploys retrieves a fact
    about a coffee machine: they share the word "cache"."""
    store.write(Fact(text="the deploy script lives in tools/ship.sh"))
    store.write(Fact(text="the coffee machine has a cache of beans"))

    hits = store.search("deploy staging cache invalidation policy")

    assert "the coffee machine has a cache of beans" not in _texts(hits)


def test_a_term_too_common_to_discriminate_does_not_drive_the_match(store):
    """A word that appears in most of the store narrows nothing. Matching on
    it is how twelve slots fill with everything."""
    for i in range(30):
        store.write(Fact(text=f"the service number {i} runs somewhere in the cluster"))
    store.write(Fact(text="the badger enclosure needs a new latch"))

    hits = store.search("service badger")

    assert _texts(hits) == ["the badger enclosure needs a new latch"]


def test_a_phrase_query_matches_the_phrase(store):
    """Two words next to each other are a different claim from the same two
    words in one paragraph."""
    store.write(Fact(text="the connection pool is capped at 20"))
    store.write(Fact(text="the connection is over TLS and the pool table is in the office"))

    hits = store.search('"connection pool"')

    assert _texts(hits) == ["the connection pool is capped at 20"]


def test_a_key_match_outranks_an_equal_prose_match(store):
    """197 facts carry dotted keys. A query that matches a fact's key is a
    much stronger signal than one matching a word in its prose, and bm25
    treats the two columns alike."""
    keyed, _ = store.write(Fact(text="the head revision is 7f3a91c",
                                key="backend.alembic.head", kind=Kind.ENVIRONMENT))
    store.write(Fact(text="alembic head output is what the deploy check reads"))

    hits = store.search("backend.alembic.head")

    assert hits[0][0].id == keyed.id
    assert hits[0][2]["key_match"] is True


def test_a_tag_match_is_weighted_like_a_key_match(store):
    """`tags_csv` is in the FTS table and has never been weighted either. A
    fact carrying the tag is a stronger answer than one that happens to say
    the words in a sentence."""
    tagged, _ = store.write(Fact(text="run the migration before the deploy",
                                 tags=["release-checklist"]))
    store.write(Fact(text="the release checklist lives in the wiki, "
                          "and the release checklist is reviewed monthly"))

    hits = store.search("release-checklist")

    assert hits[0][0].id == tagged.id
    assert hits[0][2]["tag_match"] is True


def test_a_believed_fact_outside_the_lexical_pool_can_still_surface(store):
    """The structural fault: the pool is `ORDER BY rank LIMIT limit*5`, so
    confidence is a re-ranker over a lexical pool and never a retriever. A
    fact ranked 51st lexically can never surface however strongly it is
    believed."""
    for i in range(40):
        store.write(Fact(text=f"cache note {i}", kind=Kind.PROJECT,
                         origin=Origin.AGENT_INFERRED, confidence=0.4))
    believed, _ = store.write(Fact(
        text="the cache is invalidated by hand after every release, which the "
             "team has been bitten by twice and now checks on the release call",
        origin=Origin.USER_STATED, confidence=0.95,
    ))

    hits = store.search("cache", limit=5)

    assert believed.id in [fact.id for fact, _s, _w in hits]


def test_an_unmatched_query_does_not_look_like_a_match(store):
    """The silent failure at the fallback: an unmatched query returns
    arbitrary recent facts at a flat 0.3, presented as hits. In the MCP path
    they are logged as recalls with a query attached, which poisons the exact
    population the gate reads to measure ranking."""
    store.write(Fact(text="the deploy script lives in tools/ship.sh"))
    store.write(Fact(text="the badger enclosure needs a new latch"))

    hits = store.search("xyzzy plugh frobnicate")

    assert hits == [] or all(why.get("fallback") for _f, _s, why in hits)


def test_the_fallback_is_labelled_where_a_caller_can_see_it(store):
    """Either answer is defensible — return nothing, or return the fallback
    marked as such. What is not defensible is a recency guess wearing the
    clothes of a lexical hit."""
    store.write(Fact(text="the deploy script lives in tools/ship.sh"))

    hits = store.search("xyzzy plugh frobnicate")

    for _fact, _score, why in hits:
        assert why["fallback"] is True
        assert why["lexical"] == 0.0


def test_an_empty_query_is_not_a_search(store):
    store.write(Fact(text="the deploy script lives in tools/ship.sh"))

    hits = store.search("   ")

    assert hits == [] or all(why.get("fallback") for _f, _s, why in hits)


def test_scoping_still_holds_over_the_wider_pool(store):
    """R1 widens what can be retrieved, and the scope filter is what keeps
    "right fact, wrong project" fixed while it does."""
    store.write(Fact(text="this repo runs on port 8080", scope="repo:here@aaaaaaaa"))
    store.write(Fact(text="that repo runs on port 5544", scope="repo:there@bbbbbbbb"))

    hits = store.search("port", scope=["global", "repo:here@aaaaaaaa"])

    assert _texts(hits) == ["this repo runs on port 8080"]


def test_a_retired_fact_is_still_not_retrievable(store):
    """Whatever the pool is built from, status is what decides membership."""
    fact, _ = store.write(Fact(text="the deploy script lives in tools/ship.sh"))
    store.forget(fact.id)

    assert store.search("deploy script") == []


# ---------- E7 · proximity, and what it is not allowed to do ----------


def _entity_fact(store, *, text, path, scope="global", role="subject"):
    from nenapu.entities import EntityGraph

    fact, _ = store.write(Fact(text=text, scope=scope))
    graph = EntityGraph(store.conn)
    entity = graph.upsert(kind="file", name=path, scope=scope)
    graph.attach(fact.id, entity.id, role=role, source="path")
    return fact, entity


@e7
def test_the_scoring_weights_are_the_ones_the_plan_named(store):
    """Weights on the hottest path in the system, written down in one place so
    a change to them is a change someone has to make on purpose."""
    from nenapu.store import SEARCH_WEIGHTS

    assert SEARCH_WEIGHTS == {
        "lexical": 0.45, "confidence": 0.35, "usage": 0.1, "proximity": 0.10,
    }
    assert sum(SEARCH_WEIGHTS.values()) == pytest.approx(1.0)


@e7
def test_a_fact_about_a_nearby_entity_outranks_an_equal_one_far_away(store):
    """R4 anchors injection on cwd, branch and recently edited files. E7
    extends that anchor through the entity graph, so a fact about the file
    being worked on beats an identically believed fact about another."""
    near, _ = _entity_fact(store, text="the handler validates the token twice",
                           path="services/auth/routes.py")
    far, _ = _entity_fact(store, text="the handler validates the token twice over",
                          path="services/billing/invoices.py")

    hits = store.search("handler validates token", near=["services/auth/routes.py"])
    ranked = [fact.id for fact, _s, _w in hits]

    assert ranked.index(near.id) < ranked.index(far.id)


@e7
def test_proximity_is_reported_in_the_explanation(store):
    """`explain` is how a user is shown why a memory surfaced. A term that
    moves the ranking and does not appear there is a silent reranker."""
    _entity_fact(store, text="the handler validates the token twice",
                 path="services/auth/routes.py")

    hits = store.search("handler", near=["services/auth/routes.py"])

    assert hits[0][2]["proximity"] > 0


@e7
def test_proximity_reaches_one_hop_out_with_decay(store):
    """Depth 2 with per-hop decay: a fact about a file that is touched with
    the anchor is nearer than an unrelated one, and further than the anchor's
    own facts."""
    from nenapu.entities import EntityGraph

    anchor, anchor_entity = _entity_fact(store, text="the login handler is here",
                                         path="services/auth/routes.py")
    neighbour, neighbour_entity = _entity_fact(store, text="the login handler is tested here",
                                               path="services/auth/test_routes.py")
    stranger, _ = _entity_fact(store, text="the login handler is unrelated here",
                               path="tools/unrelated.py")
    EntityGraph(store.conn).link(anchor_entity.id, neighbour_entity.id,
                                 kind="touched_with", source="observed")

    hits = store.search("login handler", near=["services/auth/routes.py"])
    by_id = {fact.id: why["proximity"] for fact, _s, why in hits}

    assert by_id[anchor.id] > by_id[neighbour.id] > by_id[stranger.id]


@e7
def test_traversal_does_not_cross_scope(store):
    """Scoping already had to fix "right fact, wrong project" once. A graph
    walk that ignores scope recreates it one layer down."""
    _entity_fact(store, text="the handler validates the token", path="app/routes.py",
                 scope="repo:here@aaaaaaaa")
    _entity_fact(store, text="the handler validates the token elsewhere",
                 path="app/routes.py", scope="repo:there@bbbbbbbb")

    hits = store.search("handler validates token", scope=["global", "repo:here@aaaaaaaa"],
                        near=["app/routes.py"])

    assert all(fact.scope in ("global", "repo:here@aaaaaaaa") for fact, _s, _w in hits)


@e7
def test_a_global_entity_is_still_reachable_from_a_project(store):
    """The one crossing that is allowed, because global facts are meant to
    surface everywhere."""
    everywhere, _ = _entity_fact(store, text="commits never carry a co-author trailer",
                                 path="tools/commit-check.sh", scope="global")

    hits = store.search("commits co-author trailer",
                        scope=["global", "repo:here@aaaaaaaa"],
                        near=["tools/commit-check.sh"])

    assert everywhere.id in [fact.id for fact, _s, _w in hits]


@e7
def test_proximity_never_promotes_a_falsified_fact(store):
    """The belief layer stays *after* ranking, as filter and warning, exactly
    as `recall_context` does today."""
    suspect, _ = _entity_fact(store, text="the handler validates the token twice",
                              path="services/auth/routes.py")
    store.set_status(suspect.id, Status.SUSPECT)
    sound, _ = _entity_fact(store, text="the handler validates the token once",
                            path="tools/unrelated.py")

    hits = store.search("handler validates token", near=["services/auth/routes.py"],
                        min_confidence=0.4)
    ranked = [fact.id for fact, _s, _w in hits]

    assert suspect.id not in ranked or ranked.index(sound.id) < ranked.index(suspect.id)


@e7
def test_a_suspect_fact_keeps_its_warning_however_near_it_is(store):
    suspect, _ = _entity_fact(store, text="the handler validates the token twice",
                              path="services/auth/routes.py")
    store.set_status(suspect.id, Status.SUSPECT)

    hits = store.search("handler validates token", near=["services/auth/routes.py"])

    assert any(fact.id == suspect.id and why.get("suspect_reason") is not None
               for fact, _s, why in hits) or hits == []


def test_search_without_an_anchor_is_unchanged(store):
    """With no entity data and no anchor, scoring degrades to what it does
    today, which is what keeps every existing test in this file true."""
    store.write(Fact(text="The API runs on port 8080", kind=Kind.ENVIRONMENT,
                     key="api.port"))

    hits = store.search("port")

    assert hits and "8080" in hits[0][0].text
