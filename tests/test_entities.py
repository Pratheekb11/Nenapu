"""The entity tier: one graph out of two node types, not two graphs side by side.

Requirement (plan "join the entity graph to the belief graph", tasks E1-E10).
Three new tables, of which `fact_entities` is the whole integration:

    BELIEF (built)          BRIDGE (new)              ENTITY (new)
    fact --derived_from-->  fact_entities             entity --contains--> entity
         cascades              fact_id                       --touched_with-->
                               entity_id                     --changed_in-->
                               role: subject|mentions        --alias_of-->

`role='subject'` is load-bearing: a fact *about* a deleted file dies with it,
a fact that merely *mentions* it does not.

Tasks pinned in this file:

* **E1** schema and the `EntityGraph` module — three `CREATE TABLE IF NOT
  EXISTS` tables, `SCHEMA_VERSION` bumped, traversal depth-capped and
  cycle-safe, all writes through `db.transaction`.
* **E2** deterministic build from the activity ledger — zero model calls, the
  same principle the ledger already runs on. 1470 `file_events` rows over 647
  distinct paths, 213 sessions, 5 commits, all already stored.
* **E3** dotted keys become subjects — 197 facts already carry them.
* **E4** text matching, against *known* names only, so this step is incapable
  of inventing an entity.
* **E5** alias resolution — `services/auth`, `auth service`, `AuthService`.
* **E6** entity death cascades into belief (the belief half is pinned in
  tests/test_graph.py; the entity and capture halves are here).
* **E8** model-extracted entities, riding the same Stop-hook call.
* **E9** edge weights learn from grades.

E10 is pinned in tests/test_graph.py and tests/test_mcp.py, E7 in
tests/test_store.py and tests/test_project_injection.py.

Assumed seam — proposed by the plan, not yet in the codebase::

    nenapu.entities.EntityGraph(conn)
        upsert(*, kind, name, scope='global', at=None) -> Entity
        get(entity_id) / find(*, kind=None, name, scope='global')
        link(src_id, dst_id, *, kind, source='observed', weight=None)
        close_edge(src_id, dst_id, kind)        # valid_to, never a delete
        neighbours(entity_id, *, depth=1) -> list[(entity_id, hops)]
        attach(fact_id, entity_id, *, role, source)
        subjects_of(entity_id) -> [fact_id]
        entities_for_fact(fact_id) -> [(Entity, role)]
        mark_gone(entity_id, *, reason) -> [fact_id]
        mark_alive(entity_id) -> [fact_id]

    nenapu.entities.build_from_activity(store, *, scope=None) -> dict
    nenapu.entities.subjects_from_keys(store) -> int
    nenapu.entities.mentions_from_text(store, *, scope=None) -> int
    nenapu.entities.resolve_aliases(store, *, scope=None, backend=None) -> int
    nenapu.entities.reward_edges_for_grades(store, *, session_id=None) -> int
    nenapu.entities.MIN_TOUCHED_OBSERVATIONS, MAX_EDGE_WEIGHT

Every test here describes behaviour that does not exist yet, so the whole
module is strict-xfail by task. Remove a marker when its task lands: a marker
that outlives its implementation fails the suite, which is the point.
"""

import json
import os
import subprocess
import sys
import time

import pytest

from nenapu import connect
from nenapu.models import Fact, Kind, Origin, Status, now
from nenapu.store import Store

e1 = pytest.mark.xfail(strict=True, reason="E1 not implemented yet: remove when it lands")
e2 = pytest.mark.xfail(strict=True, reason="E2 not implemented yet: remove when it lands")
e3 = pytest.mark.xfail(strict=True, reason="E3 not implemented yet: remove when it lands")
e4 = pytest.mark.xfail(strict=True, reason="E4 not implemented yet: remove when it lands")
e6 = pytest.mark.xfail(strict=True, reason="E6 not implemented yet: remove when it lands")
e8 = pytest.mark.xfail(strict=True, reason="E8 not implemented yet: remove when it lands")
e9 = pytest.mark.xfail(strict=True, reason="E9 not implemented yet: remove when it lands")

DAY = 86400.0
SCOPE = "repo:backend@aaaaaaaa"
OTHER = "repo:portfolio@bbbbbbbb"


@pytest.fixture
def store():
    return Store(connect(":memory:"))


@pytest.fixture
def ledger(store):
    from nenapu.activity import ActivityLedger

    return ActivityLedger(store.conn)


@pytest.fixture
def graph(store):
    from nenapu.entities import EntityGraph

    return EntityGraph(store.conn)


def _session(ledger, *, scope=SCOPE, paths=(), ago=DAY, agent="claude-code",
             commit_sha=None, commit_files=()):
    started = now() - ago
    session_id = ledger.start_session(agent=agent, project_scope=scope, cwd="/repo",
                                      git_branch="main", started_at=started)
    for path in paths:
        ledger.record_file_event(session_id, path=path, op="edited", tool="Edit",
                                 at=started + 60)
    if commit_sha:
        ledger.record_commit(session_id, sha=commit_sha, subject="a commit",
                             files_changed=list(commit_files or paths), at=started + 120)
    ledger.end_session(session_id, ended_at=started + 180)
    return session_id


def _names(graph, kind=None, scope=SCOPE):
    rows = graph.conn.execute(
        "SELECT kind, name FROM entities WHERE scope = ?", (scope,)
    ).fetchall()
    return {r["name"] for r in rows if kind is None or r["kind"] == kind}


def _edge(conn, src_id, dst_id, kind):
    return conn.execute(
        "SELECT * FROM entity_edges WHERE src_id = ? AND dst_id = ? AND kind = ?",
        (src_id, dst_id, kind),
    ).fetchone()


# ==========================================================================
# E1 · schema and the entity graph module
# ==========================================================================


def test_the_three_tables_exist_on_a_fresh_store():
    conn = connect(":memory:")
    tables = {r["name"] for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type = 'table'"
    )}

    assert {"entities", "entity_edges", "fact_entities"} <= tables


def test_the_columns_are_the_ones_the_plan_named():
    conn = connect(":memory:")

    def columns(table):
        return {r["name"] for r in conn.execute(f"PRAGMA table_info({table})")}

    assert columns("entities") >= {"id", "kind", "name", "scope", "status",
                                   "first_seen", "last_seen", "mentions"}
    assert columns("entity_edges") >= {"id", "src_id", "dst_id", "kind", "source",
                                       "weight", "observations", "valid_from", "valid_to"}
    assert columns("fact_entities") >= {"fact_id", "entity_id", "role", "source"}


def test_an_existing_store_gains_the_tables_on_reconnect(tmp_path):
    """`connect()` runs `executescript(SCHEMA)` on every open and every table
    is `CREATE TABLE IF NOT EXISTS`, so an existing store migrates with no
    extra work — `_ADDED_COLUMNS` is only for columns on tables that already
    exist."""
    path = tmp_path / "old.db"
    old = Store(connect(str(path)))
    fact, _ = old.write(Fact(text="a fact written before the entity tier"))
    old.conn.close()

    reopened = Store(connect(str(path)))
    tables = {r["name"] for r in reopened.conn.execute(
        "SELECT name FROM sqlite_master WHERE type = 'table'"
    )}

    assert {"entities", "entity_edges", "fact_entities"} <= tables
    assert reopened.get(fact.id).text == "a fact written before the entity tier"


def test_the_schema_version_is_bumped():
    from nenapu.db import SCHEMA_VERSION

    assert SCHEMA_VERSION > 7


def test_the_indexes_the_traversal_depends_on_exist():
    conn = connect(":memory:")
    indexed = {r["name"] for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type = 'index'"
    )}

    assert any("entity_edges" in name and "src" in name for name in indexed)
    assert any("entity_edges" in name and "dst" in name for name in indexed)
    assert any("fact_entities" in name for name in indexed)


def test_upsert_is_idempotent_on_kind_name_and_scope(graph):
    first = graph.upsert(kind="file", name="app/routes.py", scope=SCOPE)
    second = graph.upsert(kind="file", name="app/routes.py", scope=SCOPE)

    assert first.id == second.id
    assert len(_names(graph, kind="file")) == 1


def test_the_same_name_in_another_scope_is_another_entity(graph):
    here = graph.upsert(kind="file", name="app/routes.py", scope=SCOPE)
    there = graph.upsert(kind="file", name="app/routes.py", scope=OTHER)

    assert here.id != there.id


def test_upsert_moves_last_seen_and_counts_mentions(graph):
    graph.upsert(kind="file", name="app/routes.py", scope=SCOPE)
    again = graph.upsert(kind="file", name="app/routes.py", scope=SCOPE)

    assert again.mentions >= 2
    assert again.last_seen >= again.first_seen


def test_self_links_and_duplicate_edges_are_refused(graph):
    """The same two rejections `Graph.link` already makes, for the same
    reasons: a self-link is not a relation and a duplicate is not news."""
    a = graph.upsert(kind="file", name="a.py", scope=SCOPE)
    b = graph.upsert(kind="file", name="b.py", scope=SCOPE)

    assert graph.link(a.id, a.id, kind="touched_with") is None
    assert graph.link(a.id, b.id, kind="touched_with") is not None
    assert graph.link(a.id, b.id, kind="touched_with") is None


def test_neighbours_respects_its_depth_cap(graph):
    a = graph.upsert(kind="file", name="a.py", scope=SCOPE)
    b = graph.upsert(kind="file", name="b.py", scope=SCOPE)
    c = graph.upsert(kind="file", name="c.py", scope=SCOPE)
    graph.link(a.id, b.id, kind="touched_with")
    graph.link(b.id, c.id, kind="touched_with")

    reached = {entity_id for entity_id, _hops in graph.neighbours(a.id, depth=1)}

    assert b.id in reached
    assert c.id not in reached


def test_neighbours_terminates_on_a_cycle(graph):
    """An unbounded walk on a cyclic graph never returns, which is why
    `Graph.descendants` is a `deque` walk with a seen-set. Same shape here."""
    a = graph.upsert(kind="file", name="a.py", scope=SCOPE)
    b = graph.upsert(kind="file", name="b.py", scope=SCOPE)
    graph.link(a.id, b.id, kind="touched_with")
    graph.link(b.id, a.id, kind="touched_with")

    reached = {entity_id for entity_id, _hops in graph.neighbours(a.id, depth=6)}

    assert b.id in reached


def test_neighbours_reports_how_far_away_each_one_is(graph):
    """E7 decays proximity per hop, so the walk has to say how many hops it
    took rather than only which nodes it reached."""
    a = graph.upsert(kind="file", name="a.py", scope=SCOPE)
    b = graph.upsert(kind="file", name="b.py", scope=SCOPE)
    c = graph.upsert(kind="file", name="c.py", scope=SCOPE)
    graph.link(a.id, b.id, kind="touched_with")
    graph.link(b.id, c.id, kind="touched_with")

    hops = dict(graph.neighbours(a.id, depth=2))

    assert hops[b.id] == 1
    assert hops[c.id] == 2


def test_a_move_closes_an_edge_rather_than_deleting_it(graph):
    """`valid_to` is what makes the graph a record of what was true when,
    rather than only of what is true now."""
    a = graph.upsert(kind="file", name="a.py", scope=SCOPE)
    b = graph.upsert(kind="dir", name="app", scope=SCOPE)
    graph.link(b.id, a.id, kind="contains")

    graph.close_edge(b.id, a.id, "contains")

    row = _edge(graph.conn, b.id, a.id, "contains")
    assert row is not None
    assert row["valid_to"] is not None


def test_a_closed_edge_is_not_traversed(graph):
    a = graph.upsert(kind="file", name="a.py", scope=SCOPE)
    b = graph.upsert(kind="file", name="b.py", scope=SCOPE)
    graph.link(a.id, b.id, kind="touched_with")
    graph.close_edge(a.id, b.id, "touched_with")

    assert graph.neighbours(a.id, depth=2) == []


def test_subjects_of_returns_only_the_subject_facts(store, graph):
    about, _ = store.write(Fact(text="app/routes.py owns the login handler"))
    mentions, _ = store.write(Fact(text="the handler used to live in app/routes.py"))
    entity = graph.upsert(kind="file", name="app/routes.py", scope="global")
    graph.attach(about.id, entity.id, role="subject", source="key")
    graph.attach(mentions.id, entity.id, role="mentions", source="path")

    assert graph.subjects_of(entity.id) == [about.id]


def test_a_fact_can_be_looked_up_from_either_side(store, graph):
    fact, _ = store.write(Fact(text="app/routes.py owns the login handler"))
    entity = graph.upsert(kind="file", name="app/routes.py", scope="global")
    graph.attach(fact.id, entity.id, role="subject", source="key")

    attached = graph.entities_for_fact(fact.id)

    assert [(e.name, role) for e, role in attached] == [("app/routes.py", "subject")]


def test_attaching_the_same_pair_twice_is_one_row(store, graph):
    fact, _ = store.write(Fact(text="app/routes.py owns the login handler"))
    entity = graph.upsert(kind="file", name="app/routes.py", scope="global")
    graph.attach(fact.id, entity.id, role="subject", source="key")
    graph.attach(fact.id, entity.id, role="subject", source="key")

    rows = graph.conn.execute("SELECT COUNT(*) c FROM fact_entities").fetchone()["c"]
    assert rows == 1


def test_entity_writes_join_an_outer_transaction(store, graph):
    """All writes go through `db.transaction`, so nested calls across Store,
    Graph, Ledger and EntityGraph join one outermost transaction — which is
    what tests/test_concurrency.py exists to catch."""
    with store.transaction():
        entity = graph.upsert(kind="file", name="app/routes.py", scope=SCOPE)
        fact, _ = store.write(Fact(text="app/routes.py owns the login handler"))
        graph.attach(fact.id, entity.id, role="subject", source="key")

    assert graph.subjects_of(entity.id) == [fact.id]


def test_the_models_carry_the_new_types():
    """`Entity` and `EntityEdge` dataclasses, the enums as `_StrEnum`
    subclasses following `Kind` / `EdgeKind` / `Status`, and `row_to_entity`
    beside the existing `row_to_*` helpers."""
    from nenapu.models import Entity, EntityEdge, EntityEdgeKind, EntityKind, EntityStatus

    assert str(EntityKind.FILE) == "file"
    assert str(EntityStatus.GONE) == "gone"
    assert str(EntityEdgeKind.CONTAINS) == "contains"
    assert Entity(kind="file", name="a.py").scope == "global"
    assert EntityEdge(src_id=1, dst_id=2, kind="contains").weight == 1.0


def test_row_to_entity_reads_a_row_the_way_the_others_do():
    from nenapu.models import row_to_entity

    conn = connect(":memory:")
    conn.execute(
        "INSERT INTO entities(kind, name, scope, status, first_seen, last_seen)"
        " VALUES ('file','a.py','global','alive', 1.0, 2.0)"
    )
    row = conn.execute("SELECT * FROM entities").fetchone()

    assert row_to_entity(row).name == "a.py"


# ==========================================================================
# E2 · deterministic build from the activity ledger
# ==========================================================================


def test_the_build_calls_no_model(store, ledger, monkeypatch):
    """The same principle the activity ledger already runs on: "Deterministic
    — filled from git and transcript tool calls, never a model"."""
    from nenapu import entities

    _session(ledger, paths=["backend/app/routes.py"])
    called = []
    monkeypatch.setattr("nenapu.llm.structured", lambda *a, **k: called.append(a))

    entities.build_from_activity(store)

    assert called == []


def test_every_edited_path_becomes_a_file_entity(store, ledger, graph):
    from nenapu import entities

    _session(ledger, paths=["backend/app/routes.py", "backend/app/models.py"])

    entities.build_from_activity(store)

    assert _names(graph, kind="file") == {"backend/app/routes.py", "backend/app/models.py"}


def test_directories_contain_their_files(store, ledger, graph):
    from nenapu import entities

    _session(ledger, paths=["backend/app/routes.py"])

    entities.build_from_activity(store)

    parent = graph.find(kind="dir", name="backend/app", scope=SCOPE)
    child = graph.find(kind="file", name="backend/app/routes.py", scope=SCOPE)
    assert _edge(graph.conn, parent.id, child.id, "contains") is not None


def test_the_directory_chain_is_depth_capped(store, ledger, graph):
    """One node per path segment on a deep monorepo path is a graph nobody
    asked for, and a traversal that spends its depth budget walking upward."""
    from nenapu import entities
    from nenapu.entities import MAX_DIR_DEPTH

    _session(ledger, paths=["a/b/c/d/e/f/g/deep.py"])

    entities.build_from_activity(store)

    assert len(_names(graph, kind="dir")) <= MAX_DIR_DEPTH


def test_one_co_edit_is_not_a_relation(store, ledger, graph):
    """The exact trap `infer_edges_for` fell into, one layer up: two files
    edited together once is a coincidence."""
    from nenapu import entities

    _session(ledger, paths=["a.py", "b.py"])

    entities.build_from_activity(store)

    a = graph.find(kind="file", name="a.py", scope=SCOPE)
    b = graph.find(kind="file", name="b.py", scope=SCOPE)
    assert _edge(graph.conn, a.id, b.id, "touched_with") is None


def test_three_co_edits_are(store, ledger, graph):
    from nenapu import entities
    from nenapu.entities import MIN_TOUCHED_OBSERVATIONS

    for i in range(MIN_TOUCHED_OBSERVATIONS):
        _session(ledger, paths=["a.py", "b.py"], ago=DAY * (i + 1))

    entities.build_from_activity(store)

    a = graph.find(kind="file", name="a.py", scope=SCOPE)
    b = graph.find(kind="file", name="b.py", scope=SCOPE)
    edge = _edge(graph.conn, a.id, b.id, "touched_with")
    assert edge is not None
    assert edge["observations"] >= MIN_TOUCHED_OBSERVATIONS


def test_an_old_relation_weighs_less_than_a_recent_one(store, ledger, graph):
    """Decay on the `Decay.MEDIUM` 90-day half-life already defined in
    `HALF_LIFE_DAYS`, so a pairing from last year stops steering retrieval."""
    from nenapu import entities

    for i in range(3):
        _session(ledger, paths=["old_a.py", "old_b.py"], ago=DAY * (300 + i))
        _session(ledger, paths=["new_a.py", "new_b.py"], ago=DAY * (i + 1))

    entities.build_from_activity(store)

    old_a = graph.find(kind="file", name="old_a.py", scope=SCOPE)
    old_b = graph.find(kind="file", name="old_b.py", scope=SCOPE)
    new_a = graph.find(kind="file", name="new_a.py", scope=SCOPE)
    new_b = graph.find(kind="file", name="new_b.py", scope=SCOPE)
    assert (_edge(graph.conn, old_a.id, old_b.id, "touched_with")["weight"]
            < _edge(graph.conn, new_a.id, new_b.id, "touched_with")["weight"])


def test_a_commit_becomes_an_entity_with_changed_in_edges(store, ledger, graph):
    from nenapu import entities

    _session(ledger, paths=["backend/app/routes.py"], commit_sha="c7f1a9d4e2")

    entities.build_from_activity(store)

    commit = graph.find(kind="commit", name="c7f1a9d4e2", scope=SCOPE)
    changed = graph.find(kind="file", name="backend/app/routes.py", scope=SCOPE)
    assert commit is not None
    assert _edge(graph.conn, changed.id, commit.id, "changed_in") is not None


def test_each_project_scope_becomes_a_repo_entity(store, ledger, graph):
    from nenapu import entities

    _session(ledger, paths=["a.py"], scope=SCOPE)
    _session(ledger, paths=["b.py"], scope=OTHER)

    entities.build_from_activity(store)

    assert _names(graph, kind="repo", scope=SCOPE) == {SCOPE}
    assert _names(graph, kind="repo", scope=OTHER) == {OTHER}


def test_entities_are_partitioned_by_scope(store, ledger, graph):
    """Two repos with a `README.md` each are two files, not one."""
    from nenapu import entities

    _session(ledger, paths=["README.md"], scope=SCOPE)
    _session(ledger, paths=["README.md"], scope=OTHER)

    entities.build_from_activity(store)

    assert graph.find(kind="file", name="README.md", scope=SCOPE).id != \
        graph.find(kind="file", name="README.md", scope=OTHER).id


def test_a_rebuild_changes_nothing_the_second_time(store, ledger, graph):
    from nenapu import entities

    for i in range(3):
        _session(ledger, paths=["a.py", "b.py"], ago=DAY * (i + 1), commit_sha=f"sha{i}")

    entities.build_from_activity(store)
    before = [dict(r) for r in graph.conn.execute("SELECT * FROM entities ORDER BY id")]
    edges_before = [dict(r) for r in graph.conn.execute(
        "SELECT src_id, dst_id, kind, observations FROM entity_edges ORDER BY id")]

    entities.build_from_activity(store)

    after = [dict(r) for r in graph.conn.execute("SELECT * FROM entities ORDER BY id")]
    edges_after = [dict(r) for r in graph.conn.execute(
        "SELECT src_id, dst_id, kind, observations FROM entity_edges ORDER BY id")]
    assert [r["name"] for r in after] == [r["name"] for r in before]
    assert edges_after == edges_before


def test_a_live_session_builds_entities_as_it_lands(store, tmp_path, monkeypatch):
    """The incremental hook in `capture_session`, after the
    `ledger.record_file_event` loop, so live sessions do not wait for an
    offline rebuild."""
    from nenapu.activity import ActivityLedger
    from nenapu.capture import capture_session
    from nenapu.entities import EntityGraph

    transcript = tmp_path / "t.jsonl"
    transcript.write_text("\n".join([
        json.dumps({"type": "user", "sessionId": "s-live", "cwd": str(tmp_path),
                    "message": {"role": "user", "content": "edit it"}}),
        json.dumps({"type": "assistant", "sessionId": "s-live",
                    "message": {"role": "assistant", "content": [
                        {"type": "tool_use", "id": "t1", "name": "Edit",
                         "input": {"file_path": str(tmp_path / "app.py")}},
                    ]}}),
    ]))

    capture_session(ActivityLedger(store.conn), transcript, agent="claude-code",
                    cwd=str(tmp_path))

    names = {r["name"] for r in store.conn.execute("SELECT name FROM entities")}
    assert any("app.py" in name for name in names)


def test_the_rebuild_is_reachable_from_the_command_line(tmp_path):
    db = tmp_path / "s.db"
    Store(connect(str(db)))

    result = subprocess.run(
        [sys.executable, "-m", "nenapu.cli", "entities", "--rebuild", "--db", str(db)],
        capture_output=True, text=True,
        env={**os.environ, "PYTHONPATH": "src", "NENAPU_NO_BANNER": "1"},
    )

    assert result.returncode == 0, result.stdout + result.stderr


# ==========================================================================
# E3 · dotted keys become subjects
# ==========================================================================


def test_a_dotted_key_splits_into_a_subject_and_an_attribute(store, graph):
    """197 facts already carry keys like `backend.alembic.head`. Split on the
    last dot: everything left is the entity path, the last segment is the
    attribute."""
    from nenapu import entities

    fact, _ = store.write(Fact(text="the head revision is 7f3a91c",
                               key="backend.alembic.head", kind=Kind.ENVIRONMENT))

    entities.subjects_from_keys(store)

    attached = graph.entities_for_fact(fact.id)
    assert [(e.name, role) for e, role in attached] == [("backend.alembic", "subject")]


def test_the_subject_row_records_where_it_came_from(store, graph):
    from nenapu import entities

    fact, _ = store.write(Fact(text="the head revision is 7f3a91c",
                               key="backend.alembic.head"))

    entities.subjects_from_keys(store)

    row = store.conn.execute("SELECT * FROM fact_entities WHERE fact_id = ?",
                             (fact.id,)).fetchone()
    assert (row["role"], row["source"]) == ("subject", "key")


def test_a_single_segment_key_yields_no_entity(store):
    """`port` is an attribute of nothing. Inventing an entity called `port`
    would put a node in the graph that no traversal should ever reach."""
    from nenapu import entities

    store.write(Fact(text="the port is 8080", key="port"))

    entities.subjects_from_keys(store)

    assert store.conn.execute("SELECT COUNT(*) c FROM entities").fetchone()["c"] == 0


def test_a_fact_with_no_key_is_skipped(store):
    from nenapu import entities

    store.write(Fact(text="a fact with no key at all"))

    assert entities.subjects_from_keys(store) == 0


def test_two_facts_about_one_subject_share_the_entity(store, graph):
    from nenapu import entities

    head, _ = store.write(Fact(text="the head revision is 7f3a91c",
                               key="backend.alembic.head"))
    url, _ = store.write(Fact(text="the database url comes from the environment",
                              key="backend.alembic.url"))

    entities.subjects_from_keys(store)

    subject = graph.find(kind="concept", name="backend.alembic", scope="global")
    assert sorted(graph.subjects_of(subject.id)) == sorted([head.id, url.id])


def test_running_it_twice_writes_one_row_per_fact(store):
    from nenapu import entities

    store.write(Fact(text="the head revision is 7f3a91c", key="backend.alembic.head"))

    entities.subjects_from_keys(store)
    entities.subjects_from_keys(store)

    assert store.conn.execute("SELECT COUNT(*) c FROM fact_entities").fetchone()["c"] == 1


def test_the_subject_stays_in_the_facts_scope(store, graph):
    from nenapu import entities

    store.write(Fact(text="the head revision is 7f3a91c", key="backend.alembic.head",
                     scope=SCOPE))

    entities.subjects_from_keys(store)

    assert graph.find(kind="concept", name="backend.alembic", scope="global") is None
    assert graph.find(kind="concept", name="backend.alembic", scope=SCOPE) is not None


# ==========================================================================
# E4 · text matching
# ==========================================================================


def test_a_fact_naming_a_known_file_mentions_it(store, ledger, graph):
    from nenapu import entities

    _session(ledger, paths=["backend/app/routes.py"])
    entities.build_from_activity(store)
    fact, _ = store.write(Fact(text="the login handler is registered in "
                                    "backend/app/routes.py", scope=SCOPE))

    entities.mentions_from_text(store)

    attached = graph.entities_for_fact(fact.id)
    assert [(e.name, role) for e, role in attached] == [("backend/app/routes.py", "mentions")]


def test_matching_cannot_invent_an_entity(store):
    """Matching only against *known* names is the guard: this step must be
    incapable of inventing an entity, only of connecting a fact to something
    already observed to exist."""
    from nenapu import entities

    store.write(Fact(text="the config lives in some/path/nobody/has/touched.py",
                     scope=SCOPE))

    entities.mentions_from_text(store)

    assert store.conn.execute("SELECT COUNT(*) c FROM entities").fetchone()["c"] == 0


def test_a_mention_is_never_promoted_to_a_subject(store, ledger, graph):
    """The distinction E6 depends on: a fact *about* a deleted file dies with
    it, a fact that merely *mentions* it does not."""
    from nenapu import entities

    _session(ledger, paths=["backend/app/routes.py"])
    entities.build_from_activity(store)
    fact, _ = store.write(Fact(text="the handler used to live in backend/app/routes.py",
                               scope=SCOPE))

    entities.mentions_from_text(store)

    row = store.conn.execute("SELECT role FROM fact_entities WHERE fact_id = ?",
                             (fact.id,)).fetchone()
    assert row["role"] == "mentions"


def test_matching_does_not_cross_scope(store, ledger):
    from nenapu import entities

    _session(ledger, paths=["backend/app/routes.py"], scope=OTHER)
    entities.build_from_activity(store)
    store.write(Fact(text="the login handler is in backend/app/routes.py", scope=SCOPE))

    entities.mentions_from_text(store)

    assert store.conn.execute("SELECT COUNT(*) c FROM fact_entities").fetchone()["c"] == 0


def test_a_global_fact_can_mention_a_project_entity(store, ledger, graph):
    """Global facts surface everywhere, so the one crossing that is allowed is
    the one scoping already allows."""
    from nenapu import entities

    _session(ledger, paths=["tools/commit-check.sh"], scope=SCOPE)
    entities.build_from_activity(store)
    fact, _ = store.write(Fact(text="tools/commit-check.sh rejects co-author trailers",
                               kind=Kind.FEEDBACK, scope="global"))

    entities.mentions_from_text(store, scope=SCOPE)

    assert graph.entities_for_fact(fact.id)


def test_ordinary_prose_does_not_match_anything(store, ledger):
    """The tokeniser follows `_salient_terms`, which is why "the" and "with"
    cannot become entity references."""
    from nenapu import entities

    _session(ledger, paths=["backend/app/routes.py"])
    entities.build_from_activity(store)
    store.write(Fact(text="the user prefers short commands with no chained pipes",
                     scope=SCOPE))

    assert entities.mentions_from_text(store) == 0


# ==========================================================================
# E5 · alias resolution
# ==========================================================================


def test_the_three_spellings_resolve_to_one_node(store, graph):
    """`services/auth`, `auth service` and `AuthService` must resolve to one
    node or traversal fragments into synonyms and returns nothing."""
    from nenapu import entities

    a = graph.upsert(kind="service", name="services/auth", scope=SCOPE)
    b = graph.upsert(kind="service", name="auth service", scope=SCOPE)
    c = graph.upsert(kind="service", name="AuthService", scope=SCOPE)

    entities.resolve_aliases(store, scope=SCOPE)

    canonical = {graph.canonical(entity_id).id for entity_id in (a.id, b.id, c.id)}
    assert len(canonical) == 1


def test_the_deterministic_path_needs_no_model(store, graph, monkeypatch):
    """Basename, case fold, snake/camel split — the common cases are decided
    without asking anything, and the model is only for the leftovers."""
    from nenapu import entities

    called = []
    monkeypatch.setattr("nenapu.llm.structured", lambda *a, **k: called.append(a))
    graph.upsert(kind="service", name="services/auth", scope=SCOPE)
    graph.upsert(kind="service", name="AuthService", scope=SCOPE)

    entities.resolve_aliases(store, scope=SCOPE)

    assert called == []


def test_two_genuinely_different_entities_are_not_merged(store, graph):
    """The false-positive cost, and the reason this task is Opus 5: a wrong
    merge is not recoverable by inspection. Two different entities collapsed
    into one corrupts every traversal through them, and nothing in the system
    would report it."""
    from nenapu import entities

    a = graph.upsert(kind="service", name="services/auth", scope=SCOPE)
    b = graph.upsert(kind="service", name="services/author", scope=SCOPE)

    entities.resolve_aliases(store, scope=SCOPE)

    assert graph.canonical(a.id).id != graph.canonical(b.id).id


def test_an_alias_is_recorded_as_an_edge_not_a_deletion(store, graph):
    """The rows stay, and the graph records that one is a spelling of the
    other — the same reason `forget` retires rather than deletes."""
    from nenapu import entities

    a = graph.upsert(kind="service", name="services/auth", scope=SCOPE)
    b = graph.upsert(kind="service", name="AuthService", scope=SCOPE)

    entities.resolve_aliases(store, scope=SCOPE)

    edges = graph.conn.execute(
        "SELECT src_id, dst_id FROM entity_edges WHERE kind = 'alias_of'"
    ).fetchall()
    assert {(r["src_id"], r["dst_id"]) for r in edges} & {(a.id, b.id), (b.id, a.id)}


def test_traversal_follows_an_alias_transparently(store, graph):
    """A query anchored on one spelling has to reach facts attached to
    another, or the merge bought nothing."""
    from nenapu import entities

    canonical = graph.upsert(kind="service", name="services/auth", scope=SCOPE)
    spelled = graph.upsert(kind="service", name="AuthService", scope=SCOPE)
    other = graph.upsert(kind="file", name="services/auth/routes.py", scope=SCOPE)
    graph.link(spelled.id, other.id, kind="contains")

    entities.resolve_aliases(store, scope=SCOPE)

    reached = {entity_id for entity_id, _hops in graph.neighbours(canonical.id, depth=2)}
    assert other.id in reached


def test_aliases_do_not_cross_scope(store, graph):
    from nenapu import entities

    here = graph.upsert(kind="service", name="services/auth", scope=SCOPE)
    there = graph.upsert(kind="service", name="AuthService", scope=OTHER)

    entities.resolve_aliases(store, scope=SCOPE)

    assert graph.canonical(here.id).id != graph.canonical(there.id).id


def test_resolution_is_idempotent(store, graph):
    from nenapu import entities

    graph.upsert(kind="service", name="services/auth", scope=SCOPE)
    graph.upsert(kind="service", name="AuthService", scope=SCOPE)

    entities.resolve_aliases(store, scope=SCOPE)
    first = graph.conn.execute("SELECT COUNT(*) c FROM entity_edges").fetchone()["c"]
    entities.resolve_aliases(store, scope=SCOPE)

    assert graph.conn.execute("SELECT COUNT(*) c FROM entity_edges").fetchone()["c"] == first


# ==========================================================================
# E6 · entity death, from the entity side and from git
# ==========================================================================


@e6
def test_marking_an_entity_gone_records_the_status(store, graph):
    entity = graph.upsert(kind="file", name="services/auth/routes.py", scope=SCOPE)

    graph.mark_gone(entity.id, reason="deleted in commit abc123")

    assert graph.get(entity.id).status == "gone"


@e6
def test_the_facts_it_was_the_subject_of_come_back_as_a_list(store, graph):
    """`mark_gone` returns what it falsified, so a caller can report it rather
    than having to re-derive it."""
    fact, _ = store.write(Fact(text="services/auth/routes.py owns the login handler",
                               scope=SCOPE))
    entity = graph.upsert(kind="file", name="services/auth/routes.py", scope=SCOPE)
    graph.attach(fact.id, entity.id, role="subject", source="path")

    assert graph.mark_gone(entity.id, reason="deleted") == [fact.id]


@e6
def test_a_deleted_file_falsifies_the_fact_about_it_end_to_end(store, tmp_path):
    """Detection rides `_record_git_evidence`, which already sees deletions.
    The end-to-end shape the plan asks for: delete a tracked file, run the
    capture path, and the fact about it surfaces under the warning."""
    from nenapu.activity import ActivityLedger
    from nenapu.capture import capture_session
    from nenapu.entities import EntityGraph
    from nenapu.observer import recall_context
    from nenapu.store import project_scope

    repo = tmp_path / "repo"
    repo.mkdir()
    env = {"GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@example.com",
           "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@example.com",
           "PATH": os.environ.get("PATH", ""), "HOME": str(repo)}

    def git(*args):
        subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True, env=env)

    git("init", "-q", "-b", "main")
    (repo / "auth.py").write_text("handler\n")
    git("add", "-A")
    git("commit", "-q", "-m", "init")
    before = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo, capture_output=True,
                            text=True, env=env).stdout.strip()

    scope = project_scope(str(repo))
    fact, _ = store.write(Fact(text=f"{repo}/auth.py owns the login handler", scope=scope))
    entity = EntityGraph(store.conn).upsert(kind="file", name=f"{repo}/auth.py",
                                            scope=scope)
    EntityGraph(store.conn).attach(fact.id, entity.id, role="subject", source="path")

    (repo / "auth.py").unlink()
    git("add", "-A")
    git("commit", "-q", "-m", "remove auth")
    transcript = tmp_path / "t.jsonl"
    transcript.write_text(json.dumps({
        "type": "user", "sessionId": "s-del", "cwd": str(repo),
        "message": {"role": "user", "content": "removed it"},
    }))

    capture_session(ActivityLedger(store.conn), transcript, agent="claude-code",
                    cwd=str(repo), git_head_before=before)

    assert store.get(fact.id).status == Status.SUSPECT
    assert "falsified" in recall_context(store, scope=scope, cwd=str(repo)).lower()


@e6
def test_a_restored_file_brings_its_fact_back(store, graph):
    fact, _ = store.write(Fact(text="services/auth/routes.py owns the login handler",
                               scope=SCOPE))
    entity = graph.upsert(kind="file", name="services/auth/routes.py", scope=SCOPE)
    graph.attach(fact.id, entity.id, role="subject", source="path")
    graph.mark_gone(entity.id, reason="deleted")

    restored = graph.mark_alive(entity.id)

    assert restored == [fact.id]
    assert store.get(fact.id).status == Status.ACTIVE


@e6
def test_a_fact_already_suspect_keeps_its_first_reason(store, graph):
    """"A fact already suspect stays suspect under its first recorded reason"
    — the rule `cascade_falsification` already keeps."""
    fact, _ = store.write(Fact(text="services/auth/routes.py owns the login handler",
                               scope=SCOPE))
    store.set_status(fact.id, Status.SUSPECT)
    first_reason = store.get(fact.id).suspect_reason
    entity = graph.upsert(kind="file", name="services/auth/routes.py", scope=SCOPE)
    graph.attach(fact.id, entity.id, role="subject", source="path")

    graph.mark_gone(entity.id, reason="deleted")

    assert store.get(fact.id).suspect_reason == first_reason


@e6
def test_a_retired_fact_is_not_dragged_back_into_doubt(store, graph):
    """Only `active` facts are touched. A fact already retired has its own
    story."""
    fact, _ = store.write(Fact(text="services/auth/routes.py owns the login handler",
                               scope=SCOPE))
    store.forget(fact.id)
    entity = graph.upsert(kind="file", name="services/auth/routes.py", scope=SCOPE)
    graph.attach(fact.id, entity.id, role="subject", source="path")

    graph.mark_gone(entity.id, reason="deleted")

    assert store.get(fact.id).status == Status.RETIRED


# ==========================================================================
# E8 · model-extracted entities
# ==========================================================================


class FakeBackend:
    name = "fake"
    model = "fake"
    supports_schema = False


def _transcript(tmp_path):
    def event(role, text):
        return json.dumps({"type": role,
                           "message": {"role": role, "content": [
                               {"type": "text", "text": text}]}})

    path = tmp_path / "t.jsonl"
    path.write_text("\n".join([event("user", "the auth service owns login"),
                               event("assistant", "noted. " + "detail " * 40)]))
    return path


def _patch_structured(monkeypatch, payload):
    calls = []

    def fake(prompt, schema, system=None, backend=None, max_tokens=None):
        calls.append(prompt)
        return payload

    monkeypatch.setattr("nenapu.observer.structured", fake)
    return calls


@e8
def test_the_schema_asks_for_entities():
    """For the entities a filesystem cannot see: services, people, concepts."""
    from nenapu.observer import EXTRACT_SCHEMA

    item = EXTRACT_SCHEMA["properties"]["entities"]["items"]

    assert set(item["required"]) >= {"name", "kind"}
    assert "relation" in item["properties"]
    assert "target" in item["properties"]


@e8
def test_an_extracted_entity_is_written(store, tmp_path, monkeypatch):
    from nenapu.observer import observe_transcript

    _patch_structured(monkeypatch, {"facts": [], "entities": [
        {"name": "auth service", "kind": "service", "relation": "", "target": ""},
    ]})

    observe_transcript(store, _transcript(tmp_path), session_id="s-1",
                       backend=FakeBackend())

    names = {r["name"] for r in store.conn.execute("SELECT name FROM entities")}
    assert "auth service" in names


@e8
def test_a_target_the_model_was_not_shown_is_dropped(store, tmp_path, monkeypatch):
    """Mirrors `_proposed_id`: real ids are guessable, and the 1.5b model in
    the calibration table invented nine of them for four facts."""
    from nenapu.observer import observe_transcript

    _patch_structured(monkeypatch, {"facts": [], "entities": [
        {"name": "auth service", "kind": "service", "relation": "calls",
         "target": "99999"},
    ]})

    observe_transcript(store, _transcript(tmp_path), session_id="s-1",
                       backend=FakeBackend())

    edges = store.conn.execute("SELECT COUNT(*) c FROM entity_edges").fetchone()["c"]
    assert edges == 0


def test_extraction_still_works_when_the_entities_list_is_absent(store, tmp_path,
                                                                monkeypatch):
    """A model that does not know about the new field is proposing what it has
    always proposed."""
    from nenapu.observer import observe_transcript

    _patch_structured(monkeypatch, {"facts": [
        {"text": "The auth service owns login.", "kind": "project",
         "key": "auth.owner", "correction": False},
    ], "open_loops": [], "grades": []})

    learned = observe_transcript(store, _transcript(tmp_path), session_id="s-1",
                                 backend=FakeBackend())

    assert [f.text for f in learned] == ["The auth service owns login."]


def test_entities_ride_the_same_single_call(store, tmp_path, monkeypatch):
    """Still one call, now with a fourth output: facts, open loops, grades,
    entities."""
    from nenapu.observer import observe_transcript

    calls = _patch_structured(monkeypatch, {"facts": [], "entities": [
        {"name": "auth service", "kind": "service", "relation": "", "target": ""},
    ]})

    observe_transcript(store, _transcript(tmp_path), session_id="s-1",
                       backend=FakeBackend())

    assert len(calls) == 1


@e8
def test_an_extracted_entity_lands_in_the_sessions_scope(store, tmp_path, monkeypatch):
    from nenapu.observer import observe_transcript

    _patch_structured(monkeypatch, {"facts": [], "entities": [
        {"name": "auth service", "kind": "service", "relation": "", "target": ""},
    ]})

    observe_transcript(store, _transcript(tmp_path), session_id="s-1",
                       scope=SCOPE, backend=FakeBackend())

    row = store.conn.execute("SELECT scope FROM entities WHERE name = 'auth service'"
                             ).fetchone()
    assert row["scope"] == SCOPE


@e8
def test_a_dry_run_writes_no_entities(store, tmp_path, monkeypatch):
    from nenapu.observer import observe_transcript

    _patch_structured(monkeypatch, {"facts": [], "entities": [
        {"name": "auth service", "kind": "service", "relation": "", "target": ""},
    ]})

    observe_transcript(store, _transcript(tmp_path), session_id="s-1",
                       backend=FakeBackend(), apply=False)

    assert store.conn.execute("SELECT COUNT(*) c FROM entities").fetchone()["c"] == 0


# ==========================================================================
# E9 · edge weights learn from grades
# ==========================================================================


def _surfaced(store, graph, *, fact, entity, session_id, outcome):
    """A fact reached through an entity edge and then graded — the loop E9
    closes back into the graph."""
    from nenapu.models import Outcome

    graph.attach(fact.id, entity.id, role="subject", source="path")
    recall_id = store.ledger.log(fact.id, session_id=session_id, query="deploy")
    store.ledger.grade(recall_id, getattr(Outcome, outcome.upper()), source="observer")


@e9
def test_an_edge_that_surfaces_good_facts_gains_weight(store, graph):
    """The entity-layer counterpart of G10: traversal learns which relations
    actually carry useful memory."""
    from nenapu import entities

    anchor = graph.upsert(kind="file", name="a.py", scope=SCOPE)
    useful = graph.upsert(kind="file", name="b.py", scope=SCOPE)
    graph.link(anchor.id, useful.id, kind="touched_with")
    fact, _ = store.write(Fact(text="b.py owns the retry policy", scope=SCOPE))
    for i in range(3):
        _surfaced(store, graph, fact=fact, entity=useful, session_id=f"s-{i}",
                  outcome="good")
    before = _edge(store.conn, anchor.id, useful.id, "touched_with")["weight"]

    entities.reward_edges_for_grades(store)

    assert _edge(store.conn, anchor.id, useful.id, "touched_with")["weight"] > before


@e9
def test_an_edge_that_surfaces_unused_facts_gains_nothing(store, graph):
    from nenapu import entities

    anchor = graph.upsert(kind="file", name="a.py", scope=SCOPE)
    quiet = graph.upsert(kind="file", name="b.py", scope=SCOPE)
    graph.link(anchor.id, quiet.id, kind="touched_with")
    fact, _ = store.write(Fact(text="b.py owns the retry policy", scope=SCOPE))
    for i in range(3):
        _surfaced(store, graph, fact=fact, entity=quiet, session_id=f"s-{i}",
                  outcome="neutral")
    before = _edge(store.conn, anchor.id, quiet.id, "touched_with")["weight"]

    entities.reward_edges_for_grades(store)

    assert _edge(store.conn, anchor.id, quiet.id, "touched_with")["weight"] == before


@e9
def test_a_misleading_fact_does_not_raise_the_edge_that_surfaced_it(store, graph):
    from nenapu import entities

    anchor = graph.upsert(kind="file", name="a.py", scope=SCOPE)
    wrong = graph.upsert(kind="file", name="b.py", scope=SCOPE)
    graph.link(anchor.id, wrong.id, kind="touched_with")
    fact, _ = store.write(Fact(text="b.py owns the retry policy", scope=SCOPE))
    _surfaced(store, graph, fact=fact, entity=wrong, session_id="s-1", outcome="bad")
    before = _edge(store.conn, anchor.id, wrong.id, "touched_with")["weight"]

    entities.reward_edges_for_grades(store)

    assert _edge(store.conn, anchor.id, wrong.id, "touched_with")["weight"] <= before


@e9
def test_weights_are_bounded(store, graph):
    """Without bounding, an edge that surfaced one good fact attracts more
    traffic, earns more grades, and crowds out the rest — which is the whole
    reason this is a design task and not a counter."""
    from nenapu import entities
    from nenapu.entities import MAX_EDGE_WEIGHT

    anchor = graph.upsert(kind="file", name="a.py", scope=SCOPE)
    hot = graph.upsert(kind="file", name="b.py", scope=SCOPE)
    graph.link(anchor.id, hot.id, kind="touched_with")
    fact, _ = store.write(Fact(text="b.py owns the retry policy", scope=SCOPE))
    for i in range(200):
        _surfaced(store, graph, fact=fact, entity=hot, session_id=f"s-{i}",
                  outcome="good")
        entities.reward_edges_for_grades(store)

    assert _edge(store.conn, anchor.id, hot.id, "touched_with")["weight"] <= MAX_EDGE_WEIGHT


@e9
def test_a_store_with_no_grades_leaves_every_weight_alone(store, graph):
    from nenapu import entities

    anchor = graph.upsert(kind="file", name="a.py", scope=SCOPE)
    other = graph.upsert(kind="file", name="b.py", scope=SCOPE)
    graph.link(anchor.id, other.id, kind="touched_with")
    before = [dict(r) for r in store.conn.execute("SELECT * FROM entity_edges")]

    entities.reward_edges_for_grades(store)

    assert [dict(r) for r in store.conn.execute("SELECT * FROM entity_edges")] == before


@e9
def test_rewarding_is_idempotent_per_grade(store, graph):
    """A maintenance tick that runs hourly must not pay for the same grade
    every hour."""
    from nenapu import entities

    anchor = graph.upsert(kind="file", name="a.py", scope=SCOPE)
    useful = graph.upsert(kind="file", name="b.py", scope=SCOPE)
    graph.link(anchor.id, useful.id, kind="touched_with")
    fact, _ = store.write(Fact(text="b.py owns the retry policy", scope=SCOPE))
    _surfaced(store, graph, fact=fact, entity=useful, session_id="s-1", outcome="good")

    entities.reward_edges_for_grades(store)
    once = _edge(store.conn, anchor.id, useful.id, "touched_with")["weight"]
    entities.reward_edges_for_grades(store)

    assert _edge(store.conn, anchor.id, useful.id, "touched_with")["weight"] == once
