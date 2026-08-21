"""The entity tier: files, dirs, commits, services — joined to belief.

`fact_entities` is the whole integration: one graph out of two node types
rather than two graphs sitting beside each other.

    BELIEF (built)          BRIDGE                    ENTITY
    fact --derived_from-->  fact_entities             entity --contains--> entity
         cascades              fact_id                       --touched_with-->
                               entity_id                     --changed_in-->
                               role: subject|mentions        --alias_of-->

`role='subject'` is load-bearing: a fact *about* a deleted file dies with
it, a fact that merely *mentions* it does not (E6, cascaded through
`graph.cascade_falsification`, not built here).

Everything in this module costs zero model calls — the same principle the
activity ledger already runs on: entities and their edges are inferred
from git and transcript tool calls, never asked of a model.
"""

from __future__ import annotations

import json
import re
import sqlite3
from collections import deque

from .db import commit, transaction
from .models import (
    Decay,
    EntityEdge,
    EntityEdgeKind,
    EntityKind,
    EntityStatus,
    HALF_LIFE_DAYS,
    now,
    row_to_entity,
)

DAY = 86400.0

# A monorepo path eight directories deep should not become eight dir nodes —
# a traversal would spend its whole depth budget walking upward. Only the
# directories closest to the file are kept.
MAX_DIR_DEPTH = 4

# A pair of files edited together once is a coincidence. `infer_edges_for`
# (graph.py, see G10) fell into exactly this trap one layer down; three
# observations is the same guard one layer up.
MIN_TOUCHED_OBSERVATIONS = 3

# `touched_with` decays on the same 90-day half-life facts already use for
# `Decay.MEDIUM`, so a pairing from last year stops steering retrieval.
_TOUCHED_HALF_LIFE_DAYS = HALF_LIFE_DAYS[Decay.MEDIUM]


class EntityGraph:
    """Mirrors `graph.Graph`'s shape, one tier up: nodes are entities, not facts."""

    def __init__(self, conn: sqlite3.Connection) -> None:
        self.conn = conn

    # ---------- nodes ----------

    def upsert(self, *, kind: str, name: str, scope: str = "global", at: float | None = None):
        """Idempotent on `(kind, name, scope)`. A repeat upsert moves `last_seen`
        and counts the mention rather than creating a second row."""
        at = at if at is not None else now()
        with transaction(self.conn):
            row = self.conn.execute(
                "SELECT id FROM entities WHERE kind = ? AND name = ? AND scope = ?",
                (kind, name, scope),
            ).fetchone()
            if row:
                self.conn.execute(
                    "UPDATE entities SET last_seen = ?, mentions = mentions + 1 WHERE id = ?",
                    (at, row["id"]),
                )
                commit(self.conn)
                return self.get(row["id"])
            cur = self.conn.execute(
                "INSERT INTO entities(kind, name, scope, status, first_seen, last_seen, mentions)"
                " VALUES (?,?,?,?,?,?,1)",
                (kind, name, scope, EntityStatus.ALIVE, at, at),
            )
            commit(self.conn)
            return self.get(cur.lastrowid)

    def get(self, entity_id: int):
        row = self.conn.execute("SELECT * FROM entities WHERE id = ?", (entity_id,)).fetchone()
        return row_to_entity(row) if row else None

    def canonical(self, entity_id: int):
        """Follow `alias_of` to the node that owns this spelling.

        Aliases are edges rather than deletions, so every caller that means
        "the thing itself" has to ask. Cycle-guarded: a chain that loops
        stops at the first repeat rather than spinning.
        """
        seen = {entity_id}
        current = entity_id
        while True:
            row = self.conn.execute(
                "SELECT dst_id FROM entity_edges WHERE src_id = ? AND kind = ?"
                " AND valid_to IS NULL LIMIT 1",
                (current, EntityEdgeKind.ALIAS_OF),
            ).fetchone()
            if row is None or row["dst_id"] in seen:
                return self.get(current)
            current = row["dst_id"]
            seen.add(current)

    def find(self, *, kind: str | None = None, name: str, scope: str = "global"):
        sql = "SELECT * FROM entities WHERE name = ? AND scope = ?"
        args: list = [name, scope]
        if kind is not None:
            sql += " AND kind = ?"
            args.append(kind)
        row = self.conn.execute(sql, args).fetchone()
        return row_to_entity(row) if row else None

    # ---------- edges ----------

    def link(self, src_id: int, dst_id: int, *, kind: str, source: str = "observed",
              weight: float | None = None):
        """Record a relation. Self-links and dupes are refused, the same two
        rejections `Graph.link` already makes and for the same reasons."""
        if src_id == dst_id:
            return None
        w = weight if weight is not None else 1.0
        edge = EntityEdge(src_id=src_id, dst_id=dst_id, kind=kind, source=source, weight=w)
        try:
            cur = self.conn.execute(
                "INSERT INTO entity_edges(src_id, dst_id, kind, source, weight, observations,"
                " valid_from) VALUES (?,?,?,?,?,?,?)",
                (src_id, dst_id, kind, source, w, 1, edge.valid_from),
            )
        except sqlite3.IntegrityError:
            return None
        commit(self.conn)
        edge.id = cur.lastrowid
        return edge

    def close_edge(self, src_id: int, dst_id: int, kind: str) -> None:
        """A move closes an edge rather than deleting it — the graph records
        what was true when, not only what is true now."""
        self.conn.execute(
            "UPDATE entity_edges SET valid_to = ? WHERE src_id = ? AND dst_id = ? AND kind = ?",
            (now(), src_id, dst_id, kind),
        )
        commit(self.conn)

    def neighbours(self, entity_id: int, *, depth: int = 1) -> list[tuple[int, int]]:
        """Breadth-first, cycle-safe, undirected — a `contains` edge and a
        `touched_with` edge both mean "related", and E7's proximity term
        decays per hop rather than caring which direction the edge points."""
        seen = {entity_id}
        order: list[tuple[int, int]] = []
        queue: deque[tuple[int, int]] = deque([(entity_id, 0)])
        while queue:
            current, hops = queue.popleft()
            if hops >= depth:
                continue
            rows = self.conn.execute(
                "SELECT dst_id AS other FROM entity_edges WHERE src_id = ? AND valid_to IS NULL"
                " UNION SELECT src_id AS other FROM entity_edges"
                " WHERE dst_id = ? AND valid_to IS NULL",
                (current, current),
            ).fetchall()
            for row in rows:
                other = row["other"]
                if other in seen:
                    continue
                seen.add(other)
                order.append((other, hops + 1))
                queue.append((other, hops + 1))
        return order

    # ---------- the bridge to belief ----------

    def attach(self, fact_id: int, entity_id: int, *, role: str, source: str) -> None:
        self.conn.execute(
            "INSERT OR IGNORE INTO fact_entities(fact_id, entity_id, role, source)"
            " VALUES (?,?,?,?)",
            (fact_id, entity_id, role, source),
        )
        commit(self.conn)

    def subjects_of(self, entity_id: int) -> list[int]:
        rows = self.conn.execute(
            "SELECT fact_id FROM fact_entities WHERE entity_id = ? AND role = 'subject'",
            (entity_id,),
        )
        return [r["fact_id"] for r in rows]

    def entities_for_fact(self, fact_id: int) -> list[tuple]:
        rows = self.conn.execute(
            "SELECT e.*, fe.role AS role FROM fact_entities fe"
            " JOIN entities e ON e.id = fe.entity_id WHERE fe.fact_id = ?",
            (fact_id,),
        )
        return [(row_to_entity(r), r["role"]) for r in rows]


# ---------- E2 · deterministic build from the activity ledger ----------


def _decay(age_seconds: float) -> float:
    age_days = max(age_seconds, 0.0) / DAY
    return 0.5 ** (age_days / _TOUCHED_HALF_LIFE_DAYS)


def _dir_prefixes(path: str) -> list[str]:
    parts = path.split("/")[:-1]
    prefixes = ["/".join(parts[: i + 1]) for i in range(len(parts))]
    if len(prefixes) > MAX_DIR_DEPTH:
        prefixes = prefixes[-MAX_DIR_DEPTH:]
    return prefixes


def _upsert_file_and_dirs(graph: EntityGraph, path: str, scope: str):
    """A file entity, plus its ancestor directories `contains`-linked down to
    it — capped at `MAX_DIR_DEPTH`, keeping the directories closest to the
    file rather than the ones nearest the repo root."""
    file_entity = graph.upsert(kind=EntityKind.FILE, name=path, scope=scope)
    prev = None
    for prefix in _dir_prefixes(path):
        dir_entity = graph.upsert(kind=EntityKind.DIR, name=prefix, scope=scope)
        if prev is not None:
            graph.link(prev.id, dir_entity.id, kind=EntityEdgeKind.CONTAINS)
        prev = dir_entity
    if prev is not None:
        graph.link(prev.id, file_entity.id, kind=EntityEdgeKind.CONTAINS)
    return file_entity


def _upsert_touched_with(conn: sqlite3.Connection, src_id: int, dst_id: int, *,
                          observations: int, weight: float) -> None:
    """Set, not increment: a rebuild recomputes both from the whole history
    each time, so calling it twice must not double either one."""
    existing = conn.execute(
        "SELECT id FROM entity_edges WHERE src_id = ? AND dst_id = ? AND kind = ?",
        (src_id, dst_id, EntityEdgeKind.TOUCHED_WITH),
    ).fetchone()
    if existing:
        conn.execute(
            "UPDATE entity_edges SET observations = ?, weight = ? WHERE id = ?",
            (observations, weight, existing["id"]),
        )
    else:
        conn.execute(
            "INSERT INTO entity_edges(src_id, dst_id, kind, source, weight, observations,"
            " valid_from) VALUES (?,?,?,?,?,?,?)",
            (src_id, dst_id, EntityEdgeKind.TOUCHED_WITH, "observed", weight, observations, now()),
        )
    commit(conn)


def build_from_activity(store, *, scope: str | None = None) -> dict:
    """Bootstrap the entity graph from `sessions`, `file_events` and `commits`.

    Zero model calls — everything needed is already stored. Safe to call
    repeatedly: entities and `contains`/`changed_in` edges are idempotent by
    construction, and `touched_with` is recomputed from the whole history
    each time rather than incremented, so a rebuild changes nothing when
    nothing new has happened.
    """
    graph = EntityGraph(store.conn)
    conn = store.conn

    sql = "SELECT id, project_scope, started_at FROM sessions"
    args: list = []
    if scope is not None:
        sql += " WHERE project_scope = ?"
        args.append(scope)
    sessions = conn.execute(sql, args).fetchall()

    anchor_row = conn.execute("SELECT MAX(started_at) AS m FROM sessions").fetchone()
    anchor = anchor_row["m"] or now()

    seen_scopes: set[str] = set()
    pair_observations: dict[tuple[str, tuple[str, str]], list[float]] = {}
    for srow in sessions:
        session_id, pscope, started_at = srow["id"], srow["project_scope"], srow["started_at"]
        if pscope not in seen_scopes:
            graph.upsert(kind=EntityKind.REPO, name=pscope, scope=pscope)
            seen_scopes.add(pscope)

        paths = sorted({
            r["path"] for r in conn.execute(
                "SELECT DISTINCT path FROM file_events WHERE session_id = ?", (session_id,)
            )
        })
        for path in paths:
            _upsert_file_and_dirs(graph, path, pscope)

        for i, a in enumerate(paths):
            for b in paths[i + 1:]:
                key = (pscope, (a, b))
                pair_observations.setdefault(key, []).append(started_at)

    for (pscope, (a, b)), timestamps in pair_observations.items():
        if len(timestamps) < MIN_TOUCHED_OBSERVATIONS:
            continue
        entity_a = graph.find(kind=EntityKind.FILE, name=a, scope=pscope)
        entity_b = graph.find(kind=EntityKind.FILE, name=b, scope=pscope)
        if entity_a is None or entity_b is None:
            continue
        src_id, dst_id = sorted((entity_a.id, entity_b.id))
        weight = sum(_decay(anchor - t) for t in timestamps)
        _upsert_touched_with(conn, src_id, dst_id, observations=len(timestamps), weight=weight)

    csql = ("SELECT c.sha, c.files_changed, s.project_scope AS project_scope"
            " FROM commits c JOIN sessions s ON s.id = c.session_id")
    cargs: list = []
    if scope is not None:
        csql += " WHERE s.project_scope = ?"
        cargs.append(scope)
    for crow in conn.execute(csql, cargs).fetchall():
        pscope = crow["project_scope"]
        commit_entity = graph.upsert(kind=EntityKind.COMMIT, name=crow["sha"], scope=pscope)
        for path in json.loads(crow["files_changed"] or "[]"):
            file_entity = graph.find(kind=EntityKind.FILE, name=path, scope=pscope) \
                or _upsert_file_and_dirs(graph, path, pscope)
            graph.link(file_entity.id, commit_entity.id, kind=EntityEdgeKind.CHANGED_IN)

    return {"sessions": len(sessions), "scopes": len(seen_scopes)}


# ---------- E3 · dotted keys become subjects ----------


def subjects_from_keys(store, *, scope: str | None = None) -> int:
    """197 facts already carry dotted keys — `backend.alembic.head`. Split on
    the last dot: everything left of it is the subject, the last segment is
    the attribute. No model call, and the highest yield per line in the tier.
    """
    graph = EntityGraph(store.conn)
    sql = "SELECT id, key, scope FROM facts WHERE key IS NOT NULL AND key != ''"
    args: list = []
    if scope is not None:
        sql += " AND scope = ?"
        args.append(scope)

    written = 0
    for row in store.conn.execute(sql, args).fetchall():
        key = row["key"]
        if "." not in key:
            continue  # a single-segment key is an attribute of nothing
        subject_name = key.rsplit(".", 1)[0]
        entity = graph.upsert(kind=EntityKind.CONCEPT, name=subject_name, scope=row["scope"])
        graph.attach(row["id"], entity.id, role="subject", source="key")
        written += 1
    return written


# ---------- E4 · text matching against known names only ----------


def mentions_from_text(store, *, scope: str | None = None) -> int:
    """Path- and identifier-shaped tokens in a fact's text, matched against
    entity names already known — never invented. Matching only against
    entities that already exist is the guard: this step can connect a fact
    to something observed to exist, never conjure a new node.
    """
    conn = store.conn
    if scope is None:
        fact_sql = "SELECT id, text, scope FROM facts WHERE status = 'active'"
        fact_args: list = []
    else:
        fact_sql = ("SELECT id, text, scope FROM facts"
                    " WHERE status = 'active' AND (scope = ? OR scope = 'global')")
        fact_args = [scope]

    written = 0
    for frow in conn.execute(fact_sql, fact_args).fetchall():
        candidate_scopes = {frow["scope"]}
        if scope is not None:
            candidate_scopes.add(scope)
        placeholders = ",".join("?" for _ in candidate_scopes)
        entity_rows = conn.execute(
            f"SELECT id, name FROM entities WHERE scope IN ({placeholders})",
            list(candidate_scopes),
        ).fetchall()
        matched_names = {r["name"] for r in entity_rows if r["name"] and r["name"] in frow["text"]}
        for erow in entity_rows:
            name = erow["name"]
            if not name or name not in matched_names:
                continue
            # A path is its own ancestor's prefix — "backend/app/routes.py"
            # contains "backend" as a substring, and both would otherwise
            # match. Keep only the longest match: a mention names the most
            # specific thing the text actually said, not every directory
            # that happens to be a prefix of it.
            if any(name != other and name in other for other in matched_names):
                continue
            existing = conn.execute(
                "SELECT 1 FROM fact_entities WHERE fact_id = ? AND entity_id = ?",
                (frow["id"], erow["id"]),
            ).fetchone()
            if existing:
                continue
            conn.execute(
                "INSERT INTO fact_entities(fact_id, entity_id, role, source)"
                " VALUES (?,?,'mentions','path')",
                (frow["id"], erow["id"]),
            )
            commit(conn)
            written += 1
    return written


# ---------- E5 · alias resolution ----------
#
# `services/auth`, `auth service` and `AuthService` are one thing spelled three
# ways, and a traversal that treats them as three nodes fragments into synonyms
# and returns nothing.
#
# The cost of getting this wrong is asymmetric and that shapes the design. A
# missed alias leaves the graph as it is today. A wrong merge corrupts every
# traversal through the merged node, is invisible to inspection, and nothing in
# the system would report it. So:
#
#   * only human-named kinds are ever merged. A path is already an exact
#     identifier: `a/b.py` and `b/a.py` are two files, whatever their words
#     have in common, and the same holds for dirs, repos and commit shas.
#   * a merge needs the same scope and the same kind. "Right fact, wrong
#     project" is a failure scoping already had to fix once.
#   * an alias is an edge, never a deletion, so a wrong merge is at least
#     undoable.

# Kinds whose names are prose, and are therefore worth normalising. Everything
# else is an exact identifier where two spellings mean two things.
ALIASABLE_KINDS = frozenset({EntityKind.SERVICE, EntityKind.PERSON, EntityKind.CONCEPT})

# Words that say what sort of thing something is rather than which thing it is.
# `auth service` and `AuthService` normalise together with them kept; they are
# dropped only when a name would otherwise be nothing but these.
_NAME_NOISE = frozenset({"the", "a", "an", "of", "and"})

# How many facts two entities must both be attached to before the model is
# asked about them at all. Co-occurrence is the only evidence available that
# two names that do not normalise together might still be one thing.
MIN_COOCCURRENCE_FOR_MODEL = 3

_CAMEL_BOUNDARY = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")
_NAME_SEPARATORS = re.compile(r"[^A-Za-z0-9]+")

ALIAS_SCHEMA = {
    "type": "object",
    "properties": {
        "aliases": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "a": {"type": "integer"},
                    "b": {"type": "integer"},
                    "same": {"type": "boolean"},
                },
                "required": ["a", "b", "same"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["aliases"],
    "additionalProperties": False,
}

ALIAS_SYSTEM = """\
You are told about pairs of names that appear in one project's memory, and for
each pair you say whether the two names refer to the same thing.

Say true only when they are two spellings of one thing. Two related things, two
parts of one system, or two things that are often worked on together are not
the same thing. When you are unsure, say false: leaving two names apart costs a
missed connection, merging two different things corrupts everything recorded
about either one."""


def _alias_key(name: str) -> frozenset[str]:
    """The normal form two spellings of one name share.

    Basename, case fold, snake and camel split, and a naive singular. Order is
    dropped on purpose: `services/auth` and `auth service` are the same name
    said in two directions.
    """
    spaced = _NAME_SEPARATORS.sub(" ", _CAMEL_BOUNDARY.sub(" ", name or ""))
    words = [w.lower() for w in spaced.split() if w]
    singular = {w[:-1] if len(w) > 3 and w.endswith("s") else w for w in words}
    meaningful = singular - _NAME_NOISE
    return frozenset(meaningful or singular)


def _alias_groups(rows: list) -> list[list]:
    """Entities that share a normal form, grouped within scope and kind."""
    groups: dict[tuple, list] = {}
    for row in rows:
        key = _alias_key(row["name"])
        if not key:
            continue
        groups.setdefault((row["scope"], row["kind"], key), []).append(row)
    return [members for members in groups.values() if len(members) > 1]


def _canonical_of(members: list):
    """The spelling the others become aliases of.

    Most mentioned first, because that is the name the project actually uses;
    then oldest, then lowest id, so the choice does not move between runs.
    """
    return sorted(members, key=lambda r: (-r["mentions"], r["first_seen"], r["id"]))[0]


def _link_aliases(graph: EntityGraph, members: list) -> int:
    canonical = _canonical_of(members)
    linked = 0
    for row in members:
        if row["id"] == canonical["id"]:
            continue
        if graph.canonical(row["id"]).id == canonical["id"]:
            continue  # already resolved; a rerun must not add a second edge
        if graph.link(row["id"], canonical["id"], kind=EntityEdgeKind.ALIAS_OF,
                      source="inferred") is not None:
            linked += 1
    return linked


def _cooccurring_pairs(conn: sqlite3.Connection, rows: list) -> list[tuple]:
    """Pairs of aliasable entities attached to the same facts often enough to
    be worth a question, and which normalisation did not already join."""
    by_id = {row["id"]: row for row in rows}
    if len(by_id) < 2:
        return []
    marks = ",".join("?" * len(by_id))
    counts = conn.execute(
        f"SELECT a.entity_id AS a_id, b.entity_id AS b_id, COUNT(*) AS shared"
        f" FROM fact_entities a JOIN fact_entities b ON a.fact_id = b.fact_id"
        f" AND a.entity_id < b.entity_id"
        f" WHERE a.entity_id IN ({marks}) AND b.entity_id IN ({marks})"
        f" GROUP BY a.entity_id, b.entity_id HAVING shared >= ?",
        [*by_id, *by_id, MIN_COOCCURRENCE_FOR_MODEL],
    ).fetchall()

    pairs = []
    for row in counts:
        first, second = by_id[row["a_id"]], by_id[row["b_id"]]
        if first["scope"] != second["scope"] or first["kind"] != second["kind"]:
            continue
        if _alias_key(first["name"]) == _alias_key(second["name"]):
            continue  # the deterministic pass already has this one
        pairs.append((first, second))
    return pairs


def _model_aliases(store, graph: EntityGraph, rows: list, backend) -> int:
    """Ask about the leftovers only: names that co-occur heavily and never
    normalise together. Ids the model was not shown are dropped, mirroring the
    guard the extractor already uses on proposed ids."""
    from .llm import structured

    pairs = _cooccurring_pairs(store.conn, rows)
    if not pairs:
        return 0

    shown = {(a["id"], b["id"]) for a, b in pairs}
    listing = "\n".join(
        f"{a['id']}: {a['name']}  |  {b['id']}: {b['name']}" for a, b in pairs
    )
    result = structured(
        f"Pairs of names from one project:\n\n{listing}\n\n"
        "For each pair, is it two spellings of one thing?",
        ALIAS_SCHEMA, system=ALIAS_SYSTEM, backend=backend, max_tokens=1024,
    )

    by_id = {row["id"]: row for row in rows}
    linked = 0
    for item in result.get("aliases") or []:
        if not isinstance(item, dict) or not item.get("same"):
            continue
        try:
            pair = (int(item["a"]), int(item["b"]))
        except (KeyError, TypeError, ValueError):
            continue
        if pair not in shown and pair[::-1] not in shown:
            continue
        linked += _link_aliases(graph, [by_id[pair[0]], by_id[pair[1]]])
    return linked


def resolve_aliases(store, *, scope: str | None = None, backend=None) -> int:
    """Join the spellings of one name into one node. Returns edges added.

    Deterministic normalisation decides the common cases with no model call.
    The model is opt-in — pass a backend — and is only ever shown the
    leftovers: names that co-occur heavily and that normalisation did not
    join. Running twice adds nothing the first run did not.
    """
    graph = EntityGraph(store.conn)
    marks = ",".join("?" * len(ALIASABLE_KINDS))
    sql = f"SELECT * FROM entities WHERE kind IN ({marks}) AND status = ?"
    args: list = [*sorted(ALIASABLE_KINDS), EntityStatus.ALIVE]
    if scope is not None:
        sql += " AND scope = ?"
        args.append(scope)
    rows = store.conn.execute(sql, args).fetchall()

    linked = sum(_link_aliases(graph, members) for members in _alias_groups(rows))
    if backend is not None:
        linked += _model_aliases(store, graph, rows, backend)
    return linked
