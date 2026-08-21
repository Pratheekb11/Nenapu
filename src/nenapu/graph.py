"""The belief network: what a fact rests on, and what falls when it falls.

Every memory system stores facts as a flat list. That is fine until a root
fact goes wrong. An agent that once learned "auth lives in services/auth" will
have derived a dozen conclusions from it; when the repo moves, the root fact
fails its check and every derived conclusion keeps its old confidence and
quietly misroutes work.

Two ideas make the graph work in practice:

  * **Falsification cascades.** Retiring or failing a fact marks everything
    derived from it `suspect` — not wrong, but no longer standing on anything.
  * **Edges build themselves.** Nobody will hand-maintain a dependency graph.
    When an agent recalls facts and then writes a new one in the same session,
    that is observable causality, and the edge is inferred from it.
"""

from __future__ import annotations

import sqlite3
from collections import deque

from .models import Edge, EdgeKind, EdgeSource, Outcome, Status, now
from .db import commit, transaction

# An inferred edge is a guess from co-occurrence, so it cascades more weakly
# than one the agent declared outright.
INFERRED_WEIGHT = 0.6
DECLARED_WEIGHT = 1.0

# How much of a session's recalls become parents of a fact written in it.
# Beyond this the co-occurrence signal is too diffuse to mean anything.
MAX_INFERRED_PARENTS = 3

# A cascade stops here. Deep chains are usually a sign of runaway inference,
# and an unbounded walk on a cyclic graph never returns.
MAX_CASCADE_DEPTH = 6


class Graph:
    def __init__(self, conn: sqlite3.Connection) -> None:
        self.conn = conn

    # ---------- edges ----------

    def link(
        self,
        parent_id: int,
        child_id: int,
        *,
        kind: str = EdgeKind.DERIVED_FROM,
        source: str = EdgeSource.DECLARED,
        weight: float | None = None,
    ) -> Edge | None:
        """Record that `child` rests on `parent`. Self-links and dupes are ignored."""
        if parent_id == child_id:
            return None
        if weight is None:
            weight = DECLARED_WEIGHT if source == EdgeSource.DECLARED else INFERRED_WEIGHT

        edge = Edge(parent_id=parent_id, child_id=child_id, kind=kind, source=source,
                    weight=weight)
        try:
            cur = self.conn.execute(
                "INSERT INTO fact_edges(parent_id, child_id, kind, source, weight, created_at)"
                " VALUES (?,?,?,?,?,?)",
                (parent_id, child_id, kind, source, weight, edge.created_at),
            )
        except sqlite3.IntegrityError:
            return None  # already linked
        commit(self.conn)
        edge.id = cur.lastrowid
        return edge

    def parents(self, fact_id: int) -> list[tuple[int, str, float]]:
        rows = self.conn.execute(
            "SELECT parent_id, source, weight FROM fact_edges WHERE child_id = ?", (fact_id,)
        )
        return [(r["parent_id"], r["source"], r["weight"]) for r in rows]

    def children(self, fact_id: int) -> list[tuple[int, str, float]]:
        rows = self.conn.execute(
            "SELECT child_id, source, weight FROM fact_edges WHERE parent_id = ?", (fact_id,)
        )
        return [(r["child_id"], r["source"], r["weight"]) for r in rows]

    def descendants(self, fact_id: int, *, max_depth: int = MAX_CASCADE_DEPTH) -> list[int]:
        """Everything downstream, breadth-first, cycle-safe."""
        seen: set[int] = set()
        order: list[int] = []
        queue: deque[tuple[int, int]] = deque([(fact_id, 0)])
        while queue:
            current, depth = queue.popleft()
            if depth >= max_depth:
                continue
            for child, _source, _weight in self.children(current):
                if child in seen:
                    continue
                seen.add(child)
                order.append(child)
                queue.append((child, depth + 1))
        return order

    # ---------- cascade ----------

    def cascade_falsification(self, fact_id: int, reason: str) -> list[int]:
        """A fact stopped being trustworthy; put its dependents in doubt.

        Only `active` descendants are touched. Facts already superseded or
        retired have their own story, and a fact already suspect stays suspect
        under its first recorded reason.
        """
        affected: list[int] = []
        with transaction(self.conn):
            return self._cascade_locked(fact_id, reason, affected)

    def _cascade_locked(self, fact_id: int, reason: str, affected: list[int]) -> list[int]:
        for child_id in self.descendants(fact_id):
            row = self.conn.execute(
                "SELECT status FROM facts WHERE id = ?", (child_id,)
            ).fetchone()
            if not row or row["status"] != Status.ACTIVE:
                continue
            self.conn.execute(
                "UPDATE facts SET status=?, suspect_since=?, suspect_reason=?, updated_at=?"
                " WHERE id=?",
                (Status.SUSPECT, now(), f"rests on #{fact_id}: {reason}", now(), child_id),
            )
            self.conn.execute(
                "INSERT INTO journal(action, fact_id, actor, detail, created_at)"
                " VALUES ('cascade', ?, 'graph', ?, ?)",
                (child_id, f"suspect via #{fact_id}: {reason}", now()),
            )
            affected.append(child_id)
        commit(self.conn)
        return affected

    def clear_suspicion(self, fact_id: int) -> list[int]:
        """A root recovered (its check passes again). Reinstate dependents that
        are not propped up by some *other* broken parent."""
        restored: list[int] = []
        for child_id in self.descendants(fact_id):
            row = self.conn.execute(
                "SELECT status, suspect_reason FROM facts WHERE id = ?", (child_id,)
            ).fetchone()
            if not row or row["status"] != Status.SUSPECT:
                continue
            if self._has_broken_parent(child_id, ignore=fact_id):
                continue
            self.conn.execute(
                "UPDATE facts SET status=?, suspect_since=NULL, suspect_reason=NULL,"
                " updated_at=? WHERE id=?",
                (Status.ACTIVE, now(), child_id),
            )
            restored.append(child_id)
        commit(self.conn)
        return restored

    def _has_broken_parent(self, fact_id: int, *, ignore: int | None = None) -> bool:
        for parent_id, _source, _weight in self.parents(fact_id):
            if parent_id == ignore:
                continue
            row = self.conn.execute(
                "SELECT status, verify_status FROM facts WHERE id = ?", (parent_id,)
            ).fetchone()
            if not row:
                continue
            if row["status"] in (Status.RETIRED, Status.SUPERSEDED, Status.SUSPECT):
                return True
            if row["verify_status"] == "fail":
                return True
        return False

    # ---------- automatic edge discovery ----------

    def infer_edges_for(
        self, child_id: int, session_id: str | None, *, within_seconds: float = 3600.0
    ) -> list[Edge]:
        """Link a newly written fact to what the session actually used.

        This is the part that makes the graph survive contact with reality: an
        agent that recalled A and B and then concluded C almost certainly used
        A and B to get there, and no agent will declare that by hand.

        Prefers recalls this session graded `good` over raw co-occurrence: on
        a store where every session starts with a twelve-fact context dump,
        "whatever was recalled in the last hour" links a conclusion to the
        whole dump rather than to what was used. The co-occurrence fallback
        only applies when the session has graded nothing at all, so the
        feature does not vanish on every store that has never graded a
        recall — which today is every store.
        """
        if not session_id:
            return []

        graded_something = self.conn.execute(
            "SELECT 1 FROM recalls WHERE session_id = ? AND outcome != ? LIMIT 1",
            (session_id, Outcome.PENDING),
        ).fetchone()

        if graded_something:
            rows = self.conn.execute(
                "SELECT DISTINCT fact_id FROM recalls WHERE session_id = ? AND outcome = ?"
                " AND fact_id != ? ORDER BY created_at DESC LIMIT ?",
                (session_id, Outcome.GOOD, child_id, MAX_INFERRED_PARENTS),
            ).fetchall()
        else:
            rows = self.conn.execute(
                "SELECT DISTINCT fact_id FROM recalls WHERE session_id = ? AND created_at >= ?"
                " AND fact_id != ? ORDER BY created_at DESC LIMIT ?",
                (session_id, now() - within_seconds, child_id, MAX_INFERRED_PARENTS),
            ).fetchall()

        edges = []
        for row in rows:
            edge = self.link(row["fact_id"], child_id, source=EdgeSource.INFERRED)
            if edge:
                edges.append(edge)
        return edges

    # ---------- explanation ----------

    def why(self, fact_id: int, *, depth: int = 3) -> dict:
        """The belief chain behind a fact, for showing a human why it is trusted."""
        row = self.conn.execute("SELECT * FROM facts WHERE id = ?", (fact_id,)).fetchone()
        if not row:
            return {"error": f"no fact {fact_id}"}

        def node(fid: int, level: int) -> dict:
            r = self.conn.execute(
                "SELECT id, text, status, verify_status FROM facts WHERE id = ?", (fid,)
            ).fetchone()
            if not r:
                return {"id": fid, "missing": True}
            out = {
                "id": r["id"], "text": r["text"], "status": r["status"],
                "verify_status": r["verify_status"],
            }
            if level < depth:
                rests_on = [
                    {**node(pid, level + 1), "via": source}
                    for pid, source, _w in self.parents(fid)
                ]
                if rests_on:
                    out["rests_on"] = rests_on
            return out

        return {
            **node(fact_id, 0),
            "suspect_reason": row["suspect_reason"],
            "supports": [c for c, _s, _w in self.children(fact_id)],
        }
