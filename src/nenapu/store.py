"""Fact storage, decay, contradiction detection, and recall ranking.

The rules that make this different from a notes file:

  * Confidence decays. A claim you have not re-verified in six months is not
    worth what it was the day it was written.
  * Writes are checked against what is already known. Two facts that share a
    `key` in the same scope cannot both be active with different values.
  * Recall ranks on belief, not just lexical match. A stale, unverified,
    never-used fact loses to a fresh verified one even if it matches better.
"""

from __future__ import annotations

import contextlib
import hashlib
import math
import os
import random
import re
import sqlite3
import subprocess
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Iterable, Sequence

from .graph import Graph
from .models import (
    HALF_LIFE_DAYS,
    ORIGIN_WEIGHT,
    Conflict,
    Decay,
    Fact,
    Kind,
    Origin,
    Status,
    VerifyStatus,
    now,
    row_to_fact,
)
from .outcomes import Ledger, outcome_signal
from .db import commit
from .db import transaction as db_transaction

DAY = 86400.0
CONFIDENCE_FLOOR = 0.05

# Write contention: retry with jittered backoff rather than surfacing a lock
# error to a user who only asked to remember something.
LOCK_RETRIES = 6
LOCK_BACKOFF = 0.05

# A failed executable check is the strongest possible staleness signal, so it
# hits harder than any amount of elapsed time.
VERIFY_FAIL_PENALTY = 0.1
VERIFY_PASS_BONUS = 1.15

# A fact whose foundation was falsified is not wrong yet — it is unsupported.
# Heavy penalty, short of the outright collapse a failed check causes.
SUSPECT_PENALTY = 0.35

# A model confirming a fact against evidence is real information, but weaker
# than running a command that proves it. Each soft confirmation costs a little
# base confidence, so a fact sustained only by model opinion still declines.
SOFT_VERIFY_DISCOUNT = 0.9


def project_scope(cwd: str) -> str:
    """A stable scope id for the repo containing `cwd`, or `cwd` itself
    outside a repo. Hashes the absolute path so two clones sharing a
    directory name (`~/work/backend`, `~/other/backend`) do not collide.
    """
    path = Path(cwd).resolve()
    root = path
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            cwd=path, capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0 and result.stdout.strip():
            root = Path(result.stdout.strip()).resolve()
    except (OSError, subprocess.SubprocessError):
        pass
    digest = hashlib.sha1(str(root).encode()).hexdigest()[:8]
    return f"repo:{root.name}@{digest}"


def scope_for(kind: str, cwd: str | None = None) -> str:
    """Which tier a fact of this kind belongs in.

    Two tiers, one rule, in one place: how the user wants to be worked with
    travels with them, and what a repository does stays in the repository.
    """
    if kind in (Kind.USER, Kind.FEEDBACK):
        return "global"
    return project_scope(cwd or os.getcwd())


def decay_factor(decay_class: str, age_seconds: float) -> float:
    half_life = HALF_LIFE_DAYS.get(decay_class, HALF_LIFE_DAYS[Decay.MEDIUM])
    if half_life <= 0:  # immutable
        return 1.0
    if age_seconds <= 0:
        return 1.0
    return 0.5 ** ((age_seconds / DAY) / half_life)


def effective_confidence(fact: Fact, at: float | None = None) -> float:
    """What we should actually believe right now, all evidence folded in.

    Four independent signals, because each catches what the others miss:
    how the claim was obtained, how long since anyone checked, whether its
    check still passes, whether acting on it has gone well, and whether the
    thing it rests on still stands.
    """
    at = at or now()
    anchor = fact.last_verified_at or fact.created_at
    base = fact.confidence * ORIGIN_WEIGHT.get(fact.origin, 0.7)
    score = base * decay_factor(fact.decay_class, at - anchor)

    if fact.verify_status == VerifyStatus.PASS:
        score *= VERIFY_PASS_BONUS
    elif fact.verify_status == VerifyStatus.FAIL:
        score *= VERIFY_FAIL_PENALTY

    score *= outcome_signal(fact.good_recalls, fact.bad_recalls)

    if fact.status == Status.DISPUTED:
        score *= 0.5
    elif fact.status == Status.SUSPECT:
        score *= SUSPECT_PENALTY

    return max(CONFIDENCE_FLOOR, min(1.0, score))


_WORD = re.compile(r"[a-z0-9]+")


def _normalize_value(text: str) -> str:
    return " ".join(_WORD.findall(text.lower()))


_NUM = re.compile(r"-?\d+(?:\.\d+)?")

# Words that carry no claim. Two facts differing only in these are the same
# fact phrased twice.
_FILLER = {
    "a", "an", "the", "is", "are", "was", "were", "be", "to", "of", "in", "on",
    "at", "for", "and", "or", "our", "we", "it", "its", "this", "that", "uses",
    "use", "used", "using", "runs", "run", "lives", "set", "currently",
}


def _content(text: str) -> set[str]:
    return set(_normalize_value(text).split()) - _FILLER


def looks_contradictory(a: str, b: str) -> tuple[bool, str]:
    """Do two facts *about the same subject* disagree?

    Only ever asked of facts sharing a `key`, and that changes the burden of
    proof. A shared key is the writer declaring these are values for one
    subject, so two different values are a conflict by default — not something
    that has to clear a similarity bar. Getting this backwards is how
    "cache backend is redis" and "cache backend is memcached" end up both
    active and both believed.

    Deliberately not an LLM call: this runs on every write. Three cases are
    *not* conflicts — identical text, a rephrasing that differs only in filler,
    and an elaboration that keeps every content word and adds detail.
    """
    na, nb = _normalize_value(a), _normalize_value(b)
    if na == nb:
        return False, "identical"

    ca, cb = _content(a), _content(b)
    if ca == cb:
        return False, "same content, different phrasing"

    nums_a, nums_b = _NUM.findall(a), _NUM.findall(b)
    if nums_a and nums_b and nums_a != nums_b:
        return True, f"numeric mismatch: {nums_a} vs {nums_b}"

    if ca < cb or cb < ca:
        # One says everything the other says, plus more: an elaboration.
        return False, "elaboration, not disagreement"

    if not ca or not cb:
        return False, "nothing substantive to compare"

    differing = sorted((ca - cb) | (cb - ca))
    return True, f"same key, conflicting values: {differing}"


class Store:
    def __init__(self, conn: sqlite3.Connection) -> None:
        self.conn = conn
        self.graph = Graph(conn)
        self.ledger = Ledger(conn)

    def transaction(self):
        """Delegates to the connection-level primitive; see `db.transaction`."""
        return db_transaction(self.conn)

    # ---------- internals ----------

    def _journal(self, action: str, *, fact_id=None, skill_id=None, actor=None, detail=None):
        self.conn.execute(
            "INSERT INTO journal(action, fact_id, skill_id, actor, detail, created_at)"
            " VALUES (?,?,?,?,?,?)",
            (action, fact_id, skill_id, actor, detail, now()),
        )

    def _insert(self, fact: Fact) -> Fact:
        cur = self.conn.execute(
            """INSERT INTO facts(text, kind, scope, key, origin, origin_ref, session_id,
                   confidence, decay_class, verify_cmd, verify_expect, tags_csv, status,
                   created_at, updated_at, last_verified_at, verify_status, verify_last_run,
                   verify_detail, supersedes_id, superseded_by_id, distilled_into_id,
                   use_count, last_used_at, agent, occurrences)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                fact.text, fact.kind, fact.scope, fact.key, fact.origin, fact.origin_ref,
                fact.session_id, fact.confidence, fact.decay_class, fact.verify_cmd,
                fact.verify_expect, ",".join(fact.tags), fact.status, fact.created_at,
                fact.updated_at, fact.last_verified_at, fact.verify_status,
                fact.verify_last_run, fact.verify_detail, fact.supersedes_id,
                fact.superseded_by_id, fact.distilled_into_id, fact.use_count,
                fact.last_used_at, fact.agent, fact.occurrences,
            ),
        )
        return replace(fact, id=cur.lastrowid)

    # ---------- reads ----------

    def get(self, fact_id: int) -> Fact | None:
        row = self.conn.execute("SELECT * FROM facts WHERE id = ?", (fact_id,)).fetchone()
        return row_to_fact(row) if row else None

    def list_facts(
        self,
        scope: str | Sequence[str] | None = None,
        status: str | Sequence[str] | None = Status.ACTIVE,
        kind: str | None = None,
        limit: int = 500,
    ) -> list[Fact]:
        sql = "SELECT * FROM facts WHERE 1=1"
        args: list = []
        if scope:
            scopes = [scope] if isinstance(scope, str) else list(scope)
            sql += f" AND scope IN ({','.join('?' * len(scopes))})"
            args.extend(scopes)
        if status:
            statuses = [status] if isinstance(status, str) else list(status)
            sql += f" AND status IN ({','.join('?' * len(statuses))})"
            args.extend(statuses)
        if kind:
            sql += " AND kind = ?"
            args.append(kind)
        sql += " ORDER BY updated_at DESC LIMIT ?"
        args.append(limit)
        return [row_to_fact(r) for r in self.conn.execute(sql, args)]

    def find_by_key(self, scope: str, key: str) -> list[Fact]:
        rows = self.conn.execute(
            "SELECT * FROM facts WHERE scope = ? AND key = ? AND status IN ('active','disputed')",
            (scope, key),
        )
        return [row_to_fact(r) for r in rows]

    # ---------- write path ----------

    def write(
        self,
        fact: Fact,
        *,
        actor: str = "agent",
        derived_from: list[int] | None = None,
        infer_edges: bool = True,
    ) -> tuple[Fact, list[Conflict]]:
        """Insert a fact, resolving contradictions against what is already known.

        Returns the stored fact plus any conflicts that were recorded. A caller
        that gets a non-empty conflict list has learned something worth showing
        the user — that is the point.
        """
        with self.transaction():
            return self._write_locked(fact, actor, derived_from, infer_edges)

    def _write_locked(self, fact, actor, derived_from, infer_edges):
        dup = self.conn.execute(
            "SELECT * FROM facts WHERE scope = ? AND text = ? AND status = 'active'",
            (fact.scope, fact.text),
        ).fetchone()
        if dup:
            # Same claim asserted again is evidence, not noise: re-anchor it.
            existing = row_to_fact(dup)
            self.touch_verified(existing.id, confidence=max(existing.confidence, fact.confidence))
            # Re-asserting still tells us something about structure — this
            # session reached the same conclusion, possibly from new premises.
            for parent_id in derived_from or []:
                self.graph.link(parent_id, existing.id)
            if infer_edges:
                self.graph.infer_edges_for(existing.id, fact.session_id)
            self._journal("reassert", fact_id=existing.id, actor=actor)
            commit(self.conn)
            return self.get(existing.id), []

        conflicts: list[Conflict] = []
        stored = self._insert(fact)

        for parent_id in derived_from or []:
            self.graph.link(parent_id, stored.id)
        if infer_edges:
            # What this session just recalled is the best available evidence for
            # what the conclusion was built on.
            self.graph.infer_edges_for(stored.id, fact.session_id)

        if fact.key:
            for other in self.find_by_key(fact.scope, fact.key):
                if other.id == stored.id:
                    continue
                clash, detail = looks_contradictory(stored.text, other.text)
                if not clash:
                    continue
                conflicts.append(self._resolve(stored, other, detail))

        self._journal("write", fact_id=stored.id, actor=actor, detail=fact.key)
        commit(self.conn)
        return self.get(stored.id), conflicts

    def _resolve(self, new: Fact, old: Fact, detail: str) -> Conflict:
        """Newer evidence wins only if it is at least as trustworthy as the old.

        Otherwise the new fact is parked as `disputed` rather than silently
        overwriting a stronger claim — an agent inference should not be able to
        quietly delete something the user told us.
        """
        new_score = effective_confidence(new)
        old_score = effective_confidence(old)

        if new_score >= old_score:
            self.conn.execute(
                "UPDATE facts SET status = ?, superseded_by_id = ?, updated_at = ? WHERE id = ?",
                (Status.SUPERSEDED, new.id, now(), old.id),
            )
            self.conn.execute(
                "UPDATE facts SET supersedes_id = ? WHERE id = ?", (old.id, new.id)
            )
            # Implicit grading signal: anyone who acted on the old fact in the
            # last few hours acted on something we now know was wrong.
            self.ledger.blame_recent_recalls(
                old.id, source="correction", note=f"superseded by #{new.id}"
            )
            # The 6h window above misses a recall injected at session start
            # into a session that ran longer than that. The session doing the
            # superseding is read straight off the new fact, no window needed.
            self.ledger.blame_session_recalls(
                old.id, new.session_id, source="correction", note=f"superseded by #{new.id}"
            )
            self.graph.cascade_falsification(old.id, "superseded")
            resolution = "superseded"
        else:
            self.conn.execute(
                "UPDATE facts SET status = ?, updated_at = ? WHERE id = ?",
                (Status.DISPUTED, now(), new.id),
            )
            resolution = "disputed"

        c = Conflict(
            fact_id=new.id,
            other_id=old.id,
            key=new.key,
            detail=f"{detail} (new={new_score:.2f} old={old_score:.2f})",
            resolution=resolution,
        )
        cur = self.conn.execute(
            "INSERT INTO conflicts(fact_id, other_id, key, detail, resolution, created_at)"
            " VALUES (?,?,?,?,?,?)",
            (c.fact_id, c.other_id, c.key, c.detail, c.resolution, c.created_at),
        )
        return replace(c, id=cur.lastrowid)

    def touch_verified(self, fact_id: int, *, confidence: float | None = None) -> None:
        """Re-anchor a fact's decay clock. Called on re-assertion and on a pass."""
        if confidence is None:
            self.conn.execute(
                "UPDATE facts SET last_verified_at = ?, updated_at = ? WHERE id = ?",
                (now(), now(), fact_id),
            )
        else:
            self.conn.execute(
                "UPDATE facts SET last_verified_at = ?, updated_at = ?, confidence = ?"
                " WHERE id = ?",
                (now(), now(), confidence, fact_id),
            )
        commit(self.conn)

    def note_recurrence(self, fact_id: int, *, actor: str = "observer") -> Fact | None:
        """The same claim, said again. Bumps the counter and re-anchors decay.

        The count is what separates a preference stated once from one the user
        has now had to repeat five times, which is the most actionable signal
        the store holds and the one near-duplicate rows used to destroy: five
        rows each saying "once" instead of one row saying "five times".
        """
        fact = self.get(fact_id)
        if fact is None:
            return None
        with self.transaction():
            self.conn.execute(
                "UPDATE facts SET occurrences = occurrences + 1, last_verified_at = ?,"
                " updated_at = ? WHERE id = ?",
                (now(), now(), fact_id),
            )
            self._journal("recur", fact_id=fact_id, actor=actor)
            commit(self.conn)
        return self.get(fact_id)

    def revise(
        self,
        fact_id: int,
        *,
        text: str,
        kind: str | None = None,
        key: str | None = None,
        actor: str = "observer",
    ) -> Fact | None:
        """Re-word an existing fact in place rather than storing it twice.

        Refuses to touch a `user_stated` fact: an agent's reading of a session
        must never silently overwrite what the user said, which is the oldest
        invariant in the store. The recurrence is still counted, because being
        told the same thing again is true whoever ends up owning the wording.
        """
        fact = self.get(fact_id)
        if fact is None:
            return None
        if fact.origin == Origin.USER_STATED:
            return self.note_recurrence(fact_id, actor=actor)
        with self.transaction():
            self.conn.execute(
                "UPDATE facts SET text = ?, kind = ?, key = ?, occurrences = occurrences + 1,"
                " last_verified_at = ?, updated_at = ? WHERE id = ?",
                (text, kind or fact.kind, key if key is not None else fact.key,
                 now(), now(), fact_id),
            )
            self._journal("revise", fact_id=fact_id, actor=actor, detail=text[:80])
            commit(self.conn)
        return self.get(fact_id)

    def soft_verify(self, fact_id: int, *, actor: str = "audit") -> None:
        """Re-anchor a fact on evidence read by a model rather than a command.

        Weaker than `touch_verified`, which a passing check earns: the base
        confidence is discounted each time, so a fact kept alive purely by
        model confirmation still fades — just slowly enough that a true fact
        does not rot to the floor while nobody is looking.

        Without this the audit is a one-way ratchet: able to dispute and
        retire, never to reaffirm, so every fact decays to the minimum
        regardless of whether it is still true.
        """
        fact = self.get(fact_id)
        if not fact:
            return
        self.conn.execute(
            "UPDATE facts SET last_verified_at = ?, updated_at = ?, confidence = ? WHERE id = ?",
            (now(), now(), max(0.3, fact.confidence * SOFT_VERIFY_DISCOUNT), fact_id),
        )
        self._journal("soft_verify", fact_id=fact_id, actor=actor)
        commit(self.conn)

    def set_status(self, fact_id: int, status: str, *, actor: str = "agent",
                   session_id: str | None = None) -> None:
        self.conn.execute(
            "UPDATE facts SET status = ?, updated_at = ? WHERE id = ?", (status, now(), fact_id)
        )
        self._journal("status", fact_id=fact_id, actor=actor, detail=status)
        commit(self.conn)

        if status in (Status.RETIRED, Status.DISPUTED):
            self.graph.cascade_falsification(fact_id, status)
            if status == Status.RETIRED:
                self.ledger.blame_recent_recalls(
                    fact_id, source="correction", note="fact retired"
                )
                # A retire has no superseding fact to read a session off, so
                # the caller passes it explicitly when one applies.
                self.ledger.blame_session_recalls(
                    fact_id, session_id, source="correction", note="fact retired"
                )

    def forget(self, fact_id: int, *, actor: str = "user") -> None:
        self.set_status(fact_id, Status.RETIRED, actor=actor)

    def forget_all(self, *, scope: str | None = None, kind: str | None = None,
                   actor: str = "user") -> int:
        """Retire every active fact, optionally narrowed by scope or kind.

        Retired rather than deleted, like every other forget here: the row
        stays, the journal records who did it, and `nenapu audit` can still
        explain why the store looks the way it does. `purge` is the one that
        actually removes rows, and it is a separate word on purpose.

        One statement rather than a loop over `set_status`: retiring a hundred
        facts individually would fire a hundred cascades, and cascading a
        falsification into facts that are themselves being retired in the same
        breath is work whose result nobody will ever read.
        """
        where = ["status = ?"]
        args: list = [Status.ACTIVE]
        if scope:
            where.append("scope = ?")
            args.append(scope)
        if kind:
            where.append("kind = ?")
            args.append(kind)
        clause = " AND ".join(where)

        with self.transaction():
            count = self.conn.execute(
                f"SELECT COUNT(*) c FROM facts WHERE {clause}", args).fetchone()["c"]
            if count:
                self.conn.execute(
                    f"UPDATE facts SET status = ?, updated_at = ? WHERE {clause}",
                    [Status.RETIRED, now(), *args])
                self._journal("forget-all", actor=actor,
                              detail=f"{count} fact(s)"
                                     + (f" in scope {scope}" if scope else "")
                                     + (f" of kind {kind}" if kind else ""))
        return count

    def purge(self, *, scope: str | None = None, actor: str = "user") -> int:
        """Delete facts outright, and everything hanging off them.

        The one operation here that destroys history. Edges, recalls and
        conflicts go too — a graph edge to a row that no longer exists is worse
        than no edge, and a recall that cannot name what it recalled cannot be
        graded. The journal keeps a line saying it happened, because a store
        that cannot say why it is empty is indistinguishable from a broken one.
        """
        where, args = ("WHERE scope = ?", [scope]) if scope else ("", [])
        with self.transaction():
            ids = [r["id"] for r in
                   self.conn.execute(f"SELECT id FROM facts {where}", args).fetchall()]
            if ids:
                marks = ",".join("?" * len(ids))
                self.conn.execute(
                    f"DELETE FROM fact_edges WHERE parent_id IN ({marks})"
                    f" OR child_id IN ({marks})", ids + ids)
                self.conn.execute(f"DELETE FROM recalls WHERE fact_id IN ({marks})", ids)
                self.conn.execute(
                    f"DELETE FROM conflicts WHERE fact_id IN ({marks})"
                    f" OR other_id IN ({marks})", ids + ids)
                self.conn.execute(f"DELETE FROM facts WHERE id IN ({marks})", ids)
            self._journal("purge", actor=actor,
                          detail=f"{len(ids)} fact(s)"
                                 + (f" in scope {scope}" if scope else ""))
        return len(ids)

    def mark_used(self, fact_ids: Iterable[int]) -> None:
        ids = list(fact_ids)
        if not ids:
            return
        # One transaction, not one per row: in autocommit each statement is its
        # own durable write, so eight bumps cost eight fsyncs.
        with self.transaction():
            self.conn.executemany(
                "UPDATE facts SET use_count = use_count + 1, last_used_at = ? WHERE id = ?",
                [(now(), i) for i in ids],
            )

    # ---------- recall ----------

    def _document_frequency(self, term: str) -> int:
        """How many facts contain `term`. A term in most of the store narrows
        nothing, and matching on it is how twelve slots fill with everything."""
        try:
            row = self.conn.execute(
                "SELECT COUNT(*) c FROM facts_fts WHERE facts_fts MATCH ?", (f'"{term}"',)
            ).fetchone()
        except sqlite3.OperationalError:
            return 0
        return row["c"] if row else 0

    def _plan_query(self, raw: str) -> QueryPlan:
        """Turn user text into required and optional terms.

        Required terms are the rarest ones that actually occur in the store: a
        term nothing contains cannot be required without emptying the result,
        and a term almost everything contains cannot discriminate.
        """
        phrases, terms = _parse_query(raw)
        if not phrases and not terms:
            return QueryPlan()
        if phrases:
            # A phrase is already a strong constraint; the loose terms beside it
            # stay optional so they refine the ranking without narrowing it.
            present = [t for t in terms if self._document_frequency(t) > 0]
            return QueryPlan(phrases=tuple(phrases), optional=tuple(present), is_search=True)

        frequency = {t: self._document_frequency(t) for t in terms}
        present = [t for t in terms if frequency[t] > 0]
        total = self.conn.execute("SELECT COUNT(*) c FROM facts").fetchone()["c"]
        discriminating = [t for t in present if not _too_common(frequency[t], total)]
        if not discriminating:
            # Every term is common: the query is still what the user asked for.
            discriminating = present
        required = sorted(discriminating, key=lambda t: frequency[t])[:MAX_REQUIRED_TERMS]
        optional = [t for t in present if t not in required]
        return QueryPlan(required=tuple(required), optional=tuple(optional), is_search=True)

    def _candidate_pool(
        self,
        plan: QueryPlan,
        *,
        statuses: Sequence[str],
        scope: str | Sequence[str] | None,
        pool: int,
    ) -> list[_Candidate]:
        """Lexical hits unioned with the best-believed hits for the same query.

        R1: the pool used to be `ORDER BY rank LIMIT limit*5`, which made
        confidence a re-ranker over a lexical pool and never a retriever. A
        strongly believed fact ranked 51st lexically could not surface however
        well it answered the query. Both pools match the same expression, so
        confidence widens what is considered without inventing hits.
        """
        expression = plan.match_expression()
        if not expression:
            return []

        sql = (
            f"SELECT f.*, bm25(facts_fts, {_BM25_TEXT_WEIGHT}, {_BM25_KEY_WEIGHT},"
            f" {_BM25_TAG_WEIGHT}) AS rank FROM facts_fts"
            " JOIN facts f ON f.id = facts_fts.rowid"
            " WHERE facts_fts MATCH ? AND f.status IN ("
            + ",".join("?" * len(statuses)) + ")"
        )
        args: list = [expression, *statuses]
        if scope:
            scopes = [scope] if isinstance(scope, str) else list(scope)
            sql += f" AND f.scope IN ({','.join('?' * len(scopes))})"
            args.extend(scopes)

        orderings = (
            "rank",
            "f.confidence DESC, COALESCE(f.last_verified_at, f.created_at) DESC",
        )
        found: dict[int, _Candidate] = {}
        for order in orderings:
            try:
                rows = self.conn.execute(f"{sql} ORDER BY {order} LIMIT ?", [*args, pool])
                for row in rows:
                    if row["id"] in found:
                        continue
                    found[row["id"]] = _candidate_from(row_to_fact(row), row["rank"], plan)
            except sqlite3.OperationalError:
                return []
        return list(found.values())

    def search(
        self,
        query: str,
        *,
        scope: str | Sequence[str] | None = None,
        limit: int = 10,
        min_confidence: float = 0.0,
        include_disputed: bool = True,
        mark_used: bool = True,
        session_id: str | None = None,
        log_recall: bool = True,
    ) -> list[tuple[Fact, float, dict]]:
        """Rank by lexical match *and* current believability.

        Returns (fact, score, explain) so a caller can show the user why a
        memory surfaced — and why a stale one did not.
        """
        statuses = ["active", "suspect", "disputed"] if include_disputed else ["active"]

        # R1: the query is planned, not OR-joined. `plan.match_expression()`
        # requires the discriminating terms and leaves the rest as bm25 signal,
        # so a twelve-term query no longer matches everything sharing one word.
        plan = self._plan_query(query)
        candidates = self._candidate_pool(
            plan, statuses=statuses, scope=scope, pool=limit * 5
        )

        # R1: recency is only an answer to "no query at all". An unmatched query
        # used to return arbitrary recent facts at a flat 0.3 *presented as
        # hits*, which the MCP path then logged as recalls with a query
        # attached, poisoning the population the gate reads to measure ranking.
        fallback = not plan.is_search
        if fallback:
            candidates = [
                _Candidate(fact=f, lexical=0.0, key_match=False, tag_match=False)
                for f in self.list_facts(scope=scope, status=statuses, limit=limit * 5)
            ]

        at = now()
        scored: list[tuple[Fact, float, dict]] = []
        for candidate in candidates:
            fact, lex = candidate.fact, candidate.lexical
            conf = effective_confidence(fact, at)
            usage = min(1.0, math.log1p(fact.use_count) / math.log(11))  # saturates ~10 uses
            score = 0.5 * lex + 0.4 * conf + 0.1 * usage
            if conf < min_confidence:
                continue
            scored.append(
                (
                    fact,
                    score,
                    {
                        "lexical": round(lex, 3),
                        "confidence": round(conf, 3),
                        "usage": round(usage, 3),
                        "key_match": candidate.key_match,
                        "tag_match": candidate.tag_match,
                        "fallback": fallback,
                        "age_days": round((at - (fact.last_verified_at or fact.created_at)) / DAY, 1),
                        "verify_status": fact.verify_status,
                        "track_record": f"{fact.good_recalls}/{fact.good_recalls + fact.bad_recalls}",
                        "suspect_reason": fact.suspect_reason,
                    },
                )
            )

        scored.sort(key=lambda t: t[1], reverse=True)
        top = scored[:limit]
        if (mark_used or log_recall) and top:
            # Recall is the hottest path in the system; its bookkeeping should
            # cost one commit, not one per row.
            with self.transaction():
                if mark_used:
                    self.mark_used(f.id for f, _, _ in top)
                if log_recall:
                    recall_ids = self.ledger.log_many(top, session_id=session_id, query=query)
                    for (_f, _s, why), rid in zip(top, recall_ids):
                        why["recall_id"] = rid
        return top

    # ---------- reporting ----------

    def stats(self, scope: str | None = None) -> dict:
        where, args = ("WHERE scope = ?", [scope]) if scope else ("", [])
        rows = self.conn.execute(
            f"SELECT status, COUNT(*) c FROM facts {where} GROUP BY status", args
        ).fetchall()
        by_status = {r["status"]: r["c"] for r in rows}

        active = self.list_facts(scope=scope, limit=10_000)
        confs = [effective_confidence(f) for f in active]
        stale = sum(1 for c in confs if c < 0.3)
        return {
            "by_status": by_status,
            "active": len(active),
            "stale_active": stale,
            "verifiable": sum(1 for f in active if f.verify_cmd),
            "failing_verification": sum(1 for f in active if f.verify_status == VerifyStatus.FAIL),
            "mean_confidence": round(sum(confs) / len(confs), 3) if confs else 0.0,
            "conflicts": self.conn.execute("SELECT COUNT(*) c FROM conflicts").fetchone()["c"],
            "suspect": self.conn.execute(
                "SELECT COUNT(*) c FROM facts WHERE status = 'suspect'"
            ).fetchone()["c"],
            "edges": self.conn.execute("SELECT COUNT(*) c FROM fact_edges").fetchone()["c"],
            **self.ledger.stats(),
        }


_FTS_SAFE = re.compile(r"[^\w\s]")


def _fts_query(raw: str) -> str:
    """FTS5 MATCH syntax is its own little language; user text is not."""
    cleaned = _FTS_SAFE.sub(" ", raw or "").strip()
    terms = [t for t in cleaned.split() if len(t) > 1]
    if not terms:
        return ""
    return " OR ".join(f'"{t}"' for t in terms)


# ---------- R1 · query planning and candidate generation ----------

_PHRASE = re.compile(r'"([^"]*)"')

# Words that carry no retrieval signal: they appear in almost any sentence, so
# a fact matching one of them has told you nothing about the query.
STOPWORDS = frozenset("""
a an the and or but if then else than that this these those of in on at to for
from with without by as is are was were be been being it its do does did done
how what when where which who why can could should would will have has had not
no yes i me my we our you your they them their he she his her about into over
under out up down again more most some any all there here also very just
""".split())

# A term in more than half the store cannot discriminate between facts in it.
COMMON_TERM_FRACTION = 0.5
# ... but only once there is a store to speak of: on a handful of facts every
# term looks common, and dropping it would leave the query with nothing.
COMMON_TERM_MIN_FACTS = 10
# Two required terms is the smallest constraint that rejects a one-word
# coincidence while still answering a query whose other terms are noise.
MAX_REQUIRED_TERMS = 2
# `relevant_memory` can feed forty salient terms from a whole session; the
# planner must not turn one query into forty.
MAX_QUERY_TERMS = 24

# bm25 column weights, in the order the FTS table declares them (text, key,
# tags_csv). A query matching a fact's key or tag is a much stronger signal
# than one matching a word in its prose, and bm25 treated all three alike.
_BM25_TEXT_WEIGHT = 1.0
_BM25_KEY_WEIGHT = 5.0
_BM25_TAG_WEIGHT = 4.0

# Applied in bm25 rank space (negative is better) rather than to the squashed
# score, which saturates near 1.0 and would swallow the difference.
KEY_MATCH_BOOST = 2.0
TAG_MATCH_BOOST = 1.5


@dataclass(frozen=True)
class QueryPlan:
    """What a query is allowed to match.

    `is_search` distinguishes "the user asked for nothing" from "the user asked
    for something the store does not hold". The first may answer with recent
    facts; the second must answer with nothing.
    """

    phrases: tuple[str, ...] = ()
    required: tuple[str, ...] = ()
    optional: tuple[str, ...] = ()
    is_search: bool = False

    def match_expression(self) -> str:
        """FTS5 MATCH text: required terms conjoined, optional terms kept in an
        OR group so bm25 still scores them without narrowing the result."""
        must = [f'"{p}"' for p in self.phrases] + [f'"{t}"' for t in self.required]
        if not must:
            return ""
        expression = " AND ".join(must)
        if self.optional:
            everything = must + [f'"{t}"' for t in self.optional]
            expression += " AND (" + " OR ".join(everything) + ")"
        return expression

    def words(self) -> set[str]:
        """Every word the query mentions, for matching against key and tags."""
        found = {t.lower() for t in self.required + self.optional}
        for phrase in self.phrases:
            found.update(w.lower() for w in _FTS_SAFE.sub(" ", phrase).split())
        return found


@dataclass(frozen=True)
class _Candidate:
    fact: Fact
    lexical: float
    key_match: bool
    tag_match: bool


def _parse_query(raw: str) -> tuple[list[str], list[str]]:
    """Split user text into quoted phrases and loose terms.

    Two words next to each other are a different claim from the same two words
    in one paragraph, so quotes survive as phrases instead of being scrubbed.
    """
    text = raw or ""
    phrases = [p.strip() for p in _PHRASE.findall(text) if p.strip()]
    cleaned = _FTS_SAFE.sub(" ", _PHRASE.sub(" ", text))
    terms: list[str] = []
    for word in cleaned.split():
        lowered = word.lower()
        if len(lowered) < 2 or lowered in STOPWORDS or lowered in terms:
            continue
        terms.append(lowered)
    return phrases, terms[:MAX_QUERY_TERMS]


def _too_common(frequency: int, total: int) -> bool:
    return total >= COMMON_TERM_MIN_FACTS and frequency > total * COMMON_TERM_FRACTION


def _field_words(text: str | None) -> set[str]:
    return {w.lower() for w in _FTS_SAFE.sub(" ", text or "").split()}


def _candidate_from(fact: Fact, rank: float, plan: QueryPlan) -> _Candidate:
    """Score one row, boosting matches that landed in `key` or `tags_csv`."""
    asked = plan.words()
    key_match = bool(asked & _field_words(fact.key))
    tag_match = bool(asked & _field_words(",".join(fact.tags or [])))
    if key_match:
        rank -= KEY_MATCH_BOOST
    if tag_match:
        rank -= TAG_MATCH_BOOST
    # bm25 is negative-better; flip and squash into 0..1.
    return _Candidate(
        fact=fact,
        lexical=1.0 / (1.0 + math.exp(rank)),
        key_match=key_match,
        tag_match=tag_match,
    )
