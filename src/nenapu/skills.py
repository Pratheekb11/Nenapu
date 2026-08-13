"""Skill storage with an outcome loop.

Hermes-style agents write a skill document whenever they solve something hard.
Nothing then checks whether the skill was any good. Here every invocation is
recorded with an outcome, and skills that keep failing — or never get used at
all — are quarantined instead of quietly accumulating.
"""

from __future__ import annotations

import sqlite3
from dataclasses import replace

from .models import Skill, now, row_to_skill
from .store import _fts_query
from .db import commit

DAY = 86400.0

# Quarantine thresholds. Deliberately conservative: a skill needs a real track
# record before we stop trusting it, and "unused" needs a long window so a
# seasonal skill is not culled.
MIN_GRADED_FOR_JUDGEMENT = 4
FAILING_RATE = 0.4
UNUSED_AFTER_DAYS = 120.0


class SkillStore:
    def __init__(self, conn: sqlite3.Connection) -> None:
        self.conn = conn

    def upsert(self, skill: Skill) -> Skill:
        existing = self.get(skill.name)
        if existing:
            self.conn.execute(
                "UPDATE skills SET body=?, description=?, scope=?, tags_csv=?, updated_at=?"
                " WHERE id=?",
                (skill.body, skill.description, skill.scope, ",".join(skill.tags), now(),
                 existing.id),
            )
            commit(self.conn)
            return self.get(skill.name)

        cur = self.conn.execute(
            """INSERT INTO skills(name, description, body, scope, tags_csv, status,
                   quarantine_reason, created_at, updated_at, invocations, successes,
                   failures, last_used_at)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (skill.name, skill.description, skill.body, skill.scope, ",".join(skill.tags),
             skill.status, skill.quarantine_reason, skill.created_at, skill.updated_at,
             skill.invocations, skill.successes, skill.failures, skill.last_used_at),
        )
        commit(self.conn)
        return replace(skill, id=cur.lastrowid)

    def get(self, name: str) -> Skill | None:
        row = self.conn.execute("SELECT * FROM skills WHERE name = ?", (name,)).fetchone()
        return row_to_skill(row) if row else None

    def list_skills(self, status: str | None = "active", scope: str | None = None) -> list[Skill]:
        sql, args = "SELECT * FROM skills WHERE 1=1", []
        if status:
            sql += " AND status = ?"
            args.append(status)
        if scope:
            sql += " AND scope = ?"
            args.append(scope)
        sql += " ORDER BY updated_at DESC"
        return [row_to_skill(r) for r in self.conn.execute(sql, args)]

    def search(self, query: str, *, limit: int = 5, include_quarantined: bool = False):
        fts = _fts_query(query)
        if not fts:
            return self.list_skills()[:limit]
        sql = (
            "SELECT s.*, bm25(skills_fts) rank FROM skills_fts"
            " JOIN skills s ON s.id = skills_fts.rowid WHERE skills_fts MATCH ?"
        )
        args: list = [fts]
        if not include_quarantined:
            sql += " AND s.status = 'active'"
        sql += " ORDER BY rank LIMIT ?"
        args.append(limit)
        try:
            return [row_to_skill(r) for r in self.conn.execute(sql, args)]
        except sqlite3.OperationalError:
            return []

    def record_outcome(
        self, name: str, outcome: str, *, session_id: str | None = None, note: str | None = None
    ) -> Skill | None:
        """Log one use. `outcome` is success | failure | used (ungraded)."""
        skill = self.get(name)
        if not skill:
            return None

        self.conn.execute(
            "INSERT INTO skill_events(skill_id, outcome, session_id, note, created_at)"
            " VALUES (?,?,?,?,?)",
            (skill.id, outcome, session_id, note, now()),
        )
        col = {"success": "successes", "failure": "failures"}.get(outcome)
        bump = f", {col} = {col} + 1" if col else ""
        self.conn.execute(
            f"UPDATE skills SET invocations = invocations + 1, last_used_at = ?,"
            f" updated_at = ?{bump} WHERE id = ?",
            (now(), now(), skill.id),
        )
        commit(self.conn)
        return self.sweep_one(name)

    def sweep_one(self, name: str) -> Skill | None:
        skill = self.get(name)
        if not skill or skill.status != "active":
            return skill

        graded = skill.successes + skill.failures
        rate = skill.success_rate
        reason = None
        if graded >= MIN_GRADED_FOR_JUDGEMENT and rate is not None and rate < FAILING_RATE:
            reason = f"success rate {rate:.0%} over {graded} graded runs"
        else:
            age_days = (now() - skill.created_at) / DAY
            if skill.invocations == 0 and age_days > UNUSED_AFTER_DAYS:
                reason = f"never invoked in {age_days:.0f} days"

        if reason:
            self.conn.execute(
                "UPDATE skills SET status='quarantined', quarantine_reason=?, updated_at=?"
                " WHERE id=?",
                (reason, now(), skill.id),
            )
            commit(self.conn)
        return self.get(name)

    def sweep(self) -> list[Skill]:
        """Quarantine every skill that no longer earns its place. Returns the culled."""
        culled = []
        for skill in self.list_skills(status="active"):
            after = self.sweep_one(skill.name)
            if after and after.status == "quarantined":
                culled.append(after)
        return culled

    def revive(self, name: str) -> Skill | None:
        self.conn.execute(
            "UPDATE skills SET status='active', quarantine_reason=NULL, updated_at=? WHERE name=?",
            (now(), name),
        )
        commit(self.conn)
        return self.get(name)
