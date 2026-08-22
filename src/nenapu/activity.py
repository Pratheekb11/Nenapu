"""The activity ledger: sessions, file_events, commits.

Deterministic and free — everything here comes from git and the transcript's
tool-use blocks, never from a model call. This is what answers "where did I
do what, which agent edited which file, what got deleted" across every repo
on the machine, which no belief-network fact ever could.
"""

from __future__ import annotations

import json
import sqlite3

from .db import commit, transaction
from .models import now


class ActivityLedger:
    def __init__(self, conn: sqlite3.Connection) -> None:
        self.conn = conn

    # ---------- sessions ----------

    def start_session(
        self,
        *,
        agent: str,
        project_scope: str,
        cwd: str | None = None,
        git_branch: str | None = None,
        git_head_before: str | None = None,
        started_at: float | None = None,
        external_id: str | None = None,
        source: str | None = None,
    ) -> int:
        """`source` says whether the row was watched as it ran (`hook`) or
        reconstructed from history (`backfill`). Recorded rather than inferred,
        because the repair that moves mis-dated rows onto their transcript's
        clock has to know which rows it may touch."""
        with transaction(self.conn):
            cur = self.conn.execute(
                "INSERT INTO sessions(agent, project_scope, cwd, git_branch,"
                " git_head_before, started_at, external_id, source)"
                " VALUES (?,?,?,?,?,?,?,?)",
                (agent, project_scope, cwd, git_branch, git_head_before, started_at or now(),
                 external_id, source),
            )
            commit(self.conn)
            return cur.lastrowid

    def end_session(
        self,
        session_id: int,
        *,
        git_head_after: str | None = None,
        summary: str | None = None,
        ended_at: float | None = None,
    ) -> None:
        with transaction(self.conn):
            self.conn.execute(
                "UPDATE sessions SET git_head_after = ?, summary = ?, ended_at = ?"
                " WHERE id = ?",
                (git_head_after, summary, ended_at or now(), session_id),
            )
            commit(self.conn)

    def redate_session(self, session_id: int, *, started_at: float,
                       ended_at: float | None = None) -> None:
        """Correct when a session happened, for a row that was recorded from
        history rather than watched as it ran. `ended_at` is left alone when
        the transcript cannot say."""
        with transaction(self.conn):
            if ended_at is None:
                self.conn.execute("UPDATE sessions SET started_at = ? WHERE id = ?",
                                  (started_at, session_id))
            else:
                self.conn.execute(
                    "UPDATE sessions SET started_at = ?, ended_at = ? WHERE id = ?",
                    (started_at, ended_at, session_id))
            commit(self.conn)

    def get_session(self, session_id: int | str) -> dict | None:
        """Looks up by the internal row id or by `external_id` — a transcript's
        own session id, which backfill uses to detect a session already
        ingested without keeping a separate index of processed files."""
        try:
            id_val = int(session_id)
        except (TypeError, ValueError):
            id_val = None
        row = self.conn.execute(
            "SELECT * FROM sessions WHERE id = ? OR external_id = ?",
            (id_val, str(session_id)),
        ).fetchone()
        return dict(row) if row else None

    def sessions_for_scope(self, project_scope: str, *, limit: int = 500) -> list[dict]:
        rows = self.conn.execute(
            "SELECT * FROM sessions WHERE project_scope = ? ORDER BY started_at DESC LIMIT ?",
            (project_scope, limit),
        )
        return [dict(r) for r in rows]

    def recent_sessions(self, *, since_at: float | None = None, limit: int = 500) -> list[dict]:
        sql = "SELECT * FROM sessions WHERE 1=1"
        args: list = []
        if since_at is not None:
            sql += " AND started_at >= ?"
            args.append(since_at)
        sql += " ORDER BY started_at DESC LIMIT ?"
        args.append(limit)
        return [dict(r) for r in self.conn.execute(sql, args)]

    def known_scopes(self) -> list[str]:
        rows = self.conn.execute("SELECT DISTINCT project_scope FROM sessions")
        return [r["project_scope"] for r in rows]

    def sessions_in_range(self, start: float, end: float) -> list[dict]:
        rows = self.conn.execute(
            "SELECT * FROM sessions WHERE started_at >= ? AND started_at < ?"
            " ORDER BY started_at",
            (start, end),
        )
        return [dict(r) for r in rows]

    def delete_session(self, session_id: int) -> None:
        """Removes a session and everything keyed to it. Called once a
        session has been folded into a rollup row — the aggregate survives,
        the raw detail does not need to."""
        with transaction(self.conn):
            self.conn.execute("DELETE FROM file_events WHERE session_id = ?", (session_id,))
            self.conn.execute("DELETE FROM commits WHERE session_id = ?", (session_id,))
            self.conn.execute("DELETE FROM sessions WHERE id = ?", (session_id,))
            commit(self.conn)

    # ---------- file events ----------

    def record_file_event(
        self,
        session_id: int,
        *,
        path: str,
        op: str,
        tool: str | None = None,
        at: float | None = None,
    ) -> int:
        with transaction(self.conn):
            cur = self.conn.execute(
                "INSERT INTO file_events(session_id, path, op, tool, at) VALUES (?,?,?,?,?)",
                (session_id, path, op, tool, at or now()),
            )
            commit(self.conn)
            return cur.lastrowid

    def file_events_for_session(self, session_id: int | str) -> list[dict]:
        session = self.get_session(session_id)
        if session is None:
            return []
        rows = self.conn.execute(
            "SELECT * FROM file_events WHERE session_id = ? ORDER BY at, id", (session["id"],)
        )
        return [dict(r) for r in rows]

    def file_events_for_path(self, path: str, *, limit: int = 500) -> list[dict]:
        rows = self.conn.execute(
            "SELECT fe.*, s.agent AS agent, s.project_scope AS project_scope"
            " FROM file_events fe JOIN sessions s ON s.id = fe.session_id"
            " WHERE fe.path = ? ORDER BY fe.at DESC LIMIT ?",
            (path, limit),
        )
        return [dict(r) for r in rows]

    # ---------- commits ----------

    def record_commit(
        self,
        session_id: int | None,
        *,
        sha: str,
        subject: str | None = None,
        files_changed: list[str] | None = None,
        at: float | None = None,
    ) -> int:
        with transaction(self.conn):
            cur = self.conn.execute(
                "INSERT INTO commits(session_id, sha, subject, files_changed, at)"
                " VALUES (?,?,?,?,?)",
                (session_id, sha, subject, json.dumps(files_changed or []), at or now()),
            )
            commit(self.conn)
            return cur.lastrowid

    def commits_for_session(self, session_id: int | str) -> list[dict]:
        session = self.get_session(session_id)
        if session is None:
            return []
        rows = self.conn.execute(
            "SELECT * FROM commits WHERE session_id = ? ORDER BY at", (session["id"],)
        )
        return [_commit_row(r) for r in rows]

    def file_events_for_scope(self, project_scope: str, *, limit: int = 500) -> list[dict]:
        rows = self.conn.execute(
            "SELECT fe.* FROM file_events fe JOIN sessions s ON s.id = fe.session_id"
            " WHERE s.project_scope = ? ORDER BY fe.at DESC LIMIT ?",
            (project_scope, limit),
        )
        return [dict(r) for r in rows]

    def commits_for_scope(self, project_scope: str, *, limit: int = 500) -> list[dict]:
        rows = self.conn.execute(
            "SELECT c.* FROM commits c JOIN sessions s ON s.id = c.session_id"
            " WHERE s.project_scope = ? ORDER BY c.at DESC LIMIT ?",
            (project_scope, limit),
        )
        return [_commit_row(r) for r in rows]


    # ---------- rollups ----------

    def upsert_rollup(
        self,
        project_scope: str,
        period: str,
        period_start: float,
        period_end: float,
        *,
        session_count: int = 0,
        files_touched: int = 0,
        commits: int = 0,
    ) -> None:
        with transaction(self.conn):
            existing = self.conn.execute(
                "SELECT id FROM rollups WHERE project_scope = ? AND period = ?"
                " AND period_start = ?",
                (project_scope, period, period_start),
            ).fetchone()
            if existing:
                self.conn.execute(
                    "UPDATE rollups SET session_count = session_count + ?,"
                    " files_touched = files_touched + ?, commits = commits + ?,"
                    " period_end = ? WHERE id = ?",
                    (session_count, files_touched, commits, period_end, existing["id"]),
                )
            else:
                self.conn.execute(
                    "INSERT INTO rollups(project_scope, period, period_start, period_end,"
                    " session_count, files_touched, commits) VALUES (?,?,?,?,?,?,?)",
                    (project_scope, period, period_start, period_end,
                     session_count, files_touched, commits),
                )
            commit(self.conn)

    def rollups_for_scope(self, project_scope: str, *, period: str | None = None) -> list[dict]:
        sql = "SELECT * FROM rollups WHERE project_scope = ?"
        args: list = [project_scope]
        if period:
            sql += " AND period = ?"
            args.append(period)
        sql += " ORDER BY period_start"
        return [dict(r) for r in self.conn.execute(sql, args)]


def _commit_row(row: sqlite3.Row) -> dict:
    d = dict(row)
    d["files_changed"] = json.loads(d["files_changed"]) if d["files_changed"] else []
    return d
