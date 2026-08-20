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
    ) -> int:
        with transaction(self.conn):
            cur = self.conn.execute(
                "INSERT INTO sessions(agent, project_scope, cwd, git_branch,"
                " git_head_before, started_at) VALUES (?,?,?,?,?,?)",
                (agent, project_scope, cwd, git_branch, git_head_before, started_at or now()),
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

    def get_session(self, session_id: int) -> dict | None:
        row = self.conn.execute(
            "SELECT * FROM sessions WHERE id = ?", (session_id,)
        ).fetchone()
        return dict(row) if row else None

    def sessions_for_scope(self, project_scope: str, *, limit: int = 500) -> list[dict]:
        rows = self.conn.execute(
            "SELECT * FROM sessions WHERE project_scope = ? ORDER BY started_at DESC LIMIT ?",
            (project_scope, limit),
        )
        return [dict(r) for r in rows]

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

    def file_events_for_session(self, session_id: int) -> list[dict]:
        rows = self.conn.execute(
            "SELECT * FROM file_events WHERE session_id = ? ORDER BY at", (session_id,)
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

    def commits_for_session(self, session_id: int) -> list[dict]:
        rows = self.conn.execute(
            "SELECT * FROM commits WHERE session_id = ? ORDER BY at", (session_id,)
        )
        return [_commit_row(r) for r in rows]

    def commits_for_scope(self, project_scope: str, *, limit: int = 500) -> list[dict]:
        rows = self.conn.execute(
            "SELECT c.* FROM commits c JOIN sessions s ON s.id = c.session_id"
            " WHERE s.project_scope = ? ORDER BY c.at DESC LIMIT ?",
            (project_scope, limit),
        )
        return [_commit_row(r) for r in rows]


def _commit_row(row: sqlite3.Row) -> dict:
    d = dict(row)
    d["files_changed"] = json.loads(d["files_changed"]) if d["files_changed"] else []
    return d
