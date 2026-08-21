"""The recall ledger: did believing this actually help?

Every memory system optimizes *retrieval relevance* — did the right text come
back. None of them measure the thing that matters: whether acting on the
recalled memory worked. A fact that was surfaced into five tasks that all went
wrong should lose standing, even if it keeps matching the query beautifully.

Grading signals, cheapest first. All of them write to the same ledger, so a
deployment can use whichever it can actually supply:

  1. `verification`  — a recalled fact whose check later fails. Free, exact.
  2. `correction`    — a recalled fact contradicted or retired soon after.
                       Needs no cooperation from the harness at all.
  3. `agent`         — the harness reports task success explicitly.
  4. `human`         — `nenapu bad <id>` after a wrong answer.

Signals 1 and 2 are inferred from things that were happening anyway, which is
why the loop closes even when nobody wires anything up.
"""

from __future__ import annotations

import sqlite3

from .models import Outcome, Recall, now, row_to_recall
from .db import commit, transaction

# How far back an implicit signal reaches. A correction an hour after the recall
# is plausibly about that recall; a correction next week is just new information.
IMPLICIT_WINDOW_SECONDS = 6 * 3600.0

# A recall nobody ever graded is not evidence of anything. After this it is
# closed as neutral so it stops sitting in the pending queue forever.
PENDING_EXPIRY_SECONDS = 7 * 86400.0


class Ledger:
    def __init__(self, conn: sqlite3.Connection) -> None:
        self.conn = conn

    # ---------- writing ----------

    def log(
        self,
        fact_id: int,
        *,
        session_id: str | None = None,
        query: str = "",
        rank: int = 0,
        score: float = 0.0,
    ) -> int:
        cur = self.conn.execute(
            "INSERT INTO recalls(fact_id, session_id, query, rank, score, outcome, created_at)"
            " VALUES (?,?,?,?,?,?,?)",
            (fact_id, session_id, query, rank, score, Outcome.PENDING, now()),
        )
        return cur.lastrowid

    def log_many(self, hits, *, session_id: str | None = None, query: str = "") -> list[int]:
        """Log a whole result set. Caller is expected to hold a transaction."""
        with transaction(self.conn):
            return [
                self.log(fact.id, session_id=session_id, query=query, rank=i, score=score)
                for i, (fact, score, _why) in enumerate(hits)
            ]

    # ---------- grading ----------

    def grade(self, recall_id: int, outcome: str, *, source: str, note: str | None = None) -> bool:
        """Grade one recall. First grade wins — a later implicit signal must not
        overwrite what a human or the harness explicitly said."""
        row = self.conn.execute(
            "SELECT fact_id FROM recalls WHERE id = ?", (recall_id,)
        ).fetchone()
        if not row:
            return False

        # The pending check and the write are one statement on purpose. Reading
        # the outcome first and updating second lets several graders each see
        # "pending" and each bump the counter — eight concurrent graders scored
        # a single recall eight times before this.
        cur = self.conn.execute(
            "UPDATE recalls SET outcome=?, outcome_source=?, outcome_at=?, note=?"
            " WHERE id=? AND outcome=?",
            (outcome, source, now(), note, recall_id, Outcome.PENDING),
        )
        if cur.rowcount == 0:
            return False  # someone else graded it first; first grade wins

        self._bump_fact(row["fact_id"], outcome)
        commit(self.conn)
        return True

    def grade_session(
        self, session_id: str, outcome: str, *, source: str = "agent", note: str | None = None
    ) -> int:
        """Grade every pending recall in a session — the shape a harness reports:
        one verdict for the whole task, applied to whatever memory it used."""
        rows = self.conn.execute(
            "SELECT id FROM recalls WHERE session_id = ? AND outcome = ?",
            (session_id, Outcome.PENDING),
        ).fetchall()
        with transaction(self.conn):
            return sum(
                1 for row in rows
                if self.grade(row["id"], outcome, source=source, note=note)
            )

    def _bump_fact(self, fact_id: int, outcome: str) -> None:
        if outcome == Outcome.GOOD:
            column = "good_recalls"
        elif outcome == Outcome.BAD:
            column = "bad_recalls"
        else:
            return
        self.conn.execute(
            f"UPDATE facts SET {column} = {column} + 1, updated_at = ? WHERE id = ?",
            (now(), fact_id),
        )

    # ---------- implicit signals ----------

    def blame_recent_recalls(
        self, fact_id: int, *, source: str, note: str, window: float = IMPLICIT_WINDOW_SECONDS
    ) -> int:
        """This fact just turned out to be wrong; mark its recent recalls bad.

        Called from the write path (a contradiction superseded it) and from the
        verifier (its check failed). No harness cooperation required.
        """
        rows = self.conn.execute(
            "SELECT id FROM recalls WHERE fact_id = ? AND outcome = ? AND created_at >= ?",
            (fact_id, Outcome.PENDING, now() - window),
        ).fetchall()
        return sum(1 for r in rows if self.grade(r["id"], Outcome.BAD, source=source, note=note))

    def blame_session_recalls(
        self, fact_id: int, session_id: str | None, *, source: str, note: str,
    ) -> int:
        """This fact just turned out to be wrong; blame that session's own recalls.

        `blame_recent_recalls` reaches back a fixed 6h window, which a
        SessionStart injection outruns in any session longer than that — the
        correction signal misses its own session. Here the session id is the
        only guard, in place of a window: every pending recall of this fact
        logged under that session, however old, is blamed. A missing session
        id grades nothing, never "every pending recall of this fact" — a
        write outside a session, or a retire from the CLI, has no session to
        implicate.
        """
        if not session_id:
            return 0
        rows = self.conn.execute(
            "SELECT id FROM recalls WHERE fact_id = ? AND session_id = ? AND outcome = ?",
            (fact_id, session_id, Outcome.PENDING),
        ).fetchall()
        return sum(1 for r in rows if self.grade(r["id"], Outcome.BAD, source=source, note=note))

    def credit_recent_recalls(
        self, fact_id: int, *, source: str, note: str, window: float = IMPLICIT_WINDOW_SECONDS
    ) -> int:
        """The fact re-verified cleanly; its recent recalls were sound."""
        rows = self.conn.execute(
            "SELECT id FROM recalls WHERE fact_id = ? AND outcome = ? AND created_at >= ?",
            (fact_id, Outcome.PENDING, now() - window),
        ).fetchall()
        return sum(1 for r in rows if self.grade(r["id"], Outcome.GOOD, source=source, note=note))

    def expire_pending(self, older_than: float = PENDING_EXPIRY_SECONDS) -> int:
        """Close out recalls nobody graded, so the pending queue means something."""
        cur = self.conn.execute(
            "UPDATE recalls SET outcome=?, outcome_source='expiry', outcome_at=?"
            " WHERE outcome=? AND created_at < ?",
            (Outcome.NEUTRAL, now(), Outcome.PENDING, now() - older_than),
        )
        commit(self.conn)
        return cur.rowcount

    # ---------- reads ----------

    def get(self, recall_id: int) -> Recall | None:
        row = self.conn.execute("SELECT * FROM recalls WHERE id = ?", (recall_id,)).fetchone()
        return row_to_recall(row) if row else None

    def pending(self, session_id: str | None = None, limit: int = 50) -> list[Recall]:
        sql = "SELECT * FROM recalls WHERE outcome = ?"
        args: list = [Outcome.PENDING]
        if session_id:
            sql += " AND session_id = ?"
            args.append(session_id)
        sql += " ORDER BY created_at DESC LIMIT ?"
        args.append(limit)
        return [row_to_recall(r) for r in self.conn.execute(sql, args)]

    def for_fact(self, fact_id: int, limit: int = 20) -> list[Recall]:
        rows = self.conn.execute(
            "SELECT * FROM recalls WHERE fact_id = ? ORDER BY created_at DESC LIMIT ?",
            (fact_id, limit),
        )
        return [row_to_recall(r) for r in rows]

    def stats(self) -> dict:
        rows = self.conn.execute("SELECT outcome, COUNT(*) c FROM recalls GROUP BY outcome")
        by_outcome = {r["outcome"]: r["c"] for r in rows}
        sources = self.conn.execute(
            "SELECT outcome_source s, COUNT(*) c FROM recalls WHERE outcome_source IS NOT NULL"
            " GROUP BY outcome_source"
        )
        return {
            "recalls": by_outcome,
            "graded_by": {r["s"]: r["c"] for r in sources},
        }


def outcome_signal(good: int, bad: int) -> float:
    """Confidence multiplier from a fact's track record.

    Laplace-smoothed so a single bad recall does not condemn a fact, and an
    ungraded fact is neither rewarded nor punished.
    """
    n = good + bad
    if n == 0:
        return 1.0
    rate = (good + 1) / (n + 2)
    return 0.6 + 0.8 * rate
