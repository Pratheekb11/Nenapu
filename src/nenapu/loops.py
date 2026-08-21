"""Open loops: things the session said would happen and then did not.

The premise of the whole project is that nobody files todos by hand, so a
loop is opened from what a session says and closed from what later sessions
do. Two rules shape everything here:

**Closure is biased toward closing.** False nagging is fatal and silence is
survivable. If the agent claims work was missed that was in fact shipped,
trust is gone immediately and permanently, while a reminder that never
arrives costs one forgotten task. So any plausible evidence of completion
closes a loop: a commit touching the hinted path, a matching file written, or
a commit whose subject plainly describes the loop.

**Old loops go quiet rather than shouting.** A loop in a project nobody has
opened for three months should not be at the top of the block. It ages on the
same decay curve facts do and drops out of what gets surfaced once it falls
under the injection floor — still queryable, never deleted.
"""

from __future__ import annotations

import fnmatch
import re
import sqlite3

from .db import commit, transaction
from .models import Decay, now
from .store import decay_factor

# Below this a loop stops being surfaced. Deliberately the same floor facts
# are injected against — a reminder nobody has acted on for months is exactly
# as believable as a fact nobody has checked for months.
QUIET_BELIEF = 0.35
# Loops age on the medium curve: a 90-day half-life puts the quiet threshold
# a little over four months out, which is the "three months untouched" the
# requirement describes.
LOOP_DECAY = Decay.MEDIUM

OPEN = "open"
CLOSED = "closed"
STATED = "stated"
INTERRUPTED = "interrupted"

# Ops that count as work. Reading a file is the most common thing a session
# does and is evidence of nothing.
_WORK_OPS = ("created", "edited")

# How much of a commit subject must be about the loop before the subject
# alone closes it. Measured against the two cases that matter: "Rate limit the
# availability endpoint" closing "Add rate limiting to the availability
# endpoint" (0.75), and "Fix the pet drawing" closing nothing (0.0).
SUBJECT_MATCH_RATIO = 0.4

_WORD = re.compile(r"[a-z0-9]+")
_FILLER = {
    "a", "an", "the", "to", "of", "in", "on", "at", "for", "and", "or", "add",
    "adds", "added", "fix", "fixes", "fixed", "make", "makes", "made", "use",
    "using", "with", "into", "from", "still", "need", "needs", "needed", "is",
    "are", "be", "it", "its", "this", "that", "we", "our", "should", "must",
}


class LoopBook:
    """Storage and closure for open loops. One table, no model calls."""

    def __init__(self, conn: sqlite3.Connection) -> None:
        self.conn = conn

    # ---------- opening and closing ----------

    def open_loop(
        self,
        *,
        scope: str,
        text: str,
        resolution_hint: str | None = None,
        session_id: str | None = None,
        kind: str = STATED,
        at: float | None = None,
    ) -> int:
        with transaction(self.conn):
            cur = self.conn.execute(
                "INSERT INTO open_loops(scope, text, resolution_hint, kind, status,"
                " opened_at, session_id) VALUES (?,?,?,?,?,?,?)",
                (scope, text, resolution_hint or None, kind, OPEN, at or now(), session_id),
            )
            commit(self.conn)
            return cur.lastrowid

    def close_loop(self, loop_id: int, *, reason: str) -> bool:
        """Retire a loop, keeping why. Returns False if it was already closed.

        Folded into one statement rather than a read followed by a write: the
        maintenance worker and a hook can both reach the same loop, and two
        closes would overwrite the first reason with the second.
        """
        with transaction(self.conn):
            cur = self.conn.execute(
                "UPDATE open_loops SET status = ?, closed_at = ?, close_reason = ?"
                " WHERE id = ? AND status = ?",
                (CLOSED, now(), reason, loop_id, OPEN),
            )
            commit(self.conn)
            return cur.rowcount > 0

    def get(self, loop_id: int) -> dict | None:
        row = self.conn.execute("SELECT * FROM open_loops WHERE id = ?", (loop_id,)).fetchone()
        return dict(row) if row else None

    # ---------- what is worth saying out loud ----------

    def open_for_scope(self, scope: str, *, include_quiet: bool = False) -> list[dict]:
        return self._open_loops("scope = ?", [scope], include_quiet=include_quiet)

    def all_open(self, *, include_quiet: bool = False) -> list[dict]:
        return self._open_loops("1=1", [], include_quiet=include_quiet)

    def _open_loops(self, where: str, args: list, *, include_quiet: bool) -> list[dict]:
        rows = self.conn.execute(
            f"SELECT * FROM open_loops WHERE status = ? AND {where} ORDER BY opened_at DESC",
            [OPEN, *args],
        )
        loops = [dict(r) for r in rows]
        if include_quiet:
            return loops
        return [loop for loop in loops if loudness(loop) >= QUIET_BELIEF]

    # ---------- closure from the activity ledger ----------

    def close_satisfied(self, ledger, *, scope: str | None = None) -> list[int]:
        """Close every loop the ledger shows satisfied. Returns what it closed.

        Reported once. This runs on every maintenance tick, so returning the
        same loop forever would make anything built on the return value nag
        about work that was finished months ago.
        """
        loops = (self.open_for_scope(scope, include_quiet=True) if scope
                 else self.all_open(include_quiet=True))
        closed: list[int] = []
        for loop in loops:
            reason = self._evidence_for(ledger, loop)
            if reason and self.close_loop(loop["id"], reason=reason):
                closed.append(loop["id"])
        return closed

    def _evidence_for(self, ledger, loop: dict) -> str | None:
        """Why this loop should be considered done, if anything says so.

        Only work done *after* the loop was opened counts. Without that, a
        loop mentioned late in a long session closes itself against that same
        session's earlier edits and the whole mechanism is silently inert.
        """
        since = loop["opened_at"]
        patterns = _hint_patterns(loop["resolution_hint"])

        if patterns:
            for event in ledger.file_events_for_scope(loop["scope"]):
                if event["at"] <= since or event["op"] not in _WORK_OPS:
                    continue
                if _matches_any(event["path"], patterns):
                    return f"{event['op']} {event['path']}"

        for entry in ledger.commits_for_scope(loop["scope"]):
            if entry["at"] <= since:
                continue
            if patterns and any(_matches_any(path, patterns)
                                for path in entry["files_changed"]):
                return f"commit {entry['sha'][:8]} touched {entry['files_changed'][0]}"
            # A loop the extractor could attach no path to is still closable by
            # a commit that plainly describes it. This is the bias toward
            # closing, stated as code: prefer a missed reminder over a wrong one.
            if not patterns and _describes(entry["subject"], loop["text"]):
                return f"commit {entry['sha'][:8]}: {entry['subject']}"
        return None

    # ---------- abrupt stops ----------

    def detect_interrupted(self, ledger, session_id: int | str) -> int | None:
        """Raise a loop for work left in flight, if that is what happened.

        Deterministic and free: files were modified and the commit the session
        started on is the commit it ended on, so nothing was committed. A
        session that only read things left no work behind, and saying
        otherwise on every such session is the noise that teaches people to
        skip the block.
        """
        session = ledger.get_session(session_id)
        if session is None or session["git_head_before"] != session["git_head_after"]:
            return None

        touched = [event["path"] for event in ledger.file_events_for_session(session_id)
                   if event["op"] in _WORK_OPS]
        if not touched:
            return None

        existing = self.conn.execute(
            "SELECT id FROM open_loops WHERE kind = ? AND session_id = ? AND status = ?",
            (INTERRUPTED, str(session["id"]), OPEN),
        ).fetchone()
        if existing:
            return existing["id"]  # the worker can revisit a session; one stop, one loop

        unique = list(dict.fromkeys(touched))
        return self.open_loop(
            scope=session["project_scope"],
            text="Uncommitted work left in flight: " + ", ".join(unique[:5]),
            # The loop is closed by exactly the evidence that it was resumed,
            # or every interrupted session becomes a permanent nag.
            resolution_hint=" ".join(unique[:5]),
            session_id=str(session["id"]),
            kind=INTERRUPTED,
        )


def loudness(loop: dict) -> float:
    """How much a loop still deserves to be said out loud."""
    return decay_factor(LOOP_DECAY, now() - loop["opened_at"])


def _hint_patterns(hint: str | None) -> list[str]:
    """A hint is one or more path globs. Several arrive when the loop came
    from an abrupt stop, which knows every file that was in flight."""
    return [part for part in (hint or "").split() if part]


def _matches_any(path: str, patterns: list[str]) -> bool:
    return any(fnmatch.fnmatch(path, pattern) or fnmatch.fnmatch(path, f"*/{pattern}")
               for pattern in patterns)


def _stem(word: str) -> str:
    """Enough stemming to see that "limiting" and "limit" are one word, and no
    more: a real stemmer here would be a dependency bought for four suffixes."""
    for suffix in ("ing", "ed", "es", "s"):
        if len(word) > len(suffix) + 2 and word.endswith(suffix):
            return word[: -len(suffix)]
    return word


def _content(text: str) -> set[str]:
    words = [w for w in _WORD.findall((text or "").lower()) if w not in _FILLER]
    return {_stem(w) for w in words if len(w) > 2}


def _describes(subject: str | None, loop_text: str) -> bool:
    """Is this commit subject about this loop?

    Measured against the subject rather than the loop: a loop is a sentence
    and a subject is a summary, so asking how much of the summary is about the
    loop is the question with a stable denominator.
    """
    subject_words = _content(subject or "")
    if not subject_words:
        return False
    shared = subject_words & _content(loop_text)
    return len(shared) / len(subject_words) >= SUBJECT_MATCH_RATIO
