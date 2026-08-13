"""Approval gate for executable checks.

`verify_cmd` is the feature that makes a fact falsifiable, and it is also the
most dangerous thing in the system: it is shell, stored in a row that an agent
can write. Agents read untrusted input all day — a README in a cloned repo, a
web page, a tool result — so "the agent decided to store this command" is not a
trust signal. Without a gate, one injected `memory_write` makes `nenapu verify`
into a scheduled remote-code-execution primitive running unattended with the
user's privileges.

So: a command runs only after a human has approved that exact string. Editing
an approved command invalidates the approval, because the hash changes.
"""

from __future__ import annotations

import hashlib
import re
import sqlite3

from .models import now
from .db import commit

# Shell constructs worth pointing at during review. Not a blocklist — approval
# is the control — but a reviewer skimming should have their eye drawn to the
# pipe into a shell rather than having to spot it.
RISKY = [
    (re.compile(r"\|\s*(ba)?sh\b"), "pipes into a shell"),
    (re.compile(r"\b(curl|wget)\b"), "fetches from the network"),
    (re.compile(r"\brm\b\s+-[rf]"), "recursive or forced delete"),
    (re.compile(r"[;&]{1,2}|\$\(|`"), "chains or substitutes commands"),
    (re.compile(r">\s*/|>>\s*/"), "writes to an absolute path"),
    (re.compile(r"\b(sudo|doas)\b"), "escalates privileges"),
    (re.compile(r"\bchmod\b|\bchown\b"), "changes permissions or ownership"),
]


def fingerprint(command: str) -> str:
    return hashlib.sha256(command.encode()).hexdigest()


def concerns(command: str) -> list[str]:
    """Human-readable reasons to look twice at a command."""
    return [why for pattern, why in RISKY if pattern.search(command)]


def is_approved(conn: sqlite3.Connection, command: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM approved_commands WHERE sha256 = ?", (fingerprint(command),)
    ).fetchone()
    return row is not None


def approve(
    conn: sqlite3.Connection, command: str, *, fact_id: int | None = None, by: str = "user"
) -> str:
    digest = fingerprint(command)
    conn.execute(
        "INSERT OR REPLACE INTO approved_commands(sha256, command, fact_id, approved_at,"
        " approved_by) VALUES (?,?,?,?,?)",
        (digest, command, fact_id, now(), by),
    )
    conn.execute(
        "INSERT INTO journal(action, fact_id, actor, detail, created_at)"
        " VALUES ('approve_check', ?, ?, ?, ?)",
        (fact_id, by, command[:200], now()),
    )
    commit(conn)
    return digest


def revoke(conn: sqlite3.Connection, command: str) -> bool:
    cur = conn.execute(
        "DELETE FROM approved_commands WHERE sha256 = ?", (fingerprint(command),)
    )
    commit(conn)
    return cur.rowcount > 0


def pending(conn: sqlite3.Connection) -> list[tuple[int, str, str]]:
    """Facts carrying a check that has never been approved: (id, origin, command)."""
    rows = conn.execute(
        "SELECT id, origin, verify_cmd FROM facts"
        " WHERE verify_cmd IS NOT NULL AND status != 'retired'"
    )
    return [
        (r["id"], r["origin"], r["verify_cmd"])
        for r in rows
        if not is_approved(conn, r["verify_cmd"])
    ]


def approved_list(conn: sqlite3.Connection) -> list[tuple[str, str, float]]:
    rows = conn.execute(
        "SELECT command, approved_by, approved_at FROM approved_commands ORDER BY approved_at"
    )
    return [(r["command"], r["approved_by"], r["approved_at"]) for r in rows]
