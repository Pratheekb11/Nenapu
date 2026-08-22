"""`--dry-run` is a contract, not a per-command courtesy.

Requirement (plan "Harden the four incidents into guarantees", Phase B,
tasks B1-B3, marked **Opus** for the guard and the registry test):

    Turn the promise each command remembers to keep into a guarantee the
    connection enforces, and make it impossible for a new command to declare
    `--dry-run` without being held to it.

Four defects started this plan and the last was a `backfill --redate --dry-run`
that wrote anyway. Phase A found three more in `learn` alone. Every one of them
was the same shape: the flag is a boolean each write path has to remember to
consult, and the paths that forgot were invisible until someone ran the command
against a real store. Remembering does not scale, so the connection refuses.

Two halves, and both are needed:

* `deny_writes` installs a SQLite authorizer that refuses every statement that
  would change data. A dry run that tries to write raises rather than writes,
  which is the loud failure a silent one deserves.
* the registry test below fails if a command declares `--dry-run` and is not in
  the covered list, so the contract cannot rot the first time a command is added.

What a dry run may still do
---------------------------
Open a store that does not exist, which creates and migrates the file. That is
`db.connect`'s job, it happens before any command logic, and it is stated here
so the guarantee is exact rather than approximate: no *data* changes.

`audit --dry-run` also still spends the model call. That is deliberate and out
of scope: it suppresses the database effects, not the cost. These tests point
`NENAPU_LLM` at a backend that does not exist, so no test here reaches a model.
"""

import inspect
import json
import os
import sqlite3
import subprocess
import sys

import pytest
import typer

from nenapu import connect
from nenapu.cli import app
from nenapu.db import allow_writes, deny_writes, readonly

NO_MODEL = {"NENAPU_LLM": "nonesuch"}


# ---------- B1: the guard itself ----------


@pytest.fixture
def conn(tmp_path):
    c = connect(str(tmp_path / "guard.db"))
    c.execute(
        "INSERT INTO facts(text, kind, origin, confidence, decay_class,"
        " created_at, updated_at, status)"
        " VALUES('the staging bucket is the export target','env','user_stated',"
        "0.9,'medium',0,0,'active')"
    )
    c.commit()
    yield c
    c.close()


@pytest.mark.parametrize("sql", [
    "INSERT INTO meta(key, value) VALUES('x', '1')",
    "UPDATE facts SET text = 'something else'",
    "DELETE FROM facts",
    "DROP TABLE facts",
    "CREATE TABLE extra(a)",
    "ALTER TABLE facts ADD COLUMN extra TEXT",
])
def test_a_guarded_connection_refuses_to_change_anything(conn, sql):
    deny_writes(conn)

    with pytest.raises(sqlite3.DatabaseError):
        conn.execute(sql)


def test_a_refused_write_changes_nothing(conn):
    """Denied at prepare time, so there is no half-applied statement to undo."""
    before = conn.execute("SELECT COUNT(*) FROM facts").fetchone()[0]
    deny_writes(conn)

    with pytest.raises(sqlite3.DatabaseError):
        conn.execute("DELETE FROM facts")

    allow_writes(conn)
    assert conn.execute("SELECT COUNT(*) FROM facts").fetchone()[0] == before


def test_reads_still_work_under_the_guard(conn):
    deny_writes(conn)

    assert conn.execute("SELECT COUNT(*) FROM facts").fetchone()[0] == 1


def test_full_text_search_still_works_under_the_guard(conn):
    """The one that is easy to break: instantiating an FTS5 table declares its
    schema through what the authorizer sees as an UPDATE of `sqlite_master`.
    Deny that and every search fails with a vtable constructor error, which is
    a guard that has quietly become a read outage."""
    deny_writes(conn)

    hits = conn.execute(
        "SELECT rowid FROM facts_fts WHERE facts_fts MATCH 'staging' LIMIT 1"
    ).fetchall()

    assert len(hits) == 1


def test_a_transaction_may_still_be_opened(conn):
    """`Store.transaction` issues BEGIN IMMEDIATE before it knows whether the
    body writes. Refusing the BEGIN would break reads that take the lock."""
    deny_writes(conn)

    conn.execute("BEGIN IMMEDIATE")
    conn.execute("ROLLBACK")


def test_allow_writes_lifts_the_guard(conn):
    deny_writes(conn)
    allow_writes(conn)

    conn.execute("INSERT INTO meta(key, value) VALUES('x', '1')")
    conn.commit()

    assert conn.execute("SELECT value FROM meta WHERE key='x'").fetchone()[0] == "1"


def test_readonly_restores_the_connection_afterwards(conn):
    with readonly(conn):
        with pytest.raises(sqlite3.DatabaseError):
            conn.execute("DELETE FROM facts")

    conn.execute("DELETE FROM facts")
    conn.commit()


def test_readonly_restores_the_connection_after_a_failure(conn):
    """A command that raises inside a dry run must not leave the process's
    connection refusing writes for whatever runs next."""
    with pytest.raises(ValueError):
        with readonly(conn):
            raise ValueError("something in the command went wrong")

    conn.execute("DELETE FROM facts")
    conn.commit()


# ---------- B2/B3: no command that offers the flag may write ----------


def _turn(role: str, text: str, session: str, cwd: str) -> str:
    return json.dumps({
        "type": role, "sessionId": session, "cwd": cwd,
        "message": {"role": role, "content": [{"type": "text", "text": text}]},
    })


def _tool_turn(session: str, cwd: str, path: str) -> str:
    return json.dumps({
        "type": "assistant", "sessionId": session, "cwd": cwd,
        "message": {"role": "assistant", "content": [
            {"type": "tool_use", "name": "Edit", "input": {"file_path": path}},
        ]},
    })


@pytest.fixture
def transcript(tmp_path):
    root = tmp_path / "projects" / "repo"
    root.mkdir(parents=True)
    path = root / "s-contract.jsonl"
    path.write_text("\n".join([
        _turn("user", "always use the staging bucket for exports", "s-c", "/repo"),
        _tool_turn("s-c", "/repo", "backend/app/exports.py"),
    ]))
    return path


# Every command that declares `--dry-run`, with arguments that make it do real
# work. Adding a command with the flag and not adding it here fails the registry
# test below.
COVERED = ("audit", "backfill", "learn", "observe")


def _cases(transcript):
    glob = str(transcript.parent.parent / "**" / "*.jsonl")
    return {
        "audit": ["audit", "--dry-run"],
        "backfill": ["backfill", "--dry-run", "--glob", glob],
        "learn": ["learn", str(transcript), "--dry-run"],
        "observe": ["observe", str(transcript), "--dry-run"],
    }


@pytest.fixture
def seeded(tmp_path, transcript):
    """A store with something in it, so "nothing changed" is a real claim."""
    db = tmp_path / "s.db"
    _run(["write", "the staging bucket is the export target"], db)
    _run(["backfill", "--glob", str(transcript.parent.parent / "**" / "*.jsonl")], db)
    return db


def _run(args, db, **env):
    return subprocess.run(
        [sys.executable, "-m", "nenapu.cli", *args, "--db", str(db)],
        capture_output=True, text=True,
        env={**os.environ, "PYTHONPATH": "src", "NENAPU_NO_BANNER": "1",
             **NO_MODEL, **env},
    )


def _snapshot(db) -> dict[str, int]:
    """Row counts for every table, which is what "wrote nothing" has to mean."""
    conn = connect(str(db))
    try:
        tables = [r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table'"
            " AND name NOT LIKE 'sqlite_%'"
        )]
        return {t: conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
                for t in tables}
    finally:
        conn.close()


@pytest.mark.parametrize("command", COVERED)
def test_a_dry_run_writes_nothing(command, seeded, transcript):
    before = _snapshot(seeded)

    _run(_cases(transcript)[command], seeded)

    assert _snapshot(seeded) == before


def test_every_command_offering_the_flag_is_covered(transcript):
    """The contract rots the first time a command declares `--dry-run` and
    nobody remembers to hold it to the promise. This is what remembers."""
    declared = set()
    for command in app.registered_commands:
        for parameter in inspect.signature(command.callback).parameters.values():
            default = parameter.default
            if (isinstance(default, typer.models.OptionInfo)
                    and "--dry-run" in (default.param_decls or ())):
                declared.add(command.name or command.callback.__name__)

    assert declared == set(_cases(transcript)), (
        "a command offers --dry-run but nothing proves it writes nothing"
    )
