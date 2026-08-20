"""SQLite schema and connection handling.

One file, no server, no daemon to babysit. FTS5 ships with the stdlib sqlite3
on every platform we care about; the whole store is `~/.nenapu/nenapu.db` and is
safe to copy, diff, or check into a private repo.
"""

from __future__ import annotations

import contextlib
import os
import random
import sqlite3
import time
from pathlib import Path

SCHEMA_VERSION = 5


def default_db_path() -> Path:
    root = os.environ.get("NENAPU_HOME")
    return Path(root).expanduser() if root else Path.home() / ".nenapu"


SCHEMA = """
CREATE TABLE IF NOT EXISTS facts (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    text              TEXT NOT NULL,
    kind              TEXT NOT NULL,
    scope             TEXT NOT NULL DEFAULT 'global',
    key               TEXT,
    origin            TEXT NOT NULL,
    origin_ref        TEXT,
    session_id        TEXT,
    confidence        REAL NOT NULL DEFAULT 0.7,
    decay_class       TEXT NOT NULL,
    verify_cmd        TEXT,
    verify_expect     TEXT,
    tags_csv          TEXT NOT NULL DEFAULT '',
    status            TEXT NOT NULL DEFAULT 'active',
    created_at        REAL NOT NULL,
    updated_at        REAL NOT NULL,
    last_verified_at  REAL,
    verify_status     TEXT NOT NULL DEFAULT 'none',
    verify_last_run   REAL,
    verify_detail     TEXT,
    supersedes_id     INTEGER,
    superseded_by_id  INTEGER,
    distilled_into_id INTEGER,
    use_count         INTEGER NOT NULL DEFAULT 0,
    last_used_at      REAL
);

CREATE INDEX IF NOT EXISTS idx_facts_scope_status ON facts(scope, status);

-- Backstop for the duplicate-write race. Two processes can both look for an
-- existing copy, both miss, and both insert; the transaction in Store.write
-- prevents that, and this makes it unrepresentable even if some future caller
-- writes outside one. Partial, so superseded and retired history is unaffected.
CREATE UNIQUE INDEX IF NOT EXISTS idx_facts_active_unique
    ON facts(scope, text) WHERE status = 'active';
CREATE INDEX IF NOT EXISTS idx_facts_key          ON facts(scope, key, status);
CREATE INDEX IF NOT EXISTS idx_facts_verify       ON facts(verify_cmd) WHERE verify_cmd IS NOT NULL;

CREATE VIRTUAL TABLE IF NOT EXISTS facts_fts USING fts5(
    text, key, tags_csv,
    content='facts', content_rowid='id', tokenize='porter unicode61'
);

CREATE TRIGGER IF NOT EXISTS facts_ai AFTER INSERT ON facts BEGIN
    INSERT INTO facts_fts(rowid, text, key, tags_csv)
    VALUES (new.id, new.text, COALESCE(new.key,''), new.tags_csv);
END;
CREATE TRIGGER IF NOT EXISTS facts_ad AFTER DELETE ON facts BEGIN
    INSERT INTO facts_fts(facts_fts, rowid, text, key, tags_csv)
    VALUES ('delete', old.id, old.text, COALESCE(old.key,''), old.tags_csv);
END;
-- Scoped to the indexed columns on purpose. An unscoped AFTER UPDATE re-indexed
-- the row's full text every time recall bumped `use_count`, which made the most
-- common operation in the system pay for a full-text delete and reinsert.
CREATE TRIGGER IF NOT EXISTS facts_au AFTER UPDATE OF text, key, tags_csv ON facts BEGIN
    INSERT INTO facts_fts(facts_fts, rowid, text, key, tags_csv)
    VALUES ('delete', old.id, old.text, COALESCE(old.key,''), old.tags_csv);
    INSERT INTO facts_fts(rowid, text, key, tags_csv)
    VALUES (new.id, new.text, COALESCE(new.key,''), new.tags_csv);
END;

-- The belief network. An edge says the child would not have been concluded
-- without the parent, so falsifying a parent must put the child in doubt.
CREATE TABLE IF NOT EXISTS fact_edges (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    parent_id  INTEGER NOT NULL,
    child_id   INTEGER NOT NULL,
    kind       TEXT NOT NULL DEFAULT 'derived_from',
    source     TEXT NOT NULL DEFAULT 'declared',
    weight     REAL NOT NULL DEFAULT 1.0,
    created_at REAL NOT NULL,
    UNIQUE(parent_id, child_id, kind)
);

CREATE INDEX IF NOT EXISTS idx_edges_parent ON fact_edges(parent_id);
CREATE INDEX IF NOT EXISTS idx_edges_child  ON fact_edges(child_id);

-- Every fact surfaced into a task, graded later by whichever signal arrives.
-- This is what lets a memory lose standing because acting on it went badly,
-- rather than only because it got old.
CREATE TABLE IF NOT EXISTS recalls (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    fact_id        INTEGER NOT NULL,
    session_id     TEXT,
    query          TEXT NOT NULL DEFAULT '',
    rank           INTEGER NOT NULL DEFAULT 0,
    score          REAL NOT NULL DEFAULT 0.0,
    outcome        TEXT NOT NULL DEFAULT 'pending',
    outcome_source TEXT,
    outcome_at     REAL,
    note           TEXT,
    created_at     REAL NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_recalls_fact    ON recalls(fact_id, outcome);
CREATE INDEX IF NOT EXISTS idx_recalls_session ON recalls(session_id, created_at);
CREATE INDEX IF NOT EXISTS idx_recalls_pending ON recalls(outcome, created_at);

CREATE TABLE IF NOT EXISTS conflicts (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    fact_id    INTEGER NOT NULL,
    other_id   INTEGER NOT NULL,
    key        TEXT,
    detail     TEXT NOT NULL,
    resolution TEXT NOT NULL,
    created_at REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS skills (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    name              TEXT NOT NULL UNIQUE,
    description       TEXT NOT NULL DEFAULT '',
    body              TEXT NOT NULL,
    scope             TEXT NOT NULL DEFAULT 'global',
    tags_csv          TEXT NOT NULL DEFAULT '',
    status            TEXT NOT NULL DEFAULT 'active',
    quarantine_reason TEXT,
    created_at        REAL NOT NULL,
    updated_at        REAL NOT NULL,
    invocations       INTEGER NOT NULL DEFAULT 0,
    successes         INTEGER NOT NULL DEFAULT 0,
    failures          INTEGER NOT NULL DEFAULT 0,
    last_used_at      REAL
);

CREATE VIRTUAL TABLE IF NOT EXISTS skills_fts USING fts5(
    name, description, body,
    content='skills', content_rowid='id', tokenize='porter unicode61'
);

CREATE TRIGGER IF NOT EXISTS skills_ai AFTER INSERT ON skills BEGIN
    INSERT INTO skills_fts(rowid, name, description, body)
    VALUES (new.id, new.name, new.description, new.body);
END;
CREATE TRIGGER IF NOT EXISTS skills_ad AFTER DELETE ON skills BEGIN
    INSERT INTO skills_fts(skills_fts, rowid, name, description, body)
    VALUES ('delete', old.id, old.name, old.description, old.body);
END;
CREATE TRIGGER IF NOT EXISTS skills_au AFTER UPDATE OF name, description, body ON skills BEGIN
    INSERT INTO skills_fts(skills_fts, rowid, name, description, body)
    VALUES ('delete', old.id, old.name, old.description, old.body);
    INSERT INTO skills_fts(rowid, name, description, body)
    VALUES (new.id, new.name, new.description, new.body);
END;

CREATE TABLE IF NOT EXISTS skill_events (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    skill_id   INTEGER NOT NULL,
    outcome    TEXT NOT NULL,          -- success | failure | used
    session_id TEXT,
    note       TEXT,
    created_at REAL NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_skill_events ON skill_events(skill_id, created_at);

CREATE TABLE IF NOT EXISTS journal (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    action     TEXT NOT NULL,
    fact_id    INTEGER,
    skill_id   INTEGER,
    actor      TEXT,
    detail     TEXT,
    created_at REAL NOT NULL
);

-- Commands a human has explicitly allowed to run.
--
-- `verify_cmd` is shell, and facts are written by agents that read untrusted
-- input — a web page, a dependency README, a file in a cloned repo. Without
-- this table, one prompt-injected `memory_write` turns `nenapu check` into
-- scheduled remote code execution. A check does not run until its exact
-- command appears here.
CREATE TABLE IF NOT EXISTS approved_commands (
    sha256      TEXT PRIMARY KEY,
    command     TEXT NOT NULL,
    fact_id     INTEGER,
    approved_at REAL NOT NULL,
    approved_by TEXT NOT NULL DEFAULT 'user'
);

CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

-- The work-activity ledger: where you did what, which agent, in which repo.
-- Deterministic — filled from git and transcript tool calls, never a model.
CREATE TABLE IF NOT EXISTS sessions (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    agent             TEXT NOT NULL,
    project_scope     TEXT NOT NULL,
    cwd               TEXT,
    git_branch        TEXT,
    git_head_before   TEXT,
    git_head_after    TEXT,
    started_at        REAL NOT NULL,
    ended_at          REAL,
    summary           TEXT,
    external_id       TEXT  -- the transcript's own session id, for backfill idempotency
);

CREATE INDEX IF NOT EXISTS idx_sessions_scope ON sessions(project_scope, started_at);
CREATE INDEX IF NOT EXISTS idx_sessions_external ON sessions(external_id);

CREATE TABLE IF NOT EXISTS file_events (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id INTEGER NOT NULL,
    path       TEXT NOT NULL,
    op         TEXT NOT NULL,  -- created | edited | deleted | read
    tool       TEXT,
    at         REAL NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_file_events_session ON file_events(session_id);
CREATE INDEX IF NOT EXISTS idx_file_events_path    ON file_events(path);

CREATE TABLE IF NOT EXISTS commits (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id    INTEGER,
    sha           TEXT NOT NULL,
    subject       TEXT,
    files_changed TEXT NOT NULL DEFAULT '[]',
    at            REAL NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_commits_session ON commits(session_id);

-- Durable, single-flight ingestion queue. The Stop hook enqueues and
-- returns; one worker holding an exclusive lock drains this strictly
-- serially, so sessions ending together never fan out into concurrent
-- model calls against the one store.
CREATE TABLE IF NOT EXISTS ingest_queue (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    path        TEXT NOT NULL,
    agent       TEXT NOT NULL,
    session_id  TEXT,
    enqueued_at REAL NOT NULL,
    state       TEXT NOT NULL DEFAULT 'pending',  -- pending | claimed | done | failed
    claimed_at  REAL,
    finished_at REAL,
    detail      TEXT
);

CREATE INDEX IF NOT EXISTS idx_ingest_queue_state ON ingest_queue(state, enqueued_at);

-- Downsampled activity: a work log compressed by age, not similarity. One
-- row replaces a whole period's sessions once they age out of full detail.
CREATE TABLE IF NOT EXISTS rollups (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    project_scope TEXT NOT NULL,
    period        TEXT NOT NULL,  -- week | month
    period_start  REAL NOT NULL,
    period_end    REAL NOT NULL,
    session_count INTEGER NOT NULL DEFAULT 0,
    files_touched INTEGER NOT NULL DEFAULT 0,
    commits       INTEGER NOT NULL DEFAULT 0,
    loops_opened  INTEGER NOT NULL DEFAULT 0,
    loops_closed  INTEGER NOT NULL DEFAULT 0,
    UNIQUE(project_scope, period, period_start)
);

CREATE INDEX IF NOT EXISTS idx_rollups_scope ON rollups(project_scope, period);
"""


# Write contention: retry with jittered backoff rather than surfacing a lock
# error to a user who only asked to remember something.
LOCK_RETRIES = 6
LOCK_BACKOFF = 0.05


@contextlib.contextmanager
def transaction(conn: sqlite3.Connection):
    """Serialize a read-modify-write, and batch its writes into one commit.

    Two jobs in one primitive, because they are the same mechanism:

    * Correctness — `BEGIN IMMEDIATE` takes the write lock *before* the read a
      decision depends on. A deferred transaction takes it at the first write,
      which is too late: two processes both check for an existing fact, both
      see none, and both insert.
    * Speed — in autocommit every statement is its own durable write. Eight
      recall rows meant eight fsyncs, which made recall (the hottest path here)
      take seconds on a store of any size.

    Re-entrant via `in_transaction`, so nested calls across Store, Graph and
    Ledger join the outermost transaction instead of committing early.
    """
    if conn.in_transaction:
        yield
        return

    for attempt in range(LOCK_RETRIES):
        try:
            conn.execute("BEGIN IMMEDIATE")
            break
        except sqlite3.OperationalError as exc:
            if "locked" not in str(exc).lower() and "busy" not in str(exc).lower():
                raise
            if attempt == LOCK_RETRIES - 1:
                raise
            time.sleep(LOCK_BACKOFF * (2**attempt) * (0.5 + random.random()))

    try:
        yield
        commit(conn)
    except BaseException:
        with contextlib.suppress(sqlite3.Error):
            conn.execute("ROLLBACK")
        raise


def commit(conn: sqlite3.Connection) -> None:
    """Commit only if a transaction is open.

    Connections run in autocommit, so a bare `commit()` outside an explicit
    `BEGIN` raises "cannot commit - no transaction is active". Call sites that
    may or may not be inside `Store.transaction()` use this instead.
    """
    if conn.in_transaction:
        conn.commit()


def _make_private(target: Path) -> None:
    """Create the store owner-only, and repair a store that predates this.

    The file holds facts extracted from private sessions — paths, hostnames,
    whatever the user pasted. The process umask left it 0644 inside a 0755
    directory, so on any shared machine every account could read it. SQLite
    also writes `-wal` and `-shm` beside it, which carry the same content and
    are created by the driver, so they are narrowed too once they exist.

    Best effort by design: a store on a filesystem with no POSIX permissions
    (a mounted share, Windows) must still open. Being unable to tighten the
    mode is not a reason to refuse someone their memory.
    """
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.chmod(target.parent, 0o700)
        if not target.exists():
            # Created here rather than left to the driver: sqlite3 opens with
            # the process umask, so a file it creates is 0644 for the moment
            # before we could chmod it, and that moment is enough on a shared
            # box. An empty file is a valid empty database.
            os.close(os.open(target, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600))
        else:
            os.chmod(target, 0o600)
    except OSError:
        pass
    for sidecar in ("-wal", "-shm"):
        path = target.with_name(target.name + sidecar)
        try:
            if path.exists():
                os.chmod(path, 0o600)
        except OSError:
            pass


def connect(path: str | Path | None = None) -> sqlite3.Connection:
    """Open (creating if needed) the store and apply the schema."""
    if path is None or str(path) in ("", ":memory:"):
        target = ":memory:" if str(path) == ":memory:" else default_db_path() / "nenapu.db"
    else:
        target = Path(path).expanduser()

    if target != ":memory:":
        _make_private(Path(target))

    # isolation_level=None puts the driver in autocommit, so transactions are
    # explicit (`BEGIN IMMEDIATE` in Store.transaction) rather than implicitly
    # started on the first write. Without that, a read-modify-write's SELECT
    # happens outside the transaction it is supposed to be protected by.
    conn = sqlite3.connect(str(target), isolation_level=None, timeout=30)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys=ON")
    # Concurrent writers should queue, not fail: MCP server, CLI and cron all
    # write to the same file.
    conn.execute("PRAGMA busy_timeout=10000")
    # WAL + NORMAL is the standard durable-but-fast pairing. FULL fsyncs on
    # every commit, which cost ~85ms per remembered fact here — untenable when
    # an agent writes a fact mid-task. NORMAL cannot corrupt the database; the
    # exposure is losing the most recent commits if the machine loses power,
    # which for a memory store is an acceptable trade against being too slow to
    # use. Set NENAPU_SYNCHRONOUS=FULL to opt back in.
    conn.execute(f"PRAGMA synchronous={os.environ.get('NENAPU_SYNCHRONOUS', 'NORMAL')}")

    if _schema_version(conn) == SCHEMA_VERSION:
        # Already provisioned. Re-running the script is harmless but not free —
        # every `CREATE IF NOT EXISTS` costs a catalogue read, and the MCP
        # server opens a connection per worker thread.
        return conn

    conn.execute("PRAGMA journal_mode=WAL")
    if target != ":memory:":
        _make_private(Path(target))  # -wal and -shm exist only now
    _drop_replaced_triggers(conn)
    conn.executescript(SCHEMA)
    _add_missing_columns(conn)
    conn.execute(
        "INSERT OR REPLACE INTO meta(key, value) VALUES ('schema_version', ?)",
        (str(SCHEMA_VERSION),),
    )
    conn.commit()
    return conn


# Columns added after v1. `CREATE TABLE IF NOT EXISTS` will not add them to a
# store that already exists, so they are applied separately.
_ADDED_COLUMNS: dict[str, list[tuple[str, str]]] = {
    "facts": [
        ("good_recalls", "INTEGER NOT NULL DEFAULT 0"),
        ("bad_recalls", "INTEGER NOT NULL DEFAULT 0"),
        ("suspect_since", "REAL"),
        ("suspect_reason", "TEXT"),
        ("agent", "TEXT"),
        ("occurrences", "INTEGER NOT NULL DEFAULT 1"),
    ],
}


# Triggers cannot be altered in place, and an old store carries the unscoped
# versions that made every recall re-index. Dropped so the schema script can
# recreate them in their scoped form.
_REPLACED_TRIGGERS = ("facts_au", "skills_au")


def _drop_replaced_triggers(conn: sqlite3.Connection) -> None:
    for name in _REPLACED_TRIGGERS:
        conn.execute(f"DROP TRIGGER IF EXISTS {name}")


def _add_missing_columns(conn: sqlite3.Connection) -> None:
    for table, columns in _ADDED_COLUMNS.items():
        have = {r["name"] for r in conn.execute(f"PRAGMA table_info({table})")}
        for name, decl in columns:
            if name not in have:
                conn.execute(f"ALTER TABLE {table} ADD COLUMN {name} {decl}")


def _schema_version(conn: sqlite3.Connection) -> int | None:
    try:
        row = conn.execute("SELECT value FROM meta WHERE key = 'schema_version'").fetchone()
    except sqlite3.OperationalError:  # meta table does not exist yet
        return None
    return int(row["value"]) if row else None
