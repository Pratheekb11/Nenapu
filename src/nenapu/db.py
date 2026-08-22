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

SCHEMA_VERSION = 11


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

-- Semantic retrieval's half of the index. One row per embedded fact, keyed by
-- `fact_id` so a fact cannot carry two vectors and be returned twice at two
-- scores. `model` and `dim` are recorded because a model switch has to be
-- detectable: two embedding spaces mixed in one dot product produce numbers
-- that look fine and mean nothing. `text_sha` covers stores that carried
-- vectors before the triggers below existed.
CREATE TABLE IF NOT EXISTS fact_vectors (
    fact_id    INTEGER PRIMARY KEY,
    model      TEXT NOT NULL,
    dim        INTEGER NOT NULL,
    text_sha   TEXT NOT NULL,
    vec        BLOB NOT NULL,
    created_at REAL NOT NULL
);

-- These triggers invalidate, they never embed. SQLite cannot call the model,
-- and a trigger that tried would put inference on the write path; a missing
-- row is the signal `index_missing` looks for.
--
-- Scoped to `text` for the reason `facts_au` above is scoped to the indexed
-- columns: recall bumps `use_count` on every fact it surfaces, and an
-- unscoped AFTER UPDATE would throw away a vector on the most common write in
-- the system and then pay to recompute it. Status changes are excluded for
-- the same reason: `forget` is reversible and re-embedding is the expensive
-- half, so a retired fact keeps the vector it already paid for.
CREATE TRIGGER IF NOT EXISTS facts_vec_au AFTER UPDATE OF text ON facts BEGIN
    DELETE FROM fact_vectors WHERE fact_id = old.id;
END;

-- Deletion, unlike retirement, is final. This is also what keeps `purge`
-- correct without purge having to learn the table exists.
CREATE TRIGGER IF NOT EXISTS facts_vec_ad AFTER DELETE ON facts BEGIN
    DELETE FROM fact_vectors WHERE fact_id = old.id;
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
    external_id       TEXT, -- the transcript's own session id, for backfill idempotency
    source            TEXT   -- 'hook' if watched as it ran, 'backfill' if reconstructed
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
    detail      TEXT,
    -- Which source a grade from this job is recorded under. Null is a live
    -- session; `grade --replay` sets it so backfilled evidence stays
    -- distinguishable from evidence that arrived as the sessions ran.
    grade_source TEXT
);

CREATE INDEX IF NOT EXISTS idx_ingest_queue_state ON ingest_queue(state, enqueued_at);

-- What the watcher has seen, so a transcript is ingested once and a session
-- still being written is left alone. Keyed by path because that is the only
-- identifier every agent's transcript format is guaranteed to have.
CREATE TABLE IF NOT EXISTS watch_state (
    path          TEXT PRIMARY KEY,
    agent         TEXT NOT NULL,
    last_size     INTEGER NOT NULL DEFAULT 0,
    last_mtime    REAL,
    seen_at       REAL,          -- when the size above was observed
    ingested_at   REAL,
    ingested_size INTEGER,       -- so a resumed session is picked up again
    session_id    TEXT
);

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

-- Things said but not done. Kept out of `facts` on purpose: a loop needs the
-- evidence that would close it, a status and a reason it was retired, none of
-- which a belief about the world wants. The ageing rule it does share with
-- facts is a function, not a column.
CREATE TABLE IF NOT EXISTS open_loops (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    scope           TEXT NOT NULL,
    text            TEXT NOT NULL,
    resolution_hint TEXT,           -- path globs the work would touch
    kind            TEXT NOT NULL DEFAULT 'stated',  -- stated | interrupted
    status          TEXT NOT NULL DEFAULT 'open',    -- open | closed
    opened_at       REAL NOT NULL,
    closed_at       REAL,
    close_reason    TEXT,
    session_id      TEXT
);

CREATE INDEX IF NOT EXISTS idx_open_loops_scope ON open_loops(scope, status);

-- Working memory: verbatim turns, privacy-gated and off by default (see
-- NENAPU_STORE_MESSAGES in observer.py). Everything else in this file holds
-- extracted, redacted facts; this is the one table that can hold raw
-- conversation, so it exists behind an explicit opt-in.
CREATE TABLE IF NOT EXISTS messages (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT,
    seq        INTEGER NOT NULL,
    role       TEXT NOT NULL,
    text       TEXT NOT NULL,
    created_at REAL NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id, seq);

-- The entity tier: files, dirs, commits, services — joined to belief
-- through fact_entities. `role='subject'` is load-bearing: a fact *about*
-- a deleted file dies with it, a fact that merely *mentions* it does not.
CREATE TABLE IF NOT EXISTS entities (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    kind       TEXT NOT NULL,   -- repo|dir|file|commit|command|service|person|concept
    name       TEXT NOT NULL,
    scope      TEXT NOT NULL DEFAULT 'global',
    status     TEXT NOT NULL DEFAULT 'alive',   -- alive | gone
    first_seen REAL NOT NULL,
    last_seen  REAL NOT NULL,
    mentions   INTEGER NOT NULL DEFAULT 0,
    UNIQUE(kind, name, scope)
);

CREATE TABLE IF NOT EXISTS entity_edges (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    src_id       INTEGER NOT NULL,
    dst_id       INTEGER NOT NULL,
    kind         TEXT NOT NULL,   -- contains|touched_with|changed_in|calls|runs|owns|alias_of
    source       TEXT NOT NULL,   -- observed|declared|inferred
    weight       REAL NOT NULL DEFAULT 1.0,
    observations INTEGER NOT NULL DEFAULT 1,
    valid_from   REAL NOT NULL,
    valid_to     REAL,            -- NULL = still true
    UNIQUE(src_id, dst_id, kind)
);

CREATE TABLE IF NOT EXISTS fact_entities (
    fact_id   INTEGER NOT NULL,
    entity_id INTEGER NOT NULL,
    role      TEXT NOT NULL,   -- subject | mentions
    source    TEXT NOT NULL,   -- key | path | observed | model
    PRIMARY KEY (fact_id, entity_id, role)
);

CREATE INDEX IF NOT EXISTS idx_entity_edges_src ON entity_edges(src_id);
CREATE INDEX IF NOT EXISTS idx_entity_edges_dst ON entity_edges(dst_id);
CREATE INDEX IF NOT EXISTS idx_fact_entities_entity ON fact_entities(entity_id);
CREATE INDEX IF NOT EXISTS idx_entities_scope_status ON entities(scope, status);
CREATE INDEX IF NOT EXISTS idx_entities_name ON entities(name);
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


# Actions that change what the store holds. Everything else — SELECT, READ,
# PRAGMA, FUNCTION, TRANSACTION, SAVEPOINT — is a read or a no-op and is let
# through, so a guarded connection is still fully usable for answering.
_WRITE_ACTIONS = frozenset({
    sqlite3.SQLITE_INSERT,
    sqlite3.SQLITE_UPDATE,
    sqlite3.SQLITE_DELETE,
    sqlite3.SQLITE_ALTER_TABLE,
    sqlite3.SQLITE_CREATE_INDEX,
    sqlite3.SQLITE_CREATE_TABLE,
    sqlite3.SQLITE_CREATE_TRIGGER,
    sqlite3.SQLITE_CREATE_VIEW,
    sqlite3.SQLITE_DROP_INDEX,
    sqlite3.SQLITE_DROP_TABLE,
    sqlite3.SQLITE_DROP_TRIGGER,
    sqlite3.SQLITE_DROP_VIEW,
    sqlite3.SQLITE_REINDEX,
})

# SQLite declares a virtual table's schema by what the authorizer reports as an
# UPDATE of `sqlite_master`. Denying that breaks every FTS5 query with "vtable
# constructor failed" — a guard that has quietly become a read outage. No
# statement can write `sqlite_master` directly, so the exemption costs nothing.
_SCHEMA_TABLE = "sqlite_master"


def _refuse_writes(action: int, arg1, arg2, db_name, trigger) -> int:
    if action in _WRITE_ACTIONS and arg1 != _SCHEMA_TABLE:
        return sqlite3.SQLITE_DENY
    return sqlite3.SQLITE_OK


def deny_writes(conn: sqlite3.Connection) -> None:
    """Refuse every statement on this connection that would change data.

    `--dry-run` used to be a boolean each write path had to remember to
    consult, and the paths that forgot were invisible until someone ran the
    command against a real store: a `backfill --redate --dry-run` that wrote,
    and three more in `learn` alone. Remembering does not scale. The
    connection refuses instead, and a refused write raises
    `sqlite3.DatabaseError` at prepare time rather than writing, which is the
    loud failure a silent one deserves.

    One write a dry run still performs, and it happens before this is
    installed: `connect` creates and migrates the store if it does not exist.
    The guarantee is about data, not about the file existing.
    """
    conn.set_authorizer(_refuse_writes)


def _permit_everything(action: int, arg1, arg2, db_name, trigger) -> int:
    return sqlite3.SQLITE_OK


def allow_writes(conn: sqlite3.Connection) -> None:
    """Lift the guard. The inverse of `deny_writes`, and safe if none is set.

    A permissive callback rather than `set_authorizer(None)`: on Python 3.10
    None is accepted and then ignored, so the old callback stays installed and
    the connection keeps refusing writes for the rest of its life. Measured on
    3.10.12, and this package supports 3.10. Only a connection that was guarded
    ever carries a callback, so no ordinary connection pays for one.
    """
    conn.set_authorizer(_permit_everything)


@contextlib.contextmanager
def readonly(conn: sqlite3.Connection):
    """Hold `deny_writes` for a block, and lift it even if the block raises.

    A command that fails partway through a dry run must not leave the
    connection refusing writes for whatever runs next in the same process.
    """
    deny_writes(conn)
    try:
        yield conn
    finally:
        allow_writes(conn)


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
    "ingest_queue": [
        ("grade_source", "TEXT"),
    ],
    "sessions": [
        # Whether the row was watched as it ran or reconstructed from history.
        # `backfill --redate` used to infer this from `git_head_before`, which
        # is also NULL for a live session that ran outside a git repo, so the
        # repair moved live rows onto transcript timestamps. Rows written
        # before this column existed carry NULL and fall back to the old rule.
        ("source", "TEXT"),
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
