"""The activity ledger: sessions, file_events, commits.

Requirement (Task 3, priority-ordered task list) — this is the table the rest
of the "actual requirement" section of the plan is built on:

    "I work on multiple projects and I sometimes lose track of things —
    where I did what, which agent edited what file, which file was added,
    which file was deleted, what was implemented, what is yet to be
    implemented."

Three tables, deterministic, no model calls (`docs` "Proposed tier: the
activity ledger"):

    sessions(id, agent, project_scope, cwd, git_branch,
             git_head_before, git_head_after, started_at, ended_at, summary)
    file_events(id, session_id, path, op, tool, at)
                 op IN created | edited | deleted | read
    commits(id, session_id, sha, subject, files_changed, at)

Task 3's scope is the storage layer only: the tables exist, are additive
(migrate an existing store without dropping the 367 live facts alongside
them), and a minimal write/query API round-trips through them. Populating
`file_events`/`commits` from real transcripts and `git diff` is Task 4
(marked Opus 5 in the plan — git rename/merge/worktree edge cases decide
correctness there) and is out of scope for these tests.

This test file assumes the write/query surface lives in a new
`nenapu.activity` module, mirroring the existing `outcomes.Ledger` /
`graph.Graph` pattern (a small class wrapping the connection). That module
name is this test suite's proposal for the contract, not a fact already in
the codebase — the schema-level tests below (table/column existence) do not
depend on it and will still make sense if the implementation lands the API
somewhere else.
"""

import pytest

from nenapu import connect

# ---------- schema: implementation-location-agnostic ----------


def test_the_ledger_tables_exist_on_a_new_store():
    conn = connect(":memory:")
    tables = {r["name"] for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type = 'table'"
    )}
    assert {"sessions", "file_events", "commits"} <= tables


def test_sessions_table_has_the_documented_columns():
    conn = connect(":memory:")
    columns = {r["name"] for r in conn.execute("PRAGMA table_info(sessions)")}
    assert columns >= {
        "id", "agent", "project_scope", "cwd", "git_branch",
        "git_head_before", "git_head_after", "started_at", "ended_at", "summary",
    }


def test_file_events_table_has_the_documented_columns():
    conn = connect(":memory:")
    columns = {r["name"] for r in conn.execute("PRAGMA table_info(file_events)")}
    assert columns >= {"id", "session_id", "path", "op", "tool", "at"}


def test_commits_table_has_the_documented_columns():
    conn = connect(":memory:")
    columns = {r["name"] for r in conn.execute("PRAGMA table_info(commits)")}
    assert columns >= {"id", "session_id", "sha", "subject", "files_changed", "at"}


def test_an_existing_store_gains_the_ledger_tables_on_reconnect(tmp_path):
    """The migration must be additive, matching `_add_missing_columns` /
    `CREATE TABLE IF NOT EXISTS` for every other table in `db.py` — an
    existing store's 367 facts must survive untouched."""
    path = tmp_path / "old.db"
    conn = connect(str(path))
    conn.execute("INSERT INTO facts (text, kind, origin, decay_class, created_at, updated_at)"
                 " VALUES ('pre-existing fact', 'project', 'user_stated', 'medium', 0, 0)")
    conn.commit()

    reopened = connect(str(path))
    tables = {r["name"] for r in reopened.execute(
        "SELECT name FROM sqlite_master WHERE type = 'table'"
    )}
    assert {"sessions", "file_events", "commits"} <= tables
    assert reopened.execute("SELECT COUNT(*) c FROM facts").fetchone()["c"] == 1


# ---------- proposed write/query API ----------


@pytest.fixture
def ledger():
    from nenapu.activity import ActivityLedger

    return ActivityLedger(connect(":memory:"))


def test_recording_a_session_round_trips(ledger):
    session_id = ledger.start_session(
        agent="claude-code", project_scope="repo:demo@aaaaaaaa", cwd="/repo",
        git_branch="main", git_head_before="abc123",
    )
    ledger.end_session(session_id, git_head_after="def456", summary="fixed the bug")

    session = ledger.get_session(session_id)
    assert session["agent"] == "claude-code"
    assert session["project_scope"] == "repo:demo@aaaaaaaa"
    assert session["git_head_before"] == "abc123"
    assert session["git_head_after"] == "def456"
    assert session["ended_at"] is not None


def test_recording_file_events(ledger):
    session_id = ledger.start_session(agent="claude-code", project_scope="repo:demo@aaaaaaaa",
                                       cwd="/repo")
    ledger.record_file_event(session_id, path="backend/app/bookings.py", op="edited",
                             tool="Edit")
    ledger.record_file_event(session_id, path="backend/app/new_file.py", op="created",
                             tool="Write")

    events = ledger.file_events_for_session(session_id)
    assert {(e["path"], e["op"]) for e in events} == {
        ("backend/app/bookings.py", "edited"),
        ("backend/app/new_file.py", "created"),
    }


def test_recording_a_commit(ledger):
    session_id = ledger.start_session(agent="claude-code", project_scope="repo:demo@aaaaaaaa",
                                       cwd="/repo")
    ledger.record_commit(session_id, sha="c7f1a9d4", subject="Add booking overlap constraint",
                         files_changed=["backend/app/bookings.py"])

    commits = ledger.commits_for_session(session_id)
    assert commits[0]["sha"] == "c7f1a9d4"
    assert commits[0]["subject"] == "Add booking overlap constraint"


def test_sessions_are_queryable_by_project_scope(ledger):
    """`nenapu where <file>` / `nenapu project <name>` (Task 6) both need to
    ask "everything for this repo", so this is the read path they sit on."""
    a = ledger.start_session(agent="claude-code", project_scope="repo:a@11111111", cwd="/a")
    ledger.start_session(agent="claude-code", project_scope="repo:b@22222222", cwd="/b")

    sessions = ledger.sessions_for_scope("repo:a@11111111")
    assert [s["id"] for s in sessions] == [a]


def test_which_agent_touched_a_file_is_answerable(ledger):
    """The concrete question the plan names: "which agent edited what"."""
    claude_session = ledger.start_session(agent="claude-code",
                                          project_scope="repo:a@11111111", cwd="/a")
    codex_session = ledger.start_session(agent="codex",
                                         project_scope="repo:a@11111111", cwd="/a")
    ledger.record_file_event(claude_session, path="app.py", op="edited", tool="Edit")
    ledger.record_file_event(codex_session, path="app.py", op="edited", tool="apply_patch")

    touches = ledger.file_events_for_path("app.py")
    agents = {t["agent"] for t in touches}
    assert agents == {"claude-code", "codex"}
