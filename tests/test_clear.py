"""Clearing the store.

The one command here that can empty a store, so the tests are mostly about
what it refuses to do: retire rather than delete unless asked, keep the history
either way, and never assume consent from something that is not a person.
"""

import os
import subprocess
import sys

import pytest

from nenapu import connect
from nenapu.models import Fact, Kind, Origin, Status
from nenapu.store import Store


@pytest.fixture
def store():
    s = Store(connect(":memory:"))
    for i in range(3):
        s.write(Fact(text=f"a fact about the api, number {i}", kind=Kind.PROJECT,
                     origin=Origin.USER_STATED, scope="api"))
    for i in range(2):
        s.write(Fact(text=f"a fact about the ui, number {i}", kind=Kind.ENVIRONMENT,
                     origin=Origin.TOOL_OBSERVED, scope="ui"))
    return s


def _run(args, tmp_path, **env):
    return subprocess.run(
        [sys.executable, "-m", "nenapu.cli", *args, "--db", str(tmp_path / "m.db")],
        capture_output=True, text=True,
        env={**os.environ, "PYTHONPATH": "src", "NENAPU_NO_BANNER": "1", **env},
    )


def test_clearing_retires_rather_than_deletes(store):
    """A store that has been emptied should still be able to say what it used
    to know and who cleared it. Deleting is a different word."""
    cleared = store.forget_all()

    assert cleared == 5
    assert store.stats()["active"] == 0
    assert len(store.list_facts(status=Status.RETIRED, limit=50)) == 5
    actions = [r["action"] for r in
               store.conn.execute("SELECT action FROM journal").fetchall()]
    assert "forget-all" in actions


def test_clearing_a_scope_leaves_the_others_alone(store):
    assert store.forget_all(scope="api") == 3
    assert store.stats()["active"] == 2
    assert all(f.scope == "ui" for f in store.list_facts(limit=50))


def test_clearing_a_kind_leaves_the_others_alone(store):
    assert store.forget_all(kind=Kind.ENVIRONMENT) == 2
    assert {f.kind for f in store.list_facts(limit=50)} == {Kind.PROJECT}


def test_clearing_twice_is_not_an_error(store):
    store.forget_all()

    assert store.forget_all() == 0


def test_purge_takes_the_rows_and_what_hangs_off_them(store):
    """An edge to a row that no longer exists is worse than no edge, and a
    recall that cannot name what it recalled cannot be graded."""
    facts = store.list_facts(limit=50)
    store.graph.link(facts[0].id, facts[1].id)
    store.ledger.log_many(store.search("api"), session_id="s1", query="api")

    gone = store.purge()

    assert gone == 5
    for table in ("facts", "fact_edges", "recalls"):
        left = store.conn.execute(f"SELECT COUNT(*) c FROM {table}").fetchone()["c"]
        assert left == 0, f"{table} still has {left} row(s)"


def test_purge_still_says_it_happened(store):
    """A store that cannot say why it is empty is indistinguishable from a
    broken one."""
    store.purge()

    detail = store.conn.execute(
        "SELECT detail FROM journal WHERE action = 'purge'").fetchone()
    assert detail and "5" in detail["detail"]


def test_purge_leaves_the_search_index_consistent(store):
    """FTS is a separate table kept in step by triggers. A delete that skipped
    it would leave `search` returning rows that are not there."""
    store.purge()

    assert store.search("api") == []


# ---------- the command, and what it refuses ----------


def test_a_pipe_is_not_consent(tmp_path):
    """The one command that can empty a store, so a non-interactive run refuses
    rather than assuming. `--yes` exists for people who mean it."""
    _run(["write", "something worth keeping"], tmp_path)

    result = _run(["clear"], tmp_path)

    assert result.returncode == 1
    assert "refusing" in result.stdout
    assert _run(["list"], tmp_path).stdout.count("something worth keeping") == 1


def test_yes_means_yes(tmp_path):
    _run(["write", "something worth keeping"], tmp_path)

    result = _run(["clear", "--yes"], tmp_path)

    assert result.returncode == 0
    assert "retired 1" in result.stdout


def test_forget_all_is_the_same_thing(tmp_path):
    """`forget all` is what people type. It is `clear`, confirmation and all."""
    _run(["write", "a fact"], tmp_path)

    result = _run(["forget", "all", "--yes"], tmp_path)

    assert result.returncode == 0
    assert "retired 1" in result.stdout


def test_forget_still_wants_a_number_otherwise(tmp_path):
    result = _run(["forget", "banana"], tmp_path)

    assert result.returncode != 0
    assert "fact id" in (result.stdout + result.stderr)


def test_clearing_an_empty_store_says_so(tmp_path):
    result = _run(["clear", "--yes"], tmp_path)

    assert result.returncode == 0
    assert "nothing to clear" in result.stdout
