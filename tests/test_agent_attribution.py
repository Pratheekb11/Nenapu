"""Which agent wrote this, and which agent's write path touched the DB.

Requirement (Task 2, priority-ordered task list, Phase 3 of the plan):

* `facts.agent TEXT` — an additive column (through `_ADDED_COLUMNS`,
  `db.py:358`, the same mechanism `good_recalls`/`suspect_since` used), so an
  existing store migrates without a destructive rebuild.
* Populated from the `NENAPU_AGENT` environment variable when a caller does
  not say otherwise.
* The MCP write path (`mcp_server.memory_write`) currently calls
  `store.write(fact, derived_from=derived_from)` with no `actor` — which
  means every MCP-originated journal row is stamped with `Store.write`'s
  generic default, `"agent"` (`store.py:241`). That is indistinguishable
  from any other caller and is exactly the gap the plan calls out:
  "you cannot ask which agent told me this."

Without this column, "which agent edited what" — the actual cross-project
requirement task 3 builds on — is unanswerable, because there is nowhere to
put the answer.
"""

import pytest

from nenapu import connect
from nenapu.models import Fact
from nenapu.store import Store


@pytest.fixture
def store():
    return Store(connect(":memory:"))


def test_the_facts_table_has_an_agent_column():
    """Additive migration: a brand-new store must carry the column."""
    conn = connect(":memory:")
    columns = {r["name"] for r in conn.execute("PRAGMA table_info(facts)")}
    assert "agent" in columns


def test_an_existing_v5_store_gains_the_column_on_reconnect(tmp_path):
    """The regression that matters: a *real*, pre-existing database (367 live
    facts, in the measured store) must migrate additively, not require a
    rebuild that would drop history."""
    path = tmp_path / "old.db"
    connect(str(path))  # simulate an existing store created before this change
    conn = connect(str(path))  # reconnect, as every real invocation does

    columns = {r["name"] for r in conn.execute("PRAGMA table_info(facts)")}
    assert "agent" in columns


def test_fact_dataclass_exposes_agent(store):
    """The column is only useful if it round-trips through `Store.write` /
    `row_to_fact`, not just sitting in the schema unread."""
    fact = Fact(text="touched booking.py", agent="claude-code")
    stored, _ = store.write(fact)
    assert stored.agent == "claude-code"


def test_agent_defaults_from_the_environment_variable(store, monkeypatch):
    """`NENAPU_AGENT` is how a non-Claude-Code caller (a future watcher
    adapter, Codex, a plain CLI invocation) identifies itself without every
    call site having to pass `agent=` explicitly."""
    monkeypatch.setenv("NENAPU_AGENT", "codex")
    fact = Fact(text="ran from codex")  # agent left unset
    stored, _ = store.write(fact)
    assert stored.agent == "codex"


def test_mcp_write_path_sets_a_real_actor(tmp_path, monkeypatch):
    """`mcp_server.memory_write` must stamp the journal with something that
    identifies the MCP surface, not fall through to `Store.write`'s generic
    default. Today it does exactly that (`mcp_server.py:147`); this pins the
    fix."""
    monkeypatch.setenv("NENAPU_DB", str(tmp_path / "mcp.db"))
    import importlib

    from nenapu import mcp_server
    m = importlib.reload(mcp_server)

    result = m.memory_write("the public endpoint has no rate limit", kind="project")

    store = Store(connect(str(tmp_path / "mcp.db")))
    row = store.conn.execute(
        "SELECT actor FROM journal WHERE fact_id = ? AND action = 'write'",
        (result["id"],),
    ).fetchone()
    assert row["actor"] != "agent", "MCP writes are still stamped with the generic default"


def test_mcp_write_path_sets_agent_on_the_fact_itself(tmp_path, monkeypatch):
    monkeypatch.setenv("NENAPU_DB", str(tmp_path / "mcp.db"))
    monkeypatch.setenv("NENAPU_AGENT", "cursor")
    import importlib

    from nenapu import mcp_server
    m = importlib.reload(mcp_server)

    result = m.memory_write("cursor wrote this one", kind="project")

    store = Store(connect(str(tmp_path / "mcp.db")))
    stored = store.get(result["id"])
    assert stored.agent == "cursor"
