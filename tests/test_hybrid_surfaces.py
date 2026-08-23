"""The new ranking signals, visible on every surface that already explains one.

Requirement (Task 7, query-driven hybrid retrieval plan):

Three surfaces already answer "why did this memory surface" -- `nenapu recall`,
`memory_search(explain=True)` and `GET /facts/search`. Retrieval just gained
two terms and a mode, and a term that moves ranking while staying invisible is
a silent re-ranker, which is the thing `tests/test_store.py` already refuses on
the scoring path. So all three have to report them.

The second half matters as much as the first: none of the three may break when
the semantic leg is absent. That is the common case -- fastembed is optional and
is not installed in CI -- so the keys have to be present and zero rather than
missing, and no surface may raise looking for one.

Scope boundary
--------------
Whether the numbers are *right* is Tasks 4 to 6, pinned in their own files.
This file pins only that each surface carries them and survives their absence.
"""

import json
import os
import subprocess
import sys

import pytest
from fastapi.testclient import TestClient

from nenapu import connect, embeddings
from nenapu.api import create_app
from nenapu.models import Fact
from nenapu.store import Store

QUERY = "which database do we use"
ANSWER = "the datastore is postgres"
SIGNALS = ("semantic", "entity", "mode")


@pytest.fixture
def seeded(tmp_path):
    """A store on disk, so the subprocess and HTTP surfaces can reach it."""
    path = tmp_path / "s.db"
    store = Store(connect(str(path)))
    store.write(Fact(text=ANSWER))
    store.write(Fact(text="the deploy script lives in bin/release"))
    store.conn.close()
    return path


def _run(args, db, **env):
    return subprocess.run(
        [sys.executable, "-m", "nenapu.cli", *args, "--db", str(db)],
        capture_output=True, text=True,
        env={**os.environ, "PYTHONPATH": "src", "NENAPU_NO_BANNER": "1", **env},
    )


# --- the command line --------------------------------------------------------


def test_recall_json_carries_the_new_signals(seeded):
    out = _run(["recall", "deploy", "--json"], seeded, NENAPU_EMBEDDINGS="off")

    assert out.returncode == 0
    hits = json.loads(out.stdout)
    assert hits
    for key in SIGNALS:
        assert key in hits[0]


def test_recall_explains_the_anchor_and_the_leg_in_the_table(seeded):
    """The table is what a person reads. It gained the two columns rather than
    hiding them behind --json, which is the machine surface."""
    out = _run(["recall", "deploy", "--explain"], seeded, NENAPU_EMBEDDINGS="off")

    assert out.returncode == 0
    assert "sem" in out.stdout
    assert "near" in out.stdout


def test_the_plain_table_is_unchanged(seeded):
    """Explaining is opt-in. Six columns is already a lot for a terminal, and
    someone recalling a fact usually wants the fact."""
    out = _run(["recall", "deploy"], seeded, NENAPU_EMBEDDINGS="off")

    assert out.returncode == 0
    assert "sem" not in out.stdout


def test_recall_survives_a_missing_embedder(seeded):
    out = _run(["recall", "deploy", "--explain"], seeded, NENAPU_EMBEDDINGS="off")

    assert out.returncode == 0
    assert "Traceback" not in out.stderr


# --- the MCP surface ---------------------------------------------------------


def _server(monkeypatch, db):
    monkeypatch.setenv("NENAPU_DB", str(db))
    import importlib

    from nenapu import mcp_server

    return importlib.reload(mcp_server)


def test_memory_search_explains_the_new_signals(monkeypatch, seeded):
    m = _server(monkeypatch, seeded)

    out = m.memory_search("deploy", explain=True)

    why = out["results"][0]["why"]
    for key in SIGNALS:
        assert key in why


def test_the_lean_payload_stays_lean(monkeypatch, seeded):
    """`explain` is opt-in on this surface too. Eight results carrying five
    scoring terms each is a real cost in an agent's context window."""
    m = _server(monkeypatch, seeded)

    out = m.memory_search("deploy")

    assert "why" not in out["results"][0]


# --- the HTTP surface --------------------------------------------------------


def test_the_search_endpoint_carries_the_new_signals(seeded):
    client = TestClient(create_app(str(seeded)))

    results = client.get("/facts/search", params={"q": "deploy"}).json()["results"]

    assert results
    for key in SIGNALS:
        assert key in results[0]["why"]


# --- absence -----------------------------------------------------------------


def test_every_surface_reports_zero_rather_than_nothing(monkeypatch, seeded):
    """Keys present and zero, never missing. A caller reading `why["semantic"]`
    must not have to guard for a machine that never installed the extra."""
    monkeypatch.setattr(embeddings, "get_embedder", lambda: None)
    store = Store(connect(str(seeded)))

    (_f, _score, why), *_ = store.search("deploy", log_recall=False, mark_used=False)

    assert why["semantic"] == 0.0
    assert why["entity"] == 0.0
    assert why["mode"] == "lexical"
