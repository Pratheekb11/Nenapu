import pytest
from fastapi.testclient import TestClient

from nenapu.api import create_app


@pytest.fixture
def client(tmp_path):
    return TestClient(create_app(str(tmp_path / "api.db")))


def test_write_search_roundtrip(client):
    client.post("/facts", json={"text": "CI runs on GitHub Actions", "kind": "project"})
    results = client.get("/facts/search", params={"q": "CI GitHub"}).json()["results"]
    assert results and "GitHub" in results[0]["text"]
    assert 0.0 < results[0]["confidence"] <= 1.0


def test_conflicts_are_reported_to_the_caller(client):
    client.post("/facts", json={"text": "queue depth cap is 100", "key": "queue.cap",
                                "origin": "user_stated", "confidence": 0.9})
    body = client.post("/facts", json={"text": "queue depth cap is 500", "key": "queue.cap",
                                       "origin": "user_stated", "confidence": 0.95}).json()
    assert body["conflicts"][0]["resolution"] == "superseded"


def test_verify_endpoint_refuses_unapproved_shell(client):
    """The HTTP surface is reachable by anything on localhost, so it must not
    be a way around the approval gate."""
    fact_id = client.post(
        "/facts", json={"text": "echo works", "verify_cmd": "echo hi", "verify_expect": "hi"}
    ).json()["stored"]["id"]
    assert client.post(f"/facts/{fact_id}/verify").json()["status"] == "blocked"


def test_verify_endpoint_runs_an_approved_check(client, tmp_path):
    from nenapu.approval import approve
    from nenapu.db import connect

    fact_id = client.post(
        "/facts", json={"text": "echo works", "verify_cmd": "echo hi", "verify_expect": "hi"}
    ).json()["stored"]["id"]

    conn = connect(str(tmp_path / "api.db"))       # same store the app opened
    approve(conn, "echo hi", fact_id=fact_id, by="test")

    assert client.post(f"/facts/{fact_id}/verify").json()["status"] == "pass"


def test_missing_fact_is_404(client):
    assert client.delete("/facts/9999").status_code == 404
    assert client.post("/facts/9999/verify").status_code == 404


def test_skill_outcome_quarantines(client):
    client.post("/skills", json={"name": "flaky", "body": "..."})
    for _ in range(4):
        body = client.post("/skills/flaky/outcome", json={"outcome": "failure"}).json()
    assert body["status"] == "quarantined"
    assert client.get("/skills/search", params={"q": "flaky"}).json()["results"] == []


def test_export_endpoint(client):
    client.post("/facts", json={"text": "deploy on Tuesdays", "origin": "user_stated",
                                "confidence": 0.9})
    md = client.get("/export").json()["markdown"]
    assert "BEGIN NENAPU" in md and "Tuesdays" in md


def test_handlers_survive_the_threadpool(client):
    # Each sync endpoint runs on a worker thread; a shared sqlite connection
    # would raise ProgrammingError here.
    for _ in range(6):
        assert client.get("/health").json()["ok"] is True


def test_second_connection_skips_schema_setup(tmp_path):
    import time

    from nenapu.db import connect

    path = tmp_path / "reconnect.db"
    connect(path).close()
    start = time.perf_counter()
    conn = connect(path)
    elapsed = time.perf_counter() - start
    assert conn.execute("SELECT COUNT(*) c FROM facts").fetchone()["c"] == 0
    assert elapsed < 0.1  # provisioning is skipped on an existing store
