import asyncio
import json


def _server(tmp_path, monkeypatch):
    monkeypatch.setenv("NENAPU_DB", str(tmp_path / "mcp.db"))
    import importlib

    from nenapu import mcp_server

    return importlib.reload(mcp_server)


def _tools(m):
    return asyncio.run(m.mcp.list_tools())


def test_agent_surface_is_what_an_agent_needs_mid_task(tmp_path, monkeypatch):
    m = _server(tmp_path, monkeypatch)
    names = {t.name for t in _tools(m)}
    assert names == {
        "memory_search", "memory_write", "memory_verify", "memory_forget",
        "memory_why", "memory_loops", "task_outcome",
        "skill_search", "skill_write", "skill_record_outcome",
    }


def test_operator_jobs_stay_off_the_agent_surface(tmp_path, monkeypatch):
    # Export, audit, distill and explicit linking are cron/CLI work. Leaving
    # them registered would tax every request in the session for nothing.
    m = _server(tmp_path, monkeypatch)
    names = {t.name for t in _tools(m)}
    for operator_job in ("memory_export", "memory_audit", "memory_distill",
                         "memory_link", "memory_stats"):
        assert operator_job not in names
        assert callable(getattr(m, operator_job))  # still usable from CLI/HTTP


def test_tool_surface_stays_within_budget(tmp_path, monkeypatch):
    """Tool schemas sit in context on every request. Regressions here are paid
    on every turn of every session, so the budget is a test, not a guideline."""
    m = _server(tmp_path, monkeypatch)
    estimate = sum(
        len(json.dumps({"name": t.name, "description": t.description or "",
                        "input_schema": t.input_schema})) // 4
        for t in _tools(m)
    )
    assert estimate < 1600, f"tool surface grew to ~{estimate} tokens"


def test_recall_payload_is_lean_by_default(tmp_path, monkeypatch):
    m = _server(tmp_path, monkeypatch)
    for i in range(8):
        m.memory_write(f"service {i} listens on port {8000 + i}", kind="environment")
    results = m.memory_search("service port", limit=8)["results"]

    assert len(results) == 8
    per_result = len(json.dumps(results)) // 4 // 8
    assert per_result < 45, f"~{per_result} tokens per result"
    # Predictable values are omitted; only id, text, confidence, recall_id remain.
    assert set(results[0]) == {"id", "text", "confidence", "recall_id"}


def test_explain_is_opt_in(tmp_path, monkeypatch):
    m = _server(tmp_path, monkeypatch)
    m.memory_write("the cache is redis", kind="environment")
    plain = m.memory_search("cache")["results"][0]
    detailed = m.memory_search("cache", explain=True)["results"][0]
    assert "why" not in plain
    assert {"lexical", "confidence", "age_days"} <= set(detailed["why"])


def test_exceptions_are_surfaced_but_defaults_are_not(tmp_path, monkeypatch):
    m = _server(tmp_path, monkeypatch)
    m.memory_write("root fact", verify_cmd="false")
    m.memory_verify(scope="")
    result = m.memory_search("root fact")["results"][0]
    # An agent-written command is inert until a human approves it, and the
    # agent is told so rather than being left to assume the check passed.
    assert result["check"] == "unapproved"
    assert "scope" not in result          # default, so it is not


def test_an_agent_cannot_approve_its_own_shell_command(tmp_path, monkeypatch):
    """The whole point of the gate: nothing reachable over MCP grants
    execution. Approval is a human action at the CLI."""
    m = _server(tmp_path, monkeypatch)
    canary = tmp_path / "pwned"
    out = m.memory_write("harmless looking fact", verify_cmd=f"touch {canary}")
    assert "awaiting" in out["check"]

    report = m.memory_verify(scope="")
    assert report["awaiting_approval"] == [out["id"]]
    assert not canary.exists()

    assert not any("approve" in t.name for t in _tools(m))


def test_write_then_search_explains_itself(tmp_path, monkeypatch):
    m = _server(tmp_path, monkeypatch)
    m.memory_write("The staging database is postgres 16", kind="environment", key="db.version")
    hit = m.memory_search("staging database", explain=True)["results"][0]
    assert "postgres" in hit["text"]
    assert {"lexical", "confidence", "age_days"} <= set(hit["why"])


def test_conflicting_write_returns_a_note(tmp_path, monkeypatch):
    m = _server(tmp_path, monkeypatch)
    m.memory_write("retry limit is 3", key="retry.limit", origin="user_stated", confidence=0.9)
    out = m.memory_write("retry limit is 10", key="retry.limit", origin="user_stated",
                         confidence=0.95)
    assert out["conflicts"][0]["resolution"] == "superseded"
    assert "contradicts" in out["note"].lower()


def test_forget_and_loops(tmp_path, monkeypatch):
    m = _server(tmp_path, monkeypatch)
    fact_id = m.memory_write("temporary note")["id"]
    assert m.memory_forget(fact_id) == {"retired": fact_id}
    assert m.memory_forget(9999)["error"]
    assert m.memory_loops()["active"] == 0


def test_task_outcome_grades_a_session(tmp_path, monkeypatch):
    m = _server(tmp_path, monkeypatch)
    m.memory_write("deploy with make ship")
    m.memory_search("deploy", session_id="task-7")
    assert m.task_outcome(session_id="task-7", success=False)["graded"] == 1


# ==========================================================================
# Pre-written for the MCP half of G8 and E10.
#
# G8. The human grading path is unreachable — `nenapu helped` and `nenapu
# misled` want a recall id no command prints — and the agent path is
# reachable but unprompted. One line in the `task_outcome` description tells
# an agent to call it at task end, which is the cheapest half of closing the
# loop.
#
# E10. `Graph.why` prints belief ancestry only. Adding the fact's subject
# entity and its neighbourhood means a human sees why a fact *surfaced*, not
# only why it is *believed* — and `memory_why` is the surface an agent asks
# that question through.
#
# The tool surface is paid for on every request of every session, so the
# budget test above is the guard on both of these.
# ==========================================================================


def _tool(m, name):
    return next(t for t in _tools(m) if t.name == name)


def test_task_outcome_tells_an_agent_when_to_call_it(tmp_path, monkeypatch):
    """An agent that is never told when to grade never grades: 480 of 483
    recalls in the live store are still pending."""
    m = _server(tmp_path, monkeypatch)

    description = _tool(m, "task_outcome").description.lower()

    assert "task end" in description or "end of the task" in description


def test_the_tool_surface_did_not_grow_a_command(tmp_path, monkeypatch):
    """G8 adds a CLI listing and one line of description. A new MCP tool would
    be paid for on every request by every agent, which is not what "make the
    human path reachable" asks for."""
    m = _server(tmp_path, monkeypatch)

    assert {t.name for t in _tools(m)} == {
        "memory_search", "memory_write", "memory_verify", "memory_forget",
        "memory_why", "memory_loops", "task_outcome",
        "skill_search", "skill_write", "skill_record_outcome",
    }


def test_memory_why_shows_the_entity_layer(tmp_path, monkeypatch):
    from nenapu.entities import EntityGraph

    m = _server(tmp_path, monkeypatch)
    fact_id = m.memory_write("services/auth/routes.py owns the login handler")["id"]
    store, _ = m._stores()
    graph = EntityGraph(store.conn)
    entity = graph.upsert(kind="file", name="services/auth/routes.py", scope="global")
    graph.attach(fact_id, entity.id, role="subject", source="path")

    answer = m.memory_why(fact_id)

    assert answer["subject_entity"]["name"] == "services/auth/routes.py"


def test_memory_why_on_a_fact_with_no_entity_is_unchanged(tmp_path, monkeypatch):
    """A store that has never built the entity tier gets exactly today's
    answer, which is what keeps the rest of this file true."""
    m = _server(tmp_path, monkeypatch)
    fact_id = m.memory_write("deploy with make ship")["id"]

    answer = m.memory_why(fact_id)

    assert answer["id"] == fact_id
    assert "confidence" in answer and "track_record" in answer
    assert answer.get("subject_entity") in (None, {})
