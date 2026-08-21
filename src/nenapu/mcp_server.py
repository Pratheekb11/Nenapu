"""MCP server — the widest-reach surface.

One server, and every MCP client (Claude Code, Cursor, Windsurf, Claude
Desktop, anything else that speaks the protocol) gets the same verified memory.
That is the portability claim: memory lives here, not inside one agent.

Run:  nenapu-mcp
Or:   uvx --from nenapu nenapu-mcp
"""

from __future__ import annotations

import os

try:  # current SDK layout
    from mcp.server.mcpserver import MCPServer as _Server
except ImportError:  # SDK <= the FastMCP naming
    from mcp.server.fastmcp import FastMCP as _Server

from .audit import LLMUnavailable
from .audit import audit as run_audit
from .distill import distill as run_distill
from .export import write_file
from . import ThreadLocalStores
from .models import Fact, Skill, Status, VerifyStatus
from .store import effective_confidence
from .verify import apply_result, run_check, verify_scope

mcp = _Server("nenapu", version="0.1.0")

# Tool definitions ride in the context window on every single request, whether
# or not memory is touched. Only operations an agent performs *mid-task* earn
# that permanent cost; maintenance jobs (export, audit, distill, explicit
# linking) live in the CLI and HTTP API, where they cost nothing per turn.

# Tool handlers are synchronous, so the server may run them on a worker
# thread; sqlite connections cannot cross threads.
_stores = ThreadLocalStores(os.environ.get("NENAPU_DB"))


# Fields worth spending tokens on only when they deviate from the norm. A
# result set is read on every recall, so anything predictable is noise.
def _fact_view(
    fact: Fact,
    score: float | None = None,
    explain: dict | None = None,
    *,
    verbose: bool = False,
) -> dict:
    view: dict = {
        "id": fact.id,
        "text": fact.text,
        "confidence": round(effective_confidence(fact), 2),
    }
    if explain and "recall_id" in explain:
        view["recall_id"] = explain["recall_id"]

    # Exceptions only. `active`, `none`, and the default scope carry no
    # information, and repeating them across eight results costs more than the
    # facts themselves.
    if fact.status != Status.ACTIVE:
        view["status"] = fact.status
        if fact.suspect_reason:
            view["reason"] = fact.suspect_reason
    if fact.verify_status == VerifyStatus.FAIL:
        view["check"] = "failing"
    elif fact.verify_status == VerifyStatus.BLOCKED:
        # Distinct from failing: nothing has been tested, so the fact is
        # neither confirmed nor doubted.
        view["check"] = "unapproved"
    if fact.scope != "global":
        view["scope"] = fact.scope

    if verbose:
        view.update(
            kind=fact.kind, key=fact.key, origin=fact.origin,
            verify_status=fact.verify_status, scope=fact.scope, status=fact.status,
        )
        if score is not None:
            view["score"] = round(score, 3)
        if explain:
            view["why"] = explain
    return view


def memory_search(
    query: str,
    scope: str = "",
    limit: int = 8,
    min_confidence: float = 0.0,
    session_id: str = "",
    explain: bool = False,
) -> dict:
    """Recall facts, ranked by text match and current believability.

    Pass a stable `session_id` per task: facts you write later get linked to
    what you recalled, and `task_outcome` can grade what the task used.
    Fields appear only when notable (status, failing check). `explain=True`
    adds scoring detail.
    """
    store, _ = _stores()
    hits = store.search(
        query, scope=scope or None, limit=limit, min_confidence=min_confidence,
        session_id=session_id or None,
    )
    return {"results": [_fact_view(f, s, why, verbose=explain) for f, s, why in hits]}


def memory_write(
    text: str,
    kind: str = "project",
    scope: str = "global",
    key: str = "",
    origin: str = "agent_inferred",
    confidence: float = 0.7,
    decay_class: str = "",
    verify_cmd: str = "",
    verify_expect: str = "",
    session_id: str = "",
    derived_from: list[int] | None = None,
) -> dict:
    """Store a fact.

    key: dotted subject id (`db.port`) when the fact is one value for one
      subject — turns a later disagreement into a detected conflict.
    verify_cmd: shell command proving it; such facts never go stale silently.
      Stored inert — a human must approve it before it ever runs.
    derived_from: ids this rests on (inferred from `session_id` if omitted).

    kind: user|project|environment|feedback|reference
    origin: user_stated|tool_observed|file_derived|agent_inferred
    decay: immutable|slow|medium|volatile
    """
    store, _ = _stores()
    fact = Fact(
        text=text,
        kind=kind,
        scope=scope,
        key=key or None,
        origin=origin,
        confidence=confidence,
        decay_class=decay_class or None,
        verify_cmd=verify_cmd or None,
        verify_expect=verify_expect or None,
        session_id=session_id or None,
    )
    stored, conflicts = store.write(fact, actor="mcp", derived_from=derived_from)

    result: dict = {"id": stored.id, "confidence": round(effective_confidence(stored), 2)}
    if stored.verify_cmd:
        result["check"] = "stored, awaiting `nenapu approve` before it can run"
    parents = store.graph.parents(stored.id)
    if parents:
        result["rests_on"] = [pid for pid, _source, _w in parents]
    if conflicts:
        result["conflicts"] = [
            {"with": c.other_id, "detail": c.detail, "resolution": c.resolution}
            for c in conflicts
        ]
        result["note"] = "Contradicts existing memory — surface this to the user."
    return result


def memory_verify(fact_id: int = 0, scope: str = "", stale_after_days: float = 0.0) -> dict:
    """Re-run executable checks for one fact or a scope. Failures cascade to
    dependents. `stale_after_days` skips recently-checked facts.

    Checks run only after a human approves the exact command (`nenapu approve`);
    unapproved ones are returned in `awaiting_approval` and change nothing.
    """
    store, _ = _stores()
    if fact_id:
        fact = store.get(fact_id)
        if not fact:
            return {"error": f"no fact {fact_id}"}
        result = run_check(fact, conn=store.conn)
        fallout = apply_result(store, result)
        return {
            "results": [
                {"id": result.fact_id, "status": result.status, "detail": result.detail}
            ],
            "cascaded_to": fallout["cascaded"],
            "restored": fallout["restored"],
            "recalls_graded": fallout["graded_recalls"],
        }

    results = verify_scope(
        store,
        scope=scope or None,
        only_stale_after_days=stale_after_days or None,
    )
    cascaded = sorted({c for r in results for c in r.fallout.get("cascaded", [])})
    blocked = [r.fact_id for r in results if r.status == VerifyStatus.BLOCKED]
    return {
        "checked": len(results),
        # Surfaced so the agent tells the user rather than reporting a clean run.
        "awaiting_approval": blocked,
        "failing": [
            {"id": r.fact_id, "detail": r.detail} for r in results if r.status == "fail"
        ],
        "cascaded_to": cascaded,
        "results": [{"id": r.fact_id, "status": r.status} for r in results],
    }


def memory_forget(fact_id: int) -> dict:
    """Retire a fact. Journalled, never recalled again; dependents go suspect."""
    store, _ = _stores()
    if not store.get(fact_id):
        return {"error": f"no fact {fact_id}"}
    store.forget(fact_id)
    return {"retired": fact_id}


def memory_stats(scope: str = "") -> dict:
    """Counts: active, stale, disputed, suspect, failing, edges, recall grades."""
    store, _ = _stores()
    return store.stats(scope=scope or None)


def memory_export(path: str, scope: str = "", min_confidence: float = 0.35) -> dict:
    """Write verified memory into a CLAUDE.md / AGENTS.md managed block.

    For harnesses that read flat files rather than MCP. Content outside the
    nenapu markers is preserved.
    """
    store, _ = _stores()
    written = write_file(path, store, scope=scope or None, min_confidence=min_confidence)
    return {"written": str(written)}


def memory_audit(evidence: str = "", scope: str = "", older_than_days: float = 30.0) -> dict:
    """LLM re-check of soft facts that decay and shell checks cannot cover.

    Pass current evidence (repo state, recent findings) as `evidence`. Facts
    judged wrong are retired; stale ones are demoted to disputed for a human to
    settle. Costs tokens — this is a scheduled job, not a per-turn call.
    """
    store, _ = _stores()
    try:
        report = run_audit(
            store, evidence=evidence, scope=scope or None, older_than_days=older_than_days
        )
    except LLMUnavailable as exc:
        return {"error": str(exc)}
    return {
        "covered": f"{report.covered}/{report.requested}",
        "not_audited": report.missing,
        "findings": [
            {"id": f.fact_id, "verdict": f.verdict, "reason": f.reason} for f in report.findings
        ],
    }


def memory_distill(scope: str = "", token_budget: int = 1500, use_llm: bool = True) -> dict:
    """Compress a scope: drop near-duplicates, then merge related facts.

    Originals are archived with a pointer to the distilled fact, never deleted.
    """
    store, _ = _stores()
    try:
        report = run_distill(
            store, scope=scope or None, token_budget=token_budget, use_llm=use_llm
        )
    except LLMUnavailable as exc:
        return {"error": str(exc)}
    return {
        "tokens_before": report.tokens_before,
        "tokens_after": report.tokens_after,
        "saved_pct": round(report.saved_pct, 1),
        "deduped": report.deduped,
        "merged": report.merged,
    }


def task_outcome(
    session_id: str = "",
    recall_id: int = 0,
    success: bool = True,
    note: str = "",
) -> dict:
    """Report whether the recalled memory actually helped.

    Grade by `session_id` (everything the task used) or one `recall_id`. Facts
    that keep preceding failures lose confidence and stop surfacing. Call this
    at task end, once per task — it is what keeps recall quality honest.
    """
    store, _ = _stores()
    outcome = "good" if success else "bad"

    if recall_id:
        ok = store.ledger.grade(recall_id, outcome, source="agent", note=note or None)
        return {"graded": 1 if ok else 0, "recall_id": recall_id, "outcome": outcome}
    if not session_id:
        return {"error": "pass either session_id or recall_id"}

    graded = store.ledger.grade_session(session_id, outcome, source="agent", note=note or None)
    return {"graded": graded, "session_id": session_id, "outcome": outcome}


def memory_why(fact_id: int, depth: int = 3) -> dict:
    """Belief chain: what a fact rests on, what rests on it. Use when a memory
    matters, or turned out wrong and you need to know what else to distrust."""
    store, _ = _stores()
    fact = store.get(fact_id)
    if not fact:
        return {"error": f"no fact {fact_id}"}
    chain = store.graph.why(fact_id, depth=depth)
    chain["confidence"] = round(effective_confidence(fact), 3)
    chain["track_record"] = {"good": fact.good_recalls, "bad": fact.bad_recalls}
    return chain


def memory_link(parent_id: int, child_id: int) -> dict:
    """Declare that one fact rests on another, so falsification propagates."""
    store, _ = _stores()
    if not store.get(parent_id) or not store.get(child_id):
        return {"error": "both facts must exist"}
    edge = store.graph.link(parent_id, child_id)
    return {"linked": bool(edge), "parent": parent_id, "child": child_id}


def memory_loops(limit: int = 20) -> dict:
    """Memory the store no longer stands behind: contradicted, unsupported, or
    failing its check. Worth one call at session start."""
    store, _ = _stores()
    rows = store.conn.execute(
        "SELECT id, text, status, suspect_reason, verify_detail FROM facts"
        " WHERE status IN ('disputed','suspect') OR verify_status = 'fail'"
        " ORDER BY updated_at DESC LIMIT ?",
        (limit,),
    )
    stats = store.stats()
    return {
        "open_loops": [
            {
                "id": r["id"], "text": r["text"], "status": r["status"],
                "reason": (r["suspect_reason"] or r["verify_detail"] or "")[:120],
            }
            for r in rows
        ],
        "active": stats["active"],
        "stale": stats["stale_active"],
        "failing_checks": stats["failing_verification"],
        "pending_grades": len(store.ledger.pending(limit=200)),
    }


def skill_search(query: str, limit: int = 5) -> dict:
    """Find relevant skills. Quarantined ones are excluded."""
    _, skills = _stores()
    found = skills.search(query, limit=limit)
    return {
        "results": [
            {
                "name": s.name,
                "description": s.description,
                "body": s.body,
                "invocations": s.invocations,
                "success_rate": s.success_rate,
            }
            for s in found
        ]
    }


def skill_write(
    name: str, body: str, description: str = "", scope: str = "global", tags: list[str] | None = None
) -> dict:
    """Save or update a skill document."""
    _, skills = _stores()
    saved = skills.upsert(
        Skill(name=name, body=body, description=description, scope=scope, tags=tags or [])
    )
    return {"name": saved.name, "id": saved.id, "status": saved.status}


def skill_record_outcome(name: str, outcome: str, note: str = "") -> dict:
    """Report how a skill went: success | failure | used. Repeated failures
    quarantine it automatically."""
    _, skills = _stores()
    updated = skills.record_outcome(name, outcome, note=note or None)
    if not updated:
        return {"error": f"no skill named {name!r}"}
    return {
        "name": updated.name,
        "status": updated.status,
        "quarantine_reason": updated.quarantine_reason,
        "invocations": updated.invocations,
        "success_rate": updated.success_rate,
    }


# Tool schemas are resident context, so the right surface depends on what the
# deployment actually uses. A memory-only setup should not pay for the skill
# library, and a narrow agent should not pay for either.
TOOL_PROFILES: dict[str, tuple[str, ...]] = {
    # Enough to store, recall, and close the outcome loop. Nothing else.
    "minimal": ("memory_search", "memory_write", "task_outcome"),
    # Everything an agent does with memory mid-task.
    "memory": (
        "memory_search", "memory_write", "memory_verify", "memory_forget",
        "memory_why", "memory_loops", "task_outcome",
    ),
    # Memory plus the graded skill library.
    "full": (
        "memory_search", "memory_write", "memory_verify", "memory_forget",
        "memory_why", "memory_loops", "task_outcome",
        "skill_search", "skill_write", "skill_record_outcome",
    ),
}


def register_tools(profile: str | None = None) -> list[str]:
    """Expose one profile's tools. Operator jobs are never registered."""
    profile = (profile or os.environ.get("NENAPU_TOOLS", "full")).lower()
    if profile not in TOOL_PROFILES:
        raise SystemExit(
            f"unknown NENAPU_TOOLS={profile!r}; expected {', '.join(TOOL_PROFILES)}"
        )
    for name in TOOL_PROFILES[profile]:
        mcp.tool()(globals()[name])
    return list(TOOL_PROFILES[profile])


register_tools()


def main() -> None:
    mcp.run()


if __name__ == "__main__":
    main()
