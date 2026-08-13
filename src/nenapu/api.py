"""Local HTTP API for custom harnesses.

Same operations as the MCP server, over JSON, for agents that do not speak MCP.
Binds to localhost by default — this is a personal store, not a service.
"""

from __future__ import annotations

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from . import ThreadLocalStores
from .audit import LLMUnavailable
from .audit import audit as run_audit
from .distill import distill as run_distill
from .export import render
from .models import Fact, Skill
from .store import effective_confidence
from .verify import apply_result, run_check, verify_scope


class FactIn(BaseModel):
    text: str
    kind: str = "project"
    scope: str = "global"
    key: str | None = None
    origin: str = "agent_inferred"
    origin_ref: str | None = None
    confidence: float = 0.7
    decay_class: str | None = None
    verify_cmd: str | None = None
    verify_expect: str | None = None
    tags: list[str] = Field(default_factory=list)
    session_id: str | None = None
    derived_from: list[int] | None = None


class SkillIn(BaseModel):
    name: str
    body: str
    description: str = ""
    scope: str = "global"
    tags: list[str] = Field(default_factory=list)


class OutcomeIn(BaseModel):
    outcome: str
    note: str | None = None
    session_id: str | None = None


class GradeIn(BaseModel):
    success: bool
    session_id: str | None = None
    recall_id: int | None = None
    note: str | None = None


class LinkIn(BaseModel):
    parent_id: int
    child_id: int


def _view(fact: Fact, score: float | None = None, why: dict | None = None) -> dict:
    out = {
        "id": fact.id, "text": fact.text, "kind": fact.kind, "scope": fact.scope,
        "key": fact.key, "origin": fact.origin, "status": fact.status,
        "confidence": round(effective_confidence(fact), 3),
        "verify_status": fact.verify_status,
    }
    if score is not None:
        out["score"] = round(score, 3)
    if why:
        out["why"] = why
    return out


def create_app(db: str | None = None) -> FastAPI:
    # Endpoints are sync, so FastAPI runs them on a threadpool; each worker
    # thread needs its own sqlite connection.
    stores = ThreadLocalStores(db)
    app = FastAPI(title="nenapu", version="0.1.0")

    @app.get("/health")
    def health() -> dict:
        store, skills = stores()
        return {"ok": True, **store.stats()}

    @app.get("/facts")
    def list_facts(scope: str | None = None, status: str = "active", limit: int = 50) -> dict:
        store, skills = stores()
        return {"facts": [_view(f) for f in store.list_facts(scope=scope, status=status, limit=limit)]}

    @app.get("/facts/search")
    def search(
        q: str,
        scope: str | None = None,
        limit: int = 10,
        min_confidence: float = 0.0,
        session_id: str | None = None,
    ) -> dict:
        store, skills = stores()
        hits = store.search(
            q, scope=scope, limit=limit, min_confidence=min_confidence, session_id=session_id
        )
        return {"results": [_view(f, s, why) for f, s, why in hits]}

    @app.get("/facts/{fact_id}/why")
    def why(fact_id: int, depth: int = 3) -> dict:
        store, skills = stores()
        fact = store.get(fact_id)
        if not fact:
            raise HTTPException(404, f"no fact {fact_id}")
        chain = store.graph.why(fact_id, depth=depth)
        chain["confidence"] = round(effective_confidence(fact), 3)
        return chain

    @app.post("/links")
    def link(body: LinkIn) -> dict:
        store, skills = stores()
        if not store.get(body.parent_id) or not store.get(body.child_id):
            raise HTTPException(404, "both facts must exist")
        edge = store.graph.link(body.parent_id, body.child_id)
        return {"linked": bool(edge)}

    @app.post("/outcome")
    def grade(body: GradeIn) -> dict:
        store, skills = stores()
        verdict = "good" if body.success else "bad"
        if body.recall_id:
            ok = store.ledger.grade(body.recall_id, verdict, source="agent", note=body.note)
            return {"graded": int(ok), "outcome": verdict}
        if not body.session_id:
            raise HTTPException(400, "pass session_id or recall_id")
        graded = store.ledger.grade_session(
            body.session_id, verdict, source="agent", note=body.note
        )
        return {"graded": graded, "outcome": verdict}

    @app.get("/loops")
    def loops(limit: int = 20) -> dict:
        store, skills = stores()
        rows = store.conn.execute(
            "SELECT id, text, status, suspect_reason, verify_detail FROM facts"
            " WHERE status IN ('disputed','suspect') OR verify_status = 'fail'"
            " ORDER BY updated_at DESC LIMIT ?",
            (limit,),
        )
        return {
            "open_loops": [
                {
                    "id": r["id"], "text": r["text"], "status": r["status"],
                    "reason": r["suspect_reason"] or r["verify_detail"],
                }
                for r in rows
            ],
            "pending_grades": len(store.ledger.pending(limit=500)),
        }

    @app.post("/facts")
    def write(body: FactIn) -> dict:
        store, skills = stores()
        payload = body.model_dump()
        derived_from = payload.pop("derived_from", None)
        fact, conflicts = store.write(
            Fact(**payload), actor="http", derived_from=derived_from
        )
        return {
            "stored": _view(fact),
            "conflicts": [
                {"with_fact_id": c.other_id, "detail": c.detail, "resolution": c.resolution}
                for c in conflicts
            ],
        }

    @app.delete("/facts/{fact_id}")
    def forget(fact_id: int) -> dict:
        store, skills = stores()
        if not store.get(fact_id):
            raise HTTPException(404, f"no fact {fact_id}")
        store.forget(fact_id, actor="http")
        return {"retired": fact_id}

    @app.post("/facts/{fact_id}/verify")
    def verify_one(fact_id: int) -> dict:
        store, skills = stores()
        fact = store.get(fact_id)
        if not fact:
            raise HTTPException(404, f"no fact {fact_id}")
        result = run_check(fact, conn=store.conn)
        fallout = apply_result(store, result)
        return {
            "id": result.fact_id, "status": result.status, "detail": result.detail,
            "cascaded_to": fallout["cascaded"], "restored": fallout["restored"],
        }

    @app.post("/verify")
    def verify_all(scope: str | None = None, stale_after_days: float | None = None) -> dict:
        store, skills = stores()
        results = verify_scope(store, scope=scope, only_stale_after_days=stale_after_days)
        return {
            "checked": len(results),
            "results": [{"id": r.fact_id, "status": r.status, "detail": r.detail} for r in results],
        }

    @app.post("/audit")
    def audit(evidence: str = "", scope: str | None = None, older_than_days: float = 30.0) -> dict:
        store, skills = stores()
        try:
            report = run_audit(
                store, evidence=evidence, scope=scope, older_than_days=older_than_days
            )
        except LLMUnavailable as exc:
            raise HTTPException(503, str(exc))
        return {
            "covered": report.covered,
            "requested": report.requested,
            "not_audited": report.missing,
            "findings": [
                {"id": f.fact_id, "verdict": f.verdict, "reason": f.reason}
                for f in report.findings
            ],
        }

    @app.post("/distill")
    def distill(scope: str | None = None, token_budget: int = 1500, use_llm: bool = True) -> dict:
        store, skills = stores()
        try:
            report = run_distill(store, scope=scope, token_budget=token_budget, use_llm=use_llm)
        except LLMUnavailable as exc:
            raise HTTPException(503, str(exc))
        return {
            "tokens_before": report.tokens_before,
            "tokens_after": report.tokens_after,
            "saved_pct": round(report.saved_pct, 1),
            "deduped": report.deduped,
            "merged": report.merged,
        }

    @app.get("/export")
    def export(scope: str | None = None, min_confidence: float = 0.35) -> dict:
        store, skills = stores()
        return {"markdown": render(store, scope=scope, min_confidence=min_confidence)}

    @app.get("/skills")
    def list_skills(status: str | None = "active") -> dict:
        store, skills = stores()
        return {
            "skills": [
                {
                    "name": s.name, "description": s.description, "status": s.status,
                    "invocations": s.invocations, "success_rate": s.success_rate,
                    "quarantine_reason": s.quarantine_reason,
                }
                for s in skills.list_skills(status=status)
            ]
        }

    @app.get("/skills/search")
    def search_skills(q: str, limit: int = 5) -> dict:
        store, skills = stores()
        return {
            "results": [
                {"name": s.name, "description": s.description, "body": s.body}
                for s in skills.search(q, limit=limit)
            ]
        }

    @app.post("/skills")
    def upsert_skill(body: SkillIn) -> dict:
        store, skills = stores()
        saved = skills.upsert(Skill(**body.model_dump()))
        return {"name": saved.name, "id": saved.id, "status": saved.status}

    @app.post("/skills/{name}/outcome")
    def record_outcome(name: str, body: OutcomeIn) -> dict:
        store, skills = stores()
        updated = skills.record_outcome(
            name, body.outcome, session_id=body.session_id, note=body.note
        )
        if not updated:
            raise HTTPException(404, f"no skill named {name!r}")
        return {
            "name": updated.name, "status": updated.status,
            "invocations": updated.invocations, "success_rate": updated.success_rate,
            "quarantine_reason": updated.quarantine_reason,
        }

    return app
