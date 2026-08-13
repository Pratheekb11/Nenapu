"""LLM audit pass — the only part of verification that needs judgement.

Decay and executable checks cover facts that are dated or mechanically
checkable. Soft claims ("the team decided to drop the queue") are neither, and
they are exactly the ones that quietly go wrong. This pass re-reads them
against current evidence and marks what no longer holds.

It is a scheduled job, not a write-path hook: it costs tokens, so it runs on
the small subset of facts that are old enough or contested enough to be worth
a second look.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .llm import Backend, LLMUnavailable, detect_backend, structured
from .models import Status, now
from .store import DAY, Store, effective_confidence

# Local models pay for every generated token twice over: schema-constrained
# decoding is slow, and a 3B model wanders on long outputs. The compact schema
# drops the rewrite field and asks for a terse reason, which roughly halves
# generation without losing the verdict — the only part that drives behaviour.
COMPACT_SCHEMA = {
    "type": "object",
    "properties": {
        "findings": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "integer"},
                    "verdict": {"type": "string", "enum": ["holds", "stale", "wrong", "unclear"]},
                    "reason": {"type": "string", "maxLength": 120},
                },
                "required": ["id", "verdict", "reason"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["findings"],
    "additionalProperties": False,
}

AUDIT_SCHEMA = {
    "type": "object",
    "properties": {
        "findings": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "integer"},
                    "verdict": {"type": "string", "enum": ["holds", "stale", "wrong", "unclear"]},
                    "reason": {"type": "string"},
                    "suggested_text": {"type": "string"},
                },
                "required": ["id", "verdict", "reason", "suggested_text"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["findings"],
    "additionalProperties": False,
}

COMPACT_SYSTEM = """Check each stored fact against the evidence.

holds = evidence confirms it. stale = evidence shows it changed. wrong =
evidence contradicts it. unclear = evidence does not mention it.

Prefer unclear over guessing. Old is not the same as stale. One line per fact."""

SYSTEM = """You audit an agent's stored memory for staleness and contradiction.

For each numbered fact, decide whether it still holds given the evidence \
provided. Be conservative: 'unclear' is the correct verdict when the evidence \
does not speak to the claim. Do not mark something stale merely because it is \
old — only because the evidence contradicts it or shows it has moved on.

suggested_text: the corrected wording when the verdict is 'stale' or 'wrong', \
otherwise an empty string."""


@dataclass
class Finding:
    fact_id: int
    verdict: str
    reason: str
    suggested_text: str


@dataclass
class AuditReport:
    """Findings plus what the pass failed to cover.

    A model that returns three verdicts for eight facts has not audited eight
    facts — and the five it skipped keep their confidence with nobody the
    wiser. Silent under-coverage is the failure mode that makes an audit worse
    than useless, so it is reported, not swallowed.
    """

    findings: list[Finding]
    requested: int = 0
    covered: int = 0
    missing: list[int] = field(default_factory=list)
    invented: list[int] = field(default_factory=list)
    batches: int = 0
    backend: str = ""
    applied: bool = False

    @property
    def coverage(self) -> float:
        return 1.0 if not self.requested else self.covered / self.requested

    def __iter__(self):  # so callers can still treat it as the findings list
        return iter(self.findings)

    def __len__(self) -> int:
        return len(self.findings)


def select_candidates(
    store: Store,
    *,
    scope: str | None = None,
    older_than_days: float = 30.0,
    max_facts: int = 40,
) -> list:
    """Pick the facts worth spending tokens on: aged, disputed, or failing."""
    cutoff = now() - older_than_days * DAY
    pool = store.list_facts(scope=scope, status=[Status.ACTIVE, Status.DISPUTED], limit=5000)

    def worth_auditing(f) -> bool:
        if f.status == Status.DISPUTED:
            return True
        if f.verify_status == "fail":
            return True
        anchor = f.last_verified_at or f.created_at
        return anchor < cutoff

    candidates = [f for f in pool if worth_auditing(f)]
    # Cheapest wins first: the least-believed facts are the ones most likely wrong.
    candidates.sort(key=effective_confidence)
    return candidates[:max_facts]


def build_prompt(facts: list, evidence: str, *, compact: bool = False) -> str:
    """Input tokens are cheap next to generated ones, but a small model still
    reads a shorter prompt more reliably."""
    if compact:
        lines = ["EVIDENCE:", evidence.strip() or "(none)", "", "FACTS:"]
        lines += [f"{f.id}. {f.text}" for f in facts]
        lines.append("")
        lines.append(f"Return one verdict for each of the {len(facts)} ids above.")
        return "\n".join(lines)

    lines = ["## Evidence (current state of the world)", evidence.strip() or "(none supplied)", ""]
    lines.append("## Stored facts")
    for f in facts:
        age = (now() - (f.last_verified_at or f.created_at)) / DAY
        lines.append(
            f"- id={f.id} [{f.kind}/{f.scope}] "
            f"(origin={f.origin}, {age:.0f}d since verified, belief={effective_confidence(f):.2f})"
            f"\n  {f.text}"
        )
    lines.append("")
    lines.append("Return a verdict for every id listed above.")
    return "\n".join(lines)


def audit(
    store: Store,
    *,
    evidence: str = "",
    scope: str | None = None,
    older_than_days: float = 30.0,
    max_facts: int | None = None,
    apply: bool | None = None,
    backend: Backend | None = None,
    batch_size: int | None = None,
    trust: bool | None = None,
) -> AuditReport:
    """Run one audit pass.

    `apply` defaults to the backend's trust level: cloud backends act on their
    findings, local ones report and stop. That is not caution for its own sake
    — the best local model measured still called half of a set of *confirmed*
    facts stale, and a pass like that would leave a real store full of spurious
    doubt. Pass `apply=True` explicitly to act on a local model's findings.

    When findings are applied they are applied conservatively: a `wrong`
    verdict retires the fact on a trusted backend and merely disputes it on a
    local one, `stale` disputes, and `holds` changes nothing at all. The model
    is one more fallible source, so it can raise doubt but never manufacture
    confidence. `trust` overrides the backend-derived default.
    """
    backend = backend or detect_backend()
    local = backend.name != "anthropic"
    row = store.conn.execute(
        "SELECT value FROM meta WHERE key = ?", (f"calibration:{backend.describe()}",)
    ).fetchone()
    calibrated = bool(row and row["value"] == "pass")
    if row and row["value"] == "fail":
        # Calibration already showed this model's verdicts do not depend on the
        # evidence. Applying them would mark sound memory disputed wholesale.
        raise LLMUnavailable(
            f"{backend.describe()} failed calibration — its verdicts do not track the "
            "evidence. Re-run `nenapu doctor --calibrate`, or use a larger model."
        )

    # Trust is earned by passing calibration, not by carrying a particular
    # backend name. A CLI agent that reads evidence correctly deserves more
    # standing than a local model that does not, whatever it is called.
    trusted = (backend.trusted or calibrated) if trust is None else trust
    apply = trusted if apply is None else apply

    if max_facts is None:
        max_facts = 40 if not local else 24
    if batch_size is None:
        batch_size = max_facts if not local else 4

    candidates = select_candidates(
        store, scope=scope, older_than_days=older_than_days, max_facts=max_facts
    )
    if not candidates:
        return AuditReport([], backend=backend.describe())

    findings: list[Finding] = []
    seen: set[int] = set()
    batches = 0

    for start in range(0, len(candidates), batch_size):
        batch = candidates[start : start + batch_size]
        batches += 1
        result = structured(
            build_prompt(batch, evidence, compact=local),
            COMPACT_SCHEMA if local else AUDIT_SCHEMA,
            system=COMPACT_SYSTEM if local else SYSTEM,
            backend=backend,
            max_tokens=(60 * len(batch) + 128) if local else max(2048, 220 * len(batch)),
        )
        for item in result.get("findings", []):
            fact_id = item.get("id")
            if fact_id is None or fact_id in seen:
                continue  # models repeat themselves across batches
            seen.add(fact_id)
            findings.append(
                Finding(
                    fact_id=fact_id,
                    verdict=item.get("verdict", "unclear"),
                    reason=item.get("reason", ""),
                    suggested_text=item.get("suggested_text", ""),
                )
            )

    known = {f.id for f in candidates}
    report = AuditReport(
        findings=findings,
        requested=len(candidates),
        covered=len(seen & known),
        missing=sorted(known - seen),
        invented=sorted(seen - known),
        batches=batches,
        backend=backend.describe(),
        applied=bool(apply),
    )

    if apply:
        for finding in findings:
            if finding.fact_id not in known:
                continue  # model invented an id; ignore it
            if finding.verdict == "wrong":
                # A model asserting something is wrong is an opinion, not a
                # measurement. Only a backend we trust gets to retire a fact
                # outright; otherwise it is flagged for a human to settle.
                store.set_status(
                    finding.fact_id,
                    Status.RETIRED if trusted else Status.DISPUTED,
                    actor="audit",
                )
            elif finding.verdict == "stale":
                store.set_status(finding.fact_id, Status.DISPUTED, actor="audit")
            elif finding.verdict == "holds" and trusted:
                # A calibrated backend confirming a fact against real evidence
                # is a genuine, if weaker, check. Ignoring it entirely made the
                # audit a one-way ratchet — able to destroy confidence, never
                # to restore it — so a fact that is still true decayed to the
                # floor while the audit kept agreeing it was fine.
                store.soft_verify(finding.fact_id)
            # An uncalibrated backend's `holds` still does nothing: that is the
            # same guess that wrote the fact, made again, and letting it reset
            # the clock would keep stale memory alive forever.
            store.conn.execute(
                "INSERT INTO journal(action, fact_id, actor, detail, created_at)"
                " VALUES ('audit', ?, 'audit', ?, ?)",
                (finding.fact_id, f"{finding.verdict}: {finding.reason}", now()),
            )
        store.conn.commit()

    return report


__all__ = ["AuditReport", "Finding", "audit", "select_candidates", "build_prompt",
           "LLMUnavailable"]
