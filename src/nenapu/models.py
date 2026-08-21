"""Core data types.

A memory in nenapu is never a bare string. Every fact carries the three things
that let a later session decide whether to still believe it: where it came from,
how fast that kind of claim goes stale, and (optionally) a command that proves
it.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from dataclasses import fields as dataclasses_fields
from enum import Enum
from typing import Any


class _StrEnum(str, Enum):
    """str-valued enum that prints as its value.

    Python 3.11 formats a plain `str, Enum` member as "Kind.USER"; these values
    end up in CLI output, JSON payloads, and SQL binds, so the value is the only
    representation that should ever escape.
    """

    def __str__(self) -> str:
        return self.value

    def __format__(self, spec: str) -> str:
        return format(self.value, spec)


class Kind(_StrEnum):
    """What sort of claim this is. Drives export grouping and audit prompts."""

    USER = "user"          # who the user is, how they work
    PROJECT = "project"    # goals, constraints, decisions
    ENVIRONMENT = "environment"  # ports, paths, tool versions, hosts
    FEEDBACK = "feedback"  # corrections and confirmed approaches
    REFERENCE = "reference"  # pointers to external resources


class Origin(_StrEnum):
    """Provenance. Ranked: what the user said outranks what an agent guessed."""

    USER_STATED = "user_stated"
    TOOL_OBSERVED = "tool_observed"
    FILE_DERIVED = "file_derived"
    AGENT_INFERRED = "agent_inferred"
    DISTILLED = "distilled"


ORIGIN_WEIGHT: dict[str, float] = {
    Origin.USER_STATED: 1.00,
    Origin.TOOL_OBSERVED: 0.95,
    Origin.FILE_DERIVED: 0.85,
    Origin.DISTILLED: 0.80,
    Origin.AGENT_INFERRED: 0.65,
}


class Decay(_StrEnum):
    """How fast a claim rots without re-verification."""

    IMMUTABLE = "immutable"  # never decays: "user prefers tabs"
    SLOW = "slow"            # 1y half-life: architecture decisions
    MEDIUM = "medium"        # 90d half-life: project goals, team process
    VOLATILE = "volatile"    # 14d half-life: ports, branch names, versions


HALF_LIFE_DAYS: dict[str, float] = {
    Decay.IMMUTABLE: 0.0,  # 0 == no decay
    Decay.SLOW: 365.0,
    Decay.MEDIUM: 90.0,
    Decay.VOLATILE: 14.0,
}

DEFAULT_DECAY: dict[str, str] = {
    Kind.USER: Decay.SLOW,
    Kind.PROJECT: Decay.MEDIUM,
    Kind.ENVIRONMENT: Decay.VOLATILE,
    Kind.FEEDBACK: Decay.SLOW,
    Kind.REFERENCE: Decay.MEDIUM,
}


class Status(_StrEnum):
    ACTIVE = "active"
    SUSPECT = "suspect"        # something it rests on was falsified
    SUPERSEDED = "superseded"   # a newer fact replaced it
    DISPUTED = "disputed"       # contradicts an active fact, not yet resolved
    ARCHIVED = "archived"       # folded into a distilled fact
    RETIRED = "retired"         # explicitly forgotten, kept for audit


class VerifyStatus(_StrEnum):
    NONE = "none"
    BLOCKED = "blocked"   # a check exists but no human has approved running it
    PASS = "pass"
    FAIL = "fail"
    ERROR = "error"  # the check itself broke; says nothing about the fact


def now() -> float:
    return time.time()


@dataclass
class Fact:
    text: str
    kind: str = Kind.PROJECT
    scope: str = "global"
    key: str | None = None  # e.g. "db.port" — the contradiction join key
    origin: str = Origin.AGENT_INFERRED
    origin_ref: str | None = None  # file:line, URL, session id, command
    session_id: str | None = None
    confidence: float = 0.7  # asserted at write time, before decay
    decay_class: str | None = None  # defaults from kind
    verify_cmd: str | None = None
    verify_expect: str | None = None  # substring the command output must contain
    tags: list[str] = field(default_factory=list)

    id: int | None = None
    status: str = Status.ACTIVE
    created_at: float = field(default_factory=now)
    updated_at: float = field(default_factory=now)
    last_verified_at: float | None = None
    verify_status: str = VerifyStatus.NONE
    verify_last_run: float | None = None
    verify_detail: str | None = None
    supersedes_id: int | None = None
    superseded_by_id: int | None = None
    distilled_into_id: int | None = None
    use_count: int = 0
    last_used_at: float | None = None
    good_recalls: int = 0
    bad_recalls: int = 0
    suspect_since: float | None = None
    suspect_reason: str | None = None
    agent: str | None = None
    occurrences: int = 1

    def __post_init__(self) -> None:
        if self.decay_class is None:
            self.decay_class = DEFAULT_DECAY.get(self.kind, Decay.MEDIUM)
        if self.last_verified_at is None:
            self.last_verified_at = self.created_at
        if self.agent is None:
            self.agent = os.environ.get("NENAPU_AGENT") or None


@dataclass
class Skill:
    name: str
    body: str
    description: str = ""
    scope: str = "global"
    tags: list[str] = field(default_factory=list)

    id: int | None = None
    status: str = "active"  # active | quarantined | retired
    quarantine_reason: str | None = None
    created_at: float = field(default_factory=now)
    updated_at: float = field(default_factory=now)
    invocations: int = 0
    successes: int = 0
    failures: int = 0
    last_used_at: float | None = None

    @property
    def success_rate(self) -> float | None:
        graded = self.successes + self.failures
        return None if graded == 0 else self.successes / graded


class EdgeKind(_StrEnum):
    """How one fact depends on another."""

    DERIVED_FROM = "derived_from"  # child would not have been concluded without parent
    SUPPORTS = "supports"          # corroboration, not dependence


class EdgeSource(_StrEnum):
    DECLARED = "declared"   # the agent said so
    INFERRED = "inferred"   # observed: parent was recalled, then child was written


class Outcome(_StrEnum):
    PENDING = "pending"
    GOOD = "good"
    BAD = "bad"
    NEUTRAL = "neutral"


@dataclass
class Edge:
    parent_id: int
    child_id: int
    kind: str = EdgeKind.DERIVED_FROM
    source: str = EdgeSource.DECLARED
    weight: float = 1.0
    id: int | None = None
    created_at: float = field(default_factory=now)


@dataclass
class Recall:
    """One fact surfaced into one task. Graded later, by whatever signal arrives."""

    fact_id: int
    session_id: str | None = None
    query: str = ""
    rank: int = 0
    score: float = 0.0
    outcome: str = Outcome.PENDING
    outcome_source: str | None = None
    outcome_at: float | None = None
    note: str | None = None
    id: int | None = None
    created_at: float = field(default_factory=now)
    # Filled by queries that join the fact back in. A recall names a fact by
    # id, and a caller that has to look every one of them up cannot render a
    # list of them without a query per row.
    fact_text: str = ""


@dataclass
class Conflict:
    fact_id: int
    other_id: int
    key: str | None
    detail: str
    resolution: str  # superseded | disputed | ignored
    id: int | None = None
    created_at: float = field(default_factory=now)


def _from_row(row: Any, cls: type) -> Any:
    """Build a dataclass from a sqlite Row, ignoring extra columns.

    Ranked queries join in things like `bm25(...) AS rank`; the dataclass should
    not have to know about every projection a caller invents.
    """
    fields = {f.name for f in dataclasses_fields(cls)}
    d = {k: v for k, v in dict(row).items() if k in fields or k == "tags_csv"}
    csv = d.pop("tags_csv", None)
    if "tags" in fields:
        d["tags"] = [t for t in (csv or "").split(",") if t]
    return cls(**d)


class EntityKind(_StrEnum):
    """What sort of thing an entity node represents."""

    REPO = "repo"
    DIR = "dir"
    FILE = "file"
    COMMIT = "commit"
    COMMAND = "command"
    SERVICE = "service"
    PERSON = "person"
    CONCEPT = "concept"


class EntityEdgeKind(_StrEnum):
    """How one entity relates to another."""

    CONTAINS = "contains"
    TOUCHED_WITH = "touched_with"
    CHANGED_IN = "changed_in"
    CALLS = "calls"
    RUNS = "runs"
    OWNS = "owns"
    ALIAS_OF = "alias_of"


class EntityStatus(_StrEnum):
    ALIVE = "alive"
    GONE = "gone"


@dataclass
class Entity:
    kind: str
    name: str
    scope: str = "global"
    id: int | None = None
    status: str = EntityStatus.ALIVE
    first_seen: float = field(default_factory=now)
    last_seen: float = field(default_factory=now)
    mentions: int = 1


@dataclass
class EntityEdge:
    src_id: int
    dst_id: int
    kind: str
    source: str = "observed"
    weight: float = 1.0
    observations: int = 1
    id: int | None = None
    valid_from: float = field(default_factory=now)
    valid_to: float | None = None


def row_to_fact(row: Any) -> Fact:
    return _from_row(row, Fact)


def row_to_entity(row: Any) -> Entity:
    return _from_row(row, Entity)


def row_to_recall(row: Any) -> Recall:
    return _from_row(row, Recall)


def row_to_skill(row: Any) -> Skill:
    return _from_row(row, Skill)
