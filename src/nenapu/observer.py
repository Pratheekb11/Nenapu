"""The agentic layer: watch, learn, and put memory where the agent will read it.

The tool-call model asks the agent to remember to remember. It will not. An
agent that just got corrected is precisely the agent least likely to stop and
file a note about it, and nothing in MCP fires unless the agent decides to fire
it.

So this layer never waits to be asked:

    session starts  ──▶  relevant memory injected into context
                             (agent reads it without calling anything)
    session ends    ──▶  transcript read, corrections and decisions extracted
                             (agent never had to file a note)

Two entry points, both driven by Claude Code hooks:

* `recall_context()` — what the agent should know before it starts. Emitted at
  SessionStart, so it lands in the model's context rather than in a tool
  result it has to request.
* `observe_transcript()` — what the session taught. Runs at Stop, reads the
  finished JSONL, and writes what was learned.

The extraction is deliberately biased toward corrections. "You said X, actually
it is Y" is the highest-value thing a session produces and the thing an agent
is least likely to record about itself.
"""

from __future__ import annotations

import json
import os
import re
import sqlite3
from dataclasses import replace
from pathlib import Path
from typing import Callable

from .db import commit, transaction
from .distill import _similarity
from .entities import proximity_scores
from .llm import Backend, LLMUnavailable, detect_backend, structured
from .models import (
    EntityEdgeKind,
    EntityKind,
    EntityStatus,
    Fact,
    Kind,
    Origin,
    Outcome,
    Status,
    now,
)
from .store import Store, effective_confidence, scope_for

# Keep the injected block small. It is prepended to every session, so it is
# paid for on every request whether or not it gets used.
# Tail sizing. Start small so a hook is instant on the common case, and grow
# only when a busy transcript has not yielded enough real conversation yet.
INITIAL_TAIL_BYTES = 400_000
MAX_TAIL_BYTES = 24_000_000
MAX_CONVERSATION_CHARS = 24_000

MAX_INJECTED = 12
# Per-section ceilings for the project block. A refactor session can touch two
# hundred files and a neglected project can hold fifty loops; either would
# spend the whole context budget on one section of a block that is paid for on
# every request. These bound a single runaway section; the token budget below
# is what bounds the block.
MAX_LEFT_OFF_FILES = 6
MAX_OPEN_LOOPS = 5
MAX_CHANGED = 8

# R3: the real unit. Every cap above is a count, and a count is the wrong unit
# for something prepended to every session and paid for on every request:
# twelve facts is 200 tokens or 2000 depending on what got written. Measured
# against the live store, where the block came to 841 tokens on 2026-08-22
# while the gate reported 84% of injected facts unused.
INJECTION_TOKEN_BUDGET = 700

# Sections in the order they are paid for, which is not the order they are
# printed in. A correction the user has repeated is the most actionable line
# in the block, and a list of files that changed while they were away is the
# most disposable, so a refactor that touched two hundred files must not be
# the reason a correction falls out.
INJECTION_PRIORITY = ("corrections", "falsified", "left_off", "loops", "known", "changed")

# The order a reader sees. Unchanged: the ledger sections, then what is
# believed, then what is not.
INJECTION_RENDER_ORDER = ("left_off", "loops", "changed", "corrections", "known",
                          "falsified")
# What the extractor is shown of the store before it reads the session. The
# extraction already runs at thousands of tokens against an 83-second call, so
# this is a fixed budget rather than something that grows with the store.
RELEVANT_MEMORY_LIMIT = 12
# Warnings are worth their tokens, but a store mid-cascade can have hundreds.
MAX_SUSPECT_INJECTED = 5
MIN_INJECTED_CONFIDENCE = 0.35

EXTRACT_SCHEMA = {
    "type": "object",
    "properties": {
        "facts": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "op": {"type": "string", "enum": ["add", "update", "noop"]},
                    "target_id": {"type": "integer"},
                    "text": {"type": "string"},
                    "kind": {"type": "string",
                             "enum": ["user", "project", "environment", "feedback"]},
                    "key": {"type": "string"},
                    "correction": {"type": "boolean"},
                },
                "required": ["text", "kind", "key", "correction"],
                "additionalProperties": False,
            },
        },
        "open_loops": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "text": {"type": "string"},
                    "resolution_hint": {"type": "string"},
                },
                "required": ["text", "resolution_hint"],
                "additionalProperties": False,
            },
        },
        # E8: the entities a filesystem cannot see. Files, dirs and commits
        # are built deterministically from the activity ledger and are not
        # something a model should be inventing; services, people and
        # concepts have no other source.
        "entities": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "kind": {"type": "string", "enum": ["service", "person", "concept"]},
                    "relation": {"type": "string", "enum": ["", "calls", "runs", "owns"]},
                    "target": {"type": "integer"},
                },
                "required": ["name", "kind"],
                "additionalProperties": False,
            },
        },
        # G4: grading rides this call rather than a second one. The same read
        # of the same transcript answers "what did this session teach" and
        # "what did this session do with what it was given".
        "grades": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "fact_id": {"type": "integer"},
                    "verdict": {"type": "string", "enum": ["used", "misled", "unused"]},
                    "where": {"type": "string"},
                },
                "required": ["fact_id", "verdict", "where"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["facts"],
    "additionalProperties": False,
}

EXTRACT_SYSTEM = """\
You read a coding session and record what should be remembered next time.

Record:
- corrections the user made ("no, use X not Y") — mark correction: true
- decisions and constraints the user stated
- environment facts discovered by running something (paths, versions, ports)
- how the user wants to be worked with

Ignore: what the assistant did, narration, anything true only of this session,
and anything you are inferring rather than reading.

Write each as a standalone sentence that will still make sense in six months
with none of this conversation around it. Give a dotted key for facts that are
one value for one subject (db.port, test.command), otherwise an empty string.
Prefer few, durable facts over many disposable ones.

You may be shown what is already known, each line starting with its id. For
every fact you record, say which it is:
- op "add" for something not already in that list
- op "update" with target_id set to one of those ids, when the session says
  more about a fact already listed, or says it again in different words
- op "noop" with target_id set, when the session only confirms what is
  already recorded word for word

Only use an id you were actually shown. Never invent one.

Also record open loops: things the session said should happen and did not.
Only what was left undone, never what was completed. Give a path glob in
resolution_hint when the session named where the work would go, otherwise an
empty string.

You may also be shown the facts that were injected into this session before it
started, each line starting with its id. Grade each one:
- verdict "used" only when the transcript shows the session relying on it, and
  quote or name that moment in `where`
- verdict "misled" when the session showed the fact to be wrong or out of date
- verdict "unused" for everything else, and leave `where` empty

Default to "unused". Most injected facts are never referred to, and a fact
being true, or looking familiar, is not evidence that this session used it.
Only grade ids you were shown.

Finally, record the things the session was about that a filesystem cannot
see: services, people, concepts. Files, directories and commits are already
known and must not be listed. When the session said one of them relates to
something in the list of known things you were shown, give its id as target
and say how in relation. Only use an id you were shown, and leave target out
otherwise."""


# ---------- redaction ----------
#
# A transcript is not a document the user wrote for us. It is whatever went
# past during the session: a pasted `.env`, a curl with a bearer token, a key
# echoed by a failing command. Harvest is the only place worth doing this,
# because it is upstream of both the things that outlive the session — the
# model call, and the store. Redacting later would mean the secret had already
# been sent somewhere.
#
# Shaped keys are matched by their own prefix. Everything else is matched by
# what it is *called*, because a secret's value is not distinguishable from a
# short string of letters. That direction of error is the safe one: at worst a
# nonsense assignment is blanked out and the session's meaning survives, which
# is what the "keeps" tests pin down.

_ASSIGNED = (
    r"(?i)\b([A-Za-z0-9_.-]*(?:api[_-]?key|secret|token|password|passwd|pwd|"
    r"credential|auth)[A-Za-z0-9_.-]*)\s*[:=]\s*"
    r"[\"']?([^\s\"',;)]{6,})[\"']?"
)

REDACTIONS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----.*?-----END [A-Z ]*PRIVATE KEY-----",
                re.S), "private-key"),
    (re.compile(r"\bsk-ant-[A-Za-z0-9_-]{8,}"), "anthropic-key"),
    (re.compile(r"\bsk-[A-Za-z0-9_-]{16,}"), "api-key"),
    (re.compile(r"\b(?:ghp|gho|ghu|ghs|ghr)_[A-Za-z0-9]{16,}"), "github-token"),
    (re.compile(r"\bgithub_pat_[A-Za-z0-9_]{20,}"), "github-token"),
    (re.compile(r"\bxox[baprs]-[A-Za-z0-9-]{8,}"), "slack-token"),
    (re.compile(r"\bAKIA[0-9A-Z]{16}\b"), "aws-key-id"),
    (re.compile(r"\bAIza[0-9A-Za-z_-]{35}\b"), "google-key"),
    (re.compile(r"\beyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}"), "jwt"),
    (re.compile(r"(?i)\b(?:bearer|basic)\s+[A-Za-z0-9._~+/-]{12,}=*"), "auth-header"),
    (re.compile(r"\b([a-zA-Z][a-zA-Z0-9+.-]*://[^\s:@/]+):[^\s:@/]+@"), "url-password"),
]

# Values that are plainly not secrets, however the thing holding them is
# named. `max_tokens=2048` and `timeout=180` are conversation, not credentials.
_NOT_A_SECRET = re.compile(r"^(?:\d+|true|false|none|null|nil)$", re.I)


def redact(text: str) -> str:
    """Blank out credentials in harvested conversation.

    Returns the text with each match replaced by `[redacted:<kind>]`, which is
    left in place deliberately: a fact extracted from a line that had a secret
    in it should read as though something was removed, not as though the line
    said something else.
    """
    for pattern, label in REDACTIONS:
        if label == "url-password":
            text = pattern.sub(rf"\1:[redacted:{label}]@", text)
        else:
            text = pattern.sub(f"[redacted:{label}]", text)

    def _assignment(match: re.Match[str]) -> str:
        name, value = match.group(1), match.group(2)
        if _NOT_A_SECRET.match(value) or value.startswith("[redacted:"):
            return match.group(0)
        return f"{name}=[redacted:secret]"

    return re.sub(_ASSIGNED, _assignment, text)


def _role_text_pairs(lines: list[str]) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    for line in lines:
        if not line.strip():
            continue
        try:
            event = json.loads(line)
        except ValueError:
            continue
        if not isinstance(event, dict):
            continue  # a stray `null` or bare string is not a turn
        message = event.get("message")
        if not isinstance(message, dict):
            message = {}
        role = message.get("role") or event.get("type")
        if role not in ("user", "assistant"):
            continue
        content = message.get("content")
        if isinstance(content, list):
            text = " ".join(
                block.get("text", "") for block in content
                if isinstance(block, dict) and block.get("type") == "text"
            )
        else:
            text = content if isinstance(content, str) else ""
        text = text.strip()
        if text:
            pairs.append((role, text))
    return pairs


def _turns_from(lines: list[str]) -> list[str]:
    return [f"{role}: {text}" for role, text in _role_text_pairs(lines)]


def _read_transcript(
    path: Path,
    max_chars: int = MAX_CONVERSATION_CHARS,
    parse: Callable[[list[str]], list[str]] | None = None,
) -> str:
    """Flatten a Claude Code JSONL transcript into readable turns.

    Two things make this harder than it looks on real data:

    * Transcripts reach tens of megabytes — the largest on the machine this was
      written on is 55MB — so reading the whole file to keep its tail wastes
      memory in a hook that runs after every session.
    * A long session is mostly tool traffic. A fixed 400KB tail of a busy
      transcript yielded 2,400 characters of actual conversation, which would
      silently miss a correction made earlier.

    So the tail grows until there is enough real conversation to be worth
    reading, or the file runs out.

    `parse` is the agent's own reader. Every agent writes a different file,
    and reading a Codex rollout with Claude Code's parser harvests nothing at
    all — the watcher would discover the session, queue it, and extract an
    empty conversation while reporting success.
    """
    parse = parse or _turns_from
    try:
        size = path.stat().st_size
    except OSError:
        return ""

    window = INITIAL_TAIL_BYTES
    turns: list[str] = []
    while True:
        try:
            if size > window:
                with path.open("rb") as handle:
                    handle.seek(size - window)
                    raw = handle.read().decode("utf-8", errors="ignore")
                lines = raw.split("\n")[1:]  # drop the partial first line
            else:
                lines = path.read_text(errors="ignore").splitlines()
        except OSError:
            return ""

        turns = parse(lines)
        harvest = sum(len(t) for t in turns)
        if harvest >= max_chars or window >= size or window >= MAX_TAIL_BYTES:
            break
        window *= 4

    joined = "\n\n".join(turns)
    trimmed = joined[-max_chars:] if len(joined) > max_chars else joined
    # Last thing before the text leaves this function, so every caller — the
    # model call, the dry run, the store — sees the redacted version and there
    # is no path that forgets to ask.
    return redact(trimmed)


# ---------- working memory: verbatim turns, `--no-infer` ----------
#
# Everything above extracts and discards the raw transcript. This is the one
# path that keeps it — a real privacy change, since it is the first place
# verbatim conversation lands on disk rather than distilled facts. Gated
# behind `NENAPU_STORE_MESSAGES`, default off, and redacted the same way the
# model-facing text is: the harvest-time invariant has to hold here too, or a
# secret reaches the store by a route that did not exist before.


def messages_from_transcript(path: Path, *, max_messages: int = 200) -> list[tuple[str, str]]:
    """Every user/assistant turn in a transcript, redacted, oldest first.

    Reuses the same tail-growing read `_read_transcript` uses for the model
    prompt, so this stays a bounded read even against the largest transcripts
    on disk rather than loading the whole file to keep the last few turns.
    """
    try:
        size = path.stat().st_size
    except OSError:
        return []

    window = INITIAL_TAIL_BYTES
    pairs: list[tuple[str, str]] = []
    while True:
        try:
            if size > window:
                with path.open("rb") as handle:
                    handle.seek(size - window)
                    raw = handle.read().decode("utf-8", errors="ignore")
                lines = raw.split("\n")[1:]  # drop the partial first line
            else:
                lines = path.read_text(errors="ignore").splitlines()
        except OSError:
            return []

        pairs = _role_text_pairs(lines)
        if len(pairs) >= max_messages or window >= size or window >= MAX_TAIL_BYTES:
            break
        window *= 4

    return [(role, redact(text)) for role, text in pairs[-max_messages:]]


def store_messages(
    conn: sqlite3.Connection, session_id: str | None, pairs: list[tuple[str, str]],
    *, apply: bool = True,
) -> int:
    """Persist verbatim turns, gated by `NENAPU_STORE_MESSAGES`.

    The gate lives here rather than at each call site so it cannot be
    forgotten by a caller — a store must opt in explicitly before any raw
    conversation lands on disk, `--no-infer` alone is not enough.

    `apply=False` answers the same question without writing, the way
    `observe_transcript` and `run_audit` already do: the count is what a real
    run would store, so a dry run reports the same number it would have.
    """
    if not os.environ.get("NENAPU_STORE_MESSAGES"):
        return 0
    if not apply:
        return len(pairs)
    with transaction(conn):
        for seq, (role, text) in enumerate(pairs):
            conn.execute(
                "INSERT INTO messages(session_id, seq, role, text, created_at)"
                " VALUES (?,?,?,?,?)",
                (session_id, seq, role, text, now()),
            )
        commit(conn)
    return len(pairs)


# Words too common to narrow anything down. Anything longer than this list
# would be tuning a search that FTS already ranks.
_QUERY_STOPWORDS = {
    "user", "assistant", "the", "and", "for", "that", "this", "with", "from",
    "have", "has", "was", "were", "you", "your", "not", "but", "are", "its",
    "it", "we", "our", "they", "them", "then", "than", "into", "just", "like",
    "what", "when", "where", "which", "will", "would", "should", "could",
}
# Enough terms to describe a session, few enough that FTS stays a cheap query.
_MAX_QUERY_TERMS = 40


def _salient_terms(conversation: str) -> str:
    """The words worth searching the store for, out of a whole session.

    Order of first appearance is kept rather than frequency-ranked: the thing
    a session was about is usually named early, and the tail of a long session
    is mostly the assistant agreeing with itself.
    """
    seen: list[str] = []
    for word in re.findall(r"[A-Za-z][A-Za-z0-9_.-]{3,}", conversation.lower()):
        if word in _QUERY_STOPWORDS or word in seen:
            continue
        seen.append(word)
        if len(seen) >= _MAX_QUERY_TERMS:
            break
    return " ".join(seen)


def relevant_memory(
    store: Store, conversation: str, *, scope: str | None = None,
    limit: int = RELEVANT_MEMORY_LIMIT,
) -> list[Fact]:
    """What the store already knows about what this session was about.

    An extractor shown nothing can only ever propose `add`, which is how the
    same fact came to be learned five separate times. One FTS query over the
    session's own words is the whole cost — no second model call.
    """
    query = _salient_terms(conversation)
    if not query:
        return []
    scopes = ["global", scope] if scope else None
    hits = store.search(
        query, scope=scopes, limit=limit, mark_used=False, log_recall=False,
    )
    return [fact for fact, _score, _explain in hits][:limit]


def _known_memory_block(facts: list[Fact]) -> str:
    """The retrieved facts, with the ids an `update` has to point at."""
    if not facts:
        return ""
    lines = ["## Already known (use these ids for update/noop)", ""]
    lines += [f"[{fact.id}] {fact.text}" for fact in facts]
    return "\n".join(lines) + "\n\n"


def _proposed_id(item: dict) -> int | None:
    raw = item.get("target_id")
    try:
        return int(raw)
    except (TypeError, ValueError):
        return None


# ---------- G4 · grading what the session was given ----------
#
# The set to grade is what was actually injected, read from the ledger. It is
# not `relevant_memory`, which is a fresh search over the transcript and finds
# a different set: grading that would grade facts the session never saw.
#
# Self-confirmation is the risk worth naming. The extractor writes facts and
# now grades them, so the block below carries the id and the claim and nothing
# else. No confidence, no origin, no usage count: a grader shown how strongly
# a fact is believed can defend one it recognises, and `used` is the verdict
# that moves confidence upward.

# How many of a session's pending recalls one extraction grades. `Ledger.pending`
# defaults to 50, which is a listing default rather than a grading one: a
# resumed session can hold more, and reading fewer than were injected would
# silently leave the rest ungraded. Bounded all the same, because every one of
# them is a line in the prompt. What is left over stays pending for a later
# replay, which is a no-op on everything already graded.
GRADED_RECALL_LIMIT = 60

GRADE_VERDICTS = {
    "used": Outcome.GOOD,
    "misled": Outcome.BAD,
    "unused": Outcome.NEUTRAL,
}


def _injected_block(recalls: list) -> str:
    """The facts this session was handed, with the ids a grade points at."""
    lines = [f"[{r.fact_id}] {r.fact_text}" for r in recalls if r.fact_text]
    if not lines:
        return ""
    header = ["## Injected into this session (grade each id)", ""]
    return "\n".join(header + lines) + "\n\n"


def _graded_fact_id(item: dict, allowed: set[int]) -> int | None:
    """An id the model was not shown is not an id, mirroring `_proposed_id`.

    Real ids are guessable, so a model that invents one must not be able to
    grade a fact this session never surfaced, or a fact in someone else's
    session.
    """
    try:
        fact_id = int(item.get("fact_id"))
    except (TypeError, ValueError):
        return None
    return fact_id if fact_id in allowed else None


# G5: a fact that was in context all session and never came up is evidence too,
# and it is what fills the denominator the gate divides by. Its own source
# keeps it measurable separately from a verdict the grader actually gave, and
# lets a later audit exclude it if it proves noisy.
UNUSED_GRADE_SOURCE = "observer-unused"


# ---------- E8 · the entities a filesystem cannot see ----------

# What the extractor is shown of the entity graph. Same reasoning as
# `RELEVANT_MEMORY_LIMIT`: a fixed budget on a call that already runs at
# thousands of tokens, rather than something that grows with the store.
KNOWN_ENTITY_LIMIT = 15

# Kinds a model may propose. Files, dirs, repos and commits are built from the
# activity ledger, which sees them exactly, and a model guessing at them would
# put invented paths beside observed ones.
EXTRACTABLE_ENTITY_KINDS = frozenset({EntityKind.SERVICE, EntityKind.PERSON,
                                      EntityKind.CONCEPT})

# Relations the session can state. The rest — contains, touched_with,
# changed_in, alias_of — are derived, not reported.
EXTRACTABLE_RELATIONS = frozenset({EntityEdgeKind.CALLS, EntityEdgeKind.RUNS,
                                   EntityEdgeKind.OWNS})


def _known_entities(store: Store, scope: str) -> list:
    """The entities already recorded in this scope, most mentioned first.

    Shown so a session that says "the auth service calls the billing service"
    can point the relation at a node that exists, instead of the graph growing
    a second copy of everything under a slightly different name.
    """
    rows = store.conn.execute(
        "SELECT id, kind, name FROM entities WHERE scope IN (?, 'global')"
        " AND status = ? ORDER BY mentions DESC, last_seen DESC LIMIT ?",
        (scope, EntityStatus.ALIVE, KNOWN_ENTITY_LIMIT),
    ).fetchall()
    return list(rows)


def _known_entities_block(rows: list) -> str:
    if not rows:
        return ""
    lines = ["## Known things (use these ids for target)", ""]
    lines += [f"[{r['id']}] {r['kind']}: {r['name']}" for r in rows]
    return "\n".join(lines) + "\n\n"


def _entities_from(store: Store, result: dict, *, scope: str, shown_ids: set[int]) -> int:
    """Write the entities the extraction proposed. Returns how many landed.

    A target the model was not shown is dropped rather than created, mirroring
    the guard on proposed fact ids: real ids are guessable, and a relation
    pointing at an invented one would be an edge to whatever happens to hold
    that number.
    """
    from .entities import EntityGraph

    graph = EntityGraph(store.conn)
    written = 0
    for item in result.get("entities") or []:
        if not isinstance(item, dict):
            continue
        name = (item.get("name") or "").strip()
        kind = (item.get("kind") or "").strip().lower()
        if not name or kind not in EXTRACTABLE_ENTITY_KINDS:
            continue
        entity = graph.upsert(kind=kind, name=name, scope=scope)
        written += 1

        relation = (item.get("relation") or "").strip().lower()
        target = _proposed_target(item)
        if relation in EXTRACTABLE_RELATIONS and target in shown_ids:
            graph.link(entity.id, target, kind=relation, source="observed")
    return written


def _proposed_target(item: dict) -> int | None:
    try:
        return int(item.get("target"))
    except (TypeError, ValueError):
        return None


def _mentioned_fact_ids(result: dict) -> set[int]:
    """Every fact id the grader tried to speak about, readable verdict or not.

    A grade whose verdict is unusable is not silence: the extractor named the
    fact and we could not read what it said, which is no more evidence of
    "never came up" than a failed model call is. Those stay pending.
    """
    mentioned: set[int] = set()
    for item in result.get("grades") or []:
        if not isinstance(item, dict):
            continue
        try:
            mentioned.add(int(item.get("fact_id")))
        except (TypeError, ValueError):
            continue
    return mentioned


def _neutralise_unmentioned(store: Store, recalls: list, mentioned: set[int]) -> int:
    """Grade the injections the extractor never named.

    Only ever called after an extraction succeeded. A failed or skipped model
    call has said nothing about what was injected, and neutralising on one
    would manufacture the evidence the gate reads. `Ledger.grade` is
    first-grade-wins, so a human verdict still stands.
    """
    return sum(
        1 for recall in recalls
        if recall.fact_id not in mentioned
        and store.ledger.grade(recall.id, Outcome.NEUTRAL, source=UNUSED_GRADE_SOURCE)
    )


def _grades_from(
    store: Store, result: dict, recalls: list, *, source: str = "observer",
) -> int:
    """Apply the extractor's verdicts to the recalls it was shown.

    Returns how many recalls were graded. `Ledger.grade` is reused rather than
    re-implemented, so first-grade-wins still holds and a human verdict is
    never overwritten by the model's.
    """
    by_fact: dict[int, list] = {}
    for recall in recalls:
        by_fact.setdefault(recall.fact_id, []).append(recall)

    graded = 0
    for item in result.get("grades") or []:
        if not isinstance(item, dict):
            continue
        fact_id = _graded_fact_id(item, set(by_fact))
        outcome = GRADE_VERDICTS.get(str(item.get("verdict") or "").strip().lower())
        if fact_id is None or outcome is None:
            continue  # a malformed grade is dropped, never fatal to the hook
        note = (item.get("where") or "").strip() or None
        for recall in by_fact[fact_id]:
            if store.ledger.grade(recall.id, outcome, source=source, note=note):
                graded += 1
    return graded


def observe_transcript(
    store: Store,
    transcript: Path,
    *,
    session_id: str | None = None,
    scope: str | None = None,
    cwd: str | None = None,
    backend: Backend | None = None,
    apply: bool = True,
    parse: Callable[[list[str]], list[str]] | None = None,
    grade_source: str = "observer",
) -> list[Fact]:
    """Extract what a finished session taught, and store it.

    The model is shown what is already known about the session's subject, so
    it can say "this updates fact 12" rather than adding a fifth phrasing of
    something recorded four times already. What it says is a *proposal*: the
    ids it may point at are only the ones it was shown, and the store's own
    rules still decide what happens to the row.
    """
    conversation = _read_transcript(Path(transcript), parse=parse)
    if len(conversation) < 200:
        return []  # nothing substantive happened

    known = relevant_memory(store, conversation, scope=scope)
    shown_ids = {fact.id for fact in known}
    # Read before the call, so the block and the grades that come back are the
    # same set. Without a session id there is no injected set to grade, and
    # grading by fact id alone would reach across every session in the store.
    injected = (store.ledger.pending(session_id=session_id, limit=GRADED_RECALL_LIMIT)
                if session_id else [])
    # Which scope this session's entities belong in, resolved once: the same
    # two-tier rule the facts below are written by.
    entity_scope = scope or scope_for(Kind.PROJECT, cwd)
    known_entities = _known_entities(store, entity_scope)

    backend = backend or detect_backend()
    result = structured(
        f"{_known_memory_block(known)}{_injected_block(injected)}"
        f"{_known_entities_block(known_entities)}"
        f"## Session transcript\n\n{conversation}\n\n"
        "Record what should be remembered. Return an empty list if nothing "
        "durable was established.",
        EXTRACT_SCHEMA,
        system=EXTRACT_SYSTEM,
        backend=backend,
        max_tokens=2048,
    )

    written: list[Fact] = []
    for item in result.get("facts", []):
        text = (item.get("text") or "").strip()
        if not text:
            continue
        correction = bool(item.get("correction"))
        # A model that has not been told about the new field, or one that
        # dropped it, is proposing what it has always proposed.
        op = (item.get("op") or "add").strip().lower()
        # A target the model was not shown is not a target. Real ids are
        # guessable, and the 1.5b model in the calibration table invented nine
        # of them for four facts — harmless while every op was an add, and the
        # exact hazard this schema introduces.
        target = _proposed_id(item)
        if target not in shown_ids:
            target = None
            op = "add" if op == "update" else op

        if op == "noop":
            continue  # already recorded, word for word

        fact = Fact(
            text=text,
            kind=item.get("kind") or (Kind.FEEDBACK if correction else Kind.PROJECT),
            key=(item.get("key") or "").strip() or None,
            # Observed from what the user actually said, not inferred by an
            # agent about itself — but still a model's reading of it, so it
            # does not get user_stated's weight.
            origin=Origin.TOOL_OBSERVED,
            origin_ref=f"session {session_id}" if session_id else "observed session",
            confidence=0.8 if correction else 0.65,
            session_id=session_id,
        )
        # Same two-tier rule the CLI writes by: a correction about how the
        # user works follows them everywhere, a fact about this repo does not.
        fact = replace(fact, scope=scope_for(fact.kind, cwd) if scope is None
                       else ("global" if fact.kind in (Kind.USER, Kind.FEEDBACK) else scope))
        if not apply:
            written.append(fact)
            continue

        if op == "update" and target is not None:
            revised = store.revise(target, text=text, kind=fact.kind, key=fact.key)
            if revised is not None:
                written.append(revised)
                continue
        stored, conflicts = store.write(fact, actor="observer")
        written.append(stored)

    if apply:
        _open_loops_from(store, result.get("open_loops") or [], scope=scope,
                         session_id=session_id)
        _grades_from(store, result, injected, source=grade_source)
        _neutralise_unmentioned(store, injected, _mentioned_fact_ids(result))
        _entities_from(store, result, scope=entity_scope,
                       shown_ids={row["id"] for row in known_entities})
    return written


def _open_loops_from(
    store: Store, proposals: list[dict], *, scope: str | None, session_id: str | None,
) -> None:
    """Record what the session said it would do and did not.

    Rides the same extraction call the facts came from, per the plan: a second
    83-second model call to ask the same transcript a second question is the
    version of this feature nobody would leave switched on.
    """
    from .loops import LoopBook

    book = LoopBook(store.conn)
    for item in proposals:
        text = (item.get("text") or "").strip()
        if not text:
            continue
        book.open_loop(
            scope=scope or "global",
            text=text,
            resolution_hint=(item.get("resolution_hint") or "").strip() or None,
            session_id=session_id,
        )


def _ago(seconds: float) -> str:
    if seconds < 3600:
        return "moments ago"
    if seconds < 86400:
        return f"{int(seconds // 3600)}h ago"
    return f"{int(seconds // 86400)}d ago"


def _left_off_section(ledger, scope: str) -> list[str]:
    """The last session in this repo: what it touched, and what it committed.

    One session, not a changelog. "Where you left off" is a question about the
    most recent thing, and the sessions before it are what `nenapu activity`
    is for.
    """
    sessions = ledger.sessions_for_scope(scope, limit=1)
    if not sessions:
        return []
    session = sessions[0]

    paths: list[str] = []
    for event in ledger.file_events_for_session(session["id"]):
        if event["path"] not in paths:
            paths.append(event["path"])
    commits = ledger.commits_for_session(session["id"])
    if not paths and not commits:
        return []

    when = _ago(now() - (session["ended_at"] or session["started_at"]))
    lines = [f"Where you left off ({when}, {session['agent']}):"]
    if paths:
        lines.append("- touched " + ", ".join(paths[:MAX_LEFT_OFF_FILES]))
    if commits and commits[-1]["subject"]:
        lines.append(f"- last commit: \"{commits[-1]['subject']}\"")
    lines.append("")
    return lines


def _open_loops_section(conn, scope: str) -> list[str]:
    from .loops import LoopBook

    loops = LoopBook(conn).open_for_scope(scope)
    if not loops:
        return []
    lines = ["Open here — mentioned but not done:"]
    lines += [f"- {loop['text']}" for loop in loops[:MAX_OPEN_LOOPS]]
    lines.append("")
    return lines


def _changed_section(ledger, scope: str, cwd: str | None) -> list[str]:
    """What moved in the repo while this project was somebody else's problem.

    One git call, from the commit the last session ended on to HEAD now. A
    recorded head can become unreachable through a rebase or a pruned branch,
    in which case there is no evidence and the section is simply absent —
    a SessionStart hook that raises is a session that starts knowing nothing.
    """
    if not cwd:
        return []
    from .capture import changed_paths, git_head

    last = next((s for s in ledger.sessions_for_scope(scope, limit=10)
                 if s["git_head_after"]), None)
    if last is None:
        return []
    changed = changed_paths(cwd, last["git_head_after"], git_head(cwd))
    if not changed:
        return []

    lines = ["Changed since you were last here:"]
    lines += [f"- {op} {path}" for op, path in changed[:MAX_CHANGED]]
    if len(changed) > MAX_CHANGED:
        lines.append(f"- ... and {len(changed) - MAX_CHANGED} more")
    lines.append("")
    return lines


# ---------- R4 · anchored on the work at hand ----------
#
# The block was sorted by `(kind != FEEDBACK, -occurrences, -confidence)` and
# nothing else, so it was identical for every session in a repo until the
# store itself changed. That is the mechanical reason it read as a dump, and
# the gate measured 84% of it unused.
#
# The anchor costs no model call and no new state: the activity ledger already
# records which files the last sessions touched and which branch they were on.
# A fact that names one of them is about the work at hand.

# How many recent file events the anchor is read from. Enough to cover a
# session or two of work, few enough that last month's refactor does not
# still be steering today's block.
ANCHOR_FILE_EVENTS = 20

# Basenames only, never directory segments: `app`, `src` and `backend` appear
# in half the paths in a repo and would match half the facts in the store,
# which is a flatter signal than no signal at all.
MIN_ANCHOR_TERM = 3


def _anchor_terms(conn, scope: str | None, cwd: str | None) -> set[str]:
    """What this session is working on, in the words a fact might use."""
    if not scope:
        return set()
    from .activity import ActivityLedger

    ledger = ActivityLedger(conn)
    terms: set[str] = set()
    for event in ledger.file_events_for_scope(scope, limit=ANCHOR_FILE_EVENTS):
        name = event["path"].rsplit("/", 1)[-1]
        stem = name.rsplit(".", 1)[0]
        terms.update(t for t in (name, stem) if len(t) >= MIN_ANCHOR_TERM)
    for session in ledger.sessions_for_scope(scope, limit=1):
        branch = session["git_branch"]
        if branch and len(branch) >= MIN_ANCHOR_TERM:
            terms.add(branch)
    if cwd:
        here = Path(cwd).name
        if len(here) >= MIN_ANCHOR_TERM:
            terms.add(here)
    return {t.lower() for t in terms}


def _anchor_paths(conn, scope: str | None) -> list[str]:
    """The paths recent sessions edited, as the entity graph spells them.

    R4 reads the words out of these; E7 walks the graph from them, so the
    full path is what is wanted here rather than the basename.
    """
    if not scope:
        return []
    from .activity import ActivityLedger

    seen: list[str] = []
    for event in ActivityLedger(conn).file_events_for_scope(scope,
                                                            limit=ANCHOR_FILE_EVENTS):
        if event["path"] not in seen:
            seen.append(event["path"])
    return seen


def _anchor_score(fact: Fact, terms: set[str]) -> int:
    """How much of the work at hand this fact names."""
    if not terms:
        return 0
    haystack = f"{fact.text} {fact.key or ''}".lower()
    return sum(1 for term in terms if term in haystack)


# ---------- R2 · diversity at selection ----------
#
# Selection was top-N by score with no diversity check, and the store holds 46
# superseded rows and many near-identical actives, so twelve slots could carry
# three distinct claims. `distill.dedupe` runs at write time and per scope,
# which is a different moment doing a different job: what reaches here is the
# residue that survived it.
#
# The false-positive cost is what shapes the rule. Merging two facts that only
# look alike silently hides one, and the block is the one surface where a
# hidden fact is never noticed. So a restatement has to be evidence, not a
# guess: the store saying outright that two rows are one subject (a shared
# key), or wording so close that the difference is filler. Anything that reads
# as a disagreement — two different numbers, most of all — is two claims and
# stays two claims.

REDUNDANCY_THRESHOLD = 0.85


def _restates(fact: Fact, other: Fact) -> bool:
    """Is `fact` the same claim as `other`, said again?"""
    from .store import looks_contradictory

    if fact.key and fact.key == other.key:
        return True  # one key is one subject with one value, by construction
    # A disagreement is not a restatement. Asked here of facts that share no
    # key, where a numeric mismatch is the clearest evidence there is that two
    # sentences are about two different values.
    if looks_contradictory(fact.text, other.text)[0]:
        return False
    return _similarity(fact.text, other.text) >= REDUNDANCY_THRESHOLD


def _distinct(facts: list[Fact]) -> list[Fact]:
    """Drop restatements of claims already chosen, keeping order.

    Order is the priority order the caller sorted into, so the survivor of a
    pair is the one that earned its slot — a correction outranks the project
    fact that repeats it.
    """
    kept: list[Fact] = []
    for fact in facts:
        if any(_restates(fact, chosen) for chosen in kept):
            continue
        kept.append(fact)
    return kept


class _Section:
    """One block of the injection: a heading, its lines, and the facts behind
    them, so that what is logged as recalled is what actually got printed."""

    def __init__(self, key: str, lines: list[str], facts: list[Fact] | None = None) -> None:
        self.key = key
        self.lines = lines
        self.facts = facts or []

    def __bool__(self) -> bool:
        return bool(self.lines)


_TRUNCATION_MARK = " …"


def _token_estimate(text: str) -> int:
    """Chars over four, the same estimate `nenapu cost` reports at.

    An approximation on purpose: an exact count needs the provider's
    tokenizer, and a budget that cannot be computed without a network call is
    not a budget a SessionStart hook can keep.
    """
    return len(text) // 4


def _truncate_to(line: str, tokens: int) -> str:
    """Cut a line to fit, marked so a reader can see something was removed."""
    keep = max(tokens * 4 - len(_TRUNCATION_MARK), 0)
    return line[:keep].rstrip() + _TRUNCATION_MARK


def _fit(sections: dict[str, _Section], spent: int, budget: int) -> dict[str, _Section]:
    """Spend the budget across the sections, most valuable first.

    A section whose heading will not fit is dropped whole: a heading with
    nothing under it costs tokens and says nothing. The first line of the
    first section is truncated rather than dropped, because a block that
    disappears because one row was enormous is a session that starts knowing
    nothing, which is the failure every guard in this path exists to avoid.
    """
    kept: dict[str, _Section] = {}
    for key in INJECTION_PRIORITY:
        section = sections.get(key)
        if not section:
            continue
        heading, body = section.lines[0], section.lines[1:]
        heading_cost = _token_estimate(heading) + 1
        if spent + heading_cost > budget:
            continue
        spent += heading_cost
        lines: list[str] = [heading]
        facts: list[Fact] = []
        for i, line in enumerate(body):
            cost = _token_estimate(line) + 1
            if spent + cost > budget:
                if len(lines) > 1 or kept:
                    break  # the block already says something; stop here
                line = _truncate_to(line, budget - spent)
                cost = _token_estimate(line) + 1
                if cost <= 0:
                    break
            spent += cost
            lines.append(line)
            if i < len(section.facts):
                facts.append(section.facts[i])
        if len(lines) > 1:
            kept[key] = _Section(key, lines, facts)
    return kept


def _section_from_ledger(key: str, lines: list[str]) -> _Section:
    """The ledger sections already render themselves; drop their trailing
    blank, which the assembler puts back."""
    body = [line for line in lines if line]
    return _Section(key, body)


def recall_context(
    store: Store, *, scope: str | None = None, cwd: str | None = None,
    limit: int | None = None, session_id: str | None = None,
) -> str:
    """What the agent should know before it starts, as plain text.

    Emitted into the session's context rather than returned from a tool: the
    agent reads it whether or not it would have thought to ask.

    With a `scope` this is about one repository: the facts are that project's
    plus the global ones, and three sections come from the activity ledger
    rather than from the belief network — where the last session left off,
    what is open here, and what changed while you were elsewhere. Unscoped it
    renders exactly as before, which the MCP surface and `nenapu rules` rely
    on. `cwd` is where the one git call for "changed since" runs.

    Without `session_id` this was a plain SELECT — no recall logged, no
    `use_count` bumped — which is why the outcome ledger and the
    falsification cascade had nothing to work with under real hook-only use.
    Passing it logs the injected set as recalls with an empty query, since a
    session-start injection has no query for BM25 to have been involved in.
    """
    from .activity import ActivityLedger

    # Two-tier by construction: how the user works travels with them, what a
    # repo does stays in the repo. Unscoped means everything, as before.
    scopes = ["global", scope] if scope else None
    sections: dict[str, _Section] = {}
    if scope:
        ledger = ActivityLedger(store.conn)
        sections["left_off"] = _section_from_ledger(
            "left_off", _left_off_section(ledger, scope))
        sections["loops"] = _section_from_ledger(
            "loops", _open_loops_section(store.conn, scope))
        sections["changed"] = _section_from_ledger(
            "changed", _changed_section(ledger, scope, cwd))
    project_lines = [line for key in ("left_off", "loops", "changed")
                     if sections.get(key) for line in sections[key].lines]
    # Suspect facts are included on purpose. They are exactly the ones an agent
    # would otherwise use without knowing their foundation collapsed, and
    # listing them under a warning is the only way the cascade reaches the
    # place decisions get made. Filtering to active would have made the
    # belief network invisible where it matters most.
    facts = [
        (fact, effective_confidence(fact))
        for fact in store.list_facts(scope=scopes, status=(Status.ACTIVE, Status.SUSPECT),
                                     limit=500)
    ]
    # The confidence floor is a filter on what to *believe*, so it must not be
    # applied to the suspect list: a suspect fact is penalised precisely
    # because its foundation fell, which would push it under the floor and
    # silence the warning exactly when it is most needed.
    suspect = [(f, c) for f, c in facts if f.status == Status.SUSPECT]
    sound = [(f, c) for f, c in facts
             if f.status != Status.SUSPECT and c >= MIN_INJECTED_CONFIDENCE]
    if not sound and not suspect and not project_lines:
        return ""

    # Corrections first, and among them the ones said most often — a
    # correction repeated five times is worth the budget more than one said
    # once at the same confidence, and confidence alone cannot tell them
    # apart when both were just asserted.
    # Anchoring reorders what follows the corrections; it does not demote
    # them. A correction the user has repeated is still the most actionable
    # line in the block whatever files were edited yesterday.
    anchor = _anchor_terms(store.conn, scope, cwd)
    # E7 extends the same anchor one layer out: a fact about the module this
    # one is always edited with is about the work at hand too. Traversal is
    # scoped, so it cannot reach into another project. With no entity data
    # every score here is zero and the ordering is R4's.
    proximity = proximity_scores(store.conn, _anchor_paths(store.conn, scope),
                                 scopes) if scope else {}

    def _relevance(fact: Fact) -> float:
        return _anchor_score(fact, anchor) + proximity.get(fact.id, 0.0)

    sound.sort(key=lambda pair: (pair[0].kind != Kind.FEEDBACK,
                                 -_relevance(pair[0]),
                                 -pair[0].occurrences, -pair[1]))
    suspect.sort(key=lambda pair: (-_relevance(pair[0]), -pair[1]))
    # `limit` is still honoured when a caller sets one, but nothing sets one by
    # default any more: the token budget is what decides how much of the store
    # a session is handed.
    if limit is not None:
        sound = sound[:limit]

    # Diversity applies to what is believed, never to the warnings: a suspect
    # fact is listed precisely because an agent would otherwise act on it
    # without knowing its foundation collapsed, which is the same reason the
    # confidence floor does not apply to it either.
    chosen = _distinct([fact for fact, _ in sound])
    corrections = [f for f in chosen if f.kind == Kind.FEEDBACK]
    others = [f for f in chosen if f.kind != Kind.FEEDBACK]
    doubted = [f for f, _ in suspect[:MAX_SUSPECT_INJECTED]]

    if corrections:
        sections["corrections"] = _Section(
            "corrections",
            ["Previously corrected — do not repeat these:"]
            + [_correction_line(f) for f in corrections],
            corrections,
        )
    if others:
        sections["known"] = _Section(
            "known",
            ["Known about this work:"] + [f"- {f.text}" for f in others],
            others,
        )
    if doubted:
        sections["falsified"] = _Section(
            "falsified",
            ["Do not rely on these — what they rested on was falsified:"]
            + [f"- {f.text}" for f in doubted],
            doubted,
        )

    header = f"# Memory (nenapu) — {scope}" if scope else "# Memory (nenapu)"
    kept = _fit(sections, _token_estimate(header) + 2, INJECTION_TOKEN_BUDGET)

    lines = [header, ""]
    for key in INJECTION_RENDER_ORDER:
        section = kept.get(key)
        if not section:
            continue
        lines += section.lines
        lines.append("")

    # Log what was actually printed, not what was considered: a recall row for
    # a fact the session never saw is evidence of nothing, and the gate reads
    # these rows.
    injected = [fact for key in INJECTION_RENDER_ORDER
                for fact in kept.get(key, _Section(key, [])).facts]
    if session_id and injected:
        store.ledger.log_many([(fact, 0.0, {}) for fact in injected],
                              session_id=session_id, query="")
        store.mark_used([fact.id for fact in injected])
    return "\n".join(lines).strip()


def _correction_line(fact: Fact) -> str:
    """A correction said more than once is called out by count — the most
    actionable signal in the store, and one duplicate near-identical facts
    used to destroy silently."""
    if fact.occurrences > 1:
        return f"- {fact.text} (said {fact.occurrences} times)"
    return f"- {fact.text}"


def hook_payload(raw: str) -> dict:
    """Claude Code hands hooks a JSON object on stdin."""
    try:
        return json.loads(raw) if raw.strip() else {}
    except ValueError:
        return {}


__all__ = ["observe_transcript", "recall_context", "hook_payload", "LLMUnavailable"]
