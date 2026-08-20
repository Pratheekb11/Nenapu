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
from pathlib import Path

from .db import commit, transaction
from .llm import Backend, LLMUnavailable, detect_backend, structured
from .models import Fact, Kind, Origin, Status, now
from .store import Store, effective_confidence

# Keep the injected block small. It is prepended to every session, so it is
# paid for on every request whether or not it gets used.
# Tail sizing. Start small so a hook is instant on the common case, and grow
# only when a busy transcript has not yielded enough real conversation yet.
INITIAL_TAIL_BYTES = 400_000
MAX_TAIL_BYTES = 24_000_000
MAX_CONVERSATION_CHARS = 24_000

MAX_INJECTED = 12
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
                    "text": {"type": "string"},
                    "kind": {"type": "string",
                             "enum": ["user", "project", "environment", "feedback"]},
                    "key": {"type": "string"},
                    "correction": {"type": "boolean"},
                },
                "required": ["text", "kind", "key", "correction"],
                "additionalProperties": False,
            },
        }
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
Prefer few, durable facts over many disposable ones."""


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


def _read_transcript(path: Path, max_chars: int = MAX_CONVERSATION_CHARS) -> str:
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
    """
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

        turns = _turns_from(lines)
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
) -> int:
    """Persist verbatim turns, gated by `NENAPU_STORE_MESSAGES`.

    The gate lives here rather than at each call site so it cannot be
    forgotten by a caller — a store must opt in explicitly before any raw
    conversation lands on disk, `--no-infer` alone is not enough.
    """
    if not os.environ.get("NENAPU_STORE_MESSAGES"):
        return 0
    with transaction(conn):
        for seq, (role, text) in enumerate(pairs):
            conn.execute(
                "INSERT INTO messages(session_id, seq, role, text, created_at)"
                " VALUES (?,?,?,?,?)",
                (session_id, seq, role, text, now()),
            )
        commit(conn)
    return len(pairs)


def observe_transcript(
    store: Store,
    transcript: Path,
    *,
    session_id: str | None = None,
    backend: Backend | None = None,
    apply: bool = True,
) -> list[Fact]:
    """Extract what a finished session taught, and store it."""
    conversation = _read_transcript(Path(transcript))
    if len(conversation) < 200:
        return []  # nothing substantive happened

    backend = backend or detect_backend()
    result = structured(
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
        if not apply:
            written.append(fact)
            continue
        stored, conflicts = store.write(fact, actor="observer")
        written.append(stored)
    return written


def recall_context(
    store: Store, *, scope: str | None = None, limit: int = MAX_INJECTED,
    session_id: str | None = None,
) -> str:
    """What the agent should know before it starts, as plain text.

    Emitted into the session's context rather than returned from a tool: the
    agent reads it whether or not it would have thought to ask.

    Without `session_id` this was a plain SELECT — no recall logged, no
    `use_count` bumped — which is why the outcome ledger and the
    falsification cascade had nothing to work with under real hook-only use.
    Passing it logs the injected set as recalls with an empty query, since a
    session-start injection has no query for BM25 to have been involved in.
    """
    # Suspect facts are included on purpose. They are exactly the ones an agent
    # would otherwise use without knowing their foundation collapsed, and
    # listing them under a warning is the only way the cascade reaches the
    # place decisions get made. Filtering to active would have made the
    # belief network invisible where it matters most.
    facts = [
        (fact, effective_confidence(fact))
        for fact in store.list_facts(scope=scope, status=(Status.ACTIVE, Status.SUSPECT),
                                     limit=500)
    ]
    # The confidence floor is a filter on what to *believe*, so it must not be
    # applied to the suspect list: a suspect fact is penalised precisely
    # because its foundation fell, which would push it under the floor and
    # silence the warning exactly when it is most needed.
    suspect = [(f, c) for f, c in facts if f.status == Status.SUSPECT]
    sound = [(f, c) for f, c in facts
             if f.status != Status.SUSPECT and c >= MIN_INJECTED_CONFIDENCE]
    if not sound and not suspect:
        return ""

    # Corrections first, and among them the ones said most often — a
    # correction repeated five times is worth the budget more than one said
    # once at the same confidence, and confidence alone cannot tell them
    # apart when both were just asserted.
    sound.sort(key=lambda pair: (pair[0].kind != Kind.FEEDBACK, -pair[0].occurrences, -pair[1]))
    suspect.sort(key=lambda pair: -pair[1])
    chosen = sound[:limit] + suspect[:MAX_SUSPECT_INJECTED]

    if session_id:
        hits = [(fact, score, {}) for fact, score in chosen]
        store.ledger.log_many(hits, session_id=session_id, query="")
        store.mark_used([fact.id for fact, _ in chosen])

    lines = ["# Memory (nenapu)", ""]
    kept = [f for f, _ in chosen if f.status != Status.SUSPECT]
    doubted = [f for f, _ in chosen if f.status == Status.SUSPECT]
    corrections = [f for f in kept if f.kind == Kind.FEEDBACK]
    others = [f for f in kept if f.kind != Kind.FEEDBACK]

    if corrections:
        lines.append("Previously corrected — do not repeat these:")
        lines += [_correction_line(f) for f in corrections]
        lines.append("")
    if others:
        lines.append("Known about this work:")
        lines += [f"- {f.text}" for f in others]
        lines.append("")

    if doubted:
        lines.append("Do not rely on these — what they rested on was falsified:")
        lines += [f"- {f.text}" for f in doubted]
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
