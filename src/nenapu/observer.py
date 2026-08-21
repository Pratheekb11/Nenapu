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
from .llm import Backend, LLMUnavailable, detect_backend, structured
from .models import Fact, Kind, Origin, Status, now
from .store import Store, effective_confidence, scope_for

# Keep the injected block small. It is prepended to every session, so it is
# paid for on every request whether or not it gets used.
# Tail sizing. Start small so a hook is instant on the common case, and grow
# only when a busy transcript has not yielded enough real conversation yet.
INITIAL_TAIL_BYTES = 400_000
MAX_TAIL_BYTES = 24_000_000
MAX_CONVERSATION_CHARS = 24_000

MAX_INJECTED = 12
# Per-section caps for the project block. A refactor session can touch two
# hundred files and a neglected project can hold fifty loops; either would
# spend the whole context budget on one section of a block that is paid for on
# every request.
MAX_LEFT_OFF_FILES = 6
MAX_OPEN_LOOPS = 5
MAX_CHANGED = 8
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
empty string."""


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

    backend = backend or detect_backend()
    result = structured(
        f"{_known_memory_block(known)}## Session transcript\n\n{conversation}\n\n"
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


def recall_context(
    store: Store, *, scope: str | None = None, cwd: str | None = None,
    limit: int = MAX_INJECTED, session_id: str | None = None,
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
    project_lines: list[str] = []
    if scope:
        ledger = ActivityLedger(store.conn)
        project_lines = (
            _left_off_section(ledger, scope)
            + _open_loops_section(store.conn, scope)
            + _changed_section(ledger, scope, cwd)
        )
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
    sound.sort(key=lambda pair: (pair[0].kind != Kind.FEEDBACK, -pair[0].occurrences, -pair[1]))
    suspect.sort(key=lambda pair: -pair[1])
    chosen = sound[:limit] + suspect[:MAX_SUSPECT_INJECTED]

    if session_id:
        hits = [(fact, score, {}) for fact, score in chosen]
        store.ledger.log_many(hits, session_id=session_id, query="")
        store.mark_used([fact.id for fact, _ in chosen])

    header = f"# Memory (nenapu) — {scope}" if scope else "# Memory (nenapu)"
    lines = [header, "", *project_lines]
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
