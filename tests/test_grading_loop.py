"""Something has to grade a recall, or confidence never moves on feedback.

Requirement (plan "close the recall loop", tasks G4, G5, G6). Measured
against the live store on 2026-08-21: 483 recalls, 480 pending, 3 graded —
all `neutral`, all from `expiry`. `outcome_signal` therefore returns 1.0 for
every fact in the store, and the retrieval gate is stuck on no evidence.

The three tasks pinned here:

**G4 · the grader.** Grading rides the extraction call that already reads the
whole transcript, the way open loops do — "a second 83-second model call to
ask the same transcript a second question is the version of this feature
nobody would leave switched on". So:

    EXTRACT_SCHEMA gains
        grades: [{fact_id: int, verdict: "used"|"misled"|"unused", where: str}]
    EXTRACT_SYSTEM biases hard toward `unused`; `used` requires pointing at
        where in the transcript the fact was relied on.
    _injected_block(recalls)      renders the facts actually injected into
                                  this session, read from
                                  store.ledger.pending(session_id=...) —
                                  *not* from relevant_memory, which is a
                                  fresh FTS search over the transcript and a
                                  different set.
    _grades_from(store, result, recalls)
                                  used -> GOOD, misled -> BAD, unused ->
                                  NEUTRAL, applied via
                                  Ledger.grade(source='observer').

Ids the model was not shown are dropped, mirroring the `_proposed_id` guard.

**G5 · unmentioned injections become neutral.** Once an extraction actually
ran, every pending recall of that session the grader did not name is graded
`neutral` with `outcome_source='observer-unused'`. "Was in context all
session and never came up" is the honest reading, and it is what fills the
denominator. Only on success: a failed model call leaves recalls pending.

**G6 · replay the backlog.** `nenapu grade --replay [--since DAYS]` runs the
grader over every session that still has pending recalls and a transcript on
disk, so the gate is answerable today rather than in two weeks. Grades carry
`outcome_source='observer-replay'`.

Assumed seams, proposed by the plan rather than present in the codebase:

    observer._injected_block(recalls) -> str
    observer._grades_from(store, result, recalls) -> int
    observe_transcript(..., grade_source="observer")   # replay overrides it
    cli.replay_pending_sessions(store, *, since_days=None,
                                transcripts_root=None, db=None) -> list[str]

Transcripts resolve by matching `recalls.session_id` against the basenames of
`~/.claude/projects/*/*.jsonl`; `transcripts_root` is that directory, made
injectable so the tests do not read the developer's own sessions.

Every test that describes behaviour the code does not have yet carries a
strict xfail. Remove the marker when the task lands — a marker that outlives
its implementation fails the suite, which is the point.
"""

import json
import os
import subprocess
import sys
import time

import pytest

from nenapu import connect
from nenapu.llm import LLMUnavailable
from nenapu.models import Fact, Outcome
from nenapu.store import Store

g5 = pytest.mark.xfail(strict=True, reason="G5 not implemented yet: remove when it lands")
g6 = pytest.mark.xfail(strict=True, reason="G6 not implemented yet: remove when it lands")

DAY = 86400.0


@pytest.fixture
def store():
    return Store(connect(":memory:"))


class FakeBackend:
    name = "fake"
    model = "fake"
    supports_schema = False


def _event(role: str, text: str) -> str:
    return json.dumps({
        "type": role,
        "message": {"role": role, "content": [{"type": "text", "text": text}]},
    })


def _transcript(tmp_path, *turns, name="t.jsonl"):
    """A transcript long enough to be worth extracting from — `observe_transcript`
    returns early under 200 characters, and a test that trips that guard is
    testing the guard rather than the grader."""
    path = tmp_path / name
    lines = [_event("user", turns[0] if turns else "let us work on the deploy path")]
    lines += [_event("assistant", "working on it. " + "detail " * 40)]
    lines += [_event("user", t) for t in turns[1:]]
    path.write_text("\n".join(lines))
    return path


def _patch_structured(monkeypatch, payload):
    """Stand in for the one model call, and keep the prompt it was given."""
    prompts = []

    def fake(prompt, schema, system=None, backend=None, max_tokens=None):
        prompts.append(prompt)
        if isinstance(payload, Exception):
            raise payload
        return payload

    monkeypatch.setattr("nenapu.observer.structured", fake)
    return prompts


def _inject(store, texts, *, session_id="s-1"):
    """Facts injected at SessionStart: written, then logged as recalls with an
    empty query, which is exactly what `recall_context` does."""
    facts = []
    for text in texts:
        fact, _ = store.write(Fact(text=text))
        facts.append(fact)
    store.ledger.log_many(
        [(f, 0.5, {}) for f in facts], session_id=session_id, query="",
    )
    return facts


def _outcomes(store, session_id="s-1"):
    rows = store.conn.execute(
        "SELECT outcome, outcome_source FROM recalls WHERE session_id = ?", (session_id,)
    )
    return [(r["outcome"], r["outcome_source"]) for r in rows]


# ==========================================================================
# G4 · the grader
# ==========================================================================


def test_the_schema_asks_the_extractor_to_grade_what_it_was_shown():
    from nenapu.observer import EXTRACT_SCHEMA

    grades = EXTRACT_SCHEMA["properties"]["grades"]
    item = grades["items"]

    assert grades["type"] == "array"
    assert set(item["required"]) >= {"fact_id", "verdict", "where"}
    assert set(item["properties"]["verdict"]["enum"]) == {"used", "misled", "unused"}
    assert item["properties"]["fact_id"]["type"] == "integer"


def test_grading_does_not_become_a_second_model_call(store, tmp_path, monkeypatch):
    """The design constraint the whole task hangs on: one call already reads
    the whole transcript, and a second 83-second call to ask it another
    question is the version of this feature nobody would leave switched on."""
    from nenapu.observer import observe_transcript

    _inject(store, ["the deploy script lives in tools/ship.sh"])
    prompts = _patch_structured(monkeypatch, {"facts": [], "grades": [
        {"fact_id": 1, "verdict": "used", "where": "user asked about ship.sh"},
    ]})

    observe_transcript(store, _transcript(tmp_path), session_id="s-1",
                       backend=FakeBackend())

    assert len(prompts) == 1


def test_the_prompt_biases_toward_unused_and_asks_where(store):
    """`used` is the verdict that moves confidence upward, so it is the one
    that has to be paid for with evidence from the transcript."""
    from nenapu.observer import EXTRACT_SYSTEM

    system = EXTRACT_SYSTEM.lower()

    assert "unused" in system
    assert "used" in system and "where" in system


def test_a_fact_the_session_relied_on_is_graded_good(store, tmp_path, monkeypatch):
    from nenapu.observer import observe_transcript

    fact = _inject(store, ["the deploy script lives in tools/ship.sh"])[0]
    _patch_structured(monkeypatch, {"facts": [], "grades": [
        {"fact_id": fact.id, "verdict": "used",
         "where": "the user ran tools/ship.sh after reading it"},
    ]})

    observe_transcript(store, _transcript(tmp_path), session_id="s-1",
                       backend=FakeBackend())

    assert _outcomes(store) == [(Outcome.GOOD, "observer")]


def test_a_fact_the_session_contradicted_is_graded_bad(store, tmp_path, monkeypatch):
    from nenapu.observer import observe_transcript

    fact = _inject(store, ["the deploy script lives in tools/ship.sh"])[0]
    _patch_structured(monkeypatch, {"facts": [], "grades": [
        {"fact_id": fact.id, "verdict": "misled",
         "where": "the user said it moved to scripts/ship.sh months ago"},
    ]})

    observe_transcript(store, _transcript(tmp_path), session_id="s-1",
                       backend=FakeBackend())

    assert _outcomes(store) == [(Outcome.BAD, "observer")]


def test_a_fact_named_as_unused_is_graded_neutral(store, tmp_path, monkeypatch):
    from nenapu.observer import observe_transcript

    fact = _inject(store, ["the deploy script lives in tools/ship.sh"])[0]
    _patch_structured(monkeypatch, {"facts": [], "grades": [
        {"fact_id": fact.id, "verdict": "unused", "where": ""},
    ]})

    observe_transcript(store, _transcript(tmp_path), session_id="s-1",
                       backend=FakeBackend())

    assert _outcomes(store) == [(Outcome.NEUTRAL, "observer")]


def test_a_bad_grade_costs_the_fact_confidence(store, tmp_path, monkeypatch):
    """The loop closing: a graded recall has to reach `outcome_signal`, or the
    grader is a report rather than feedback."""
    from nenapu.observer import observe_transcript
    from nenapu.store import effective_confidence

    fact, twin = _inject(store, ["the deploy script lives in tools/ship.sh",
                                 "an identically believed fact nobody graded"])
    _patch_structured(monkeypatch, {"facts": [], "grades": [
        {"fact_id": fact.id, "verdict": "misled", "where": "it moved"},
    ]})

    observe_transcript(store, _transcript(tmp_path), session_id="s-1",
                       backend=FakeBackend())

    # Measured against an ungraded twin rather than against a reading taken a
    # moment earlier: decay alone moves the second number, and a test that
    # cannot tell decay from feedback would pass before the grader exists.
    assert store.get(fact.id).bad_recalls == 1
    assert effective_confidence(store.get(fact.id)) < effective_confidence(store.get(twin.id))


def test_an_id_never_injected_into_this_session_is_ignored(store, tmp_path, monkeypatch):
    """Mirrors the `_proposed_id` guard: real ids are guessable, and a model
    that invents one must not be able to grade a fact it never saw."""
    from nenapu.observer import observe_transcript

    mine = _inject(store, ["the deploy script lives in tools/ship.sh"])[0]
    theirs = _inject(store, ["another repo uses make ship"], session_id="s-other")[0]
    _patch_structured(monkeypatch, {"facts": [], "grades": [
        {"fact_id": theirs.id, "verdict": "misled", "where": "invented"},
        {"fact_id": 99999, "verdict": "used", "where": "invented"},
        {"fact_id": mine.id, "verdict": "used", "where": "real"},
    ]})

    observe_transcript(store, _transcript(tmp_path), session_id="s-1",
                       backend=FakeBackend())

    assert _outcomes(store, "s-1") == [(Outcome.GOOD, "observer")]
    assert _outcomes(store, "s-other") == [(Outcome.PENDING, None)]


def test_a_malformed_grade_is_dropped_rather_than_fatal(store, tmp_path, monkeypatch):
    """A hook must never break a session over what a model returned."""
    from nenapu.observer import observe_transcript

    fact = _inject(store, ["the deploy script lives in tools/ship.sh"])[0]
    _patch_structured(monkeypatch, {"facts": [], "grades": [
        {"verdict": "used", "where": "no id at all"},
        {"fact_id": "not a number", "verdict": "used", "where": "x"},
        {"fact_id": fact.id, "verdict": "sideways", "where": "unknown verdict"},
    ]})

    observe_transcript(store, _transcript(tmp_path), session_id="s-1",
                       backend=FakeBackend())

    assert _outcomes(store) == [(Outcome.PENDING, None)]


def test_the_injected_block_is_read_from_the_ledger(store):
    """`relevant_memory` is a fresh FTS search over the transcript, which is a
    different set from what was actually injected. Grading the wrong set would
    grade facts this session never saw."""
    from nenapu.observer import _injected_block

    facts = _inject(store, ["the deploy script lives in tools/ship.sh",
                            "staging runs on box-7"])
    recalls = store.ledger.pending(session_id="s-1")

    block = _injected_block(recalls)

    for fact in facts:
        assert str(fact.id) in block
        assert fact.text in block


def test_an_empty_injection_renders_nothing(store):
    from nenapu.observer import _injected_block

    assert _injected_block([]) == ""


def test_the_extraction_prompt_shows_what_was_injected(store, tmp_path, monkeypatch):
    """End of the same thread: the block has to reach the prompt, and it has
    to carry the facts the session was given rather than the ones a search
    over the transcript happens to find."""
    from nenapu.observer import observe_transcript

    injected = _inject(store, ["the badger enclosure needs a new latch"])[0]
    store.write(Fact(text="deploy runs from tools/ship.sh"))  # only FTS would find this
    prompts = _patch_structured(monkeypatch, {"facts": [], "grades": []})

    observe_transcript(store, _transcript(tmp_path, "how do we deploy?"),
                       session_id="s-1", backend=FakeBackend())

    assert injected.text in prompts[0]
    assert str(injected.id) in prompts[0]


def test_the_grader_is_not_shown_confidence_or_provenance(store, tmp_path, monkeypatch):
    """Self-confirmation risk: the extractor writes facts and now grades them.
    `used` needs transcript evidence, and the grader is never shown a fact's
    confidence or provenance, so it cannot defend one it recognises."""
    from nenapu.observer import _injected_block

    _inject(store, ["the deploy script lives in tools/ship.sh"])
    block = _injected_block(store.ledger.pending(session_id="s-1")).lower()

    for tell in ("confidence", "origin", "user_stated", "tool_observed", "occurrences"):
        assert tell not in block


def test_an_unavailable_backend_leaves_every_recall_pending(store, tmp_path, monkeypatch):
    """The hook must never break a session, and it must never invent evidence
    out of a failure either."""
    from nenapu.observer import observe_transcript

    _inject(store, [f"fact number {i}" for i in range(5)])
    _patch_structured(monkeypatch, LLMUnavailable("no backend"))

    with pytest.raises(LLMUnavailable):
        observe_transcript(store, _transcript(tmp_path), session_id="s-1",
                           backend=FakeBackend())

    assert _outcomes(store) == [(Outcome.PENDING, None)] * 5


def test_a_dry_run_grades_nothing(store, tmp_path, monkeypatch):
    from nenapu.observer import observe_transcript

    fact = _inject(store, ["the deploy script lives in tools/ship.sh"])[0]
    _patch_structured(monkeypatch, {"facts": [], "grades": [
        {"fact_id": fact.id, "verdict": "used", "where": "used it"},
    ]})

    observe_transcript(store, _transcript(tmp_path), session_id="s-1",
                       backend=FakeBackend(), apply=False)

    assert _outcomes(store) == [(Outcome.PENDING, None)]


def test_a_session_without_an_id_grades_nothing(store, tmp_path, monkeypatch):
    """Without a session there is no injected set to grade, and grading by
    fact id alone would reach across every session in the store."""
    from nenapu.observer import observe_transcript

    fact = _inject(store, ["the deploy script lives in tools/ship.sh"])[0]
    _patch_structured(monkeypatch, {"facts": [], "grades": [
        {"fact_id": fact.id, "verdict": "misled", "where": "x"},
    ]})

    observe_transcript(store, _transcript(tmp_path), backend=FakeBackend())

    assert _outcomes(store) == [(Outcome.PENDING, None)]


def test_an_extraction_without_grades_still_writes_facts(store, tmp_path, monkeypatch):
    """A model that does not know about the new field is proposing what it has
    always proposed — the same contract `open_loops` already keeps."""
    from nenapu.observer import observe_transcript

    _inject(store, ["the deploy script lives in tools/ship.sh"])
    _patch_structured(monkeypatch, {"facts": [
        {"text": "The user wants pnpm, not npm.", "kind": "feedback",
         "key": "pkg.manager", "correction": True},
    ]})

    learned = observe_transcript(store, _transcript(tmp_path), session_id="s-1",
                                 backend=FakeBackend())

    assert [f.text for f in learned] == ["The user wants pnpm, not npm."]


def test_a_human_grade_is_not_overwritten_by_the_grader(store, tmp_path, monkeypatch):
    """First grade wins, as everywhere else: `Ledger.grade` is reused rather
    than re-implemented."""
    from nenapu.observer import observe_transcript

    fact = _inject(store, ["the deploy script lives in tools/ship.sh"])[0]
    recall_id = store.ledger.pending(session_id="s-1")[0].id
    store.ledger.grade(recall_id, Outcome.BAD, source="human")
    _patch_structured(monkeypatch, {"facts": [], "grades": [
        {"fact_id": fact.id, "verdict": "used", "where": "x"},
    ]})

    observe_transcript(store, _transcript(tmp_path), session_id="s-1",
                       backend=FakeBackend())

    assert _outcomes(store) == [(Outcome.BAD, "human")]


def test_grades_are_applied_by_the_helper_the_plan_names(store):
    """`_grades_from` is the seam the replay path reuses, so it is tested on
    its own rather than only through the hook."""
    from nenapu.observer import _grades_from

    facts = _inject(store, ["one fact", "another fact"])
    recalls = store.ledger.pending(session_id="s-1")
    result = {"grades": [{"fact_id": facts[0].id, "verdict": "used", "where": "here"}]}

    graded = _grades_from(store, result, recalls)

    assert graded == 1
    assert store.ledger.get(recalls[-1].id).outcome in (Outcome.GOOD, Outcome.PENDING)


# ==========================================================================
# G5 · unmentioned injections become neutral
# ==========================================================================


@g5
def test_every_injected_recall_is_graded_after_a_successful_extraction(
    store, tmp_path, monkeypatch
):
    """"Was in context all session and never came up" is the honest reading,
    and it is what fills the denominator the gate divides by."""
    from nenapu.observer import observe_transcript

    facts = _inject(store, [f"injected fact number {i}" for i in range(15)])
    _patch_structured(monkeypatch, {"facts": [], "grades": [
        {"fact_id": facts[0].id, "verdict": "used", "where": "used it"},
        {"fact_id": facts[1].id, "verdict": "misled", "where": "it was wrong"},
    ]})

    observe_transcript(store, _transcript(tmp_path), session_id="s-1",
                       backend=FakeBackend())

    outcomes = _outcomes(store)
    assert sum(1 for _o, src in outcomes if src == "observer") == 2
    assert sum(1 for _o, src in outcomes if src == "observer-unused") == 13
    assert all(outcome != Outcome.PENDING for outcome, _src in outcomes)


@g5
def test_the_unnamed_ones_are_neutral_and_carry_their_own_source(
    store, tmp_path, monkeypatch
):
    """The distinct source is what lets G7 measure the population separately,
    and lets a later audit exclude it if it proves noisy."""
    from nenapu.observer import observe_transcript

    _inject(store, ["a fact nobody mentioned"])
    _patch_structured(monkeypatch, {"facts": [], "grades": []})

    observe_transcript(store, _transcript(tmp_path), session_id="s-1",
                       backend=FakeBackend())

    assert _outcomes(store) == [(Outcome.NEUTRAL, "observer-unused")]


def test_a_failed_extraction_neutralises_nothing(store, tmp_path, monkeypatch):
    """Only fires when the extraction succeeded. A failed model call that
    silently neutralised fifteen recalls would manufacture the evidence the
    gate reads."""
    from nenapu.observer import observe_transcript

    _inject(store, [f"injected fact number {i}" for i in range(15)])
    _patch_structured(monkeypatch, LLMUnavailable("backend down"))

    with pytest.raises(LLMUnavailable):
        observe_transcript(store, _transcript(tmp_path), session_id="s-1",
                           backend=FakeBackend())

    assert _outcomes(store) == [(Outcome.PENDING, None)] * 15


def test_a_skipped_extraction_neutralises_nothing(store, tmp_path, monkeypatch):
    """A transcript too short to extract from never ran a grader over
    anything, so it has said nothing about what was injected."""
    from nenapu.observer import observe_transcript

    _inject(store, ["a fact nobody mentioned"])
    calls = _patch_structured(monkeypatch, {"facts": [], "grades": []})
    short = tmp_path / "short.jsonl"
    short.write_text(_event("user", "hi"))

    observe_transcript(store, short, session_id="s-1", backend=FakeBackend())

    assert calls == []
    assert _outcomes(store) == [(Outcome.PENDING, None)]


def test_a_dry_run_neutralises_nothing(store, tmp_path, monkeypatch):
    from nenapu.observer import observe_transcript

    _inject(store, ["a fact nobody mentioned"])
    _patch_structured(monkeypatch, {"facts": [], "grades": []})

    observe_transcript(store, _transcript(tmp_path), session_id="s-1",
                       backend=FakeBackend(), apply=False)

    assert _outcomes(store) == [(Outcome.PENDING, None)]


@g5
def test_only_this_session_is_neutralised(store, tmp_path, monkeypatch):
    """Another session's pending recalls are another session's evidence."""
    from nenapu.observer import observe_transcript

    _inject(store, ["mine"], session_id="s-1")
    _inject(store, ["theirs"], session_id="s-2")
    _patch_structured(monkeypatch, {"facts": [], "grades": []})

    observe_transcript(store, _transcript(tmp_path), session_id="s-1",
                       backend=FakeBackend())

    assert _outcomes(store, "s-1") == [(Outcome.NEUTRAL, "observer-unused")]
    assert _outcomes(store, "s-2") == [(Outcome.PENDING, None)]


def test_an_already_graded_recall_is_left_alone(store, tmp_path, monkeypatch):
    from nenapu.observer import observe_transcript

    _inject(store, ["graded by a human already"])
    recall_id = store.ledger.pending(session_id="s-1")[0].id
    store.ledger.grade(recall_id, Outcome.GOOD, source="human")
    _patch_structured(monkeypatch, {"facts": [], "grades": []})

    observe_transcript(store, _transcript(tmp_path), session_id="s-1",
                       backend=FakeBackend())

    assert _outcomes(store) == [(Outcome.GOOD, "human")]


# ==========================================================================
# G6 · replay the backlog
# ==========================================================================


def _projects_root(tmp_path, sessions):
    """`~/.claude/projects/<project>/<session-id>.jsonl`, which is where the
    18 surviving transcripts of the 20 sessions with pending recalls live."""
    root = tmp_path / "projects"
    (root / "a-project").mkdir(parents=True)
    for session_id in sessions:
        (root / "a-project" / f"{session_id}.jsonl").write_text(
            "\n".join([_event("user", "we talked about the deploy path"),
                       _event("assistant", "yes. " + "detail " * 40)])
        )
    return root


def _pending_recall(store, text, *, session_id, days_ago=1.0):
    fact, _ = store.write(Fact(text=text))
    recall_id = store.ledger.log(fact.id, session_id=session_id)
    store.conn.execute(
        "UPDATE recalls SET created_at = ? WHERE id = ?",
        (time.time() - days_ago * DAY, recall_id),
    )
    store.conn.commit()
    return recall_id


def _queued(store):
    return [dict(r) for r in store.conn.execute("SELECT * FROM ingest_queue")]


@g6
def test_replay_queues_every_session_that_still_has_pending_recalls(store, tmp_path):
    """Reuses the ingest queue on purpose: 18 extractions drain serially
    through the one worker holding the lock, rather than fanning out 18
    concurrent model calls at one store."""
    from nenapu.cli import replay_pending_sessions

    _pending_recall(store, "one", session_id="s-a")
    _pending_recall(store, "two", session_id="s-b")
    root = _projects_root(tmp_path, ["s-a", "s-b"])

    queued = replay_pending_sessions(store, transcripts_root=root)

    assert sorted(queued) == ["s-a", "s-b"]
    assert len(_queued(store)) == 2


@g6
def test_a_session_whose_transcript_is_gone_is_skipped(store, tmp_path):
    """18 of the 20 sessions have a transcript on disk. The other two are not
    an error, they are two sessions that cannot be replayed."""
    from nenapu.cli import replay_pending_sessions

    _pending_recall(store, "one", session_id="s-a")
    _pending_recall(store, "gone", session_id="s-vanished")
    root = _projects_root(tmp_path, ["s-a"])

    queued = replay_pending_sessions(store, transcripts_root=root)

    assert queued == ["s-a"]


@g6
def test_a_session_with_nothing_pending_is_not_replayed(store, tmp_path):
    """Replay is for the backlog. Re-reading a session whose recalls are all
    graded buys an 83-second call for no new evidence."""
    from nenapu.cli import replay_pending_sessions

    recall_id = _pending_recall(store, "already graded", session_id="s-done")
    store.ledger.grade(recall_id, Outcome.GOOD, source="human")
    root = _projects_root(tmp_path, ["s-done"])

    assert replay_pending_sessions(store, transcripts_root=root) == []


@g6
def test_replaying_twice_changes_nothing_the_second_time(store, tmp_path):
    """`enqueue_once` dedupes unfinished work and `Ledger.grade`'s
    first-grade-wins covers the rest, so a repeat replay is a no-op rather
    than a second pass over 18 transcripts."""
    from nenapu.cli import replay_pending_sessions

    _pending_recall(store, "one", session_id="s-a")
    root = _projects_root(tmp_path, ["s-a"])

    replay_pending_sessions(store, transcripts_root=root)
    before = _queued(store)
    replay_pending_sessions(store, transcripts_root=root)

    assert _queued(store) == before


@g6
def test_since_bounds_which_sessions_are_picked_up(store, tmp_path):
    from nenapu.cli import replay_pending_sessions

    _pending_recall(store, "recent", session_id="s-recent", days_ago=2)
    _pending_recall(store, "ancient", session_id="s-ancient", days_ago=40)
    root = _projects_root(tmp_path, ["s-recent", "s-ancient"])

    queued = replay_pending_sessions(store, since_days=7, transcripts_root=root)

    assert queued == ["s-recent"]


def test_a_replayed_grade_is_distinguishable_from_a_live_one(store, tmp_path, monkeypatch):
    """`observer-replay` is to `observer` what `expiry` already is to a real
    grade: an audit has to be able to tell backfilled evidence from evidence
    that arrived as the sessions ran."""
    from nenapu.observer import observe_transcript

    fact = _inject(store, ["the deploy script lives in tools/ship.sh"])[0]
    _patch_structured(monkeypatch, {"facts": [], "grades": [
        {"fact_id": fact.id, "verdict": "used", "where": "ran it"},
    ]})

    observe_transcript(store, _transcript(tmp_path), session_id="s-1",
                       backend=FakeBackend(), grade_source="observer-replay")

    assert _outcomes(store) == [(Outcome.GOOD, "observer-replay")]


@g6
def test_the_replay_flag_is_registered_on_grade():
    """`nenapu grade --replay` rather than a new command: it is the same
    question the OUTCOMES panel already answers, asked over the backlog."""
    import nenapu.cli as cli

    command = next(c for c in cli.app.registered_commands if c.name == "grade")
    source = command.callback.__code__.co_varnames

    assert "replay" in source


@g6
def test_the_replay_run_is_a_registered_command_path(tmp_path):
    db = tmp_path / "s.db"
    Store(connect(str(db)))

    result = subprocess.run(
        [sys.executable, "-m", "nenapu.cli", "grade", "--replay", "--db", str(db)],
        capture_output=True, text=True,
        env={**os.environ, "PYTHONPATH": "src", "NENAPU_NO_BANNER": "1"},
    )

    assert result.returncode == 0, result.stdout + result.stderr
