"""Showing the extractor what is already known, so it can update instead of add.

Requirement (Task 9, priority-ordered task list, marked **Opus 5**):

    "Feed relevant memory to the extractor + `op: add|update|noop` | kills
    275 near-duplicate pairs; yields the `occurrences` counter | M | Opus 5
    | depends on 8"

The measured failure, from "Mode 2 — msg → LLM → structured JSON: mechanism
✅, inputs ❌":

    "**Existing memories are never shown to the extractor.** So it can only
    ever emit ADD. It cannot say 'this updates fact #12' or 'already known,
    skip'."

    "Over your 367 live facts: 12 groups / 29 facts are exact duplicates once
    filler words are stripped. 275 near-duplicate pairs at Jaccard ≥ 0.6. The
    same Ollama fact was independently re-learned **five times**."

The plan's implementation sketch ("How it could be implemented", item 2):

    "Before the extraction call, run the existing `Store.search()` over
    salient terms from the transcript and pass the top-N facts *with their
    ids* into the prompt. Then widen `EXTRACT_SCHEMA` from
    `{text, kind, key, correction}` to carry an operation:
    `{op: add|update|noop, target_id, text, kind, key, correction}`. This is
    one schema change plus one cheap FTS query — no second model call."

and the safety rule that makes it survivable (same section, and item 5):

    "The LLM's `update` becomes a *proposal*; `looks_contradictory` and the
    unique index still have final say. A model that hallucinates `target_id`
    must not be able to overwrite a `user_stated` fact."

    "...have the writer reject any `target_id` not present in the facts it
    was shown."

Proposed seam
-------------
Inside `nenapu.observer`, keeping the one-call shape:

    RELEVANT_MEMORY_LIMIT: int
    relevant_memory(store, conversation, *, scope=None, limit=...) -> list[Fact]
    EXTRACT_SCHEMA facts[] gains "op" and "target_id"

`observe_transcript` gains `scope=` (the session's project, so retrieval and
writes stay two-tier), shows the retrieved facts with their ids, applies the
returned ops, and rejects any `target_id` it did not show.

Remove the `pytestmark` line below when Task 9 lands.
"""

import json

import pytest

from nenapu import connect
from nenapu.models import Fact, Kind, Origin
from nenapu.observer import observe_transcript
from nenapu.store import Store

pytestmark = pytest.mark.xfail(
    reason="Task 9 (Opus 5) not implemented — tests written first", strict=False
)


class FakeBackend:
    name = "fake"
    model = "fake"
    supports_schema = False


@pytest.fixture
def store():
    return Store(connect(":memory:"))


def _event(role: str, text: str) -> str:
    return json.dumps({
        "type": role,
        "message": {"role": role, "content": [{"type": "text", "text": text}]},
    })


def _transcript(tmp_path, *texts):
    path = tmp_path / "t.jsonl"
    # 200 characters is the floor below which `observe_transcript` never calls
    # the model at all.
    path.write_text("\n".join(
        [_event("user", text) for text in texts]
        + [_event("assistant", "understood, applying that " + "detail " * 40)]
    ))
    return path


def _patch_structured(monkeypatch, payload):
    """Captures every prompt so the tests can assert what the model was shown,
    which is the entire subject of this task."""
    calls = []

    def fake(prompt, schema, system=None, backend=None, max_tokens=None):
        calls.append({"prompt": prompt, "schema": schema, "system": system})
        return payload

    monkeypatch.setattr("nenapu.observer.structured", fake)
    return calls


# ---------- the schema carries an operation ----------


def test_the_schema_offers_add_update_and_noop():
    from nenapu.observer import EXTRACT_SCHEMA

    item = EXTRACT_SCHEMA["properties"]["facts"]["items"]["properties"]
    assert set(item["op"]["enum"]) == {"add", "update", "noop"}


def test_the_schema_carries_a_target_id():
    from nenapu.observer import EXTRACT_SCHEMA

    item = EXTRACT_SCHEMA["properties"]["facts"]["items"]["properties"]
    assert "target_id" in item


# ---------- what the extractor is shown ----------


def test_relevant_memory_is_retrieved_from_the_transcripts_own_words(store):
    from nenapu.observer import relevant_memory

    store.write(Fact(text="Ollama defaults to CONTEXT 4096 and truncates silently.",
                     kind=Kind.ENVIRONMENT, confidence=0.9))
    store.write(Fact(text="The landing page must use real photos, not AI imagery.",
                     kind=Kind.FEEDBACK, confidence=0.9))

    found = relevant_memory(store, "user: the ollama context window truncated my prompt again")

    assert any("Ollama" in fact.text for fact in found)
    assert not any("photos" in fact.text for fact in found)


def test_the_prompt_shows_known_facts_with_their_ids(store, tmp_path, monkeypatch):
    """Ids are the whole mechanism: without them `update` has nothing to point
    at and the model is back to emitting ADD forever."""
    fact, _ = store.write(Fact(text="Ollama defaults to CONTEXT 4096.",
                               kind=Kind.ENVIRONMENT, confidence=0.9))
    calls = _patch_structured(monkeypatch, {"facts": []})

    observe_transcript(
        store, _transcript(tmp_path, "the ollama context window truncated it again " * 6),
        backend=FakeBackend(),
    )

    assert str(fact.id) in calls[0]["prompt"]
    assert "CONTEXT 4096" in calls[0]["prompt"]


def test_retrieval_is_scoped_to_the_session(store, tmp_path, monkeypatch):
    """Showing another project's facts would invite cross-project `update`
    proposals — the same bug as unscoped injection, on the write path."""
    store.write(Fact(text="Ollama defaults to CONTEXT 4096.", scope="repo:other@bbbbbbbb",
                     kind=Kind.ENVIRONMENT, confidence=0.9))
    calls = _patch_structured(monkeypatch, {"facts": []})

    observe_transcript(
        store, _transcript(tmp_path, "the ollama context window truncated it again " * 6),
        backend=FakeBackend(), scope="repo:here@aaaaaaaa",
    )

    assert "CONTEXT 4096" not in calls[0]["prompt"]


def test_an_empty_store_still_produces_one_clean_call(store, tmp_path, monkeypatch):
    calls = _patch_structured(monkeypatch, {"facts": [
        {"op": "add", "text": "The user wants pnpm, not npm.", "kind": "feedback",
         "key": "pkg.manager", "correction": True},
    ]})

    learned = observe_transcript(
        store, _transcript(tmp_path, "no, use pnpm not npm " * 20), backend=FakeBackend(),
    )

    assert len(calls) == 1
    assert [f.text for f in learned] == ["The user wants pnpm, not npm."]


def test_retrieval_costs_no_extra_model_call(store, tmp_path, monkeypatch):
    """"one schema change plus one cheap FTS query — no second model call.\""""
    store.write(Fact(text="Ollama defaults to CONTEXT 4096.", kind=Kind.ENVIRONMENT,
                     confidence=0.9))
    calls = _patch_structured(monkeypatch, {"facts": []})

    observe_transcript(
        store, _transcript(tmp_path, "ollama context window " * 30), backend=FakeBackend(),
    )

    assert len(calls) == 1


def test_the_shown_facts_are_capped(store, tmp_path, monkeypatch):
    """Extraction already runs at 6k tokens against an 83-second call. The
    known-memory section cannot grow with the store."""
    from nenapu.observer import RELEVANT_MEMORY_LIMIT

    for i in range(60):
        store.write(Fact(text=f"Ollama fact number {i} about the context window.",
                         kind=Kind.ENVIRONMENT, confidence=0.9))
    calls = _patch_structured(monkeypatch, {"facts": []})

    observe_transcript(
        store, _transcript(tmp_path, "ollama context window " * 30), backend=FakeBackend(),
    )

    shown = sum(1 for i in range(60) if f"number {i} " in calls[0]["prompt"])
    assert shown <= RELEVANT_MEMORY_LIMIT


def test_the_prompt_is_still_redacted(store, tmp_path, monkeypatch):
    """Redaction stays at harvest, upstream of the model call — this task adds
    a second thing to the prompt and must not open a second route."""
    calls = _patch_structured(monkeypatch, {"facts": []})

    observe_transcript(
        store,
        _transcript(tmp_path, "here is the config DB_PASSWORD=hunter2swordfish " * 12),
        backend=FakeBackend(),
    )

    assert "hunter2swordfish" not in calls[0]["prompt"]


# ---------- applying the operations ----------


def test_add_writes_a_new_fact(store, tmp_path, monkeypatch):
    _patch_structured(monkeypatch, {"facts": [
        {"op": "add", "text": "The app listens on 8080.", "kind": "environment",
         "key": "app.port", "correction": False},
    ]})

    observe_transcript(store, _transcript(tmp_path, "the app listens on 8080 " * 20),
                       backend=FakeBackend())

    assert len(store.list_facts()) == 1


def test_a_missing_op_is_treated_as_add(store, tmp_path, monkeypatch):
    """Back-compat with every backend that has not been told about the new
    field, including the ones the calibration table scores badly."""
    _patch_structured(monkeypatch, {"facts": [
        {"text": "The app listens on 8080.", "kind": "environment", "key": "app.port",
         "correction": False},
    ]})

    observe_transcript(store, _transcript(tmp_path, "the app listens on 8080 " * 20),
                       backend=FakeBackend())

    assert len(store.list_facts()) == 1


def test_update_merges_into_the_existing_fact_instead_of_duplicating(store, tmp_path,
                                                                     monkeypatch):
    """The headline: the Ollama fact was re-learned five times, each phrased
    differently. One fact, one counter, is the correct outcome."""
    fact, _ = store.write(Fact(text="Ollama defaults to CONTEXT 4096.",
                               kind=Kind.ENVIRONMENT, confidence=0.9))
    _patch_structured(monkeypatch, {"facts": [
        {"op": "update", "target_id": fact.id,
         "text": "Ollama defaults to CONTEXT 4096 and truncates the oldest turns.",
         "kind": "environment", "key": "", "correction": False},
    ]})

    observe_transcript(
        store, _transcript(tmp_path, "ollama context window truncated it " * 20),
        backend=FakeBackend(),
    )

    assert len(store.list_facts()) == 1


def test_update_increments_the_occurrences_counter(store, tmp_path, monkeypatch):
    """"kills 275 near-duplicate pairs; yields the `occurrences` counter" —
    which Task 10 already renders as "you have had to say this 3 times"."""
    fact, _ = store.write(Fact(text="The pet must be a dog, not a bear.",
                               kind=Kind.FEEDBACK, confidence=0.9))
    _patch_structured(monkeypatch, {"facts": [
        {"op": "update", "target_id": fact.id,
         "text": "The pet must be a dog, not a bear.", "kind": "feedback",
         "key": "", "correction": True},
    ]})

    observe_transcript(store, _transcript(tmp_path, "no, a dog, not a bear " * 20),
                       backend=FakeBackend())

    assert store.get(fact.id).occurrences == 2


def test_noop_writes_nothing(store, tmp_path, monkeypatch):
    fact, _ = store.write(Fact(text="Ollama defaults to CONTEXT 4096.",
                               kind=Kind.ENVIRONMENT, confidence=0.9))
    _patch_structured(monkeypatch, {"facts": [
        {"op": "noop", "target_id": fact.id, "text": "Ollama defaults to CONTEXT 4096.",
         "kind": "environment", "key": "", "correction": False},
    ]})

    learned = observe_transcript(
        store, _transcript(tmp_path, "ollama context window " * 20), backend=FakeBackend(),
    )

    assert learned == []
    assert len(store.list_facts()) == 1
    assert store.get(fact.id).occurrences == 1


# ---------- the safety rules, which is why this task is Opus-tier ----------


def test_a_hallucinated_target_id_cannot_write_over_anything(store, tmp_path, monkeypatch):
    """1.5b "invents ids — 9 verdicts for 4 facts" (implementation notes,
    calibration table). Inventing ids is survivable for pure ADD and is
    exactly what this schema change makes dangerous."""
    _patch_structured(monkeypatch, {"facts": [
        {"op": "update", "target_id": 9999, "text": "Something newly claimed.",
         "kind": "project", "key": "", "correction": False},
    ]})

    learned = observe_transcript(
        store, _transcript(tmp_path, "we decided something " * 20), backend=FakeBackend(),
    )

    assert [f.text for f in learned] == ["Something newly claimed."]
    assert [f.text for f in store.list_facts()] == ["Something newly claimed."]


def test_an_update_targeting_a_fact_that_was_not_shown_is_rejected(store, tmp_path,
                                                                   monkeypatch):
    """"reject any `target_id` not present in the facts it was shown" — a real
    id is not authority to edit a fact this session never saw."""
    unseen, _ = store.write(Fact(text="The portfolio hero uses a real photo.",
                                 scope="repo:portfolio@bbbbbbbb", kind=Kind.FEEDBACK,
                                 confidence=0.9))
    _patch_structured(monkeypatch, {"facts": [
        {"op": "update", "target_id": unseen.id, "text": "The hero uses AI art.",
         "kind": "feedback", "key": "", "correction": True},
    ]})

    observe_transcript(
        store, _transcript(tmp_path, "the ollama context window truncated it " * 20),
        backend=FakeBackend(), scope="repo:here@aaaaaaaa",
    )

    assert store.get(unseen.id).text == "The portfolio hero uses a real photo."


def test_an_update_cannot_rewrite_what_the_user_stated(store, tmp_path, monkeypatch):
    """The project's oldest invariant: "An agent's guess must never silently
    overwrite what you said" (implementation notes, the `origin` row)."""
    fact, _ = store.write(Fact(text="Never add a Claude co-author trailer.",
                               kind=Kind.FEEDBACK, origin=Origin.USER_STATED,
                               confidence=0.95))
    _patch_structured(monkeypatch, {"facts": [
        {"op": "update", "target_id": fact.id,
         "text": "Co-author trailers are fine now.", "kind": "feedback",
         "key": "", "correction": True},
    ]})

    observe_transcript(
        store, _transcript(tmp_path, "about the co-author trailer " * 20),
        backend=FakeBackend(),
    )

    kept = store.get(fact.id)
    assert kept.text == "Never add a Claude co-author trailer."
    assert kept.origin == Origin.USER_STATED
    assert kept.occurrences == 2


def test_a_dry_run_applies_no_operation_at_all(store, tmp_path, monkeypatch):
    """`--dry-run` is one of the two defences named against poisoned content;
    it must stay honest once the schema can mutate rows rather than only
    insert them."""
    fact, _ = store.write(Fact(text="Ollama defaults to CONTEXT 4096.",
                               kind=Kind.ENVIRONMENT, confidence=0.9))
    _patch_structured(monkeypatch, {"facts": [
        {"op": "update", "target_id": fact.id, "text": "Ollama now defaults to 8192.",
         "kind": "environment", "key": "", "correction": False},
    ]})

    proposals = observe_transcript(
        store, _transcript(tmp_path, "ollama context window " * 20),
        backend=FakeBackend(), apply=False,
    )

    assert proposals
    assert store.get(fact.id).text == "Ollama defaults to CONTEXT 4096."
    assert store.get(fact.id).occurrences == 1


def test_a_target_id_on_an_add_is_ignored_not_followed(store, tmp_path, monkeypatch):
    """A model that fills every field regardless must not be able to turn an
    `add` into an overwrite by leaving a stale id in the payload."""
    fact, _ = store.write(Fact(text="Ollama defaults to CONTEXT 4096.",
                               kind=Kind.ENVIRONMENT, confidence=0.9))
    _patch_structured(monkeypatch, {"facts": [
        {"op": "add", "target_id": fact.id, "text": "The app listens on 8080.",
         "kind": "environment", "key": "app.port", "correction": False},
    ]})

    observe_transcript(store, _transcript(tmp_path, "the app listens on 8080 " * 20),
                       backend=FakeBackend())

    assert store.get(fact.id).text == "Ollama defaults to CONTEXT 4096."
    assert len(store.list_facts()) == 2
