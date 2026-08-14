"""The agentic layer: reading transcripts, and what gets put back in context.

These tests use transcripts shaped like the real ones — mostly tool traffic,
with the conversation scattered thinly through it — because that shape is what
broke the naive implementations. A fixed tail read passed against a synthetic
ten-turn transcript and harvested 2,400 characters of a real one.
"""

import json
import time

import pytest

from nenapu import connect
from nenapu.models import Fact, Kind, Origin, Status
from nenapu.observer import (
    MIN_INJECTED_CONFIDENCE,
    _read_transcript,
    _turns_from,
    hook_payload,
    observe_transcript,
    recall_context,
)
from nenapu.store import Store


@pytest.fixture
def store():
    return Store(connect(":memory:"))


def _event(role: str, text: str) -> str:
    return json.dumps({
        "type": role,
        "message": {"role": role, "content": [{"type": "text", "text": text}]},
    })


def _tool_event(size: int = 400) -> str:
    """A tool call, which is what real transcripts are mostly made of."""
    return json.dumps({
        "type": "assistant",
        "message": {"role": "assistant", "content": [
            {"type": "tool_use", "id": "t1", "name": "Bash",
             "input": {"command": "x" * size}},
        ]},
    })


# ---------- parsing ----------


def test_turns_keep_text_and_drop_tool_traffic():
    lines = [_event("user", "use pnpm not npm"), _tool_event(), _event("assistant", "ok")]
    assert _turns_from(lines) == ["user: use pnpm not npm", "assistant: ok"]


def test_malformed_lines_are_skipped_not_fatal():
    lines = ["", "{not json", _event("user", "hello"), "null"]
    assert _turns_from(lines) == ["user: hello"]


def test_string_content_is_read_as_well_as_block_lists():
    line = json.dumps({"type": "user", "message": {"role": "user", "content": "plain"}})
    assert _turns_from([line]) == ["user: plain"]


# ---------- the tail read ----------


def test_short_transcript_is_read_whole(tmp_path):
    path = tmp_path / "t.jsonl"
    path.write_text("\n".join([_event("user", "port is 8080"), _event("assistant", "noted")]))
    text = _read_transcript(path)
    assert "port is 8080" in text and "noted" in text


def test_tail_grows_until_it_finds_real_conversation(tmp_path):
    """A conversation buried under megabytes of tool output must still surface.

    The correction sits at the very front, behind ~2MB of tool calls. A fixed
    400KB tail cannot see it; the window has to grow.
    """
    path = tmp_path / "busy.jsonl"
    lines = [_event("user", "always run the tests with pytest -x")]
    lines += [_tool_event(2000) for _ in range(1200)]
    lines.append(_event("assistant", "understood"))
    path.write_text("\n".join(lines))
    assert path.stat().st_size > 2_000_000

    text = _read_transcript(path)
    assert "pytest -x" in text


def test_read_is_capped_and_keeps_the_end(tmp_path):
    path = tmp_path / "t.jsonl"
    path.write_text("\n".join(_event("user", "x" * 500) for _ in range(200)))
    text = _read_transcript(path, max_chars=1000)
    assert len(text) <= 1000
    # The tail is what a session most recently established, so it is what
    # survives truncation.
    assert text.endswith("x")


def test_missing_transcript_is_not_an_error(tmp_path):
    assert _read_transcript(tmp_path / "nope.jsonl") == ""


# ---------- extraction ----------


class FakeBackend:
    name = "fake"
    model = "fake"
    supports_schema = False


def _patch_structured(monkeypatch, payload):
    calls = []

    def fake(prompt, schema, system=None, backend=None, max_tokens=None):
        calls.append(prompt)
        return payload

    monkeypatch.setattr("nenapu.observer.structured", fake)
    return calls


def test_extraction_writes_facts_with_observed_provenance(store, tmp_path, monkeypatch):
    path = tmp_path / "t.jsonl"
    path.write_text("\n".join([
        _event("user", "no, use pnpm not npm, I have told you this before"),
        _event("assistant", "switching to pnpm " + "detail " * 40),
    ]))
    _patch_structured(monkeypatch, {"facts": [
        {"text": "The user wants pnpm, not npm.", "kind": "feedback",
         "key": "pkg.manager", "correction": True},
        {"text": "The app listens on 8080.", "kind": "environment",
         "key": "app.port", "correction": False},
    ]})

    learned = observe_transcript(store, path, session_id="s1", backend=FakeBackend())
    assert [f.origin for f in learned] == [Origin.TOOL_OBSERVED] * 2
    # A correction is the highest-value thing a session produces, and is
    # trusted further than an incidental observation.
    assert learned[0].confidence > learned[1].confidence
    assert store.search("pnpm")


def test_dry_run_stores_nothing(store, tmp_path, monkeypatch):
    path = tmp_path / "t.jsonl"
    path.write_text("\n".join([_event("user", "x " * 200), _event("assistant", "y " * 200)]))
    _patch_structured(monkeypatch, {"facts": [
        {"text": "Something durable.", "kind": "project", "key": "", "correction": False},
    ]})
    learned = observe_transcript(store, path, backend=FakeBackend(), apply=False)
    assert len(learned) == 1
    assert store.list_facts() == []


def test_empty_session_never_calls_the_model(store, tmp_path, monkeypatch):
    path = tmp_path / "t.jsonl"
    path.write_text(_event("user", "hi"))
    calls = _patch_structured(monkeypatch, {"facts": []})
    assert observe_transcript(store, path, backend=FakeBackend()) == []
    assert calls == []


def test_blank_facts_are_dropped(store, tmp_path, monkeypatch):
    path = tmp_path / "t.jsonl"
    path.write_text("\n".join([_event("user", "x " * 200), _event("assistant", "y " * 200)]))
    _patch_structured(monkeypatch, {"facts": [
        {"text": "   ", "kind": "project", "key": "", "correction": False},
        {"text": "Real one.", "kind": "project", "key": "", "correction": False},
    ]})
    learned = observe_transcript(store, path, backend=FakeBackend())
    assert [f.text for f in learned] == ["Real one."]


# ---------- what goes back into the session ----------


def test_corrections_are_injected_before_everything_else(store):
    store.write(Fact(text="The repo uses uv.", kind=Kind.PROJECT, confidence=0.95))
    store.write(Fact(text="Do not add a Claude co-author trailer.", kind=Kind.FEEDBACK,
                     confidence=0.7))
    block = recall_context(store)
    assert block.index("co-author") < block.index("uses uv")
    assert "do not repeat these" in block.lower()


def test_barely_believed_facts_are_not_injected(store):
    store.write(Fact(text="A guess nobody stands behind.", kind=Kind.PROJECT,
                     confidence=MIN_INJECTED_CONFIDENCE / 4))
    assert "guess" not in recall_context(store)


def test_empty_store_injects_nothing(store):
    assert recall_context(store) == ""


def test_injection_is_capped(store):
    for i in range(40):
        store.write(Fact(text=f"Fact number {i} about the project.", kind=Kind.PROJECT,
                         confidence=0.9))
    block = recall_context(store, limit=5)
    assert block.count("\n- ") == 5


def test_suspect_facts_are_flagged_not_silently_included(store):
    fact, _ = store.write(Fact(text="Endpoints go in services/auth/routes.",
                               kind=Kind.PROJECT, confidence=0.9))
    store.conn.execute("UPDATE facts SET status=? WHERE id=?", (Status.SUSPECT, fact.id))
    store.conn.commit()
    block = recall_context(store)
    assert "Do not rely on these" in block


# ---------- the hook contract ----------


def test_hook_payload_survives_whatever_it_is_handed():
    assert hook_payload('{"session_id": "abc"}') == {"session_id": "abc"}
    assert hook_payload("") == {}
    assert hook_payload("not json") == {}


# ---------- the Stop hook must not block the session it is attached to ----------


def test_detached_observe_returns_immediately(tmp_path):
    """The hook hands the work off and gets out of the way.

    Extraction is a model call over a whole session — 83 seconds against real
    transcripts. Doing it inline means Claude Code kills the hook at its
    timeout and nothing is ever written, which looks exactly like a memory
    layer that works and learns nothing.
    """
    import os
    import subprocess
    import sys

    transcript = tmp_path / "t.jsonl"
    transcript.write_text("\n".join(
        _event("user", "the deploy command is `make ship`") for _ in range(50)))

    payload = json.dumps({"transcript_path": str(transcript), "session_id": "s9"})
    started = time.time()
    result = subprocess.run(
        [sys.executable, "-c",
         "import sys; sys.argv[0]='nenapu'; from nenapu.cli import app; app()",
         "observe", "--stdin", "--detach", "--db", str(tmp_path / "s.db")],
        input=payload, capture_output=True, text=True,
        env={**os.environ, "PYTHONPATH": "src", "NENAPU_NO_BANNER": "1",
             # Anything slow, so a child that was waited on would be obvious.
             "NENAPU_LLM": "exec", "NENAPU_LLM_CMD": "sleep 30"},
    )
    assert result.returncode == 0
    assert time.time() - started < 10, "the hook waited for the extraction"


def test_the_detached_child_outlives_the_hook_and_writes(tmp_path):
    """Returning fast is only half of it — the work has to actually land.

    A detached child that dies with the hook, or writes to a database the
    parent never sees, produces exactly the same fast exit as a working one.
    """
    import os
    import subprocess
    import sys

    from nenapu import connect
    from nenapu.store import Store

    transcript = tmp_path / "t.jsonl"
    transcript.write_text("\n".join(
        _event("user", "the deploy command is `make ship`, not `make deploy`")
        for _ in range(50)))
    db = tmp_path / "s.db"

    # A stub model, so this tests the plumbing rather than an LLM: sleep past
    # any plausible hook timeout, then print the extraction.
    stub = tmp_path / "stub.sh"
    stub.write_text(
        "#!/bin/sh\ncat > /dev/null\nsleep 3\n"
        'printf \'{"facts":[{"text":"Deploy with make ship.","kind":"project",\''
        '\'"key":"deploy.cmd","correction":true}]}\'\n'
    )
    stub.chmod(0o755)

    payload = json.dumps({"transcript_path": str(transcript), "session_id": "s9"})
    subprocess.run(
        [sys.executable, "-c",
         "import sys; sys.argv[0]='nenapu'; from nenapu.cli import app; app()",
         "observe", "--stdin", "--detach", "--db", str(db)],
        input=payload, capture_output=True, text=True, check=True,
        env={**os.environ, "PYTHONPATH": "src", "NENAPU_NO_BANNER": "1",
             # Strip any installed `nenapu` from PATH so the child runs the
             # source under test, not whatever version happens to be on this
             # machine — the test passed against a stale install first.
             "PATH": os.pathsep.join(
                 d for d in os.environ["PATH"].split(os.pathsep)
                 if not (d and os.path.exists(os.path.join(d, "nenapu")))),
             "NENAPU_LLM": "exec", "NENAPU_LLM_CMD": str(stub)},
    )

    deadline = time.time() + 30
    while time.time() < deadline:
        if db.exists() and Store(connect(str(db))).search("make ship"):
            break
        time.sleep(0.5)
    else:
        pytest.fail("the detached extraction never landed")


def test_a_hook_with_no_transcript_exits_clean(tmp_path):
    """A malformed or empty payload must never fail the session's teardown."""
    import os
    import subprocess
    import sys

    result = subprocess.run(
        [sys.executable, "-c",
         "import sys; sys.argv[0]='nenapu'; from nenapu.cli import app; app()",
         "observe", "--stdin", "--db", str(tmp_path / "s.db")],
        input="", capture_output=True, text=True,
        env={**os.environ, "PYTHONPATH": "src", "NENAPU_NO_BANNER": "1"},
    )
    assert result.returncode == 0


def test_a_hook_firing_inside_an_extraction_stops_there(tmp_path):
    """The recursion the exec backend makes possible, closed off.

    Extraction through an agent CLI ends that CLI's own session, which fires
    its Stop hook, which is us. Measured: `claude -p` does fire Stop. Without a
    marker the chain has no end — every observation starts another one.
    """
    import os
    import subprocess
    import sys

    transcript = tmp_path / "t.jsonl"
    transcript.write_text(_event("user", "the deploy command is `make ship`"))
    db = tmp_path / "s.db"
    payload = json.dumps({"transcript_path": str(transcript), "session_id": "s9"})

    proc = subprocess.run(
        [sys.executable, "-m", "nenapu.cli", "observe", "--stdin", "--detach",
         "--db", str(db)],
        input=payload, capture_output=True, text=True,
        env={**os.environ, "PYTHONPATH": "src", "NENAPU_NO_BANNER": "1",
             "NENAPU_OBSERVING": "1"},
    )

    assert proc.returncode == 0
    time.sleep(1)
    assert not db.exists(), "the inner hook started work instead of standing down"


def test_the_extraction_itself_is_not_blocked_by_the_marker(tmp_path):
    """The guard covers the hook path only.

    The detached child runs with the marker set — it *is* the extraction.
    Reading the marker as "never observe" would mean the layer learns nothing
    at all, which is the failure it exists to prevent.
    """
    import os
    import subprocess
    import sys

    from nenapu import connect
    from nenapu.store import Store

    transcript = tmp_path / "t.jsonl"
    transcript.write_text("\n".join(
        _event("user", "the deploy command is `make ship`, not `make deploy`")
        for _ in range(50)))
    stub = tmp_path / "stub.sh"
    stub.write_text(
        "#!/bin/sh\ncat > /dev/null\n"
        'printf \'{"facts":[{"text":"Deploy with make ship.","kind":"project",\''
        '\'"key":"deploy.cmd","correction":true}]}\'\n'
    )
    stub.chmod(0o755)
    db = tmp_path / "s.db"

    proc = subprocess.run(
        [sys.executable, "-m", "nenapu.cli", "observe", str(transcript), "--db", str(db)],
        capture_output=True, text=True,
        env={**os.environ, "PYTHONPATH": "src", "NENAPU_NO_BANNER": "1",
             "NENAPU_OBSERVING": "1",
             "NENAPU_LLM": "exec", "NENAPU_LLM_CMD": str(stub)},
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert Store(connect(str(db))).search("make ship")
