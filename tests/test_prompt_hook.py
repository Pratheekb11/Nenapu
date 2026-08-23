"""The hook that puts a prompt's memory in front of the model.

Requirement (Task 9, query-driven hybrid retrieval plan):

`recall-hook` is the template and the contract is the same: this runs inside
somebody's editor, on every turn, and it must never be the reason a session
breaks. So the interesting assertions here are not about what it emits when
things go well. They are about the six ways it can be handed nothing useful --
no store, unreadable store, no embedder, empty prompt, malformed JSON, no stdin
at all -- and exiting zero and silent for every one of them.

Two differences from `recall-hook`, both deliberate.

**It speaks JSON.** `recall-hook` writes bare text to stdout and
`tests/test_project_injection.py` asserts on that raw output, so changing it
would buy nothing and break something. This one is new, so it can use the
structured `hookSpecificOutput` envelope the harness actually documents, which
removes any question about what counts as context and what is noise.

**It says nothing rather than saying nothing usefully.** An empty envelope on
every turn where memory had no answer is still a per-turn cost, and a reader
who sees an empty block often enough stops reading the block.

Assumed seam, proposed by the plan and not yet in the codebase::

    nenapu prompt-hook   # hidden, reads {prompt, session_id, cwd} on stdin
"""

import json
import os
import subprocess
import sys
import time

import pytest

from nenapu import connect
from nenapu.models import Fact
from nenapu.store import Store, project_scope

PROMPT = "tell me about the billing service"


@pytest.fixture
def repo(tmp_path):
    """A store whose scope matches the cwd the hook will be handed."""
    work = tmp_path / "work"
    work.mkdir()
    db = tmp_path / "s.db"
    store = Store(connect(str(db)))
    store.write(Fact(text="the billing service runs on postgres",
                     scope=project_scope(str(work))))
    store.conn.close()
    return db, work


def _hook(db, payload, *, work=None, **env):
    return subprocess.run(
        [sys.executable, "-m", "nenapu.cli", "prompt-hook", "--db", str(db)],
        input=payload, capture_output=True, text=True,
        cwd=str(work) if work else None,
        env={**os.environ, "PYTHONPATH": os.path.abspath("src"),
             "NENAPU_NO_BANNER": "1", **env},
    )


def _payload(**kwargs):
    return json.dumps(kwargs)


# --- the happy path ----------------------------------------------------------


def test_it_emits_the_structured_envelope(repo):
    db, work = repo

    out = _hook(db, _payload(prompt=PROMPT, session_id="s-1", cwd=str(work)),
                work=work, NENAPU_EMBEDDINGS="off")

    assert out.returncode == 0, out.stdout + out.stderr
    body = json.loads(out.stdout)
    assert body["hookSpecificOutput"]["hookEventName"] == "UserPromptSubmit"
    assert "postgres" in body["hookSpecificOutput"]["additionalContext"]


def test_stdout_carries_the_envelope_and_nothing_else(repo):
    """A downstream parser gets JSON or gets nothing. A version stamp or a
    stray log line in the same stream is a parse error in someone's editor."""
    db, work = repo

    out = _hook(db, _payload(prompt=PROMPT, session_id="s-1", cwd=str(work)),
                work=work, NENAPU_EMBEDDINGS="off")

    assert json.loads(out.stdout)  # the whole stream parses, not just a prefix


def test_the_session_id_reaches_the_ledger(repo):
    """Dedup against the session-start block and against earlier prompts both
    depend on it, and it is the only thing tying these recalls to a session."""
    db, work = repo

    _hook(db, _payload(prompt=PROMPT, session_id="hook-prompt-1", cwd=str(work)),
          work=work, NENAPU_EMBEDDINGS="off")

    conn = connect(str(db))
    row = conn.execute(
        "SELECT query FROM recalls WHERE session_id = 'hook-prompt-1'"
    ).fetchone()
    assert row is not None
    assert row["query"]           # a query recall, which the gate counts
    assert row["query"] != PROMPT  # terms, not the prompt


# --- silence -----------------------------------------------------------------


def test_nothing_to_say_prints_nothing(repo):
    db, work = repo

    out = _hook(db, _payload(prompt="what is the weather like today",
                             session_id="s-1", cwd=str(work)),
                work=work, NENAPU_EMBEDDINGS="off")

    assert out.returncode == 0
    assert out.stdout.strip() == ""


# --- the six ways to be handed nothing ---------------------------------------

@pytest.mark.parametrize("payload", [
    "",                                   # no stdin content
    "not json at all",                    # malformed
    "{}",                                 # no prompt
    '{"prompt": ""}',                     # empty prompt
    '{"prompt": null, "session_id": 1}',  # wrong types
    '[1, 2, 3]',                          # valid JSON, wrong shape
])
def test_it_exits_clean_on_anything_it_is_handed(repo, payload):
    db, work = repo

    out = _hook(db, payload, work=work, NENAPU_EMBEDDINGS="off")

    assert out.returncode == 0, out.stdout + out.stderr
    assert "Traceback" not in out.stderr


def test_a_missing_store_is_not_an_error(tmp_path):
    """`connect` creates it. What matters is that the turn is not interrupted
    to tell the user their memory layer had nothing yet."""
    out = _hook(tmp_path / "absent.db",
                _payload(prompt=PROMPT, session_id="s-1"),
                NENAPU_EMBEDDINGS="off")

    assert out.returncode == 0
    assert out.stdout.strip() == ""


def test_an_unreadable_store_is_not_an_error(tmp_path):
    db = tmp_path / "corrupt.db"
    db.write_bytes(b"this is not a sqlite file at all, not even close")

    out = _hook(db, _payload(prompt=PROMPT, session_id="s-1"),
                NENAPU_EMBEDDINGS="off")

    assert out.returncode == 0
    assert "Traceback" not in out.stderr


# --- shape -------------------------------------------------------------------


def test_the_command_is_hidden(tmp_path):
    """Machine-to-machine, like every other hook here."""
    out = subprocess.run(
        [sys.executable, "-m", "nenapu.cli", "--help"],
        capture_output=True, text=True,
        env={**os.environ, "PYTHONPATH": os.path.abspath("src"),
             "NENAPU_NO_BANNER": "1"},
    )

    assert out.returncode == 0
    assert [ln for ln in out.stdout.splitlines()
            if ln.strip().startswith("prompt-hook")] == []


def test_the_banner_never_reaches_the_stream(repo):
    """`tests/test_banner.py` documents this exact bug class: a hook leaks the
    version stamp because the suppression tuple was not updated."""
    db, work = repo
    env = {k: v for k, v in os.environ.items() if k != "NENAPU_NO_BANNER"}

    out = subprocess.run(
        [sys.executable, "-m", "nenapu.cli", "prompt-hook", "--db", str(db)],
        input=_payload(prompt=PROMPT, session_id="s-1", cwd=str(work)),
        capture_output=True, text=True, cwd=str(work),
        env={**env, "PYTHONPATH": os.path.abspath("src"),
             "NENAPU_EMBEDDINGS": "off"},
    )

    assert out.returncode == 0
    assert json.loads(out.stdout)
    assert "nenapu" not in out.stderr.lower()


def test_it_finishes_well_inside_the_hook_timeout(repo):
    """The hook budget is ten seconds and this fires on every turn. Measured
    without an embedder, which is the floor: the real number with fastembed
    warm is recorded in the notes, not asserted here, because a wall-clock
    assertion over an ONNX load would be a flaky test on someone else's
    machine."""
    db, work = repo

    started = time.time()
    out = _hook(db, _payload(prompt=PROMPT, session_id="s-1", cwd=str(work)),
                work=work, NENAPU_EMBEDDINGS="off")
    elapsed = time.time() - started

    assert out.returncode == 0
    assert elapsed < 5.0
