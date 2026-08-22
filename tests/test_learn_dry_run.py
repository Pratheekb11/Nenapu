"""`nenapu learn --dry-run` must write nothing at all.

Requirement (plan "Harden the four incidents into guarantees", Phase A,
tasks A1-A3, marked **Sonnet 5**):

    A `--dry-run` that writes is worse than no `--dry-run` at all. The flag is
    honoured per command, and three paths in `learn` defeat it today.

The four incidents that started this plan were all found by running commands
against the real store rather than in review, and the last of them was a
`backfill --redate --dry-run` that wrote anyway. That was not a `--redate`
bug. `learn` has three more instances of the same class:

* `_capture_activity` runs before `dry_run` is ever consulted, so a dry run
  writes a `sessions` row, its `file_events`, the git evidence and the
  entities built from them.
* `--no-infer` calls `store_messages` with no dry-run consideration at all.
* `--detach` returns early into `_queue_and_detach`, which enqueues a real job
  and spawns a real worker. The detached run applies everything, so the flag
  that promises nothing happens buys a full extraction.

Scope boundary with the existing dry-run tests
----------------------------------------------
`tests/test_observer.py:153`, `tests/test_extractor_context.py:374` and the
others call `observe_transcript(apply=False)` directly. That is the half of
`learn` that already honours the flag, which is precisely why none of them
sees any of the three bypasses above: they never go through the command. This
file drives the CLI, as `tests/test_backfill_command.py` does.

No model is needed to prove any of this. `NENAPU_LLM` is pointed at a backend
that does not exist, so the extraction raises `LLMUnavailable` — and the
writes under test have all already happened by then, which is the sharpest
statement of the bug: a dry run that could not even reach the model still
changed the store.
"""

import json
import os
import subprocess
import sys

import pytest

from nenapu import connect

# An unknown backend makes `detect_backend` raise rather than reach a network.
NO_MODEL = {"NENAPU_LLM": "nonesuch"}


def _turn(role: str, text: str, session: str, cwd: str) -> str:
    return json.dumps({
        "type": role, "sessionId": session, "cwd": cwd,
        "message": {"role": role, "content": [{"type": "text", "text": text}]},
    })


def _tool_turn(session: str, cwd: str, path: str) -> str:
    return json.dumps({
        "type": "assistant", "sessionId": session, "cwd": cwd,
        "message": {"role": "assistant", "content": [
            {"type": "tool_use", "name": "Edit", "input": {"file_path": path}},
        ]},
    })


@pytest.fixture
def transcript(tmp_path):
    """One session that edited a file, so capture has something to record."""
    path = tmp_path / "s-dry.jsonl"
    path.write_text("\n".join([
        _turn("user", "always use the staging bucket for exports", "s-dry", "/repo"),
        _tool_turn("s-dry", "/repo", "backend/app/exports.py"),
        _turn("assistant", "switched the export target", "s-dry", "/repo"),
    ]))
    return path


@pytest.fixture
def db(tmp_path):
    return tmp_path / "s.db"


def _run(args, db, **env):
    return subprocess.run(
        [sys.executable, "-m", "nenapu.cli", *args, "--db", str(db)],
        capture_output=True, text=True,
        env={**os.environ, "PYTHONPATH": "src", "NENAPU_NO_BANNER": "1", **env},
    )


def _count(db, table: str) -> int:
    if not db.exists():
        return 0
    conn = connect(str(db))
    try:
        return conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
    finally:
        conn.close()


# ---------- A1: the activity ledger is not written either ----------


def test_a_dry_run_records_no_session(transcript, db):
    """The ledger half is still a write, and `--dry-run` covers the command."""
    _run(["learn", str(transcript), "--dry-run"], db, **NO_MODEL)

    assert _count(db, "sessions") == 0


def test_a_dry_run_records_no_file_events(transcript, db):
    """Capture writes a row per edited file. A dry run must leave none."""
    _run(["learn", str(transcript), "--dry-run"], db, **NO_MODEL)

    assert _count(db, "file_events") == 0


def test_a_dry_run_builds_no_entities(transcript, db):
    """Entities are derived from the file events, so they follow the same rule."""
    _run(["learn", str(transcript), "--dry-run"], db, **NO_MODEL)

    assert _count(db, "entities") == 0


def test_a_real_run_still_records_the_session(transcript, db):
    """The guard must not cost the ledger on the run that is not a dry one.

    The extraction still fails without a model, and that is the point: the
    ledger half deliberately runs first so a missing model does not lose what
    the session did to the files.
    """
    _run(["learn", str(transcript)], db, **NO_MODEL)

    assert _count(db, "sessions") == 1
    assert _count(db, "file_events") >= 1


# ---------- A2: --no-infer stores no messages ----------


def test_a_dry_run_stores_no_messages_verbatim(transcript, db):
    """`--no-infer` skips the model, not the flag that promises nothing happens."""
    _run(["learn", str(transcript), "--no-infer", "--dry-run"], db,
         NENAPU_STORE_MESSAGES="1")

    assert _count(db, "messages") == 0


def test_a_dry_run_says_what_it_would_have_stored(transcript, db):
    """A dry run that reports nothing is indistinguishable from one that found
    nothing, which is the whole reason to offer the flag."""
    result = _run(["learn", str(transcript), "--no-infer", "--dry-run"], db,
                  NENAPU_STORE_MESSAGES="1")

    assert "would be" in result.stdout


def test_no_infer_without_dry_run_still_stores(transcript, db):
    """The guard must not disable the mode it is guarding."""
    _run(["learn", str(transcript), "--no-infer"], db, NENAPU_STORE_MESSAGES="1")

    assert _count(db, "messages") >= 1


# ---------- A3: --detach is refused, not silently honoured ----------


def test_a_dry_run_cannot_be_detached(transcript, db):
    """One flag promises nothing happens, the other hands the work to a process
    that applies everything. The combination has no honest meaning."""
    result = _run(["learn", str(transcript), "--dry-run", "--detach"], db)

    assert result.returncode != 0


def test_a_refused_detach_says_why(transcript, db):
    """A hook passing both should hear about it rather than guess."""
    result = _run(["learn", str(transcript), "--dry-run", "--detach"], db)

    assert "--dry-run" in (result.stdout + result.stderr)


def test_a_refused_detach_queues_nothing(transcript, db):
    """The refusal has to happen before `_queue_and_detach`, or a worker is
    already running the extraction the flag said would not happen."""
    _run(["learn", str(transcript), "--dry-run", "--detach"], db)

    assert _count(db, "ingest_queue") == 0


def test_detach_on_its_own_still_queues(transcript, db):
    """The refusal is about the pair, not about `--detach`."""
    _run(["learn", str(transcript), "--detach"], db)

    assert _count(db, "ingest_queue") == 1
