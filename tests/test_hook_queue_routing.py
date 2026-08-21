"""Route the Stop hook through the ingest queue.

Requirement (Task 19, "Next up" list, 2026-08-21, marked **Sonnet 5**,
depends on 8 and 15):

    "Route the Stop hook through the ingest queue | the single-flight queue
    only serialises the watcher; `learn --detach` still forks an inline
    extraction, so two sessions ending together are still two concurrent
    83-second model calls against one store — the exact fan-out task 8 was
    built to end | M | Sonnet 5 | 8, 15"

And the sentence that makes this the top of the list:

    "**19 is the one that matters.** Everything downstream of the queue —
    `run_maintenance_tick`, loop closure, ledger capture on the extraction
    side — runs only when `worker.drain` runs, and `drain` runs only from
    `nenapu watch`. On a machine that uses hooks and never starts the
    watcher, none of the self-maintenance built in tasks 11 and 12 ever
    fires."

Today (`cli.py:_detach_observe`) the Stop hook re-execs `nenapu observe
<path>` with `start_new_session=True` and no lock, no queue and no wait. Two
sessions ending together therefore run two concurrent extractions against
one store and one log file, and nothing on a hook-only machine ever calls
`worker.drain`, so `run_maintenance_tick` and loop closure never run there.

The contract these tests pin
----------------------------
The Stop hook's job shrinks to *enqueue and hand off*:

1. `nenapu learn --stdin --detach` writes one `ingest_queue` row and returns.
   It does not extract, does not capture the ledger itself, and prints
   nothing — the hook's stdout is read by the harness.
2. It spawns one detached **worker**, not an inline extraction: the child's
   argv is the drain command, so whatever it picks up is serialised by
   `WorkerLock` against every other worker on the machine.
3. A second worker started while one is draining finds the lock taken and
   returns, leaving the jobs where they are — that is the fan-out fix.
4. Draining is what runs capture, extraction, loop closure and the
   maintenance tick, so all of it now fires on a hook-only machine.

New CLI surface assumed: `nenapu drain [--limit N] [--db PATH]`, a thin
wrapper over `worker.drain`, whose lock lives beside the store
(`<db>.parent/worker.lock`) so two stores on one machine do not block each
other and a test can point at its own.

The installed hook command itself does **not** change — `nenapu learn
--stdin --detach` keeps its meaning — so no machine has to re-run `nenapu
init` to get this. That is pinned below and is green today.
"""

import json
import os
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

from nenapu import connect


# ---------- helpers ----------


def _turn(role: str, text: str, session: str = "s-1") -> str:
    return json.dumps({
        "type": role, "sessionId": session, "cwd": "/repo",
        "message": {"role": role, "content": [{"type": "text", "text": text}]},
    })


def _transcript(tmp_path, name="s-1.jsonl", session="s-1", turns=None):
    path = tmp_path / name
    path.write_text("\n".join(turns or [
        _turn("user", "use pnpm not npm", session),
        _turn("assistant", "understood, pnpm from here", session),
    ]))
    return path


def _payload(transcript, session_id="s-1", cwd="/repo") -> str:
    return json.dumps({
        "transcript_path": str(transcript), "session_id": session_id, "cwd": cwd,
    })


def _hook(transcript, db, *, session_id="s-1", env=None):
    """Fire the Stop hook in-process, with `Popen` stubbed out.

    In-process (CliRunner) rather than subprocess: the whole point is to look
    at what the hook *would* spawn and what it did not do itself, and a patch
    here would not reach another interpreter.
    """
    from typer.testing import CliRunner

    from nenapu.cli import app

    with patch("subprocess.Popen") as popen, patch(
        "nenapu.observer.observe_transcript"
    ) as observe:
        with patch.dict(os.environ, env or {}, clear=False):
            result = CliRunner().invoke(
                app,
                ["learn", "--stdin", "--detach", "--db", str(db)],
                input=_payload(transcript, session_id),
            )
    return result, popen, observe


def _queue_rows(db):
    return [dict(r) for r in connect(str(db)).execute(
        "SELECT * FROM ingest_queue ORDER BY id"
    )]


@pytest.fixture
def db(tmp_path):
    return tmp_path / "s.db"


# ---------- the hook enqueues ----------


def test_the_stop_hook_enqueues_the_transcript(tmp_path, db):
    transcript = _transcript(tmp_path)

    result, _popen, _observe = _hook(transcript, db)

    assert result.exit_code == 0, result.output
    rows = _queue_rows(db)
    assert len(rows) == 1
    assert rows[0]["path"] == str(transcript)
    assert rows[0]["state"] == "pending"


def test_the_queued_job_carries_the_agent_and_session_id(tmp_path, db):
    """`_ingest` reads both off the job: the agent stamps the ledger row and
    the session id is what ties extracted facts back to the conversation."""
    transcript = _transcript(tmp_path, session="s-42")

    _hook(transcript, db, session_id="s-42")

    row = _queue_rows(db)[0]
    assert row["agent"] == "claude-code"
    assert row["session_id"] == "s-42"


def test_the_hook_does_not_extract_anything_itself(tmp_path, db):
    """The 83-second model call must not happen on the hook's own thread —
    that is the timeout the whole `--detach` mechanism exists to survive."""
    transcript = _transcript(tmp_path)

    _result, _popen, observe = _hook(transcript, db)

    observe.assert_not_called()


def test_the_hook_prints_nothing(tmp_path, db):
    """A Stop hook that prints corrupts whatever is reading it."""
    transcript = _transcript(tmp_path)

    result, _popen, _observe = _hook(transcript, db)

    assert result.output.strip() == ""


# ---------- and hands off to a worker, not to an extraction ----------


def test_the_hook_spawns_a_worker_rather_than_an_inline_extraction(tmp_path, db):
    transcript = _transcript(tmp_path)

    _result, popen, _observe = _hook(transcript, db)

    assert popen.called, "the hook must start a worker, or a hook-only machine never drains"
    argv = popen.call_args.args[0]
    assert "drain" in argv
    assert "observe" not in argv and "learn" not in argv


def test_the_spawned_worker_is_pointed_at_the_same_store(tmp_path, db):
    transcript = _transcript(tmp_path)

    _result, popen, _observe = _hook(transcript, db)

    argv = popen.call_args.args[0]
    assert "--db" in argv and str(db) in argv


def test_the_spawned_worker_outlives_the_session_process_group(tmp_path, db):
    """The reason `--detach` exists at all: the harness tears down the
    session's process tree at the hook timeout, and an extraction in that
    tree dies with it."""
    transcript = _transcript(tmp_path)

    _result, popen, _observe = _hook(transcript, db)

    kwargs = popen.call_args.kwargs
    assert kwargs.get("start_new_session") is True
    assert kwargs.get("stdin") is subprocess.DEVNULL


def test_the_spawned_worker_carries_the_recursion_guard(tmp_path, db):
    """The worker calls the model, and when the backend is an agent CLI that
    CLI fires its own Stop hook. `NENAPU_OBSERVING` is what keeps the chain
    one level deep instead of unbounded."""
    transcript = _transcript(tmp_path)

    _result, popen, _observe = _hook(transcript, db)

    assert popen.call_args.kwargs["env"].get("NENAPU_OBSERVING") == "1"


def test_a_hook_firing_inside_an_extraction_queues_nothing(tmp_path, db):
    """Measured behaviour: `claude -p` fires Stop. Without this guard the
    queue would grow a job for every extraction, forever."""
    transcript = _transcript(tmp_path)

    result, popen, _observe = _hook(transcript, db, env={"NENAPU_OBSERVING": "1"})

    assert result.exit_code == 0
    assert _queue_rows(db) == []
    popen.assert_not_called()


# ---------- one queue, one worker ----------


def test_two_sessions_ending_together_queue_two_jobs(tmp_path, db):
    first = _transcript(tmp_path, name="a.jsonl", session="s-a")
    second = _transcript(tmp_path, name="b.jsonl", session="s-b")

    _hook(first, db, session_id="s-a")
    _hook(second, db, session_id="s-b")

    assert [r["path"] for r in _queue_rows(db)] == [str(first), str(second)]
    assert {r["state"] for r in _queue_rows(db)} == {"pending"}


def test_a_second_worker_leaves_the_jobs_alone_while_one_is_draining(tmp_path, db):
    """The fan-out fix, stated directly: whoever holds the lock does the
    work, and everyone else returns rather than doubling the load on the
    store."""
    from nenapu.ingest_queue import WorkerLock, enqueue
    from nenapu.worker import drain
    from nenapu.store import Store

    store = Store(connect(str(db)))
    enqueue(store.conn, path=str(_transcript(tmp_path)), agent="claude-code")

    held = WorkerLock(tmp_path / "worker.lock")
    assert held.try_acquire()
    try:
        assert drain(store, lock_path=tmp_path / "worker.lock") == 0
    finally:
        held.release()

    assert _queue_rows(db)[0]["state"] == "pending"

    with patch("nenapu.worker.observe_transcript", return_value=[]):
        assert drain(store, lock_path=tmp_path / "worker.lock") == 1
    assert _queue_rows(db)[0]["state"] == "done"


def test_the_same_transcript_is_not_queued_twice_while_a_job_is_pending(tmp_path, db):
    """Two hooks for one session — a retry, or the watcher and the hook
    reaching the same file — must not buy two 83-second extractions of
    identical content."""
    transcript = _transcript(tmp_path)

    _hook(transcript, db)
    _hook(transcript, db)

    assert len(_queue_rows(db)) == 1


def test_a_resumed_session_is_queued_again_once_its_job_finished(tmp_path, db):
    """The other direction of the same rule: dedupe covers *pending* work
    only. A session that was resumed and appended to is new material, and
    the watcher already re-ingests on exactly this signal."""
    from nenapu.ingest_queue import mark_done

    transcript = _transcript(tmp_path)
    _hook(transcript, db)
    conn = connect(str(db))
    mark_done(conn, _queue_rows(db)[0]["id"])

    transcript.write_text(transcript.read_text() + "\n" + _turn("user", "and use uv"))
    _hook(transcript, db)

    rows = _queue_rows(db)
    assert len(rows) == 2
    assert rows[-1]["state"] == "pending"


# ---------- what draining is now responsible for ----------


def test_draining_records_the_session_in_the_ledger(tmp_path, db):
    """Capture moved off the hook and onto the worker, so it has to still
    happen exactly once, from the job."""
    from nenapu.activity import ActivityLedger
    from nenapu.store import Store

    transcript = _transcript(tmp_path)
    _hook(transcript, db)

    from nenapu.worker import drain

    store = Store(connect(str(db)))
    with patch("nenapu.worker.observe_transcript", return_value=[]):
        drain(store, lock_path=tmp_path / "worker.lock")

    assert ActivityLedger(store.conn).get_session("s-1") is not None


def test_the_ledger_is_recorded_even_when_the_extraction_fails(tmp_path, db):
    """Deterministic and free runs first: a model backend that is down costs
    the facts, not the record of what the session touched."""
    from nenapu.activity import ActivityLedger
    from nenapu.observer import LLMUnavailable
    from nenapu.store import Store
    from nenapu.worker import drain

    transcript = _transcript(tmp_path)
    _hook(transcript, db)

    store = Store(connect(str(db)))
    with patch("nenapu.worker.observe_transcript", side_effect=LLMUnavailable("no backend")):
        drain(store, lock_path=tmp_path / "worker.lock")

    assert ActivityLedger(store.conn).get_session("s-1") is not None
    assert _queue_rows(db)[0]["state"] == "failed"


def test_one_unreadable_transcript_does_not_stop_the_queue(tmp_path, db):
    from nenapu.store import Store
    from nenapu.worker import drain

    _hook(tmp_path / "gone.jsonl", db, session_id="s-missing")
    good = _transcript(tmp_path, name="good.jsonl", session="s-good")
    _hook(good, db, session_id="s-good")

    store = Store(connect(str(db)))
    with patch("nenapu.worker.observe_transcript", return_value=[]):
        drain(store, lock_path=tmp_path / "worker.lock")

    states = [r["state"] for r in _queue_rows(db)]
    assert states.count("done") >= 1
    assert "pending" not in states


def test_the_maintenance_tick_fires_on_a_hook_only_machine(tmp_path, db):
    """The reason this task is first. `expire_pending`, dedupe and loop
    closure all ride on `worker.drain`, which on a hook-only machine used to
    never run at all."""
    from nenapu.store import Store
    from nenapu.worker import drain

    _hook(_transcript(tmp_path), db)
    store = Store(connect(str(db)))
    with patch("nenapu.worker.observe_transcript", return_value=[]):
        drain(store, lock_path=tmp_path / "worker.lock")

    marks = {r["key"] for r in store.conn.execute(
        "SELECT key FROM meta WHERE key LIKE 'maintenance:last_run:%'"
    )}
    assert "maintenance:last_run:expire_pending" in marks


# ---------- the drain command ----------


def _run(args, db, **env):
    return subprocess.run(
        [sys.executable, "-m", "nenapu.cli", *args, "--db", str(db)],
        capture_output=True, text=True,
        env={**os.environ, "PYTHONPATH": "src", "NENAPU_NO_BANNER": "1", **env},
    )


def test_drain_is_a_real_command(tmp_path, db):
    result = _run(["drain"], db)
    assert result.returncode == 0, result.stdout + result.stderr


def test_drain_reports_what_it_processed(tmp_path, db):
    from nenapu.ingest_queue import enqueue
    from nenapu.store import Store

    store = Store(connect(str(db)))
    enqueue(store.conn, path=str(_transcript(tmp_path)), agent="claude-code")

    from typer.testing import CliRunner

    from nenapu.cli import app

    with patch("nenapu.worker.observe_transcript", return_value=[]):
        result = CliRunner().invoke(app, ["drain", "--db", str(db)])

    assert result.exit_code == 0, result.output
    assert "1" in result.output
    assert _queue_rows(db)[0]["state"] == "done"


def test_drain_takes_the_lock_beside_the_store(tmp_path, db):
    """Two stores on one machine must not block each other, and the lock has
    to be findable from `--db` alone — the hook passes nothing else."""
    from nenapu.ingest_queue import WorkerLock, enqueue
    from nenapu.store import Store

    store = Store(connect(str(db)))
    enqueue(store.conn, path=str(_transcript(tmp_path)), agent="claude-code")

    held = WorkerLock(Path(db).parent / "worker.lock")
    assert held.try_acquire()
    try:
        from typer.testing import CliRunner

        from nenapu.cli import app

        with patch("nenapu.worker.observe_transcript", return_value=[]) as observe:
            result = CliRunner().invoke(app, ["drain", "--db", str(db)])
        assert result.exit_code == 0, result.output
        observe.assert_not_called()
    finally:
        held.release()

    assert _queue_rows(db)[0]["state"] == "pending"


def test_drain_limits_how_much_it_takes_in_one_pass(tmp_path, db):
    """A backlog of a hundred transcripts discovered at once must not become
    a hundred model calls in one process."""
    from nenapu.ingest_queue import enqueue
    from nenapu.store import Store

    store = Store(connect(str(db)))
    for i in range(3):
        enqueue(store.conn, path=str(_transcript(tmp_path, name=f"t{i}.jsonl",
                                                 session=f"s-{i}")), agent="claude-code")

    from typer.testing import CliRunner

    from nenapu.cli import app

    with patch("nenapu.worker.observe_transcript", return_value=[]):
        result = CliRunner().invoke(app, ["drain", "--limit", "1", "--db", str(db)])

    assert result.exit_code == 0, result.output
    assert [r["state"] for r in _queue_rows(db)].count("pending") == 2


# ---------- what must not change ----------


def test_the_installed_stop_hook_command_is_unchanged(tmp_path):
    """Green, and must stay so: `learn --stdin --detach` keeps its meaning,
    so nobody has to re-run `nenapu init` to get the queue. Changing the
    command string would leave every existing machine on the old path
    silently."""
    from nenapu.setup_wizard import hook_config

    assert "nenapu learn --stdin --detach" in json.dumps(hook_config()["Stop"])


def test_learn_with_a_path_and_no_detach_still_extracts_inline(tmp_path, db):
    """Green pin. `nenapu learn <path>` is the human-facing and
    worker-facing form; routing it through the queue would make the queue
    its own consumer."""
    from typer.testing import CliRunner

    from nenapu.cli import app

    transcript = _transcript(tmp_path)
    with patch("nenapu.observer.observe_transcript", return_value=[]) as observe:
        result = CliRunner().invoke(app, ["learn", str(transcript), "--db", str(db)])

    assert result.exit_code == 0, result.output
    observe.assert_called_once()
    assert _queue_rows(db) == []


def test_no_infer_neither_queues_nor_calls_the_model(tmp_path, db):
    """Green pin (task 16). `--no-infer` is the cheap inspection path; it
    must not acquire a model call by way of the queue."""
    from typer.testing import CliRunner

    from nenapu.cli import app

    transcript = _transcript(tmp_path)
    with patch("nenapu.llm.structured") as structured:
        result = CliRunner().invoke(
            app, ["learn", str(transcript), "--no-infer", "--db", str(db)]
        )

    assert result.exit_code == 0, result.output
    structured.assert_not_called()
    assert _queue_rows(db) == []


def test_a_hook_with_no_transcript_path_queues_nothing_and_exits_zero(db):
    from typer.testing import CliRunner

    from nenapu.cli import app

    result = CliRunner().invoke(
        app, ["learn", "--stdin", "--detach", "--db", str(db)],
        input=json.dumps({"session_id": "s-1"}),
    )

    assert result.exit_code == 0
    assert _queue_rows(db) == []


def test_a_broken_store_never_breaks_the_session(tmp_path):
    """A hook must fail silently and exit 0 whatever happens underneath it —
    the queue is one more thing that can be unwritable."""
    from typer.testing import CliRunner

    from nenapu.cli import app

    unwritable = tmp_path / "dir-not-a-db"
    unwritable.mkdir()
    with patch("subprocess.Popen"):
        result = CliRunner().invoke(
            app, ["learn", "--stdin", "--detach", "--db", str(unwritable)],
            input=_payload(_transcript(tmp_path)),
        )

    assert result.exit_code == 0
