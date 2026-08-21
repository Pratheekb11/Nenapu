"""What `worker._ingest` and `worker.drain` must recover on the second pass.

Three findings from the review of the queue routing work, all of them about
a job that carries less than the first, happy job did:

1. `capture_session` returns `None` for a session it has already recorded —
   a `claude --resume`, or a retry after a model backend outage, since
   `capture_session` calls `end_session` before the extraction runs. Today
   `_ingest` reads that `None` as "no session at all" and falls back to
   `scope="global"`, so every fact a resumed session teaches is stored where
   every other project on the machine will recall it. That is the bug
   project scoping was built to end.

2. The watcher enqueues with no `session_id` (`watch.tick` calls `enqueue`
   without one), so facts learned from a watcher-queued session land with
   `fact.session_id = NULL`. `Graph.infer_edges_for` returns `[]` on a null
   session id, which silently disables inferred edges for the whole Codex
   path and for any Claude Code machine without the Stop hook. The session
   id is on disk in the transcript and in the ledger row `capture_session`
   just wrote; `_ingest` has to go get it.

3. `drain` runs `run_maintenance_tick` after its loop has already broken but
   while it still holds `WorkerLock`. A hook firing in that window enqueues
   fine and spawns a worker that finds the lock taken and returns, and the
   worker holding the lock never looks at the queue again — so the job sits
   pending until some later, unrelated trigger.
"""

import json
from unittest.mock import patch

import pytest

from nenapu import connect
from nenapu.ingest_queue import enqueue
from nenapu.store import Store, project_scope
from nenapu.worker import drain


CWD = "/repo"


def _turn(role: str, text: str, session: str = "s-1") -> str:
    return json.dumps({
        "type": role, "sessionId": session, "cwd": CWD,
        "message": {"role": role, "content": [{"type": "text", "text": text}]},
    })


def _transcript(tmp_path, name="s-1.jsonl", session="s-1"):
    path = tmp_path / name
    path.write_text("\n".join([
        _turn("user", "use pnpm not npm", session),
        _turn("assistant", "understood, pnpm from here", session),
    ]))
    return path


def _queue_rows(db):
    return [dict(r) for r in connect(str(db)).execute(
        "SELECT * FROM ingest_queue ORDER BY id"
    )]


@pytest.fixture
def db(tmp_path):
    return tmp_path / "s.db"


@pytest.fixture
def store(db):
    return Store(connect(str(db)))


def _drain_capturing(store, tmp_path):
    """Drain once with the extraction stubbed, returning the stub so the test
    can read the scope, cwd and session id `_ingest` decided on."""
    with patch("nenapu.worker.observe_transcript", return_value=[]) as observe:
        drain(store, lock_path=tmp_path / "worker.lock")
    return observe


# ---------- 1. a session already in the ledger keeps its scope ----------


def test_a_resumed_session_is_extracted_into_its_project_scope(tmp_path, store, db):
    """The second job for one session must not land in `global`."""
    transcript = _transcript(tmp_path)
    enqueue(store.conn, path=str(transcript), agent="claude-code", session_id="s-1")
    _drain_capturing(store, tmp_path)

    transcript.write_text(transcript.read_text() + "\n" + _turn("user", "and use uv"))
    enqueue(store.conn, path=str(transcript), agent="claude-code", session_id="s-1")
    observe = _drain_capturing(store, tmp_path)

    assert observe.call_args.kwargs["scope"] == project_scope(CWD)
    assert observe.call_args.kwargs["scope"] != "global"


def test_a_resumed_session_is_extracted_with_its_own_cwd(tmp_path, store, db):
    """`observe_transcript` uses `cwd` for the scope of anything it has to
    scope itself, so dropping it re-opens the same hole one level down."""
    transcript = _transcript(tmp_path)
    enqueue(store.conn, path=str(transcript), agent="claude-code", session_id="s-1")
    _drain_capturing(store, tmp_path)

    enqueue(store.conn, path=str(transcript), agent="claude-code", session_id="s-1")
    observe = _drain_capturing(store, tmp_path)

    assert observe.call_args.kwargs["cwd"] == CWD


def test_a_retry_after_a_failed_extraction_keeps_the_scope(tmp_path, store, db):
    """`capture_session` ends the session row before the extraction runs, so
    a backend outage leaves exactly the state a resume does."""
    from nenapu.observer import LLMUnavailable

    transcript = _transcript(tmp_path)
    enqueue(store.conn, path=str(transcript), agent="claude-code", session_id="s-1")
    with patch("nenapu.worker.observe_transcript", side_effect=LLMUnavailable("down")):
        drain(store, lock_path=tmp_path / "worker.lock")
    assert _queue_rows(db)[0]["state"] == "failed"

    enqueue(store.conn, path=str(transcript), agent="claude-code", session_id="s-1")
    observe = _drain_capturing(store, tmp_path)

    assert observe.call_args.kwargs["scope"] == project_scope(CWD)


def test_the_scope_a_resumed_session_returns_is_the_project_one(tmp_path, store, db):
    """`_ingest` returns the scope it used, and `drain` hands that to the
    maintenance tick as a touched scope — a `global` answer here would spend
    the tick on the wrong scope as well as storing the facts in it."""
    transcript = _transcript(tmp_path)
    enqueue(store.conn, path=str(transcript), agent="claude-code", session_id="s-1")
    _drain_capturing(store, tmp_path)
    enqueue(store.conn, path=str(transcript), agent="claude-code", session_id="s-1")

    seen = []
    with patch("nenapu.worker.observe_transcript", return_value=[]), patch(
        "nenapu.worker.run_maintenance_tick",
        side_effect=lambda _store, touched_scopes=(): seen.append(list(touched_scopes)),
    ):
        drain(store, lock_path=tmp_path / "worker.lock")

    assert seen and seen[0] == [project_scope(CWD)]


# ---------- 2. a watcher-queued job recovers the session id ----------


def test_a_watcher_queued_job_extracts_with_the_transcripts_session_id(tmp_path, store):
    """`watch.tick` enqueues with no session id. Without recovery every fact
    it teaches gets `session_id = NULL`, which turns off inferred edges."""
    transcript = _transcript(tmp_path, session="s-watched")
    enqueue(store.conn, path=str(transcript), agent="claude-code")

    observe = _drain_capturing(store, tmp_path)

    assert observe.call_args.kwargs["session_id"] == "s-watched"


def test_a_watcher_queued_resume_still_recovers_the_session_id(tmp_path, store):
    """Both gaps at once: no session id on the job *and* `capture_session`
    returning `None` because the session is already in the ledger. The
    transcript still says who it is."""
    transcript = _transcript(tmp_path, session="s-watched")
    enqueue(store.conn, path=str(transcript), agent="claude-code")
    _drain_capturing(store, tmp_path)

    enqueue(store.conn, path=str(transcript), agent="claude-code")
    observe = _drain_capturing(store, tmp_path)

    assert observe.call_args.kwargs["session_id"] == "s-watched"
    assert observe.call_args.kwargs["scope"] == project_scope(CWD)


def test_the_jobs_own_session_id_still_wins(tmp_path, store):
    """The hook passes the id the harness reported; recovery is a fallback,
    not a replacement."""
    transcript = _transcript(tmp_path, session="s-1")
    enqueue(store.conn, path=str(transcript), agent="claude-code", session_id="s-1")

    observe = _drain_capturing(store, tmp_path)

    assert observe.call_args.kwargs["session_id"] == "s-1"


def test_a_transcript_naming_no_session_extracts_with_none(tmp_path, store):
    """Recovery must not invent an id: a transcript with no session id is
    still extractable, just without edges to tie it to."""
    path = tmp_path / "anon.jsonl"
    path.write_text(json.dumps({
        "type": "user", "cwd": CWD,
        "message": {"role": "user", "content": [{"type": "text", "text": "hi"}]},
    }))
    enqueue(store.conn, path=str(path), agent="claude-code")

    observe = _drain_capturing(store, tmp_path)

    assert observe.call_args.kwargs["session_id"] is None


def test_facts_from_a_watcher_queued_session_carry_the_session_id(tmp_path, store):
    """The end the recovery is for: `Graph.infer_edges_for` returns nothing
    on a null session id, so a null here is the whole edge mechanism off."""
    from nenapu.models import Fact, Kind

    transcript = _transcript(tmp_path, session="s-watched")
    enqueue(store.conn, path=str(transcript), agent="claude-code")

    def fake_extract(store_, path_, *, session_id=None, scope=None, cwd=None, parse=None):
        store_.write(Fact(
            text="use pnpm, not npm", kind=Kind.PROJECT, scope=scope,
            confidence=0.9, session_id=session_id, origin_ref=f"session {session_id}",
        ))
        return []

    with patch("nenapu.worker.observe_transcript", side_effect=fake_extract):
        drain(store, lock_path=tmp_path / "worker.lock")

    row = store.conn.execute("SELECT session_id FROM facts LIMIT 1").fetchone()
    assert row["session_id"] == "s-watched"


# ---------- 3. a job arriving during maintenance is not orphaned ----------


def test_a_job_enqueued_during_maintenance_is_drained_by_the_same_worker(tmp_path, store, db):
    """The lock-holder is the only one that can drain, so it has to look
    again before letting go of it."""
    first = _transcript(tmp_path, name="a.jsonl", session="s-a")
    second = _transcript(tmp_path, name="b.jsonl", session="s-b")
    enqueue(store.conn, path=str(first), agent="claude-code", session_id="s-a")

    ticks = []

    def tick(_store, touched_scopes=()):
        ticks.append(list(touched_scopes))
        if len(ticks) == 1:  # a hook fires while upkeep is still running
            enqueue(store.conn, path=str(second), agent="claude-code", session_id="s-b")

    with patch("nenapu.worker.observe_transcript", return_value=[]), patch(
        "nenapu.worker.run_maintenance_tick", side_effect=tick
    ):
        processed = drain(store, lock_path=tmp_path / "worker.lock")

    assert processed == 2
    assert [r["state"] for r in _queue_rows(db)] == ["done", "done"]


def test_the_maintenance_tick_still_runs_once_when_nothing_arrives(tmp_path, store, db):
    """Re-checking the queue must not turn one drain into repeated upkeep."""
    enqueue(store.conn, path=str(_transcript(tmp_path)), agent="claude-code",
            session_id="s-1")

    ticks = []
    with patch("nenapu.worker.observe_transcript", return_value=[]), patch(
        "nenapu.worker.run_maintenance_tick",
        side_effect=lambda _store, touched_scopes=(): ticks.append(1),
    ):
        drain(store, lock_path=tmp_path / "worker.lock")

    assert len(ticks) == 1


def test_the_limit_still_bounds_one_drain(tmp_path, store, db):
    """The re-check must not become a way around `limit` — a worker that
    never stops while jobs keep arriving is the fan-out again."""
    for i in range(4):
        enqueue(store.conn, path=str(_transcript(tmp_path, name=f"{i}.jsonl",
                                                 session=f"s-{i}")),
                agent="claude-code", session_id=f"s-{i}")

    with patch("nenapu.worker.observe_transcript", return_value=[]):
        processed = drain(store, lock_path=tmp_path / "worker.lock", limit=2)

    assert processed == 2
    assert [r["state"] for r in _queue_rows(db)].count("pending") == 2
