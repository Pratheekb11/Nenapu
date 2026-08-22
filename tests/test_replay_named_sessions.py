"""Replaying the sessions you meant, rather than the ones that are recent.

Requirement (plan "Harden the four incidents into guarantees", Phase E task
E1, marked **Sonnet 5**):

    "`grade --replay` queued 52 sessions when you asked for 7."

`--limit` fixed the cost and `2bfb51a` landed it: the bound counts sessions
actually queued, newest first, and the default stays unbounded. What it did
not fix is the aim. `--limit` and `--since` both bound by recency, and the
request that started this was "the seven that failed" — a set that recency
cannot describe. `_sessions_with_pending_recalls` hard-codes
`outcome = 'pending'` and offers no way to name a session at all, so the only
route to seven specific sessions is to queue everything newer than them.

Naming them is the missing axis. The ids go through the same `enqueue_once`
dedupe, so a repeat is still a no-op and a session already waiting is not
queued twice.

Scope boundary with `tests/test_grading_loop.py`
------------------------------------------------
That file pins the unbounded replay and the `--limit` bound. Nothing here
re-tests either.
"""

import os
import subprocess
import sys
import time

import pytest

from nenapu import Store, connect
from nenapu.cli import replay_pending_sessions
from nenapu.models import Fact
from nenapu.store import DAY


@pytest.fixture
def store():
    return Store(connect(":memory:"))


def _root(tmp_path, sessions):
    root = tmp_path / "projects"
    (root / "a-project").mkdir(parents=True)
    for session_id in sessions:
        (root / "a-project" / f"{session_id}.jsonl").write_text(
            '{"type":"user","sessionId":"%s","message":{"role":"user",'
            '"content":[{"type":"text","text":"the deploy path"}]}}' % session_id
        )
    return root


def _pending_recall(store, text, *, session_id, days_ago=1.0):
    fact, _ = store.write(Fact(text=text))
    recall_id = store.ledger.log(fact.id, session_id=session_id)
    store.conn.execute("UPDATE recalls SET created_at = ? WHERE id = ?",
                       (time.time() - days_ago * DAY, recall_id))
    store.conn.commit()
    return recall_id


def _queued(store):
    return [r["session_id"] for r in store.conn.execute(
        "SELECT session_id FROM ingest_queue")]


def test_naming_sessions_queues_those_and_nothing_else(store, tmp_path):
    """The seven that failed, not everything newer than them."""
    for name in ("s-a", "s-b", "s-c"):
        _pending_recall(store, f"fact for {name}", session_id=name)
    root = _root(tmp_path, ["s-a", "s-b", "s-c"])

    queued = replay_pending_sessions(store, sessions=["s-a", "s-c"],
                                     transcripts_root=root)

    assert sorted(queued) == ["s-a", "s-c"]
    assert sorted(_queued(store)) == ["s-a", "s-c"]


def test_a_named_session_is_queued_even_though_it_is_the_oldest(store, tmp_path):
    """The point of naming: recency is exactly the wrong axis here."""
    _pending_recall(store, "old", session_id="s-old", days_ago=90)
    _pending_recall(store, "new", session_id="s-new", days_ago=1)
    root = _root(tmp_path, ["s-old", "s-new"])

    queued = replay_pending_sessions(store, sessions=["s-old"], transcripts_root=root)

    assert queued == ["s-old"]


def test_naming_a_session_twice_queues_it_once(store, tmp_path):
    """Same `enqueue_once` dedupe the unbounded replay relies on."""
    _pending_recall(store, "one", session_id="s-a")
    root = _root(tmp_path, ["s-a"])

    queued = replay_pending_sessions(store, sessions=["s-a", "s-a"],
                                     transcripts_root=root)

    assert queued == ["s-a"]
    assert len(_queued(store)) == 1


def test_a_named_session_with_no_transcript_is_skipped(store, tmp_path):
    """A session whose transcript is gone cannot be replayed, which is not an
    error: the same rule the unbounded replay already follows."""
    _pending_recall(store, "one", session_id="s-a")
    root = _root(tmp_path, ["s-a"])

    queued = replay_pending_sessions(store, sessions=["s-a", "s-missing"],
                                     transcripts_root=root)

    assert queued == ["s-a"]


def test_a_named_session_with_nothing_pending_is_still_replayed(store, tmp_path):
    """Naming is an instruction, not a filter to be second-guessed. A session
    whose recalls were all graded by some other signal can still be worth
    reading again, and the caller said so."""
    root = _root(tmp_path, ["s-quiet"])

    queued = replay_pending_sessions(store, sessions=["s-quiet"], transcripts_root=root)

    assert queued == ["s-quiet"]


def test_naming_nothing_still_replays_the_backlog(store, tmp_path):
    """The default must not change: replaying the whole backlog is what the
    command is for."""
    _pending_recall(store, "one", session_id="s-a")
    _pending_recall(store, "two", session_id="s-b")
    root = _root(tmp_path, ["s-a", "s-b"])

    queued = replay_pending_sessions(store, transcripts_root=root)

    assert sorted(queued) == ["s-a", "s-b"]


def test_grade_takes_session_ids_on_the_command_line(tmp_path):
    result = subprocess.run(
        [sys.executable, "-m", "nenapu.cli", "grade", "--replay", "s-a", "s-b",
         "--db", str(tmp_path / "s.db")],
        capture_output=True, text=True,
        env={**os.environ, "PYTHONPATH": "src", "NENAPU_NO_BANNER": "1"},
    )

    assert result.returncode == 0, result.stderr
    assert "queued" in result.stdout


def test_grading_one_session_by_hand_still_works(tmp_path):
    """The argument carries both meanings now, so the single-session form has
    to keep working exactly as it did."""
    db = tmp_path / "s.db"

    def _run(args):
        return subprocess.run(
            [sys.executable, "-m", "nenapu.cli", *args, "--db", str(db)],
            capture_output=True, text=True,
            env={**os.environ, "PYTHONPATH": "src", "NENAPU_NO_BANNER": "1"},
        )

    _run(["remember", "the staging bucket is the export target"])
    result = _run(["grade", "s-one", "--success"])

    assert result.returncode == 0, result.stderr


def test_grading_by_hand_refuses_more_than_one_session(tmp_path):
    """Two ids and one verdict is ambiguous. Only `--replay` takes a set."""
    result = subprocess.run(
        [sys.executable, "-m", "nenapu.cli", "grade", "s-one", "s-two", "--success",
         "--db", str(tmp_path / "s.db")],
        capture_output=True, text=True,
        env={**os.environ, "PYTHONPATH": "src", "NENAPU_NO_BANNER": "1"},
    )

    assert result.returncode != 0
