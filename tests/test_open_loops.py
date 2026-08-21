"""Open loops: "hey, you missed implementing X", and knowing when not to say it.

Requirement (Task 11, priority-ordered task list, marked **Opus 5** — "the
hardest design here: closure heuristics, bias-toward-closing, false-nag
avoidance"):

    "Open loops + automatic closure (incl. `interrupted` from abrupt stops) |
    'hey, you missed implementing X' — the headline feature | L | Opus 5 |
    depends on 3, 9"

From "Open loops: the mechanism behind 'you missed implementing X'":

    **Capture.** "The user has said they will not file todos by hand — that
    is the whole premise — so this must come from the transcript."

    **Closure.** "An open loop carries a `resolution_hint`: a path glob, a
    `key`, or a verify command. It is closed automatically when the activity
    ledger shows it satisfied — a commit touching the path, a file created, a
    check passing."

    **Ageing.** "An open loop in a project untouched for three months should
    go quiet, not shout. Reuse `decay_class` and the belief floor... a loop
    below the injection threshold stops being surfaced but stays queryable
    via `nenapu pending --all`."

    **Risk.** "False nagging is fatal, silence is survivable... closure
    detection must be *biased toward closing*: any plausible evidence of
    completion closes the loop... Prefer a missed reminder over a wrong one."

And from "Abrupt stops — deterministic and free":

    "`git_head_before == git_head_after` **and** `file_events` non-empty →
    files were modified and never committed. Work in flight... Either
    condition raises an open loop of kind `interrupted`, with the file list
    attached."

Proposed seam
-------------
A new table and a small class, `nenapu.loops.LoopBook`, rather than a new
`Kind` on `facts`. The plan offers both ("New `Kind.OPEN_LOOP` (or a
`pending` status)"); a loop needs `resolution_hint`, `status`, `closed_at`
and `close_reason`, which are four columns no fact wants, while the ageing
rule it does reuse — `decay_class` and the belief floor — is a function, not
a table.

    open_loop(*, scope, text, resolution_hint=None, session_id=None,
              kind="stated") -> int
    close_loop(loop_id, *, reason) -> bool
    open_for_scope(scope, *, include_quiet=False) -> list[dict]
    all_open(*, include_quiet=False) -> list[dict]
    close_satisfied(ledger, *, scope=None) -> list[int]
    detect_interrupted(ledger, session_id) -> int | None

Rendering into the injected block is Task 7's contract and is tested in
`tests/test_project_injection.py`.

Remove the `pytestmark` line below when Task 11 lands.
"""

import json
import os
import subprocess
import sys

import pytest

from nenapu import connect
from nenapu.models import now
from nenapu.store import Store

pytestmark = pytest.mark.xfail(
    reason="Task 11 (Opus 5) not implemented — tests written first", strict=False
)

DAY = 86400.0
SCOPE = "repo:backend@aaaaaaaa"
OTHER = "repo:portfolio@bbbbbbbb"


class FakeBackend:
    name = "fake"
    model = "fake"
    supports_schema = False


@pytest.fixture
def store():
    return Store(connect(":memory:"))


@pytest.fixture
def book(store):
    from nenapu.loops import LoopBook

    return LoopBook(store.conn)


@pytest.fixture
def ledger(store):
    from nenapu.activity import ActivityLedger

    return ActivityLedger(store.conn)


# ---------- storage ----------


def test_the_open_loops_table_exists_on_a_new_store():
    conn = connect(":memory:")
    tables = {r["name"] for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type = 'table'")}
    assert "open_loops" in tables


def test_a_loop_records_the_evidence_it_would_be_closed_by():
    conn = connect(":memory:")
    columns = {r["name"] for r in conn.execute("PRAGMA table_info(open_loops)")}
    assert columns >= {
        "id", "scope", "text", "resolution_hint", "kind", "status",
        "opened_at", "closed_at", "close_reason", "session_id",
    }


def test_opening_and_listing_a_loop_round_trips(book):
    book.open_loop(scope=SCOPE, text="Rate limiting on the availability endpoint",
                   resolution_hint="backend/app/ratelimit*")

    loops = book.open_for_scope(SCOPE)

    assert [loop["text"] for loop in loops] == ["Rate limiting on the availability endpoint"]
    assert loops[0]["resolution_hint"] == "backend/app/ratelimit*"


def test_loops_are_scoped_to_their_project(book):
    book.open_loop(scope=SCOPE, text="Rate limiting")
    book.open_loop(scope=OTHER, text="Hero animation")

    assert [loop["text"] for loop in book.open_for_scope(SCOPE)] == ["Rate limiting"]
    assert len(book.all_open()) == 2


def test_closing_a_loop_removes_it_from_what_is_open_and_keeps_why(book):
    """Nothing in this project is deleted; a closed loop is the record that
    the reminder was retired for a reason."""
    loop_id = book.open_loop(scope=SCOPE, text="Rate limiting")

    book.close_loop(loop_id, reason="commit c7f1a9d4 touched backend/app/ratelimit.py")

    assert book.open_for_scope(SCOPE) == []
    closed = book.get(loop_id)
    assert closed["status"] == "closed"
    assert "ratelimit.py" in closed["close_reason"]
    assert closed["closed_at"] is not None


def test_closing_twice_is_harmless(book):
    loop_id = book.open_loop(scope=SCOPE, text="Rate limiting")
    book.close_loop(loop_id, reason="first")

    assert book.close_loop(loop_id, reason="second") is False
    assert "first" in book.get(loop_id)["close_reason"]


# ---------- closure from the activity ledger ----------


def _session(ledger, *, scope=SCOPE, ago=0.0, head_before="a" * 40, head_after="b" * 40):
    session_id = ledger.start_session(
        agent="claude-code", project_scope=scope, cwd="/repo", git_branch="main",
        git_head_before=head_before, started_at=now() - ago,
    )
    ledger.end_session(session_id, git_head_after=head_after, ended_at=now() - ago + 60)
    return session_id


def test_a_commit_touching_the_hinted_path_closes_the_loop(book, ledger):
    loop_id = book.open_loop(scope=SCOPE, text="Rate limiting on the endpoint",
                             resolution_hint="backend/app/ratelimit*")
    session_id = _session(ledger)
    ledger.record_commit(session_id, sha="c7f1a9d4e2", subject="Add rate limiting",
                         files_changed=["backend/app/ratelimit.py"], at=now())

    closed = book.close_satisfied(ledger, scope=SCOPE)

    assert closed == [loop_id]
    assert book.open_for_scope(SCOPE) == []


def test_a_created_file_matching_the_hint_closes_the_loop(book, ledger):
    """"a commit touching the path, a file created" — a session that wrote the
    file but has not committed yet is still evidence the work happened."""
    loop_id = book.open_loop(scope=SCOPE, text="Rate limiting",
                             resolution_hint="backend/app/ratelimit*")
    session_id = _session(ledger)
    ledger.record_file_event(session_id, path="backend/app/ratelimit.py", op="created",
                             tool="Write", at=now())

    assert book.close_satisfied(ledger, scope=SCOPE) == [loop_id]


def test_merely_reading_the_file_does_not_close_the_loop(book, ledger):
    """The bias is toward closing, not toward pretending. Opening a file to
    look at it is the most common thing a session does."""
    book.open_loop(scope=SCOPE, text="Rate limiting", resolution_hint="backend/app/ratelimit*")
    session_id = _session(ledger)
    ledger.record_file_event(session_id, path="backend/app/ratelimit.py", op="read",
                             tool="Read", at=now())

    assert book.close_satisfied(ledger, scope=SCOPE) == []


def test_work_that_happened_before_the_loop_was_opened_does_not_close_it(book, ledger):
    """Otherwise every loop mentioned in a long session closes itself against
    that same session's earlier edits, and the feature is silently inert."""
    session_id = _session(ledger, ago=7 * DAY)
    ledger.record_file_event(session_id, path="backend/app/ratelimit.py", op="created",
                             tool="Write", at=now() - 7 * DAY)
    book.open_loop(scope=SCOPE, text="Rate limiting is still missing",
                   resolution_hint="backend/app/ratelimit*")

    assert book.close_satisfied(ledger, scope=SCOPE) == []


def test_another_projects_activity_does_not_close_this_projects_loop(book, ledger):
    book.open_loop(scope=SCOPE, text="Rate limiting", resolution_hint="*ratelimit*")
    session_id = _session(ledger, scope=OTHER)
    ledger.record_file_event(session_id, path="portfolio/ratelimit.py", op="created",
                             tool="Write", at=now())

    assert book.close_satisfied(ledger, scope=SCOPE) == []


def test_a_plausible_commit_subject_closes_a_loop_with_no_path_hint(book, ledger):
    """Bias toward closing, stated as a rule: a loop the extractor could not
    attach a path to still gets closed by work that plainly describes it.
    "Prefer a missed reminder over a wrong one.\""""
    loop_id = book.open_loop(scope=SCOPE, text="Add rate limiting to the availability endpoint")
    session_id = _session(ledger)
    ledger.record_commit(session_id, sha="deadbeef12",
                         subject="Rate limit the availability endpoint",
                         files_changed=["backend/app/limits.py"], at=now())

    assert book.close_satisfied(ledger, scope=SCOPE) == [loop_id]


def test_unrelated_work_leaves_the_loop_open(book, ledger):
    """The other half of the same rule: closing on any activity at all would
    make the whole mechanism a coin flip."""
    book.open_loop(scope=SCOPE, text="Add rate limiting to the availability endpoint")
    session_id = _session(ledger)
    ledger.record_commit(session_id, sha="deadbeef12", subject="Fix the pet drawing",
                         files_changed=["src/nenapu/pet_art.py"], at=now())

    assert book.close_satisfied(ledger, scope=SCOPE) == []


def test_closure_is_reported_once_and_not_re_reported(book, ledger):
    """`close_satisfied` runs on every maintenance tick. Returning the same
    loop forever would make anything built on its return value nag."""
    book.open_loop(scope=SCOPE, text="Rate limiting", resolution_hint="*ratelimit*")
    session_id = _session(ledger)
    ledger.record_file_event(session_id, path="backend/app/ratelimit.py", op="created",
                             tool="Write", at=now())

    first = book.close_satisfied(ledger, scope=SCOPE)
    second = book.close_satisfied(ledger, scope=SCOPE)

    assert len(first) == 1
    assert second == []


def test_the_closing_evidence_is_recorded(book, ledger):
    """"with the evidence for why they are still believed open" cuts both
    ways — a closure the user disagrees with must be inspectable."""
    loop_id = book.open_loop(scope=SCOPE, text="Rate limiting", resolution_hint="*ratelimit*")
    session_id = _session(ledger)
    ledger.record_commit(session_id, sha="c7f1a9d4e2", subject="Add rate limiting",
                         files_changed=["backend/app/ratelimit.py"], at=now())

    book.close_satisfied(ledger, scope=SCOPE)

    assert "c7f1a9d4" in book.get(loop_id)["close_reason"]


# ---------- abrupt stops ----------


def test_uncommitted_work_raises_an_interrupted_loop(book, ledger):
    session_id = _session(ledger, head_before="a" * 40, head_after="a" * 40)
    ledger.record_file_event(session_id, path="backend/app/bookings.py", op="edited",
                             tool="Edit", at=now())

    loop_id = book.detect_interrupted(ledger, session_id)

    loop = book.get(loop_id)
    assert loop["kind"] == "interrupted"
    assert "bookings.py" in loop["text"]
    assert loop["scope"] == SCOPE


def test_a_session_that_committed_is_not_interrupted(book, ledger):
    session_id = _session(ledger, head_before="a" * 40, head_after="b" * 40)
    ledger.record_file_event(session_id, path="backend/app/bookings.py", op="edited",
                             tool="Edit", at=now())

    assert book.detect_interrupted(ledger, session_id) is None


def test_a_session_that_changed_nothing_is_not_interrupted(book, ledger):
    """A session spent reading and answering questions left no work in
    flight, and saying otherwise on every such session is the noise that
    teaches the user to skip the block."""
    session_id = _session(ledger, head_before="a" * 40, head_after="a" * 40)
    ledger.record_file_event(session_id, path="README.md", op="read", tool="Read", at=now())

    assert book.detect_interrupted(ledger, session_id) is None


def test_interrupted_detection_is_idempotent(book, ledger):
    """The worker can revisit a session — a re-run backfill, a retried job.
    Two loops for one stop is two nags for one thing."""
    session_id = _session(ledger, head_before="a" * 40, head_after="a" * 40)
    ledger.record_file_event(session_id, path="backend/app/bookings.py", op="edited",
                             tool="Edit", at=now())

    book.detect_interrupted(ledger, session_id)
    book.detect_interrupted(ledger, session_id)

    assert len(book.open_for_scope(SCOPE)) == 1


def test_an_interrupted_loop_closes_when_the_files_are_committed(book, ledger):
    """The abrupt-stop loop must be closable by exactly the evidence that it
    was resumed, or every interrupted session becomes a permanent nag."""
    session_id = _session(ledger, head_before="a" * 40, head_after="a" * 40)
    ledger.record_file_event(session_id, path="backend/app/bookings.py", op="edited",
                             tool="Edit", at=now())
    book.detect_interrupted(ledger, session_id)

    later = _session(ledger, head_before="a" * 40, head_after="c" * 40)
    ledger.record_commit(later, sha="c0ffee1234", subject="Finish the booking work",
                         files_changed=["backend/app/bookings.py"], at=now() + 1)

    book.close_satisfied(ledger, scope=SCOPE)

    assert book.open_for_scope(SCOPE) == []


# ---------- ageing: go quiet, do not shout ----------


def test_an_old_loop_in_an_untouched_project_goes_quiet(book):
    """"An open loop in a project untouched for three months should go quiet,
    not shout." Below the floor it is not surfaced..."""
    book.open_loop(scope=SCOPE, text="Rate limiting", at=now() - 200 * DAY)

    assert book.open_for_scope(SCOPE) == []


def test_a_quiet_loop_is_still_queryable(book):
    """...but "stays queryable via `nenapu pending --all`"; nothing here is
    deleted, it is only de-prioritised."""
    book.open_loop(scope=SCOPE, text="Rate limiting", at=now() - 200 * DAY)

    assert len(book.open_for_scope(SCOPE, include_quiet=True)) == 1


def test_a_recent_loop_is_loud(book):
    book.open_loop(scope=SCOPE, text="Rate limiting", at=now() - 2 * DAY)

    assert len(book.open_for_scope(SCOPE)) == 1


# ---------- capture rides the existing extraction call ----------


def test_the_extraction_schema_carries_open_loops():
    """"Add `open_loop` to the extraction schema (riding the same call, per
    Addendum 2)" — not a second 83-second model call."""
    from nenapu.observer import EXTRACT_SCHEMA

    assert "open_loops" in EXTRACT_SCHEMA["properties"]


def test_a_loop_mentioned_in_a_session_is_stored(store, book, tmp_path, monkeypatch):
    def fake(prompt, schema, system=None, backend=None, max_tokens=None):
        return {"facts": [], "open_loops": [
            {"text": "Rate limiting on the public availability endpoint",
             "resolution_hint": "backend/app/ratelimit*"},
        ]}

    monkeypatch.setattr("nenapu.observer.structured", fake)
    from nenapu.observer import observe_transcript

    transcript = tmp_path / "t.jsonl"
    transcript.write_text("\n".join(json.dumps({
        "type": role, "message": {"role": role, "content": [{"type": "text", "text": text}]},
    }) for role, text in [
        ("user", "we still need rate limiting on the availability endpoint " * 8),
        ("assistant", "noted, not doing it now " * 30),
    ]))

    observe_transcript(store, transcript, session_id="s-1", backend=FakeBackend(),
                       scope=SCOPE)

    assert [loop["text"] for loop in book.open_for_scope(SCOPE)] == [
        "Rate limiting on the public availability endpoint"
    ]


def test_a_dry_run_opens_no_loops(store, book, tmp_path, monkeypatch):
    def fake(prompt, schema, system=None, backend=None, max_tokens=None):
        return {"facts": [], "open_loops": [{"text": "Rate limiting", "resolution_hint": ""}]}

    monkeypatch.setattr("nenapu.observer.structured", fake)
    from nenapu.observer import observe_transcript

    transcript = tmp_path / "t.jsonl"
    transcript.write_text("\n".join(json.dumps({
        "type": role, "message": {"role": role, "content": [{"type": "text", "text": text}]},
    }) for role, text in [("user", "x " * 200), ("assistant", "y " * 200)]))

    observe_transcript(store, transcript, backend=FakeBackend(), scope=SCOPE, apply=False)

    assert book.all_open(include_quiet=True) == []


# ---------- the command that was left reporting an honest zero ----------


def _run(args, db, cwd=None):
    env = {**os.environ, "PYTHONPATH": os.path.abspath("src"), "NENAPU_NO_BANNER": "1"}
    return subprocess.run(
        [sys.executable, "-m", "nenapu.cli", *args, "--db", str(db)],
        capture_output=True, text=True, env=env, cwd=cwd,
    )


def test_pending_lists_open_loops_across_projects(tmp_path):
    """`cli.pending` currently prints "No open loops tracked yet" because
    there was nothing to list. This is the task that gives it content."""
    from nenapu.loops import LoopBook

    db = tmp_path / "s.db"
    book = LoopBook(connect(str(db)))
    book.open_loop(scope=SCOPE, text="Rate limiting on the availability endpoint")
    book.open_loop(scope=OTHER, text="Hero animation on the portfolio")

    result = _run(["pending"], db)

    assert result.returncode == 0
    assert "Rate limiting" in result.stdout
    assert "Hero animation" in result.stdout


def test_pending_filters_to_one_project(tmp_path):
    from nenapu.loops import LoopBook

    db = tmp_path / "s.db"
    book = LoopBook(connect(str(db)))
    book.open_loop(scope=SCOPE, text="Rate limiting on the availability endpoint")
    book.open_loop(scope=OTHER, text="Hero animation on the portfolio")

    result = _run(["pending", "--project", "backend"], db)

    assert "Rate limiting" in result.stdout
    assert "Hero animation" not in result.stdout


def test_pending_all_shows_the_quiet_ones(tmp_path):
    from nenapu.loops import LoopBook

    db = tmp_path / "s.db"
    book = LoopBook(connect(str(db)))
    book.open_loop(scope=SCOPE, text="Rate limiting", at=now() - 200 * DAY)

    assert "Rate limiting" not in _run(["pending"], db).stdout
    assert "Rate limiting" in _run(["pending", "--all"], db).stdout
