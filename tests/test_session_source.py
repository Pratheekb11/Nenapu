"""Which sessions a repair may touch, recorded rather than guessed.

Requirement (plan "Harden the four incidents into guarantees", Phase D,
tasks D1 and D2, marked **Opus** for the migration):

    `--redate` must stop mis-targeting live sessions that ran outside a git
    repo.

`redate_backfilled_sessions` moves a session onto the clock its transcript
carries, and skips rows the hook recorded because those carry a start of their
own. It decides which is which by asking whether `git_head_before` is set —
a stand-in, and a leaky one. `capture.open_session` writes
`git_head_before=git_head(cwd) if cwd else None`, so a live session that ran
outside a git repo, or with no cwd, has NULL there. The repair reads that as
"backfilled", moves the row onto transcript timestamps and stamps `ended_at`
onto it as well.

The fix is to record the answer instead of inferring it: `sessions.source` says
whether a row was watched as it ran or reconstructed from history, and the
repair targets the reconstructed ones. Rows written before the column existed
have no answer to give, so the old heuristic stays as the fallback for those
and only those.

Scope boundary with `tests/test_backfill_command.py`
----------------------------------------------------
That file drives `backfill --redate` through the CLI and pins what it prints.
This file pins the column, both branches of the targeting rule, and the library
functions the repair is built from, which until now were reachable only through
the command.
"""

import pytest

from nenapu import connect
from nenapu.activity import ActivityLedger
from nenapu.backfill import _already_dated, redate_backfilled_sessions
from nenapu.capture import open_session

# The one timestamp every transcript below carries, and what a repaired row
# must end up holding. The mis-dated stamp is deliberately a different year, so
# "it moved" and "it moved to the right place" are not the same assertion.
TRANSCRIPT_AT = 1_782_900_000.0  # 2026-07-01T10:00:00Z
MIS_DATED_AT = 1_700_500_000.0   # the afternoon an earlier backfill ran


@pytest.fixture
def ledger(tmp_path):
    return ActivityLedger(connect(str(tmp_path / "s.db")))


def _column_names(ledger) -> set[str]:
    return {r["name"] for r in ledger.conn.execute("PRAGMA table_info(sessions)")}


# ---------- D1: the column, and what writes it ----------


def test_sessions_record_where_they_came_from(ledger):
    assert "source" in _column_names(ledger)


def test_a_backfilled_session_says_so(ledger, tmp_path):
    from nenapu.backfill import backfill_transcript

    transcript = tmp_path / "s-old.jsonl"
    transcript.write_text(
        '{"type":"user","sessionId":"s-old","cwd":"/repo",'
        '"timestamp":"2026-07-01T10:00:00Z",'
        '"message":{"role":"user","content":[{"type":"text","text":"hello"}]}}'
    )

    backfill_transcript(ledger, transcript, agent="claude-code")

    assert ledger.get_session("s-old")["source"] == "backfill"


def test_a_watched_session_says_so(ledger, tmp_path):
    """`open_session` is what the SessionStart hook calls."""
    row_id = open_session(ledger, agent="claude-code", cwd=str(tmp_path),
                          external_id="s-live")

    assert ledger.get_session(row_id)["source"] == "hook"


# ---------- D1: what the repair may touch ----------


def _mis_dated(ledger, external_id: str, *, source: str | None,
               git_head_before: str | None) -> int:
    """A row stamped with the moment a backfill ran rather than the session."""
    row_id = ledger.start_session(
        agent="claude-code", project_scope="repo", cwd="/repo",
        started_at=MIS_DATED_AT, external_id=external_id,
        git_head_before=git_head_before,
    )
    ledger.conn.execute("UPDATE sessions SET source = ? WHERE id = ?", (source, row_id))
    ledger.conn.commit()
    return row_id


@pytest.fixture
def transcripts(tmp_path):
    """One transcript per session, on a clock of their own."""
    root = tmp_path / "projects"
    root.mkdir()

    def _write(external_id: str) -> None:
        (root / f"{external_id}.jsonl").write_text(
            '{"type":"user","sessionId":"%s","cwd":"/repo",'
            '"timestamp":"2026-07-01T10:00:00Z",'
            '"message":{"role":"user","content":[{"type":"text","text":"hi"}]}}'
            % external_id
        )

    _write.root = root
    return _write


def _glob(transcripts) -> str:
    return str(transcripts.root / "*.jsonl")


def test_a_backfilled_row_is_repaired(ledger, transcripts):
    transcripts("s-back")
    row_id = _mis_dated(ledger, "s-back", source="backfill", git_head_before=None)

    moved = redate_backfilled_sessions(ledger, _glob(transcripts))

    assert moved == 1
    assert ledger.get_session(row_id)["started_at"] == TRANSCRIPT_AT


def test_a_watched_row_outside_a_git_repo_is_left_alone(ledger, transcripts):
    """The bug this task exists for: no `git_head_before` because there was no
    repo to read one from, not because the row was reconstructed."""
    transcripts("s-nogit")
    row_id = _mis_dated(ledger, "s-nogit", source="hook", git_head_before=None)
    before = ledger.get_session(row_id)

    moved = redate_backfilled_sessions(ledger, _glob(transcripts))

    assert moved == 0
    assert ledger.get_session(row_id)["started_at"] == before["started_at"]
    assert ledger.get_session(row_id)["ended_at"] == before["ended_at"]


def test_a_row_written_before_the_column_existed_falls_back_to_git(ledger, transcripts):
    """An old store has no answer to give, so the previous rule still decides
    for those rows and only those."""
    transcripts("s-legacy-hook")
    transcripts("s-legacy-backfill")
    watched = _mis_dated(ledger, "s-legacy-hook", source=None,
                         git_head_before="a" * 40)
    reconstructed = _mis_dated(ledger, "s-legacy-backfill", source=None,
                               git_head_before=None)
    watched_before = ledger.get_session(watched)["started_at"]

    moved = redate_backfilled_sessions(ledger, _glob(transcripts))

    assert moved == 1
    assert ledger.get_session(watched)["started_at"] == watched_before
    assert ledger.get_session(reconstructed)["started_at"] == TRANSCRIPT_AT


# ---------- D2: the repair path, tested directly ----------


def test_a_dry_run_counts_what_it_would_move_and_moves_nothing(ledger, transcripts):
    transcripts("s-back")
    row_id = _mis_dated(ledger, "s-back", source="backfill", git_head_before=None)
    before = ledger.get_session(row_id)["started_at"]

    moved = redate_backfilled_sessions(ledger, _glob(transcripts), apply=False)

    assert moved == 1, "a dry run must report the same count a real run would"
    assert ledger.get_session(row_id)["started_at"] == before


def test_repairing_twice_moves_nothing_the_second_time(ledger, transcripts):
    transcripts("s-back")
    _mis_dated(ledger, "s-back", source="backfill", git_head_before=None)

    redate_backfilled_sessions(ledger, _glob(transcripts))

    assert redate_backfilled_sessions(ledger, _glob(transcripts)) == 0


def test_a_transcript_with_no_row_is_not_ingested(ledger, transcripts):
    """A repair repairs. A transcript with no row yet is a plain backfill's job."""
    transcripts("s-unknown")

    moved = redate_backfilled_sessions(ledger, _glob(transcripts))

    assert moved == 0
    assert ledger.get_session("s-unknown") is None


def test_a_row_already_on_its_own_clock_is_left_alone():
    """The tolerance exists so a re-run is a no-op rather than a rewrite of the
    same value with a slightly different float."""
    row = {"started_at": TRANSCRIPT_AT, "ended_at": TRANSCRIPT_AT + 60}

    assert _already_dated(row, TRANSCRIPT_AT + 0.5, TRANSCRIPT_AT + 60.5)
    assert not _already_dated(row, TRANSCRIPT_AT + 5.0, TRANSCRIPT_AT + 60.0)


def test_an_ended_at_the_transcript_cannot_say_is_not_invented(ledger):
    """`redate_session` leaves `ended_at` alone when there is nothing to put
    there, rather than stamping the row as finished now."""
    row_id = ledger.start_session(agent="claude-code", project_scope="repo",
                                  started_at=MIS_DATED_AT, external_id="s-x")

    ledger.redate_session(row_id, started_at=TRANSCRIPT_AT)

    assert ledger.get_session(row_id)["started_at"] == TRANSCRIPT_AT
    assert ledger.get_session(row_id)["ended_at"] is None
