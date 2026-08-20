"""Downsampling the activity ledger by age — the "compressed cache" for activity.

Requirement (Task 13, priority-ordered task list): "Activity rollup /
downsampling (14d -> weekly -> monthly) | the 'compressed cache'; keeps
`standup` readable." Depends on Task 3 (the ledger tables) and Task 12 (the
maintenance tick this rides on).

Distinguished explicitly from fact distillation (which compresses by
*similarity*): a work log needs compressing by *age*.

    0-14 days    every file_event, every session, full detail
    14-90 days   one rollup row per project per week
                    (sessions, files touched, commits, loops opened/closed)
    > 90 days    one rollup row per project per month

The user's constraint this satisfies directly: "I might work on a project
for a year but I don't need the whole year... More than deleted, I would
like a compressed cache." Rollups are rows, not deletions — "what did I do
in March" must still answer, at month granularity, which is the property
these tests check rather than raw row counts.

Assumes a new `rollups(id, project_scope, period, period_start, period_end,
session_count, files_touched, commits, loops_opened, loops_closed)` table
and a `nenapu.rollup` module with `rollup_activity(ledger, *, now=None)`.
"""

import time

import pytest

from nenapu import connect

DAY = 86400.0


@pytest.fixture
def ledger():
    from nenapu.activity import ActivityLedger

    return ActivityLedger(connect(":memory:"))


def _session_days_ago(ledger, days_ago, *, project_scope="repo:demo@aaaaaaaa", paths=()):
    at = time.time() - days_ago * DAY
    session_id = ledger.start_session(agent="claude-code", project_scope=project_scope,
                                      cwd="/repo", started_at=at)
    for path in paths:
        ledger.record_file_event(session_id, path=path, op="edited", tool="Edit", at=at)
    ledger.end_session(session_id, ended_at=at + 60)
    return session_id


# ---------- schema ----------


def test_rollups_table_exists():
    conn = connect(":memory:")
    columns = {r["name"] for r in conn.execute("PRAGMA table_info(rollups)")}
    assert columns >= {
        "id", "project_scope", "period", "period_start", "period_end",
        "session_count", "files_touched", "commits",
    }


# ---------- rollup behaviour ----------


def test_recent_sessions_are_left_at_full_detail(ledger):
    """Nothing inside the 14-day window should be rolled up — `standup` and
    `activity` need full detail for "what happened yesterday"."""
    from nenapu.rollup import rollup_activity

    recent = _session_days_ago(ledger, 2, paths=["app.py"])

    rollup_activity(ledger)

    assert ledger.get_session(recent) is not None


def test_sessions_between_14_and_90_days_roll_up_weekly(ledger):
    from nenapu.rollup import rollup_activity

    for offset in (20, 21, 22):
        _session_days_ago(ledger, offset, paths=[f"file{offset}.py"])

    rollup_activity(ledger)

    rollups = ledger.rollups_for_scope("repo:demo@aaaaaaaa", period="week")
    assert len(rollups) >= 1
    assert sum(r["session_count"] for r in rollups) == 3


def test_sessions_older_than_90_days_roll_up_monthly(ledger):
    from nenapu.rollup import rollup_activity

    for offset in (100, 105, 110):
        _session_days_ago(ledger, offset, paths=[f"old{offset}.py"])

    rollup_activity(ledger)

    rollups = ledger.rollups_for_scope("repo:demo@aaaaaaaa", period="month")
    assert len(rollups) >= 1
    assert sum(r["session_count"] for r in rollups) == 3


def test_rollup_is_additive_not_destructive_of_the_answer(ledger):
    """"What did I do in March" must still answer after rollup, at month
    granularity — rollups summarise, they do not silently make history
    unanswerable."""
    from nenapu.rollup import rollup_activity

    for offset in (100, 101):
        _session_days_ago(ledger, offset, paths=["march_file.py"])

    rollup_activity(ledger)

    rollups = ledger.rollups_for_scope("repo:demo@aaaaaaaa", period="month")
    assert any(r["files_touched"] >= 1 for r in rollups)


def test_rollup_is_idempotent(ledger):
    """Running the maintenance tick's rollup pass twice must not double-count
    sessions already summarised."""
    from nenapu.rollup import rollup_activity

    for offset in (95, 96):
        _session_days_ago(ledger, offset, paths=["x.py"])

    rollup_activity(ledger)
    rollup_activity(ledger)

    rollups = ledger.rollups_for_scope("repo:demo@aaaaaaaa", period="month")
    assert sum(r["session_count"] for r in rollups) == 2


def test_rollups_are_scoped_per_project(ledger):
    """Two projects' history must not blend into one rollup row."""
    from nenapu.rollup import rollup_activity

    _session_days_ago(ledger, 100, project_scope="repo:a@11111111", paths=["a.py"])
    _session_days_ago(ledger, 100, project_scope="repo:b@22222222", paths=["b.py"])

    rollup_activity(ledger)

    a_rollups = ledger.rollups_for_scope("repo:a@11111111", period="month")
    b_rollups = ledger.rollups_for_scope("repo:b@22222222", period="month")
    assert sum(r["session_count"] for r in a_rollups) == 1
    assert sum(r["session_count"] for r in b_rollups) == 1
