"""Downsample the activity ledger by age.

Fact distillation compresses by similarity; a work log needs compressing by
age instead:

    0-14 days    every session and file_event, full detail
    14-90 days   one rollup row per project per ISO week
    > 90 days    one rollup row per project per calendar month

Rollups are rows, not deletions of the answer — "what did I do in March"
still answers, at month granularity. But the raw session and file_event rows
that fed a rollup are removed once summarised, which is the actual
compression the user asked for: a year of work should not have to sit on
disk as a year of individual sessions to stay queryable.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from .activity import ActivityLedger
from .models import now as _now

DAY = 86400.0
RECENT_WINDOW_DAYS = 14
MONTHLY_THRESHOLD_DAYS = 90


def _week_bounds(ts: float) -> tuple[float, float]:
    d = datetime.fromtimestamp(ts, tz=timezone.utc)
    monday = (d - timedelta(days=d.weekday())).replace(
        hour=0, minute=0, second=0, microsecond=0
    )
    start = monday.timestamp()
    return start, start + 7 * DAY


def _month_bounds(ts: float) -> tuple[float, float]:
    d = datetime.fromtimestamp(ts, tz=timezone.utc)
    start = datetime(d.year, d.month, 1, tzinfo=timezone.utc)
    end = (
        datetime(d.year + 1, 1, 1, tzinfo=timezone.utc)
        if d.month == 12
        else datetime(d.year, d.month + 1, 1, tzinfo=timezone.utc)
    )
    return start.timestamp(), end.timestamp()


def _fold(ledger: ActivityLedger, sessions: list[dict], *, period: str, bounds_fn) -> None:
    buckets: dict[tuple[str, float], dict] = {}
    for session in sessions:
        start, end = bounds_fn(session["started_at"])
        key = (session["project_scope"], start)
        bucket = buckets.setdefault(
            key, {"period_end": end, "session_count": 0, "paths": set(), "commits": 0}
        )
        bucket["session_count"] += 1
        bucket["paths"].update(e["path"] for e in ledger.file_events_for_session(session["id"]))
        bucket["commits"] += len(ledger.commits_for_session(session["id"]))

    for (scope, start), bucket in buckets.items():
        ledger.upsert_rollup(
            scope, period, start, bucket["period_end"],
            session_count=bucket["session_count"],
            files_touched=len(bucket["paths"]),
            commits=bucket["commits"],
        )

    for session in sessions:
        ledger.delete_session(session["id"])


def rollup_activity(ledger: ActivityLedger, *, now: float | None = None) -> None:
    """Fold everything outside the 14-day full-detail window into weekly or
    monthly rollup rows, then remove the raw rows it summarised.

    Safe to call repeatedly: a session already folded away is gone from the
    ledger, so a later pass has nothing left to double-count.
    """
    at = now if now is not None else _now()
    recent_cutoff = at - RECENT_WINDOW_DAYS * DAY
    monthly_cutoff = at - MONTHLY_THRESHOLD_DAYS * DAY

    weekly = ledger.sessions_in_range(monthly_cutoff, recent_cutoff)
    _fold(ledger, weekly, period="week", bounds_fn=_week_bounds)

    monthly = ledger.sessions_in_range(0, monthly_cutoff)
    _fold(ledger, monthly, period="month", bounds_fn=_month_bounds)
