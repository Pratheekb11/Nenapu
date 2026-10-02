"""Usage credit fades with time since the fact was last used.

The `usage` ranking term used to read `use_count` alone, so a fact used ten
times a year ago kept full usage credit forever and outranked one used twice
this week. `last_used_at` was recorded on every bump and never read.

Usage is now `saturation(use_count) * 0.5 ** (days_since_last_use / half_life)`,
computed at query time. No background job and no stored rank: recall is
query-dependent, so a periodically rewritten global order would only go stale.

What must not change:

  * A fact used just now scores exactly what it scored before.
  * A fact never used scores zero, however old it is.
  * Usage stays a small additive term; this changes its value, not its weight.
"""

from __future__ import annotations

import math

import pytest

from nenapu import connect, embeddings
from nenapu.models import Fact, now
from nenapu.store import DAY, SEARCH_WEIGHTS, USAGE_HALF_LIFE_DAYS, Store, usage_score


@pytest.fixture
def store(monkeypatch):
    # Pin the degraded profile so both suite runs (with and without the
    # embedder installed) rank on the same terms.
    monkeypatch.setattr(embeddings, "get_embedder", lambda: None)
    return Store(connect(":memory:"))


def _saturation(uses: int) -> float:
    return min(1.0, math.log1p(uses) / math.log(11))


def _fact(uses: int, last_used_days_ago: float | None, at: float) -> Fact:
    last = None if last_used_days_ago is None else at - last_used_days_ago * DAY
    return Fact(text="x", use_count=uses, last_used_at=last, created_at=at - 400 * DAY)


def _set_usage(store: Store, fact_id: int, uses: int, days_ago: float) -> None:
    store.conn.execute(
        "UPDATE facts SET use_count = ?, last_used_at = ? WHERE id = ?",
        (uses, now() - days_ago * DAY, fact_id),
    )


# --- the score ----------------------------------------------------------------


def test_a_fact_used_just_now_scores_what_it_always_did():
    at = now()
    for uses in (1, 3, 10, 40):
        assert usage_score(_fact(uses, 0, at), at) == pytest.approx(_saturation(uses))


def test_a_never_used_fact_scores_zero():
    at = now()
    assert usage_score(_fact(0, None, at), at) == 0.0


def test_usage_halves_after_one_half_life():
    at = now()
    fresh = usage_score(_fact(10, 0, at), at)
    stale = usage_score(_fact(10, USAGE_HALF_LIFE_DAYS, at), at)
    assert stale == pytest.approx(fresh / 2)


def test_heavy_old_use_falls_below_light_recent_use():
    """The case the change exists for."""
    at = now()
    old_heavy = usage_score(_fact(30, 365, at), at)
    recent_light = usage_score(_fact(2, 1, at), at)
    assert old_heavy < recent_light


def test_a_used_fact_with_no_last_used_time_fades_from_creation():
    """Rows from before `last_used_at` was written have a count but no time.
    Anchoring on creation, as `effective_confidence` does for verification,
    keeps them from holding full credit forever."""
    at = now()
    legacy = Fact(text="x", use_count=10, last_used_at=None, created_at=at - 400 * DAY)
    assert usage_score(legacy, at) < _saturation(10) / 2


def test_a_last_used_time_in_the_future_is_not_a_bonus():
    """Clock skew between machines must not push usage above saturation."""
    at = now()
    assert usage_score(_fact(10, -5, at), at) == pytest.approx(_saturation(10))


# --- recall -------------------------------------------------------------------


def test_recall_reports_the_faded_usage(store):
    fact, _ = store.write(Fact(text="the deploy script lives in bin/release"))
    _set_usage(store, fact.id, 10, USAGE_HALF_LIFE_DAYS)

    (_f, _score, why), = store.search("deploy", log_recall=False, mark_used=False)

    assert why["usage"] == pytest.approx(_saturation(10) / 2, abs=0.002)


def test_recently_used_fact_outranks_an_equal_one_used_long_ago(store):
    stale, _ = store.write(Fact(text="the release tag is cut from main by ci alpha"))
    fresh, _ = store.write(Fact(text="the release tag is cut from main by ci beta"))
    _set_usage(store, stale.id, 10, 365)
    _set_usage(store, fresh.id, 10, 0)

    results = store.search("release tag", log_recall=False, mark_used=False)
    by_id = {f.id: (score, why) for f, score, why in results}

    assert by_id[fresh.id][1]["usage"] > by_id[stale.id][1]["usage"]
    assert results[0][0].id == fresh.id


def test_stale_usage_never_beats_relevance(store):
    """Usage is the smallest term. A fresh, heavily used fact that does not
    match the query must not outrank one that does."""
    match, _ = store.write(Fact(text="alembic migrations run before the api starts"))
    other, _ = store.write(Fact(text="the frontend builds with vite"))
    _set_usage(store, other.id, 50, 0)

    results = store.search("alembic migrations", log_recall=False, mark_used=False)

    assert results[0][0].id == match.id


def test_weights_are_untouched():
    assert SEARCH_WEIGHTS["usage"] == 0.1
