import time

import pytest

from nenapu import connect
from nenapu.distill import dedupe, distill, estimate_tokens
from nenapu.export import BEGIN, END, render, write_file
from nenapu.models import Decay, Fact, Kind, Origin, Status
from nenapu.store import Store


@pytest.fixture
def store():
    return Store(connect(":memory:"))


def test_export_groups_and_omits_stale(store):
    store.write(Fact(text="user prefers terse output", kind=Kind.USER,
                     origin=Origin.USER_STATED, confidence=0.95))
    old = time.time() - 500 * 86400
    store.write(Fact(text="staging runs on box-7", kind=Kind.ENVIRONMENT,
                     decay_class=Decay.VOLATILE, created_at=old, last_verified_at=old))

    block = render(store, min_confidence=0.35)
    assert BEGIN in block and END in block
    assert "terse output" in block
    assert "box-7" not in block


def test_export_preserves_surrounding_file(store, tmp_path):
    target = tmp_path / "CLAUDE.md"
    target.write_text("# My project\n\nHand written notes.\n")
    store.write(Fact(text="tests run with pytest", origin=Origin.USER_STATED, confidence=0.9))

    write_file(target, store)
    once = target.read_text()
    assert "Hand written notes." in once and "pytest" in once

    store.write(Fact(text="linting uses ruff", origin=Origin.USER_STATED, confidence=0.9))
    write_file(target, store)
    twice = target.read_text()
    assert twice.count(BEGIN) == 1  # replaced, not appended
    assert "Hand written notes." in twice and "ruff" in twice


def test_dedupe_archives_near_duplicates(store):
    store.write(Fact(text="the build uses hatchling as the backend",
                     origin=Origin.USER_STATED, confidence=0.9))
    store.write(Fact(text="the build uses hatchling backend"))
    assert dedupe(store) == 1
    active = store.list_facts()
    assert len(active) == 1
    archived = store.list_facts(status=Status.ARCHIVED)
    assert archived[0].distilled_into_id == active[0].id


def test_distill_without_llm_only_dedupes(store):
    store.write(Fact(text="deploy script lives in scripts/deploy.sh"))
    store.write(Fact(text="the deploy script lives in scripts/deploy.sh"))
    report = distill(store, use_llm=False)
    assert report.deduped == 1 and report.merged == 0
    assert report.tokens_after < report.tokens_before


def test_estimate_tokens_tracks_length(store):
    assert estimate_tokens([Fact(text="x" * 400)]) == 100


def test_dedupe_keeps_the_more_specific_fact(store):
    store.write(Fact(text="we use postgres", origin=Origin.USER_STATED, confidence=0.95))
    store.write(Fact(text="we use postgres for the analytics warehouse only",
                     origin=Origin.USER_STATED, confidence=0.6))
    assert dedupe(store) == 0
    assert len(store.list_facts()) == 2
