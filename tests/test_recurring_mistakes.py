"""Surfacing "you have had to say this N times" in the injected block.

Requirement (Task 10, priority-ordered task list): "Recurring-mistake
surfacing — occurrences weighting in injection | 'you have had to say this 3
times'; free once 9 lands." Depends on Task 9 (Opus 5: feed relevant memory
to the extractor + `op: add|update|noop`), which is where a repeated
correction becomes one fact with a growing `occurrences` counter instead of
several near-duplicate facts at middling confidence — the bug the plan
measured directly against the live store:

    3x | The Nenapu CLI pet must be a cute dog, not a bear and not a kaomoji...
    2x | The user wants security and privacy gaps fixed rather than only...
    2x | The Nenapu CLI pet must be a cute puppy — big floppy filled ears...

Task 10's own scope, once that counter exists, is narrow: weight it into
`recall_context()`'s ordering and render the count in the text a session
actually reads. These tests construct facts with `occurrences` set directly
(bypassing Task 9's dedupe/merge machinery, which is out of scope here) so
they isolate exactly Task 10's contract.

Assumes `Fact` gains an `occurrences: int = 1` field (the plan's own wording
for the counter) and that `recall_context()` renders corrections with
`occurrences > 1` using an explicit count rather than silently.
"""

import pytest

from nenapu import connect
from nenapu.models import Fact, Kind
from nenapu.observer import recall_context
from nenapu.store import Store


@pytest.fixture
def store():
    return Store(connect(":memory:"))


def test_fact_carries_an_occurrences_count():
    fact = Fact(text="always squash before merging", kind=Kind.FEEDBACK, occurrences=3)
    assert fact.occurrences == 3


def test_occurrences_defaults_to_one():
    """A correction said once should not read as "you have had to say this
    1 time" — the default must not trigger the recurring-mistake framing."""
    fact = Fact(text="use tabs not spaces", kind=Kind.FEEDBACK)
    assert fact.occurrences == 1


def test_a_repeated_correction_is_called_out_by_count(store):
    store.write(Fact(
        text="the CLI pet must be a dog, not a bear",
        kind=Kind.FEEDBACK, occurrences=3, confidence=0.9,
    ))

    block = recall_context(store)

    assert "3 times" in block


def test_a_correction_said_once_is_not_called_out_by_count(store):
    store.write(Fact(
        text="use tabs not spaces", kind=Kind.FEEDBACK, confidence=0.9,
    ))

    block = recall_context(store)

    assert "1 time" not in block
    assert "times" not in block


def test_high_occurrence_corrections_are_prioritised_in_the_injection_budget(store):
    """`MAX_INJECTED` caps the block at 12 facts (`observer.py:46`); when
    corrections compete for the budget, the one the user has repeated most is
    the one most worth the tokens."""
    from nenapu.observer import MAX_INJECTED

    for i in range(MAX_INJECTED + 2):
        store.write(Fact(
            text=f"minor correction number {i}", kind=Kind.FEEDBACK, confidence=0.9,
        ))
    store.write(Fact(
        text="the recurring one, said five times", kind=Kind.FEEDBACK,
        occurrences=5, confidence=0.9,
    ))

    block = recall_context(store)

    assert "the recurring one, said five times" in block
    assert "5 times" in block
