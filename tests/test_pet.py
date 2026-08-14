"""The pet is a readout, so these tests are about whether it can lie.

Each mood is driven by putting a real store into the state that should cause
it, never by handing `assess` a dictionary. A face that says "content" while a
check is failing is the only real bug this feature can have.
"""

import time

import pytest

from nenapu import connect
from nenapu.models import Decay, Fact, Kind, Origin, Status, VerifyStatus
from nenapu.pet import DAY, HUNGRY_AFTER_DAYS, RESTLESS_PENDING, assess, line, render
from nenapu.store import Store


@pytest.fixture
def store():
    return Store(connect(":memory:"))


def _fact(store, text="deploys go through make ship", **kwargs):
    return store.write(Fact(text=text, kind=Kind.PROJECT, origin=Origin.USER_STATED,
                            **kwargs))[0]


def test_an_empty_store_is_new_rather_than_happy(store):
    pet = assess(store)
    assert pet.mood == "new"
    assert "nenapu remember" in " ".join(pet.notes)


def test_a_store_with_nothing_active_left_is_not_reported_as_new(store):
    """The first bug these tests found. Every fact suspect means an empty
    active set, which looked exactly like a fresh install — the most reassuring
    possible face on the least reassuring possible store."""
    fact = _fact(store)
    store.conn.execute("UPDATE facts SET status = ? WHERE id = ?",
                       (Status.SUSPECT, fact.id))

    assert assess(store).mood == "spooked"


def test_a_failing_check_makes_it_sick(store):
    fact = _fact(store, verify_cmd="test -f /nope")
    store.conn.execute("UPDATE facts SET verify_status = ? WHERE id = ?",
                       (VerifyStatus.FAIL, fact.id))

    pet = assess(store)

    assert pet.mood == "sick"
    assert pet.unwell
    assert "stopped being true" in pet.blurb


def test_a_failing_check_outranks_everything_cheerful(store):
    """The ordering is the feature. A store can be several things at once and
    the one worth showing is the worst — anything else teaches you to ignore
    the face."""
    for i in range(5):
        _fact(store, text=f"fresh fact number {i}")
    fact = _fact(store, text="the one that broke", verify_cmd="test -f /nope")
    store.conn.execute("UPDATE facts SET verify_status = ? WHERE id = ?",
                       (VerifyStatus.FAIL, fact.id))

    assert assess(store).mood == "sick"


def test_a_falsified_foundation_spooks_it(store):
    fact = _fact(store)
    store.conn.execute("UPDATE facts SET status = ? WHERE id = ?",
                       (Status.SUSPECT, fact.id))

    pet = assess(store)

    assert pet.mood == "spooked"
    assert "lost the ground" in pet.blurb


def test_it_gets_hungry_when_nothing_is_learned(store):
    fact = _fact(store)
    old = time.time() - (HUNGRY_AFTER_DAYS + 1) * DAY
    store.conn.execute("UPDATE facts SET created_at = ? WHERE id = ?", (old, fact.id))

    pet = assess(store)

    assert pet.mood == "hungry"
    assert "Stop hook" in " ".join(pet.notes), "hunger should point at why it is not eating"


def test_decayed_memory_makes_it_drowsy(store):
    """Belief decaying is by design; most of the store being past the floor is
    a thing to act on.

    The fresh fact at the end matters: without it this store is also starving,
    and hunger outranks drowsiness. Cold memory that is *still being fed* is
    the state this mood is actually for.
    """
    for i in range(5):
        fact = _fact(store, text=f"a thing that was true once, number {i}",
                     decay_class=Decay.VOLATILE, confidence=0.4)
        # Both fields: confidence decays from `last_verified_at or created_at`,
        # so backdating only one of them ages nothing.
        store.conn.execute(
            "UPDATE facts SET created_at = ?, last_verified_at = NULL WHERE id = ?",
            (time.time() - 120 * DAY, fact.id))
    _fact(store, text="something learned just now")

    pet = assess(store)

    assert pet.mood == "drowsy"
    assert "gone cold" in pet.blurb


def test_ungraded_recalls_make_it_restless(store):
    _fact(store)
    hits = store.search("deploys")
    for _ in range(RESTLESS_PENDING + 1):
        store.ledger.log_many(hits, session_id="s1", query="deploys")

    pet = assess(store)

    assert pet.mood == "restless"
    assert "helped" in pet.blurb


def test_a_healthy_store_is_allowed_to_be_happy(store):
    """The counterpart to every test above: if it is never cheerful it is just
    an alarm, and an alarm that is always on is furniture."""
    for i in range(3):
        _fact(store, text=f"a solid fact number {i}", confidence=0.9)

    pet = assess(store)

    assert pet.mood in ("content", "delighted")
    assert not pet.unwell


def test_the_status_line_stays_one_line_and_carries_the_alarm(store):
    fact = _fact(store, verify_cmd="test -f /nope")
    store.conn.execute("UPDATE facts SET verify_status = ? WHERE id = ?",
                       (VerifyStatus.FAIL, fact.id))

    text = line(assess(store))

    assert "\n" not in text
    assert "1 failing" in text
    assert "[" not in text, "a status bar gets no markup"


def test_every_mood_renders(store):
    """A mood with no face or no colour would only show up in the one state
    nobody tested, which is the state it matters in."""
    from nenapu.pet import BLINKS, FACES, MOOD_COLOUR, Pet

    for mood in FACES:
        assert mood in BLINKS and mood in MOOD_COLOUR
        out = render(Pet(mood, FACES[mood], "blurb", ["a note"], {"active": 1}))
        assert FACES[mood] in out and "blurb" in out


def test_freshness_counts_only_what_is_still_known(store):
    """"80 facts, 92 learned today" is nonsense a reader has to stop and parse."""
    first = store.write(Fact(text="the port is 5432", kind=Kind.ENVIRONMENT,
                             key="db.port", origin=Origin.USER_STATED))[0]
    store.write(Fact(text="the port is 6543", kind=Kind.ENVIRONMENT, key="db.port",
                     origin=Origin.USER_STATED))

    pet = assess(store)

    assert store.get(first.id).status == Status.SUPERSEDED
    assert pet.stats["learned_today"] <= pet.stats["active"]


# ---------- the drawing ----------


def test_every_mood_is_baked_at_every_size():
    """The art is generated by `tools/render_pet.py` and baked in as text. A
    mood missing at one size would only show up on the terminal that happens to
    pick that size."""
    from nenapu.pet import FACES
    from nenapu.pet_art import ART, SIZES

    for size in SIZES:
        assert set(ART[size]) == set(FACES), f"{size} columns is missing moods"


def test_each_baked_drawing_is_rectangular():
    """Rows of different lengths would step the right edge in and out, which
    is what made the hand-set versions look homemade."""
    from nenapu.pet_art import ART

    for size, moods in ART.items():
        for mood, rows in moods.items():
            widths = {len(row) for row in rows}
            assert widths == {size}, f"{mood} at {size}: ragged rows {widths}"


def test_the_drawing_fits_beside_its_status():
    from nenapu.pet import FULL_MIN_WIDTH
    from nenapu.pet_art import draw

    rows = draw("content")

    assert len(rows) <= 18
    assert max(len(r) for r in rows) < FULL_MIN_WIDTH // 2


@pytest.mark.parametrize("mood", ["sick", "spooked", "drowsy", "delighted",
                                  "restless"])
def test_each_mood_is_visibly_a_different_face(mood):
    """The moods share one body on purpose. If they also shared a face, the
    drawing would be decoration rather than a readout."""
    from nenapu.pet_art import draw

    assert draw(mood) != draw("content")


def test_blinking_only_moves_eyes_that_were_open():
    """A dog whose eyes are already shut should not flicker; that reads as a
    glitch rather than as a creature."""
    from nenapu.pet_art import draw

    assert draw("content", blink=True) != draw("content")
    assert draw("drowsy", blink=True) == draw("drowsy")
    assert draw("sick", blink=True) == draw("sick")


def test_the_art_carries_no_markup_of_its_own():
    """Rich would swallow a stray bracket, and the dog would lose a row."""
    from nenapu.pet_art import draw

    assert not any("[" in row or "]" in row for row in draw("spooked"))


def test_an_unwell_dog_does_not_get_the_theme_colour():
    """The whole point is that a bad store cannot look like a good one, and a
    calm teal dog with its eyes crossed still reads as fine at a glance."""
    from nenapu.banner import THEMES
    from nenapu.pet_art import MOOD_SHADES, coloured

    teal = THEMES["teal"]
    sick = "\n".join(coloured("sick", teal))
    content = "\n".join(coloured("content", teal))

    assert MOOD_SHADES["sick"][0] in sick
    assert teal[0] not in sick
    assert teal[0] in content


def test_every_row_of_the_drawing_is_coloured():
    from nenapu.banner import THEMES
    from nenapu.pet_art import coloured, draw

    rows = coloured("content", THEMES["teal"])

    assert len(rows) == len(draw("content"))
    assert all(row.startswith("[#") for row in rows)
