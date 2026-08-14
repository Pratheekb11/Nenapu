"""The landing view has one job beyond looking right: it must fit the screen.

The bug these are for: stacked, the wordmark, the store readout and Typer's
grouped help came to seventy-three rows. On a twenty-four row terminal that put
the mark someone had just run `nenapu` to look at fifty lines above the top of
the screen, where they could never see it.
"""

import pytest
from rich.console import Console

import nenapu.cli as cli
from nenapu import connect, open_store
from nenapu.models import Fact, Kind, Origin
from nenapu.store import Store

# Every braille cell, so a line of drawing can be told from a line of text.
BRAILLE = "".join(chr(c) for c in range(0x2800, 0x2900))


def render(width: int, height: int, store=None) -> list[str]:
    console = Console(width=width, height=height, record=True, file=open("/dev/null", "w"))
    original = cli.console
    cli.console = console
    try:
        console.print()
        console.print(cli._landing(store))
        console.print()
    finally:
        cli.console = original
    return console.export_text().rstrip("\n").split("\n")


@pytest.fixture
def store(tmp_path):
    s = Store(connect(str(tmp_path / "m.db")))
    s.write(Fact(text="deploys go through make ship", kind=Kind.PROJECT,
                 origin=Origin.USER_STATED))
    return s


@pytest.mark.parametrize("width, height", [
    (120, 40), (110, 30), (110, 24), (110, 20), (100, 24),
    (96, 24), (80, 30), (80, 24), (72, 24), (60, 20), (60, 14), (40, 20),
])
def test_the_landing_view_never_scrolls(width, height, store):
    """Every size a terminal is plausibly at. One row too tall and the
    wordmark is gone, which is the whole failure."""
    lines = render(width, height, store)
    assert len(lines) <= height, f"{len(lines)} lines in {height} rows"


def test_the_mark_survives_a_short_terminal(store):
    """Shedding parts is only worth doing if the thing being protected stays.
    On a short screen the block letters go and the one-line mark takes over —
    what must never happen is the name being absent entirely."""
    lines = render(110, 18, store)
    text = "\n".join(lines)
    assert "nenapu" in text
    assert len(lines) <= 18


def test_a_wide_terminal_puts_the_dog_beside_the_readout(store):
    """The side-by-side layout is what buys the room. If it silently stopped
    happening, the view would still fit and would have lost the dog."""
    lines = render(120, 40, store)
    dog_rows = [line for line in lines if "⣿" in line or "⠿" in line]
    assert dog_rows, "no drawing on a wide terminal"
    assert any("facts" in line for line in dog_rows), \
        "the readout should sit on the same rows as the dog, not below it"


def test_a_narrow_terminal_truncates_nothing_it_wrote_as_prose(store):
    """Below the side-by-side width there is no room for the three-line pitch,
    which is hand-set at 76 characters. Left in, Rich cuts it mid-sentence and
    the summary reads as a bug. The store path is a different matter — it is
    deliberately elided by `_shorten`, and that ellipsis is meant."""
    lines = [line for line in render(72, 30, store) if "store" not in line]

    assert "…" not in "\n".join(lines)
    assert "search" in "\n".join(lines)


def test_every_command_is_named_somewhere(store):
    """The landing view replaced the full help, so it is now the only place a
    reader is told what exists. A command missing from it is invisible."""
    listed = {name for rows in cli._command_groups().values() for name, _ in rows}
    text = "\n".join(render(120, 40, store))

    for name in listed:
        assert name in text, f"{name} is registered but never shown"
    assert "write" in listed and "pet" in listed


def test_the_command_list_is_read_from_the_app(store):
    """Written out by hand it would go stale the first time someone adds a
    command, and nothing would fail to say so."""
    groups = cli._command_groups()

    assert groups, "no commands found"
    assert "recall-hook" not in {n for rows in groups.values() for n, _ in rows}, \
        "hidden commands are machine-to-machine and do not belong on a landing page"


def test_the_view_survives_a_store_that_will_not_open(tmp_path):
    """A landing screen that raises is worse than one with an empty readout."""
    lines = render(120, 40, None)

    assert any("nenapu" in line for line in lines)


def test_an_unwell_store_shows_it_on_the_landing_screen(tmp_path):
    """The dog is a readout here too. A failing check should be visible from
    the screen someone lands on, not only from `nenapu pet`."""
    from nenapu.models import VerifyStatus

    store, _ = open_store(str(tmp_path / "m.db"))
    fact = store.write(Fact(text="cache lives in /tmp/cache", kind=Kind.PROJECT,
                            origin=Origin.USER_STATED, verify_cmd="test -d /nope"))[0]
    store.conn.execute("UPDATE facts SET verify_status = ? WHERE id = ?",
                       (VerifyStatus.FAIL, fact.id))

    healthy = "\n".join(render(120, 40, store=None))
    unwell = "\n".join(render(120, 40, store=store))

    assert unwell != healthy, "the drawing did not change with the store's health"


# ---------- using the room, not just surviving it ----------


@pytest.mark.parametrize("width, height", [(120, 40), (110, 30), (100, 36), (96, 30)])
def test_a_tall_terminal_is_not_mostly_empty(width, height, store):
    """The first fix over-corrected: twenty-three rows on a forty-row screen,
    most of it blank, which reads as a program with nothing to say. The view
    is built at several sizes now and takes the largest that fits."""
    for i in range(12):
        store.write(Fact(text=f"a fact worth showing on the landing screen, number {i}",
                         kind=Kind.PROJECT, origin=Origin.TOOL_OBSERVED))

    used = len(render(width, height, store))

    assert used <= height
    assert used >= height * 0.7, f"only {used} of {height} rows used"


def test_the_dog_grows_with_the_room(store):
    """Every measurement in the drawing is a multiplication of one number,
    which is the point of drawing it rather than typing it."""
    from nenapu.pet_art import draw

    small, large = draw("content"), draw("content", scale=2.0)

    assert len(large) > len(small)
    assert max(len(r) for r in large) > max(len(r) for r in small)


def test_the_dog_is_capped_by_width_not_only_height(store):
    """Sized against rows alone it grew until the column beside it could not
    hold a sentence, and Rich answered by cutting every line off with an
    ellipsis — a bigger drawing bought with the text that says what it knows.
    """
    for i in range(12):
        store.write(Fact(text=f"a fact long enough to be clipped if the column is thin {i}",
                         kind=Kind.PROJECT, origin=Origin.TOOL_OBSERVED))

    for width in (96, 100, 110, 120):
        lines = render(width, 44, store)
        drawn = max((len(line) - len(line.lstrip(BRAILLE)) for line in lines
                     if line[:1] in BRAILLE), default=0)
        assert width - drawn >= cli.MIN_TEXT_COLUMN, \
            f"at {width} columns the drawing left only {width - drawn} for the text"


def test_what_it_learned_lately_shows_when_there_is_room(store):
    """Better filler than more art: someone landing here wants to know the
    thing is working, and sentences it picked up on its own answer that faster
    than any number can."""
    store.write(Fact(text="the deploy command is make ship, not make deploy",
                     kind=Kind.PROJECT, origin=Origin.TOOL_OBSERVED))

    roomy = "\n".join(render(120, 40, store))
    cramped = "\n".join(render(110, 20, store))

    assert "Lately" in roomy and "make ship" in roomy
    assert "Lately" not in cramped, "a short screen should spend its rows on the basics"
