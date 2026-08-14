"""A creature that is the store's health, rather than a mascot beside it.

`nenapu stats` already prints everything in here. Nobody reads it. A table of
eleven numbers has no opinion about whether anything is wrong, so it gets
skimmed and the memory debt sits there for a week.

So the same numbers get a face. Every mood below is a real signal with a
threshold you can argue with — the pet is hungry because nothing has been
learned in days, sick because a check it trusted started failing, spooked
because a falsification cascade knocked facts over. It cannot be cheerful while
the store is unwell, which is the entire point: cosmetic pets teach you to
ignore them.

Priority matters more than the moods do. A store can be several things at once,
and what you want to see is the worst one — a happy face beside a failing check
is a lie, so the order below runs from most alarming to least.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

# Thresholds. Each is a judgement call, so each says what it is for.
HUNGRY_AFTER_DAYS = 3.0      # nothing learned in this long and it starts asking
DROWSY_STALE_SHARE = 0.4     # this share of memory decayed below useful
RESTLESS_PENDING = 10        # recalls waiting on a grade before it gets antsy
STUFFED_FACTS = 800          # past here, distilling is overdue more than not
DAY = 86400.0


@dataclass
class Pet:
    mood: str
    face: str
    blurb: str
    notes: list[str] = field(default_factory=list)
    stats: dict = field(default_factory=dict)

    @property
    def unwell(self) -> bool:
        """Whether the mood is one that wants something done about it."""
        return self.mood in ("sick", "spooked", "hungry", "drowsy", "stuffed")


# Ears down, matching the drawing. Only the eyes move between moods: a face you
# recognise at a glance is a face you read at a glance. (The project's own mark
# stays ᐡ•ᴥ•ᐡ — that is the wordmark, not the pet.)
FACES = {
    "sick": "▽×ᴥ×▽",
    "spooked": "▽⊙ᴥ⊙▽",
    "hungry": "▽•ᗝ•▽",
    "drowsy": "▽˘ᴥ˘▽",
    "restless": "▽˙ᴥ˙▽",
    "stuffed": "▽-ᴥ-▽",
    "content": "▽•ᴥ•▽",
    "delighted": "▽^ᴥ^▽",
    "new": "▽·ᴥ·▽",
}

# Second frame for the blink. Anything already squinting stays put — a sleeping
# dog that flickers reads as a glitch rather than as a dog.
BLINKS = {
    "sick": "▽×ᴥ×▽",
    "spooked": "▽－ᴥ－▽",
    "hungry": "▽－ᗝ－▽",
    "drowsy": "▽˘ᴥ˘▽",
    "restless": "▽－ᴥ－▽",
    "stuffed": "▽-ᴥ-▽",
    "content": "▽－ᴥ－▽",
    "delighted": "▽－ᴥ－▽",
    "new": "▽－ᴥ－▽",
}


def _ago(seconds: float | None) -> str:
    if seconds is None:
        return "never"
    if seconds < 90:
        return "just now"
    if seconds < 5400:
        return f"{int(seconds // 60)}m ago"
    if seconds < 2 * DAY:
        return f"{int(seconds // 3600)}h ago"
    return f"{int(seconds // DAY)}d ago"


def _freshness(store) -> dict:
    """When it last ate, and what the last meal was.

    "Ate" means a fact was written. Sessions observed by the hook are counted
    separately because that is the food it gets without being asked — a store
    that only grows when a human types into it is not being kept by the layer,
    it is being kept by the human.
    """
    row = store.conn.execute(
        "SELECT MAX(created_at) fed, MAX(CASE WHEN origin = 'tool_observed'"
        " THEN created_at END) observed FROM facts WHERE status = 'active'"
    ).fetchone()
    now = time.time()
    fed, observed = row["fed"], row["observed"]
    today = store.conn.execute(
        # Active only, so it cannot read "80 facts, 92 learned today" — the
        # ones that were superseded within the day are not still known.
        "SELECT COUNT(*) c FROM facts WHERE status = 'active' AND created_at > ?",
        (now - DAY,),
    ).fetchone()["c"]
    return {
        "fed_seconds_ago": (now - fed) if fed else None,
        "observed_seconds_ago": (now - observed) if observed else None,
        "learned_today": today,
    }


def assess(store) -> Pet:
    """Read the store and decide how the creature is doing.

    Order is most alarming first. A pet that looks happy while a verification
    is failing has taught you to stop believing it.
    """
    s = store.stats()
    fresh = _freshness(store)
    active = s.get("active", 0)
    stale = s.get("stale_active", 0)
    pending = (s.get("recalls") or {}).get("pending", 0)
    failing = s.get("failing_verification", 0)
    suspect = s.get("suspect", 0)
    fed_ago = fresh["fed_seconds_ago"]
    stats = {**s, **fresh}

    notes = [
        f"{active} fact{'s' if active != 1 else ''}, "
        f"{fresh['learned_today']} learned today",
        f"last fed {_ago(fed_ago)}"
        + (f", last observed a session {_ago(fresh['observed_seconds_ago'])}"
           if fresh["observed_seconds_ago"] is not None else ", never observed a session"),
    ]

    # "Nothing active" is not the same as "nothing here". A store whose facts
    # have all been knocked over by a cascade has an empty active set and is
    # the opposite of new — reporting it as a fresh install was the first bug
    # this module's own tests found.
    if sum(s.get("by_status", {}).values()) == 0:
        return Pet("new", FACES["new"], "brand new, and knows nothing yet",
                   ["tell it something: nenapu write \"...\"",
                    "or let it watch: nenapu init"], stats)

    if failing:
        return Pet("sick", FACES["sick"],
                   f"{failing} thing{'s' if failing > 1 else ''} it believed "
                   f"stopped being true",
                   notes + ["a check that used to pass is failing — nenapu loops"], stats)

    if suspect:
        return Pet("spooked", FACES["spooked"],
                   f"{suspect} fact{'s' if suspect > 1 else ''} lost the ground "
                   f"{'they' if suspect > 1 else 'it'} stood on",
                   notes + ["something they rested on was falsified — nenapu loops"], stats)

    if fed_ago is not None and fed_ago > HUNGRY_AFTER_DAYS * DAY:
        return Pet("hungry", FACES["hungry"],
                   f"nothing new in {_ago(fed_ago).replace(' ago', '')}",
                   notes + ["either you have been away, or the Stop hook is not "
                            "firing — nenapu doctor"], stats)

    if active and stale / active > DROWSY_STALE_SHARE:
        return Pet("drowsy", FACES["drowsy"],
                   f"{stale} of {active} facts have gone cold",
                   notes + ["belief decays on purpose; re-verify or forget them "
                            "— nenapu verify"], stats)

    if active > STUFFED_FACTS:
        return Pet("stuffed", FACES["stuffed"], f"carrying {active} facts around",
                   notes + ["a lot of that is probably the same thing twice "
                            "— nenapu distill"], stats)

    if pending > RESTLESS_PENDING:
        return Pet("restless", FACES["restless"],
                   f"waiting to hear whether {pending} recalls helped",
                   notes + ["grading is what keeps recall honest — nenapu outcome"], stats)

    mean = s.get("mean_confidence", 0.0)
    if mean >= 0.7 and fresh["learned_today"]:
        return Pet("delighted", FACES["delighted"], "well fed, and sure of itself",
                   notes, stats)

    return Pet("content", FACES["content"], "nothing to worry about", notes, stats)


def line(pet: Pet) -> str:
    """One line, for a status bar. No colour, no width assumptions."""
    active = pet.stats.get("active", 0)
    bits = [pet.face, f"{active} fact{'s' if active != 1 else ''}"]
    if pet.stats.get("failing_verification"):
        bits.append(f"{pet.stats['failing_verification']} failing")
    if pet.stats.get("suspect"):
        bits.append(f"{pet.stats['suspect']} suspect")
    bits.append(f"fed {_ago(pet.stats.get('fed_seconds_ago'))}")
    return " · ".join(bits)


MOOD_COLOUR = {
    "sick": "red",
    "spooked": "yellow",
    "hungry": "yellow",
    "drowsy": "cyan",
    "restless": "cyan",
    "stuffed": "cyan",
    "content": "green",
    "delighted": "green",
    "new": "dim",
}


def render(pet: Pet, *, blink: bool = False) -> str:
    """The compact view: the mark, the mood, the numbers.

    Deliberately not a box. The pet is looked at in the middle of doing
    something else, and a panel border makes a glance feel like a report.
    """
    colour = MOOD_COLOUR.get(pet.mood, "white")
    face = BLINKS.get(pet.mood, pet.face) if blink else pet.face
    accent = {"drowsy": " [dim]z z[/]", "hungry": " [dim]...[/]",
              "spooked": " [dim]![/]", "sick": " [dim]?[/]"}.get(pet.mood, "")

    lines = [f"    [{colour}]{face}[/]{accent}   [italic]{pet.blurb}[/]", ""]
    lines += [f"      [dim]{note}[/]" for note in pet.notes]
    return "\n".join(lines)


# Below this width the drawn dog and its status cannot sit side by side, and
# stacking a 44-column animal on top of the numbers pushes them off a small
# screen. The compact view is not a fallback, it is the right answer there.
FULL_MIN_WIDTH = 78


def render_full(pet: Pet, shades: list[str], *, blink: bool = False):
    """The drawn creature with its readout beside it.

    Rich renderable rather than a string: the art is coloured per row and the
    status has to line up next to it without either being padded by hand.
    """
    from rich.table import Table
    from rich.text import Text

    from .pet_art import coloured

    art = coloured(pet.mood, shades, blink=blink)
    colour = MOOD_COLOUR.get(pet.mood, "white")

    # No manual padding: the grid column is vertically centred, and doing
    # both leaves the text sitting below the dog it belongs to.
    status = Text()
    status.append_text(Text.from_markup(f"[bold {colour}]{pet.mood}[/]  "))
    status.append_text(Text.from_markup(f"[italic]{pet.blurb}[/]\n\n"))
    for note in pet.notes:
        status.append_text(Text.from_markup(f"[dim]{note}[/]\n"))

    layout = Table.grid(padding=(0, 3))
    layout.add_column()
    layout.add_column(vertical="middle")
    layout.add_row(Text.from_markup("\n".join(art)), status)
    return layout
