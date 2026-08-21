"""Project-scoped injection: what a returning session is told before it starts.

Requirement (Task 7, priority-ordered task list, marked **Opus 5**
*(borderline)*):

    "Project-scoped injection — sections *where you left off* and *changed
    since you were last here* | the daily payoff; both are pure ledger
    queries, no LLM needed | M | Opus 5 (borderline) | depends on 3, 6"

The plan's framing, from "The payoff: context-switch-aware auto-injection":

    "So the ledger is not the deliverable. **Injection is.** The user should
    be able to open an agent in any repo, say something incomplete, and have
    the agent already know where things stood and what was left hanging."

and the bug it fixes:

    "`recall_context()` is called with **no scope** (`cli.py:925`) and returns
    the top 12 facts by belief... Across 11 projects that means a session in
    the OOH backend gets injected with Nenapu's Ollama context-window facts."

The block the plan specifies has four sections, in this order:

    # Memory (nenapu) — physical-ads-and-OOH-MVP-backend

    Where you left off (3 days ago, claude-code):
    - touched backend/app/bookings.py, ...
    - last commit: "Add booking overlap constraint"

    Open here — mentioned but not done:
    - Rate limiting on the public availability endpoint

    Changed since you were last here:
    - 4 commits on main by another session; backend/app/models.py moved

    Previously corrected — do not repeat these:
    - Commits without a Co-Authored-By trailer.

Three of the four come from the activity ledger, not from a model. The
fourth is what `recall_context` already emits.

Proposed seam
-------------
`observer.recall_context(store, *, scope=None, cwd=None, session_id=None,
limit=...)` keeps its existing arguments and grows ledger-driven sections
whenever `scope` is given — `cwd` is what the "changed since" section runs
its one git call in — plus one per-section cap each, because the plan is explicit that this
is paid for on every request:

    MAX_LEFT_OFF_FILES, MAX_OPEN_LOOPS, MAX_CHANGED

`cli.recall_hook` derives the scope from the session's cwd via
`store.project_scope` and passes `["global", scope]`, which is the line that
actually fixes the reported bug.

The "Open here" section is fed by Task 11's open loops; only its rendering
is pinned here, its capture and closure are pinned in
`tests/test_open_loops.py`.

Remove the `pytestmark` line below when Task 7 lands.
"""

import json
import os
import subprocess
import sys

import pytest

from nenapu import connect
from nenapu.models import Fact, Kind
from nenapu.store import Store

DAY = 86400.0


def _git(repo, *args):
    return subprocess.run(
        ["git", *args], cwd=repo, check=True, capture_output=True, text=True,
        env={"GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@example.com",
             "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@example.com",
             "PATH": os.environ.get("PATH", ""), "HOME": str(repo)},
    )


@pytest.fixture
def store():
    return Store(connect(":memory:"))


@pytest.fixture
def ledger(store):
    from nenapu.activity import ActivityLedger

    return ActivityLedger(store.conn)


SCOPE = "repo:backend@aaaaaaaa"
OTHER = "repo:portfolio@bbbbbbbb"


def _finished_session(ledger, *, scope=SCOPE, agent="claude-code", ago=3 * DAY,
                      files=("backend/app/bookings.py",), head_after=None,
                      subject="Add booking overlap constraint"):
    from nenapu.models import now

    started = now() - ago
    session_id = ledger.start_session(
        agent=agent, project_scope=scope, cwd="/repo", git_branch="main",
        git_head_before="a" * 40, started_at=started,
    )
    for path in files:
        ledger.record_file_event(session_id, path=path, op="edited", tool="Edit",
                                 at=started + 60)
    if subject:
        ledger.record_commit(session_id, sha="c7f1a9d4e2", subject=subject,
                             files_changed=list(files), at=started + 120)
    ledger.end_session(session_id, git_head_after=head_after or "b" * 40,
                       ended_at=started + 180)
    return session_id


# ---------- the block is about one project ----------


def test_the_header_names_the_project_when_scoped(store, ledger):
    _finished_session(ledger)

    block = recall(store, scope=SCOPE)

    assert block.splitlines()[0].startswith("# Memory (nenapu)")
    assert SCOPE in block.splitlines()[0]


def test_an_unscoped_call_keeps_the_old_header(store):
    """Back-compat: the MCP surface and `nenapu rules` both render this block
    without a project."""
    store.write(Fact(text="The repo uses uv.", kind=Kind.PROJECT, confidence=0.9))

    assert recall(store).splitlines()[0] == "# Memory (nenapu)"


def test_facts_from_another_project_are_not_injected(store):
    """The reported bug, stated as a test: a session in one repo was being
    told about another repo's Ollama context window."""
    store.write(Fact(text="Ollama defaults to CONTEXT 4096.", scope=OTHER,
                     kind=Kind.ENVIRONMENT, confidence=0.9))
    store.write(Fact(text="Bookings use an overlap constraint.", scope=SCOPE,
                     kind=Kind.PROJECT, confidence=0.9))

    block = recall(store, scope=SCOPE)

    assert "overlap constraint" in block
    assert "Ollama" not in block


def test_global_facts_are_still_injected_alongside_the_project(store):
    """Scope is two-tier (Task 1): how the user works travels with them, what
    a repo does stays in the repo."""
    store.write(Fact(text="Never add a Claude co-author trailer.", scope="global",
                     kind=Kind.FEEDBACK, confidence=0.9))
    store.write(Fact(text="Bookings use an overlap constraint.", scope=SCOPE,
                     kind=Kind.PROJECT, confidence=0.9))

    block = recall(store, scope=SCOPE)

    assert "co-author" in block
    assert "overlap constraint" in block


# ---------- section 1: where you left off ----------


def test_where_you_left_off_names_the_files_and_the_agent(store, ledger):
    _finished_session(ledger, files=("backend/app/bookings.py", "backend/app/models.py"))

    block = recall(store, scope=SCOPE)

    assert "Where you left off" in block
    assert "backend/app/bookings.py" in block
    assert "claude-code" in block


def test_where_you_left_off_shows_the_last_commit_subject(store, ledger):
    _finished_session(ledger)

    assert "Add booking overlap constraint" in recall(store, scope=SCOPE)


def test_where_you_left_off_reads_the_latest_session_only(store, ledger):
    """Two sessions in the same repo: the block is "where you left off", not
    a changelog."""
    _finished_session(ledger, ago=10 * DAY, files=("old.py",), subject="old work")
    _finished_session(ledger, ago=1 * DAY, files=("new.py",), subject="new work")

    block = recall(store, scope=SCOPE)

    assert "new.py" in block
    assert "old.py" not in block


def test_another_projects_session_is_not_where_you_left_off(store, ledger):
    _finished_session(ledger, scope=OTHER, ago=1 * DAY, files=("portfolio/app.tsx",))
    _finished_session(ledger, scope=SCOPE, ago=5 * DAY, files=("backend/app/bookings.py",))

    block = recall(store, scope=SCOPE)

    assert "backend/app/bookings.py" in block
    assert "portfolio/app.tsx" not in block


def test_the_section_is_absent_when_the_project_has_no_history(store):
    """An empty header is worse than no header: it spends tokens to say
    nothing, on every request."""
    store.write(Fact(text="Bookings use an overlap constraint.", scope=SCOPE,
                     confidence=0.9))

    block = recall(store, scope=SCOPE)

    assert "overlap constraint" in block
    assert "Where you left off" not in block


def test_the_file_list_is_capped(store, ledger):
    """A refactor session can touch two hundred files. This block is prepended
    to every session, so no section may be unbounded."""
    from nenapu.observer import MAX_LEFT_OFF_FILES

    _finished_session(ledger, files=tuple(f"src/mod{i}.py" for i in range(60)))

    block = recall(store, scope=SCOPE)

    assert sum(1 for i in range(60) if f"src/mod{i}.py" in block) <= MAX_LEFT_OFF_FILES


# ---------- section 3: changed since you were last here ----------


def test_changed_since_you_were_last_here_diffs_from_the_last_head(store, ledger, tmp_path):
    """"the single highest-value item for context switching and it costs one
    git call" — from the last session's `git_head_after` to HEAD now."""
    from nenapu.capture import git_head
    from nenapu.store import project_scope

    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q", "-b", "main")
    (repo / "models.py").write_text("v1\n")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "init")
    left_at = git_head(str(repo))

    scope = project_scope(str(repo))
    _finished_session(ledger, scope=scope, head_after=left_at, files=("models.py",))

    (repo / "handlers.py").write_text("new\n")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "someone else's work")

    block = recall(store, scope=scope, cwd=str(repo))

    assert "Changed since you were last here" in block
    assert "handlers.py" in block


def test_nothing_changed_means_no_section(store, ledger, tmp_path):
    from nenapu.capture import git_head
    from nenapu.store import project_scope

    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q", "-b", "main")
    (repo / "models.py").write_text("v1\n")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "init")

    scope = project_scope(str(repo))
    _finished_session(ledger, scope=scope, head_after=git_head(str(repo)))

    assert "Changed since you were last here" not in recall(store, scope=scope, cwd=str(repo))


def test_a_missing_or_stale_head_does_not_break_the_block(store, ledger, tmp_path):
    """A rebase or a pruned branch can make the recorded head unreachable. The
    hook must still emit everything else — a SessionStart hook that raises
    means the session starts with no memory at all."""
    from nenapu.store import project_scope

    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q", "-b", "main")
    (repo / "a.py").write_text("v1\n")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "init")

    scope = project_scope(str(repo))
    _finished_session(ledger, scope=scope, head_after="0" * 40)
    store.write(Fact(text="Never add a co-author trailer.", kind=Kind.FEEDBACK,
                     confidence=0.9))

    block = recall(store, scope=scope, cwd=str(repo))

    assert "co-author" in block


def test_the_changed_list_is_capped(store, ledger, tmp_path):
    from nenapu.capture import git_head
    from nenapu.observer import MAX_CHANGED
    from nenapu.store import project_scope

    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q", "-b", "main")
    (repo / "seed.py").write_text("v1\n")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "init")
    left_at = git_head(str(repo))

    scope = project_scope(str(repo))
    _finished_session(ledger, scope=scope, head_after=left_at)

    for i in range(40):
        (repo / f"f{i}.py").write_text("x\n")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "big change")

    block = recall(store, scope=scope, cwd=str(repo))

    assert sum(1 for i in range(40) if f"f{i}.py" in block) <= MAX_CHANGED


# ---------- section 2: open loops, rendered here, captured in Task 11 ----------


def test_open_loops_for_this_project_are_listed(store, ledger):
    from nenapu.loops import LoopBook

    LoopBook(store.conn).open_loop(
        scope=SCOPE, text="Rate limiting on the public availability endpoint",
        resolution_hint="backend/app/ratelimit*",
    )
    _finished_session(ledger)

    block = recall(store, scope=SCOPE)

    assert "Open here" in block
    assert "Rate limiting" in block


def test_another_projects_open_loops_are_not_listed(store, ledger):
    from nenapu.loops import LoopBook

    LoopBook(store.conn).open_loop(scope=OTHER, text="Ship the portfolio hero animation")
    _finished_session(ledger)

    assert "hero animation" not in recall(store, scope=SCOPE)


def test_a_closed_loop_is_never_mentioned_again(store, ledger):
    """"If the agent claims you missed something you already shipped, trust is
    gone immediately and permanently.\""""
    from nenapu.loops import LoopBook

    book = LoopBook(store.conn)
    loop_id = book.open_loop(scope=SCOPE, text="Rate limiting on the endpoint")
    book.close_loop(loop_id, reason="commit touched backend/app/ratelimit.py")
    _finished_session(ledger)

    assert "Rate limiting" not in recall(store, scope=SCOPE)


# ---------- the block as a whole ----------


def test_the_four_sections_appear_in_the_documented_order(store, ledger):
    from nenapu.loops import LoopBook

    LoopBook(store.conn).open_loop(scope=SCOPE, text="Rate limiting on the endpoint")
    _finished_session(ledger)
    store.write(Fact(text="Never add a co-author trailer.", kind=Kind.FEEDBACK,
                     confidence=0.9))

    block = recall(store, scope=SCOPE)

    assert block.index("Where you left off") < block.index("Open here")
    assert block.index("Open here") < block.index("Previously corrected")


def test_the_whole_block_stays_within_a_measured_budget(store, ledger):
    """"Four sections will blow past today's `MAX_INJECTED = 12`, and this
    block is prepended to every session." The point of per-section caps is
    that a pathological store still produces a bounded block."""
    from nenapu.loops import LoopBook

    book = LoopBook(store.conn)
    for i in range(50):
        book.open_loop(scope=SCOPE, text=f"loop number {i} that was never finished")
        store.write(Fact(text=f"Correction number {i} about how to work here.",
                         scope=SCOPE, kind=Kind.FEEDBACK, confidence=0.9))
    _finished_session(ledger, files=tuple(f"src/mod{i}.py" for i in range(80)))

    block = recall(store, scope=SCOPE)

    assert len(block) < 4000


def test_suspect_facts_are_still_listed_under_their_warning(store, ledger):
    """Regression on the rule the implementation notes call out: suspect facts
    are exempt from the confidence floor, and scoping must not quietly
    reintroduce the filter."""
    from nenapu.models import Status

    fact, _ = store.write(Fact(text="The DB port is 5432.", scope=SCOPE, confidence=0.9))
    store.conn.execute("UPDATE facts SET status=? WHERE id=?", (Status.SUSPECT, fact.id))

    block = recall(store, scope=SCOPE)

    assert "Do not rely on these" in block
    assert "5432" in block


def test_the_injected_facts_are_still_logged_as_recalls(store, ledger):
    """Regression on Task 14: the hook path is the only path that logs
    recalls, and it is the path this task rewrites."""
    store.write(Fact(text="Bookings use an overlap constraint.", scope=SCOPE,
                     confidence=0.9))

    recall(store, scope=SCOPE, session_id="s-1")

    rows = store.conn.execute(
        "SELECT COUNT(*) AS n FROM recalls WHERE session_id = 's-1'"
    ).fetchone()
    assert rows["n"] >= 1


# ---------- the wiring that actually fixes the reported bug ----------


def test_the_session_start_hook_scopes_itself_to_the_repo_it_starts_in(tmp_path):
    """`cli.py` calls `recall_context(store, session_id=...)` with no scope.
    Everything above is inert until this line changes, so it is tested
    through the real CLI rather than against the function."""
    from nenapu.store import project_scope

    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q", "-b", "main")
    (repo / "a.py").write_text("v1\n")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "init")

    db = tmp_path / "s.db"
    store = Store(connect(str(db)))
    store.write(Fact(text="Bookings use an overlap constraint.",
                     scope=project_scope(str(repo)), confidence=0.9))
    store.write(Fact(text="Ollama defaults to CONTEXT 4096.", scope=OTHER,
                     kind=Kind.ENVIRONMENT, confidence=0.9))
    store.conn.close()

    result = subprocess.run(
        [sys.executable, "-m", "nenapu.cli", "recall-hook", "--db", str(db)],
        cwd=repo, input=json.dumps({"session_id": "s-1", "cwd": str(repo)}),
        capture_output=True, text=True,
        env={**os.environ, "PYTHONPATH": os.path.abspath("src"), "NENAPU_NO_BANNER": "1"},
    )

    assert result.returncode == 0
    assert "overlap constraint" in result.stdout
    assert "Ollama" not in result.stdout


def recall(store, **kwargs):
    from nenapu.observer import recall_context

    return recall_context(store, **kwargs)


# ==========================================================================
# Pre-written for R3 · injection budget in tokens, and R4 · injection
# anchored on the work at hand. Both held until G9's verdict is recorded,
# because both change what gets injected and shipping either mid-measurement
# moves the thing being measured.
#
# R3. Every cap in the injection path is a count: MAX_INJECTED = 12,
# MAX_SUSPECT_INJECTED = 5, MAX_LEFT_OFF_FILES = 6, MAX_OPEN_LOOPS = 5,
# MAX_CHANGED = 8. The block is prepended to every session and paid for on
# every request, so a count is the wrong unit — twelve facts is 200 tokens or
# 2000 depending on what got written. Replace the count caps with a token
# budget across the whole block, sections keeping their relative priority.
#
# R4. `recall_context` sorts by `(kind != FEEDBACK, -occurrences,
# -confidence)`. Nothing about cwd, branch, or which files the last sessions
# touched, so the block is deterministic and identical for every session in a
# repo until the store itself changes — the mechanical reason it reads as a
# dump. Anchor it on what the activity ledger already holds. E7 extends this
# anchor through the entity graph rather than replacing it.
#
# Assumed seams: `observer.INJECTION_TOKEN_BUDGET` and an
# `observer._token_estimate(text)` the budget is counted in.
# ==========================================================================

r3 = pytest.mark.xfail(strict=True, reason="R3 not implemented yet: remove when it lands")
r4 = pytest.mark.xfail(strict=True, reason="R4 not implemented yet: remove when it lands")


def _fact_lines(block):
    return [line for line in block.splitlines() if line.startswith("- ")]


def _known_lines(block):
    """Only the facts, not the ledger sections above them.

    "Where you left off" already varies with what was edited; a test satisfied
    by that would pass today while the memory half of the block stayed
    identical for every session in the repo.
    """
    if "Known about this work:" not in block:
        return []
    tail = block.split("Known about this work:", 1)[1]
    lines = []
    for line in tail.splitlines():
        if line.startswith("- "):
            lines.append(line)
        elif lines:
            break
    return lines


def _tokens(block):
    """The budget's own unit, so a test cannot pass by measuring a different
    thing from the implementation."""
    from nenapu.observer import _token_estimate

    return _token_estimate(block)


# ---------- R3 · the budget is tokens, not rows ----------


@r3
def test_a_block_of_long_facts_costs_no_more_than_a_block_of_short_ones(store):
    """Twelve facts is 200 tokens or 2000 depending on what got written, and
    the difference is paid on every request of every session."""
    from nenapu.observer import INJECTION_TOKEN_BUDGET

    for i in range(12):
        store.write(Fact(text=f"a long fact number {i}: " + "with a great deal of detail " * 30,
                         kind=Kind.PROJECT, confidence=0.9))

    block = recall(store)

    assert _tokens(block) <= INJECTION_TOKEN_BUDGET


@r3
def test_the_budget_is_never_exceeded_by_any_section(store, ledger):
    """Facts, warnings, open loops, changed files, where you left off: five
    sections that each used to have their own count cap and no shared
    ceiling."""
    from nenapu.observer import INJECTION_TOKEN_BUDGET
    from nenapu.loops import LoopBook

    _finished_session(ledger, files=tuple(f"backend/app/file_{i}.py" for i in range(40)))
    book = LoopBook(store.conn)
    for i in range(20):
        book.open_loop(scope=SCOPE, text=f"an open loop number {i} " + "with detail " * 20)
    for i in range(30):
        store.write(Fact(text=f"a fact number {i} " + "with detail " * 20, scope=SCOPE,
                         kind=Kind.PROJECT, confidence=0.9))

    block = recall(store, scope=SCOPE)

    assert _tokens(block) <= INJECTION_TOKEN_BUDGET


def test_corrections_are_not_starved_by_a_long_changed_files_list(store, ledger):
    """Section priority holds under pressure. A refactor that touched two
    hundred files must not be the reason a correction the user has repeated
    five times falls out of the block."""
    _finished_session(ledger, files=tuple(f"backend/app/file_{i}.py" for i in range(60)))
    store.write(Fact(text="commit messages never carry a co-author trailer",
                     kind=Kind.FEEDBACK, confidence=0.95))
    for i in range(30):
        store.write(Fact(text=f"a project fact number {i} " + "with detail " * 20,
                         scope=SCOPE, kind=Kind.PROJECT, confidence=0.9))

    block = recall(store, scope=SCOPE)

    assert "co-author" in block


@r3
def test_one_fact_longer_than_the_whole_budget_is_truncated_not_dropped(store):
    """A block that disappears because one row is enormous is a session that
    starts knowing nothing, which is the failure mode every guard in this
    path exists to avoid."""
    from nenapu.observer import INJECTION_TOKEN_BUDGET

    store.write(Fact(text="the deploy rule: " + "one more clause " * 4000,
                     kind=Kind.FEEDBACK, confidence=0.95))

    block = recall(store)

    assert "the deploy rule" in block
    assert _tokens(block) <= INJECTION_TOKEN_BUDGET


@r3
def test_short_facts_are_not_cut_at_twelve_when_the_budget_allows_more(store):
    """The count cap was a proxy for cost. Once cost is measured directly, a
    store of one-line facts should be allowed to send more of them."""
    for i in range(30):
        store.write(Fact(text=f"port {8000 + i} is taken", kind=Kind.ENVIRONMENT,
                         confidence=0.9))

    block = recall(store)

    assert len(_fact_lines(block)) > 12


@r3
def test_the_measured_budget_is_written_down(store):
    """The plan asks this task to answer the "Token cost" section of
    IMPLEMENTATION_NOTES.md with a measured number rather than an intention."""
    from pathlib import Path

    from nenapu.observer import INJECTION_TOKEN_BUDGET

    notes = (Path(__file__).resolve().parent.parent / "IMPLEMENTATION_NOTES.md").read_text()

    assert str(INJECTION_TOKEN_BUDGET) in notes


# ---------- R4 · anchored on the work at hand ----------


def _touched(ledger, paths, *, scope=SCOPE, ago=DAY):
    return _finished_session(ledger, scope=scope, files=tuple(paths), ago=ago,
                             subject=None)


@r4
def test_two_sessions_in_one_repo_with_different_recent_files_differ(store, ledger):
    """The mechanical reason the block reads as a dump: it is identical for
    every session in a repo until the store itself changes."""
    store.write(Fact(text="the bookings module owns the overlap constraint",
                     scope=SCOPE, kind=Kind.PROJECT, confidence=0.9))
    store.write(Fact(text="the invoices module rounds to two decimals",
                     scope=SCOPE, kind=Kind.PROJECT, confidence=0.9))

    _touched(ledger, ["backend/app/bookings.py"])
    on_bookings = _known_lines(recall(store, scope=SCOPE))
    _touched(ledger, ["backend/app/invoices.py"], ago=60.0)
    on_invoices = _known_lines(recall(store, scope=SCOPE))

    assert on_bookings and on_bookings != on_invoices


@r4
def test_a_fact_about_a_recently_edited_file_leads_the_others(store, ledger):
    store.write(Fact(text="the invoices module rounds to two decimals",
                     scope=SCOPE, kind=Kind.PROJECT, confidence=0.95))
    store.write(Fact(text="the bookings module owns the overlap constraint",
                     scope=SCOPE, kind=Kind.PROJECT, confidence=0.9))
    _touched(ledger, ["backend/app/bookings.py"])

    block = recall(store, scope=SCOPE)

    assert block.index("bookings module") < block.index("invoices module")


@r4
def test_the_branch_is_part_of_the_anchor(store, ledger):
    """`sessions.git_branch` is already recorded and already unused. Work on
    `release-3` is different work from work on `main`."""
    from nenapu.models import now

    store.write(Fact(text="release-3 is cut from main every second Tuesday",
                     scope=SCOPE, kind=Kind.PROJECT, confidence=0.9))
    store.write(Fact(text="the invoices module rounds to two decimals",
                     scope=SCOPE, kind=Kind.PROJECT, confidence=0.95))
    session = ledger.start_session(agent="claude-code", project_scope=SCOPE, cwd="/repo",
                                   git_branch="release-3", started_at=now() - 600)
    ledger.end_session(session, ended_at=now() - 300)

    block = recall(store, scope=SCOPE)

    assert block.index("release-3") < block.index("invoices module")


def test_the_anchor_does_not_reach_into_another_project(store, ledger):
    """Anchoring reads the activity ledger, which spans every repo on the
    machine. Scope is what keeps that from undoing the fix it already made."""
    store.write(Fact(text="the portfolio site is deployed from netlify",
                     scope=OTHER, kind=Kind.PROJECT, confidence=0.95))
    store.write(Fact(text="the bookings module owns the overlap constraint",
                     scope=SCOPE, kind=Kind.PROJECT, confidence=0.9))
    _touched(ledger, ["portfolio/src/index.astro"], scope=OTHER)

    block = recall(store, scope=SCOPE)

    assert "netlify" not in block


def test_corrections_still_lead_the_block(store, ledger):
    """Anchoring reorders what follows the corrections; it does not demote
    them. A correction the user repeated is still the most actionable line in
    the block whatever files were edited yesterday."""
    store.write(Fact(text="do not add a Claude co-author trailer", kind=Kind.FEEDBACK,
                     confidence=0.7))
    store.write(Fact(text="the bookings module owns the overlap constraint",
                     scope=SCOPE, kind=Kind.PROJECT, confidence=0.95))
    _touched(ledger, ["backend/app/bookings.py"])

    block = recall(store, scope=SCOPE)

    assert block.index("co-author") < block.index("bookings module")


def test_with_no_activity_history_the_block_is_todays_block(store):
    """A fresh store has no anchor to read, and must be unaffected — which is
    what keeps every existing test in this file true."""
    store.write(Fact(text="The repo uses uv.", kind=Kind.PROJECT, confidence=0.95))
    store.write(Fact(text="Do not add a Claude co-author trailer.", kind=Kind.FEEDBACK,
                     confidence=0.7))

    block = recall(store, scope=SCOPE)

    assert block.splitlines()[0] == f"# Memory (nenapu) — {SCOPE}"
    assert "co-author" in block


# ==========================================================================
# Pre-written for the injection half of E7 · entity-anchored retrieval.
#
# R4 anchors the block on cwd, branch and recently edited files. E7 extends
# that anchor through the entity graph — traverse `entity_edges` to depth 2
# with per-hop decay, join through `fact_entities` to candidate facts — rather
# than replacing it, which is why E7 depends on R4. The scoring half is pinned
# in tests/test_store.py.
#
# Held until G9's verdict is recorded. The verdict also scopes the task: a
# `coverage-problem` or a high injection unused-rate means anchoring is the
# fix and vectors may never be needed.
# ==========================================================================

e7 = pytest.mark.xfail(strict=True, reason="E7 not implemented yet: remove when it lands")


@e7
def test_the_block_reaches_a_fact_about_a_neighbouring_file(store, ledger):
    """Graph distance, not a second lexical pass: the session edited
    `bookings.py`, and the fact worth injecting is about the module it is
    always changed with."""
    from nenapu.entities import EntityGraph

    graph = EntityGraph(store.conn)
    neighbour_fact, _ = store.write(Fact(text="availability windows are half-open",
                                         scope=SCOPE, kind=Kind.PROJECT, confidence=0.6))
    store.write(Fact(text="an unrelated fact about the mailer", scope=SCOPE,
                     kind=Kind.PROJECT, confidence=0.6))
    edited = graph.upsert(kind="file", name="backend/app/bookings.py", scope=SCOPE)
    neighbour = graph.upsert(kind="file", name="backend/app/availability.py", scope=SCOPE)
    graph.link(edited.id, neighbour.id, kind="touched_with", source="observed")
    graph.attach(neighbour_fact.id, neighbour.id, role="subject", source="path")
    _touched(ledger, ["backend/app/bookings.py"])

    block = recall(store, scope=SCOPE)

    assert block.index("availability windows") < block.index("unrelated fact about the mailer")


@e7
def test_proximity_does_not_promote_a_falsified_fact(store, ledger):
    """The belief layer stays after ranking, as filter and warning, exactly as
    the block does today. Being near the work is not a reason to believe
    something whose foundation collapsed."""
    from nenapu.entities import EntityGraph
    from nenapu.models import Status

    graph = EntityGraph(store.conn)
    doubted, _ = store.write(Fact(text="availability windows are half-open", scope=SCOPE,
                                  kind=Kind.PROJECT, confidence=0.9))
    store.set_status(doubted.id, Status.SUSPECT)
    entity = graph.upsert(kind="file", name="backend/app/bookings.py", scope=SCOPE)
    graph.attach(doubted.id, entity.id, role="subject", source="path")
    _touched(ledger, ["backend/app/bookings.py"])

    block = recall(store, scope=SCOPE)

    assert "falsified" in block.lower()
    assert block.index("falsified") < block.index("availability windows")


@e7
def test_the_anchor_does_not_traverse_out_of_the_project(store, ledger):
    """Traversal must not cross scope except through `global`, or this
    recreates the "right fact, wrong project" failure scoping already fixed."""
    from nenapu.entities import EntityGraph

    graph = EntityGraph(store.conn)
    elsewhere, _ = store.write(Fact(text="the portfolio deploys from netlify",
                                    scope=OTHER, kind=Kind.PROJECT, confidence=0.9))
    here = graph.upsert(kind="file", name="backend/app/bookings.py", scope=SCOPE)
    there = graph.upsert(kind="file", name="portfolio/src/index.astro", scope=OTHER)
    graph.link(here.id, there.id, kind="touched_with", source="observed")
    graph.attach(elsewhere.id, there.id, role="subject", source="path")
    _touched(ledger, ["backend/app/bookings.py"])

    block = recall(store, scope=SCOPE)

    assert "netlify" not in block


def test_a_store_with_no_entities_renders_the_block_it_renders_today(store, ledger):
    """With no entity data the scoring degrades to R4's behaviour, which is
    what keeps every existing test in this file true unmodified."""
    store.write(Fact(text="Bookings use an overlap constraint.", scope=SCOPE,
                     kind=Kind.PROJECT, confidence=0.9))
    _finished_session(ledger)

    block = recall(store, scope=SCOPE)

    assert "Where you left off" in block
    assert "overlap constraint" in block
