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

pytestmark = pytest.mark.xfail(
    reason="Task 7 (Opus 5) not implemented — tests written first", strict=False
)

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
