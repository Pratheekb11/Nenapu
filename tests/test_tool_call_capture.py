"""Capturing what a session actually did: tool calls, and git for deletes.

Requirement (Task 4, priority-ordered task list, marked **Opus 5**):

    "Capture tool calls — stop discarding `tool_use` in `_turns_from`; add
    git `HEAD` diffing for deletes | the data is already in the transcript
    and is being thrown away | M | Opus 5 | depends on 3"

Task 3 built the tables; nothing fills them from a live session yet, and
`backfill.py` reads only `Write`/`Edit`/`NotebookEdit`. The plan's "Proposed
tier: the activity ledger" section names two sources and ranks them:

    1. Git — `git diff --name-status <before>..<after>` for the authoritative
       created/modified/**deleted** set. "Deletion is the one op that tool
       calls alone cannot reliably give you: files die via `Bash rm` or
       `git rm`, and parsing shell strings for that is fragile."
    2. Tool calls — `Edit`/`Write`/`NotebookEdit` `file_path` for per-action
       attribution and ordering, "including edits later reverted and files
       touched outside git".

The plan marks this Opus 5 for one reason, quoted directly: "git edge cases
decide correctness: renames, detached HEAD, merge commits, and **worktrees**
(`~/.claude/projects/` already contains a `-claude-worktrees-` entry, so this
is real, not hypothetical)." Each of those four has a test below.

Proposed seam
-------------
A new module `nenapu.capture`, holding the two sources behind one surface:

    TOOL_OPS: dict[str, str]                       # tool name -> ledger op
    file_events_from(lines, *, cwd=None) -> list[dict]   # {path, op, tool, at}
    git_head(cwd) -> str | None
    git_branch(cwd) -> str | None                  # None when detached
    changed_paths(cwd, before, after) -> list[tuple[str, str]]   # (op, path)
    capture_session(ledger, transcript, *, agent, cwd=None) -> int | None

`backfill.py` is expected to route through the same parser rather than keep
its own `_TOOL_OP` table, which is what the backfill test at the bottom
pins. The module name is this file's proposal for the contract, in the same
way `tests/test_activity_ledger.py` proposed `nenapu.activity`.

Remove the `pytestmark` line below when Task 4 lands: until then these are
the specification, and the suite reports them as expected failures rather
than as breakage.
"""

import json
import os
import subprocess

import pytest

from nenapu import connect

# ---------- transcript fixtures, shaped like the real thing ----------


def _tool_use(name: str, inputs: dict, *, timestamp: str | None = None) -> str:
    event = {
        "type": "assistant",
        "message": {"role": "assistant", "content": [
            {"type": "tool_use", "id": "t1", "name": name, "input": inputs},
        ]},
    }
    if timestamp:
        event["timestamp"] = timestamp
    return json.dumps(event)


def _meta(session_id="s-1", cwd="/repo", branch="main") -> str:
    return json.dumps({
        "type": "user", "sessionId": session_id, "cwd": cwd, "gitBranch": branch,
        "message": {"role": "user", "content": [{"type": "text", "text": "go"}]},
    })


def _git(repo, *args, check=True):
    return subprocess.run(
        ["git", *args], cwd=repo, check=check, capture_output=True, text=True,
        env={"GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@example.com",
             "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@example.com",
             "PATH": os.environ.get("PATH", ""), "HOME": str(repo)},
    )


@pytest.fixture
def repo(tmp_path):
    path = tmp_path / "repo"
    path.mkdir()
    _git(path, "init", "-q", "-b", "main")
    (path / "kept.py").write_text("keep\n")
    (path / "doomed.py").write_text("doomed\n")
    _git(path, "add", "-A")
    _git(path, "commit", "-q", "-m", "init")
    return path


@pytest.fixture
def ledger():
    from nenapu.activity import ActivityLedger

    return ActivityLedger(connect(":memory:"))


# ---------- source 2: tool calls ----------


def test_every_file_touching_tool_maps_to_a_ledger_op():
    """`op` is constrained by the schema to created | edited | deleted | read,
    so a tool table that invents a fifth value would write rows no query
    understands."""
    from nenapu.capture import TOOL_OPS

    assert set(TOOL_OPS.values()) <= {"created", "edited", "deleted", "read"}
    assert TOOL_OPS["Write"] == "created"
    assert TOOL_OPS["Edit"] == "edited"
    assert TOOL_OPS["NotebookEdit"] == "edited"


def test_a_read_is_captured_as_its_own_op():
    """The schema has always had `read`; the backfill parser never emitted it.
    "which agent looked at this file" is half of `nenapu where`."""
    from nenapu.capture import file_events_from

    events = file_events_from([_tool_use("Read", {"file_path": "/repo/a.py"})])

    assert [(e["path"], e["op"]) for e in events] == [("/repo/a.py", "read")]


def test_tool_name_is_kept_for_attribution():
    from nenapu.capture import file_events_from

    events = file_events_from([_tool_use("Edit", {"file_path": "/repo/a.py"})])

    assert events[0]["tool"] == "Edit"


def test_tools_that_touch_no_file_are_ignored():
    """`Bash` dominates real transcripts (27 of 29 tool calls in the measured
    sample). Recording it as a file event would drown the ledger."""
    from nenapu.capture import file_events_from

    lines = [
        _tool_use("Bash", {"command": "rm -rf /repo/doomed.py"}),
        _tool_use("Grep", {"pattern": "x"}),
        _tool_use("Edit", {"file_path": "/repo/a.py"}),
    ]

    assert len(file_events_from(lines)) == 1


def test_order_of_actions_is_preserved():
    """Ordering is the whole reason tool calls are kept alongside git: git
    knows the net effect, only the transcript knows the sequence."""
    from nenapu.capture import file_events_from

    lines = [
        _tool_use("Write", {"file_path": "/repo/a.py"}),
        _tool_use("Edit", {"file_path": "/repo/b.py"}),
        _tool_use("Edit", {"file_path": "/repo/a.py"}),
    ]

    assert [e["path"] for e in file_events_from(lines)] == [
        "/repo/a.py", "/repo/b.py", "/repo/a.py",
    ]


def test_a_relative_path_is_resolved_against_the_session_cwd():
    """`nenapu where backend/app/models.py` has to match rows written by two
    different sessions in the same repo, so paths cannot be stored however
    each tool call happened to spell them."""
    from nenapu.capture import file_events_from

    events = file_events_from(
        [_tool_use("Edit", {"file_path": "src/a.py"})], cwd="/repo",
    )

    assert events[0]["path"] == "/repo/src/a.py"


def test_the_transcript_timestamp_is_used_when_present():
    """`standup` and `activity` order by `at`. Stamping ingestion time would
    make a backfill of 232 transcripts look like one busy afternoon."""
    from nenapu.capture import file_events_from

    events = file_events_from(
        [_tool_use("Edit", {"file_path": "/repo/a.py"},
                   timestamp="2026-08-20T10:00:00.000Z")]
    )

    assert events[0]["at"] == pytest.approx(1787220000.0, abs=1.0)


def test_malformed_and_partial_events_are_skipped_not_fatal():
    """A tail read starts mid-line by construction, and a killed session
    leaves a truncated one at the end."""
    from nenapu.capture import file_events_from

    lines = [
        "", "{not json", "null",
        json.dumps({"type": "assistant", "message": {"content": "plain string"}}),
        _tool_use("Edit", {}),                      # no file_path
        _tool_use("Edit", {"file_path": ""}),       # empty file_path
        _tool_use("Edit", {"file_path": "/repo/a.py"}),
    ]

    assert [e["path"] for e in file_events_from(lines)] == ["/repo/a.py"]


# ---------- source 1: git, which is where the edge cases live ----------


def test_git_head_reads_the_current_commit(repo):
    from nenapu.capture import git_head

    assert git_head(str(repo)) == _git(repo, "rev-parse", "HEAD").stdout.strip()


def test_git_head_outside_a_repo_is_none_not_an_error(tmp_path):
    """Sessions run in `/tmp` and in `~/Downloads`. A capture path that raises
    outside a repo would lose the tool events too."""
    from nenapu.capture import git_head

    assert git_head(str(tmp_path)) is None


def test_a_deleted_file_is_only_visible_to_git(repo):
    """The headline case: the file died via `Bash rm`, so no tool call names
    it, and the plan refuses to parse shell strings for deletes."""
    from nenapu.capture import changed_paths, git_head

    before = git_head(str(repo))
    (repo / "doomed.py").unlink()
    _git(repo, "commit", "-q", "-am", "drop it")
    after = git_head(str(repo))

    assert ("deleted", "doomed.py") in changed_paths(str(repo), before, after)


def test_a_modified_file_is_edited_and_a_new_one_is_created(repo):
    from nenapu.capture import changed_paths, git_head

    before = git_head(str(repo))
    (repo / "kept.py").write_text("changed\n")
    (repo / "fresh.py").write_text("new\n")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "work")
    after = git_head(str(repo))

    changed = dict((path, op) for op, path in changed_paths(str(repo), before, after))
    assert changed["kept.py"] == "edited"
    assert changed["fresh.py"] == "created"


def test_a_rename_becomes_a_delete_and_a_create(repo):
    """Edge case 1 of 4. `git diff --name-status` reports `R100 old new` when
    rename detection is on; the ledger's op vocabulary has no `renamed`, and
    "where did models.py go" must be answerable from both ends."""
    from nenapu.capture import changed_paths, git_head

    before = git_head(str(repo))
    _git(repo, "mv", "kept.py", "moved.py")
    _git(repo, "commit", "-q", "-m", "move")
    after = git_head(str(repo))

    changed = changed_paths(str(repo), before, after)
    assert ("deleted", "kept.py") in changed
    assert ("created", "moved.py") in changed


def test_a_merge_commit_reports_the_net_change(repo):
    """Edge case 2 of 4. Diffing a merge commit against its first parent is
    the trap — `before..after` must be used, or everything the side branch
    did disappears from the ledger."""
    from nenapu.capture import changed_paths, git_head

    before = git_head(str(repo))
    _git(repo, "checkout", "-q", "-b", "side")
    (repo / "side.py").write_text("side\n")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "side work")
    _git(repo, "checkout", "-q", "main")
    (repo / "kept.py").write_text("main work\n")
    _git(repo, "commit", "-q", "-am", "main work")
    _git(repo, "merge", "--no-ff", "-q", "side", "-m", "merge side")
    after = git_head(str(repo))

    changed = dict((path, op) for op, path in changed_paths(str(repo), before, after))
    assert changed["side.py"] == "created"
    assert changed["kept.py"] == "edited"


def test_a_detached_head_has_a_commit_but_no_branch(repo):
    """Edge case 3 of 4. `git rev-parse --abbrev-ref HEAD` answers the literal
    string "HEAD" when detached, which would land in the ledger as a branch
    called HEAD and group unrelated sessions together."""
    from nenapu.capture import git_branch, git_head

    _git(repo, "checkout", "-q", "--detach")

    assert git_head(str(repo)) is not None
    assert git_branch(str(repo)) is None


def test_a_linked_worktree_is_captured_as_its_own_checkout(repo, tmp_path):
    """Edge case 4 of 4, and the one the plan says is already real on this
    machine. A worktree has its own HEAD and branch while sharing the object
    store, so both must be read from the worktree path, not the main repo."""
    from nenapu.capture import git_branch, git_head

    tree = tmp_path / "wt"
    _git(repo, "worktree", "add", "-q", "-b", "feature", str(tree))
    (tree / "kept.py").write_text("in the worktree\n")
    _git(tree, "commit", "-q", "-am", "worktree work")

    assert git_branch(str(tree)) == "feature"
    assert git_head(str(tree)) != git_head(str(repo))


def test_diffing_against_an_unknown_commit_is_empty_not_an_exception(repo):
    """A session's recorded `git_head_before` can vanish — a rebase, a reset,
    a pruned branch. Capture must degrade to "no git evidence", never take
    the session's tool events down with it."""
    from nenapu.capture import changed_paths, git_head

    assert changed_paths(str(repo), "0" * 40, git_head(str(repo))) == []


# ---------- the two sources, combined ----------


def test_capture_records_both_tool_events_and_git_deletes(repo, ledger):
    """The integration the task exists for: an `Edit` the transcript saw, and
    a `Bash rm` only git saw, both land on one session row."""
    from nenapu.capture import capture_session, git_head

    before = git_head(str(repo))
    transcript = repo / "t.jsonl"
    transcript.write_text("\n".join([
        _meta(cwd=str(repo)),
        _tool_use("Edit", {"file_path": str(repo / "kept.py")}),
        _tool_use("Bash", {"command": "rm doomed.py"}),
    ]))
    (repo / "kept.py").write_text("edited\n")
    (repo / "doomed.py").unlink()
    _git(repo, "commit", "-q", "-am", "work")

    session_id = capture_session(
        ledger, transcript, agent="claude-code", cwd=str(repo), git_head_before=before,
    )

    events = ledger.file_events_for_session(session_id)
    by_path = {e["path"].split("/")[-1]: e for e in events}
    assert by_path["kept.py"]["op"] == "edited"
    assert by_path["kept.py"]["tool"] == "Edit"
    assert by_path["doomed.py"]["op"] == "deleted"


def test_a_file_seen_by_both_sources_is_recorded_once(repo, ledger):
    """Git reports `kept.py` as modified and the transcript reports one Edit
    of it. Two rows for one action would inflate every `files_touched` count
    the rollups and `standup` are built on."""
    from nenapu.capture import capture_session, git_head

    before = git_head(str(repo))
    transcript = repo / "t.jsonl"
    transcript.write_text("\n".join([
        _meta(cwd=str(repo)),
        _tool_use("Edit", {"file_path": str(repo / "kept.py")}),
    ]))
    (repo / "kept.py").write_text("edited\n")
    _git(repo, "commit", "-q", "-am", "work")

    session_id = capture_session(
        ledger, transcript, agent="claude-code", cwd=str(repo), git_head_before=before,
    )

    kept = [e for e in ledger.file_events_for_session(session_id)
            if e["path"].endswith("kept.py")]
    assert len(kept) == 1


def test_capture_records_the_commits_the_session_made(repo, ledger):
    """`commits(session_id, sha, subject, files_changed, at)` has been empty
    since Task 3 created it — "what was implemented" is the question it
    answers."""
    from nenapu.capture import capture_session, git_head

    before = git_head(str(repo))
    (repo / "kept.py").write_text("edited\n")
    _git(repo, "commit", "-q", "-am", "Add booking overlap constraint")
    transcript = repo / "t.jsonl"
    transcript.write_text("\n".join([_meta(cwd=str(repo))]))

    session_id = capture_session(
        ledger, transcript, agent="claude-code", cwd=str(repo), git_head_before=before,
    )

    commits = ledger.commits_for_session(session_id)
    assert [c["subject"] for c in commits] == ["Add booking overlap constraint"]
    assert commits[0]["files_changed"] == ["kept.py"]


def test_capture_closes_the_session_with_the_head_it_ended_on(repo, ledger):
    """`git_head_after` is what "changed since you were last here" (Task 7)
    diffs from, and what the abrupt-stop check (Task 11) compares against
    `git_head_before`."""
    from nenapu.capture import capture_session, git_head

    before = git_head(str(repo))
    (repo / "kept.py").write_text("edited\n")
    _git(repo, "commit", "-q", "-am", "work")
    transcript = repo / "t.jsonl"
    transcript.write_text(_meta(cwd=str(repo)))

    session_id = capture_session(
        ledger, transcript, agent="claude-code", cwd=str(repo), git_head_before=before,
    )

    session = ledger.get_session(session_id)
    assert session["git_head_before"] == before
    assert session["git_head_after"] == git_head(str(repo))
    assert session["ended_at"] is not None


def test_capture_outside_a_repo_still_records_tool_events(tmp_path, ledger):
    """Sessions in a scratch directory are still work. Losing them because
    git had nothing to say would be the ledger silently under-reporting."""
    from nenapu.capture import capture_session

    transcript = tmp_path / "t.jsonl"
    transcript.write_text("\n".join([
        _meta(cwd=str(tmp_path)),
        _tool_use("Write", {"file_path": str(tmp_path / "notes.md")}),
    ]))

    session_id = capture_session(ledger, transcript, agent="claude-code", cwd=str(tmp_path))

    assert len(ledger.file_events_for_session(session_id)) == 1


def test_capture_is_idempotent_for_one_transcript(repo, ledger):
    """The Stop hook, the watcher and a backfill can all reach the same
    transcript. `external_id` already exists for exactly this."""
    from nenapu.capture import capture_session

    transcript = repo / "t.jsonl"
    transcript.write_text("\n".join([
        _meta(cwd=str(repo)),
        _tool_use("Edit", {"file_path": str(repo / "kept.py")}),
    ]))

    first = capture_session(ledger, transcript, agent="claude-code", cwd=str(repo))
    second = capture_session(ledger, transcript, agent="claude-code", cwd=str(repo))

    assert second is None
    assert len(ledger.file_events_for_session(first)) == 1


def test_backfill_uses_the_same_parser_and_now_sees_reads(tmp_path, ledger):
    """`backfill.py` keeps its own three-entry `_TOOL_OP` table today. Two
    parsers means the 232 backfilled sessions and every future session
    disagree about what a session contains."""
    from nenapu.backfill import backfill_transcript

    transcript = tmp_path / "t.jsonl"
    transcript.write_text("\n".join([
        _meta(cwd=str(tmp_path)),
        _tool_use("Read", {"file_path": "/repo/a.py"}),
        _tool_use("Edit", {"file_path": "/repo/a.py"}),
    ]))

    session_id = backfill_transcript(ledger, transcript, agent="claude-code")

    assert [e["op"] for e in ledger.file_events_for_session(session_id)] == ["read", "edited"]
