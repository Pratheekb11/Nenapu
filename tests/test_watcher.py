"""The watcher: capture from agents that have no hook API.

Requirement (Task 15, priority-ordered task list, marked **Opus 5** —
"reverse-engineering undocumented transcript formats per agent"):

    "The watcher — Codex/Gemini/Cursor adapters | multi-agent capture; only
    Claude Code feeds the store today | L | Opus 5 | depends on 8, 4"

From "Phase 4 — `nenapu watch`: provider-agnostic auto-ingestion":

    **Adapters.** "`observer._turns_from` is Claude-JSONL specific. Refactor
    it into a registry of `TranscriptFormat` adapters, each with a glob and a
    parser... Every adapter must be probed against a real file before it is
    written; do not ship a glob nobody has seen match."

    **Mechanism.** "Poll mtimes on a timer rather than adding a `watchdog`
    dependency... A session is 'finished' when its size and mtime have been
    stable for ~120s."

    **State.** "New table `watch_state(path PRIMARY KEY, agent, last_size,
    last_mtime, ingested_at, session_id)` so a file is extracted once and
    incrementally."

    **Reuse, do not duplicate.** "`redact()`, the growing-tail reader, and
    `observe_transcript` are all correct and stay as-is behind the adapter
    seam. Redaction must remain at the harvest boundary."

    **Overlap with the Stop hook.** "Default the watcher to skipping any agent
    whose hook is installed... The `(scope, text)` unique index would de-dupe
    the rows anyway, but not the 83-second model call."

    **Cost.** "Add a floor between extractions and a `--batch` mode that
    drains the queue on a schedule instead of instantly."

    **Supervision.** "`nenapu watch` runs foreground; `nenapu init --watch`
    installs a systemd user unit... consent-gated exactly like the hooks —
    TTY required, backup first, idempotent, non-TTY prints what it would do."

Scope note
----------
These tests deliberately assert **nothing about any specific glob for Codex,
Gemini, OpenCode or Cursor**. The plan forbids shipping a glob nobody has
seen match, so the per-agent formats are a probing exercise against real
files on a machine that has those agents installed — not something a test
can pin in advance without inventing the very thing the plan says to verify.
What is pinned here is everything that is provider-agnostic: the registry
shape, the finished-file rule, the state table, the hook-overlap skip, the
extraction floor, and the consent rules for supervision. The Claude Code
adapter is pinned in full, because its format is already parsed by
`observer._turns_from` and is on this machine.

Proposed seam
-------------
A new module `nenapu.watch`:

    TranscriptFormat(agent, glob, parse)
    ADAPTERS: list[TranscriptFormat]
    QUIET_SECONDS, MIN_SECONDS_BETWEEN_EXTRACTIONS
    discover(adapters=ADAPTERS) -> list[tuple[TranscriptFormat, Path]]
    is_finished(path, state, *, at=None) -> bool
    get_state(conn, path) / record_state(conn, ...)
    agents_with_hooks(settings_path) -> set[str]
    tick(conn, *, adapters=..., settings_path=..., at=..., batch=False) -> list[int]

`tick` enqueues into the Task 8 queue rather than extracting inline, which is
what makes the watcher and the Stop hook share one serialized worker.

Remove the `pytestmark` line below when Task 15 lands.
"""

import json
import os
import subprocess
import sys

import pytest

from nenapu import connect

pytestmark = pytest.mark.xfail(
    reason="Task 15 (Opus 5) not implemented — tests written first", strict=False
)


def _line(role: str, text: str) -> str:
    return json.dumps({
        "type": role, "sessionId": "s-1", "cwd": "/repo",
        "message": {"role": role, "content": [{"type": "text", "text": text}]},
    })


@pytest.fixture
def conn():
    return connect(":memory:")


@pytest.fixture
def transcript(tmp_path):
    path = tmp_path / "claude" / "projects" / "proj" / "s-1.jsonl"
    path.parent.mkdir(parents=True)
    path.write_text("\n".join([_line("user", "use pnpm not npm"), _line("assistant", "ok")]))
    return path


def _claude_adapter(root):
    from nenapu.watch import TranscriptFormat
    from nenapu.observer import _turns_from

    return TranscriptFormat(agent="claude-code", glob=f"{root}/**/*.jsonl", parse=_turns_from)


# ---------- the registry ----------


def test_the_registry_is_data_not_a_branch():
    """The point of the refactor: adding Codex is registering an adapter, not
    editing `observer`."""
    from nenapu.watch import ADAPTERS, TranscriptFormat

    assert ADAPTERS
    assert all(isinstance(adapter, TranscriptFormat) for adapter in ADAPTERS)


def test_claude_code_is_registered_and_reuses_the_existing_parser():
    from nenapu.observer import _turns_from
    from nenapu.watch import ADAPTERS

    claude = next(a for a in ADAPTERS if a.agent == "claude-code")
    lines = [_line("user", "use pnpm not npm")]
    assert claude.parse(lines) == _turns_from(lines)


def test_every_registered_glob_is_an_absolute_path():
    """A relative glob would resolve against whatever directory the watcher
    happened to be started from."""
    from nenapu.watch import ADAPTERS

    for adapter in ADAPTERS:
        assert os.path.isabs(os.path.expanduser(adapter.glob)), adapter.agent


def test_discovery_tolerates_an_agent_that_is_not_installed(tmp_path):
    """Most machines have one or two of these five. A missing directory is
    the normal case, not an error."""
    from nenapu.watch import TranscriptFormat, discover

    missing = TranscriptFormat(agent="codex", glob=str(tmp_path / "nope" / "**" / "*.jsonl"),
                               parse=lambda lines: [])

    assert discover([missing]) == []


def test_discovery_finds_a_transcript_and_names_its_agent(tmp_path, transcript):
    from nenapu.watch import discover

    found = discover([_claude_adapter(tmp_path / "claude")])

    assert [(adapter.agent, path) for adapter, path in found] == [("claude-code", transcript)]


def test_a_parser_given_a_file_it_does_not_understand_returns_nothing(tmp_path):
    """A misfiring glob must produce an empty harvest, not an exception that
    stops the watcher for every other agent."""
    from nenapu.watch import ADAPTERS

    claude = next(a for a in ADAPTERS if a.agent == "claude-code")

    assert claude.parse(["not json at all", "", "{"]) == []


# ---------- "finished" is a measurement, not a guess ----------


def test_a_file_that_just_changed_is_not_finished(conn, transcript):
    """A session in progress is written continuously; ingesting it would
    spend an 83-second extraction on half a conversation."""
    from nenapu.watch import is_finished, record_state
    from nenapu.models import now

    record_state(conn, path=str(transcript), agent="claude-code",
                 last_size=transcript.stat().st_size - 10, last_mtime=now())

    assert is_finished(transcript, conn, at=now()) is False


def test_a_file_stable_for_the_quiet_window_is_finished(conn, transcript):
    from nenapu.watch import QUIET_SECONDS, is_finished, record_state
    from nenapu.models import now

    seen_at = now() - QUIET_SECONDS - 1
    record_state(conn, path=str(transcript), agent="claude-code",
                 last_size=transcript.stat().st_size, last_mtime=seen_at, seen_at=seen_at)

    assert is_finished(transcript, conn, at=now()) is True


def test_a_file_seen_for_the_first_time_is_recorded_not_ingested(conn, tmp_path, transcript):
    """The first tick has nothing to compare against, so it can only start the
    clock."""
    from nenapu.watch import get_state, tick

    assert tick(conn, adapters=[_claude_adapter(tmp_path / "claude")], settings_path=None) == []
    assert get_state(conn, str(transcript)) is not None


# ---------- state, so a transcript is extracted once ----------


def test_the_watch_state_table_exists_with_the_documented_columns():
    columns = {r["name"] for r in connect(":memory:").execute("PRAGMA table_info(watch_state)")}
    assert columns >= {"path", "agent", "last_size", "last_mtime", "ingested_at", "session_id"}


def test_a_finished_transcript_is_enqueued_rather_than_extracted_inline(conn, tmp_path,
                                                                       transcript):
    """The watcher's whole job is to feed the Task 8 queue; one serialized
    worker is what keeps concurrent 83-second calls off the store."""
    from nenapu.models import now
    from nenapu.watch import QUIET_SECONDS, record_state, tick

    seen_at = now() - QUIET_SECONDS - 1
    record_state(conn, path=str(transcript), agent="claude-code",
                 last_size=transcript.stat().st_size, last_mtime=seen_at, seen_at=seen_at)

    tick(conn, adapters=[_claude_adapter(tmp_path / "claude")], settings_path=None)

    rows = [dict(r) for r in conn.execute("SELECT * FROM ingest_queue")]
    assert [r["path"] for r in rows] == [str(transcript)]
    assert rows[0]["agent"] == "claude-code"


def test_the_same_transcript_is_never_enqueued_twice(conn, tmp_path, transcript):
    from nenapu.models import now
    from nenapu.watch import QUIET_SECONDS, record_state, tick

    seen_at = now() - QUIET_SECONDS - 1
    record_state(conn, path=str(transcript), agent="claude-code",
                 last_size=transcript.stat().st_size, last_mtime=seen_at, seen_at=seen_at)
    adapters = [_claude_adapter(tmp_path / "claude")]

    tick(conn, adapters=adapters, settings_path=None)
    tick(conn, adapters=adapters, settings_path=None, batch=True)

    assert conn.execute("SELECT COUNT(*) AS n FROM ingest_queue").fetchone()["n"] == 1


def test_a_transcript_that_grew_after_ingestion_is_picked_up_again(conn, tmp_path, transcript):
    """"so a file is extracted once and incrementally" — a resumed session
    appends to the same file."""
    from nenapu.models import now
    from nenapu.watch import QUIET_SECONDS, record_state, tick

    adapters = [_claude_adapter(tmp_path / "claude")]
    seen_at = now() - QUIET_SECONDS - 1
    record_state(conn, path=str(transcript), agent="claude-code",
                 last_size=transcript.stat().st_size, last_mtime=seen_at, seen_at=seen_at)
    tick(conn, adapters=adapters, settings_path=None)

    with transcript.open("a") as handle:
        handle.write("\n" + _line("user", "actually, use bun"))
    later = now() + QUIET_SECONDS + 2
    tick(conn, adapters=adapters, settings_path=None, at=later)
    tick(conn, adapters=adapters, settings_path=None, at=later + QUIET_SECONDS + 2, batch=True)

    assert conn.execute("SELECT COUNT(*) AS n FROM ingest_queue").fetchone()["n"] == 2


# ---------- not doing the Stop hook's job twice ----------


def _settings(tmp_path, *, with_hook: bool) -> str:
    path = tmp_path / "settings.json"
    hooks = {"Stop": [{"hooks": [{"type": "command", "command": "nenapu learn --stdin --detach"}]}]}
    path.write_text(json.dumps({"hooks": hooks} if with_hook else {}))
    return str(path)


def test_an_agent_whose_hook_is_installed_is_skipped(conn, tmp_path, transcript):
    """"The `(scope, text)` unique index would de-dupe the rows anyway, but
    not the 83-second model call.\""""
    from nenapu.models import now
    from nenapu.watch import QUIET_SECONDS, record_state, tick

    seen_at = now() - QUIET_SECONDS - 1
    record_state(conn, path=str(transcript), agent="claude-code",
                 last_size=transcript.stat().st_size, last_mtime=seen_at, seen_at=seen_at)

    tick(conn, adapters=[_claude_adapter(tmp_path / "claude")],
         settings_path=_settings(tmp_path, with_hook=True))

    assert conn.execute("SELECT COUNT(*) AS n FROM ingest_queue").fetchone()["n"] == 0


def test_without_the_hook_the_watcher_covers_claude_code_too(conn, tmp_path, transcript):
    from nenapu.models import now
    from nenapu.watch import QUIET_SECONDS, record_state, tick

    seen_at = now() - QUIET_SECONDS - 1
    record_state(conn, path=str(transcript), agent="claude-code",
                 last_size=transcript.stat().st_size, last_mtime=seen_at, seen_at=seen_at)

    tick(conn, adapters=[_claude_adapter(tmp_path / "claude")],
         settings_path=_settings(tmp_path, with_hook=False))

    assert conn.execute("SELECT COUNT(*) AS n FROM ingest_queue").fetchone()["n"] == 1


def test_hook_detection_reads_the_settings_file_the_installer_writes(tmp_path):
    from nenapu.watch import agents_with_hooks

    assert agents_with_hooks(_settings(tmp_path, with_hook=True)) == {"claude-code"}
    assert agents_with_hooks(_settings(tmp_path, with_hook=False)) == set()
    assert agents_with_hooks(str(tmp_path / "absent.json")) == set()


# ---------- cost ----------


def test_only_one_extraction_is_enqueued_per_tick_by_default(conn, tmp_path):
    """"Every finished session is one extraction (83s via `claude -p`, ~6k
    tokens in). Add a floor between extractions." Discovering a backlog at
    startup must not queue a hundred model calls at once."""
    from nenapu.models import now
    from nenapu.watch import QUIET_SECONDS, record_state, tick

    root = tmp_path / "claude"
    (root / "projects").mkdir(parents=True)
    seen_at = now() - QUIET_SECONDS - 1
    for i in range(3):
        path = root / "projects" / f"s-{i}.jsonl"
        path.write_text(_line("user", "hello"))
        record_state(conn, path=str(path), agent="claude-code",
                     last_size=path.stat().st_size, last_mtime=seen_at, seen_at=seen_at)

    tick(conn, adapters=[_claude_adapter(root)], settings_path=None)

    assert conn.execute("SELECT COUNT(*) AS n FROM ingest_queue").fetchone()["n"] == 1


def test_batch_mode_drains_the_whole_backlog(conn, tmp_path):
    """"...and a `--batch` mode that drains the queue on a schedule instead
    of instantly.\""""
    from nenapu.models import now
    from nenapu.watch import QUIET_SECONDS, record_state, tick

    root = tmp_path / "claude"
    (root / "projects").mkdir(parents=True)
    seen_at = now() - QUIET_SECONDS - 1
    for i in range(3):
        path = root / "projects" / f"s-{i}.jsonl"
        path.write_text(_line("user", "hello"))
        record_state(conn, path=str(path), agent="claude-code",
                     last_size=path.stat().st_size, last_mtime=seen_at, seen_at=seen_at)

    tick(conn, adapters=[_claude_adapter(root)], settings_path=None, batch=True)

    assert conn.execute("SELECT COUNT(*) AS n FROM ingest_queue").fetchone()["n"] == 3


# ---------- supervision, under the project's existing consent rules ----------


def _run(args, env=None, cwd=None):
    base = {**os.environ, "PYTHONPATH": os.path.abspath("src"), "NENAPU_NO_BANNER": "1"}
    return subprocess.run(
        [sys.executable, "-m", "nenapu.cli", *args],
        capture_output=True, text=True, env={**base, **(env or {})}, cwd=cwd,
    )


def test_watch_once_runs_a_single_tick_and_exits(tmp_path):
    """A foreground daemon needs a way to be exercised without becoming one."""
    result = _run(["watch", "--once", "--db", str(tmp_path / "s.db")])

    assert result.returncode == 0


def test_installing_the_unit_without_a_tty_changes_nothing(tmp_path):
    """The project's standing rule: "A non-TTY run (a pipe, a CI job) is
    **not** consent: it prints what it would do and changes nothing.\""""
    home = tmp_path / "home"
    (home / ".config").mkdir(parents=True)

    result = _run(["init", "--watch", "--db", str(tmp_path / "s.db")],
                  env={"HOME": str(home)})

    unit = home / ".config" / "systemd" / "user" / "nenapu-watch.service"
    assert result.returncode == 0
    assert not unit.exists()
