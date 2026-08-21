"""Probe and register the remaining watcher adapters.

Requirement (Task 22, "Next up" list, 2026-08-21, marked **Opus 5**,
depends on 15):

    "Probe and register the remaining watcher adapters | Codex, Gemini,
    OpenCode, Cursor. Needs a machine with them installed; the plan's own
    rule forbids shipping a glob nobody has seen match | M | Opus 5 | 15"

The watcher ships one adapter (`watch.ADAPTERS`, claude-code). The plan's
constraint is not a style preference — "a glob nobody has watched match is a
feature that reports success and captures nothing" — so this task is a
probing session, and the thing tests can pin is **the discipline**, not the
globs.

What this file does, therefore
------------------------------
1. Turns "must be probed against a real file" into a rule a machine checks:
   every registered adapter must ship a sample transcript in
   `tests/fixtures/transcripts/<agent>.<ext>`, its glob must match that
   sample's name, and its parser must read turns out of it. Registering
   Codex without ever having seen a Codex transcript then fails a test
   instead of shipping quietly.
2. Pins the provider-agnostic behaviour that must survive a second adapter
   arriving: per-adapter agent labelling through a tick, hook-skip applying
   only to the agent that has hooks, and `discover`/`tick` needing no edit to
   accept a new format.
3. Adds the tool that makes the probing session possible on someone else's
   machine: `nenapu watch --probe`, which reports what each registered glob
   matches here and enqueues nothing.

Adding an adapter is therefore: probe a real file, drop it in
`tests/fixtures/transcripts/`, register the `TranscriptFormat`. The fixture
is evidence, and adding one is the intended way to satisfy these tests —
editing the assertions below is not.

Tests 1-2 below are green for the shipped claude-code adapter and exist to
stay that way; the rest are the contract a new adapter has to meet.
"""

import fnmatch
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from nenapu import connect

FIXTURES = Path(__file__).parent / "fixtures" / "transcripts"

# The four the task names. Each is either absent or fully registered — there
# is no half-state where a glob ships without a probed sample behind it.
PLANNED_AGENTS = {"codex", "gemini", "opencode", "cursor"}


def _adapters():
    from nenapu.watch import ADAPTERS

    return list(ADAPTERS)


def _fixture_for(agent: str) -> Path | None:
    matches = sorted(FIXTURES.glob(f"{agent}.*"))
    return matches[0] if matches else None


# ---------- the registry is still data ----------


def test_every_adapter_carries_an_agent_a_glob_and_a_parser():
    for adapter in _adapters():
        assert adapter.agent and isinstance(adapter.agent, str)
        assert adapter.glob.startswith("~") or adapter.glob.startswith("/")
        assert callable(adapter.parse)


def test_no_agent_is_registered_twice():
    """Two adapters for one agent means every transcript it writes is
    discovered twice and enqueued twice — the duplicate 83 seconds the
    hook-skip rule exists to avoid."""
    names = [a.agent for a in _adapters()]
    assert len(names) == len(set(names))


# ---------- the probing rule, mechanised ----------


def test_every_registered_adapter_ships_the_transcript_it_was_probed_against():
    """The plan's rule as a test. Without evidence on disk, "we probed it" is
    a claim in a commit message that nobody can re-check when the format
    changes."""
    missing = [a.agent for a in _adapters() if _fixture_for(a.agent) is None]
    assert not missing, (
        f"no probed sample in {FIXTURES} for: {missing}. "
        "Register an adapter only with a real transcript from that agent."
    )


def test_every_registered_glob_matches_its_own_sample():
    """Catches the actual failure mode of a hand-written glob: right
    directory, wrong extension — `*.json` against a `.jsonl` file matches
    nothing, forever, silently."""
    for adapter in _adapters():
        sample = _fixture_for(adapter.agent)
        pattern = os.path.basename(adapter.glob)
        assert fnmatch.fnmatch(sample.name, pattern), (
            f"{adapter.agent}: glob {adapter.glob!r} does not match its own "
            f"sample {sample.name!r}"
        )


def test_every_parser_reads_turns_out_of_its_own_sample():
    """A registered parser that returns nothing is the same feature failure
    one layer down: discovery works, extraction gets an empty conversation,
    and the store stays empty while everything reports success."""
    for adapter in _adapters():
        lines = _fixture_for(adapter.agent).read_text().splitlines()
        turns = adapter.parse(lines)
        assert turns, f"{adapter.agent}: parser found no turns in its own sample"
        assert all(isinstance(t, str) for t in turns)
        assert any(t.strip() for t in turns)


def test_every_parser_survives_a_file_it_does_not_understand():
    """Globs overlap on a real machine — a JSONL file in a watched directory
    may belong to another tool entirely, and one adapter raising takes the
    whole tick down with it."""
    junk = ["not json", "", "{}", json.dumps({"unexpected": "shape"})]
    for adapter in _adapters():
        assert adapter.parse(junk) == [] or all(isinstance(t, str) for t in adapter.parse(junk))
        assert adapter.parse([]) == []


def test_a_planned_agent_is_either_fully_registered_or_absent():
    """Codex, Gemini, OpenCode and Cursor are named in the task. Whichever
    ones a probing session reaches must arrive complete — glob, parser and
    sample — rather than as a placeholder that reports coverage it does not
    have."""
    for adapter in _adapters():
        if adapter.agent in PLANNED_AGENTS:
            assert _fixture_for(adapter.agent) is not None
            assert adapter.parse(_fixture_for(adapter.agent).read_text().splitlines())


# ---------- a second adapter changes no code ----------


@pytest.fixture
def conn():
    return connect(":memory:")


def _line(session: str, text: str) -> str:
    return json.dumps({
        "type": "user", "sessionId": session, "cwd": "/repo",
        "message": {"role": "user", "content": [{"type": "text", "text": text}]},
    })


def _fake_adapter(agent: str, root: Path, ext: str = "jsonl"):
    from nenapu.observer import _turns_from
    from nenapu.watch import TranscriptFormat

    return TranscriptFormat(agent=agent, glob=f"{root}/**/*.{ext}", parse=_turns_from)


def _transcript(root: Path, name: str, session: str) -> Path:
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_line(session, "some work happened here"))
    return path


def _settle(conn, path, adapters, at):
    """Two ticks: first sight starts the quiet clock, the second ingests."""
    from nenapu.watch import QUIET_SECONDS, tick

    tick(conn, adapters=adapters, settings_path=None, at=at, batch=True)
    return tick(conn, adapters=adapters, settings_path=None,
                at=at + QUIET_SECONDS + 1, batch=True)


def test_two_agents_in_one_tick_are_labelled_separately(conn, tmp_path):
    """The point of the whole task: `standup` has to be able to say which
    agent did what, and the job carries the only label there is."""
    a_root, b_root = tmp_path / "a", tmp_path / "b"
    _transcript(a_root, "s-a.jsonl", "s-a")
    _transcript(b_root, "s-b.jsonl", "s-b")
    adapters = [_fake_adapter("agent-a", a_root), _fake_adapter("agent-b", b_root)]

    _settle(conn, None, adapters, at=1000.0)

    queued = {(r["agent"], Path(r["path"]).name) for r in conn.execute(
        "SELECT agent, path FROM ingest_queue"
    )}
    assert queued == {("agent-a", "s-a.jsonl"), ("agent-b", "s-b.jsonl")}


def test_an_agent_with_no_hook_api_is_never_skipped(conn, tmp_path):
    """The skip exists because Claude Code reports its own sessions. Applying
    it to an agent that cannot report anything would silently uninstall the
    only capture it has."""
    root = tmp_path / "codexish"
    _transcript(root, "s-x.jsonl", "s-x")
    settings = tmp_path / "settings.json"
    settings.write_text(json.dumps({"hooks": {"Stop": [{"hooks": [
        {"type": "command", "command": "nenapu learn --stdin --detach"},
    ]}]}}))

    from nenapu.watch import QUIET_SECONDS, tick

    adapters = [_fake_adapter("agent-without-hooks", root)]
    tick(conn, adapters=adapters, settings_path=settings, at=1000.0, batch=True)
    queued = tick(conn, adapters=adapters, settings_path=settings,
                  at=1000.0 + QUIET_SECONDS + 1, batch=True)

    assert len(queued) == 1


def test_a_new_adapter_needs_no_change_to_discovery_or_the_tick(conn, tmp_path):
    """"An adapter is data, not a branch." If registering one required an
    edit to `discover` or `tick`, every future agent would be a code change
    in the module the design says it should not touch."""
    from nenapu.watch import discover

    root = tmp_path / "newagent"
    path = _transcript(root, "deep/nested/s-n.jsonl", "s-n")
    adapters = [_fake_adapter("newagent", root)]

    found = discover(adapters)

    assert [(a.agent, p) for a, p in found] == [("newagent", path)]


def test_an_uninstalled_agent_contributes_nothing_rather_than_failing(conn, tmp_path):
    """Most machines have one or two of these installed. A missing directory
    is the normal case, not an error."""
    from nenapu.watch import discover

    assert discover([_fake_adapter("ghost", tmp_path / "not-here")]) == []


# ---------- the probe command ----------


def _run(args, db, **env):
    return subprocess.run(
        [sys.executable, "-m", "nenapu.cli", *args, "--db", str(db)],
        capture_output=True, text=True,
        env={**os.environ, "PYTHONPATH": "src", "NENAPU_NO_BANNER": "1", **env},
    )


def test_watch_probe_reports_every_registered_adapter(tmp_path):
    """The tool the probing session needs: run it on a machine that has
    Codex or Cursor and it says whether the glob you are about to register
    matches anything there."""
    db = tmp_path / "s.db"

    result = _run(["watch", "--probe"], db, HOME=str(tmp_path))

    assert result.returncode == 0, result.stdout + result.stderr
    for adapter in _adapters():
        assert adapter.agent in result.stdout


def test_watch_probe_counts_what_each_glob_matches_here(tmp_path):
    db = tmp_path / "s.db"
    projects = tmp_path / ".claude" / "projects" / "repo"
    projects.mkdir(parents=True)
    (projects / "s-1.jsonl").write_text(_line("s-1", "hello"))

    result = _run(["watch", "--probe"], db, HOME=str(tmp_path))

    assert "1" in result.stdout


def test_watch_probe_enqueues_nothing(tmp_path):
    """A probe that ingests is not a probe — someone running it to check a
    glob would spend an extraction per matched file finding out."""
    db = tmp_path / "s.db"
    projects = tmp_path / ".claude" / "projects" / "repo"
    projects.mkdir(parents=True)
    (projects / "s-1.jsonl").write_text(_line("s-1", "hello"))

    _run(["watch", "--probe"], db, HOME=str(tmp_path))

    queued = connect(str(db)).execute("SELECT COUNT(*) c FROM ingest_queue").fetchone()["c"]
    assert queued == 0


def test_watch_probe_says_so_when_a_glob_matches_nothing(tmp_path):
    """The answer the probing session is actually looking for, and the one a
    silent zero would hide."""
    db = tmp_path / "s.db"

    result = _run(["watch", "--probe"], db, HOME=str(tmp_path))

    assert "0" in result.stdout or "none" in result.stdout.lower()
