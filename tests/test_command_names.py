"""The command names, and the promise the old ones still keep.

Nine commands were renamed to plain words — `write` became `remember`, `loops`
became `doubts`, `observe` became `learn`. Renaming commands in a tool people
have already wired into hooks and shell scripts is a breaking change unless the
old word keeps working, so every old name survives as a hidden alias.
"""

import os
import subprocess
import sys

import pytest

import nenapu.cli as cli

RENAMED = {
    "write": "remember",
    "search": "recall",
    "verify": "check",
    "loops": "doubts",
    "distill": "tidy",
    "observe": "learn",
    "good": "helped",
    "bad": "misled",
    "outcome": "grade",
}


def registered() -> dict[str, bool]:
    """Every command name, mapped to whether it is hidden from the listing."""
    names = {}
    for command in cli.app.registered_commands:
        name = command.name or (command.callback.__name__.replace("_", "-")
                                if command.callback else "")
        names[name] = bool(command.hidden)
    return names


@pytest.mark.parametrize("old, new", RENAMED.items())
def test_the_new_name_is_the_one_advertised(old, new):
    names = registered()

    assert names.get(new) is False, f"{new} should be listed"
    assert names.get(old) is True, f"{old} should survive, hidden"


def test_the_landing_view_lists_each_command_once():
    """An unhidden alias would put every renamed command on the screen twice,
    which is worse than the jargon it replaced."""
    listed = [name for rows in cli._command_groups().values() for name, _ in rows]

    assert len(listed) == len(set(listed))
    for old in RENAMED:
        assert old not in listed


def test_the_stop_hook_uses_the_new_name():
    """The hook is written into someone's settings.json and read back by the
    upgrade check. If it kept saying `observe`, every install would look stale
    forever and be rewritten on every run."""
    from nenapu.setup_wizard import hook_config

    stop = str(hook_config()["Stop"])

    assert "nenapu learn --stdin --detach" in stop
    assert "observe" not in stop


@pytest.mark.parametrize("old", ["write", "search", "loops"])
def test_an_old_name_still_runs(old, tmp_path):
    """Someone's shell history, someone's script, someone's blog post."""
    db = str(tmp_path / "m.db")
    env = {**os.environ, "PYTHONPATH": "src", "NENAPU_NO_BANNER": "1"}
    args = {"write": ["write", "a fact"], "search": ["search", "fact"],
            "loops": ["loops"]}[old]

    result = subprocess.run([sys.executable, "-m", "nenapu.cli", *args, "--db", db],
                            capture_output=True, text=True, env=env)

    assert result.returncode == 0, result.stdout + result.stderr


def test_the_recall_hook_keeps_its_name():
    """This one is wired into ~/.claude/settings.json on every machine that has
    run `nenapu init`. Renaming it would break memory injection silently — the
    session would simply start knowing nothing."""
    assert registered().get("recall-hook") is True
