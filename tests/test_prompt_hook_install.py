"""Installing the prompt hook, and why it is not on by default.

Requirement (Task 10, query-driven hybrid retrieval plan):

`hook_config()` has always returned exactly two events, and
`tests/test_setup_wizard.py` pins that. It is not an incidental assertion: two
hooks is what this tool asks of somebody's editor today, and adding a third
uninvited is a change to that bargain.

Measurement is what settles it. The prompt hook runs at 1167ms p50 and 1996ms
p95 on the live store with the model warm, against 160ms with the semantic leg
off, because a fresh CLI process pays the ONNX session load on every prompt and
has nothing to amortise it with. That is comfortably inside the ten second hook
timeout and is also more than a second added to every turn the user takes.
Comfortably inside a timeout is not the same as free.

So it ships opt-in, `hook_config()` still returns two events by default, and
the pinned assertion stays true rather than being edited to match a decision
that had not been measured yet.

Assumed seam, proposed by the plan and not yet in the codebase::

    setup_wizard.hook_config(prompt_hook: bool = False)
    setup_wizard.install_hooks(path, prompt_hook: bool = False)
    nenapu init --prompt-hook
"""

import json
import os
import subprocess
import sys

import pytest

from nenapu.setup_wizard import hook_config, install_hooks, remove_hooks


# --- the default is unchanged ------------------------------------------------


def test_the_default_is_still_two_hooks():
    """Asserted here as well as in tests/test_setup_wizard.py, because this is
    the file where someone would come looking to change it."""
    assert set(hook_config()) == {"SessionStart", "Stop"}


def test_asking_for_it_adds_a_third():
    config = hook_config(prompt_hook=True)

    assert set(config) == {"SessionStart", "Stop", "UserPromptSubmit"}
    entry = config["UserPromptSubmit"][0]["hooks"][0]
    assert entry["command"] == "nenapu prompt-hook"
    assert entry["timeout"] <= 10


# --- installing --------------------------------------------------------------


def test_installing_with_the_flag_writes_all_three(tmp_path):
    path = tmp_path / "settings.json"

    ok, _ = install_hooks(path, prompt_hook=True)

    assert ok
    settings = json.loads(path.read_text())
    assert set(settings["hooks"]) == {"SessionStart", "Stop", "UserPromptSubmit"}


def test_installing_without_the_flag_leaves_it_out(tmp_path):
    path = tmp_path / "settings.json"

    install_hooks(path)

    settings = json.loads(path.read_text())
    assert "UserPromptSubmit" not in settings["hooks"]


def test_a_users_own_prompt_hook_is_left_alone(tmp_path):
    """Somebody else's hook on the same event is none of our business, and
    replacing it would be a data loss bug in a file people hand edit."""
    path = tmp_path / "settings.json"
    theirs = {"hooks": {"UserPromptSubmit": [
        {"hooks": [{"type": "command", "command": "their-own-tool"}]}]}}
    path.write_text(json.dumps(theirs))

    install_hooks(path, prompt_hook=True)

    entries = json.loads(path.read_text())["hooks"]["UserPromptSubmit"]
    commands = [h["command"] for e in entries for h in e["hooks"]]
    assert "their-own-tool" in commands
    assert "nenapu prompt-hook" in commands


def test_an_older_version_of_ours_is_replaced_in_place(tmp_path):
    """The rule install_hooks already keeps: a stale entry of ours is worse
    than none, because it looks installed and behaves wrongly."""
    path = tmp_path / "settings.json"
    stale = {"hooks": {"UserPromptSubmit": [
        {"hooks": [{"type": "command", "command": "nenapu prompt-hook --old",
                    "timeout": 60}]}]}}
    path.write_text(json.dumps(stale))

    install_hooks(path, prompt_hook=True)

    entries = json.loads(path.read_text())["hooks"]["UserPromptSubmit"]
    commands = [h["command"] for e in entries for h in e["hooks"]]
    assert commands == ["nenapu prompt-hook"]


def test_removing_takes_the_prompt_hook_with_it(tmp_path):
    path = tmp_path / "settings.json"
    install_hooks(path, prompt_hook=True)

    remove_hooks(path)

    settings = json.loads(path.read_text())
    assert "UserPromptSubmit" not in settings.get("hooks", {})


def test_a_backup_is_written_before_touching_an_existing_file(tmp_path):
    path = tmp_path / "settings.json"
    path.write_text(json.dumps({"model": "opus"}))

    install_hooks(path, prompt_hook=True)

    assert path.with_suffix(".json.nenapu-backup").exists()
    assert json.loads(path.read_text())["model"] == "opus"


@pytest.mark.parametrize("prompt_hook", [False, True])
def test_installing_twice_changes_nothing_the_second_time(tmp_path, prompt_hook):
    path = tmp_path / "settings.json"
    install_hooks(path, prompt_hook=prompt_hook)
    first = path.read_text()

    ok, message = install_hooks(path, prompt_hook=prompt_hook)

    assert ok
    assert path.read_text() == first
    assert "already installed" in message


# --- telling the user where they stand ---------------------------------------


def test_doctor_reports_whether_the_model_is_warm(tmp_path):
    """Someone who turns the hook on needs to know this before their first
    prompt does. Every read path refuses to download, so an uncached model
    means the semantic leg is silently off rather than slow."""
    out = subprocess.run(
        [sys.executable, "-m", "nenapu.cli", "doctor", "--db", str(tmp_path / "s.db")],
        capture_output=True, text=True,
        env={**os.environ, "PYTHONPATH": os.path.abspath("src"),
             "NENAPU_NO_BANNER": "1"},
    )

    assert out.returncode == 0
    assert "embedding" in out.stdout.lower()


def test_doctor_survives_the_extra_being_absent(tmp_path):
    out = subprocess.run(
        [sys.executable, "-m", "nenapu.cli", "doctor", "--db", str(tmp_path / "s.db")],
        capture_output=True, text=True,
        env={**os.environ, "PYTHONPATH": os.path.abspath("src"),
             "NENAPU_NO_BANNER": "1", "NENAPU_EMBEDDINGS": "off"},
    )

    assert out.returncode == 0
    assert "embedding" in out.stdout.lower()
    assert "Traceback" not in out.stderr
