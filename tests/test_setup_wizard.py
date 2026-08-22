"""Wiring Nenapu into someone else's editor config.

Every function here writes to a file the user owns and may have hand-edited.
The tests are mostly about what must *not* happen: no clobbering, no
duplicating, no leaving a broken JSON file behind.
"""

import json


from nenapu.setup_wizard import (
    HOOK_MARKER,
    TARGETS,
    hook_config,
    install_hooks,
    remove_hooks,
    wire_json_client,
)


# ---------- the hook contract ----------


def test_stop_hook_detaches():
    """Extraction is an 83-second model call; a blocking Stop hook is killed.

    If this ever reverts to a plain `observe --stdin`, the layer silently stops
    learning: the hook is terminated at its timeout and nothing is written.
    """
    stop = hook_config()["Stop"][0]["hooks"][0]
    assert "--detach" in stop["command"]
    assert stop["timeout"] <= 15


def test_session_start_hook_is_the_injecting_half():
    start = hook_config()["SessionStart"][0]["hooks"][0]
    assert "recall-hook" in start["command"]


# ---------- installing into settings.json ----------


def test_install_creates_settings_when_absent(tmp_path):
    path = tmp_path / "settings.json"
    ok, _ = install_hooks(path)
    assert ok
    settings = json.loads(path.read_text())
    assert set(settings["hooks"]) == {"SessionStart", "Stop"}


def test_install_preserves_unrelated_settings_and_hooks(tmp_path):
    path = tmp_path / "settings.json"
    path.write_text(json.dumps({
        "model": "opus",
        "hooks": {"Stop": [{"hooks": [{"type": "command", "command": "my-own-thing"}]}]},
    }))
    ok, _ = install_hooks(path)
    assert ok
    settings = json.loads(path.read_text())
    assert settings["model"] == "opus"
    commands = json.dumps(settings["hooks"]["Stop"])
    assert "my-own-thing" in commands and HOOK_MARKER in commands


def test_install_is_idempotent(tmp_path):
    path = tmp_path / "settings.json"
    install_hooks(path)
    ok, detail = install_hooks(path)
    assert ok and "already" in detail
    assert json.dumps(json.loads(path.read_text())).count("nenapu learn") == 1


def test_a_backup_is_written_before_an_existing_file_is_touched(tmp_path):
    path = tmp_path / "settings.json"
    original = json.dumps({"model": "opus"})
    path.write_text(original)
    install_hooks(path)
    assert (tmp_path / "settings.json.nenapu-backup").read_text() == original


def test_unparseable_settings_are_left_alone(tmp_path):
    path = tmp_path / "settings.json"
    path.write_text("{ not json")
    ok, detail = install_hooks(path)
    assert not ok and "not valid JSON" in detail
    assert path.read_text() == "{ not json"


def test_remove_takes_out_ours_and_only_ours(tmp_path):
    path = tmp_path / "settings.json"
    path.write_text(json.dumps({
        "hooks": {"Stop": [{"hooks": [{"type": "command", "command": "my-own-thing"}]}]},
    }))
    install_hooks(path)
    ok, _ = remove_hooks(path)
    assert ok
    settings = json.loads(path.read_text())
    assert "my-own-thing" in json.dumps(settings)
    assert HOOK_MARKER not in json.dumps(settings)


def test_remove_drops_an_event_left_empty(tmp_path):
    path = tmp_path / "settings.json"
    install_hooks(path)
    remove_hooks(path)
    assert json.loads(path.read_text()).get("hooks", {}) == {}


def test_remove_on_a_missing_file_is_not_an_error(tmp_path):
    ok, _ = remove_hooks(tmp_path / "nothing.json")
    assert ok


# ---------- MCP client configs ----------


def test_wiring_an_editor_keeps_its_other_servers(tmp_path):
    path = tmp_path / "mcp.json"
    path.write_text(json.dumps({"mcpServers": {"other": {"command": "other-server"}}}))
    ok, _ = wire_json_client(path)
    assert ok
    servers = json.loads(path.read_text())["mcpServers"]
    assert set(servers) == {"other", "nenapu"}


def test_wiring_creates_missing_parent_directories(tmp_path):
    path = tmp_path / "deep" / "nested" / "mcp.json"
    ok, _ = wire_json_client(path)
    assert ok and path.exists()


def test_wiring_twice_does_not_duplicate(tmp_path):
    path = tmp_path / "mcp.json"
    wire_json_client(path)
    ok, detail = wire_json_client(path)
    assert ok and "already" in detail


def test_broken_editor_config_is_left_alone(tmp_path):
    path = tmp_path / "mcp.json"
    path.write_text("]]not json[[")
    ok, _ = wire_json_client(path)
    assert not ok
    assert path.read_text() == "]]not json[["


# ---------- the honest-menu rule ----------


def test_model_hosts_are_not_advertised_as_plugin_hosts():
    """Ollama and LM Studio serve models; they have no plugin surface.

    Listing them as somewhere to "install" memory would be a menu entry that
    does nothing, so they carry a different kind and a note that says so.
    """
    hosts = {t.key: t for t in TARGETS}
    assert hosts["ollama"].kind == "backend"
    assert hosts["lmstudio"].kind == "backend"
    assert hosts["claude"].kind == "mcp"


def test_a_stale_hook_from_an_earlier_version_is_replaced(tmp_path):
    """The upgrade path, which is the one that actually bites.

    A user who installed before `--detach` existed has a Stop hook capped at 60
    seconds — the case that is killed mid-extraction and writes nothing. If
    install skips because the word "nenapu" appears somewhere in the event, that
    user stays silently broken forever.
    """
    path = tmp_path / "settings.json"
    path.write_text(json.dumps({"hooks": {"Stop": [
        {"hooks": [{"type": "command", "command": "nenapu learn --stdin", "timeout": 60}]},
    ]}}))

    ok, detail = install_hooks(path)

    assert ok and "replaced" in detail
    stop = json.loads(path.read_text())["hooks"]["Stop"]
    assert stop == hook_config()["Stop"]
    assert not any("timeout" in json.dumps(e) and '"timeout": 60' in json.dumps(e) for e in stop)


def test_the_real_pre_rename_hook_is_upgraded(tmp_path):
    """Requirement (Task 0c, priority-ordered task list): re-running `nenapu
    init` must repair the exact stale hook this machine's own
    `~/.claude/settings.json` carries today.

    The command word was `observe` before commit `8a33995` renamed it to
    `learn`; `install_hooks` never rewrote already-installed hooks on
    upgrade, so this literal string is what is currently sitting on disk:
    `nenapu observe --stdin --detach`. This test pins that the upgrade path
    actually replaces it once the install is repaired (Task 0) and this is
    re-run (Task 0c) — not a hypothetical, a fixture of a real file.
    """
    path = tmp_path / "settings.json"
    path.write_text(json.dumps({"hooks": {"Stop": [
        {"hooks": [{"type": "command", "command": "nenapu observe --stdin --detach",
                    "timeout": 10}]},
    ]}}))

    ok, detail = install_hooks(path)

    assert ok and "replaced" in detail
    stop = json.loads(path.read_text())["hooks"]["Stop"]
    assert stop == hook_config()["Stop"]
    assert "observe" not in json.dumps(stop)


def test_replacing_ours_leaves_someone_elses_hook_on_the_same_event(tmp_path):
    path = tmp_path / "settings.json"
    theirs = {"hooks": [{"type": "command", "command": "make lint"}]}
    path.write_text(json.dumps({"hooks": {"Stop": [
        theirs,
        {"hooks": [{"type": "command", "command": "nenapu learn --stdin", "timeout": 60}]},
    ]}}))

    install_hooks(path)

    stop = json.loads(path.read_text())["hooks"]["Stop"]
    assert theirs in stop
    assert stop == [theirs] + hook_config()["Stop"]
