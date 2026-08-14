"""First-run setup: wire Nenapu into whatever you actually use.

An honest note on what is possible, because the difference decides the design:

* **MCP clients** (Claude Code, Cursor, VS Code, Codex) talk to `nenapu-mcp`
  over the Model Context Protocol. MCP is request/response — the server sees a
  tool call or nothing. It **cannot** watch a conversation go by. So memory
  gets written when the agent decides to write it, which means the agent has to
  be told to.
* **Claude Code hooks** are the exception, and the only place real passive
  observation exists. A `SessionStart` hook can surface memory before the agent
  does anything, and a `Stop` hook can read the finished transcript and extract
  what was learned. Both are installed here.
* **Ollama and LM Studio are not plugin hosts.** They serve models. Nenapu uses
  them for its scheduled audit pass, not as somewhere to plug into. Pretending
  otherwise would be a nice-looking menu entry that does nothing.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Target:
    key: str
    label: str
    kind: str  # "mcp" | "backend"
    detect: str  # command or path to look for
    note: str

    @property
    def present(self) -> bool:
        if self.detect.startswith("~") or self.detect.startswith("/"):
            return Path(self.detect).expanduser().exists()
        return shutil.which(self.detect) is not None


TARGETS = [
    Target("claude", "Claude Code", "mcp", "claude",
           "MCP tools + hooks that observe sessions and surface memory"),
    Target("cursor", "Cursor", "mcp", "cursor", "MCP tools"),
    Target("vscode", "VS Code", "mcp", "code", "MCP tools"),
    Target("codex", "Codex CLI", "mcp", "codex", "MCP tools"),
    Target("ollama", "Ollama", "backend", "ollama",
           "used as the model for scheduled audits, not a plugin host"),
    Target("lmstudio", "LM Studio", "backend", "~/.lmstudio",
           "used as the model for scheduled audits, not a plugin host"),
]


def detected() -> list[Target]:
    return [t for t in TARGETS if t.present]


# ---------- MCP wiring ----------


def wire_claude_code() -> tuple[bool, str]:
    """Register the server with the Claude Code CLI."""
    if not shutil.which("claude"):
        return False, "claude CLI not found"
    existing = subprocess.run(["claude", "mcp", "list"], capture_output=True, text=True)
    if "nenapu" in existing.stdout:
        return True, "already registered"
    result = subprocess.run(
        ["claude", "mcp", "add", "nenapu", "--", "nenapu-mcp"],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        return False, result.stderr.strip()[:160] or "claude mcp add failed"
    return True, "registered with `claude mcp add`"


MCP_SNIPPET = {"mcpServers": {"nenapu": {"command": "nenapu-mcp", "args": []}}}


def wire_json_client(path: Path, key: str = "mcpServers") -> tuple[bool, str]:
    """Merge our server into an editor's MCP config without disturbing others."""
    path = path.expanduser()
    try:
        config = json.loads(path.read_text()) if path.exists() else {}
    except ValueError:
        return False, f"{path} is not valid JSON — left alone"

    servers = config.setdefault(key, {})
    if "nenapu" in servers:
        return True, f"already present in {path}"
    servers["nenapu"] = {"command": "nenapu-mcp", "args": []}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(config, indent=2) + "\n")
    return True, f"added to {path}"


CLIENT_CONFIGS = {
    "cursor": Path("~/.cursor/mcp.json"),
    "vscode": Path("~/.config/Code/User/mcp.json"),
    "codex": Path("~/.codex/config.json"),
}


# ---------- Claude Code hooks: the observing half ----------

HOOK_MARKER = "nenapu"


def hook_config() -> dict:
    """SessionStart surfaces memory; Stop extracts it.

    This is the only place Nenapu sees a conversation it was not explicitly
    asked about, which is what makes "stop repeating the same mistake" possible
    rather than aspirational.

    `--detach` is not an optimisation. Extraction is a model call over a whole
    session; measured against real transcripts through `claude -p` it takes 83
    seconds. A Stop hook that blocks for 83 seconds is unusable, and one capped
    at 60 is simply killed before it writes anything — which is how a memory
    layer ends up looking like it works while learning nothing. So the hook
    hands the work to a detached child and returns immediately.
    """
    return {
        "SessionStart": [{
            "hooks": [{
                "type": "command",
                "command": "nenapu recall-hook",
                "timeout": 10,
            }],
        }],
        "Stop": [{
            "hooks": [{
                "type": "command",
                "command": "nenapu observe --stdin --detach",
                "timeout": 10,
            }],
        }],
    }


def install_hooks(settings_path: Path | None = None) -> tuple[bool, str]:
    path = (settings_path or Path("~/.claude/settings.json")).expanduser()
    try:
        settings = json.loads(path.read_text()) if path.exists() else {}
    except ValueError:
        return False, f"{path} is not valid JSON — left alone"

    hooks = settings.setdefault("hooks", {})
    changed = []
    for event, entries in hook_config().items():
        existing = hooks.setdefault(event, [])
        ours = [e for e in existing if HOOK_MARKER in json.dumps(e)]
        if ours == entries:
            continue
        # An earlier version of ours is present but out of date. Replacing it
        # in place is the point: a stale Stop hook is the 60-second timeout
        # that kills extraction silently, and skipping because "nenapu is in
        # there somewhere" is how a user stays broken across every upgrade.
        kept = [e for e in existing if HOOK_MARKER not in json.dumps(e)]
        hooks[event] = kept + entries
        changed.append(f"{event} (replaced)" if ours else event)

    if not changed:
        return True, "hooks already installed"

    path.parent.mkdir(parents=True, exist_ok=True)
    # Back up before touching a file the user may have hand-edited.
    if path.exists():
        shutil.copy(path, path.with_suffix(".json.nenapu-backup"))
    path.write_text(json.dumps(settings, indent=2) + "\n")
    return True, f"installed {', '.join(changed)} (backup written)"


def remove_hooks(settings_path: Path | None = None) -> tuple[bool, str]:
    path = (settings_path or Path("~/.claude/settings.json")).expanduser()
    if not path.exists():
        return True, "nothing to remove"
    try:
        settings = json.loads(path.read_text())
    except ValueError:
        return False, "settings.json is not valid JSON"

    removed = 0
    for event, entries in list(settings.get("hooks", {}).items()):
        kept = [e for e in entries if HOOK_MARKER not in json.dumps(e)]
        removed += len(entries) - len(kept)
        if kept:
            settings["hooks"][event] = kept
        else:
            settings["hooks"].pop(event, None)
    path.write_text(json.dumps(settings, indent=2) + "\n")
    return True, f"removed {removed} hook(s)"


# ---------- the instruction block, for harnesses without hooks ----------

AGENT_RULES = """\
## Memory (nenapu)

You have a persistent memory store. Use it or you will repeat work and repeat
mistakes.

- At the start of a task, call `memory_search` with a stable `session_id`.
- When the user corrects you, store it: `memory_write(kind="feedback",
  origin="user_stated")`. A correction you do not store is one you will make
  again.
- Store environment facts with a `key` (e.g. `db.port`) so a later contradiction
  is detected rather than stacked, and a `verify_cmd` when a shell command can
  prove it.
- When you conclude something from what you recalled, pass `derived_from` so it
  is flagged if the foundation turns out to be wrong.
- After the task, call `task_outcome` with the same `session_id`. That is what
  keeps recall quality honest.
"""
