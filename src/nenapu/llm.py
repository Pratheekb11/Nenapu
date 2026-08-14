"""Model backends for the audit and distill passes.

Nenapu's core loop — decay, contradiction detection, executable checks,
falsification cascades, recall grading — never calls a model at all. Only two
scheduled jobs do, and neither should force a cloud account on someone running
a local store.

Four backends, one interface:

  * `anthropic`  — the Anthropic SDK, when an API key or profile is present.
  * `ollama`     — http://localhost:11434, native API with schema support.
  * `lmstudio`   — http://localhost:1234/v1, OpenAI-compatible.
  * `openai`     — any other OpenAI-compatible server (vLLM, llama.cpp, LiteLLM).
  * `exec`       — any CLI that takes a prompt on stdin and prints a reply,
                   e.g. `claude -p`. Lets someone who already has an agent CLI
                   installed run audits without provisioning a second
                   credential.

Selection is `NENAPU_LLM`; `auto` (the default) prefers a configured Anthropic
key and otherwise probes for a local server. Everything but the Anthropic
backend goes through stdlib HTTP, so a local-only install has no extra deps.

Note on editors: VS Code, Cursor, and Claude Code are MCP *clients* — they talk
to `nenapu-mcp` and are already supported. They are not model providers, and
nothing here concerns them.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import time
import urllib.error
import urllib.request
from dataclasses import dataclass

DEFAULT_TIMEOUT = int(os.environ.get("NENAPU_LLM_TIMEOUT", "180"))

# Auditing a fact is a classification, not a piece of writing. Sampling buys
# nothing here and costs reproducibility: the same store audited twice would
# otherwise give different verdicts, and a user cannot act on a finding that
# might evaporate on re-run. Greedy decoding with a fixed seed makes the pass
# repeatable.
DECODING = {"temperature": 0.0, "top_p": 1.0, "seed": 1}

# No num_ctx override here on purpose. Capping it to 8192 was measured against
# the model default on this workload and made no difference (29.1s vs 31.3s for
# identical work), while risking truncation on stores with long facts. Ollama
# evidently sizes the KV cache lazily. Left to the model default.

# Local defaults, overridable per backend with NENAPU_LLM_URL.
BACKEND_URLS = {
    "ollama": "http://localhost:11434",
    "lmstudio": "http://localhost:1234/v1",
    "openai": "http://localhost:8000/v1",
}

# Sensible small-model defaults. A 3B model handles these jobs adequately
# because both prompts are short and the output schema is tiny — the work is
# classification, not composition. Bigger local models mostly buy latency:
# schema-constrained decoding is token-bound, so halving the output matters
# more than halving the parameters.
DEFAULT_MODELS = {
    "anthropic": "claude-opus-5",
    "ollama": "qwen2.5:3b",
    "lmstudio": "local-model",
    "openai": "local-model",
    "exec": "cli",
}

# Command for the `exec` backend. Anything that reads a prompt on stdin and
# writes the reply to stdout.
EXEC_COMMAND = os.environ.get("NENAPU_LLM_CMD", "claude -p")

# Retries for the `exec` backend. Nothing here is idempotency-sensitive — the
# call is a read that produces JSON — so retrying is safe.
EXEC_ATTEMPTS = int(os.environ.get("NENAPU_LLM_ATTEMPTS", "3"))

# Ceiling on the context window we will ask Ollama for. Past this the KV cache
# costs more memory than a background pass has any business taking.
MAX_NUM_CTX = int(os.environ.get("NENAPU_NUM_CTX_MAX", "16384"))
EXEC_BACKOFF = 2.0


class LLMUnavailable(RuntimeError):
    pass


@dataclass
class Backend:
    name: str
    model: str
    url: str | None = None

    def describe(self) -> str:
        return f"{self.name}:{self.model}" + (f" @ {self.url}" if self.url else "")

    @property
    def trusted(self) -> bool:
        """Backends whose findings may be applied without asking.

        `exec` is deliberately excluded: the command is arbitrary, so nothing
        can be assumed about what is on the other end until calibration says so.
        """
        return self.name == "anthropic"


def _exec_backend(prompt: str, schema: dict, system: str | None, backend: Backend) -> dict:
    """Drive a CLI that speaks prose in and prose out.

    There is no schema-enforcement hook here, so the shape is requested in the
    prompt and recovered by `extract_json` — which is what a CLI agent's
    conversational wrapping requires anyway.
    """
    import subprocess

    instruction = (
        f"{system}\n\n" if system else ""
    ) + prompt + (
        "\n\nReply with a single JSON object matching this schema, and nothing "
        f"else — no prose, no code fences:\n{json.dumps(schema)}"
    )
    # An agent CLI is a network client behind a process boundary. Rate limits,
    # a dropped connection and an overloaded upstream all arrive here as the
    # same thing: a non-zero exit. Three identical prompts that failed this way
    # all succeeded unchanged on a later attempt, so a single try turns a
    # transient into a permanent "cannot observe".
    last = ""
    for attempt in range(EXEC_ATTEMPTS):
        try:
            proc = subprocess.run(
                backend.model, shell=True, input=instruction, capture_output=True,
                text=True, timeout=DEFAULT_TIMEOUT,
            )
        except subprocess.TimeoutExpired as exc:
            raise LLMUnavailable(
                f"`{backend.model}` did not finish within {DEFAULT_TIMEOUT}s"
            ) from exc
        if proc.returncode == 0:
            return extract_json(proc.stdout)
        # Agent CLIs routinely print their diagnostics on stdout and leave
        # stderr empty. Reporting stderr alone produced `exited 1: ` with
        # nothing after the colon, which is the least useful error possible.
        last = (proc.stderr.strip() or proc.stdout.strip() or "no output")[:300]
        if attempt + 1 < EXEC_ATTEMPTS:
            time.sleep(EXEC_BACKOFF * (2 ** attempt))
    raise LLMUnavailable(
        f"`{backend.model}` exited {proc.returncode} after {EXEC_ATTEMPTS} attempts: {last}"
    )


def _exec_available() -> bool:
    """Whether the configured agent CLI is actually on PATH.

    Only the program name is checked — the rest of `EXEC_COMMAND` is flags. A
    command with a shell operator in it is never auto-selected: `auto` should
    not be the thing that decides to run a pipeline, only a plain CLI someone
    already installed and signed into.
    """
    command = EXEC_COMMAND.strip()
    if not command or any(ch in command for ch in "|&;<>$`"):
        return False
    return shutil.which(command.split()[0]) is not None


def _probe(url: str, path: str, timeout: float = 1.5) -> bool:
    try:
        with urllib.request.urlopen(f"{url}{path}", timeout=timeout) as r:
            return r.status < 500
    except Exception:
        return False


def detect_backend() -> Backend:
    """Resolve which model serves the scheduled passes.

    Explicit configuration always wins. `auto` prefers Anthropic only when
    credentials actually exist, then looks for something listening locally, so
    an offline machine does the right thing without being told.
    """
    choice = os.environ.get("NENAPU_LLM", "auto").lower()
    model = os.environ.get("NENAPU_MODEL", "")
    url = os.environ.get("NENAPU_LLM_URL", "")

    def build(name: str) -> Backend:
        if name == "anthropic":
            return Backend(name=name, model=model or DEFAULT_MODELS[name])
        if name == "exec":
            return Backend(name=name, model=model or EXEC_COMMAND)
        return Backend(name=name, model=model or DEFAULT_MODELS[name],
                       url=url or BACKEND_URLS[name])

    if choice in DEFAULT_MODELS:
        return build(choice)
    if choice != "auto":
        raise LLMUnavailable(
            f"unknown NENAPU_LLM={choice!r}; expected one of {', '.join(DEFAULT_MODELS)} or auto"
        )

    if os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN"):
        return build("anthropic")
    if _exec_available():
        # Preferred over a local server, and the ordering was decided by
        # measurement rather than taste. Extraction sends ~6k tokens of
        # conversation and asks for structured JSON back. Through `claude -p`
        # that takes 83s. Through the default local 3B on a CPU-only host it
        # did not finish at all: 180s, twice, on transcripts of 20k and 24k
        # characters. A backend that always times out is not a cheaper answer,
        # it is no answer — the session ends and the store stays empty.
        return build("exec")
    if _probe(url or BACKEND_URLS["ollama"], "/api/tags"):
        return build("ollama")
    if _probe(url or BACKEND_URLS["lmstudio"], "/models"):
        return build("lmstudio")
    if os.path.exists(os.path.expanduser("~/.config/anthropic")):
        return build("anthropic")  # `ant auth login` profile on disk

    raise LLMUnavailable(
        "no model backend found. Start Ollama or LM Studio, set ANTHROPIC_API_KEY, "
        "or set NENAPU_LLM explicitly."
    )


# ---------- response parsing ----------

_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.S)


def extract_json(text: str) -> dict:
    """Pull an object out of whatever the model actually returned.

    Small local models fence their output, prepend commentary, or emit a
    trailing explanation no matter how the prompt is worded. Schema enforcement
    handles the well-behaved case; this handles the rest.
    """
    text = (text or "").strip()
    if not text:
        raise LLMUnavailable("empty response")

    for candidate in (text, *(m.group(1) for m in _FENCE.finditer(text))):
        try:
            parsed = json.loads(candidate.strip())
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            continue

    start = text.find("{")
    if start >= 0:
        depth, in_string, escaped = 0, False, False
        for i, ch in enumerate(text[start:], start):
            if in_string:
                if escaped:
                    escaped = False
                elif ch == "\\":
                    escaped = True
                elif ch == '"':
                    in_string = False
                continue
            if ch == '"':
                in_string = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[start : i + 1])
                    except json.JSONDecodeError:
                        break

    raise LLMUnavailable(f"no JSON object in response: {text[:200]}")


def _post(url: str, payload: dict, *, timeout: int, headers: dict | None = None) -> dict:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json", **(headers or {})},
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode())
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode()[:300]
        raise LLMUnavailable(f"HTTP {exc.code} from {url}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise LLMUnavailable(f"cannot reach {url}: {exc.reason}") from exc
    except TimeoutError as exc:
        # An 8B model on CPU can exceed the read timeout on a single call. That
        # is a capacity answer, not a crash: surface it as one.
        raise LLMUnavailable(
            f"{url} did not respond within {timeout}s — the model is too slow for this "
            "machine. Use a smaller model, or raise NENAPU_LLM_TIMEOUT."
        ) from exc
    except OSError as exc:
        raise LLMUnavailable(f"connection to {url} failed: {exc}") from exc


# ---------- backends ----------


def _anthropic(prompt: str, schema: dict, system: str | None, backend: Backend,
               max_tokens: int, effort: str) -> dict:
    try:
        import anthropic
    except ImportError as exc:  # pragma: no cover - dependency guard
        raise LLMUnavailable("pip install anthropic, or use a local backend") from exc

    kwargs: dict = {
        "model": backend.model,
        "max_tokens": max_tokens,
        "thinking": {"type": "adaptive"},
        "output_config": {"effort": effort, "format": {"type": "json_schema", "schema": schema}},
        "messages": [{"role": "user", "content": prompt}],
    }
    if system:
        kwargs["system"] = system

    response = anthropic.Anthropic().messages.create(**kwargs)
    if response.stop_reason == "refusal":
        raise LLMUnavailable("model declined the request")
    text = next((b.text for b in response.content if b.type == "text"), "")
    return extract_json(text)


def _num_ctx(messages: list[dict], max_tokens: int) -> int:
    """Size the context window to the prompt instead of taking Ollama's default.

    Ollama defaults to 4096 tokens and, when the prompt is longer, **silently
    drops the front of it**. A harvested session runs to 24,000 characters —
    comfortably past that — so the oldest part of the conversation was being
    discarded by the server without a word. That is the part where a correction
    usually is: the user objects, then the work continues for another twenty
    minutes. Nothing in the response says this happened.
    """
    chars = sum(len(m["content"]) for m in messages)
    needed = chars // 3 + max_tokens + 512  # ~3 chars/token, then headroom
    return max(4096, min(1 << (needed - 1).bit_length(), MAX_NUM_CTX))


def _ollama(prompt: str, schema: dict, system: str | None, backend: Backend,
            max_tokens: int) -> dict:
    """Ollama's native endpoint takes a JSON schema in `format` directly.

    Streamed deliberately, even though only the final text is wanted. With
    `stream: false` Ollama generates the whole response before writing a byte,
    so abandoning a slow request leaves the server generating into the void —
    an 8B model on CPU kept a core pinned for minutes after a timeout, starving
    everything else on the machine. Streaming lets us close the socket, which
    fails Ollama's next write and cancels the generation.
    """
    messages = ([{"role": "system", "content": system}] if system else []) + [
        {"role": "user", "content": prompt}
    ]
    request = urllib.request.Request(
        f"{backend.url}/api/chat",
        data=json.dumps({
            "model": backend.model,
            "messages": messages,
            "stream": True,
            "format": schema,
            "options": {"num_predict": max_tokens, "num_ctx": _num_ctx(messages, max_tokens),
                        **DECODING},
        }).encode(),
        headers={"Content-Type": "application/json"},
    )

    deadline = time.monotonic() + DEFAULT_TIMEOUT
    parts: list[str] = []
    try:
        with urllib.request.urlopen(request, timeout=DEFAULT_TIMEOUT) as response:
            for line in response:
                if time.monotonic() > deadline:
                    # Closing here is the point: it aborts generation server-side
                    # rather than leaving it running unattended.
                    response.close()
                    raise LLMUnavailable(
                        f"{backend.model} exceeded {DEFAULT_TIMEOUT}s — generation "
                        "cancelled. Use a smaller model, or raise NENAPU_LLM_TIMEOUT."
                    )
                if not line.strip():
                    continue
                chunk = json.loads(line)
                if chunk.get("error"):
                    raise LLMUnavailable(f"ollama: {chunk['error']}")
                parts.append(chunk.get("message", {}).get("content", ""))
                if chunk.get("done"):
                    break
    except urllib.error.HTTPError as exc:
        raise LLMUnavailable(f"HTTP {exc.code} from ollama: {exc.read().decode()[:200]}") from exc
    except urllib.error.URLError as exc:
        raise LLMUnavailable(f"cannot reach {backend.url}: {exc.reason}") from exc
    except TimeoutError as exc:
        raise LLMUnavailable(
            f"{backend.model} stopped responding after {DEFAULT_TIMEOUT}s"
        ) from exc

    return extract_json("".join(parts))


def _openai_compatible(prompt: str, schema: dict, system: str | None, backend: Backend,
                       max_tokens: int) -> dict:
    """LM Studio, vLLM, llama.cpp, LiteLLM — the common wire format.

    Schema enforcement is requested but not required: servers that reject
    `response_format` are retried plain, with `extract_json` doing the work.
    """
    messages = ([{"role": "system", "content": system}] if system else []) + [
        {"role": "user", "content": prompt}
    ]
    payload = {
        "model": backend.model,
        "messages": messages,
        "max_tokens": max_tokens,
        "stream": False,
        **DECODING,
        "response_format": {
            "type": "json_schema",
            "json_schema": {"name": "nenapu", "schema": schema, "strict": True},
        },
    }
    url = f"{backend.url}/chat/completions"
    try:
        body = _post(url, payload, timeout=DEFAULT_TIMEOUT)
    except LLMUnavailable as exc:
        if "HTTP 4" not in str(exc):
            raise
        payload.pop("response_format")
        payload["messages"][-1]["content"] += "\n\nReply with JSON only. No prose, no fences."
        body = _post(url, payload, timeout=DEFAULT_TIMEOUT)

    choices = body.get("choices") or []
    if not choices:
        raise LLMUnavailable(f"no choices in response: {str(body)[:200]}")
    return extract_json(choices[0].get("message", {}).get("content", ""))


def structured(
    prompt: str,
    schema: dict,
    *,
    system: str | None = None,
    backend: Backend | None = None,
    max_tokens: int = 4096,
    effort: str = "high",
) -> dict:
    """One schema-constrained call against whichever backend is configured."""
    backend = backend or detect_backend()

    if backend.name == "anthropic":
        return _anthropic(prompt, schema, system, backend, max_tokens, effort)
    if backend.name == "ollama":
        return _ollama(prompt, schema, system, backend, max_tokens)
    if backend.name == "exec":
        return _exec_backend(prompt, schema, system, backend)
    return _openai_compatible(prompt, schema, system, backend, max_tokens)


def available() -> tuple[bool, str]:
    """(usable, description) — for `nenapu doctor` and error messages."""
    try:
        backend = detect_backend()
    except LLMUnavailable as exc:
        return False, str(exc)
    return True, backend.describe()
