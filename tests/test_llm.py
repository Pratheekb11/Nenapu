"""Backend selection and the parsing that local models make necessary."""

import json

import pytest

from nenapu import llm
from nenapu.llm import (
    BACKEND_URLS,
    DEFAULT_MODELS,
    LLMUnavailable,
    detect_backend,
    extract_json,
)


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for var in ("NENAPU_LLM", "NENAPU_MODEL", "NENAPU_LLM_URL",
                "ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN"):
        monkeypatch.delenv(var, raising=False)


def test_explicit_backend_wins(monkeypatch):
    monkeypatch.setenv("NENAPU_LLM", "ollama")
    monkeypatch.setenv("NENAPU_MODEL", "qwen3:4b")
    backend = detect_backend()
    assert (backend.name, backend.model, backend.url) == (
        "ollama", "qwen3:4b", BACKEND_URLS["ollama"]
    )


def test_local_backends_get_sane_defaults(monkeypatch):
    monkeypatch.setenv("NENAPU_LLM", "lmstudio")
    backend = detect_backend()
    assert backend.url == BACKEND_URLS["lmstudio"]
    assert backend.model == DEFAULT_MODELS["lmstudio"]


def test_custom_url_is_honoured(monkeypatch):
    monkeypatch.setenv("NENAPU_LLM", "openai")
    monkeypatch.setenv("NENAPU_LLM_URL", "http://gpu-box:8000/v1")
    assert detect_backend().url == "http://gpu-box:8000/v1"


def test_auto_prefers_anthropic_when_credentials_exist(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    assert detect_backend().name == "anthropic"


def test_unknown_backend_is_rejected(monkeypatch):
    monkeypatch.setenv("NENAPU_LLM", "gpt5-turbo-ultra")
    with pytest.raises(LLMUnavailable, match="unknown NENAPU_LLM"):
        detect_backend()


# --- parsing: small models do not return clean JSON ---


def test_plain_json():
    assert extract_json('{"findings": []}') == {"findings": []}


def test_fenced_json():
    assert extract_json('```json\n{"a": 1}\n```') == {"a": 1}


def test_json_after_commentary():
    assert extract_json('Sure! Here is the result:\n{"a": 1}\nHope that helps.') == {"a": 1}


def test_braces_inside_strings_do_not_confuse_the_scan():
    assert extract_json('{"note": "a } inside", "b": 2}') == {"note": "a } inside", "b": 2}


def test_nested_objects():
    payload = {"merged": [{"text": "x", "source_ids": [1, 2]}]}
    assert extract_json(f"blah {json.dumps(payload)} blah") == payload


def test_unparseable_response_raises():
    with pytest.raises(LLMUnavailable):
        extract_json("I cannot help with that.")
    with pytest.raises(LLMUnavailable):
        extract_json("")


# ---------- the exec backend, which is a network client behind a process ----------


def _fake_run(results):
    """Return a subprocess.run stand-in that plays back a list of outcomes."""
    import subprocess

    calls = []

    def run(cmd, **kwargs):
        calls.append(cmd)
        code, out, err = results[min(len(calls) - 1, len(results) - 1)]
        return subprocess.CompletedProcess(cmd, code, out, err)

    run.calls = calls
    return run


def test_exec_retries_a_transient_failure(monkeypatch):
    """Rate limits and dropped connections arrive as a non-zero exit.

    Three real prompts failed this way and every one succeeded unchanged on a
    later attempt, so a single try turns a transient into a hard error.
    """
    from nenapu import llm

    monkeypatch.setattr(llm, "EXEC_BACKOFF", 0)
    run = _fake_run([(1, "", ""), (0, '{"ok": true}', "")])
    monkeypatch.setattr("subprocess.run", run)

    backend = llm.Backend(name="exec", model="claude -p")
    assert llm._exec_backend("p", {}, None, backend) == {"ok": True}
    assert len(run.calls) == 2


def test_exec_gives_up_and_says_what_it_saw(monkeypatch):
    from nenapu import llm

    monkeypatch.setattr(llm, "EXEC_BACKOFF", 0)
    monkeypatch.setattr("subprocess.run", _fake_run([(1, "", "quota exceeded")]))

    backend = llm.Backend(name="exec", model="claude -p")
    with pytest.raises(LLMUnavailable, match="quota exceeded"):
        llm._exec_backend("p", {}, None, backend)


def test_exec_reports_stdout_when_stderr_is_empty(monkeypatch):
    """Agent CLIs print diagnostics on stdout. Reporting stderr alone produced
    `exited 1: ` with nothing after the colon — the least useful error there is.
    """
    from nenapu import llm

    monkeypatch.setattr(llm, "EXEC_BACKOFF", 0)
    monkeypatch.setattr("subprocess.run", _fake_run([(1, "Error: overloaded", "")]))

    backend = llm.Backend(name="exec", model="claude -p")
    with pytest.raises(LLMUnavailable, match="overloaded"):
        llm._exec_backend("p", {}, None, backend)


def test_exec_never_reports_an_empty_reason(monkeypatch):
    from nenapu import llm

    monkeypatch.setattr(llm, "EXEC_BACKOFF", 0)
    monkeypatch.setattr("subprocess.run", _fake_run([(1, "", "")]))

    backend = llm.Backend(name="exec", model="claude -p")
    with pytest.raises(LLMUnavailable, match="no output"):
        llm._exec_backend("p", {}, None, backend)


def test_exec_does_not_retry_a_timeout(monkeypatch):
    """A command that ran out of time will run out of time again; retrying it
    three times just triples how long the user waits for the same failure."""
    import subprocess

    from nenapu import llm

    calls = []

    def run(cmd, **kwargs):
        calls.append(cmd)
        raise subprocess.TimeoutExpired(cmd, 1)

    monkeypatch.setattr("subprocess.run", run)
    backend = llm.Backend(name="exec", model="claude -p")
    with pytest.raises(LLMUnavailable, match="did not finish"):
        llm._exec_backend("p", {}, None, backend)
    assert len(calls) == 1


# ---------- what `auto` picks, and why ----------


def test_auto_prefers_an_agent_cli_over_a_local_server(monkeypatch):
    """Ordering decided by measurement, not by cost.

    A local 3B on a CPU-only host did not finish an extraction at all: 180s,
    twice, on transcripts of 20k and 24k characters. The same prompts through
    `claude -p` take 83s. A backend that always times out is not the cheaper
    answer, it is no answer.
    """
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)
    monkeypatch.setenv("NENAPU_LLM", "auto")
    monkeypatch.setattr(llm, "_exec_available", lambda: True)
    monkeypatch.setattr(llm, "_probe", lambda *a, **k: True)  # ollama is up too

    assert llm.detect_backend().name == "exec"


def test_auto_still_falls_back_to_a_local_server(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)
    monkeypatch.setenv("NENAPU_LLM", "auto")
    monkeypatch.setattr(llm, "_exec_available", lambda: False)
    monkeypatch.setattr(llm, "_probe", lambda url, path, **k: "11434" in url)

    assert llm.detect_backend().name == "ollama"


def test_credentials_still_win_over_a_cli(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    monkeypatch.setenv("NENAPU_LLM", "auto")
    monkeypatch.setattr(llm, "_exec_available", lambda: True)

    assert llm.detect_backend().name == "anthropic"


def test_a_command_with_shell_operators_is_never_auto_selected(monkeypatch):
    """`auto` may pick up a CLI someone installed. It may not decide to run a
    pipeline — that is a choice the user makes explicitly with NENAPU_LLM."""
    monkeypatch.setattr(llm, "EXEC_COMMAND", "claude -p | tee /tmp/log")
    assert llm._exec_available() is False


def test_a_missing_cli_is_not_available(monkeypatch):
    monkeypatch.setattr(llm, "EXEC_COMMAND", "definitely-not-installed-xyz -p")
    assert llm._exec_available() is False


# ---------- the context window ----------


def test_the_context_window_covers_a_whole_harvested_session():
    """Ollama defaults to 4096 tokens and silently drops the front of anything
    longer. A harvested session is 24,000 characters — about 8,000 tokens — so
    the oldest part of the conversation was being discarded by the server
    without a word, which is exactly where a correction tends to be."""
    ctx = llm._num_ctx([{"content": "x" * 24000}], 1024)
    assert ctx >= 24000 // 3 + 1024


def test_a_short_prompt_does_not_inflate_the_window():
    """Asking for a large window on a small prompt buys nothing and costs KV
    cache on a machine that is also running the user's editor."""
    assert llm._num_ctx([{"content": "hello"}], 512) == 4096


def test_the_window_is_capped():
    assert llm._num_ctx([{"content": "x" * 10_000_000}], 1024) == llm.MAX_NUM_CTX


def test_ollama_is_told_the_window(monkeypatch):
    """The fix is only real if it reaches the wire."""
    sent = {}

    class _Response:
        status = 200

        def __enter__(self):
            return iter([json.dumps(
                {"message": {"content": '{"ok": true}'}, "done": True}
            ).encode()])

        def __exit__(self, *a):
            return False

    def _fake_urlopen(request, timeout=None):
        sent.update(json.loads(request.data))
        return _Response()

    monkeypatch.setattr(llm.urllib.request, "urlopen", _fake_urlopen)
    backend = llm.Backend(name="ollama", model="qwen2.5:3b", url="http://x")
    llm._ollama("y" * 24000, {"type": "object"}, None, backend, 1024)

    assert sent["options"]["num_ctx"] >= 8000
