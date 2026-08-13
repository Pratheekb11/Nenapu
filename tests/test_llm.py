"""Backend selection and the parsing that local models make necessary."""

import json

import pytest

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
