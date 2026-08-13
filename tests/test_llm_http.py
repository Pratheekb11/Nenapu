"""The local-backend wire paths, exercised against a stub server.

Covers the two things that actually break in the field: a server that rejects
`response_format`, and a model that wraps its JSON in prose.
"""

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from nenapu.llm import Backend, LLMUnavailable, structured

SCHEMA = {
    "type": "object",
    "properties": {"findings": {"type": "array", "items": {"type": "object"}}},
    "required": ["findings"],
}


class Stub(BaseHTTPRequestHandler):
    mode = "ok"
    seen: list = []

    def log_message(self, *args):  # keep pytest output clean
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        Stub.seen.append((self.path, body))

        if Stub.mode == "reject_schema" and "response_format" in body:
            self.send_response(400)
            self.end_headers()
            self.wfile.write(b'{"error":"response_format unsupported"}')
            return

        if self.path.endswith("/api/chat"):  # ollama native, NDJSON stream
            body_text = '{"findings": [{"id": 1}]}'
            lines = b"".join(
                json.dumps({"message": {"content": ch}, "done": False}).encode() + b"\n"
                for ch in body_text
            ) + json.dumps({"message": {"content": ""}, "done": True}).encode() + b"\n"
            self.send_response(200)
            self.send_header("Content-Type", "application/x-ndjson")
            self.send_header("Content-Length", str(len(lines)))
            self.end_headers()
            self.wfile.write(lines)
            return
        else:  # openai-compatible
            content = '{"findings": []}'
            if Stub.mode in ("chatty", "reject_schema"):
                content = 'Sure!\n```json\n{"findings": []}\n```\nLet me know.'
            payload = {"choices": [{"message": {"content": content}}]}

        raw = json.dumps(payload).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)


@pytest.fixture
def server():
    Stub.seen = []
    Stub.mode = "ok"
    httpd = HTTPServer(("127.0.0.1", 0), Stub)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}"
    httpd.shutdown()


def test_ollama_sends_the_schema_natively(server):
    backend = Backend("ollama", "qwen2.5:3b", server)
    assert structured("audit these", SCHEMA, backend=backend) == {"findings": [{"id": 1}]}
    path, body = Stub.seen[0]
    assert path == "/api/chat"
    assert body["format"] == SCHEMA  # ollama enforces it server-side
    # Streamed on purpose: it is the only way an abandoned request stops
    # burning CPU server-side. See test_abandoning_a_slow_stream_cancels_it.
    assert body["stream"] is True


def test_openai_compatible_path(server):
    backend = Backend("lmstudio", "local-model", server + "/v1")
    assert structured("audit these", SCHEMA, backend=backend) == {"findings": []}
    path, body = Stub.seen[0]
    assert path == "/v1/chat/completions"
    assert body["response_format"]["json_schema"]["schema"] == SCHEMA


def test_server_that_rejects_response_format_is_retried_plain(server):
    Stub.mode = "reject_schema"
    backend = Backend("openai", "local-model", server + "/v1")
    assert structured("audit these", SCHEMA, backend=backend) == {"findings": []}
    assert len(Stub.seen) == 2                       # rejected, then retried
    assert "response_format" not in Stub.seen[1][1]  # without the schema
    assert "JSON only" in Stub.seen[1][1]["messages"][-1]["content"]


def test_prose_wrapped_json_is_still_parsed(server):
    Stub.mode = "chatty"
    backend = Backend("lmstudio", "local-model", server + "/v1")
    assert structured("audit these", SCHEMA, backend=backend) == {"findings": []}


def test_unreachable_server_is_a_clear_error():
    backend = Backend("lmstudio", "local-model", "http://127.0.0.1:1/v1")
    with pytest.raises(LLMUnavailable, match="cannot reach"):
        structured("hi", SCHEMA, backend=backend)


def test_system_prompt_is_passed_through(server):
    backend = Backend("ollama", "m", server)
    structured("prompt", SCHEMA, system="be terse", backend=backend)
    assert Stub.seen[0][1]["messages"][0] == {"role": "system", "content": "be terse"}


# --- audit coverage: the failure the 0.5B eval exposed ---


class PartialStub(Stub):
    """A model that answers about one fact when asked about four."""

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        PartialStub.seen.append((self.path, body))
        # Echo a verdict for the first id mentioned in the prompt, ignore the rest.
        prompt = body["messages"][-1]["content"]
        first = next((int(t.rstrip(".")) for t in prompt.split()
                      if t.rstrip(".").isdigit()), 1)
        content = json.dumps(
            {"findings": [{"id": first, "verdict": "holds", "reason": "looks fine"}]})
        raw = json.dumps({"message": {"content": content}, "done": True}).encode() + b"\n"
        self.send_response(200)
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)


@pytest.fixture
def partial_server():
    PartialStub.seen = []
    httpd = HTTPServer(("127.0.0.1", 0), PartialStub)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}"
    httpd.shutdown()


def _seeded_store():
    import time

    from nenapu import connect
    from nenapu.models import Fact
    from nenapu.store import Store

    store = Store(connect(":memory:"))
    old = time.time() - 200 * 86400
    for i in range(4):
        store.write(Fact(text=f"aged claim number {i}", created_at=old, last_verified_at=old))
    return store


def test_partial_coverage_is_reported_not_swallowed(partial_server):
    from nenapu.audit import audit

    store = _seeded_store()
    backend = Backend("ollama", "tiny", partial_server)
    report = audit(store, evidence="some evidence", older_than_days=30, apply=False,
                   backend=backend, batch_size=4)

    assert report.requested == 4
    assert report.covered == 1
    assert len(report.missing) == 3      # named, so a partial audit cannot read as clean
    assert report.coverage == 0.25


def test_small_batches_recover_coverage(partial_server):
    # One fact per call is exactly how you get a weak model to answer about all
    # of them — which is what the local default does.
    from nenapu.audit import audit

    store = _seeded_store()
    backend = Backend("ollama", "tiny", partial_server)
    report = audit(store, evidence="some evidence", older_than_days=30, apply=False,
                   backend=backend, batch_size=1)

    assert report.batches == 4
    assert report.covered == 4
    assert report.missing == []


def test_invented_ids_are_reported_and_never_applied(partial_server):
    from nenapu.audit import audit
    from nenapu.models import Status

    store = _seeded_store()
    ids = [f.id for f in store.list_facts()]
    backend = Backend("ollama", "tiny", partial_server)
    # PartialStub echoes the first integer in the prompt; point it at a bogus id.
    report = audit(store, evidence="ignore 9999 entirely", older_than_days=30,
                   apply=True, backend=backend, batch_size=4)
    assert 9999 in report.invented or report.covered <= 1
    for fact_id in ids:
        assert store.get(fact_id).status in (Status.ACTIVE, Status.DISPUTED)


class VerdictStub(Stub):
    """A model that returns a fixed verdict for every fact — including the
    confidently wrong `holds` a 1.5B produced on contradicted evidence."""

    verdict = "holds"

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        prompt = body["messages"][-1]["content"]
        ids = [int(t.rstrip(".")) for t in prompt.split() if t.rstrip(".").isdigit()]
        payload = {"message": {"content": json.dumps({"findings": [
            {"id": i, "verdict": VerdictStub.verdict, "reason": "so says the model"}
            for i in ids
        ]})}}
        raw = json.dumps(payload).encode()
        self.send_response(200)
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)


@pytest.fixture
def verdict_server():
    httpd = HTTPServer(("127.0.0.1", 0), VerdictStub)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}"
    httpd.shutdown()


def test_holds_never_refreshes_a_stale_fact(verdict_server):
    """The decay clock is evidence of checking. A model's opinion is not.

    A weak model that answers `holds` on everything must not be able to keep
    stale memory alive forever — that is precisely what the audit exists to
    stop.
    """
    from nenapu.audit import audit
    from nenapu.store import effective_confidence

    VerdictStub.verdict = "holds"
    store = _seeded_store()
    fact_id = store.list_facts()[0].id
    before = effective_confidence(store.get(fact_id))
    anchor_before = store.get(fact_id).last_verified_at

    audit(store, evidence="anything", older_than_days=30, apply=True,
          backend=Backend("ollama", "tiny", verdict_server), batch_size=4)

    assert store.get(fact_id).last_verified_at == anchor_before
    assert effective_confidence(store.get(fact_id)) == pytest.approx(before, abs=0.01)


def test_untrusted_backend_disputes_rather_than_retires(verdict_server):
    from nenapu.audit import audit
    from nenapu.models import Status

    VerdictStub.verdict = "wrong"
    store = _seeded_store()
    fact_id = store.list_facts()[0].id
    audit(store, evidence="anything", older_than_days=30, apply=True,
          backend=Backend("ollama", "tiny", verdict_server), batch_size=4)
    assert store.get(fact_id).status == Status.DISPUTED  # flagged, not destroyed


def test_trusted_backend_may_retire(verdict_server):
    from nenapu.audit import audit
    from nenapu.models import Status

    VerdictStub.verdict = "wrong"
    store = _seeded_store()
    fact_id = store.list_facts()[0].id
    audit(store, evidence="anything", older_than_days=30, apply=True,
          backend=Backend("ollama", "tiny", verdict_server), batch_size=4, trust=True)
    assert store.get(fact_id).status == Status.RETIRED


def test_a_slow_model_is_reported_not_crashed():
    """An 8B model on CPU can blow through the read timeout mid-call. That is a
    capacity answer, not a stack trace — gemma4:8b produced a raw
    socket.TimeoutError here before this was handled."""
    import socket
    import threading

    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    listener.listen(1)
    port = listener.getsockname()[1]

    def accept_and_stall():
        conn, _ = listener.accept()
        threading.Event().wait(5)  # accept, then never reply
        conn.close()

    threading.Thread(target=accept_and_stall, daemon=True).start()
    try:
        import nenapu.llm as llm_mod
        original, llm_mod.DEFAULT_TIMEOUT = llm_mod.DEFAULT_TIMEOUT, 1
        try:
            with pytest.raises(LLMUnavailable, match="stopped responding|did not respond"):
                structured("hi", SCHEMA,
                           backend=Backend("ollama", "slow", f"http://127.0.0.1:{port}"))
        finally:
            llm_mod.DEFAULT_TIMEOUT = original
    finally:
        listener.close()


# --- exec backend: any CLI that takes a prompt and prints a reply ---


def test_exec_backend_runs_a_command(tmp_path):
    """Someone with an agent CLI installed should not need a second credential
    to run audits."""
    script = tmp_path / "fake_cli.sh"
    script.write_text('#!/bin/sh\ncat > /dev/null\necho \'{"findings": [{"id": 1}]}\'\n')
    script.chmod(0o755)

    backend = Backend("exec", str(script))
    assert structured("audit these", SCHEMA, backend=backend) == {"findings": [{"id": 1}]}


def test_exec_backend_passes_the_prompt_on_stdin(tmp_path):
    seen = tmp_path / "seen.txt"
    script = tmp_path / "echo_cli.sh"
    script.write_text(f'#!/bin/sh\ncat > {seen}\necho \'{{"findings": []}}\'\n')
    script.chmod(0o755)

    structured("THE PROMPT", SCHEMA, system="THE SYSTEM", backend=Backend("exec", str(script)))
    sent = seen.read_text()
    assert "THE PROMPT" in sent and "THE SYSTEM" in sent
    assert "json" in sent.lower()  # the schema is requested inline


def test_exec_backend_tolerates_a_chatty_cli(tmp_path):
    script = tmp_path / "chatty.sh"
    script.write_text(
        '#!/bin/sh\ncat > /dev/null\n'
        'printf "Sure, here you go:\\n\\`\\`\\`json\\n{\\"findings\\": []}\\n\\`\\`\\`\\n"\n'
    )
    script.chmod(0o755)
    assert structured("x", SCHEMA, backend=Backend("exec", str(script))) == {"findings": []}


def test_exec_backend_reports_a_failing_command(tmp_path):
    script = tmp_path / "broken.sh"
    script.write_text('#!/bin/sh\ncat > /dev/null\necho "not logged in" >&2\nexit 3\n')
    script.chmod(0o755)
    with pytest.raises(LLMUnavailable, match="exited 3"):
        structured("x", SCHEMA, backend=Backend("exec", str(script)))


def test_exec_backend_is_never_trusted_implicitly():
    """The command is arbitrary, so nothing may be assumed about what is behind
    it until calibration says otherwise."""
    assert Backend("exec", "claude -p").trusted is False
    assert Backend("anthropic", "claude-opus-5").trusted is True


def test_abandoning_a_slow_stream_cancels_it_server_side():
    """Measured problem, not a hypothetical: with `stream: false` Ollama
    generates the whole reply before writing anything, so a client timeout left
    an 8B model pinning a CPU core for minutes afterwards and starving every
    later job. Streaming means dropping the socket breaks the server's next
    write, which ends the generation.
    """
    import socketserver
    import threading
    import time as _time

    stopped_writing = threading.Event()

    class Endless(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            self.rfile.read(int(self.headers["Content-Length"]))
            self.send_response(200)
            self.send_header("Content-Type", "application/x-ndjson")
            self.end_headers()
            try:
                for _ in range(2000):        # far more than the client will read
                    self.wfile.write(
                        json.dumps({"message": {"content": "x"}, "done": False}).encode()
                        + b"\n"
                    )
                    self.wfile.flush()
                    _time.sleep(0.01)
            except (BrokenPipeError, ConnectionResetError):
                stopped_writing.set()        # the client hung up; we stopped

    class Threaded(socketserver.ThreadingMixIn, HTTPServer):
        daemon_threads = True

    httpd = Threaded(("127.0.0.1", 0), Endless)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{httpd.server_address[1]}"

    import nenapu.llm as llm_mod
    original, llm_mod.DEFAULT_TIMEOUT = llm_mod.DEFAULT_TIMEOUT, 1
    try:
        started = _time.time()
        with pytest.raises(LLMUnavailable, match="exceeded|stopped responding"):
            structured("x", SCHEMA, backend=Backend("ollama", "endless", url))
        elapsed = _time.time() - started
    finally:
        llm_mod.DEFAULT_TIMEOUT = original
        httpd.shutdown()

    assert elapsed < 5, "client should abort at its deadline, not read to the end"
    assert stopped_writing.wait(timeout=5), "server kept generating after the client left"
