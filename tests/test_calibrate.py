"""Calibration must catch the failure that accuracy scoring misses.

A model answering `stale` unconditionally scores well against contradicting
evidence and is worthless. These stubs reproduce both behaviours exactly.
"""

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from nenapu.calibrate import calibrate
from nenapu.llm import Backend


class _Handler(BaseHTTPRequestHandler):
    verdict_for = staticmethod(lambda prompt: "stale")

    def log_message(self, *args):
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        prompt = body["messages"][-1]["content"]
        ids = [int(t.rstrip(".")) for t in prompt.split() if t.rstrip(".").isdigit() and len(t.rstrip(".")) >= 3]
        verdict = type(self).verdict_for(prompt)
        content = json.dumps({"findings": [
            {"id": i, "verdict": verdict, "reason": "r"} for i in ids
        ]})
        raw = (json.dumps({"message": {"content": content}, "done": True})
               .encode() + b"\n")
        self.send_response(200)
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)


def _serve(verdict_for):
    handler = type("H", (_Handler,), {"verdict_for": staticmethod(verdict_for)})
    httpd = HTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    return httpd, f"http://127.0.0.1:{httpd.server_address[1]}"


@pytest.fixture
def constant_model():
    """The 0.5B failure: says 'stale' no matter what the evidence says."""
    httpd, url = _serve(lambda prompt: "stale")
    yield url
    httpd.shutdown()


@pytest.fixture
def reading_model():
    """A model whose verdict actually depends on the evidence it was given."""

    def verdict_for(prompt: str) -> str:
        if ">=3.10" in prompt:
            return "stale"       # contradicting
        if "Jenkinsfile" in prompt:
            return "holds"       # confirming
        return "unclear"         # absent

    httpd, url = _serve(verdict_for)
    yield url
    httpd.shutdown()


def test_constant_model_fails_calibration(constant_model):
    result = calibrate(Backend("ollama", "tiny", constant_model), batch_size=4)
    assert not result.passed
    assert not result.responsive
    assert any("not reading it" in f for f in result.failures)


def test_constant_model_fails_despite_scoring_well_on_one_condition(constant_model):
    """The whole point: it looks fine if you only test contradicting evidence."""
    result = calibrate(Backend("ollama", "tiny", constant_model), batch_size=4)
    contradicting = next(r for r in result.results if r.condition == "contradicting")
    confirming = next(r for r in result.results if r.condition == "confirming")
    assert contradicting.agreement == 1.0   # perfect, if that is all you measure
    assert confirming.agreement == 0.0      # and completely wrong here
    assert not result.passed


def test_reading_model_passes(reading_model):
    result = calibrate(Backend("ollama", "tiny", reading_model), batch_size=4)
    assert result.responsive
    assert result.passed, result.failures
    assert result.accuracy == 1.0


def test_unreachable_backend_reports_errors_not_a_pass():
    result = calibrate(Backend("ollama", "tiny", "http://127.0.0.1:1"), batch_size=4)
    assert not result.passed
    assert all(r.error for r in result.results)


def test_failed_calibration_blocks_audit(constant_model, tmp_path):
    from nenapu import connect
    from nenapu.audit import audit
    from nenapu.llm import LLMUnavailable
    from nenapu.models import Fact
    from nenapu.store import Store

    backend = Backend("ollama", "tiny", constant_model)
    store = Store(connect(str(tmp_path / "s.db")))
    store.write(Fact(text="some aged claim"))
    store.conn.execute(
        "INSERT OR REPLACE INTO meta(key, value) VALUES (?, 'fail')",
        (f"calibration:{backend.describe()}",),
    )
    store.conn.commit()

    with pytest.raises(LLMUnavailable, match="failed calibration"):
        audit(store, evidence="anything", backend=backend)


class _InventHandler(_Handler):
    """A model that answers about facts nobody asked about."""

    def do_POST(self):
        self.rfile.read(int(self.headers["Content-Length"]))
        payload = {"message": {"content": json.dumps({"findings": [
            {"id": 9000 + i, "verdict": "stale", "reason": "r"} for i in range(6)
        ]})}}
        raw = json.dumps(payload).encode()
        self.send_response(200)
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)


def test_invented_ids_cannot_inflate_agreement():
    """Six 'stale' verdicts for four facts must not score 150% on a condition
    that expects 'stale'."""
    httpd = HTTPServer(("127.0.0.1", 0), _InventHandler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{httpd.server_address[1]}"
    try:
        result = calibrate(Backend("ollama", "tiny", url), batch_size=4)
    finally:
        httpd.shutdown()

    contradicting = next(r for r in result.results if r.condition == "contradicting")
    assert contradicting.agreement == 0.0     # none of the real facts were answered
    assert contradicting.invented == 6
    assert not result.passed
    assert any("do not exist" in f for f in result.failures)


def test_local_backend_reports_without_acting_by_default(constant_model, tmp_path):
    """Measured behaviour: the best local model tested still called half of a
    set of confirmed facts stale. Applying that unattended fills a store with
    spurious doubt, so local findings are advisory until asked for."""
    from nenapu import connect
    from nenapu.audit import audit
    from nenapu.models import Fact, Status
    from nenapu.store import Store

    import time

    store = Store(connect(str(tmp_path / "s.db")))
    # Match the probe store's id range; the stub keys off 3-digit ids.
    store.conn.execute("INSERT INTO sqlite_sequence(name, seq) VALUES ('facts', 100)")
    old = time.time() - 200 * 86400
    fact, _ = store.write(Fact(text="a perfectly good fact", created_at=old,
                               last_verified_at=old))

    report = audit(store, evidence="anything",
                   backend=Backend("ollama", "tiny", constant_model), batch_size=4)
    assert report.findings                                  # it reported
    assert not report.applied
    assert store.get(fact.id).status == Status.ACTIVE       # but changed nothing

    audit(store, evidence="anything", apply=True,
          backend=Backend("ollama", "tiny", constant_model), batch_size=4)
    assert store.get(fact.id).status == Status.DISPUTED     # opt in, and it acts


def test_a_model_that_doubts_unrelated_facts_fails():
    """The false-positive test. Evidence that says nothing about a fact is not
    grounds for doubting it — a model that fails here disparages a healthy
    store. An earlier version of this gate let exactly that through."""

    def verdict_for(prompt: str) -> str:
        if ">=3.10" in prompt:
            return "stale"        # correct on contradicting
        if "Jenkinsfile" in prompt:
            return "holds"        # correct on confirming
        return "stale"            # but manufactures doubt from a coffee machine

    httpd, url = _serve(verdict_for)
    try:
        result = calibrate(Backend("ollama", "tiny", url), batch_size=4, repeats=1)
    finally:
        httpd.shutdown()

    assert result.responsive          # it does read the evidence
    assert not result.passed          # and is still unfit to audit
    assert any("unrelated evidence" in f for f in result.failures)


def test_repeats_average_out_a_flapping_model():
    """A model that answers differently run to run is not one to trust with an
    audit, and a single sample cannot tell."""
    state = {"n": 0}

    def verdict_for(prompt: str) -> str:
        state["n"] += 1
        return "holds" if state["n"] % 2 else "wrong"   # alternates every call

    httpd, url = _serve(verdict_for)
    try:
        result = calibrate(Backend("ollama", "tiny", url), batch_size=4, repeats=4)
    finally:
        httpd.shutdown()

    assert all(len(r.runs) == 4 for r in result.results)
    assert any(r.unstable for r in result.results)
    assert not result.passed


def test_a_hopelessly_slow_model_aborts_early():
    """Nine probe runs against a model that cannot finish one is an hour spent
    reaching a conclusion the first call already gave."""
    import time as _time

    import nenapu.calibrate as cal

    class Slow(_Handler):
        def do_POST(self):
            _time.sleep(1.2)
            super().do_POST()

    httpd = HTTPServer(("127.0.0.1", 0), Slow)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{httpd.server_address[1]}"
    original, cal.SLOW_CALL_SECONDS = cal.SLOW_CALL_SECONDS, 1.0
    try:
        started = _time.time()
        result = cal.calibrate(Backend("ollama", "slow", url), batch_size=4, repeats=3)
        elapsed = _time.time() - started
    finally:
        cal.SLOW_CALL_SECONDS = original
        httpd.shutdown()

    assert result.aborted
    assert not result.passed
    assert "too slow" in result.failures[0]
    assert elapsed < 4          # stopped after the first call, not all nine
    assert len(result.results) == 1


def test_a_model_that_mislabels_half_of_confirmed_facts_fails():
    """A prompt tweak got qwen2.5:3b to exactly 50% here and 'passed' under the
    old 0.5 bar, while producing the same number of false-doubt errors as the
    baseline. Half wrong on facts the evidence supports is not a pass."""
    calls = {"n": 0}

    def verdict_for(prompt: str) -> str:
        if "Jenkinsfile" in prompt:            # confirming: alternate holds/stale
            calls["n"] += 1
            return "holds" if calls["n"] % 2 else "stale"
        if ">=3.10" in prompt:
            return "stale"
        return "unclear"

    httpd, url = _serve(verdict_for)
    try:
        result = calibrate(Backend("ollama", "tiny", url), batch_size=1, repeats=2)
    finally:
        httpd.shutdown()

    confirming = next(r for r in result.results if r.condition == "confirming")
    assert confirming.agreement == pytest.approx(0.5, abs=0.01)
    assert not result.passed
    assert any("supported facts stale" in f for f in result.failures)


def test_passing_calibration_earns_the_right_to_apply(reading_model, tmp_path):
    """A backend that demonstrably reads evidence should be allowed to act,
    whatever it is called — trust is earned by the probe, not by the name."""
    import time

    from nenapu import connect
    from nenapu.audit import audit
    from nenapu.models import Fact
    from nenapu.store import Store

    backend = Backend("ollama", "proven", reading_model)
    store = Store(connect(str(tmp_path / "s.db")))
    store.conn.execute("INSERT INTO sqlite_sequence(name, seq) VALUES ('facts', 100)")
    old = time.time() - 200 * 86400
    store.write(Fact(text="an aged claim", created_at=old, last_verified_at=old))

    # Untested backend: reports, does not act.
    assert not audit(store, evidence="x", backend=backend, batch_size=4).applied

    store.conn.execute(
        "INSERT OR REPLACE INTO meta(key, value) VALUES (?, 'pass')",
        (f"calibration:{backend.describe()}",),
    )
    store.conn.commit()
    assert audit(store, evidence="x", backend=backend, batch_size=4).applied
