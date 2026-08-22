"""The embedder seam, and the promise that its absence changes nothing.

Requirement (Task 2, query-driven hybrid retrieval plan):

* Embedding is an *optional* capability. `nenapu` installs without it, CI runs
  without it, and a store copied to a machine that lacks it must still open,
  search and answer. Every entry point here therefore reports unavailability
  rather than raising, and every caller downstream is allowed to treat `None`
  as "no semantic leg today".
* Vectors are unit-normalised at pack time, so cosine similarity is a bare dot
  product. This is not a micro-optimisation; it is what makes the pure-Python
  fallback cheap enough to be a real fallback rather than a theoretical one.
* Two numeric paths exist -- NumPy when it is importable, plain Python when it
  is not -- and they must agree. A store that ranks differently depending on
  whether NumPy happened to be installed is a store nobody can debug.
* Nothing here may import `fastembed` or `numpy` at module import time.
  `nenapu.embeddings` is reachable from `nenapu.store`, which is reachable
  from the banner, and putting an ONNX runtime import in the path of every CLI
  invocation would be felt on every single command.
* No test in this file may touch the network or load a real model. The whole
  file runs against a deterministic fake.

Assumed seam, proposed by the plan and not yet in the codebase::

    nenapu.embeddings
        MODEL_NAME, DIM
        available()        -> (bool, reason)      never raises
        model_ready()      -> bool                is the model cached locally
        get_embedder()     -> object | None       memoised per process
        embed_one(text)    -> list[float] | None
        embed_many(texts)  -> list[list[float]]
        embed_query(text)  -> list[float] | None  deadline-bounded
        pack(vec) -> bytes / unpack(blob) -> list[float]
        dot(a, b) / cosine(a, b) / similarity_score(a, b)
        text_sha(text) -> str
"""

import hashlib
import math
import struct
import subprocess
import sys
import time

import pytest

from nenapu import embeddings


class _FakeEmbedder:
    """Deterministic stand-in for fastembed.

    Hash-seeded so the same text always yields the same vector, which is what
    lets later tasks assert on ranking without a model in the loop.
    """

    dim = 8

    def __init__(self, *, delay: float = 0.0, fail: bool = False):
        self._delay = delay
        self._fail = fail
        self.calls = 0

    def embed(self, texts):
        self.calls += 1
        if self._delay:
            time.sleep(self._delay)
        if self._fail:
            raise RuntimeError("backend exploded")
        return [self._vector(t) for t in texts]

    def _vector(self, text):
        digest = hashlib.sha256(text.encode()).digest()
        return [digest[i] / 255.0 for i in range(self.dim)]


@pytest.fixture
def fake(monkeypatch):
    embedder = _FakeEmbedder()
    monkeypatch.setattr(embeddings, "get_embedder", lambda: embedder)
    return embedder


def _unit(values):
    norm = math.sqrt(sum(v * v for v in values))
    return [v / norm for v in values]


# --- packing -----------------------------------------------------------------


def test_a_vector_round_trips_through_the_blob():
    vec = _unit([0.1, -0.2, 0.3, 0.4])

    out = embeddings.unpack(embeddings.pack(vec))

    assert len(out) == len(vec)
    for got, want in zip(out, vec):
        assert got == pytest.approx(want, abs=1e-6)


def test_the_blob_is_little_endian_float32():
    """Pinned so a store file stays readable on a machine with a different
    byte order, and so the width cannot drift to float64 unnoticed."""
    vec = _unit([1.0, 0.0, 0.0, 0.0])

    blob = embeddings.pack(vec)

    assert len(blob) == 4 * len(vec)
    assert struct.unpack("<4f", blob)[0] == pytest.approx(1.0, abs=1e-6)


def test_unpacking_a_truncated_blob_raises():
    """Returning a short vector would silently score a fact against a
    different number of dimensions, which reads as a plausible number."""
    blob = embeddings.pack(_unit([1.0, 2.0, 3.0, 4.0]))

    with pytest.raises(ValueError):
        embeddings.unpack(blob[:-1])


def test_packing_normalises_to_unit_length():
    """Cosine is a bare dot product only if this holds. Every consumer of
    `dot` in the retrieval path depends on it."""
    blob = embeddings.pack([3.0, 4.0, 0.0, 0.0])

    out = embeddings.unpack(blob)

    assert math.sqrt(sum(v * v for v in out)) == pytest.approx(1.0, abs=1e-6)
    assert out[0] == pytest.approx(0.6, abs=1e-6)
    assert out[1] == pytest.approx(0.8, abs=1e-6)


def test_a_zero_vector_does_not_divide_by_zero():
    out = embeddings.unpack(embeddings.pack([0.0, 0.0, 0.0, 0.0]))

    assert all(v == 0.0 for v in out)


# --- similarity --------------------------------------------------------------


def test_cosine_of_a_vector_with_itself_is_one():
    vec = _unit([0.2, 0.5, -0.1, 0.9])

    assert embeddings.cosine(vec, vec) == pytest.approx(1.0, abs=1e-6)


def test_orthogonal_is_zero_and_opposite_is_minus_one():
    a = [1.0, 0.0]
    b = [0.0, 1.0]

    assert embeddings.cosine(a, b) == pytest.approx(0.0, abs=1e-6)
    assert embeddings.cosine(a, [-1.0, 0.0]) == pytest.approx(-1.0, abs=1e-6)


def test_the_similarity_score_clamps_negatives_to_zero():
    """The fusion *adds* semantic to confidence rather than gating on it, so a
    negative term would subtract standing from a fact for being unrelated,
    which is not the same claim as being contradicted."""
    assert embeddings.similarity_score([1.0, 0.0], [-1.0, 0.0]) == 0.0
    assert embeddings.similarity_score([1.0, 0.0], [1.0, 0.0]) == pytest.approx(1.0)


def test_the_numpy_and_pure_python_paths_agree(monkeypatch):
    """Forced rather than skipped: the fallback has to be exercised on
    machines that do have NumPy, or it is only tested where nobody looks."""
    pairs = [
        (_unit([0.1, 0.2, 0.3, 0.4]), _unit([0.4, 0.3, 0.2, 0.1])),
        (_unit([1.0, 0.0, 0.0, 0.0]), _unit([0.0, 1.0, 0.0, 0.0])),
        (_unit([-0.5, 0.5, -0.5, 0.5]), _unit([0.5, 0.5, 0.5, 0.5])),
    ]
    fast = [embeddings.dot(a, b) for a, b in pairs]

    monkeypatch.setattr(embeddings, "_numpy", lambda: None)
    slow = [embeddings.dot(a, b) for a, b in pairs]

    for quick, plain in zip(fast, slow):
        assert quick == pytest.approx(plain, abs=1e-6)


# --- availability ------------------------------------------------------------


def test_availability_reports_rather_than_raises(monkeypatch):
    def explode(*_args, **_kwargs):
        raise ImportError("no fastembed here")

    monkeypatch.setattr(embeddings, "_load_backend", explode)
    embeddings.reset_cache()

    ok, reason = embeddings.available()

    assert ok is False
    assert reason


def test_the_kill_switch_turns_the_leg_off(monkeypatch):
    """An escape hatch that does not depend on uninstalling anything, for the
    case where the embedder is installed and misbehaving."""
    monkeypatch.setenv("NENAPU_EMBEDDINGS", "off")
    embeddings.reset_cache()

    ok, reason = embeddings.available()

    assert ok is False
    assert reason
    assert embeddings.get_embedder() is None


def test_the_model_is_not_ready_when_the_cache_is_absent(monkeypatch, tmp_path):
    """The guard behind "never download inside a hook". A cold first prompt
    must decline, not spend the hook timeout fetching from HuggingFace."""
    monkeypatch.setattr(embeddings, "model_cache_dir", lambda: tmp_path / "absent")

    assert embeddings.model_ready() is False


# --- embedding ---------------------------------------------------------------


def test_embedding_is_deterministic_for_the_same_text(fake):
    first = embeddings.embed_one("the datastore is postgres")
    second = embeddings.embed_one("the datastore is postgres")

    assert first == second


def test_embed_many_returns_one_vector_per_text(fake):
    out = embeddings.embed_many(["one", "two", "three"])

    assert len(out) == 3
    assert all(len(v) == _FakeEmbedder.dim for v in out)


def test_embedding_returns_none_when_there_is_no_backend(monkeypatch):
    monkeypatch.setattr(embeddings, "get_embedder", lambda: None)

    assert embeddings.embed_one("anything") is None
    assert embeddings.embed_many(["anything"]) == []
    assert embeddings.embed_query("anything") is None


def test_a_throwing_backend_is_a_none_not_an_exception(monkeypatch):
    monkeypatch.setattr(embeddings, "get_embedder", lambda: _FakeEmbedder(fail=True))

    assert embeddings.embed_one("anything") is None
    assert embeddings.embed_query("anything") is None


def test_the_query_deadline_is_honoured(monkeypatch):
    """This runs inside a hook that fires on every prompt the user types. A
    slow model must cost the prompt its memory block, never its latency."""
    monkeypatch.setenv("NENAPU_EMBED_DEADLINE_MS", "50")
    monkeypatch.setattr(embeddings, "get_embedder", lambda: _FakeEmbedder(delay=2.0))

    started = time.time()
    out = embeddings.embed_query("a query the model is too slow to embed")
    elapsed = time.time() - started

    assert out is None
    assert elapsed < 1.0


def test_a_query_within_the_deadline_still_returns(monkeypatch, fake):
    monkeypatch.setenv("NENAPU_EMBED_DEADLINE_MS", "5000")

    assert embeddings.embed_query("a query the model can embed in time") is not None


# --- text hashing ------------------------------------------------------------


def test_the_text_hash_is_stable_and_whitespace_sensitive():
    """Whitespace changes the tokens the model sees, so it has to count as a
    different text or `index_missing` would leave a stale vector in place."""
    assert embeddings.text_sha("a fact") == embeddings.text_sha("a fact")
    assert embeddings.text_sha("a fact") != embeddings.text_sha("a  fact")
    assert embeddings.text_sha("a fact") != embeddings.text_sha("a fact ")


# --- import cost -------------------------------------------------------------


def test_importing_the_module_does_not_import_the_backend():
    """`nenapu.embeddings` is reachable from the store, which is reachable
    from the banner. A top-level fastembed import would put an ONNX runtime
    load in front of every `nenapu` command, including `--help`."""
    code = (
        "import sys; import nenapu.embeddings; "
        "print('fastembed' in sys.modules, 'numpy' in sys.modules, "
        "'onnxruntime' in sys.modules)"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )

    assert out.stdout.strip() == "False False False"
