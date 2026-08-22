"""Optional semantic embedding, and the promise that its absence changes nothing.

Nenapu's retrieval works without this module. FTS5 and the belief layer are the
floor; embedding is a leg added on top for the one failure they cannot cover --
a query that shares no word with the fact that answers it. "How should I write
commit messages" and "commit messages must carry no em dashes" have no term in
common, and no amount of stemming will join them.

Three rules shape everything here.

**It never raises.** The caller is usually a hook that must not break a session,
so unavailability is a return value. `available()` reports, `embed_*` returns
`None`, and a store on a machine with no embedder searches exactly as it did
before this module existed.

**It never downloads on a read path.** fastembed fetches from HuggingFace on
first use, and a cold `UserPromptSubmit` that did so would spend the hook's
whole timeout on it. `_load_backend` refuses to construct a model that is not
already cached; `warm()` is the one entry point permitted to fetch, and only
`nenapu index --warm` calls it.

**It imports nothing heavy at module scope.** This module is reachable from the
store, which is reachable from the banner. A top-level `import fastembed` would
load an ONNX runtime in front of `nenapu --help`.

Vectors are unit-normalised at pack time, so cosine similarity is a bare dot
product. That is what keeps the pure-Python fallback cheap enough to be a real
fallback rather than a theoretical one, which matters because a store file
copied to a machine without NumPy still has to be readable.
"""

from __future__ import annotations

import hashlib
import math
import os
import struct
import threading
from pathlib import Path
from typing import Sequence

# bge-small over nomic-v1.5 deliberately. At this corpus size 384 dimensions
# are plenty, and nomic requires `search_query:` / `search_document:` prefix
# discipline whose failure mode is silent degradation -- the worst kind in a
# component whose output nobody can eyeball.
MODEL_NAME = "BAAI/bge-small-en-v1.5"
DIM = 384

# Generous, because it only bites when the model is pathologically slow. The
# hook's own timeout is 10s and it has other work to do.
DEFAULT_DEADLINE_MS = 2000

_OFF_VALUES = {"0", "off", "false", "no"}

# Memoised per process: constructing an ONNX session is the expensive part, and
# the MCP server is long-lived enough to amortise it across many queries.
_STATE: dict[str, object] = {}
_LOCK = threading.Lock()


# --- configuration -----------------------------------------------------------


def model_cache_dir() -> Path:
    """Where the model files live.

    Ours rather than fastembed's default, so `model_ready()` is a question we
    can answer without importing the library it would otherwise ask.
    """
    override = os.environ.get("NENAPU_EMBED_CACHE")
    if override:
        return Path(override).expanduser()
    root = os.environ.get("NENAPU_HOME")
    base = Path(root).expanduser() if root else Path.home() / ".nenapu"
    return base / "models"


def model_ready() -> bool:
    """True when the model is already on disk.

    The guard behind "never download inside a hook". A cold first prompt
    declines the semantic leg rather than spending the timeout fetching.
    """
    cache = model_cache_dir()
    try:
        return cache.is_dir() and any(cache.iterdir())
    except OSError:
        return False


def _switched_off() -> bool:
    return os.environ.get("NENAPU_EMBEDDINGS", "").strip().lower() in _OFF_VALUES


def reset_cache() -> None:
    """Drop the memoised backend. For tests, and for `--warm` re-probing."""
    with _LOCK:
        _STATE.clear()


# --- backend -----------------------------------------------------------------


class _FastEmbedBackend:
    """Adapter over fastembed, shaped like the fake the tests use.

    One interface, many backends -- the same arrangement `llm.Backend` uses,
    and for the same reason: the thing under test should be the retrieval
    logic, not whichever library happens to produce the floats.
    """

    def __init__(self, model, dim: int = DIM):
        self._model = model
        self.dim = dim

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        return [list(map(float, vec)) for vec in self._model.embed(list(texts))]


def _load_backend():
    """Import fastembed and build an embedder over the cached model.

    Raises rather than returning a sentinel: every caller already wraps this,
    and an exception carries the reason, which `available()` reports.
    """
    from fastembed import TextEmbedding

    if not model_ready():
        raise RuntimeError(
            f"model {MODEL_NAME} is not cached; run `nenapu index --warm` once"
        )
    return _FastEmbedBackend(
        TextEmbedding(model_name=MODEL_NAME, cache_dir=str(model_cache_dir()))
    )


def available() -> tuple[bool, str]:
    """Whether the semantic leg can run, and if not, why.

    Reports rather than raises. Callers are hooks and search paths that must
    degrade, not fail.
    """
    if _switched_off():
        return False, "disabled by NENAPU_EMBEDDINGS"
    if get_embedder() is not None:
        return True, ""
    return False, str(_STATE.get("reason") or "no embedding backend")


def get_embedder():
    """The memoised backend, or `None` when there is not one."""
    if _switched_off():
        return None
    with _LOCK:
        if "embedder" in _STATE:
            return _STATE["embedder"]
        try:
            _STATE["embedder"] = _load_backend()
        except Exception as exc:  # ImportError, missing model, ONNX failure
            _STATE["embedder"] = None
            _STATE["reason"] = f"{type(exc).__name__}: {exc}"
        return _STATE["embedder"]


def warm() -> tuple[bool, str]:
    """Fetch the model. The only place a download is permitted.

    Kept out of every read path on purpose: this is the call that can take a
    minute on a slow connection, and a hook that made it would hang a prompt.
    """
    try:
        from fastembed import TextEmbedding
    except ImportError as exc:
        return False, f"fastembed is not installed: {exc}"
    cache = model_cache_dir()
    cache.mkdir(parents=True, exist_ok=True)
    try:
        TextEmbedding(model_name=MODEL_NAME, cache_dir=str(cache))
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"
    reset_cache()
    return True, ""


# --- embedding ---------------------------------------------------------------


def embed_many(texts: Sequence[str]) -> list[list[float]]:
    """Embed a batch. Empty list when there is no backend or it fails."""
    items = list(texts)
    if not items:
        return []
    embedder = get_embedder()
    if embedder is None:
        return []
    try:
        return embedder.embed(items)
    except Exception:
        return []


def embed_one(text: str) -> list[float] | None:
    out = embed_many([text]) if text else []
    return out[0] if out else None


def embed_query(text: str, *, deadline_ms: int | None = None) -> list[float] | None:
    """Embed a query under a wall-clock deadline.

    This runs inside a hook that fires on every prompt the user types. A slow
    model must cost the prompt its memory block, never its latency, so the work
    happens on a daemon thread the caller is free to abandon.
    """
    if not text:
        return None
    embedder = get_embedder()
    if embedder is None:
        return None

    budget = deadline_ms if deadline_ms is not None else _deadline_from_env()
    box: dict[str, list[float]] = {}

    def run() -> None:
        try:
            vectors = embedder.embed([text])
            if vectors:
                box["vec"] = vectors[0]
        except Exception:
            pass  # a failed embed is a missing leg, not an error

    worker = threading.Thread(target=run, daemon=True)
    worker.start()
    worker.join(budget / 1000.0)
    if worker.is_alive():
        return None  # abandoned; the daemon thread cannot hold up exit
    return box.get("vec")


def _deadline_from_env() -> int:
    raw = os.environ.get("NENAPU_EMBED_DEADLINE_MS", "")
    try:
        return max(1, int(raw))
    except ValueError:
        return DEFAULT_DEADLINE_MS


# --- storage -----------------------------------------------------------------


def pack(vec: Sequence[float]) -> bytes:
    """Normalise to unit length and pack as little-endian float32.

    Normalising here rather than at read time is what lets `dot` stand in for
    cosine on the hot path. Little-endian and float32 are pinned so a store
    file stays readable on another machine and the width cannot drift.
    """
    values = [float(v) for v in vec]
    norm = math.sqrt(sum(v * v for v in values))
    if norm > 0.0:
        values = [v / norm for v in values]
    return struct.pack(f"<{len(values)}f", *values)


def unpack(blob: bytes) -> list[float]:
    """Read a packed vector back.

    Raises on a length that is not a whole number of floats. Returning a short
    vector would score a fact against a different number of dimensions and
    produce a plausible-looking number.
    """
    if len(blob) % 4:
        raise ValueError(f"vector blob of {len(blob)} bytes is not whole float32s")
    return list(struct.unpack(f"<{len(blob) // 4}f", blob))


def text_sha(text: str) -> str:
    """Hash of the exact text embedded, so a stale vector is detectable.

    Whitespace-sensitive by construction: it changes the tokens the model sees,
    so it has to count as different text.
    """
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# --- similarity --------------------------------------------------------------


def _numpy():
    """NumPy if it is importable, else `None`.

    Indirected through a function so the import stays lazy and so the
    pure-Python path can be forced in a test on a machine that has NumPy.
    """
    try:
        import numpy
    except ImportError:
        return None
    return numpy


def dot(a: Sequence[float], b: Sequence[float]) -> float:
    np = _numpy()
    if np is not None:
        return float(np.dot(np.asarray(a), np.asarray(b)))
    return float(sum(x * y for x, y in zip(a, b)))


def cosine(a: Sequence[float], b: Sequence[float]) -> float:
    norm_a = math.sqrt(sum(v * v for v in a))
    norm_b = math.sqrt(sum(v * v for v in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot(a, b) / (norm_a * norm_b)


def similarity_score(a: Sequence[float], b: Sequence[float]) -> float:
    """Cosine clamped at zero.

    The fusion *adds* semantic to confidence rather than gating on it, so a
    negative term would subtract standing from a fact for being unrelated --
    which is not the same claim as being contradicted.
    """
    return max(0.0, cosine(a, b))
