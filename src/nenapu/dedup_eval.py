"""A labelled set of fact pairs, for deciding when two facts are one rule.

Dedupe today is lexical: `distill._similarity` at 0.85. It misses reworded
corrections, and the fix on the table is an embedding cutoff. A cutoff picked
by eye is how the store ended up with duplicates in the first place, so this
builds the yardstick first: real pairs from the live store, labelled, and a
scorer that reports what any decider would merge at any threshold.

Mining draws from four places, each answering a different question:

  * `distilled`  - pairs the current dedupe merged. Did it merge correctly?
  * `superseded` - same subject, changed value. These must never be merged.
  * `near_miss`  - semantically close pairs nothing ever linked. The
                   paraphrases the current dedupe misses live here.
  * `random`     - unrelated pairs of the same kind. Easy negatives.

Exact textual repeats are left out. Every decider gets them right, and they
would only inflate the scores.

The file holds real user facts, so it lives under NENAPU_HOME and never in the
repository. Run as `python -m nenapu.dedup_eval mine|score`.
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path
from typing import Iterable

from . import embeddings
from .db import connect, default_db_path
from .distill import _similarity
from .store import _normalize_value

LABELS = ("same", "update", "different")

# Below this, bge-small calls unrelated English related about as often as not
# (the store's own SEMANTIC_FLOOR measurement), so a pair down there is not a
# near miss of anything.
NEAR_MISS_FLOOR = 0.60

# Near misses are spread across these cosine bands rather than drawn uniformly,
# because a threshold is chosen *somewhere* in this range and every band needs
# examples for its precision to mean anything.
NEAR_MISS_BANDS = (0.60, 0.70, 0.80, 0.90, 1.01)

DEFAULT_PER_SOURCE = 60
THRESHOLDS = tuple(round(0.50 + 0.025 * i, 3) for i in range(20))

_USE_DEFAULT = object()


def default_path() -> Path:
    return default_db_path() / "eval" / "dedup_pairs.jsonl"


# --- mining -------------------------------------------------------------------


def _key(a: int, b: int) -> tuple[int, int]:
    return (a, b) if a < b else (b, a)


def _pair(rows: dict[int, dict], a: int, b: int, source: str) -> dict:
    a, b = _key(a, b)
    return {
        "a_id": a, "b_id": b, "a": rows[a]["text"], "b": rows[b]["text"],
        "kind": rows[a]["kind"], "source": source, "cosine": None,
        "jaccard": _similarity(rows[a]["text"], rows[b]["text"]),
        "label": None, "note": None,
    }


def _linked(conn, column: str) -> list[tuple[int, int]]:
    return [(r[0], r[1]) for r in conn.execute(
        f"SELECT id, {column} FROM facts WHERE {column} IS NOT NULL"
        f" AND {column} IN (SELECT id FROM facts)")]


def _cosines(texts: list[str], embedder) -> list[list[float]]:
    """Pairwise cosine over every fact. numpy when it is there, which it is
    whenever fastembed is; the pure loop only ever runs on small stores."""
    vectors = embedder.embed(texts)
    try:
        import numpy as np
    except ImportError:  # pragma: no cover - fastembed brings numpy
        return [[embeddings.cosine(u, v) for v in vectors] for u in vectors]
    m = np.asarray(vectors, dtype=float)
    norms = np.linalg.norm(m, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    m = m / norms
    return (m @ m.T).tolist()


def mine_pairs(conn, *, embedder=_USE_DEFAULT, per_source: int = DEFAULT_PER_SOURCE,
               seed: int = 0) -> list[dict]:
    """Candidate pairs from the store, unlabelled, at most `per_source` each."""
    if embedder is _USE_DEFAULT:
        embedder = embeddings.get_embedder()
    rng = random.Random(seed)
    rows = {r["id"]: dict(r) for r in conn.execute("SELECT id, text, kind FROM facts ORDER BY id")}
    norm = {i: _normalize_value(r["text"]) for i, r in rows.items()}

    taken: set[tuple[int, int]] = set()
    out: list[dict] = []

    def usable(a: int, b: int) -> bool:
        return a != b and norm[a] != norm[b] and _key(a, b) not in taken

    def take(source: str, candidates: list[tuple[int, int]]) -> None:
        for a, b in candidates[:per_source]:
            taken.add(_key(a, b))
            out.append(_pair(rows, a, b, source))

    # Linked pairs first, so a pair the store already has a verdict on is
    # filed under that verdict and not drawn again as a near miss.
    linked: set[tuple[int, int]] = set()
    for source, column in (("distilled", "distilled_into_id"), ("superseded", "superseded_by_id")):
        found = sorted({_key(a, b) for a, b in _linked(conn, column)})
        linked.update(found)
        found = [p for p in found if usable(*p)]
        rng.shuffle(found)
        take(source, found)

    ids = sorted(rows)
    sims: dict[tuple[int, int], float] = {}
    if embedder is not None and len(ids) > 1:
        matrix = _cosines([rows[i]["text"] for i in ids], embedder)
        bands: list[list[tuple[int, int]]] = [[] for _ in NEAR_MISS_BANDS[:-1]]
        for x, a in enumerate(ids):
            for y in range(x + 1, len(ids)):
                b, c = ids[y], matrix[x][y]
                sims[(a, b)] = c
                if c < NEAR_MISS_FLOOR or (a, b) in linked or not usable(a, b):
                    continue
                for band, (lo, hi) in enumerate(zip(NEAR_MISS_BANDS, NEAR_MISS_BANDS[1:])):
                    if lo <= c < hi:
                        bands[band].append((a, b))
                        break
        for band in bands:
            rng.shuffle(band)
        # Round-robin across bands so a cap still covers the whole range.
        spread = [band[i] for i in range(max(map(len, bands), default=0))
                  for band in bands if i < len(band)]
        take("near_miss", spread)

    by_kind: dict[str, list[int]] = {}
    for i in ids:
        by_kind.setdefault(rows[i]["kind"], []).append(i)
    drawn: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
    kinds = sorted(k for k, members in by_kind.items() if len(members) > 1)
    for _ in range(per_source * 20):
        if len(drawn) >= per_source or not kinds:
            break
        a, b = rng.sample(by_kind[rng.choice(kinds)], 2)
        key = _key(a, b)
        if key in seen or key in linked or not usable(a, b):
            continue
        seen.add(key)
        drawn.append(key)
    take("random", drawn)

    for pair in out:
        c = sims.get((pair["a_id"], pair["b_id"]))
        pair["cosine"] = c
    return out


# --- storage ------------------------------------------------------------------


def save_pairs(pairs: Iterable[dict], path: Path | None = None) -> Path:
    path = Path(path or default_path())
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(p, ensure_ascii=False) + "\n" for p in pairs))
    return path


def load_pairs(path: Path | None = None) -> list[dict]:
    path = Path(path or default_path())
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def merge_labels(new: list[dict], old: list[dict]) -> list[dict]:
    """Fresh scores, kept labels: re-mining must never throw away judgement."""
    given = {(p["a_id"], p["b_id"]): p for p in old if p.get("label")}
    merged = []
    for pair in new:
        prior = given.get((pair["a_id"], pair["b_id"]))
        merged.append({**pair, "label": prior["label"], "note": prior.get("note")}
                      if prior else pair)
    return merged


# --- scoring ------------------------------------------------------------------


def evaluate(pairs: list[dict], field: str, thresholds: Iterable[float]) -> list[dict]:
    """What a decider merging at `score >= threshold` would get right.

    Only `same` is a correct merge. An `update` merged is counted again on its
    own, because merging one silently drops the newer value.
    """
    scored = [p for p in pairs if p.get("label") in LABELS and p.get(field) is not None]
    rows = []
    for t in thresholds:
        tp = fp = fn = tn = updates = 0
        for p in scored:
            merge, same = p[field] >= t, p["label"] == "same"
            tp += merge and same
            fp += merge and not same
            fn += same and not merge
            tn += not same and not merge
            updates += merge and p["label"] == "update"
        rows.append({
            "threshold": t, "n": len(scored), "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": tp / (tp + fp) if tp + fp else 1.0,
            "recall": tp / (tp + fn) if tp + fn else 0.0,
            "updates_merged": updates,
        })
    return rows


def pick_threshold(rows: list[dict], *, min_precision: float = 0.95) -> dict | None:
    """Best recall among thresholds that keep precision. A wrong merge destroys
    a fact; a missed one only costs a duplicate line."""
    ok = [r for r in rows if r["tp"] and r["precision"] >= min_precision]
    if not ok:
        return None
    return max(ok, key=lambda r: (r["recall"], -r["threshold"]))


# --- command line -------------------------------------------------------------


def _report(pairs: list[dict]) -> str:
    labelled = [p for p in pairs if p.get("label") in LABELS]
    lines = [f"{len(labelled)} labelled of {len(pairs)} pairs"]
    for field in ("jaccard", "cosine"):
        rows = evaluate(labelled, field, THRESHOLDS)
        if not rows or not rows[0]["n"]:
            continue
        lines.append(f"\n{field} (n={rows[0]['n']})")
        lines.append("  thr    prec  recall  fp  updates_merged")
        for r in rows:
            lines.append(f"  {r['threshold']:.3f}  {r['precision']:.2f}  {r['recall']:.2f}"
                         f"  {r['fp']:>3}  {r['updates_merged']:>3}")
        best = pick_threshold(rows)
        lines.append(f"  pick: {best['threshold']:.3f} (recall {best['recall']:.2f})"
                     if best else "  pick: none reaches 0.95 precision")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    command = argv[0] if argv else "score"
    path = default_path()
    if command == "mine":
        pairs = merge_labels(mine_pairs(connect()), load_pairs(path))
        save_pairs(pairs, path)
        print(f"{len(pairs)} pairs written to {path}")
    elif command == "score":
        print(_report(load_pairs(path)))
    else:
        print("usage: python -m nenapu.dedup_eval mine|score", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
