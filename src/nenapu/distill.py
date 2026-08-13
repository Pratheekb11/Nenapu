"""Compression and distillation.

Left alone, a memory store grows monotonically and the export block turns into
a wall. Distillation folds a cluster of related facts into one that carries the
same information in fewer tokens. Originals are archived, not deleted — the
distilled fact points back at its sources, so nothing is unrecoverable.

Two tiers, cheapest first:
  1. `dedupe` — exact and near-exact duplicates, no model call.
  2. `distill` — LLM merge of a related cluster, used only when a scope is
     actually over budget.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from .llm import Backend, detect_backend, structured
from .models import Fact, Origin, Status, now
from .store import Store, _content, _normalize_value, effective_confidence

# Rough token estimate; good enough to decide whether to spend a model call.
CHARS_PER_TOKEN = 4


def estimate_tokens(facts: list[Fact]) -> int:
    return sum(len(f.text) for f in facts) // CHARS_PER_TOKEN


# When one text is much shorter than the other it is usually *more specific*
# ("we use postgres" vs "we use postgres for the analytics warehouse only"), so
# containment would archive the informative one. Only treat containment as
# duplication when the two are close to the same length.
LENGTH_PARITY = 0.8


def _similarity(a: str, b: str) -> float:
    ta, tb = set(_normalize_value(a).split()), set(_normalize_value(b).split())
    if not ta or not tb:
        return 0.0
    shorter, longer = sorted((len(ta), len(tb)))
    jaccard = len(ta & tb) / len(ta | tb)
    if shorter >= 4 and shorter / longer >= LENGTH_PARITY:
        # Same length, same content words, different filler: a real duplicate.
        return max(jaccard, len(ta & tb) / shorter)
    return jaccard


def _prefix_tokens(ordered: list[str], threshold: float) -> list[str]:
    """The tokens a candidate must share to have any chance of matching.

    Standard prefix filter: order both token sets the same way and keep the
    first `len - ceil(threshold * len) + 1`. Two sets overlapping by at least
    `threshold` must then share a prefix token, so indexing on the prefix alone
    cannot miss a real duplicate.
    """
    if not ordered:
        return []
    keep = len(ordered) - math.ceil(threshold * len(ordered)) + 1
    return ordered[: max(1, keep)]


def _candidate_pairs(facts: list[Fact], threshold: float):
    """Yield (fact, plausible twins) using an inverted index, not every pair.

    Brute force is O(n^2) with a token-set comparison per pair: fine for the
    twenty facts it was written against, 103s at three thousand, and it did not
    return at all on five thousand.

    Tokens are ordered by global rarity so the discriminating words land in the
    prefix; indexing on "the" would hand back the entire store as candidates.
    """
    frequency: dict[str, int] = {}
    contents: dict[int, set[str]] = {}
    for fact in facts:
        content = _content(fact.text)
        contents[fact.id] = content
        for token in content:
            frequency[token] = frequency.get(token, 0) + 1

    index: dict[str, list[Fact]] = {}
    for fact in facts:
        own = contents[fact.id]
        ordered = sorted(own, key=lambda t: (frequency[t], t))
        candidates: dict[int, Fact] = {}
        for token in _prefix_tokens(ordered, threshold):
            for other in index.get(token, ()):
                # Free length filter: sets of very different sizes cannot
                # overlap enough to matter.
                a, b = len(own), len(contents[other.id])
                if a and b and min(a, b) / max(a, b) >= threshold:
                    candidates[other.id] = other
            index.setdefault(token, []).append(fact)
        yield fact, list(candidates.values())


def dedupe(store: Store, *, scope: str | None = None, threshold: float = 0.85) -> int:
    """Archive near-duplicates, keeping the most-believed copy. No model call."""
    facts = sorted(store.list_facts(scope=scope, limit=10_000),
                   key=effective_confidence, reverse=True)

    # Exact repeats are the common case and need no comparison at all.
    doomed: list[tuple[Fact, Fact]] = []
    by_exact: dict[str, Fact] = {}
    survivors: list[Fact] = []
    for fact in facts:
        exact = _normalize_value(fact.text)
        twin = by_exact.get(exact)
        if twin is None:
            by_exact[exact] = fact
            survivors.append(fact)
        else:
            doomed.append((fact, twin))

    # The containment branch of `_similarity` can accept a pair whose Jaccard
    # sits below `threshold`, so the filter runs looser than the verifier does.
    # A filter that matched the verifier exactly would drop real duplicates.
    kept: set[int] = set()
    for fact, candidates in _candidate_pairs(survivors, threshold * LENGTH_PARITY):
        twin = next(
            (c for c in candidates
             if c.id in kept and _similarity(c.text, fact.text) >= threshold),
            None,
        )
        if twin is None:
            kept.add(fact.id)
        else:
            doomed.append((fact, twin))

    with store.transaction():
        for fact, twin in doomed:
            store.conn.execute(
                "UPDATE facts SET status=?, distilled_into_id=?, updated_at=? WHERE id=?",
                (Status.ARCHIVED, twin.id, now(), fact.id),
            )
            store._journal("dedupe", fact_id=fact.id, actor="distill",
                           detail=f"into {twin.id}")
    return len(doomed)


DISTILL_SCHEMA = {
    "type": "object",
    "properties": {
        "merged": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "text": {"type": "string"},
                    "kind": {"type": "string"},
                    "key": {"type": "string"},
                    "source_ids": {"type": "array", "items": {"type": "integer"}},
                },
                "required": ["text", "kind", "key", "source_ids"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["merged"],
    "additionalProperties": False,
}

SYSTEM = """You compress an agent's memory without losing information.

Merge only facts that are genuinely about the same thing. Preserve every \
specific: numbers, names, paths, versions, and the reason behind a decision. \
Drop only redundancy and filler. A fact that has no good merge partner must be \
left out of your output entirely — do not restate it alone.

key: a short dotted identifier for what the merged fact is about (e.g. \
'db.connection', 'user.editor'), used later to detect contradictions."""


@dataclass
class DistillReport:
    scope: str | None
    tokens_before: int
    tokens_after: int
    deduped: int
    merged: int
    created_ids: list[int]

    @property
    def saved_pct(self) -> float:
        if not self.tokens_before:
            return 0.0
        return 100.0 * (1 - self.tokens_after / self.tokens_before)


def distill(
    store: Store,
    *,
    scope: str | None = None,
    token_budget: int = 1500,
    max_facts: int | None = None,
    use_llm: bool = True,
    backend: Backend | None = None,
) -> DistillReport:
    """Bring a scope under `token_budget`, cheapest tier first."""
    before = estimate_tokens(store.list_facts(scope=scope, limit=10_000))
    deduped = dedupe(store, scope=scope)

    facts = store.list_facts(scope=scope, limit=10_000)
    merged_count = 0
    created: list[int] = []

    if use_llm and estimate_tokens(facts) > token_budget:
        backend = backend or detect_backend()
        if max_facts is None:
            max_facts = 60 if backend.name == "anthropic" else 15
        # Oldest-and-least-believed first: recent, high-confidence facts keep
        # their exact wording.
        pool = sorted(facts, key=effective_confidence)[:max_facts]
        listing = "\n".join(f"- id={f.id} [{f.kind}] {f.text}" for f in pool)
        prompt = (
            f"Compress these memory entries to fit roughly {token_budget} tokens total.\n\n"
            f"{listing}\n"
        )
        result = structured(prompt, DISTILL_SCHEMA, system=SYSTEM, backend=backend,
                            max_tokens=max(2048, 120 * len(pool)))
        by_id = {f.id: f for f in pool}

        for item in result.get("merged", []):
            sources = [i for i in item.get("source_ids", []) if i in by_id]
            if len(sources) < 2:
                continue  # a "merge" of one is just a rewrite; skip it
            new = Fact(
                text=item["text"],
                kind=item.get("kind") or by_id[sources[0]].kind,
                scope=scope or by_id[sources[0]].scope,
                key=item.get("key") or None,
                origin=Origin.DISTILLED,
                origin_ref=f"distilled from {sources}",
                confidence=max(effective_confidence(by_id[i]) for i in sources),
            )
            stored, _ = store.write(new, actor="distill")
            created.append(stored.id)
            merged_count += len(sources)
            for sid in sources:
                store.conn.execute(
                    "UPDATE facts SET status=?, distilled_into_id=?, updated_at=? WHERE id=?",
                    (Status.ARCHIVED, stored.id, now(), sid),
                )
            store._journal("distill", fact_id=stored.id, actor="distill", detail=str(sources))
        store.conn.commit()

    after = estimate_tokens(store.list_facts(scope=scope, limit=10_000))
    return DistillReport(scope, before, after, deduped, merged_count, created)
