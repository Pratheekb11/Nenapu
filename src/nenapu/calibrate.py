"""Does the configured model actually read the evidence?

An audit backend is trusted with the store's contents, so it should have to
earn that. The failure mode this catches is not "gets some verdicts wrong" —
it is a model whose verdicts do not depend on the input at all. A 0.5B model
tested against three different evidence sets returned:

    contradicting  [stale, stale, stale, unclear]
    confirming     [stale, stale, stale, wrong]     <- every fact was true
    absent         [stale, stale, stale, stale]     <- evidence about a coffee machine

Scored against the contradicting set alone it looks like 50% accuracy. It is
actually answering `stale` unconditionally, and turned loose on a real store it
would mark every memory disputed in a single pass.

Three probes over the same four facts, no user data involved. A backend that
fails should not be running audits.
"""

from __future__ import annotations

import os
import time
from collections import Counter
from dataclasses import dataclass, field

from .db import connect
from .llm import Backend, LLMUnavailable, detect_backend
from .models import Fact
from .store import Store

PROBE_FACTS = [
    "The project uses Python 3.9 and setuptools",
    "Deployments are manual, run by hand from a laptop",
    "CI runs on Jenkins",
    "The build backend is hatchling",
]

CONTRADICTING = """Current repository state:
- pyproject.toml requires-python = ">=3.10", build backend is hatchling.
- .github/workflows/ci.yml runs the tests. No Jenkins config exists.
- Deploys happen automatically from CI on a tagged release."""

CONFIRMING = """Current repository state:
- setup.py declares python_requires='>=3.9'; the project uses setuptools.
- CI is configured in Jenkinsfile; Jenkins runs every build.
- There is no deploy automation; releases are copied up by hand from a laptop.
- The build backend is hatchling."""

ABSENT = """Current repository state:
- The logo was redesigned last quarter.
- The office coffee machine is on the third floor."""

# condition -> (evidence, verdicts a correct auditor would mostly give)
CONDITIONS: dict[str, tuple[str, set[str]]] = {
    "contradicting": (CONTRADICTING, {"stale", "wrong"}),
    "confirming": (CONFIRMING, {"holds"}),
    "absent": (ABSENT, {"unclear", "holds"}),
}

MIN_COVERAGE = 0.75      # must answer about most of what it was asked
# 0.5 was too lenient: a model that mislabels half of the facts its evidence
# explicitly supports will still fill a store with false doubt, and a threshold
# that a knife-edge 50% clears is not a threshold. Set above the coin flip.
MIN_CONFIRMING = 0.6     # must recognise a fact the evidence supports
MIN_ABSENT = 0.5         # must not manufacture doubt from unrelated evidence
MIN_ACCURACY = 0.6       # overall, across all three conditions
MAX_INVENTED = 1         # answering about facts that do not exist is disqualifying

# A model too slow to answer one four-fact probe is too slow to audit a store,
# so there is nothing to learn from the remaining eight runs. gemma4:8b on CPU
# took longer than this per call and would otherwise have kept the machine busy
# for an hour to reach a conclusion available after the first call.
SLOW_CALL_SECONDS = float(os.environ.get("NENAPU_CALIBRATE_ABORT", "150"))


@dataclass
class Run:
    verdicts: list[str] = field(default_factory=list)
    covered: int = 0
    requested: int = 0
    invented: int = 0
    seconds: float = 0.0
    error: str | None = None


@dataclass
class ConditionResult:
    condition: str
    expected: set[str]
    runs: list[Run] = field(default_factory=list)

    # --- aggregates over runs; a single sample is not a measurement ---

    @property
    def agreements(self) -> list[float]:
        return [
            0.0 if (r.error or not r.requested)
            else sum(1 for v in r.verdicts if v in self.expected) / r.requested
            for r in self.runs
        ]

    @property
    def spread(self) -> tuple[float, float]:
        scores = self.agreements
        return (min(scores), max(scores)) if scores else (0.0, 0.0)

    @property
    def unstable(self) -> bool:
        """Did the same question get materially different answers across runs?"""
        low, high = self.spread
        return (high - low) >= 0.5

    @property
    def verdicts(self) -> list[str]:
        return [v for r in self.runs for v in r.verdicts]

    @property
    def covered(self) -> int:
        return min((r.covered for r in self.runs), default=0)

    @property
    def requested(self) -> int:
        return max((r.requested for r in self.runs), default=0)

    @property
    def invented(self) -> int:
        return max((r.invented for r in self.runs), default=0)

    @property
    def seconds(self) -> float:
        return sum(r.seconds for r in self.runs)

    @property
    def error(self) -> str | None:
        return next((r.error for r in self.runs if r.error), None)

    @property
    def coverage(self) -> float:
        return 0.0 if not self.requested else self.covered / self.requested

    @property
    def agreement(self) -> float:
        """Mean fraction of *real* facts given an acceptable verdict.

        Scored over facts asked, not answers returned — a model emitting six
        verdicts for four facts must not score above 100% by hallucinating ids
        that happen to carry the right word.
        """
        scores = self.agreements
        return sum(scores) / len(scores) if scores else 0.0


@dataclass
class Calibration:
    backend: str
    results: list[ConditionResult] = field(default_factory=list)
    aborted: bool = False

    @property
    def responsive(self) -> bool:
        """Do the verdicts change when the evidence changes?

        The single most important question, and the one accuracy against a
        fixed evidence set cannot answer.
        """
        signatures = {
            tuple(sorted(Counter(r.verdicts).items())) for r in self.results if r.verdicts
        }
        return len(signatures) > 1

    @property
    def failures(self) -> list[str]:
        problems: list[str] = []
        if self.aborted:
            slow = next((r for r in self.results if r.error), None)
            return [slow.error if slow else "calibration aborted"]
        for r in self.results:
            if r.error:
                problems.append(f"{r.condition}: {r.error}")
            elif r.coverage < MIN_COVERAGE:
                problems.append(
                    f"{r.condition}: answered about {r.covered}/{r.requested} facts"
                )
            if r.invented > MAX_INVENTED:
                problems.append(
                    f"{r.condition}: returned verdicts for {r.invented} facts that do "
                    "not exist"
                )
        if not self.responsive:
            problems.append(
                "verdicts did not change when the evidence changed — the model is not "
                "reading it"
            )
        for r in self.results:
            if not r.error and r.unstable:
                low, high = r.spread
                problems.append(
                    f"{r.condition}: unstable across runs ({low:.0%}-{high:.0%})"
                )
        confirming = next((r for r in self.results if r.condition == "confirming"), None)
        if confirming and not confirming.error and confirming.agreement < MIN_CONFIRMING:
            problems.append(
                f"called supported facts stale ({confirming.agreement:.0%} correct on "
                "evidence that confirms every one)"
            )

        absent = next((r for r in self.results if r.condition == "absent"), None)
        if absent and not absent.error and absent.agreement < MIN_ABSENT:
            # The false-positive test, and the one that matters most for a
            # memory store: evidence that says nothing about a fact is not
            # grounds for doubting it. A model that fails here manufactures
            # doubt out of unrelated text and will disparage a healthy store.
            problems.append(
                f"invented doubt from unrelated evidence ({absent.agreement:.0%} correct "
                "on evidence that mentions none of the facts)"
            )

        if not any(r.error for r in self.results) and self.accuracy < MIN_ACCURACY:
            problems.append(
                f"overall agreement {self.accuracy:.0%}, below the {MIN_ACCURACY:.0%} bar"
            )
        return problems

    @property
    def passed(self) -> bool:
        return not self.failures

    @property
    def accuracy(self) -> float:
        graded = [r for r in self.results if r.verdicts]
        return 0.0 if not graded else sum(r.agreement for r in graded) / len(graded)


def _probe_store() -> Store:
    store = Store(connect(":memory:"))
    old = time.time() - 200 * 86400
    # Non-contiguous ids: numbering probe facts 1..4 invites a small model to
    # confuse the id with the list position, which fails the probe for the
    # wrong reason.
    store.conn.execute("INSERT INTO sqlite_sequence(name, seq) VALUES ('facts', 100)")
    for text in PROBE_FACTS:
        store.write(Fact(text=text, created_at=old, last_verified_at=old, confidence=0.8))
        store.conn.execute("UPDATE sqlite_sequence SET seq = seq + 37 WHERE name='facts'")
    return store


def calibrate(
    backend: Backend | None = None,
    *,
    batch_size: int | None = None,
    repeats: int = 3,
) -> Calibration:
    """Run the three probes `repeats` times each. Never touches the user's store.

    Repeats matter even under greedy decoding: batching, prompt ordering, and
    server-side caching all leave enough nondeterminism that a single sample
    can be off by half the scale. A model whose score swings across identical
    runs is not one to hand an audit to, so instability is itself a failure.
    """
    from .audit import audit  # imported here to avoid a cycle

    backend = backend or detect_backend()
    calibration = Calibration(backend=backend.describe())

    for condition, (evidence, expected) in CONDITIONS.items():
        result = ConditionResult(condition=condition, expected=expected)
        for _ in range(max(1, repeats)):
            store = _probe_store()
            run = Run()
            started = time.time()
            try:
                report = audit(
                    store, evidence=evidence, older_than_days=30, apply=False,
                    backend=backend, max_facts=8, batch_size=batch_size,
                )
                invented = set(report.invented)
                run.verdicts = [
                    f.verdict for f in report.findings if f.fact_id not in invented
                ]
                run.invented = len(report.invented)
                run.covered, run.requested = report.covered, report.requested
            except LLMUnavailable as exc:
                run.error = str(exc)
                run.seconds = time.time() - started
                result.runs.append(run)
                break  # a dead backend will not recover on the next attempt
            run.seconds = time.time() - started
            result.runs.append(run)

            if run.seconds > SLOW_CALL_SECONDS:
                run.error = (
                    f"one probe took {run.seconds:.0f}s — too slow to audit a real store"
                )
                calibration.results.append(result)
                calibration.aborted = True
                return calibration

        calibration.results.append(result)

    return calibration
