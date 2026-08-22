"""Three things about the repo itself that nothing in the suite watches.

Requirement (follow-on to the plan "Harden the four incidents into
guarantees", items left open after Phase F):

    1. No test file may define an `xfail` marker that is applied to nothing.
    2. `ruff check` must pass, and the rules it runs must be declared in the
       repo rather than in whoever's shell.
    3. The local agent instruction files must stay out of the repo.

The first is the same failure as F2, one level up. F2 was a test that could not
fail; this is a marker that says "not implemented yet" about work that landed
weeks ago. `MEMORY.md` records a reader being misled by exactly that: thirteen
marker variables survive across seven files, every one applied to zero tests,
every one still carrying its original reason string. A marker definition is not
evidence of anything, and the only way to keep it from being read as evidence
is to make an unapplied one fail.

The second is a gate the suite cannot give. Every test here passes with an
unused import three lines above it, and 18 of them are in the tree right now,
one of them in `src/nenapu/store.py`. `F` is the ruff group that catches the
ones that are bugs rather than taste: names imported and never used, names used
and never defined, a second definition shadowing the first. Line length and
import order are deliberately not selected, because this repo's own test names
run past 88 columns on purpose.

The third is bookkeeping with teeth: `AGENTS.md`, `CLAUDE.md` and `.claude/`
are written by whichever agent runs here and are not part of the project. They
are ignored, and an ignore rule that nothing checks is one `git add -A` away
from being wrong.

What is *not* here: the fourth open item, the 85% injection unused-rate. Every
graded recall in the store ran against the pre-anchoring block, so the number
cannot be judged until roughly a week of sessions has accumulated under the
current one. That is a measurement to re-run, not code to write, and there is
nothing to assert about it that `tests/test_retrieval_decision.py` does not
already assert.
"""

import ast
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
TESTS = ROOT / "tests"


# ---------- 1: a marker that marks nothing ----------


def _marker_definitions(tree: ast.Module) -> dict[str, ast.Assign]:
    """Module-level names bound to a `pytest.mark.xfail(...)` call."""
    found = {}
    for node in tree.body:
        if not isinstance(node, ast.Assign) or not isinstance(node.value, ast.Call):
            continue
        func = node.value.func
        parts = []
        while isinstance(func, ast.Attribute):
            parts.append(func.attr)
            func = func.value
        if isinstance(func, ast.Name):
            parts.append(func.id)
        if "xfail" in parts and "mark" in parts:
            for target in node.targets:
                if isinstance(target, ast.Name):
                    found[target.id] = node
    return found


def _names_read(tree: ast.Module) -> set[str]:
    """Every name the module reads, which is where an applied marker shows up.

    A marker is applied as `@e7`, as `pytestmark = e7`, or inside a
    `pytest.param(..., marks=e7)`. All three read the name; the assignment that
    defines it stores instead, so it does not count itself.
    """
    return {node.id for node in ast.walk(tree)
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)}


def _unapplied(source: str) -> set[str]:
    tree = ast.parse(source)
    return set(_marker_definitions(tree)) - _names_read(tree)


def _test_modules() -> list[Path]:
    return sorted(TESTS.glob("test_*.py"))


def test_there_are_test_modules_to_check():
    """The check below is vacuously true if the glob finds nothing."""
    assert len(_test_modules()) > 20


@pytest.mark.parametrize("module", _test_modules(), ids=lambda p: p.name)
def test_no_test_module_defines_a_marker_it_never_applies(module):
    dead = _unapplied(module.read_text())

    assert not dead, (
        f"{module.name} defines {sorted(dead)} and applies them to nothing. "
        "A marker reading 'not implemented yet' about work that landed is a "
        "false claim about the repo; delete the definition."
    )


def test_the_detector_sees_a_marker_that_is_applied():
    """Guards the check above against passing because it finds nothing.

    All three ways a marker gets applied, so tightening the detector later
    cannot quietly start deleting live markers.
    """
    source = (
        "import pytest\n"
        "live = pytest.mark.xfail(strict=True, reason='pending')\n"
        "also = pytest.mark.xfail(strict=True, reason='pending')\n"
        "third = pytest.mark.xfail(strict=True, reason='pending')\n"
        "pytestmark = also\n"
        "@live\n"
        "def test_one():\n"
        "    pass\n"
        "@pytest.mark.parametrize('x', [pytest.param(1, marks=third)])\n"
        "def test_two(x):\n"
        "    pass\n"
    )

    assert _unapplied(source) == set()


def test_the_detector_catches_a_marker_that_is_applied_to_nothing():
    source = (
        "import pytest\n"
        "e7 = pytest.mark.xfail(strict=True, reason='E7 not implemented yet')\n"
        "def test_one():\n"
        "    pass\n"
    )

    assert _unapplied(source) == {"e7"}


# ---------- 2: the lint gate ----------


def _ruff() -> str | None:
    """The ruff the current interpreter would run, venv first."""
    beside = Path(sys.executable).parent / "ruff"
    if beside.exists():
        return str(beside)
    return shutil.which("ruff")


def test_ruff_is_declared_as_a_dependency():
    """The gate has to be reproducible from the project, not from a machine
    that happens to have ruff on its PATH."""
    pyproject = (ROOT / "pyproject.toml").read_text()

    assert re.search(r"^lint = \[.*\bruff\b.*\]", pyproject, re.M), (
        "pyproject.toml declares no lint extra pinning ruff"
    )


def test_the_selected_rules_are_declared_in_the_repo():
    """Which rules run is part of the project, so two people running
    `ruff check` in this tree get the same answer."""
    pyproject = (ROOT / "pyproject.toml").read_text()

    assert "[tool.ruff.lint]" in pyproject
    assert re.search(r'^select = \[[^]]*"F"', pyproject, re.M), (
        "the F group is what catches unused imports and undefined names"
    )


def test_the_tree_is_clean_under_ruff():
    ruff = _ruff()
    if ruff is None:
        pytest.skip("ruff is not installed in this environment")

    result = subprocess.run(
        [ruff, "check", ".", "--output-format", "concise"],
        cwd=ROOT, capture_output=True, text=True,
    )

    assert result.returncode == 0, result.stdout or result.stderr


# ---------- 3: the agent files are not the project ----------


AGENT_FILES = ("AGENTS.md", "CLAUDE.md", ".claude/settings.json")


@pytest.mark.parametrize("path", AGENT_FILES)
def test_an_agent_instruction_file_is_ignored(path):
    result = subprocess.run(
        ["git", "check-ignore", "-q", "--no-index", path],
        cwd=ROOT, capture_output=True, text=True,
    )

    assert result.returncode == 0, f"{path} is not ignored"


@pytest.mark.parametrize("path", AGENT_FILES)
def test_an_agent_instruction_file_is_not_tracked(path):
    """Ignoring a file that is already tracked changes nothing, so the ignore
    rule only means something while the file stays out of the index."""
    result = subprocess.run(
        ["git", "ls-files", "--error-unmatch", path],
        cwd=ROOT, capture_output=True, text=True,
    )

    assert result.returncode != 0, f"{path} is tracked despite being ignored"
