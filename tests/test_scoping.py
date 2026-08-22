"""Two-tier scope: user-level facts follow you everywhere, project facts stay
in their repo.

Requirement (Task 1, priority-ordered task list, Phase 2 of the plan):

* A new `project_scope(cwd)` helper resolves the git root (falling back to
  `cwd` outside a repo) and returns a stable id shaped like
  `repo:<basename>@<sha1(abspath)[:8]>`, so two different clones that happen
  to share a directory name do not collide.
* Facts tier by `kind`: `user` and `feedback` stay `global`; `project`,
  `environment` and `reference` default to the caller's project scope.
* `Store.list_facts` and `Store.search` accept a *sequence* of scopes, not
  just one string, because recall needs `["global", project_scope(cwd)]` in
  one query.

Measured against the live store before this change: 367/367 facts are
`scope='global'` — scoping exists in the schema (`db.py:30`) but nothing
ever derives or filters on it. `docs/plan` Verification step 3 is the
acceptance test for the whole feature: two repos, same DB, an `environment`
fact written in each must not leak into the other, while a `feedback` fact
written in either must be visible from both.
"""

import subprocess

import pytest

from nenapu import connect
from nenapu.models import Fact, Kind
from nenapu.store import Store


def _git_repo(path):
    path.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "init", "-q"], cwd=path, check=True)
    subprocess.run(["git", "config", "user.email", "t@example.com"], cwd=path, check=True)
    subprocess.run(["git", "config", "user.name", "t"], cwd=path, check=True)
    (path / "README.md").write_text("x")
    subprocess.run(["git", "add", "README.md"], cwd=path, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "init"], cwd=path, check=True)
    return path


@pytest.fixture
def store():
    return Store(connect(":memory:"))


# ---------- project_scope() ----------


def test_project_scope_is_stable_for_the_same_repo(tmp_path):
    from nenapu.store import project_scope

    repo = _git_repo(tmp_path / "proj")
    assert project_scope(str(repo)) == project_scope(str(repo))


def test_project_scope_is_the_same_from_a_subdirectory(tmp_path):
    """A recall-hook running from `backend/app/` must resolve the same
    project scope as one running from the repo root, or the two-tier split
    silently fragments into one scope per subdirectory."""
    from nenapu.store import project_scope

    repo = _git_repo(tmp_path / "proj")
    sub = repo / "backend" / "app"
    sub.mkdir(parents=True)

    assert project_scope(str(sub)) == project_scope(str(repo))


def test_two_repos_with_the_same_basename_do_not_collide(tmp_path):
    """The whole reason the id hashes the absolute path: `basename` alone
    would merge every checkout named e.g. `backend` into one scope."""
    from nenapu.store import project_scope

    a = _git_repo(tmp_path / "clones" / "backend")
    b = _git_repo(tmp_path / "elsewhere" / "backend")

    assert project_scope(str(a)) != project_scope(str(b))


def test_project_scope_falls_back_to_cwd_outside_a_repo(tmp_path):
    """No `.git` anywhere above `cwd`: must not raise, and must still be
    deterministic for the same directory."""
    from nenapu.store import project_scope

    plain = tmp_path / "not-a-repo"
    plain.mkdir()

    first = project_scope(str(plain))
    second = project_scope(str(plain))
    assert first == second
    assert first != "global"


def test_project_scope_is_never_the_bare_global_string(tmp_path):
    from nenapu.store import project_scope

    repo = _git_repo(tmp_path / "proj")
    assert project_scope(str(repo)) != "global"


# ---------- list_facts / search accept a sequence of scopes ----------


def test_list_facts_accepts_a_sequence_of_scopes(store):
    store.write(Fact(text="global corrections apply everywhere", scope="global"))
    store.write(Fact(text="this repo uses alembic", scope="repo:a@11111111"))
    store.write(Fact(text="that repo uses a different migration tool",
                     scope="repo:b@22222222"))

    facts = store.list_facts(scope=["global", "repo:a@11111111"])
    texts = {f.text for f in facts}

    assert "global corrections apply everywhere" in texts
    assert "this repo uses alembic" in texts
    assert "that repo uses a different migration tool" not in texts


def test_list_facts_still_accepts_a_single_scope_string(store):
    """Backward compatible: every existing caller passes a bare string."""
    store.write(Fact(text="only in global", scope="global"))
    store.write(Fact(text="only in a repo", scope="repo:a@11111111"))

    facts = store.list_facts(scope="global")

    assert {f.text for f in facts} == {"only in global"}


def test_search_accepts_a_sequence_of_scopes(store):
    store.write(Fact(text="ollama context window is 4096", scope="global"))
    store.write(Fact(text="ollama is not used in this repo", scope="repo:a@11111111"))
    store.write(Fact(text="ollama runs on a different port here",
                     scope="repo:b@22222222"))

    hits = store.search("ollama", scope=["global", "repo:a@11111111"])
    texts = {f.text for f, _score, _why in hits}

    assert "ollama context window is 4096" in texts
    assert "ollama is not used in this repo" in texts
    assert "ollama runs on a different port here" not in texts


# ---------- tiering by kind ----------


@pytest.mark.parametrize("kind", [Kind.USER, Kind.FEEDBACK])
def test_user_and_feedback_facts_default_to_global(tmp_path, kind, monkeypatch):
    """A correction about how the user likes commits must follow them into
    every repo, not stay pinned to whichever one they said it in."""
    from typer.testing import CliRunner

    from nenapu.cli import app

    repo = _git_repo(tmp_path / "proj")
    db = tmp_path / "s.db"
    monkeypatch.chdir(repo)
    runner = CliRunner()
    result = runner.invoke(app, [
        "remember", "always squash before merging", "--kind", str(kind),
        "--db", str(db),
    ])

    assert result.exit_code == 0, result.output
    store = Store(connect(str(db)))
    facts = store.list_facts(scope="global", kind=str(kind))
    assert any(f.text == "always squash before merging" for f in facts)


@pytest.mark.parametrize("kind", [Kind.PROJECT, Kind.ENVIRONMENT, Kind.REFERENCE])
def test_project_environment_and_reference_facts_default_to_the_project_scope(
    tmp_path, kind, monkeypatch,
):
    """An Alembic revision or a `DATABASE_URL` belongs to the repo it was
    learned in, not to every session on the machine — the OOH_Marketplace
    leak the plan measured against the live store."""
    from typer.testing import CliRunner

    from nenapu.cli import app
    from nenapu.store import project_scope

    repo = _git_repo(tmp_path / "proj")
    db = tmp_path / "s.db"
    monkeypatch.chdir(repo)
    runner = CliRunner()
    result = runner.invoke(app, [
        "remember", "the alembic head is c7f1a9d4", "--kind", str(kind),
        "--db", str(db),
    ])

    assert result.exit_code == 0, result.output
    store = Store(connect(str(db)))
    expected_scope = project_scope(str(repo))
    facts = store.list_facts(scope=expected_scope, kind=str(kind))
    assert any(f.text == "the alembic head is c7f1a9d4" for f in facts)
    assert not store.list_facts(scope="global", kind=str(kind), status=None) or all(
        f.text != "the alembic head is c7f1a9d4"
        for f in store.list_facts(scope="global", kind=str(kind))
    )
