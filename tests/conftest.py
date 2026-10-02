"""Shared test helpers."""

import pytest

from nenapu.approval import approve, pending


@pytest.fixture
def approve_all():
    """Approve every check currently awaiting review in a store.

    Explicit rather than autouse: tests that exercise the approval gate itself
    must see the default-deny behaviour, and a fixture that silently blessed
    every command would quietly delete the guarantee it is meant to protect.
    """

    def _approve(store) -> int:
        waiting = pending(store.conn)
        for fact_id, _origin, command in waiting:
            approve(store.conn, command, fact_id=fact_id, by="test")
        return len(waiting)

    return _approve


@pytest.fixture(autouse=True)
def _plain_cli_output(monkeypatch):
    """Keep CLI help and errors plain text under CI.

    Typer forces a rich terminal when GITHUB_ACTIONS, FORCE_COLOR or PY_COLORS
    is set, and rich then styles `--redate` as `-` + `-redate`, so a substring
    check for the flag fails on GitHub Actions and passes locally. Subprocesses
    inherit os.environ, so clearing it here covers them too.
    """
    for name in ("GITHUB_ACTIONS", "FORCE_COLOR", "PY_COLORS"):
        monkeypatch.delenv(name, raising=False)
