"""nenapu — portable, self-verifying memory and skill substrate for AI agents.

Not an agent. A store that any harness plugs into: MCP for Claude Code / Cursor,
a CLI and HTTP API for everything else, and a file exporter for harnesses that
only read CLAUDE.md.
"""

import threading

from .db import connect
from .models import Decay, Fact, Kind, Origin, Skill, Status, VerifyStatus
from .skills import SkillStore
from .store import Store, effective_confidence

__version__ = "0.1.0"

__all__ = [
    "connect",
    "Store",
    "SkillStore",
    "Fact",
    "Skill",
    "Kind",
    "Origin",
    "Decay",
    "Status",
    "VerifyStatus",
    "effective_confidence",
    "ThreadLocalStores",
    "__version__",
]


def open_store(path=None) -> tuple[Store, SkillStore]:
    """Convenience: one connection, both stores. Single-threaded callers only."""
    conn = connect(path)
    return Store(conn), SkillStore(conn)


class ThreadLocalStores:
    """Hands out stores bound to the calling thread.

    sqlite3 connections are thread-affine, and both the HTTP API and the MCP
    server dispatch synchronous handlers onto a worker threadpool. WAL mode
    makes one connection per thread the simplest correct answer — writers
    serialize at the file level, readers never block.
    """

    def __init__(self, path=None) -> None:
        self.path = path
        self._local = threading.local()

    def __call__(self) -> tuple[Store, SkillStore]:
        pair = getattr(self._local, "pair", None)
        if pair is None:
            conn = connect(self.path)
            pair = (Store(conn), SkillStore(conn))
            self._local.pair = pair
        return pair
