"""The missing working-memory tier: `messages` + a real `--no-infer`.

Requirement (Task 16, priority-ordered task list, Addendum 1 point 5 /
Addendum 2 point 4 of the plan): "`messages` tier + real `--no-infer` |
working memory / 'last 10 messages'; privacy-gated, default off." Depends
on Task 8 (the ingest queue this rides through).

Today the DB holds only extracted, redacted *facts*. `_turns_from`
(`observer.py:156-184`) flattens a transcript in memory at Stop, the model
reads it, and the turns are discarded — there is no message or turn ever
persisted, and no short-term/working-memory tier at all.

This is a real privacy change, not a free extension: storing raw turns means
verbatim conversation lands on disk for the first time. The plan is explicit
that this must be:

* **config-gated, default off** — a store must not start persisting raw
  conversation just because this code shipped;
* **redacted before insert** — the harvest-time redaction invariant
  (`IMPLEMENTATION_NOTES.md:179-197`) has to hold for this path too, using
  the existing `observer.redact()`, or secrets reach the store by a route
  that did not exist before;
* reachable via `nenapu learn <path> --no-infer`, which stores turns
  verbatim and skips the model entirely — useful for debugging extraction
  quality without spending a call.

Assumes a `messages(id, session_id, seq, role, text, created_at)` table and
a `NENAPU_STORE_MESSAGES` environment gate (following the existing
`NENAPU_AGENT` / `NENAPU_NO_BANNER` convention), with a `--no-infer` option
added to `nenapu learn`.
"""

import os
import subprocess
import sys

from nenapu import connect

# ---------- schema ----------


def test_messages_table_exists():
    conn = connect(":memory:")
    columns = {r["name"] for r in conn.execute("PRAGMA table_info(messages)")}
    assert columns >= {"id", "session_id", "seq", "role", "text", "created_at"}


def test_an_existing_store_gains_the_messages_table_on_reconnect(tmp_path):
    path = tmp_path / "old.db"
    connect(str(path))
    reopened = connect(str(path))
    tables = {r["name"] for r in reopened.execute(
        "SELECT name FROM sqlite_master WHERE type = 'table'"
    )}
    assert "messages" in tables


# ---------- the CLI surface ----------


def _run(args, tmp_path, **env):
    return subprocess.run(
        [sys.executable, "-m", "nenapu.cli", *args],
        capture_output=True, text=True,
        env={**os.environ, "PYTHONPATH": "src", "NENAPU_NO_BANNER": "1", **env},
    )


def _transcript(tmp_path, lines):
    path = tmp_path / "t.jsonl"
    path.write_text("\n".join(lines) + "\n")
    return path


def _turn(role, text):
    import json

    return json.dumps({
        "type": role, "message": {"role": role, "content": [{"type": "text", "text": text}]},
    })


def test_no_infer_is_off_by_default_even_with_the_flag(tmp_path):
    """The gate is the load-bearing part: `--no-infer` alone must not be
    enough to start writing verbatim conversation to disk. A store must opt
    in explicitly (`NENAPU_STORE_MESSAGES=1`) before any raw text lands."""
    db = tmp_path / "s.db"
    transcript = _transcript(tmp_path, [
        _turn("user", "the staging DB password is hunter2, please note it"),
        _turn("assistant", "noted"),
    ])

    _run(["learn", str(transcript), "--no-infer", "--db", str(db)], tmp_path)

    conn = connect(str(db))
    count = conn.execute("SELECT COUNT(*) c FROM messages").fetchone()["c"]
    assert count == 0, "messages were stored despite the privacy gate being off"


def test_no_infer_stores_verbatim_turns_when_the_gate_is_on(tmp_path):
    db = tmp_path / "s.db"
    transcript = _transcript(tmp_path, [
        _turn("user", "fix the booking overlap bug"),
        _turn("assistant", "done, added the constraint"),
    ])

    result = _run(["learn", str(transcript), "--no-infer", "--db", str(db)], tmp_path,
                  NENAPU_STORE_MESSAGES="1")

    assert result.returncode == 0, result.stdout + result.stderr
    conn = connect(str(db))
    rows = conn.execute("SELECT role, text FROM messages ORDER BY seq").fetchall()
    texts = [r["text"] for r in rows]
    assert any("booking overlap" in t for t in texts)
    assert any("added the constraint" in t for t in texts)


def test_no_infer_never_calls_the_model(tmp_path, monkeypatch):
    """The whole point: cheap, useful for debugging extraction quality
    without spending a call. Driven in-process (CliRunner) rather than via
    subprocess, since a subprocess would run in its own interpreter and a
    patch here would not reach it."""
    from unittest.mock import patch

    monkeypatch.setenv("NENAPU_STORE_MESSAGES", "1")
    db = tmp_path / "s.db"
    transcript = _transcript(tmp_path, [_turn("user", "hello")])

    from typer.testing import CliRunner

    from nenapu.cli import app

    with patch("nenapu.llm.structured") as mock_structured:
        result = CliRunner().invoke(app, [
            "learn", str(transcript), "--no-infer", "--db", str(db),
        ])
        assert result.exit_code == 0, result.output
        mock_structured.assert_not_called()


def test_messages_are_redacted_before_insert(tmp_path):
    """Same invariant `observer.redact()` already enforces for the harvest
    that feeds the model — it has to hold here too, or raw secrets reach the
    store by a route that did not exist before."""
    db = tmp_path / "s.db"
    transcript = _transcript(tmp_path, [
        _turn("user", "AWS_SECRET_ACCESS_KEY=wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY"),
    ])

    _run(["learn", str(transcript), "--no-infer", "--db", str(db)], tmp_path,
        NENAPU_STORE_MESSAGES="1")

    conn = connect(str(db))
    rows = conn.execute("SELECT text FROM messages").fetchall()
    joined = " ".join(r["text"] for r in rows)
    assert "wJalrXUtnFEMI" not in joined


def test_messages_preserve_role_and_order(tmp_path):
    db = tmp_path / "s.db"
    transcript = _transcript(tmp_path, [
        _turn("user", "first message"),
        _turn("assistant", "second message"),
        _turn("user", "third message"),
    ])

    _run(["learn", str(transcript), "--no-infer", "--db", str(db)], tmp_path,
        NENAPU_STORE_MESSAGES="1")

    conn = connect(str(db))
    rows = conn.execute("SELECT role, text FROM messages ORDER BY seq").fetchall()
    assert [r["role"] for r in rows] == ["user", "assistant", "user"]
    assert [r["text"] for r in rows] == ["first message", "second message", "third message"]
