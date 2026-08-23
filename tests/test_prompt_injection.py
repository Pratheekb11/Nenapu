"""Memory chosen for the prompt the user just typed.

Requirement (Task 8, query-driven hybrid retrieval plan):

`recall_context` runs at SessionStart, before any user text exists, so its idea
of relevance is built from what happened *before*: the last twenty file events,
the branch, the directory name. It has never been able to answer "what is being
asked right now", and the ledger shows the cost -- 2077 injection recalls, 85%
of them never used.

`prompt_context` is the other half. Same store, same budget discipline, same
never-break-the-session contract, but the selection comes from the prompt.

Three properties make it safe to run on every turn rather than once a session.

**It is small and it repeats nothing.** A fact already injected at session
start, or at an earlier prompt in the same session, is already in the model's
context. Sending it again spends tokens to say something twice.

**It does not ratchet.** `search` bumps `use_count`, which feeds the `usage`
ranking term. Firing that on every prompt would inflate the counter for
whatever the ranker already favours, roughly twenty times more often than
today, and the ranker would then favour it more.

**It does not become a transcript.** The store gates verbatim conversation
behind `NENAPU_STORE_MESSAGES` and an opt-in table. Writing the user's literal
prompt into `recalls.query` on every turn would route around that boundary, so
the ledger records the planned terms instead -- stopword-stripped, lowercased,
capped. `retrieval_report` only tests `query != ''`, so grading is unaffected
and the query population it reads finally starts growing.

Scope boundary
--------------
`recall_context` is the highest-blast-radius function in the repo and four test
files pin its exact output. Nothing here touches it. `prompt_context` is a
sibling that reuses `_fit`, `_distinct` and `_token_estimate` unchanged.

Assumed seam, proposed by the plan and not yet in the codebase::

    observer.prompt_context(store, prompt, *, scope, session_id, cwd=None) -> str
    observer.PROMPT_TOKEN_BUDGET, observer.MAX_PROMPT_INJECTED
"""

import pytest

from nenapu import connect
from nenapu.models import Fact, Kind, Origin
from nenapu.observer import (
    INJECTION_TOKEN_BUDGET,
    MAX_PROMPT_INJECTED,
    PROMPT_TOKEN_BUDGET,
    _token_estimate,
    prompt_context,
    recall_context,
)
from nenapu.store import Store

SCOPE = "repo:test@abcd1234"
PROMPT = "please tell me which database the billing service uses"


@pytest.fixture
def store():
    return Store(connect(":memory:"))


def _recall_queries(store):
    return [r["query"] for r in store.conn.execute("SELECT query FROM recalls")]


# --- selection ---------------------------------------------------------------


def test_the_block_answers_the_prompt(store):
    store.write(Fact(text="the billing service stores rows in postgres", scope=SCOPE))
    store.write(Fact(text="the release notes live in docs/releases", scope=SCOPE))

    block = prompt_context(store, PROMPT, scope=SCOPE, session_id="s-1")

    assert "postgres" in block
    assert "docs/releases" not in block


def test_global_and_project_facts_are_both_in_reach(store):
    """Two-tier by construction, the same as the session-start block: how the
    user works travels with them, what a repo does stays in the repo."""
    store.write(Fact(text="the billing service runs on postgres", scope=SCOPE))
    store.write(Fact(text="always mention the billing service in commit messages",
                     kind=Kind.FEEDBACK, origin=Origin.USER_STATED, scope="global"))

    # Both facts carry both of the query's rare terms on purpose. The planner
    # requires the two rarest present terms (`MAX_REQUIRED_TERMS`), so fixtures
    # that split them between two facts would match neither, and the test would
    # be measuring the planner rather than the scoping it is about.
    block = prompt_context(store, "tell me about the billing service",
                           scope=SCOPE, session_id="s-1")

    assert "postgres" in block
    assert "commit messages" in block


def test_another_projects_fact_never_appears(store):
    store.write(Fact(text="the billing service stores rows in mysql",
                     scope="repo:other@9999"))

    block = prompt_context(store, PROMPT, scope=SCOPE, session_id="s-1")

    assert "mysql" not in block


def test_a_weakly_believed_fact_is_not_injected(store):
    """Injection is unasked-for. A fact the store would show on request, it
    should not push into a context window unprompted."""
    store.write(Fact(text="the billing service stores rows in postgres",
                     scope=SCOPE, confidence=0.05))

    block = prompt_context(store, PROMPT, scope=SCOPE, session_id="s-1")

    assert "postgres" not in block


def test_nothing_worth_saying_prints_nothing(store):
    """A hook that emits noise on every prompt is worse than one that emits
    nothing: the reader learns to skip the block."""
    store.write(Fact(text="the release notes live in docs/releases", scope=SCOPE))

    block = prompt_context(store, "what is the weather like today",
                           scope=SCOPE, session_id="s-1")

    assert block == ""


# --- repeating nothing -------------------------------------------------------


def test_a_fact_from_session_start_is_not_repeated(store):
    """It is already in the model's context. Saying it again spends tokens to
    say the same thing twice."""
    store.write(Fact(text="the billing service stores rows in postgres", scope=SCOPE))
    recall_context(store, scope=SCOPE, session_id="s-1")

    block = prompt_context(store, PROMPT, scope=SCOPE, session_id="s-1")

    assert "postgres" not in block


def test_a_fact_from_an_earlier_prompt_is_not_repeated(store):
    store.write(Fact(text="the billing service stores rows in postgres", scope=SCOPE))

    first = prompt_context(store, PROMPT, scope=SCOPE, session_id="s-1")
    second = prompt_context(store, PROMPT, scope=SCOPE, session_id="s-1")

    assert "postgres" in first
    assert second == ""


def test_another_session_still_gets_the_fact(store):
    """Dedup is per session, not global. A new session has an empty context
    and needs to be told."""
    store.write(Fact(text="the billing service stores rows in postgres", scope=SCOPE))
    prompt_context(store, PROMPT, scope=SCOPE, session_id="s-1")

    block = prompt_context(store, PROMPT, scope=SCOPE, session_id="s-2")

    assert "postgres" in block


# --- cost --------------------------------------------------------------------


def test_the_prompt_budget_is_smaller_than_the_session_budget():
    """Paid on every turn rather than once, so it has to be cheaper than the
    block that is paid once."""
    assert PROMPT_TOKEN_BUDGET < INJECTION_TOKEN_BUDGET


def test_the_block_stays_inside_its_budget(store):
    for i in range(40):
        store.write(Fact(
            text=f"the billing service database note number {i} " + "x" * 300,
            scope=SCOPE))

    block = prompt_context(store, PROMPT, scope=SCOPE, session_id="s-1")

    assert _token_estimate(block) <= PROMPT_TOKEN_BUDGET


def test_the_block_carries_at_most_a_few_facts(store):
    for i in range(40):
        store.write(Fact(text=f"the billing service database note number {i}",
                         scope=SCOPE))

    block = prompt_context(store, PROMPT, scope=SCOPE, session_id="s-1")

    assert 0 < len([ln for ln in block.splitlines() if ln.startswith("-")]) \
        <= MAX_PROMPT_INJECTED


def test_recall_does_not_bump_the_usage_counter(store):
    """`usage` is a ranking term. Bumping it on every prompt would inflate it
    for whatever the ranker already likes, which would then rank higher."""
    fact, _ = store.write(Fact(text="the billing service stores rows in postgres",
                               scope=SCOPE))

    prompt_context(store, PROMPT, scope=SCOPE, session_id="s-1")

    assert store.get(fact.id).use_count == 0


# --- the ledger --------------------------------------------------------------


def test_the_ledger_records_terms_and_not_the_prompt(store):
    """The store gates verbatim conversation behind an opt-in table. A hook
    writing the literal prompt here on every turn would route around it."""
    store.write(Fact(text="the billing service stores rows in postgres", scope=SCOPE))

    prompt_context(store, PROMPT, scope=SCOPE, session_id="s-1")

    queries = [q for q in _recall_queries(store) if q]
    assert queries
    assert all(q != PROMPT for q in queries)
    assert all("please" not in q and "tell" not in q for q in queries)
    assert any("billing" in q for q in queries)


def test_a_word_the_store_does_not_hold_never_reaches_the_ledger(store):
    """The property that makes this safe to run on every turn. Terms are kept
    only if the store already contains them, so a prompt carrying a secret
    cannot write the secret here -- there is nothing for it to match."""
    store.write(Fact(text="the billing service stores rows in postgres", scope=SCOPE))

    prompt_context(store, "does the billing service use hunter2 as a password",
                   scope=SCOPE, session_id="s-1")

    assert all("hunter2" not in q for q in _recall_queries(store))


def test_the_recall_counts_as_a_query_not_an_injection(store):
    """`retrieval_report` splits its populations on `query != ''`. These rows
    are what finally give the query population something to measure."""
    store.write(Fact(text="the billing service stores rows in postgres", scope=SCOPE))

    prompt_context(store, PROMPT, scope=SCOPE, session_id="s-1")

    assert all(q for q in _recall_queries(store))


def test_a_prompt_of_stopwords_searches_nothing(store):
    """`plan.is_search` is false here and the recency fallback must not fire:
    every prompt would otherwise log arbitrary recent facts as query recalls
    and poison the population the gate reads."""
    store.write(Fact(text="the billing service stores rows in postgres", scope=SCOPE))

    block = prompt_context(store, "the", scope=SCOPE, session_id="s-1")

    assert block == ""
    assert _recall_queries(store) == []


# --- never breaking the session ----------------------------------------------


@pytest.mark.parametrize("prompt", ["", "   ", None])
def test_an_empty_prompt_is_not_an_error(store, prompt):
    assert prompt_context(store, prompt, scope=SCOPE, session_id="s-1") == ""


def test_a_broken_store_returns_an_empty_block(store):
    store.conn.execute("DROP TABLE fact_entities")
    store.conn.commit()

    assert prompt_context(store, PROMPT, scope=SCOPE, session_id="s-1") == ""


def test_a_throwing_embedder_returns_a_block_anyway(store, monkeypatch):
    """Degradation, not failure: the lexical and belief legs still work."""
    from nenapu import embeddings

    class _Broken:
        def embed(self, texts):
            raise RuntimeError("onnx fell over")

    monkeypatch.setattr(embeddings, "get_embedder", lambda: _Broken())
    store.write(Fact(text="the billing service stores rows in postgres", scope=SCOPE))

    block = prompt_context(store, PROMPT, scope=SCOPE, session_id="s-1")

    assert "postgres" in block
