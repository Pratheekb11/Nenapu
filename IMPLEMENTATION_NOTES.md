# Implementation notes

Why things are built the way they are, and what measurement changed my mind.
Read this before altering the mechanisms — several of them look arbitrary until
you know which failure they exist to prevent.

## Data model

Every fact is a row in `facts` carrying, beyond its text:

| field | why it exists |
|---|---|
| `origin` | `user_stated` outranks `agent_inferred`. An agent's guess must never silently overwrite what you said. |
| `decay_class` | Half-lives: volatile 14d, medium 90d, slow 365d, immutable never. A port number rots faster than a coding preference. |
| `key` | A dotted subject id (`db.port`). Two facts sharing a key are competing values for one subject — that is what makes contradiction detectable. |
| `verify_cmd` | A shell command that proves the fact. The only signal in the system that is not an opinion. |
| `good_recalls` / `bad_recalls` | Track record: did acting on this fact go well? |
| `suspect_since` / `suspect_reason` | Set when something this fact rests on was falsified. |

Nothing is deleted. `superseded`, `archived`, `suspect` and `retired` rows stay
with pointers to what replaced them, plus a `journal` row per action.

## Confidence

```
belief = asserted × origin_weight × decay(age) × verify_signal
                  × outcome_signal(good, bad) × support_signal
```

Six independent signals because each catches what the others miss. A passing
check resets the decay clock; a failing one collapses belief 10×. Recall track
record is Laplace-smoothed so a single bad run does not condemn a fact and an
ungraded fact is neither rewarded nor punished.

## The two mechanisms nobody else has

**Falsification cascade** (`graph.py`). Facts declare what they rest on. When a
root fails its check, is superseded, or is retired, every descendant becomes
`suspect` — visibly resting on nothing rather than quietly keeping its old
confidence. Recovery reinstates dependents unless some *other* parent is still
broken. The walk is depth-capped and cycle-safe.

**Edges build themselves.** Nobody hand-maintains a dependency graph, so the
graph is inferred: recall facts A and B, write C in the same session, and C is
linked to them as observed causality, weighted below a declared edge. Without
this the feature would be technically present and practically unused.

**Recall ledger** (`outcomes.py`). Every fact surfaced into a task is logged
with a `recall_id` and graded later by whichever signal arrives — a failing
check, a contradiction, an explicit report from the harness, or a human. Two of
the four need no cooperation from anything, which is why the loop closes even
when nobody wires it up.

## The agentic layer (`observer.py`, `setup_wizard.py`)

The original surface was MCP. That was wrong, and the reason is worth writing
down: **MCP is request/response, so the server only ever sees a tool call.** It
cannot watch a conversation go past. Memory therefore gets written when the
agent decides to write it — and the agent that has just been corrected is
precisely the one least likely to stop and file a note about it. A memory layer
built on that premise measures well and learns nothing.

Claude Code hooks are the exception, and the whole design now rests on them:

| hook | command | what it does |
|---|---|---|
| `SessionStart` | `nenapu recall-hook` | prints memory to stdout; the harness puts it in the model's context. The agent reads it without asking. |
| `Stop` | `nenapu learn --stdin --detach` | reads the finished transcript, extracts corrections and decisions, writes them. |

MCP is kept for Cursor/VS Code/Codex, which have no hook API. It is the weaker
mode and the docs say so.

### Three things real transcripts broke

**A fixed tail read harvests almost nothing.** Transcripts here reach 55MB and
are overwhelmingly tool traffic. A 400KB tail of a busy one yielded 2,400
characters of actual conversation — a correction made twenty minutes earlier
was simply invisible. The window now starts at 400KB and grows ×4 until enough
real conversation has been harvested. Reading the whole file instead was the
obvious fix and the wrong one: 187MB RSS in a hook that runs after every
session. Seeking to the tail keeps it at 1MB.

**The Stop hook was being killed before it finished.** Extraction is a model
call over an entire session; measured through `claude -p` on real transcripts
it takes **83 seconds**. The hook was configured with a 60-second timeout, so
on any real session it was terminated and nothing was ever written. The failure
is silent — the session ends normally and the store is simply empty. Hence
`--detach`: the hook re-execs itself with `start_new_session=True` and returns
in milliseconds, so the extraction outlives both the hook timeout and the
teardown of the session's process group.

**"Exited 1" with an empty reason.** Two of three real transcripts failed with
`` `claude -p` exited 1: `` and nothing after the colon. Two bugs stacked: the
error reported `stderr` only, and agent CLIs print their diagnostics on
*stdout*. The same prompts then succeeded unchanged on a later attempt — they
were transient upstream failures. Fixed in both directions: the reason now
falls back to stdout, and the `exec` backend retries three times with
exponential backoff. Timeouts are deliberately *not* retried; a command that
ran out of time will run out of time again, and retrying only triples the wait
for the same failure.

### Suspect facts are injected below the confidence floor

`recall_context` filters on believability — except for `suspect` facts, which
are exempt on purpose. A suspect fact is penalised *because* its foundation
fell, which pushes it under the floor and silences the warning exactly when it
matters most. They are listed separately, under "do not rely on these", never
mixed into what the agent is told it knows. Without this the falsification
cascade would be visible in `nenapu doubts` and invisible where decisions are
actually made.

### Consent

`nenapu init` and the first-run walkthrough edit `~/.claude/settings.json` and
editor MCP configs — files the user owns and may have hand-edited. Every write
backs up first, merges rather than replaces, and is idempotent. A non-TTY run
(a pipe, a CI job) is **not** consent: it prints what it would do and changes
nothing. `--yes` exists so automation can opt in explicitly.

### Three more things the first real install broke

The hooks were wired, the tests were green, and the layer was still learning
nothing. Each of these was found by running it, not by reading it.

**An installed hook is a version too.** The `Stop` hook on this machine still
read `nenapu learn --stdin` with a 60-second timeout — the exact
silently-killed configuration `--detach` was written to fix. `install_hooks`
matched on "does an entry mention nenapu" and skipped, so no upgrade would ever
have repaired it. It now compares against the config it wants and replaces its
own stale entries in place, leaving anyone else's hooks on the same event
alone. A memory layer that cannot fix its own installation is a memory layer
that stays broken quietly.

**Ollama drops the front of a long prompt without saying so.** `ollama ps`
reported `CONTEXT 4096` while a harvested session runs to 24,000 characters —
roughly 8,000 tokens. Everything past the window is discarded server-side, and
nothing in the response mentions it. What gets discarded is the *oldest* part
of the conversation, which is where a correction usually is: the user objects,
then the work goes on for another twenty minutes. `num_ctx` is now sized to the
prompt, rounded to a power of two, capped at 16,384.

**`auto` never considered the CLI already installed on the machine.** It went
straight to the local 3B, which on a CPU-only host did not finish an extraction
at all — 180s, twice, on transcripts of 20,000 and 24,000 characters. The same
prompt through `claude -p` takes 83s. `auto` now prefers the `exec` backend
when its command is on PATH, and refuses to auto-select anything containing a
shell operator: picking up a CLI someone installed is reasonable, deciding to
run a pipeline is not. On the transcript that previously extracted nothing, the
new default extracted twelve facts.

### The recursion that preferring an agent CLI creates

An agent CLI is a harness. It fires its own `Stop` hook when it finishes, and
our `Stop` hook is what started it — so an extraction through `claude -p` would
start another extraction, forever. This was measured, not assumed: a bare
`claude -p "reply with exactly: ok"` wrote to `~/.nenapu/observe.log`.

The extraction carries `NENAPU_OBSERVING=1`, and the **hook path only** stands
down when it sees it. Guarding the whole command instead was the first attempt
and was wrong in the way that matters: the detached child is invoked with a
plain path argument and is itself the extraction, so refusing it meant never
observing anything at all. The test suite caught that within a minute, which is
the argument for the test existing.

### What the store and the harvest give away

Two things a stranger's machine made obvious that this one never would.

**The store was world-readable.** `~/.nenapu/nenapu.db` came out `0644` inside
a `0755` directory, because nothing ever set a mode and the process umask
decided. That file holds facts extracted from private sessions. The file is now
created by `os.open(..., 0o600)` rather than left to the driver — sqlite would
create it with the umask, and the gap between that and a `chmod` is enough on a
shared box — the directory is `0700`, and `-wal`/`-shm` are narrowed once the
driver has made them. An existing loose store is tightened on the next open,
because fixing only new installs leaves exactly the people who trusted it
earliest exposed.

**A transcript is not a document the user wrote for us.** It is whatever went
past: a pasted `.env`, a curl with a bearer token, a key echoed by a failing
command. Redaction runs at harvest and nowhere else, because harvest is
upstream of both things that outlive the session — the model call and the
store. Anywhere later and the secret has already been sent.

Shaped credentials are matched by prefix. Everything else is matched by *what
it is called*, since a secret's value is not distinguishable from any other
short string: `DB_PASSWORD=hunter2swordfish` goes, `max_tokens=2048` and
`NENAPU_LLM_TIMEOUT=180` stay. That direction of error is the safe one, and
both directions are pinned by tests — over-redacting produces a memory layer
that records nothing worth having, which is its own kind of failure.

A URL keeps everything but the password: `postgres://admin:pw@db:5432/app`
becomes `postgres://admin:[redacted:url-password]@db:5432/app`, because the
host and database are the part worth remembering. It is a deny list, and a deny
list is never finished — a credential with no recognisable shape, in something
not named like one, gets through. The README says so rather than implying
otherwise.

## The activity ledger and what closes a loop

Facts answer "what is true here". They cannot answer "where did I leave off",
because that is a record of work rather than a claim about the world. So
sessions, file events and commits are their own tables, filled from two
sources with no model call on the path.

**Tool calls and git are both needed, and neither is enough.** Git knows the
net effect and is the only source that can report a *deletion* — files die by
`rm` or `git rm`, and nothing in the transcript names them. Tool calls know
the sequence, the tool, and everything touched outside a repository. A path
both sources see is recorded once, from the tool event, or every
`files_touched` count doubles.

Four git edge cases decide correctness, and each one is a test. A rename is
stored as a delete plus a create, because the ledger has no `renamed` op and
"where did models.py go" has to answer from both ends. A merge is diffed
`before..after` rather than against its first parent, or everything the side
branch did disappears. A detached HEAD reports no branch, because
`--abbrev-ref` answers the literal string `HEAD` there and would group
unrelated sessions under a branch by that name. Everything is read through
`git -C <path>`, so a linked worktree reports its own HEAD rather than the
main checkout's — `~/.claude/projects/` on this machine already contains a
worktree entry, so that one is real rather than hypothetical.

**`git_head_before` is the field that cannot be recovered later.** By the time
a session ends, the commit it started from is only knowable if something wrote
it down, so the SessionStart hook opens the ledger row and the Stop path
finishes it.

**Closure is biased toward closing.** An open loop is closed by a commit
touching the path it named, a file written that matches it, or — for a loop
nobody could attach a path to — a commit whose subject is plainly about it.
Reading a file closes nothing; opening a file to look at it is the most common
thing a session does. Neither does work that happened *before* the loop was
opened, or the whole mechanism closes every loop against the same session's
earlier edits and is silently inert. The asymmetry is deliberate: being told
you missed something you shipped last month destroys trust in the block
permanently, while a reminder that never arrives costs one forgotten task.

**Old loops go quiet rather than shouting.** They age on the same medium decay
curve facts use and drop out of injection under the same floor, staying
visible to `nenapu pending --all`. Nothing is deleted.

## The extractor was never shown the store

The extraction could only ever propose ADD, because it had never seen a fact.
Measured over 367 live facts: 12 groups of exact duplicates once filler words
are stripped, 275 near-duplicate pairs at Jaccard ≥ 0.6, and the same Ollama
context-window fact independently re-learned **five times**.

The fix is one FTS query over the session's own words, and the retrieved facts
go into the same prompt with their ids. The schema then carries `op` and
`target_id`, so the model can say "this updates 12" or "already recorded".

**What comes back is a proposal.** A `target_id` the model was not shown is
rejected and demoted to an add — real ids are guessable, and the 1.5b in the
calibration table invented nine of them for four facts, which was harmless
while every op was an add and is exactly what this schema makes dangerous. A
`user_stated` fact is never reworded by a model's reading of a session; the
recurrence is still counted, because being told the same thing again is true
whoever owns the wording. A missing `op` means add, so a backend that knows
nothing about the field behaves as it did before.

## The injected block was about every project at once

`recall_context` was called with no scope, so a session in one backend was
told about another project's Ollama context window. It now derives the project
from the session's cwd and asks for that project plus global.

Three of the four sections come from the ledger rather than from belief, and
every one of them is capped. The block is prepended to every request, so one
refactor session touching two hundred files would otherwise spend the whole
budget saying so. An empty section is omitted rather than rendered as a header
with nothing under it — a header that says nothing still costs tokens on every
request.

"Changed since you were last here" is one git call, from the commit the last
session ended on to HEAD now. A recorded commit can become unreachable after a
rebase or a pruned branch, and there the section is simply absent: a
SessionStart hook that raises is a session that starts knowing nothing.

## The hook queues; the worker does everything else

The queue was built so that sessions ending together could not fan out into
concurrent extractions, and then the Stop hook kept forking one of its own —
so the queue only ever serialised the watcher. Worse, everything downstream
of the queue (`run_maintenance_tick`, loop closure, ledger capture on the
extraction side) rides on `worker.drain`, and `drain` only ran from `nenapu
watch`. On a machine that uses hooks and never starts the watcher, none of
the self-maintenance ran at all.

The hook now writes one row and spawns `nenapu drain`. Three details decide
whether that is an improvement or a rename:

**Capture moved with it.** Leaving the ledger write on the hook would mean
two processes reading the same transcript for two halves of the same job.
The worker does both, in order: deterministic and free first, so a model
backend that is down costs the facts rather than the record.

**Dedupe covers pending work only.** Two hooks for one session — a retry, or
the watcher and the hook reaching the same file — must not buy two
extractions of identical content. A *resumed* session has appended new
material and has to be read again, so a finished job is no reason to refuse.

**The lock lives beside the store.** One fixed path meant a test store and
the real one blocking each other, and the hook passes the worker nothing but
`--db`. `worker.lock` sits next to the database file it protects.

The installed hook command string did not change, so no machine has to
re-run `nenapu init` to get any of this.

## The watcher polls, and refuses to guess

Only Claude Code can announce that a session ended. Everything else writes a
file and says nothing.

**An adapter is data.** A glob and a parser in a list, so adding an agent is
registering an entry rather than editing the observer. Claude Code and Codex
are registered, and both ship the transcript they were probed against under
`tests/fixtures/transcripts/`; a test refuses any adapter that does not.
Gemini, OpenCode and Cursor are absent because this machine does not have
them — a glob nobody has watched match is a feature that reports success and
captures nothing. `nenapu watch --probe` is how someone else checks a glob
against their own machine without ingesting anything.

**The parser was decorative until it was used.** `TranscriptFormat.parse`
was stored and never called: the tick enqueued a path and the extraction
re-read it with Claude Code's parser whatever wrote it. Measured on a real
Codex rollout, that harvests **0 characters** against 5,120 for the Codex
parser, so a registered adapter would have discovered sessions, queued them,
spent nothing and reported success. `worker._ingest` now resolves the parser
by the job's agent.

A Codex rollout also spells its own metadata differently — everything is
wrapped in `payload`, with `session_id` on `session_meta` and `cwd` on both
that and `turn_context` — so `session_meta_from` reads both spellings.
Without it a Codex session gets no ledger row and its facts fall back to the
`global` scope. Its *file events* are still unread: Codex records what it
touched in `patch_apply_end` stdout and in `custom_tool_call` arguments,
neither of which is a `tool_use` block, so `nenapu where` answers for Claude
Code only.

**"Finished" is measured across ticks, not read off the mtime.** A session
that ended two minutes ago and one still being written look the same to
`stat`; only the tick history separates them, so a transcript is read once its
size has held still for the quiet window. The recorded ingestion length is
kept as well, so a resumed session that appends is picked up again while an
untouched one is never read twice.

**An agent whose Stop hook is installed is skipped.** The unique index would
absorb the duplicate facts, but not the 83 seconds spent producing them.

## Decisions that came from measurement

Each of these was wrong on first attempt. The measurement is recorded because
the reasoning that produced the wrong answer was perfectly plausible.

**Contradiction has an inverted burden of proof.** Facts sharing a `key` are
competing values, so a *different* value is a conflict by default — not
something that must clear a similarity bar. The first version required low word
overlap, which let "cache backend is redis" and "cache backend is memcached"
both stay active.

**`holds` from a model does not reset the decay clock unless the backend is
calibrated.** A model saying "looks fine" is the same guess that wrote the
fact, made again. But making it *always* inert turned the audit into a one-way
ratchet: a true fact the audit had just confirmed still decayed to the floor.
The compromise is a discounted soft-verification, worth strictly less than a
passing shell command.

**Local audits are report-only by default.** The best local model tested still
called half of a set of *confirmed* facts stale. Applying that unattended fills
a store with spurious doubt.

**Trust is earned by calibration, not by backend name.** The first version
hardcoded `anthropic` as trusted, so a CLI agent that demonstrably read
evidence was permanently distrusted while a cloud model got trust without ever
being probed.

## Calibration (`calibrate.py`)

An audit backend gets to mark your memory wrong, so it has to earn that.
Accuracy against contradicting evidence cannot distinguish a careful auditor
from one that answers `stale` unconditionally — both score well. So the probe
runs the same four facts past three evidence sets (refuting, confirming,
irrelevant) and checks whether the **verdicts change with the evidence**.

Measured, greedy decoding, three runs per probe:

| model | contradicting | confirming | absent | |
|---|---|---|---|---|
| qwen2.5:0.5b | 2/4 | 0/4 | 0/4 | answers `stale` regardless of input |
| qwen2.5:1.5b | 2/4 | 2/4 | 0/4 | invents ids — 9 verdicts for 4 facts |
| qwen2.5:3b | 75% | **25%** | 100% | reads, but treats *mention* as contradiction |
| frontier via `exec` | 75% | 75% | 100% | passes at 83–92% |

The discriminating axis is `confirming` alone. Prompt engineering did not
rescue 3B: an explicit prior made it *worse*, and adding a worked example
produced identical overall accuracy with the errors merely redistributed.

## Security: executable checks are gated

`verify_cmd` is shell, and facts are written by agents that read untrusted
input. Without a gate, one prompt-injected `memory_write` turns `nenapu check`
into scheduled remote code execution.

- Nothing runs until a human approves that exact command.
- Approval binds to a hash, so editing a blessed command revokes it.
- Omitting the ledger connection fails closed — refusal, not a shell.
- No path over MCP or HTTP can approve anything.
- Non-interactive shells refuse rather than defaulting to yes.
- A blocked check is absence of evidence, never failure, so it cannot demote a
  fact or trigger a cascade.

## Concurrency

MCP server, CLI and cron all write the same file. Reproduced before fixing:
eight writers produced eight identical active facts; eight graders each scored
one recall.

- `BEGIN IMMEDIATE` takes the write lock *before* the read a decision depends
  on. Python's sqlite3 starts transactions on the first *write*, which is too
  late.
- Grading folds check and write into one statement (`UPDATE ... WHERE
  outcome='pending'`, then test `rowcount`).
- A partial unique index on `(scope, text) WHERE status='active'` makes
  duplicates unrepresentable even outside a transaction.
- Bounded retry with jittered backoff for lock contention.

## Performance

At 3,000 facts. Three of the four causes were regressions introduced by the
concurrency fix, which was correct and unmeasured.

| operation | before | after | cause |
|---|---:|---:|---|
| write a fact | 180 ms | 2.2 ms | `synchronous=FULL` fsyncs every commit |
| recall | 5,745 ms | 87 ms | one fsync per result row; FTS reindex on every use-count bump |
| cascade (900 deps) | 345,850 ms | 102 ms | one durable write per affected node |
| dedupe | 103,478 ms | 3,230 ms | O(n²) all-pairs comparison |

Dedupe now uses a prefix filter over rarest tokens, run deliberately looser
than the verifier so containment matches cannot slip through, and verified
against the original implementation: the fast path archives exactly the same
set.

## Token cost

The core loop never calls a model. What is spent is MCP tool schemas, which sit
in context on every request whether memory is touched or not — hence a small
surface (operator jobs live in the CLI, never registered as tools), lean recall
results that omit predictable fields, and a test asserting the surface stays
under budget.

The extraction call is the other place tokens are spent, once per session, and
it now carries four outputs instead of two. Measured at 4 chars/token:

| part of the extraction prompt | tokens |
|---|---|
| schema, whole | 336 (grades 67, entities 80) |
| system prompt | 605 |
| injected block, at its cap of 17 recalls | 318 |
| known entities block, at its cap of 15 | 125 |

So grading and entity extraction together add roughly 590 tokens to a call
that already sends a whole session's conversation, and they buy the recall
ledger its evidence without a second model call. Both blocks are capped by
count rather than by length, which is the same unit R3 replaces in the
injection path.

## The pet: five versions, and the one lesson worth keeping

Four were hand-drawn and every one looked homemade: three braille bitmaps
plotted dot by dot (a filled silhouette, chibi proportions, then outlines with
the eyes as the only filled shapes), then line art set in ordinary characters.
The verdicts were "looks pirated", "still ugly" and "very very bad", and all
three were fair.

**The lesson is about resolution, not about geometry.** Art that reads as a
logo is not drawn at the resolution it will be displayed at. It is drawn large,
by someone who can draw, and then *reduced* — and the reduction is where the
quality comes from, because averaging and thresholding a big smooth curve
produces a clean edge, while plotting the same curve directly onto a 2x4 dot
grid produces a staircase of unrelated glyphs. Every hand-plotted version was
fighting that and losing.

So the shape now comes from the Noto Emoji dog face: rendered at 400px, given a
different expression per mood by wiping and redrawing the eyes and mouth, and
reduced to a braille grid at three widths by `tools/render_pet.py`. The output
is baked into `pet_art.py` as text, so Pillow and the font are build-time only.
Nothing is resized at runtime either — reducing a reduction is the same mistake
one step later.

What survived all five versions: the moods, their priority order, the colour
override when something is wrong, the blink, and the layout around it. Only the
drawing was ever thrown away, which is the argument for having kept the readout
and the picture separate from the start.

**Escaping the rows is not a precaution.** A row ending in a backslash escapes
the closing markup tag, so a literal `[/]` printed itself at the start of the
next line and shoved the whole drawing a column sideways. It was invisible
until a test looked at rendered text rather than at the drawing.

## The landing view is measured, not guessed

A bare `nenapu` printed seventy-three rows: the hero panel, then Typer's
grouped help. On a twenty-four row terminal that put the wordmark someone had
just run the command to look at fifty lines above the top of the screen. It was
reported as "the logo scrolls away", which is exactly what it was.

Two columns fixed it rather than trimming did. The block letters span the top,
the dog takes the left, and the store readout and command names take the right
— under twenty rows, and the full descriptions live in `--help` where they cost
nothing. Below 96 columns the layout stacks instead, because two thin columns
make Rich cut command names mid-word.

The parts are shed in a fixed order as the screen shrinks: the three-line pitch
first, then the block letters, then the command list. The wordmark is last
because it is the thing being protected.

**Then the fix over-corrected**, which is worth recording because it is the
same mistake in the other direction: twenty-three rows on a forty-row screen,
most of it empty, which reads as a program with nothing to say. So the view is
now built at a ladder of sizes and the *largest* that fits is printed. It grows
into the room it has — a bigger dog, more of what it learned lately — rather
than only shrinking out of trouble. Fill runs 72–96% across every terminal size
tested.

The drawing is capped by width as well as by height. Sized against rows alone
it grew until the column beside it could not hold a sentence, and Rich answered
by cutting every line of the readout off with an ellipsis: a bigger drawing
bought with the text that says what the thing actually knows.

A ladder of fixed sizes was still not enough — it lands on whichever rung is
closest and leaves the rest blank, which is what "there is space below it"
meant. The drawing cannot fill a tall screen by itself at any width, so the
frame is measured once and facts are added one at a time until the screen is
full, keeping the last version that fits. Grown and measured rather than
calculated: the drawing can be taller than the column beside it, in which case
the first few facts cost no height at all, and arithmetic that assumed
otherwise left a quarter of the screen empty.

**Which candidate to print is measured, not calculated.** `_first_that_fits`
renders each version and counts the lines. How tall any of them is depends on
how many commands are registered and where the terminal wraps them, and
arithmetic that is one row wrong scrolls the wordmark off the top — which is
the entire bug.

The tests print the view at a dozen terminal sizes and assert it fits every
one. They immediately found a second truncation: the three-line pitch is
hand-set at 76 characters, so under 82 columns Rich cut it mid-sentence, which
reads as a bug rather than as a summary. It is dropped below that width now.

## The command names are plain words

Nine were renamed: `write` → `remember`, `search` → `recall`, `verify` →
`check`, `loops` → `doubts`, `distill` → `tidy`, `observe` → `learn`, `good` →
`helped`, `bad` → `misled`, `outcome` → `grade`.

The old names were the vocabulary of the implementation rather than of the
person typing them. `loops` meant unresolved memory debt to whoever wrote the
graph code and meant nothing to anyone else; `distill` was the deduplication
pass; `observe` was reading a transcript. Someone reaching for this at a prompt
is trying to say *remember this*, *what did it learn*, *that one misled me*.

**Every old name survives as a hidden alias.** Renaming commands in a tool
people have already wired into hooks and shell scripts is a breaking change
unless the old word keeps working — and the aliases are hidden rather than
listed because an advertised alias puts every renamed command on the screen
twice, which is worse than the jargon it replaced.

Two names deliberately did not change. `recall-hook` is written into
`~/.claude/settings.json` on every machine that has run `nenapu init`, and
renaming it would break memory injection silently: the session would simply
start knowing nothing. `init`, `doctor` and `serve` are conventions people
already know from other tools, and inventing new words for them would be
originality in the wrong place.

## Things still open

- The repo is private, so the install URL in the README 404s for anyone else.
- Not published to PyPI; install is git-only. `uv.lock` is used by CI and by
  anyone working in a clone; `uv tool install git+...` still resolves fresh
  within the pyproject bounds, so the caps are what protect a plain install.
- Poisoned *content* is not covered by the approval gate. A session that read a
  hostile file can leave a misleading fact behind — not code, but something the
  agent will believe later. Origin is recorded, `--dry-run` shows what a
  transcript would produce, and that is the whole of the defence.
- No full-store backup/restore; export is a filtered Markdown block.
- The `anthropic` backend proper is unexercised — `exec` covers the same path.
- Vectors and an entity tier are designed but deliberately unbuilt: the plan
  gates both on the recall ledger showing that retrieval is what fails, and
  building them first would invent the design that evidence is supposed to
  choose. `nenapu retrieval` is that gate, executed rather than argued —
  today it answers `insufficient-evidence` on 0 graded recalls out of 327
  logged, so the open question is what grades a recall, not what indexes it.
- The watcher ships two adapters, Claude Code and Codex. Gemini, OpenCode and
  Cursor need a probing session against a machine that has them installed;
  `nenapu watch --probe` is the tool for it. Codex sessions reach the ledger
  and the extractor but contribute no file events yet.
- Ollama keeps generating after a client timeout; handled, not elegant.
- Python 3.10 through 3.13 are run in CI; nothing outside that range is.
