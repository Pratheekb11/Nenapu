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
| `Stop` | `nenapu observe --stdin --detach` | reads the finished transcript, extracts corrections and decisions, writes them. |

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
cascade would be visible in `nenapu loops` and invisible where decisions are
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
read `nenapu observe --stdin` with a 60-second timeout — the exact
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
input. Without a gate, one prompt-injected `memory_write` turns `nenapu verify`
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

## The pet is drawn, not typed

Braille cells carry 2x4 dots, so a 32-column block of them is an 84x58 bitmap —
enough resolution for something that reads as an animal rather than as
punctuation. The dog is therefore assembled from ellipses at runtime and a mood
is a handful of pixel edits: eyes shut, brows down, mouth open, tongue out.
Nine hand-typed drawings would have started drifting apart the first time an
eye needed to move one dot left.

That the animal changed species late is the argument for the approach. Going
from a bear to a dog was ears, a snout, a collar and a tail — a few dozen lines
of geometry — and every mood, blink and colour followed along. Hand-typed
braille would have been nine drawings to redo.

Three things the first drafts got wrong, each visible the moment it was
printed:

**A filled silhouette cannot be cute.** Three compositions were drawn before
this one: a filled body with features carved out of it, then the same thing
with chibi proportions, then a head on a small body. The first read as a stamp,
and all three had the same flaw — parts competing for a small canvas, every
feature shrinking until it turned to mush against the fill. The verdict on the
second was "looks pirated", which is exactly right for a knock-off plush.

What worked was inverting it: outlines and arcs, with the eyes as the only
large filled shapes. An empty face means the eyes are the only thing to look
at, and no body means nothing crowds the mouth. Cute is uncluttered before it
is anything else.

**A catchlight goes beside the eye, not inside it.** Braille has no grey to
soften a hole with, so a 3px bite out of a 12px eye is a chunk missing — the
eye reads as cracked rather than as shiny.

**Ears carry the species.** Filled and hanging, the same head is a puppy;
outlined and perched, it is a balloon with two rings on it. They are the only
heavy shapes besides the eyes for exactly that reason.

**`⠀` is U+2800, not a space.** The canvas is sized for the shapes rather than
the result, so the dog arrived inside a wide frame of blank braille that
`strip()` will not touch. It silently ate terminal width and pushed the status
column off the screen. The drawing is cropped to its ink.

An unwell dog is not drawn in the user's theme colour. The point of the
creature is that a bad store cannot look like a good one, and a calm teal dog
with its eyes crossed still reads as fine at a glance.

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
bought with the text that says what the thing actually knows. Where the dog is
already at its width cap the spare rows go into more facts instead, which is
the more useful thing to be looking at anyway.

**Which candidate to print is measured, not calculated.** `_first_that_fits`
renders each version and counts the lines. How tall any of them is depends on
how many commands are registered and where the terminal wraps them, and
arithmetic that is one row wrong scrolls the wordmark off the top — which is
the entire bug.

The tests print the view at a dozen terminal sizes and assert it fits every
one. They immediately found a second truncation: the three-line pitch is
hand-set at 76 characters, so under 82 columns Rich cut it mid-sentence, which
reads as a bug rather than as a summary. It is dropped below that width now.

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
- Ollama keeps generating after a client timeout; handled, not elegant.
- Python 3.10 through 3.13 are run in CI; nothing outside that range is.
