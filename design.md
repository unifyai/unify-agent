# cleanslate-v1: an agent harness that remembers what actually worked

Status: phase 0 approved by MAIN overnight; paid phases need a frozen prereg + MAIN GO. Updated 7 October 2026
with two rounds of offline engineering.

Phase 0:

- bubblewrap confinement of every workspace;
- file deliverables;
- the cost record;
- the office adapter and its dry run.

Phase 0.5:

- cgroup resource limits;
- the real-model client, with per-instance caps and the 40 USD/h runaway guard;
- the lab-ledger journal;
- the worker run kit;
- the analysis script;
- the prereg draft.

Phase 1b, offline, 7 October:

- AppWorld API record and replay, with secrets kept out of memory;
- the AppWorld and ARC systems for the benchmark runners;
- the ARC binding probe;
- the cost-curve note (`cost-curve.md`);
- the AppWorld prereg draft.

**Correctness fix (7 Oct).** The binder and the retriever had ignored words such as "at", "or", "with" and "for".
So "at or before 2021" and "before 2021" counted as the same request. Now only filler words are ignored: articles,
pronouns, copulas, question words and request verbs. Every boundary, comparison, quantifier, negation and
conjunction word counts. `tests/test_binding.py` covers the sibling pairs and value-only returns. MAIN will freeze
a v1.1 prereg from the snapshot `frozen/office-v1.1/`, which is office-v1 plus this fix only
(`frozen/README-office-v1.1.txt`).

**Offers off (7 Oct).** A run-time switch, `Agent(offers=False)` or `kit/run_cell.sh --no-offers`, shows stored
procedures only as reference code. Nothing is bound to the new request, pre-run or placed in the workspace. MAIN
chose this for the office test, because the binder's word alignment falls under the lead's 5 Oct ruling. The
snapshot `frozen/office-v1.2/` is office-v1.1 plus only this switch (`frozen/README-office-v1.2.txt`). The tests
are in `tests/test_no_offers.py`.

**The office prereg is frozen. Its exact source is the snapshot in `frozen/office-v1/`**, verified file by file
against the prereg's hashes (`frozen/README-office-v1.txt`). The working tree has moved on.

Nothing here has been run against a paid model, a worker or a benchmark run. The prototype in `cleanslate/` runs
with a scripted fake model, against a local stand-in for OpenRouter on 127.0.0.1, or against a fake API world.
All 76 offline tests in `tests/` pass. This design was written from a clean slate. It does not copy the existing Unify harness. The
only things taken from Unify are its measured problems, which are listed in section 1.

## 0. The idea in one paragraph

The model has one tool: a Python workspace whose variables persist between turns. It answers by calling
`deliver(value)` from code. Everything else is bookkeeping, and the harness does it, not the model:

- It works out which code produced the delivered answer. The answer is traced backwards through the cells that
  ran.
- It checks the answer for common smells, such as a zero, an empty result, or a filter on a value the data does
  not contain. A smelly answer is held once and the model is shown evidence.
- It saves the producing code as a procedure. The values that came from the request become parameters. The
  procedure is saved only if it gives the same answer again when replayed in a fresh workspace.
- When a similar request arrives later, it re-checks the procedure first and then runs it. The result is put in
  the workspace ready to use, before the model's first turn.
- When the history gets long, it folds old turns into one line each. Nothing is lost, because computed state lives
  in the workspace, not in the transcript.

The model is never asked to remember a convention, use an optional tool, write a skill or summarise its own work.

## 1. The problems this has to solve

These come from measurements of the existing system and from the model's known habits.

| # | Problem | What it looks like |
|---|---|---|
| P1 | Repeat visits are not cheaper | The fifth time the same job comes in, it costs about as much as the first. |
| P2 | Saved procedures drift from what was delivered | The stored "skill" is a rewrite of the work, not the code that actually produced the answer. |
| P3 | Retrieval trades precision for recall | Look-alike requests get the wrong procedure, and the model uses it. |
| P4 | Long sessions hit a step cap and lose context | The model forgets what it found 30 turns ago. |
| P5 | The model accepts empty or zero results from reused code | It delivers `0.00` after filtering on `"meal"` when the data says `"meals"`. |
| P6 | 52% of code cells do nothing in some builds | Examples: a function defined but never called, a loop over nothing, a bare `pass`. |
| P7 | The model ignores new optional tools and conventions | A `save_skill` or `remember` tool goes unused. |
| P8 | The model rarely keeps state between cells, and often answers without computing | It retypes numbers it saw earlier, or answers in prose. |

## 2. The core loop

### 2.1 What the model sees each turn

The prompt is built in the same order every time. Earlier parts change least, which keeps the provider's prompt
cache warm.

1. **System text** (fixed, eight lines). It says: you are in a Python workspace; reply with one code block;
   variables persist; the harness shows what each cell changed; deliver the answer with `deliver(value)` from
   code.
2. **The request block** (fixed for the whole task):
   - the request, word for word;
   - a list of the files in the working directory;
   - any **offers**, which are stored procedures that the harness has already re-checked and run on this request
     (section 3.4);
   - any **references**, which are similar past procedures shown as code only.
3. **The state card**, but only after compaction has happened (section 4). It holds pinned findings, the variables
   that currently exist, and one line for each folded turn.
4. **The recent turns**, word for word. Each turn is the model's code plus the harness's observation of what
   happened.

The observation of a cell is written by the harness, not by the model's print statements. It contains:

- the printed output (long output keeps its head and tail, with the cut size stated);
- the value of the cell's last expression, as in a notebook;
- every variable the cell created or changed, with its type, length and a short preview;
- the files it wrote;
- the error, trimmed to the last frames;
- what it delivered.

A cell that changed nothing visible is named as such, with a one-line hint (P6).

### 2.2 Tools

There is one tool, the workspace. There are no other tools to learn or ignore (P7). File, data, shell and API
work all happen in Python, using `subprocess` and HTTP clients inside the workspace. When the harness has
something to give the model, it does not offer a tool. It puts a value into the workspace (`offer_1`) or a line
into the prompt.

Why there is only one tool: every extra tool is an optional convention, and this model ignores optional
conventions. A persistent workspace also fixes P8 without asking the model to change its behaviour. The variables
are simply still there, and the harness lists them.

### 2.3 How an answer is delivered

The model calls `deliver(value)` inside a code cell. This is the only way to finish. If the model replies in
prose with no code, the harness treats the prose as a typed-in answer. Typed-in answers go through the same smell
gate. When the task has input files and nothing was computed, the gate holds the answer once with "the answer was
typed in rather than computed from the files" (P8).

For work whose result is a file (a report, a cleaned CSV), the model calls `deliver(path)`, or passes a list of
paths. The path may be a file or a folder inside the working folder.

- The worker records each delivered file's sha256, size and the first 4,000 characters.
- The delivered value becomes `{"__files__": {path: sha256}}`, so replay compares file contents exactly.
- A path outside the working folder, or one that does not exist, is treated as an ordinary string value.

File smells are checked:

- an empty file;
- a CSV or TSV with a header and no rows;
- an empty JSONL file, or a JSON file holding `[]`, `{}` or `null`;
- a file that holds a single value which smells, for example `0.00` in `answer.txt`.

For file deliveries, the slice is seeded with every cell that changed files:

- through `open()`, unless a later cell rewrote the same file from scratch;
- through calls that bypass `open()`: `rename`, `move`, `unlink`, `write_text`, `subprocess.run`, and so on.

Replay of a file procedure runs on the whole start-of-task text snapshot. This is because moves and deletes do not
show up as reads.

### 2.4 Provenance: what produced the answer

The harness keeps a log of every cell that ran successfully. When something is delivered, it computes a
*backward slice*. Starting from the names and files that the `deliver` cell reads, it walks back through the log.
It keeps only the cells that defined or changed those names, or wrote those files. This repeats until nothing more
is needed. A cell that fully
overwrites a name ends the search for that name. That is how a wrong first attempt drops out once the model fixes
it.

Inspection cells, no-op cells and abandoned attempts are not part of the answer, so they never enter memory.
This is the fix for P2. A stored procedure is, by construction, the code that produced the answer, and it is
checked by replay before it is stored (section 3.2).

## 3. Memory

### 3.1 What is stored

| Form | Content | Written by | Status |
|---|---|---|---|
| **Procedure** | The sliced code, with request values lifted into parameters `p1, p2, …`. Each one has an id, a lineage, a version, the version it supersedes, a status and counters. | The harness, automatically, after every accepted delivery | Prototype |
| **Case** (belongs to a procedure) | The request text, the parameter values, the input files the code read (a snapshot), the delivered answer, the run id and a timestamp | The harness | Prototype |
| **Negative request** (belongs to a procedure) | A request for which this procedure was shown, and the work delivered had a different *shape* (section 3.2) | The harness | Prototype |
| **Data note** | A plain-language fact about a *resource*, keyed by the resource's identity (for example the file path plus a hash of its header), not by the task. Example: "expenses.csv: categories are plural, e.g. `meals`". The note is created when a smell hold led to a changed answer that then passed the gate. Its text is the harness's own finding. | The harness | Design only |
| **Guidance** | Plain-language advice that the user states ("always round to cents"), kept word for word and attached to the resources it mentions | The user, through the request | Design only |

There are no model-written "lessons" or skills. Those cannot be checked, and they drift (section 8).

### 3.2 When and how a procedure is stored

This happens after every accepted delivery, with no model call. There are six steps.

1. Compute the slice and drop display-only lines (a bare `len(rows)` or `total`).
2. **Lift parameters.** Every string or number in the code that also appears as a whole word in the request
   becomes a named parameter. For the request "total spend on meals in 2026-03 according to expenses.csv", the
   strings `"meals"`, `"2026-03"` and `"expenses.csv"` become `p2`, `p3` and `p1`. Strings that are not in the
   request, such as the column name `"amount"`, stay as constants. Parameters therefore come from the request's
   content, never from a task id.
3. **Replay check.** Run the lifted code with the original parameters in a fresh, empty workspace that holds only
   a copy of the files the slice read. Store the procedure only if this reproduces the delivered answer exactly.
   Floats are compared to 9 decimal places. Code that is random, depends on lost state, or depends on files the
   session created is refused, and the reason is logged.
4. If the lifted code is identical to an existing procedure (same syntax tree), add a **case** to it. A procedure
   becomes *trusted* when it has cases from two different requests.
5. **Store a new version in the same lineage** (`L1v2` supersedes `L1v1`) in two cases:
   - The model was offered, or tried with, a procedure for this same request shape, and solved it with different
     code. A typical cause is that the data changed, for example a renamed column.
   - The new code has the same **shape** as an existing procedure. Shape means the syntax tree with every constant
     and parameter blanked. Same shape with different values is what a paraphrased return of the same job
     produces. For example, office-v1's first expense visit says "travel" and its return says "meal spend", so
     different words get lifted into parameters.
6. A procedure that was shown (as an offer or a reference) gets this request recorded as a **negative**, but only
   when the delivered work has a *different shape*. In the first version of this rule, any procedure that was shown
   and not used got the negative. The office dry run showed the problem: that rule marked a paraphrased return of
   the same job as "not applicable".

The prototype keeps the store in one JSON file, with snapshots stored inline. A real store would be an append-only
event log plus content-addressed file snapshots. Nothing is ever deleted, so every version, case and failure keeps
its identity.

### 3.3 How a procedure is found

1. **Candidates.** Score the new request against each stored procedure's case requests by word overlap (Jaccard,
   ignoring stopwords). Keep at most 2 that score above 0.3. Skip a procedure if the new request is at least as
   close to one of its **negative** requests as to its positive cases. This is how look-alikes stop being
   retrieved after one mistake. Ties go to the newer version.
2. Retrieval only proposes. Everything after this step decides.

Lexical retrieval is enough for v1 because the gate after it is strict. Embeddings can replace step 1 later. That
change is a measured experiment, not an assumption.

### 3.4 How a procedure is trusted: verify before every reuse

For each candidate, in order:

1. **Replay its recorded cases.** At most the 2 newest are replayed, in fresh workspaces. If any case no longer
   reproduces its answer, the procedure is marked *stale* and is never offered. Causes include a changed library,
   a changed interpreter, or a store edited by hand. This step costs CPU only.
2. **Bind parameters to the new request.** Align the old and new request word by word. Each parameter's value in
   the old request is mapped to the word in the same position in the new one, so "meals in 2026-03" becomes
   "travel in 2026-04". If any parameter cannot be placed, there is no ready answer.
3. **Shape check (precision by construction).** If the two requests differ anywhere *outside* the parameter
   positions, there is no ready answer. Stopwords are ignored. For example, "total spend" → "number of rows" is a
   difference outside the parameters.
4. **Run it on today's files** in a fresh workspace, and apply the smell tests from section 5 to the result.
5. **The outcome decides what the model sees.**
   - If all checks pass, the value goes into the workspace as `offer_1`. The prompt shows the value, the
     procedure's status and number of cases, the parameter binding, the last recorded case and the code.
     Delivering it is one line: `deliver(offer_1)`.
   - If the procedure produces **files**, the harness cannot hand over a value. Instead, the prompt shows the
     verified cell with the parameters filled in. It also shows what the cell produced on a copy of today's files
     (name, size and first characters). Running that cell is the one-turn path. The harness then records it
     through the normal capture route as another case of the same procedure.
   - If any check fails, the procedure is shown as **code for reference only**, with the reason, for example
     "differs beyond the parameters: total spend → number", or "KeyError: 'amount'" on today's data. The model
     can reuse the code, but no ready-made answer invites blind acceptance.

This splits P3 into two tiers that are measured separately. *Ready answers* are offered only when the request is
the same shape and every check passed. Here precision should be very high, and recall is limited to rewordings of
values. *References* cover paraphrases and look-alikes. They cost tokens but cannot be accepted blindly. Pre-running
on paraphrases (for example with a cheap yes/no judge) is deliberately left out until an offline precision
measurement supports it.

### 3.5 Versioning and lifecycle

| Status | Meaning |
|---|---|
| candidate | 1 case. It can be offered, and the offer is labelled with its single case. |
| trusted | 2 or more cases from different requests, all replay-verified |
| stale | One of its recorded cases no longer replays. It is never offered again, and it is kept for the record. |
| superseded (design) | A newer version in the same lineage exists. Both stay. At reuse time, the newest version that passes on today's files wins. |
| retired (design) | Never offered again. This happens after 3 negatives with no positive case since, or on a user correction. |

A user correction ("that was wrong") is recorded as a failed case against the procedure that produced the answer,
and that procedure is not offered again until a new verified case exists. That part is design only.

## 4. Long tasks

**Where state lives.** Computed state lives in the workspace's variables, not in the transcript. That makes the
transcript disposable, and that one decision does most of the work for P4.

**Compaction is deterministic and needs no model call.** When the rendered prompt exceeds the budget (24,000
characters by default), the harness folds the oldest half of the unfolded turns. Folding in large chunks means the
prompt prefix changes rarely, so the prompt cache is invalidated rarely. Each folded turn becomes one line: the
first line of the code and the start of the observation. The **state card** replaces the folded turns. It is held
to half the budget and contains:

- **pinned findings**: smell findings, the result of any stored procedure withheld because it smelled, workspace
  restarts, and the model's latest `Plan:` line if it wrote one;
- **live variables**, listed from the workspace: name, type, length and a preview. When there are too many, the
  rest are listed by name only.
- **one-line digests of folded cells**, newest first, as many as fit.

The request block always stays word for word.

**Plans.** No plan tool exists. If the model writes a line starting `Plan:`, the harness pins the latest one and
keeps it through compaction. A design option is to ask for a plan once, at the first compaction. It is not in the
prototype.

**No step cap. A budget and a stall detector instead.**

- The task ends when an answer is accepted, or the money or token budget runs out.
- After three cells in a row that fail or have no effect, the harness adds a note: look at the data or the error
  before trying again.
- A design option is to stop after a longer stall.

The prototype also has a safety ceiling on the number of steps (`max_steps=40`). It is a backstop, not a planning
device.

**Checkpoints (design).** The successful-cell log plus the start-of-task file snapshot is a checkpoint. If the
workspace process dies (a timeout or a memory limit), the harness can rebuild the variables by replaying the slices
that define them. Cells that wrote files or called APIs are skipped, and replay uses their recorded results
instead. The prototype only reports the restart and pins it. It does not rebuild yet.

## 5. Self-checks before an answer is accepted

The **smell gate** runs on every delivery. It looks at the value and at the code that produced it:

- **Empty or degenerate value**: `None`, `NaN`, infinity, zero, `"0.00"`, an empty string or collection, or a
  collection whose numbers are all zero.
- **Ungrounded literal (the 'meal' vs 'meals' check)**:
  - The code compares a raw field to a string with `==` or `!=`, and that string is not a whole value in any file
    the slice read. The harness lists the closest real values (for example "the data has 'meals'").
  - Substring and prefix comparisons (`in`, `startswith`, `x[:7] ==`) only need the literal to appear somewhere in
    the files, so date prefixes do not trigger it.
- **Typed-in answer**: input files exist, but the answer was a literal and no data was read.

**The gate holds a delivery once.** It holds a given (value, smells) pair one time, shows the evidence (the smells
and a preview of each file read), pins the findings, and asks the model to inspect the data. If the model delivers
the same value again, the answer is accepted with a pinned note. Correct zeros exist, and an endless gate would only
waste money.

The same smell tests run on **offers** before the model sees them (section 3.4). This targets P5: reused code that
returns a zero on new data is never presented as a ready answer.

**How it is measured** (from traces, with labels from the checker added offline):

- hold rate;
- how often a hold leads to a changed answer, and how often that changed answer is correct;
- **false-hold rate**: correct answers that were held, which costs one turn each;
- smelly answers that were accepted and wrong, before and after the gate.

## 6. Multi-agent coordination

There is none in v1, deliberately. The failures listed in section 1 are bookkeeping failures, not failures of
breadth of reasoning. Extra agents multiply cost and add hand-off points where context is lost (P4).

Parallel work, when it is needed, happens inside the workspace: threads or subprocesses run by the model's code.
If sub-agents are added later, each gets its own workspace, and they share the same rules for delivery, provenance
and the gate. The parent receives only *delivered* values with their provenance, never summaries.

## 7. Cost control

| Lever | Mechanism |
|---|---|
| Repeat visits cost one turn | A verified offer makes the cheapest action `deliver(offer_1)`. The target is a return visit that costs about one model call. |
| No bookkeeping calls | Slicing, packaging, replay checks, binding, smell tests and compaction are all CPU. The harness makes no summariser, reflection or skill-writing calls. |
| Cache-friendly prompt | Fixed system text, a fixed request block, and folding in chunks, so the prefix changes at most a few times per task |
| Short observations | The harness shows changes and the value of the last expression. Long output keeps its head and tail with the cut size stated. Previews are capped. |
| Fewer wasted cells | No-op cells are flagged at once, and three bad cells in a row trigger a nudge |
| Bounded retrieval | At most 2 candidates, at most 2 replayed cases each, and per-cell timeouts |
| Cost record | `cleanslate/cost.py` keeps one line per model request, with the run, attempt, solve and request ids and the provider's generation id. USD is a decimal string, and it is the provider-reported charge (OpenRouter `usage.cost`). A price-table figure is kept separately as `usd_estimate` and is never written as a charge. A request with no reported charge, or a failed one, is `usd: null` and is counted as unpriced. The total is `null` while any call is unpriced. The same requests are also written in the lab journal format (`runs/<run>/attempts/<attempt>/costs.jsonl`), which `cost_ledger.py` counts. Scripted-fake calls are zero-charge local replays. |
| Caps and guard | These are checked before every model call. The per-instance caps are USD 0.50, 200 calls and 900 s, the same as the matched Unify office cells; a cap ends the task, which is still scored. For the money cap, a call with no reported charge counts at its estimate. The runaway guard stops the whole cell above 40 USD/h, measured over the last 10 minutes of this process's cost record. |

## 8. Each component: the failure it prevents and how we would measure it

| Component | Prevents | Measure (per run, from traces) |
|---|---|---|
| Persistent workspace + change listing | P8, P6 | Share of cells with no effect; share of answers that were retyped (a literal in `deliver` while inputs exist) |
| `deliver()` from code only | P8 | Share of deliveries with a non-empty slice that read the inputs |
| Backward slice + replay before storing | P2 | Share of stored procedures that reproduce their answer (should be 100%); share of deliveries that could not be stored, and why |
| Parameters lifted from the request | P1 | Share of return visits where binding succeeds; **fit rate**: of return visits whose job has a stored procedure, how often the lifted procedure bound to the next visit gives the right answer (offline, scored by the checker; see section 12) |
| Verify before reuse (replay of cases) | P5, P2 | Stale detections; offers whose recorded cases fail (should be 0) |
| Shape check + negative requests | P3 | Offer precision (accepted and correct ÷ offered) on look-alike and same-job returns; how many returns fall back to references |
| Smell tests on offers | P5 | Smelly offers withheld; accepted offers that were wrong |
| Smell gate on deliveries | P5 | Holds, changed-after-hold, correct-after-hold, false holds |
| Deterministic compaction + state card | P4 | Completion of long tasks; answers to probe questions about early findings after compaction; prompt size distribution; cache-hit rate |
| Stall detector | P4, cost | Longest run of failing or no-effect cells; cost of tasks that never finish |
| Offers placed in the workspace | P1, P7 | Cost and number of calls on return visits ÷ first visit, per recurring job |

## 9. What we deliberately do NOT have

- **No memory tools for the model**: no "save skill", "search memory" or "remember". The model ignores optional
  tools, and the harness has better information about what worked than the model does.
- **No model-written skills or lessons.** They cannot be checked, they drift from what was delivered, and every
  one costs a call. Plain-language notes exist only as harness findings (data notes) or the user's own words.
- **No model-written summaries for compaction.** State lives in the workspace, so folding needs no call and loses
  no computed fact.
- **No step cap as a planning device**, only a budget, a stall detector and a backstop.
- **No ready answers from fuzzy matches.** A paraphrase or look-alike gets code to read, never a value to accept.
- **No task ids, benchmark names or benchmark-specific rules anywhere** in the harness. Retrieval, binding and
  parameters key only on the request's content and the files.
- **No LLM judge of correctness.** The checks are mechanical: replay, binding, smells.
- **No planner/executor split and no multi-agent system** (section 6).
- **No enforced reply format** beyond "one code block". Prose is accepted and treated as a typed-in answer.
- **No vector database or embeddings in v1.** Lexical retrieval plus a strict gate is the baseline any learned
  retriever must beat.
- **No silent fallbacks.** A procedure that failed today is shown with its failure. It is not hidden, and it is not
  quietly retried.

## 10. Where I disagree with, or sharpen, the starting principles

1. **"Make the right behaviour the path of least resistance" has a dangerous side.** A ready-made `offer_1` also
   makes *blind acceptance* the easiest path, and P5 says this model accepts what it is given. So ready answers have
   to be precise by construction: same request shape, replayed cases, a run on today's data, and smell tests.
   Everything else is demoted to a reference. Recall is traded for precision on purpose, and both are measured.
2. **"Plain-language guidance" should not come from the model.** Guidance the model writes about itself after a
   task is the main way drift enters the system. This design keeps plain language only where it is grounded: the
   harness's own data findings and the user's words.
3. **"The harness does the bookkeeping" goes further than capture.** The harness also decides the *parameters* (by
   matching literals to the request) and the *lineage* (by noting which offer was rejected). The model's only job
   is the work itself.
4. **Executable procedures fit data and file work best.** They fit open-ended reasoning tasks such as ARC much less
   well: there, the request contains grids rather than lifted literals, and binding will usually fail. On those
   tasks, expect this design to give references rather than one-turn reuse. The experiment plan says so up front,
   so a weak ARC result is not a surprise.

## 11. The prototype

| Path | What it is |
|---|---|
| `cleanslate/sandbox.py` | A persistent workspace in a **bubblewrap** child process. It reports changes, the last expression, files read, written and truncated, and deliveries (values, or files with their hashes). It has a per-cell timeout that kills the process group, then restarts and reports the restart. Termination is verified. |
| `cleanslate/analysis.py` | Def-use analysis with scopes; the backward slice, which knows about files and about cells that overwrite a name or file; parameter lifting; shape; value and file smells; the ungrounded-literal check; and file previews |
| `cleanslate/memory.py` | The procedure store, confined replay, binding with the shape check, retrieval with negatives, verify-before-reuse, and versions decided by shape |
| `cleanslate/limits.py` | Resource-limit modes on top of bwrap: `scope`, `prlimit-user` and `rlimit-only`, the worker and local profiles, reading the limits back from the cgroup, verified scope removal, and the task-folder size check |
| `cleanslate/cost.py` | The cost record (section 7), the lab-journal writer and the runaway guard |
| `cleanslate/llm.py` | `ChatClient`: OpenRouter chat completions using only the standard library. Model `openai/gpt-6-luna`, `reasoning: {effort: low}`, usage accounting on. The key is read from the environment at each request. |
| `cleanslate/agent.py` | The loop, per-instance caps, the smell gate (with a hold hook), value offers, file-cell offers, references, the learning step, compaction, the state card and the transcript. It also has `ScriptedModel` (offline). |
| `kit/run_cell.sh`, `kit/run_cell.py`, `kit/README.md` | The worker run kit: venv from the system interpreter; one cell (arm, run index, instances, frozen order) in the lab's layout; preflight; verified cleanup; refusals; exit codes. OPS runs it. |
| `analysis/office_analysis.py` | The preregistered analysis: metrics per run and arm, bootstrap intervals, false holds, fit rate, A0/A1 from EVAL's readout, and evaluation of the criteria |
| `prereg/PREREG-office-v1.md` | The office prereg, frozen by MAIN. Its source is `frozen/office-v1/`. |
| `cleanslate/world.py` | API worlds: the recording gateway (live, read-only and replay modes), secret labels, credential recipes, and the workspace's API client (section 13) |
| `adapters/cleanslate_systems.py` | `CleanslateAppWorld` and `CleanslateARC`, the systems for the benchmark runners (section 13) |
| `analysis/binding_probe.py` | Whether lifting can ever give a ready offer on a stream's logged requests (section 13) |
| `prereg/PREREG-appworld-train-canary-DRAFT.md` | The draft AppWorld prereg |
| `adapters/office.py` | The office-v1 adapter and the `dryrun` command. It is benchmark code, and it is kept outside the harness package. |
| `adapters/office_fake.py` | The scripted fake used by the office dry run. It sees only the messages. |
| `tests/` | 76 offline tests. Run them with `nice /usr/bin/python3 -m unittest -q` from `tests/`. They take about 30 seconds. They use the system interpreter, so nothing from another project's virtual environment is involved. One test calls `cost_ledger.py`'s reader under Python 3.12, which that file needs. |

What the tests show:

- **Capture** (`test_capture.py`): the no-op and inspection cells are left out of the procedure. The request values
  become parameters. A random answer that cannot be reproduced is refused.
- **Verify before reuse** (`test_reuse.py`):
  - a return visit with new values is solved in one turn by `deliver(offer_1)`, and the procedure becomes trusted;
  - a procedure that has drifted fails its replay and is marked stale;
  - a look-alike gets only a reference, is stored as a new lineage, and is then excluded by the negative;
  - a smelly offer (a zero for an unseen category) is withheld;
  - a renamed column produces version 2 (`L1v2`), which is offered on the next visit to the new data.
- **Smell gate** (`test_smell.py`):
  - the 'meal' vs 'meals' zero is held, with the evidence "the data has 'meals'", and the stored procedure contains
    only the fixed cell;
  - a model that insists gets its zero accepted after one hold;
  - a prose answer given without computing is held.
- **Compaction** (`test_compaction.py`): a 28-turn session under a 4,000-character budget keeps the request word for
  word, the variable holding the early finding (`account_id = 'AC-7731'`), and the pinned plan line.
- **Sandbox** (`test_sandbox.py`): state persists, the environment is not inherited, the timeout kills and
  restarts, and deliveries are reported.
- **Confinement** (`test_confinement.py`). Each check runs code *inside* the workspace:
  - (a) no parent environment variable is visible, through `os.environ` or through any `/proc/*/environ`. Host
    processes are invisible and cannot be signalled.
  - (b) `$HOME`, the repository and its `AGENTS.md`, and the benchmark data folder do not exist inside. HOME is an
    empty tmpfs, and `/` holds only `usr bin lib lib64 proc dev tmp work`.
  - (c) a TCP connect fails, DNS fails, and the only network interface is `lo`.
  - (d) writes to `/`, `/usr`, `/etc`, `/bin`, `/lib`, `$HOME` and the repository fail. A write to the private
    `/tmp` never reaches the host. The task directory is the one writable host path.
  - (e) a checker folder (with `expected.json` and `check.py`) that is named as hidden is not visible. The sandbox
    also refuses a task directory that would contain, or sit inside, a hidden path or `$HOME`.
  - Termination: after a timeout, every process of the old pid namespace has gone, including a background child the
    code started. A new namespace replaces it.
- **Files** (`test_files.py`):
  - `deliver('answer.txt')` records the hash;
  - a folder delivery lists its files, and a path outside the working folder is just a string;
  - the four kinds of file smell are detected;
  - a '0.00' answer file built from the 'meal' filter is held with "the data has 'meals'", and then fixed;
  - a CSV with only a header is held;
  - a file-producing procedure is replayed, offered as a verified cell, and run in one turn on the next visit.
- **Cost, caps and guard** (`test_cost.py`, 10 tests):
  - The provider-reported charge is the charge. A price-table figure is kept as an estimate only.
  - Unknown is never zero.
  - The run, attempt, solve and request ids are kept.
  - **The lab parser accepts the journal:** `journal_accounting.read_journal`, unchanged, reads it with complete
    coverage. It counts charged, failed, unpriced and fake (local replay) requests correctly.
  - Per-instance caps stop the task at the right point: calls; USD, with unpriced calls counted by their
    estimate; and wall time.
  - The 40 USD/h guard stops the cell before the 8th call of 1 USD each in 10 minutes.
- **Real-model client** (`test_llm.py`, against a local stand-in for OpenRouter):
  - the request shape: model, `reasoning.effort=low`, usage accounting and the bearer key from the environment;
  - with no key, nothing is sent;
  - error messages have the key redacted;
  - **the workspace cannot find the key** in its environment or in any `/proc/*/environ` or `cmdline` it can see;
  - no host process has the key in its arguments;
  - no file the run writes contains it;
  - the charge reaches the cost record and the lab journal.
- **Resource limits** (`test_limits.py`). The limits are proved by exceeding them from inside:
  - In scope mode, `memory.max`, `memory.swap.max` and `pids.max` are read back from the workspace's own cgroup,
    and the scope's cgroup is gone after `close()`.
  - Allocating 400 MB under a 128 MB limit kills the workspace, and a fresh one replaces it.
  - Filling the private `/tmp` past the memory limit kills it too.
  - The 9th process fails with `BlockingIOError` under `TasksMax=8`.
  - Requiring the cpu control where it is not delegated (this laptop) is refused.
  - `RLIMIT_CPU=2` kills a busy loop.
  - A task folder over its size limit ends the task.
  - The worker profile refuses rlimit-only, and refuses prlimit mode unless running as the named dedicated user.
- **Worker kit** (`test_kit.py`):
  - a fake cell through `run_cell.sh`: it creates the venv, writes the lab layout, and leaves no preflight folder.
    The journal is read by `read_journal`, and `cost_ledger.runs_under` (the lab ledger's own reader) lists the
    cell with USD 0 and 2 instances.
  - a **paid-path rehearsal** against a local stand-in, using a loopback `--base-url` and the `rehearse-` prefix:
    the no-memory arm, real `ChatClient`, priced journal fully covered, transcripts kept, and no key in any file;
  - paid refusals: no key, no prereg, wrong hash, not frozen, or not the worker profile.
- **Analysis** (`test_analysis.py`):
  - the report on a fake cell, with the held state scored, the fit rate computed and the criteria evaluated;
  - EVAL's A0/A1 readout loads with the solved counts and USD that EVAL reported.
- **Office dry run** (`test_office.py`). Two real office-v1 instances are used: the first expense-total visit and its
  paraphrased return.
  - The fake model receives the stream prompt byte for byte, and never a task id, job name or data path.
  - Both instances are scored by `python -m everyday.score` in a separate process. Both pass.
  - The return says "meal" where the data says "meals". The smell gate held the `0.00`, and the fake fixed it.
  - Every task directory is removed, and no workspace process survives.
  - An `--out` inside the repository is refused.

**Confinement now in place.** Every workspace, including replay workspaces, runs under `bwrap` with these
settings:

- `--unshare-all`, `--die-with-parent`, `--new-session` and `--clearenv`;
- `/usr` read-only, with the interpreter at `/usr/bin/python3`;
- a fresh `/proc` and `/dev`;
- `/tmp` as a private tmpfs, with HOME at `/tmp/home`;
- the task directory bound read-write at `/work`;
- `/` remounted read-only;
- the environment set to exactly PATH, HOME, LANG and PYTHONDONTWRITEBYTECODE.

There is no unconfined fallback. If `bwrap` is missing, the sandbox refuses to start.

**Resource limits** (`cleanslate/limits.py`). These apply on top of bwrap in every mode:

- **Per-process rlimits:** address space (2 GB), single-file size (64 MB), open files (256), core files (0), and
  total CPU seconds of the workspace process (900).
- **A wall-clock timeout per cell.**
- **A task-folder size check after every cell** (512 MB). It ends the task.

The modes:

| Mode | How | Bounds |
|---|---|---|
| `scope` (preferred) | `systemd-run --user --scope` with `MemoryMax`, `MemorySwapMax=0`, `TasksMax` and `CPUQuota`. After start, the harness **reads the values back** from the workspace's own cgroup and refuses to continue if a required control is missing. After `close()`, it verifies that the scope's cgroup is gone. | Total memory, including the private `/tmp`; processes and threads; CPU share |
| `prlimit-user` | Per-uid `RLIMIT_NPROC`, plus the rlimits above. Allowed only when running as the named dedicated unprivileged user, because the limit counts every process of that uid. | Process count; CPU seconds |
| `rlimit-only` | The rlimits above only. Allowed on a laptop, never in the worker profile. | — |

Profiles:

- **Worker profile:** `scope` with memory, pids and **cpu** all verified, or `prlimit-user`. Anything else is
  refused.
- **Local profile:** `scope` with memory and pids, because the laptop's user manager does not delegate cpu.
  Otherwise `rlimit-only`.

**Verified only on the worker:**

- that the worker's user systemd can create scopes, and delegates `cpu` as well as `memory` and `pids` (root
  configuration; the preflight refuses otherwise);
- the `prlimit-user` mode, which needs a dedicated user created as root;
- a hard disk bound: the kit's `--max-fs-mb` refuses unless the run folder sits on a size-limited filesystem
  (loop mount, set up as root);
- bwrap and unprivileged user namespaces on the worker's kernel;
- the worker's `/usr/bin/python3` version (3.10 or later);
- that OpenRouter accepts the request as sent (`reasoning.effort`, `usage.include`) and reports `usage.cost` for
  Luna. The first paid canary instance checks this.

The read-only `/usr` exposes system packages and binaries. That is what the code is meant to use, but it is more
than the interpreter alone.

The parent harness process reads only the task directory, and only as regular files: it skips links, and it never
follows a path that leads outside the replay folder.

**Known gaps in the prototype** (all of them in the design):

- recording and replaying API calls (deferred by MAIN);
- the trace smell audit (deferred by MAIN);
- binary files in snapshots: a procedure whose replay needs a binary input is refused, not stored;
- office clarification tasks, which need a second user turn;
- data notes;
- rebuilding the workspace after a crash;
- the retired status;
- running more than one task at a time against the store (it has no locking).

## 12. Risks (evidence from Unify, capture-v1, 6 October 2026)

MEMORY's offline capture-v1 study traced Unify's delivered answers back to code cells. It then replayed the code
and tried the lifted code on the same job's next visit. The study used bwrap and made no model calls. The report is
`artifacts/research-regression-diagnosis-20261001/memory-a-v1/capture-v1/REPORT.md`.

**Capture only exists when answers come from code.** Only 26.3% of delivered ARC answers trace to a code cell.
On AppWorld it is 0.9%, and on ScienceWorld 3.8%, because their replies are narrative or game commands. The
"deliver only from code" rule (section 2.3) is what makes capture possible at all. It is also a behavioural change
that this model may resist, and that must be measured: the share of deliveries with a non-empty slice. For
environments driven by state-changing actions (AppWorld), the anchor would have to be the cells that made the
state-changing calls. That needs the API recording described in the experiment plan.

**Replaying a recorded case is weak evidence of fit.** When a capture existed, it replayed 91 of 91 times. But the
captured code, lifted and applied to the next visit, was right only 6 of 59 times. Functions written in Unify's
review step managed 10 of 59. The reason is that 256 of 258 ARC next visits changed the input's shape, and the
cells were scripts for one instance, with sizes, coordinates and assertions built in.

This means step 1 of section 3.4 (replaying recorded cases) protects against drift, but it says almost nothing
about whether the procedure fits a new request. In this design, fit rests on three later steps:

- binding (step 2);
- the request-shape check (step 3);
- the run on today's files with smell tests (step 4).

Data and file work, where the next visit usually keeps the file's structure, is the setting where lifting can
work. Whether it does is an open question.

**What we will measure.** The experiment plan reports the **fit rate** on its own: of the return visits whose job
has a stored procedure, how often the lifted procedure, bound to the next visit, gives the checker's right answer.
It is computed offline, with no model call, beside the share of returns where binding succeeds at all. If office
fit is as low as ARC's, the memory half of this design does not pay. In that case the loop and the gate have to
carry it, and the "without memory" arm will show that.

## 13. API worlds, and the AppWorld and ARC systems (phase 1b)

**One door to the world.** The workspace reaches an API world only through one Unix socket. The harness binds it
into the sandbox at `/run/world/gw.sock`, and the network stays unshared. Behind the socket is a gateway that the
harness runs outside the sandbox. It speaks the AppWorld relay's JSON-lines protocol. The workspace gets an
`apis` object with AppWorld's calling convention, loaded as a preamble and hidden from the variable listing.

**The gateway's three modes:**

| Mode | What it does | Used for |
|---|---|---|
| live | Forwards each call to the runner's relay and records the request, the answer, the cell that made the call, and whether the call changes state (method other than GET or HEAD) | The model's own work |
| read-only | Forwards only calls that do not change state. It captures the completion call and never sends it. | Pre-running a read-only procedure for an offer |
| replay | Answers from a recording and has **no upstream at all**. It refuses any call that is not in the recording. Each recorded state-changing call is answered once, in order. | Replay verification of stored procedures |

**Secrets.**

- **Detection.** Values under credential-like keys are found in requests and responses. The keys searched for are
  `password`, `token`, `secret`, `api_key`, `otp`, `cvv` and `pin_code`.
- **In recordings,** these values are replaced with consistent labels such as `<secret:1>`. A replay that passes
  a label back therefore matches.
- **In stored procedures, a literal secret becomes a credential parameter.** The parameter is bound at call time
  by a recipe: the recorded call that first returned the secret, plus the path to it. An example is
  `cred1 = _cred(apis.supervisor.show_account_passwords(), [{"match": {"account_name": p1}}, "password"])`.
- **Live,** the recipe fetches today's value. In replay, it gets the label back from the recording. This answers
  EVAL's finding that Unify's stored functions received `{{token}}` placeholders: here credentials are explicit,
  and they are bound when the procedure runs.
- **Calls that issue credentials,** such as a login, are not effects on the task. They may be repeated in
  replay, and they do not make a procedure "state-changing".
- **No source, no store.** A secret with no recorded source, such as one typed from the instruction text, makes
  its procedure unstorable.

**Answers and effects.**

- In a world, the work finishes with the world's completion call: AppWorld's
  `apis.supervisor.complete_task(...)`. The harness does not accept `deliver()` there.
- **The gate.** The completion call is gated once, inside the gateway and before anything is sent. The gate
  checks:
  - value smells on the answer, when an answer is given;
  - ungrounded literals, checked against the text of every API response the session saw. This is the
    'meal' versus 'meals' check, with API data in place of files.
- **A procedure's answer** is its effects: the state-changing calls in order, with their redacted arguments,
  plus the completion answer. Replay must reproduce exactly that.
- **The slice** is seeded with every cell that changed the world. Calling `apis` is not a data dependency between
  cells.

**Offers.**

- A procedure that changes the world is **never pre-run**. It is offered as a verified cell for the model to run,
  with its recorded state-changing calls listed.
- A read-only procedure is pre-run through the read-only gateway.

**The systems.** `CleanslateAppWorld` and `CleanslateARC` are `SystemLearner`s for the runners that Unify's cells
use.

- **The request text is the runner's message, byte for byte**, because the runner renders it. A logged
  AppWorld message and a logged ARC message both reach the model unchanged (tests).
- **No example checking is prompted.** The harness adds no words about examples, demonstrations or checking.
- **ARC:** a delivered list of lists becomes `{"action": "submit", "grid": ...}`. A dict with an `action` passes
  through. Follow-up messages within an instance continue it, with the runner's messages and the system's
  replies as the request.
- **Model calls** go to the runner's tracking proxy, so the runner's own journal records the spend.
- **Limits:** `start()` sets the worker limits profile and proves it with one confined workspace, or refuses.

**ARC binding, measured offline.** Prediction: lifting never gives a ready offer on ARC.

- Data: 36 same-task request pairs from Unify's logged ARC first turns (`arc-ufix4-h-low-s0`).
- **0 of 36 pairs can ever be shape-equal.** Every pair differs in single-digit grid cells, which are never
  parameters.
- Even lifting every liftable word binds only 4 of the 36, and none has the same request shape.
- So, as predicted, ARC returns get references and no one-turn reuse. References only add tokens there
  (cost-curve.md).

**Tests (phase 1b):**

- `test_world.py` (5 tests):
  - capture with two credentials becoming parameters, and no secret anywhere in the store;
  - **replay cannot reach the live world**: it has no upstream, an off-recording call is refused, the live relay's
    socket is absent from the sandbox, and the live call count is unchanged;
  - a return visit in a *new* world, with another password, runs the offered cell, and the credentials are
    fetched live;
  - the completion gate holds a 'Rock' versus 'rock' zero, and nothing is sent until the fix;
  - `deliver()` is not the completion call in a world.
- `test_systems.py` (5 tests): the ARC and AppWorld systems, the logged requests unchanged, no example wording,
  and the worker-profile preflight refusing on this laptop.

**Verified only on a worker or in the runner:**

- nested bwrap inside the runner's outer sandbox;
- the user scope or dedicated user reachable from inside it;
- the learner inside the real `continual-appworld run`;
- the gateway against the real relay;
- OpenRouter's acceptance of the request shape.
