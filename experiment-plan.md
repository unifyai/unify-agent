# Experiment plan: cleanslate-v1 against Unify

Status: phase 0 approved by MAIN overnight; paid phases need a frozen prereg + MAIN GO. (Proposal written 7 October
2026; phase 0 progress is recorded in section 4.) No paid call, worker job or benchmark run has been made for this
plan. The cost figures below are estimates. They are based on per-run costs
already measured for the existing arms (`benchmarks/everyday/docs/matrix-v1.md`).

## 1. The question

Does a harness that keeps the code that actually produced each answer, checks that code before reusing it, and
checks answers for smells get **cheaper on recurring work** without losing accuracy? Does it avoid **harm on
look-alike requests**? The comparison is against Unify as the team knows it (upstream@main) and against the
current best Unify configuration (lean-all).

## 2. Arms

Every arm uses the same model, the same reasoning effort, the same per-instance caps on time and money, the same
instances and the same order. Arms run in alternation on the bench_workers VMs so that they see the same provider
conditions. Every arm is identified by its source commit or file hashes, which are recorded at launch.

| Arm (descriptive name) | What it is |
|---|---|
| **Unify upstream@main** | The A0 rows of the existing matrix. Existing runs are reused if the model, effort and prices match; otherwise there are 2 new runs. |
| **Unify lean-all** | The A1 rows (base configuration af8958e5d). Reuse rule as above. |
| **cleanslate-v1 full** | This prototype, with procedure memory, offers and the smell gate |
| **cleanslate-v1 without memory** | The same loop and smell gate, with no procedure store. This separates the effect of the loop from the effect of memory. |
| Paper harnesses (ARC only) | The existing paper rerun bands, shown for reference. No new runs. |

A cleanslate row is a separate harness. It is reported beside the paper-matched rows, never inside them.

## 3. Benchmarks, in order

1. **Everyday office core** (primary). 24 tasks in a frozen order, each with a deterministic checker. The tasks
   are questions over CSV, JSON and log files, file and shell chores, and seven recurring jobs that come back in
   other words over new data. This set tests exactly what the design is for.
2. **AppWorld 12-task train canary** (secondary). This is the same subset as the existing AppWorld screen. If
   office passes, an extension is the everyday AppWorld train-v1 set: 30 jobs and 150 instances, 120 of them
   returns. Only training and development scenarios are used. The test set stays blind.
3. **Continual-ARC LOW-25** (transfer check). The first 25 instances of the 9-rule workstreams stream, seed 0.
   The prediction, stated before any run: ARC requests contain grids rather than values that can be lifted into
   parameters, so binding will usually fail. The expected result is references, not one-turn reuse. Any gain
   should therefore come from the loop and the gate, not from memory.

## 4. Work needed before any paid run (phase 0, no paid calls)

MAIN approved phase 0 as offline engineering only. The order was bubblewrap, then file deliverables, then the cost
record, then the office adapter with a dry run. API recording and the trace smell audit stay as written below and
are not built yet.

| Item | Why | State on 7 Oct (offline only) |
|---|---|---|
| Run the workspace under bubblewrap: `--unshare-all` (no network for now), read-only `/usr` and interpreter, read-write task directory only, a cleared environment, HOME in a private tmpfs, the existing time and memory limits. The same applies to replay workspaces. | Required by the repo's rules. Benchmark checkers and generators must never be visible inside the workspace. | **Done.** 7 tests prove it from inside: no parent environment, no home or repository, no network, no writes outside the task directory, hidden checker paths invisible, termination verified. |
| **Phase 0.5: resource limits.** `systemd-run --user --scope` with MemoryMax, MemorySwapMax=0, TasksMax and CPUQuota, read back from the cgroup. Otherwise prlimit as a dedicated user. Plus rlimits (CPU seconds, file size, address space) and a task-folder size check. | Bound memory (including `/tmp`), processes, CPU and disk | **Done** (8 tests: exceeding each limit from inside). **Verified only on the worker:** cpu delegation (not delegated on the laptop, where the worker profile correctly refuses), the dedicated-user mode, and the hard disk bound (`--max-fs-mb` on a size-limited filesystem). |
| File deliverables: `deliver(path)`, with a content hash and file smells (empty, header-only, zero rows) | Office tasks are scored on the files they produce | **Done** (6 tests) |
| A cost record with decimal-string USD; unpriced calls counted, never zero; full run, attempt and request identities | Repo rules, and the cost comparison | **Done** (10 tests in phase 0.5): the provider charge is the charge; the journal is written in the lab format and accepted by `journal_accounting.read_journal` and `cost_ledger.runs_under`; the per-instance caps (USD 0.50, 200 calls, 900 s) and the 40 USD/h runaway guard work. |
| **Phase 0.5: real-model client.** OpenRouter, `openai/gpt-6-luna`, effort low (the matched A0/A1 office cells). The key comes from the environment and never reaches the workspace. | The paid route | **Done** (3 tests against a local stand-in, plus a paid-path rehearsal through the kit). Whether OpenRouter accepts the request as sent and reports `usage.cost` is verified on the first paid instance. |
| **Phase 0.5: worker run kit** (`kit/`) | OPS runs cells | **Done**: venv, preflight, one cell, verified cleanup, refusals, lab layout (3 tests) |
| Office adapter: the drive is the workdir. The model gets the stream prompt byte for byte, checked against stream, task.json and manifest. Task ids stay offline labels. Scoring uses the benchmark's own scorer in a separate process. | Same request text that Unify receives | **Done**, plus a dry run with the scripted fake on 2 instances (`ofc-d01`, `ofc-r01`): 2/2 scored as passed by `everyday.score`, cleanup verified, USD "0" (fake). Clarification tasks (second turn) are not supported yet. |
| API recording and replay inside the workspace, for AppWorld | Without it, a procedure that calls an API cannot be replay-checked, so nothing would be stored. State-changing calls must never be repeated live during verification. | **Phase 1b, done offline:** a gateway with live, read-only and replay modes; the replay gateway has no upstream; secrets are labelled in recordings and become credential parameters fetched at call time (5 tests against a fake relay) |
| AppWorld and ARC adapters. AppWorld: the client in the workspace, the server outside it behind an allow-listed port. ARC: the answer delivered as a list of lists. | As for office | **Phase 1b, offline:** both are `SystemLearner`s for the runners Unify's cells use, so the request text is byte-identical. AppWorld goes through the recording gateway (one Unix socket). ARC: 0/36 logged same-task pairs could ever bind (design.md §13). Not yet run inside the real runners (worker). |
| Offline smell audit. Run `value_smells`, `file_smells` and `ungrounded_literals` over the final answers and code of existing Unify office and AppWorld traces, on the workers, with no model calls. | Estimates the hold rate and false-hold rate before spending anything. If the false-hold rate on correct answers is above 20%, the gate is tuned first. | Not started (deferred by MAIN) |
| **Fit rate.** For each recurring office job, take the procedure stored at visit *k* and replay it against visit *k+1*'s request and files: lift, bind, run, then score with the checker. No model call is needed. | MEMORY's capture-v1 study found that lifted Unify captures fit the next ARC visit only 6/59 times (design.md §12). This number decides whether memory can pay at all. | **Done** in `analysis/office_analysis.py --fit`. It runs on each paid cell's stored procedures and on the per-task store snapshots. |
| Pre-registration: this plan, the source hashes, the instance lists and the analysis script | So the success criteria cannot move after the results are seen | **Draft** at `prereg/PREREG-office-v1-DRAFT.md`. MAIN freezes it. |

What remains before the paid office phase:

- MAIN reviews and freezes the prereg, and gives GO;
- OPS prepares a worker: cpu delegation or a dedicated user, optionally a size-limited run filesystem, and the
  office data;
- OPS runs a fake cell, then a paid canary.

AppWorld still needs API recording.

## 5. Phases (each only after the previous one passes its checks)

| Phase | What | Runs | Estimated model spend |
|---|---|---|---|
| 1. Canary | Office, first 6 instances, cleanslate-v1 full, 1 run. This is a plumbing check and is not reported as a result. | 1 | about $0.05 |
| 2. Office core (prereg draft) | cleanslate-v1 full and without memory, **2 runs each**, frozen order. Unify A0 and A1: reuse EVAL's 3 runs each from 6 Oct (conf-office-core p1–p3). | 4 | about $0.1 per cell (range $0.05–0.25); $0.2–1.0 in total. The cap ceiling is $12 per cell. |
| 3. AppWorld train canary (prereg draft `prereg/PREREG-appworld-train-canary-DRAFT.md`) | cleanslate-v1 full and without memory, plus a fresh, paired lean-all, 2 runs each. This is A3's 12-task stream. | 6 | $0.4–1.0 |
| 4. ARC LOW-25 | cleanslate-v1 full, 2 runs, plus without memory if phase 2 showed memory has an effect. A1 on the same model if the existing runs do not match it. | 2–4 (+2) | $0.8–2.0 (+$1.1) |

**Total: about $2–7 in model spend (the range of the rows above), or up to about $9 with 30% contingency.** There is also worker VM time of a few hours. The
existing per-run costs at LOW effort are:

- office: A0 about $0.18–0.19 and A1 about $0.09;
- ARC LOW-25: A0 about $1.01 and A1 about $0.39–0.54;
- AppWorld 12 at HIGH effort: about $0.19.

Each cell runs its own runaway guard: it stops the cell above 40 USD/h over 10 minutes, reading its own cost
record. The lab's `cost_ledger.py --watch` covers everything else.

## 6. What is reported

Every number appears beside Unify upstream@main, Unify lean-all and, for ARC, the paper harness bands. Each comes
with all runs listed, the mean, the spread, and a run-and-task resampled interval. No headline rests on a single
run.

- **Accuracy**: solved per run. Office: first visits, returns and look-alikes, reported separately. ARC: the
  exposure-index breakdown, a per-rule plot and a leave-one-rule-out check. Best-of-N is shown only when labelled
  as such.
- **Cost**: USD per run, calls per instance, and prompt and completion tokens. For each recurring job: the cost of
  a return visit divided by the cost of the first visit.
- **Fit rate, reported on its own.** Of return visits whose job has a stored procedure, the share where the lifted
  procedure, bound to the return's request and run on its files, gives the checker's right answer. It is reported
  beside two other shares: returns where binding succeeds at all, and returns that got a ready offer, a file-cell
  offer or only a reference. This is computed offline, per job, for both stored-procedure arms. ARC is reported
  the same way, so it can be compared with capture-v1's 6/59.
- **Mechanism counts** (from traces). Each comes with two or three real trace excerpts, so the reasons are visible
  and not only the totals:
  - ready offers made, accepted, and accepted but wrong;
  - references shown;
  - stale detections;
  - holds, changed answers after a hold, correct answers after a hold, and false holds;
  - share of no-op cells;
  - compaction events and the largest prompt size;
  - deliveries that could not be stored, and why.

## 7. Success criteria (all must hold to call office a gain)

- **S1 Cheaper returns.** On the seven recurring office jobs, the mean cost of a return visit for cleanslate-v1
  full is ≤ 50% of its own first-visit cost and ≤ 70% of lean-all's return-visit cost.
- **S2 No accuracy loss.** Mean solved for cleanslate-v1 full is ≥ lean-all's mean minus 1 task (of 24), and its
  interval does not lie entirely below lean-all's. Return accuracy is ≥ first-visit accuracy.
- **S3 Precise reuse.** At least 95% of accepted ready offers are correct, across all runs. No accepted ready offer
  is wrong on a look-alike instance.
- **S4 Mechanisms behave as designed.** 100% of stored procedures replay to their answers. The share of no-op
  cells is ≤ 15%. False holds are ≤ 10% of correct deliveries.

AppWorld uses the same criteria: non-inferior accuracy, cheaper returns and precise reuse. ARC is reported as a
transfer check. It counts as a gain only if both runs exceed the top of the lean-all band (18 of 25) at no higher
cost. It counts as harm if the mean falls more than 2 below lean-all's mean.

## 8. Harm and stop criteria (stop the arm, keep all evidence, report)

- **H1** Any sign of a confinement problem: access outside the task directory in the bubblewrap audit, network
  access outside the allow-list, or a credential-like string in a trace. Stop everything.
- **H2** Two or more accepted wrong ready offers in one run. Stop the full arm and continue only without memory.
- **H3** Holds add more than 15% to cost, or more than 20% of holds are false.
- **H4** A run's cost goes above twice lean-all's on the same benchmark, or any instance hits its money cap
  repeatedly.
- **H5** After 2 office runs, mean solved is 3 or more tasks below upstream@main.

## 9. What would change the design

| Finding | Change |
|---|---|
| S3 holds but returns are rarely offered (paraphrased returns) | Test a cheap yes/no applicability judge *offline* on the recorded references first. Only then consider pre-running on paraphrases. |
| Many false holds | Narrow the gate. For example, hold zero only when the code also filtered on something. |
| Many deliveries cannot be stored | Look at the reasons, for example files created in the session or randomness, and widen the snapshot or replay rules |
| ARC: no difference between full and without memory | This is expected. It confirms that executable procedures fit data work, and the ARC row stays a transfer check. |
| Office fit rate below 25%: lifted procedures rarely fit the next visit, as with ARC in capture-v1 | Memory does not pay through one-turn reuse. Decide on evidence between three options: store procedures as functions with explicit inputs (for example "the file and the filters"), not only as lifted scripts; show references only; or drop procedure memory and keep the loop and the gate. |
