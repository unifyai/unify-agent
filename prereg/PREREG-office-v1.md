# PREREG: cleanslate-v1 on the everyday office-v1 core stream

Status: FROZEN 2026-10-06T22:23Z by MAIN
Drafted by the cleanslate sub-agent (PREREG-office-v1-DRAFT.md, kept unchanged beside this file). MAIN's changes
from the draft: the A1 pairing decision in §2.1 and the review-defect note in §2.2. Nothing else changed.

## 1. Question

On recurring office work, does cleanslate-v1 become **cheaper on return visits** without solving fewer tasks
than Unify? cleanslate-v1 is a harness with one Python workspace, answers delivered from code, a smell gate, and
procedures captured by the harness and checked by replay before reuse. The two Unify arms are upstream@main and
lean-all.

Does its procedure memory, as opposed to its loop and gate alone, cause the saving?

## 2. Arms

| Arm | What | Identity |
|---|---|---|
| **cleanslate-full** | cleanslate-v1 with procedure memory (offers, references, capture), the smell gate and compaction | Source digest `d3ceac9b0760a33f55db95bc3bfbc47af898f9c6b86721006f1368da4fea3825`. The file hashes are in §9. |
| **cleanslate-no-memory** | The same loop, gate and compaction, with no procedure store | The same source |
| **A0: Unify @ main** | Commit 592685713, build `unify-agent-prelimup-v1`, built-in guidance off | EVAL's office confirmation, 3 runs on 6 Oct (§2.1) |
| **A1: lean-all** | Commit af8958e5d, build `unify-agent-overhaul-v4b-af8958e5d`, the lean-all settings plus the placeholder note | EVAL's office confirmation, 3 runs on 6 Oct (§2.1) |

### 2.1 Baseline runs reused (not re-run)

EVAL's readout is `artifacts/everyday-v1/readout-confirmation-v1/office-rows.jsonl`, sha256
`7d6ddecdf9354a5afcba80968858f960c7e0607c493c3adb4a6614f71066197c`. The cells are
`cells-conf-office-core-p{1,2,3}.json`.

| Arm | Run ids (everyday-office-persistent-…) | Solved (of 24) | USD per run |
|---|---|---|---|
| A0 | `…unify-prelimup-v1-gpt-6-luna-low-ue3b1b8785-all-20261006T112434Z`, `…T112903Z`, `…T114928Z` | 22, 22, 22 | 0.191214595, 0.179645295, 0.180568090 |
| A1 | `…unify-overhaul-v4b-af8958e5d-gpt-6-luna-low-ue06512b09-all-20261006T112452Z`, `…T112920Z`, `…T114947Z` | 22, 21, 23 | 0.092791620, 0.093580275, 0.092471285 |

These baseline values come from `analysis/office_analysis.py --baseline-readout`:

| Arm | Return USD ÷ the same jobs' first-visit USD, per run | USD per instance | Calls per instance |
|---|---|---|---|
| A0 | 1.145, 1.045, 0.900 | about 0.0075–0.0080 | about 10.8 |
| A1 | 0.757, 0.968, 0.737 | about 0.0039 | about 8.2 |

**Decision (MAIN): A1 is re-run fresh and paired.** 2 fresh A1 runs (lean-all af8958e5d, the same build, settings,
caps and office runner as the 6 Oct A1 cells) run on the same worker in the same wave as the cleanslate cells: one
fresh A1 run per run index. Estimated USD 0.19. **All A1 comparisons in §6 (S1b, S2, H4) use these 2 fresh runs.**
The 3 runs from 6 Oct are reported beside them, and so is the spread of all 5. A0 is not re-run; its 6 Oct runs are
reference rows only (H5).

### 2.2 Known difference in the Unify rows

Every A0 and A1 office run (6 Oct and fresh) has the review defect found on 6 Oct: the after-task storage review
sees "processed stopped early, no result" instead of the answer, because the office runner refuses UNIFY_OUTCOME.
This affects what Unify saves, not how a task is scored. cleanslate has no after-task review. This is reported
as a named difference beside every comparison of memory or return cost; UNIFY_REVIEW_LAST_REPLY is not
added to A1 here, so A1 stays the recorded lean-all configuration.

## 3. Instances and order

- **Instances:** the office-v1 core stream, all 24 instances, in the frozen order of `stream.json`
  (`~/.local/share/continual-harness-research/everyday/office-v1/stream.json`, sha256
  `7ee55096e7ddcd2b453473989be562de86f84deaff0401d0eac0f160ccac6bb9`; `benchmarks/everyday/MANIFEST.json`, sha256
  `be1682931e53ccf9d1202d4da10abee319bf3b372a219d0b0a9fa8565e50b24c`):
  ofc-s01, ofc-d01, ofc-s04, ofc-d09, ofc-d06, ofc-s03, ofc-s07, ofc-s05, ofc-s08, ofc-s02, ofc-d03, ofc-r04,
  ofc-r05, ofc-s06, ofc-d04, ofc-d08, ofc-r01, ofc-r02, ofc-r07, ofc-d07, ofc-d05, ofc-d02, ofc-r03, ofc-r06.
  There are 17 first visits and 7 returns, from 7 recurring jobs. The optional clarification tasks are excluded,
  as they were in A0 and A1.
- **What each instance receives:**
  - The model gets the stream's prompt byte for byte. The prompt is checked against `stream.json`, `task.json` and
    the manifest.
  - The workspace is the task's `workspace/`. Its tree hash is checked against `task.json`.
  - Task ids are offline labels only.
- **Scoring:** `python -m everyday.score <task> <ws>`, the same scorer the A0 and A1 runs used, in a separate
  process. `score.py` sha256 `f7c72740…501cb`; `office/tasks.py` (checkers) sha256 `a1fe2a44…439d2`.
- **Memory:** persistent across the stream, as in A0 and A1. The store starts empty in each cell.

## 4. Runs, settings and caps

| Setting | Value |
|---|---|
| Runs | 2 cells per cleanslate arm: run index 1 and run index 2. Cells with the same run index (full and no-memory) run on the same worker in the same wave. |
| Command | `kit/run_cell.sh --arm full\|no-memory --run-index 1\|2 --tasks all --order frozen --confirm-paid --prereg <this file once frozen> --prereg-sha256 <its hash>` |
| Model | `openai/gpt-6-luna` through OpenRouter, `reasoning.effort = low` (as A0 and A1) |
| Per-instance caps | USD 0.50, 200 model calls, 900 s wall (as A0 and A1). A cap ends the task, which is still scored. |
| Per-cell runaway guard | Stop the cell above 40 USD/h over 10 minutes (exit 4) |
| Per-cell timeout | 60 s for each code cell |
| Context budget | 24,000 characters, with deterministic compaction |
| Confinement | bubblewrap. Worker limits profile: a user scope with memory, pids and cpu verified, or a dedicated user. The preflight refuses otherwise. |
| Content capture | **On.** Transcripts (the prompt, the model's replies, the harness's observations), final workspaces, the scored states of held deliveries, and per-task procedure-store snapshots. No credential appears in any of them: the tests check that no file the run writes contains the key. |
| Cost record | Provider-reported charges as decimal strings. Unpriced calls are counted and never zero. Journal in the lab format, counted by `cost_ledger.py` under `everyday`. |

## 5. Outcomes

**Primary**, per run and per arm:

1. **Solved**, out of 24, by the checker. An unknown grade stays unknown.
2. **USD per instance:** the provider-reported charges. A run with unpriced calls reports its priced subtotal and
   its unpriced count, and its total stays unknown.
3. **Return-visit cost:** USD of the 7 returns ÷ USD of the same 7 jobs' first visits. Also mean USD per return
   and per first visit.

**Secondary:**

- **Hold rate:** the share of instances with at least one smell hold.
- **False holds:** holds whose held state the checker passes. The held state is copied and scored. Reported as a
  share of correct deliveries and as a share of holds.
- **Fit rate** (cleanslate-full; `--fit`): of the returns whose job has a procedure stored by an earlier visit in
  the same run, the share where that procedure, bound to the return's request and run on the return's own files,
  passes the checker. This needs no model call. Bindable and same-request-shape counts are reported beside it.
- **Offers:** made, accepted, accepted and correct, and references shown.
- **Other:** calls per instance, no-op cells, and caps hit.

**Intervals.** Solved uses a run-then-task bootstrap: 2,000 draws with seed 0. Each draw resamples runs with
replacement, then tasks within each run. Every run is listed, with its spread, and no headline rests on a single
run.

## 6. Pre-stated criteria (`analysis/office_analysis.py` THRESHOLDS)

**Success.** Office counts as a gain only if all of the following hold for **cleanslate-full**:

| Id | Criterion | Threshold |
|---|---|---|
| S1a | Return USD ÷ the same jobs' first-visit USD (mean over runs) | ≤ 0.50. Reference values: A1 0.74–0.97, A0 0.90–1.15. |
| S1b | Mean USD per return ÷ A1's mean USD per return | ≤ 0.70 |
| S2 | Mean solved − A1's mean solved | ≥ −1 (A1's mean is 22.0) |
| S3 | Accepted ready offers that pass | ≥ 95%. Not evaluated if none were accepted. |
| S4 | False holds ÷ correct deliveries | ≤ 10% |

**Memory's effect.** This is reported beside the criteria, not as a criterion. It is the difference between
cleanslate-full and cleanslate-no-memory in return-visit USD and in return solved. Memory is credited with a saving
only if the full arm's return/first ratio is below the no-memory arm's in both runs.

**Harm.** If any of the following happens, the arm stops (after it has been seen) and the evidence is kept:

| Id | Condition |
|---|---|
| H1 | Any sign of a confinement problem |
| H2 | 2 or more accepted ready offers that fail, in one run |
| H3 | False holds above 20% of holds |
| H4 | USD per run above 2× A1's |
| H5 | Mean solved 3 or more below A0's |

A criterion whose inputs are missing is reported as "not evaluated". It is never reported as a pass.

## 7. Cost

**Estimate per cell:** about USD 0.10, with a range of 0.05–0.25. This assumes 4–8 calls per instance at about
the same cost per call as A1 (USD 0.00047). The prompts are shorter, because there are no tool schemas. A1 costs
about USD 0.093 per cell and A0 about 0.18.

**4 cells:** about USD 0.2–1.0.

**Bounds:**

- The caps bound a cell at USD 12 (24 × 0.50).
- The 40 USD/h guard stops a runaway within about 10 minutes.

**Optional:** 2 fresh A1 cells, about USD 0.19.

## 8. Order of operations

1. **Phase 0 checks**, already passing offline. The test suite (57 tests) passes on the laptop.
2. **On the worker:**
   1. A fake cell (`--fake`) must exit 0 with clean cleanup. Its preflight reads back memory, pids and cpu limits.
   2. A paid canary: `--tasks ofc-s01,ofc-d01`, one cell, cleanslate-full, run index 0, reported apart. It must
      show that OpenRouter accepted the request and reported `usage.cost`, and that the journal is counted by the
      ledger.
3. **The four cells:** r1 for both arms in one wave, then r2 for both arms.
4. **Analysis:**
   `analysis/office_analysis.py --cell <4 cells> --baseline-readout artifacts/everyday-v1/readout-confirmation-v1/office-rows.jsonl --fit --out <report>`.
   Each verdict is explained with trace excerpts: transcripts, held states and fit items.

## 9. Source hashes (sha256)

The kit records the same hashes in each `cell.json`. A mismatch with this table is a deviation that must be
reported.

```
2b33f9a74137d75c1feca0b81fc18593bfbc1cfe349b639e053ec5ca23dce372  adapters/office.py
dcbe2fc2460d101dbaf3487b93b82c48f4fd6e72e2854dab8e8fe6519212eb86  adapters/office_fake.py
2a1cd62695fb79ad6733f1048ff98b0085ae2bfd9e4b9621512f31bd659b5626  analysis/office_analysis.py
35044065de79059760446f4b8b2f7cab92cf07f2f30cd4926e7f7aeed8027fc9  cleanslate/__init__.py
fff9fb0d9c3b3b8631dbb5e988fd32f57c8541193c2c6b9b998c9bdaeed1b769  cleanslate/agent.py
f2f2ad7fc50f4a49ee68d2da0bad7927808d1c92b71c726f5bef26ff295e8588  cleanslate/analysis.py
f9a19816c0f483260a7d6c12a85aa7da307e0ba64fdea1ef2d7ea8c6dd5e69dd  cleanslate/cost.py
8c799a2558bb54c20797e3dc8c0e1a3c4948d2945735315c6182c78083f8c6f6  cleanslate/limits.py
0ec7dbb924b001b4c7ae2e14d39b487d28cb736057d2e0d889506e8ebed18c72  cleanslate/llm.py
278e22a953d4bb7281d0c65511646afaf3d1eab6359f8f631faf9bc6a2c5519f  cleanslate/memory.py
50bcebd9f74a3f475a5dd0368626684faa68955f9e4f8b12b8443237f25a3589  cleanslate/sandbox.py
f578259d757d60e192db628d3a0631bd39ada4d0ad175bb94cab01bae811f142  kit/README.md
4beed746330800d5cefa621c69f48dd73faf4546a4ef5e09620b61cd4200adc4  kit/run_cell.py
c6a6c3c3b50baa1130813da22fabae672d147574681fa444e347d42bb287947e  kit/run_cell.sh
digest d3ceac9b0760a33f55db95bc3bfbc47af898f9c6b86721006f1368da4fea3825  (sha256 of the lines above, joined by newlines)
```

## 10. Known limits stated before the run

- **Untested model behaviour.** Neither cleanslate arm has met the real model yet. Its adherence to "one code
  block" and `deliver()` is unmeasured.
- **The fit rate may be low.** The office returns are paraphrased, so strict binding may rarely succeed. In
  that case the full arm gets references, not one-turn offers, and S1 is unlikely to hold. That would itself be
  the answer to the memory question (design.md §12).
- **Not confined on any host:** the read-only `/usr` is visible to the workspace. Disk is hard-bounded only when
  the run folder sits on a size-limited filesystem.
