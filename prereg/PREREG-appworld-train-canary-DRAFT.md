# PREREG (DRAFT): cleanslate-v1 on the AppWorld train-canary stream

Status: DRAFT, written 7 October 2026 by the cleanslate sub-agent for MAIN's review. It is not frozen and not
authorised. To freeze it, MAIN removes "DRAFT" from the title and status line, writes
`Status: FROZEN <UTC time> by MAIN`, and records the file's sha256.

This prereg mirrors PREREG-office-v1. It runs on the AppWorld stream that A3 uses, and its baseline is a
**fresh, paired** lean-all arm.

## 1. Question

In AppWorld, a task's variants are generated from one template with new values. Does cleanslate-v1 get
cheaper on the return visits without solving fewer tasks than lean-all? Here cleanslate-v1 means:

- an API gateway that records every call;
- procedures captured by the harness and checked by replay against the recording (never against the live world);
- credentials fetched at call time instead of stored;
- a gate on the completion call.

Is any saving due to its procedure memory?

## 2. Arms (all paired: the same worker and the same wave for each run index)

| Arm | What | Identity |
|---|---|---|
| **cleanslate-full** | `--learner cleanslate_systems:CleanslateAppWorld`, learner kwargs `{"memory": true, "model": "openai/gpt-6-luna", "reasoning_effort": "low", "max_task_usd": "1.00", "max_wall_s": 1200}` | Source hashes in §9 (working tree of 7 Oct; it is **not** the office snapshot) |
| **cleanslate-no-memory** | The same kwargs, with `"memory": false` | The same |
| **lean-all (fresh)** | Unify A1: `unify-agent-overhaul-v4b-af8958e5d` with lean settings (`make_cells_ov1.env_for('appworld', 'lean')`) plus `UNIFY_PLACEHOLDER_NOTE=1` (matrix-v1 §2) | Build af8958e5d. **MAIN to confirm** whether A3's own lean arm (L, on `unify-agent-overhaul-mema-ec328ed87`) should be used instead, so the row is directly comparable to A3. |

All arms use `openai/gpt-6-luna` with reasoning effort `low`, the relay route, and the persistent arm, as A3's
AppWorld LOW cells do.

## 3. Instances and order

- **The stream:** `appworld_campaign.py --subset train-canary`: 12 train tasks, in a fixed order with the
  scenarios interleaved.

  e3d6c94_1, 692c77d_1, ce359b5_1, e7a10f8_1, e3d6c94_2, 692c77d_2, ce359b5_2, e7a10f8_2, e3d6c94_3, 692c77d_3,
  ce359b5_3, e7a10f8_3.

  That is 4 first visits and 8 returns. The scenarios were drawn with `random.Random(0).sample` over the sorted
  train scenario ids (docs/appworld-adapter.md §19). This is the A3 stream (BOARD, 6 Oct: "12, not 30").
- **Splits:** only training data is used. The test split is refused in code by the adapter.
- **Grading:** AppWorld's own grader, run by the runner (task goal completion per task). An unknown grade stays
  unknown.
- **What the harness sees:** the runner renders every message (`continual_appworld.protocol`). The cleanslate
  system receives the same text as Unify, byte for byte, and passes it on unchanged (test:
  `tests/test_systems.py`, with a logged AppWorld message). Task ids are offline labels only.

## 4. Runs, settings and caps

| Setting | Value |
|---|---|
| Runs | 2 cells per arm: run index 1 and run index 2. Each run index puts its three arms on one worker in one wave. |
| Runner | `continual-appworld run --learner … --learner-kwargs … --schedule train-canary --arm persistent --route relay --max-nudges 2 --max-wall-s 1200 --max-task-usd 1.00` (the AppWorld LOW cells' defaults) |
| cleanslate per-task caps | USD 1.00 (also enforced at the runner's proxy), 200 calls, 1,200 s |
| Runaway guard | 40 USD/h over 10 minutes. The lab's `cost_ledger.py --watch` also runs. |
| Model route | The runner's tracking proxy (`ARC_LLM_BASE_URL` and `ARC_LLM_API_KEY`), so spend lands in the runner's own journal, as for Unify |
| Confinement | Each cleanslate workspace runs under bubblewrap. The gateway socket is its only way out. Worker limits profile, which `start()` proves or refuses. |
| Content capture | **On**: the runner's `--capture-content`, plus cleanslate's per-call recordings (redacted) |

## 5. Outcomes

**Primary**, per run and per arm:

1. **Solved,** out of 12.
2. **USD per instance.**
3. **USD of the 8 returns ÷ USD of their scenarios' first visits.**

**Secondary:**

- Cell offers made and run, and references shown.
- Hold rate, and held completions re-sent unchanged and then passed. This is a lower bound on false holds: a
  held call never reaches the world, so it cannot be graded directly.
- Calls per instance, and API calls per instance.
- **Fit rate:** of the returns with a stored procedure from an earlier variant, the share where that procedure,
  bound to the return's request and run against the return's own fresh world, passes AppWorld's grader. This is
  a worker-side analysis with no model calls. It needs AppWorld's task worlds and is **not built yet** (§10).

**Safety outcomes** (must be zero): any secret value in a stored procedure or recording (scanned), and any call
reaching a live world during replay verification (gateway logs).

**Intervals:** the same bootstrap as the office prereg: 2,000 draws, seed 0, resampling runs and then tasks.

## 6. Pre-stated criteria

| Id | Criterion | Threshold |
|---|---|---|
| S1 | cleanslate-full: return USD ÷ the same scenarios' first-visit USD | ≤ 0.50 |
| S1b | cleanslate-full mean USD per return ÷ lean-all's | ≤ 0.70 |
| S2 | Mean solved − lean-all's mean | ≥ −1 (of 12) |
| S3 | Cell offers that were run and then passed | ≥ 90%. Not evaluated if none. |
| H1 | Any secret in a stored procedure or recording, or any live call during replay | Stop |
| H2 | Two or more run offers that fail in one run | Stop the full arm |
| H3 | USD per run > 2× lean-all's | Stop |
| H4 | Mean solved ≥ 3 below lean-all's | Stop |

Memory's effect is reported beside these, not as a criterion: full against no-memory, return USD and return
solved.

## 7. Cost

| Item | Estimate |
|---|---|
| Measured anchor | AppWorld LOW over 12 tasks costs about USD 0.09–0.13 per cell (unify-harness-ledger) |
| cleanslate, per cell (cost-curve.md §3) | About USD 0.07, range 0.05–0.20 |
| Lean-all, per cell | About USD 0.10 |
| **6 cells** | **About USD 0.4–1.0** |
| Bound | The caps allow at most USD 12 per cell. The guard stops a runaway. |

## 8. Order of operations

1. **Prerequisites** (none done yet):
   - an AppWorld cell generator for the cleanslate learner, from OPS or EVAL;
   - the learner importable in the runtime venv (`PYTHONPATH` to `prototypes/cleanslate-v1/adapters`);
   - on the worker, **nested bwrap inside the runner's outer sandbox**, and the worker limits profile (user scope
     or dedicated user) reachable from inside it;
   - the fit-rate analysis.
2. **Fake or rehearsal:** a zero-cost cell against AppWorld's scripted fake upstream.
3. **Paid canary:** the first scenario's 3 variants, cleanslate-full, reported apart.
4. **The 6 cells.**

## 9. Source hashes

These were generated at drafting time from the working tree of 7 October, by the same procedure as the office
prereg's: the sha256 of each file under `cleanslate/`, `adapters/`, `analysis/` and `kit/`, and a digest of
those lines. MAIN regenerates them at freeze if the tree has changed.

```
f6f6e36d3cea643858ebf5e7e9cb6af1307c3686f4e5a8f031cdadb106331990  adapters/cleanslate_systems.py
2b33f9a74137d75c1feca0b81fc18593bfbc1cfe349b639e053ec5ca23dce372  adapters/office.py
dcbe2fc2460d101dbaf3487b93b82c48f4fd6e72e2854dab8e8fe6519212eb86  adapters/office_fake.py
cfdd1a31620d0a6592fa30dab272f907c3e3636ede6dcb7aee6667eba614c0a6  analysis/binding_probe.py
2a1cd62695fb79ad6733f1048ff98b0085ae2bfd9e4b9621512f31bd659b5626  analysis/office_analysis.py
35044065de79059760446f4b8b2f7cab92cf07f2f30cd4926e7f7aeed8027fc9  cleanslate/__init__.py
b4316da14a1f33c8d035681878051615ce21deda84c2f66022338ed35d4aab47  cleanslate/agent.py
d8d860fed9581e34b55ffa20d317a1115f8685df64d6d77460e9e70ac30f19c8  cleanslate/analysis.py
f9a19816c0f483260a7d6c12a85aa7da307e0ba64fdea1ef2d7ea8c6dd5e69dd  cleanslate/cost.py
8c799a2558bb54c20797e3dc8c0e1a3c4948d2945735315c6182c78083f8c6f6  cleanslate/limits.py
22f406870aa5dd53690278079a59c062e68b4678df8a95b27169aa87c7a57fde  cleanslate/llm.py
e104f915e763a5f3e398a33d4f3fab0fc6fe624da46564d98831501c67d14446  cleanslate/memory.py
30589162f0f31a409bd757e87c544999905681cadd244b71c312366ca1f11c7d  cleanslate/sandbox.py
9593ffbc3ac11d04ce1e6b383a25bc38607cb092d7231effc3b72f81cacf3541  cleanslate/world.py
d19d50beadc29b2307773aa0c1c4b8a13a27624ecb7e15b81e0329c03665f935  kit/README.md
4beed746330800d5cefa621c69f48dd73faf4546a4ef5e09620b61cd4200adc4  kit/run_cell.py
c6a6c3c3b50baa1130813da22fabae672d147574681fa444e347d42bb287947e  kit/run_cell.sh
digest fd251f4021973f0bb54bbacaabb7044bffd906914d27de1ccff210fea2e70717  (sha256 of the lines above, joined by newlines)
```

## 10. Known limits stated before the run

- **Untested inside the real runner.** The learner has not run inside the real AppWorld runner. Its
  `SystemLearner` surface was tested against a stand-in base class, and its gateway against a fake relay that
  speaks the relay's protocol.
- **Credential recipes need a recorded source.** A recipe needs the call that first returned the secret to be
  in the recording. A secret the model typed from the instruction text has no source, so its procedure is
  refused, not stored.
- **A held completion call is never sent.** This is a cleanslate behaviour Unify does not have. It is
  reported, not hidden.
- **State-changing procedures are never pre-run.** They are offered as cells to run, so a wrong binding is seen
  by the model, not caught by the harness.
