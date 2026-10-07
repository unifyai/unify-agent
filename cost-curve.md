# Cost per completed task over a lifelong stream (cleanslate-v1)

Status: design note, 7 October 2026, offline. No paid call has been made with cleanslate yet. The numbers below
are of two kinds, and every row says which:

- **Estimates** from a token model of the harness.
- **Measured anchors** from the Unify cells that EVAL and OPS already ran.

The first paid office cells (prereg PREREG-office-v1, frozen) will replace these estimates with measurements.

## 1. Why the shape matters more than the level

A lifelong stream serves the same kinds of work again and again. What matters is how cost changes along the
stream, for one job:

- the first visit;
- the k-th return;
- the visit after the job's data or interface has changed, so that its stored procedure has gone stale.

Cost per completed task is the stream's spend divided by its solved instances. A harness can lower it in two ways.
It can make returns cheap, and it can make them succeed at least as often as first visits. A cheap return that
fails is the worst case.

## 2. The token model

Prices for `openai/gpt-6-luna`, from the pinned OpenRouter catalogue: input USD 0.10/M, cached input USD 0.01/M,
output USD 0.50/M.

One call costs (input tokens × price) + (output tokens × price).

- **Input** on call *t* of an instance is P0 + (t − 1) · T, up to the compaction ceiling B.
  - P0 is the fixed prefix: system text, request, and offers or references.
  - T is the growth per turn: the code plus the harness's observation.
  - B is about 6,000 tokens (24,000 characters). Compaction folds older turns once the prompt would exceed it.
- **Output** is O per call: code plus low-effort reasoning, about 400–600 tokens.
- **Caching.** The prefix is built to be stable. The system text and request block never change within an
  instance, and turns are folded in large chunks. So most of each call's input repeats the previous call's
  prefix, and can be served from the provider's cache at a tenth of the price. Whether OpenRouter applies caching
  for Luna on this route is unverified. Both columns are given below.

| Benchmark | P0 | T | Calls, first visit (n) |
|---|---|---|---|
| Office | ≈ 400 | ≈ 450 | ≈ 6 |
| AppWorld | ≈ 1,700 (the rules block, about 1,300) | ≈ 900 (API-doc observations are long) | ≈ 12 |
| ARC | ≈ 1,200 (the test grid) | ≈ 500 | ≈ 5 |

## 3. The curve, per benchmark (estimates; anchors are marked)

USD per instance. "Ratio" is the cost divided by the same job's first visit.

| Visit | Office | Ratio | AppWorld | Ratio | ARC | Ratio |
|---|---|---|---|---|---|---|
| First visit, no cache | 0.0024 | 1 | 0.0096 | 1 | 0.0020 | 1 |
| First visit, cached prefix | 0.0018 | 1 | 0.0055 | 1 | 0.0014 | 1 |
| *Anchor (measured): Unify lean-all A1, all visits* | *0.0039* | | *≈ 0.0077 (AppWorld LOW, 12 tasks)* | | | |
| *Anchor (measured): A1 return USD ÷ the same jobs' first visits* | | *0.74–0.97* | | | | |
| k-th return with a **ready offer** (value) or **verified cell**: 1–2 calls | 0.0003–0.0006 | **0.12–0.25** | 0.0018 (2–3 calls; the rules block is still sent) | **0.2–0.3** | never | — |
| k-th return with **references only** (paraphrase or look-alike): about 4 calls, plus about 600 tokens of code per call | 0.0017 | **≈ 0.7** | 0.0070 | ≈ 0.75 | 0.0022 | **≈ 1.05–1.1** |
| The visit when a procedure has **gone stale** (it fails on today's data and is shown with its error), plus the replay CPU | 0.0026–0.0030 | **1.1–1.2** | 0.010–0.011 | 1.1 | — | — |
| The return after a stale visit: the new version is offered | as a ready offer | 0.12–0.25 | as a cell | 0.2–0.3 | — | — |
| A smell hold, added to any visit | +1 call ≈ +0.0004 | +15% | +0.0008 | +10% | +0.0004 | +20% |

Replay verification before reuse costs no money. It is CPU time: at most 2 candidates, at most 2 recorded cases
each, plus one pre-run, which is at most 6 sandbox runs of about 0.1–1 s each. In AppWorld it never touches the
live world.

**Expected return cost** is ρ·R_offer + (1 − ρ)·R_ref, plus σ·(R_stale − that). Here ρ is the share of returns
that get a ready offer or cell, and σ is the staleness rate.

| Benchmark | ρ | Expected return ratio | Basis |
|---|---|---|---|
| **Office core** | probably low, about 0.1–0.3 | **about 0.55–0.65** | The 7 returns are rewordings over new data. The dry run's `ofc-r01` could not bind, so it got a reference. |
| **AppWorld train canary** | possibly high | **0.3–0.5 if ρ ≈ 0.7** | Variants are generated from one template with new values. Not yet measured: the AppWorld logs on this laptop hide the scenario behind task aliases. The fit-rate analysis on the first paid cells will measure it. |
| **ARC** | **0, measured** | **about 1.05** | Of 36 same-task pairs in Unify's logged first turns (run `arc-ufix4-h-low-s0`), 0 could ever give a ready offer: every pair differs in single-digit grid cells, which are never parameters. A generous lifting binds 4 of 36, but none has the same request shape. References only add tokens. (`analysis/binding_probe.py`; prediction confirmed.) |

**Cost per completed task.** Suppose return accuracy equals first-visit accuracy *a*. Then cost per completed
task over a stream with a share *r* of returns is about F · (1 − r + r·ratio) / a.

| Stream | r | Cost per completed task | Compared with "no memory" (ratio 1) |
|---|---|---|---|
| Office core | 7/24 | 0.0024 · (0.71 + 0.29 · 0.6) / a ≈ 0.0021 / a | about 12% lower |
| AppWorld canary | 8/12 | 0.0096 · (0.33 + 0.67 · 0.4) / a ≈ 0.0058 / a | about 40% lower, if ρ holds |
| ARC | — | about 0.0021 / a | about 5% **higher**, from reference tokens |

On ARC the remaining levers are accuracy *a*, and not showing references whose lineage has never fitted.

## 4. The two biggest remaining cost sinks

1. **Returns that cannot bind.** These are paraphrased office returns and every ARC return. They get references
   and pay 70–110% of a first visit, which is the dominant term everywhere except template-generated streams.
   - MEMORY's capture-v1 found the same on Unify: lifted captures fit the next ARC visit 6/59 times.
   - The levers, in order of evidence needed:
     1. Stop showing references whose lineage has a measured fit rate of 0. This is free: it removes the ARC
        cost increase.
     2. A cheap yes/no applicability judge on references, which is allowed only after offline precision of at
        least 90% and the lead's re-approval of harness-side previews.
     3. Procedures stored as functions with explicit inputs (for example "the file, the filters"), not
        single-instance scripts. That is the "functions-first" direction, and it needs its own screen.
2. **Re-sent context on exploration-heavy first visits.** In AppWorld, most calls of a first visit run at the
   6,000-token compaction ceiling. The 1,300-token rules block and the API-documentation observations are
   re-sent on every call, and input tokens are about 60% of the bill.
   - The levers:
     1. Confirm and keep prompt caching. The design keeps the prefix stable; with caching this is about 40%
        cheaper.
     2. A smaller output cap for documentation calls.
     3. Data notes keyed by resource (design.md §3.1), so that what one first visit learned about an API's
        shape is a few lines on the next first visit of a *different* job using the same API. This is the only
        lever here that helps first visits themselves.

Smaller sinks: smell holds (one extra call each), and prose replies that the harness treats as typed-in answers.

## 5. How the paid cells will check this

The office analysis (`analysis/office_analysis.py`) already reports, per run:

- return USD divided by the same jobs' first-visit USD (the curve's main point);
- offers made and accepted, and references shown (ρ);
- the fit rate;
- hold costs.

AppWorld uses the same analysis once its cells exist. A per-visit-index cost plot of USD against stream
position, for each job, is the figure to add when the first paid stream lands.
