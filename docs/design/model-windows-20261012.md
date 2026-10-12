# Context windows for the OpenRouter models the harness runs (12 Oct 2026)

## Why

The compaction trigger reads the model's context window: it fires when a call's prompt reaches a share of
that window (0.7 when this was written; a rule with a fixed size as well is being introduced), and the
writer's end event reports the window as `context_window`. The window comes from
`unillm.get_max_input_tokens(endpoint)`, which returned None for `openai/gpt-6-sol@openrouter` (the memory
writer and curator) and `openai/gpt-6-luna@openrouter` (the helpers and the agent loop) with the pinned
unillm on the test worker (the control run below). With no window, no compaction ever fired. unillm now
registers the window for both, so compaction can trigger for the agent loop and for every memory agent. The memory agents still compact only with
`UNIFY_MEMORY_LIBRARY_COMPACT=on` (off by default).

## Values

| Model (OpenRouter id) | `context_length` (public list, 12 Oct) | `context_length` (pinned snapshot) | Registered `max_input_tokens` |
| --- | --- | --- | --- |
| `openai/gpt-6-sol` | 1,050,000 | 1,050,000 | 1,050,000 |
| `openai/gpt-6-luna` | 1,050,000 | 1,050,000 | 1,050,000 |

Both sources agree. `top_provider.max_completion_tokens` in the public list is 128,000 for both.

## Sources

1. OpenRouter's public model list, `https://openrouter.ai/api/v1/models`, fetched without a key on
   2026-10-12 at 18:53:49Z (HTTP 200). The saved response (769,471 bytes) has sha256
   `4e80515b15d51d4e3632f8e38c9c473d851fd215077b04e8025caf640c7db517`. It was saved outside Git, in the
   controller's session scratch directory.
2. The pinned catalogue snapshot the team cites by its sha256 prefix `cbb212a9`:
   `continual-harness-research/artifacts/research-regression-diagnosis-20261001/arc-consolidated-package-preparation-v1/public-catalogue-v1/openrouter_models.json`,
   sha256 `cbb212a90a75ba8b7326ba14dd15eb3548dde6923846c005c2f71eb22a50beae` (278,932 bytes; its
   `fetched_at` is 2026-09-24T16:06:17Z, source `https://openrouter.ai/api/v1/models`).

## Derivation

`max_input_tokens = context_length`, with no output reserve subtracted. This follows unillm's own
convention: every OpenRouter entry in `unillm/endpoints/openrouter_overrides.py` uses the catalog's
context length as `max_input_tokens` (for example `openai/gpt-5.6-sol` at 1,050,000), and unillm's catalog
registration (`catalog_pricing_as_litellm_info`) maps `context_length` to `max_input_tokens` directly. The
compaction trigger fires below the full window, which leaves room for the reply.

## How it is registered

Only the window is registered, as a `model@provider` override (`register_model_info("openrouter", ...)`)
that `get_model_info` lays over LiteLLM's entry; `get_max_input_tokens` reads it from there. Nothing else
changes: no pricing, routing, key handling or other model. Costs and request preparation read LiteLLM's
registry, which this override does not touch, so prices and reasoning support still come from the
catalog. The module's full-entry helper (`register_openrouter_model_info` with `_openai_chat`) was not
used: it would also make the ids known to LiteLLM, which stops the catalog from registering their prices
and reasoning support, and it would require hand-written prices.

## unillm commit and deployment

- Commit `8f99d889d16391af7bc1f31446df3b5a96edf593` on branch `harness-cache-windows-20261012`, on top of
  the pinned harness-cache commit `02b08f4`: `unillm/endpoints/openrouter_overrides.py` (the two
  entries) and `tests/test_tokens.py` (the window holds when neither LiteLLM nor the catalog describes
  the id).
- Deployed to the test worker as `~/unillm-worktrees/harness-cache-8f99d889`. The deploy script stopped
  at its checksum step (`sha256sum: .agents/global-rules: Is a directory`: the repository's
  `.agents/global-rules` is a Git submodule, which `git ls-files` lists). By the script's order of steps,
  the clone and checkout had completed; its `SHA256SUMS` is incomplete, it was not made read-only, and no
  SUMS digest was printed (not inspected on the worker).

## Tests

`tests/memory/test_model_windows.py`: unillm returns the window for both models; the real turn built by
`writer_pass.unillm_turn` holds it; through `run_due_passes` the writer's end event reports it as
`context_window`; the helper turn `_helper_turns` builds holds it; a pass whose prompt reaches the
trigger compacts exactly once (the summary call made through the writer's own turn and booked), and one
that stays just below never does. The trigger point is found from `writer_pass.compaction_due` itself, so
these tests do not depend on the rule's share or fixed size.

With the pinned unillm instead of this commit, every test in the file fails: `get_max_input_tokens`
returns None for both models.
