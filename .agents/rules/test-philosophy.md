---
description: Test Philosophy: stubbed models only, deterministic ordering, root-cause fixes
---

# Test Philosophy

## Stubbed Models, Never Real Providers

Tests always use stubbed models and never reach a real provider: "we do not add tests that need a real provider key
and always mock those tests with stubbed LLM calls" (the lead, 13 Oct 2026). `tests/conftest.py` enforces it in every
test, with or without a key loaded:

- unillm's provider calls (`acompletion`, `completion`, `aembedding`, `embedding`) raise;
- a socket may connect only to this machine;
- a test carrying `live_provider` or `requires_provider_key` fails.

Script the model instead:

- `tests/scripted_model.py`: `ScriptedModel`, one script per kind of call (actor, helper, `query_llm`, compaction),
  classified from the request alone. A kind the test did not script fails the test, naming the kind.
- `tests/cache_discipline_helpers.py`: `scripted(...)`.

Both replace unillm's transport below its request building, so every request is built exactly as in production and
nothing leaves the process. A streamed request gets LiteLLM's own mock stream of the scripted reply
(`scripted_model.as_requested`). Embeddings are faked at `unify.common.embeddings.embed` or at the endpoint it posts
to.

## Trigger-Based Synchronization, Not Sleeps

Never rely on sleeps to align events in tests. Use the trigger helpers in `tests/async_helpers.py` so each event
occurs in the order the test needs. Anchor a trigger on something that happens while a tool runs, not on wall-clock
time.

## Fix the Cause, Keep the Assertion

A failing test is fixed at its root cause: a real code bug is fixed in the code; a test of behaviour that changed is
migrated with every assertion intact; a test is deleted only when its subject was removed, with the reason recorded.
Never loosen an assertion, skip, xfail, raise a timeout or add a retry to get a pass. Security and confinement tests
pass as written, or the code gets fixed. The only skips are environment-conditional ones (an optional tool or extra
that is not installed).

## When Debugging Failing Tests

1. Read the test's log under `logs/pytest/<run>/` (see `log-directory-navigation.md`).
2. A scripted test is deterministic: the same failure on a re-run is a code or test bug, never a model's whim.
3. A test that fails only inside a large parallel run can be a load flake. Re-run it on its own before debugging it.
