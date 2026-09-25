---
description: Test Philosophy – Symbolic vs Eval Spectrum
---

# Test Philosophy

## Real LLMs, Never Mocked

We *never* stub the LLM client. Tests always use a real LLM via `unillm.AsyncUnify`. Responses are cached per unique input, so repeated runs are fast on CI. The LLM responses become "locked in" after the first successful run.

## Trigger-Based Synchronization, Not Sleeps

Never rely on sleeps to align events in tests. Always use the trigger helpers in `tests/async_helpers.py` to ensure each event occurs in the necessary order. This makes tests robust to significant timing differences between cached responses (milliseconds) and live LLM calls (up to a minute).

Anchor a trigger on something that happens while a tool runs, not before or during an LLM call. Tools run for real in both a live run and a cached replay, so only LLM latency differs between the two: an event keyed to an LLM call lands at a different point on replay, and one that cancels a call mid-flight leaves no recording for the replay to hit.

## Symbolic ↔ Eval Spectrum

Tests fall on a **spectrum** between two paradigms. Understanding where a test sits on this spectrum is essential for writing, debugging, and interpreting test results.

### Symbolic Tests (Infrastructure-Focused)

At one end, **symbolic tests** use the LLM purely as a stub. The LLM receives minimal "dummy" instructions designed to trigger specific code paths.

**Characteristics:**
- LLM behavior is deterministic and predictable
- Focus is on testing *infrastructure*: async tool loops, steering (pause/resume/interject/stop), state mutations
- The LLM's "intelligence" is irrelevant—we just need it to call the right tools in the right order
- **Failures indicate regressions in symbolic/programmatic logic**

### Eval Tests (Capability-Focused)

At the other end, **eval tests** exercise the system end-to-end. We ask a high-level question or give a directive, then verify the outcome—regardless of internal tool calls.

**Characteristics:**
- Focus is on *capability*: "Did the assistant correctly answer the question?" or "Did it complete the task?"
- Internal implementation details (tool call order, number of steps) don't matter
- Tests LLM reasoning and decision-making in realistic scenarios
- **Failures may indicate prompt issues, tool design problems, or capability gaps**

### The Spectrum (Not Binary)

Most tests sit **somewhere between** these extremes:
- Realistic prompts but only verify specific tool calls were made
- End-to-end behavior but with constrained, predictable inputs
- Combine symbolic infrastructure checks with high-level outcome assertions

Think of each test as having a "slider" between symbolic and eval—not a binary classification.

## Caching and Determinism (`UNILLM_CACHE`)

When `UNILLM_CACHE="true"` (the default), all LLM responses are cached:

1. **First run**: LLM executes normally; responses stored in cache
2. **Subsequent runs**: Cached responses replayed—no actual LLM calls

**Implications:**
- Both symbolic and eval tests become deterministic once the cache is populated
- After caching, both test types effectively verify that *symbolic logic has not regressed*
- Tests run fast on CI (milliseconds vs seconds/minutes for real LLM calls)
- To re-evaluate LLM behavior: delete `.cache.ndjson`, set `UNILLM_CACHE="false"`, or use `--no-cache`

A test replays only if its LLM input is identical on every run, so nothing that varies between runs (a random or clock-derived value, rows read without an `ORDER BY`) may reach a prompt, a tool argument or a tool result. Canonical keying (`UNILLM_CACHE_KEYING=canonical`; the default is `exact`) scrubs ISO timestamps, UUIDs, hex runs of 32 or more characters and `/tmp` paths, but never plain numbers or shorter hex.

### The Cache Is Never the Problem

"We just need to update the cache" is **never** a valid conclusion when debugging test failures. The cache is a faithful replay mechanism, not a source of bugs.

Cache keys are the exact LLM input. A cache hit means the code is sending **byte-for-byte identical** input to a previous run. If you modify prompts, tool docstrings, or system messages, the cache key changes automatically and you get fresh inference.

If a cached response causes a failure, an LLM **actually made that decision** given that exact input. This is a prompt issue, not a stale cache issue. Clearing the cache just re-runs inference on the same input—likely producing the same flawed output.

Only clear the cache to **re-evaluate** LLM behavior after prompt changes (`--no-cache`), not as a fix for failing tests.

## Tagging Tests as Eval

Mark a test file as eval with a module-level marker:

```python
import pytest

pytestmark = pytest.mark.eval  # All tests in this file are eval tests
```

For mixed files, use test-level markers:

```python
@pytest.mark.eval
@pytest.mark.asyncio
async def test_natural_language_query():
    ...
```

## Running Test Categories

```bash
# Run only eval tests
tests/parallel_run.sh --eval-only tests

# Run only symbolic tests
tests/parallel_run.sh --symbolic-only tests

# Re-evaluate LLM behavior (fresh calls, no cache)
tests/parallel_run.sh --no-cache tests

# Statistical sampling for eval reliability
tests/parallel_run.sh --no-cache --repeat 10 --eval-only tests/function_manager/
```

## When Debugging Failing Tests

1. **Determine test type**: Is it symbolic (infrastructure) or eval (capability)?
2. **For symbolic failures**: Logic bug or regression in programmatic code—debug the infrastructure
3. **For eval failures (cached)**: The cached LLM path broke—likely a code change affected tool behavior
4. **For eval failures (uncached)**: LLM reasoning changed—may need prompt tuning, tool docstring improvements, or acceptance of variance
