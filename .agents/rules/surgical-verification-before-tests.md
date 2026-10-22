---
description: Write quick verification scripts before running slow tests
---

# Surgical Verification Before Running Tests

When fixing infrastructure issues (especially concurrency, race conditions, or fixture problems), consider writing a quick verification script before running the full test suite.

## Why This Matters

Tests in this repo have significant overhead:
- Sandbox start-up (every pytest process runs inside bubblewrap, and each cell's interpreter is a confined worker)
- Fixture set-up (homes, run records, memory homes, scripted models)

A targeted verification script can validate a fix in **seconds** rather than waiting minutes for tests that may not even reliably reproduce the issue.

## When to Use This Pattern

✅ **Good candidates:**
- Race conditions and concurrency fixes (can simulate parallel execution directly)
- Fixture/conftest changes (test the set-up logic in isolation)
- Infrastructure plumbing (context management, caching, file locking)
- Non-deterministic failures that are hard to reproduce reliably in tests

❌ **Skip this for:**
- Simple bug fixes where one test runs quickly enough
- Prompt changes (a scripted-model test of the prompt's text is the check)
- Cases where the verification script would be more complex than the test itself

## How to Write a Verification Script

1. **Name it clearly**: Use `tests/_verify_*.py` (underscore prefix signals temporary/debug)
2. **Keep it self-contained**: Minimal dependencies, no pytest fixtures
3. **Simulate the failure condition directly**: e.g., spawn threads to test race conditions
4. **Run fast**: Should complete in seconds, not minutes
5. **Print clear pass/fail output**: Make success/failure obvious

## Example Sequence

```bash
# 1. Implement the fix

# 2. Write quick verification script
#    tests/_verify_race_fix.py - spawns 5 threads, verifies no duplicates

# 3. Run verification (seconds)
.venv/bin/python tests/_verify_race_fix.py

# 4. If verification passes, run actual test(s)
tests/parallel_run.sh tests/path/to/relevant_test.py

# 5. Delete verification script (it served its purpose)
rm tests/_verify_race_fix.py
```

## Key Principle

This pattern **complements** running actual tests—it doesn't replace them. The goal is faster feedback during development, not skipping proper test validation.
