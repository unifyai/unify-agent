# Tests

This directory contains the test suite for Unify. Everything runs locally:
each pytest process opens its own SQLite store (`UNIFY_STORE_PATH`), seeds
the primitive and builtin guidance catalogues into it, and empties the user
tables before each test with `db.clear()`. There is no backend to start, no
credentials beyond an LLM provider key, and no shared state between sessions.

## Table of Contents

- [Quick Start](#quick-start)
- [Tools at a Glance](#tools-at-a-glance)
- [Test Philosophy](#test-philosophy-symbolic--eval-spectrum)
- [Parallel Runner Reference](#parallel-runner-reference)
- [Stores](#stores)
- [Common Workflows](#common-workflows)
- [Worktree Support](#worktree-support)
- [Troubleshooting](#troubleshooting)
- [Requirements](#requirements)
- [Environment Variable Propagation](#environment-variable-propagation)
- [Detailed Documentation](#detailed-documentation)

---

## Quick Start

```bash
# Run all tests in parallel (blocks until completion)
tests/parallel_run.sh tests/

# Run a specific folder
tests/parallel_run.sh tests/function_manager/

# Run with a whole-run timeout
tests/parallel_run.sh --timeout 300 tests/
```

**Optional shell aliases** (for convenience):

```bash
# Add to ~/.zshrc for permanent aliases
source /path/to/unify/tests/shell_init.zsh

# Then use shorter commands
parallel_run tests/
watch_tests
kill_failed
```

---

## Tools at a Glance

| Command | Purpose |
|---------|---------|
| `parallel_run <tests>` | Run tests in parallel tmux sessions |
| `watch_tests` | Monitor test progress in real-time |
| `attach '<name>'` | Attach to a tmux session |
| `list_runs` | List all active test runs across terminals |
| `kill_failed` | Kill all failed sessions |
| `kill_server` | Kill tmux server + purge orphaned processes |
| `monitor_resources` | Launch resource monitoring dashboard |
| `grid_search` | Run tests across setting combinations |

All commands support `--help` for usage details.

---

## Test Philosophy: Symbolic ↔ Eval Spectrum

Tests fall on a spectrum between two paradigms:

**Symbolic Tests** use the LLM purely as a stub—minimal "dummy" instructions trigger specific code paths. Focus is on testing *infrastructure*: async tool loops, steering, state mutations. Failures indicate regressions in symbolic/programmatic logic.

**Eval Tests** exercise the system end-to-end. We ask a high-level question, then verify the outcome—regardless of internal tool calls. Focus is on *capability*: "Did the assistant complete the task?" Failures may indicate prompt issues or capability gaps.

Most tests sit somewhere between these extremes.

### Caching and Determinism

When `UNILLM_CACHE="true"` (the default), all LLM responses are cached:
- **First run**: LLM executes normally; responses stored in `.cache.ndjson`
- **Subsequent runs**: Cached responses replayed—no actual LLM calls

Both test types become deterministic after caching. To re-evaluate LLM behavior:
```bash
parallel_run --no-cache tests/guidance_manager/test_skill_migration.py
```

The cache is keyed on the exact LLM input, so a prompt or docstring change
gets fresh inference on its own. Clearing the cache never fixes a failing
test; it only re-runs the same decision.

### Marking Tests as Eval

```python
import pytest

pytestmark = pytest.mark.eval  # All tests in file are eval

# Or per-test:
@pytest.mark.eval
async def test_natural_language_query():
    ...
```

### `eval` vs `llm_call`

Two independent questions, not two levels of one:

| Marker | Asks | Effect |
|---|---|---|
| `llm_call` | does this test reach a model? | cost/deselection only |
| `eval` | is the model's answer the thing under test? | eval tier, plus cache lookups pinned to `exact` keying |

A test can reach a model and still assert something exact — "did it call the
right tool?" — and that is `llm_call` without `eval`. An eval is scoring the
answer itself, which is why it refuses a canonically-equivalent recording: a
reworded tool description is an ordinary way to change what a model picks, so
a canonical hit would score the trajectory from before the change.

Neither marker implies the other. A test carrying neither is one you are
asserting never reaches a model; if you find a model-reaching test with
neither, that is a missing marker rather than a free test.

### Running by Category

```bash
parallel_run --eval-only tests           # Only eval tests
parallel_run --symbolic-only tests       # Only symbolic tests
parallel_run --deterministic-only tests  # Only tests carrying neither marker
```

---

## Parallel Runner Reference

The script always blocks until all tests complete, streaming pass/fail results inline.

```bash
parallel_run [options] <targets>

# Targeting
parallel_run tests/                              # Directory
parallel_run tests/foo.py                        # File
parallel_run tests/foo.py::test_bar              # Specific test

# Common flags
parallel_run --timeout 300 tests/                # Abort after 5 minutes
parallel_run --session-timeout 600 tests/        # Kill any one session after 10 minutes
parallel_run -s tests/                           # Serial (per-file, not per-test)
parallel_run -j 8 tests/                         # Limit to 8 concurrent
parallel_run --eval-only tests/                  # Only eval tests
parallel_run --symbolic-only tests/              # Only symbolic tests
parallel_run --env KEY=VALUE tests/              # Set environment variable
parallel_run --no-cache tests/                   # Fresh LLM inference
parallel_run --repeat 5 tests/                   # Run each test 5 times

# Pass extra args directly to pytest (after --)
parallel_run tests/ -- -v --tb=short            # Verbose with short tracebacks
parallel_run tests/ -- --pdb                    # Drop into debugger on failure
parallel_run tests/ -- -k 'pattern'             # Filter by test name
```

| Flag | Description |
|------|-------------|
| `-t`, `--timeout N` | Abort if tests don't complete within N seconds (exit 2) |
| `--session-timeout N` | Kill each session's pytest after N seconds (`UNIFY_TEST_SESSION_TIMEOUT` sets the default) |
| `-s`, `--serial` | One session per file (default: one per test) |
| `-j N`, `--jobs N` | Limit concurrent sessions (default: CPU cores) |
| `-m`, `--match PATTERN` | Filter files by glob pattern |
| `--eval-only` | Only `@pytest.mark.eval` tests |
| `--symbolic-only` | Only non-eval tests |
| `--deterministic-only` | Only tests with no model in the loop |
| `--env K=V` | Set environment variable (repeatable) |
| `--no-cache` | Shorthand for `--env UNILLM_CACHE=false` |
| `--repeat N` | Run each test N times |
| `--` | Pass remaining args to pytest |

Exit codes: `0` all passed, `1` something failed, `2` whole-run timeout.

Each run writes to `logs/pytest/<YYYY-MM-DDTHH-MM-SS_socket>/`: one
`<session>.txt` log per session, `duration_summary.txt` with the sorted
duration/cache/cost table, and `stores/` (see below). The runner prints the
directory at the start and end of every run.

---

## Stores

Every session opens the SQLite store named by `UNIFY_STORE_PATH`. The runner
gives each session its own file, `logs/pytest/<run>/stores/<session>.sqlite`,
so sessions never share tables and each store stays on disk next to the
session's log — open it with `sqlite3` to inspect what a test left behind.

Within a session every test starts from an empty store: `tests/conftest.py`
calls `db.clear()` before each test, which deletes the rows of `functions`,
`guidance` and `messages` and restarts their id sequences. The seeded
catalogues (`primitives`, `builtin_guidance`) persist across tests. After a
run the file holds whatever the last test left behind:

```bash
sqlite3 logs/pytest/<run>/stores/<session>.sqlite 'SELECT name FROM functions;'
```

Set `UNIFY_STORE_PATH` yourself (in the environment, in `.env`, or via
`--env`) and the runner honours it instead: every session then shares that
one store, and nothing deletes it afterwards.

```bash
parallel_run --env UNIFY_STORE_PATH=/tmp/shared.sqlite tests/guidance_manager/
```

A bare `pytest` invocation (no runner) gets a per-process store under the
system temp directory, removed when the process exits.

---

## Common Workflows

### Run tests and watch progress

```bash
# Terminal 1: Run tests
parallel_run tests/function_manager/

# Terminal 2: Watch (optional - inline feedback is shown by default)
watch_tests
```

### Debug a failing test

```bash
# Find failing sessions
watch_tests                    # Look for f ❌ prefix

# Attach to see full output
attach 'f ❌ guidance_manager-test_skill_migration'

# Or check the log file
ls logs/pytest/*/             # Find the run directory
cat logs/pytest/2025-12-05T14-30-22_unity_dev_ttys042/guidance_manager-test_skill_migration.txt

# Query the store the session left behind
sqlite3 logs/pytest/2025-12-05T14-30-22_unity_dev_ttys042/stores/guidance_manager-test_skill_migration.sqlite
```

### Clean up after tests

```bash
kill_failed           # Kill failed sessions (keep passing ones)
kill_server           # Kill this terminal's tmux server
kill_server --all     # Kill all unity* tmux servers
kill_server --global  # Kill ALL tmux servers
```

### Run with different settings

```bash
# Fresh LLM calls
parallel_run --no-cache tests/guidance_manager/test_skill_migration.py

# Compare models (grid search)
grid_search --env UNIFY_MODEL="gpt-4o|claude-3" tests/
```

---

## Worktree Support

All test commands **automatically detect the current git repository** and use that repo's scripts. This means:

- Commands work correctly in **git worktrees**
- Tests run against the **current repo's code**, not a hardcoded path
- Logs and stores appear in the **current repo's** `logs/pytest/` directory
- No manual path adjustments needed

**How it works:** When you run `parallel_run`, the shell function checks `git rev-parse --show-toplevel` to find the current repo root, then uses that repo's `tests/parallel_run.sh`. If you're not in a git repo, it falls back to the originally configured path.

Worktrees share the main repo's LLM cache (`UNILLM_CACHE_DIR` resolves to the main checkout), so cache hits carry across worktrees.

### Browsing All Worktree Logs from Main Repo

When tests run from a worktree (via any method - `parallel_run`, direct `pytest`, etc.), **symlinks are automatically created** in the main repo's log directories pointing to each worktree's logs:

```
/Users/you/unify/logs/pytest/
├── 2025-12-05T14-30-45_unity_dev_ttys042/   # main repo's own logs
├── worktree-oty/  →  ~/.cursor/worktrees/unify/oty/logs/pytest/
└── ...
```

This lets you browse **all logs from all worktrees** in one place (the main repo), while each worktree still maintains its own isolated log directories.

**Note:** Symlinks are created by `conftest.py` during pytest session start, so they work regardless of how pytest was invoked.

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `tmux: command not found` | `brew install tmux` |
| High resource usage after tests | `kill_server --global` |
| "error connecting to ... (No such file or directory)" | Socket was deleted; re-run tests |
| Tests not found | Check that path exists and isn't in `EXCLUDE_DIRS` |
| Permission denied | `chmod +x tests/*.sh` |
| `--session-timeout` has no effect on macOS | `brew install coreutils` (provides `timeout`) |
| `create window failed: fork failed: Device not configured` | The machine is out of pseudo-terminals, usually held by failed sessions from earlier runs: `kill_failed --all` |
| Every model-reaching test fails with a 403 `Key limit exceeded` | The OpenRouter key hit its spending limit; the code is not at fault |

---

## Requirements

- **tmux**: `brew install tmux`
- **coreutils** (macOS): `brew install coreutils` — provides `timeout` for the per-session hang guard and helper scripts
- **Python virtualenv**: Repo-local `.venv/` (create/sync via `uv sync --all-groups`)
- **Environment**: Optional `.env` file at repo root for the LLM provider key and other settings

---

## Environment Variable Propagation

Understanding how env vars flow from `.env` to your test process avoids subtle "variable is set but tests don't see it" bugs.

```
.env  ──(sourced)──>  parallel_run.sh  ──(inherited + re-exported)──>  tmux session  ──>  pytest
```

1. `parallel_run.sh` sources the repo-root `.env` file (via `set -a; source .env; set +a`), exporting all variables into its own process.
2. It creates one tmux session per test. Sessions run `bash -c`, so they **inherit the full environment** from `parallel_run.sh` — every variable from `.env` is available.
3. The variables the runner owns (`UNIFY_STORE_PATH`, `UNIFY_TEST_SOCKET`, `UNIFY_LOG_SUBDIR`, the OTel switches, and every `--env`) are re-exported **inside** the session command, after the shell's own init files have run, so a `~/.zshenv` that exports one of them cannot clobber the runner's value.
4. `--env KEY=VALUE` flags override inherited values.

### pydantic-settings (Python side)

Once inside a pytest process, `unify/settings.py` uses pydantic-settings with `env_file=".env"`. This reads the repo-root `.env` **again** at Python import time, so a setting present only in `.env` is still picked up by the `SETTINGS` object.

---

## Detailed Documentation

- **[Logging & Data](../logs/README.md)** — Log directory structure and analyzing test data
