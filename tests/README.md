# Tests

This directory contains the test suite for unify-agent. Everything runs
locally: every model reply is scripted, each test gets a home of its own, and
no test needs a provider key or reaches the network. There is no backend to
start and no shared state between sessions.

## Table of Contents

- [Quick Start](#quick-start)
- [Tools at a Glance](#tools-at-a-glance)
- [What a Test Asserts](#what-a-test-asserts)
- [Parallel Runner Reference](#parallel-runner-reference)
- [Homes](#homes)
- [Sandbox and Time Limits](#sandbox-and-time-limits)
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

## What a Test Asserts

A test that involves a model writes the model's replies itself and asserts
what the harness does with them (the scripted transport turns unillm's
response cache off), so every run of it is the same run.

### No test reaches a model provider or the network

No test needs a provider key (the rule since 13 October 2026): a test
that involves a model scripts the reply (`tests/scripted_model.py`;
`tests/cache_discipline_helpers.py`'s `scripted(...)`; in
`tests/async_tool_loop`, the `loop_model` fixture) and asserts what the
harness does with it. The guard in `tests/conftest.py` holds in every
session, with or without a key loaded: unillm's provider calls
(completion and embedding) fail with an error that says how to script
the model, and a socket may connect only to this machine. A test that
still carries `live_provider` or `requires_provider_key` fails. An eval
whose subject is a live model's answers lives in `model_evals/`, run by
hand with a key (`model_evals/README.md`); the suite never collects it.

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
parallel_run --env KEY=VALUE tests/              # Set environment variable
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
| `--env K=V` | Set environment variable (repeatable) |
| `--repeat N` | Run each test N times |
| `--` | Pass remaining args to pytest |

Exit codes: `0` all passed, `1` something failed, `2` whole-run timeout.

Each run writes to `logs/pytest/<YYYY-MM-DDTHH-MM-SS_socket>/`: one
`<session>.txt` log per session, `duration_summary.txt` with the sorted
duration/cache/cost table. The runner prints the directory at the start and
end of every run.

---

## Homes

Each test also gets a home of its own (`UNIFY_HOME`, which holds the actor's
`workspace/` and the workspace environment) at `unity_test_homes/<hash of the
node id>` in the system temp directory, emptied before the test and removed
after it. The actor's system prompt embeds the workspace path, so the path
depends only on the test, never on the process, and a rerun replays its
recorded LLM responses. A run of the same test that starts while another
holds that home (`--repeat`, another checkout) gets `<hash>-1` instead and
records afresh.

---

## Sandbox and Time Limits

Tests drive real models, and the code a model writes runs inside the pytest
process. So on Linux every pytest process, however it is started (directly or
by `parallel_run.sh`), re-executes itself inside a
[bubblewrap](https://github.com/containers/bubblewrap) sandbox before it imports
anything (`tests/_test_sandbox.py`, called from the root `conftest.py`). Inside:

- the system directories are read-only; the home directory, `/mnt` (the
  Windows drives under WSL), `/run` and `/sys` are absent, apart from the paths
  below; `~/.ssh`, `~/.config` and other credential files are out of reach;
- the checkout is read-only except `logs/` and `.pytest_cache/`; the
  interpreter, editable installs outside the checkout, `uv` and `rg` are
  read-only; the checkout's `.env` is readable; a worktree's git metadata is
  replaced by an empty repository;
- the shared LLM cache (`.cache.ndjson*` in the main checkout of a worktree)
  is writable; everything else in that directory is hidden;
- `/tmp` is private, except `/tmp/unity_test_home` and `/tmp/data-gym-cache`;
- PID, IPC and hostname namespaces are private, and everything a test starts
  dies with the pytest process.

What it does not do: the network namespace is shared (the guard in
`tests/conftest.py` refuses provider calls and connections off this machine),
and the test process can read a provider key in its environment or `.env`.
Model-written code runs in the sandboxed worker, without either.

`UNIFY_TEST_SANDBOX=auto` (default) confines where bubblewrap is installed and
warns elsewhere; `required` refuses to run unconfined; `off` disables it.

Each test phase (setup, call, teardown) has a time limit: its
`@pytest.mark.timeout(N)`, else `UNIFY_TEST_TIMEOUT` seconds (900; `0` means
none). At the limit the test fails; `UNIFY_TEST_TIMEOUT_GRACE` seconds later
(60) the process prints every thread's stack and exits, even if the test's code
never yields (`tests/_test_timeouts.py`). `UNIFY_TEST_TIMEOUTS=off` or `--pdb`
disables both.

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
attach 'f ❌ actor-code_act-test_trimmed_loop_edges-test_2_a_text_reply_is_the_final_answer'

# Or check the log file
ls logs/pytest/*/             # Find the run directory
cat logs/pytest/2025-12-05T14-30-22_unity_dev_ttys042/actor-code_act-test_trimmed_loop_edges-test_2_a_text_reply_is_the_final_answer.txt
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
# Compare settings (grid search)
grid_search --env UNIFY_MAX_TOOL_LOOP_STEPS="50|300" tests/async_tool_loop/
```

---

## Worktree Support

All test commands **automatically detect the current git repository** and use that repo's scripts. This means:

- Commands work correctly in **git worktrees**
- Tests run against the **current repo's code**, not a hardcoded path
- Logs appear in the **current repo's** `logs/pytest/` directory
- No manual path adjustments needed

**How it works:** When you run `parallel_run`, the shell function checks `git rev-parse --show-toplevel` to find the current repo root, then uses that repo's `tests/parallel_run.sh`. If you're not in a git repo, it falls back to the originally configured path.

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
| `tmux: command not found` | `sudo apt install tmux` |
| High resource usage after tests | `kill_server --global` |
| "error connecting to ... (No such file or directory)" | Socket was deleted; re-run tests |
| Tests not found | Check that path exists and isn't in `EXCLUDE_DIRS` |
| Permission denied | `chmod +x tests/*.sh` |
| `create window failed: fork failed: Device not configured` | The machine is out of pseudo-terminals, usually held by failed sessions from earlier runs: `kill_failed --all` |
| `bubblewrap cannot start the test sandbox` | Install bubblewrap and allow unprivileged user namespaces, or set `UNIFY_TEST_SANDBOX=off` to run unconfined |
| A test fails with `Timeout: the call of ...` | It exceeded its `timeout` marker (or `UNIFY_TEST_TIMEOUT`); a `Timeout (h:mm:ss)!` stack dump means it was killed |

---

## Requirements

- **Linux, or Windows with WSL2** (the harness and the tests need bubblewrap)
- **tmux**: `sudo apt install tmux`
- **bubblewrap**: `sudo apt install bubblewrap` — confines each pytest process (see [Sandbox and Time Limits](#sandbox-and-time-limits))
- **Python virtualenv**: Repo-local `.venv/` (create/sync via `uv sync --group dev`)
- **Environment**: Optional `.env` file at repo root for settings; no key is needed

---

## Environment Variable Propagation

Understanding how env vars flow from `.env` to your test process avoids subtle "variable is set but tests don't see it" bugs.

```
.env  ──(sourced)──>  parallel_run.sh  ──(inherited + re-exported)──>  tmux session  ──>  pytest
```

1. `parallel_run.sh` sources the repo-root `.env` file (via `set -a; source .env; set +a`), exporting all variables into its own process.
2. It creates one tmux session per test. Sessions run `bash -c`, so they **inherit the full environment** from `parallel_run.sh` — every variable from `.env` is available.
3. The variables the runner owns (`UNIFY_TEST_SOCKET`, `UNIFY_LOG_SUBDIR` and every `--env`) are re-exported **inside** the session command, after the shell's own init files have run, so a `~/.zshenv` that exports one of them cannot clobber the runner's value.
4. `--env KEY=VALUE` flags override inherited values.

### pydantic-settings (Python side)

Inside a pytest process, `unify/settings.py` reads each setting from the environment the session inherited; it does not read `.env` itself.

---

## Detailed Documentation

- **[Logging & Data](../logs/README.md)** — Log directory structure and analyzing test data
