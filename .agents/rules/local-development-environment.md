---
description: Instructions for local development, test execution, and environment management
---

# Local Development Environment

## Python Interpreter
- **ALWAYS** use the project's virtual environment interpreter: `.venv/bin/python`.
- Do not use global python or other system interpreters.

## Environment Bootstrap (fresh clone / Cloud Agents)
- The repo virtualenv lives at **`.venv/`** and is intentionally not committed.
- If `.venv/` is missing (common in fresh clones and Cursor Cloud Agents), bootstrap it with:
  - `pip install uv && uv sync --all-groups`
- `tests/parallel_run.sh` will also auto-bootstrap `.venv/` (and install `uv` via `pip --user` if needed).
- Prefer `python3` over `python` in shell scripts; some environments don't provide a `python` shim.

## Running Tests

### Terminal Isolation (Automatic)
Each terminal session (including each A coding agent) automatically gets its own **isolated tmux server**. This means:
- Your tests don't interfere with other agents' tests
- `tmux kill-server` only affects YOUR terminal's sessions
- No configuration needed - it's automatic

### Choosing the Right Command

The script **always blocks** until all tests complete (or timeout), streaming pass/fail results inline as tests finish.

| Scenario | Command |
|----------|---------|
| **Default** | `tests/parallel_run.sh [path]` |
| **Serial mode** (one session per file) | `tests/parallel_run.sh -s [path]` |
| **With timeout** | `tests/parallel_run.sh --timeout 300 [path]` |

- The script blocks until all tests complete, then reports success (exit 0), failure (exit 1), or timeout (exit 2).
- `--timeout N` aborts if tests don't complete within N seconds.

#### Parallelism Behavior

- By default: One tmux session per *test*. All tests run concurrently (maximum speed).
- With `-s`: One tmux session per *file*. Tests within a file run serially.
- `-s` runs all of a file's tests in one process, each on its own event loop (`asyncio_default_test_loop_scope = function` in `pytest.ini`). That exposes module-level asyncio state (a queue, lock or task) left bound to an earlier test's loop, which the per-test default hides. Such state needs the dead-loop guard that `_adopt_running_loop` applies in `unify/conversation_manager/domains/managers_utils.py`.

**Examples:**
```bash
# Single test file with multiple tests (default: runs all tests concurrently)
tests/parallel_run.sh tests/function_manager/storage/test_primitives.py

# Specific test functions
tests/parallel_run.sh tests/test_foo.py::test_one tests/test_bar.py::test_two

# Small directory
tests/parallel_run.sh tests/actor/

# Large test suite with serial mode (one session per file, fewer total sessions)
tests/parallel_run.sh -s tests/

# With timeout (abort after 5 minutes)
tests/parallel_run.sh --timeout 300 tests/function_manager/
```

### Failure Handling
- If the script exits with code 1, failures were detected.
- Do **NOT** inspect `tmux` panes directly.
- **ALWAYS** read the corresponding log file in `logs/pytest/` for the failed session.
- A test that fails only inside a large parallel run can be a load flake. Re-run it on its own before debugging it.
- When every model-reaching test fails at once with `APIError(status=403)` and `Key limit exceeded`, the OpenRouter key has hit its spending limit. The code is not at fault.

### Log Directory Naming
Log directories use a **datetime-prefixed format** for natural time-based ordering in the filesystem:
- Format: `YYYY-MM-DDTHH-MM-SS_{socket_name}` (e.g., `2025-12-05T14-30-45_unity_dev_ttys042`)
- The datetime is when the test run started
- The socket name identifies the terminal session (for isolation)

**Finding your logs:**
- The script prints the log directory path when tests start
- Directories are sorted chronologically, so recent runs appear at the bottom of `ls` output
- Each run gets its own directory, even from the same terminal

**Example directory listing:**
```
logs/pytest/
├── 2025-12-05T09-15-22_unity_dev_ttys004/
├── 2025-12-05T10-30-45_unity_dev_ttys026/
├── 2025-12-05T14-22-18_unity_dev_ttys004/
└── 2025-12-05T15-00-00_unity_dev_ttys042/
```

**Environment variables:**
- `UNIFY_LOG_SUBDIR`: The full datetime-prefixed log directory name (set by `parallel_run.sh`)
- `UNIFY_TEST_SOCKET`: The terminal socket name for tmux isolation (e.g., `unity_dev_ttys004`)

### Cleanup (REQUIRED)
- **ALWAYS** kill failed tmux sessions after extracting failure info from `logs/pytest/`.
- Logs are persisted in `logs/pytest/`; keeping sessions open is unnecessary.
- Run: `tests/kill_failed.sh` to kill all failed sessions from YOUR terminal.
- Run: `tests/kill_server.sh` to kill the entire tmux server for YOUR terminal.
- For cross-terminal cleanup: `tests/kill_failed.sh --all` or `tests/kill_server.sh --all`

Every session owns a pseudo-terminal. A passing session closes itself after ten seconds, but a failed one keeps its shell open until it is killed. macOS caps pseudo-terminals at `kern.tty.ptmx_max` (511 by default), so failed sessions left over from earlier runs, in any terminal, eventually make tmux fail with `create window failed: fork failed: Device not configured`. The error names whichever test was starting, so it reads like that test's failure. Check `ls /dev/ttys* | wc -l` before a big run; `tests/kill_failed.sh --all` frees what failed sessions hold without stopping anyone's running tests.

### Permissions
- Use `required_permissions: ['all']` to ensure access to `.env` and log files.

## Pre-commit Hooks
- The `pre-commit` tool is installed in the project `dev` dependencies.
- **Execution**: Run via the python module to ensure path visibility:
  - `.venv/bin/python -m pre_commit run --all-files`
- **When to run**: run pre-commit *before* committing so the hooks never surprise you.
- On newly wrapped code, `black` and `add-trailing-comma` each rewrite the other's output once, so the hooks can fail twice before they pass. Re-stage and run them again until they pass; never bypass them.

## Dependencies
- This project uses `uv` for dependency management.
- Config file: `pyproject.toml`

## Edit Safety
- **Protected Files**: Do not edit `uv.lock` manually. Use `uv` to change dependencies.
- **Sensitive Files**: Do not output the contents of `.env` or `*.key` files to the chat.
