<!--
    GENERATED FILE - DO NOT EDIT DIRECTLY.

    Regenerate from the repo's root with build_agents_md.py from
    unifyai/global-agent-rules: python3 ../global-agent-rules/build_agents_md.py
    where it is a clone beside the repo, or
    python3 .agents/global-rules/build_agents_md.py where it is a submodule.

    Edit the sources instead:
      .agents/repo.md       this repo's overview and always-on guidance
      .agents/rules/*.md    this repo's own rules
      .agents/shared.txt    which shared rules this repo includes
      rules/*.md in unifyai/global-agent-rules, the rules shared across
      all unifyai repos
-->

# Unify

A code-acting agent harness: one actor answers a request by writing Python that runs in a sandbox, and a memory
turns finished requests into a tested library that later requests read.

Read [`ARCHITECTURE.md`](ARCHITECTURE.md) first for the system design, [`docs/memory.md`](docs/memory.md) for the
memory and [`CONFIG.md`](CONFIG.md) for every setting. This file covers *how to work on the code*, not *what the code
does*.

## What Unify is

- **`unify act`** (`unify/cli.py`) runs one request, or a session (`--persist`, `--jsonl`), through one
  **`CodeActActor`** (`unify/actor/`).
- **Cells.** The actor's only model-API tool is `execute_code(code)`. Each call is one cell in a persistent Python
  interpreter inside bubblewrap (`unify/common/box_interpreter.py`, `unify/sandbox.py`). Every other tool is a call
  `tools.<name>(...)` from a cell, served on the host through the tool channel from one registry
  (`unify/common/tool_registry.py`). An embedding application adds its own `primitives.<name>` namespaces
  (`unify/function_manager/primitives/`, `unify/actor/environments/`).
- **The run record** (`unify/agents/`). One append-only record per run that the actor, its teammates, the requester,
  the librarian and the harness post to. Its own tools are `send`, `messages`, `wait` and `who`, the same for every
  agent (`unify/agents/run_record_tools.py`).
- **Memory** (`unify/memory/`). One switch, `UNIFY_MEMORY`: `off` or `on` (the default). With memory
  on, each request is recorded as an episode, a fork of the actor's conversation captures what it did (a capture is
  never run), a detached writer service keeps the library under a gate that labels and only refuses for safety, and
  later requests read the library as the `memory` package and ask the librarian (`memory.ask`).
- **The lanes and the relay** (`unify/lanes/`). With memory on and `OPENROUTER_API_KEY` set, `unify act` starts a
  local relay that holds the key and runs the controller as its child. The writer, its helpers, the episode-end fork
  and embeddings each get their own lane and token under one spend cap, `UNIFY_MEMORY_CAP_USD` (20.00 USD per run by
  default); memory's processes never hold the key.
- **State** lives under `UNIFY_HOME` (default `~/.unify`): the workspace, transcripts, the run record, the memory
  home and the logs (`unify/db.py`'s `store_home`). There is no backend service and no database server; the only
  external dependency is the model provider, reached through the sibling `unillm` repo.

## Run the agent locally

```bash
uv sync --all-groups
cp .env.example .env        # add one LLM provider key
.venv/bin/unify act "request"   # `act` is the default; with no request it reads stdin
```

- `unify --debug` streams runtime logs to the terminal as well as the log files; `unify --home DIR` uses another home.
- `unify act` options: `--json` (the result as a JSON object, with token accounting), `--jsonl` (newline-delimited
  JSON on stdin and stdout, for a driving program), `--persist` (further lines are follow-ups), `--no-clarify`
  (unattended: questions get no reply), `--timeout SECONDS` and `--quiet`. Memory has no command-line flag; it is
  `UNIFY_MEMORY`.

## Development environment

```bash
# First-time setup (fresh clone)
pip install uv && uv sync --all-groups
```

- **Python interpreter:** always use `.venv/bin/python`. Never the system Python.
- **Bootstrap:** if `.venv/` is missing, `uv sync --all-groups` recreates it.
- **`uv.lock` is protected** — never hand-edit it. Use the package manager.
- **Do not output `.env` or `*.key` contents to chat.**

### Running tests

Tests run in tmux sessions, with logs streamed to `logs/pytest/`. The runner blocks until everything completes.

```bash
# Default: one session per test, max concurrency
tests/parallel_run.sh tests/actor/

# Specific test
tests/parallel_run.sh tests/test_cli_act.py::test_name

# Serial mode (one session per file) for large suites
tests/parallel_run.sh -s tests/

# With timeout
tests/parallel_run.sh --timeout 300 tests/memory/
```

Each agent (or terminal) gets an **isolated tmux server automatically**, so concurrent agents don't collide. Every
pytest process runs inside bubblewrap, so the tests need it too.

### When a test fails

1. **Never inspect tmux panes directly.** Read the corresponding log in `logs/pytest/<YYYY-MM-DDTHH-MM-SS_socket>/`.
2. **Use `Read` (not `cat`/`tail`)**: `logs/` is gitignored, so `Grep`/`Glob` won't find files there.
3. **Temporary debug logging** follows the shared temporary-debug-logging rule below; remove every call before
   finalizing the fix.
4. **Clean up failed sessions** with `tests/kill_failed.sh` (or `tests/kill_server.sh` for everything).

### Pre-commit

Install the hooks once per checkout (they run automatically on commit):

```bash
./scripts/install-git-hooks.sh   # or: pre-commit install
```

Run them manually any time:

```bash
pre-commit run --all-files
```

## Testing philosophy

**Tests always use stubbed models and never reach a real provider.** This is the lead's rule ("we do not add tests
that need a real provider key and always mock those tests with stubbed LLM calls"), and `tests/conftest.py` enforces
it in every test: unillm's provider calls raise, a socket may connect only to this machine, and the markers that once
let a test reach a provider (`live_provider`, `requires_provider_key`) fail the test that carries them. No test needs a
key; with none set, the root conftest sets a placeholder that is plainly not a secret.

- Script the model with `tests/scripted_model.py` (`ScriptedModel`, one script per kind of call: actor, helper,
  `query_llm`, compaction) or `tests/cache_discipline_helpers.py` (`scripted(...)`). Both replace unillm's transport
  below its request building, so every request is built exactly as in production.
- A test fakes embeddings (`unify.common.embeddings.embed`, or the endpoint it posts to); it never posts to a provider.
- **Never rely on sleeps**: use the trigger helpers in `tests/async_helpers.py` for deterministic ordering.
- A failing assertion is fixed at its root cause. Never loosen an assertion, skip, xfail, raise a timeout or add a
  retry to get a pass; security and confinement tests pass as written, or the code gets fixed.

## No fast paths or heuristics

If a component needs to respond correctly to a class of input, address it by prompting the model and/or improving
tool descriptions. Never apply regex-based or substring-based routing on user requests. The model is the router.

## Tools and prompts

Every tool's name, description, parameter schema and refusal texts live in one registry
(`unify/common/tool_registry.py`), for the actor, its teammates and the memory agents. The box module `tools` is
generated from it, and `tools.help(name)` reads it.

- **A tool's own guidance lives in its registry description**, never in a prompt builder.
- **Compositional guidance** (when to use tool A versus B, multi-tool patterns) lives in the prompt builders
  (`unify/actor/prompt_builders.py`, the memory agents' prompts), never in one tool's description.
- **Descriptions are implementation-agnostic**: they say what the tool does for the caller, not which module serves
  it.

### Who owns what

| Concern | Owner |
|---|---|
| Anything the request needs | `execute_code` (a cell of plain Python in the sandbox) |
| The embedding application's own operations | `primitives.<name>` namespaces |
| Installing packages | `tools.install(packages)` |
| Teammates, questions to the requester, messages | the run record's tools (`send`, `messages`, `wait`, `who`) |
| What the library already holds | `memory.ask(...)` (the librarian) and the `memory` package |
| Keeping the library | the writer service (`unify/memory/writer_service.py`), never the actor |

Full role descriptions are in [`.agents/rules/component-roles.md`](.agents/rules/component-roles.md).

## Additional git constraints

Beyond the shared git rules below: never use `git rebase -i` or `git add -i`
(interactive flags don't work in non-interactive shells), and never edit
`git config`.

This code is `unifyai/unify-agent` on GitHub. `unifyai/unify` is a public,
archived snapshot of the earlier hosted product's code, which the essays in
`docs/writeups/` link to.

`unillm` is installed editable from the sibling checkout `../unillm`, so every
run here uses whatever that checkout holds; keep it on `main`. A change here
that needs a unillm change is finished only when that unillm commit is on
unillm's `main` on GitHub: before pushing, check
`git -C ../unillm status -sb` for `[ahead N]`.

## Repo map

```
unify-agent/
├── unify/                   # Main package
│   ├── cli.py               # The command line (`unify act`)
│   ├── actor/               # CodeActActor, its prompts, in-code tools and the worker
│   ├── agents/              # The run record, teammates and their tools
│   ├── common/              # The tool loop, the registry and tool channel, the interpreter, model calls
│   ├── function_manager/    # primitives: the embedding application's namespaces
│   ├── lanes/               # The local lane relay and the launcher that starts it
│   ├── memory/              # Episodes, captures, the writer service, the gate, the librarian
│   ├── sandbox.py           # The bubblewrap confinement
│   ├── environment.py       # The workspace environment (package installs)
│   ├── db.py                # Where state lives (`UNIFY_HOME`)
│   └── settings.py          # Every setting (CONFIG.md lists them)
├── tests/                   # Pytest suite (stubbed models only)
├── scripts/                 # Git hooks and identity checks
├── docs/                    # memory.md and design notes
├── ARCHITECTURE.md          # System design (read first)
├── CONFIG.md                # Every environment variable
├── KNOWN-ISSUES.md          # Evaluated behaviour with its open fixes
├── README.md
└── pyproject.toml
```

## When in doubt

- Check [`.agents/rules/`](.agents/rules/) for fuller context on any topic above.
- `ARCHITECTURE.md` is canonical for design questions.
- Code is canonical when this document and the implementation disagree — open a PR to update this doc.

---

# Repository rules

# Component Roles

Use this to decide which component owns a change and where its jurisdiction ends. ARCHITECTURE.md and docs/memory.md
are canonical for how each one works.

### The CLI (`unify/cli.py`)
- **Role**: `unify act`: one request, or a session (`--persist`, `--jsonl`), through one actor.
- **Owns**: argument parsing, the driver's stdin/stdout contract, and, with memory on, starting the lane relay
  (`unify/lanes/launch.py`) and the episode line.

### The actor (`unify/actor/`)
- **Role**: answers a request by writing Python, one cell per `execute_code` call.
- **Scope**: the tool loop (`unify/common/async_tool_loop.py`), its prompt (`prompt_builders.py`), its in-code tools
  (`in_code_tools.py`, `tool_handlers.py`) and the worker that runs cells in the sandbox (`execution/`).
- **Reaches**: `tools.<name>` (the registry), the embedding application's `primitives.<name>` namespaces, the run
  record, and with memory on the `memory` package and `memory.ask`.

### The run record (`unify/agents/`)
- **Role**: one append-only record per run for the actor, its teammates, the requester, the librarian and the harness.
- **Owns**: `send`, `messages`, `wait` and `who`, the same tools for every agent (`run_record_tools.py`); teammates
  and their lifetimes (`pool.py`).

### Memory (`unify/memory/`)
- **Role**: turns finished requests into a tested library that later requests read. One switch, `UNIFY_MEMORY`
  (`off`, `on`).
- **Parts**: episodes and the request's memory (`integration/request.py`), the episode-end fork and captures
  (`integration/fork.py`, `captures.py`, `capture_trace.py`; a capture is never run), the attention ledger
  (`attention.py`), the writer service and its passes (`writer_service.py`, `writer_pass.py`), the gate (`gate.py`,
  `gate_policy.py`: labels, refusing only for safety, paths, syntax and layout) and the librarian (`librarian.py`).
- **Boundary**: only the writer edits the library; the actor reads a pinned copy.

### The lanes (`unify/lanes/`)
- **Role**: with memory on and `OPENROUTER_API_KEY` set, a local relay that holds the key and gives the writer, its
  helpers, the episode-end fork and embeddings their own lanes and tokens under one cap, `UNIFY_MEMORY_CAP_USD`.
- **Boundary**: the key stays in the relay; memory's processes get only their lane's token, on an inherited
  descriptor.

This project is a code-acting agent harness. One `CodeActActor` answers a request by writing Python: its only model-API tool is `execute_code`, each call is a cell in a persistent interpreter inside bubblewrap, and every other tool is a `tools.<name>(...)` call from a cell, served on the host through one tool registry. The actor, its teammates, the requester and the librarian share one append-only run record per run. With `UNIFY_MEMORY` on (the default), each request is recorded as an episode, a fork of the actor's conversation captures what it did without running it, a detached writer service keeps a tested library under a gate that labels its findings and refuses only for safety, and later requests read that library. Memory's model calls go through a local lane relay that holds the provider key, under one spend cap. There are no "fast paths" or heuristics based on regex or substring detection of the request, no world vocabulary and no task ids in the harness: when a component must respond correctly to a class of input, the fix is a prompt or a tool description that **nudges** the model. No mechanism executes or re-runs a capture or a function to classify, deduplicate, rank or test it.

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
- `-s` runs all of a file's tests in one process, each on its own event loop (`asyncio_default_test_loop_scope = function` in `pytest.ini`). That exposes module-level asyncio state (a queue, lock or task) left bound to an earlier test's loop, which the per-test default hides; such state must be created per loop.

**Examples:**
```bash
# Single test file with multiple tests (default: runs all tests concurrently)
tests/parallel_run.sh tests/function_manager/primitives/test_environment_namespaces.py

# Specific test functions
tests/parallel_run.sh tests/test_foo.py::test_one tests/test_bar.py::test_two

# Small directory
tests/parallel_run.sh tests/actor/

# Large test suite with serial mode (one session per file, fewer total sessions)
tests/parallel_run.sh -s tests/

# With timeout (abort after 5 minutes)
tests/parallel_run.sh --timeout 300 tests/memory/
```

### Failure Handling
- If the script exits with code 1, failures were detected.
- Do **NOT** inspect `tmux` panes directly.
- **ALWAYS** read the corresponding log file in `logs/pytest/` for the failed session.
- A test that fails only inside a large parallel run can be a load flake. Re-run it on its own before debugging it.
- No test reaches a model provider (see `test-philosophy.md`), so a provider error in a test log is a test that escaped its stub: fix the test's script, never by adding a key.

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

## Dependencies
- This project uses `uv` for dependency management.
- Config file: `pyproject.toml`

## Edit Safety
- **Protected Files**: Do not edit `uv.lock` manually. Use `uv` to change dependencies.
- **Sensitive Files**: Do not output the contents of `.env` or `*.key` files to the chat.

# Log Directory Navigation

## Tool Behavior for Logs

The `logs/` directory is gitignored, which affects tool availability:

| Tool | Works? | Notes |
|------|--------|-------|
| **Read** | ✅ Yes | Preferred for reading log file contents |
| **Shell** | ✅ Yes | Use `ls` to explore directory structure |
| **Glob** | ❌ No | Git-aware index excludes gitignored paths |
| **Grep** | ❌ No | Git-aware index excludes gitignored paths; use `rg -uu` from the shell |

## Log Directories

| Directory | Purpose |
|-----------|---------|
| `logs/pytest/` | Test output logs (datetime-prefixed subdirs per run, one file per tmux session) |
| `logs/unillm/` | Raw LLM request/response traces |

`unify act` writes its runtime logs to `UNIFY_LOG_DIR`, by default `<UNIFY_HOME>/logs` (`~/.unify/logs`), not under the repo.

## Practical Steps

**Step 1: Explore with Shell**
```bash
# List log directories (sorted by time, newest last)
ls logs/pytest/

# List contents of a specific run
ls logs/pytest/2025-12-05T14-30-45_unify_dev_ttys042/
```

**Step 2: Read with Read tool**
```
Read: logs/pytest/2025-12-05T14-30-45_unify_dev_ttys042/function_manager-primitives-test_environment_namespaces.txt
```

## Worktree Symlinks

In worktrees, log directories contain a `_root` symlink pointing to the main repository's logs. Use this when looking for logs from tests run in the main repo.

```bash
# List main repo's logs from a worktree
ls logs/pytest/_root/
```

## Example: Debugging a Test Failure

```bash
# 1. Find recent log directories
ls logs/pytest/

# 2. List logs in the most recent run
ls logs/pytest/2025-12-21T16-00-00_unify_dev_ttys042/
```

Then use the Read tool:
```
Read: logs/pytest/2025-12-21T16-00-00_unify_dev_ttys042/function_manager-primitives-test_environment_namespaces.txt
```

# Parallelize Test Analysis

Tests run in tmux sessions and stream results inline. For long test runs, you can start them in the background and analyze early failures while remaining tests complete.

## The Mental Model

When you run `parallel_run.sh`, each test spawns in its own tmux session. The script blocks until all tests complete, streaming pass/fail results inline as tests finish. Log files are written to `logs/pytest/` as each test completes.

For long-running test suites, you can run in the background and analyze failures incrementally:
1. Start tests with `block_until_ms: 0` in the Shell tool call
2. Read log files as they appear in `logs/pytest/`
3. Analyze failures immediately
4. Check back later for final results

## Steps (Background Mode)

```bash
# 1. Start tests in background (use block_until_ms: 0 in Shell tool call)
tests/parallel_run.sh tests/some_module/

# 2. Check what logs exist (tests write here as they complete)
ls logs/pytest/<latest-run-dir>/

# 3. Read and analyze any failures that have appeared
# (Use the Read tool on specific log files)

# 4. Continue reasoning about the failure while tests run
# 5. Check back later for more results if needed
```

## The Principle: Time-to-Solution Over Turns

Prioritize **minimizing time to find a solution** rather than minimizing turns or waiting for "complete" information. Early failures often provide enough signal to begin investigation. You can always check for additional failures later.

This doesn't mean being hasty or sacrificing thoroughness. It means: **don't wait for information you don't need yet**.

## Relationship to Other Rules

- **`surgical-verification-before-tests.md`**: Covers *pre-test* optimization (quick verification scripts)
- **This rule**: Covers *mid-test* optimization (incremental analysis)
- **`log-directory-navigation.md`**: Explains how to read from `logs/pytest/` (use Shell for `ls`, Read tool for file contents)

# Product Vocabulary: One Noun Per Concept

The model reasons in the words we hand it. When two concepts share a word, the model has to guess which one a prompt
means, and the guess is invisible until it calls the wrong tool or reads the wrong thing.

## The canonical nouns

| Concept | Word | Where |
|---|---|---|
| One request and everything the actor did for it | **episode** | `unify/memory/` |
| One `execute_code` call: a program run in the persistent interpreter | **cell** | `unify/actor/` |
| A host-side operation a cell calls as `tools.<name>(...)` | **tool** | `unify/common/tool_registry.py` |
| The embedding application's own operations, `primitives.<name>` | **namespace** (its members: **methods**) | `unify/function_manager/primitives/` |
| The shared, append-only message log of a run | **run record** | `unify/agents/` |
| What the episode-end fork wrote down that an episode did | **capture** | `unify/memory/captures.py` |
| A tested function or note the writer landed | **library item** | `unify/memory/` |
| The agent that keeps the library | **writer** (its unit of work: a **pass**) | `unify/memory/writer_pass.py` |
| The agent that answers what the library holds | **librarian** | `unify/memory/librarian.py` |
| What the gate records about an item instead of refusing it | **label** | `unify/memory/gate.py` |
| A model-call route of its own for one memory agent | **lane** | `unify/lanes/` |

Do not write "workflow" to mean a multi-step anything: it is an episode, a pass or a procedure. Do not call a capture
a "skill", and do not call the library a "store".

## Before naming a new first-class type

1. **Grep first.** `rg -ic "<candidate>" unify/ tests/`. If the word appears in prompts, descriptions or examples
   meaning something else, you have a collision.
2. **Prefer an unused word** over renaming existing prose. An unused word is free; a rename costs a sweep through
   every prompt that uses it.
3. **If you keep the colliding word, do the cleanup in the same change.** A half-renamed codebase teaches the model
   both meanings at once.
4. **Watch for the near-synonym trap.** A new noun must not collide *conceptually* either.
5. **No world vocabulary.** The harness's own words name no benchmark, environment or task, and nothing keys on a
   task id.

## What gets renamed, and what does not

Rename **our** vocabulary: prompts, tool descriptions, example titles, section headers, internal identifiers, test
names and fixtures.

Leave alone:

- **Third-party product names.**
- **The shared `global-agent-rules` submodule.** Edit it in its own repo, never here.

After editing anything under `.agents/`, regenerate the aggregate:
`python3 .agents/global-rules/build_agents_md.py`. Never hand-edit `AGENTS.md`.

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

---

# Situational rules

These are not loaded by default. Read the file when the work matches its description.

- [`.agents/global-rules/rules/shared-agent-context.md`](.agents/global-rules/rules/shared-agent-context.md) — Questions about what the team did or decided in earlier agent sessions (did we set up X, who changed Y, why Z) - every Cursor, Claude Code and Codex transcript is in ~/shared_context
- [`.agents/global-rules/rules/rented-compute.md`](.agents/global-rules/rules/rented-compute.md) — Renting, extending or ending any cloud machine, GPU, volume, disk, IP or database (Vast, RunPod, Nebius, Hetzner, GCP, Lambda, Prime) - every machine has a lease that ends, and you end what your work leaves behind

---

# Shared rules

These apply across every unifyai repo. Edit them in the `unifyai/global-agent-rules`, not here.

# Aggressive Refactoring, No Backward Compatibility

The code is a prototype in rapid development: clean, minimal and correct
matters more than stability or caution.

- **No backward compatibility** unless it is asked for. Change an interface
  and update every call site at once, with no adapters, aliases or optional
  parameters to soften it. A new pattern replaces the old one everywhere.
- **Rewrite, don't staple.** When requirements change, rewrite the code as
  if the new requirements had been the first ones, rather than adding logic
  on top. Delete what is no longer the best way; never comment it out or
  keep it "just in case".
- **No defensive coding.** No try/except except around a specific, expected,
  recoverable error such as a network timeout, and no null or type checks
  the logic does not need. Fail loud and fast.
- **Simplify first.** Before adding a feature, ask whether simplifying the
  code would make it natural, and remove redundant code, unused variables
  and needless abstraction as you go.

# Timeless Comments

Comments and docstrings describe the code as it is now, never its history or
the conversation that produced it.

- Never mark code "new", "updated", "added" or "refactored"; git records that.
- Never cite the chat or the request ("as the user asked"). Give the
  technical why ("retries with backoff because the API drops connections"),
  never the conversational one.
- Say why tricky code exists or how it works, not that it is an
  implementation of something.

If a test is failing, we should **never** add test-specific information or shortcuts to production code as a hack to get the test passing. No details about specific test cases should ever make their way into production code—no special-case branches, no hardcoded values that match test inputs, no conditional logic that only exists to satisfy a test.

All fixes must be fully general and **much** broader than the specific failing test. We do **not** want to overfit production code to a specific set of tests.

# Git Safety

- **Pull before editing.** Run `git pull --rebase origin main` once per repo
  per session before changing files, so you never work on a stale tree.
  After a rejected push and a rebase, re-read any file you are about to edit:
  the rebase changed it on disk.
- **Commit explicit paths:** `git commit <files> -m "…"`, never a bare
  `git commit`. Agents working in parallel share one index, so a bare commit
  takes in whatever another agent staged. A new file needs `git add <file>`
  first, then the same explicit commit.
- **Push to `main` as the work lands,** without being asked:
  `git push origin HEAD:main`, which works from `main` and from a worktree's
  own branch alike. Never push a worktree's branch itself, since nobody
  merges it. If the push is rejected, `git pull --rebase origin main`,
  re-read what you are about to edit, and push again.
- **Never force-push** unless the user asks and understands that it rewrites
  history other people have pulled.
- Start a branch only when a change has to sit apart from `main`, and merge
  and delete it as soon as it lands.

# Worktrees: No Feature Branches, No Pull Requests

A worktree is isolation enough: agents in separate worktrees are
collaborators on `main`. Commit on the branch the worktree has and push it
to `main` (see Git Safety). Never create a feature branch or a pull request,
or suggest one, unless the user asks for it.

# Python Formatting & Pre-commit

Every Python repo that includes this rule enforces formatting with **black**
(plus `isort`/`autoflake` where configured), and CI rejects unformatted code.
A missing local hook or a drifting Black target/Python version is the single
most common avoidable CI failure. This rule keeps local and CI identical so
it stops blocking us — for Cursor, Claude Code, Codex, and humans alike.

## Single source of truth: the locked `lint` group

The formatters are ordinary, locked dependencies — not a version hardcoded in
the pre-commit hook or in CI YAML.

- Each repo declares its formatters in a dedicated **`lint` dependency
  group**, pinned and committed to `uv.lock`. Every first-party Python repo
  is uv-managed with a repo-local `.venv` — there are no poetry repos.
  The `dev` group includes `lint` so a normal sync gives developers everything:
  `[dependency-groups]` → `lint = ["black==X", "isort>=…", "autoflake>=…"]`,
  and `dev = [ …, {include-group = "lint"} ]`.
- **Both** the pre-commit hook and CI run that **same locked** tool via uv —
  never a separate pin:
  - pre-commit hook: `entry: uv run black`, `language: system`
    (no `additional_dependencies`).
  - CI: `uv sync --only-group lint --no-install-project --frozen` then
    `uv run --no-sync black --check .` on **Python 3.12**.
- Pin Black's language target in every Python repo so local Mac Pythons and
  CI 3.12 cannot disagree (Black 26+ defaults toward newer targets):

```toml
[tool.black]
target-version = ["py312"]
```

- Never introduce a second black version or a parallel invocation anywhere
  (CI YAML, Dockerfiles, docs, ad-hoc `pip install black`, hook
  `additional_dependencies`). The locked `lint` group is authoritative; this
  rule deliberately does not restate the number — look it up in the repo's
  `pyproject.toml` / lockfile so guidance can never drift from reality.
- Keep the version **in lockstep across all the Python repos**: a bump is one
  coordinated change per repo (the `lint` pin + lockfile) applied to every
  repo so they don't diverge.

## Committed hooks (required once per clone / worktree)

`.git/hooks/` is a per-checkout artifact. Fresh clones, Cursor/Codex/Claude
worktrees, and cloud agents start with **no** hooks, which is why unformatted
code reaches CI.

Each Python repo commits `.githooks/pre-commit`. Enable it with the shared
helper (idempotent, tool-agnostic):

```bash
python3 .agents/global-rules/ensure_git_hooks.py
```

That sets local `core.hooksPath=.githooks`. Do this before the first commit
in any new clone or worktree. Coding agents (Cursor, Claude Code, Codex)
MUST run it at session start when working in a checkout that has
`.pre-commit-config.yaml`.

`pre-commit install` alone is no longer enough — it writes into `.git/hooks/`,
which worktrees and new clones miss. Prefer `ensure_git_hooks.py`.

## Before you commit (required)

1. Ensure committed hooks are wired (above).
2. Let the hook run on `git commit`, or run it explicitly on what you changed:

```bash
pre-commit run --files <changed-files>   # or: pre-commit run --all-files
```

3. Never bypass hooks: do not use `git commit -n` / `--no-verify`.

On newly wrapped code, `black` and `add-trailing-comma` each rewrite the
other's output once, so the hooks can fail twice before they pass. Re-stage
and run them again until they pass.

## Formatting across multiple repos

When juggling several repos, do not invoke a globally-installed `black` —
versions drift between machines and repos and produce diffs CI rejects.
Always format through the repo's pinned tooling, which uses that repo's
locked version:

```bash
pre-commit run black --all-files          # or: uv run black .
```

## The hook is the gate

Work lands on `main` by direct push, so the committed git hook is what keeps
unformatted code off `main`. CI runs `black` on every push, which shows a
slip at once, but only after it has landed.

## Why this matters

Without committed `.githooks` + `ensure_git_hooks.py`, the first place
formatting is checked is CI — which then burns agent turns on mundane
reformats. Pinning Black's target and CI Python to 3.12 removes the
"works on my Mac, fails in Actions" class of failures.

This rule standardizes how we add temporary debug logging during failing tests and how we clean it up afterwards. ALWAYS use this process in agent sessions WHENEVER A TEST FAILS. This is the ONLY permitted way to address failing tests.

1) Always start with hardcoded, unconditional debug logs
- Add logs immediately, without flags or guards. Do not gate behind environment variables or configuration.
- Use **only** the `CURSOR_DEBUG_LOG` function. No other logging method is permitted.
- **Finding the function**: Search for it with `rg "CURSOR_DEBUG_LOG"` to locate the utility in your project, then import and use it.
  - **Python**: `from <module> import CURSOR_DEBUG_LOG` then `CURSOR_DEBUG_LOG("message", variable)`
  - **JavaScript/TypeScript**: `import { CURSOR_DEBUG_LOG } from "<module>"` then `CURSOR_DEBUG_LOG("message", { variable })`
- Behavior: Prints an entry to stderr/stdout, making it easy to correlate with test runs.

2) Python-specific import discipline
- **Self-contained imports**: Each debug snippet must include ALL its own imports inline (e.g., `import json as _json; import os as _os;`). Never rely on the file's existing imports.
- **Prefixed names**: Use underscore-prefixed aliases (`_json`, `_os`, `_pid`) to avoid shadowing.
- **Region markers**: Wrap in `# #region agent log` / `# #endregion` for easy identification and removal.
- This prevents `NameError` crashes when debug snippets reference modules that aren't imported at that location.

3) Investigation workflow
- Step A: Add targeted debug calls around suspected code paths.
- Step B: Re-run the failing test(s) and inspect the new logs.
- Step C: If you are not 100% certain of the root cause, add more debug entries and repeat.
- Step D: Only when you are 100% confident of the cause, implement a direct fix (with or without keeping some logs briefly for confirmation).
- Step E: The user may repeatedly re-run tests and paste logs; continue iterating until the issue is definitively fixed.

4) Cleanup policy
- After the fix is confirmed, remove all temporary logging.
- Grep to find every occurrence:
  - ripgrep: `rg -n "CURSOR_DEBUG_LOG" -S`
  - grep:    `grep -Rin "CURSOR_DEBUG_LOG" .`
- Delete each call site (and any now-unused imports) before finalizing the fix.

5) Alignment with workspace rules
- No fast paths or heuristics: Logging should not add conditional shortcuts; it merely reports state unconditionally.
- No exception-handling shields: Do not add defensive exception handling (try/catch, try/except) around the logs. Keep failures visible.
- No test details in production prompts: Temporary logs must not leak test-specific information into production prompts or docstrings.
- Rapid evolution: This logging is temporary by design; remove it once the issue is resolved—do not preserve backward compatibility.

6) Intent and scope
- This logging exists solely for interactive debugging in agent sessions.
- The function name `CURSOR_DEBUG_LOG` is intentionally unique and grep-friendly to ensure quick cleanup on request.

# Knowledge Lives in the Repo

An agent's private memory is invisible to the team. Claude Code's
auto-memory, Cursor's memories and any notes file outside the repo are read
only by one person's sessions on one machine: nobody else, and none of their
agents, ever sees them. A fact kept there is lost to the company the moment
it is written, and it drifts from the truth because nobody can correct it.

- **Record what is worth remembering in the repo it concerns, in the same
  change as the work:** a wiki page, a README or docs section, a rule under
  `.agents/rules/`, or a comment beside the code it explains. Company-wide
  facts (people, accounts, legal state, decisions, how a job is run) go in
  `brain`'s wiki.
- **Corrections that teach something count.** When someone tells you what
  the next agent could not work out for itself (how they like a thing done,
  a fact, a limit to respect, a gotcha nobody would guess), write it down
  where the next agent will read it. A correction for a lapse in ordinary
  judgement or effort is not a rule to write: fix the work. Rules that
  restate common sense bury the ones that matter, and teach agents to wait
  for a rule instead of thinking (Dan, 12 October 2026).
- **Commit and push it as the work lands,** so the next agent on any machine
  starts from it.
- **Never write to private memory.** Claude Code's auto-memory is off in
  every repo through `"autoMemoryEnabled": false` in the committed
  `.claude/settings.json`; keep it off.
- **Treat anything found in private memory as unverified.** Check it against
  the repo, move what is still true into the repo, and delete the rest.

Two things never go in the repo: secrets, which live in Secret Manager, and
material a repo keeps out of git on purpose through `.gitignore`, which stays
in its ignored folder.

# Replying to the Team

The people reading an agent's replies act on them, often in another window
while the agent waits.

- **Lead with the answer or the decision.** Background comes after it, or
  goes in a file the reply points to.
- **When someone has to do something, give exact numbered steps:** one action
  per step, with the literal button, menu path or text to paste. When they are
  carrying the steps out as they read, give one step and wait for them.
- **Only the caveat that changes what they do next.** No stream of
  consciousness, no narration of tools or process, and no alternatives nobody
  asked for; they will ask if they want more.

# OpenAI is reached only through OpenRouter

Every OpenAI LLM call in every repo routes through **OpenRouter**, using
`OPENROUTER_API_KEY`. UniLLM does not expose a native OpenAI chat provider, and
the company's own direct OpenAI account is inactive — it answers
`429 billing_not_active` — so a native route is dead in practice as well as
unregistered.

## Canonical endpoint form

```
openai/<model-id>@openrouter      # openai/gpt-5.6-terra@openrouter
```

Never `<model-id>@openai`. `@openrouter` resolves dynamically through the
OpenRouter catalog; `@openai` is not registered and fails endpoint resolution.

Source, defaults, examples, and tests use the canonical OpenRouter form
directly.

## Hard rules

- New LLM call sites use `openai/<id>@openrouter`. Non-OpenAI providers
  (Anthropic, Google, …) are unaffected by this rule and keep their own routing.
- Never use `OPENAI_API_KEY` or a native `openai.OpenAI()` client for LLM chat
  or text generation. Non-chat integrations such as speech-to-text, realtime
  voice, or embeddings may use that key when the call site documents its
  distinct purpose — that key is the deployment's own (self-host / BYOK), not
  the company account above.
- Env defaults and `.env.example` entries carry the `@openrouter` form, so a
  fresh checkout cannot inherit a dead route.
- Treat any surviving `@openai` model string as a bug; UniLLM rejects it.

## The one legitimate direct-OpenAI path

Masked image edits (`images.edit` with `gpt-image-2`) have no OpenRouter
equivalent — OpenRouter's unified Image API does not expose the mask parameter.
That path may use a separately-named credential (`OPENAI_DIRECT_API_KEY`), must
never fall back to reading `OPENAI_API_KEY`, and must degrade loudly when the
credential is absent. It is the only exception; adding another needs an
explicit reason, not convenience.
