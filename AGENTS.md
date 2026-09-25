<!--
    GENERATED FILE - DO NOT EDIT DIRECTLY.

    Regenerate with:  python3 .agents/global-rules/build_agents_md.py

    Edit the sources instead:
      .agents/repo.md              this repo's overview and always-on guidance
      .agents/rules/*.md           this repo's own rules
      .agents/shared.txt           which shared rules this repo includes
      .agents/global-rules/rules/  rules shared across all unifyai repos
                                   (submodule: unifyai/global-agent-rules)
-->

# Unify

The brain of a local AI assistant: a persistent conversation loop above a
code-writing actor, two skill libraries, and one SQLite store.

Read [`ARCHITECTURE.md`](ARCHITECTURE.md) first for the system design. This file
covers *how to work on the code*, not *what the code does*.

## What Unify is

Unify implements an AI assistant's brain as a persistent **conversation loop** above a code-writing **`Actor`**, with two **skill libraries** behind it: `FunctionManager` (stored Python functions, the *what*) and `GuidanceManager` (procedures, the *how*). The actor discovers skills before it writes code, runs plans in a persistent Python sandbox, and a storage review after each run distils what worked back into the libraries. Manager methods run inside an **async LLM tool loop** and return a **steerable handle** that supports `ask`, `interject`, `pause`, `resume`, `stop` — all the way down the nesting tree. The skill libraries expose direct CRUD methods as Actor JSON tools (`FunctionManager_*`, `GuidanceManager_*`).

Everything persists in one SQLite file, `unify/db.py` (imported as `from unify import db`): five tables (`functions`, `primitives`, `guidance`, `builtin_guidance`, `messages`) and two views (`all_functions`, `all_guidance`) over each catalogue and its seeded rows. Managers write plain SQL; every filter the model writes is a SQL `WHERE` clause run read-only. There is no backend service, no accounts and no infrastructure: the only external dependency is an LLM provider reached through the sibling `unillm` repo (editable install via `[tool.uv.sources]` in `pyproject.toml`).

The assistant is **reactive**: it acts on the messages it receives and on the work those messages start. There is no scheduler, no timer wheel and no inbound channel other than the in-app chat.

## Run the agent locally

```bash
uv sync --all-groups
cp .env.example .env        # add one LLM provider key
.venv/bin/python -m unify   # chat; `unify` is the same entry point on PATH inside the venv
```

- The runtime's home is `UNIFY_HOME` (default `~/.unify`): `store.sqlite`, `workspace/` (the actor's working directory, attachments and downloads) and `logs/`.
- `unify --debug` streams runtime logs to the terminal as well as the log files; `unify --home DIR` uses another home.
- The chat CLI lives in `unify/cli.py`. It publishes `UnifyMessageReceived` events on the in-memory event broker and renders the `UnifyMessageSent` replies, so it is one front end over the same loop any client can drive.
- `unify act "request"` bypasses the conversation loop: one `CodeActActor` takes the request, progress streams to stderr from the handle's notification queue, a question the actor asks is answered from the terminal through `next_clarification`/`answer_clarification`, and the result goes to stdout (`--json` adds token accounting, `--persist` keeps the sandbox for follow-ups, `--no-store` skips the storage review). Use it for like-for-like comparisons with single-loop harnesses and for benchmark runners.

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

Tests run in tmux sessions, each test in its own session against its own SQLite store, with logs streamed to `logs/pytest/`. The runner blocks until everything completes.

```bash
# Default — one session per test, max concurrency
tests/parallel_run.sh tests/function_manager/

# Specific test
tests/parallel_run.sh tests/function_manager/storage/test_primitives.py::test_name

# Serial mode (one session per file) for large suites
tests/parallel_run.sh -s tests/

# With timeout
tests/parallel_run.sh --timeout 300 tests/function_manager/
```

Each agent (or terminal) gets an **isolated tmux server automatically**, so concurrent agents don't collide.

### When a test fails

1. **Never inspect tmux panes directly.** Read the corresponding log in `logs/pytest/<YYYY-MM-DDTHH-MM-SS_socket>/`.
2. **Use `Read` (not `cat`/`tail`)** — `logs/` is gitignored, so `Grep`/`Glob` won't find files there.
3. **Add temporary debug logs via `CURSOR_DEBUG_LOG`** — the only permitted logging mechanism for debugging. Grep for it (`rg CURSOR_DEBUG_LOG`) to find the project's util, then import and use it. Remove all calls before finalizing the fix.
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

We **never** mock the LLM client. All tests use real LLM calls via `unillm.AsyncUnify`, with responses cached per unique input (`UNILLM_CACHE=true`, the default). First run is slow; subsequent runs replay from cache in milliseconds.

Tests sit on a **spectrum** between two paradigms — there's no binary classification:

- **Symbolic tests** use the LLM as a deterministic stub to exercise infrastructure (async tool loops, steering, state mutations). Failures = regression in programmatic code.
- **Eval tests** verify end-to-end *capability* ("did the assistant answer correctly?"). Failures may indicate prompt issues, tool design problems, or capability gaps.

**Never rely on sleeps** — use the trigger helpers in `tests/async_helpers.py` for deterministic ordering across cached (ms) and live (sec–min) timing.

### The cache is never the problem

"We just need to update the cache" is **never** a valid conclusion when debugging failures. The cache is a faithful replay mechanism keyed on the exact LLM input. If you change prompts or docstrings, the cache key changes automatically and you get fresh inference. If a cached response causes a failure, an LLM *actually made that decision* given that exact input — that's a prompt issue, not a stale-cache issue. Clearing the cache to "fix" a failing test is a category error.

### Tagging eval tests

```python
import pytest
pytestmark = pytest.mark.eval  # whole file

@pytest.mark.eval                # single test
async def test_natural_language_query(): ...
```

## No fast paths or heuristics

If a method needs to respond correctly to a class of user input, **always** address this by prompting the model and/or improving tool docstrings. Never apply regex-based or substring-based routing on user commands. The LLM is the router.

## State manager design

The public API of each state manager is defined by the abstract methods on `Base{SomeManager}` in `base.py`. These docstrings are the **LLM-facing contract** — they're attached to concrete implementations via `@functools.wraps`.

### Docstring rules

- **Implementation-agnostic.** Public docstrings must never reference other managers (cross-references rot) or the manager's own internal tools.
- **Tool-specific guidance lives in the tool's own docstring** — never in the prompt builder.
- **Compositional guidance (when to use tool A vs B, multi-tool patterns) lives in the prompt builder** — never in individual tool docstrings.

### Routing playbook (who owns what)

| Concern | Owner |
|---|---|
| Stored functions (find, run, store) | `FunctionManager_*` JSON tools and `execute_function` |
| Procedures, how-tos | `GuidanceManager_*` JSON tools |
| Anything else the request needs | `execute_code` (plain Python in the sandbox) |
| Parallel or delegated work | `primitives.actor.act` |
| Live action from chat | `Actor.act` (via ConversationManager) |

Full role descriptions are in [`.agents/rules/state-manager-roles.md`](.agents/rules/state-manager-roles.md).

## Additional git constraints

Beyond the shared git rules below: never use `git rebase -i` or `git add -i`
(interactive flags don't work in non-interactive shells), and never edit
`git config`.

## Repo map

```
unify/
├── unify/                   # Main package
│   ├── cli.py               # Terminal chat (`python -m unify`)
│   ├── actor/               # CodeAct Actor, central orchestrator
│   ├── conversation_manager/ # The persistent interaction loop (slow brain)
│   ├── db.py                # The local SQLite store: five tables, two views
│   ├── guidance_manager/    # Procedures, SOPs
│   ├── function_manager/    # Stored Python functions and their dependencies
│   ├── workspace.py         # The assistant's working directory
│   ├── events/              # Typed event bus
│   └── common/              # Async tool loop, shared infra
├── tests/                   # Pytest suite
├── scripts/                 # Skill import, git hooks
├── docs/                    # Design writeups
├── ARCHITECTURE.md          # System design (read first)
├── README.md
├── CONTRIBUTING.md
└── pyproject.toml
```

## When in doubt

- Check [`.agents/rules/`](.agents/rules/) for fuller context on any topic above.
- `ARCHITECTURE.md` is canonical for design questions.
- Code is canonical when this document and the implementation disagree — open a PR to update this doc.

---

# Repository rules

This project is an AI assistant implemented as a back office of cooperating managers, each dealing with one aspect of the assistant's persistent state, and each exposing an English-language public API. The public methods of most managers are asynchronous tool loops: a central LLM handles the English request by orchestrating lower-level tools that read and mutate that manager's rows in the local SQLite store (`unify/db.py`). These methods are dynamic, and return handles for mid-flight steering, question answering, pausing, resuming and stopping. They are also often **nested**: the public API of one manager appears in the tool set of a higher-level manager, and a tool loop can steer its own in-flight inner tools, so steering reaches arbitrary depth. The `Actor` is the central intelligence, orchestrating the managers through code-first plans, and the `ConversationManager` is the persistent interaction loop above it. There are no "fast paths" or heuristics based on regex or substring detection of user input: when a method must respond correctly to a class of input, the fix is always a prompt or a tool docstring that **nudges** the LLM.

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
| `logs/pytest/` | Test output logs (datetime-prefixed subdirs per run, one file per tmux session, one SQLite store per session under `stores/`) |
| `logs/unify/` | Runtime LOGGER output (async tool loop, managers) when `UNIFY_LOG_DIR` points here |
| `logs/unillm/` | Raw LLM request/response traces |

The chat CLI writes its runtime logs under `<UNIFY_HOME>/logs` (default `~/.unify/logs`), not under the repo.

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
Read: logs/pytest/2025-12-05T14-30-45_unify_dev_ttys042/function_manager-storage-test_primitives.txt
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
Read: logs/pytest/2025-12-21T16-00-00_unify_dev_ttys042/function_manager-storage-test_primitives.txt
```

Each session's store file sits next to its log, so a failing test's rows can be inspected with `sqlite3` after the run.

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

The assistant reasons in the words we hand it. When two concepts share a
word, the model has to guess which one a prompt means, and the guess is
invisible until it routes to the wrong manager.

## The canonical nouns

| Concept | Word | Owner |
|---|---|---|
| Written-down multi-step how-to | **procedure** | `GuidanceManager` |
| Executable unit the assistant calls | **function** | `FunctionManager` |
| What the actor writes and runs to satisfy one request | **plan** | `Actor` |
| One dispatched, steerable piece of work | **action** | `ConversationManager` |
| Ordered instructions inside a docstring | **Steps** (section header) | — |

Do not write "workflow" to mean a multi-step anything: it is a procedure,
an action, or a plan. Do not title a docstring section `### Workflow`; use
`Steps` or `Procedure`.

## The three stores, and how to tell them apart

`FunctionManager` and `GuidanceManager` are the two places durable know-how
lives, and they are distinguished by *what kind of thing they hold*, not by
topic:

| | FunctionManager | GuidanceManager |
|---|---|---|
| Role | the **what** | the **how** |
| Holds | one callable | a multi-step procedure |
| Content | executable implementation | natural-language recipe |
| Analogy | a tool's docstring | a prompt that references tools |

Deciding where something belongs:

- **Can it run?** Function. If it is code that executed successfully and
  would be worth calling again, it is a function — not a procedure
  describing the code.
- **Does it tell someone how to act?** Procedure (guidance). Reach for it
  when composing several functions is non-obvious, or when a durable rule
  or policy governs how work is done. A procedure links the functions it
  composes via `function_ids`, which is also how a rule change finds every
  implementation that embeds it.
The common error is storing a procedure that merely restates one
function's docstring. If a single function's docstring already explains
its inputs, behaviour and use, store the function and stop.

Neither store is a dumping ground for "stuff we learned": a fact with no
procedure attached belongs in the procedure that uses it, or nowhere.

**`skill` is an umbrella, not a synonym.** "Skills" means *anything worth
storing across the three stores* — the `store_skills` tool, the
`"Storing reusable skills"` review label. That is a legitimate superset
covering functions and procedures together. It is wrong only when
used for one specific member.

## Before naming a new first-class type

1. **Grep first.** `rg -ic "<candidate>" unify/ tests/`. If the word
   appears in prompts, docstrings or examples meaning something else, you
   have a collision.
2. **Prefer an unused word** over renaming existing prose. An unused word
   is free; a rename costs a sweep plus fresh LLM inference everywhere a
   prompt changed.
3. **If you keep the colliding word, do the cleanup in the same change.**
   A half-renamed codebase teaches the model both meanings at once.
4. **Watch for the near-synonym trap.** A new noun must not collide
   *conceptually* either: "recipe" and "playbook" are both unused as
   types, but the three-stores table calls guidance a "natural-language
   recipe" and a playbook *is* a procedure.

## What gets renamed, and what does not

Rename **our** vocabulary: prompts, docstrings, tool labels, example
titles, section headers, internal identifiers, test names and fixtures.

Leave alone:

- **Simulated user speech in eval tests.** Users really do say "workflow"
  loosely about business processes. Scrubbing it removes exactly the
  ambiguity the assistant must survive.
- **Third-party product names.**
- **The shared `global-agent-rules` submodule.** Edit it in its own repo,
  never here.

After editing anything under `.agents/`, regenerate the aggregate:
`python3 .agents/global-rules/build_agents_md.py`. Never hand-edit
`AGENTS.md`.

# State Manager Interface Design

## Base Class Public APIs

The public API for all state managers (`FunctionManager`, `GuidanceManager`) is fully contained in the docstrings of the abstract methods defined on the base class `Base{SomeManager}` in `base.py`. All high level usage instructions should be fully encapsulated in these docstrings. These docstrings are then attached to the public methods of any derived class via `@functools.wraps(Base{StateManager}.{public_method}, updated=())`. These docstrings should not make **any** reference to **other managers** (we don't want to lock in any brittle cross-references, as other managers may change) and should also not make any reference to their **internal implementation**, including the private tools used for any particular instantiation of this abstract base class, with a consistent implementation agnostic public API.

## Prompts vs Tool Docstrings

The prompts in each prompt builder file should focus on the high level usage patterns, general guidance to the LLM, and specifically how to reason about the **composition** of tools, which tool to use in which scenario with contrastive explanations etc. However, in order to have a fully modular design and maximise our separation of concerns, it's very important that we do **not** bloat these prompts with any purely tool-specific information. This belongs exclusively in the tool's unique docstring (which the LLM gets access to). If the guidance is about deciding between two tools or using these tools together for complex composite behaviour, then it belongs in the prompt for the high-level public method in `prompt_builders.py`. If it's purely tool-specific, then it belongs in the tools own docstring.

Use this to decide which component owns what and where its jurisdiction ends. Keep manager docstrings implementation‑agnostic; this guide is only for high‑level routing and composition.

### ConversationManager
- **Role**: The persistent interaction loop. Reads the chat, decides whether to speak or wait, routes work to `Actor` for code-first execution and steers it (pause/resume/interject/stop/ask) while it runs.
- **Scope**: One user, one chat. Owns the chat history (in memory, persisted in its own store table), the notification bar and the in-flight and completed actions. Attachments are file paths passed along with a message.
- **Connections**:
  - **Steered by**: the chat front end (`unify/cli.py`, or any client that publishes `UnifyMessageReceived` on the in-memory event broker).
  - **Steers**: `Actor.act`; relays in‑flight handles.

### Actor
- **Role**: Central intelligence. Generates and executes Python programs in a persistent sandbox to satisfy a request, discovering stored skills first.
- **Scope**: Code-first execution via `act()`. A plan is plain Python plus the sandbox's namespaces: `primitives.actor.*` (nested actors for parallel or delegated work), `execute_function` for stored functions, and the `FunctionManager_*` / `GuidanceManager_*` JSON tools for the skill libraries. Files anywhere on the machine are read where they are; outputs go into the workspace. Wires in‑flight handles back to `ConversationManager` for real‑time steering.
- **Connections**:
  - **Steered by**: `ConversationManager` (primary caller of `act()`).
  - **Steers**: nested actors, the skill libraries' JSON tools, and the `ConversationManager` handle (`ask`/`interject`/`get_full_transcript`).

### Actor routing playbook
- **Before writing code**: `FunctionManager_search_functions` / `FunctionManager_filter_functions` for a stored function that already does it; `GuidanceManager_search` / `GuidanceManager_filter` for a procedure that says how. The discovery-first policy gates the other tools until both were consulted.
- **A single stored function or primitive call** → `execute_function` (a bare steerable handle; never `execute_code` for one call).
- **Anything else** → `execute_code`: plain Python, loops, control flow, files, packages installed into the workspace environment.
- **Parallel or delegated work** → `primitives.actor.act`.
- **Asking the user** → `request_clarification` (bubbles up through every layer to the chat).
- **After a run** the storage review decides what to keep: a callable that worked becomes a function, a non-obvious composition becomes guidance.

### FunctionManager
- **Role**: Catalogue of stored Python functions (the **what**) and their pip dependencies.
- **Scope**: add/list/filter/search/delete over functions, execution in-process with dependencies ensured in the workspace environment, and the read-only `primitives` table of every primitive the Actor can call, seeded from the primitive registry.
- **Connections**:
  - **Steered by**: `Actor` (discovers and executes functions during plans; the storage review stores new ones).
  - **Steers**: —

### GuidanceManager
- **Role**: Owner of procedural how-to information (the **how**): step-by-step instructions, walkthroughs, and strategies for composing functions together.
- **Scope**: CRUD (search, filter, add_guidance, update_guidance, delete_guidance) exposed as `GuidanceManager_*` JSON tools on the Actor. Read tools are gated by the discovery-first policy.
- **Builtins library**: reads also cover the read-only `builtin_guidance` table (through the `all_guidance` view, `is_builtin = 1`) holding entries imported from the Agent Skills ecosystem with stable hash-based ids. Seeded from the committed snapshot `unify/guidance_manager/builtins_guidance.json` when a `GuidanceManager` is constructed; `update_guidance`/`delete_guidance` refuse builtin ids.
- **Connections**:
  - **Steered by**: `Actor` (via `GuidanceManager_*` JSON tools).
  - **Steers**: reads the `functions` table to surface linked functions.

### EventBus
- **Role**: Cross‑cutting, in‑process publish/subscribe backbone and searchable event log used by every component for telemetry and coordination.
- **Scope**: Components publish structured events (notably `ManagerMethod` for incoming/outgoing method calls) via a thin logging wrapper; the bus supports `publish`, `search` over a bounded in-memory ring, `register_callback`/`unregister_callback`, and `join_callbacks` for deterministic joins.
- **Connections**:
  - **Steered by**: all public manager methods (through the logging decorator).
  - **Steers**: —

### Precedence and source of truth
- **Code is canonical**: This guide is descriptive. If the implementation contradicts it, the code takes precedence.
- **Keep in sync**: As components evolve, update this document alongside changes to cross‑component wiring, public surfaces, or prompt composition.
- **Where to update**: Prefer updating manager base docstrings (public API contracts) and prompt builders for tool composition guidance, and reflect those changes here.

# Surgical Verification Before Running Tests

When fixing infrastructure issues (especially concurrency, race conditions, or fixture problems), consider writing a quick verification script before running the full test suite.

## Why This Matters

Tests in this repo have significant overhead:
- LLM calls (cached: milliseconds, uncached: seconds to minutes)
- Fixture initialisation and catalogue seeding

A targeted verification script can validate a fix in **seconds** rather than waiting minutes for tests that may not even reliably reproduce the issue.

## When to Use This Pattern

✅ **Good candidates:**
- Race conditions and concurrency fixes (can simulate parallel execution directly)
- Fixture/conftest changes (test the seeding logic in isolation)
- Infrastructure plumbing (context management, caching, file locking)
- Non-deterministic failures that are hard to reproduce reliably in tests

❌ **Skip this for:**
- Simple bug fixes where one test runs quickly enough
- LLM behavior changes (need real eval tests with actual prompts)
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

## Real LLMs, Never Mocked

We *never* stub the LLM client. Tests always use a real LLM via `unillm.AsyncUnify`. Responses are cached per unique input, so repeated runs are fast on CI. The LLM responses become "locked in" after the first successful run.

## Trigger-Based Synchronization, Not Sleeps

Never rely on sleeps to align events in tests. Always use the trigger helpers in `tests/async_helpers.py` to ensure each event occurs in the necessary order. This makes tests robust to significant timing differences between cached responses (milliseconds) and live LLM calls (up to a minute).

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

---

# Shared rules

These apply across every unifyai repo. Edit them in the `unifyai/global-agent-rules` submodule, not here.

# Aggressive Refactoring & Zero Backward Compatibility

## Context
This project is a prototype in rapid development. We prioritize a clean, minimal, and correct codebase over stability, backward compatibility, or risk aversion.

## Critical Rules

### 1. Zero Backward Compatibility
- **Assume NO Backward Compatibility**: Unless explicitly requested, break APIs, data structures, and protocols freely.
- **Immediate Updates**: When changing an interface, update all call sites immediately. Do not create adapters, aliases, or optional parameters to "soften" the change.
- **Purge Old Patterns**: If you introduce a new design pattern (e.g., for state management), strictly remove all instances of the old pattern. Do not leave mixed patterns in the codebase.

### 2. Destructive vs. Additive Editing
- **Avoid "Stapling"**: Do not merely add new logic on top of old logic (additive editing). This creates bloat and "staples upon staples".
- **Rewrite and Simplify**: When requirements change, **rewrite** the affected code to optimally support the *new* requirements as if they were the original ones.
- **Delete Aggressively**: If code is no longer the "best" way to do something, delete it. Do not comment it out. Do not keep it "just in case".

### 3. No Defensive Coding
- **No Preemptive Exception Handling**: Do not wrap code in try/catch or try/except blocks to prevent crashes unless you are handling a specific, expected, and recoverable runtime error (e.g., network timeout). Fail loud and fast.
- **No Defensive Checks**: Do not add null checks or type checks unless strictly necessary for the logic. Trust the type system and the caller contract.
- **Clean Implementation**: Code should look like the "happy path". Avoid cluttering logic with defensive branches for edge cases that shouldn't happen in a correct system.

### 4. Review and Reflect
- **Simplify First**: Before adding a feature, ask: "Can I simplify the existing code to make this feature natural to implement?"
- **Remove Bloat**: actively look for and remove redundant code, unused variables, and overly complex abstractions.

# No Temporal or Chat-Specific Comments

## Context
Code comments and docstrings must be **timeless** and describe the code as it currently exists. They must **never** reflect the history of changes, the current chat session, or the fact that code is "new".

## Critical Rules

### 1. No "New" or "Updated" Markers
- **Forbidden**: Never use words like "New", "Updated", "Added", "Modified", "Refactored" in comments to mark changes.
- **Reasoning**: Code is only "new" for the moment it is written. Next week, it is old. These comments rot immediately and create noise.
- **Correct Approach**: Just write the comment describing what the code does. Git history tracks what is new.

### 2. No Chat Context
- **Forbidden**: Do not reference "the user request", "this chat", "per instruction", or specific reasoning from the current conversation.
- **Reasoning**: The codebase must stand alone. Context from a chat session is lost to future readers.
- **Correct Approach**: If a complex decision needs explanation, document the *technical why* (e.g., "Use X because Y is slow"), not the *conversational why* ("User asked for X").

### 3. Clean Documentation
- **Focus**: Comments should explain **why** tricky code exists or **how** it functions.
- **Avoid**: "Here is the implementation of..." or "Standardized composer utilities". The code itself shows it is an implementation.
- **Example**:
  - BAD: `NEW: Added this function to handle retries`
  - BAD: `Updated to support the new API`
  - GOOD: `Retries the request with exponential backoff to handle transient network errors.`

If a test is failing, we should **never** add test-specific information or shortcuts to production code as a hack to get the test passing. No details about specific test cases should ever make their way into production code—no special-case branches, no hardcoded values that match test inputs, no conditional logic that only exists to satisfy a test.

All fixes must be fully general and **much** broader than the specific failing test. We do **not** want to overfit production code to a specific set of tests.

# Git Safety

## Rule: Pull Before Editing a Repository

Before making any file edits in a repository, run `git pull --rebase` once to sync with the remote. This prevents the agent from working on a stale branch and silently overwriting others' commits.

- **Once per repo per session** is sufficient — no need to pull on every turn.
- **After a push rejection + rebase**: re-read any files you plan to edit next. The rebase changed them on disk but your in-memory copies are stale.
- **Exception**: only skip if the user explicitly asks you not to pull.

## Context: Explicit Path Commits
When multiple agents run in parallel in `local` mode, there is a race condition risk if they use the shared git index (staging area).
- Agent A: `git add fileA`
- Agent B: `git add fileB`
- Agent A: `git commit -m "msg"` -> Commits BOTH fileA and fileB!

## Rule: Explicit Path Commits
To eliminate this risk, **NEVER** run `git commit` without explicit file arguments.

### Incorrect
```bash
git add myfile.json
git commit -m "Update myfile"
```

### Correct
```bash
# For modified files:
git commit myfile.json -m "Update myfile"

# For new (untracked) files:
git add myfile.json
git commit myfile.json -m "Add myfile"
```

### Reasoning
Passing filenames to `git commit` bypasses the shared index for that specific commit operation, ensuring that Agent A only commits what it intends to, regardless of what Agent B has staged.

## Rule: Push When the Work Lands

Commit to `main` and push as the work lands, without waiting to be asked. An
unpushed commit, or a branch nobody merges, is invisible to every other
checkout and every other agent: a branch in `continual-arc-baselines` once
drifted ten commits ahead of an untouched `main` before anyone noticed.

- Push the current branch only: `git push origin HEAD`.
- If the push is rejected, `git pull --rebase`, re-read what you are about to
  edit, and push again.
- **Never** force-push unless the user explicitly asks for it and understands
  that it rewrites history other people have pulled.
- Start a branch only when a change genuinely has to sit apart from `main`,
  and merge and delete it as soon as it lands.

# Worktree Mode: Direct Commits, No Feature Branches

## Context
When running in **worktree mode**, the mental model is fundamentally different from traditional feature development:

- **Worktrees are for small-scale parallel fixes**, not large-scale feature development
- **Multiple agents on the same branch** = multiple collaborators working in parallel, each with their own local working directory
- The overhead of `feature branch → PR → merge → cleanup` is **overkill** for this workflow

## Critical Rules

### 1. NEVER Create Feature Branches
When asked to make changes, commit, or push:
- **DO NOT** create a new branch (e.g., `git checkout -b feature/...`)
- **DO NOT** suggest creating a branch for the work
- **COMMIT DIRECTLY** to whatever branch is currently checked out

The worktree already provides isolation. Creating additional branches defeats the purpose.

### 2. NEVER Create Pull Requests
- **DO NOT** use `gh pr create` or suggest creating a PR
- **DO NOT** push to a new remote branch with the intent of opening a PR
- If the user wants changes merged, they will handle the merge strategy themselves

### 3. The Correct Workflow
```bash
# 1. Make your changes to files

# 2. Commit directly to the current branch (following git-commit-safety rules)
git commit <specific-files> -m "Description of change"

# 3. Push the current branch (see git-commit-safety)
git push origin HEAD
```

### 4. Mental Model
Think of worktree agents as **multiple developers pair-programming on the same branch**:
- Each has their own local checkout (the worktree)
- All commit to the same branch
- No one creates personal feature branches for small fixes
- Coordination happens through communication, not branch isolation

## Why This Matters
The alternative workflow creates significant noise:
1. **Stale branches accumulate** - agents create branches, users forget to delete them
2. **PR overhead** - reviewing, merging, and closing PRs for trivial fixes wastes time
3. **Context switching** - users must mentally track multiple branches for what should be one stream of work
4. **Merge conflicts** - more branches = more opportunities for conflicts

## Exception
If the user **explicitly asks** for a feature branch or PR workflow, follow their instructions. But **never default to this behavior** in worktree mode.

# Git History for Context

## Context
This rule applies when you are trying to understand the *rationale* behind specific code blocks, the evolution of a module, or when deciding whether "weird" looking code is essential or legacy technical debt.

## Rules

### 1. Strategic Git Usage
- **Use as a Second Level of Analysis**: If the code's purpose isn't clear from the current state alone (static analysis), use `git blame` or `git log` to uncover the "why".
- **Not a Mandate**: Do not check git history for every file you touch. This creates noise. Use it selectively when you lack context.

### 2. Understanding Code Evolution
- **Identify Legacy Code**: If you suspect code is redundant or outdated, check its commit date and message. If it was added months ago for a feature that is no longer relevant, this confirms it can likely be purged.
- **Find the "Why"**: Expressive commit messages often contain the reasoning that comments lack. Use them to understand the author's original intent before refactoring or deleting complex logic.

### 3. Targeted Queries
- **Be Surgical**: When querying git, look for the history of specific lines or changes (e.g., `git blame -L n,m filename` or `git log -p filename`) rather than dumping the entire history into the context.
- **Synthesize**: Use the information to form a narrative about the code's lifecycle (e.g., "This was added in commit X to fix bug Y, but since we rewrote the bug Y subsystem, this is now dead code").

### 4. Investigating Regressions with Git Diff

When debugging test failures or regressions, git history can pinpoint exactly what changed.

**When the user proactively provides context:**
If the user says something like "the test was passing at commit `<hash>`, and the relevant changes are in `<path>`", use this optimally:
- Run `git log --oneline <hash>..HEAD -- <path>` to see which commits touched the area
- Run `git diff <hash>..HEAD -- <path>` to get the **aggregate diff** (not serial diffs commit-by-commit)
- Cross-reference the diff with commit messages to understand developer intent
- The overall diff is mathematically equivalent to composing serial diffs, but far more token-efficient and cognitively cleaner

**When debugging hits a roadblock:**
If direct code analysis and debug logging (`CURSOR_DEBUG_LOG`) aren't yielding answers, *then* ask the user:
- "Do you know when this test was last passing? If you have a commit hash and know which files/folders are likely involved, that would help narrow down what changed."
- Don't front-load this question—often the user doesn't know the answer. Try direct debugging first.

**Avoid wasteful patterns:**
- Don't ask the user to provide diffs—ask for the commit hash and run git commands yourself
- Don't read diffs commit-by-commit and mentally compose them; use the aggregate diff
- Don't dump entire file histories; scope queries to the relevant path(s)

# Python Formatting & Pre-commit

Every first-party Python repo (`orchestra`, `unify`, `unisdk`, `unillm`,
`unify-deploy`, `docs`) enforces formatting with **black** (plus
`isort`/`autoflake` where configured), and CI rejects unformatted code. A
missing local hook or a drifting Black target/Python version is the single
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

## Formatting across multiple repos

When juggling several repos, do not invoke a globally-installed `black` —
versions drift between machines and repos and produce diffs CI rejects.
Always format through the repo's pinned tooling, which uses that repo's
locked version:

```bash
pre-commit run black --all-files          # or: uv run black .
```

## Release gates

`black` is a required status check on `staging → main` (ruleset and/or branch
protection) for the Python repos. Direct pushes to `staging` stay open for
the worktree workflow — the committed git hook is the staging-side gate.
CI still runs `black` on every push so failures are visible immediately.

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

# Shared agent conversation archive

Unify keeps a private repo of **raw** agent transcripts at **`~/shared_context`**
(GitHub: `unifyai/shared_context`), keyed by **GitHub login** (e.g. `djl11`).

## Design (important)

- **Adjacent clone, not a submodule.** `shared_context` sits next to the
  product checkouts (`~/brain`, `~/unillm`, …). It is **not** nested under any
  public or private product repo.
- **Why:** public repos such as `unillm` stay public; transcript data stays
  private. Public cloners never need or see this tree. One clone serves agents
  in **every** repo that pulls `unifyai/global-agent-rules`.

## When to load this

Use before answering questions about past investigations or decisions across the
team — e.g. "did we set up X?", "who changed Y?", "why did we do Z?" — when the
answer might live in someone else's Cursor / Claude Code / Codex session, not
only the current chat.

## How to search

Prefer ripgrep over reading whole files. Search **tracked** login trees only —
**do not** search `yours/` unless the user explicitly asks about their local /
unexported chats:

```bash
rg -n -i "keyword" ~/shared_context/derived/index.jsonl
rg -n -i "keyword" ~/shared_context -g '!yours/**' -g '!tools/**' -g '!.git/**'
```

`derived/index.jsonl` is rebuilt locally by `tools/sync.sh` / `tools/export.py`
after pull (gitignored). If it is missing, search tracked trees directly or run:

```bash
python3 ~/shared_context/tools/export.py --index-only
```

Sessions live at
`<github_login>/{cursor|codex|claude-code}/<yyyy-mm>/<id>/{meta.json,transcript.jsonl}`.

`yours/{cursor,codex,claude-code}` are local symlinks to personal stores and are
gitignored.

If `~/shared_context` is missing, say so and suggest:

```bash
git clone git@github.com:unifyai/shared_context.git ~/shared_context
```

Do **not** suggest `git submodule add` / nesting it under a product repo.

## Citing

Cite **user**, **tool**, **date**, and **path** so a human can open the same session.

## Do not

- Do not confuse this with `brain` (curated company memory).
- Do not scrub or rewrite historical transcripts.
- Do not push/sync unless the user asked you to.
- Do not grep `yours/` unless the user asked for local-only context.

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
