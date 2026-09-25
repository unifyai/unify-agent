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
