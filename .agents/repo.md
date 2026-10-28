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

**Tests always use stubbed models and never reach a real provider.** No test needs a real provider key; every model
call in a test is a stubbed one. `tests/conftest.py` enforces it in every test: unillm's provider calls raise, a socket may connect only to this machine, and the markers that once
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
