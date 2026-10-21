# Configuration

These are every environment variable the package reads. `tests/test_env_allowlist.py` fails on any variable that
the code reads and this file does not list. A variable may also come from the `.env` file in the working directory.

Nothing else is configurable. Every other number is a constant in the code, set to the value it was evaluated
with.

## The memory switch

| Variable | Values | Default | What it does |
|---|---|---|---|
| `UNIFY_MEMORY` | `off`, `on`, `in_episode` | `on` | `on` keeps memory between episodes. `in_episode` adds memory during an episode as well, at about twice the cost per visit. `off` is the no-memory baseline. Any other value is refused, and the error lists the three. See [docs/memory.md](docs/memory.md). |

## Models and the provider

| Variable | Default | What it is |
|---|---|---|
| `OPENROUTER_API_KEY`, `ANTHROPIC_API_KEY` | none | The provider key; unillm reads it. |
| `UNIFY_MODEL` | `openai/gpt-5.6-sol@openrouter` | The model every request runs on. |
| `UNIFY_REASONING_EFFORT` | `high` | The reasoning effort that goes with it. |
| `UNIFY_MEMORY_WRITER_MODEL` | empty, which means `UNIFY_MODEL` | The model the memory writer and its helpers run on. It must be a model the pinned price catalogue bounds, or the writer does not start. |

## Bounds and spend

| Variable | Default | What it bounds |
|---|---|---|
| `UNIFY_MAX_OUTPUT_TOKENS` | `32768` | One turn's output, so a degenerate turn cannot bill a whole output window. |
| `UNIFY_MAX_TOOL_LOOP_STEPS` | `300` | The steps of one request. |
| `UNIFY_MEMORY_CAP_USD` | `20.00` | Memory's spend over the memory home's life: the writer and its helpers, and the episode-end forks when a lane relay holds them. It takes a plain decimal string and fails closed: an unpriced call is booked at its worst case, and once the cap is reached the library stops changing while the actor is still served from it. |

## Paths

| Variable | Default | What it is |
|---|---|---|
| `UNIFY_HOME` | `~/.unify` | Where the state lives, including the memory home `<UNIFY_HOME>/memory`. |
| `UNIFY_LOCAL_ROOT` | `<UNIFY_HOME>/workspace` | The workspace directory the actor works in. |
| `UNIFY_LOG_DIR` | `<UNIFY_HOME>/logs` | Where the log files go. |

## What cell code may reach

| Variable | Default | What it allows |
|---|---|---|
| `UNIFY_CELL_LLM_MODELS` | empty | The model endpoints that cell code may name. This governs both spend and confinement. |
| `UNIFY_CELL_ENV_ALLOW` | empty | A runner's own variables that a cell receives, in addition to the allow-list. |
| `UNIFY_WORKSPACE_NETWORK`, `UNIFY_WORKSPACE_PROXY_PORT` | off | Whether cells reach the network. When they do, it is only through the host's proxy on that port. |
| `UNIFY_ENV_NAMESPACES` | empty | An embedding application's own `primitives.<name>` namespaces. |

## Model-call lanes

A launcher sets these. Nobody sets them by hand.

Each lane is a loopback relay that holds the provider key and gives memory's calls their own token and spend cap. The
controller reads each token from an inherited descriptor once, closes the descriptor and drops the variable from its
environment. A lane's base URL and its descriptor are set together or not at all.

| Variable | What it is |
|---|---|
| `UNIFY_MEMORY_WRITER_BASE_URL`, `UNIFY_MEMORY_WRITER_TOKEN_FD` | The writer's lane. Without it, the writer calls on the controller's own provider route. |
| `UNIFY_MEMORY_WRITER_JOURNAL` | The writer lane's JSONL journal (an absolute path), which prices the writer's calls. |
| `UNIFY_MEMORY_WRITER_PRICES_USD_PER_MILLION` | The writer model's pinned prices, as JSON in USD per million tokens. |
| `UNIFY_MEMORY_LIBRARY_FORK_BASE_URL`, `UNIFY_MEMORY_LIBRARY_FORK_TOKEN_FD` | The episode-end fork's lane. Without it no fork runs, so no capture is made. |
| `UNIFY_MEMORY_LIBRARY_HELPER_BASE_URL`, `UNIFY_MEMORY_LIBRARY_HELPER_TOKEN_FD` | The writer helpers' lane. Without it the writer is not offered `delegate`. |

The relay's own contract is in `unify/memory/integration/switch.py`.

## Test-only

`UNIFY_TEST_*`, `UNIFY_RECORD_GOLDEN` and `SKIP_UNIFY_TEST_INIT` are read only by the test suite. They are not
configuration.
