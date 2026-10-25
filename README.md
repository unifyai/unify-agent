<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/unifyai/unify-agent/main/assets/brand/unify-readme-banner-dark.png">
    <img src="https://raw.githubusercontent.com/unifyai/unify-agent/main/assets/brand/unify-readme-banner-light.png" alt="unify" width="100%">
  </picture>
</p>

<p align="center">
  <a href="https://unify.ai">unify.ai</a> · <a href="LICENSE">MIT license</a>
</p>

# unify-agent

unify-agent is an agent harness that learns from use. You give it a task in
plain words. A model, the actor, solves it by writing Python, one program per
turn, which the harness runs in a sandbox on your machine. With memory on, each
finished task is written down, and a background writer turns those records into
a library of tested functions and notes that later tasks can ask for and call.

It runs locally with one model-provider key, and keeps everything it learns in
a git repository under `~/.unify`. It is research code, published so the design
can be read, run and measured.

## Quickstart

You need:

- **Linux, or Windows with WSL2.** macOS is not supported: every program the
  agent runs is confined with bubblewrap, which needs Linux, and the harness
  refuses to run without it.
- **bubblewrap, util-linux (for `prlimit`) and git.** On Debian or Ubuntu:
  `sudo apt install bubblewrap util-linux git`. Check that bubblewrap can start
  a sandbox with `bwrap --ro-bind / / true`; some distributions (Ubuntu 24.04,
  for one) restrict the user namespaces it needs.
- **Python 3.12 and [uv](https://docs.astral.sh/uv/).**
- **One model-provider key:** `OPENROUTER_API_KEY` for the default model, or
  `ANTHROPIC_API_KEY` with an Anthropic model named in `UNIFY_MODEL`.

> **Install layout.** unify-agent pins its model client, unillm, to commit
> `9d85422` of unillm's `main`, and uv installs it from a unillm checkout
> beside this one (`[tool.uv.sources]` in `pyproject.toml`), at that commit.
> Clone both as below.

```bash
git clone https://github.com/unifyai/unify-agent.git
git clone https://github.com/unifyai/unillm.git
git -C unillm checkout 9d8542206047d8131cf23f5511f4d6afcc81fc75
cd unify-agent
uv sync --no-dev
```

Put your key in a `.env` file in the checkout (git ignores it), as
`OPENROUTER_API_KEY=...`, or export it in your shell. Then give the actor a
file and a task:

> **Memory spends on your key.** Memory is on by default, and it makes model
> calls of its own on your provider key, beside the actor's: the writer, its
> helpers, the episode-end fork and embeddings. Together they are capped at
> `UNIFY_MEMORY_CAP_USD`, 20.00 USD per run (one `unify act`) by default.
> Set `UNIFY_MEMORY=off` to make no memory calls at all.

```bash
mkdir -p ~/.unify/workspace
cp data.csv ~/.unify/workspace/
.venv/bin/unify act "How many rows does data.csv have, and which columns are mostly empty?"
```

Progress goes to stderr and the answer to stdout. The actor's programs see the
system's read-only directories, their workspace `~/.unify/workspace` and a
private `/tmp`, and can write only the last two. Everything the agent keeps
lives under `~/.unify` (`UNIFY_HOME`); delete that directory to start from
nothing.

With memory on and `OPENROUTER_API_KEY` set, `unify act` starts a local lane
relay that holds the key and gives memory's writer, helpers, fork and
embeddings their own lanes, all under that one cap; memory's processes never
hold the key. Memory's search needs embeddings, and the harness never sends the
provider key to a provider for them directly: they go through the relay's
embedding lane, or you compute them in process (install the extra with
`uv sync --no-dev --extra local-embeddings` and set
`UNIFY_LOCAL_EMBEDDINGS=true`), or you set `UNIFY_EMBED_URL` to a proxy lane of
your own on `http://127.0.0.1:<port>/...`. With none of these, a search that
needs a new embedding is refused.

## What the agent and its memory do

**The actor.** Each turn, the model writes one Python program. The harness runs
it in a persistent sandboxed interpreter, so variables carry over from one
program to the next. Shell commands run from Python through `subprocess`.
Packages a program needs are installed into one shared environment under
`~/.unify`, from prebuilt wheels only. The actor answers when it is done; with
`--persist` it then waits for a follow-up in the same sandbox.

**Memory** is on by default. After each task:

1. **A capture.** A confined copy of the actor's conversation, a fork, is asked
   to write down what it did: a small runnable procedure, what it was for, and
   which parts of the task it rests on. The fork needs a fork lane, which
   `unify act` starts with memory on and `OPENROUTER_API_KEY` set (or a
   launcher gives). Without one no fork runs and no capture is made;
   `ANTHROPIC_API_KEY` alone stores none.
2. **The writer.** A background writer reads the captures and failures that
   need attention, and keeps a library in a git repository at
   `~/.unify/memory`: tested functions, and notes that link to the functions
   they use (and back). It states tests for what it writes, and the harness
   runs them in a sandbox. A gate checks every change and records what it
   finds as labels on the item. The writer spends under memory's cap.
   With the lane relay, `unify act` stops the writer when it ends: a pass in
   flight gets up to 5 minutes, and what it held stays owed for the next run.
   A writer service that no run owns exits after 30 idle minutes.
3. **The librarian.** At the start of a later task, a librarian agent reads
   the library and tells the actor which items fit; the actor can ask it
   again with `memory.ask("...")`. Each item comes with a recorded example
   call and can be called directly, as `item(...)`. The actor decides
   whether to use it.
4. **A shared run record.** The actor, its teammates, the forks, the writer
   and the librarian post to one append-only record. They can address one
   another (`@writer`, `@librarian`), so a question reaches the agent that
   can answer it.

[`docs/memory.md`](docs/memory.md) explains the memory in full.

## Configuration

Memory has one switch, `UNIFY_MEMORY`:

- `UNIFY_MEMORY=off`: no memory of any kind. Each task costs only the actor's
  model calls.
- `UNIFY_MEMORY=on`, the default (also when unset or empty): memory between
  tasks. It adds the capture, writer and librarian calls to the actor's.

Everything else is essential configuration only: the provider key, the model
and reasoning effort, the paths, where embeddings come from, and the writer's
spend cap.
[`CONFIG.md`](CONFIG.md) lists every environment variable, one line each.

## Using it in your own workflow

### As a subprocess

`unify act --persist --jsonl "first request"` keeps one session open and speaks
newline-delimited JSON. Progress still goes to stderr.

Lines your program writes to its stdin:

| Line | Effect |
|---|---|
| `{"message": "..."}` | A follow-up request in the same session. |
| `{"cancel": true}` | Ends the running turn at once and keeps the session. |
| `{"quit": true}` | Ends the session. |
| `{"outcome": {"solved": true, ...}}` | Your verdict on the work, so memory learns what worked. The full form is in `unify/outcome.py`. |
| `{"checker": {"label": "pass"}}` | With memory on: a verdict the environment showed the actor during the request, `"pass"` or `"fail"`. It may also carry `about` (an action or item) and `episode` (an earlier episode's id). |

Lines it writes to stdout, each with a `type`:

| `type` | When |
|---|---|
| `response` | At the end of every turn, with the turn's text in `content`; `"cancelled": true` if a cancel ended it. |
| `record` | An entry in the run record addressed to `@user`. |
| `outcome` | The answer to an outcome line: `accepted`, or a `reason` if it was refused. |
| `result` | The session's final result, with `run_stats`. |
| `ended` | The last line: the session is over. |

With memory on, other types report memory's progress; ignore types you do not
know. A minimal driver, with `unify` on the `PATH` (or `.venv/bin/unify`):

```python
import json, subprocess

proc = subprocess.Popen(
    ["unify", "act", "--persist", "--jsonl", "How many rows does data.csv have?"],
    stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True,
)

def send(line: dict) -> None:
    proc.stdin.write(json.dumps(line) + "\n")
    proc.stdin.flush()

for raw in proc.stdout:
    event = json.loads(raw)
    if event["type"] == "response":
        print(event["content"])
        send({"outcome": {"solved": True, "source": "user"}})
        send({"quit": True})
    elif event["type"] == "ended":
        break
proc.wait()
```

### From Python

```python
import asyncio

import unify
from unify.actor.code_act_actor import CodeActActor
from unify.actor.environments.actor import top_level_environments


async def main() -> None:
    unify.init()
    actor = CodeActActor(environments=top_level_environments())
    handle = await actor.act("How many rows does data.csv have?")
    print(await handle.result())
    await actor.close()


asyncio.run(main())
```

`act()` returns a handle with `result()`, `done()`, `stop(reason)`, `submit(text)`
(a follow-up, read at the next turn boundary), `cancel_request()`,
`next_notification()`, and `next_clarification()` with
`answer_clarification(call_id, answer)`. This runs the actor alone: memory is
driven by the `unify act` session (`unify/cli.py`), so to use memory from your
program, drive the command as above.

### Your own tools

Give the actor your own functions as `primitives.<name>`. Point
`UNIFY_ENV_NAMESPACES` at a factory, `package.module:factory`, importable by the
harness:

```python
from unify.function_manager.primitives.environment import (
    EnvironmentMethod,
    EnvironmentNamespace,
    EnvironmentSurface,
)


def lookup(order_id: str) -> dict:
    """Return the order with this id from the order service."""
    ...


def surface() -> EnvironmentSurface:
    return EnvironmentSurface(
        namespaces=(
            EnvironmentNamespace(
                name="orders",
                description="Our order service.",
                methods=(
                    EnvironmentMethod(
                        name="lookup",
                        call=lookup,
                        effect="read",  # or "write" or "destructive"
                        signature="(order_id: str) -> dict",
                        docstring=lookup.__doc__,
                    ),
                ),
            ),
        ),
    )
```

With `UNIFY_ENV_NAMESPACES=mytools:surface`, the actor's programs call
`primitives.orders.lookup("A-17")` and its prompt lists the namespace. Your
functions run in the harness process, outside the sandbox and with its access;
arguments and results cross from the sandbox as plain data. A factory that
fails stops the harness at start-up. The details are in
`unify/function_manager/primitives/environment.py`.

### Network

The actor's programs have no network by default. With
`UNIFY_WORKSPACE_NETWORK=proxy`, their only route out is a proxy you run on
the host at `127.0.0.1:UNIFY_WORKSPACE_PROXY_PORT`; `HTTP_PROXY` and
`HTTPS_PROXY` point at it inside the sandbox, so your proxy decides what is
reachable. Package installs are separate: they go through the harness's own
proxy, which reaches only the package index (PyPI, or the index your
`UV_INDEX` settings name).

## Safety model

- Every program the actor writes, every shell command and every subprocess
  runs inside bubblewrap. The host root is mounted read-only and only as an
  allow-list; the workspace and a private `/tmp` are the only writable places.
  `~/.unify`, credential directories (`~/.ssh`, `~/.config`, `~/.aws` and
  others) and `.env` files are hidden. Environment variables that look like
  credentials are removed, a seccomp filter limits sockets and system calls,
  and resource limits apply. Without bubblewrap nothing runs.
- Model-written code never sees the provider key. Model calls are made by
  harness code; a program the actor writes reaches a model only by asking the
  harness.
- Memory's agents run their code in the same kind of box, with no network and
  an empty environment.
- Your own tools (`primitives.<name>`) run outside the sandbox, with the
  harness's access. They are the openings you choose.
- The sandbox limits what injected instructions can reach; it does not stop a
  model from following them. Review what you feed the agent.

[`SECURITY.md`](SECURITY.md) has the full trust model.

## Testing

```bash
uv sync --group dev
uv run pytest -n auto              # the whole suite
uv run pytest -n auto tests/actor  # one area
```

The tests need no key and make no network calls. Every model reply is
scripted (`tests/scripted_model.py`), provider calls are refused, and sockets
may connect only to this machine. When no key is set, the root `conftest.py`
sets a placeholder that is plainly not a secret. Each pytest process runs
inside bubblewrap, so the tests need it too. The tests of the in-process
embedder also need its weights in the Hugging Face cache.

- `tests/parallel_run.sh tests/` runs one tmux session per test file (it needs
  tmux); see [`tests/README.md`](tests/README.md).
- [`model_evals/`](model_evals/README.md) holds checks of a live model. They
  need a key, spend money and are run by hand.

## Project status

This is research code, published as it is.
TODO(lead): maintenance status after the wind-down (who reviews pull requests,
whether issues are answered).

Please report security issues privately, not in a public issue; see
[`SECURITY.md`](SECURITY.md). Contributions: [`CONTRIBUTING.md`](CONTRIBUTING.md).

## License

MIT. See [LICENSE](LICENSE).
