# Architecture

This document describes how the harness works, for anyone who wants to study it, change it or measure it. The README
says what the harness is for and how to run it; `docs/memory.md` describes the memory; `CONFIG.md` lists every
setting.

## Mental model

One **actor** answers a request by writing Python. Its only model-API tool is `execute_code(code)`. Each cell runs in
one persistent interpreter inside a **sandbox** (bubblewrap). Every other tool is a call `tools.<name>(...)` from a
cell, which crosses a narrow **tool channel** to a handler on the host. The actor and any teammates it starts share
one **agent record** for messages. With memory on, a **memory** package turns finished requests into a tested library
that later requests read.

```
unify act (the CLI)
 │  one request, or a persistent session (--persist, --jsonl)
 ▼
CodeActActor ── the tool loop: one model turn at a time
 │  execute_code(code)
 ▼
BoxInterpreter ── one persistent interpreter per agent
 │  PythonWorker: python -I -S inside bubblewrap
 │  tools.<name>(...) ──► tool channel (two pipes) ──► host handlers
 ▼
the agent record ── teammates, questions to the requester, the librarian
memory (UNIFY_MEMORY) ── the library the next request reads
```

## The CLI

**File:** `unify/cli.py`

`unify act "request"` builds one `CodeActActor` over the environments the process registered, runs the request and
prints the result. `--persist` keeps the session open: each further line is a new request in the same interpreter.
`--jsonl` speaks newline-delimited JSON for a program driving the actor: stdin lines are `{"message": ...}`,
`{"cancel": true}`, `{"quit": true}` and, with memory on, `{"outcome": ...}` and
`{"checker": {"label": "pass"|"fail"}}` (optionally with `text`, `about` and `episode`); stdout lines are `result`, `response`,
`record` and `ended`. A cancel ends the running request and keeps the session. When stdin is not a terminal it is
read from a private copy and descriptor 0 becomes `/dev/null`, so nothing model code starts can read the driver's
lines.

## The actor

**Files:** `unify/actor/code_act_actor.py`, `unify/actor/prompt_builders.py`, `unify/actor/in_code_tools.py`

`CodeActActor.act(request)` refuses to start without bubblewrap, joins the run's agent record, builds the system
prompt and starts the tool loop. The system prompt is a pure function of the configuration: the shared opening that
names the agent's tools, the actor's role and rules, each environment's section, and with memory on a fixed guide to
the library at the end, so the prefix stays the same for the whole run. The first user message carries, before the
request, the record's section and, with memory on, the library's size and the librarian's answer about what in the
library fits the request.

`InCodeTools` is the request's in-code surface: it opens the interpreter with the tool channel, runs each
`execute_code` call as one cell, writes one row per call to the request's `tool-events.jsonl`, and refuses a cell
that only shows constant text (`unify/actor/no_op_cells.py`). `tools.answer(answer)` gives the request's answer from
a cell: the loop ends the request with it when the cell returns, without another model call; with a response format
the answer is validated first.

## The tool loop

**Files:** `unify/common/async_tool_loop.py`, `unify/common/_async_tool/loop.py`, `unify/common/_async_tool/`

The loop sends the conversation, runs the turn's tool calls in call order, each to completion, appends the results
and repeats until the model answers in text (or a cell gave the answer). Nothing the requester sends interrupts a
model call or a cell: a new message joins at the next turn boundary. `stop()` cancels what is in flight; a persistent
loop's `cancel_request()` ends only the current request.

- **Limits.** `UNIFY_MAX_TOOL_LOOP_STEPS` bounds a request's messages: at the limit the request ends with a reply
  that says so and quotes the latest draft, and a persistent session takes the next request. Three back-to-back tool
  crashes end the loop. A refused call (invalid JSON, a parameter the tool does not have) never reaches the tool; the
  model reads what to change.
- **Provider errors.** Transient errors are retried in place. A reply the provider withheld (`content_filter` with no
  text) is sent again at most twice.
- **Compaction.** Before the next model call, once a prompt reaches `COMPACTION_AT_TOKENS` (150,000) or 85 % of the
  model's window, the handle asks a fork of the last request for a summary and restarts the session from its own
  system prompt, the same tool list and the summary, which ends with a pointer to the session's transcript. When the
  fork gives no summary, `context_compression.compress_and_rebuild` builds the new context instead.
- **Cache discipline.** `unify/common/_async_tool/cache_discipline.py`: the tool list is fixed at a session's first
  call, a message once sent is never edited (the append-only check runs on every dispatch), and sessions with the
  same prefix share one cache-affinity key. Each reply's cached share is logged.
- **Request headers.** Every model call carries `X-Unify-Session`, `X-Unify-Request`, `X-Unify-Call-Kind` and
  `X-Unify-Msg-Count` (`unify/common/llm_client.py`), so a proxy can attribute calls. Every call is streamed.

## The interpreter and the sandbox

**Files:** `unify/common/box_interpreter.py`, `unify/actor/execution/worker.py`, `unify/actor/execution/worker_child.py`,
`unify/sandbox.py`

Each agent's cells run in one `PythonWorker`: `python -I -S` inside bubblewrap, started on the first cell and kept
until it is restarted. A cell's variables persist from cell to cell. A cell past its limit, a thread a cell left
running, an idle interpreter or a worker that exited is killed with every process in its pid namespace (verified,
`unify/common/process_tree.py`), and the next cell's output says the variables are gone.

The sandbox's root is an allow-list: the system directories, the interpreter and the package, read-only; the
workspace at `/workspace` is writable; `/tmp` is private; the transcripts are readable at `/transcripts`. The Unify
state directory, the log directories, credential locations and secret-named files are hidden behind notices that name
the rule. The environment is built from an allow-list (`UNIFY_CELL_ENV_ALLOW` adds a runner's own names); a
credential's name or a key-shaped value never passes. There is no network, or with `UNIFY_WORKSPACE_NETWORK=proxy`
one loopback port to the host's proxy. Resource limits are set inside with prlimit.

Harness objects never enter the worker. The names a cell sees (`tools`, `query_llm`, `display`, an environment's
`primitives` and globals) are served from a host-side namespace through the worker's channel: data crosses by
value, other objects by reference, and modules, the harness's environment and credential-shaped values are refused by
name.

## The in-code tools

**Files:** `unify/common/tool_registry.py`, `unify/common/tool_channel.py`, `unify/common/tool_channel_client.py`,
`unify/actor/tool_handlers.py`

One registry holds every tool's name, description, parameter schema and refusal texts, for the actor, its teammates
and the memory agents. The box module `tools` is generated from it; `tools.list()` and `tools.help(name)` read it.
A call is validated on the host before its handler runs, and a refusal carries a stable form code. Each agent's
allow-list says which tools it has; `tools.check_examples` reaches the actor and its teammates only with
`UNIFY_ACTOR_EXAMPLES_CHECK=true`, for a world whose requests give input/output examples.

## The workspace environment

**File:** `unify/environment.py`

`tools.install(packages)` installs wheels into one persistent venv under `<UNIFY_HOME>/venv`, which every later cell
can import. The installer runs `uv` in its own sandbox with no credentials; its only network is an allow-listing
CONNECT proxy to the package index hosts.

## Environments

**Files:** `unify/function_manager/primitives/`, `unify/actor/environments/`

An embedding application registers its own callable surface as `primitives.<name>` namespaces, from code
(`register_environment`) or with factories named by `UNIFY_ENV_NAMESPACES`. Each method declares its signature, its
documentation and its effect (`read`, `write`, `destructive`). The actor's prompt lists the namespaces; a cell reads a
method's documentation with `help(primitives.<name>.<method>)`.

## Model calls from cells

**Files:** `unify/common/reasoning.py`, `unify/common/cell_models.py`

`query_llm` and the `unillm` clients a cell builds run on the host. By default a cell may name only the session's
model (`UNIFY_CELL_LLM_MODELS` widens that). A provider's refusal of the request's own content is
`ModelRequestError` in the cell; any other failure is `ModelCallFailed` with a fixed text.

## The agent record

**Files:** `unify/agents/`

One append-only record per run that every agent, the requester and the harness post to: the actor and its
teammates, and with memory on the episode-end forks, the writer, its helpers and the librarian. An agent reads new
entries that concern it only at its turn boundary, and addresses others by name or by role (`@writer`, `@librarian`,
`@actor`, `@fork`, `@helpers`). The actor can start teammates with a task, ask its requester a question and ask the
librarian what the library holds.

## Transcripts and logs

**Files:** `unify/transcripts.py`, `unify/logger.py`

Every conversation is one append-only JSONL file under `<UNIFY_HOME>/transcripts`, with an index line when it ends;
credential values are replaced before a line is written. The log files go to `UNIFY_LOG_DIR` (the CLI's default is
`<UNIFY_HOME>/logs`).

## Memory

**Files:** `unify/memory/`, `unify/memory/integration/hooks.py`; see `docs/memory.md`.

`UNIFY_MEMORY` is `off` or `on` (the default). Off is no memory of any kind. On, each request reads a pinned,
read-only copy of the library (functions and linked notes, with their generated `INDEX.md` and `links.json`) and the
captures served beside it. After the request a fork of the actor writes down what it did, and the detached writer
service turns that into tested library items in passes it runs in a confined box; a gate checks every change before
it lands. `docs/memory.md` walks through it.
