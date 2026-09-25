<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/unifyai/unify/main/assets/brand/unify-readme-banner-dark.png">
    <img src="https://raw.githubusercontent.com/unifyai/unify/main/assets/brand/unify-readme-banner-light.png" alt="unify" width="100%">
  </picture>
</p>

<p align="center">
  <a href="https://unify.ai">unify.ai</a> · <a href="https://discord.com/invite/sXyFF8tDtm">Discord</a> · <a href="LICENSE">MIT license</a>
</p>

# unify

**unify is a self-improving agent harness for research on how an assistant learns tasks and skills. It runs entirely on your machine: a persistent conversation loop above a code-writing actor, two skill libraries the actor consults before it writes code and distils into after work that went well, and a steering protocol that reaches into any running operation. One LLM key and `python -m unify`.**

The repo is deliberately small. Everything that is not the harness itself has been cut, so that what is left can be studied, changed and measured: how skills are discovered, how a run is distilled into a function or a procedure, how a correction reaches work that is already running, and how all of that composes when actors nest.

## The design

Seven decisions, each of which the rest of the code serves:

- **Everything is code.** The actor writes one Python program per turn in a persistent sandbox. Files are read where they are, shell commands run through `subprocess`, packages install into one workspace environment, and nested actors are spawned from code with `primitives.actor.act(...)`. There is no tool per capability.
- **Two skill libraries, discovered first.** Functions are the *what*: executable Python with a docstring and pip dependencies. Guidance is the *how*: procedures for composing them. The actor searches both before it writes new code.
- **A run is distilled, not remembered.** After a run completes, a storage review reads the trajectory and decides whether anything is worth keeping: a callable that worked becomes a function, a non-obvious composition becomes guidance. Often nothing is. That review is the only way the libraries grow.
- **A conversation loop above the actor.** A slow brain stays present across the conversation and keeps deciding whether to speak or wait while work is in flight. Dispatching work and replying to you are different decisions, made by different loops. This is the interaction-model split [described by Thinking Machines](https://thinkingmachines.ai/blog/interaction-models/), arrived at in the harness rather than in the model.
- **Steering is a protocol.** Every operation returns the same steerable handle: `ask`, `interject`, `pause`, `resume`, `stop`. Handles nest, so a correction made in chat propagates down the live call stack into whatever loop is running, and a clarification from the innermost loop bubbles up to you. Corrections reach code that is already executing, not just the next tool boundary.
- **One process, one file.** Runtime, chat history and both libraries run in one process against one SQLite file under `~/.unify`. No services, no accounts, no infrastructure.
- **Reactive.** Every piece of work starts from a message you sent or from work that message started. There is no scheduler and no inbound channel other than the chat.

unify shares the stance of [Prime Agent](https://github.com/PrimeIntellect-ai/prime-agent)'s recursive language model: a persistent Python control environment, subagents as function calls, skills as importable code, and harness state that improves with use. It differs in the conversation layer above the actor, in steering as a first-class protocol, and in keeping everything in one process over one store.

---

## Install

**Prerequisites:** Python 3.12+, [uv](https://docs.astral.sh/uv/), and one LLM provider key (OpenRouter, Anthropic, or DeepSeek; OpenAI models are reached through OpenRouter). macOS, Linux, or WSL2.

`unify` and its LLM client `unillm` are sibling checkouts linked by an editable install:

```bash
git clone https://github.com/unifyai/unify.git
git clone https://github.com/unifyai/unillm.git
cd unify
uv sync --all-groups
cp .env.example .env      # add your provider key
```

Then start chatting:

```bash
.venv/bin/python -m unify
```

```text
> Here's the benchmark spreadsheet, which configs regressed?
> Turn that regression check into something you can rerun next week.
> Run it against ~/exports/run-42.csv and plot the deltas.
```

To measure the actor on its own, without the conversation loop above it, hand it one request directly:

```bash
.venv/bin/python -m unify act "Count the rows in ~/exports/run-42.csv"
```

Progress streams to stderr and the result to stdout, so it slots into a benchmark runner. The request can come from stdin, `--json` adds the token accounting, `--no-store` skips the storage review, `--persist` keeps the sandbox alive for follow-up lines, and a question the actor asks is answered by typing at the terminal. This is the like-for-like unit against single-loop harnesses such as Prime Agent; the conversation loop is what unify adds on top.

Everything the assistant keeps lives under `~/.unify/` (`UNIFY_HOME`): the SQLite store, the embeddings cache, the workspace environment, the `workspace/` directory the actor reads and writes files in, and the runtime logs. Delete the directory and you have a fresh assistant. `/help` inside the chat lists the few slash commands (attach a file, quit); `unify --debug` streams the runtime logs to the terminal.

<details>
<summary>Configuration</summary>

`.env` holds the whole configuration; `.env.example` documents every key. The ones that matter:

| Variable | Purpose |
|---|---|
| `OPENROUTER_API_KEY` / `ANTHROPIC_API_KEY` / `DEEPSEEK_API_KEY` | At least one provider key |
| `UNIFY_MODEL`, `UNIFY_REASONING_EFFORT` | The default model (a unillm `model@provider` endpoint) and effort |
| `UNIFY_HOME` | Where the store, embeddings cache, environment and workspace live (default `~/.unify`) |
| `UNIFY_STORE_PATH` | An explicit path for the SQLite store |
| `UNIFY_EMBED_MODEL` | The `<model>@openrouter` embedding model for skill search (default `openai/text-embedding-3-small@openrouter`) |
| `UNILLM_CACHE` | Cache LLM responses locally; later runs replay identical calls |
| `ASSISTANT_FIRST_NAME`, `USER_FIRST_NAME`, … | Optional identity for the assistant and its user |

</details>

---

## Steering while work is in flight

When the assistant is mid-task, steer it the way you would steer a colleague: send another message.

```text
> Actually, narrow it to ones with Rust bindings.
> What step are you on?
> Pause that, something urgent.
```

Each message wakes the slow brain, which can answer you directly or redirect in-flight work through its steering tools (`interject_action`, `ask_action`, `pause_action`, `resume_action`, `stop_action`, `answer_clarification_action`), each addressed by the action's id. Nothing restarts; the correction propagates down the live call stack into whatever loop is currently running.

```text
You          ▸  "Find me high-throughput vector DBs under Apache 2."
Assistant    ▸  (start searching)
You          ▸  "Actually, narrow it to ones with Rust bindings."
Assistant    ▸  (adjust the in-flight search, don't restart)
You          ▸  "Pause that, something urgent."
Assistant    ▸  (freeze exactly where they are)
... five minutes later ...
You          ▸  "OK, resume. How's it going?"
Assistant    ▸  (pick up where they left off, give you a status update)
```

Several actions can run at once, each independently inspectable, steerable and pausable:

```text
┌─ In-Flight Actions ────────────────────────────────┐
│  [0] watch_pr_reviews    ██████████░░░  In progress │
│  [1] digest_releases     ████████████░  In progress │
│  [2] retry_failed_build  ██░░░░░░░░░░  Starting     │
└─────────────────────────────────────────────────────┘
```

---

## How it works

```text
You ──► ConversationManager (slow brain: event-driven, single-shot tool decisions)
            │
            │  act(...) / interject / ask / pause / resume / stop
            ▼
        CodeActActor (writes one Python program per turn in a persistent sandbox)
            │
            ▼
        Skill libraries (stored functions + procedures, discovered before writing code)
            │
            ▼
        unify.db (one SQLite file: functions, primitives, guidance, builtin_guidance, messages)
```

**Dispatch flows down; steering flows back up the same path.** Every level returns the same `SteerableToolHandle`, so a mid-flight redirect doesn't abort the run, doesn't append a second prompt, and doesn't wait for the next tool boundary. It propagates through the live nested call stack as a typed signal any inner loop can act on.

### Steerable handles

```python
handle = await actor.act("Survey high-throughput vector DBs and draft a comparison")
await handle.interject("Only ones with Rust bindings")   # mid-flight redirect
await handle.pause(); ...; await handle.resume()         # freeze and resume
answer = await handle.ask("what step are you on and why?")  # a read-only inspection loop over the live transcript
```

When the actor calls `primitives.actor.act(...)`, the nested actor returns its own handle, nested inside the actor's, which is nested inside the `ConversationManager`'s. Steering at any level propagates down through the live call stack. When an inner loop hits genuine ambiguity, its clarification bubbles up through every intervening layer to you, and your answer flows back down to the loop that asked, without unwinding the stack. `stop()` on one nested branch leaves its siblings running.

### CodeAct: the actor writes programs

Most agents emit one JSON tool call at a time and let the LLM stitch results across turns. unify's actor writes a single Python program per turn in a persistent sandbox, calling stored functions and nested actors from code:

```python
rows = load_orders("~/exports/orders.csv")          # a stored function, discovered first
by_status = {}
for row in rows:
    by_status[row["status"]] = by_status.get(row["status"], 0) + row["amount"]
note = await primitives.actor.act(f"Write a short note explaining {by_status}")
```

A load, reshape, delegate sequence becomes one plan with real variables, loops and control flow, rather than separate tool-selection turns round-tripping through tool messages. Corrections reach a running program too: an AST pass adds probes at function entry and loop iteration, dispatches are memoised, and a patched function is spliced into the source and re-run from where the change first matters.

### Functions and guidance

Two libraries the actor consults before reaching for raw code:

- **Functions**: executable Python with a docstring and pip dependencies, run in-process with dependencies ensured in the workspace environment.
- **Guidance**: procedural how-to prose (walkthroughs, multi-step strategies), linked to the functions it composes.

Search is semantic: functions (name, signature, docstring) and procedures (title, content) are embedded as they are written, with `openai/text-embedding-3-small` through OpenRouter and the same key as the LLM calls, and a query is ranked by cosine similarity, so a paraphrase finds a skill that words the task differently. Without an OpenRouter key search falls back to a plain word match, and the log says so. Skills in the [Agent Skills](https://agentskills.io) format import as guidance through `scripts/skill_migration`.

### The local store

`unify.db` is one SQLite file with five tables: `functions`, `primitives`, `guidance`, `builtin_guidance` and `messages`. The `primitives` table is seeded from the primitive registry. The `builtin_guidance` table is seeded from the committed Agent Skills snapshot. Two views, `all_functions` (`is_primitive`) and `all_guidance` (`is_builtin`), read each seeded catalogue alongside the assistant's own rows. Managers write plain SQL. Filters the model writes are SQL `WHERE` clauses (`docstring LIKE '%csv%' AND is_primitive = 0`) run through a read-only authorizer, so a clause can only read.

Chat history, functions and procedures all live in this one file, so the whole assistant is something you can back up, inspect with `sqlite3` or delete.

For the full breakdown (async tool loop internals, event bus, primitive registry, context propagation) see [`ARCHITECTURE.md`](ARCHITECTURE.md).

---

## The runtime stack

| Repo | Role |
|------|------|
| **unify** (this) | The harness: conversation loop, actor, skill libraries, the store |
| **[unillm](https://github.com/unifyai/unillm)** | LLM access layer: OpenRouter, Anthropic, DeepSeek, or any compatible endpoint, with response caching |

---

## Running the tests

Tests exercise the real system (steerable handles, CodeAct, nested tool loops, the storage review) against a per-session SQLite store with cached LLM responses. The LLM is never mocked: responses are cached per exact input, so a first run is slow and later runs replay in milliseconds.

```bash
uv sync --all-groups

tests/parallel_run.sh -s -j 36 tests/           # everything, one session per file
tests/parallel_run.sh tests/actor/              # one area, one session per test
```

See [tests/README.md](tests/README.md) for the philosophy and the runner.

---

## Where to start reading

| File | What's there |
|------|-------------|
| `unify/common/async_tool_loop.py` | `SteerableToolHandle`: the protocol everything returns |
| `unify/common/_async_tool/loop.py` | The async tool loop engine: nesting, steering, context propagation |
| `unify/actor/code_act_actor.py` | CodeAct: plan generation, sandbox, the storage review |
| `unify/function_manager/steering.py` | Steering code that is already running |
| `unify/conversation_manager/conversation_manager.py` | The slow brain: debouncing, in-flight actions, event loop |
| `unify/conversation_manager/domains/brain_action_tools.py` | How the brain starts, steers and tracks concurrent work |
| `unify/db.py` | The local store: the five tables, the two views, the read-only path for model-written SQL |
| `unify/environment.py` | The workspace environment: one venv, packages installed once |

## Project structure

```text
unify/
├── unify/             # The harness: cli, actor, conversation_manager, function_manager, guidance_manager, db, common
├── tests/             # Pytest suite (cached LLM responses, per-session SQLite store)
├── scripts/           # Skill import, git hooks
└── docs/              # Design writeups
```

## License

MIT. See [LICENSE](LICENSE).
