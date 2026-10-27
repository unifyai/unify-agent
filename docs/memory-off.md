# Memory off

`UNIFY_MEMORY=off` runs the harness with no memory: every request runs as if nothing had come before it. It is the
baseline that memory (`on`, the default, [memory.md](memory.md)) is measured against. This page says what runs with
memory off, what does not and where each part is switched off. Paths are under `unify/` unless they say otherwise.

## How to run it

```bash
UNIFY_MEMORY=off .venv/bin/unify act "How many rows does data.csv have?"
```

or put `UNIFY_MEMORY=off` in the `.env` file. The setting takes `off` or `on`; unset or empty is `on`, and any other
value is refused when the settings are read, with an error that lists the two (`memory/integration/switch.py:109-117`).
It is read in one place, `memory_mode`; every other reader asks `memory_on` (`switch.py:120-127`). With `off`,
`unify act` skips the check of the writer's model that can refuse a memory run (`cli.py:626-631`) and starts no lane
relay (`lanes/launch.py:161`), so none of the writer's or the lanes' settings is used.

## What runs

Everything that is not memory, with the same code as memory on:

- **The actor.** One model turn at a time, with `execute_code(code)` its only model-API tool; its answer is its final
  reply or `tools.answer(...)` from a cell (`actor/code_act_actor.py:220-283`). Its system prompt is built by the same
  function in both modes (`actor/prompt_builders.py:145-177`); the sentences that differ are quoted below.
- **Tools in code.** Cells call `tools.<name>(...)`, served from the one registry (`common/tool_registry.py`). The
  actor and its teammates keep every tool they have with memory on except the library's: `ask_library` needs the
  librarian flag (`tool_registry.py:949`), which their allow-list gets only with memory on
  (`actor/tool_handlers.py:118-121`). `tools.check_examples` still follows `UNIFY_ACTOR_EXAMPLES_CHECK`
  (`tool_handlers.py:122-124`).
- **Cell state.** Each agent's cells run in one persistent interpreter in the same sandbox, so variables carry over
  from cell to cell. The worker's start message differs only in the memory keys it leaves out
  (`actor/execution/worker.py:252-286`), and its mounts only in the library copy it does not bind (`worker.py:305-309`).
- **Compaction.** Every agent compacts its context once a prompt reaches `COMPACTION_AT_TOKENS` (150,000) or 85 % of
  the model's window (`common/_async_tool/context_compression.py:16-22`); the actor's loop does it before its next
  model call (`common/async_tool_loop.py:268`). Neither reads the memory switch.
- **Step caps.** `UNIFY_MAX_TOOL_LOOP_STEPS` (default 300, `settings.py:65`) bounds a request's messages
  (`common/_async_tool/loop.py:553`), and `UNIFY_MAX_OUTPUT_TOKENS` (default 32768, `settings.py:57`) one turn's
  output.
- **The agents' run record, with the actor and its helpers only.** The record's files are under
  `<UNIFY_HOME>/memory-state/agents/` in both modes (`agents/run_record.py:11-17`, `run_record.py:248-250`): it is the
  team's record, not memory. With memory off, its common section names only the actor and the helpers it starts
  (`run_record.py:192-206`), and `tools.who()` describes each agent's role as actor or helper
  (`tool_registry.py:2103-2109`).
- **Helpers.** The actor and a teammate start helpers with `tools.delegate(task)` (`tool_registry.py:1737-1741`). Each
  is a sub-actor in the same environment as its starter (`agents/binding.py:83-94`), on the run's model
  (`code_act_actor.py:146`), and reads the same memory-off texts as the actor (`tool_registry.py:99-102`).

## What does not run, and where it is switched off

| Part | Where it is switched off | What happens instead |
|---|---|---|
| The request's memory run | `cli.py:295-296`, `cli.py:513`; `memory/integration/hooks.py:37-45` | `begin_request` returns `None`: no memory home is opened, exported or recorded, and no episode is written. |
| Episode and verdict lines | `cli.py:513-516`, `cli.py:489-503`, `cli.py:272-277`, `cli.py:414-428` | No `{"type": "episode"}` line is written. An `{"outcome": ...}` line is answered `"accepted": false` with the reason `this session records no memory`, so no typed outcome is recorded. A `{"checker": ...}` line is no control line at all; with no message in it, it is skipped (`cli.py:446-449`). |
| The fork and the writer | `hooks.py:37-45`; `memory/integration/request.py:1007`, `request.py:1034` | The fork and the writer service are started by the request's memory run, which never begins: no capture, no attention entry, no writer pass, no gate. |
| Memory's model lanes | `lanes/launch.py:161` | `plan` returns no plan: no lane relay, no writer, helper, fork or embedding lanes, and no writer drain when `unify act` ends. |
| The librarian | `memory/librarian.py:87-92`; `code_act_actor.py:262-265` | Never asked: no request-start answer, and no `memory.ask`. |
| The library in cells | `worker.py:252-258`; `hooks.py:81-102` | No import path to a library copy and no `memory` global: `memory` in a cell raises `NameError`. |
| The library copy's mounts | `worker.py:305-309`; `hooks.py:202-216` | No library is bound, read-only or otherwise. |
| Audit hooks | `worker.py:277-286`; `hooks.py:218-229`, `hooks.py:330-336` | The worker's start message has no `no_bytecode` and no `audit` key, so the child installs no audit hook, and a finished cell's audit records go nowhere. |
| The tool-result hook | `common/_async_tool/tools_data.py:72-78`; `hooks.py:278-286` | `None`: the loop never enters it, so a tool call is processed exactly as without memory. |
| Memory's prompt parts | `hooks.py:48-60`; `code_act_actor.py:236-247` | The system prompt gets no memory guide, and the first user message no library line and no standing-note digest. |
| The library tool | `tool_handlers.py:118`; `tool_registry.py:949` | No librarian flag, so no `ask_library`. |
| The record's memory agents | `run_record.py:192-206`, `run_record.py:288-303`; `tool_registry.py:1957-1962` | No writer, fork or librarian on the record, and the common section names none (quoted below). |

The common section's check also reads the record's roster: when it already lists a writer, fork or librarian, or the
librarian's role is seeded under that home, an agent reads the full section (`run_record.py:297-303`).

## What the actor and its teammates read differently

Every other byte of their prompts and tool texts is the same in both modes. With memory off:

**The role's first sentence** (`prompt_builders.py:19-27`, chosen at `prompt_builders.py:105`). Off, `_ROLE_FIRST`:

```
You solve the request in this conversation by writing and running
Python cells with `execute_code`.
```

On, `_ROLE_FIRST_MEMORY`:

```
You solve the request in this conversation by writing and running
Python cells with `execute_code`. A library of functions and notes from
earlier work, when one exists, is `memory` in your cells, already bound.
```

**The record's common section** (`tool_registry.py:1948-1962`, chosen at `tool_registry.py:1964-1967`;
`COMMON_SECTION_NO_MEMORY`, `run_record.py:194`). Two passages of the on text are replaced. On:

> the actor doing each request, the fork that reviews each episode after it, the librarian when the run has one, the
> writer that keeps the library, and the helpers any of them starts. The librarian answers `memory.ask`; an episode's
> fork runs after its episode, and the writer reaches it as @fork.

Off:

> the actor doing each request and the helpers it starts.

On:

> a role: @writer; @actor (every running actor); @helpers (every running helper).

Off:

> a role: @actor (every running actor); @helpers (every running helper).

**The names bound in cells** (`tool_registry.py:2893-2901`, from `actor/in_code_tools.py:253-261`). On: "`tools` and
`memory` are already available as variables in every cell; no import needed." Off: "`tools` is already available as a
variable in every cell; no import needed."

**Tool texts** (`tool_registry.py:105-112`: a run without memory reads a tool's off text first).

- `tools.read` (`tool_registry.py:1444-1474`, used at `tool_registry.py:1593-1603`). On, its summary is "read a file or
  list a directory under /workspace or of the library (/memory-library), 8000 bytes a page." Off: "read a file or list
  a directory under /workspace, 8000 bytes a page." On, a path outside its roots is refused with "{path} is outside
  /workspace and the library (/memory-library, read-only); an entry of the library opens with memory.get(<its id>)";
  off with "{path} is outside /workspace".
- `tools.grep` (`tool_registry.py:1513-1540`, used at `tool_registry.py:1617-1626`). On, its summary is "search files
  with a regular expression (default under /workspace; the library under /memory-library)." Off: "search files with a
  regular expression (default under /workspace)." Its refusal of a path outside its roots changes as `tools.read`'s
  does.
- `tools.send` (`tool_registry.py:2010-2033`). On, its description's addresses are "@<name>, or a role: @writer, @actor
  (every running actor), @helpers (every running helper); the library is asked with `memory.ask`." Off: "@<name>, or a
  role: @actor (every running actor), @helpers (every running helper)." On, its example is
  `tools.send("@writer the served item gave a wrong total here: action:7")`; off,
  `tools.send("@helpers the totals are on page 2 of the listing: action:7")`.
- `tools.who` (`tool_registry.py:2095-2109`). On, each agent's "role (actor, fork, librarian, writer or helper)"; off,
  "role (actor or helper)".

**Parts that are only there with memory on** and absent with it off: the memory guide that ends the system prompt
(`hooks.py:48-60`, built at `memory/integration/request.py:1177`), the library line and the standing-note digest that
open the first user message (`code_act_actor.py:236-247`) and the librarian's answer that ends it
(`code_act_actor.py:262-265`). With memory off, the first user message carries only the record's section before the
request (`code_act_actor.py:252`).

## Cost

Only the actor's model calls: its turns, its teammates' turns (on the same model), the model calls its cells make
(`query_llm`, `unillm`) and its compaction summaries. Memory makes no calls of its own (no writer, writer's helpers,
fork, librarian or embeddings), and nothing is counted against `UNIFY_MEMORY_CAP_USD`, because no lane relay or writer
starts (`lanes/launch.py:161`, `hooks.py:37-45`).

## What it is the baseline for

The same harness with memory taken out, and nothing else changed: the same actor, model, sandbox, tools (less the
library's), step caps and compaction. Running the same requests with `UNIFY_MEMORY=on` and with `UNIFY_MEMORY=off`
measures what memory adds and what it costs (`switch.py:16-17`). `tests/system/test_s06_memory_off.py` checks that a
memory-off actor and teammate are shown no memory text, that no memory agent runs and that nothing of memory is
written, and that their tools with memory on are the memory-off tools plus memory's own.

## Side by side

| Component | `UNIFY_MEMORY=on` (default) | `UNIFY_MEMORY=off` |
|---|---|---|
| Actor, tool loop, sandbox, cell state | yes | yes, the same |
| Compaction and step caps | yes | yes, the same |
| Helpers (`tools.delegate`) | yes | yes |
| Run record | actor, helpers, forks, writer, librarian | actor and helpers |
| Episode recorded (`episodes.git`, evidence store) | yes | no |
| `{"outcome": ...}` and `{"checker": ...}` lines | taken | outcome refused; checker not a control line |
| Fork at episode end, captures | yes | no |
| Attention ledger, writer service, gate | yes | no |
| Library copy bound as `memory` in cells | yes | no (`NameError`) |
| Librarian, `memory.ask`, `ask_library` | yes | no |
| Memory guide, library line, standing-note digest | yes | no |
| Lane relay and embeddings | yes, with `OPENROUTER_API_KEY` | no |
| Model spend | actor, plus memory under `UNIFY_MEMORY_CAP_USD` | actor only |
