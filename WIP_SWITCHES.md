# Switches still in progress

At the code freeze of 7 October 2026 every other research switch was either made the default behaviour or set to be removed. The switches below stay as switches, at the defaults shown, until the experiment named for each one decides them. Each is read from `unify/settings.py`, where its comment gives the full behaviour, and each is set through the environment variable of the same name.

## UNIFY_CODE_PROJECTION

- **Values:** empty (the same as `legacy`) or `notebook`.
- **Default:** empty.
- **What it does:** With `notebook`, `execute_code` takes a single `code` field, and where a cell runs is written inside the cell as Jupyter magics on its first lines (for example `%%bash` or `%%scratch`) instead of as separate arguments. The cell's result then reads like a notebook cell: its output, its errors and its last value.
- **Evidence:** This is the "code alone" schema of design B, which the 5 October Python-tool-mode debugging lane ranked first. Its Stage 2/3 screen had not reported when the freeze was made. The proposal was to make `notebook` the default if the Stage 3 readout is not worse.

## UNIFY_STATEFUL_CELLS

- **Values:** a boolean.
- **Default:** off.
- **What it does:** When on, every `execute_code` cell runs in the task's one persistent session, like a notebook, and the tool no longer offers a stateless mode or a choice of session. The session-management tools are also withdrawn.
- **Evidence:** This switch is an arm of the E4 workbench screen (build de97caa57; preregistration `workbench-screen-v1/PREREG-v1.md`). Its canaries passed on ARC and AppWorld, but no counted readout existed at the freeze. The proposal was to turn it on if the readout is inconclusive, because the change only removes options and no measured run had made a stateless choice useful. Under the core tool surface, the choice it removes is already small.

## UNIFY_VARIABLE_INVENTORY

- **Values:** empty (the same as `off`) or `on`.
- **Default:** empty.
- **What it does:** When on, a cell that ran in a persistent session ends its result with one line naming the variables the session has bound, each with its type and a short description of its shape. The line appears only when those names or shapes have changed since the last one shown.
- **Evidence:** This switch is an arm of the E4 workbench screen (the W arm against the W-without-inventory arm). Its canaries passed, but no counted readout existed at the freeze. It is to be removed unless E4 shows a gain.

## UNIFY_BIND_REQUEST

- **Values:** empty (the same as `off`) or `on`.
- **Default:** empty.
- **What it does:** When on, code in a cell can read the current request as `request`: `request.text` is the requester's latest message, and `request.data` is the list of JSON objects and arrays found in it. The object is read-only, and a fresh copy is given to each cell.
- **Evidence:** This switch is an arm of the E4 workbench screen. No counted readout existed at the freeze. It is to be removed unless E4 shows a gain.

## UNIFY_REPLY_CHANNEL

- **Values:** empty (the same as `text`) or `code+text`.
- **Default:** empty.
- **What it does:** With `code+text`, code in a cell can send the turn's reply by calling `reply(text)`. The cell ends at once and the text becomes the reply, without another model call.
- **Evidence:** This switch is in the E4 workbench screen. In the earlier WI-2 screen it scored 11 of 25 on a single run, and a trace review judged that difference to be noise rather than a defect. It is to be removed unless E4 shows a gain.

## UNIFY_CELL_SCOPE_FIX

- **Values:** a boolean.
- **Default:** on.
- **What it does:** A cell runs as the body of a wrapper function, so a name the cell binds outside a plain top-level assignment used to be lost when the cell ended. When on, every name the cell's own scope binds is kept in the session, as the prompt promises, including names bound inside `if`, `for`, `with` and `try` blocks.
- **Evidence:** This is a bug fix and is already on by default. In measured runs, where about 95% of cells were stateless, it never mattered. Its exposure rises when every cell is stateful.

## UNIFY_STEP_CAP_COMPACT

- **Values:** empty (the same as `off`), `on` or `continue`.
- **Default:** empty.
- **What it does:** When on, a task loop that can compress its context compacts it when it reaches its step limit, instead of stopping, and carries on with the same request. This happens at most twice per request. At the third limit, the request stops as it would without the switch.
- **`continue`:** one long-horizon mode for the actor's task loop, the one that answers the requester (every other loop runs as shipped). Each part keys only on the request and the session, never on a task stream:
  - The step budget (`UNIFY_MAX_TOOL_LOOP_STEPS`, a count of messages) is counted per request, from the requester's message, and afresh after each compaction.
  - At the limit, the conversation is compacted as under `on`, with no bound on compactions per request, and the request goes on. The summary is a loop-authored message, so it starts no request and `UNIFY_LOOP_STOP`'s count carries across it.
  - A compaction whose rebuilt context is not smaller than the one it replaced (serialised message characters) is ineffective. A second ineffective compaction in a row in one request ends that request.
  - That end, a compaction that fails, a loop stop and the loop's timeout all end the request through `UNIFY_STEP_CAP_REPLY`'s reply path, in its `draft` mode when that switch is empty. The requester always gets an answer, and a persistent session waits for its next request.
- **Evidence:** long-horizon WIP, unscreened; to be tested on the long-horizon beds.

## UNIFY_COMPACTION_KEEP_PREFIX

- **Values:** empty (the same as `off`) or `on`.
- **Default:** empty.
- **What it does:** When on, a context compaction keeps what the session already sent at the start of the conversation, byte for byte: the system prompt, the session's first user message and every requester message of the current request, unchanged and in their original order. The summary follows them as one loop-authored message, and the tools stay the same. As shipped, the conversation restarts from the system prompt and the summary alone, so the request's own words are replaced by the model's paraphrase of them, and only the tools and the system prompt stay cached. The current request is read from the session's own messages: it starts at the latest requester message. If the first call after such a compaction is still over the compression threshold, what was kept is too large on its own, and the next compaction rebuilds as shipped from the summary alone. That way a request is never compacted around the same prefix again and again. "Byte for byte" covers the system prompt and the kept user messages. The loop's runtime-context system messages are rebuilt at the restart: they are the same for the actor, but not for a nested loop given a parent chat context.
- **Evidence:** long-horizon WIP, unscreened; part of the cache-preserving compaction design of 8 October, to be tested on the long-horizon beds.

## UNIFY_LOOP_STOP and UNIFY_LOOP_STOP_K

- **Values:** `UNIFY_LOOP_STOP` is empty (the same as `off`) or `on`. `UNIFY_LOOP_STOP_K` is a whole number of at least 1.
- **Defaults:** empty and 10.
- **What it does:** When on, a request ends early once the model has made `UNIFY_LOOP_STOP_K` calls in a row that make no progress. A call makes no progress when it runs a cell that does nothing, or when it repeats one of the two calls before it and gets the same result. The request then ends the way the step limit ends it under `UNIFY_STEP_CAP_REPLY`.
- **Evidence:** long-horizon WIP, unscreened; to be tested on the long-horizon beds.

## UNIFY_STEP_CAP_REPLY

- **Values:** empty, `draft` or `last_word`. The boolean spellings are also accepted, with true meaning `draft`.
- **Default:** empty.
- **What it does:** With `draft`, reaching the step limit in a persistent session ends only the current request. The reply quotes the latest reply text the model drafted, and the next message starts a new request with its own step budget. With `last_word`, the model first gets one call with no tools to give its best answer.
- **Evidence:** long-horizon WIP, unscreened; to be tested on the long-horizon beds.

The three long-horizon switches share the step-limit reply path.

## UNIFY_MEMORY_V2

- **Values:** empty (the same as `off`) or `on`.
- **Default:** empty.
- **What it does:** When on, no storage review runs, and the `functions` and `guidance` objects and the library shortlist are gone. Each request instead imports from a scratch export of the memory repo's `main`, mounted read-write in the worker and discarded after the request, and sees the memory index at the end of the system prompt. The index is rendered only from the memory commit, so two requests on the same commit send the same prompt prefix. The whole request (its messages, replies, cells, actions, the work tree's before and after snapshots and what it wrote to the export) is recorded as one episode in `<UNIFY_HOME>/episodes.git`; only the checker's pass/fail is kept from a posted outcome. After the request, due consolidation passes by Sol run behind a deterministic gate, blocking before the session ends; each pass writes a start and an end event (`--jsonl` output and `<UNIFY_HOME>/memory-v2/events.jsonl`).
- **Evidence:** matched screens pending (memory-v2 preregistrations, continual-harness-research). Design: `docs/design/memory-redesign-spec.md` there.

## UNIFY_MEMORY_V2_E, UNIFY_MEMORY_V2_SOL_MODEL, UNIFY_MEMORY_V2_SOL_ALLOWANCE_USD_PER_TOKENS, UNIFY_MEMORY_V2_SOL_RUN_GUARD_USD and UNIFY_MEMORY_V2_SOL_EFFORT

- **Values:** `UNIFY_MEMORY_V2_E` is a positive whole number of tokens. `UNIFY_MEMORY_V2_SOL_MODEL` is a model id. `UNIFY_MEMORY_V2_SOL_ALLOWANCE_USD_PER_TOKENS` is a positive plain decimal string. `UNIFY_MEMORY_V2_SOL_RUN_GUARD_USD` is a non-negative plain decimal string, or empty. `UNIFY_MEMORY_V2_SOL_EFFORT` is `actor`, `low`, `medium` or `high`. Exponent forms (`7.3e-7`) are refused.
- **Defaults:** `150000`, `openai/gpt-6-sol`, `0.00000073`, empty (no guard) and `actor`.
- **What they do:** They apply only with `UNIFY_MEMORY_V2=on`. The trigger is size-based and batched (the spec's F1): one pass becomes due once the experience recorded since the last pass reaches E tokens, and it covers every channel with new evidence. E times the allowance is one pass's USD cap. The passes run on the Sol model at the actor's reasoning effort for the run (`actor`, the default; the lead, 8 Oct); a fixed `low`, `medium` or `high` is for a declared mismatch ablation only. With a run guard set, no further pass starts once the run's committed Sol USD plus the next pass's cap would exceed it.
- **Evidence:** E and the allowance follow the offline cadence replay and the memory-v2 preregistrations (continual-harness-research); the guard is a runaway stop, not the expected cutoff.

## UNIFY_MEMORY_V2_SOL_BASE_URL, UNIFY_MEMORY_V2_SOL_TOKEN and UNIFY_MEMORY_V2_SOL_TOKEN_FD

- **Values:** both empty, or both set: `UNIFY_MEMORY_V2_SOL_BASE_URL` an http(s) URL with a host and no user, password, query or fragment (plain http only to exactly `127.0.0.1` with an explicit port, the launcher's loopback bridge; https to any host); `UNIFY_MEMORY_V2_SOL_TOKEN` a bearer token of at least 16 URL-safe characters, `A-Z a-z 0-9 . _ ~ -` (a secret; `secrets.token_urlsafe` output fits).
- **Where to set them:** in the controller process's own environment, never in `.env`: settings are read before the CLI loads `.env`, so a value found there (or under another letter case, with a different value) refuses every pass with an error naming the setting (checked again whenever a pass is about to start, so an embedder that loads `.env` late is refused too), and its token is removed from the environment. Hand the token to the controller process only, not to a shared driver environment (`/proc/<pid>/environ` keeps a process's initial environment).
- **Token by descriptor:** `UNIFY_MEMORY_V2_SOL_TOKEN_FD=<n>` (a descriptor number above 2) replaces `UNIFY_MEMORY_V2_SOL_TOKEN`; set exactly one of the two (both refuse every pass), and set it with `UNIFY_MEMORY_V2_SOL_BASE_URL`. The launcher passes the controller an inherited descriptor (a pipe or a file) holding the token, optionally followed by one newline, and closes its own write end; the environment carries only the number. The controller reads it once when its settings are first settled (at most 4096 bytes and 5 s), closes it, and holds the token outside `os.environ`, registered with the redactors. A descriptor that is not open, was not inherited, is neither a pipe nor a file, or holds nothing, more than 4096 bytes or not a bearer token refuses every pass the same way, naming the rule, never the value.
- **Defaults:** empty (Sol's calls go as shipped).
- **What they do:** They apply only with `UNIFY_MEMORY_V2=on`. Set, Sol's model calls go to that OpenAI-compatible base URL with that token, and carry `X-Unify-Call-Kind: memory_v2.sol`: a proxy listener of Sol's own, so the actor's route (and its allow-list) never carries Sol's model. Exactly one set, or a value in a wrong form, starts no pass (the request stays due); no error quotes either value, and each such request appends `{"type": "consolidation", "phase": "refused", "consolidation_refused": "route_not_in_effect", ...}` to the run's `events.jsonl` (and the `--jsonl` stream), so the run is flagged rather than read as a null result. With `UNILLM_OTEL` on, the route is refused the same way (unillm's spans keep error text as is). The token is read into the controller's settings only, removed from its environment (every letter case) and kept registered with the value-based redactors; a failed Sol call is recorded as its exception class and a fixed category, never its text, and unillm's own copies of that text (its `unillm.retry` warnings and Sol's per-call `UNILLM_LOG_DIR` file) are redacted. Cells and Sol's box never receive either name. The header enters unillm's response-cache key, so a cache recorded with the route unset misses with it set (unset sends no header, so unset recordings are unaffected).
- **Evidence:** a deployment route for the paired memory-v2 screens, not a behaviour under test.

## UNIFY_MEMORY_V2_SURFACING, UNIFY_MEMORY_V2_DOCSTRINGS and UNIFY_MEMORY_V2_SOFT_BUDGET

- **Values:** `UNIFY_MEMORY_V2_SURFACING` is `index` or `catalogue`; `UNIFY_MEMORY_V2_DOCSTRINGS` and `UNIFY_MEMORY_V2_SOFT_BUDGET` are `off` or `on`. Empty means the default.
- **Defaults:** `index`, `off` and `off`: the v2 screen build's behaviour (`9deefbfd1`), so a paired v2 vs v2.1 comparison runs on one build.
- **What they do:** They apply only with `UNIFY_MEMORY_V2=on` (memory v2.1, lane S1). `index`: the system prompt ends with the v2 per-function index and the export line, and Sol's first message carries the index. `catalogue`: the prompt ends with one constant guide paragraph (the same bytes for the whole run, from the first request whose library lists anything; no count, channel, function, drift flag or path), and memory is discovered in cells: every export holds a generated `README.md`, `.memory/catalog.json`, `.memory/shapes.py` and a `memory` helper (`import memory; print(memory.catalog())`, `memory.find(value)`, `memory.describe(name)`), input shapes are recorded at each merge and frozen per commit, a suspect channel's refusal says so in the cell's error, and Sol's first message carries the README. The use record (memory-v2.1-tele) then records the guide as shown and no function or channel, since the guide names none; per-function exposure comes from the cells. `UNIFY_MEMORY_V2_DOCSTRINGS=on`: the gate's lean docstring standard (G1) and its examples run as doctests (G3), stated in Sol's brief. `UNIFY_MEMORY_V2_SOFT_BUDGET=on`: G4 notes that hygiene is due past the 4,000-token budget instead of refusing growth. Not switched (a safety fix in every mode): the gate refuses bytecode, native code, start-up hooks and root entries outside the layout, and changes to the paths reserved for the generated catalogue.
- **Evidence:** none yet (memory v2.1 stage 1-2); offline replay (`memory_v2_offline/cadence_replay.py` builds the gate from these switches) and a paired screen pending (`docs/design/memory-v2.1-plan.md`, continual-harness-research).

## UNIFY_MEMORY_V2_SOL_USAGE

- **Values:** empty (the same as `off`) or `on`.
- **Default:** empty.
- **What it does:** It applies only with `UNIFY_MEMORY_V2=on`. When on, each consolidation pass's first message ends with a table of how the pass's requests used each library function: requests whose pin held it, requests whose prompt showed its own index line, requests whose prompt showed its channel, requests that called it, call sites, refusals (`MemoryInputError` raised out of the function), refusals followed by a successful action on its channel, other errors, dynamic calls in its channel and requests since its last call. Refusals and errors from a request that edited the function's channel in its scratch copy are left out (they are recorded apart, as the edit's). Refusals and other errors read `k of n known (+u unknown)`: requests with one, of the requests whose outcome for the function was known, and the requests whose outcome for it was unknown because a cell with no recorded outcome (cancelled, timed out, stopped at the step cap, a tool call that raised, an executor failure) could reach it; such a cell leaves unknown only the functions its code imports, calls or references, or whose channel it uses dynamically, in that request only. A "then accepted" count marked `+?` is at least that number. Closing notes say how many requests had such a cell, how many requests' shown counts came from the harness's record of what it rendered rather than from the prompt's text (legacy) or nowhere (no record, or no system prompt recorded), and how many recorded prompts did not end with the recorded section. The numbers come from the harness's use record (`memory_use.json` per episode, the evidence store's `item_use` table), never from a checker. When off, the first message is byte for byte as before. The use record is kept either way.
- **Evidence:** none yet (memory v2.1 stage 1); offline comparison and a paired screen pending (`docs/design/memory-v2.1-plan.md`, continual-harness-research).

## UNIFY_MEMORY_V2_QA_FIXTURES, UNIFY_MEMORY_V2_QA_MUTATION, UNIFY_MEMORY_V2_QA_MUTATION_MIN_KILL, UNIFY_MEMORY_V2_QA_DETERMINISM, UNIFY_MEMORY_V2_QA_REPLAY and UNIFY_MEMORY_V2_QA_FIXTURE_SIZE

- **Values:** `UNIFY_MEMORY_V2_QA_FIXTURES` is empty (the same as `off`), `on` or `strict`. `UNIFY_MEMORY_V2_QA_MUTATION_MIN_KILL` is a plain decimal string from 0 to 1. The others are empty (the same as `off`) or `on`.
- **Defaults:** all empty (off), and `0.5`.
- **What they do:** They apply only with `UNIFY_MEMORY_V2=on`, inside the consolidation gate (memory v2.1 stage 5, `unify/memory_v2/qa.py`), and judge the quality of the tests Sol writes. With `_QA_FIXTURES`, the gate draws a seeded random sample of recorded inputs of each new or changed function's family (the seed is the candidate commit) and calls the function on each: it must return or raise `MemoryInputError`, and refusing an input shaped like its covers refuses the pass. Arguments beyond the input come from the recordings only. Tests parametrised over `memlab.inputs.inputs` also run on the drawn inputs; `strict` refuses a drawn input no test of the function read. With `_QA_MUTATION`, the function's tests must kill at least `_QA_MUTATION_MIN_KILL` of the mutants that change its outputs on the recorded inputs or on structurally broken copies of its covers (a field dropped, retyped, emptied or added; at most 2 MiB of them per function); a mutation check the probe could not judge is refused. With `_QA_DETERMINISM`, every gate test run pins the clock, the hash seed and `random`, and each new test file runs a second time under different pins (epoch, clock step, seeds, time zone); different outcomes are refused (flaky, or dependent on the pinned values). With `_QA_REPLAY`, a test passing an environment function a stand-in of its own instead of the kit's replay (`memlab.replay.env_from` or `RecordedEnv`: exact-call replay keyed by channel, method, arguments and sorted keyword arguments; identical calls answered in recorded order, the last recording repeating past the end and listed in `env.repeats`; an unrecorded call raises, a recorded error re-raises, `env.issued()` lists the calls served so a test holds a function to exactly its own recorded calls; an effect filter raises where a recorded effect is `unknown`, as for every dialogue action) is refused; a root `unify_memory_testkit.py` that re-exports it is accepted, one faking the environment is not. With any switch on, Sol's brief points at `env_from` and no longer asks for a root test kit. With `_QA_FIXTURE_SIZE`, a test or data file over 64 KiB is refused, recorded payloads are referenced by blob id, and a test asserting on the cut of a truncated recording is refused. Any of them also mounts the library test kit (`unify/memory_v2/testkit.py`: `memlab`, the pin plugin, the referenced blobs) at `/inputs` in the gate's runs and adds the matching paragraph to Sol's brief. The kit is also mounted, with every switch off, when the library's tests use it, and it ships in Sol's box and the actor's export, so a stored library never depends on a switch setting (dynamic imports with a constant name count; a test importing a module named at run time is refused); library code using the kit is refused. In every gate run with the kit mounted (and only there), a test or module skipped because an import failed counts as failed; a pytest plugin marks such skips by their exception (`pytest.importorskip`, or an ImportError being handled) in the junit report, never by message text. The dynamic checks share a 900 s budget, and a pass it leaves unjudged is refused. All off, for a library whose tests do not use the kit: the gate and Sol's brief are as at the screen build.
- **Evidence:** unscreened. To be judged offline first by replaying the gate with each switch on and off over the screen replay's recorded passes (refusals, Sol repair cost, later-use precision), then paired.

## UNIFY_MEMORY_V2_DIALOGUE

- **Values:** empty (the same as `off`) or `env`.
- **Default:** empty.
- **What it does:** It applies only with `UNIFY_MEMORY_V2=on`. With `env`, each request's episode also records its dialogue actions: every turn-ending reply (text, no tool call) is one action on the channel `env` (memory channel `env/env/`), paired with the counterpart's next message as its observation. The action is read from the reply's structure only: a trailing JSON object names its method by its first member whose value is an identifier (`{"action": "submit", "grid": g}` is `submit(g)`), else the reply's last line is `act(line)`. The observation is the counterpart's message, parsed when it is wholly JSON, redacted, and kept whole up to 65,536 characters. A benchmark whose actions are text in the replies (Continual-ARC) then leaves actions a consolidation item can cover. Off, episodes are recorded as without the setting.
- **Evidence:** keyless tests only (`tests/memory_v2/test_arc_dialogue.py`, `tests/memory_v2/integration/test_cli_e2e_arc.py`); the Continual-ARC memory-v2 arm needs it.

## UNIFY_CLOCK_PLACEMENT

- **Values:** empty (the same as `system`) or `first_message`.
- **Default:** empty.
- **What it does:** As shipped, the system prompt carries a "Current Time" section that tells the model to resolve "today" against it and to prefer it over any clock read in code. With `first_message`, the system prompt has no clock section, and the session's first user message opens with one line: "The host clock reads <time>. Dates stated in the request or in the files and records you work with take precedence." The time is sampled once, when the session starts. Later requests in a persistent session do not repeat the line. The system prompt, and the cache affinity key derived from it, are then the same for every session of one configuration, whenever it starts.
- **Evidence:** office-v2 defect note P2c (the clock read as an authority over dates in the work, and a per-minute timestamp that breaks the cross-session prompt cache); unscreened. To be compared with the shipped placement under the per-instance fake clock, reporting the cache-hit rate and clock-attributable losses from traces.
