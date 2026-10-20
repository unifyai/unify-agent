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
- **Default:** `draft` (since 13 October 2026; empty before).
- **What it does:** With `draft`, reaching the step limit in a persistent session ends only the current request. The reply quotes the latest reply text the model drafted, and the next message starts a new request with its own step budget. With `last_word`, the model first gets one call with no tools to give its best answer. Empty: the limit counts every message of a persistent session and ends the session.
- **Evidence:** a benchmark adapter that sends each step as another request of one session reaches a session-wide count however short each request is. In the Continual-ARC runs of 12-13 October, it ended a session in each of the two medium-effort runs. `last_word` is still unscreened.

The three long-horizon switches share the step-limit reply path.

## UNIFY_MEMORY

- **Values:** empty (the same as `off`) or `on`.
- **Default:** empty.
- **What it does:** When on, no storage review runs, and the `functions` and `guidance` objects and the library shortlist are gone. Each request instead imports from a scratch export of the memory repo's `main`, mounted read-write in the worker and discarded after the request, and sees the memory index at the end of the system prompt. The index is rendered only from the memory commit, so two requests on the same commit send the same prompt prefix. The whole request (its messages, replies, cells, actions, the work tree's before and after snapshots and what it wrote to the export) is recorded as one episode in `<UNIFY_HOME>/episodes.git`; only the checker's pass/fail is kept from a posted outcome. After the request, due consolidation passes by Sol run behind a deterministic gate, blocking before the session ends; each pass writes a start and an end event (`--jsonl` output and `<UNIFY_HOME>/memory-state/events.jsonl`).
- **Evidence:** matched screens pending (memory preregistrations, continual-harness-research). Design: `docs/design/memory-redesign-spec.md` there.

## UNIFY_MEMORY_E, UNIFY_MEMORY_SOL_MODEL, UNIFY_MEMORY_SOL_ALLOWANCE_USD_PER_TOKENS, UNIFY_MEMORY_SOL_RUN_GUARD_USD and UNIFY_MEMORY_SOL_EFFORT

- **Values:** `UNIFY_MEMORY_E` is a positive whole number of tokens. `UNIFY_MEMORY_SOL_MODEL` is a model id. `UNIFY_MEMORY_SOL_ALLOWANCE_USD_PER_TOKENS` is a positive plain decimal string. `UNIFY_MEMORY_SOL_RUN_GUARD_USD` is a non-negative plain decimal string, or empty. `UNIFY_MEMORY_SOL_EFFORT` is `actor`, `low`, `medium` or `high`. Exponent forms (`7.3e-7`) are refused.
- **Defaults:** `150000`, `openai/gpt-6-sol`, `0.00000073`, empty (no guard) and `actor`.
- **What they do:** They apply only with `UNIFY_MEMORY=on`. The trigger is size-based and batched (the spec's F1): one pass becomes due once the experience recorded since the last pass reaches E tokens, and it covers every channel with new evidence. E times the allowance is one pass's USD cap. The passes run on the Sol model at the actor's reasoning effort for the run (`actor`, the default; the lead, 8 Oct); a fixed `low`, `medium` or `high` is for a declared mismatch ablation only. With a run guard set, no further pass starts once the run's committed Sol USD plus the next pass's cap would exceed it.
- **Evidence:** E and the allowance follow the offline cadence replay and the memory preregistrations (continual-harness-research); the guard is a runaway stop, not the expected cutoff.

## UNIFY_MEMORY_SOL_EFFORT_SCALE and UNIFY_MEMORY_SOL_MAX_CALLS

- **Values:** each a map from the three Sol efforts to a value, `low:<v>,medium:<v>,high:<v>` (each effort exactly once, any order, spaces and letter case ignored). `UNIFY_MEMORY_SOL_EFFORT_SCALE` takes positive plain decimal strings; `UNIFY_MEMORY_SOL_MAX_CALLS` takes positive whole numbers. Anything else is refused.
- **Defaults:** `low:1,medium:2,high:5` and `low:40,medium:80,high:80`.
- **What they do:** They apply only with `UNIFY_MEMORY=on`. Each consolidation pass takes its limits from its own Sol effort (the actor's for the run unless `UNIFY_MEMORY_SOL_EFFORT` fixes one): its USD cap is E times the allowance times the effort's scale, and it makes at most the effort's calls (model calls and `check` calls). One rule for every bed, never per stream. The run guard compares against the scaled cap. An effort with no entry (neither `low`, `medium` nor `high`) starts no pass; the request stays due and the error log says why. Each pass's start event and its ledger reservation record `cap_usd`, `max_calls` and `effort_scale`.
- **Evidence:** an offline cadence replay diagnostic (8 Oct) measured USD per pass 0.095 (ARC MEDIUM), 0.178 (ARC HIGH), 0.116 (office MEDIUM) and 0.30 (office HIGH), with 10, 14, 13 and 23 calls per pass; at 40 calls and the unscaled cap (USD 0.1095 at E = 150,000) Sol at HIGH merged nothing. Matched screens pending.

## UNIFY_MEMORY_SOL_BASE_URL, UNIFY_MEMORY_SOL_TOKEN and UNIFY_MEMORY_SOL_TOKEN_FD

- **Values:** both empty, or both set: `UNIFY_MEMORY_SOL_BASE_URL` an http(s) URL with a host and no user, password, query or fragment (plain http only to exactly `127.0.0.1` with an explicit port, the launcher's loopback bridge; https to any host); `UNIFY_MEMORY_SOL_TOKEN` a bearer token of at least 16 URL-safe characters, `A-Z a-z 0-9 . _ ~ -` (a secret; `secrets.token_urlsafe` output fits).
- **Where to set them:** in the controller process's own environment, never in `.env`: settings are read before the CLI loads `.env`, so a value found there (or under another letter case, with a different value) refuses every pass with an error naming the setting (checked again whenever a pass is about to start, so an embedder that loads `.env` late is refused too), and its token is removed from the environment. Hand the token to the controller process only, not to a shared driver environment (`/proc/<pid>/environ` keeps a process's initial environment).
- **Token by descriptor:** `UNIFY_MEMORY_SOL_TOKEN_FD=<n>` (a descriptor number above 2) replaces `UNIFY_MEMORY_SOL_TOKEN`; set exactly one of the two (both refuse every pass), and set it with `UNIFY_MEMORY_SOL_BASE_URL`. The launcher passes the controller an inherited descriptor (a pipe or a file) holding the token, optionally followed by one newline, and closes its own write end; the environment carries only the number. The controller reads it once when its settings are first settled (at most 4096 bytes and 5 s), closes it, and holds the token outside `os.environ`, registered with the redactors. A descriptor that is not open, was not inherited, is neither a pipe nor a file, or holds nothing, more than 4096 bytes or not a bearer token refuses every pass the same way, naming the rule, never the value.
- **Defaults:** empty (Sol's calls go as shipped).
- **What they do:** They apply only with `UNIFY_MEMORY=on`. Set, Sol's model calls go to that OpenAI-compatible base URL with that token, and carry `X-Unify-Call-Kind: memory.sol`: a proxy listener of Sol's own, so the actor's route (and its allow-list) never carries Sol's model. Exactly one set, or a value in a wrong form, starts no pass (the request stays due); no error quotes either value, and each such request appends `{"type": "consolidation", "phase": "refused", "consolidation_refused": "route_not_in_effect", ...}` to the run's `events.jsonl` (and the `--jsonl` stream), so the run is flagged rather than read as a null result. With `UNILLM_OTEL` on, the route is refused the same way (unillm's spans keep error text as is). The token is read into the controller's settings only, removed from its environment (every letter case) and kept registered with the value-based redactors; a failed Sol call is recorded as its exception class and a fixed category, never its text, and unillm's own copies of that text (its `unillm.retry` warnings and Sol's per-call `UNILLM_LOG_DIR` file) are redacted. Cells and Sol's box never receive either name. The header enters unillm's response-cache key, so a cache recorded with the route unset misses with it set (unset sends no header, so unset recordings are unaffected).
- **Evidence:** a deployment route for the paired memory screens, not a behaviour under test.

## UNIFY_MEMORY_SURFACING, UNIFY_MEMORY_DOCSTRINGS and UNIFY_MEMORY_SOFT_BUDGET

- **Values:** `UNIFY_MEMORY_SURFACING` is `index` or `catalogue`; `UNIFY_MEMORY_DOCSTRINGS` and `UNIFY_MEMORY_SOFT_BUDGET` are `off` or `on`. Empty means the default.
- **Defaults:** `index`, `off` and `off`: the v2 screen build's behaviour (`9deefbfd1`), so a paired v2 vs v2.1 comparison runs on one build.
- **What they do:** They apply only with `UNIFY_MEMORY=on` (memory v2.1, lane S1). `index`: the system prompt ends with the v2 per-function index and the export line, and Sol's first message carries the index. `catalogue`: the prompt ends with one constant guide paragraph (the same bytes for the whole run, from the first request whose library lists anything; no count, channel, function, drift flag or path), and memory is discovered in cells: every export holds a generated `README.md`, `.memory/catalog.json`, `.memory/shapes.py` and a `memory` helper (`import memory; print(memory.catalog())`, `memory.find(value)`, `memory.describe(name)`), input shapes are recorded at each merge and frozen per commit, a suspect channel's refusal says so in the cell's error, and Sol's first message carries the README. The use record (memory-v2.1-tele) then records the guide as shown and no function or channel, since the guide names none; per-function exposure comes from the cells. `UNIFY_MEMORY_DOCSTRINGS=on`: the gate's lean docstring standard (G1) and its examples run as doctests (G3), stated in Sol's brief. `UNIFY_MEMORY_SOFT_BUDGET=on`: G4 notes that hygiene is due past the 4,000-token budget instead of refusing growth. Not switched (a safety fix in every mode): the gate refuses bytecode, native code, start-up hooks and root entries outside the layout, and changes to the paths reserved for the generated catalogue.
- **Evidence:** none yet (memory v2.1 stage 1-2); offline replay (`memory_offline/cadence_replay.py` builds the gate from these switches) and a paired screen pending (`docs/design/memory-v2.1-plan.md`, continual-harness-research).

## UNIFY_MEMORY_SOL_USAGE

- **Values:** empty (the same as `off`) or `on`.
- **Default:** empty.
- **What it does:** It applies only with `UNIFY_MEMORY=on`. When on, each consolidation pass's first message ends with a table of how the pass's requests used each library function: requests whose pin held it, requests whose prompt showed its own index line, requests whose prompt showed its channel, requests that called it, call sites, refusals (`MemoryInputError` raised out of the function), refusals followed by a successful action on its channel, other errors, dynamic calls in its channel and requests since its last call. Refusals and errors from a request that edited the function's channel in its scratch copy are left out (they are recorded apart, as the edit's). Refusals and other errors read `k of n known (+u unknown)`: requests with one, of the requests whose outcome for the function was known, and the requests whose outcome for it was unknown because a cell with no recorded outcome (cancelled, timed out, stopped at the step cap, a tool call that raised, an executor failure) could reach it; such a cell leaves unknown only the functions its code imports, calls or references, or whose channel it uses dynamically, in that request only. A "then accepted" count marked `+?` is at least that number. Closing notes say how many requests had such a cell, how many requests' shown counts came from the harness's record of what it rendered rather than from the prompt's text (legacy) or nowhere (no record, or no system prompt recorded), and how many recorded prompts did not end with the recorded section. The numbers come from the harness's use record (`memory_use.json` per episode, the evidence store's `item_use` table), never from a checker. When off, the first message is byte for byte as before. The use record is kept either way.
- **Evidence:** none yet (memory v2.1 stage 1); offline comparison and a paired screen pending (`docs/design/memory-v2.1-plan.md`, continual-harness-research).

## UNIFY_MEMORY_QA_FIXTURES, UNIFY_MEMORY_QA_MUTATION, UNIFY_MEMORY_QA_MUTATION_MIN_KILL, UNIFY_MEMORY_QA_DETERMINISM, UNIFY_MEMORY_QA_REPLAY and UNIFY_MEMORY_QA_FIXTURE_SIZE

- **Values:** `UNIFY_MEMORY_QA_FIXTURES` is empty (the same as `off`), `on` or `strict`. `UNIFY_MEMORY_QA_MUTATION_MIN_KILL` is a plain decimal string from 0 to 1. The others are empty (the same as `off`) or `on`.
- **Defaults:** all empty (off), and `0.5`.
- **What they do:** They apply only with `UNIFY_MEMORY=on`, inside the consolidation gate (memory v2.1 stage 5, `unify/memory/qa.py`), and judge the quality of the tests Sol writes. With `_QA_FIXTURES`, the gate draws a seeded random sample of recorded inputs of each new or changed function's family (the seed is the candidate commit) and calls the function on each: it must return or raise `MemoryInputError`, and refusing an input shaped like its covers refuses the pass. Arguments beyond the input come from the recordings only. Tests parametrised over `memlab.inputs.inputs` also run on the drawn inputs; `strict` refuses a drawn input no test of the function read. With `_QA_MUTATION`, the function's tests must kill at least `_QA_MUTATION_MIN_KILL` of the mutants that change its outputs on the recorded inputs or on structurally broken copies of its covers (a field dropped, retyped, emptied or added; at most 2 MiB of them per function); a mutation check the probe could not judge is refused. With `_QA_DETERMINISM`, every gate test run pins the clock, the hash seed and `random`, and each new test file runs a second time under different pins (epoch, clock step, seeds, time zone); different outcomes are refused (flaky, or dependent on the pinned values). With `_QA_REPLAY`, a test passing an environment function a stand-in of its own instead of the kit's replay (`memlab.replay.env_from` or `RecordedEnv`: exact-call replay keyed by channel, method, arguments and sorted keyword arguments; identical calls answered in recorded order, the last recording repeating past the end and listed in `env.repeats`; an unrecorded call raises, a recorded error re-raises, `env.issued()` lists the calls served so a test holds a function to exactly its own recorded calls; an effect filter raises where a recorded effect is `unknown`, as for every dialogue action) is refused; a root `unify_memory_testkit.py` that re-exports it is accepted, one faking the environment is not. With any switch on, Sol's brief points at `env_from` and no longer asks for a root test kit. With `_QA_FIXTURE_SIZE`, a test or data file over 64 KiB is refused, recorded payloads are referenced by blob id, and a test asserting on the cut of a truncated recording is refused. Any of them also mounts the library test kit (`unify/memory/testkit.py`: `memlab`, the pin plugin, the referenced blobs) at `/inputs` in the gate's runs and adds the matching paragraph to Sol's brief. The kit is also mounted, with every switch off, when the library's tests use it, and it ships in Sol's box and the actor's export, so a stored library never depends on a switch setting (dynamic imports with a constant name count; a test importing a module named at run time is refused); library code using the kit is refused. In every gate run with the kit mounted (and only there), a test or module skipped because an import failed counts as failed; a pytest plugin marks such skips by their exception (`pytest.importorskip`, or an ImportError being handled) in the junit report, never by message text. The dynamic checks share a 900 s budget, and a pass it leaves unjudged is refused. All off, for a library whose tests do not use the kit: the gate and Sol's brief are as at the screen build.
- **Evidence:** unscreened. To be judged offline first by replaying the gate with each switch on and off over the screen replay's recorded passes (refusals, Sol repair cost, later-use precision), then paired.

## UNIFY_MEMORY_DIALOGUE

- **Values:** empty (the same as `off`) or `env`.
- **Default:** empty.
- **What it does:** It applies only with `UNIFY_MEMORY=on`. With `env`, each request's episode also records its dialogue actions: every turn-ending reply (text, no tool call) is one action on the channel `env` (memory channel `env/env/`), paired with the counterpart's next message as its observation. The action is read from the reply's structure only: a trailing JSON object names its method by its first member whose value is an identifier (`{"action": "submit", "grid": g}` is `submit(g)`), else the reply's last line is `act(line)`. The observation is the counterpart's message, parsed when it is wholly JSON, redacted, and kept whole up to 65,536 characters. A benchmark whose actions are text in the replies (Continual-ARC) then leaves actions a consolidation item can cover. Off, episodes are recorded as without the setting.
- **Evidence:** keyless tests only (`tests/memory/test_arc_dialogue.py`, `tests/memory/integration/test_cli_e2e_arc.py`); the Continual-ARC memory arm needs it.

## UNIFY_CLOCK_PLACEMENT

- **Values:** empty (the same as `system`) or `first_message`.
- **Default:** empty.
- **What it does:** As shipped, the system prompt carries a "Current Time" section that tells the model to resolve "today" against it and to prefer it over any clock read in code. With `first_message`, the system prompt has no clock section, and the session's first user message opens with one line: "The host clock reads <time>. Dates stated in the request or in the files and records you work with take precedence." The time is sampled once, when the session starts. Later requests in a persistent session do not repeat the line. The system prompt, and the cache affinity key derived from it, are then the same for every session of one configuration, whenever it starts.
- **Evidence:** office-v2 defect note P2c (the clock read as an authority over dates in the work, and a per-minute timestamp that breaks the cross-session prompt cache); unscreened. To be compared with the shipped placement under the per-instance fake clock, reporting the cache-hit rate and clock-attributable losses from traces.

## UNIFY_TOOLS_IN_CODE

- **Values:** `on` or `off`.
- **Default:** `off`.
- **What it does:** When on, every agent (the actor, its teammates, and the memory writer, curator and helpers) keeps exactly one model-API tool, `execute_code`. Every other tool is a Python call in a cell, `tools.<name>(...)`, defined once in the shared registry (`unify/common/tool_registry.py`) and run on the host with the same handler, limits and refusal texts. For the actor and its teammates the loop also compacts by itself when the prompt reaches the compaction point (no forced `compress_context` turn), and `tools.answer(answer)` replaces the response tool. When off, every agent's requests are byte for byte as before.
- **Evidence:** It is the treatment arm of a paired replay against the build without it (same requests and passes, LOW and HIGH), reported with task outcome, cost, prompt tokens, cache-read share, turns and compactions per request.

## UNIFY_CELL_STATE

- **Values:** `persistent` or `fresh`.
- **Default:** `persistent`.
- **What it does:** With `persistent`, each agent's cells run in one interpreter that keeps its variables between cells until it is restarted (a cell that runs out of time restarts it; files stay). With `fresh`, every cell starts in a new interpreter (files stay; variables do not).
- **Evidence:** Set together with `UNIFY_TOOLS_IN_CODE=on` in the same paired replay.

## UNIFY_STREAM_MODEL_CALLS

- **Values:** empty (the same as `off`) or `on`.
- **Default:** empty.
- **What it does:** When on, every model call of a client made by `new_llm_client` (or forked from one) goes out as a stream and is read to its end, and unillm reassembles the reply into the completion the same call returns unstreamed (its `stream_transport` flag). Transcripts, caching, cost and postprocessing are unchanged. Bytes arrive while the model is still generating, so a proxy that cancels a call after a spell with no traffic does not cancel a long one. Nothing is sent but the model's own reply: no keepalive or filler bytes.
- **Evidence:** A benchmark adapter's proxy cancels a turn after 300 s with no bytes, and an actor call of 377 s was cancelled by it, because Unify's calls are never streamed while the other agent harnesses compared with it all stream. Keyless tests only so far (unillm's `tests/test_clients/test_stream_transport.py`, on a hand-written OpenRouter stream, and `tests/common/test_stream_model_calls.py`). A recorded real stream is to confirm that the reassembled reply matches the unstreamed one before the switch becomes the default.

## UNIFY_MEMORY_LIBRARY, UNIFY_MEMORY_LIBRARY_E, UNIFY_MEMORY_LIBRARY_FORK, UNIFY_MEMORY_LIBRARY_POOL, UNIFY_MEMORY_LIBRARY_EDIT_TOOLS, UNIFY_MEMORY_LIBRARY_COMPACT, UNIFY_MEMORY_LIBRARY_ACTOR_INDEX and UNIFY_MEMORY_WRITER_USAGE

- **Values:** `UNIFY_MEMORY_LIBRARY`, `_FORK`, `_POOL`, `_EDIT_TOOLS`, `_COMPACT` and `UNIFY_MEMORY_WRITER_USAGE` are `on` or `off` (empty is the default); `UNIFY_MEMORY_LIBRARY_E` is a positive whole number of recorded tokens; `UNIFY_MEMORY_LIBRARY_ACTOR_INDEX` is `inline` or `pointer`.
- **Defaults:** `off`, `100000`, `off`, `off`, `off`, `on`, `inline` and empty (off).
- **What they do:** `UNIFY_MEMORY_LIBRARY=on` is the library mode: the writer reads a batch map and bounded views and keeps a function library (off: the earlier memory mode as built). In it, `_E` is the recorded experience at which a pass becomes due; `_FORK=on` forks the actor's conversation once at each episode's end into a confined process that writes only to the episode's staging directory; `_POOL=on` keeps every proposal open across passes in the candidate pool until the writer closes it; `_EDIT_TOOLS=on` gives the writer and the curator the `write` and `edit` file tools; `_COMPACT` (on by default) lets the memory agents compact their conversations as the agent loop does; `_ACTOR_INDEX` chooses whether the actor's prompt ends with the index view (`inline`) or a constant guide (`pointer`); `UNIFY_MEMORY_WRITER_USAGE=on` ends each pass's first message with the table of how requests used each library function. Each switch's comment in `unify/settings.py` gives the full behaviour.
- **Evidence:** the memory design under evaluation (continual-harness-research: the memory preregistrations and the build ledgers); a run's configuration sets them, and each default is the build's behaviour without the switch.

## UNIFY_MEMORY_NOTEBOOK

- **Values:** `on` or `off` (empty is `off`).
- **Default:** `off`.
- **What it does:** When on (with the library mode), the world notebook (`unify/memory/notebook.py`) runs: after each episode the harness marks its consequence moments from observable changes only (the end, a shown bounded quantity falling faster than its drift or once an earlier episode ended with it at its lowest, an action that changed nothing it usually changes, a failed call) and lists them in the episode-end fork's message; the writer keeps lessons (situation, action, consequence, alternative) with per-episode counts of occurrences and outcomes, which the harness sums into rates and support/contradiction tallies, and may propose untried remedies as trials; established lessons and open trials are appended to the actor's memory section at every request start. Off, nothing of it runs and every prompt is byte for byte as before.
- **Evidence:** offline replays of recorded episodes (continual-harness-research, the 18 Oct world-memory study: R-C2 counting, R-C3 moments, R-C5 remedies in the real engine); keyless tests only so far (`tests/memory/test_notebook.py`).

## UNIFY_MEMORY_LIBRARY_REVIEW_EACH_CAPTURE and UNIFY_MEMORY_LIBRARY_CAPTURES_PER_PASS

- **Values:** `UNIFY_MEMORY_LIBRARY_REVIEW_EACH_CAPTURE` is `on` or `off` (empty is `off`); `UNIFY_MEMORY_LIBRARY_CAPTURES_PER_PASS` is a positive whole number.
- **Defaults:** `off` and `1`.
- **What they do:** With the library mode and the fork on, every stored capture makes a writer pass due at once, whatever its outcome and whether it reproduced the value its attempt handed back; the experience budget alone no longer makes a pass due. One pass covers the pending episodes up to its `UNIFY_MEMORY_LIBRARY_CAPTURES_PER_PASS`-th capture episode; the rest stay due, so the worker runs the next pass right after it (kind `capture` in the pass events). Off, a pass is due when the recorded experience reaches E or when a capture reproduced its value with no contrary signal, which depends on whether the environment hands back a value the capture can reproduce. Refused at start-up without `UNIFY_MEMORY_LIBRARY=on` and `UNIFY_MEMORY_LIBRARY_FORK=on`. With `UNIFY_MEMORY_WRITER_PASSES=off` no pass runs at all, so these settings change nothing: they decide when a pass is due, never whether passes run.
- **Evidence:** where captures rarely reproduce a handed-back value, passes fell back to the experience budget and the writer never opened most capture episodes, while where they do it ran about one pass per capture and opened nearly all of them (continual-harness-research: the writer generalisation audit and the safe-serving replay). Keyless tests only so far (`tests/memory/test_trigger_per_capture.py`).

## UNIFY_MEMORY_LIBRARY_FIXED_ANSWERS_AS_EXAMPLES

- **Values:** `on` or `off` (empty is `off`).
- **Default:** `off`.
- **What it does:** With the library mode and the fork on, a captured procedure whose `run` never reads its input, or whose every return is a literal (`unify/memory/fixed_answer.py`: a rule on the code's syntax only), is offered as a worked example: an item of kind note whose text says why, what it handed back, its evidence and its source. It is never matched as applying, has no reader in the dispatch file, and calling it or checking `fits` refuses with what to do instead. Each request records in a `capture_serving` event how many captures it offered this way. Refused at start-up without `UNIFY_MEMORY_LIBRARY=on` and `UNIFY_MEMORY_LIBRARY_FORK=on`.
- **Evidence:** on some worlds nearly every capture stored one attempt's answer rather than a procedure (99-100% of captures on a card-game world), and in a replay of finished runs no call of such a capture solved its task where solving is checked, and none beat the memory-off baseline where one exists (continual-harness-research: the writer generalisation audit and the safe-serving replay). Keyless tests only so far (`tests/memory/test_capture_examples.py`).

## UNIFY_MEMORY_LIBRARY_WITHDRAW_CLOSED_CAPTURES

- **Values:** `on` or `off` (empty is `off`, or `on` under `UNIFY_MEMORY_PROFILE=redesign`).
- **Default:** empty (`off`).
- **What it does:** With the library mode and the candidate pool on, a captured procedure whose pool entry the writer closed as superseded or retired is no longer offered to the actor: it leaves the copy's capture snapshot, so `memory.items`, `memory.get`, the structural match, the librarian's catalogue, search index and dispatch file, and so `memory.ask`, no longer offer it. The item that superseded it (a library item, or an open captured procedure, followed through entries that were merged or superseded in turn) is offered in its place: `memory.get` of the withdrawn id gives it with a line saying so, the structural match lists it where the withdrawn capture would have matched, and its catalogue card and search text carry what the withdrawn capture was asked. A capture merged into an item stays offered. Each request records in a `capture_serving` event how many captures it withdrew and which items it offered in their place. Refused at start-up without `UNIFY_MEMORY_LIBRARY=on` and `UNIFY_MEMORY_LIBRARY_POOL=on`.
- **Evidence:** a read-only replay of finished runs found captures the writer had already superseded or retired still listed in 26-37% of later visits and called in some of them; the superseding item, called on the same kind of request after the decision, gave the accepted answer in 85-86% of calls (continual-harness-research: the safe-serving replay). Keyless tests only so far (`tests/memory/test_capture_withdrawal.py`).

## UNIFY_MEMORY_PROFILE

- **Values:** empty or `redesign`.
- **Default:** empty (each switch as set).
- **What it does:** `redesign` turns on every switch of the general harness redesign together: `UNIFY_AGENTS_RUN_RECORD`, `UNIFY_MEMORY_WRITER_SERVICE`, `UNIFY_MEMORY_METERING`, `UNIFY_MEMORY_CAPTURE_TRACE`, `UNIFY_MEMORY_ATTENTION_LEDGER`, `UNIFY_MEMORY_LEDGER_K=4`, `UNIFY_MEMORY_IMPURE_ITEMS`, `UNIFY_MEMORY_WRITER_TESTS=value` (the replay mode is refused until its stand-in's confinement review is recorded), `UNIFY_MEMORY_IN_EPISODE`, `UNIFY_MEMORY_SERVING_FAILURE_RECORD`, `UNIFY_MEMORY_LIBRARY_WITHDRAW_CLOSED_CAPTURES` and `UNIFY_MEMORY_REGIME=inferred`. A switch left empty takes the profile's value; a run that sets one of them to another value, or sets `UNIFY_MEMORY_LIBRARY_REVIEW_EACH_CAPTURE=on`, `UNIFY_MEMORY_LIBRARY_FIXED_ANSWERS_AS_EXAMPLES=on`, `UNIFY_MEMORY_NOTEBOOK=on`, `UNIFY_MEMORY_PASSES=off` or `UNIFY_MEMORY_WRITER_PASSES=off`, is refused at start-up, naming the settings and never a value, and so is a run that lacks what one of its switches needs. With `UNIFY_MEMORY` off the profile gives the memory switches nothing, so the no-memory arm keeps only the run record. Empty, nothing changes.
- **Evidence:** the redesign build plan (continual-harness-research: REDESIGN-BUILD-PLAN-20261020) puts every change of the redesign in one bundle for the offline replay and the live screen; keyless tests only so far (`tests/memory/integration/test_redesign_profile.py`).

## UNIFY_MEMORY_SERVING_FAILURE_RECORD

- **Values:** `on` or `off` (empty is `off`, or `on` under `UNIFY_MEMORY_PROFILE=redesign`).
- **Default:** empty (`off`).
- **What it does:** In the library mode, each request's library copy carries every stored item's evidence (`.memory/item_evidence.json`): its failure record (each failed use: the episode, the library version that was served, and why: it raised, its stated result condition was false, a contrary signal its regime observes followed it, that signal came only after the episode ended, or an actor reported a problem with it on the run record), what contradicted it, what its status rests on (verdicts, outcomes at episode end, implicit signals, or no outcome signal at all) and the conditions it states. Item cards, `Item.fits` (structurally similar or not, with the recorded failures, never "fits"), `match`, the librarian's catalogue and search cards, and each question to the librarian (with the situation: the latest message after the request) show that evidence; a reader that parsed the request or an `applies` that held is stated as evidence about the input's shape only. With `UNIFY_MEMORY_LIBRARY_WITHDRAW_CLOSED_CAPTURES` also on, library items the writer retired or superseded are withdrawn as captures are: no listing shows them and asking for one by its id gives what replaced it. Refused at start-up without `UNIFY_MEMORY=on` and `UNIFY_MEMORY_LIBRARY=on`. Off, every byte of the copy and every prompt is as before.
- **Evidence:** an office transcript served an item that hard-coded a year as fitting ("the stored reader parsed this request and its applies check held") and it failed on two repeats (continual-harness-research: NONARC-WRITER-GENERALISATION-20261019, E4); keyless tests only so far (`tests/memory/test_serving_failure_record.py`).

## UNIFY_MEMORY_REGIME

- **Values:** `inferred` or `off` (empty is `off`, or `inferred` under `UNIFY_MEMORY_PROFILE=redesign`).
- **Default:** empty (`off`: every episode is `dense`, as built).
- **What it does:** Each episode's feedback regime is inferred from the signals that arrive for it, never defaulted: `dense` when the actor saw a verdict after an answered action within the episode, `sparse` when an outcome came for the episode as a whole, `implicit` when only errors, stated conditions or corrections did, otherwise the run's own history (the most common regime its earlier episodes' own signals gave) or `none`. A hidden grader's verdict decides nothing. Every signal records when it arrived, when its episode ended, whether it came within or after the episode, and what it is about (`action:<n>`, `item:<id>`, `capture:<id>`), and each arrival is a row of `memory-state/signal-arrivals.jsonl`. A later request's session takes a verdict on an earlier episode by its id (`{"checker": {"label": "fail"}, "episode": "<id>"}`; the CLI announces each episode's id as `{"type": "episode"}`): the earlier episode's regime is inferred again (each change logged in the evidence store) and the captures judged on it are judged again. Refused at start-up without `UNIFY_MEMORY=on`.
- **Evidence:** the coverage probe found no late signal recorded in any world (no record type existed for one), and every episode was `dense` whatever its world gave; keyless tests only so far (`tests/memory/test_inferred_regime.py`).

## UNIFY_MEMORY_FROZEN

- **Values:** `on` or `off` (empty is `off`). Not one of the redesign profile's switches: allowed beside the profile and every arm, and never set by the profile.
- **Default:** empty (`off`).
- **What it does:** Held-out evaluation. Memory is served as the home holds it and nothing changes it: the request's episode is not recorded (so its uses, its signals and a late verdict it carries reach no store, and a dead process's journal is not recovered), no writer runs (no writer service, pass, episode-end fork or segment fork), pass results still unapplied stay unapplied, the library copy is pinned once and never re-pinned, `tools.checkpoint` is refused ("memory is frozen in this run"), and a message addressed to the writer on the run record is not posted: the sender is told "@writer: not available: memory is frozen" (to anyone else it is posted as usual). Serving counts an actor's report on the record only from a recorded episode. Each frozen request leaves one `{"type": "consolidation", "phase": "frozen"}` event. The kit stops the writer service (its drain) when it freezes a cell. Off, nothing changes.
- **Evidence:** the redesign build plan (continual-harness-research: REDESIGN-BUILD-PLAN-20261020, §4 C4 and §10) defines held-out gain on a frozen snapshot; keyless tests only so far (`tests/memory/test_frozen_memory.py`, conformance `tests/memory/conformance/test_surfaces.py`).

## UNIFY_MEMORY_CAPTURE_TRACE

- **Values:** `on` or `off` (empty is `off`, or `on` under `UNIFY_MEMORY_PROFILE=redesign`).
- **Default:** empty (`off`).
- **What it does:** The episode-end fork's capture records what the episode did, not only a value it handed back: the procedure that did the work, in one of three forms (it returns the value from the request, it takes the episode's state, or it takes the state and acts through the environment's tools), with a value only when one was handed back. The harness checks each capture against its own episode in the writer's box, every tool call answered only from the episode's recorded calls (nothing is executed), and keeps the result as evidence, never as a gate; a capture whose episode a signal contradicted is kept as what went wrong. Its outcome comes from the episode's signals only. Refused at start-up without `UNIFY_MEMORY_LIBRARY=on` and `UNIFY_MEMORY_LIBRARY_FORK=on`. Off, the capture contract and its records are as before.
- **Evidence:** the writer coverage design (continual-harness-research: WRITER-COVERAGE-DESIGN-20261019) found captures dropped whenever an episode handed back no single value, so work done through tools never reached the writer; keyless tests only so far (`tests/memory/test_capture_trace.py`).

## UNIFY_MEMORY_ATTENTION_LEDGER and UNIFY_MEMORY_LEDGER_K

- **Values:** `UNIFY_MEMORY_ATTENTION_LEDGER`: `on` or `off` (empty is `off`, or `on` under `UNIFY_MEMORY_PROFILE=redesign`); `UNIFY_MEMORY_LEDGER_K`: a positive whole number (empty is 4).
- **Default:** empty (`off`; 4 entries per pass).
- **What it does:** Every event that needs the writer's attention becomes an entry of the ledger (`memory-state/memory-attention/ledger.jsonl`): new experience (a settled episode or segment fork), a use that went badly, a capture whose own check failed, late evidence, repeated use, the gate's deferrals and messages to the writer on the run record. The writer service hands each pass `UNIFY_MEMORY_LEDGER_K` entries, bad news first; the writer reads their material, decides each with `tools.decide` (alone or together with entries it takes from the open list, `tools.open_entries`, `tools.find`, `tools.take`) and the decision is refused until the pass's own read records show the material opened. Every entry ends closed with a decision or listed with the reason it was not reached; an entry handed and not read goes back, and after repeated misses it is handed alone. The harness never groups entries for the writer. Refused at start-up without the library mode, the fork, `UNIFY_TOOLS_IN_CODE=on`, `UNIFY_MEMORY_CAPTURE_TRACE=on` and the writer service, or with `UNIFY_MEMORY_LIBRARY_REVIEW_EACH_CAPTURE=on`.
- **Evidence:** the writer coverage design (continual-harness-research: WRITER-COVERAGE-DESIGN-20261019) found settled forks never read by any pass and passes triggered by token counts rather than by what needed reading; keyless tests only so far (`tests/memory/test_attention_ledger.py`, `tests/memory/test_attention_tools.py`).

## UNIFY_MEMORY_IMPURE_ITEMS

- **Values:** `on` or `off` (empty is `off`, or `on` under `UNIFY_MEMORY_PROFILE=redesign`).
- **Default:** empty (`off`).
- **What it does:** The library holds items of any effect: a function may take the current state and act through the environment's tools, notes and checklists are items too, and nothing is filtered by purity. The writer's brief says so and asks, for each item it lands or changes, which episodes it is meant to cover, which values are inputs and which constants (with each constant's source) and what its calls do (`none`, `reads`, `writes` or `unknown`). The landing gate records its quality findings (reproduction, the held-out perturbation, example checks) on the item's label instead of refusing it; safety, confinement, credentials, the path rules and syntax still refuse. Every item's record and card carry its evidence label: form, declared effects against what its tests saw, the episodes it covers, its inputs and constants, its stated tests, its uses since its current version, its failure record (each with the version served) and what its status rests on. Refused at start-up without `UNIFY_MEMORY_LIBRARY=on` and `UNIFY_TOOLS_IN_CODE=on`.
- **Evidence:** the writer coverage design (continual-harness-research: WRITER-COVERAGE-DESIGN-20261019) found work that acts through tools refused or never written because the brief forbade side effects; keyless tests only so far (`tests/memory/test_impure_items.py`).

## UNIFY_MEMORY_WRITER_TESTS

- **Values:** `off`, `value` or `value,replay` (empty is `off`, or `value` under `UNIFY_MEMORY_PROFILE=redesign`; `value,replay` is refused at start-up until the replay stand-in's confinement review (G2) is recorded).
- **Default:** empty (`off`).
- **What it does:** The writer's code runs only as a test it states first (`tools.test`): the expectation is written to the pass's `tests.jsonl` before anything runs and never changes. `value` runs an item in the writer's box on a recorded input of an episode; `replay` runs it against that episode's recorded tool calls, every call (read, write or unknown effect) answered only from an identical recorded call and never executed, an unrecorded call being a replay miss. Each result is evidence on the item's label, never a gate. `replay` is refused, with what to do instead, until its confinement review is recorded. Refused at start-up without `UNIFY_MEMORY_LIBRARY=on` and `UNIFY_TOOLS_IN_CODE=on`.
- **Evidence:** the writer coverage design (continual-harness-research: WRITER-COVERAGE-DESIGN-20261019): the writer ran code freely with no stated expectation, and items with side effects had no way to be tested without effects; keyless tests only so far (`tests/memory/test_writer_tests.py`).

## UNIFY_MEMORY_IN_EPISODE

- **Values:** `on` or `off` (empty is `off`, or `on` under `UNIFY_MEMORY_PROFILE=redesign`).
- **Default:** empty (`off`).
- **What it does:** Memory can change while an episode runs. At each natural pause (the actor calls `tools.checkpoint`, the harness compacts the actor's context, a new message arrives after a cell, or an in-episode verdict arrives), a segment fork writes down the cells since the last pause in its own staging dir (`<episode>.s<n>`) and the writer service is called; never a count or a timer. When the writer lands a commit while the episode is open, the next cell boundary (never mid-cell) exports it into a fresh directory, swaps the actor's library link to it atomically, makes every worker drop the library's modules, and ends that cell's result with one line naming what changed (the first such line also says that names bound earlier keep their old code); at most one re-pin per boundary, to the latest commit, and none while another cell runs. The episode records its pins (`memory_pins`), and every use is counted under the version whose code it ran. Refused at start-up without the library mode, the fork, `UNIFY_TOOLS_IN_CODE=on` and `UNIFY_MEMORY_ATTENTION_LEDGER=on`. Off, every request reads one pin, as before.
- **Evidence:** the redesign build plan (continual-harness-research: REDESIGN-BUILD-PLAN-20261020, R6) for long episodes and long conversations, where nothing learnt reaches the episode that taught it; keyless tests only so far (`tests/memory/integration/test_in_episode.py`).

## UNIFY_MEMORY_LIBRARIAN

- **Values:** `on` or `off` (empty is `off`).
- **Default:** `off`.
- **What it does:** When on, the actor asks a librarian agent which stored items fit its request (`memory.ask()` in a cell, through the in-code tool `ask_library`); the library export also writes the librarian's catalogue and the procedures' dispatch file, and the actor's prompt shows a constant pointer in place of the index. It requires `UNIFY_TOOLS_IN_CODE=on` (refused at start-up otherwise). Off, nothing of it exists or runs.
- **Evidence:** the memory design under evaluation, as for the library switches above; keyless tests only so far (the librarian, catalogue, dispatch and ask tests under `tests/memory/`, on records the gate writes).

## UNIFY_MEMORY_PASSES

- **Values:** `on` or `off` (empty is `on`).
- **Default:** empty (`on`).
- **What it does:** When off, the actor's memory surface is exactly as with it on (its tools, the library pointer or index, the librarian when that is on), and no memory work runs after a request: no consolidation pass of any kind starts (a batch's WRITE, the CURATE it makes due, a pass a verdict makes due), no writer call is made, the episode-end fork does not start, and the library never changes. Every other memory setting keeps its value. The episode is still recorded, its meta saying `"memory_passes": "off"`.
- **Evidence:** the no-pass twin of the memory arm: the same surface with no learning, so a paired comparison separates what the library adds from what the surface alone does. No counted readout yet; keyless tests only.

## UNIFY_MEMORY_WRITER_PASSES

- **Values:** `on` or `off` (empty is `on`).
- **Default:** empty (`on`).
- **What it does:** When off (in the library mode with the episode-end fork on), the settled forks are still captured and the actor still runs the captures, and nothing after the capture step runs: no WRITE, CURATE or verdict pass starts and no writer or helper call is made, so the library never changes. It is refused without the library mode, without the fork, or with `UNIFY_MEMORY_PASSES=off`. On, every pass runs as before.
- **Evidence:** the captures-only arm of the cross-world memory screen (captures and librarian without the writer), against the full memory arm on one build; keyless tests only so far (`tests/memory/integration/test_writer_passes_off.py`).

## UNIFY_LIBRARY

- **Values:** `on` or `off` (empty is `on`).
- **Default:** empty (`on`).
- **What it does:** When off, the actor has no library: it removes memory and library surface only. The sandbox has no `functions` or `guidance` object, so the prompt has no library sections, no shortlist of stored entries and no line giving the library's size; the session may not store, so there is no `store_skills`, no storage review at a turn or at the end and no library write; and the system prompt no longer points to `functions.search` or says the actor has stored functions to draw on. Everything else is as it is (`execute_code`, `install`, `read_file`, `grep`, `sh`, `tools.check_examples`, the record, teammates). A run that also sets `UNIFY_MEMORY=on` is refused at start-up.
- **Evidence:** the minimal-agent arms: j3 with only memory and the library removed, so a paired comparison with j3 isolates what memory adds. No counted readout yet; keyless tests only.

## UNIFY_SYSTEM_SUFFIX and UNIFY_SYSTEM_SUFFIX_SHA256

- **Values:** `UNIFY_SYSTEM_SUFFIX` a text; `UNIFY_SYSTEM_SUFFIX_SHA256` the lowercase hex sha256 of its UTF-8 bytes.
- **Default:** both empty (no suffix).
- **What it does:** The text ends the system prompt of every `act()` (the actor's and a teammate's), after one blank line and after everything else the prompt carries. The two are set together and must match, or the run is refused at start-up, naming the settings and never the text. The CLI's `ended` line carries `system_suffix_sha256` when a suffix is set.
- **Evidence:** the minimal-agent arms' one paragraph (M0, and M1 with the cost sentence); a paired comparison of the two isolates the sentence. No counted readout yet; keyless tests only.

## UNIFY_AGENTS_RUN_RECORD

- **Values:** `on` or `off`; empty is `off`, or `on` under `UNIFY_MEMORY_PROFILE=redesign`.
- **Default:** empty.
- **What it does:** Every agent of the run shares one append-only record for the whole run, across processes: the actor of each request (`actor-<hex>`), the fork that reviews each episode (`fork-<hex>`), the librarian (`librarian-<hex>`), the writer (`writer`) and every helper any of them starts (`h<k>`). The record lives at `<UNIFY_HOME>/memory-state/agents/run-record.jsonl`; entries are numbered under a file lock, scrubbed of credentials, and never bound into a box. Every agent gets the same comms tools (`send`, `messages`, `wait`, `who`), the same common prompt section and a roster saying whom it should mention and when; new entries for an agent arrive between its steps. A pair of agents that only message each other for 20 entries with no tool call or file read in between gets one notice. Requires `UNIFY_TOOLS_IN_CODE=on` (refused at start-up otherwise). Off, the actor keeps its session record and each writer pass its own record, as built.
- **Evidence:** the lead's mesh ruling (20 Oct 2026: every agent on one record, with a roster and a common section); keyless tests only so far (`tests/agents/test_run_record*.py`).

## UNIFY_MEMORY_WRITER_SERVICE and UNIFY_MEMORY_WRITER_SERVICE_CAP_USD

- **Values:** `on` or `off` (empty is `off`, or `on` under `UNIFY_MEMORY_PROFILE=redesign`); the cap is a plain decimal string in USD, required with the service on.
- **Default:** empty.
- **What it does:** The writer runs as one detached service per memory home, spawned by the harness at request start, at request end and at a checkpoint, never by a runner: a lock, a heartbeat every 30 s (a stale one is taken over after 90 s), its own spend cap (a pass starts only when the room left under the cap holds one writer call at its worst case, 13.125 USD; the pass books that whole room and no call of its writer or helpers starts past it, counting unpriced and in-flight calls at the worst case; it settles from the writer lane's journal plus its helpers' priced calls, or from its own figures, after it; so the cap must be at least 13.125 USD for any pass to run), an idle exit after 30 minutes with nothing open, a 36 h lifetime that never runs past the host's lease less 30 minutes (`UNIFY_HOST_LEASE_END`), a stop marker with verified termination, and resumable passes (each model turn checkpointed before its call, so a killed service resumes at the turn it lost). Shutdown (`async_pass drain`) stops it and returns only once no process of it remains, before the key proxy is torn down. Requires the library mode, `UNIFY_MEMORY_ATTENTION_LEDGER=on`, the cap and `UNIFY_HOST_LEASE_END` (refused at start-up otherwise, each by name: no service runs without knowing when its host goes away). With `UNIFY_AGENTS_RUN_RECORD=on` it holds the writer's place on the run record; with it off (the mesh-off arm) each pass keeps its own record.
- **Evidence:** the lead's writer-service ruling (R4); the paper adapter's 600 s session kill lost a third to a half of the request-scoped pass workers' writer passes at HIGH and MED; keyless tests only so far (`tests/memory/test_writer_service*.py`).

## UNIFY_MEMORY_METERING

- **Values:** `on` or `off` (empty is `off`, or `on` under `UNIFY_MEMORY_PROFILE=redesign`).
- **Default:** empty.
- **What it does:** Every writer, helper and fork model call carries its pass id (or episode and segment for a fork) and the ledger entries it serves as request metadata headers (ids only), and each pass's result row records its entries, items, known USD, unpriced calls booked at the worst case, killed turns, tokens, turns and messages; `memory-state/metering/` holds the per-entry and per-item views derived from them. Requires `UNIFY_REQUEST_METADATA_HEADERS=on` (refused at start-up otherwise).
- **Evidence:** the lead's attribution ruling (R4, R13: whole-run all-lane USD with the writer included and labelled); keyless tests only so far (`tests/memory/test_metering.py`).
