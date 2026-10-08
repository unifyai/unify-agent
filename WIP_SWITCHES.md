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

## UNIFY_MEMORY_V2_E, UNIFY_MEMORY_V2_SOL_MODEL, UNIFY_MEMORY_V2_SOL_ALLOWANCE_USD_PER_TOKENS and UNIFY_MEMORY_V2_SOL_RUN_GUARD_USD

- **Values:** `UNIFY_MEMORY_V2_E` is a positive whole number of tokens. `UNIFY_MEMORY_V2_SOL_MODEL` is a model id. `UNIFY_MEMORY_V2_SOL_ALLOWANCE_USD_PER_TOKENS` is a positive plain decimal string. `UNIFY_MEMORY_V2_SOL_RUN_GUARD_USD` is a non-negative plain decimal string, or empty. Exponent forms (`7.3e-7`) are refused.
- **Defaults:** `150000`, `openai/gpt-6-sol`, `0.00000073` and empty (no guard).
- **What they do:** They apply only with `UNIFY_MEMORY_V2=on`. The trigger is size-based and batched (the spec's F1): one pass becomes due once the experience recorded since the last pass reaches E tokens, and it covers every channel with new evidence. E times the allowance is one pass's USD cap. The passes run on the Sol model at the actor's reasoning effort for the run (there is no effort switch). With a run guard set, no further pass starts once the run's committed Sol USD plus the next pass's cap would exceed it.
- **Evidence:** E and the allowance follow the offline cadence replay and the memory-v2 preregistrations (continual-harness-research); the guard is a runaway stop, not the expected cutoff.

## UNIFY_MEMORY_V2_SOL_BASE_URL and UNIFY_MEMORY_V2_SOL_TOKEN

- **Values:** both empty, or both set: `UNIFY_MEMORY_V2_SOL_BASE_URL` an http(s) URL with a host and no user, password, query or fragment (plain http only to exactly `127.0.0.1`, the launcher's loopback bridge; https to any host); `UNIFY_MEMORY_V2_SOL_TOKEN` a bearer token of at least 16 characters from `A-Z a-z 0-9 - . _ ~ + /` plus `=` padding (a secret).
- **Where to set them:** in the controller process's own environment, never in `.env`: settings are read before the CLI loads `.env`, so a value found there (or under another letter case, with a different value) refuses every pass with an error naming the setting, and its token is removed from the environment. Hand the token to the controller process only, not to a shared driver environment (`/proc/<pid>/environ` keeps a process's initial environment).
- **Defaults:** empty (Sol's calls go as shipped).
- **What they do:** They apply only with `UNIFY_MEMORY_V2=on`. Set, Sol's model calls go to that OpenAI-compatible base URL with that token, and carry `X-Unify-Call-Kind: memory_v2.sol`: a proxy listener of Sol's own, so the actor's route (and its allow-list) never carries Sol's model. Exactly one set, or a value in a wrong form, starts no pass (the request stays due); no error quotes either value. The token is read into the controller's settings only, removed from its environment (every letter case) and kept registered with the value-based redactors; a failed Sol call is recorded as its exception class and a fixed category, never its text. Cells and Sol's box never receive either name. The header enters unillm's response-cache key, so a cache recorded with the route unset misses with it set (unset sends no header, so unset recordings are unaffected).
- **Evidence:** a deployment route for the paired memory-v2 screens, not a behaviour under test.

## UNIFY_CLOCK_PLACEMENT

- **Values:** empty (the same as `system`) or `first_message`.
- **Default:** empty.
- **What it does:** As shipped, the system prompt carries a "Current Time" section that tells the model to resolve "today" against it and to prefer it over any clock read in code. With `first_message`, the system prompt has no clock section, and the session's first user message opens with one line: "The host clock reads <time>. Dates stated in the request or in the files and records you work with take precedence." The time is sampled once, when the session starts. Later requests in a persistent session do not repeat the line. The system prompt, and the cache affinity key derived from it, are then the same for every session of one configuration, whenever it starts.
- **Evidence:** office-v2 defect note P2c (the clock read as an authority over dates in the work, and a per-minute timestamp that breaks the cross-session prompt cache); unscreened. To be compared with the shipped placement under the per-instance fake clock, reporting the cache-hit rate and clock-attributable losses from traces.
