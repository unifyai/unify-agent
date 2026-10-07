# Code freeze: what is done and what remains (7 Oct 2026)

This branch is the overhauled Unify harness. **Lean-all is the base config, in Python tool mode:** `execute_code` is the actor's only tool, with the shared agent record for communication. The research switches are being removed so that the chosen behaviour is the code's only path. Everything removed stays recoverable from the tag `pre-freeze-3bb760e54`.

`WIP_SWITCHES.md` lists the switches that stay on purpose (work in progress). This file lists what the freeze has not finished. Work through it on branches off `harness-learning`, landing each item as a focused PR once its tests pass.

## Done
- **Step 3 (bake):** every approved value is the default. `PROMPT_TRIM` is baked on: under the new defaults it still removes 839 characters per first request.
- **Step 4 (strip), part:** 34 research switches removed or collapsed to their baked value, with their losing paths and tests.
- **Step 5 (dead code), part:**
  - the actor's JSON tools for FunctionManager and GuidanceManager, and the `execute_function` tool;
  - the 8 loop switches: REPEAT_GUARD, PENDING_TIMEOUT_S, PENDING_REQUIRED, BATCH_WAKE, LIFECYCLE_NOTICES, WAIT_FOR_BATCH, WAIT_CEILING_SECONDS, DISCOVERY_SPECULATIVE_TURN.
- **Tests:**
  - the switches-off equivalence test is replaced by one baked-prompt golden (`tests/actor_baked_prompt_golden.json`, checked by `tests/actor/code_act/test_baked_prompt_golden.py`);
  - `tests/test_no_research_switches.py` fails while any research switch remains;
  - provider-key tests carry `requires_provider_key`: skipped by name without a key, and failing under `UNIFY_TEST_REQUIRE_PROVIDER_KEY=1` if the key is missing.
- **Untouched:** ordinary context compression and `unify/conversation_manager/`.

## Remaining research switches (17)

Groups 1–5 of the first list (the lexical and per-task-review memory cluster; the function summary and notices; INLINE_CURATION; REVIEW_FAILED and its lessons path; OUTCOME and STORE_TRUST) were removed in follow-up PRs after the freeze push.

What is left is the tool-surface tangle below. `tests/test_no_research_switches.py` lists these switches until they are gone.

### 1. The tool-surface tangle (prompt building)

**Switches:** PROMPT_PROFILE (lean), DISCOVERY_GATE (off), DELEGATION (off), AGENTS (record) + AGENTS_OPTIONS, CACHE_DISCIPLINE, REVIEW_FORK + REVIEW_FORK_CORE, TOOL_SURFACE (core) with CORE_BIND_LISTED, CORE_CALL_EXAMPLE, GUIDANCE_LINKED_NAMES and FUNCTION_HELPERS, WORKSPACE (sandboxed), WORKSPACE_PYTHON (worker), LIBRARY_SHORTLIST, TRANSCRIPTS.

**Known couplings:**
- `lean_prompt()` is still a settings method, and `prompt_builders` has `lean` branches;
- `delegation_mode()` lives in `environments/actor.py`;
- `agents.enabled()` is read by `sandbox.py`, `cli.py` and `loop.py`;
- `core_surface` refuses any non-core combination, so the switch can simply go.
- **CACHE_DISCIPLINE also selects the compression fork summary. That part must stay,** as a fixed constant inside the compression module.

### 2. Leftovers from the JSON-tool removal
- `close_session` / `close_all_sessions`, which two WIP tests still pin;
- `_synthesize_python_call`;
- `placeholder_note.py`, whose only effect was on `execute_function`, so it is now dead;
- `cell_state.describe_execute_function`;
- the `execute_function` entries in `_correct_tool_docs` and `_hide_parent_chat_context`;
- `execution/session.py` still names `close_session` in an error message;
- `install_python_packages` and the workspace `read_file` / `grep` JSON tools.

### 3. The loop (a design task, not a strip)

**What stays, and why:** the steerable-handle machinery (steer/wait/ask, check_status, pending placeholders, multi_handle, interjection channels). `conversation_manager` runs its brain through `start_async_tool_loop` and imports `SteerableToolHandle`, and an interjection into a running `execute_code` cell goes through the steering patcher.

**The goal:** a minimal synchronous `execute_code` loop. It needs a separate actor entry point. Compression, the agent record's interjections and the WIP loop switches (LOOP_STOP, STEP_CAP_*, REPLY_CHANNEL) are wired into the async loop.

**Small dead pieces:**
- the `interrupt_llm_on_tool_completion` parameter, which is now always overridden;
- `BatchHold.install()`'s time-from-install branch.

## Known defects and test status
- **`library_shortlist.shortlist_block` (lines ~435–437) and `_gated_block` swallow any exception at DEBUG level.** A missing embedding key silently drops the shortlist. Log a warning at least.
- **`test_can_store_true_merges_redundant_functions`** passes 3/5 live under the old defaults. The review sometimes searches only by the new function's exact name, so it never sees the narrower variants.
  - The baked review (fork + shortlist) should expose them. This has to be measured (n ≥ 10 live, ≥ 9/10).
  - If it doesn't pass, the fix is a general review change with its own evidence.
- **conversation_manager's actor now inherits the core surface and the agent record.** That has not been reviewed.
- **Tests still pin a removed switch to its LOSING value** (about 26 sites, e.g. `test_prompt_accuracy`, `test_patch_tools`, `test_prompt_trim`, `test_functions_first`, `test_library_snapshot`). Those tests exercise deleted paths: delete them, don't skip them. `tests/baked_defaults.py`'s `as_shipped` pins (29 modules, all tied to the JSON-tool and handle machinery) go with those paths.
- **`ARCHITECTURE.md` and `README.md`** still describe the old JSON-tool, in-process behaviour.
- **The final full test run** happens on branch `freeze-final-tests`, then as follow-up PRs. The lead's standard is zero failures, no allowlists, and skips only by design, each with its reason.
