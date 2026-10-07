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
- **Untouched:** ordinary context compression. `conversation_manager` moved to `unify/legacy/` (see Legacy below).

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

## Legacy (8 Oct 2026, branch `freeze-legacy-cm`)

**Legacy, unused, unsupported.** The harness uses only the actor; the conversation manager served an older product.
- **Moved:** `unify/conversation_manager/` to `unify/legacy/conversation_manager/` (files byte-identical apart from import paths), and its tests to `tests/legacy/conversation_manager/` (with the CM-only tests from `tests/actor/code_act/`, `tests/common/`, `tests/event_bus/` and the CM parts of `tests/test_test_configuration.py`, `tests/test_prompt_token_budgets.py` and `tests/actor/code_act/test_delegation_switch.py`).
- **Out of the default discovery and the gate:** `tests/conftest.py` skips `tests/legacy/` unless a path inside it is named, and `tests/parallel_run.sh` prunes `legacy` directories from a sweep.
- **Isolated:** `tests/test_legacy_isolation.py` fails if importing `unify.cli` or `unify.actor` (or populating the manager registry) loads any `unify.legacy` module. `ConversationSettings` (read by the slow-brain resolver in `unify/common/llm_client.py`) now lives in `unify/settings.py`; the registry registers the CM handles only inside `get_conversation_manager_handle`.
- **Still reaching legacy:** `unify chat`, the CLI's default command, imports it lazily when it starts. Removing that command (or making `act` the default) is a product decision for the lead.
- **After the loop trim** legacy is reference-only and not expected to run: it drives the pre-trim loop with steering tools.

## Security: the store check and confinement (7 Oct 2026)

**Found and fixed (PR "SECURITY: keep the store check from executing model code in the harness"):** storing a function executed its `def`, and its stored callees, in the process serving the call (`_store_check` → `_inject_dependencies` / `_create_in_process_callable`). That runs default values and decorators. With Python in the sandboxed worker, a cell's `functions.add` is served by the harness, so model-written code ran outside the sandbox, beside the credentials. With the worker on, the check is now static only. `tests/actor/code_act/test_store_check_confinement.py` fails without the fix.

**Found and fixed (PR "SECURITY: keep the store verifier and the case replay from executing model code in the harness"):**
- **The `UNIFY_STORE_VERIFY` hook** (benchmark configuration): a verifier loads and calls each candidate in the process that asks (`Candidate.load`), which is the harness for the review's `FunctionManager_check_function` (the re-check before reuse went with STORE_TRUST). With the worker on, `store_verify.verifier()` now refuses the switch before the factory is called, and start-up stops. Running a verifier beside the worker is a design choice, still open (below). `tests/function_manager/core/test_store_verify_confinement.py` fails without the fix.
- **The case replay (`UNIFY_FUNCTION_CASES`, on by default):** an overwrite or patch replayed recorded calls by loading and calling the new source in the harness, so a cell could add a function, call it, overwrite it with code of its own and read the result back in the refusal. With the worker on nothing is replayed: a case that returned refuses the change (a new name, or retire the case), a case that raised does not block. The candidate loader also refuses with the worker on. `tests/actor/code_act/test_case_replay_confinement.py` fails without the fix.

**Found and fixed (PR "SECURITY: bind stored functions without executing them in the harness", branch `fix-bind-load-confinement`, 8 Oct 2026):**
- **Bind loads.** `functions.get` / `search` / `filter` / `list` from a cell, the start-of-task shortlist (`UNIFY_CORE_BIND_LISTED`) and a guidance read's linked names (`UNIFY_GUIDANCE_LINKED_NAMES`) all reach `FunctionLibrary._load` (`core_surface.py`), which bound the returned functions in the harness's shadow namespace through `_inject_callables_for_functions` → `_inject_dependencies` / `_create_in_process_callable`. That executed each stored `def` and its stored callees in the harness (defaults, decorators, module-level code), and called `environment.ensure` there. With the worker on, a bind now stores a `StoredSource` (`source_labels.py`) per function and callee, built from the stored row without executing anything: the worker's `_describe_global` sends its source, and the worker defines and runs it, recorded as before. What a load did besides executing stays: declared dependencies are installed (by the confined installer), annotation names get their placeholders, and a source that does not compile or does not define its name drops the row with the "unloadable" warning. A `def` that raises when defined is now reported by the worker on use, not dropped from the read.
- **Choke points.** With the worker on, `_create_in_process_callable`, `_inject_dependencies`, `execute_function` (stored, before a case is opened) and `_execute_python_function` raise "… would execute model-written code in the harness; with Python in the sandboxed worker it runs only there". Nothing legitimate reaches them with the worker on (trace below).
- **The workspace environment off the harness's `sys.path`.** `environment.activate()` refuses with the worker on; the act start and `install()`'s `_create()` no longer call it; `missing()` reads distribution metadata on the worker's path (`importlib.metadata.distributions(name=, path=)`), and the store check's import lookup finds a module in the environment by path (`PathFinder.find_spec`), importing nothing.
- Tests, each failing without the fix: `tests/actor/code_act/test_bind_load_confinement.py` (every library call from cells with the loaders and `activate` watched, a scripted `act`, the choke points; plus an in-process control) and `test_workspace_environment.py::test_with_python_in_the_worker_the_harness_reads_the_environment_by_path`. That file's in-process tests, and the tests of the in-process loaders (`test_callable_return`, `test_in_process_proxy_state_modes`, two in `test_basics`, three in `test_env_observers`, the in-process prompt-function test in `test_sub_agent`), now pin `UNIFY_WORKSPACE_PYTHON=""`, the mode whose behaviour they test; `test_search_skip_unloadable` and `test_environment_namespaces`' stored-function test run in both modes, and `test_sub_agent` gained the worker case of a prompt function (refused, nothing executed).

**Trace (who reaches the in-process loaders and `activate`, worker on):**

| Caller | Reached from | Before | Now |
|---|---|---|---|
| `_inject_callables_for_functions` ← `list/filter/search_functions(_return_callable=True)` ← `FunctionLibrary._load` | cells (`functions.get/search/filter/list`), CM's actor (same surface), task start (`_bind_names`: CORE_BIND_LISTED), guidance reads (GUIDANCE_LINKED_NAMES); not the review fork (its reads bind nothing) | executed in harness | `_bind_stored_sources`, nothing executed |
| `_inject_callables_for_functions` ← `FunctionStoreEnvironment.get_sandbox_instance` | `primitives.actor.act(prompt_functions=[stored name])`, only with `can_spawn_sub_agents` | executed in harness; calls ran there | bound as `StoredSource`; a call from the sub-actor's cell is refused (open item below) |
| `_store_check` → both loaders | `functions.add/patch` from cells, review fork, CM | static only (PR #198) | unchanged |
| candidate loader (`store_verify`, case replay) | review's check, overwrite/patch | refused (PR #201) | unchanged |
| `function_helpers.Helpers.define` | no caller in `unify/` (went with the `execute_function` tool) | — | choke point raises |
| `fm.execute_function` / `_execute_python_function` | only `_InProcessFunctionProxy` (made by `_create_in_process_callable`) | unreachable | choke point raises |
| `environment.activate` | act start (`code_act_actor.py`); `missing()` ← `ensure()` ← `functions.run` (`_begin`), bind, `execute_function`; `_create()` ← `install()` ← cell `install`, `install_python_packages`, `%pip` | venv on harness `sys.path` | not called; refuses if reached |

**Open design choices (lead):**
- **A store verifier with the worker on:** (a) keep the refusal (the switch needs Python in process, in a process the benchmark confines as a whole); (b) split the verifier: held-out description and verdict stay harness-side, talking to the trusted side directly, and only the candidate runs, in a fresh sandboxed worker that reaches the held-out world through a socket bound into the sandbox; (c) change the contract so the verifier receives the source only and the benchmark's trusted side runs it in its own confinement.
- **The case replay with the worker on:** (a) keep the fail-closed interim (changes of a function with passing cases need a new name or a retired case); (b) fail open (inconclusive, as the store check does); (c) replay in a fresh sandboxed worker holding only the new source, its callees' sources and the case's recorded trace, with the verdict computed harness-side from what the worker reports.

**Still open, same class:**
- **Declared dependencies** were installed by the harness with its full environment and no confinement. Partly fixed; see "Package installs" below.
- **Restore later:** `tests/function_manager/core/test_stored_metadata.py`'s `default` and `decorator` cases ("add_functions does not execute"). They were dropped while the check executed; restore them once the test-suite PR lands.

### Package installs (8 Oct 2026, branch `fix-dependency-install-confinement`)

**How a model-chosen package reaches `uv`.** Every route ends in `environment.install` (`unify/environment.py`), which runs `uv pip install --python <venv>/bin/python SPEC...` in the harness process:
- `functions.run(name)` from a cell → `FunctionLibrary._begin` (`actor/core_surface.py:667`) → `environment.ensure` of the function's and its helpers' `dependencies`;
- `functions.get` / `functions.search` / `functions.filter` from a cell → `FunctionLibrary._load` (`core_surface.py:328`, `_return_callable=True` into the harness's shadow namespace) → `_inject_callables_for_functions` → `_create_in_process_callable` (`function_manager.py:2342`) → `environment.ensure`. **This path also executes the stored `def` (and its callees) in the harness**: a probe on this branch (core world, worker on) stored a function with `dependencies=["humanize"]`, then a cell's `await functions.get('double')` recorded both `_inject_dependencies`/`_create_in_process_callable` in the harness and an `ensure(["humanize"])`. Same class as the store-check escape; not fixed here.
- `install(...)` from a cell (`core_surface.py:1037`), the `install_python_packages` JSON tool off the core surface (`code_act_actor.py:4375`), and `%pip install` under `UNIFY_CODE_PROJECTION=notebook` (`notebook_cells.py:777`) → `environment.install` directly;
- `fm.execute_function` (sub-agents, proxy re-entry; `function_manager.py:3441`) → `environment.ensure`;
- `functions.add(..., dependencies=...)` only validates PEP 508 (`function_manager.py:1188`); with the worker on the store check no longer loads or installs (PR #198); with Python in-process it still installs through `_create_in_process_callable`. The review has no `install` and cannot `run`, but what it stores is installed by later sessions' `run`/`get`/`search`.
- The worker has no install of its own: `PythonWorker._start` (`actor/execution/worker.py:263`) creates the venv directory empty so it can be mounted read-only, and the harness fills it in place.

**What `uv` ran with, before this branch.** The harness process, under `sandbox.unconfined()` (deliberately outside a cell's subprocess confinement), with no `env=`: the harness's whole environment, provider keys included; the host network; the host filesystem with the harness user's rights; cwd the harness's. `install()` passed its list straight into argv, so `install(["--index-url", URL, ...])` or `["-r", PATH]` were installer options. Build steps (sdist `setup.py`, PEP 517 backends, git checkouts) run as uv's children and inherit all of it. A real offline install of a local package whose build backend probes its world saw `ANTHROPIC_API_KEY`, `OPENROUTER_API_KEY` and the rest, and could write outside the workspace (`test_a_package_build_step_runs_confined`, failing before the change).

**Fixed on this branch** (`tests/actor/code_act/test_install_confinement.py`, four tests, all failing before):
- (a) `uv venv` and `uv pip install` get an explicit environment (`environment.installer_env`): `PATH`, `HOME`, locale, `TZ`, CA bundle, proxy and `UV_INDEX*`/`UV_NATIVE_TLS`/`UV_OFFLINE`/cache variables, then credential-named names dropped. Applies with or without the sandbox.
- (b) With `UNIFY_WORKSPACE=sandboxed`, `uv pip install` runs in bubblewrap under the workspace policy (`sandbox.wrap_argv` gained `writable=` and `share_network=`): `/` read-only, `UNIFY_HOME` hidden, credential directories and `.env` files masked, private `/tmp` and `/proc`; writable are only the venv and a dedicated cache `<UNIFY_HOME>/uv-cache` (never the host's uv cache, which unconfined installs reuse); cwd is the venv, so no `pyproject.toml`/`uv.toml` a cell wrote applies. The network is the host's, as before. `uv venv` names no package and stays unconfined, with the minimal environment.
- Specifiers starting with `-` are refused before anything runs.
- Behaviour changes: no reuse of the host's uv cache (first installs download again); `UV_LINK_MODE=copy` inside the sandbox; an operator's `~/.config/uv/uv.toml` is masked in sandboxed mode (environment variables still pass); variables outside the list (e.g. `PIP_*`, other `UV_*`) no longer reach uv.

**Fixed since (branch `fix-bind-load-confinement`, above):** the bind loads that executed stored `def`s in the harness, and `activate()` putting the venv on the harness's `sys.path`.

**Still open after the bind-load fix:**
- **`prompt_functions` naming stored functions, worker on.** `FunctionStoreEnvironment` hands the sub-actor a namespace object, which the worker reaches as a remote object; its functions are now `StoredSource`s, so a call fails with "runs only in the sandboxed worker" instead of executing in the harness. Making it work needs a worker message kind for "a namespace of stored functions defined here" (or binding them as top-level names); a design choice (lead/COMMS, since sub-actors are theirs).
- **Context compression executes model-written code in the harness.** `compress_messages` (`common/_async_tool/context_compression.py`) gives the compression model an `update(index, transformation)` tool whose Python runs by `exec` in the harness (`_eval_transformation`), over transcript text that holds untrusted content. Reached through `compress_and_rebuild` whenever the fork summary (`UNIFY_CACHE_DISCIPLINE`) is off or returns nothing. Compression is untouchable in the freeze, so not changed here; options: run the transformation in a sandboxed worker, or restrict it to a non-executing edit form (literal replace / slice).
- **Installer egress and the metadata server** (options 2 below) and the other policy options are unchanged.
- **Dead code:** `function_helpers.Helpers.define` / `plan` have no caller left in `unify/` (they served the removed `execute_function` tool); with the worker on `define` would hit the choke point. Delete them with the JSON-tool leftovers.

**Options for the lead (policy; not chosen here):**
1. **Build steps.** (i) Keep them, confined as above: any sdist, git or path specifier works. (ii) `--only-binary :all:`: no build step ever runs; packages without a wheel for this platform, `pkg @ git+...` and local paths fail with uv's error, which the model can read. (iii) Binary-only plus an allow-list of names allowed to build. (iv) A prebuilt wheelhouse (`--no-index --find-links DIR`) curated offline: the strongest, and closes the network question too, but packages outside it are unavailable. Wheels still run code where they are imported: the worker, confined and without network (with the worker on the harness no longer imports from the environment).
2. **Installer egress.** The confined installer keeps the host network, including host loopback services and, on GCP workers, the metadata server (`169.254.169.254`), which hands out the attached service account's token. A build step can reach it. Options: (i) keep it, accepting that risk on GCP; (ii) route the installer through `UNIFY_WORKSPACE_NETWORK=proxy` with the proxy allowing only the index hosts (`pypi.org`, `files.pythonhosted.org`, or a mirror); `wrap_argv`'s proxy forwarder exists, and the proxy is the operator's; (iii) no network: offline from the wheelhouse of 1(iv); (iv) block link-local and loopback with a netns plus slirp4netns/pasta, which is new machinery. With 1(ii), (i) still exposes the metadata server to the resolver and downloader (uv itself), but no package code runs at install time.
3. **Which packages.** An allow-list of names (or an internal index mirror via `UV_INDEX_URL`) bounds typosquats and malicious uploads; it limits what tasks can use.
4. **Credentials in URLs.** A proxy or index URL with `user:password@` passes to the installer (and to build steps) unchanged; scrubbing is by name. Options: strip userinfo, or require index authentication through a keyring or netrc that stays outside the sandbox.
5. **Without the sandbox** (`UNIFY_WORKSPACE` unset), installs still run unconfined (only the environment is minimal); there, cells run in-process with the harness anyway.

## Known defects and test status
- **`library_shortlist.shortlist_block` (lines ~435–437) and `_gated_block` swallow any exception at DEBUG level.** A missing embedding key silently drops the shortlist. Log a warning at least.
- **`test_can_store_true_merges_redundant_functions`** passes 3/5 live under the old defaults. The review sometimes searches only by the new function's exact name, so it never sees the narrower variants.
  - The baked review (fork + shortlist) should expose them. This has to be measured (n ≥ 10 live, ≥ 9/10).
  - If it doesn't pass, the fix is a general review change with its own evidence.
- **conversation_manager's actor now inherits the core surface and the agent record.** That has not been reviewed.
- **Tests still pin a removed switch to its LOSING value** (about 26 sites, e.g. `test_prompt_accuracy`, `test_patch_tools`, `test_prompt_trim`, `test_functions_first`, `test_library_snapshot`). Those tests exercise deleted paths: delete them, don't skip them. `tests/baked_defaults.py`'s `as_shipped` pins (29 modules, all tied to the JSON-tool and handle machinery) go with those paths.
- **`ARCHITECTURE.md` and `README.md`** still describe the old JSON-tool, in-process behaviour.
- **The final full test run** happens on branch `freeze-final-tests`, then as follow-up PRs. The lead's standard is zero failures, no allowlists, and skips only by design, each with its reason.
