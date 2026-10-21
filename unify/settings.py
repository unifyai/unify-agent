"""
unify/settings.py
==================

Centralized runtime settings using pydantic-settings.

Each setting is read from the environment variable of the same name (the CLI
also loads a ``.env`` file from the working directory into the environment
first). CONFIG.md lists every variable the package reads.
"""

import os
import re
from typing import Any

import unillm
from pydantic import SecretStr, field_validator
from pydantic_settings import BaseSettings

# The reasoning efforts unillm forwards to providers (it maps them per
# provider, e.g. DeepSeek's high/max), with "" for the model's own default.
_REASONING_EFFORTS = ("", "none", "minimal", "low", "medium", "high", "xhigh", "max")


class ProductionSettings(BaseSettings):
    """The package's runtime configuration: models, paths, limits and the memory mode."""

    # ─────────────────────────────────────────────────────────────────────────
    # Local Workspace
    # ─────────────────────────────────────────────────────────────────────────
    # Root directory for local file operations, CodeActActor working directory
    # and virtual environments. Defaults to ``<UNIFY_HOME>/workspace`` (with
    # ``UNIFY_HOME`` defaulting to ``~/.unify``) when empty.
    UNIFY_LOCAL_ROOT: str = ""

    # ─────────────────────────────────────────────────────────────────────────
    # Core LLM Settings
    # ─────────────────────────────────────────────────────────────────────────
    UNIFY_MODEL: str = "openai/gpt-5.6-sol@openrouter"
    # Reasoning effort paired with UNIFY_MODEL. Empty leaves the model's own
    # default.
    UNIFY_REASONING_EFFORT: str = "high"
    # The LLM endpoints model-written code may name (the ``query_llm``,
    # ``list_llms`` and ``unillm`` globals of a cell, unify/common/cell_models.py).
    # Empty: only the session's configured model (what ``model=None``
    # resolves to). ``any``: every endpoint.
    # Otherwise a comma list of ``model@provider`` endpoints allowed besides
    # the session's model. Harness calls (the actor's loop, reviews,
    # compression) never read it.
    UNIFY_CELL_LLM_MODELS: str = ""
    # Ceiling on output tokens for one actor turn. Unset, the provider ceiling
    # applies (128k on current OpenAI models), so a turn that degenerates into
    # repetition bills and blocks for the full window — observed at eight
    # minutes. Sized to bound that, not to shape normal turns: reasoning tokens
    # count toward it, so keep it well above what a long reasoning turn plus a
    # large code block needs. Set to 0 to restore the provider ceiling.
    UNIFY_MAX_OUTPUT_TOKENS: int = 32768

    # Ceiling on tool-calling iterations for one agent run. Unset, a run ends
    # only when the model decides to stop, so a loop that never converges
    # bills indefinitely. Sized to bound that rather than to shape normal
    # runs: a long agentic task uses far fewer steps than this, and a caller
    # that genuinely needs more can pass ``max_steps`` explicitly. Set to 0 to
    # restore unbounded iteration.
    UNIFY_MAX_TOOL_LOOP_STEPS: int = 300

    # ─────────────────────────────────────────────────────────────────────────
    # Environment Namespaces
    # ─────────────────────────────────────────────────────────────────────────
    # ``package.module:factory`` entries, comma-separated: factories that
    # register the environment's own callable surface as ``primitives.<name>``
    # namespaces at start-up (unify/function_manager/primitives/environment.py).
    # Empty registers nothing.
    UNIFY_ENV_NAMESPACES: str = ""
    # Memory v2's dialogue capture (applies only with UNIFY_MEMORY on;
    # parsed by unify/memory/integration/switch.py): ``env`` also records
    # each request's dialogue actions (every turn-ending reply's action with
    # the counterpart's next message) on the channel ``env``, so a counterpart
    # whose actions are text in the replies leaves coverable
    # actions. Empty (also ``off``): episodes as shipped.
    UNIFY_MEMORY_DIALOGUE: str = ""
    # Memory v2 (continual-harness-research docs/design/memory-redesign-spec.md):
    # ``on`` replaces the storage review, the ``functions``/``guidance``
    # objects and the library shortlist with a per-request export of the
    # memory repo (``<UNIFY_HOME>/memory``) on the worker's import path and
    # its index at the end of the system prompt (unify/memory/integration).
    # Needs the sandboxed worker and the core tool surface. Empty or ``off``:
    # as shipped. The companions (parsed by unify/memory/integration/
    # switch.py) apply only when it is on: ``_E`` is the experience budget in
    # tokens at which a batched consolidation pass becomes due; ``_WRITER_MODEL``
    # runs the passes, at the actor's reasoning effort for the run;
    # ``_WRITER_ALLOWANCE_USD_PER_TOKENS`` (a decimal string) times E caps one
    # pass's USD; ``_WRITER_RUN_GUARD_USD`` (a decimal string, empty for none)
    # stops further passes once the run's writer USD plus the next cap would
    # exceed it. ``_WRITER_BASE_URL`` and ``_WRITER_TOKEN`` (both or neither) send
    # the writer's calls to a route of their own (a proxy listener with its own
    # token); empty, they go as shipped. They are checked when a pass starts
    # (a settings error would print the value), the token is a SecretStr and
    # leaves this process's environment once read (below). ``_WRITER_TOKEN_FD``
    # (instead of ``_WRITER_TOKEN``, never both) names an inherited descriptor
    # holding the token: read once when settings are settled (below), then
    # closed; the token never enters the environment. ``_WRITER_EFFORT`` is
    # ``actor`` (the writer's effort is the actor's for the run) or a fixed
    # ``low``/``medium``/``high`` for a declared mismatch ablation.
    # ``_WRITER_EFFORT_SCALE`` (``low:1,medium:2,high:5``) multiplies a pass's
    # USD cap and ``_WRITER_MAX_CALLS`` (``low:40,medium:80,high:80``) bounds its
    # calls, each by the pass's writer effort. The library
    # surfacing switches (same parser; each default is
    # the v2 screen build's behaviour): ``_SURFACING`` is ``index`` (the v2
    # per-function index in the prompt, nothing generated in the export) or
    # ``catalogue`` (a constant guide in the prompt; generated
    # README, catalogue and ``memory`` helper in the export; input shapes
    # recorded per commit); ``_DOCSTRINGS`` ``off``/``on`` is the gate's lean
    # docstring standard and examples run; ``_SOFT_BUDGET`` ``off``/``on``
    # turns G4's 4,000-token index refusal into a hygiene note.
    # ``_WRITER_USAGE`` (empty/off or on) ends each pass's first message with
    # the table of how requests used each library function.
    UNIFY_MEMORY: str = ""
    UNIFY_MEMORY_E: int = 150000
    UNIFY_MEMORY_WRITER_MODEL: str = "openai/gpt-6-sol"
    UNIFY_MEMORY_WRITER_ALLOWANCE_USD_PER_TOKENS: str = "0.00000073"
    UNIFY_MEMORY_WRITER_RUN_GUARD_USD: str = ""
    # In the library mode a pass with a writer route runs with no pass cap, bounded in flight only by the route's
    # ceilings; point this only at a proxy with a spend ceiling (the runner's writer lane, the replay proxy), never
    # at an uncapped relay.
    UNIFY_MEMORY_WRITER_BASE_URL: str = ""
    UNIFY_MEMORY_WRITER_TOKEN: SecretStr = SecretStr("")
    UNIFY_MEMORY_WRITER_TOKEN_FD: str = ""
    UNIFY_MEMORY_WRITER_EFFORT: str = "actor"
    UNIFY_MEMORY_WRITER_EFFORT_SCALE: str = "low:1,medium:2,high:5"
    UNIFY_MEMORY_WRITER_MAX_CALLS: str = "low:40,medium:80,high:80"
    UNIFY_MEMORY_SURFACING: str = "index"
    UNIFY_MEMORY_DOCSTRINGS: str = "off"
    UNIFY_MEMORY_SOFT_BUDGET: str = "off"
    UNIFY_MEMORY_WRITER_USAGE: str = ""
    # The library mode (unify/memory): ``off`` (default; empty means it) or
    # ``on``. On: the writer reads a batch map and bounded views and keeps a
    # function library. Off: the earlier memory mode as built.
    UNIFY_MEMORY_LIBRARY: str = "off"
    # Checker visibility: ``off`` (default) or ``on``. On (with UNIFY_MEMORY_LIBRARY), a
    # ``{"checker": {"label": ...}}`` stdin line records the verdict the bed
    # showed the actor as an agent-visible checker signal. Off: refused.
    UNIFY_MEMORY_LIBRARY_CHECKER_VISIBLE: str = "off"
    # The comparison profile: empty (default: today's behaviour) or ``paper``, the one setting every reported row
    # runs under: the memory writer on the actor's model, at the actor's effort, through its own accounting lane
    # (the writer route, required), and no checker line. A paper run that also sets one of those settings to another
    # value is refused before anything runs (unify/memory/integration/switch.py, ``profile_conflicts``). Recorded in
    # each episode's meta and each pass's start event.
    UNIFY_PROFILE: str = ""
    # Memory work after a request: empty or ``on`` (default: today's behaviour) or ``off``: the actor's memory
    # surface is unchanged, and no pass, writer call or episode-end fork runs, so the library never changes; the
    # other memory settings keep their values. Recorded in each episode's meta when off.
    UNIFY_MEMORY_PASSES: str = ""
    # Consolidation passes: empty or ``on`` (default: today's behaviour) or ``off``: in the library mode with the
    # episode-end fork on, the settled forks are still captured and the actor still runs the captures, and no pass,
    # writer or helper call runs, so the library never changes. Refused without the library mode, the fork, or with
    # UNIFY_MEMORY_PASSES=off.
    UNIFY_MEMORY_WRITER_PASSES: str = ""
    # The library mode's E (recorded tokens at which a pass becomes due), one pass's wall-clock bound (s), and the
    # writer route proxy's journal (absolute path; empty: none) used only to reconcile cancelled calls.
    UNIFY_MEMORY_LIBRARY_E: int = 100000
    # The library mode: the writer's reader calls per pass (PassConfig.max_reads).
    UNIFY_MEMORY_LIBRARY_MAX_READS: int = 400
    # The library mode: ``on`` makes the CLI wait for the pass slot it spawned before exiting (sandboxes whose PID
    # namespace ends with the controller); ``off`` (default): the request returns at once.
    UNIFY_MEMORY_LIBRARY_WAIT_SLOT: str = "off"
    # The library mode: ``on`` forks the actor's conversation once at each episode's end, in a confined process
    # writing only to the episode's staging dir (integration/fork.py); ``off`` (default): nothing happens.
    UNIFY_MEMORY_LIBRARY_FORK: str = "off"
    # The world notebook (unify/memory/notebook.py): ``on`` marks each episode's consequence moments for the fork,
    # lets the writer keep lessons as counted rates and propose untried remedies as trials, and appends the
    # established lessons and open trials to the actor's memory section; ``off`` (default): nothing of it runs and
    # every prompt is as before.
    UNIFY_MEMORY_NOTEBOOK: str = "off"
    # The library mode with the fork on: ``on`` makes every stored capture make a writer pass due at once, at most
    # UNIFY_MEMORY_LIBRARY_CAPTURES_PER_PASS capture episodes per pass (default 1), and the experience budget alone
    # makes no pass due; ``off`` (default): a pass is due at E or on a verdict, as built.
    UNIFY_MEMORY_LIBRARY_REVIEW_EACH_CAPTURE: str = "off"
    UNIFY_MEMORY_LIBRARY_CAPTURES_PER_PASS: int = 1
    # The library mode with the fork on: ``on`` offers a capture whose run() never reads its input or only returns
    # literals as a worked example to read, never as a procedure to call; ``off`` (default): as built.
    UNIFY_MEMORY_LIBRARY_FIXED_ANSWERS_AS_EXAMPLES: str = "off"
    # The library mode with the pool on: ``on`` stops offering a capture whose pool entry the writer superseded or
    # retired, and offers the item that superseded it in its place; ``off`` (default; also empty without the redesign
    # profile): as built. Empty stays empty so that UNIFY_MEMORY_PROFILE=redesign can give it ``on``.
    UNIFY_MEMORY_LIBRARY_WITHDRAW_CLOSED_CAPTURES: str = ""
    # ``redesign`` turns on every switch of the general harness redesign together (switch.REDESIGN_SWITCHES): a switch
    # left empty takes the profile's value, and one set to another value, or a setting the profile cannot run with, is
    # refused before anything runs; empty (default): each switch as set.
    UNIFY_MEMORY_PROFILE: str = ""
    # What a capture records, what makes the writer read, what the library holds, and memory during an episode
    # (unify/memory/integration/switch.py). Each is empty (off) by default; under UNIFY_MEMORY_PROFILE=redesign an empty
    # value takes the profile's value (on; LEDGER_K 4; WRITER_TESTS value; replay refused until G2). ``UNIFY_MEMORY_CAPTURE_TRACE``: a capture records what its episode
    # did (the trace, a value only when one was handed back) in one of three forms, its own check as evidence.
    # ``UNIFY_MEMORY_ATTENTION_LEDGER``: the attention ledger makes the writer read every entry that needs it, in
    # passes of ``UNIFY_MEMORY_LEDGER_K`` entries (empty: 4). ``UNIFY_MEMORY_IMPURE_ITEMS``: the library holds
    # side-effecting items too, and every quality result is a label. ``UNIFY_MEMORY_WRITER_TESTS``: ``value`` or
    # ``value,replay``, the tests the writer states and runs. ``UNIFY_MEMORY_IN_EPISODE``: memory changes during an
    # episode (checkpoints, segment forks, a re-pin at the next cell boundary).
    UNIFY_MEMORY_CAPTURE_TRACE: str = ""
    UNIFY_MEMORY_ATTENTION_LEDGER: str = ""
    UNIFY_MEMORY_LEDGER_K: str = ""
    UNIFY_MEMORY_IMPURE_ITEMS: str = ""
    UNIFY_MEMORY_WRITER_TESTS: str = ""
    UNIFY_MEMORY_IN_EPISODE: str = ""
    # Rev 2's gate (unify/memory/gate_policy.py): ``UNIFY_MEMORY_GATE_LABELS_ONLY=on`` makes every check but the hard
    # ones a label and runs only statically pure items; ``UNIFY_MEMORY_GATE_RECORDED_REPLAYS=on`` adds the replays of
    # recorded task data, for comparisons. Empty: off (the redesign profile gives on and off).
    UNIFY_MEMORY_GATE_LABELS_ONLY: str = ""
    UNIFY_MEMORY_GATE_RECORDED_REPLAYS: str = ""
    # Serving on the situation plus the failure record: ``on`` gives the librarian each item's failure record and
    # contradictions beside the situation, shows match and applicability results as evidence (never as a claim of
    # fit), states each status by what it rests on, and withdraws retired and superseded library items as well as
    # captures; ``off`` (default; empty without the redesign profile): as built.
    UNIFY_MEMORY_SERVING_FAILURE_RECORD: str = ""
    # Each episode's feedback regime: ``inferred`` infers it from the signals that arrive (in the episode or after it),
    # with the run's own history as the prior, and takes late verdicts on earlier episodes; ``off`` (default; empty
    # without the redesign profile): every episode is ``dense``, as built.
    UNIFY_MEMORY_REGIME: str = ""
    # Held-out evaluation: ``on`` freezes memory as the home holds it. The request is served from what is stored and
    # changes nothing: no episode is recorded, no writer runs (service, pass, fork, segment fork), the copy is never
    # re-pinned, and a message to the writer is answered "not available: memory is frozen". ``off`` (default): as built.
    UNIFY_MEMORY_FROZEN: str = ""
    # The fork lane's route: the key proxy's /fork/v1 base URL and the inherited
    # descriptor carrying its token, read once at start-up and then removed from the environment (empty: no fork lane).
    UNIFY_MEMORY_LIBRARY_FORK_BASE_URL: str = ""
    UNIFY_MEMORY_LIBRARY_FORK_TOKEN_FD: str = ""
    # Retired: helpers are launched by the writer with its delegate tool; empty or ``off`` only (anything else is
    # refused, so a configuration that still asks for harness-launched helpers fails loudly).
    UNIFY_MEMORY_LIBRARY_ANALYSTS: str = "off"
    # The model of writer-launched helpers; empty: the model the episode-end fork uses (the actor's UNIFY_MODEL).
    UNIFY_MEMORY_LIBRARY_HELPER_MODEL: str = ""
    # The helper lane's route: the key proxy's /helper/v1 base URL and the inherited descriptor carrying its token,
    # read once at start-up and then removed from the environment (empty: no helper lane).
    UNIFY_MEMORY_LIBRARY_HELPER_BASE_URL: str = ""
    UNIFY_MEMORY_LIBRARY_HELPER_TOKEN_FD: str = ""
    # ``on`` keeps every proposal open across passes in the candidate pool (memory repo
    # refs/heads/pool) until the writer closes it; ``off`` (default): none.
    UNIFY_MEMORY_LIBRARY_POOL: str = "off"
    # ``on`` gives the library writer and the curator the ``write`` and ``edit`` file tools on /memory; ``off``
    # (default): neither tool exists.
    UNIFY_MEMORY_LIBRARY_EDIT_TOOLS: str = "off"
    # On by default (also when empty): the memory agents compact their conversations as the agent loop compacts its
    # own (same trigger, summary and restart; UNIFY_COMPACTION_KEEP_PREFIX applies); ``off``: never.
    UNIFY_MEMORY_LIBRARY_COMPACT: str = "on"
    # The former reading rule: empty or ``optional`` (a no-op; finish never waits on reading). ``required`` is
    # refused at start-up.
    UNIFY_MEMORY_LIBRARY_COVERAGE: str = ""
    # How the actor's prompt reaches the library index: ``inline`` (default) ends it with the index view;
    # ``pointer`` ends it with a constant guide pointing to memory.index(), memory.show() and memory.find().
    UNIFY_MEMORY_LIBRARY_ACTOR_INDEX: str = "inline"
    # ``on``: the actor asks a librarian agent which stored items fit (``memory.ask()`` in a cell, through the
    # in-code tool ``ask_library``); the export also writes the librarian's catalogue and the procedures' dispatch
    # file, and the actor's prompt shows a constant pointer in place of the index. Requires UNIFY_TOOLS_IN_CODE=on.
    # ``off`` (the default): nothing of it exists or runs.
    UNIFY_MEMORY_LIBRARIAN: str = "off"
    # The library mode: USD per recorded token of a pass's cap (the earlier mode's rate until an offline replay
    # sizes it).
    UNIFY_MEMORY_LIBRARY_WRITER_USD_PER_TOKEN: str = "0.00000073"
    UNIFY_MEMORY_LIBRARY_PASS_WALL_S: int = 2700
    UNIFY_MEMORY_WRITER_JOURNAL: str = ""
    # The writer model's prices in USD per million tokens, as the run pins them (JSON: input, output, optionally
    # cache_write). A writer call the journal shows with tokens but no charge is priced from them; empty: it stays
    # unpriced and the run guard books it at the worst case.
    UNIFY_MEMORY_WRITER_PRICES_USD_PER_MILLION: str = ""
    # Stage-5 test checks in the memory v2 gate (unify/memory/qa.py), each
    # off by default and read only while UNIFY_MEMORY is on: ``_QA_FIXTURES``
    # (``on``/``strict``) draws seeded random recorded inputs per new function;
    # ``_QA_MUTATION`` mutation-tests it, refusing below
    # ``_QA_MUTATION_MIN_KILL`` (a decimal string share, 0.5); ``_QA_DETERMINISM``
    # pins clock and randomness and runs new tests twice; ``_QA_REPLAY`` refuses
    # stand-in environments; ``_QA_FIXTURE_SIZE`` bounds test files and
    # references recorded payloads by blob id.
    UNIFY_MEMORY_QA_FIXTURES: str = ""
    UNIFY_MEMORY_QA_MUTATION: str = ""
    UNIFY_MEMORY_QA_MUTATION_MIN_KILL: str = "0.5"
    UNIFY_MEMORY_QA_DETERMINISM: str = ""
    UNIFY_MEMORY_QA_REPLAY: str = ""
    UNIFY_MEMORY_QA_FIXTURE_SIZE: str = ""

    # ─────────────────────────────────────────────────────────────────────────
    # Workspace Sandbox
    # ─────────────────────────────────────────────────────────────────────────
    # The Python worker that runs every cell, and every subprocess a cell
    # starts, are confined by bubblewrap (unify/sandbox.py,
    # unify/actor/execution/worker.py); without bubblewrap they are refused,
    # never run unconfined.
    # ``proxy``: the sandbox's only network is one loopback port forwarded to
    # the proxy listening on 127.0.0.1:UNIFY_WORKSPACE_PROXY_PORT on the host.
    # Empty: no network at all.
    UNIFY_WORKSPACE_NETWORK: str = ""
    UNIFY_WORKSPACE_PROXY_PORT: int = 0
    # Variables of the harness's environment a sandboxed command (the Python
    # worker that runs every cell, and what it starts) gets besides the built-in
    # allow-list (unify/sandbox.py, CELL_ENV_NAMES and CELL_ENV_PATTERNS): a
    # comma list of names and ``PREFIX_*`` patterns, for a runner's own cell
    # variables. A credential's name or a key-shaped value is never passed.
    UNIFY_CELL_ENV_ALLOW: str = ""

    # ─────────────────────────────────────────────────────────────────────────
    # Logging / Observability
    # ─────────────────────────────────────────────────────────────────────────
    # Directory of the package's log files (``unify.log`` and
    # ``unify_info_only.log``, unify/logger.py). Empty: the ``unify`` command
    # writes them under ``<UNIFY_HOME>/logs``; a program importing the package
    # gets none.
    UNIFY_LOG_DIR: str = ""

    # The redesign's mesh, writer service and metering (unify/memory/integration/switch.py, builder A). Each is empty
    # by default: off, unless UNIFY_MEMORY_PROFILE=redesign turns it on; ``off`` keeps it off.
    # ``UNIFY_AGENTS_RUN_RECORD``: every agent of the run on one run-long record (unify/agents/run_record.py).
    UNIFY_AGENTS_RUN_RECORD: str = ""
    # ``UNIFY_MEMORY_WRITER_SERVICE``: the writer as one detached service per memory home
    # (unify/memory/writer_service.py), with its own spend cap ``UNIFY_MEMORY_WRITER_SERVICE_CAP_USD`` (a plain decimal
    # string, required with the service on).
    UNIFY_MEMORY_WRITER_SERVICE: str = ""
    UNIFY_MEMORY_WRITER_SERVICE_CAP_USD: str = ""
    # ``UNIFY_MEMORY_METERING``: pass, entry, episode and segment ids on the writer's, helpers' and forks' calls, and
    # the per-pass, per-entry and per-item views (unify/memory/metering.py).
    UNIFY_MEMORY_METERING: str = ""
    # When the host's lease ends (UTC, ``…Z``), set by the kit: the writer service stops 30 minutes before it.
    UNIFY_HOST_LEASE_END: str = ""
    # Who starts the writer service: empty/``harness`` (a request) or ``runner`` (one per cell, in the runner's own
    # sandbox that outlives every session; a request never starts it but as the fallback after it crashed).
    UNIFY_MEMORY_WRITER_SERVICE_OWNER: str = ""

    # ─────────────────────────────────────────────────────────────────────────
    # Validators
    # ─────────────────────────────────────────────────────────────────────────
    @field_validator("UNIFY_REASONING_EFFORT", mode="before")
    @classmethod
    def parse_reasoning_effort(cls, v: Any) -> str:
        value = str(v or "").strip().lower()
        if value not in _REASONING_EFFORTS:
            raise ValueError(
                "UNIFY_REASONING_EFFORT must be empty or one of "
                f"{', '.join(_REASONING_EFFORTS[1:])}, not {v!r}",
            )
        return value

    @field_validator(
        "UNIFY_MEMORY",
        "UNIFY_MEMORY_E",
        "UNIFY_MEMORY_WRITER_MODEL",
        "UNIFY_MEMORY_WRITER_ALLOWANCE_USD_PER_TOKENS",
        "UNIFY_MEMORY_WRITER_RUN_GUARD_USD",
        "UNIFY_MEMORY_WRITER_BASE_URL",
        "UNIFY_MEMORY_WRITER_TOKEN",
        "UNIFY_MEMORY_WRITER_TOKEN_FD",
        "UNIFY_MEMORY_WRITER_EFFORT",
        "UNIFY_MEMORY_WRITER_EFFORT_SCALE",
        "UNIFY_MEMORY_WRITER_MAX_CALLS",
        "UNIFY_MEMORY_SURFACING",
        "UNIFY_MEMORY_DOCSTRINGS",
        "UNIFY_MEMORY_SOFT_BUDGET",
        "UNIFY_MEMORY_WRITER_USAGE",
        "UNIFY_MEMORY_LIBRARY",
        "UNIFY_MEMORY_LIBRARY_CHECKER_VISIBLE",
        "UNIFY_MEMORY_LIBRARY_E",
        "UNIFY_MEMORY_LIBRARY_MAX_READS",
        "UNIFY_MEMORY_LIBRARY_WAIT_SLOT",
        "UNIFY_MEMORY_LIBRARY_FORK",
        "UNIFY_MEMORY_LIBRARY_REVIEW_EACH_CAPTURE",
        "UNIFY_MEMORY_LIBRARY_CAPTURES_PER_PASS",
        "UNIFY_MEMORY_LIBRARY_FIXED_ANSWERS_AS_EXAMPLES",
        "UNIFY_MEMORY_LIBRARY_WITHDRAW_CLOSED_CAPTURES",
        "UNIFY_MEMORY_LIBRARY_FORK_BASE_URL",
        "UNIFY_MEMORY_LIBRARY_FORK_TOKEN_FD",
        "UNIFY_MEMORY_LIBRARY_ANALYSTS",
        "UNIFY_MEMORY_LIBRARY_HELPER_MODEL",
        "UNIFY_MEMORY_LIBRARY_HELPER_BASE_URL",
        "UNIFY_MEMORY_LIBRARY_HELPER_TOKEN_FD",
        "UNIFY_MEMORY_LIBRARY_POOL",
        "UNIFY_MEMORY_LIBRARY_EDIT_TOOLS",
        "UNIFY_MEMORY_LIBRARY_COMPACT",
        "UNIFY_MEMORY_LIBRARY_ACTOR_INDEX",
        "UNIFY_MEMORY_LIBRARY_WRITER_USD_PER_TOKEN",
        "UNIFY_MEMORY_LIBRARY_PASS_WALL_S",
        "UNIFY_MEMORY_WRITER_JOURNAL",
        "UNIFY_MEMORY_WRITER_PRICES_USD_PER_MILLION",
        "UNIFY_MEMORY_QA_FIXTURES",
        "UNIFY_MEMORY_QA_MUTATION",
        "UNIFY_MEMORY_QA_MUTATION_MIN_KILL",
        "UNIFY_MEMORY_QA_DETERMINISM",
        "UNIFY_MEMORY_QA_REPLAY",
        "UNIFY_MEMORY_QA_FIXTURE_SIZE",
        "UNIFY_PROFILE",
        "UNIFY_MEMORY_PASSES",
        "UNIFY_MEMORY_NOTEBOOK",
        "UNIFY_MEMORY_WRITER_PASSES",
        "UNIFY_MEMORY_PROFILE",
        "UNIFY_MEMORY_SERVING_FAILURE_RECORD",
        "UNIFY_MEMORY_REGIME",
        "UNIFY_MEMORY_FROZEN",
        mode="before",
    )
    @classmethod
    def parse_memory(cls, v: Any, info: Any) -> Any:
        from unify.memory.integration import switch

        return switch.PARSERS[info.field_name](v)

    @field_validator(
        "UNIFY_MEMORY_CAPTURE_TRACE",
        "UNIFY_MEMORY_ATTENTION_LEDGER",
        "UNIFY_MEMORY_LEDGER_K",
        "UNIFY_MEMORY_IMPURE_ITEMS",
        "UNIFY_MEMORY_WRITER_TESTS",
        "UNIFY_MEMORY_IN_EPISODE",
        "UNIFY_MEMORY_GATE_LABELS_ONLY",
        "UNIFY_MEMORY_GATE_RECORDED_REPLAYS",
        mode="before",
    )
    @classmethod
    def parse_memory_learning(cls, v: Any, info: Any) -> Any:
        from unify.memory.integration import switch

        return switch.PARSERS[info.field_name](v)

    @field_validator(
        "UNIFY_AGENTS_RUN_RECORD",
        "UNIFY_MEMORY_WRITER_SERVICE",
        "UNIFY_MEMORY_WRITER_SERVICE_CAP_USD",
        "UNIFY_MEMORY_METERING",
        "UNIFY_HOST_LEASE_END",
        "UNIFY_MEMORY_WRITER_SERVICE_OWNER",
        mode="before",
    )
    @classmethod
    def parse_redesign_a(cls, v: Any, info: Any) -> Any:
        from unify.memory.integration import switch

        return switch.PARSERS[info.field_name](v)

    @field_validator("UNIFY_MEMORY_DIALOGUE", mode="before")
    @classmethod
    def parse_memory_dialogue(cls, v: Any) -> str:
        from unify.memory.integration import switch

        return switch.parse_dialogue(v)

    @field_validator("UNIFY_MEMORY_LIBRARIAN", mode="before")
    @classmethod
    def parse_memory_librarian(cls, v: Any) -> str:
        value = str(v or "off").strip().lower()
        if value not in ("on", "off"):
            raise ValueError(
                f"UNIFY_MEMORY_LIBRARIAN must be 'on' or 'off', not {v!r}",
            )
        return value

    @field_validator("UNIFY_CELL_LLM_MODELS", mode="before")
    @classmethod
    def parse_cell_llm_models(cls, v: Any) -> str:
        text = str(v or "").strip()
        if text.lower() == "any":
            return "any"
        endpoints = [part.strip() for part in text.split(",") if part.strip()]
        for endpoint in endpoints:
            model, _, provider = endpoint.rpartition("@")
            if endpoint.lower() == "any" or not model or not provider:
                raise ValueError(
                    "UNIFY_CELL_LLM_MODELS must be empty, 'any' or a comma list "
                    f"of 'model@provider' endpoints, not {v!r}",
                )
        return ",".join(endpoints)

    @field_validator("UNIFY_CELL_ENV_ALLOW", mode="before")
    @classmethod
    def parse_cell_env_allow(cls, v: Any) -> str:
        entries = [part.strip() for part in str(v or "").split(",") if part.strip()]
        for entry in entries:
            if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*\*?", entry):
                raise ValueError(
                    "UNIFY_CELL_ENV_ALLOW must be a comma list of variable names "
                    f"and PREFIX_* patterns, not {v!r}",
                )
        return ",".join(entries)

    @field_validator("UNIFY_WORKSPACE_NETWORK", mode="before")
    @classmethod
    def parse_workspace_network(cls, v: Any) -> str:
        value = str(v or "").strip().lower()
        if value not in ("", "proxy"):
            raise ValueError(
                f"UNIFY_WORKSPACE_NETWORK must be empty or 'proxy', not {v!r}",
            )
        return value

    def validate_llm_providers(self) -> None:
        """Validate that unillm holds a key it can reach an LLM provider with.

        unillm calls Anthropic models directly and every other model through
        OpenRouter, so those two keys are the only ones that serve a call.

        Raises:
            RuntimeError: If unillm resolved neither key.
        """
        keys = (unillm.SETTINGS.OPENROUTER_API_KEY, unillm.SETTINGS.ANTHROPIC_API_KEY)
        if not any(key.get_secret_value() for key in keys):
            raise RuntimeError(
                "At least one LLM provider credential is required. "
                "Set OPENROUTER_API_KEY and/or ANTHROPIC_API_KEY.",
            )


# Singleton instance for production code (former setting names are read as their current ones first)
from unify.setting_aliases import apply_former_names  # noqa: E402

apply_former_names(os.environ)
SETTINGS = ProductionSettings()
# UNIFY_MEMORY_WRITER_TOKEN lives in SETTINGS (this, the controller process)
# only: every case variant is removed from the environment once read, so no
# subprocess inherits it, and its value stays registered with the value-based
# redactors. With UNIFY_MEMORY_WRITER_TOKEN_FD set, this first settle reads the
# token from that inherited descriptor and closes it, before the CLI or any
# controller code can spawn a child. The CLI checks again after loading .env.
# Unset, nothing changes.
from unify.memory.integration.switch import settle_writer_route_env  # noqa: E402

settle_writer_route_env(os.environ, SETTINGS)
# The fork lane's token likewise: read once from its descriptor, both route names out of the environment.
from unify.memory.integration.switch import settle_fork_route_env  # noqa: E402

settle_fork_route_env(os.environ, SETTINGS)
# The helper lane's token likewise: read once from its descriptor, both route names out of the environment.
from unify.memory.integration.switch import settle_helper_route_env  # noqa: E402

settle_helper_route_env(os.environ, SETTINGS)
