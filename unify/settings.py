"""
unify/settings.py
==================

Centralized runtime settings using pydantic-settings.

All settings can be overridden via environment variables or the ``.env`` file
in the working directory.
"""

from typing import Any, Optional

import unillm
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from unify.actor.settings import ActorSettings
from unify.function_manager.settings import FunctionSettings
from unify.guidance_manager.settings import GuidanceSettings

# The reasoning efforts unillm forwards to providers (it maps them per
# provider, e.g. DeepSeek's high/max), with "" for "as shipped".
_REVIEW_EFFORTS = ("", "none", "low", "medium", "high", "xhigh", "max")


def _parse_bool(v: Any) -> bool:
    """Parse a value as boolean."""
    if isinstance(v, bool):
        return v
    if isinstance(v, str):
        return v.lower() in ("true", "yes", "1", "on")
    return bool(v)


def _step_cap_reply_mode(v: Any) -> Optional[str]:
    """``UNIFY_STEP_CAP_REPLY`` as ``""``, ``"draft"`` or ``"last_word"``.

    The switch was a bool, so the bool spellings keep their meaning (true is
    ``"draft"``). ``None`` for anything else.
    """
    if isinstance(v, bool) or v is None:
        return "draft" if v else ""
    value = str(v).strip().lower()
    if value in ("", "false", "0", "no", "off"):
        return ""
    if value in ("draft", "true", "1", "yes", "on"):
        return "draft"
    if value == "last_word":
        return "last_word"
    return None


class ConversationSettings(BaseSettings):
    """The conversation-layer settings (env prefix ``UNIFY_CONVERSATION_``).

    Defined here, not in the legacy conversation manager
    (``unify/legacy/conversation_manager/settings.py`` holds the same class),
    because the slow-brain model resolver in ``unify.common.llm_client`` and
    the manager registry read ``SETTINGS.conversation``, and nothing on the
    actor or CLI path may import ``unify.legacy``.

    Attributes:
        IMPL: Implementation type - "real" or "simulated".
        SLOW_BRAIN_MODEL: Shared ConversationManager slow-brain model. Empty
            falls back to the global shared model (UNIFY_MODEL / assistant
            default resolution). Override via
            UNIFY_CONVERSATION_SLOW_BRAIN_MODEL.
        SLOW_BRAIN_REASONING_EFFORT: Reasoning effort paired with
            SLOW_BRAIN_MODEL when that setting is non-empty. Empty leaves
            call-site effort intact. Override via
            UNIFY_CONVERSATION_SLOW_BRAIN_REASONING_EFFORT.
    """

    SLOW_BRAIN_MODEL: str = "openai/gpt-5.6-terra@openrouter"
    SLOW_BRAIN_REASONING_EFFORT: str = "high"
    IMPL: str = "real"

    model_config = SettingsConfigDict(
        env_prefix="UNIFY_CONVERSATION_",
        case_sensitive=True,
        extra="ignore",
    )


class ProductionSettings(BaseSettings):
    """Runtime settings; test settings (TestingSettings) inherit from this class.

    The code freeze (7 Oct 2026) made the chosen configuration the only code
    path: the lean-all recipe, Python tool mode (the core tool surface, with
    Python in the sandboxed worker) and the shared agent record. What remains
    here is configuration and the switches still in progress (listed in
    WIP_SWITCHES.md at the repository root); in the comments below, "as
    shipped" names upstream's behaviour, which a WIP switch's off value
    selects.
    """

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
    # Reasoning effort paired with UNIFY_MODEL when no per-assistant default is
    # set. Empty leaves per-call-site effort levels untouched.
    UNIFY_REASONING_EFFORT: str = "high"
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
    # Reaching ``max_steps`` ends the loop with a stop notice, so a persistent
    # session (``unify act --persist``) that reaches it takes no further
    # messages. Set true and, in a persistent session, the limit ends only
    # the current request: the reply says the session stopped at its step
    # limit and quotes the latest reply text it drafted for the request, the
    # pending tool calls are cancelled and answered as such, and the next
    # message starts a request with its own ``max_steps`` (the limit then
    # counts the messages of one request instead of the whole session). A
    # loop that is not persistent still ends at the limit, its stop notice
    # followed by that draft. Off: as shipped.
    # ``draft`` (also ``true``/``1``/``yes``/``on``, as when this was a bool)
    # is the behaviour above. ``last_word``: the same, except that before the
    # reply the model is given one model call with no tools offered, after a
    # notice that the request's step limit is reached and it should reply now
    # with its best answer; the stop reply carries that answer in place of
    # the draft, and the draft when the call fails or returns no text.
    # Empty (also ``false``/``0``/``no``/``off``): as shipped.
    UNIFY_STEP_CAP_REPLY: str = ""
    # ``on``: a loop that can compress its context (the actor's task loop)
    # compacts it when it reaches ``max_steps`` instead of stopping, with the
    # same compression a full context gets, and the same request goes on in
    # the same loop, its steps counted from the compacted conversation. Calls
    # still running are cancelled and answered as such first. A request is
    # compacted at most twice; at its third limit, or when the compaction
    # fails or runs past the loop's timeout, the limit stops it as it would
    # without this switch (with UNIFY_STEP_CAP_REPLY's reply when that is on).
    # A reply the request has already given (a text reply, or a cell's
    # reply()) is given, not compacted for. A loop without compression is
    # unchanged. Empty (also ``off``): as shipped.
    UNIFY_STEP_CAP_COMPACT: str = ""
    # ``on``: a request to the actor's task loop (the one that answers the
    # requester; never a sub-agent's, a review's or its fork's) whose tool
    # calls stop making progress ends early. A
    # model call makes no progress when every tool call it makes either runs
    # a Python cell that does nothing (only ``pass``, comments, prints of
    # constant text, bare constants; magics ignored) or repeats one of the
    # two calls before it in the request (tool and arguments without the
    # thought; a cell's code without comments, whitespace or magics; string
    # and number literals ignored) and gets the same result (times, ids and
    # durations ignored). A result not known yet never counts, nor does a
    # call made while other calls are still running. UNIFY_LOOP_STOP_K such
    # calls in a row end the request as the step limit does under
    # UNIFY_STEP_CAP_REPLY: ``draft`` quotes the request's latest draft;
    # ``last_word``, and also an empty UNIFY_STEP_CAP_REPLY, first gives the
    # model one tool-less turn to reply with its best answer. A persistent
    # session then waits for the next request. Any other model call, and
    # every requester message, starts the count again. Empty (also ``off``):
    # as shipped.
    UNIFY_LOOP_STOP: str = ""
    # How many no-progress model calls in a row UNIFY_LOOP_STOP allows: a
    # whole number, at least 1.
    UNIFY_LOOP_STOP_K: int = 10

    # Fail init when unillm holds no provider key. The keys live only in
    # unillm's settings, which read them from the environment, ``.env`` and, on
    # a machine with the team's service-account key, Google Secret Manager.
    UNIFY_VALIDATE_LLM_PROVIDERS: bool = True
    # Storage review of a persistent session: by default it runs once, when the session
    # ends, over the whole trajectory. Set true to also review at every completed turn
    # that ran tools (for sessions that live long enough that waiting for the end would
    # defer distillation indefinitely).
    UNIFY_TURN_STORAGE_REVIEWS: bool = False

    # ─────────────────────────────────────────────────────────────────────────
    # Skill Search
    # ─────────────────────────────────────────────────────────────────────────
    # Embed skills for semantic search with BAAI/bge-small-en-v1.5 in process
    # instead of openai/text-embedding-3-small through OpenRouter. It needs no
    # network once its weights (134 MB) are downloaded, but reads only English
    # and the first 512 tokens of each text.
    UNIFY_LOCAL_EMBEDDINGS: bool = False
    # The endpoint OpenRouter-style embedding requests are posted to, for
    # example a proxy that tracks their cost. Empty posts to openrouter.ai.
    UNIFY_EMBED_URL: str = ""

    # ─────────────────────────────────────────────────────────────────────────
    # Environment Namespaces and the Storage Check
    # ─────────────────────────────────────────────────────────────────────────
    # ``package.module:factory`` entries, comma-separated: factories that
    # register the environment's own callable surface as ``primitives.<name>``
    # namespaces at start-up (unify/function_manager/primitives/environment.py).
    # Empty registers nothing.
    UNIFY_ENV_NAMESPACES: str = ""
    # ``code+text``: model code in a cell can send the turn's reply with
    # ``reply(text)`` (in the sandboxed worker).
    # It takes only a ``str``, ends the cell at once (its output so far is
    # kept), and the turn ends with exactly that text as the reply, as if the
    # model had replied with it, without another model call (UniLLM strips
    # the whitespace around a model's text; reply() keeps it). A second
    # reply() in one turn, reply() inside a stored function or through
    # execute_function, in a session that is not answering a requester (the
    # storage review) or in a request whose answer is a ``final_response``
    # call is refused with the reason.
    # The prompt states it in one sentence where it states the reply rule,
    # and the code tool's description mentions it. Which kind of reply ended
    # a turn is recorded on its final assistant message (``_reply_source``:
    # "cell" or "text"; ``_reply_from_value``: whether reply()'s argument was
    # computed rather than a string literal), never sent to the model, and
    # counted in the CLI's run stats (``replies_from_cell``,
    # ``replies_from_value``) (unify/common/_async_tool/cell_reply.py).
    # Empty: replies are text only, as shipped.
    UNIFY_REPLY_CHANNEL: str = ""
    # ``on``: model code in a cell reads the current request as ``request``
    # (in the sandboxed worker): ``request.text``
    # is the requester's latest message, the request or a later message of a
    # persistent session, as the model reads it (without the session context
    # the harness opens the first message with); ``request.data`` is the
    # list of JSON objects and arrays found in that text, in order of
    # appearance, parsed with the standard json module. Nothing else is
    # parsed. Each cell gets a fresh, read-only ``request`` (its attributes
    # cannot be set; a cell's changes to ``request.data`` last for that cell
    # only). A variable of the model's own named ``request`` is never
    # replaced; after ``del request`` the next cell has the current request
    # again. A loop that answers no requester
    # (the storage review) has none. The prompt says so in one sentence in
    # its Sandbox Environment section; the tools are unchanged
    # (unify/common/_async_tool/bound_request.py). Empty: as shipped.
    UNIFY_BIND_REQUEST: str = ""
    # ``on``: an ``execute_code`` cell that ran in a persistent session ends
    # its result with one line naming the variables that session's cells
    # have bound, each with its type and a short shape (length, rows x
    # columns of a list of equal rows, key count, an array's shape and
    # dtype; a scalar's or short string's value, truncated), most recently
    # bound first: at most 12 names and 400 characters, then "…and N more".
    # It comes only when those names or their shapes changed since the last
    # line shown, and never when there are none; a stateless or read-only
    # cell keeps nothing, so its result has none. Names a cell did not bind
    # are left out (primitives, request, reply, the libraries, injected
    # stored functions), as are modules and names starting with ``_``. No
    # value is printed whole. It is computed where the cell ran (the sandboxed
    # worker), is the tool's own
    # result, and the prompt and tools are unchanged (unify/actor/execution/worker_child.py
    # ``Inventory``). Empty: results as shipped.
    UNIFY_VARIABLE_INVENTORY: str = ""
    # Path of a JSON file in which an external check of the session's outcome
    # admits (``{"admit": true}``) the review that runs when a session ends.
    # A missing, unreadable or malformed file, or any other ``admit``, skips
    # that review. While set, the session's own library write tools are
    # withheld and turn-level reviews are off, so the libraries change only
    # through an admitted review. ``never`` is a frozen library: writes are
    # withheld the same way, no review is ever admitted and no file is read.
    # Empty reviews every session as shipped.
    UNIFY_STORE_ADMISSION: str = ""
    # ``package.module:factory``: a verifier for the storage review. A
    # function is stored only after its exact source has passed a run on a
    # held-out task of the same kind (FunctionManager_check_function, offered
    # to the review only while this is set) and static checks against request
    # details and credentials (unify/function_manager/store_verify.py). Needs
    # UNIFY_STORE_ADMISSION. The verifier loads and calls candidates in this
    # process, so with Python in the sandboxed worker it is refused and
    # start-up stops. Empty stores without the check.
    UNIFY_STORE_VERIFY: str = ""
    # Memory v2 (continual-harness-research docs/design/memory-redesign-spec.md):
    # ``on`` replaces the storage review, the ``functions``/``guidance``
    # objects and the library shortlist with a per-request export of the
    # memory repo (``<UNIFY_HOME>/memory``) on the worker's import path and
    # its index at the end of the system prompt (unify/memory_v2/integration).
    # Needs the sandboxed worker and the core tool surface. Empty or ``off``:
    # as shipped. The companions name Sol's model, its USD budget per run (a
    # decimal string) and the consolidation trigger (``d6`` or ``batched``).
    UNIFY_MEMORY_V2: str = ""
    UNIFY_MEMORY_V2_SOL_MODEL: str = "openai/gpt-6-sol"
    UNIFY_MEMORY_V2_SOL_BUDGET_USD: str = "2.50"
    UNIFY_MEMORY_V2_TRIGGER: str = "d6"
    # When a provider refuses a forced tool choice ("required", "any" or one
    # named tool) with HTTP 400 because the model does not support it, retry
    # that call once with tool_choice "auto" and an instruction to make the
    # required call first (re-prompting once if the reply makes none), and
    # send later forced calls to that model this way for the rest of the
    # process (unify/common/tool_choice_fallback.py). Off: the error
    # propagates as shipped.
    UNIFY_TOOL_CHOICE_FALLBACK: bool = False
    # ``stored``: a guidance search without reference text returns the stored
    # guidance entries, newest first, instead of the built-in catalogue (whose
    # hash-derived ids otherwise sort ahead of every stored entry). Empty
    # searches as shipped.
    UNIFY_GUIDANCE_EMPTY_QUERY: str = ""
    # execute_code takes state_mode (an untyped optional string), session_id
    # and session_name, and four tools manage sessions. An omitted mode runs
    # the cell in session 0, the task's persistent session, but a model that
    # fills in every argument writes a mode into every cell, and next to
    # execute_function (whose own default is "stateless") and inspect_state
    # ("run stateless") it writes "stateless": nothing a cell computed is
    # there for the next. On: execute_code has no state_mode or session
    # argument and every cell runs in session 0, like a notebook;
    # list_sessions, inspect_state, close_session and close_all_sessions are
    # not offered; execute_function keeps its state_mode (stateless by
    # default; stateful and read_only use session 0) and has no session
    # argument; the prompt names no cell mode and no session tool
    # (unify/actor/cell_state.py). Off: as shipped.
    UNIFY_STATEFUL_CELLS: bool = False
    # A cell runs as the body of an async wrapper function, and a name it
    # binds reaches the session only if the wrapper declares it global. On:
    # the declaration is every name Python's symbol table finds the cell's
    # own scope binding (under if/for/while/with/try/except/match, a walrus,
    # also in a comprehension, del, nested def and class), and an annotated
    # assignment there drops its annotation (an annotated name cannot be
    # declared global, so today the whole cell is a SyntaxError); nested
    # functions, classes, lambdas and comprehensions keep their own names.
    # This is what the prompt promises ("a notebook"). If the symbol table
    # cannot be built, the declaration is as shipped. A `del primitives` is
    # renamed like an assignment to it (`del _primitives_local`), so a cell
    # cannot delete the injected global. Off: only names bound
    # by top-level assignments, imports, defs and classes are kept; any other
    # is lost when the cell ends (a later cell gets NameError), as shipped.
    # Read per cell; in-process and worker cells alike.
    UNIFY_CELL_SCOPE_FIX: bool = True

    # ─────────────────────────────────────────────────────────────────────────
    # Workspace Sandbox
    # ─────────────────────────────────────────────────────────────────────────
    # Bash cells, every subprocess a Python cell starts, and the Python worker
    # that runs every cell are confined by bubblewrap (unify/sandbox.py,
    # unify/actor/execution/worker.py); without bubblewrap they are refused,
    # never run unconfined.
    # ``proxy``: the sandbox's only network is one loopback port forwarded to
    # the proxy listening on 127.0.0.1:UNIFY_WORKSPACE_PROXY_PORT on the host.
    # Empty: no network at all.
    UNIFY_WORKSPACE_NETWORK: str = ""
    UNIFY_WORKSPACE_PROXY_PORT: int = 0
    # What the model is asked to fill in to run a cell. Empty (or "legacy"):
    # ``execute_code`` as shipped, with ``thought``, ``state_mode``,
    # ``session_id``, ``session_name`` (and ``language`` in a sandboxed
    # workspace) as fields, the session JSON tools, and a JSON envelope
    # before each cell's output. "notebook": the tool takes one field,
    # ``code``, and where a cell runs is written in it as Jupyter magics on
    # its first lines -- ``%%bash``, ``%pip install PKG``, ``%%scratch``
    # (stateless), ``%%what_if`` (read-only), ``%%session NAME`` and
    # ``%sessions`` (the session tools' data) -- mapped onto the unchanged
    # function behind the tool; an unknown or misplaced magic is refused
    # with what to write instead. The cell's first comment (or first code
    # line) becomes its ``thought``; its result reads as a notebook cell
    # (stdout, ``[stderr]``, ``Out: <repr>``, the traceback), the
    # ``ExecutionResult`` object being unchanged; the session JSON tools are
    # not offered and the prompt names the magics where it named the fields
    # (unify/actor/notebook_cells.py). On both tool surfaces.
    UNIFY_CODE_PROJECTION: str = ""

    # ─────────────────────────────────────────────────────────────────────────
    # Builtins Catalogue
    # ─────────────────────────────────────────────────────────────────────────
    # Name of the project holding the builtins catalogues (function primitives
    # and guidance), seeded from the committed snapshots at start-up.
    UNIFY_BUILTINS_PROJECT: str = "Builtins"

    # ─────────────────────────────────────────────────────────────────────────
    # Logging / Observability
    # ─────────────────────────────────────────────────────────────────────────
    PYTEST_LOG_TO_FILE: bool = True
    # Directory for Unify LOGGER file output (async tool loop, managers, etc.)
    # When set, logs are written to {UNIFY_LOG_DIR}/unify.log
    # Default: None (console only)
    UNIFY_LOG_DIR: str = ""

    # ─────────────────────────────────────────────────────────────────────────
    # Terminal Logging
    # ─────────────────────────────────────────────────────────────────────────
    UNIFY_TERMINAL_LOG: bool = True
    UNIFY_TERMINAL_LOG_LEVEL: str = "INFO"

    # ─────────────────────────────────────────────────────────────────────────
    # Composed Manager Settings
    # ─────────────────────────────────────────────────────────────────────────
    # Each manager owns its settings in its own settings.py file.
    # Access via SETTINGS.function.IMPL, SETTINGS.guidance.IMPL, etc.
    actor: ActorSettings = Field(default_factory=ActorSettings)
    conversation: ConversationSettings = Field(default_factory=ConversationSettings)
    function: FunctionSettings = Field(default_factory=FunctionSettings)
    guidance: GuidanceSettings = Field(default_factory=GuidanceSettings)

    # ─────────────────────────────────────────────────────────────────────────
    # Validators
    # ─────────────────────────────────────────────────────────────────────────
    @field_validator(
        "UNIFY_TERMINAL_LOG",
        "PYTEST_LOG_TO_FILE",
        "UNIFY_VALIDATE_LLM_PROVIDERS",
        "UNIFY_TURN_STORAGE_REVIEWS",
        "UNIFY_LOCAL_EMBEDDINGS",
        "UNIFY_TOOL_CHOICE_FALLBACK",
        "UNIFY_CELL_SCOPE_FIX",
        mode="before",
    )
    @classmethod
    def parse_bool_fields(cls, v: Any) -> bool:
        return _parse_bool(v)

    @field_validator("UNIFY_LOOP_STOP", mode="before")
    @classmethod
    def parse_loop_stop(cls, v: Any) -> str:
        value = str(v or "").strip().lower()
        value = "" if value == "off" else value
        if value not in ("", "on"):
            raise ValueError(f"UNIFY_LOOP_STOP must be empty, 'off' or 'on', not {v!r}")
        return value

    @field_validator("UNIFY_LOOP_STOP_K", mode="before")
    @classmethod
    def parse_loop_stop_k(cls, v: Any) -> int:
        if v is None or v == "":
            return 10
        value: Any = v
        if isinstance(value, str) and value.strip().lstrip("-").isdigit():
            value = int(value.strip())
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(
                f"UNIFY_LOOP_STOP_K must be a whole number of at least 1, not {v!r}",
            )
        return value

    @field_validator("UNIFY_STEP_CAP_REPLY", mode="before")
    @classmethod
    def parse_step_cap_reply(cls, v: Any) -> str:
        value = _step_cap_reply_mode(v)
        if value is None:
            raise ValueError(
                "UNIFY_STEP_CAP_REPLY must be empty, 'draft' or 'last_word' "
                f"(or a boolean, true meaning 'draft'), not {v!r}",
            )
        return value

    @field_validator("UNIFY_STEP_CAP_COMPACT", mode="before")
    @classmethod
    def parse_step_cap_compact(cls, v: Any) -> str:
        value = str(v or "").strip().lower()
        value = "" if value == "off" else value
        if value not in ("", "on"):
            raise ValueError(
                f"UNIFY_STEP_CAP_COMPACT must be empty, 'off' or 'on', not {v!r}",
            )
        return value

    @field_validator("UNIFY_MEMORY_V2", "UNIFY_MEMORY_V2_TRIGGER", mode="before")
    @classmethod
    def parse_memory_v2(cls, v: Any, info: Any) -> str:
        from unify.memory_v2.integration.switch import parse_choice

        return parse_choice(info.field_name, v)

    @field_validator(
        "UNIFY_MEMORY_V2_SOL_MODEL",
        "UNIFY_MEMORY_V2_SOL_BUDGET_USD",
        mode="before",
    )
    @classmethod
    def parse_memory_v2_sol(cls, v: Any, info: Any) -> str:
        from unify.memory_v2.integration.switch import parse_sol

        return parse_sol(info.field_name, v)

    @field_validator("UNIFY_GUIDANCE_EMPTY_QUERY", mode="before")
    @classmethod
    def parse_guidance_empty_query(cls, v: Any) -> str:
        value = str(v or "").strip().lower()
        if value not in ("", "stored"):
            raise ValueError(
                f"UNIFY_GUIDANCE_EMPTY_QUERY must be empty or 'stored', not {v!r}",
            )
        return value

    @field_validator("UNIFY_BIND_REQUEST", mode="before")
    @classmethod
    def parse_bind_request(cls, v: Any) -> str:
        value = str(v or "").strip().lower()
        value = "" if value == "off" else value
        if value not in ("", "on"):
            raise ValueError(
                f"UNIFY_BIND_REQUEST must be empty, 'off' or 'on', not {v!r}",
            )
        return value

    @field_validator("UNIFY_VARIABLE_INVENTORY", mode="before")
    @classmethod
    def parse_variable_inventory(cls, v: Any) -> str:
        value = str(v or "").strip().lower()
        value = "" if value == "off" else value
        if value not in ("", "on"):
            raise ValueError(
                f"UNIFY_VARIABLE_INVENTORY must be empty, 'off' or 'on', not {v!r}",
            )
        return value

    @field_validator("UNIFY_REPLY_CHANNEL", mode="before")
    @classmethod
    def parse_reply_channel(cls, v: Any) -> str:
        value = str(v or "").strip().lower()
        value = "" if value == "text" else value
        if value not in ("", "code+text"):
            raise ValueError(
                "UNIFY_REPLY_CHANNEL must be empty, 'text' or 'code+text', "
                f"not {v!r}",
            )
        return value

    @field_validator("UNIFY_WORKSPACE_NETWORK", mode="before")
    @classmethod
    def parse_workspace_network(cls, v: Any) -> str:
        value = str(v or "").strip().lower()
        if value not in ("", "proxy"):
            raise ValueError(
                f"UNIFY_WORKSPACE_NETWORK must be empty or 'proxy', not {v!r}",
            )
        return value

    @field_validator("UNIFY_CODE_PROJECTION", mode="before")
    @classmethod
    def parse_code_projection(cls, v: Any) -> str:
        value = str(v or "").strip().lower()
        value = "" if value == "legacy" else value
        if value not in ("", "notebook"):
            raise ValueError(
                "UNIFY_CODE_PROJECTION must be empty, 'legacy' or 'notebook', "
                f"not {v!r}",
            )
        return value

    def step_cap_reply(self) -> str:
        """The ``UNIFY_STEP_CAP_REPLY`` mode: ``""``, ``"draft"`` or ``"last_word"``."""
        return _step_cap_reply_mode(self.UNIFY_STEP_CAP_REPLY) or ""

    def validate_llm_providers(self) -> None:
        """Validate that unillm holds a key it can reach an LLM provider with.

        unillm calls Anthropic models directly and every other model through
        OpenRouter, so those two keys are the only ones that serve a call.

        Raises:
            RuntimeError: If unillm resolved neither key.
        """
        if not self.UNIFY_VALIDATE_LLM_PROVIDERS:
            return
        keys = (unillm.SETTINGS.OPENROUTER_API_KEY, unillm.SETTINGS.ANTHROPIC_API_KEY)
        if not any(key.get_secret_value() for key in keys):
            raise RuntimeError(
                "At least one LLM provider credential is required. "
                "Set OPENROUTER_API_KEY and/or ANTHROPIC_API_KEY.",
            )


# Singleton instance for production code
SETTINGS = ProductionSettings()
