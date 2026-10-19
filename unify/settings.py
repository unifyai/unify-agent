"""
unify/settings.py
==================

Centralized runtime settings using pydantic-settings.

All settings can be overridden via environment variables or the ``.env`` file
in the working directory.
"""

import math
import os
import re
from typing import Any, Optional

import unillm
from pydantic import Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from unify.actor.settings import ActorSettings
from unify.function_manager.settings import FunctionSettings
from unify.guidance_manager.settings import GuidanceSettings

# The reasoning efforts unillm forwards to providers (it maps them per
# provider, e.g. DeepSeek's high/max), with "" for "as shipped". "provider" sends
# no effort at all on any call of the run (the provider's own default), as a
# published row that set none (unify/common/llm_client.py PROVIDER_EFFORT).
_REASONING_EFFORTS = ("", "none", "minimal", "low", "medium", "high", "xhigh", "max", "provider")


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
    # set. Empty leaves per-call-site effort levels untouched. ``provider``: no
    # call of the run sends a reasoning effort (each provider's default applies).
    UNIFY_REASONING_EFFORT: str = "high"
    # The LLM endpoints model-written code may name (the ``query_llm``,
    # ``list_llms`` and ``unillm`` globals of a cell, unify/common/cell_models.py).
    # Empty: only the session's configured model (the act profile's model, or
    # what ``model=None`` resolves to). ``any``: every endpoint, as before.
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
    # ``on``: every model call of a client made by ``new_llm_client`` (or
    # forked from one) is sent as a stream and read to its end, and unillm
    # reassembles the reply into the completion the same call returns
    # unstreamed (its ``stream_transport``), so transcripts, caching, cost
    # and postprocessing are unchanged. Bytes then arrive while the model is
    # still generating, so a proxy in front that cancels a call after a spell
    # without traffic does not cancel a long one. Nothing is sent but the
    # model's own reply. Empty (or ``off``): calls are not streamed, as
    # shipped (unify/common/llm_client.py).
    UNIFY_STREAM_MODEL_CALLS: str = ""
    # How long, in seconds, a streamed model call may go without receiving a
    # single byte before unillm aborts it and retries it within the same turn
    # (unillm's ``UNILLM_STREAM_IDLE_S``, which every client this process
    # makes passes on). Every byte counts, keep-alive comments included, so
    # only a call on which nothing at all arrives is cut. Empty leaves
    # unillm's own value (120 s unless set); ``0`` or ``off`` turns it off.
    # Only streamed calls (UNIFY_STREAM_MODEL_CALLS=on) have a byte to wait for.
    UNIFY_STREAM_IDLE_S: str = ""

    # Ceiling on tool-calling iterations for one agent run. Unset, a run ends
    # only when the model decides to stop, so a loop that never converges
    # bills indefinitely. Sized to bound that rather than to shape normal
    # runs: a long agentic task uses far fewer steps than this, and a caller
    # that genuinely needs more can pass ``max_steps`` explicitly. Set to 0 to
    # restore unbounded iteration.
    UNIFY_MAX_TOOL_LOOP_STEPS: int = 300
    # ``draft`` (the default; also ``true``/``1``/``yes``/``on``, as when
    # this was a bool): in a persistent session (``unify act --persist``) the
    # limit ends only the current request: the reply says the session stopped
    # at its step limit and quotes the latest reply text it drafted for the
    # request, the pending tool calls are cancelled and answered as such, and
    # the next message starts a request with its own ``max_steps`` (the limit
    # counts the messages of one request). A loop that is not persistent
    # still ends at the limit, its stop notice followed by that draft.
    # ``last_word``: the same, except that before the reply the model is
    # given one model call with no tools offered, after a notice that the
    # request's step limit is reached and it should reply now with its best
    # answer; the stop reply carries that answer in place of the draft, and
    # the draft when the call fails or returns no text. Empty (also
    # ``false``/``0``/``no``/``off``): reaching ``max_steps`` ends the loop
    # with a stop notice, and in a persistent session the limit counts every
    # message of the session, so a long session ends at it and takes no
    # further messages.
    UNIFY_STEP_CAP_REPLY: str = "draft"
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
    # unchanged. ``continue``: one long-horizon mode for the actor's task
    # loop (the one that answers the requester; any other loop runs as
    # shipped). The step budget (``max_steps``, a count of messages) is
    # counted per request, from the request's own message, and afresh after
    # each compaction. At the limit the conversation is compacted as under
    # ``on``, with no bound on compactions per request, and the request goes
    # on; the compaction's summary is a loop-authored message, so it starts
    # no request (UNIFY_LOOP_STOP's count carries across it). A compaction
    # whose rebuilt context is not smaller than the one it replaced
    # (serialised message characters) is ineffective, and a second
    # ineffective one in a row in a request ends it. That end, a failed
    # compaction, a loop stop and the loop's timeout all end the request
    # through UNIFY_STEP_CAP_REPLY's reply path, in its ``draft`` mode when
    # that switch is empty: the requester always gets an answer and a
    # persistent session waits for its next request. Empty (also ``off``):
    # as shipped.
    UNIFY_STEP_CAP_COMPACT: str = ""
    # ``on``: a context compaction (at the context threshold, on the model's
    # own ``compress_context`` call, or at ``max_steps`` under
    # UNIFY_STEP_CAP_COMPACT) rebuilds the conversation so that it starts
    # with what the session already sent, byte for byte, instead of with the
    # system prompt alone: the system prompt, the session's first user
    # message, then every requester message of the current request (a user
    # message the loop did not author), each unchanged and in its original
    # order, and only then the summary, as one loop-authored user message
    # (the compressed-context header, the summary, "Context was compressed.
    # Continue from where you left off." and the transcript's path). The
    # current request is read from the session's own messages: it starts at
    # the latest requester message, together with the requester messages
    # just before it that no model turn separates from it, as
    # UNIFY_LOOP_STOP's tracker counts a request; earlier requests of a
    # persistent session are in the summary. The summary is asked for as
    # shipped (a fork of the last request). The tools and their order stay
    # the same, the fallback compactor's rebuild included: it then neither
    # rewrites the system prompt nor adds ``unpack_messages``, and its
    # compressed entries are the summary. "Byte for byte" covers the system
    # prompt, the first message and the kept requester messages; the loop's
    # own runtime-context system messages are rebuilt at the restart (the
    # same bytes for the actor, not for a nested loop given a
    # ``parent_chat_context``, which a restart drops). When the first call
    # after such a compaction is still over the threshold, what was kept is
    # too large on its own, and the next compaction rebuilds as shipped, so a
    # request is never compacted around the same prefix twice in a row.
    # Empty (also ``off``): as shipped.
    UNIFY_COMPACTION_KEEP_PREFIX: str = ""
    # The fixed prompt size, in tokens, at which every agent (the agent loop and the memory agents) compacts its
    # context, unless 85% of the model's context window comes first; with no known window it applies alone. A
    # context-management point, not a limit on what an agent reads or writes. A positive whole number.
    UNIFY_COMPACTION_AT_TOKENS: int = 150_000
    # Where the actor's session states the host clock. Empty (also
    # ``system``): as shipped, a "Current Time" section near the end of the
    # system prompt, which tells the model to resolve "today" against it and
    # to prefer it over a clock read in ``execute_code``. ``first_message``:
    # the system prompt has no clock section, and the session's first user
    # message opens with one line, "The host clock reads <time>. Dates stated
    # in the request or in the files and records you work with take
    # precedence.", the time sampled once when the session starts (the
    # moment the system prompt would have sampled it), before the library's
    # size and the rest of the first message's context. The line is in the
    # first user message only: a persistent session's later requests do not
    # repeat it, and a session restarted after context compression opens its
    # new first message with the same line. The system prompt, and so the
    # cache affinity key derived from it, is then the same for every session
    # of one configuration whenever it starts.
    UNIFY_CLOCK_PLACEMENT: str = ""
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
    # ``on``: every agent keeps one model-API tool, ``execute_code``; every
    # other tool is a Python call ``tools.<name>(...)`` in a cell, defined
    # once in the shared registry (unify/common/tool_registry.py) and run on
    # the host. For the actor and its teammates the loop also compacts by
    # itself at the compaction point (no forced ``compress_context`` turn)
    # and ``tools.answer(answer)`` replaces the response tool. ``off`` (the
    # default): every agent's requests are as before.
    UNIFY_TOOLS_IN_CODE: str = "off"
    # ``persistent`` (the default): each agent's cells run in one
    # interpreter that keeps its variables between cells until it restarts.
    # ``fresh``: every cell starts in a new interpreter.
    UNIFY_CELL_STATE: str = "persistent"
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
    # The actor's library: empty or ``on`` (default: today's behaviour) or ``off``, which removes memory and library
    # surface only: no ``functions`` or ``guidance`` in the sandbox, no ``store_skills``, no storage review, no
    # shortlist or library line, and no prompt text naming stored functions. Refused with UNIFY_MEMORY on
    # (unify/actor/library_switch.py).
    UNIFY_LIBRARY: str = ""
    # Text appended at the very end of the system prompt (empty: none), with UNIFY_SYSTEM_SUFFIX_SHA256 the lowercase
    # hex sha256 of its UTF-8 bytes: set together, and a run whose text does not match is refused before anything
    # runs (unify/actor/system_suffix.py). The sha256 is on the CLI's ``ended`` line.
    UNIFY_SYSTEM_SUFFIX: str = ""
    UNIFY_SYSTEM_SUFFIX_SHA256: str = ""
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
    # Serving on the situation plus the failure record: ``on`` gives the librarian each item's failure record and
    # contradictions beside the situation, shows match and applicability results as evidence (never as a claim of
    # fit), states each status by what it rests on, and withdraws retired and superseded library items as well as
    # captures; ``off`` (default; empty without the redesign profile): as built.
    UNIFY_MEMORY_SERVING_FAILURE_RECORD: str = ""
    # Each episode's feedback regime: ``inferred`` infers it from the signals that arrive (in the episode or after it),
    # with the run's own history as the prior, and takes late verdicts on earlier episodes; ``off`` (default; empty
    # without the redesign profile): every episode is ``dense``, as built.
    UNIFY_MEMORY_REGIME: str = ""
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
    # Variables of the harness's environment a sandboxed command (the Python
    # worker that runs every cell, a bash cell) gets besides the built-in
    # allow-list (unify/sandbox.py, CELL_ENV_NAMES and CELL_ENV_PATTERNS): a
    # comma list of names and ``PREFIX_*`` patterns, for a runner's own cell
    # variables. A credential's name or a key-shaped value is never passed.
    UNIFY_CELL_ENV_ALLOW: str = ""
    # The address-space limit (RLIMIT_AS, bytes) of every process in the
    # workspace sandbox, set inside it with prlimit beside its other limits
    # (unify/sandbox.py). A positive whole number; 16 GiB by default, which
    # leaves normal work unconstrained.
    UNIFY_SANDBOX_ADDRESS_SPACE_BYTES: int = 16 * 1024**3
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
    # ``on``: every model call carries four HTTP headers, so a proxy in front
    # of the provider can attribute it (for example, measure the cache hits of
    # each session's first call and of the calls after a compaction):
    # ``X-Unify-Session``, a random id per model client (a fork gets its own,
    # plus ``X-Unify-Parent``, its parent's); ``X-Unify-Request``, how many
    # requester messages the loop answering a requester had received when
    # the call was made (1 for the first request, 0 where no such loop drives
    # the client; a fork starts from its parent's count); ``X-Unify-Call-Kind``,
    # the client's ``origin`` label; and ``X-Unify-Msg-Count``, the number of
    # messages in the request. Every value is random or counted by the
    # harness, matches ``[A-Za-z0-9_.:-]{1,64}`` and is never request
    # content. The request body is unchanged, so the provider's prompt cache
    # is not affected; unillm's response cache (UNILLM_CACHE) keys on the
    # headers, so a recorded response is not replayed for a call that
    # carries them. With no proxy in front that strips them, the headers
    # reach the model provider (harmless, but they do). Empty (or ``off``):
    # no header is added and the call's arguments are as shipped
    # (unify/common/llm_client.py).
    UNIFY_REQUEST_METADATA_HEADERS: str = ""

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
        if value not in ("", "on", "continue"):
            raise ValueError(
                "UNIFY_STEP_CAP_COMPACT must be empty, 'off', 'on' or 'continue', "
                f"not {v!r}",
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
        mode="before",
    )
    @classmethod
    def parse_memory(cls, v: Any, info: Any) -> Any:
        from unify.memory.integration import switch

        return switch.PARSERS[info.field_name](v)

    @field_validator(
        "UNIFY_AGENTS_RUN_RECORD",
        "UNIFY_MEMORY_WRITER_SERVICE",
        "UNIFY_MEMORY_WRITER_SERVICE_CAP_USD",
        "UNIFY_MEMORY_METERING",
        "UNIFY_HOST_LEASE_END",
        mode="before",
    )
    @classmethod
    def parse_redesign_a(cls, v: Any, info: Any) -> Any:
        from unify.memory.integration import switch

        return switch.PARSERS[info.field_name](v)

    @field_validator("UNIFY_LIBRARY", mode="before")
    @classmethod
    def parse_library(cls, v: Any) -> str:
        from unify.actor import library_switch

        return library_switch.parse(v)

    @field_validator("UNIFY_SYSTEM_SUFFIX_SHA256", mode="before")
    @classmethod
    def parse_system_suffix_sha256(cls, v: Any) -> str:
        from unify.actor import system_suffix

        return system_suffix.parse_sha256(v)

    @field_validator("UNIFY_COMPACTION_AT_TOKENS", mode="before")
    @classmethod
    def parse_compaction_at_tokens(cls, v: Any) -> int:
        text = str(v).strip() if v is not None else ""
        if isinstance(v, bool) or not text.isdigit() or int(text) <= 0:
            raise ValueError(
                f"UNIFY_COMPACTION_AT_TOKENS must be a positive whole number of tokens, not {v!r}",
            )
        return int(text)

    @field_validator("UNIFY_COMPACTION_KEEP_PREFIX", mode="before")
    @classmethod
    def parse_compaction_keep_prefix(cls, v: Any) -> str:
        value = str(v or "").strip().lower()
        value = "" if value == "off" else value
        if value not in ("", "on"):
            raise ValueError(
                "UNIFY_COMPACTION_KEEP_PREFIX must be empty, 'off' or 'on', "
                f"not {v!r}",
            )
        return value

    @field_validator("UNIFY_MEMORY_DIALOGUE", mode="before")
    @classmethod
    def parse_memory_dialogue(cls, v: Any) -> str:
        from unify.memory.integration import switch

        return switch.parse_dialogue(v)

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

    @field_validator("UNIFY_REQUEST_METADATA_HEADERS", mode="before")
    @classmethod
    def parse_request_metadata_headers(cls, v: Any) -> str:
        value = str(v or "").strip().lower()
        value = "" if value == "off" else value
        if value not in ("", "on"):
            raise ValueError(
                "UNIFY_REQUEST_METADATA_HEADERS must be empty, 'off' or 'on', "
                f"not {v!r}",
            )
        return value

    @field_validator("UNIFY_STREAM_MODEL_CALLS", mode="before")
    @classmethod
    def parse_stream_model_calls(cls, v: Any) -> str:
        value = str(v or "").strip().lower()
        value = "" if value == "off" else value
        if value not in ("", "on"):
            raise ValueError(
                f"UNIFY_STREAM_MODEL_CALLS must be empty, 'off' or 'on', not {v!r}",
            )
        return value

    @field_validator("UNIFY_STREAM_IDLE_S", mode="before")
    @classmethod
    def parse_stream_idle_s(cls, v: Any) -> str:
        text = str(v if v is not None else "").strip().lower()
        if text in ("", "off"):
            return "0" if text == "off" else ""
        try:
            seconds = float(text)
        except ValueError:
            seconds = float("nan")
        if isinstance(v, bool) or not math.isfinite(seconds) or seconds < 0:
            raise ValueError(
                "UNIFY_STREAM_IDLE_S must be empty, 'off' or a number of seconds, "
                f"not {v!r}",
            )
        return format(seconds, "g")

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

    @field_validator("UNIFY_TOOLS_IN_CODE", mode="before")
    @classmethod
    def parse_tools_in_code(cls, v: Any) -> str:
        value = str(v or "off").strip().lower()
        if value not in ("on", "off"):
            raise ValueError(
                f"UNIFY_TOOLS_IN_CODE must be 'on' or 'off', not {v!r}",
            )
        return value

    @field_validator("UNIFY_MEMORY_LIBRARIAN", mode="before")
    @classmethod
    def parse_memory_librarian(cls, v: Any) -> str:
        value = str(v or "off").strip().lower()
        if value not in ("on", "off"):
            raise ValueError(
                f"UNIFY_MEMORY_LIBRARIAN must be 'on' or 'off', not {v!r}",
            )
        return value

    @model_validator(mode="after")
    def librarian_needs_tools_in_code(self) -> "ProductionSettings":
        """The librarian is reached from a cell through the box-to-host tool channel, which exists only with the
        tools called from code: ``UNIFY_MEMORY_LIBRARIAN=on`` with ``UNIFY_TOOLS_IN_CODE=off`` is refused."""
        if getattr(self, "UNIFY_MEMORY_LIBRARIAN", "off") == "on" and getattr(self, "UNIFY_TOOLS_IN_CODE", "off") != "on":
            raise ValueError(
                "UNIFY_MEMORY_LIBRARIAN=on requires UNIFY_TOOLS_IN_CODE=on: the librarian is asked from a cell "
                "through the tool channel, which exists only with the tools called from code",
            )
        return self

    @field_validator("UNIFY_CELL_STATE", mode="before")
    @classmethod
    def parse_cell_state(cls, v: Any) -> str:
        value = str(v or "persistent").strip().lower()
        if value not in ("persistent", "fresh"):
            raise ValueError(
                f"UNIFY_CELL_STATE must be 'persistent' or 'fresh', not {v!r}",
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

    @field_validator("UNIFY_SANDBOX_ADDRESS_SPACE_BYTES", mode="before")
    @classmethod
    def parse_sandbox_address_space(cls, v: Any) -> int:
        text = str(v).strip() if v is not None else ""
        if isinstance(v, bool) or not re.fullmatch(r"[0-9]+", text) or int(text) <= 0:
            raise ValueError(
                "UNIFY_SANDBOX_ADDRESS_SPACE_BYTES must be a positive whole number "
                f"of bytes, not {v!r}",
            )
        return int(text)

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

    @field_validator("UNIFY_CLOCK_PLACEMENT", mode="before")
    @classmethod
    def parse_clock_placement(cls, v: Any) -> str:
        value = str(v or "").strip().lower()
        value = "" if value == "system" else value
        if value not in ("", "first_message"):
            raise ValueError(
                "UNIFY_CLOCK_PLACEMENT must be empty, 'system' or "
                f"'first_message', not {v!r}",
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
