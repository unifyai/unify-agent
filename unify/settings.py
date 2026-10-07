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
from unify.conversation_manager.settings import ConversationSettings
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


class ProductionSettings(BaseSettings):
    """Runtime settings; test settings (TestingSettings) inherit from this class.

    The research switches default to the configuration chosen at the code
    freeze (7 Oct 2026): the lean-all recipe, Python tool mode
    (``UNIFY_TOOL_SURFACE=core`` with worker Python in the sandboxed
    workspace) and the shared agent record. In the comments below, "as
    shipped" names upstream's behaviour, which a switch's off value still
    selects until the switch is removed. The switches still in progress are
    listed in WIP_SWITCHES.md at the repository root.
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
    # ``reply(text)`` (in process and under UNIFY_WORKSPACE_PYTHON=worker).
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
    # (in process and under UNIFY_WORKSPACE_PYTHON=worker): ``request.text``
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
    # value is printed whole. It is computed where the cell ran (in process
    # or in the UNIFY_WORKSPACE_PYTHON=worker child), is the tool's own
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
    # UNIFY_STORE_ADMISSION. Empty stores without the check.
    UNIFY_STORE_VERIFY: str = ""
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
    # With UNIFY_FUNCTION_CASES on, every recorded call of a stored function
    # also leaves one small row (store home, function_runs.sqlite): per
    # environment endpoint, calls made and items per answer; a digest per
    # argument; whether the trace is complete; and its request. Cases keep
    # only the latest few calls, so facts over many runs come from these
    # rows, summarised when read over the calls whose request was accepted
    # (unify/function_manager/run_summary.py). Missing evidence reads
    # "unknown". Nothing is shown to the agent yet. Off: no table, no row.
    UNIFY_FUNCTION_SUMMARY: bool = False
    # With UNIFY_FUNCTION_SUMMARY on: when a stored function returns empty
    # ([], {}, "", None or 0) and every earlier call of the same source whose
    # request was accepted returned something non-empty (at least one, from
    # complete traces, none of unknown shape), one plain line follows the
    # call in the cell's output saying so, with no instruction. Off: no line.
    UNIFY_FUNCTION_EMPTY_NOTICE: bool = False
    # With UNIFY_FUNCTION_SUMMARY on: while a recorded stored function runs,
    # the files it opens for reading are noted, and when it returns they are
    # scanned (bounded) for the values it matches on -- its short string
    # arguments and the string literals its code compares against -- as
    # whole values, never substrings. A value that matched in every earlier
    # call of the same source whose request was accepted, and now occurs 0
    # times, gets one plain line after the call, with the closest values of a
    # small table column. Silent when the scan was cut short, the call started
    # a process or read no files. Informational only: nothing enforces on it
    # (unify/function_manager/value_notice.py). Off: no hook, scan or line.
    UNIFY_FUNCTION_VALUE_NOTICE: bool = False
    # ``on``: the actor's prompt says that during the task it may store a
    # unit that ran and worked and repair a stored function that failed,
    # keeping its behaviour on the inputs it handled (a behaviour change gets
    # a new name); guidance likewise. Every
    # function the actor itself adds must have a name that says what it does
    # (snake_case, at least two words, one a real word that is no
    # placeholder such as ``tmp`` or ``unused``) and passes the storage check
    # of UNIFY_STORE_CHECK=resolve whether or not that is set; with
    # UNIFY_FUNCTION_CASES its replay gate applies as usual. The post-task
    # review still runs. ``only``: the same, and no review curates the
    # libraries (none after the task or a turn, no ``store_skills``). Ignored,
    # with a log line, while UNIFY_STORE_ADMISSION withholds the session's
    # writes (unify/function_manager/inline_curation.py). Empty: as shipped.
    UNIFY_INLINE_CURATION: str = ""
    # Keep every tool loop's requests a growing, byte-stable prefix, so the
    # provider's prompt cache is reused call after call: the tool list is
    # computed once per session and a tool the phase does not allow is
    # refused by rule instead of removed; messages already sent are never
    # edited; compression asks for its summary as a fork of the conversation;
    # a cache affinity key (UNIFY_CACHE_AFFINITY_SCOPE) is passed when the
    # LLM client takes one; and each call logs how much of its input came
    # from the cache (unify/common/_async_tool/cache_discipline.py). Off: as
    # shipped.
    UNIFY_CACHE_DISCIPLINE: bool = True
    # Run the storage review that follows a session as a fork of the session's
    # own conversation: its request is the actor's system prompt, messages,
    # last tools and tool choice, plus one user message with the review
    # rulebook, so it is served from the cache the actor built. Tools outside
    # the library are refused. Needs UNIFY_CACHE_DISCIPLINE (for the fixed
    # tool list); without it, or when the session was compressed or its
    # history changed since its last request, the review runs as shipped and
    # the log says why. With it, a message sent to a persistent session after
    # its task loop ended on its own (a step limit, say) is refused rather than
    # read by the review, forked or not, as an interjection to answer.
    # Off: as shipped.
    UNIFY_REVIEW_FORK: bool = True
    # Delegation for the actor `unify act` and the conversation manager
    # build: ``on`` installs the sub-actor primitive (``primitives.actor``)
    # and its 2.3k-token docs in the prompt, as shipped. ``off`` installs
    # no sub-actor: no ``primitives`` global, no delegation docs, no
    # sub-actor notch in the query_llm doctrine, and execute_function
    # refuses ``primitives.actor.*``, for runs that are one task with no
    # use for delegates. ``on_demand`` installs it, and the prompt carries a
    # three-line pointer instead of the docs, which ``help(primitives.actor.act)``
    # returns in the sandbox.
    UNIFY_DELEGATION: str = "off"
    # The shared agent record. ``record``: every agent of a run, the user (or
    # the benchmark driving the run) and the harness talk through one
    # append-only thread, read only at turn boundaries; the steering tools,
    # interjections that cancel calls, clarification and notification
    # channels, lifecycle notices and primitives.actor are not used, and
    # ``agents.spawn`` starts helpers. Empty: as shipped.
    UNIFY_AGENTS: str = "record"
    # Tunables for UNIFY_AGENTS=record, ``key=value,…`` (unify/agents/options.py).
    # Empty: the documented defaults.
    UNIFY_AGENTS_OPTIONS: str = ""
    # ``lean``: an actor prompt for a non-interactive session (one requester,
    # no reader of progress notifications), which describes the session's
    # mechanisms and states few rules. It opens with the role and the
    # requester's reply format (followed at once by the reply-protocol note
    # when UNIFY_REPLY_PROTOCOL_NOTE is on); has no notification rule and no
    # send_notification tool, no Uncertainties ending (the reply follows the
    # requester's format), no clarification norms, a one-line workspace
    # instead of the attachments table, "verify before scaling" instead of
    # the pacing rules for browser and UI work, a short query_llm doctrine,
    # and no preference for execute_function over execute_code in the
    # prompt or the two tools' descriptions; and includes every
    # UNIFY_PROMPT_ACCURACY fix. The library, discovery, storage and steering
    # sections are unchanged (their own switches govern them). Empty: as
    # shipped.
    UNIFY_PROMPT_PROFILE: str = "lean"
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
    # Off: the actor's library searches are the model's choice. As shipped the
    # default tool policy opens every task with a discovery-first gate: until
    # each present library family (FunctionManager, GuidanceManager) has been
    # searched, the model is offered only their search tools with
    # tool_choice "required", and the system prompt tells it to search both
    # before deciding how to execute. Off, no turn is gated or forced: every
    # turn offers the actor's full (statically filtered) tool list with
    # tool_choice "auto", the completion mutator that adds a missing search
    # family is not installed, the gate's refusal rule under
    # UNIFY_CACHE_DISCIPLINE never applies, and the prompt's discovery-first
    # section and its library section's search-first paragraph ("Always
    # search ...", "A no-hit is not permission ...") give way to one sentence
    # saying the library exists and can be searched with the listed tools
    # when useful (what the paragraph says about using a result is kept).
    # The library tools, their schemas and stored functions are unchanged,
    # as is UNIFY_LIBRARY_SNAPSHOT's line (without its note on skipping an
    # empty library's search, which describes the gate). A caller's own
    # tool_policy is unaffected. On: as shipped.
    UNIFY_DISCOVERY_GATE: bool = False
    # On: at the start of each act() (sub-agents' included) the harness ranks the stored
    # functions and guidance entries in scope against the request by
    # embedding similarity (no model call; primitives and lapsed functions
    # left out; no search hit counted) and lists the closest five, functions
    # and guidance together, one line each, in the task's first user message:
    # a function's name, signature and first docstring line; a guidance
    # entry's id, title and first content line. The list is written once,
    # after the UNIFY_LIBRARY_SNAPSHOT line, and asks nothing: reading,
    # calling or searching stays the model's choice, and no turn is forced. An empty or unranked library adds
    # nothing. Off: as shipped.
    UNIFY_LIBRARY_SHORTLIST: bool = True

    # ─────────────────────────────────────────────────────────────────────────
    # Session Transcripts
    # ─────────────────────────────────────────────────────────────────────────
    # Append every agent conversation (actor, sub-agents, storage review,
    # compressor) as JSON lines to ``<UNIFY_HOME>/transcripts/<session>.jsonl``
    # and one line per ended session to ``transcripts/index.jsonl``; after a
    # context compression the compressed context points at the file
    # (unify/transcripts.py). Off: nothing is written.
    UNIFY_TRANSCRIPTS: bool = True

    # ─────────────────────────────────────────────────────────────────────────
    # Workspace Sandbox
    # ─────────────────────────────────────────────────────────────────────────
    # ``sandboxed``: execute_code also takes ``language="bash"`` (a persistent
    # bash session), the actor gets ``read_file`` and ``grep``, and bash cells
    # and every subprocess a Python cell starts run inside bubblewrap: ``/``
    # read-only, only the workspace and a private /tmp writable, Unify's
    # state, credential directories and .env files hidden, credential-named
    # variables removed, no network (unify/sandbox.py). Without bubblewrap
    # those commands are refused, never run unconfined. Python cells
    # themselves still run in this process unless UNIFY_WORKSPACE_PYTHON says
    # otherwise. Empty: none of this exists.
    UNIFY_WORKSPACE: str = "sandboxed"
    # ``worker`` (with ``sandboxed``): each Python session runs its cells in a
    # persistent child process inside the same bubblewrap policy, and reaches
    # ``primitives``, steering and the other harness objects only through a
    # proxy the harness serves (unify/actor/execution/worker.py). Empty: Python
    # cells run by ``exec`` in this process.
    UNIFY_WORKSPACE_PYTHON: str = "worker"
    # ``proxy``: the sandbox's only network is one loopback port forwarded to
    # the proxy listening on 127.0.0.1:UNIFY_WORKSPACE_PROXY_PORT on the host.
    # Empty: no network at all.
    UNIFY_WORKSPACE_NETWORK: str = ""
    UNIFY_WORKSPACE_PROXY_PORT: int = 0
    # ``core``: the actor's only JSON tool is ``execute_code`` (Python, and
    # bash); its answer is a reply without tool calls, or ``final_response``
    # when the caller set a response format. Everything else is Python in
    # the sandbox, reached through the sandboxed worker's harness proxy:
    # ``functions`` (search, filter, list, get, run, add, patch, delete,
    # retire, reconcile_dependencies) and ``guidance`` (search, filter, get,
    # add, update, patch, delete, reconcile_dependencies), whose writes
    # refuse at call time, with the reason, whatever this session may not
    # write; ``install``, ``read_file`` and ``grep``; and
    # ``request_clarification`` where the session can ask. ``functions.run``
    # (replacing ``execute_function``) runs a stored function in the worker
    # and records usage, cases and trust as ``execute_function`` does; so is
    # a stored function called by name. The prompt names these objects in a
    # short index and ``help(obj)`` prints their docs as cell output, so the
    # tool list and the system prompt never change during a session.
    # ``wait``, ``steer`` and ``ask_about_completed_tool`` (and the call
    # announcements that name them) are offered only to an actor that can
    # start sub-actors. Compression is as shipped, except that
    # ``compress_context`` (and ``store_skills``) are offered only on the
    # turn the loop asks for it, not on every turn. The session tools,
    # ``send_notification`` and ``install_python_packages`` are not offered.
    # Needs UNIFY_WORKSPACE=sandboxed, UNIFY_WORKSPACE_PYTHON=worker
    # (bubblewrap installed) and UNIFY_DISCOVERY_GATE off: an actor refuses
    # to start otherwise, and never runs model code unconfined
    # (unify/actor/core_surface.py). Empty: the JSON tools as shipped.
    UNIFY_TOOL_SURFACE: str = "core"
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
    # With UNIFY_TOOL_SURFACE=core and UNIFY_REVIEW_FORK, run the storage
    # review as a fork of the session too, instead of falling back to the
    # standalone librarian: the session's last request (its execute_code-only
    # tool list included) plus one user message with the rulebook, naming the
    # libraries as the sandbox does (``functions.add``, ``guidance.add``).
    # The review's execute_code runs its cells in a new sandboxed worker that
    # holds only ``functions`` and ``guidance``, with the review's writes --
    # no task environment, files or variables, and no stored function is run.
    # Any other tool in the list is refused. Falls back to the standalone
    # review, saying why, without worker Python or with UNIFY_STORE_VERIFY
    # (its check has no sandbox method). Off: the core surface's review is
    # standalone, as shipped.
    UNIFY_REVIEW_FORK_CORE: bool = True
    # With UNIFY_TOOL_SURFACE=core and UNIFY_LIBRARY_SHORTLIST (gated or
    # not): the stored functions the shortlist lists are bound in the
    # sandbox when the task starts, exactly as a ``functions.get`` would bind
    # them (no search hit is counted), and the list's header says how to
    # call one: ``call directly: `name(...)`, or `await functions.run("name",
    # arg=...)` ``. A listed function that is ``async def`` is marked
    # ``(async)`` after its signature. Nothing is called and no turn is
    # forced. As shipped, a listed function raises NameError until something
    # reads it, and the list does not say how to call it. No effect on the
    # JSON surface. Off: as shipped.
    UNIFY_CORE_BIND_LISTED: bool = True
    # With UNIFY_TOOL_SURFACE=core: the prompt's index line for
    # ``functions`` ends with one example of calling a found function, by
    # ``functions.run`` and by name (the core counterpart of the JSON
    # prompt's ``execute_function`` sentence). No effect on the JSON
    # surface. Off: as shipped.
    UNIFY_CORE_CALL_EXAMPLE: bool = True
    # A guidance read (search, filter, get: ``guidance.*`` under the core
    # surface, ``GuidanceManager_*`` otherwise) also shows the functions an
    # entry links, as ``linked_functions``: each one's name and signature
    # (``(async)`` for an ``async def``), beside the bare ``function_ids``.
    # Under UNIFY_TOOL_SURFACE=core the read also binds those functions in
    # the sandbox, as a ``functions.get`` would, so a name it shows is
    # callable from the next cell. A linked id with no stored function is left out (its
    # ``stale_reasons`` already say so). Off: as shipped.
    UNIFY_GUIDANCE_LINKED_NAMES: bool = True
    # On: ``execute_function`` of a stored function defines the stored
    # functions it calls, transitively and each once, in the namespace the
    # call runs in (in process and under worker Python), and installs their
    # declared dependencies with its own, as ``functions.run`` does under
    # ``UNIFY_TOOL_SURFACE=core``; a helper the library no longer holds is
    # named in the error. Under worker Python a stored function called by
    # name (a helper, or one a read bound) is recorded as the in-process
    # boundary wrapper records it: usage, trust, a case
    # (unify/actor/function_helpers.py). Off: only the entry point is
    # defined, so its helpers are a NameError until a read loads them.
    UNIFY_FUNCTION_HELPERS: bool = True

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
        "UNIFY_FUNCTION_SUMMARY",
        "UNIFY_FUNCTION_EMPTY_NOTICE",
        "UNIFY_FUNCTION_VALUE_NOTICE",
        "UNIFY_CACHE_DISCIPLINE",
        "UNIFY_REVIEW_FORK",
        "UNIFY_REVIEW_FORK_CORE",
        "UNIFY_TRANSCRIPTS",
        "UNIFY_DISCOVERY_GATE",
        "UNIFY_LIBRARY_SHORTLIST",
        "UNIFY_CORE_BIND_LISTED",
        "UNIFY_CORE_CALL_EXAMPLE",
        "UNIFY_GUIDANCE_LINKED_NAMES",
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

    @field_validator("UNIFY_INLINE_CURATION", mode="before")
    @classmethod
    def parse_inline_curation(cls, v: Any) -> str:
        value = str(v or "").strip().lower()
        # A boolean spelling is the plain on/off of the switch.
        value = {"true": "on", "1": "on", "false": "", "0": ""}.get(value, value)
        if value not in ("", "on", "only"):
            raise ValueError(
                f"UNIFY_INLINE_CURATION must be empty, 'on' or 'only', not {v!r}",
            )
        return value

    @field_validator("UNIFY_PROMPT_PROFILE", mode="before")
    @classmethod
    def parse_prompt_profile(cls, v: Any) -> str:
        value = str(v or "").strip().lower()
        if value not in ("", "lean"):
            raise ValueError(
                f"UNIFY_PROMPT_PROFILE must be empty or 'lean', not {v!r}",
            )
        return value

    @field_validator("UNIFY_AGENTS", mode="before")
    @classmethod
    def parse_agents(cls, v: Any) -> str:
        value = str(v or "").strip().lower()
        if value not in ("", "record"):
            raise ValueError(f"UNIFY_AGENTS must be empty or 'record', not {v!r}")
        return value

    @field_validator("UNIFY_AGENTS_OPTIONS", mode="before")
    @classmethod
    def parse_agents_options(cls, v: Any) -> str:
        from unify.agents.options import parse_options

        text = str(v or "").strip()
        parse_options(text)
        return text

    @field_validator("UNIFY_DELEGATION", mode="before")
    @classmethod
    def parse_delegation(cls, v: Any) -> str:
        value = str("on" if v is None else v).strip().lower() or "on"
        # A boolean spelling is the plain on/off of the switch.
        value = {
            "true": "on",
            "1": "on",
            "yes": "on",
            "false": "off",
            "0": "off",
            "no": "off",
        }.get(value, value)
        if value not in ("on", "off", "on_demand"):
            raise ValueError(
                f"UNIFY_DELEGATION must be 'on', 'off' or 'on_demand', not {v!r}",
            )
        return value

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

    @field_validator("UNIFY_WORKSPACE", mode="before")
    @classmethod
    def parse_workspace(cls, v: Any) -> str:
        value = str(v or "").strip().lower()
        if value not in ("", "sandboxed"):
            raise ValueError(
                f"UNIFY_WORKSPACE must be empty or 'sandboxed', not {v!r}",
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

    @field_validator("UNIFY_WORKSPACE_PYTHON", mode="before")
    @classmethod
    def parse_workspace_python(cls, v: Any) -> str:
        value = str(v or "").strip().lower()
        if value not in ("", "worker"):
            raise ValueError(
                f"UNIFY_WORKSPACE_PYTHON must be empty or 'worker', not {v!r}",
            )
        return value

    @field_validator("UNIFY_TOOL_SURFACE", mode="before")
    @classmethod
    def parse_tool_surface(cls, v: Any) -> str:
        value = str(v or "").strip().lower()
        if value not in ("", "core"):
            raise ValueError(
                f"UNIFY_TOOL_SURFACE must be empty or 'core', not {v!r}",
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

    def lean_prompt(self) -> bool:
        """``UNIFY_PROMPT_PROFILE=lean``."""
        return self.UNIFY_PROMPT_PROFILE == "lean"

    model_config = SettingsConfigDict(
        env_file=".env",
        case_sensitive=True,
        extra="ignore",
    )

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
