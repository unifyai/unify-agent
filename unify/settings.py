"""
unify/settings.py
==================

Centralized runtime settings using pydantic-settings.

All settings can be overridden via environment variables or the ``.env`` file
in the working directory.
"""

from typing import Any

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


class ProductionSettings(BaseSettings):
    """Runtime settings; test settings (TestingSettings) inherit from this class."""

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
    # Leave a stored function that cannot be loaded out of a search, list or
    # filter that loads its results, naming it in a warning, instead of
    # failing the whole result.
    UNIFY_SEARCH_SKIP_UNLOADABLE: bool = False

    # ─────────────────────────────────────────────────────────────────────────
    # Environment Namespaces and the Storage Check
    # ─────────────────────────────────────────────────────────────────────────
    # ``package.module:factory`` entries, comma-separated: factories that
    # register the environment's own callable surface as ``primitives.<name>``
    # namespaces at start-up (unify/function_manager/primitives/environment.py).
    # Empty registers nothing.
    UNIFY_ENV_NAMESPACES: str = ""
    # ``resolve``: before a function is stored, every name and every
    # ``primitives.*`` reference in it must resolve against the sandbox's
    # globals and the registered namespaces, and it must load as a search
    # would load it; otherwise ``add_functions`` refuses it and says why.
    # Empty stores without the check.
    UNIFY_STORE_CHECK: str = ""
    # On: a function or guidance entry is checked, before it is stored, for
    # identifiers of the session's own task instance: id-like tokens (hex
    # runs, UUIDs, ``word-<hex>`` aliases, long digit runs) and quoted
    # titles taken from the session's first request, and task-alias or UUID
    # shapes in any function name. ``add_functions`` refuses a function whose
    # name or code (literals and defaults) carries one; a docstring or a
    # guidance entry that names one is stored with a warning. Off stores
    # without the check.
    UNIFY_STORE_INSTANCE_LINT: bool = False
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
    # Off: the built-in guidance catalogue (the Agent Skills snapshot in
    # guidance_manager/builtins_guidance.json) is neither seeded nor read, so
    # every guidance search, filter, lookup and count sees only the entries
    # the assistant stored itself; rows an earlier process seeded stay in the
    # store, untouched. On reads them alongside the stored entries as shipped.
    UNIFY_BUILTIN_GUIDANCE: bool = True
    # Offer FunctionManager_patch_function and GuidanceManager_patch_guidance,
    # which replace excerpts of a stored entry in place -- one edit or an
    # ordered, all-or-nothing batch, each matched exactly or, failing that,
    # with whitespace differences tolerated while it stays unique (the patched
    # function is stored through add_functions, so its checks still apply),
    # and keep the previous version of every overwritten function or guidance
    # entry in function_history / guidance_history. Off: no patch tools and
    # nothing is written to history.
    UNIFY_FUNCTION_PATCH: bool = False
    # Record each stored function's calls as cases in function_cases (its
    # arguments, what it returned or raised, and the environment calls it
    # made with their answers; the latest 3 that returned and 3 that raised,
    # one per input), and before an overwrite or a patch stores a different
    # source, replay the cases that returned against it with the recorded
    # answers served in place of the environment (nothing reaches the
    # environment, the network or a model). A change that makes another
    # environment call, or returns or raises something else, is refused,
    # naming the case: store the new behaviour under a new name, or retire
    # the case with FunctionManager_retire_case, offered while this is on.
    # A case that cannot be replayed faithfully (the clock, randomness, an
    # unstubbed import) does not block and is reported. Search and filter
    # results show up to two cases per function
    # (unify/function_manager/store_cases.py). Off: nothing is recorded or
    # replayed and every tool is as shipped.
    UNIFY_FUNCTION_CASES: bool = False
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
    # ``warn``: adding a new function whose normalised code nearly matches a
    # stored one (token Jaccard >= 0.9) stores it and returns a warning naming
    # the stored function. Empty adds as shipped.
    UNIFY_STORE_DEDUPE: str = ""
    # ``ramp``: keep a trust record per stored function (probation, trusted,
    # quarantined) in function_trust. Every reuse is evidence: a call that
    # returns is a pass, one that raises quarantines the function, which is
    # then left out of the searches that load functions and listed to the
    # next storage review as needing repair. A function is trusted after 3
    # passes over 2 distinct inputs (5 over 3 if it can change anything); a
    # changed source or callee, or an overwrite, puts it back on probation
    # with its passes cleared and its failure history kept. With
    # UNIFY_STORE_VERIFY set, a reuse is also re-checked in a fresh world
    # with probability 1/2^k after k clean uses
    # (unify/function_manager/store_trust.py). Empty keeps no record.
    UNIFY_STORE_TRUST: str = ""
    # Keep every tool loop's requests a growing, byte-stable prefix, so the
    # provider's prompt cache is reused call after call: the tool list is
    # computed once per session and a tool the phase does not allow is
    # refused by rule instead of removed; messages already sent are never
    # edited; compression asks for its summary as a fork of the conversation;
    # a cache affinity key (UNIFY_CACHE_AFFINITY_SCOPE) is passed when the
    # LLM client takes one; and each call logs how much of its input came
    # from the cache (unify/common/_async_tool/cache_discipline.py). Off: as
    # shipped.
    UNIFY_CACHE_DISCIPLINE: bool = False
    # What the cache affinity key of UNIFY_CACHE_DISCIPLINE is shared by:
    # ``prefix`` (the default), every session whose model, system prompt and
    # tool list are the same, so a new session reaches the replica an earlier
    # one cached that prefix on; ``session``, one key per session; ``run``,
    # one key for every session of this process; ``static``, like ``prefix``
    # but over the actor's system prompt without its per-session sections
    # (the clock and the filesystem context), so sessions of one
    # configuration share a key across minutes and workspaces. Ignored with
    # the switch off.
    UNIFY_CACHE_AFFINITY_SCOPE: str = "prefix"
    # Tell the actor how large its libraries are and skip searching an empty
    # one: the discovery-first gate counts the stored functions (primitives
    # excluded) and the guidance entries in scope each time it is evaluated,
    # and a library with none is treated as already searched, so with both
    # empty the first turn is ``auto`` instead of a forced search. Only the
    # tool choice changes: under UNIFY_CACHE_DISCIPLINE the tool list sent is
    # the same. The session's first user message starts with one line giving
    # both counts at task start. Off: as shipped.
    UNIFY_LIBRARY_SNAPSHOT: bool = False
    # Where the actor's per-session prompt sections go: the clock (minute
    # resolution) and the filesystem context (workspace paths). Empty: at the
    # tail of the system prompt, as shipped, so the system prompt differs
    # from minute to minute and workspace to workspace. ``message``: the same
    # sections, sampled once per session, open the session's first user
    # message (and the message that restarts a compressed session), so the
    # system prompt is the same for every session of one configuration.
    UNIFY_PROMPT_CLOCK: str = ""
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
    UNIFY_REVIEW_FORK: bool = False
    # ``unified``: the storage review is framed as the agent's own curation
    # step after the task rather than a separate "skill librarian" (whose
    # text opens with "Often nothing is"): the actor's prompt says it will
    # curate the libraries from its trajectory when the task ends, a forked
    # review is told the task is finished and this is that step, and it ends
    # with the closing instruction the standalone review already gets. The
    # rulebook is unchanged. Empty frames the review as shipped.
    UNIFY_REVIEW_FRAMING: str = ""
    # ``compose``: the storage rulebook asks for small, parametrised units
    # composed into larger ones, stored only from code the trajectory ran,
    # with names and signatures that describe behaviour; a patch must keep an
    # entry's behaviour on the inputs it already handled (a behaviour change
    # gets a new name and the old entry is retired), a guidance entry stays
    # short and is split rather than grown, and the update-first order no
    # longer sends a fix into a broader existing entry. ``minimal``: the
    # same rules, with the rest of the rulebook cut to what storage needs
    # (what can be stored and how it runs, dependencies, what guidance is
    # for): no user-notification, recurring-deliverable, specialist
    # sub-agent, model-choice-trial, logging-marker or distillation-dial
    # sections. Empty: as shipped.
    UNIFY_CURATION_DOCTRINE: str = ""
    # The actor's prompt states that actions the requester asks to be taken by
    # replying in a stated format are only ever the actor's own final reply:
    # they are not functions or primitives, and no code or sub-agent can take
    # them, so a sub-agent whose task seems to need one reports that instead
    # of calling a function that does not exist. Off: as shipped.
    UNIFY_REPLY_PROTOCOL_NOTE: bool = False
    # The actor states only what the session actually has: the skill-storage
    # notice of a persistent session describes the review it gets (once,
    # when the session ends, unless UNIFY_TURN_STORAGE_REVIEWS), not one per
    # turn with results as background notes; the steering docs of
    # execute_code and execute_function name the `steer` tool, not the
    # stop_* tools it replaced; and a loop that no other loop started and
    # that was given no parent context is not told it runs inside a parent
    # conversation; the execution rules mention request_clarification only
    # when the session has it; and a sub-actor gets request_clarification
    # only when the actor that started it could ask. Off: as shipped.
    UNIFY_PROMPT_ACCURACY: bool = False
    # Delegation for the actor `unify act` and the conversation manager
    # build: ``on`` installs the sub-actor primitive (``primitives.actor``)
    # and its 2.3k-token docs in the prompt, as shipped. ``off`` installs
    # no sub-actor: no ``primitives`` global, no delegation docs, no
    # sub-actor notch in the query_llm doctrine, and execute_function
    # refuses ``primitives.actor.*``, for runs that are one task with no
    # use for delegates. ``on_demand`` installs it, and the prompt carries a
    # three-line pointer instead of the docs, which ``help(primitives.actor.act)``
    # returns in the sandbox.
    UNIFY_DELEGATION: str = "on"
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
    UNIFY_PROMPT_PROFILE: str = ""
    # Before the storage review that follows a session, one tool-free call
    # (the review's model, at low effort unless UNIFY_REVIEW_REASONING_EFFORT
    # sets the review's) reads the end of the trajectory, the checked outcome
    # (UNIFY_OUTCOME) and the final reply, and answers whether the session
    # left reusable working code, a lesson found by trial and error, or a
    # stored entry needing repair; the review runs only on a yes
    # (unify/actor/review_gate.py). A failed call or an unreadable reply runs
    # the review. While the library holds nothing (0 functions, 0 guidance
    # entries) the gate is not asked and the review runs. Turn reviews and
    # store_skills are not gated. Off: as shipped.
    UNIFY_REVIEW_GATE: bool = False
    # The actor's prompt asks it to compute a result that can be computed
    # with a program and answer with the program's output instead of a
    # result worked out in text, with judgment steps kept as query_llm calls
    # inside the program, so that a solved task leaves a program the review
    # can store. It does not ask the actor to check the program against
    # examples it was given: stored functions are checked mechanically
    # (UNIFY_FUNCTION_CASES replays their recorded calls). Conversational and single-step requests, and tasks one
    # stored function or primitive call completes, are answered as before.
    # Off: as shipped.
    UNIFY_CODE_FIRST: bool = False
    # The loop's `wait` tool takes ``until="all"``: a turn that calls several
    # tools and adds wait(until="all") is woken once, when every call from
    # that turn has finished, instead of when the first one does (a turn
    # started then is cancelled, and still billed, when a sibling lands).
    # The model decides; the declaration also skips the eager turn a
    # discovery gate would grant. A new message, a clarification request, a
    # progress notification or a stop still wakes it at once. The tool's
    # description and one line of the actor prompt say so. Off: as shipped.
    UNIFY_WAIT_FOR_BATCH: bool = False
    # The longest a wait(until="all") holds the next turn back (seconds):
    # its own max_seconds is clamped to this, so a slow call never keeps the
    # model from results that have landed for longer. Between 1 and 120.
    UNIFY_WAIT_CEILING_SECONDS: float = 15.0
    # No model turn starts while a tool call is still running: the model is
    # woken once, when every running call has finished, as if each turn added
    # wait(until="all") over everything in flight. A landed result is held
    # back at most UNIFY_WAIT_CEILING_SECONDS, counted from the first one held,
    # and the model is then woken with the results so far. Only tool results
    # are held: a message from the user, the environment or another agent (an
    # interjection, a clarification request or a progress notification) still
    # wakes the model at once, as does a stop. A tool result or a notification
    # that lands while a model turn is in flight never cancels it (the
    # provider bills it anyway): the turn finishes and the model gets the
    # result or the message next. An interjection, a clarification and a stop
    # still cancel a turn in flight, as shipped. No eager turn is granted
    # while calls run. Requests, tools and the prompt are unchanged; only when
    # the model is called changes. Read once per loop. Off: as shipped.
    UNIFY_BATCH_WAKE: bool = False
    # While a tool call is still running, every model turn is sent with
    # tool_choice "required", so the model has to call some tool (often a
    # bare `wait`, which the loop prunes) instead of replying. It dates from
    # a final-answer tool the actor no longer has: its answer is a reply
    # without tool calls. Off: such a turn keeps the tool_choice its policy
    # gave it ("auto" unless a gate requires a call). Read once per loop. On:
    # as shipped.
    UNIFY_PENDING_REQUIRED: bool = True
    # Each tool call the loop schedules appends a user-role
    # "[steerable <call_id>] <tool> started." message, a call that becomes a
    # handle appends "[steerable <call_id>] now supports ...", and a finished
    # call that can be asked about appends "[askable <call_id>] ...". The
    # first of them also appends the "User Visibility Context" system
    # message. Off: none of these is appended; the call ids stay in the
    # model's own tool calls, and progress, clarification and interjection
    # messages (with the visibility message they bring) are unchanged. Read
    # once per loop, so a session never changes mid-way, and nothing already
    # in a transcript is removed. On: as shipped.
    UNIFY_LIFECYCLE_NOTICES: bool = True
    # Off: while the actor's discovery gate is open, no model turn starts
    # before the library searches a turn scheduled have returned. As shipped
    # the gate grants one at once, while they run, and the model is woken as
    # soon as the first of two searches lands; either turn is cancelled, and
    # still billed, when a search lands during it. Off, the searches the gate
    # forces are one unit: the model is woken once all a turn made have
    # returned, or at UNIFY_WAIT_CEILING_SECONDS, or at once on a stop, a new
    # message, a clarification or a notification; other calls are not held.
    # The mutator that adds the missing family to a turn that searched one
    # also recognises the actor's gate request, which lists wait, steer and
    # ask_about_completed_tool. The gate still requires the searches
    # (tool_choice "required", only the gated tools, parallel calls asked
    # for), and a turn that leaves a family out is followed, once its calls
    # return, by one that requires it. On: as shipped.
    UNIFY_DISCOVERY_SPECULATIVE_TURN: bool = True
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
    UNIFY_DISCOVERY_GATE: bool = True
    # On: at the start of each act() (sub-agents' included) the harness ranks the stored
    # functions and guidance entries in scope against the request by
    # embedding similarity (no model call; primitives and lapsed functions
    # left out; no search hit counted) and lists the closest five, functions
    # and guidance together, one line each, in the task's first user message:
    # a function's name, signature, first docstring line and, under
    # UNIFY_TRY_FIRST, similar_request; a guidance entry's id, title and first
    # content line. The list is written once, after the UNIFY_LIBRARY_SNAPSHOT
    # line, and asks nothing: reading, calling or searching stays the model's
    # choice, and no turn is forced. An empty or unranked library adds
    # nothing. Off: as shipped.
    UNIFY_LIBRARY_SHORTLIST: bool = False
    # In a persistent session, a turn's final reply identical (whitespace
    # collapsed, JSON compared with sorted keys) to an earlier reply the
    # requester has already answered is held back once: the loop appends a
    # note quoting the requester's answer to it and the model takes another
    # step, and sending the same reply again surfaces it. Replies that differ,
    # and one-shot runs, are unaffected. Off: as shipped.
    UNIFY_REPEAT_GUARD: bool = False
    # The actor's prompt asks it to use what is free before an action that
    # costs something (a paid request, a scored submission, an irreversible
    # effect): run a stored function that fits on inputs it already has and
    # act on its result when it works. A function stored while handling a
    # request records that request in its metadata (a hash, and a copy of at
    # most 4,000 characters, never shown in library results), and a search
    # adds ``similar_request: <score>`` to its result when the current
    # request is close to one it was stored from: 1 for the same text, or a
    # weighted overlap of their letter and digit runs, each weighted by its
    # rarity among the library's requests, of at least 0.24
    # (unify/function_manager/task_origin.py). Off: as shipped.
    UNIFY_TRY_FIRST: bool = False
    # Take the session's checked outcome from the environment (unify/outcome.py:
    # ``unify.outcome.post``, or an ``{"outcome": {...}}`` line on the stdin of
    # ``unify act --jsonl``), held in memory, never in a file. The storage review
    # then reads it in a section marked as the checker's verdict, not the
    # agent's, and its "Final Result" is the agent's last reply before the
    # outcome arrived instead of the stop notice of a persistent session.
    # Off: no outcome is taken and the review is as shipped.
    UNIFY_OUTCOME: bool = False
    # ``lessons``: a run whose outcome says it failed (``solved`` false, with
    # UNIFY_OUTCOME), or whose admission verdict is ``{"admit": "lessons"}``,
    # is reviewed with function writes refused and guidance writes allowed,
    # to record what went wrong. An admission verdict of false still skips the
    # review. Empty: failed runs are reviewed, or skipped, as shipped.
    UNIFY_REVIEW_FAILED: str = ""

    # ─────────────────────────────────────────────────────────────────────────
    # Session Transcripts
    # ─────────────────────────────────────────────────────────────────────────
    # Append every agent conversation (actor, sub-agents, storage review,
    # compressor) as JSON lines to ``<UNIFY_HOME>/transcripts/<session>.jsonl``
    # and one line per ended session to ``transcripts/index.jsonl``; after a
    # context compression the compressed context points at the file
    # (unify/transcripts.py). Off: nothing is written.
    UNIFY_TRANSCRIPTS: bool = False
    # Reasoning effort of every storage review (forked, standalone, after a
    # turn, or asked for mid-task): one of the efforts unillm forwards
    # (``none``, ``low``, ``medium``, ``high``, ``xhigh``, ``max``). A forked
    # review sends the session's messages and tools unchanged and keeps its
    # cache affinity key; only the effort of its requests differs. Empty: each
    # review runs at the effort it gets as shipped.
    UNIFY_REVIEW_REASONING_EFFORT: str = ""
    # Model of every storage review, as a unillm endpoint
    # (``provider/model@host``). A model other than the session's cannot share
    # its cache, so with UNIFY_REVIEW_FORK the review runs standalone and the
    # log says why. Without UNIFY_REVIEW_REASONING_EFFORT it runs at the
    # effort a client named for a model gets (``high``). Empty: the review
    # uses the actor's model, as shipped.
    UNIFY_REVIEW_MODEL: str = ""

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
    UNIFY_WORKSPACE: str = ""
    # ``worker`` (with ``sandboxed``): each Python session runs its cells in a
    # persistent child process inside the same bubblewrap policy, and reaches
    # ``primitives``, steering and the other harness objects only through a
    # proxy the harness serves (unify/actor/execution/worker.py). Empty: Python
    # cells run by ``exec`` in this process.
    UNIFY_WORKSPACE_PYTHON: str = ""
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
    UNIFY_TOOL_SURFACE: str = ""

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
        "UNIFY_SEARCH_SKIP_UNLOADABLE",
        "UNIFY_TOOL_CHOICE_FALLBACK",
        "UNIFY_REPEAT_GUARD",
        "UNIFY_BATCH_WAKE",
        "UNIFY_PENDING_REQUIRED",
        "UNIFY_LIFECYCLE_NOTICES",
        "UNIFY_TRY_FIRST",
        "UNIFY_FUNCTION_PATCH",
        "UNIFY_FUNCTION_CASES",
        "UNIFY_CACHE_DISCIPLINE",
        "UNIFY_LIBRARY_SNAPSHOT",
        "UNIFY_REVIEW_FORK",
        "UNIFY_TRANSCRIPTS",
        "UNIFY_OUTCOME",
        "UNIFY_BUILTIN_GUIDANCE",
        "UNIFY_REPLY_PROTOCOL_NOTE",
        "UNIFY_PROMPT_ACCURACY",
        "UNIFY_REVIEW_GATE",
        "UNIFY_CODE_FIRST",
        "UNIFY_STORE_INSTANCE_LINT",
        "UNIFY_DISCOVERY_SPECULATIVE_TURN",
        "UNIFY_DISCOVERY_GATE",
        "UNIFY_LIBRARY_SHORTLIST",
        mode="before",
    )
    @classmethod
    def parse_bool_fields(cls, v: Any) -> bool:
        return _parse_bool(v)

    @field_validator("UNIFY_STORE_CHECK", mode="before")
    @classmethod
    def parse_store_check(cls, v: Any) -> str:
        value = str(v or "").strip().lower()
        if value not in ("", "resolve"):
            raise ValueError(f"UNIFY_STORE_CHECK must be empty or 'resolve', not {v!r}")
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

    @field_validator("UNIFY_CACHE_AFFINITY_SCOPE", mode="before")
    @classmethod
    def parse_cache_affinity_scope(cls, v: Any) -> str:
        value = str(v or "").strip().lower() or "prefix"
        if value not in ("prefix", "session", "run", "static"):
            raise ValueError(
                "UNIFY_CACHE_AFFINITY_SCOPE must be 'prefix', 'session', 'run' "
                f"or 'static', not {v!r}",
            )
        return value

    @field_validator("UNIFY_PROMPT_CLOCK", mode="before")
    @classmethod
    def parse_prompt_clock(cls, v: Any) -> str:
        value = str(v or "").strip().lower()
        if value not in ("", "message"):
            raise ValueError(
                f"UNIFY_PROMPT_CLOCK must be empty or 'message', not {v!r}",
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

    @field_validator("UNIFY_REVIEW_FAILED", mode="before")
    @classmethod
    def parse_review_failed(cls, v: Any) -> str:
        value = str(v or "").strip().lower()
        if value not in ("", "lessons"):
            raise ValueError(
                f"UNIFY_REVIEW_FAILED must be empty or 'lessons', not {v!r}",
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

    @field_validator("UNIFY_STORE_DEDUPE", mode="before")
    @classmethod
    def parse_store_dedupe(cls, v: Any) -> str:
        value = str(v or "").strip().lower()
        if value not in ("", "warn"):
            raise ValueError(f"UNIFY_STORE_DEDUPE must be empty or 'warn', not {v!r}")
        return value

    @field_validator("UNIFY_WAIT_CEILING_SECONDS", mode="before")
    @classmethod
    def parse_wait_ceiling_seconds(cls, v: Any) -> float:
        value = float(15.0 if v in (None, "") else v)
        if not 1 <= value <= 120:
            raise ValueError(
                f"UNIFY_WAIT_CEILING_SECONDS must be between 1 and 120, not {v!r}",
            )
        return value

    @field_validator("UNIFY_REVIEW_REASONING_EFFORT", mode="before")
    @classmethod
    def parse_review_reasoning_effort(cls, v: Any) -> str:
        value = str(v or "").strip().lower()
        if value not in _REVIEW_EFFORTS:
            raise ValueError(
                "UNIFY_REVIEW_REASONING_EFFORT must be empty or one of "
                f"{', '.join(_REVIEW_EFFORTS[1:])}, not {v!r}",
            )
        return value

    @field_validator("UNIFY_REVIEW_MODEL", mode="before")
    @classmethod
    def parse_review_model(cls, v: Any) -> str:
        value = str(v or "").strip()
        if value and "@" not in value:
            raise ValueError(
                "UNIFY_REVIEW_MODEL must be empty or a unillm endpoint "
                f"('provider/model@host'), not {v!r}",
            )
        return value

    @field_validator("UNIFY_REVIEW_FRAMING", mode="before")
    @classmethod
    def parse_review_framing(cls, v: Any) -> str:
        value = str(v or "").strip().lower()
        if value not in ("", "unified"):
            raise ValueError(
                f"UNIFY_REVIEW_FRAMING must be empty or 'unified', not {v!r}",
            )
        return value

    @field_validator("UNIFY_CURATION_DOCTRINE", mode="before")
    @classmethod
    def parse_curation_doctrine(cls, v: Any) -> str:
        value = str(v or "").strip().lower()
        if value not in ("", "compose", "minimal"):
            raise ValueError(
                "UNIFY_CURATION_DOCTRINE must be empty, 'compose' or 'minimal', "
                f"not {v!r}",
            )
        return value

    @field_validator("UNIFY_STORE_TRUST", mode="before")
    @classmethod
    def parse_store_trust(cls, v: Any) -> str:
        value = str(v or "").strip().lower()
        if value not in ("", "ramp"):
            raise ValueError(f"UNIFY_STORE_TRUST must be empty or 'ramp', not {v!r}")
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

    def prompt_accuracy(self) -> bool:
        """Whether the UNIFY_PROMPT_ACCURACY fixes apply (the switch, or the lean profile)."""
        return bool(self.UNIFY_PROMPT_ACCURACY) or self.UNIFY_PROMPT_PROFILE == "lean"

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
