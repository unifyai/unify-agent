import asyncio
import contextvars
import copy
import functools
import inspect
import json
import re
import textwrap
import traceback
import types
import uuid
import weakref
from secrets import token_hex as _token_hex
import logging
from typing import (
    Annotated,
    Any,
    Callable,
    Awaitable,
    Dict,
    Optional,
    Type,
    Union,
    TYPE_CHECKING,
)
from pydantic import BaseModel

from unify.actor.base import BaseCodeActActor
from unify.common._async_tool import cell_reply
from unify.actor import core_surface
from unify.common.context_dump import make_messages_safe_for_context_dump
from unify import environment, transcripts
from unify.actor.workspace_tools import workspace_tools as _workspace_tools
from unify.actor.grants import CALLER_GRANTS, ActorGrants
from unify.actor.execution import (
    ExecutionResult,
    PythonExecutionSession,
    SessionExecutor,
    SessionKey,
    _CAN_CLARIFY,
    _CURRENT_ENVIRONMENTS,
    _CURRENT_SANDBOX,
    _PARENT_CHAT_CONTEXT,
    _validate_execution_params,
)
from unify.actor.execution.session import inventory_enabled
from unify.common.async_tool_loop import (
    AsyncToolLoopHandle,
    ToolLoopHandle,
    start_async_tool_loop,
)
from unify.events.event_bus import EVENT_BUS, Event
from unify.common.llm_client import fork_llm_client, new_llm_client
from unify.common.llm_meter import RunMeter, current_run_meter, new_run_meter
from unify.common.act_llm_profiles import (
    CURRENT_ACT_LLM_PROFILE,
    resolve_act_llm_profile,
)
from unify.common.llm_helpers import methods_to_tool_dict
from unify.common.tool_spec import ToolSpec, llm_soft_required
from unify.function_manager import escape_drift as _escape_drift
from unify.function_manager import instance_lint as _instance_lint
from unify.function_manager.primitives.registry import get_registry
from unify.actor.prompt_builders import build_code_act_prompt
from unify.events.manager_event_logging import log_manager_call
from unify.common._async_tool.loop_config import TOOL_LOOP_LINEAGE, _PENDING_LOOP_SUFFIX
from unify.common.hierarchical_logger import log_boundary_event
from unify.events.manager_event_logging import (
    new_call_id,
    publish_manager_method_event,
)
from unify.events.active_work import ACTIVE_WORK, ActiveWorkHandle

if TYPE_CHECKING:
    from unify.actor.environments.base import BaseEnvironment
    from unify.function_manager.function_manager import FunctionManager
    from unify.guidance_manager.guidance_manager import GuidanceManager


# ---------------------------------------------------------------------------
# Tool-policy type alias and sentinel
# ---------------------------------------------------------------------------

ToolPolicyFn = Callable[[int, Dict[str, Any]], tuple[str, Dict[str, Any]]]
"""Signature for a tool-policy callback.

Receives ``(step_index, tools_dict)`` and returns ``(tool_choice_mode,
filtered_tools_dict)`` where *tool_choice_mode* is ``"auto"`` or
``"required"``.  An optional third dict ``{"eager": True}`` may be returned
to request immediate follow-up LLM turns while the policy remains eager
(see the async tool loop ``tool_policy`` docs).
"""

_USE_DEFAULT: object = object()
"""Sentinel indicating 'use the default tool policy' (the static filters only)."""

_UNSET: object = object()
"""Sentinel indicating 'parameter was not explicitly provided'."""


class _ActiveWorkNotificationQueue:
    def __init__(
        self,
        target: asyncio.Queue[dict],
        active_work: ActiveWorkHandle,
    ) -> None:
        self._target = target
        self._active_work = active_work

    async def put(self, item: dict) -> None:
        self._active_work.record_user_notification()
        await self._target.put(item)

    def put_nowait(self, item: dict) -> None:
        self._active_work.record_user_notification()
        self._target.put_nowait(item)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._target, name)


def _library_counts(
    function_manager: Any,
    guidance_manager: Any,
) -> tuple[Optional[int], Optional[int]]:
    """The stored functions and guidance entries in scope now.

    Functions exclude primitives; guidance counts the built-in entries only
    where ``UNIFY_BUILTIN_GUIDANCE`` shows them. A count a manager cannot
    give (no manager, no counter, a failed read) is ``None``: unknown.
    """

    def _count(manager: Any) -> Optional[int]:
        counter = getattr(manager, "_num_items", None) if manager else None
        if not callable(counter):
            return None
        try:
            return int(counter())
        except Exception as exc:
            logger.debug(f"library count unavailable: {type(exc).__name__}: {exc}")
            return None

    return _count(function_manager), _count(guidance_manager)


def _library_snapshot_line(
    counts: tuple[Optional[int], Optional[int]],
    *,
    has_fm_tools: bool,
    has_gm_tools: bool,
) -> Optional[str]:
    """``UNIFY_LIBRARY_SNAPSHOT``: one line giving the library's size at task start."""
    functions, guidance = counts
    parts: list[str] = []
    if has_fm_tools and functions is not None:
        parts.append(f"{functions} stored function{'' if functions == 1 else 's'}")
    if has_gm_tools and guidance is not None:
        parts.append(f"{guidance} guidance entr{'y' if guidance == 1 else 'ies'}")
    if not parts:
        return None
    return f"Library at task start: {', '.join(parts)}."


# ---------------------------------------------------------------------------
# Agent context for tracking execution depth and providing handle access
# ---------------------------------------------------------------------------
from dataclasses import dataclass, field as dataclass_field


@dataclass
class AgentContext:
    """Runtime context for agent execution, accessible via get_current_agent_context().

    Attributes:
        depth: Nesting level (0 = root agent, 1 = first subagent, etc.)
        agent_id: Unique identifier for this agent run
        handle: Reference to the AsyncToolLoopHandle (for accessing history, etc.)
    """

    depth: int = 0
    agent_id: str = dataclass_field(default_factory=lambda: str(uuid.uuid4()))
    handle: "AsyncToolLoopHandle | None" = None
    proactive_storage_summaries: list[str] = dataclass_field(default_factory=list)


_CURRENT_AGENT_CONTEXT: contextvars.ContextVar[AgentContext] = contextvars.ContextVar(
    "code_act_agent_context",
    default=AgentContext(),
)


def get_current_agent_context() -> AgentContext:
    """Get the current agent execution context.

    Use this inside service methods to:
    - Check agent depth and prevent infinite recursion
    - Access the current agent's handle for message history, etc.

    Returns:
        AgentContext with depth, agent_id, and handle

    Example:
        ctx = get_current_agent_context()
        if ctx.depth >= 2:
            raise RuntimeError("Max depth exceeded")
        if ctx.handle:
            history = ctx.handle.get_history()
    """
    return _CURRENT_AGENT_CONTEXT.get()


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Shared storage-review prompt sections
# ---------------------------------------------------------------------------

_DEFAULT_STORAGE_REVIEW_LABEL = "Storing reusable skills"
_DEFAULT_STORAGE_REVIEW_INSTRUCTIONS = (
    "Review the trajectory and store any reusable functions and "
    "compositional guidance."
)


_STORAGE_WHAT_CAN_BE_STORED = (
    "## What Can Be Stored\n\n"
    "Any code that executed successfully in `execute_code` during "
    "this trajectory can be stored as a function. Environment-provided "
    "namespaces (`primitives`, `primitives.actor`) and "
    "other stored functions referenced in the code are automatically "
    "detected from the source and injected at runtime — you do not "
    "need to add imports or worry about whether these names will be "
    "available when the function runs later. Focus on whether a "
    "pattern is *worth* reusing, not whether it is *technically "
    "executable* in isolation.\n\n"
    "### Bake configuration into reusable callables\n\n"
    "Stored functions should NOT be verbatim copies of code blocks "
    "from the trajectory. During execution, the agent discovered the "
    "right combination of parameters, tool selections, and strategies "
    "through reasoning — that configuration knowledge is the valuable "
    "part. A stored function should **bake in** the hard-won "
    "configuration as fixed values and **expose** only the parts "
    "that genuinely vary between uses (typically the task-specific "
    "input). This produces a function that future callers can use "
    "without rediscovering the right setup.\n\n"
    "The same applies to trajectories that unrolled as "
    "`execute_code` -> observe -> agent reasoning -> `execute_code` loops. "
    "Do not assume the future procedure needs a full CodeActActor. First "
    "ask whether the agent's intermediate reasoning was open-ended "
    "planning or bounded semantic judgment inside an otherwise stable "
    "control flow. If it was bounded, distill the trajectory into one "
    "function: deterministic Python for control flow, managed primitives "
    "for side effects, and focused `query_llm(...)` calls with "
    "structured outputs, low temperature, and an explicit model for "
    "classification, summarization, drafting, ranking, or source "
    "selection. Leave it live-agent or guidance-driven when the reasoning "
    "involved changing tool discovery, unknown-state debugging, user "
    "clarification, or broad strategy selection.\n\n"
    "Semantic downgrades are bugs. When a live trajectory interpreted or "
    "produced unstructured data — classification, extraction, routing, "
    "summarization, drafting, rewriting, personalization, or other "
    "human-facing synthesis — the stored function should preserve that "
    "fuzzy step as `query_llm(...)` with a stable contract. Do not "
    "replace it with keyword ladders, regex classifiers, label-specific "
    "canned prose, or templates inferred from observed examples unless "
    "the user explicitly requested fixed deterministic rules/templates. "
    "Generalize by preserving the LLM call, not by memorizing the sample "
    "cases.\n\n"
    "### The distillation dial\n\n"
    "Skill maturation is a dial, not a switch. Every procedure "
    "decomposes into a skeleton — control flow that is the same on "
    "every run — and joints — the substeps where meaning must be "
    "judged. Each substep sits independently at one notch: plain "
    "deterministic Python, a focused stateless `query_llm(...)` call, "
    "a sub-agent via `primitives.actor.act(...)`, or left to the live "
    "agent. Distilling a trajectory means choosing a notch per "
    "substep, not one mode for the whole task: freeze the skeleton, "
    "and keep every genuine judgment fluid at the cheapest notch that "
    "preserves it.\n\n"
    "Know what each direction costs. Distilling too little is loud "
    "and cheap to fix: planning cost is paid again on every run, "
    "forever. Distilling too much is quiet and expensive: "
    "everything stays cheap and correct until the first input the "
    "frozen structure was never validated on — and a frozen function "
    "processes an anomaly as if it were normal, because the fluid "
    "intelligence that would have noticed was distilled away. The "
    "same asymmetry governs placement: freezing a semantic joint into "
    "keyword ladders or templates fails silently on the first "
    "paraphrase, while leaving exact work inside an LLM call merely "
    "wastes tokens. When in doubt, a joint stays semantic — an "
    "unnecessary `query_llm(...)` call costs cents; an unnecessary "
    "regex costs correctness.\n\n"
    "Freeze only observed invariance. The evidence for a notch is the "
    "trajectory itself: structure you watched hold across the run(s) "
    "may be frozen; structure inferred from a single example may not. "
    "A branch the agent reasoned about once is still a judgment, not "
    "yet control flow.\n\n"
    "Preserve the surprise signal. A live agent notices when an input "
    "is strange; a stored function must be given that noticing back "
    "explicitly. State the envelope the structure was validated "
    "inside — type hints, checkable postconditions, precondition "
    "guards on the ranges, shapes, and assumptions the trajectory "
    "actually exhibited — and make the function raise or return early "
    "on inputs outside it, so the calling actor (or a `query_llm(...)` "
    "judgment) handles the anomaly instead of the frozen path "
    "swallowing it.\n\n"
    "Distillation is reversible. Stored functions are inspectable and "
    "revisable; re-opening one joint later is routine maintenance, not "
    "a failure. Do not under-distill out of caution — freeze the "
    "skeleton the evidence supports, guard it, and let future "
    "corrections move the dial.\n\n"
    "### Model choice is part of distillation\n\n"
    "Every `query_llm(...)` call you bake into a stored function is a "
    "standing model choice. Choose `model=` deliberately per the "
    '"Choosing A Model" section of `help(query_llm)` — bounded, '
    "repeated classification/extraction rarely needs the default "
    "high-reasoning model. When candidates look close, trial them "
    "against the trajectory itself: it already contains the concrete "
    "inputs and known-good outputs for each semantic substep, so "
    "replay those cases through each candidate with the same prompt "
    "and `response_format`, keep the cheapest model that passes, and "
    "record the rationale and the trial cases in the function's "
    "docstring, so a later model or prompt change can replay them.\n\n"
    "### Preserving user-facing communication points\n\n"
    "When wrapping a procedure into a stored function, pay attention to "
    "points where the original code depended on the user being "
    "informed — especially states that block until the user takes an "
    "external action. Stored functions cannot emit user-facing "
    "notifications mid-execution, so do not bake an indefinite wait on "
    "external human action (approving an auth prompt, granting a "
    "permission, confirming a destructive operation) into the function "
    "body. Instead, have the function return early or raise with a "
    "clear message describing what the user must do, so the calling "
    "actor can inform the user (via the `send_notification` tool "
    "between `execute_code` blocks) and resume afterwards. A silent "
    "in-function wait is a deadlock: the function blocks on a "
    "condition the user does not know about.\n\n"
    "### Expressive logging in stored functions\n\n"
    "Soft failures (empty results, skipped branches, degraded fallbacks, "
    "status dicts that report problems without raising) are the common "
    "bug shape. Stored functions must leave a reconstructable trail with "
    "the stdlib `logging` module — not via user-facing notifications. Use "
    "markers `PHASE`, `SKIP`, and `SOFT_FAIL` in the message text so "
    "Job/EventBus captures stay greppable, e.g. "
    "`logging.getLogger(__name__).info('PHASE load_rows count=%s', n)` "
    "or `.warning('SKIP empty_result')` / `.error('SOFT_FAIL partial …')`. "
    "Log every meaningful stage boundary, every intentional skip, and "
    "every soft failure. Do **not** strip PHASE/SKIP/SOFT_FAIL trails when "
    "distilling a live trajectory into a stored function. You may remove "
    "dead exploratory `print`s, duplicated setup, or formatting noise, "
    "but never remove validation gates, recovery branches, or diagnostic "
    "logging that explains why a path returned early or returned empty.\n\n"
    "### Async / event-loop safety in stored functions\n\n"
    "The runtime already owns an event loop via "
    "`asyncio.run`. Nested `asyncio.run(...)` inside a sync helper then "
    "raises `RuntimeError: asyncio.run() cannot be called from a running "
    "event loop`. Prefer `async def` entrypoints / helpers and `await` "
    "end-to-end (including `await query_llm(...)`). When a sync façade is "
    "required, call the injected `run_coro_sync(factory)` helper (also "
    "`from unify.common.asyncio_compat import run_coro_sync`) instead of "
    "nesting `asyncio.run`.\n\n"
    "### Functions that are easy to get right\n\n"
    "A stored function is a tool the actor will call without re-reading "
    "its body, so store functions that are cheap to check and hard to "
    "get wrong:\n\n"
    "- **Thin effects.** A function that performs an irreversible effect "
    "(send, post, delete, pay) must do only that. Compute "
    "in one function, perform the effect in another, and let the root "
    "compose them. This is what makes a failure cheap to fix and "
    "blame precise: the computation can be re-run and corrected without "
    "the effect ever having happened. Example — instead of one "
    "`send_weekly_summary(week)` that fetches, totals and posts, store "
    "`compute_weekly_summary(week) -> dict` (read-only, checkable "
    "against its inputs) and `post_summary(channel: str, text: str) -> "
    "dict` (the one effect), with a root that calls the first, then the "
    "second.\n"
    "- **Type hints on every parameter and the return.** The signature is "
    "what a caller reads before calling; an unhinted function tells it "
    "nothing about what goes in or what comes out.\n"
    "- **A docstring whose first sentence is a checkable postcondition** "
    '("Return the sum of `amount` over the rows, in minor units, as an '
    'int"), not a paraphrase of the name. Where the trajectory contains '
    "concrete inputs and the exact output a pure function reproduces, "
    "record those pairs in the docstring as well, so a later change can "
    "be checked against them.\n\n"
    "### Third-party package dependencies\n\n"
    "If the trajectory used `install_python_packages` and the function "
    "you want to store imports any of those packages (anything beyond "
    "the Python standard library and the environment-provided "
    "namespaces `primitives` and `pydantic`), **record the "
    "dependencies**: pass the pip specifiers that installed them — the "
    "same strings the `install_python_packages` call used, e.g. "
    '`dependencies=["google-cloud-storage>=2.0.0"]` — to '
    "`FunctionManager_add_functions`. They are installed into the "
    "workspace environment before the function runs, wherever it runs. "
    "`FunctionManager_add_functions` rejects a function whose "
    "third-party imports come without `dependencies`. Detection covers "
    "every import form — `import`, `from … import` and dynamic imports "
    'with a literal name such as `importlib.import_module("pkg")` — '
    "because the dependency is a property of the package, not of the "
    "syntax. Never rewrite an import to slip past the check: a function "
    "stored without its dependencies fails on the first call from a "
    "machine that lacks the package. If a function already stored this "
    "way appears in the trajectory, repair it — overwrite it with a "
    "plain `import` and the right `dependencies` — rather than leaving "
    "it. Declare only the packages the function actually imports, "
    "pinned as loosely as the trajectory justifies.\n\n"
)


def _environment_method_kinds(namespaces: tuple[Any, ...]) -> str:
    """Which registered environment methods are asynchronous, read from the callables.

    A method is asynchronous when calling it gives an awaitable
    (``environment.is_async_method``); every other method returns its value
    directly, so ``await`` on it raises ``TypeError``.
    """
    from unify.function_manager.primitives.environment import is_async_method

    sync_only: list[str] = []
    async_only: list[str] = []
    mixed: list[str] = []
    for namespace in namespaces:
        kinds = {m.name: is_async_method(m) for m in namespace.methods}
        label = f"`primitives.{namespace.name}`"
        if not any(kinds.values()):
            sync_only.append(label)
        elif all(kinds.values()):
            async_only.append(label)
        else:
            # Name the smaller group, so the sentence stays short.
            is_async = sum(kinds.values()) <= len(kinds) / 2
            named = sorted(name for name, kind in kinds.items() if kind is is_async)
            listed = ", ".join(f"`{name}`" for name in named)
            verb = "is" if len(named) == 1 else "are"
            if is_async:
                mixed.append(
                    f"In {label}, {listed} {verb} asynchronous (`await` "
                    f"{'it' if len(named) == 1 else 'them'}) and the other "
                    "methods are synchronous.",
                )
            else:
                mixed.append(
                    f"In {label}, {listed} {verb} synchronous (no `await`) "
                    "and the other methods are asynchronous.",
                )
    if not async_only and not mixed:
        return (
            "Every method of these namespaces is synchronous: it returns its "
            "value directly, so call it without `await`."
        )
    if not sync_only and not mixed:
        return "Every method of these namespaces is asynchronous: `await` it."
    sentences: list[str] = []
    if sync_only:
        sentences.append(
            f"The methods of {', '.join(sync_only)} are synchronous: call "
            "them without `await`.",
        )
    if async_only:
        sentences.append(
            f"The methods of {', '.join(async_only)} are asynchronous: "
            "`await` them.",
        )
    sentences.extend(mixed)
    return " ".join(sentences)


def _storage_environment_note() -> str:
    """The storage review's note on the environment's namespaces and the storage check.

    It always states the storage check, and adds the environment's
    namespaces (``UNIFY_ENV_NAMESPACES``) and the verification before
    storing (``UNIFY_STORE_VERIFY``) when they are set.
    """
    from unify.function_manager.primitives.environment import environment_surface

    surface = environment_surface()
    parts: list[str] = []
    if surface is not None and surface.namespaces:
        names = ", ".join(f"`primitives.{n.name}`" for n in surface.namespaces)
        parts.append(
            "This environment registered its own namespaces beside "
            f"`primitives.actor`: {names}. Code calls them exactly so "
            "(`primitives.<namespace>.<method>(...)`), and a stored function "
            "that does is recorded and injected like `primitives.actor`: it "
            "needs no import and no dependency for them. No other "
            "`primitives.*` name exists.",
        )
        # The minimal rulebook says to await only what is asynchronous;
        # this says which of the environment's methods are.
        parts.append(_environment_method_kinds(surface.namespaces))
        if surface.globals:
            listed = ", ".join(f"`{g}`" for g in sorted(surface.globals))
            parts.append(f"The environment also binds the sandbox globals {listed}.")
        if surface.modules:
            listed = ", ".join(f"`{m}`" for m in sorted(surface.modules))
            parts.append(
                f"Modules it supplies ({listed}) are importable wherever a "
                "stored function runs; never declare them as `dependencies`.",
            )
    parts.append(
        "`FunctionManager_add_functions` checks each function before "
        "storing it: every name it reads and every `primitives.*` "
        "reference must exist where it will run, and it must load. A "
        "function that fails is not stored, and the error names what "
        "failed; fix the function and add it again.",
    )
    from unify.function_manager import store_verify

    if store_verify.enabled():
        parts.append(store_verify.doctrine())
    if not parts:
        return ""
    return "### This environment\n\n" + " ".join(parts) + "\n\n"


# UNIFY_REPLY_CHANNEL=code+text: execute_function runs a stored function,
# which returns its result rather than replying.
_REPLY_REFUSED_IN_FUNCTION = (
    "reply() cannot be called through execute_function, which runs a stored "
    "function: call reply() in an execute_code cell"
)
_EXECUTE_CODE_REPLY_DOC = """
Replying from a cell
--------------------
``reply(text)`` sends ``text`` (a str) as your reply and ends your turn,
as replying with that text would; the cell stops there. For example
``reply(answer)`` when the answer is in a variable."""


def _storage_review_client(actor: "CodeActActor", *, origin: str) -> Any:
    """A standalone storage review's client: the actor's model, as shipped."""
    return new_llm_client(
        actor._model,
        purpose="planning",
        origin=origin,
    )


def _review_gate_client(actor: "CodeActActor", session_client: Any = None) -> Any:
    """The ``UNIFY_REVIEW_GATE`` call's client: the review's model, at the
    effort the session ran at.

    Effort is a fixed condition of a run, never the harness's to change: with
    the session's client, the gate's request carries that client's reasoning
    effort (none when it has none); without one, the effort the standalone
    review's client gets from the actor's model."""
    from unify.actor import review_gate

    client = _storage_review_client(actor, origin=review_gate.ORIGIN)
    if session_client is not None:
        client.set_reasoning_effort(getattr(session_client, "reasoning_effort", None))
    # Its prompt carries the checked outcome: never where a cell can read it.
    return transcripts.mark_internal(client)


# UNIFY_CURATION_DOCTRINE=compose: how the library is built and kept.
GUIDANCE_ENTRY_TARGET_CHARS = 2000

_STORAGE_COMPOSE_DOCTRINE = (
    "## Building The Library\n\n"
    "A finished trajectory usually holds at least one reusable unit: a step "
    "that ran successfully and that another task of the same kind would "
    "perform again. Store it. A pass that changes nothing is right only "
    "when nothing in the trajectory ran successfully or when everything "
    "reusable is already in the library; say which in your summary.\n\n"
    "- **Small units, composed.** Break the work into the smallest units "
    "that each do one thing, and expose as parameters what varies between "
    "tasks (inputs, names, thresholds, identifiers). When the trajectory "
    "solved a whole procedure, also store a root function that calls the "
    "stored units in order, so the next task runs the procedure in one "
    "call. A unit already in the library is called, not copied.\n"
    "- **Only what ran.** Store code the trajectory executed and whose "
    "result it observed; a unit that worked inside a task that failed "
    "overall still qualifies.\n"
    "- **Named for behaviour.** A name, signature and docstring describe "
    "what the unit does for any caller. Values specific to one instance "
    "(an id, a file name, a literal answer) are parameters or are left "
    "out. Ask of each entry: would a different task of this kind call "
    "this as it stands?\n"
    "- **Never break what works.** A patch must keep the entry's behaviour "
    "on the inputs it already handled: fix a defect, widen what it "
    "accepts, or clarify it. When the behaviour itself must change, store "
    "the new behaviour under a new name and retire the old entry (delete "
    "it, or say in its docstring which entry replaces it), because callers "
    "and guidance written against the old behaviour still expect it.\n"
    f"- **Short guidance.** Keep a guidance entry under about "
    f"{GUIDANCE_ENTRY_TARGET_CHARS:,} characters and to one subject. When "
    "a lesson would grow an entry past that, or is about a different "
    "subject, write a new focused entry and link it, rather than "
    "appending. Do not add run-by-run narrative to an entry.\n"
    "- **Cost.** Prefer units that replace several reasoning steps or tool "
    "calls with one call; a unit the next task would not call is clutter."
    "\n\n"
)


def _storage_compose_note() -> str:
    """The compose doctrine."""
    return _STORAGE_COMPOSE_DOCTRINE


# UNIFY_CURATION_DOCTRINE=minimal: the rulebook keeps what storage needs --
# what can be stored and how it runs, the compose rules, dependencies -- and
# drops what was written for an office assistant (user notifications,
# recurring weekly deliverables, specialist sub-agents, model-choice trials,
# logging markers, the distillation essay).
_STORAGE_MINIMAL_WHAT = (
    "## What Can Be Stored\n\n"
    "Code that ran successfully in this trajectory can be stored as a "
    "function with `FunctionManager_add_functions`. The `primitives.*` "
    "namespaces and the stored functions it calls are detected from its "
    "source and injected when it runs, so it needs no imports for them. A "
    "step that judged meaning in the trajectory (classifying, extracting, "
    "drafting) stays a `query_llm(...)` call in the stored function. "
    "Await only what is asynchronous: `query_llm(...)`, the "
    "`primitives.actor` methods and stored functions defined with "
    "`async def`; a function that awaits one is itself `async def`. A "
    "synchronous method returns its value directly, and awaiting that "
    "value raises `TypeError`, so call it without `await`. The runtime "
    "owns the event loop: synchronous code that must run a coroutine uses "
    "the injected `run_coro_sync(factory)`, not `asyncio.run`. A function "
    "that imports a third-party "
    "package is stored with `dependencies` set to the pip specifiers "
    "`install_python_packages` used; `FunctionManager_add_functions` "
    "refuses it without them.\n\n"
)
_STORAGE_MINIMAL_GUIDANCE = (
    "Guidance (`GuidanceManager_add_guidance`, linked to the functions it "
    "uses through `function_ids`) is short prose for what code cannot "
    "carry: a composition that would be hard to rediscover, or an approach "
    "that failed in a non-obvious way and what worked instead. A function "
    "whose docstring covers its use needs no guidance entry.\n\n"
)


def _storage_doctrine_sections() -> str:
    """The rulebook sections before the instructions (the minimal rulebook)."""
    return (
        f"{_STORAGE_MINIMAL_WHAT}"
        f"{_STORAGE_MINIMAL_GUIDANCE}"
        f"{_storage_environment_note()}"
        f"{_storage_compose_note()}"
        f"{_storage_update_first_note()}"
    )


def _storage_update_first_note() -> str:
    """The review's update-before-add order."""
    return (
        "### Update before you add\n\n"
        "When the trajectory shows a stored entry that was wrong, "
        "incomplete or failed, (1) patch the entry the trajectory used "
        "(`FunctionManager_patch_function` / "
        "`GuidanceManager_patch_guidance`) when the fix keeps its "
        "behaviour on the inputs it already handled; (2) otherwise add a "
        "new focused entry, under a new name when the behaviour changes. "
        "Do not move a fix into a broader entry the trajectory did not "
        "use. A patch replaces excerpts of the entry: read its current "
        "text first, copy each `old` with enough context to occur once, "
        "and say `why`. Make several changes to one entry in one call as "
        "`edits` (`[{old, new}, ...]`, applied in order, all or none). "
        "The entry keeps its id, precondition, dependencies and links, a "
        "patched function is checked like any function you add, and the "
        "replaced version is kept in history. Rewrite a whole function "
        "with `overwrite=True` only when most of it changes.\n\n"
    )


_STORAGE_TWO_STORES = (
    "## Two Stores\n\n"
    "### Function Store — the *what*\n\n"
    "The FunctionManager stores concrete reusable callables. Add a "
    "genuinely new function with `FunctionManager_add_functions` "
    "(`dependencies` required for third-party imports). Revise an "
    "existing function in "
    "place with `overwrite=True`. When a new function subsumes narrower "
    "variants, delete the superseded entries "
    "(`FunctionManager_delete_function`). Do NOT store trivial "
    "one-liners, test scaffolding, or functions too task-specific to be "
    "reusable.\n\n"
    "### Guidance Store — the *how*\n\n"
    "The GuidanceManager stores procedural recipes: multi-step "
    "compositions, SOPs, and decision points — prose that references "
    "functions, not executable code (`GuidanceManager_add_guidance` / "
    "`GuidanceManager_update_guidance` / `GuidanceManager_delete_guidance`, "
    "cross-referencing concrete functions via `function_ids`).\n\n"
    "Guidance earns its entry only when a composition strategy is "
    "non-obvious and would be hard to rediscover — or when a simple "
    "*domain* operation required a non-obvious correction (an "
    "error-recovery loop against an external API, a silent data failure "
    "mode, a precondition discovered by trial and error): there the "
    "domain insight is the value. Do NOT store agent-runtime or tooling "
    "meta-tips (tool-loop plumbing, namespace-injection quirks, "
    "clarification-tool usage) — those are session mechanics, not domain "
    "playbooks. Do NOT duplicate what a function docstring already "
    "explains: when the only reusable artifact is one standalone function "
    "whose docstring fully covers its use, store the function and finish "
    "the review — no wrapper procedure restating its contract.\n\n"
    "**A procedure the requester spelled out is guidance's canonical "
    "input, not a one-off.** When the request itself lays out phases, "
    "thresholds, quality gates or branch conditions, and the trajectory "
    "executed them inline as decision logic *between* function calls "
    "(or the requester asked that they not be collapsed into one "
    "function), the procedure is the reusable half of the deliverable: "
    "store it as a guidance entry — the phases in order, each decision "
    "point with its threshold and the branch each outcome selects, linked "
    "via `function_ids` to the functions it composes. Being handed the "
    "steps does not make them rediscoverable: the future session asked "
    "to run the same process will not have the spec in front of it, and "
    "the requester dictating a procedure is the strongest evidence of "
    "what they expect followed again. Framing such as 'demonstrate', "
    "'walk through' or 'example' describes the sample data, not the "
    "procedure's lifespan — omit the sample data and the run's specific "
    "outputs, keep the procedure. Numbered steps that are merely the "
    "internal algorithm of one function are that function's docstring, "
    "not guidance.\n\n"
    "**Shared rules and policies are the other first-class use of "
    "guidance.** A durable rule that could equally govern other "
    "procedures (thresholds, routing or escalation criteria, formatting "
    "or tone conventions, approval rules) belongs in ONE canonical "
    "guidance entry, linked via `function_ids` to every stored function "
    "that applies it — even when the procedure itself is simple. Search "
    "guidance for an existing statement of the rule first and link the "
    "new function into it rather than writing a second copy. Functions "
    "may bake the rule's current parameters into their implementation; "
    "name the linked entry in the function's docstring. When the rule "
    "changes later, the entry's `function_ids` enumerate exactly which "
    "functions must be revised — complete links at storage time are what "
    "make that maintenance reliable.\n\n"
    "### Composing the stores\n\n"
    "Function = executable *what*; guidance = natural-language *how* "
    "referencing functions. When a trajectory reveals both a useful "
    "function and a non-trivial procedure using it, store the function "
    "first, then a guidance entry referencing it via `function_ids`. A "
    "durable domain fact worth keeping (a rate limit, a data quirk, a "
    "convention an API enforces) lives in the guidance entry or the "
    "function docstring that acts on it — there is no separate fact "
    "store.\n\n"
)

_STORAGE_SUB_AGENT_PATTERNS = (
    "## Sub-Agent Delegation Patterns\n\n"
    'First apply the distillation dial (see "What Can Be Stored"): a '
    "`primitives.actor.act(...)` call is worth storing *as an agent* only "
    "when the sub-task genuinely needed its plan discovered at runtime. "
    "When the sub-agent's work was actually bounded judgment inside "
    "stable control flow, distill it down the dial — a function with "
    "`query_llm(...)` at the joints — instead of preserving the agent "
    "wrapper. Note also that a stored function which spawns an agent is "
    "the hardest kind to inspect or reason about; thin, bounded "
    "functions are far easier to check and reuse.\n\n"
    "Calls that survive this test are especially "
    "high-value storage candidates because they represent **pre-configured "
    "specialist agents**. Each `primitives.actor.act` invocation encodes a curated "
    "combination of `prompt_functions` (which tools the sub-agent sees), "
    "`guidelines` (how it should reason and compose those tools), "
    "`discovery_scope` (what it can find via search), and permission "
    "flags — together these define a specialist that can handle a "
    "particular *class* of tasks, not just the single task it was "
    "originally invoked for.\n\n"
    "### When to store\n\n"
    "Not every `primitives.actor.act` call is worth storing. Use this spectrum:\n\n"
    "- **Low value** — broad, unscoped delegation: every stored function "
    "in `prompt_functions`, generic or no `guidelines`, no "
    "`discovery_scope`, trivial `request`. This is just a passthrough "
    "that any future agent could reconstruct trivially.\n"
    "- **High value** — curated specialist: a carefully selected set of "
    "`prompt_functions`, detailed `guidelines` explaining how to compose "
    "those specific tools, a narrowed `discovery_scope`, and a non-trivial "
    "task that the sub-agent solved successfully. The configuration "
    "required real reasoning to discover and would be hard to "
    "rediscover from scratch.\n\n"
    "The more curation and domain knowledge went into the `primitives.actor.act` "
    "parameters, the more valuable it is to store.\n\n"
    "### What to bake in vs expose\n\n"
    "The parameters split naturally into two categories:\n\n"
    "- **Bake in** (agent specification): `guidelines`, "
    "`prompt_functions`, `discovery_scope`, `can_compose`, `can_store`, "
    "`can_spawn_sub_agents`, `timeout` — these define *what kind of "
    "specialist* this is and should be fixed in the stored function.\n"
    "- **Expose** (task specification): `request` — this defines *what "
    "to ask the specialist to do* and should be a parameter of the "
    "stored function.\n\n"
    "The result is a function that future callers can invoke with just "
    "a `request` string, without needing to know anything about the "
    "right tool selection, scoping, or behavioral guidelines.\n\n"
)


_STORAGE_COMPOSE_STEP_3 = (
    "3. Decide what would improve the library: new units and the root that "
    "composes them, patches that keep existing behaviour, new names for "
    "changed behaviour, and the retirement of entries they supersede. A "
    "clean library is one whose every entry is small, general and correct, "
    "not one with few entries. Add guidance when a composition is "
    "genuinely non-obvious or when the requester specified a multi-phase "
    "procedure with decision points, and factor any durable shared rule "
    "into a single linked guidance entry per the Shared rules section.\n"
)


def _storage_base_instructions() -> str:
    start = _STORAGE_BASE_INSTRUCTIONS.index("3. Decide")
    end = _STORAGE_BASE_INSTRUCTIONS.index("4. **Delete")
    return (
        _STORAGE_BASE_INSTRUCTIONS[:start]
        + _STORAGE_COMPOSE_STEP_3
        + _STORAGE_BASE_INSTRUCTIONS[end:]
    )


_STORAGE_BASE_INSTRUCTIONS = (
    "## Instructions\n\n"
    "1. Review the trajectory for reusable patterns — including **pitfall "
    "patterns**, where an obvious approach failed in a non-obvious way (a "
    "silent data loss, a precondition an API doesn't enforce, an "
    "error-recovery loop after a misleading tool contract). Corrected "
    "pitfalls have high reuse value even when the fix is simple, because "
    "every future actor will attempt the obvious approach first; a brief "
    "guidance entry — what fails, why, the correct approach — saves them "
    "the discovery cycle.\n"
    "2. Search the existing stores to understand what already exists "
    "(use each store's search/filter tools).\n"
    "3. Decide what would improve the library. Prefer a clean, "
    "non-redundant library over a large one — most trajectories warrant "
    "function changes at most. Add guidance when a composition is "
    "genuinely non-obvious or when the requester specified a multi-phase "
    "procedure with decision points, and factor any durable shared rule "
    "into a single linked guidance entry per the Shared rules section.\n"
    "4. **Delete superseded functions when you add a generalization** "
    "(`FunctionManager_delete_function` on the now-redundant "
    "`function_id`s) — the same for outright duplicates and narrow "
    "special cases the new function handles.\n"
    "5. When done (or if there is nothing worth changing), respond with "
    "a brief summary of what you did (or that nothing was needed)."
)

# ---------------------------------------------------------------------------
# Shared tool docstrings
# ---------------------------------------------------------------------------

# No delegation through primitives.actor: execute_function's docs do not
# offer the sub-actor primitive as their example of a primitive.
_SUB_ACTOR_EXAMPLES_DOC = (
    (
        re.compile(r"a primitive\s+\(``primitives\.actor\.act``\) or a stored"),
        "a primitive or a stored",
    ),
    (
        re.compile(
            r"\(dotted path for primitives, e\.g\.\s+``\"primitives\.actor\.act\"``\)",
        ),
        "(dotted path for primitives)",
    ),
)


# The lean profile: the code tools describe what they do, without
# preferring one over the other, and the install tool says why installs go
# through it instead of ordering it.
_LEAN_TOOL_DOCS = (
    (
        re.compile(
            r"\*\*IMPORTANT — single-call rule\*\*: If the task requires only a"
            r"\s+single function or primitive call with no surrounding logic,"
            r"\s+use ``execute_function`` instead\. ``execute_code`` is for"
            r"\s+\*\*multi-step composition\*\* — conditional logic, loops, or"
            r"\s+combining multiple primitives/functions where intermediate"
            r"\s+results are needed within the same code block\.\s+",
        ),
        "",
    ),
    (
        re.compile(
            r"\*\*This is the preferred tool for any task that maps to a single"
            r"\s+function or primitive call\*\* — (?P<what>.*?)\. It"
            r"\s+\*\*structurally guarantees\*\* the returned handle is exposed to"
            r"\s+the outer loop for steering \(ask, stop, pause, resume,"
            r"\s+interject\); inside ``execute_code`` a handle is only adopted"
            r"\s+if it happens to be the last expression\. Use ``execute_code``"
            r"\s+only for genuine multi-step composition \(conditional logic,"
            r"\s+loops, combining intermediate results\)\.",
            re.DOTALL,
        ),
        lambda m: (
            "It runs one callable -- "
            + " ".join(m.group("what").split())
            + " -- and exposes the handle it returns to the outer loop for "
            "steering (ask, stop, pause, resume, interject)."
        ),
    ),
)
_LEAN_INSTALL_DOC = (
    re.compile(
        r"\*\*You MUST use this tool whenever you need a Python package that is not"
        r"\s+already available\.\*\* Never install via ``execute_code`` \(``!pip install``,"
        r"\s+``subprocess\.run\(\[\"pip\", \.\.\.\]\)``, ``uv pip install``, or any other"
        r"\s+shell-based method\) — direct installs bypass the managed environment and"
        r"\s+leave it in an inconsistent state\.",
    ),
    "Packages that are not already available are installed with this tool. "
    "An install from ``execute_code`` (``pip``, ``uv``, a subprocess) bypasses "
    "the managed environment and leaves it inconsistent.",
)


# UNIFY_PROMPT_TRIM, no environment in the ``primitives`` namespace: the
# code tools name no primitive and no handle only a primitive returns.
# Applied after the other rewrites, so each pattern takes the shipped form
# and the forms they leave.
_TRIM_NO_PRIMITIVES_DOC = (
    (
        re.compile(r"a primitive(?:\s+\(``primitives\.actor\.act``\))?\s+or a stored"),
        "a stored",
    ),
    (re.compile(r"function or primitive"), "function"),
    (re.compile(r"primitives/functions"), "functions"),
    (re.compile(r"function\s+or primitive call"), "function call"),
    (
        re.compile(
            r"\s*\(dotted path for primitives(?:, e\.g\.\s+``\"primitives\.actor\.act\"``)?\)",
        ),
        "",
    ),
    (
        re.compile(
            r"It runs one callable -- (?P<what>.*?) -- and exposes the handle it"
            r" returns to the outer loop for steering \(ask, stop, pause, resume,"
            r" interject\)\.",
            re.DOTALL,
        ),
        lambda m: f"It runs one callable: {m.group('what')}.",
    ),
    (
        re.compile(
            r"It\s+\*\*structurally guarantees\*\* the returned handle is exposed to"
            r"\s+the outer loop for steering \(ask, stop, pause, resume,"
            r"\s+interject\); inside ``execute_code`` a handle is only adopted"
            r"\s+if it happens to be the last expression\.\s+",
        ),
        "",
    ),
    (
        re.compile(
            r",(?P<ws>\s+)at the top of every loop body, and before\s+every"
            r" ``primitives\.\*`` call\.",
        ),
        lambda m: f" and{m.group('ws')}at the top of every loop body.",
    ),
)


_TRIM_STORE_SKILLS_EXAMPLE = re.compile(
    r"a non-obvious\s+configuration of primitives\.actor\.act,\s+",
)


def _correct_tool_docs(
    tools: Dict[str, Any],
    *,
    environments: Optional[Dict[str, Any]] = None,
) -> None:
    """Correct the docstrings (tool descriptions) of *tools* in place, per the switches.

    The tools are built per actor, so a rewrite never reaches another actor.
    With the switches off the docstrings are as shipped.
    """
    from unify.actor import placeholder_note

    rewrites: list = []
    rewrites.extend(_SUB_ACTOR_EXAMPLES_DOC)
    rewrites.extend(_LEAN_TOOL_DOCS)
    rewrites.append(_LEAN_INSTALL_DOC)
    if "primitives" not in (environments or {}):
        rewrites.extend(_TRIM_NO_PRIMITIVES_DOC)
    for name in ("execute_code", "execute_function", "install_python_packages"):
        tool = tools.get(name)
        fn = tool.fn if isinstance(tool, ToolSpec) else tool
        if fn is None or not fn.__doc__:
            continue
        doc = fn.__doc__
        for pattern, replacement in rewrites:
            doc = pattern.sub(replacement, doc)
        if name == "execute_function":
            doc = placeholder_note.correct_doc(doc)
        fn.__doc__ = doc
    if "primitives" not in (environments or {}):
        tool = tools.get("store_skills")
        fn = tool.fn if isinstance(tool, ToolSpec) else tool
        if fn is not None and fn.__doc__:
            fn.__doc__ = _TRIM_STORE_SKILLS_EXAMPLE.sub("", fn.__doc__, count=1)


def _hide_parent_chat_context(tools: Dict[str, Any]) -> None:
    """UNIFY_PROMPT_TRIM without a primitives environment: the conversation a
    code tool is given reaches only the primitives it forwards to, so the
    code tools do not offer ``include_parent_chat_context``. The parameter is
    left out of the signature the loop reads; the functions are unchanged."""
    for name in ("execute_code", "execute_function"):
        tool = tools.get(name)
        fn = tool.fn if isinstance(tool, ToolSpec) else tool
        if fn is None:
            continue
        sig = inspect.signature(fn)
        if "_parent_chat_context" not in sig.parameters:
            continue
        fn.__signature__ = sig.replace(
            parameters=[
                p for p in sig.parameters.values() if p.name != "_parent_chat_context"
            ],
        )


# One contract for the package-install tool.
_INSTALL_PYTHON_PACKAGES_DOC = """Install Python packages into the workspace environment.

**You MUST use this tool whenever you need a Python package that is not
already available.** Never install via ``execute_code`` (``!pip install``,
``subprocess.run(["pip", ...])``, ``uv pip install``, or any other
shell-based method) — direct installs bypass the managed environment and
leave it in an inconsistent state.

Installed packages are immediately importable in subsequent ``execute_code``
Python calls and stay installed: the workspace environment is one persistent
venv shared by every task and session, so a package installed once is
available from then on. Try the import first — it may already be there.
If a requested package conflicts with one of the runtime's own
dependencies, the runtime's version takes precedence. When a stored
function needs a package, record the specifier used here as one of its
``dependencies`` so the install repeats wherever the function runs.

Parameters
----------
packages : list[str]
    pip/uv specifiers, e.g. ``"pandas"``, ``"pandas==2.1.0"``,
    ``"pandas>=2.0,<3.0"``, ``"pandas[sql]"``, ``"./path/to/wheel.whl"``.
    Only wheels are installed, from the package index: a package with no
    wheel for this platform, a git URL or a local source directory fails,
    as does any other host.

Returns
-------
dict
    ``success`` (bool), ``stdout`` / ``stderr`` (installer output — on
    failure inspect ``stderr`` and adjust the specifiers), and ``packages``
    (the requested specifiers).
"""

# ---------------------------------------------------------------------------
# Trajectory compaction for storage review prompts
# ---------------------------------------------------------------------------

# Read-only store operations whose results the librarian can (and should)
# re-derive live with its own tools. Write operations (add/update/delete/…)
# stay verbatim — their results are small and record what changed.
_STORE_READ_TOOL_RE = re.compile(
    r"^(?:FunctionManager|GuidanceManager)" r"_(?:search|filter|list|get)",
)
_TRAJ_SYSTEM_STUB_THRESHOLD = 2_000
_TRAJ_STORE_READ_STUB_THRESHOLD = 300
_TRAJ_TOOL_RESULT_HEAD = 4_000
_TRAJ_TOOL_RESULT_TAIL = 1_000


def _prepare_trajectory_for_storage_review(
    messages: list[dict] | None,
) -> list[dict]:
    """Compact a trajectory snapshot for a skill-librarian prompt.

    The librarian judges what the actor *did*, not what it was told it
    could do, and it queries the stores live with its own tools. Three
    rewrites keep the review prompt bounded without hiding decision
    signal:

    * Large system messages collapse to a one-line stub — the review
      prompt's own storage doctrine is authoritative. Small system
      messages (e.g. parent-chat context) stay verbatim.
    * Large results of store *reads* (FunctionManager / GuidanceManager
      search/filter/list/get, including results
      delivered through ``check_status_*`` placeholders) collapse to an
      entry count. The call and its arguments stay visible: "searched
      the store, found nothing, built it by hand" is exactly the signal
      that something is worth storing, and empty/short results stay
      verbatim for that reason.
    * Any other oversized tool result keeps its head and tail around an
      elision marker.

    Provider reasoning payloads (encrypted blobs, reasoning summaries) are
    dropped outright: the librarian judges visible actions and results,
    and an encrypted chain of thought is unreadable bulk in its prompt.
    """
    from unify.common._async_tool.messages import strip_reasoning_payloads

    prepared = make_messages_safe_for_context_dump(messages)
    for _msg in prepared:
        if isinstance(_msg, dict):
            strip_reasoning_payloads(_msg)

    name_by_call_id: dict[str, str] = {}
    for msg in prepared:
        for tc in msg.get("tool_calls") or []:
            if not isinstance(tc, dict):
                continue
            tc_id = tc.get("id")
            fn_name = (tc.get("function") or {}).get("name")
            if tc_id and fn_name:
                name_by_call_id[str(tc_id)] = str(fn_name)

    def _origin_tool_name(msg: dict) -> str | None:
        name = msg.get("name") or name_by_call_id.get(
            str(msg.get("tool_call_id") or ""),
        )
        if not name:
            return None
        name = str(name)
        # ``check_status_<call_id>`` delivers an async tool's real result;
        # resolve back to the originating tool for classification.
        if name.startswith("check_status_"):
            return name_by_call_id.get(name[len("check_status_") :], name)
        return name

    def _entry_count(content: str) -> str | None:
        try:
            parsed = json.loads(content)
        except (ValueError, TypeError):
            return None
        if isinstance(parsed, list):
            return f"{len(parsed)} entries"
        return None

    def _content_text(content: object) -> str | None:
        """Flatten message content to plain text for measurement/rewrites.

        Tool results arrive either as a plain string or as an OpenAI
        content-parts list (``[{"type": "text", "text": ...}, ...]``) —
        execute_code results use the latter. Rewrites always store back a
        plain string; that is fine for a serialized trajectory dump.
        """
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts: list[str] = []
            for part in content:
                if isinstance(part, dict):
                    text = part.get("text")
                    parts.append(
                        (
                            text
                            if isinstance(text, str)
                            else f"[{part.get('type') or 'non-text'} part]"
                        ),
                    )
                else:
                    parts.append(str(part))
            return "\n".join(parts)
        return None

    for msg in prepared:
        content = _content_text(msg.get("content"))
        if content is None:
            continue
        role = msg.get("role")
        if role == "system":
            if len(content) > _TRAJ_SYSTEM_STUB_THRESHOLD:
                msg["content"] = (
                    f"[System prompt omitted ({len(content):,} chars). The "
                    "storage doctrine in this review prompt is "
                    "authoritative.]"
                )
            continue
        if role != "tool":
            continue
        origin = _origin_tool_name(msg)
        if (
            origin
            and _STORE_READ_TOOL_RE.match(origin)
            and len(content) > _TRAJ_STORE_READ_STUB_THRESHOLD
        ):
            count = _entry_count(content)
            size = f"{len(content):,} chars"
            detail = f"{count}, {size}" if count else size
            msg["content"] = (
                f"[{origin} result omitted ({detail}). Query the stores "
                "directly with your tools; they are the source of truth.]"
            )
        elif len(content) > _TRAJ_TOOL_RESULT_HEAD + _TRAJ_TOOL_RESULT_TAIL:
            omitted = len(content) - _TRAJ_TOOL_RESULT_HEAD - _TRAJ_TOOL_RESULT_TAIL
            msg["content"] = (
                content[:_TRAJ_TOOL_RESULT_HEAD]
                + f"\n… [{omitted:,} chars omitted] …\n"
                + content[-_TRAJ_TOOL_RESULT_TAIL:]
            )
    return prepared


# ---------------------------------------------------------------------------
# Shared storage tool construction
# ---------------------------------------------------------------------------


def _build_storage_tools(
    *,
    actor: "CodeActActor",
) -> Dict[str, Callable]:
    """Build the tool dict shared by both post-processing and proactive storage loops.

    Tool docstrings deliberately stay static — per-run listings belong in the (volatile tail of the) system
    prompt so the serialized tool schemas are byte-identical across loops
    and stay prompt-cache-friendly.
    """
    fm = actor.function_manager
    gm = actor.guidance_manager

    storage_methods: list[Any] = [
        fm.search_functions,
        fm.filter_functions,
        fm.list_functions,
        fm.add_functions,
        fm.delete_function,
        fm.reconcile_dependencies,
        gm.search,
        gm.filter,
        gm.get_guidance,
        gm.add_guidance,
        gm.update_guidance,
        gm.delete_guidance,
        gm.reconcile_dependencies,
    ]

    # UNIFY_STORE_VERIFY: the review checks a function on a held-out task
    # before add_functions will store it; unset, the tools are as shipped.
    from unify.function_manager import store_verify

    if store_verify.enabled():
        storage_methods.append(fm.check_function)
    # UNIFY_FUNCTION_PATCH: the review can fix an entry in place by replacing
    # excerpts. Simulated managers have none.
    storage_methods.extend(
        method
        for method in (
            getattr(fm, "patch_function", None),
            getattr(gm, "patch_guidance", None),
        )
        if method is not None
    )
    # UNIFY_FUNCTION_CASES: the review can retire a recorded case that a
    # change no longer reproduces.
    if hasattr(fm, "retire_case"):
        storage_methods.append(fm.retire_case)

    tools: Dict[str, Callable] = {
        **methods_to_tool_dict(
            *storage_methods,
            include_class_name=True,
        ),
    }

    return tools


# ---------------------------------------------------------------------------
# Storage check: start a review loop and return its handle
# ---------------------------------------------------------------------------


# Stop reason of a persistent session that ended normally: its final storage review
# then reads the trajectory as one finished piece of work, not as an interrupted one.
SESSION_ENDED = "session ended"

# What a persistent session's result() returns when it is ended by a stop
# (unify/common/async_tool_loop.py), and what the review reads when the
# agent's last reply had no text.
_STOPPED_NOTICE = "processed stopped early, no result"
_EMPTY_REPLY = "(the agent's last reply had no text)"


def review_final_result(
    original_result: Any,
    *,
    last_reply: Optional[str],
    stop_reason: Optional[str],
    reply_at_outcome: Optional[str],
) -> str:
    """The "Final Result" a session's storage review reads.

    Pure: the live handle and a replay of a recorded session decide alike.
    ``original_result`` is what the session's task loop returned (for a
    persistent session ended by a stop, ``_STOPPED_NOTICE``); ``last_reply``
    the content of its latest ``response`` notification (None before the
    first); ``stop_reason`` the reason of the stop that ended it (None when
    nothing stopped it); ``reply_at_outcome`` the reply an outcome arrived
    after.

    It is the reply the outcome arrived after or, with no outcome, the last
    reply in place of the stop notice; otherwise the session's result.
    """
    result = str(original_result)
    if reply_at_outcome is not None:
        return reply_at_outcome or _EMPTY_REPLY
    if result == _STOPPED_NOTICE and last_reply is not None:
        return last_reply or _EMPTY_REPLY
    return result


# The largest admission verdict read; anything bigger is not a verdict.
_STORE_ADMISSION_MAX_BYTES = 65536


def _store_admission_path() -> str:
    """The verdict file named by ``UNIFY_STORE_ADMISSION``; empty when unset."""
    from unify.settings import SETTINGS

    return str(SETTINGS.UNIFY_STORE_ADMISSION or "").strip()


# ``UNIFY_STORE_ADMISSION=never``: a frozen library. Writes are withheld as
# for any admission-gated run, and no review is ever admitted, so no verdict
# file is read.
_STORE_ADMISSION_NEVER = "never"
_STORE_ADMISSION_NEVER_REASON = (
    "admission is never granted (UNIFY_STORE_ADMISSION=never: the library is "
    "frozen for this run)"
)


def _store_admission_never(path: Optional[str] = None) -> bool:
    """Whether ``UNIFY_STORE_ADMISSION`` says no review is ever admitted."""
    value = _store_admission_path() if path is None else path
    return value.strip().lower() == _STORE_ADMISSION_NEVER


def _load_store_admission(path: str) -> tuple[Optional[dict], str]:
    """The admission verdict object at *path*, or ``None`` and why there is none."""
    try:
        with open(path, "rb") as fh:
            raw = fh.read(_STORE_ADMISSION_MAX_BYTES + 1)
    except FileNotFoundError:
        return None, f"no admission verdict at {path}"
    except OSError as exc:
        return None, f"admission verdict unreadable: {type(exc).__name__}: {exc}"
    if len(raw) > _STORE_ADMISSION_MAX_BYTES:
        return None, "admission verdict larger than 64 KiB"
    try:
        verdict = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, ValueError) as exc:
        return None, f"admission verdict is not JSON: {type(exc).__name__}"
    if not isinstance(verdict, dict):
        return None, "admission verdict is not a JSON object"
    return verdict, ""


def _admission_why(verdict: dict) -> str:
    why = verdict.get("reason")
    return f" ({str(why)[:200]})" if why else ""


def _read_store_admission(path: str) -> tuple[bool, str]:
    """Whether an external check of the session's outcome admits its review.

    The file must hold a JSON object whose ``admit`` is ``true``. Every other
    state -- no file, one that cannot be read, is too large, is not JSON, is
    not an object, or has any other ``admit`` -- does not admit (fail-closed).
    Returns ``(admitted, reason)``; the reason names what was found.
    """
    verdict, failure = _load_store_admission(path)
    if verdict is None:
        return False, failure
    why = _admission_why(verdict)
    if verdict.get("admit") is True:
        return True, f"admitted{why}"
    return False, f"not admitted{why}"


def _storage_review_outcome_note(outcome: Optional[dict] = None) -> str:
    """The storage review's section on the session's checked outcome.

    Filled from the outcome the environment posted (:mod:`unify.outcome`).
    Empty when there is none, so the review is then the shipped text.
    """
    from unify import outcome as outcome_mod

    return outcome_mod.render(outcome)


# UNIFY_REVIEW_FRAMING=unified: the fork is the agent's own curation step.
_REVIEW_FORK_ROLE_UNIFIED = (
    "## Curating The Library\n\n"
    "The task above is finished. This is the curation step that follows "
    "it: you did this work, so turn what it taught you into library "
    "entries a future task can reuse. The conversation above -- what was "
    "asked, what you did and what came of it -- is the trajectory the "
    "rules below refer to.\n\n"
    "Your tool list is the one the task used, but only the function and "
    "guidance library tools work now; any other tool is refused. Library "
    "writes that were read-only during the task are available to you now.\n\n"
)

_REVIEW_CLOSING_UNIFIED = (
    "\n\n## Now\n\n"
    "Review the trajectory and store any reusable functions and "
    "compositional guidance, following the rules above. Then reply with a "
    "brief summary naming what changed (function names, guidance titles) "
    "or, if nothing qualified, why."
)


# What the fork's tools do, for a core session.
_REVIEW_FORK_TOOLS = (
    "Your tool list is the one the task used, but only the function and "
    "guidance library tools work now; any other tool is refused. Library "
    "writes that were read-only during the task are available to you now.\n\n"
)
_REVIEW_FORK_TOOLS_CORE = (
    "Your tool list is the one the task used. `execute_code` now runs each "
    "cell in the review's own sandbox, which holds only the `functions` and "
    "`guidance` libraries, awaited as in the task (`await "
    "functions.search(...)`, `await functions.add(...)`, `await "
    "guidance.add(...)`): the task's environment, files and variables are "
    "not in it, and stored functions are not run. Any other tool is refused. "
    "Library writes that were read-only during the task are available to you "
    "now.\n\n"
)

_REVIEW_FORK_MASK_RULE_CORE = (
    "the storage review runs only execute_code, in a sandbox that holds the "
    "function and guidance libraries; the task's other tools are not "
    "available to it"
)


def _review_fork_role(*, core: bool = False) -> str:
    role = _review_fork_role_shipped()
    if core:
        role = role.replace(_REVIEW_FORK_TOOLS, _REVIEW_FORK_TOOLS_CORE)
    return role


def _review_fork_role_shipped() -> str:
    return _REVIEW_FORK_ROLE_UNIFIED


_LATE_SESSION_MESSAGE_REFUSAL = (
    "The session has ended, so this message was not delivered: it would have "
    "reached the storage review, which does not take the session's messages. "
    "Start a new session to continue."
)

_REVIEW_FORK_MASK_RULE = (
    "the storage review can call only the function and guidance library "
    "tools; the task's other tools are not available to it"
)


def _review_fork_source(
    inner: Any,
    actor: "CodeActActor",
) -> tuple[Optional[dict], Optional[str]]:
    """What a forked storage review continues from, or why it cannot fork.

    Returns ``(source, None)`` for a fork, and ``(None, reason)`` when the
    review has to run standalone. The fork reuses the session's fixed tool
    list and needs its last request as recorded; it is refused when the
    session was compressed, when its history no longer starts with that request (something rewrote it), or
    when it ends with unanswered tool calls, which the review loop would try
    to run with its own tools.
    """
    from unify.common._async_tool import cache_discipline

    # The review stores through the list's execute_code, in a sandbox holding
    # only the libraries (the core surface).
    source, why = _session_fork_source(inner, actor)
    if source is None:
        return None, why
    why = core_surface.review_fork_refusal(
        cache_discipline.schema_names(source["tools"]),
    )
    if why is not None:
        return None, why
    return {**source, "core": True}, None


def _session_fork_source(
    inner: Any,
    actor: "CodeActActor",
) -> tuple[Optional[dict], Optional[str]]:
    """The session's conversation for a fork, or why it cannot be continued.

    The source holds the session's ``client``, its raw ``messages``, the
    ``tools`` and ``tool_choice`` of its last request, and ``sent_messages``:
    its history as that request sent it (system prompt first, preprocessed)
    followed by what came after, so a fork that sends them unchanged starts
    with the last request's bytes.
    """
    from unify.common._async_tool import cache_discipline
    from unify.common._async_tool.messages import find_unreplied_assistant_entries

    client = getattr(inner, "_client", None)
    if client is None:
        return None, "the session has no LLM client"
    if getattr(getattr(inner, "_compression", None), "count", 0):
        return None, "the session's history was compressed"
    last = cache_discipline.last_sent_request(client)
    if last is None or not last.get("messages"):
        return None, "the session recorded no request"
    if not last.get("tools"):
        return None, "the session's last request carried no tools"

    raw = copy.deepcopy(list(getattr(client, "messages", None) or []))
    as_sent = raw
    preprocess = getattr(actor, "_preprocess_msgs", None)
    if preprocess is not None:
        try:
            as_sent = preprocess(copy.deepcopy(raw)) or as_sent
        except Exception:
            return None, "the session's message preprocessor failed on its history"
    system = getattr(client, "system_message", None)
    if system and not (as_sent and as_sent[0].get("role") == "system"):
        as_sent = [{"role": "system", "content": system}, *as_sent]
    sent = last["messages"]

    def _bytes(messages: list) -> list[str]:
        return [json.dumps(m, default=str) for m in messages]

    if _bytes(as_sent[: len(sent)]) != _bytes(sent):
        return None, "the session's history changed after its last request"
    if find_unreplied_assistant_entries(types.SimpleNamespace(messages=raw)):
        return None, "the session ended with unanswered tool calls"
    return (
        {
            "client": client,
            "messages": raw,
            "sent_messages": as_sent,
            "tools": last["tools"],
            "tool_choice": last.get("tool_choice"),
        },
        None,
    )


def _start_storage_review_fork(
    *,
    fork_source: dict,
    actor: "CodeActActor",
    tools: Dict[str, Callable],
    message: str,
    parent_lineage: list[str] | None,
    mask_rule: str = _REVIEW_FORK_MASK_RULE,
) -> "AsyncToolLoopHandle":
    """Start the storage review as a fork of the session's conversation.

    Its first request is the session's system prompt, messages, last tools
    and tool choice -- as the session sent them -- plus *message*, so the
    provider serves all but that message from the session's cache. The loop
    adds nothing else: no runtime-context header, no parent context, no
    compression. Library tools the list advertises run as the review's own;
    everything else in the list is refused by rule.
    """
    from unify.common._async_tool.propagation_mode import ChatContextPropagation

    client = fork_llm_client(
        fork_source["client"],
        origin="StorageCheck",
        purpose="planning",
        messages=fork_source["messages"],
    )
    # The review is harness-internal (its message carries the checked
    # outcome): transcribed where no cell can read it.
    transcripts.mark_internal(client)
    first_choice = fork_source.get("tool_choice")
    first_choice = first_choice if isinstance(first_choice, str) else "auto"
    review_tools = dict(tools)

    opts: dict = {"mask_rule": mask_rule}

    def _review_policy(step: int, visible: Dict[str, Any]):
        return (
            first_choice if step == 0 else "auto",
            visible,
            dict(opts),
        )

    return start_async_tool_loop(
        client=client,
        message=message,
        tools=review_tools,
        loop_id="StorageCheck(CodeActActor.act)",
        parent_lineage=parent_lineage,
        tool_policy=_review_policy,
        propagate_chat_context=ChatContextPropagation.NEVER,
        caller_description="",
        preprocess_msgs=getattr(actor, "_preprocess_msgs", None),
        prompt_caching=getattr(actor, "_prompt_caching", None),
        enable_compression=False,
        fixed_tools_schema=fork_source["tools"],
    )


def _close_with_result(handle: Any, close: Callable[[], Awaitable[None]]) -> None:
    """Run *close* once *handle*'s result has been awaited, however it ends."""
    original = handle.result

    async def _result_then_close() -> Any:
        try:
            return await original()
        finally:
            try:
                await close()
            except Exception as exc:  # cleanup never masks the review's result
                logger.warning(
                    f"StorageCheck sandbox close failed: {type(exc).__name__}: {exc}",
                )

    handle.result = _result_then_close


def _start_storage_check_loop(
    *,
    trajectory: list[dict],
    actor: "CodeActActor",
    original_result: str,
    parent_lineage: list[str] | None = None,
    stop_reason: str | None = None,
    proactive_summaries: list[str] | None = None,
    live_session: bool = False,
    fork_source: dict | None = None,
    outcome: dict | None = None,
    origin_note: str = "",
) -> "AsyncToolLoopHandle | None":
    """Start a loop that reviews a completed trajectory for reusable knowledge.

    *outcome* is the session's checked outcome (:mod:`unify.outcome`), shown in
    its own section before the final result.

    With *fork_source* (see :func:`_review_fork_source`) the review is a fork
    of the session's own conversation, and the rulebook arrives as one
    appended user message instead of a system prompt around a trajectory
    dump.

    With ``live_session=True`` the trajectory belongs to a persistent
    session that is still running: the review covers the turns completed
    so far, ``original_result`` is the latest turn's response, and the
    librarian's summary is delivered back into the live session as a
    background note.

    The loop maintains two complementary stores:

    * **FunctionManager** — stores the *what*: concrete, reusable function
      implementations (the building blocks).
    * **GuidanceManager** — stores the *how*: high-level guidance on
      composing multiple functions together to accomplish broader tasks
      (the recipes / playbooks).

    Both are required. Returns ``None`` when either is missing.
    """
    fm = actor.function_manager
    gm = actor.guidance_manager
    if fm is None or gm is None:
        return None
    generalise_note = ""
    # UNIFY_STORE_FROM_SESSION: a function the session ran may be stored by name.
    from unify.function_manager import session_source

    generalise_note += session_source.review_note()
    tools = _build_storage_tools(actor=actor)
    outcome_note = _storage_review_outcome_note(outcome)

    # ── Build prompt ──────────────────────────────────────────────────

    trajectory_json = json.dumps(
        _prepare_trajectory_for_storage_review(trajectory),
        default=str,
    )

    # ── Proactive storage awareness ───────────────────────────────────
    proactive_storage_section = ""
    if proactive_summaries:
        summaries_text = "\n\n".join(
            f"**Proactive pass {i + 1}:**\n{s}"
            for i, s in enumerate(proactive_summaries)
        )
        proactive_storage_section = (
            "## Proactive Storage Already Performed\n\n"
            "The executing agent proactively triggered skill storage during "
            "this run via the `store_skills` tool. Below are the summaries "
            "from each proactive storage pass:\n\n"
            f"{summaries_text}\n\n"
            "Check the function and guidance stores to confirm "
            "what was already added. Do not duplicate existing entries. "
            "Focus on any additional reusable patterns — especially from "
            "sections of the trajectory *after* the last `store_skills` "
            "call — that the proactive passes may have missed.\n\n"
        )

    instructions = _storage_base_instructions()
    if proactive_summaries:
        instructions = (
            "## Instructions\n\n"
            "1. Skill storage was proactively triggered during this run. "
            "Start by reviewing the proactive storage summaries below and "
            "checking the function and guidance stores to see "
            "what was already added.\n"
            "2. Search the existing stores to confirm exactly what was stored "
            "(use the search/filter tools for each store).\n"
            "3. Review the full trajectory — especially sections after the "
            "last `store_skills` call — for any additional reusable patterns "
            "the proactive passes may have missed.\n"
            "4. Do not duplicate entries that already exist. Only add, update, "
            "or merge if there is genuinely new value.\n"
            "5. When done (or if there is nothing more to add), respond "
            "with a brief summary of what you did (or that nothing additional "
            "was needed)."
        )

    stop_context_section = ""
    if stop_reason == SESSION_ENDED:
        stop_context_section = (
            "## Session End\n\n"
            "This persistent session has ended normally: the work it was opened "
            "for is finished. The trajectory below is the complete record of the "
            "session, so review it as one finished piece of work and store what "
            "will be reusable in future sessions.\n\n"
        )
    elif stop_reason:
        stop_context_section = (
            "## Session Termination Context\n\n"
            "This session was explicitly stopped by the user. The stop reason "
            "provides important signal about whether the user intended the "
            "work to be saved:\n\n"
            f"> {stop_reason}\n\n"
            "Weigh this context when deciding what to store. If the reason "
            "indicates the user wanted the procedure remembered or saved, that "
            "is a strong positive signal — look for reusable patterns in the "
            "trajectory. If the reason indicates cancellation or abandonment, "
            "the trajectory is less likely to contain patterns worth "
            "persisting, though genuinely reusable sub-patterns may still "
            "be worth storing.\n\n"
        )

    live_session_section = ""
    if live_session:
        live_session_section = (
            "## Live Session Turn Review\n\n"
            "The trajectory below is a persistent interactive session that "
            "is still running; the agent has just completed a request turn "
            "and is waiting for the next instruction. You are reviewing "
            "mid-session — earlier turns are included as context and may "
            "already have been reviewed (prior passes appear under "
            "'Proactive Storage Already Performed').\n\n"
            "- Focus on what the latest turn(s) added since the last "
            "review pass.\n"
            "- Steady state is cheap: when the latest turn(s) only "
            "re-executed already-stored procedures and the requester added "
            "no new requirement, amendment or correction, there is nothing "
            "to do — say so in one sentence and finish immediately, "
            "without searching the stores first.\n"
            "- Prefer updating an existing stored entry over adding a "
            "near-duplicate: when this session already stored the "
            "procedure and a later turn refined its spec, apply the "
            "refinement with `FunctionManager_add_functions` "
            "(`overwrite=True`).\n"
            "- Your final summary is delivered to the live session as a "
            "background note. Make it actionable: name what changed "
            "(function names, numeric ids, calling conventions) so the "
            "session can execute stored functions on the next request "
            "rather than re-deriving procedures.\n\n"
        )

    role_line = (
        (
            "You are the agent running the persistent interactive "
            "session below, and you have just completed a request turn. "
            "This is the curation step that follows it: turn what the "
            "latest work taught you into library entries a future "
            "request can reuse.\n\n"
        )
        if live_session
        else (
            "You are the agent that just completed the task below. This "
            "is the curation step that follows it: turn what the work "
            "taught you into library entries a future task can reuse."
            "\n\n"
        )
    )
    trajectory_header = (
        "## Session Trajectory So Far\n\n"
        if live_session
        else "## Completed Trajectory\n\n"
    )
    result_header = (
        "## Latest Turn Response\n\n" if live_session else "## Final Result\n\n"
    )

    if fork_source is not None and fork_source.get("core"):
        # A core session: the same message, naming the libraries as
        # the sandbox does; the review stores through execute_code, whose
        # cells run in a sandbox of their own. The final result is the
        # session's, unchanged.
        review_sandbox = core_surface.ReviewSandbox(
            actor,
            core_surface.review_policy(),
        )
        # The outcome section as the rulebook carries it, renamed with it.
        from unify import outcome as outcome_mod

        outcome_mod.remember(core_surface.python_names(outcome_note))
        rulebook = core_surface.python_names(
            f"{_review_fork_role(core=True)}"
            f"{_storage_doctrine_sections()}"
            f"{instructions}"
            "\n\n"
            f"{stop_context_section}"
            f"{proactive_storage_section}"
            f"{generalise_note}"
            f"{origin_note}"
            f"{outcome_note}"
            f"{result_header}",
        )
        closing = core_surface.python_names(
            _REVIEW_CLOSING_UNIFIED,
        )
        handle = _start_storage_review_fork(
            fork_source=fork_source,
            actor=actor,
            tools={"execute_code": review_sandbox.execute_code},
            message=f"{rulebook}{original_result}{closing}",
            parent_lineage=parent_lineage,
            mask_rule=_REVIEW_FORK_MASK_RULE_CORE,
        )
        _close_with_result(handle, review_sandbox.close)
        return handle

    if fork_source is not None:
        # The conversation is the trajectory. The completed-tool and inner
        # storage sections name tools the fork's list does not carry.
        return _start_storage_review_fork(
            fork_source=fork_source,
            actor=actor,
            tools=tools,
            message=(
                f"{_review_fork_role()}"
                f"{_storage_doctrine_sections()}"
                f"{instructions}"
                "\n\n"
                f"{stop_context_section}"
                f"{proactive_storage_section}"
                f"{generalise_note}"
                f"{origin_note}"
                f"{outcome_note}"
                f"{result_header}"
                f"{original_result}"
                f"{_REVIEW_CLOSING_UNIFIED}"
            ),
            parent_lineage=parent_lineage,
        )

    # Static doctrine first, volatile trajectory last: every storage loop
    # shares the same byte-identical prefix (role + doctrine + instructions),
    # so provider prompt caching only pays cold tokens for the per-run tail.
    system_prompt = (
        f"{role_line}"
        f"{_storage_doctrine_sections()}"
        f"{instructions}"
        "\n\n"
        f"{stop_context_section}"
        f"{live_session_section}"
        f"{proactive_storage_section}"
        f"{generalise_note}"
        f"{trajectory_header}"
        f"{trajectory_json}\n\n"
        f"{origin_note}"
        f"{outcome_note}"
        f"{result_header}"
        f"{original_result}"
    )

    client = _storage_review_client(actor, origin="StorageCheck")
    client.set_system_message(system_prompt)
    # The review is harness-internal (its prompt carries the checked
    # outcome): transcribed where no cell can read it.
    transcripts.mark_internal(client)

    return start_async_tool_loop(
        client=client,
        message=(
            "Review the trajectory and store any reusable functions and "
            "compositional guidance."
        ),
        tools=tools,
        loop_id="StorageCheck(CodeActActor.act)",
        parent_lineage=parent_lineage,
    )


# ---------------------------------------------------------------------------
# Proactive storage: on-demand storage loop triggered from the doing loop
# ---------------------------------------------------------------------------


def _start_proactive_storage_loop(
    *,
    trajectory: list[dict],
    actor: "CodeActActor",
    request: str,
    parent_lineage: list[str] | None = None,
) -> "AsyncToolLoopHandle | None":
    """Start an on-demand storage review loop triggered mid-flight by the doing loop.

    Shares the same tool set and core prompt sections as the post-processing
    ``_start_storage_check_loop``, but uses a distinct prompt framing:
    the trajectory is partial (task still in progress), there is no final
    result, and the ``request`` parameter focuses the reviewer on specific
    skills worth storing.

    Returns ``None`` when either FunctionManager or GuidanceManager is
    missing.
    """
    fm = actor.function_manager
    gm = actor.guidance_manager
    if fm is None or gm is None:
        return None

    tools = _build_storage_tools(actor=actor)

    # ── Build prompt ──────────────────────────────────────────────────

    trajectory_json = json.dumps(
        _prepare_trajectory_for_storage_review(trajectory),
        default=str,
    )

    instructions = (
        "## Instructions\n\n"
        "1. Review the trajectory so far, focusing on the storage request.\n"
        "2. Search the existing stores to understand what already exists "
        "(use the search/filter tools for each store).\n"
        "3. Decide what actions (if any) would improve the library based on "
        "the requested skill(s). Prefer a clean, non-redundant library over "
        "a large one.\n"
        "4. When done (or if there is nothing worth storing), respond "
        "with a brief, concrete summary of what you stored (function names, "
        "guidance titles) or that nothing was needed. "
        "This summary will be visible to both the executing agent and a "
        "follow-up storage review, so be specific."
    )

    # Static doctrine first, volatile trajectory last — same prompt-cache
    # prefix as the post-run storage check.
    proactive_role = (
        "You are the agent executing the task below, and you asked to "
        "store skills before finishing it. This is that curation step: "
        "store the requested skill(s) for future reuse.\n\n"
    )
    system_prompt = (
        f"{proactive_role}"
        f"{_STORAGE_WHAT_CAN_BE_STORED}"
        f"{_storage_environment_note()}"
        f"{_STORAGE_TWO_STORES}"
        f"{_storage_compose_note()}"
        f"{_storage_update_first_note()}"
        f"{_STORAGE_SUB_AGENT_PATTERNS}"
        f"{instructions}"
        "\n\n"
        "## Storage Request\n\n"
        f"{request}\n\n"
        "## Trajectory So Far\n\n"
        f"{trajectory_json}"
    )

    client = _storage_review_client(actor, origin="ProactiveStorage")
    client.set_system_message(system_prompt)

    return start_async_tool_loop(
        client=client,
        message=(
            f"The executing agent has proactively requested skill storage: "
            f"{request!r}. Review the trajectory so far and store the "
            f"relevant functions and guidance."
        ),
        tools=tools,
        loop_id="ProactiveStorage(CodeActActor.act)",
        parent_lineage=parent_lineage,
    )


class _StorageCheckHandle(ToolLoopHandle):
    """Wraps an inner handle and runs a storage check after task completion.

    Lifecycle phases:

    * **task** -- the inner tool loop is running.  ``submit``, ``stop`` and
      ``cancel_request`` forward to the inner handle.  Notifications from
      the inner handle are relayed to consumers.
    * **storage** -- the task has completed.  ``result()`` has already
      resolved with the original task result.  A second loop reviews the
      trajectory for reusable skills.  The handle remains live: ``submit``
      and ``stop`` operate on the storage loop, and ``done()`` returns
      ``False``.
    * **done** -- both phases have completed (or were stopped/skipped).
      ``done()`` returns ``True``.

    ``result()`` resolves at the end of Phase 1 — callers get the task
    result without waiting for storage.  ``done()`` reflects full
    lifecycle completion (including storage).  This means nested actor
    loops propagate results immediately while storage runs concurrently
    in the background.
    """

    def __init__(
        self,
        *,
        inner: "AsyncToolLoopHandle",
        actor: "CodeActActor",
        meter: Optional[RunMeter] = None,
        turn_reviews_enabled: bool = False,
        persist: bool = False,
    ) -> None:
        self._inner = inner
        self._actor = actor
        self._meter = meter
        self._persist = bool(persist)
        self._notification_q: asyncio.Queue[dict] = asyncio.Queue()
        self._task_done_event = asyncio.Event()
        self._completion_event = asyncio.Event()
        self._original_result: Optional[str] = None
        self._task_failure: Optional[BaseException] = None
        self._storage_handle: Optional["AsyncToolLoopHandle"] = None
        self._phase: str = "task"  # "task" | "storage" | "done"
        self._stopped: bool = False
        self._stop_reason: Optional[str] = None
        self._active_relay: Optional[asyncio.Task] = None

        # Optional turn-boundary reviews for persistent sessions (off unless
        # UNIFY_TURN_STORAGE_REVIEWS is set). A persist=True loop never
        # self-completes, so by default distillation happens once, in Phase 2,
        # when the session ends: one review of the whole trajectory. A session
        # that lives long enough for that to be too late can opt into a
        # mid-session review at each completed turn that ran tools; its summary
        # is recorded for the final review and delivered back into the live
        # loop as a transcript note.
        self._turn_reviews_enabled = bool(turn_reviews_enabled)
        self._turn_review_task: Optional[asyncio.Task] = None
        self._turn_review_handle: Optional["AsyncToolLoopHandle"] = None
        self._turn_review_rerun: bool = False
        self._latest_turn_response: str = ""
        self._reviewed_tool_msg_count: int = 0

        # The environment's checked outcome for this session, posted under
        # ``outcome_session_id`` (unify/outcome.py), and the agent's replies
        # it is read against.
        self.outcome_session_id: str = uuid.uuid4().hex
        self._outcome: Optional[dict] = None
        self._last_reply: Optional[str] = None
        self._reply_at_outcome: Optional[str] = None
        from unify import outcome as outcome_mod

        outcome_mod.register(self.outcome_session_id, self)

        # Start the two-phase lifecycle manager.
        self._lifecycle_task = asyncio.create_task(self._run_lifecycle())

    @property
    def run_stats(self) -> dict[str, Any]:
        """Token accounting for the execution row (planning tokens for an agentic run)."""
        stats = (
            {} if self._meter is None else {"tokens": self._meter.snapshot()["tokens"]}
        )
        # UNIFY_REPLY_CHANNEL=code+text: the turns a cell's reply() ended.
        if cell_reply.enabled():
            runtime_state = getattr(self._inner, "_runtime_state", None)
            stats.update(cell_reply.run_stats(runtime_state))
        # UNIFY_LOOP_STOP: the requests ended for making no progress.
        from unify.common._async_tool import loop_stop

        if loop_stop.enabled():
            runtime_state = getattr(self._inner, "_runtime_state", None)
            stats.update(loop_stop.run_stats(runtime_state))
        return stats

    # ── Internal helpers ──────────────────────────────────────────────

    @property
    def _active_handle(self) -> Optional["AsyncToolLoopHandle"]:
        """The currently active inner handle the caller's calls go to."""
        if self._phase == "task":
            return self._inner
        if self._phase == "storage":
            return self._storage_handle
        return None

    async def _relay_notifications_from(
        self,
        source: "ToolLoopHandle",
    ) -> None:
        """Forward notifications from *source* into our queue until cancelled.

        ``type="response"`` notifications are the persist-mode turn
        boundary — the loop has finished a request and re-entered its wait
        state — so they are also the trigger for mid-session storage
        reviews when those are enabled.
        """
        try:
            while True:
                notif = await source.next_notification()
                if isinstance(notif, dict) and notif.get("type") == "response":
                    self._last_reply = str(notif.get("content") or "")
                await self._notification_q.put(notif)
                if (
                    self._turn_reviews_enabled
                    and isinstance(notif, dict)
                    and notif.get("type") == "response"
                ):
                    self._note_turn_boundary(str(notif.get("content") or ""))
        except asyncio.CancelledError:
            pass
        except Exception:
            pass

    def receive_outcome(self, outcome: dict) -> None:
        """Take the session's checked outcome (see :func:`unify.outcome.post`).

        The agent's latest reply is kept with it: an environment posts the
        outcome once the task is over and before any closing message, so that
        reply is the one the task ended on, and the review reads it as the
        final result. The latest outcome wins; once the session has ended it
        is too late and the outcome is refused.
        """
        from unify import outcome as outcome_mod

        if self._task_done_event.is_set():
            raise outcome_mod.OutcomeError(
                "the session has already ended; its review has started",
            )
        self._outcome = dict(outcome)
        self._reply_at_outcome = self._last_reply
        logger.info(
            "StorageCheck outcome received: solved="
            f"{outcome.get('solved')} score={outcome.get('score')} "
            f"source={outcome.get('source')}",
        )

    def _review_final_result(self) -> str:
        """The "Final Result" the storage review reads (see :func:`review_final_result`)."""
        return review_final_result(
            self._original_result,
            last_reply=self._last_reply,
            stop_reason=self._stop_reason,
            reply_at_outcome=self._reply_at_outcome,
        )

    def _note_turn_boundary(self, latest_response: str) -> None:
        """Schedule a mid-session storage review for a completed turn.

        At most one review runs at a time; a boundary that arrives while
        one is in flight coalesces into a single re-run against the
        then-current trajectory.
        """
        if self._phase != "task":
            return
        self._latest_turn_response = latest_response
        if self._turn_review_task is not None and not self._turn_review_task.done():
            self._turn_review_rerun = True
            return
        self._turn_review_task = asyncio.create_task(self._run_turn_reviews())

    @staticmethod
    def _tool_activity_count(messages: list) -> int:
        """Completed tool results in the transcript — the 'work happened' signal."""
        return sum(
            1 for m in messages if isinstance(m, dict) and m.get("role") == "tool"
        )

    def _snapshot_inner_trajectory(self) -> list[dict]:
        try:
            client = getattr(self._inner, "_client", None)
            if client is not None:
                return make_messages_safe_for_context_dump(
                    list(getattr(client, "messages", []) or []),
                )
        except Exception:
            pass
        return []

    async def _run_turn_reviews(self) -> None:
        """Run mid-session storage reviews until no boundary is pending.

        A turn with no new completed tool activity (pure conversation) is
        skipped — there is nothing new to distill. Compaction can shrink
        the transcript; the watermark follows it down so counting stays
        monotone against the live message list.
        """
        while True:
            self._turn_review_rerun = False
            trajectory = self._snapshot_inner_trajectory()
            tool_count = self._tool_activity_count(trajectory)
            if tool_count < self._reviewed_tool_msg_count:
                self._reviewed_tool_msg_count = tool_count
            if tool_count > self._reviewed_tool_msg_count:
                await self._run_one_turn_review(
                    trajectory,
                    tool_count,
                    reviewed_messages=len(trajectory),
                )
            if not self._turn_review_rerun:
                return

    async def _run_one_turn_review(
        self,
        trajectory: list[dict],
        tool_count: int,
        *,
        reviewed_messages: int,
    ) -> None:
        proactive_summaries: list[str] = []
        _ctx = _CURRENT_AGENT_CONTEXT.get(None)
        if _ctx is not None:
            proactive_summaries = list(_ctx.proactive_storage_summaries)

        _tr_suffix = _token_hex(2)
        _tr_call_id = new_call_id()
        _tr_parent = TOOL_LOOP_LINEAGE.get([])
        _tr_parent_lineage = list(_tr_parent) if isinstance(_tr_parent, list) else []
        _tr_hierarchy = [
            *_tr_parent_lineage,
            f"StorageCheck(CodeActActor.act)({_tr_suffix})",
        ]
        _tr_lineage_token = TOOL_LOOP_LINEAGE.set(_tr_hierarchy)
        _tr_suffix_token = _PENDING_LOOP_SUFFIX.set(_tr_suffix)
        try:
            await publish_manager_method_event(
                _tr_call_id,
                "CodeActActor",
                "StorageCheck",
                phase="incoming",
                display_label=_DEFAULT_STORAGE_REVIEW_LABEL,
                hierarchy=_tr_hierarchy,
                instructions=_DEFAULT_STORAGE_REVIEW_INSTRUCTIONS,
            )
            storage_handle = _start_storage_check_loop(
                trajectory=trajectory,
                actor=self._actor,
                original_result=self._latest_turn_response,
                parent_lineage=_tr_parent_lineage,
                proactive_summaries=proactive_summaries or None,
                live_session=True,
            )
            if storage_handle is None:
                return
            self._turn_review_handle = storage_handle
            try:
                summary = await storage_handle.result()
            except Exception as exc:
                logger.warning(
                    f"Turn StorageCheck failed: {type(exc).__name__}: {exc}",
                )
                await self._notification_q.put(
                    {
                        "type": "turn_storage_review_complete",
                        "message": (
                            f"StorageCheck failed: {type(exc).__name__}: {exc}"
                        ),
                        "success": False,
                    },
                )
                return
            finally:
                self._turn_review_handle = None

            self._reviewed_tool_msg_count = tool_count
            if _ctx is not None:
                _ctx.proactive_storage_summaries.append(summary)
            await self._notification_q.put(
                {
                    "type": "turn_storage_review_complete",
                    "message": summary,
                    "success": True,
                },
            )
            # Leave the librarian's summary in the live session's transcript
            # so the next request can execute what was stored instead of
            # re-deriving the procedure, and let the reviewed turns shed
            # their raw tool payloads — the review is the checkpoint that
            # makes them safe to compact. Both are transcript-only: no LLM
            # turn fires.
            queue = getattr(self._inner, "_queue", None)
            if queue is not None:
                queue.put_nowait(
                    {
                        "_transcript_note": {
                            "text": (
                                "[background skill consolidation — automated "
                                "note, not a user message]\n"
                                f"{summary}"
                            ),
                        },
                    },
                )
                queue.put_nowait(
                    {
                        "_compact_transcript": {
                            "reviewed_messages": reviewed_messages,
                        },
                    },
                )
        finally:
            await publish_manager_method_event(
                _tr_call_id,
                "CodeActActor",
                "StorageCheck",
                phase="outgoing",
                display_label=_DEFAULT_STORAGE_REVIEW_LABEL,
                hierarchy=_tr_hierarchy,
            )
            TOOL_LOOP_LINEAGE.reset(_tr_lineage_token)
            _PENDING_LOOP_SUFFIX.reset(_tr_suffix_token)

    async def abandon_storage_review(self, *, reason: str) -> None:
        """End the storage phase now, without waiting for the review to finish.

        Called when the actor the review depends on is closing. A review needs
        that actor's sandboxes to do anything useful,
        so once they are torn down the review cannot succeed -- it can only
        keep retrying against them. A review left running that way stays
        busy indefinitely, still issuing inference for a run already recorded
        as finished, because nothing connects the two lifetimes: the actor
        closes its pools and walks away from the review.

        The bound this gives a review is its actor's lifetime, not a clock. A
        review that is genuinely working is never interrupted -- the process
        that owns the run owns the actor, and only ends it when the run is
        done with it.

        ``stop`` is cooperative and a loop wedged in a retry against something
        already gone never notices, so the lifecycle task is cancelled after
        it. That is what actually ends the inference.
        """

        if self._completion_event.is_set():
            return
        turn_handle = self._turn_review_handle
        if turn_handle is not None:
            try:
                await turn_handle.stop(reason=reason)
            except Exception:
                pass
        turn_task = self._turn_review_task
        if turn_task is not None and not turn_task.done():
            turn_task.cancel()
            await asyncio.gather(turn_task, return_exceptions=True)
        handle = self._storage_handle
        if handle is not None:
            try:
                await handle.stop(reason=reason)
            except Exception:
                pass
        lifecycle = self._lifecycle_task
        if lifecycle is not None and not lifecycle.done():
            lifecycle.cancel()
            try:
                await lifecycle
            except (asyncio.CancelledError, Exception):
                pass
        # The lifecycle task owns these; setting them here covers the case
        # where it was cancelled before reaching its own ``finally``.
        self._phase = "done"
        self._task_done_event.set()
        self._completion_event.set()

    async def _cancel_relay(self) -> None:
        """Cancel the active notification relay task, if any."""
        relay = self._active_relay
        if relay is not None and not relay.done():
            relay.cancel()
            try:
                await relay
            except (asyncio.CancelledError, Exception):
                pass
        self._active_relay = None

    # ── Lifecycle ─────────────────────────────────────────────────────

    async def _run_lifecycle(self) -> None:
        """Manage the two-phase lifecycle: task -> storage check -> done."""
        if self._meter is not None:
            # The librarian's calls are planning tokens of this run.
            current_run_meter.set(self._meter)
        try:
            # ── Phase 1: task execution ───────────────────────────────
            self._active_relay = asyncio.create_task(
                self._relay_notifications_from(self._inner),
            )

            try:
                self._original_result = await self._inner.result()
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                # Kept so ``result()`` can re-raise. Flattening the failure into
                # a string here is what let a crashed run reach the scheduler
                # looking like a normal return, and be recorded as completed.
                self._task_failure = exc
                self._original_result = (
                    f"Error: inner task failed: {type(exc).__name__}: {exc}"
                )
                logger.error(
                    f"_StorageCheckHandle: inner result raised "
                    f"{type(exc).__name__}: {exc}",
                )
            await self._cancel_relay()
            self._task_done_event.set()

            # Snapshot the trajectory (client/messages are still valid after
            # result() returns -- cleanup only resets context vars and
            # releases the semaphore).
            trajectory: list[dict] = []
            try:
                client = getattr(self._inner, "_client", None)
                if client is not None:
                    trajectory = make_messages_safe_for_context_dump(
                        list(getattr(client, "messages", []) or []),
                    )
            except Exception:
                pass

            # ── Phase 2: storage check ────────────────────────────────
            # A crashed trajectory is not a source of reusable knowledge — the
            # librarian would derive functions, guidance, and claims from work
            # that never completed. A deliberate stop still reviews, because the
            # work up to that point was real.
            if self._task_failure is not None:
                return

            # With an admission file configured, the review runs only when an
            # external check of the session's outcome admits it, read now that
            # the session has ended; anything else skips it.
            admission_path = _store_admission_path()
            if admission_path and _store_admission_never(admission_path):
                logger.info(f"StorageCheck skipped: {_STORE_ADMISSION_NEVER_REASON}")
                await self._notification_q.put(
                    {
                        "type": "storage_review_skipped",
                        "message": _STORE_ADMISSION_NEVER_REASON,
                    },
                )
                return
            if admission_path:
                admitted, admission_reason = _read_store_admission(admission_path)
                if not admitted:
                    logger.info(f"StorageCheck skipped: {admission_reason}")
                    await self._notification_q.put(
                        {
                            "type": "storage_review_skipped",
                            "message": admission_reason,
                        },
                    )
                    return
                logger.info(f"StorageCheck {admission_reason}")

            self._phase = "storage"

            # A mid-session turn review still in flight finishes first: its
            # summary joins ``proactive_storage_summaries``, so the final
            # review below builds on it instead of running concurrently
            # against the same trajectory.
            turn_task = self._turn_review_task
            if turn_task is not None and not turn_task.done():
                await asyncio.gather(turn_task, return_exceptions=True)

            # UNIFY_REVIEW_GATE: one tool-free yes/no call decides whether the
            # review runs; a failed or unreadable gate runs it as shipped. While
            # the library holds nothing, the gate is not asked: the review runs.
            from unify.actor import review_gate

            ask_gate = True
            if review_gate.library_is_empty(
                _library_counts(
                    getattr(self._actor, "function_manager", None),
                    getattr(self._actor, "guidance_manager", None),
                ),
            ):
                logger.info(
                    "StorageCheck gate not asked: the library is empty; reviewing",
                )
                ask_gate = False
            if ask_gate:
                gate_outcome_note = _storage_review_outcome_note(self._outcome)
                decision = await review_gate.decide(
                    client_factory=lambda: _review_gate_client(
                        self._actor,
                        getattr(self._inner, "_client", None),
                    ),
                    trajectory=trajectory,
                    final_result=self._review_final_result(),
                    outcome_note=gate_outcome_note,
                )
                logger.info(
                    f"StorageCheck gate: review={decision.review} "
                    f"decided={decision.decided} ({decision.reason})",
                )
                if not decision.review:
                    await self._notification_q.put(
                        {
                            "type": "storage_review_skipped",
                            "message": f"review gate: {decision.reason}",
                        },
                    )
                    return

            _sc_suffix = _token_hex(2)
            _sc_call_id = new_call_id()
            _sc_parent = TOOL_LOOP_LINEAGE.get([])
            _sc_parent_lineage = (
                list(_sc_parent) if isinstance(_sc_parent, list) else []
            )
            _sc_hierarchy = [
                *_sc_parent_lineage,
                f"StorageCheck(CodeActActor.act)({_sc_suffix})",
            ]
            _sc_lineage_token = TOOL_LOOP_LINEAGE.set(_sc_hierarchy)
            _sc_suffix_token = _PENDING_LOOP_SUFFIX.set(_sc_suffix)
            # UNIFY_ESCAPE_DRIFT_CHECK: the review's writes are checked against
            # the session's own cells.
            _sc_drift_token = _escape_drift.enter(trajectory)

            try:
                review_display_label = _DEFAULT_STORAGE_REVIEW_LABEL
                review_instructions = _DEFAULT_STORAGE_REVIEW_INSTRUCTIONS
                await publish_manager_method_event(
                    _sc_call_id,
                    "CodeActActor",
                    "StorageCheck",
                    phase="incoming",
                    display_label=review_display_label,
                    hierarchy=_sc_hierarchy,
                    instructions=review_instructions,
                )

                proactive_summaries: list[str] = []
                try:
                    _ctx = _CURRENT_AGENT_CONTEXT.get(None)
                    if _ctx is not None:
                        proactive_summaries = list(
                            _ctx.proactive_storage_summaries,
                        )
                except Exception:
                    pass

                # Continue the session's own conversation when it can be
                # continued exactly; otherwise say why not.
                fork_source, fork_skipped = _review_fork_source(
                    self._inner,
                    self._actor,
                )
                if fork_skipped:
                    logger.info(
                        f"StorageCheck fork skipped: {fork_skipped}; running "
                        "the standalone review",
                    )

                # UNIFY_STORE_FROM_SESSION: the review's tools inherit the
                # session's cells, so a function it names is stored as it ran.
                from unify.function_manager import session_source as _session_source

                with _session_source.reviewing(trajectory):
                    storage_handle = _start_storage_check_loop(
                        trajectory=trajectory,
                        actor=self._actor,
                        original_result=self._review_final_result(),
                        parent_lineage=_sc_parent_lineage,
                        stop_reason=self._stop_reason,
                        proactive_summaries=proactive_summaries or None,
                        fork_source=fork_source,
                        outcome=self._outcome,
                    )

                if storage_handle is None:
                    await publish_manager_method_event(
                        _sc_call_id,
                        "CodeActActor",
                        "StorageCheck",
                        phase="outgoing",
                        display_label=review_display_label,
                        hierarchy=_sc_hierarchy,
                    )
                else:
                    self._storage_handle = storage_handle
                    storage_success = True
                    try:
                        storage_summary = await self._storage_handle.result()
                    except Exception as exc:
                        storage_success = False
                        storage_summary = (
                            f"StorageCheck failed: {type(exc).__name__}: {exc}"
                        )
                        logger.warning(
                            f"StorageCheck failed: {type(exc).__name__}: {exc}",
                        )
                    await publish_manager_method_event(
                        _sc_call_id,
                        "CodeActActor",
                        "StorageCheck",
                        phase="outgoing",
                        display_label=review_display_label,
                        hierarchy=_sc_hierarchy,
                    )

                    # ponytail: single-consumer signal — see event_handlers.py
                    # ActorNotification handler for the wake gate. Upgrade to
                    # a dedicated event class if a second consumer appears.
                    await self._notification_q.put(
                        {
                            "type": "storage_review_complete",
                            "message": storage_summary,
                            "success": storage_success,
                        },
                    )
            finally:
                _PENDING_LOOP_SUFFIX.reset(_sc_suffix_token)
                TOOL_LOOP_LINEAGE.reset(_sc_lineage_token)
                _escape_drift.leave(_sc_drift_token)

        except asyncio.CancelledError:
            pass
        except Exception:
            pass
        finally:
            self._phase = "done"
            self._task_done_event.set()
            self._completion_event.set()

    # ── The caller's side: phase-aware forwarding ─────────────────────

    async def submit(self, text: str) -> None:
        """Queue *text* for the active loop's next turn boundary."""
        if self._refuses_late_session_message():
            logger.info(
                "Message not delivered: the persistent session's task loop "
                "has ended, so the message has no session to go to, and the "
                "storage review does not take the session's messages "
                f"({len(text)} chars)",
            )
            await self._notification_q.put(
                {
                    "type": "interjection_refused",
                    "message": _LATE_SESSION_MESSAGE_REFUSAL,
                },
            )
            return None
        handle = self._active_handle
        if handle is not None:
            return await handle.submit(text)

    def _refuses_late_session_message(self) -> bool:
        """Whether a message arrives after a persistent session ended.

        A persistent session takes each follow-up as its next request. Once
        its task loop has ended (at a step or time limit, or by a stop), a
        follow-up has no session to go to, and forwarded to the storage
        review it is read there as a user message the review must answer, so
        it is refused instead. A handle that was not persistent forwards it to
        the review, which reads it at its next boundary.
        """
        return self._persist and self._task_done_event.is_set()

    async def stop(self, reason: Optional[str] = None, **kwargs) -> None:
        self._stopped = True
        self._stop_reason = reason
        handle = self._active_handle
        if handle is not None:
            await handle.stop(reason=reason, **kwargs)

    async def cancel_request(self, reason: Optional[str] = None) -> bool:
        # Only the task loop serves requests; the storage review that runs
        # after the session has none to cancel.
        if self._phase != "task":
            return False
        return await self._inner.cancel_request(reason)

    # ── Completion ────────────────────────────────────────────────────

    def done(self) -> bool:
        return self._completion_event.is_set()

    async def result(self) -> str:
        await self._task_done_event.wait()
        # An explicit stop is an outcome the caller asked for, so it reports as
        # one even if the actor then raised on its way down.
        if self._stopped and self._stop_reason:
            return (
                f"Task stopped as requested. Reason: {self._stop_reason}\n"
                f"Background skill storage is reviewing the completed work."
            )
        # Re-raise rather than returning the flattened error string: callers
        # decide success from whether this raises, so swallowing it here makes a
        # crashed run indistinguishable from one that returned normally.
        if self._task_failure is not None:
            raise self._task_failure
        return self._original_result or ""

    # ── Events ────────────────────────────────────────────────────────

    async def next_clarification(self) -> dict:
        handle = self._active_handle
        if handle is not None:
            return await handle.next_clarification()
        # Done: block forever (no more clarifications expected).
        await asyncio.Event().wait()
        return {}

    async def next_notification(self) -> dict:
        return await self._notification_q.get()

    async def answer_clarification(self, call_id: str, answer: str) -> None:
        handle = self._active_handle
        if handle is not None:
            return await handle.answer_clarification(call_id, answer)

    def get_history(self) -> list[dict]:
        return self._inner.get_history()


# ---------------------------------------------------------------------------
# Code synthesis helpers for execute_function
# ---------------------------------------------------------------------------


def _synthesize_python_call(
    *,
    function_name: str,
    call_kwargs: Dict[str, Any],
    function_manager: Optional["FunctionManager"] = None,
) -> str:
    """Build a Python code snippet that calls *function_name* with *call_kwargs*.

    This function only synthesises the **code string** — it does not handle
    dependency injection.  Transitive dependencies (both bare compositional
    functions and dotted environment namespaces like ``actor`` or
    ``primitives``) are injected into the sandbox namespace *before* this
    code runs, through a separate path:

    * When the LLM discovers a function via ``FunctionManager_search_functions``
      (or filter/list), the FM's ``_inject_callables_for_functions`` calls
      ``_inject_dependencies``, which resolves every entry in ``depends_on``
      and places the result into the sandbox's ``global_state``.
    * Environment namespaces (``actor``, ``primitives``, etc.) are also
      already present in the sandbox if the CodeActActor was constructed
      with the corresponding environments.

    So by the time this synthesised code executes, all names the function
    references — whether bare helpers or dotted environment calls — are
    already available in scope.

    Resolution order for the *function itself*:
    1. Emit a plain call expression.  The sandbox namespace already contains
       environment-injected callables and previously-discovered FM functions,
       so this is the common-case fast path.
    2. If the FunctionManager has a stored implementation, prepend it as a
       preamble (defining the function) so the call works even in a fresh
       stateless session where discovery hasn't run yet.

    The call expression is always the **last expression** so that
    ``PythonExecutionSession``'s REPL semantics return its value (including
    steerable handles from primitives).
    """
    kwargs_repr = repr(call_kwargs) if call_kwargs else "{}"
    # Determine whether the function is async by inspecting the stored impl.
    # Default to ``await`` — environment-injected callables (primitives,
    # manager methods) are async, and ``await`` on a sync return value
    # produces a clear ``TypeError`` rather than silently discarding a
    # coroutine.
    is_async = True
    preamble = ""

    if function_manager is not None:
        func_data = function_manager._get_function_data_by_name(name=function_name)
        if func_data is None and getattr(
            function_manager,
            "_include_primitives",
            False,
        ):
            func_data = function_manager._get_primitive_data_by_name(name=function_name)
        if func_data is None and getattr(
            function_manager,
            "_include_primitives",
            False,
        ):
            get_stored_primitive = getattr(
                function_manager,
                "_get_stored_primitive_data_by_name",
                None,
            )
            if callable(get_stored_primitive):
                func_data = get_stored_primitive(name=function_name)

        if func_data is not None:
            impl = func_data.get("implementation")
            if impl and isinstance(impl, str) and impl.strip():
                is_async = "async def" in impl
                preamble = impl + "\n\n"
            elif func_data.get("is_primitive"):
                is_async = True

    call_expr = f"{'await ' if is_async else ''}{function_name}(**{kwargs_repr})"
    return f"{preamble}{call_expr}"


async def _end_agents_request(
    agents_binding,
    *,
    reply: Optional[str] = None,
    reason: str = "",
) -> None:
    """End the main agent's request in its agent record.

    With a reply, the reply is recorded and running helpers are stopped;
    without one, the helpers are stopped with ``reason``. Only the main agent
    ends a request, and a failure here never costs the caller its result.
    """
    if agents_binding is None or agents_binding.name != "root":
        return
    try:
        if reply is not None:
            await agents_binding.pool.finish_request(reply)
        else:
            await agents_binding.pool.close(reason)
    except Exception:
        logger.warning("could not end the request in the agent record", exc_info=True)


class CodeActActor(BaseCodeActActor):
    """
    An actor that uses a conversational tool loop and a stateful code execution
    sandbox to accomplish tasks. It acts as a baseline for code-centric agents.
    """

    def __init__(
        self,
        *,
        environments: Optional[list["BaseEnvironment"]] = None,
        function_manager: Optional["FunctionManager"] = None,
        guidance_manager: Optional["GuidanceManager"] = None,
        can_compose: object = _UNSET,
        can_store: object = _UNSET,
        timeout: object = _UNSET,
        model: object = _UNSET,
        preprocess_msgs: Optional[Callable[[list[dict]], list[dict]]] = None,
        prompt_caching: object = _UNSET,
        guidelines: object = _UNSET,
        tool_policy: Union[ToolPolicyFn, None, object] = _USE_DEFAULT,
    ):
        """
        Initializes the CodeActActor.

        Args:
            environments: List of execution environments to install. Each environment
                injects a namespace into the sandbox (e.g. ``primitives``,
                ``primitives.actor``). Pass ``None`` or ``[]``
                for a bare actor with no environments.
            function_manager: Manages a library of reusable functions. Exposes read-only tools
                (list_functions, search_functions, filter_functions) to the LLM.
                The LLM can call these tools to discover and retrieve reusable function implementations.
            guidance_manager: Manages high-level guidance entries that describe *how* to
                compose functions together for tasks. Exposes read/write tools in the
                post-completion storage check loop alongside FunctionManager tools.
            can_compose: Whether the LLM can write and execute arbitrary code via
                ``execute_code``. Set to False for function-execution-only mode.
            can_store: Whether a post-completion review loop should run to
                identify and store reusable functions and guidance from the
                trajectory. Storage is always deferred to a dedicated second
                loop after the main task completes — the main loop never
                exposes storage tools.
            timeout: Maximum seconds for individual code execution in sessions.
            model: Optional LLM model identifier. If None, uses the assistant's
                default model when set, otherwise SETTINGS.UNIFY_MODEL.
            preprocess_msgs: Optional callback to modify messages before each LLM call.
                Receives a list of message dicts and returns a modified list.
                Useful for pruning old messages, adding context, or transforming content.
            prompt_caching: Optional list of cache targets (e.g. ["system", "messages"]).
                Enables Anthropic prompt caching for the specified components to reduce
                costs and latency. Valid values: "tools", "system", "messages".
            guidelines: Persistent behavioral guidelines applied to every ``act()``
                invocation.  Per-invocation ``guidelines`` passed to ``act()`` are
                appended after these, so the constructor value acts as a baseline
                and ``act()`` adds task-specific refinements on top.
            tool_policy: Controls per-turn dynamic tool filtering and tool-choice mode.
                - ``_USE_DEFAULT`` (default): the static filters only, as
                  ``None``, and the prompt leaves the library searches to the
                  model.
                - A custom ``ToolPolicyFn`` callable: receives ``(step, tools)`` and
                  returns ``(mode, filtered_tools)``.  Static filters (``can_compose``,
                  ``can_store``, etc.) are always applied before the custom policy sees
                  the tools.
                - ``None``: no dynamic policy; only the static ``can_compose`` /
                  ``can_store`` filters apply.
        """
        super().__init__(
            environments=environments or [],
            function_manager=function_manager,
            guidance_manager=guidance_manager,
        )

        can_compose = can_compose if can_compose is not _UNSET else True
        can_store = can_store if can_store is not _UNSET else True
        timeout = timeout if timeout is not _UNSET else 3600.0
        model = model if model is not _UNSET else None
        prompt_caching = (
            prompt_caching
            if prompt_caching is not _UNSET
            else ("system", "tools", "messages")
        )
        guidelines = guidelines if guidelines is not _UNSET else None
        self._base_guidelines = guidelines

        # Collect function_ids from all environments, split by context, and set
        # the discovery exclusions on the FunctionManager via setters. A stored
        # function an environment documents in the prompt is excluded so it
        # does not appear twice. A primitive stays discoverable only where an
        # environment provides it without documenting it: the sandbox holds
        # exactly the primitives its environments inject, so any other one
        # cannot be called from this actor. We update in-place rather than
        # replacing the FM instance so that callers who pass a custom FM (e.g.,
        # SimulatedFunctionManager) keep their instance intact.
        if self.function_manager is not None:
            _searchable_primitive: set[int] = set()
            _excl_compositional: set[int] = set()
            for env in self.environments.values():
                # When an environment declares `prompt_documented_names`, only
                # that subset is documented — undocumented callables must stay
                # searchable.
                _documented = getattr(env, "prompt_documented_names", None)
                for tool_name, tool_meta in env.get_tools().items():
                    if tool_meta.function_id is None:
                        continue
                    if _documented is not None and tool_name not in _documented:
                        if tool_meta.function_context == "primitive":
                            _searchable_primitive.add(tool_meta.function_id)
                        continue
                    if tool_meta.function_context == "compositional":
                        _excl_compositional.add(tool_meta.function_id)

            _excl_primitive = {
                int(row["function_id"])
                for row in get_registry().collect_primitives().values()
            } - _searchable_primitive
            if _excl_primitive:
                self.function_manager.exclude_primitive_ids = frozenset(
                    _excl_primitive,
                )
            if _excl_compositional:
                self.function_manager.exclude_compositional_ids = frozenset(
                    _excl_compositional,
                )

        self._session_executor = SessionExecutor(
            environments=self.environments,
            timeout=timeout,
        )

        # Session name registry: name -> session_id
        self._session_names: Dict[str, SessionKey] = {}
        # Reverse map: session_id -> set(names)
        self._session_names_rev: Dict[SessionKey, set[str]] = {}
        # Actor-level session cap for this actor instance.
        self._max_sessions_total: int = 20
        self._next_session_id: int = 1
        # Storage reviews started by this actor and not yet finished, so
        # ``close()`` can end them rather than leave them running against
        # pools it is about to tear down.
        self._live_storage_handles: "weakref.WeakSet[_StorageCheckHandle]" = (
            weakref.WeakSet()
        )

        self.can_compose: bool = bool(can_compose)
        self.can_store: bool = bool(can_store)
        self.tool_policy: Union[ToolPolicyFn, None, object] = tool_policy
        self._model = model
        self._preprocess_msgs = preprocess_msgs
        self._prompt_caching = prompt_caching
        self.add_tools("act", self._build_tools())

        self._main_event_loop: Optional[asyncio.AbstractEventLoop] = None
        try:
            self._main_event_loop = asyncio.get_running_loop()
        except RuntimeError:
            pass

        # Concurrency guard: limit active sandboxes per actor instance.
        self._act_semaphore = asyncio.Semaphore(20)
        # Timeout used when acquiring the semaphore (prevents unbounded waits).
        self._act_semaphore_timeout_s: float = 30.0
        self._active_work_heartbeat_interval_s: float = 60.0
        self._active_work_fallback_initial_delay_s: float = 120.0
        self._active_work_fallback_repeat_interval_s: float = 300.0

    # ───────────────────────── Session name registry ─────────────────────── #

    def _register_session_name(self, *, name: str, session_id: int) -> None:
        key: SessionKey = int(session_id)
        existing = self._session_names.get(name)
        if existing is not None and existing != key:
            raise ValueError(
                f"Session name {name!r} is already bound to {existing}, cannot rebind to {key}.",
            )
        self._session_names[name] = key
        self._session_names_rev.setdefault(key, set()).add(name)

    def _resolve_session_name(self, name: str) -> SessionKey | None:
        return self._session_names.get(name)

    def _get_session_name(self, *, session_id: int) -> str | None:
        key: SessionKey = int(session_id)
        names = self._session_names_rev.get(key)
        if not names:
            return None
        # Prefer stable ordering for determinism.
        return sorted(names)[0]

    def _unregister_all_names_for_session(self, *, key: SessionKey) -> None:
        names = self._session_names_rev.pop(key, None)
        if not names:
            return
        for n in list(names):
            self._session_names.pop(n, None)

    def _count_active_sessions_total(self) -> int:
        return len(
            self._session_executor._python_sessions,
        )  # pylint: disable=protected-access

    def _session_exists(self, *, session_id: int) -> bool:
        return self._session_executor.has_python_session(session_id=int(session_id))

    def _validate_execution_params(
        self,
        *,
        state_mode: str,
        session_id: int | None,
        session_name: str | None,
    ) -> dict | None:
        return _validate_execution_params(
            state_mode=state_mode,
            session_id=session_id,
            session_name=session_name,
            resolve_session_name=self._resolve_session_name,
            get_session_name_for_id=lambda s: self._get_session_name(session_id=s),
            session_exists=lambda s: self._session_exists(session_id=s),
            max_sessions_total=self._max_sessions_total,
            active_session_count=self._count_active_sessions_total(),
        )

    def _resolve_session(
        self,
        *,
        state_mode: str,
        session_id: int | None,
        session_name: str | None,
    ) -> int | None:
        """Resolve/allocate a session and validate execution params.

        Handles the full session resolution flow used by both ``execute_code``
        and ``execute_function``:

        1. For stateful mode: resolve an existing session name, allocate a new
           session id, or default to session 0.
        2. Register session name aliases when both name and id are provided.
        3. Validate the resulting execution parameters.

        Returns the resolved session id, or ``None`` for a stateless call.
        """
        # Resolve / allocate sessions for stateful.
        if state_mode == "stateful":
            if session_name:
                resolved = self._resolve_session_name(session_name)
                if resolved is not None:
                    session_id = resolved
                elif session_id is None:
                    session_id = self._next_session_id
                    self._next_session_id += 1
                    self._register_session_name(
                        name=session_name,
                        session_id=int(session_id),
                    )
            elif session_id is None:
                session_id = 0
        # UNIFY_STATEFUL_CELLS: one session, so read_only reads it.
        if state_mode == "read_only" and session_id is None and not session_name:
            from unify.actor import cell_state

            if cell_state.enabled():
                session_id = 0

        # If name + id are both set but not registered yet, register alias.
        if state_mode == "stateful" and session_name and session_id is not None:
            if self._resolve_session_name(session_name) is None:
                self._register_session_name(
                    name=session_name,
                    session_id=int(session_id),
                )

        # Refuses by raising if the parameters cannot be executed as given.
        self._validate_execution_params(
            state_mode=state_mode,
            session_id=session_id,
            session_name=session_name,
        )

        return session_id

    async def _run_active_work_heartbeat(
        self,
        active_work: ActiveWorkHandle,
        notification_q: asyncio.Queue[dict] | None,
    ) -> None:
        try:
            while True:
                await asyncio.sleep(self._active_work_heartbeat_interval_s)
                active_work.heartbeat()
                if (
                    notification_q is not None
                    and active_work.fallback_notification_due(
                        initial_delay_s=self._active_work_fallback_initial_delay_s,
                        repeat_interval_s=self._active_work_fallback_repeat_interval_s,
                    )
                ):
                    await notification_q.put(
                        {
                            "type": "notification",
                            "message": "Still working on the code step...",
                            "source": "active_work",
                            "completed": False,
                            "active_work_id": active_work.work_id,
                        },
                    )
                    active_work.record_fallback_notification()
        except asyncio.CancelledError:
            pass

    @staticmethod
    def _sandbox_call_binding(
        *,
        clarification_up_q: asyncio.Queue[str] | None,
        clarification_down_q: asyncio.Queue[str] | None,
        notification_q: asyncio.Queue | None = None,
    ):
        """Bind one tool call's channels onto the live sandbox.

        Both per-call and both restored on exit:

        * clarification queues, so nested manager clarifications write into the
          outer tool's ``clar_up_queue`` (mailbox A) watched by the async tool
          loop
        * a :class:`SteeringSession` with no correction channel: nothing
          interrupts the running block, and its checkpoints only record how
          far it got (reported when the block fails) and carry its progress
          notifications

        Yields the session so the caller can report progress once execution
        ends, however it ended.
        """
        from contextlib import contextmanager

        from unify.actor.environments.base import (
            bind_sandbox_clarification_queues,
            restore_sandbox_clarification_queues,
        )
        from unify.function_manager.steering import SteeringSession, use_session

        @contextmanager
        def _binding():
            try:
                sb = _CURRENT_SANDBOX.get()
            except Exception:
                yield None
                return

            clar_token = None
            if clarification_up_q is not None and clarification_down_q is not None:
                clar_token = bind_sandbox_clarification_queues(
                    sb.global_state,
                    clarification_up_q,
                    clarification_down_q,
                )
            # The core tool surface: the cell's request_clarification is the
            # session's (or absent where it cannot ask); None otherwise.
            core_clarification = core_surface.bind_clarification(
                sb.global_state,
                clarification_up_q,
                clarification_down_q,
            )

            steering = SteeringSession(notification_q=notification_q)
            # Carried by context rather than installed on this sandbox:
            # stateless cells build a fresh sandbox per call that would
            # never see anything installed here.
            try:
                with use_session(steering):
                    yield steering
            finally:
                if core_clarification is not None:
                    core_clarification()
                if clar_token is not None:
                    restore_sandbox_clarification_queues(sb.global_state, clar_token)

        return _binding()

    def _build_tools(self) -> Dict[str, Callable[..., Awaitable[Any]]]:
        """Builds the dictionary of tools available to the LLM."""

        @llm_soft_required(thought="")
        async def execute_code(
            thought: Annotated[
                str,
                "A brief, first-person, one-sentence explanation of what this "
                'code does and why you are running it right now (e.g. "Loading '
                'the data and computing the summary the user asked for."). Shown '
                "to the user as the rationale for this step; always provide it.",
            ],
            code: Optional[str] = None,
            *,
            state_mode: str | None = None,
            session_id: int | None = None,
            session_name: str | None = None,
            _notification_up_q: asyncio.Queue[dict] | None = None,
            _clarification_up_q: asyncio.Queue[str] | None = None,
            _clarification_down_q: asyncio.Queue[str] | None = None,
            _parent_chat_context: list[dict] | None = None,
            _language: str = "python",
        ) -> Any:
            """
            Execute arbitrary Python code in a specified state mode.

            **IMPORTANT — single-call rule**: If the task requires only a
            single function or primitive call with no surrounding logic,
            use ``execute_function`` instead. ``execute_code`` is for
            **multi-step composition** — conditional logic, loops, or
            combining multiple primitives/functions where intermediate
            results are needed within the same code block.

            Key concepts
            -----------
            - **state_mode**: omit it and the cell runs **stateful in
              session 0** — the current per-call sandbox, so variables
              persist across cells. Pass "stateless" for an isolated
              fresh run (environment globals and FunctionManager-discovered
              functions still available), "read_only" to read an existing
              session without persisting, or "stateful" with a session
              selector to target a named session.
            - **session_id/session_name**: stateful/read_only only.
              Stateful defaults to **session_id=0** — inside a running
              act() loop, the current per-call sandbox. Create an
              additional session with a fresh ``session_name`` (recommended)
              or an explicit ``session_id`` > 0; choose via
              ``list_sessions()`` / ``inspect_state()``.
            - **Shell commands** run from Python via ``subprocess`` (or
              ``asyncio.create_subprocess_exec``); there is no shell cell.

            Output
            ------
            An ExecutionResult with: ``stdout`` / ``stderr`` (rich
            List[TextPart | ImagePart]), ``result`` (last expression's
            value), ``error``, ``state_mode``, ``session_id``,
            ``session_name``, ``session_created``, ``duration_ms``.
            """
            _ = thought  # Thought is logged by the LLM; not used programmatically.
            if state_mode is None:
                state_mode = "stateful"
            if code is None or code.strip() == "":
                return {
                    "stdout": "",
                    "stderr": "",
                    "result": None,
                    "error": None,
                    "state_mode": state_mode,
                    "session_id": session_id,
                    "session_name": session_name,
                    "session_created": False,
                    "duration_ms": 0,
                }

            # ──────────────────────────────────────────────────────────────
            # Boundary wrapper: execute_code (lineage + events + terminal log)
            # ──────────────────────────────────────────────────────────────

            _suffix = _token_hex(2)
            _call_id = new_call_id()
            _parent = TOOL_LOOP_LINEAGE.get([])
            _parent_lineage = list(_parent) if isinstance(_parent, list) else []
            _hierarchy = [*_parent_lineage, f"execute_code({_suffix})"]
            # Establish a boundary lineage frame so nested calls (e.g., FunctionManager-injected
            # functions calling primitives) keep a consistent parent->child chain.
            _lineage_token = TOOL_LOOP_LINEAGE.set(_hierarchy)

            async def _pub_safe(**payload: Any) -> None:
                try:
                    await publish_manager_method_event(
                        _call_id,
                        "CodeActActor",
                        "execute_code",
                        hierarchy=_hierarchy,
                        display_label="Running code",
                        **payload,
                    )
                except Exception as e:
                    log_boundary_event(
                        "->".join(_hierarchy),
                        f"Warning: failed to publish event: {type(e).__name__}: {e}",
                        icon="⚠️",
                        level="warning",
                    )

            try:
                await _pub_safe(phase="incoming")
            except Exception:
                pass
            log_boundary_event("->".join(_hierarchy), "Executing code...", icon="🛠️")

            out: dict[str, Any] | None = None
            tb_str: str | None = None
            exec_exc: Exception | None = None

            active_work = ACTIVE_WORK.begin(
                label="execute_code",
                metadata={
                    "state_mode": state_mode,
                    "session_id": session_id,
                    "session_name": session_name,
                    "thought": thought[:500],
                },
            )
            # The agent record: nothing reaches the model while its cell
            # runs, so neither the heartbeat nor in-cell progress is wired.
            _notification_up_q = None
            heartbeat_task: asyncio.Task[None] | None = None
            try:
                heartbeat_task = asyncio.create_task(
                    self._run_active_work_heartbeat(active_work, _notification_up_q),
                )
                notification_q = (
                    _ActiveWorkNotificationQueue(_notification_up_q, active_work)
                    if _notification_up_q is not None
                    else None
                )
                session_id = self._resolve_session(
                    state_mode=state_mode,
                    session_id=session_id,
                    session_name=session_name,
                )

                _pcc_token = _PARENT_CHAT_CONTEXT.set(_parent_chat_context)
                _steering = None
                try:
                    with self._sandbox_call_binding(
                        clarification_up_q=_clarification_up_q,
                        clarification_down_q=_clarification_down_q,
                        notification_q=notification_q,
                    ) as _steering:
                        # The workspace tools route another language here
                        # (see _workspace_tools).
                        _lang_kw = (
                            {"language": _language} if _language != "python" else {}
                        )
                        # UNIFY_VARIABLE_INVENTORY: a cell that keeps what it
                        # binds ends its result with the session's variables.
                        if _language == "python" and inventory_enabled():
                            _lang_kw["inventory"] = True
                        try:
                            out = await self._session_executor.execute(
                                code=code,
                                state_mode=state_mode,  # type: ignore[arg-type]
                                session_id=session_id,
                                **_lang_kw,
                            )
                        except Exception as e:
                            exec_exc = e
                            tb = traceback.format_exc()
                            tb_str = tb
                            out = {
                                "stdout": "",
                                "stderr": "",
                                "result": None,
                                "error": tb,
                                "state_mode": state_mode,
                                "session_id": session_id,
                                "session_name": session_name,
                                "session_created": False,
                                "duration_ms": 0,
                            }
                finally:
                    _PARENT_CHAT_CONTEXT.reset(_pcc_token)

                # Only when something actually steered this block, or when it
                # failed — an uninterrupted success reports nothing, so the
                # common case costs no transcript weight.
                #
                # On failure the report earns its place: instrumentation shifts
                # the line numbers in a "<string>" traceback, which carries no
                # source text to cross-reference, whereas the checkpoint's
                # ``last_line_reached`` is in the coordinates of the code as
                # written.
                if _steering is not None and (_steering.messages or out.get("error")):
                    out["steering"] = _steering.progress()

                # Enrich with session name.
                if out.get("session_id") is not None:
                    out["session_name"] = self._get_session_name(
                        session_id=int(out["session_id"]),
                    )
                else:
                    out["session_name"] = None

                # Wrap in ExecutionResult for proper LLM image formatting.
                if isinstance(out.get("stdout"), list):
                    out = ExecutionResult(**out)

                return out
            finally:
                active_work.end()
                if heartbeat_task is not None and not heartbeat_task.done():
                    heartbeat_task.cancel()
                    try:
                        await heartbeat_task
                    except (asyncio.CancelledError, Exception):
                        pass
                try:
                    _out_err = (
                        (
                            out.get("error")
                            if isinstance(out, dict)
                            else getattr(out, "error", None)
                        )
                        if out is not None
                        else None
                    )
                    if _out_err:
                        await _pub_safe(
                            phase="outgoing",
                            status="error",
                            error=str(_out_err),
                            error_type=(
                                type(exec_exc).__name__
                                if exec_exc is not None
                                else "Error"
                            ),
                            traceback=(tb_str or "")[:2000],
                        )
                    else:
                        await _pub_safe(phase="outgoing", status="ok")
                except Exception:
                    pass
                try:
                    TOOL_LOOP_LINEAGE.reset(_lineage_token)
                except Exception:
                    pass

        # UNIFY_REPLY_CHANNEL=code+text: the description says a cell can reply.
        if cell_reply.enabled():
            execute_code.__doc__ = (
                execute_code.__doc__.rstrip()
                + "\n\n"
                + textwrap.indent(_EXECUTE_CODE_REPLY_DOC.strip("\n"), " " * 12)
                + "\n"
            )

        # ───────────────────────── Package installation tool ────────────────── #

        async def install_python_packages(
            packages: list[str],
        ) -> dict:
            return await asyncio.to_thread(environment.install, list(packages))

        install_python_packages.__doc__ = _INSTALL_PYTHON_PACKAGES_DOC

        tools: Dict[str, Callable[..., Awaitable[Any]]] = {
            "execute_code": ToolSpec(fn=execute_code),
            "install_python_packages": ToolSpec(
                fn=install_python_packages,
                display_label="Installing Python packages",
            ),
        }
        tools.update(_workspace_tools(execute_code))

        # ── Proactive skill storage tool ──────────────────────────────
        if self.function_manager and self.guidance_manager:
            _actor_ref = self

            async def store_skills(request: str) -> Any:
                """Proactively store reusable skills from the current execution trajectory.

                Triggers a skill-storage review of the trajectory so far. A dedicated
                reviewer will examine the execution history and store any reusable
                functions and compositional guidance based on your request.

                Use this when you have just completed a complex subtask and recognize
                a reusable pattern worth preserving — for example, a non-obvious
                configuration of primitives.actor.act, a multi-step procedure, or a
                function that bakes in hard-won configuration.

                Parameters
                ----------
                request : str
                    Describe the skill(s) you want stored. Be specific about which
                    part of the trajectory contains the reusable pattern and what
                    makes it valuable. For example: "Store the report-rendering
                    function that bakes in the discovered pandoc flags" or "Store
                    the multi-step data pipeline that combines CSV parsing with
                    the retry loop around the export API."

                Returns
                -------
                str
                    A summary of what was stored (functions and/or guidance), or
                    a note that nothing was worth storing.
                """
                ctx = get_current_agent_context()
                handle = ctx.handle
                if handle is None:
                    return "No active execution context to snapshot."

                _client = getattr(handle, "_client", None)
                _trajectory = (
                    make_messages_safe_for_context_dump(
                        list(getattr(_client, "messages", []) or []),
                    )
                    if _client
                    else []
                )

                _ps_call_id = new_call_id()
                _ps_parent = TOOL_LOOP_LINEAGE.get([])
                _ps_parent_lineage = (
                    list(_ps_parent) if isinstance(_ps_parent, list) else []
                )
                _ps_suffix = _token_hex(2)
                _ps_hierarchy = [
                    *_ps_parent_lineage,
                    f"ProactiveStorage(CodeActActor.act)({_ps_suffix})",
                ]

                await publish_manager_method_event(
                    _ps_call_id,
                    "CodeActActor",
                    "ProactiveStorage",
                    phase="incoming",
                    display_label="Proactive skill storage",
                    hierarchy=_ps_hierarchy,
                    instructions=request,
                )

                storage_handle = _start_proactive_storage_loop(
                    trajectory=_trajectory,
                    actor=_actor_ref,
                    request=request,
                    parent_lineage=_ps_parent_lineage,
                )

                if storage_handle is None:
                    await publish_manager_method_event(
                        _ps_call_id,
                        "CodeActActor",
                        "ProactiveStorage",
                        phase="outgoing",
                        display_label="Proactive skill storage",
                        hierarchy=_ps_hierarchy,
                    )
                    return (
                        "Skill storage unavailable "
                        "(FunctionManager or GuidanceManager missing)."
                    )

                _orig_result_fn = storage_handle.result

                async def _tracking_result():
                    try:
                        result = await _orig_result_fn()
                        ctx.proactive_storage_summaries.append(result)
                        return result
                    finally:
                        await publish_manager_method_event(
                            _ps_call_id,
                            "CodeActActor",
                            "ProactiveStorage",
                            phase="outgoing",
                            display_label="Proactive skill storage",
                            hierarchy=_ps_hierarchy,
                        )

                storage_handle.result = _tracking_result  # type: ignore[assignment]

                return storage_handle

            tools["store_skills"] = store_skills

        # ───────────────────────── Session management tools ────────────────── #

        async def list_sessions(detail: str = "summary") -> Dict[str, Any]:
            """
            List all active sessions.

            Use this to choose which session a subsequent
            `execute_code(..., state_mode="stateful"/"read_only")` call
            should target.

            Parameters
            ----------
            detail:
                "summary" (default): metadata + a short `state_summary`;
                "full": best-effort enrichment via cheap inspection.

            Returns
            -------
            dict:
                {"sessions": [...]}; each entry carries session_id,
                session_name, created_at / last_used, and state_summary.
                The default per-call sandbox appears as session_id=0 when
                bound.
            """
            detail = (detail or "summary").strip()

            sessions: list[dict[str, Any]] = []

            # Default sandbox (current act sandbox) as session 0.
            try:
                sb = _CURRENT_SANDBOX.get()
                sessions.append(
                    {
                        "session_id": 0,
                        "session_name": self._get_session_name(session_id=0),
                        "created_at": None,
                        "last_used": None,
                        "state_summary": f"{len(sb.global_state)} globals",
                    },
                )
            except Exception:
                pass

            # In-process sessions created via SessionExecutor.
            for s in self._session_executor.list_in_process_python_sessions():
                s = dict(s)
                s["session_name"] = self._get_session_name(
                    session_id=int(s["session_id"]),
                )
                sessions.append(s)

            return {"sessions": sessions}

        async def inspect_state(
            session_name: str | None = None,
            session_id: int | None = None,
            detail: str = "summary",
        ) -> Dict[str, Any]:
            """
            Inspect the state of a specific session.

            Use it to decide whether to continue in a session, start fresh,
            run stateless, or do a read_only what-if.

            Parameters
            ----------
            session_name:
                Human-friendly alias (preferred when available).
            session_id:
                Direct identity.
            detail:
                "summary" (quick context) | "names" (variable names only) |
                "full" (sparingly; values truncated/redacted best-effort).

            With no selector, inspects the **current per-call sandbox**
            (session_id=0) when bound.

            Returns
            -------
            dict with `session` ({session_id, session_name}) and
            `state` (the session's variables).
            """
            detail = (detail or "summary").strip()

            # Resolve session; with no selector, session 0.
            resolved: SessionKey | None = 0
            if session_name:
                resolved = self._resolve_session_name(session_name)
                if resolved is None:
                    return {
                        "error": f"Session {session_name!r} not found",
                        "error_type": "validation",
                    }
            elif session_id is not None:
                resolved = int(session_id)

            sb = self._session_executor.python_session(session_id=resolved)
            if sb is None:
                return {
                    "error": f"Session {resolved} not found",
                    "error_type": "validation",
                }
            # The variables live in the session's worker.
            in_worker = await sb.worker_variables()
            full_map: dict[str, str] = dict(in_worker)
            names = sorted(in_worker)
            state_obj = {
                "variables": full_map if detail == "full" else names,
                "functions": [],
            }
            return {
                "session": {
                    "session_id": resolved,
                    "session_name": self._get_session_name(session_id=resolved),
                },
                "state": state_obj,
            }

        async def close_session(
            session_name: str | None = None,
            session_id: int | None = None,
        ) -> Dict[str, Any]:
            """
            Close a specific session and free resources.

            **Idempotent**: closing an already-closed/non-existent session
            returns `closed=False, reason="not_found"` rather than raising.

            Parameters
            ----------
            session_name:
                Preferred: close by human-friendly alias.
            session_id:
                Close by canonical identity.

            Returns
            -------
            dict:
                closed (bool), reason ("success" | "not_found" | "error"),
                session ({session_id, session_name}).
            """
            resolved: SessionKey | None = None
            if session_name:
                resolved = self._resolve_session_name(session_name)
                if resolved is None:
                    return {
                        "closed": False,
                        "reason": "not_found",
                        "session": {
                            "session_id": session_id,
                            "session_name": session_name,
                        },
                    }
            elif session_id is not None:
                resolved = int(session_id)
            else:
                return {
                    "closed": False,
                    "reason": "error",
                    "error": "Must provide session_name or session_id.",
                }

            sid = int(resolved)
            closed = await self._session_executor.close_in_process_python_session(
                session_id=sid,
            )

            # Unregister all aliases for this session.
            self._unregister_all_names_for_session(key=sid)

            return {
                "closed": bool(closed),
                "reason": "success" if closed else "not_found",
                "session": {
                    "session_id": sid,
                    "session_name": session_name
                    or self._get_session_name(session_id=sid),
                },
            }

        async def close_all_sessions() -> Dict[str, Any]:
            """
            Close all active sessions.

            Blunt cleanup — prefer `close_session(...)` to discard one
            specific polluted/unused session.

            Returns
            -------
            dict:
                closed_count (int).
            """
            closed_count = 0

            for s in list(self._session_executor.list_in_process_python_sessions()):
                sid = int(s.get("session_id", 0))
                if await self._session_executor.close_in_process_python_session(
                    session_id=sid,
                ):
                    closed_count += 1
                    self._unregister_all_names_for_session(key=sid)

            # Clear any remaining aliases.
            self._session_names.clear()
            self._session_names_rev.clear()

            return {"closed_count": closed_count}

        tools["list_sessions"] = ToolSpec(
            fn=list_sessions,
            display_label="Listing active sessions",
        )
        tools["inspect_state"] = ToolSpec(
            fn=inspect_state,
            display_label="Inspecting session state",
        )
        tools["close_session"] = ToolSpec(
            fn=close_session,
            display_label="Closing a session",
        )
        tools["close_all_sessions"] = ToolSpec(
            fn=close_all_sessions,
            display_label="Closing all sessions",
        )

        _correct_tool_docs(tools, environments=self.environments)
        # UNIFY_PROMPT_TRIM: only primitives read the conversation a code
        # tool is given.
        if "primitives" not in self.environments:
            _hide_parent_chat_context(tools)
        from unify.actor import cell_state

        if cell_state.enabled():
            cell_state.correct_tools(tools)
        return tools

    @functools.wraps(BaseCodeActActor.act, updated=())
    @log_manager_call(
        "CodeActActor",
        "act",
        payload_key="request",
        display_label=lambda kw: "Session" if kw.get("persist") else "Taking action",
        forward_kwargs=("persist",),
    )
    async def act(
        self,
        request: str | dict | list[str | dict],
        *,
        guidelines: Optional[str] = None,
        clarification_enabled: bool = True,
        response_format: Optional[Type[BaseModel]] = None,
        _parent_chat_context: list[dict] | None = None,
        _clarification_up_q: Optional[asyncio.Queue[str]] = None,
        _clarification_down_q: Optional[asyncio.Queue[str]] = None,
        _call_id: Optional[str] = None,
        persist: Optional[bool] = None,
        can_compose: Optional[bool] = None,
        can_store: Optional[bool] = None,
        llm_profile: Optional[str] = None,
    ) -> ToolLoopHandle:
        if not self._main_event_loop:
            self._main_event_loop = asyncio.get_running_loop()

        import time as _act_time

        _act_t0 = _act_time.perf_counter()

        def _act_ms() -> str:
            return f"{(_act_time.perf_counter() - _act_t0) * 1000:.0f}ms"

        logger.debug(f"⏱️ [CodeActActor.act +{_act_ms()}] entered")

        effective_can_compose = (
            self.can_compose if can_compose is None else bool(can_compose)
        )
        effective_can_store = self.can_store if can_store is None else bool(can_store)
        # UNIFY_STORE_ADMISSION: the post-session review is the only writer,
        # and it runs only when an external check of the outcome admits it.
        admission_gated = effective_can_store and bool(_store_admission_path())
        act_llm_profile = resolve_act_llm_profile(llm_profile)

        # can_compose=False requires a FunctionManager so the LLM has execute_function
        # and the discovery tools available. Without it there are no usable tools.
        # The core tool surface: refuse what cannot run confined.
        core_surface.require_prerequisites(can_compose=effective_can_compose)

        if not effective_can_compose and self.function_manager is None:
            raise RuntimeError(
                "CodeActActor cannot run with can_compose=False: "
                "function_manager is required so execute_function and "
                "FunctionManager discovery tools are available.",
            )

        initial_prompt = (
            "This is an interactive session. Acknowledge that you are ready and "
            "wait for the user to provide instructions via interjection."
        )

        # The agent record: this act() joins its run's shared record, as the
        # main agent or as the helper a pool started.
        from unify.agents.binding import bind_for_act

        _agents = bind_for_act(
            request=str(request or ""),
            user_reads=bool(clarification_enabled),
        )
        # A question to the requester is a record post.
        clarification_enabled = False

        # Clarification queues for sandbox env injection (managers called from
        # execute_code). Separate from the tool-loop clarification_queues below:
        # auto-created env queues are unread on the CM→act path, so the loop
        # gets (None, None) unless the caller supplied queues explicitly.
        caller_supplied_clarification_queues = (
            clarification_enabled
            and _clarification_up_q is not None
            and _clarification_down_q is not None
        )
        env_clarification_up_q: Optional[asyncio.Queue[str]]
        env_clarification_down_q: Optional[asyncio.Queue[str]]
        if clarification_enabled:
            env_clarification_up_q = _clarification_up_q or asyncio.Queue()
            env_clarification_down_q = _clarification_down_q or asyncio.Queue()
        else:
            env_clarification_up_q = None
            env_clarification_down_q = None

        # Create per-call environments so clarification queues are not stored on shared actor environments.
        logger.debug(f"⏱️ [CodeActActor.act +{_act_ms()}] copying environments")
        sandbox_envs: Dict[str, "BaseEnvironment"] = {}
        try:
            from unify.actor.environments.base import (
                _CompositeEnvironment as _CompositeEnv,
            )
            from unify.actor.environments import (
                ActorEnvironment as _ActorEnvironment,
            )
        except Exception:
            _CompositeEnv = None  # type: ignore
            _ActorEnvironment = None  # type: ignore

        for ns, env in self.environments.items():
            # Prefer explicit reconstruction for known env types.
            try:
                if _CompositeEnv is not None and isinstance(env, _CompositeEnv):
                    sandbox_envs[ns] = _CompositeEnv(
                        env.sub_environments,
                        clarification_up_q=env_clarification_up_q,
                        clarification_down_q=env_clarification_down_q,
                    )
                    continue
                if _ActorEnvironment is not None and isinstance(
                    env,
                    _ActorEnvironment,
                ):
                    sandbox_envs[ns] = _ActorEnvironment(
                        allowed_methods=env.allowed_methods,
                        clarification_up_q=env_clarification_up_q,
                        clarification_down_q=env_clarification_down_q,
                    )
                    continue
            except Exception:
                pass

            # Fallback: shallow-copy and set private queue attrs on the copy only.
            try:
                env_copy = copy.copy(env)
                if hasattr(env_copy, "_clarification_up_q"):
                    setattr(env_copy, "_clarification_up_q", env_clarification_up_q)
                if hasattr(env_copy, "_clarification_down_q"):
                    setattr(env_copy, "_clarification_down_q", env_clarification_down_q)
                sandbox_envs[ns] = env_copy
            except Exception:
                sandbox_envs[ns] = env

        # Concurrency/backpressure guard for externally started actor runs.
        logger.debug(
            f"⏱️ [CodeActActor.act +{_act_ms()}] envs copied, preparing actor slot",
        )
        acquired_actor_slot = False
        try:
            await asyncio.wait_for(
                self._act_semaphore.acquire(),
                timeout=float(getattr(self, "_act_semaphore_timeout_s", 30.0)),
            )
            acquired_actor_slot = True
        except asyncio.TimeoutError:
            raise RuntimeError(
                "CodeActActor is at capacity (too many concurrent sessions). "
                "Try again later or reduce concurrency.",
            )
        logger.debug(
            f"⏱️ [CodeActActor.act +{_act_ms()}] actor slot ready, creating sandbox",
        )
        # Packages installed by earlier tasks and sessions are importable
        # before the first cell runs: the sandboxed worker puts them on its
        # own path.
        sandbox = PythonExecutionSession(environments=sandbox_envs)
        token = _CURRENT_SANDBOX.set(sandbox)
        env_token = _CURRENT_ENVIRONMENTS.set(sandbox_envs)
        can_clarify_token = _CAN_CLARIFY.set(bool(clarification_enabled))
        llm_profile_token = CURRENT_ACT_LLM_PROFILE.set(act_llm_profile)
        # What this run may do; every sub-actor it starts is bounded by it.
        grants_token = CALLER_GRANTS.set(
            ActorGrants.of_actor(
                self,
                can_compose=effective_can_compose,
                can_store=effective_can_store,
            ),
        )

        # Set agent context for depth tracking and handle access
        parent_ctx = _CURRENT_AGENT_CONTEXT.get()
        new_ctx = AgentContext(
            depth=parent_ctx.depth + 1,
            agent_id=str(uuid.uuid4()),
            handle=None,  # Will be set after handle is created
        )
        ctx_token = _CURRENT_AGENT_CONTEXT.set(new_ctx)

        async def _cleanup() -> None:
            try:
                # Best-effort cleanup
                if hasattr(sandbox, "close") and callable(getattr(sandbox, "close")):
                    await sandbox.close()  # type: ignore[misc]
            except Exception:
                pass
            try:
                _CURRENT_SANDBOX.reset(token)
            except Exception:
                pass
            try:
                _CURRENT_ENVIRONMENTS.reset(env_token)
            except Exception:
                pass
            try:
                _CAN_CLARIFY.reset(can_clarify_token)
            except Exception:
                pass
            try:
                CURRENT_ACT_LLM_PROFILE.reset(llm_profile_token)
            except Exception:
                pass
            try:
                CALLER_GRANTS.reset(grants_token)
            except Exception:
                pass
            try:
                _CURRENT_AGENT_CONTEXT.reset(ctx_token)
            except Exception:
                pass
            if acquired_actor_slot:
                try:
                    self._act_semaphore.release()
                except Exception:
                    pass

        # Build the tool set for this call. When can_compose=False the LLM
        # can_compose=False: specialist may only discover and execute stored
        # functions — no arbitrary code, no function persistence.
        # can_store=False: function/guidance library is read-only.
        # Session tools are kept because execute_function supports the same
        # session/state_mode semantics.
        _compose_only_tools = {
            "execute_code",
            "install_python_packages",
        }
        _store_only_tools = {"store_skills"}
        # Admission-gated sessions write nothing in-session either.
        _admission_withheld_tools = _store_only_tools

        def _filter_tools(
            tool_dict: Dict[str, Any],
            *,
            withhold_admission: bool = True,
        ) -> Dict[str, Any]:
            """Apply static per-call filters (can_compose, can_store)."""
            out = dict(tool_dict)
            if not effective_can_compose:
                for name in _compose_only_tools:
                    out.pop(name, None)
                for name in _store_only_tools:
                    out.pop(name, None)
            if not effective_can_store:
                for name in _store_only_tools:
                    out.pop(name, None)
            if admission_gated and withhold_admission:
                for name in _admission_withheld_tools:
                    out.pop(name, None)
            return out

        _act_tools = self.get_tools("act")
        base_tools = _filter_tools(_act_tools)

        # The core tool surface: execute_code is the only JSON tool; the
        # libraries, install, read_file and grep are objects in the sandbox,
        # whose writes refuse at call time what this session may not do.
        from unify.settings import SETTINGS as _CORE_SETTINGS

        _core_reviews = effective_can_store and not admission_gated
        core_session = core_surface.start_session(
            self,
            sandbox=sandbox,
            tools=base_tools,
            policy=core_surface.WritePolicy(
                can_store=effective_can_store,
                admission_gated=admission_gated,
            ),
            store_skills=_core_reviews,
            clarification_enabled=clarification_enabled,
            caller_queues=(
                (env_clarification_up_q, env_clarification_down_q)
                if caller_supplied_clarification_queues
                else None
            ),
            # Defined below, before the loop can call them.
            on_clarification_request=lambda q: (
                _on_clar_req(q) if _on_clar_req is not None else None
            ),
            on_clarification_answer=lambda a: (
                _on_clar_ans(a) if _on_clar_ans is not None else None
            ),
            structured=response_format is not None,
            turn_reviews=(
                _core_reviews
                and bool(persist)
                and bool(_CORE_SETTINGS.UNIFY_TURN_STORAGE_REVIEWS)
            ),
        )
        base_tools = dict(core_session.tools)

        # UNIFY_CODE_PROJECTION=notebook: execute_code takes one field, the
        # cell, whose first-line magics map onto the same function's
        # arguments; the session tools' data is the %sessions magic.
        from unify.actor import notebook_cells

        if notebook_cells.enabled() and "execute_code" in base_tools:
            from unify.actor.prompt_builders import _injects_actor_primitives
            from unify.actor.workspace_tools import _network_text

            base_tools = notebook_cells.project_tools(
                base_tools,
                caps=notebook_cells.Capabilities(bash=True),
                structured=response_format is not None,
                parent_context=_injects_actor_primitives(sandbox_envs),
                resolve_session_name=self._resolve_session_name,
                session_tools=_act_tools,
                network=_network_text(),
            )

        effective_guidelines = (
            "\n\n".join(filter(None, [self._base_guidelines, guidelines])) or None
        )

        from unify.settings import SETTINGS

        # The default policy leaves the library searches to the model (no
        # gated or forced turn).
        default_policy = self.tool_policy is _USE_DEFAULT
        logger.debug(f"⏱️ [CodeActActor.act +{_act_ms()}] building system prompt")
        system_prompt = build_code_act_prompt(
            environments=sandbox_envs,
            core=core_session.prompt,
            # An admission-gated session has no in-session storage tools to
            # describe; it is told the libraries are read-only instead.
            can_store=effective_can_store and not admission_gated,
            guidelines=effective_guidelines,
            persist=bool(persist),
            library_read_only=admission_gated,
        )
        if notebook_cells.enabled() and "execute_code" in base_tools:
            # UNIFY_CODE_PROJECTION=notebook: the magics, where the prompt
            # named the session fields and tools.
            system_prompt = notebook_cells.rewrite_prompt(system_prompt)
        # What opens the session's first user message (first_message_context),
        # in this order: the library's size (UNIFY_LIBRARY_SNAPSHOT), then a
        # rule and the request.
        first_message_parts: list[str] = []
        logger.debug(
            f"⏱️ [CodeActActor.act +{_act_ms()}] prompt built "
            f"({len(system_prompt)} chars, {len(base_tools)} tools)",
        )

        # Tool policy controls which tools are visible per turn, and whether a
        # tool call is required.  The static _filter_tools (can_compose,
        # can_store) is always applied regardless of the dynamic policy.
        if self.tool_policy is None or default_policy:
            # No dynamic policy -- only static filtering on every turn.
            def _static_only_policy(step: int, tools: Dict[str, Any]):
                return "auto", _filter_tools(tools)

            tool_policy: Optional[ToolPolicyFn] = _static_only_policy
        else:
            # Custom caller-provided policy.  Wrap it so that _filter_tools
            # is always applied first (static filters are never bypassed).
            _user_policy = self.tool_policy

            def _wrapped_policy(step: int, tools: Dict[str, Any]):
                return _user_policy(step, _filter_tools(tools))

            tool_policy = _wrapped_policy

        # Build an LLM client for this act() call. The profile is per-call so
        # concurrent runs on the same actor can use different models safely.
        client_model = act_llm_profile.model or self._model
        client = new_llm_client(
            client_model,
            purpose="planning",
            origin="CodeActActor.act",
            **act_llm_profile.client_kwargs,
        )
        if system_prompt:
            client.set_system_message(system_prompt)

        # UNIFY_LIBRARY_SNAPSHOT: the first user message says how large the
        # libraries are at task start.
        snapshot = _library_snapshot_line(
            _library_counts(self.function_manager, self.guidance_manager),
            has_fm_tools=core_session.prompt.functions,
            has_gm_tools=core_session.prompt.guidance,
        )
        if snapshot:
            first_message_parts.append(snapshot)

        tools = dict(base_tools)

        # Build event bus callbacks for clarification and notification tools
        # (the loop creates the tools; we just provide the event hooks).
        _on_clar_req = None
        _on_clar_ans = None
        if clarification_enabled:

            async def _on_clar_req(q: str):
                try:
                    await EVENT_BUS.publish(
                        Event(
                            type="ManagerMethod",
                            calling_id=_call_id,
                            payload={
                                "manager": "CodeActActor",
                                "method": "act",
                                "action": "clarification_request",
                                "question": q,
                            },
                        ),
                    )
                except Exception:
                    pass

            async def _on_clar_ans(ans: str):
                try:
                    await EVENT_BUS.publish(
                        Event(
                            type="ManagerMethod",
                            calling_id=_call_id,
                            payload={
                                "manager": "CodeActActor",
                                "method": "act",
                                "action": "clarification_answer",
                                "answer": ans,
                            },
                        ),
                    )
                except Exception:
                    pass

        logger.debug(f"⏱️ [CodeActActor.act +{_act_ms()}] starting async tool loop")
        run_meter = new_run_meter()
        meter_token = current_run_meter.set(run_meter)
        # UNIFY_STORE_INSTANCE_LINT: the task loop, its tools and its storage
        # review inherit the identifiers of this request (set until the handle
        # is built); a sub-agent keeps those of the task it works for.
        instance_token = _instance_lint.enter(request)
        core_token = core_session.enter()
        try:
            # The library entries closest to the request, after the snapshot line.
            from unify.actor.library_shortlist import shortlist_block

            shortlist = shortlist_block(
                self.function_manager,
                self.guidance_manager,
                request,
                functions=core_session.prompt.functions,
                guidance=core_session.prompt.guidance,
                # The listed functions are bound as a read binds them, and
                # the header says how to call.
                bind=core_session.listed_binder(sandbox),
            )
            if shortlist:
                first_message_parts.append(shortlist)
            sandbox.global_state.update(_agents.globals())
            if isinstance(getattr(sandbox, "core_globals", None), dict):
                sandbox.core_globals.update(_agents.globals())
            first_message_parts.append(_agents.prompt_section())
            handle = start_async_tool_loop(
                client,
                request or initial_prompt,
                tools,
                loop_id=f"CodeActActor.act",
                # The record carries what a helper is told, so no parent
                # chat context reaches the loop.
                parent_chat_context=None,
                log_steps=True,
                tool_policy=tool_policy,
                response_format=response_format,
                persist=persist,
                # UNIFY_REPLY_CHANNEL=code+text: a cell's reply() ends a turn.
                reply_channel=True,
                # UNIFY_BIND_REQUEST=on: a cell reads the request as ``request``.
                bind_request=True,
                preprocess_msgs=self._preprocess_msgs,
                prompt_caching=self._prompt_caching,
                extra_compression_tools=(
                    ["store_skills"]
                    if effective_can_store and not admission_gated
                    else None
                ),
                # request_clarification is the sandbox's (the core surface),
                # and there is no send_notification tool.
                clarification_queues=None,
                on_clarification_request=_on_clar_req,
                on_clarification_answer=_on_clar_ans,
                # No notification channel: nothing reads progress
                # notifications under the agent record.
                on_notify=None,
                compression_tools_on_demand=True,
                on_turn_boundary=_agents.on_turn_boundary,
                **(
                    {
                        "first_message_context": "\n\n".join(
                            part for part in first_message_parts if part
                        ),
                    }
                    if any(first_message_parts)
                    else {}
                ),
            )
        except BaseException:
            _instance_lint.leave(instance_token)
            raise
        finally:
            current_run_meter.reset(meter_token)
            if core_token is not None:
                core_surface.Session.leave(core_token)
        handle.run_meter = run_meter  # type: ignore[attr-defined]
        logger.debug(
            f"⏱️ [CodeActActor.act +{_act_ms()}] loop started, returning handle",
        )

        # Wrap result() to run cleanup when the loop finishes
        _original_result = handle.result
        _loop_handle = handle

        async def _result_with_cleanup() -> str:
            try:
                try:
                    result = await _original_result()
                except BaseException:
                    await _end_agents_request(
                        _agents,
                        reason="the main agent's run failed",
                    )
                    raise
                # The agent record: the main agent's answer ends its request;
                # helpers still running are stopped and checked to have ended. A
                # stopped run or a finished session has no answer to record.
                stop_event = getattr(_loop_handle, "_stop_event", None)
                if persist or (stop_event is not None and stop_event.is_set()):
                    await _end_agents_request(
                        _agents,
                        reason="the main agent's session ended or was stopped",
                    )
                else:
                    await _end_agents_request(_agents, reply=str(result))
                return result
            finally:
                await _cleanup()

        handle.result = _result_with_cleanup  # type: ignore[assignment]

        # Update agent context with handle reference
        new_ctx.handle = handle
        handle.agents_pool = _agents.pool  # type: ignore[attr-defined]

        # Wrap in StorageCheckHandle for post-completion function review. A
        # persistent session is reviewed once, when it ends (its stop is a
        # deliberate one, which Phase 2 still reviews); turn-boundary reviews
        # are an opt-in for sessions that live too long to wait for that.
        if effective_can_store:
            from unify.settings import SETTINGS

            handle = _StorageCheckHandle(
                inner=handle,
                actor=self,
                meter=run_meter,
                turn_reviews_enabled=(
                    effective_can_store
                    and not admission_gated
                    and bool(persist)
                    and bool(SETTINGS.UNIFY_TURN_STORAGE_REVIEWS)
                ),
                persist=bool(persist),
            )
            # Tracked so ``close()`` can end a review still in flight. The
            # set is weak: a finished handle the caller has dropped must not
            # be kept alive by this bookkeeping.
            self._live_storage_handles.add(handle)

        _instance_lint.leave(instance_token)
        # The handle the caller holds (the storage wrapper by default).
        handle.agents_pool = _agents.pool  # type: ignore[attr-defined]
        return handle

    async def close(self):
        """Shuts down the actor and its associated resources gracefully."""
        # End any storage review still running before the resources it needs
        # are torn down below. Left alone, a review outlives the actor: it
        # keeps issuing inference against dead sandboxes for a run already
        # recorded as finished.
        for storage_handle in list(self._live_storage_handles):
            await storage_handle.abandon_storage_review(
                reason="The actor running this review is shutting down.",
            )
        self._live_storage_handles.clear()

        # Close any in-process session sandboxes owned by the session executor.
        try:
            await self._session_executor.close()
        except Exception:
            pass

        # Clear session name registry.
        try:
            self._session_names.clear()
            self._session_names_rev.clear()
        except Exception:
            pass
