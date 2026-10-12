from __future__ import annotations

import asyncio
import logging
from abc import ABC, abstractmethod
from typing import Dict, Optional, Type, TYPE_CHECKING
from pydantic import BaseModel

from unify.common.async_tool_loop import SteerableToolHandle
from unify.common.state_managers import BaseStateManager

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from unify.actor.environments.base import BaseEnvironment
    from unify.function_manager.function_manager import FunctionManager
    from unify.guidance_manager.guidance_manager import GuidanceManager

__all__ = [
    "BaseActor",
    "BaseActorHandle",
    "BaseCodeActActor",
]

# --------------------------------------------------------------------------- #
# BaseActor
# --------------------------------------------------------------------------- #


class BaseActorHandle(SteerableToolHandle, ABC):
    """
    Marker base class for all actor handles returned by Actor.act().

    This provides a common nominal type across actor implementations while
    preserving the unified steerable surface inherited from SteerableToolHandle.
    Implementations are free to add additional helpers or properties, but the
    core pause/resume/stop/interject/ask/result interface must remain intact.
    """


class BaseActor(ABC):
    """
    Abstract contract that every concrete actor must satisfy.

    An actor is a component capable of performing work based on a natural
    language request. It returns a steerable handle that can be paused,
    resumed, interjected, or stopped. This type is intentionally decoupled
    from any task-specific terminology or lifecycle.

    Purpose and positioning
    -----------------------
    The Actor provides a direct, real-time handle to "act" in the world and
    get things done – e.g. open a web page, click UI elements, or perform a
    short-lived sandbox session during a conversation.

    Intended use
    ------------
    Use the Actor for interactive, ephemeral sessions within a live
    conversation (onboarding, guided walkthroughs, ad‑hoc demonstrations).
    It returns a steerable handle suitable for pause/resume/interject/stop.

    Usage guidance (LLM‑facing)
    ---------------------------
    Prefer calling ``Actor.act`` when the user's instruction implies a live,
    ad‑hoc, conversational session that should happen "now" inside the current
    chat, especially when the activity involves controlling tools or a UI in
    short iterative steps. Typical phrasings include:

    - "open a web page", "open a window", "navigate/click/show me"
    - "walk me through", "let's set this up together", "guide me live"
    - "troubleshoot together", "pair on this", "step‑by‑step now"

    This interface starts a live session and returns a steerable handle; it does
    not create durable records or schedules.
    """

    _as_caller_description: str = (
        "the Actor, performing a live action on behalf of the end user"
    )

    def __init__(
        self,
        *,
        environments: Optional[list["BaseEnvironment"]] = None,
        function_manager: Optional["FunctionManager"] = None,
        guidance_manager: Optional["GuidanceManager"] = None,
    ) -> None:
        """
        Shared initialization for concrete actor implementations.

        This centralizes:
        - Environment setup (grouping by namespace, composite merging)
        - FunctionManager resolution (registry fallback)
        - GuidanceManager resolution (registry fallback)
        """
        self.environments: Dict[str, "BaseEnvironment"] = self._setup_environments(
            environments=environments if environments is not None else [],
        )

        from unify.memory.integration import hooks as _memory

        if _memory.enabled():
            # A memory run has no function or guidance store: the library is
            # the memory package, so none is built and none is used.
            self.function_manager = None
            self.guidance_manager = None
            return

        from unify.manager_registry import ManagerRegistry

        self.function_manager = (
            function_manager or ManagerRegistry.get_function_manager()
        )
        self.guidance_manager = (
            guidance_manager or ManagerRegistry.get_guidance_manager()
        )

    def _setup_environments(
        self,
        *,
        environments: list["BaseEnvironment"],
    ) -> Dict[str, "BaseEnvironment"]:
        """
        Build the environment namespace dict from the provided list.

        When multiple environments share a namespace (e.g. ``"primitives"``),
        they are merged into a ``_CompositeEnvironment`` that aggregates
        their tools, prompt context, and state capture.

        Returns:
            Dict keyed by environment namespace.
        """
        from unify.actor.environments.base import _CompositeEnvironment

        # Group by namespace.
        by_ns: Dict[str, list["BaseEnvironment"]] = {}
        for env in environments:
            by_ns.setdefault(env.namespace, []).append(env)

        env_map: Dict[str, "BaseEnvironment"] = {}
        for ns, envs in by_ns.items():
            if len(envs) == 1:
                env_map[ns] = envs[0]
            else:
                env_map[ns] = _CompositeEnvironment(envs)
        return env_map

    # ─────────────────────────── Work management ────────────────────────── #

    @abstractmethod
    async def act(
        self,
        request: str,
        *,
        guidelines: Optional[str] = None,
        clarification_enabled: bool = True,
        response_format: Optional[Type[BaseModel]] = None,
        _parent_chat_context: list[dict] | None = None,
        _clarification_up_q: Optional[asyncio.Queue[str]] = None,
        _clarification_down_q: Optional[asyncio.Queue[str]] = None,
    ) -> SteerableToolHandle:
        """
        Perform work from a natural language request and return a steerable handle.

        This is the all-purpose method for engaging with resources and the world
        beyond immediate conversational context. Use ``act`` for any work that
        requires computing, retrieving, manipulating, or acting on information.

        **Capabilities include (but are not limited to):**

        - **Retrieval**: Read and analyse files in the workspace, fetch data from
          APIs and the web, run stored functions that look things up
        - **Action**: Write and run code, install packages, produce files, call
          external services, store reusable functions and procedures
        - **Combined**: Find information and then act on it (e.g., "read the CSV
          in the workspace and produce a summary chart")

        **When to use ``act``:**

        Call ``act`` whenever you need to access or manipulate anything beyond your
        immediate context. When uncertain whether information exists or an action is
        possible, **call ``act`` anyway** — if it cannot help, it will simply report
        back explaining what it couldn't do. There is no penalty for speculative
        delegation; it is better to try and fail than to not try at all.

        **Key properties:**

        - The returned handle supports pause/resume/interject/stop for mid-flight control
        - Results are returned as strings (or structured output if ``response_format`` specified)
        - The actor has access to a persistent workspace, a library of stored functions and procedures, external APIs, and system capabilities
        - Multiple ``act`` calls can run concurrently

        Args:
            request: Natural language request specifying what to do. Can be a question
                ("How many rows does report.csv have?"), a command ("Convert the
                attached spreadsheet to JSON"), or a combination ("Fetch today's
                exchange rates and write them to rates.csv").
            guidelines: Optional meta-guidance on *how* to approach the task, as
                opposed to *what* to do. Examples: "don't install new python packages",
                "use sub-agents for solving this task", "prefer simple solutions over
                complex ones". When provided, these are injected into the system prompt
                so the actor follows them throughout the session.
            clarification_enabled: Whether the actor can request clarification from
                its **caller** (i.e. whichever process holds the returned handle).
                This does NOT surface questions to the end user directly — the
                caller decides how (or whether) to resolve them.
            response_format: Optional Pydantic model for structured output.
            _parent_chat_context: Optional conversation context for continuity.
            _clarification_up_q: Queue for clarification requests (internal).
            _clarification_down_q: Queue for clarification answers (internal).

        Returns:
            A SteerableToolHandle for controlling and awaiting the result.
        """


class BaseCodeActActor(BaseActor, BaseStateManager, ABC):
    """
    Abstract contract for the CodeAct-style actor.

    Notes
    -----
    - Still shares the global actor base class: `BaseActor`.
    - Adds CodeAct-specific `act()` parameters (persistence, composition and
      storage permissions, model profile) while preserving manager
      tool-registration patterns via `BaseStateManager`.
    """

    _as_caller_description: str = (
        "the CodeActActor, executing code-first actions on behalf of the end user"
    )

    def __init__(
        self,
        *,
        environments: Optional[list["BaseEnvironment"]] = None,
        function_manager: Optional["FunctionManager"] = None,
        guidance_manager: Optional["GuidanceManager"] = None,
    ) -> None:
        BaseActor.__init__(
            self,
            environments=environments,
            function_manager=function_manager,
            guidance_manager=guidance_manager,
        )
        BaseStateManager.__init__(self)

    @abstractmethod
    async def act(
        self,
        request: str,
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
    ) -> SteerableToolHandle:
        """Perform work from a natural-language request and return a steerable handle.

        Args:
            request: Natural-language or structured request describing the work.
            llm_profile: Optional curated model profile for this actor run.
                Leave unset for the default profile, which uses the actor's
                configured model (normally ``openai/gpt-5.6-sol@openrouter`` at high
                reasoning effort). Available premium profiles are
                ``gpt_5_5_low``, ``gpt_5_5_medium``, and ``gpt_5_5_high``. Use
                ``gpt_5_5_high`` when the user explicitly asks for maximum
                thinking effort or the task is highly ambiguous/high-stakes
                enough to justify premium cost and latency. GPT-5.5 profiles
                use ``openai/gpt-5.5@openrouter`` and are priced similarly to the Sol
                default on a per-token basis; prefer them when the caller
                wants the GPT-5.5 family specifically.
        """
        raise NotImplementedError
