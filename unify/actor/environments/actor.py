"""Actor execution environment for CodeActActor.

Provides ``primitives.actor.act()`` in the sandbox for spawning isolated
inner CodeActActors for focused sub-tasks.  It is the sole occupant of the
``primitives`` namespace.

Stored functions that call ``primitives.actor.act(...)`` work through the
standard ``depends_on`` pipeline: detected at storage time by
``DependencyVisitor``, injected at runtime by ``_inject_dependencies``
via ``construct_sandbox_root("primitives")`` → ``Primitives()``.  This
is why ``_ActorRunner`` must be fully stateless (no ContextVars, no
parent state).
"""

from __future__ import annotations

import asyncio
import inspect
import logging
from typing import Dict, List, Optional, TYPE_CHECKING

from unify.actor.environments.base import (
    BaseEnvironment,
    ToolMetadata,
    build_filtered_method_docs,
)
from unify.function_manager.primitives.registry import get_registry
from unify.function_manager.primitives.scope import default_runtime_scope

if TYPE_CHECKING:
    from unify.common.async_tool_loop import SteerableToolHandle
    from unify.function_manager.function_manager import FunctionManager
    from unify.guidance_manager.guidance_manager import GuidanceManager
    from unify.function_manager.primitives import Primitives

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Parent environment resolution
# ---------------------------------------------------------------------------


def _resolve_parent_environments(
    prompt_functions: list[str] | None,
) -> tuple[list[BaseEnvironment], list[str]]:
    """Split *prompt_functions* into parent-env matches and DB-resolvable names.

    Checks the ``_CURRENT_ENVIRONMENTS`` ContextVar (set by the enclosing
    ``CodeActActor.act()``) for namespace matches.  Custom environments
    (e.g. those created via ``create_env("examplecorp", ...)``) are forwarded
    directly to the inner actor, while ``primitives`` is always left to
    the DB resolution path (it has its own scoping via ``PrimitiveScope``).

    Returns
    -------
    tuple[list[BaseEnvironment], list[str]]
        ``(forwarded_envs, remaining_prompt_functions)``
    """
    from unify.actor.execution import _CURRENT_ENVIRONMENTS

    parent_envs = _CURRENT_ENVIRONMENTS.get({})

    if not prompt_functions or not parent_envs:
        return [], list(prompt_functions or [])

    forwarded: list[BaseEnvironment] = []
    remaining: list[str] = []
    seen_namespaces: set[str] = set()

    for pf in prompt_functions:
        namespace = pf.split(".")[0]
        if namespace in parent_envs and namespace not in ("primitives",):
            if namespace not in seen_namespaces:
                forwarded.append(parent_envs[namespace])
                seen_namespaces.add(namespace)
        else:
            remaining.append(pf)

    return forwarded, remaining


# ---------------------------------------------------------------------------
# DB-based environment resolution
# ---------------------------------------------------------------------------


def _build_scoped_fm(
    discovery_scope: str | None,
) -> "FunctionManager":
    """Build a fresh FunctionManager with an optional discovery clause.

    Constructs a FunctionManager with the canonical role-gated primitive scope.
    If *discovery_scope* is provided it is applied as the ``filter_scope``
    so the inner actor's search/list/filter results are restricted
    accordingly.
    """
    from unify.function_manager.function_manager import FunctionManager as _FM

    return _FM(
        primitive_scope=default_runtime_scope(),
        filter_scope=discovery_scope,
        include_primitives=True,
    )


def _build_scoped_gm(
    guidance_scope: str | None,
) -> "GuidanceManager":
    """Build a fresh GuidanceManager with an optional discovery filter.

    Mirrors :func:`_build_scoped_fm` for the guidance store.  If
    *guidance_scope* is provided it is applied as ``filter_scope`` so the
    inner actor's search/filter results are restricted accordingly.

    Uses ``_force_new=True`` to bypass ``SingletonABCMeta`` caching —
    without this, every call returns the same global instance and the
    ``filter_scope`` argument is silently ignored.
    """
    from unify.manager_registry import ManagerRegistry

    gm = ManagerRegistry.get_guidance_manager(_force_new=True)
    if guidance_scope is not None:
        gm.filter_scope = guidance_scope
    return gm


def _resolve_prompt_guidance(
    prompt_guidance: list[str | int] | None,
) -> tuple[str | None, frozenset[int]]:
    """Resolve guidance entries by title or ID and return formatted text.

    Each identifier is looked up from the default ``GuidanceManager``.
    Strings are matched against the ``title`` column; integers against
    ``guidance_id``.  Resolved entries are concatenated as Markdown
    sections suitable for injection into the system prompt.

    Returns a tuple of ``(text, resolved_ids)`` where *text* is ``None``
    when *prompt_guidance* is empty or no entries match, and *resolved_ids*
    is the set of ``guidance_id`` values that were successfully resolved.
    """
    if not prompt_guidance:
        return None, frozenset()

    from unify.guidance_manager.guidance_manager import (
        GuidanceManager as _GM,
    )

    gm = _GM()

    sections: list[str] = []
    resolved_ids: set[int] = set()
    for identifier in prompt_guidance:
        if isinstance(identifier, int):
            rows = gm.filter(filter=f"guidance_id = {int(identifier)}", limit=1)
        else:
            title = str(identifier).replace("'", "''")
            rows = gm.filter(filter=f"title = '{title}'", limit=1)
        # Explicitly pinned guidance is injected with its complete content;
        # list reads only carry previews, so re-fetch each match in full.
        rows = [gm.get_guidance(guidance_id=g.guidance_id) for g in rows]
        for g in rows:
            parts = [f"## {g.title} [guidance_id: {g.guidance_id}]"]
            parts.append(f"\n{g.content}")
            if g.function_ids:
                parts.append(f"\nRelated functions: {g.function_ids}")
            sections.append("\n".join(parts))
            resolved_ids.add(g.guidance_id)

    text = "\n\n---\n\n".join(sections) if sections else None
    return text, frozenset(resolved_ids)


def _build_environments_from_db(
    prompt_functions: list[str] | None,
    function_manager: Optional["FunctionManager"],
) -> List[BaseEnvironment]:
    """Resolve *prompt_functions* patterns against the FunctionManager DB.

    Returns a list of environments for the inner actor.  Supports the
    ``primitives.actor`` namespace as well as compositional functions
    stored in the FunctionManager.
    """
    from unify.actor.environments.base import resolve_directly_callable
    from unify.actor.environments.function_store import FunctionStoreEnvironment

    if not prompt_functions:
        return []

    # Collect all known names from the DB via FunctionManager.
    all_known_names: set[str] = set()
    if function_manager is not None:
        try:
            fm_listing = function_manager.list_functions()
            all_known_names = set(fm_listing.keys())
        except Exception:
            pass

    if not all_known_names:
        return []

    # Resolve dotted-segment patterns to canonical names.
    matched_names = resolve_directly_callable(prompt_functions, all_known_names)

    # Bucket by environment type.
    actor_prefix = f"primitives.{ActorEnvironment.MANAGER_ALIAS}."
    actor_methods: set[str] = set()
    fm_function_names: list[str] = []

    for name in matched_names:
        if name.startswith(actor_prefix):
            actor_methods.add(name)
        elif "." not in name:
            fm_function_names.append(name)
        else:
            logger.debug(
                "Skipping dotted name %r in prompt_functions — only "
                "primitives.actor.* and bare compositional names can be "
                "resolved from DB",
                name,
            )

    # Build environments.
    envs: list[BaseEnvironment] = []

    if actor_methods:
        envs.append(
            ActorEnvironment(allowed_methods=actor_methods),
        )

    if fm_function_names and function_manager is not None:
        envs.append(
            FunctionStoreEnvironment(
                function_manager,
                function_names=fm_function_names,
            ),
        )

    return envs


# ---------------------------------------------------------------------------
# Actor runner (injected into the sandbox as ``actor``)
# ---------------------------------------------------------------------------


class _ActorRunner:
    """Runtime object accessible as ``primitives.actor`` in the sandbox.

    Fully stateless: constructs its own FunctionManager from defaults.
    No ambient ContextVars required — can be called in total isolation.

    Statelessness is load-bearing: stored compositional functions that call
    ``primitives.actor.act(...)`` are executed via
    ``FunctionManager._inject_dependencies`` →
    ``construct_sandbox_root("primitives")`` → ``Primitives()`` →
    ``primitives.actor`` → ``_ActorRunner()``.  That freshly constructed
    instance has no enclosing ``CodeActActor`` and no ContextVar state,
    so every piece of context the inner actor needs (FM scope, environments,
    permissions) must be derived from the explicit parameters passed to
    ``act()``.
    """

    _PRIMITIVE_METHODS = ("act",)

    async def act(
        self,
        request: str,
        *,
        guidelines: str | None = None,
        prompt_guidance: list[str | int] | None = None,
        guidance_scope: str | None = None,
        prompt_functions: list[str] | None = None,
        discovery_scope: str | None = None,
        response_format: type | None = None,
        timeout: float | None = None,
        can_compose: bool = True,
        can_store: bool = False,
        can_spawn_sub_agents: bool = False,
        llm_profile: str | None = None,
        _clarification_up_q: Optional[asyncio.Queue[str]] = None,
        _clarification_down_q: Optional[asyncio.Queue[str]] = None,
    ) -> "SteerableToolHandle":
        """Spawn an actor to work on a focused sub-task.

        The actor is an independent CodeActActor with its own sandbox,
        prompt, and (optionally) a curated set of directly callable
        functions.  It returns a steerable handle, allowing the caller
        to monitor progress and steer (stop, pause, resume, interject) the
        actor mid-flight.

        Actors are **steerable** — once spawned, dynamic steering helpers
        appear (stop, pause, resume, interject) so you can monitor progress and
        redirect the actor mid-flight, just like any other steerable handle.

        When to use
        -----------
        - The sub-task is genuinely open-ended: its plan must be discovered
          at runtime — changing tool discovery, unknown-state debugging,
          broad strategy selection, or exploration where each step depends
          on what the last one revealed.
        - The overall task decomposes naturally into independent sub-problems
          that benefit from focused, isolated reasoning.
        - A sub-task requires multi-step work that would clutter or distract
          the main agent's context window, or you want to isolate its
          execution state (sessions, variables) from the main sandbox and
          keep the main context clean for high-level orchestration.

        When NOT to use
        ---------------
        - The sub-step is a bounded semantic judgment inside control flow
          you already command — classify, extract, score, route, summarize
          into fields, draft. That is a stateless
          ``query_llm(..., response_format=...)`` call nested in your own
          code, not an agent: an actor per bounded judgment pays agent
          overhead (its own prompt, loop, and reasoning) for work a single
          call does better. Plain code -> ``query_llm(...)`` -> an actor is
          a dial — take the lowest notch that preserves the judgment.
        - A stable procedure for the sub-task already exists or should:
          discover and run the stored function (with ``query_llm(...)`` at
          its semantic joints) instead of re-deriving the procedure
          agentically on every run.
        - The task is simple enough to handle directly with ``execute_code``.
        - You need intermediate results from the sub-task to inform logic in
          the same code block (use ``execute_code`` with stateful sessions).
        - The sub-task is trivial (single tool call) — the overhead of an
          actor is not worth it.
        - The sub-task is your own task, or most of it, restated. A child
          actor has the same tools and the same unknowns you have, so
          handing it the task you were given adds a layer of prompts and
          no knowledge, and the child may do the same again. Explore
          yourself with ``execute_code`` (list the directory, read the
          files, grep) and spawn actors only for parts that are
          independent of each other or of your own next step.

        Best practices
        --------------
        - Write a **clear, self-contained request description**. The actor
          does not see the parent's conversation or sandbox state. Include
          all relevant context in the request string.
        - Use **guidelines** to inject persistent behavioral directives
          (e.g., "Be extremely thorough", "Respond in JSON") that the
          actor must follow throughout its session.
        - Use **prompt_guidance** to pass specific procedural guides from
          the GuidanceManager that the actor should follow.
        - Set an appropriate **timeout** for the expected complexity.

        Parameters
        ----------
        request : str
            A clear, self-contained description of what the actor should
            accomplish. Be specific and include all necessary context, because
            the actor does **not** share the parent agent's conversation
            history or session state.
        guidelines : str, optional
            Persistent behavioral directives injected into the actor's
            system prompt. Unlike the request (which describes *what* to
            do), guidelines describe *how* to behave throughout the
            session — e.g., output format, thoroughness level, tone, or
            constraints. When omitted, no additional guidelines are set.
        prompt_guidance : list[str | int], optional
            Guidance entries to resolve from the GuidanceManager and
            inject into the actor's system prompt.  Each element is
            either a guidance **title** (``str``) or a **guidance_id**
            (``int``).  Resolved entries are formatted as Markdown and
            prepended to the ``guidelines`` section so the actor sees
            them as persistent behavioral context.

            Use this to pass curated procedural knowledge (step-by-step
            instructions, SOPs, composition strategies) to the actor
            without requiring it to discover them via search.

            Prompt-injected guidance entries are automatically excluded
            from the actor's GuidanceManager search/filter results, so
            they will not appear as duplicates during discovery.

            Example::

                prompt_guidance=["Excel Processing Guide", 42]

            When omitted, no guidance is injected and the actor relies
            on GuidanceManager discovery tools (subject to
            ``guidance_scope``).
        guidance_scope : str, optional
            A SQL ``WHERE`` clause that restricts which guidance entries the
            actor can discover via GuidanceManager search/filter (e.g.
            ``"title LIKE '%financial%'"`` or ``"guidance_id < 100"``).

            Mirrors ``discovery_scope`` for functions.  When provided,
            only guidance matching this expression is visible to the
            actor's GuidanceManager tools.  When omitted, all stored
            guidance is discoverable.
        prompt_functions : list[str], optional
            Functions or environments to place directly in the actor's
            system prompt, making them immediately callable without any
            discovery step.

            Use this for the functions most critical to the sub-task —
            the ones you actively want the actor to reach for.
            Because they appear directly in the prompt, they receive the
            actor's full attention and are the first tools it will
            consider.

            Prompt-injected functions are automatically excluded from the
            actor's FunctionManager search index, so they will not
            appear as duplicate results during discovery.

            Names are resolved in two stages:

            1. **Parent environments** — if the name matches a custom
               environment namespace from the calling agent (e.g., a
               namespace registered via ``create_env("examplecorp", ...)``),
               that environment is forwarded directly to the actor.
               The ``"primitives"`` namespace is never forwarded this
               way (it always resolves through the DB path).

            2. **FunctionManager DB** — remaining names use
               dotted-segment matching against function names stored in
               the database:

               - ``"primitives"`` — all primitives
               - ``"primitives.actor"`` — actor delegation
               - ``"primitives.actor.act"`` — just actor.act
               - ``"alpha"`` — a specific stored function

            Any function stored in the database (primitives or
            compositional) can be listed here.
            Functions not listed are still discoverable via
            FunctionManager search (subject to ``discovery_scope``).

            When omitted, the actor receives no prompt-injected
            functions and relies entirely on FunctionManager discovery.
        discovery_scope : str, optional
            A SQL ``WHERE`` clause that restricts which stored functions the
            actor can discover via FunctionManager search/list/filter (e.g.
            ``"docstring LIKE '%data%'"`` or ``"name LIKE 'report_%'"``).

            When provided, only functions matching this expression are
            visible to the actor's discovery tools.  When omitted, all
            stored functions are discoverable.
        response_format : type (BaseModel subclass), optional
            A Pydantic ``BaseModel`` subclass that defines the required
            structure of the actor's final response.  When set, the
            actor is forced to submit its answer via a dedicated
            ``final_response`` tool whose schema matches the model.
            Validation failures are fed back to the actor so it can
            retry.  The caller receives a parsed Pydantic instance from
            ``await handle.result()``.
        timeout : float, optional
            Maximum seconds for the actor to complete.  When omitted
            the actor runs without a time limit.
        can_compose : bool, default True
            Whether the actor can write and execute arbitrary code via
            ``execute_code``. Set to False to restrict the actor to
            only discovering and executing stored functions.
        can_store : bool, default False
            Whether a post-completion review loop should run to identify
            and store reusable functions and guidance from the actor's
            trajectory. Storage is always deferred to a dedicated second
            loop after the main task completes.
        can_spawn_sub_agents : bool, default False
            Whether the actor can itself spawn deeper actors.
            Use with caution to avoid excessive nesting.
        llm_profile : str, optional
            Optional curated model profile for this actor run. Leave unset for
            the default profile, normally ``openai/gpt-5.6-sol@openrouter`` at high
            reasoning effort. Available premium profiles are ``gpt_5_5_low``,
            ``gpt_5_5_medium``, and ``gpt_5_5_high``. Use ``gpt_5_5_high``
            when the sub-task itself needs maximum thinking effort. This
            parameter is explicit: sub-agents do not automatically inherit the
            parent actor's profile.

        Returns
        -------
        SteerableToolHandle
            A live handle to the running actor.  The handle supports
            mid-flight steering (stop, pause, resume, interject).  The
            final string result is surfaced when the actor completes.
        """
        from unify.actor.code_act_actor import CodeActActor
        from unify.actor.execution import _PARENT_CHAT_CONTEXT

        effective_timeout = timeout if timeout is not None else 1000

        # Pick up parent chat context if running inside a CodeActActor
        # sandbox.  Gracefully degrades to None in standalone usage.
        _parent_chat_context = _PARENT_CHAT_CONTEXT.get(None)

        # Build a fresh FM scoped by discovery_scope (no parent inheritance).
        inner_fm = _build_scoped_fm(discovery_scope)

        # Resolve prompt_functions: first check parent environments for
        # custom namespaces (e.g. create_env-based services), then resolve
        # the remainder against the FunctionManager DB.
        forwarded_envs, db_prompt_functions = _resolve_parent_environments(
            prompt_functions,
        )
        inner_envs = _build_environments_from_db(db_prompt_functions, inner_fm)
        inner_envs.extend(forwarded_envs)

        # Optionally allow nested actor spawning (skip if prompt_functions
        # already resolved an ActorEnvironment to avoid duplicates).
        if can_spawn_sub_agents and not any(
            isinstance(e, ActorEnvironment) for e in inner_envs
        ):
            inner_envs.append(ActorEnvironment())

        # Resolve prompt_guidance entries and merge with guidelines.
        guidance_text, resolved_guidance_ids = _resolve_prompt_guidance(
            prompt_guidance,
        )
        effective_guidelines = guidelines or ""
        if guidance_text:
            effective_guidelines = (
                f"{guidance_text}\n\n{effective_guidelines}".strip() or None
            )
        else:
            effective_guidelines = effective_guidelines or None

        # Build a scoped GuidanceManager for subagent discovery.
        inner_gm = _build_scoped_gm(guidance_scope)
        if resolved_guidance_ids:
            inner_gm.exclude_ids = resolved_guidance_ids

        # Create inner CodeActActor.
        inner_actor = CodeActActor(
            environments=inner_envs,
            function_manager=inner_fm,
            guidance_manager=inner_gm,
            can_compose=bool(can_compose),
            can_store=bool(can_store),
            timeout=effective_timeout,
            prompt_caching=["system", "tools", "messages"],
        )

        handle = await inner_actor.act(
            request,
            guidelines=effective_guidelines,
            response_format=response_format,
            clarification_enabled=True,
            _parent_chat_context=_parent_chat_context,
            _clarification_up_q=_clarification_up_q,
            _clarification_down_q=_clarification_down_q,
            llm_profile=llm_profile,
        )

        # Attach inner actor cleanup to the handle's lifecycle.
        _unwrapped = getattr(handle, "__wrapped__", handle)

        if hasattr(_unwrapped, "_lifecycle_task"):
            # can_store=True: handle is a _StorageCheckHandle.  result()
            # resolves after the task phase; storage runs in the background.
            # Tie cleanup to the full lifecycle so close() waits for storage.
            def _cleanup_when_done(_task: asyncio.Task) -> None:
                asyncio.ensure_future(inner_actor.close())

            _unwrapped._lifecycle_task.add_done_callback(_cleanup_when_done)
        else:
            # can_store=False: no storage phase — cleanup when result returns.
            _original_result = _unwrapped.result

            async def _result_with_cleanup():
                try:
                    return await _original_result()
                finally:
                    await inner_actor.close()

            _unwrapped.result = _result_with_cleanup  # type: ignore[assignment]

        return handle


# ---------------------------------------------------------------------------
# Environment wrapper
# ---------------------------------------------------------------------------


class ActorEnvironment(BaseEnvironment):
    """The ``primitives`` environment: actor spawning via ``primitives.actor``.

    Injects a ``Primitives`` object into the sandbox so that
    ``primitives.actor.act(...)`` spawns isolated inner CodeActActors.
    Its method docs are inlined in the prompt, so its tools are excluded
    from FunctionManager search (``prompt_documented_names`` is unset).

    Parameters
    ----------
    allowed_methods : set[str] | None
        Optional set of fully-qualified method names to expose (e.g.,
        ``{"primitives.actor.act"}``).  When set, only these methods
        appear in ``get_tools()`` and ``get_prompt_context()``.
        When ``None`` (default), all actor methods are exposed.
    clarification_up_q / clarification_down_q : asyncio.Queue | None
        Per-call clarification channel injected into sandbox calls.
    """

    NAMESPACE = "primitives"
    MANAGER_ALIAS = "actor"

    def __init__(
        self,
        *,
        allowed_methods: Optional[set[str]] = None,
        clarification_up_q: Optional[asyncio.Queue[str]] = None,
        clarification_down_q: Optional[asyncio.Queue[str]] = None,
    ) -> None:
        from unify.function_manager.primitives import Primitives, PrimitiveScope

        self._allowed_methods = frozenset(allowed_methods) if allowed_methods else None
        self._primitives = Primitives(
            primitive_scope=PrimitiveScope(
                scoped_managers=frozenset({self.MANAGER_ALIAS}),
            ),
        )
        super().__init__(
            instance=self._primitives,
            namespace=self.NAMESPACE,
            clarification_up_q=clarification_up_q,
            clarification_down_q=clarification_down_q,
        )

    @property
    def allowed_methods(self) -> Optional[frozenset[str]]:
        """The per-method filter, or ``None`` when every actor method is exposed."""
        return self._allowed_methods

    def get_instance(self) -> "Primitives":
        """Return the scoped ``Primitives`` instance injected into the sandbox."""
        return self._primitives

    def get_tools(self) -> Dict[str, ToolMetadata]:
        registry = get_registry()
        tools: Dict[str, ToolMetadata] = {}
        for method_name in _ActorRunner._PRIMITIVE_METHODS:
            fq_name = f"{self.NAMESPACE}.{self.MANAGER_ALIAS}.{method_name}"
            if (
                self._allowed_methods is not None
                and fq_name not in self._allowed_methods
            ):
                continue
            tools[fq_name] = ToolMetadata(
                name=fq_name,
                is_impure=True,
                is_steerable=True,
                function_id=registry.get_function_id(
                    self.MANAGER_ALIAS,
                    method_name,
                ),
                function_context="primitive",
            )
        return tools

    def get_prompt_context(self) -> str:
        """Generate prompt context from the ``act()`` method's docstring."""
        if self._allowed_methods is not None:
            filtered_docs = build_filtered_method_docs(
                self._allowed_methods,
                self.NAMESPACE,
            )
            return filtered_docs

        registry = get_registry()
        sig_str = registry._format_method_signature(
            _ActorRunner,
            "act",
        )
        full_doc = inspect.getdoc(_ActorRunner.act) or ""
        filtered_doc = registry._filter_internal_params_from_docstring(full_doc)

        fq_prefix = f"{self.NAMESPACE}.{self.MANAGER_ALIAS}"
        lines = [
            f"### `{fq_prefix}` — Actor Delegation\n",
            "The one `primitives.*` surface. Calls return a "
            "`SteerableToolHandle`: make it the last expression of "
            "`execute_code` (or call it via `execute_function`) so the outer "
            "loop can steer it — `await handle.result()` only when the code "
            "itself composes on the result.\n",
        ]
        lines.append(f"**`{fq_prefix}.act{sig_str}`**")
        if filtered_doc:
            for doc_line in filtered_doc.splitlines():
                lines.append(f"  {doc_line}")
        return "\n".join(lines)
