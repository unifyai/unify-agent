from __future__ import annotations

import asyncio
import inspect
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Literal, Optional, Set

from pydantic import BaseModel


def matches_segment(pattern: str, canonical_name: str) -> bool:
    """Check if *pattern* matches *canonical_name* using dotted-segment rules.

    A pattern matches a canonical name if it is either an exact match or a
    dotted ancestor (i.e., a complete prefix up to a ``.`` boundary).

    Examples::

        matches_segment("primitives", "primitives.actor.act")      # True
        matches_segment("primitives.actor", "primitives.actor.act")  # True
        matches_segment("primitives.actor.act", "primitives.actor.act")  # True
        matches_segment("primitives.act", "primitives.actor.act")   # False
        matches_segment("functions", "functions.alpha")                 # True
        matches_segment("functions.alpha", "functions.alpha")           # True
    """
    return canonical_name == pattern or canonical_name.startswith(pattern + ".")


def resolve_directly_callable(
    patterns: List[str],
    all_tool_names: Set[str],
) -> Set[str]:
    """Expand a list of dotted-segment patterns into matching canonical tool names.

    Args:
        patterns: List of patterns (e.g., ``["primitives.actor", "alpha"]``).
        all_tool_names: Complete set of canonical tool names across all environments.

    Returns:
        Set of canonical tool names matched by the patterns.

    Raises:
        ValueError: If any pattern matches zero tool names (likely a typo or
            a function the agent hasn't encountered).
    """
    matched: Set[str] = set()
    for pat in patterns:
        hits = {name for name in all_tool_names if matches_segment(pat, name)}
        if not hits:
            raise ValueError(
                f"environment pattern {pat!r} did not match any known "
                f"tool. Available tools: {sorted(all_tool_names)}",
            )
        matched |= hits
    return matched


def build_filtered_method_docs(
    allowed_methods: frozenset[str],
    namespace: str = "primitives",
) -> str:
    """Build method-level documentation for only the specified fully-qualified methods.

    Uses the ``ToolSurfaceRegistry`` to introspect method signatures and
    docstrings for each allowed method.
    """
    from unify.function_manager.primitives.registry import get_registry

    registry = get_registry()

    allowed_aliases: dict[str, list[str]] = {}
    for fq in sorted(allowed_methods):
        parts = fq.split(".")
        if len(parts) != 3 or parts[0] != namespace:
            continue
        alias, method = parts[1], parts[2]
        allowed_aliases.setdefault(alias, []).append(method)

    if not allowed_aliases:
        return ""

    lines = ["### Method Reference\n"]
    for alias in sorted(allowed_aliases):
        spec = registry.get_manager_spec(alias)
        mgr_cls = (
            registry._load_manager_class(spec.primitive_class_path) if spec else None
        )

        lines.append(f"\n#### `{namespace}.{alias}`")
        if spec:
            lines.append(f"*{spec.domain}* — {spec.description}")

        for method_name in sorted(allowed_aliases[alias]):
            heading = registry._format_method_heading(
                mgr_cls,
                method_name,
                f"{namespace}.{alias}.{method_name}",
            )
            full_doc = registry._extract_method_docstring(mgr_cls, method_name)
            compact_doc = registry._extract_summary_and_params(full_doc)
            lines.append(f"\n**`{heading}`**")
            if compact_doc:
                for doc_line in compact_doc.splitlines():
                    lines.append(f"  {doc_line}")

    return "\n".join(lines)


class ToolMetadata(BaseModel):
    """Metadata describing a tool's behavior and safety characteristics.

    Attributes:
        name: Fully-qualified tool name as used in the Actor execution sandbox.
        is_impure: True if the tool can cause side effects.
        is_steerable: True if calling the tool may return a steerable handle.
        docstring: Tool documentation string (if available).
        signature: Human-readable signature string (if available).
        function_id: Optional cross-reference to a stored FunctionManager function.
            When set, indicates this environment tool corresponds to a function in
            the FunctionManager backend, enabling automatic exclusion from
            FunctionManager search/list/filter results to prevent overlap.
            Must be paired with ``function_context`` to identify which table
            the ID belongs to (IDs are only unique within a table).
        function_context: Which table ``function_id`` belongs to. Required
            when ``function_id`` is set. ``"primitive"`` for primitive
            namespace methods (the ``primitives`` table), ``"compositional"``
            for stored functions (the ``functions`` table).
    """

    name: str
    is_impure: bool
    is_steerable: bool = False
    docstring: Optional[str] = None
    signature: Optional[str] = None
    function_id: Optional[int] = None
    function_context: Optional[Literal["primitive", "compositional"]] = None


def _callable_accepts_clarification_kwargs(fn: Any) -> bool:
    """
    Return True if `fn` appears to accept clarification queue kwargs.

    We only inject queues into callables that declare `_clarification_up_q` /
    `_clarification_down_q` explicitly or accept `**kwargs`. This avoids breaking
    other async utilities that do not accept these keyword arguments.
    """
    try:
        sig = inspect.signature(fn)
    except Exception:
        return False

    params = sig.parameters
    if "_clarification_up_q" in params or "_clarification_down_q" in params:
        return True

    for p in params.values():
        if p.kind == inspect.Parameter.VAR_KEYWORD:
            return True

    return False


_CLAR_GLOBAL_MISSING = object()


def _make_sandbox_clarification_fn(global_state: Dict[str, Any]):
    """An awaitable the sandbox can call to ask the user and block on it.

    The queues already reached the sandbox as ``__clarification_up_q__`` /
    ``__clarification_down_q__``, but nothing exposed a way to use them, so
    generated code had no route to the clarification channel. Its only option
    was to write the question out through whatever API the task happened to
    describe — which asks without waiting, because a script cannot receive a
    reply. The question goes out, execution continues, and the work proceeds
    on a guess that looks like it was checked.

    Awaiting this suspends the Python program at the call site, exactly as the
    JSON tool suspends the loop. The actor is code-first, so an ambiguity
    found halfway through a program should not require abandoning the program
    to ask about it.

    Queues are read at call time rather than captured, so the function stays
    correct across the bind/restore cycle.

    Deadlock exposure is the same as the JSON ``request_clarification`` tool:
    both put on the same per-call up-queue and await the same down-queue with
    no timeout, drained by the same ``_handle_clarification`` →
    ``handle._clar_q`` path. The sandbox call sits under more stack frames,
    but it is one await on one queue either way, and whatever answers one
    answers the other.

    It does differ in one respect, and not in its favour: the JSON tool fires
    ``on_request`` / ``on_answer``, which publish ``ManagerMethod`` events
    carrying the call id. A clarification raised from code is therefore
    invisible on the event bus while an identical one raised from the tool is
    visible. Closing that needs the call id threaded down to this bind, which
    is a separate change.
    """

    async def request_clarification(question: str) -> str:
        """Ask the caller a question and wait for their answer.

        Blocks this call site until an answer arrives and then returns it, so
        the surrounding code resumes with the answer in hand.
        """
        up = global_state.get("__clarification_up_q__")
        down = global_state.get("__clarification_down_q__")
        if up is None or down is None:
            raise RuntimeError(
                "No clarification channel is available in this context — "
                "proceed with a stated assumption instead of asking.",
            )
        await up.put(str(question))
        return await down.get()

    return request_clarification


def bind_sandbox_clarification_queues(
    global_state: Dict[str, Any],
    up_q: asyncio.Queue[str],
    down_q: Optional[asyncio.Queue[str]],
) -> Dict[str, Any]:
    """Point sandbox manager injectors at one tool-call's clarification queues.

    Used by ``execute_code`` / ``execute_function`` so nested manager
    clarifications write into the outer tool's per-call channel (mailbox A)
    that the async tool loop already watches. Returns a token for
    :func:`restore_sandbox_clarification_queues`.
    """
    previous_injectors: Dict[str, tuple[Any, Any]] = {}
    for key, val in list(global_state.items()):
        if isinstance(val, _ClarificationQueueInjector):
            previous_injectors[key] = (val._clar_up_q, val._clar_down_q)
            object.__setattr__(val, "_clar_up_q", up_q)
            object.__setattr__(val, "_clar_down_q", down_q)
    previous_globals = (
        global_state.get("__clarification_up_q__", _CLAR_GLOBAL_MISSING),
        global_state.get("__clarification_down_q__", _CLAR_GLOBAL_MISSING),
    )
    previous_fn = global_state.get("request_clarification", _CLAR_GLOBAL_MISSING)
    global_state["__clarification_up_q__"] = up_q
    global_state["__clarification_down_q__"] = down_q
    global_state["request_clarification"] = _make_sandbox_clarification_fn(global_state)
    return {
        "injectors": previous_injectors,
        "globals": previous_globals,
        "fn": previous_fn,
    }


def restore_sandbox_clarification_queues(
    global_state: Dict[str, Any],
    token: Dict[str, Any],
) -> None:
    """Undo :func:`bind_sandbox_clarification_queues`."""
    for key, (prev_up, prev_down) in (token.get("injectors") or {}).items():
        val = global_state.get(key)
        if isinstance(val, _ClarificationQueueInjector):
            object.__setattr__(val, "_clar_up_q", prev_up)
            object.__setattr__(val, "_clar_down_q", prev_down)
    prev_up, prev_down = token.get("globals") or (
        _CLAR_GLOBAL_MISSING,
        _CLAR_GLOBAL_MISSING,
    )
    if prev_up is _CLAR_GLOBAL_MISSING:
        global_state.pop("__clarification_up_q__", None)
    else:
        global_state["__clarification_up_q__"] = prev_up
    if prev_down is _CLAR_GLOBAL_MISSING:
        global_state.pop("__clarification_down_q__", None)
    else:
        global_state["__clarification_down_q__"] = prev_down
    prev_fn = token.get("fn", _CLAR_GLOBAL_MISSING)
    if prev_fn is _CLAR_GLOBAL_MISSING:
        global_state.pop("request_clarification", None)
    else:
        global_state["request_clarification"] = prev_fn


class _ClarificationQueueInjector:
    """
    Lightweight wrapper that injects clarification queues into manager calls.

    This is intentionally minimal:
    - No caching
    - No logging
    - No pane registration
    - Just queue injection (when supported by the target callable)
    """

    _DO_NOT_WRAP_TYPES: tuple[type, ...] = (
        str,
        bytes,
        bytearray,
        int,
        float,
        bool,
        dict,
        list,
        tuple,
        set,
        frozenset,
        type(None),
    )

    def __init__(
        self,
        *,
        target: Any,
        clarification_up_q: asyncio.Queue[str],
        clarification_down_q: Optional[asyncio.Queue[str]],
    ):
        object.__setattr__(self, "_target", target)
        object.__setattr__(self, "_clar_up_q", clarification_up_q)
        object.__setattr__(self, "_clar_down_q", clarification_down_q)

    def __getattr__(self, name: str) -> Any:
        attr = getattr(self._target, name)

        # Pass through private/dunder attributes directly.
        if name.startswith("_"):
            return attr

        # Wrap callables so we can inject queues at call time.
        if callable(attr):
            return self._wrap_callable(attr)

        # For nested objects (e.g. `primitives.actor` returning a runner), return
        # another injector so `primitives.actor.act(...)` also gets queue injection.
        return self._maybe_wrap_object(attr)

    def __setattr__(self, name: str, value: Any) -> None:
        # Keep wrapper transparent to normal attribute assignment.
        if name.startswith("_"):
            object.__setattr__(self, name, value)
        else:
            setattr(self._target, name, value)

    def __repr__(self) -> str:
        return f"<ClarificationQueueInjector target={type(self._target).__name__}>"

    def _maybe_wrap_object(self, obj: Any) -> Any:
        if isinstance(obj, self._DO_NOT_WRAP_TYPES):
            return obj
        if inspect.ismodule(obj):
            return obj
        if isinstance(obj, type):
            return obj
        if isinstance(obj, _ClarificationQueueInjector):
            return obj
        return _ClarificationQueueInjector(
            target=obj,
            clarification_up_q=self._clar_up_q,
            clarification_down_q=self._clar_down_q,
        )

    def _inject_queues(self, *, fn: Any, kwargs: Dict[str, Any]) -> None:
        if "_clarification_up_q" in kwargs or "_clarification_down_q" in kwargs:
            return
        if not _callable_accepts_clarification_kwargs(fn):
            return
        kwargs["_clarification_up_q"] = self._clar_up_q
        kwargs["_clarification_down_q"] = self._clar_down_q

    def _wrap_callable(self, fn: Any) -> Any:
        if asyncio.iscoroutinefunction(fn):

            async def _async_wrapper(*args: Any, **kwargs: Any) -> Any:
                self._inject_queues(fn=fn, kwargs=kwargs)
                return await fn(*args, **kwargs)

            return _async_wrapper

        def _sync_wrapper(*args: Any, **kwargs: Any) -> Any:
            self._inject_queues(fn=fn, kwargs=kwargs)
            return fn(*args, **kwargs)

        return _sync_wrapper


class BaseEnvironment(ABC):
    """Abstract interface for execution environments.

    An environment encapsulates a domain of tools (nested actors, the function
    store, custom adapters) and provides:
    - a namespace to inject into the plan execution sandbox
    - metadata for tools (purity/steerability)
    - a prompt context section describing usage patterns for those tools

    NOTE: proxying/caching/logging is owned by the Actor, not the environment.
    """

    def __init__(
        self,
        *,
        instance: Any,
        namespace: str,
        clarification_up_q: Optional[asyncio.Queue[str]] = None,
        clarification_down_q: Optional[asyncio.Queue[str]] = None,
    ) -> None:
        self._instance = instance
        self._namespace = namespace
        self._clarification_up_q = clarification_up_q
        self._clarification_down_q = clarification_down_q

    @property
    def namespace(self) -> str:
        """Global variable name injected into the sandbox (e.g. "primitives")."""
        return self._namespace

    def get_instance(self) -> Any:
        """Return the object injected into the sandbox under `namespace`."""
        return self._instance

    def get_sandbox_instance(self) -> Any:
        """
        Return instance for sandbox injection.

        If clarification queues are configured, returns a lightweight wrapper
        that injects `_clarification_up_q` / `_clarification_down_q` into manager
        method calls (when supported).
        """
        instance = self.get_instance()
        clar_up_q = getattr(self, "_clarification_up_q", None)
        if clar_up_q is None:
            return instance

        clar_down_q = getattr(self, "_clarification_down_q", None)
        return _ClarificationQueueInjector(
            target=instance,
            clarification_up_q=clar_up_q,
            clarification_down_q=clar_down_q,
        )

    @abstractmethod
    def get_tools(self) -> Dict[str, ToolMetadata]:
        """Return metadata for tools exposed by this environment.

        The returned keys MUST be fully-qualified tool names as used in execution,
        so callers can look up metadata by the same string that appears in logs.
        """

    @abstractmethod
    def get_prompt_context(self) -> str:
        """Return Markdown-formatted rules/examples for using this environment."""


class _CompositeEnvironment(BaseEnvironment):
    """Merges multiple environments sharing the same namespace.

    When several environments share the ``"primitives"`` namespace, this
    wrapper aggregates their tool metadata and prompt context while
    injecting a single ``Primitives`` instance into the sandbox.
    """

    def __init__(
        self,
        envs: List["BaseEnvironment"],
        *,
        clarification_up_q: Optional[asyncio.Queue[str]] = None,
        clarification_down_q: Optional[asyncio.Queue[str]] = None,
    ) -> None:
        self._envs = envs
        primary = self._build_primary_instance(envs)
        super().__init__(
            instance=primary,
            namespace=envs[0].namespace,
            clarification_up_q=clarification_up_q,
            clarification_down_q=clarification_down_q,
        )

    @staticmethod
    def _build_primary_instance(envs: List["BaseEnvironment"]) -> Any:
        """Return the broadest-scoped Primitives instance from sub-envs."""
        from unify.function_manager.primitives import Primitives, PrimitiveScope

        merged_managers: Set[str] = set()
        for env in envs:
            instance = env.get_instance()
            if isinstance(instance, Primitives):
                merged_managers |= instance.primitive_scope.scoped_managers

        if merged_managers:
            return Primitives(
                primitive_scope=PrimitiveScope(
                    scoped_managers=frozenset(merged_managers),
                ),
            )
        # Fallback: first environment's instance.
        return envs[0].get_instance()

    @property
    def sub_environments(self) -> List["BaseEnvironment"]:
        """Expose wrapped environments for introspection."""
        return list(self._envs)

    @property
    def prompt_documented_names(self) -> frozenset[str]:
        """Tool names the merged prompt documents inline.

        The actor's FunctionManager search-overlap exclusion reads this to
        decide which tools to hide from search results. Sub-environments
        that declare ``prompt_documented_names`` contribute exactly that
        subset; ones that don't are treated as fully prompt-documented
        (their whole tool surface is contributed), preserving each
        sub-environment's standalone semantics. Without this forwarding the
        composite would fall back to "everything documented" and re-exclude
        primitives its sub-environments deliberately left searchable.
        """
        documented: Set[str] = set()
        for env in self._envs:
            names = getattr(env, "prompt_documented_names", None)
            if names is None:
                documented |= set(env.get_tools().keys())
            else:
                documented |= set(names)
        return frozenset(documented)

    def get_tools(self) -> Dict[str, ToolMetadata]:
        merged: Dict[str, ToolMetadata] = {}
        for env in self._envs:
            merged.update(env.get_tools())
        return merged

    def get_prompt_context(self) -> str:
        parts = [env.get_prompt_context() for env in self._envs]
        return "\n\n".join(p for p in parts if p and p.strip())
