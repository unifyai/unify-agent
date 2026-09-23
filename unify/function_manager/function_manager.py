import ast
import asyncio
import builtins
import concurrent.futures
from datetime import datetime, timezone
import inspect
import functools
import logging
import sqlite3
import threading

from secrets import token_hex
from typing import (
    Any,
    Callable,
    Dict,
    FrozenSet,
    List,
    Literal,
    Optional,
    Set,
    Sequence,
    Tuple,
    Union,
)
from unify import db
from ..common.sql_filters import and_clauses, invalid_filter_error, not_in, or_clauses
from ..common.text_search import SIMILARITY_FIELD, rank_by_text
from ..common.tool_outcome import ToolErrorException
from .activation import (
    ActivationSettings,
    activation,
    in_scope,
    merged_usage,
    rank_score,
)
from .execution_env import ENVIRONMENT_MODULES, create_base_globals
from .steering import (
    DEFAULT_TOOL_NAMESPACES,
    ExecutionStopped,
    MemoisedDispatch,
    active_session,
    bind_session,
    instrument,
    restore_session,
    run_with_steering,
)
from unify import environment
from packaging.requirements import InvalidRequirement
from .dependency_analysis import (
    collect_dependencies_from_function_node,
    detect_third_party_imports,
)
from .types.function import Function
from .source_labels import compile_function_source
from .base import BaseFunctionManager
from ..common.stale_reason import (
    StaleReason,
    coerce_stale_reasons,
    merge_stale_reasons,
)
from unify.function_manager.primitives.scope import (
    PrimitiveScope,
    default_runtime_scope,
)
from unify.function_manager.primitives.registry import get_registry

logger = logging.getLogger(__name__)

_FUNCTION_SELECT = f"SELECT {', '.join(db.FUNCTION_COLUMNS)} FROM all_functions"

_PRIMITIVES_SEEDED_FOR: set[str] = set()


def _seed_primitives() -> None:
    """Make the ``primitives`` table equal to the registry, once per store."""
    path = db.store_path()
    if path in _PRIMITIVES_SEEDED_FOR:
        return
    rows = get_registry().collect_primitives().values()
    with db.transaction() as conn:
        conn.execute("DELETE FROM primitives")
        conn.executemany(
            "INSERT INTO primitives (function_id, name, argspec, docstring,"
            " primitive_class, primitive_method, metadata) VALUES (?, ?, ?, ?, ?, ?, ?)",
            [
                (
                    int(row["function_id"]),
                    row["name"],
                    row.get("argspec") or "",
                    row.get("docstring") or "",
                    row["primitive_class"],
                    row["primitive_method"],
                    db.dumps(row.get("metadata") or {}),
                )
                for row in rows
            ],
        )
    _PRIMITIVES_SEEDED_FOR.add(path)


def _decode_function_row(row: Dict[str, Any]) -> Dict[str, Any]:
    db.decode(row, db.FUNCTION_JSON_COLUMNS)
    row["is_primitive"] = bool(row.get("is_primitive"))
    for column in (
        "depends_on",
        "stale_reasons",
        "dependencies",
        "third_party_imports",
        "usage_recent_calls",
    ):
        if row.get(column) is None:
            row[column] = []
    if row.get("metadata") is None:
        row["metadata"] = {}
    return row


def _attach_guidance_ids(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Add ``guidance_ids`` (the guidance entries citing each function) to rows."""
    linked: Dict[int, List[int]] = {}
    for link in db.query(
        "SELECT g.guidance_id AS guidance_id, je.value AS function_id"
        " FROM guidance g, json_each(g.function_ids) je",
    ):
        linked.setdefault(int(link["function_id"]), []).append(int(link["guidance_id"]))
    for row in rows:
        row["guidance_ids"] = (
            []
            if row.get("is_primitive")
            else sorted(linked.get(int(row["function_id"]), []))
        )
    return rows


_FUNCTION_JSON = set(db.FUNCTION_JSON_COLUMNS)


def _encode_function_values(entry: Dict[str, Any]) -> Dict[str, Any]:
    return {
        column: (
            db.dumps(value) if column in _FUNCTION_JSON and value is not None else value
        )
        for column, value in entry.items()
    }


# The fields a search query's words are looked for in, per function row.
SEARCHED_FUNCTION_FIELDS = ("name", "docstring", "metadata")


class _LineageTrackedFunction:
    """Boundary wrapper for FunctionManager callables injected into CodeActActor sandboxes.

    This wrapper preserves hierarchical lineage across mixed execution, e.g.:

        CodeActActor.act -> execute_code -> <function> -> primitives.actor.act -> ...

    It is injected into the Python namespace **in place of** the raw callable so that
    inter-function calls (function A calling function B) still pass through a boundary that:
    - updates `TOOL_LOOP_LINEAGE` (ContextVar)
    - emits a concise boundary log line for terminal debugging

    Note: async functions can be awaited in a different task context than the call-site.
    ContextVar tokens are only valid in the context they were created, so this wrapper sets
    lineage around coroutine construction (call-site) and again inside the awaited coroutine
    (execution-site).
    """

    def __init__(
        self,
        wrapped_callable: Callable[..., Any],
        function_name: str,
        on_call: Optional[Callable[[], None]] = None,
    ):
        self._wrapped = wrapped_callable
        self._function_name = function_name
        # Usage-trace hook: this class is the one layer every boundary call
        # already passes through, and its __getattr__ delegation keeps proxy
        # identity intact, which is why the trace records here.
        self._on_call = on_call

        # Preserve introspection attributes.
        self.__name__ = function_name
        self.__doc__ = getattr(wrapped_callable, "__doc__", None)
        self.__wrapped__ = wrapped_callable

    def __getattr__(self, name: str) -> Any:
        # Preserve wrapped callable API (e.g. proxy state-mode helpers).
        return getattr(self._wrapped, name)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        if self._on_call is not None:
            try:
                self._on_call()
            except Exception:  # noqa: BLE001 - metering must never break a call
                pass
        # Local imports to avoid import-time cycles.
        from unify.common._async_tool.loop_config import TOOL_LOOP_LINEAGE
        from unify.common.hierarchical_logger import log_boundary_event

        suffix = token_hex(2)

        parent = TOOL_LOOP_LINEAGE.get([])
        parent_lineage = list(parent) if isinstance(parent, list) else []
        hierarchy = [*parent_lineage, f"{self._function_name}({suffix})"]

        try:
            log_boundary_event("->".join(hierarchy), "Executing function...", icon="🛠️")
        except Exception:
            pass

        # Ensure synchronous work at call-time (if any) happens under the lineage frame.
        token_call = TOOL_LOOP_LINEAGE.set(hierarchy)
        try:
            result = self._wrapped(*args, **kwargs)
        except Exception:
            TOOL_LOOP_LINEAGE.reset(token_call)
            raise
        finally:
            # For async results we only needed the lineage during coroutine construction.
            # The actual awaited execution will run under a new token created in the
            # awaiting task context below.
            try:
                TOOL_LOOP_LINEAGE.reset(token_call)
            except Exception:
                pass

        if inspect.isawaitable(result):

            async def _await_and_finalize():
                token_run = TOOL_LOOP_LINEAGE.set(hierarchy)
                try:
                    return await result
                finally:
                    TOOL_LOOP_LINEAGE.reset(token_run)

            return _await_and_finalize()

        return result


class _InProcessFunctionProxy:
    """Proxy that wraps an in-process function with state mode support.

    This proxy lets a stored function be called with an explicit state mode.
    It supports three execution modes for fine-grained control over state
    management:

    Execution Modes
    ---------------
    **stateful** (default via ``__call__``, or explicit via ``.stateful()``):
        Executes in a persistent in-process session. Variables defined in previous
        calls persist across executions within the same session_id. Use this for
        iterative sessions where you want to build up state incrementally.

    **stateless** (via ``.stateless()``):
        Executes in a fresh globals dict with no inherited state. Each call starts
        with a clean environment. Use this for pure functions that should not
        depend on or affect any global state - guarantees reproducible results.

    **read_only** (via ``.read_only()``):
        Reads the current global state from the persistent session but executes
        in a fresh globals dict. Changes made during execution are NOT persisted
        back to the session. Use this for "what-if" exploration.

    Usage Examples
    --------------
    ```python
    # Stateful (default) - state persists between calls
    await set_config(key="debug", value=True)
    await run_analysis()  # can access 'config' from previous call

    # Explicit stateful (equivalent to default __call__)
    result = await compute.stateful(x=1, y=2)

    # Stateless - fresh environment each time
    result = await compute.stateless(x=1, y=2)

    # Read-only - see current state but don't modify it
    preview = await transform.read_only(factor=2)
    ```

    Design Note
    -----------
    The proxy is returned to callers (e.g., CodeActActor) for state mode control,
    but the **raw function** remains in the execution namespace. This allows:
    - Inter-function calls to work naturally (``await b()`` calls raw ``b``)
    - ``typing.get_type_hints(fn_name)`` to resolve correctly
    - Custom decorators (``@my_decorator``) to work during exec()

    The proxy exposes ``__wrapped__`` pointing to the raw function for introspection.
    """

    def __init__(
        self,
        *,
        function_manager: "FunctionManager",
        func_data: Dict[str, Any],
        namespace: Dict[str, Any],
        raw_callable: Callable[..., Any],
    ):
        self._function_manager = function_manager
        self._func_data = func_data
        self._namespace = namespace
        self._raw_callable = raw_callable

        # Copy key attributes from raw callable for introspection
        self.__name__ = str(func_data.get("name") or "unknown")
        self.__doc__ = str(func_data.get("docstring") or "")
        self.__wrapped__ = raw_callable  # Standard Python convention for wrapper chains

    async def _execute_with_mode(
        self,
        state_mode: Literal["stateful", "read_only", "stateless"],
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        """
        Execute the function with the specified state mode.

        Args:
            state_mode: How to handle global state during execution.
            *args: Positional arguments passed to the function.
            **kwargs: Keyword arguments passed to the function.

        Returns:
            The function's return value.
        """
        if state_mode == "stateful":
            # Execute directly using the raw callable in the shared namespace.
            # This is the existing behavior - state naturally persists in the namespace.
            result = self._raw_callable(*args, **kwargs)
            if asyncio.iscoroutine(result):
                result = await result
            return result

        # For stateless and read_only, use execute_function with appropriate
        # mode. Forward environment namespace objects (primitives, etc.) but
        # NOT user-defined state variables -- those are managed by state_mode.
        proxy_ns: Dict[str, Any] = {}
        val = self._namespace.get("primitives")
        if val is not None:
            proxy_ns["primitives"] = val
        result = await self._function_manager.execute_function(
            function_name=self.__name__,
            call_kwargs=kwargs,
            state_mode=state_mode,
            session_id=0,  # Default session for read_only state source
            extra_namespaces=proxy_ns if proxy_ns else None,
        )

        if result.get("error"):
            raise RuntimeError(str(result.get("error")))
        return result.get("result")

    async def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """
        Execute the function in stateful mode (default).

        State persists across calls within the shared namespace. Variables
        defined in previous executions remain accessible. This is the default
        behavior, suitable for iterative/interactive sessions.

        Equivalent to calling ``.stateful()`` explicitly.

        Args:
            *args: Positional arguments passed to the function.
            **kwargs: Keyword arguments passed to the function.

        Returns:
            The function's return value.
        """
        return await self._execute_with_mode("stateful", *args, **kwargs)

    def stateful(self, *args: Any, **kwargs: Any):
        """
        Execute the function in stateful mode (explicit form of default ``__call__``).

        State persists across calls within the shared namespace. Variables
        defined in previous executions remain accessible. Use this when you want
        to be explicit about the execution mode in your code.

        Equivalent to ``await fn()`` but more self-documenting.

        Args:
            *args: Positional arguments passed to the function.
            **kwargs: Keyword arguments passed to the function.

        Returns:
            Awaitable that resolves to the function's return value.
        """
        return self._execute_with_mode("stateful", *args, **kwargs)

    def stateless(self, *args: Any, **kwargs: Any):
        """
        Execute the function in stateless mode (fresh environment).

        Each call executes with fresh globals and no inherited state.
        The function cannot see or modify any variables from previous executions.
        Use this for pure functions that should produce identical results
        regardless of execution history.

        Args:
            *args: Positional arguments passed to the function.
            **kwargs: Keyword arguments passed to the function.

        Returns:
            Awaitable that resolves to the function's return value.
        """
        return self._execute_with_mode("stateless", *args, **kwargs)

    def read_only(self, *args: Any, **kwargs: Any):
        """
        Execute the function in read-only mode (sees state, no persistence).

        Reads the current global state from the persistent in-process session but
        executes in a fresh globals dict. Any modifications during execution are
        discarded - the persistent session state remains unchanged.

        Args:
            *args: Positional arguments passed to the function.
            **kwargs: Keyword arguments passed to the function.

        Returns:
            Awaitable that resolves to the function's return value.
        """
        return self._execute_with_mode("read_only", *args, **kwargs)


class FunctionManager(BaseFunctionManager):
    """
    Keeps a catalogue of user-supplied Python functions and system primitives.

    Stored functions live in the ``functions`` table with auto-incrementing
    IDs. System primitives (the ``primitives.*`` namespace methods) live in the
    read-only ``primitives`` table with explicit stable IDs.

    This separation ensures:
    - User function IDs are stable (adding/removing primitives doesn't affect them)
    - Primitive IDs are consistent across all users (hash-based stable IDs)
    - No ID collisions between the two namespaces
    """

    # ------------------------------------------------------------------ #
    #  Construction                                                      #
    # ------------------------------------------------------------------ #

    def __init__(
        self,
        *,
        primitive_scope: Optional[PrimitiveScope] = None,
        filter_scope: Optional[str] = None,
        exclude_primitive_ids: Optional[FrozenSet[int]] = None,
        exclude_compositional_ids: Optional[FrozenSet[int]] = None,
        include_primitives: bool = True,
    ) -> None:
        # Store the scope - this FunctionManager instance is permanently scoped
        # Default to the canonical role-scoped manager set when not specified.
        self._primitive_scope = primitive_scope or default_runtime_scope()
        self._filter_scope = filter_scope
        self._exclude_primitive_ids = (
            frozenset(exclude_primitive_ids) if exclude_primitive_ids else None
        )
        self._exclude_compositional_ids = (
            frozenset(exclude_compositional_ids) if exclude_compositional_ids else None
        )
        self._include_primitives = include_primitives
        self._registry = get_registry()
        # ToDo: expose tools to LLM once needed
        self._tools: Dict[str, callable] = {}

        # Internal monotonically-increasing function-id counter.  We keep it local
        # to the manager to avoid an expensive scan across *all* logs every
        # time we create a function.  Initialised lazily on first use.
        self._next_id: Optional[int] = None

        _seed_primitives()

        # ------------------------------------------------------------------ #
        #  In-process session state (for stateful/read_only modes)           #
        # ------------------------------------------------------------------ #
        # Dict[session_id, Dict[str, Any]] - persistent globals per session
        self._in_process_sessions: Dict[int, Dict[str, Any]] = {}

    @property
    def primitive_scope(self) -> PrimitiveScope:
        """The scope controlling which managers' primitives are accessible."""
        return self._primitive_scope

    @property
    def filter_scope(self) -> Optional[str]:
        """A SQL WHERE clause permanently applied to all compositional read queries."""
        return self._filter_scope

    @filter_scope.setter
    def filter_scope(self, value: Optional[str]) -> None:
        self._filter_scope = value

    @property
    def exclude_primitive_ids(self) -> Optional[FrozenSet[int]]:
        """Primitive function IDs excluded from primitive catalogue queries."""
        return self._exclude_primitive_ids

    @exclude_primitive_ids.setter
    def exclude_primitive_ids(self, value: Optional[FrozenSet[int]]) -> None:
        self._exclude_primitive_ids = frozenset(value) if value else None

    @property
    def exclude_compositional_ids(self) -> Optional[FrozenSet[int]]:
        """Stored function IDs excluded from every read of the ``functions`` table."""
        return self._exclude_compositional_ids

    @exclude_compositional_ids.setter
    def exclude_compositional_ids(self, value: Optional[FrozenSet[int]]) -> None:
        self._exclude_compositional_ids = frozenset(value) if value else None

    def _compositional_scope(self, caller_filter: Optional[str] = None) -> str:
        """The clause selecting this manager's visible stored functions."""
        return and_clauses(
            "is_primitive = 0",
            caller_filter,
            self._filter_scope,
            not_in("function_id", self._exclude_compositional_ids),
        )

    def _discovery_scope(self, caller_filter: Optional[str] = None) -> str:
        """The clause selecting everything discovery may return: stored functions
        under ``filter_scope`` plus, when enabled, the scoped primitives."""
        populations = [self._compositional_scope()]
        if self._include_primitives:
            populations.append(self._scoped_primitive_filter())
        return and_clauses(caller_filter, or_clauses(*populations))

    def _scoped_primitive_filter(self, caller_filter: Optional[str] = None) -> str:
        """The clause selecting the primitives in scope, minus the excluded ids."""
        return and_clauses(
            "is_primitive = 1",
            caller_filter,
            self._registry.primitive_row_filter(self._primitive_scope),
            not_in("function_id", self._exclude_primitive_ids),
        )

    def _rows(
        self,
        where: Optional[str],
        params: Sequence[Any] = (),
        *,
        limit: Optional[int] = None,
        offset: int = 0,
        readonly: bool = False,
    ) -> List[Dict[str, Any]]:
        """Read function rows from ``all_functions`` under ``where``."""
        sql = _FUNCTION_SELECT
        if where:
            sql += f" WHERE {where}"
        sql += " ORDER BY is_primitive, function_id"
        if limit is not None:
            sql += f" LIMIT {int(limit)} OFFSET {int(offset)}"
        rows = db.query_readonly(sql, params) if readonly else db.query(sql, params)
        return _attach_guidance_ids([_decode_function_row(row) for row in rows])

    def _primitive_logs(
        self,
        *,
        extra_filter: Optional[str] = None,
        params: Sequence[Any] = (),
        limit: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Fetch the primitive rows in scope."""
        return self._rows(
            self._scoped_primitive_filter(extra_filter),
            params,
            limit=limit,
        )

    @property
    def _dangerous_builtins(self) -> Set[str]:
        """
        A minimal set of truly dangerous built-ins that should never be allowed.
        These could compromise security or system integrity.
        """
        return {
            "eval",
            "exec",
            "compile",
            "__import__",
            "open",  # File system access should go through proper APIs
            "input",  # No interactive input in automated functions
            "breakpoint",  # No debugging breakpoints
            "exit",
            "quit",
        }

    def _parse_implementation(
        self,
        source: str,
    ) -> Tuple[str, ast.Module, ast.FunctionDef, str]:
        """
        Common syntactic checks (unchanged, but now returns the stripped
        source verbatim so we can persist it later).
        """
        stripped = source.lstrip("\n")
        first_line = stripped.splitlines()[0] if stripped else ""
        if first_line.startswith((" ", "\t")):
            raise ValueError(
                "Function definition must start at column 0 (no indentation).",
            )

        try:
            tree = ast.parse(source)
        except SyntaxError as e:
            raise ValueError(f"Syntax error:\n{e.text}") from e

        if len(tree.body) != 1 or not isinstance(
            tree.body[0],
            (ast.FunctionDef, ast.AsyncFunctionDef),
        ):
            if any(
                isinstance(node, (ast.Import, ast.ImportFrom)) for node in tree.body
            ):
                raise ValueError(
                    "Implementation must be a single top-level function definition "
                    "with no module-level imports or other statements.",
                )
            raise ValueError(
                "Each implementation must contain exactly one top-level function.",
            )

        fn_node: Union[ast.FunctionDef, ast.AsyncFunctionDef] = tree.body[0]
        if fn_node.col_offset != 0:
            raise ValueError(
                f"Function {fn_node.name!r} must start at column 0 (no indentation).",
            )

        return fn_node.name, tree, fn_node, source

    def _collect_function_calls(
        self,
        fn_node: Union[ast.FunctionDef, ast.AsyncFunctionDef],
    ) -> Set[str]:
        calls: Set[str] = set()
        for node in ast.walk(fn_node):
            if isinstance(node, ast.Call):
                name = self._format_callable_name(node.func)
                if name:
                    calls.add(name)
        return calls

    @staticmethod
    def _format_callable_name(callable_node: ast.AST) -> Optional[str]:
        """Return a best-effort fully qualified name for a callable.

        Handles both simple names (e.g., ``foo()``) and nested attributes
        (e.g., ``a.b.c()``). If the base of the attribute chain is not a simple
        ``ast.Name`` (e.g., ``get().b()``), this falls back to ``ast.unparse``
        when available.
        """
        # Simple function call: foo()
        if isinstance(callable_node, ast.Name):
            return callable_node.id

        # Attribute access: a.b.c()
        if isinstance(callable_node, ast.Attribute):
            parts: List[str] = []
            current: ast.AST = callable_node
            while isinstance(current, ast.Attribute):
                parts.append(current.attr)
                current = current.value
            if isinstance(current, ast.Name):
                parts.append(current.id)
                return ".".join(reversed(parts))
            # Fallback to unparse for complex bases like calls/subscripts
            try:
                return ast.unparse(callable_node)
            except Exception:
                pass
            return ".".join(reversed(parts)) if parts else None

        try:
            return ast.unparse(callable_node)
        except Exception:
            return None

    def _validate_function_calls(
        self,
        fn_name: str,
        calls: Set[str],
    ) -> None:
        """
        Validates function calls to prevent dangerous operations.

        Allows:
        - Built-in functions (except dangerous ones)
        - Any method calls on objects (e.g., primitives.*, call_handle.*, call.*)
        - User-defined functions (tracked as dependencies)

        Disallows:
        - Dangerous built-in functions (eval, exec, etc.)
        """
        dangerous = self._dangerous_builtins

        for called in calls:
            # Allow all method calls (anything with a dot)
            # This includes primitives.*, call_handle.*, obj.method(), etc.
            if "." in called:
                continue

            # Block only truly dangerous built-ins
            if called in dangerous:
                raise ValueError(
                    f"Dangerous built-in '{called}' is not permitted in {fn_name}(). "
                    f"Functions cannot use: {', '.join(sorted(dangerous))}",
                )

    # ------------------------------------------------------------------ #
    #  Private helpers for persistence                                    #
    # ------------------------------------------------------------------ #

    def _get_log_by_function_id(
        self,
        *,
        function_id: int,
        raise_if_missing: bool = True,
    ) -> Optional[Dict[str, Any]]:
        rows = self._rows("is_primitive = 0 AND function_id = ?", (int(function_id),))
        if not rows:
            if raise_if_missing:
                raise ValueError(f"No function with id {function_id!r} exists.")
            return None
        return rows[0]

    # ------------------------------------------------------------------ #
    #  Activation: the usage trace behind memory-weighted retrieval       #
    # ------------------------------------------------------------------ #

    @property
    def activation_settings(self) -> "ActivationSettings":
        from unify.settings import SETTINGS

        return SETTINGS.function.activation

    def _note_function_use(self, func_data: Dict[str, Any]) -> None:
        """Record one invocation on the function's usage trace.

        Fire-and-forget off the caller's loop: metering must never slow or
        break execution, and a lost count only under-reports standing.
        Primitives are platform surface, not library memory, and are never
        traced.
        """
        settings = self.activation_settings
        if not settings.enabled:
            return
        if func_data.get("is_primitive"):
            return
        fid = func_data.get("function_id")
        if fid is None:
            return
        now_iso = datetime.now(timezone.utc).isoformat()
        kept = settings.recent_calls_kept

        def _write() -> None:
            row = db.query_one(
                "SELECT usage_recent_calls FROM functions WHERE function_id = ?",
                (int(fid),),
            )
            if row is None:
                return
            recents = list(db.loads(row["usage_recent_calls"]) or [])
            recents.append(now_iso)
            db.execute(
                "UPDATE functions SET usage_calls = usage_calls + 1,"
                " usage_last_called_at = ?, usage_recent_calls = ? WHERE function_id = ?",
                (now_iso, db.dumps(recents[-kept:]), int(fid)),
            )

        try:
            self._write_off_loop(_write, what=f"usage:{func_data.get('name')}")
        except Exception:  # noqa: BLE001 - metering must never break a call
            pass

    def _bump_search_hits(self, rows: List[Dict[str, Any]]) -> None:
        """Retrieved-but-never-called is a signal of its own; count it."""
        settings = self.activation_settings
        if not settings.enabled:
            return
        ids = [
            int(row["function_id"])
            for row in rows
            if row.get("function_id") is not None and not row.get("is_primitive")
        ]
        if not ids:
            return

        def _write() -> None:
            db.executemany(
                "UPDATE functions SET usage_search_hits = usage_search_hits + 1"
                " WHERE function_id = ?",
                [(fid,) for fid in ids],
            )

        try:
            self._write_off_loop(_write, what="search_hits")
        except Exception:  # noqa: BLE001 - metering must never break search
            pass

    def _stamp_new_function_usage(
        self,
        entry_data: Dict[str, Any],
        name: str,
    ) -> None:
        """Creation is the row's first activation event; a same-name
        delete-then-add (the librarian's supersede flow) inherits the
        deleted row's usage trace so the replacement stands where its
        predecessor stood."""
        entry_data["created_at"] = datetime.now(timezone.utc).isoformat()
        inherited = self._take_usage_legacy(name)
        if inherited:
            entry_data.update(
                {
                    "usage_calls": int(inherited.get("calls") or 0),
                    "usage_last_called_at": inherited.get("last_called_at"),
                    "usage_recent_calls": inherited.get("recent_calls") or [],
                    "usage_search_hits": int(inherited.get("search_hits") or 0),
                },
            )

    def _stash_usage_legacy(
        self,
        name: object,
        entries: Dict[str, Any],
    ) -> None:
        if not self.activation_settings.enabled or not name:
            return
        trace = {
            "calls": entries.get("usage_calls"),
            "last_called_at": entries.get("usage_last_called_at"),
            "recent_calls": entries.get("usage_recent_calls"),
            "search_hits": entries.get("usage_search_hits"),
        }
        if not any(v for v in trace.values()):
            return
        store: Dict[str, Dict[str, Any]] = getattr(self, "_usage_legacies", None) or {}
        store[str(name)] = merged_usage(
            store.get(str(name)),
            trace,
            self.activation_settings,
        )
        # Bounded, same-process-only memory: oldest entries fall off.
        while len(store) > 64:
            store.pop(next(iter(store)))
        self._usage_legacies = store

    def _take_usage_legacy(self, name: object) -> Optional[Dict[str, Any]]:
        store = getattr(self, "_usage_legacies", None)
        return store.pop(str(name), None) if store else None

    def _activation_rank(
        self,
        rows: List[Dict[str, Any]],
        *,
        n: int,
        include_dormant: bool,
    ) -> List[Dict[str, Any]]:
        """Order search results by word match × standing; drop the lapsed.

        The match dominates (the activation term is capped in settings) and
        backfilled rows — which matched nothing — keep their tail position.
        Primitives never drop out of scope: platform surface is not memory.
        Each surviving row is annotated with the components — ``_similarity``,
        ``_standing``, ``_retrieval_score`` — so the querying model sees WHY
        a result ranked where it did (relevant-but-lapsed reads differently
        from mediocre-but-battle-tested) instead of re-deriving the math.
        """
        settings = self.activation_settings
        if not settings.enabled:
            return rows[:n]
        now = datetime.now(timezone.utc)
        ranked: List[tuple[float, int, Dict[str, Any]]] = []
        for idx, row in enumerate(rows):
            standing = activation(
                now=now,
                created_at=row.get("created_at"),
                call_count=int(row.get("usage_calls") or 0),
                recent_calls=row.get("usage_recent_calls") or [],
                settings=settings,
            )
            if (
                not include_dormant
                and not row.get("is_primitive")
                and not in_scope(standing, settings)
            ):
                continue
            similarity = float(row.get(SIMILARITY_FIELD) or 0.0)
            score = rank_score(similarity, standing, settings)
            row["_similarity"] = round(similarity, 4)
            row["_standing"] = round(standing, 4)
            row["_retrieval_score"] = round(score, 4)
            ranked.append((-score, idx, row))
        ranked.sort(key=lambda item: (item[0], item[1]))
        return [row for _, _, row in ranked[:n]]

    _refold_lock = threading.Lock()

    def _write_off_loop(
        self,
        fn: Callable[[], None],
        *,
        what: str,
    ) -> "concurrent.futures.Future[None]":
        """Run a usage-trace write without blocking the caller's event loop.

        Bounded retry: one immediate retry, then the failure is logged. A
        lost write only under-reports standing. The returned future
        resolves once the attempt (including its retry) has finished,
        carrying the final failure if the write was lost; fire-and-forget
        callers ignore it.
        """
        done: "concurrent.futures.Future[None]" = concurrent.futures.Future()

        def _attempt() -> None:
            for attempt in (1, 2):
                try:
                    fn()
                    done.set_result(None)
                    return
                except Exception as exc:
                    if attempt == 2:
                        logger.warning("%s failed after retry: %s", what, exc)
                        done.set_exception(exc)

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            _attempt()
            return done
        loop.run_in_executor(None, _attempt)
        return done

    def _boundary(
        self,
        raw: Callable[..., Any],
        func_data: Dict[str, Any],
    ) -> Callable[..., Any]:
        """The namespace-facing callable for a compositional function.

        Every call site — a plan namespace or a symbolic closure — goes
        through the same lineage-tracking wrapper, which also records the
        function's usage trace.
        """
        if isinstance(raw, _LineageTrackedFunction):
            return raw
        return _LineageTrackedFunction(
            raw,
            str(func_data.get("name")),
            on_call=lambda: self._note_function_use(func_data),
        )

    # ------------------------------------------------------------------ #
    #  Public API                                                        #
    # ------------------------------------------------------------------ #

    @functools.wraps(BaseFunctionManager.clear, updated=())
    def clear(self) -> None:
        with db.transaction() as conn:
            conn.execute("DELETE FROM functions")
            conn.execute("DELETE FROM sqlite_sequence WHERE name = 'functions'")
        self._next_id = None
        self._in_process_sessions.clear()

    def clear_in_process_sessions(self, session_id: Optional[int] = None) -> None:
        """
        Clear in-process session state.

        Parameters
        ----------
        session_id : int | None, default ``None``
            If provided, clear only the specified session. If None, clear all sessions.
        """
        if session_id is not None:
            self._in_process_sessions.pop(session_id, None)
        else:
            self._in_process_sessions.clear()

    @staticmethod
    def _compact_function_search_rows(
        rows: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Return actor-facing discovery rows without large structured payloads."""

        compact_rows: list[dict[str, Any]] = []
        for row in rows:
            compact = {
                key: value for key, value in row.items() if key != "implementation"
            }
            if compact.get("is_primitive"):
                # Primitive docstrings are full manual pages; discovery
                # results must not re-import what the actor prompt
                # deliberately leaves out. Summary + params is enough to
                # call the method.
                doc = str(compact.get("docstring") or "")
                if doc:
                    summary = get_registry()._extract_summary_and_params(doc)
                    compact["docstring"] = summary or doc[:800]
            compact_rows.append(compact)
        return compact_rows

    def list_primitives(self) -> Dict[str, Dict[str, Any]]:
        """
        Return a mapping of primitive name to primitive metadata.

        Only returns primitives for namespaces in this FunctionManager's scope,
        read from the builtins catalogue.

        Returns:
            Dict mapping primitive name to metadata dict (includes function_id).
        """
        entries: Dict[str, Dict[str, Any]] = {}
        try:
            for row in self._primitive_logs():
                data = {
                    "function_id": row.get("function_id"),
                    "name": row["name"],
                    "argspec": row.get("argspec", ""),
                    "docstring": row.get("docstring", ""),
                    "is_primitive": True,
                    "primitive_class": row.get("primitive_class"),
                    "primitive_method": row.get("primitive_method"),
                }
                for key in ("metadata",):
                    if key in row:
                        data[key] = row.get(key)
                entries.setdefault(row["name"], data)
        except Exception as e:
            logger.warning(f"Failed to list primitives: {e}")
        return entries

    # 1. Add / register ------------------------------------------------- #

    @functools.wraps(BaseFunctionManager.add_functions, updated=())
    def add_functions(
        self,
        *,
        implementations: Union[str, List[str]],
        preconditions: Optional[Dict[str, Dict]] = None,
        overwrite: bool = False,
        raise_on_error: bool = True,
        dependencies: Optional[List[str]] = None,
    ) -> Dict[str, str]:
        """
        Add or update functions in batch.

        Args:
            implementations: Function source code (single string or list of strings).
            preconditions: Optional preconditions for functions.
            overwrite: If True, update existing functions; if False, skip duplicates.
            raise_on_error: If True (default), raise ValueError when any function
                fails to add. If False, errors are returned in the result dict.
            dependencies: PEP 508 requirement strings for the third-party
                packages the functions import. Required when any function
                imports a package beyond the standard library and the
                execution environment; recorded on every function in the batch.

        Returns:
            Dictionary mapping function names to status ("added", "updated", "skipped", or "error").

        Raises:
            ValueError: If raise_on_error=True and any function fails to add,
                if third-party imports are detected without ``dependencies``,
                or if a dependency is not a valid requirement string.
        """

        if preconditions is None:
            preconditions = {}
        requirements = list(dependencies or [])
        for specifier in requirements:
            try:
                environment.parse_requirement(specifier)
            except InvalidRequirement as e:
                raise ValueError(
                    f"Dependency {specifier!r} is not a valid requirement "
                    f"string ({e}). Use PEP 508 form, e.g. 'pandas>=2.0' or "
                    f"'pkg @ git+https://github.com/user/repo.git'.",
                )
        if isinstance(implementations, str):
            implementations = [implementations]

        parsed: List[Tuple[str, ast.Module, ast.FunctionDef, str]] = []
        parse_errors: Dict[str, str] = {}
        temp_names: Set[str] = set()

        # Parse all implementations
        for i, source in enumerate(implementations):
            try:
                # _parse_implementation validates basic structure (one func at col 0)
                name, tree, node, src = self._parse_implementation(source)
                parsed.append((name, tree, node, src))
                temp_names.add(name)
            except ValueError as e:
                # Associate error with name or index
                potential_name = f"implementation_{i+1}"
                try:
                    name_in_error = ast.parse(source).body[0].name
                except:
                    name_in_error = None
                key = name_in_error or potential_name
                parse_errors[key] = f"error: {e}"

        results: Dict[str, str] = parse_errors

        # Get existing functions for duplicate detection and dependency checking
        try:
            existing_functions = self.list_functions()
            existing_names = set(existing_functions.keys())
            all_known_function_names = existing_names.union(temp_names)
        except Exception as e:
            logger.warning(
                f"Failed to list existing functions for dependency check: {e}",
            )
            existing_functions = {}
            existing_names = set()
            all_known_function_names = temp_names

        # Check for duplicates and separate into new vs. existing functions
        duplicates_to_skip: Set[str] = set()
        existing_to_update: Set[str] = set()

        for name in temp_names:
            if name in existing_names:
                if overwrite:
                    # Mark for in-place update
                    existing_to_update.add(name)
                else:
                    # Skip this function - already exists
                    duplicates_to_skip.add(name)
                    results[name] = "skipped: already exists"

        # Validate dependencies and prepare entries for batch operations
        entries_to_create: List[Dict[str, Any]] = []
        entries_to_update: List[Dict[str, Any]] = []
        log_ids_to_update: List[int] = []
        log_id_to_name: Dict[int, str] = {}

        # Sandbox namespace roots whose dotted calls should be recorded in
        # depends_on (e.g. "primitives.actor.act" → depends_on includes
        # "primitives.actor.act").  At runtime, _inject_dependencies reads
        # these entries and calls construct_sandbox_root() to materialise
        # the root object.  All primitives live under a single "primitives"
        # namespace.
        env_namespaces = frozenset({"primitives"})

        for name, tree, node, source in parsed:
            if name in duplicates_to_skip:
                continue

            try:
                dependencies = collect_dependencies_from_function_node(
                    node,
                    all_known_function_names,
                    environment_namespaces=env_namespaces,
                )
                dependencies_list = sorted(list(dependencies))

                tp_imports = detect_third_party_imports(
                    node,
                    environment_modules=ENVIRONMENT_MODULES,
                )
                if tp_imports and not requirements:
                    raise ValueError(
                        f"Function '{name}' imports third-party packages "
                        f"{sorted(tp_imports)} but no dependencies were "
                        f"provided. Pass the pip specifiers that supply "
                        f"them as `dependencies` (e.g. ['pandas>=2.0']); "
                        f"they are installed into the workspace environment "
                        f"before the function runs. Every import form "
                        f"counts, including importlib.import_module and "
                        f"__import__ with a literal name.",
                    )

                all_calls = self._collect_function_calls(node)
                self._validate_function_calls(name, all_calls)
                namespace = create_base_globals()
                exec(source, namespace)
                fn_obj = namespace[name]
                signature = str(inspect.signature(fn_obj))
                docstring = inspect.getdoc(fn_obj) or ""
                precondition = preconditions.get(name)

                prior = None
                if name in existing_to_update:
                    prior = self._get_log_by_function_id(
                        function_id=existing_functions[name]["function_id"],
                        raise_if_missing=True,
                    )

                entry_data = {
                    "argspec": signature,
                    "docstring": docstring,
                    "implementation": source,
                    "depends_on": dependencies_list,
                    "third_party_imports": sorted(tp_imports),
                    "dependencies": requirements,
                    "precondition": precondition,
                    "stale_reasons": [
                        reason.model_dump(mode="json")
                        for reason in self._dependency_stale_reasons(
                            dependencies_list,
                            available_names=all_known_function_names,
                        )
                    ],
                }

                if prior is not None:
                    # Update existing function
                    log_id = int(prior["function_id"])
                    log_ids_to_update.append(log_id)
                    log_id_to_name[log_id] = name
                    entries_to_update.append(entry_data)
                    results[name] = "updated"
                else:
                    # Create new function
                    entry_data["name"] = name
                    self._stamp_new_function_usage(entry_data, name)
                    entries_to_create.append(entry_data)
                    results[name] = "added"
            except ValueError as e:
                results[name] = f"error: {e}"
            except Exception as e:
                results[name] = f"error: Unexpected error - {e}"
                logger.error(
                    f"Unexpected error processing function {name}: {e}",
                    exc_info=True,
                )

        # Batch create new functions
        if entries_to_create:
            try:
                with db.transaction():
                    for entry in entries_to_create:
                        self._insert_function(entry)
            except Exception as e:
                logger.error(
                    f"Failed to batch create function logs: {e}",
                    exc_info=True,
                )
                for entry in entries_to_create:
                    name = entry["name"]
                    if results.get(name) == "added":
                        results[name] = f"error: Failed to create log - {e}"

        # Batch update existing functions
        if log_ids_to_update and entries_to_update:
            try:
                with db.transaction():
                    for function_id, entry in zip(log_ids_to_update, entries_to_update):
                        self._update_function(function_id, entry)
            except Exception as e:
                logger.error(
                    f"Failed to batch update function logs: {e}",
                    exc_info=True,
                )
                for log_id in log_ids_to_update:
                    name = log_id_to_name.get(log_id)
                    if name and results.get(name) == "updated":
                        results[name] = f"error: Failed to update log - {e}"

        # Check for errors and raise if requested
        if raise_on_error:
            errors = {k: v for k, v in results.items() if v.startswith("error")}
            if errors:
                error_details = "; ".join(f"{k}: {v}" for k, v in errors.items())
                raise ValueError(f"Failed to add function(s): {error_details}")

        return results

    # ------------------------------------------------------------------ #
    #  Callable return + dependency injection                             #
    # ------------------------------------------------------------------ #

    def _get_function_data_by_name(self, *, name: str) -> Optional[Dict[str, Any]]:
        """Fetch a stored function's full record by exact name, or ``None``."""
        rows = self._rows("is_primitive = 0 AND name = ?", (name,), limit=1)
        return rows[0] if rows else None

    @staticmethod
    def _insert_function(entry: Dict[str, Any]) -> int:
        values = _encode_function_values(entry)
        values.setdefault("created_at", datetime.now(timezone.utc).isoformat())
        columns = list(values)
        cursor = db.execute(
            f"INSERT INTO functions ({', '.join(columns)})"
            f" VALUES ({', '.join('?' for _ in columns)})",
            [values[column] for column in columns],
        )
        return int(cursor.lastrowid)

    @staticmethod
    def _update_function(function_id: int, changes: Dict[str, Any]) -> None:
        values = _encode_function_values(changes)
        assignments = ", ".join(f"{column} = ?" for column in values)
        db.execute(
            f"UPDATE functions SET {assignments} WHERE function_id = ?",
            [*values.values(), int(function_id)],
        )

    def _create_in_process_callable(
        self,
        func_data: Dict[str, Any],
        *,
        namespace: Dict[str, Any],
    ) -> _InProcessFunctionProxy:
        """Create an in-process callable wrapped in a proxy with state mode support.

        The function is exec'd into the provided ``namespace``, placing the **raw
        function** there. The caller should NOT overwrite ``namespace[func_name]``
        with the returned proxy - this allows:

        - Inter-function calls to work naturally (``await b()`` calls raw ``b``)
        - ``typing.get_type_hints(fn_name)`` to resolve correctly via ``__wrapped__``
        - Custom decorators (``@my_decorator``) to work during exec()

        The returned proxy provides state mode control:
        ``.stateful()`` / ``.stateless()`` / ``.read_only()``.
        """
        func_name = func_data.get("name")
        if not isinstance(func_name, str) or not func_name:
            raise ValueError("func_data missing valid 'name'")

        implementation = func_data.get("implementation")
        if not isinstance(implementation, str) or not implementation.strip():
            raise ValueError(f"Function '{func_name}' has no implementation")

        # Ensure user-defined annotation symbols don't cause NameErrors when callers
        # (e.g., CodeActActor) later resolve type hints via typing.get_type_hints().
        self._inject_forward_ref_annotation_placeholders(
            implementation,
            namespace=namespace,
        )

        environment.ensure(func_data.get("dependencies") or [])
        exec(compile_function_source(func_name, implementation), namespace)
        raw_fn = namespace.get(func_name)
        if not callable(raw_fn):
            raise ValueError(
                f"Function '{func_name}' not found after exec() into namespace",
            )

        # Wrap in proxy to provide state mode API (.stateless(), .read_only())
        return _InProcessFunctionProxy(
            function_manager=self,
            func_data=func_data,
            namespace=namespace,
            raw_callable=raw_fn,
        )

    @staticmethod
    def _inject_forward_ref_annotation_placeholders(
        implementation: str,
        *,
        namespace: Dict[str, Any],
    ) -> None:
        """
        Inject placeholder types for missing symbols referenced in annotations.

        Motivation:
        - `exec()` succeeds when annotations are strings, but later calls to
          `typing.get_type_hints(fn)` will evaluate forward-ref strings in
          `fn.__globals__` and can raise NameError if the referenced types are not
          present.
        - CodeActActor wants a callable that "just works" without manual seeding
          of domain-specific types into the namespace.

        This only attempts to satisfy *annotation resolution* (not runtime logic).
        If the function body actually uses a type (e.g. `Role.ADMIN`), the
        function must still import/define it itself.
        """
        try:
            tree = ast.parse(implementation)
        except Exception:
            return

        if not tree.body:
            return

        fn_node: Optional[Union[ast.FunctionDef, ast.AsyncFunctionDef]] = None
        if len(tree.body) == 1 and isinstance(
            tree.body[0],
            (ast.FunctionDef, ast.AsyncFunctionDef),
        ):
            fn_node = tree.body[0]
        else:
            for node in tree.body:
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    fn_node = node
                    break
        if fn_node is None:
            return

        ann_exprs: List[ast.AST] = []
        args = fn_node.args
        for arg in list(args.posonlyargs) + list(args.args) + list(args.kwonlyargs):
            if getattr(arg, "annotation", None) is not None:
                ann_exprs.append(arg.annotation)  # type: ignore[arg-type]
        if (
            args.vararg is not None
            and getattr(args.vararg, "annotation", None) is not None
        ):
            ann_exprs.append(args.vararg.annotation)  # type: ignore[arg-type]
        if (
            args.kwarg is not None
            and getattr(args.kwarg, "annotation", None) is not None
        ):
            ann_exprs.append(args.kwarg.annotation)  # type: ignore[arg-type]
        if getattr(fn_node, "returns", None) is not None:
            ann_exprs.append(fn_node.returns)  # type: ignore[arg-type]

        annotation_names: Set[str] = set()
        for expr in ann_exprs:
            # Forward-ref strings: parse the string itself as a Python expression.
            if isinstance(expr, ast.Constant) and isinstance(expr.value, str):
                ref = expr.value.strip()
                if not ref:
                    continue
                try:
                    ref_tree = ast.parse(ref, mode="eval")
                except Exception:
                    continue
                for node in ast.walk(ref_tree):
                    if isinstance(node, ast.Name) and node.id:
                        annotation_names.add(node.id)
                continue

            for node in ast.walk(expr):
                if isinstance(node, ast.Name) and node.id:
                    annotation_names.add(node.id)

        if not annotation_names:
            return

        # exec() installs the real builtins module into the namespace, so real
        # builtin names (dict, str, list, ...) are always resolvable even when
        # the caller passes a fresh namespace without __builtins__. Shadowing
        # them with placeholder classes breaks subscripted annotations like
        # ``dict[str, Any]`` at def-evaluation time.
        builtin_names = set(dir(builtins))
        builtins_obj = namespace.get("__builtins__")
        if isinstance(builtins_obj, dict):
            builtin_names |= set(builtins_obj.keys())
        elif builtins_obj is not None:
            builtin_names |= set(dir(builtins_obj))

        typing_mod = namespace.get("typing")
        pydantic_mod = namespace.get("pydantic")

        for name in sorted(annotation_names):
            if name == "typing":
                continue
            if name in namespace:
                continue
            if name in builtin_names:
                continue

            # Common typing helpers can be recovered from typing when present.
            if typing_mod is not None and hasattr(typing_mod, name):
                try:
                    namespace[name] = getattr(typing_mod, name)
                    continue
                except Exception:
                    pass

            # Some common pydantic types may appear in annotations.
            if pydantic_mod is not None and hasattr(pydantic_mod, name):
                try:
                    namespace[name] = getattr(pydantic_mod, name)
                    continue
                except Exception:
                    pass

            # Fall back to a placeholder type to avoid NameError during hint resolution.
            try:
                namespace[name] = type(name, (), {})
            except Exception:
                # If something extremely unusual happens, just skip.
                continue

    def _inject_dependencies(
        self,
        func_data: Dict[str, Any],
        *,
        namespace: Dict[str, Any],
        visited: Set[str],
    ) -> None:
        """Inject transitive dependencies into ``namespace`` (breadth-first).

        This is the runtime counterpart to the AST-based dependency detection
        in ``dependency_analysis.py``.  Every name that ``add_functions``
        recorded in ``depends_on`` is resolved here into a live object in the
        execution namespace.  The two categories:

        **Bare names** (e.g. ``"helper"``) — other compositional functions.
        The stored implementation is exec'd into the namespace so inter-
        function calls resolve naturally.

        **Dotted names** (e.g. ``"primitives.actor.act"``) —
        environment-provided namespaces.
        Only the *root* segment matters for injection (``"primitives"``).
        If the root is not already present in the namespace,
        ``construct_sandbox_root()`` from the primitive registry constructs
        a fresh ``Primitives`` instance on demand.  ``Primitives`` is
        fully stateless, so a freshly constructed instance works in
        isolation without any ambient ContextVars or parent actor state.
        """
        from collections import deque

        from unify.function_manager.primitives.registry import construct_sandbox_root

        deps = func_data.get("depends_on") or []
        if not isinstance(deps, list):
            return

        q = deque([d for d in deps if isinstance(d, str) and d])
        while q:
            dep_name = q.popleft()
            if dep_name in visited:
                continue
            visited.add(dep_name)

            # ── Dotted dependency (e.g. "primitives.actor.act") ──
            if "." in dep_name:
                root = dep_name.split(".")[0]
                if root not in namespace:
                    root_obj = construct_sandbox_root(
                        root,
                        primitive_scope=self._primitive_scope,
                    )
                    if root_obj is not None:
                        namespace[root] = root_obj
                    else:
                        logger.warning(
                            "Dotted dependency %r for %r: root %r could not "
                            "be constructed and is not in namespace, skipping",
                            dep_name,
                            func_data.get("name"),
                            root,
                        )
                continue

            # ── Bare dependency (compositional function) ─────────────────────
            dep_data = self._get_function_data_by_name(name=dep_name)
            if not dep_data:
                logger.warning(
                    f"Dependency '{dep_name}' not found for '{func_data.get('name')}', skipping",
                )
                continue

            # exec puts the raw function in the namespace. We call
            # _create_in_process_callable to exec the function, but we DON'T
            # overwrite the namespace with the proxy - the raw function stays
            # for inter-function calls, decorators, and introspection.
            self._create_in_process_callable(
                dep_data,
                namespace=namespace,
            )
            # replace namespace[dep_name] with wrapper so inter-function calls
            # also flow through lineage/event boundaries.
            raw_dep = namespace.get(dep_name)
            if callable(raw_dep):
                namespace[dep_name] = self._boundary(raw_dep, dep_data)

            nested = dep_data.get("depends_on") or []
            if isinstance(nested, list):
                for child in nested:
                    if isinstance(child, str) and child and child not in visited:
                        q.append(child)

    def _inject_callables_for_functions(
        self,
        func_rows: List[Dict[str, Any]],
        *,
        namespace: Dict[str, Any],
    ) -> List[Callable[..., Any]]:
        """Convert function records into callables and return proxies to caller.

        The raw function (from exec) remains in the namespace for
        inter-function calls, decorators, and introspection. The returned
        proxies provide state mode control (.stateful/.stateless/.read_only).

        For primitives, the callable is resolved from the live runtime registry
        via ``get_primitive_callable``. Primitives are NOT injected into the
        namespace (they are already accessible via the ``primitives`` object).
        """
        callables: List[Callable[..., Any]] = []
        visited: Set[str] = set()

        for func_data in func_rows:
            name = func_data.get("name")
            if not isinstance(name, str) or not name:
                raise ValueError("Function record missing valid 'name'")

            # Primitives: resolve callable from the live runtime registry.
            # They are already accessible via the ``primitives`` object in the
            # namespace, so we don't inject a new namespace entry.
            if func_data.get("is_primitive") is True:
                from unify.function_manager.primitives.runtime import (
                    get_primitive_callable,
                )

                primitives_obj = namespace.get("primitives")
                fn = get_primitive_callable(
                    func_data,
                    primitives=primitives_obj,
                )
                if fn is not None:
                    fn = _LineageTrackedFunction(fn, name)
                callables.append(fn)
                continue

            # Check if we've already processed this function (e.g., duplicate in results)
            if name in visited:
                # Already exec'd - just create a new proxy wrapping existing raw fn
                raw_fn = namespace.get(name)
                if callable(raw_fn):
                    # If the namespace contains our wrapper, unwrap for the proxy.
                    try:
                        if hasattr(raw_fn, "__wrapped__"):
                            raw_fn_for_proxy = getattr(raw_fn, "__wrapped__")
                            if callable(raw_fn_for_proxy):
                                raw_fn = raw_fn_for_proxy
                    except Exception:
                        pass
                    fn = _InProcessFunctionProxy(
                        function_manager=self,
                        func_data=func_data,
                        namespace=namespace,
                        raw_callable=raw_fn,
                    )
                else:
                    # Shouldn't happen, but fallback to full creation
                    fn = self._create_in_process_callable(
                        func_data,
                        namespace=namespace,
                    )
                callables.append(fn)
                continue

            visited.add(name)  # Prevent cycles from re-injecting the root function.
            self._inject_dependencies(func_data, namespace=namespace, visited=visited)

            # Create callable for the root function: exec puts the raw function
            # in the namespace, the proxy goes back to the caller. DON'T
            # overwrite the namespace - the raw function stays for internal use.
            fn = self._create_in_process_callable(func_data, namespace=namespace)
            # replace namespace[name] with wrapper so inter-function calls
            # also flow through lineage/event boundaries.
            raw_root = namespace.get(name)
            if callable(raw_root):
                namespace[name] = self._boundary(raw_root, func_data)

            callables.append(fn)

        return callables

    # 2. Listing -------------------------------------------------------- #

    def list_function_name_to_ids(self) -> Dict[str, int]:
        """Return the complete ``{name: function_id}`` catalogue.

        Prefer this over :meth:`list_functions` when only ids are needed.
        Stored functions are read without ``filter_scope`` or environment
        exclusions, so a reference resolves even when its function is hidden
        from discovery; ordinary list/filter/search operations remain scoped.
        """
        mapping: Dict[str, int] = {
            str(row["name"]): int(row["function_id"])
            for row in db.query(
                "SELECT name, function_id FROM functions ORDER BY function_id",
            )
        }
        if self._include_primitives:
            for ent in self._primitive_logs():
                name = ent.get("name")
                function_id = ent.get("function_id")
                if (
                    isinstance(name, str)
                    and name
                    and function_id is not None
                    and name not in mapping
                ):
                    mapping[name] = int(function_id)
        return mapping

    @functools.wraps(BaseFunctionManager.list_functions, updated=())
    def list_functions(
        self,
        *,
        include_implementations: bool = False,
        _return_callable: bool = False,
        _namespace: Optional[Dict[str, Any]] = None,
        _also_return_metadata: bool = False,
    ) -> Dict[str, Dict[str, Any]]:
        if _also_return_metadata and not _return_callable:
            raise ValueError("_also_return_metadata requires _return_callable=True")

        if _return_callable and _namespace is None:
            raise ValueError("_namespace required when _return_callable=True")

        compositional_rows = self._rows(self._compositional_scope())

        primitive_rows: List[Dict[str, Any]] = []
        if self._include_primitives:
            primitive_rows = self._primitive_logs()

        all_rows = compositional_rows + primitive_rows

        metadata: Dict[str, Dict[str, Any]] = {}
        func_rows: List[Dict[str, Any]] = []
        seen_names: set[str] = set()
        for ent in all_rows:
            name = ent.get("name")
            if not isinstance(name, str):
                continue
            if name in seen_names:
                continue
            seen_names.add(name)
            func_rows.append(ent)

            data: Dict[str, Any] = {
                "function_id": ent.get("function_id"),
                "argspec": ent.get("argspec"),
                "docstring": ent.get("docstring", ""),
                "depends_on": ent.get("depends_on", []),
                "stale_reasons": ent.get("stale_reasons", []),
                "guidance_ids": ent.get("guidance_ids", []),
                "dependencies": ent.get("dependencies", []),
                "third_party_imports": ent.get("third_party_imports", []),
                "is_primitive": ent.get("is_primitive", False),
            }
            for key in (
                "primitive_class",
                "primitive_method",
                "metadata",
            ):
                if key in ent:
                    data[key] = ent.get(key)
            if include_implementations:
                data["implementation"] = ent.get("implementation")
            metadata[name] = data

        if not _return_callable:
            return metadata

        assert _namespace is not None  # validated above
        callables_list = self._inject_callables_for_functions(
            func_rows,
            namespace=_namespace,
        )
        callables_map = {
            row["name"]: cb
            for row, cb in zip(func_rows, callables_list)
            if isinstance(row.get("name"), str)
        }

        if _also_return_metadata:
            return {"callables": callables_map, "metadata": metadata}  # type: ignore[return-value]

        return callables_map  # type: ignore[return-value]

    @functools.wraps(BaseFunctionManager.get_precondition, updated=())
    def get_precondition(self, *, function_name: str) -> Optional[Dict[str, Any]]:
        rows = self._rows(
            self._compositional_scope("name = ?"),
            (function_name,),
            limit=1,
        )
        if not rows and self._include_primitives:
            rows = self._primitive_logs(
                extra_filter="name = ?",
                params=(function_name,),
                limit=1,
            )
        if not rows:
            return None
        return rows[0].get("precondition")

    @staticmethod
    def _dependency_stale_reasons(
        depends_on: List[str],
        *,
        available_names: set[str],
        existing: Optional[List[StaleReason]] = None,
    ) -> List[StaleReason]:
        preserved = [
            reason
            for reason in coerce_stale_reasons(existing)
            if reason.dep_kind != "depends_on"
        ]
        # Primitive references ("primitives.<manager>.<method>") are resolved
        # against whatever environments the runtime injects, not against this
        # FunctionManager instance's own primitive catalog. Whether that
        # catalog contains a given name reflects this instance's own
        # scope/sync state, not whether the primitive actually exists and
        # runs -- so it can never be an authoritative "missing" signal.
        # Compositional dependencies are the only category this FunctionManager
        # owns outright (the ``functions`` table), so only those are checked.
        missing = [
            StaleReason(
                dep_kind="depends_on",
                name=name,
                message=f"missing dependency name={name}",
            )
            for name in depends_on
            if not name.startswith("primitives.") and name not in available_names
        ]
        return merge_stale_reasons(preserved, *missing)

    def _available_dependency_names(
        self,
        rows: Optional[List[Dict[str, Any]]] = None,
    ) -> set[str]:
        if rows is None:
            rows = db.query("SELECT name FROM functions")
        return {str(row["name"]) for row in rows if row.get("name")}

    def _append_missing_dependency_reasons(
        self,
        *,
        rows: List[Dict[str, Any]],
        missing_names: set[str],
    ) -> None:
        for row in rows:
            dependencies = set(row.get("depends_on") or [])
            matched = sorted(dependencies.intersection(missing_names))
            if not matched:
                continue
            existing = coerce_stale_reasons(row.get("stale_reasons"))
            merged = merge_stale_reasons(
                existing,
                *[
                    StaleReason(
                        dep_kind="depends_on",
                        name=name,
                        message=f"missing dependency name={name}",
                    )
                    for name in matched
                ],
            )
            if [reason.model_dump(mode="json") for reason in merged] == [
                reason.model_dump(mode="json") for reason in existing
            ]:
                continue
            self._update_function(
                row["function_id"],
                {
                    "stale_reasons": [
                        reason.model_dump(mode="json") for reason in merged
                    ],
                },
            )

    @staticmethod
    def _unlink_deleted_functions_from_guidance(
        deleted_functions: List[tuple[int, str]],
    ) -> None:
        """Drop deleted ids from every guidance entry citing them, recording why."""
        for function_id, name in deleted_functions:
            rows = db.query(
                "SELECT guidance_id, function_ids, stale_reasons FROM guidance"
                " WHERE EXISTS (SELECT 1 FROM json_each(function_ids) WHERE value = ?)",
                (int(function_id),),
            )
            for row in rows:
                db.decode(row, db.GUIDANCE_JSON_COLUMNS)
                merged = merge_stale_reasons(
                    coerce_stale_reasons(row.get("stale_reasons")),
                    StaleReason(
                        dep_kind="function",
                        id=int(function_id),
                        name=name,
                        message=f"missing function_id={int(function_id)} name={name}",
                    ),
                )
                db.execute(
                    "UPDATE guidance SET function_ids = ?, stale_reasons = ? WHERE guidance_id = ?",
                    (
                        db.dumps(
                            [
                                fid
                                for fid in row["function_ids"]
                                if int(fid) != int(function_id)
                            ],
                        ),
                        db.dumps([reason.model_dump(mode="json") for reason in merged]),
                        int(row["guidance_id"]),
                    ),
                )

    # 3. Deletion ------------------------------------------------------- #

    @functools.wraps(BaseFunctionManager.delete_function, updated=())
    def delete_function(
        self,
        *,
        function_id: Union[int, List[int]],
        delete_dependents: bool = True,
    ) -> Dict[str, str]:
        """
        Delete one or more functions and optionally their dependents in a single batch operation.

        Args:
            function_id: Function ID (int) or list of function IDs to delete.
            delete_dependents: If True, also delete all functions that depend on target(s).

        Returns:
            Dictionary mapping function names to "deleted" or "already_deleted".

        Raises:
            ValueError: If any of the requested function_ids correspond to
                primitives (system-owned functions that cannot be deleted).
        """
        function_ids = (
            [function_id] if isinstance(function_id, int) else list(function_id)
        )
        if not function_ids:
            return {}

        if self._include_primitives:
            placeholders = ", ".join("?" for _ in function_ids)
            prim_rows = self._primitive_logs(
                extra_filter=f"function_id IN ({placeholders})",
                params=[int(fid) for fid in function_ids],
            )
            if prim_rows:
                prim_names = [
                    row.get("name", row.get("function_id")) for row in prim_rows
                ]
                raise ValueError(
                    f"Cannot delete primitives (system-owned): {prim_names}",
                )

        all_rows = self._rows("is_primitive = 0")
        by_id = {int(row["function_id"]): row for row in all_rows}
        requested = [int(fid) for fid in function_ids if int(fid) in by_id]
        results: Dict[str, str] = {
            f"function_{fid}": "already_deleted"
            for fid in function_ids
            if int(fid) not in by_id
        }
        if not requested:
            return results

        ids_to_delete = set(requested)
        for fid in requested:
            results[by_id[fid]["name"]] = "deleted"

        if delete_dependents:
            # BFS over the depends_on graph: every function calling a deleted
            # name is deleted too, transitively.
            to_process = {by_id[fid]["name"] for fid in requested}
            processed: set[str] = set()
            while to_process:
                current_name = to_process.pop()
                if current_name in processed:
                    continue
                processed.add(current_name)
                for fid, row in by_id.items():
                    if (
                        current_name in set(row.get("depends_on") or [])
                        and fid not in ids_to_delete
                    ):
                        ids_to_delete.add(fid)
                        results[row["name"]] = "deleted"
                        to_process.add(row["name"])
        else:
            # Keep dependents, but record link debt on rows that still
            # reference the deleted name(s).
            self._append_missing_dependency_reasons(
                rows=[row for fid, row in by_id.items() if fid not in ids_to_delete],
                missing_names={by_id[fid]["name"] for fid in requested},
            )

        deleted = sorted(ids_to_delete)
        self._unlink_deleted_functions_from_guidance(
            [(fid, str(by_id[fid]["name"])) for fid in deleted],
        )

        # The librarian's supersede flow is delete-then-add: stash each
        # deleted row's usage trace by name so a same-process replacement
        # inherits its standing instead of restarting the popularity
        # contest from zero.
        for fid in deleted:
            self._stash_usage_legacy(by_id[fid]["name"], by_id[fid])

        db.execute(
            f"DELETE FROM functions WHERE function_id IN ({', '.join('?' for _ in deleted)})",
            deleted,
        )
        return results

    @functools.wraps(BaseFunctionManager.reconcile_dependencies, updated=())
    def reconcile_dependencies(
        self,
        *,
        function_ids: Optional[List[int]] = None,
    ) -> Dict[str, Any]:
        all_rows = self._rows("is_primitive = 0")
        selected_ids = (
            {int(function_id) for function_id in function_ids}
            if function_ids is not None
            else None
        )
        selected = [
            row
            for row in all_rows
            if selected_ids is None or int(row["function_id"]) in selected_ids
        ]
        available = self._available_dependency_names(all_rows)
        stale_function_ids: list[int] = []
        for row in selected:
            function = Function(**row)
            refreshed = self._dependency_stale_reasons(
                function.depends_on,
                available_names=available,
                existing=function.stale_reasons,
            )
            if refreshed:
                stale_function_ids.append(int(function.function_id))
            serialized = [reason.model_dump(mode="json") for reason in refreshed]
            if serialized == [
                reason.model_dump(mode="json") for reason in function.stale_reasons
            ]:
                continue
            self._update_function(function.function_id, {"stale_reasons": serialized})
        return {
            "outcome": "dependencies reconciled",
            "details": {
                "checked": len(selected),
                "stale_function_ids": stale_function_ids,
                "stale_count": len(stale_function_ids),
            },
        }

    # 4. Filter --------------------------------------------------------- #

    @functools.wraps(BaseFunctionManager.filter_functions, updated=())
    def filter_functions(
        self,
        *,
        filter: Optional[str] = None,
        offset: int = 0,
        limit: int = 100,
        include_implementations: bool = True,
        _return_callable: bool = False,
        _namespace: Optional[Dict[str, Any]] = None,
        _also_return_metadata: bool = False,
    ) -> List[Dict[str, Any]]:
        if _also_return_metadata and not _return_callable:
            raise ValueError("_also_return_metadata requires _return_callable=True")

        if _return_callable and _namespace is None:
            raise ValueError("_namespace required when _return_callable=True")

        try:
            rows = self._rows(
                self._discovery_scope(filter),
                limit=limit,
                offset=offset,
                readonly=True,
            )
        except sqlite3.Error as exc:
            return invalid_filter_error(exc, filter, db.FUNCTION_COLUMNS).payload

        if not _return_callable:
            if not include_implementations:
                rows = [
                    {k: v for k, v in row.items() if k != "implementation"}
                    for row in rows
                ]
            return rows

        assert _namespace is not None  # validated above
        callables_list = self._inject_callables_for_functions(
            rows,
            namespace=_namespace,
        )
        if _also_return_metadata:
            metadata_rows = rows
            if not include_implementations:
                metadata_rows = [
                    {k: v for k, v in row.items() if k != "implementation"}
                    for row in rows
                ]
            return {"callables": callables_list, "metadata": metadata_rows}  # type: ignore[return-value]
        return callables_list  # type: ignore[return-value]

    # 5. Text Search ---------------------------------------------------- #
    @functools.wraps(BaseFunctionManager.search_functions, updated=())
    def search_functions(
        self,
        *,
        query: str = "",
        n: int = 5,
        include_implementations: bool = True,
        include_dormant: bool = False,
        _return_callable: bool = False,
        _namespace: Optional[Dict[str, Any]] = None,
        _also_return_metadata: bool = False,
    ) -> List[Dict[str, Any]]:
        if _also_return_metadata and not _return_callable:
            raise ValueError("_also_return_metadata requires _return_callable=True")

        if _return_callable and _namespace is None:
            raise ValueError("_namespace required when _return_callable=True")

        # Soft models sometimes call search with ``{}`` / empty query during
        # discovery; an empty query has nothing to match, so return a plain
        # catalogue sample instead (or, with UNIFY_REQUIRE_SEARCH_QUERY, an error
        # that asks for the query, so the model names what it is looking for).
        if not str(query or "").strip():
            from unify.settings import SETTINGS

            if SETTINGS.UNIFY_REQUIRE_SEARCH_QUERY:
                return ToolErrorException(
                    {
                        "error_kind": "missing_query",
                        "message": (
                            "query is required: no functions were searched. Give the words a "
                            "matching function's name or docstring would contain, such as the "
                            "task name or id this work is for, and call search_functions again."
                        ),
                        "details": {"query": query, "n": n},
                    },
                ).payload
            return self.filter_functions(
                filter=None,
                offset=0,
                limit=n,
                include_implementations=include_implementations,
                _return_callable=_return_callable,
                _namespace=_namespace,
                _also_return_metadata=_also_return_metadata,
            )

        # Overfetch so the activation pass has candidates to rank and drop:
        # the text-match cut to `limit` happens before standing is known, and
        # a scope-filtered result set must still be able to fill n slots.
        activation_settings = self.activation_settings
        fetch_limit = (
            min(
                max(n + 4, n * activation_settings.search_overfetch_factor),
                activation_settings.search_overfetch_cap,
            )
            if activation_settings.enabled
            else n
        )
        results = rank_by_text(
            self._rows(self._discovery_scope()),
            {field: query for field in SEARCHED_FUNCTION_FIELDS},
            limit=fetch_limit,
            id_field="function_id",
            backfill=True,
        )
        results = self._activation_rank(
            results,
            n=n,
            include_dormant=include_dormant,
        )
        self._bump_search_hits(results)

        if not _return_callable:
            compact_results = self._compact_function_search_rows(results)
            if include_implementations:
                for compact, full in zip(compact_results, results, strict=True):
                    if "implementation" in full:
                        compact["implementation"] = full["implementation"]
            return compact_results

        assert _namespace is not None  # validated above
        callables_list = self._inject_callables_for_functions(
            results,
            namespace=_namespace,
        )

        if _also_return_metadata:
            metadata_rows = self._compact_function_search_rows(results)
            if include_implementations:
                for compact, full in zip(metadata_rows, results, strict=True):
                    if "implementation" in full:
                        compact["implementation"] = full["implementation"]
            return {"callables": callables_list, "metadata": metadata_rows}  # type: ignore[return-value]

        return callables_list  # type: ignore[return-value]

    # ------------------------------------------------------------------ #
    #  Inverse linkage: Functions → Guidance                              #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _get_guidance_ids_for_function(*, function_id: int) -> List[int]:
        rows = db.query(
            "SELECT guidance_id FROM guidance"
            " WHERE EXISTS (SELECT 1 FROM json_each(function_ids) WHERE value = ?)"
            " ORDER BY guidance_id",
            (int(function_id),),
        )
        return [int(row["guidance_id"]) for row in rows]

    def _get_guidance_for_function(
        self,
        *,
        function_id: int,
        limit: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Return guidance records linked to the function.

        Each dict includes: guidance_id, title, content.
        """
        gids = self._get_guidance_ids_for_function(function_id=function_id)
        if limit is not None and limit >= 0:
            gids = gids[:limit]
        if not gids:
            return []
        return db.query(
            "SELECT guidance_id, title, content FROM guidance"
            f" WHERE guidance_id IN ({', '.join('?' for _ in gids)}) ORDER BY guidance_id",
            gids,
        )

    async def execute_function(
        self,
        *,
        function_name: str,
        call_kwargs: Optional[Dict[str, Any]] = None,
        state_mode: Literal["stateful", "read_only", "stateless"] = "stateless",
        session_id: int = 0,
        extra_namespaces: Optional[Dict[str, Any]] = None,
        _parent_chat_context: Optional[list] = None,
    ) -> Any:
        """
        Execute a stored function by name with optional state mode overrides.

        This method looks up a function by name from the function table and
        executes it. It automatically routes to the appropriate executor based
        on the function's type:

        - **Primitives** (``is_primitive=True``): Resolved to their live
          callable via ``get_primitive_callable`` and invoked directly. The
          raw return value is passed through unmodified, which is critical
          for primitives that return ``SteerableToolHandle`` instances.
        - **Composed functions**: Their ``dependencies`` are ensured present
          in the workspace environment, then the implementation is exec'd
          in-process and wrapped in a ``{"result", "error", "stdout",
          "stderr"}`` dict.

        State modes (composed functions only):
        - "stateless" (default): Fresh globals with no inherited state. Pure
          function behavior.
        - "stateful": Persistent globals per session. Variables from previous
          executions persist.
        - "read_only": Reads current session state but executes in fresh
          globals. Changes are NOT persisted. Useful for "what-if" exploration.

        Args:
            function_name: Name of the function to execute.
            call_kwargs: Keyword arguments to pass to the function.
            state_mode: How to handle global state ("stateful", "read_only", "stateless").
            session_id: The session ID (default 0). Multiple sessions allow
                independent stateful execution contexts.
                Only applies to stateful/read_only modes.
            extra_namespaces: Named objects to inject into the function's
                execution globals.

        Returns:
            For composed functions: dict with keys result, error, stdout, stderr.
            For primitives: the raw return value of the callable (may be a
            SteerableToolHandle or any other type).

        Raises:
            ValueError: If the function doesn't exist or has no implementation.
            RuntimeError: If a dependency cannot be installed.

        Anti-patterns:
            - Nesting ``asyncio.run(...)`` inside sync helpers called from
              offline Jobs / actor sandboxes (already under a running loop).
              Prefer ``async def`` + ``await``, or ``run_coro_sync``.
        """
        ns = extra_namespaces or {}
        # Look up function by name (compositional first, then optionally primitives).
        func_data = self._get_function_data_by_name(name=function_name)

        if func_data is None and self._include_primitives:
            func_data = self._get_primitive_data_by_name(name=function_name)

        if func_data is None and self._include_primitives:
            func_data = self._get_stored_primitive_data_by_name(name=function_name)

        if func_data is None:
            raise ValueError(f"Function '{function_name}' not found")

        # Direct executions (sub-agents, proxy re-entry)
        # feed the usage trace here; primitives are filtered inside.
        self._note_function_use(func_data)

        # Primitive execution: resolve callable and invoke directly.
        if func_data.get("is_primitive"):
            return await self._execute_primitive(
                func_data=func_data,
                call_kwargs=call_kwargs,
                extra_namespaces=ns,
                _parent_chat_context=_parent_chat_context,
            )

        implementation = func_data.get("implementation")
        if not isinstance(implementation, str) or not implementation.strip():
            raise ValueError(f"Function '{function_name}' has no implementation")

        environment.ensure(func_data.get("dependencies") or [])
        return await self._execute_python_function(
            implementation=implementation,
            call_kwargs=call_kwargs or {},
            state_mode=state_mode,
            session_id=session_id,
            extra_namespaces=ns,
            _parent_chat_context=_parent_chat_context,
        )

    # ------------------------------------------------------------------ #
    #  Primitive Execution Helpers                                       #
    # ------------------------------------------------------------------ #

    def _get_primitive_data_by_name(self, *, name: str) -> Optional[Dict[str, Any]]:
        """Look up primitive metadata by name from the in-memory registry."""
        primitives = self._registry.collect_primitives(self._primitive_scope)
        return primitives.get(name)

    def _get_stored_primitive_data_by_name(
        self,
        *,
        name: str,
    ) -> Optional[Dict[str, Any]]:
        """Look up a primitive row by exact name from the primitive catalogue."""
        rows = self._primitive_logs(extra_filter="name = ?", params=(name,), limit=1)
        return dict(rows[0]) if rows else None

    async def _execute_primitive(
        self,
        *,
        func_data: Dict[str, Any],
        call_kwargs: Optional[Dict[str, Any]],
        extra_namespaces: Dict[str, Any],
        _parent_chat_context: Optional[list] = None,
    ) -> Any:
        """Resolve a primitive callable and invoke it directly.

        Returns the raw result of the callable (which may be a
        SteerableToolHandle for async tool loop primitives).
        """
        from unify.function_manager.primitives.runtime import get_primitive_callable

        callable_fn = get_primitive_callable(
            func_data,
            primitives=extra_namespaces.get("primitives"),
        )
        if callable_fn is None:
            raise ValueError(
                f"Could not resolve primitive callable for '{func_data.get('name')}'",
            )

        kwargs = call_kwargs or {}

        # Forward _parent_chat_context if the callable accepts it.
        if _parent_chat_context is not None:
            sig = inspect.signature(callable_fn)
            params = sig.parameters
            if "_parent_chat_context" in params or any(
                p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()
            ):
                kwargs["_parent_chat_context"] = _parent_chat_context

        result = callable_fn(**kwargs)
        if inspect.isawaitable(result):
            result = await result
        return result

    async def _execute_python_function(
        self,
        *,
        implementation: str,
        call_kwargs: Dict[str, Any],
        state_mode: Literal["stateful", "read_only", "stateless"] = "stateless",
        session_id: int = 0,
        extra_namespaces: Optional[Dict[str, Any]] = None,
        _parent_chat_context: Optional[list] = None,
    ) -> Dict[str, Any]:
        """
        Execute a stored function's implementation in-process.

        State modes:
        - stateless: Fresh globals each time (pure function behavior)
        - stateful: Persistent globals per session_id (Jupyter-notebook style)
        - read_only: Reads existing state but doesn't persist changes
        """
        from .execution_env import create_base_globals
        import io
        import traceback
        from contextlib import redirect_stdout, redirect_stderr

        # Determine which globals dict to use based on state_mode
        if state_mode == "stateful":
            # Use persistent session globals
            if session_id not in self._in_process_sessions:
                self._in_process_sessions[session_id] = create_base_globals()
            globals_dict = self._in_process_sessions[session_id]
        elif state_mode == "read_only":
            # Copy state from persistent session into fresh globals
            globals_dict = create_base_globals()
            if session_id in self._in_process_sessions:
                # Copy user-defined state (excluding base globals and dunder names)
                base_keys = set(create_base_globals().keys())
                for key, value in self._in_process_sessions[session_id].items():
                    if key not in base_keys and not key.startswith("_"):
                        # Shallow copy - sufficient for most state
                        globals_dict[key] = value
        else:  # stateless
            globals_dict = create_base_globals()

        # Inject all extra namespaces into globals (always, since they may
        # change between calls).
        if extra_namespaces:
            globals_dict.update(extra_namespaces)

        # Wrap primitives with ContextForwardingProxy so that environment
        # methods called from composed functions receive _parent_chat_context
        # (mirroring the PythonExecutionSession.execute() pattern).
        _orig_prims = globals_dict.get("primitives")
        _wrapped_prims = _orig_prims
        if _parent_chat_context is not None and _wrapped_prims is not None:
            from .primitives.context_proxy import ContextForwardingProxy

            _wrapped_prims = ContextForwardingProxy(
                _wrapped_prims,
                _parent_chat_context=_parent_chat_context,
            )

        # Steering, when a call is in flight. This path runs a stored function
        # directly rather than through the sandbox, so without its own probes
        # the same function would be correctable or not depending purely on
        # which route reached it.
        steering = active_session()
        steering_token = None
        if steering is not None:
            steering_token = bind_session(
                globals_dict,
                steering,
                tool_namespaces=[],
            )
            if _wrapped_prims is not None:
                # Outermost, so memoisation sees a dispatch before context
                # forwarding does.
                _wrapped_prims = MemoisedDispatch(_wrapped_prims, steering)

        if _wrapped_prims is not _orig_prims:
            globals_dict["primitives"] = _wrapped_prims

        stdout_capture = io.StringIO()
        stderr_capture = io.StringIO()
        result = None
        error = None

        async def _run_implementation(source: str) -> Any:
            """Define the function from *source* and call it.

            Name and asyncness are re-derived per attempt: a correction
            rewrites the definition, and nothing guarantees the replacement
            keeps the shape the caller was told about.
            """
            attempt_tree = ast.parse(source)
            definition = next(
                (
                    node
                    for node in attempt_tree.body
                    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                ),
                None,
            )
            if definition is None:
                raise ValueError("No function definition found in implementation")

            to_exec = source
            if steering is not None:
                to_exec = ast.unparse(
                    instrument(
                        attempt_tree,
                        tool_namespaces=set(DEFAULT_TOOL_NAMESPACES),
                    ),
                )

            exec(compile_function_source(definition.name, to_exec), globals_dict)
            fn = globals_dict.get(definition.name)
            if fn is None:
                raise ValueError(
                    f"Function '{definition.name}' not found after exec",
                )
            if isinstance(definition, ast.AsyncFunctionDef):
                return await fn(**call_kwargs)
            return fn(**call_kwargs)

        try:
            with (
                redirect_stdout(
                    stdout_capture,
                ),
                redirect_stderr(stderr_capture),
            ):
                if steering is None:
                    result = await _run_implementation(implementation)
                else:
                    result = await run_with_steering(
                        implementation,
                        _run_implementation,
                        session=steering,
                    )
        except ExecutionStopped as stopped:
            result = stopped.outcome
        except Exception:
            error = traceback.format_exc()
        finally:
            # Stateful sessions reuse this dict across calls, so the probes and
            # the memoised namespace must not outlive the call that installed
            # them.
            if _wrapped_prims is not _orig_prims:
                if _orig_prims is None:
                    globals_dict.pop("primitives", None)
                else:
                    globals_dict["primitives"] = _orig_prims
            if steering_token is not None:
                restore_session(globals_dict, steering_token)

        return {
            "result": self._make_json_serializable(result),
            "error": error,
            "stdout": stdout_capture.getvalue(),
            "stderr": stderr_capture.getvalue(),
        }

    def _make_json_serializable(self, obj: Any) -> Any:
        """Convert an object to a JSON-serializable form."""
        if obj is None or isinstance(obj, (bool, int, float, str)):
            return obj
        if isinstance(obj, (list, tuple)):
            return [self._make_json_serializable(item) for item in obj]
        if isinstance(obj, dict):
            return {str(k): self._make_json_serializable(v) for k, v in obj.items()}
        # Handle Pydantic models
        try:
            from pydantic import BaseModel

            if isinstance(obj, BaseModel):
                return self._make_json_serializable(obj.model_dump())
        except ImportError:
            pass
        # For other types, convert to string representation
        return str(obj)
