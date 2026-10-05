"""``UNIFY_TOOL_SURFACE=core``: ``execute_code`` is the actor's only JSON tool.

As shipped the actor sends 31 JSON tool schemas, about 11.7k tokens, with every
request: the function and guidance libraries alone are 16 of them, though the
harness's own design is that everything is code. With the switch on the model
sees ``execute_code`` (and ``final_response`` when the caller set a response
format, and the loop's steering tools only for an actor that can start
sub-actors); everything else is a Python object in the sandbox:

* ``functions`` -- search, filter, list, get, run, add, patch, delete, retire and
  reconcile_dependencies over the function library;
* ``guidance`` -- search, filter, get, add, update, patch, delete and
  reconcile_dependencies over the guidance library;
* ``install``, ``read_file`` and ``grep`` -- the JSON tools of those names;
* ``request_clarification`` -- where the session can ask.

These are harness objects. A cell reaches them only through the proxy of the
sandboxed Python worker (``UNIFY_WORKSPACE_PYTHON=worker``,
unify/actor/execution/worker.py): a library write or an install runs here, in
the harness, never in model-written code, and a cell never holds the store or
the harness's environment. An actor therefore refuses to start with the switch
on unless worker Python is configured, rather than fall back to in-process
cells, which would hold the real objects; and with the discovery gate on,
which can only force JSON tools.

Writes this session may not make (``can_store``, ``UNIFY_STORE_ADMISSION``,
switches that are off) are refused at call time with the reason, instead of
being left out of a tool list, so the list and the system prompt are fixed for
the session. A short index in the prompt names the objects, and ``help(obj)``
prints their full documentation as cell output.

``functions.run`` and direct calls of stored functions run in the worker,
confined; this module records them (usage, ``UNIFY_FUNCTION_CASES`` cases with
the environment calls they make, ``UNIFY_STORE_TRUST`` evidence, the declared
dependencies installed first) exactly as the ``execute_function`` tool does.

Compression is unchanged: the loop asks for it at the same threshold and
offers ``compress_context`` (and ``store_skills``) on that turn as shipped;
only on the other turns are they left out of the tool list.
"""

from __future__ import annotations

import asyncio
import contextvars
import dataclasses
import inspect
import itertools
import logging
import re
import textwrap
import types
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple, Union

logger = logging.getLogger(__name__)

CORE = "core"

FUNCTIONS = "functions"
GUIDANCE = "guidance"
INSTALL = "install"

#: The state modes ``functions.run`` takes, as ``execute_function`` did.
RUN_STATES = ("stateless", "stateful", "read_only")


def enabled() -> bool:
    """Whether ``UNIFY_TOOL_SURFACE=core`` is set."""
    from unify.settings import SETTINGS

    return getattr(SETTINGS, "UNIFY_TOOL_SURFACE", "") == CORE


class ToolSurfaceError(RuntimeError):
    """A configuration ``UNIFY_TOOL_SURFACE=core`` cannot run under."""


def require_prerequisites(*, can_compose: bool) -> None:
    """Refuse to start a core-surface session the harness cannot confine.

    Raises :class:`ToolSurfaceError` naming what to change. Nothing falls back
    to running model code in this process.
    """
    from unify import sandbox
    from unify.actor.execution import worker
    from unify.settings import SETTINGS

    if SETTINGS.UNIFY_DISCOVERY_GATE:
        raise ToolSurfaceError(
            "UNIFY_TOOL_SURFACE=core needs UNIFY_DISCOVERY_GATE=0: the "
            "discovery gate offers and forces the library's JSON search tools, "
            "which this surface replaces with the sandbox's `functions` and "
            "`guidance` objects.",
        )
    if not worker.enabled():
        raise ToolSurfaceError(
            "UNIFY_TOOL_SURFACE=core needs UNIFY_WORKSPACE=sandboxed and "
            "UNIFY_WORKSPACE_PYTHON=worker: library writes and package installs "
            "are Python calls from cells, and only a sandboxed worker reaches "
            "them through the harness's proxy. Cells run in this process "
            "would hold the real store and the harness's environment.",
        )
    try:
        sandbox.require_bwrap()
    except Exception as exc:
        raise ToolSurfaceError(
            "UNIFY_TOOL_SURFACE=core needs the sandboxed worker, and bubblewrap "
            f"is not available: {exc}",
        ) from None
    if not can_compose:
        raise ToolSurfaceError(
            "UNIFY_TOOL_SURFACE=core needs execute_code (can_compose=True): it "
            "is the surface's only tool.",
        )


def offers_steering(environments: Mapping[str, Any]) -> bool:
    """Whether the loop's steering tools (``wait``, ``steer``,
    ``ask_about_completed_tool``) are offered: only to an actor that can
    delegate to sub-actors, whose calls run alongside its own."""
    from unify.actor.prompt_builders import _injects_actor_primitives

    return _injects_actor_primitives(environments)


# ---------------------------------------------------------------------------
# Documentation
# ---------------------------------------------------------------------------

# The JSON tool names the managers' docstrings use, as the Python surface names them.
_DOC_NAMES = (
    ("``FunctionManager_add_functions``", "``functions.add``"),
    ("FunctionManager_add_functions", "functions.add"),
    ("FunctionManager_retire_case", "functions.retire"),
    ("FunctionManager_search_functions", "functions.search"),
    (
        "``filter_functions``/``list_functions``",
        "``functions.filter``/``functions.list``",
    ),
    ("``get_guidance``", "``guidance.get``"),
)


def public_doc(doc: Optional[str]) -> str:
    """*doc* without the entries of ``_``-prefixed parameters, cleaned, with
    the libraries' JSON tool names given as the Python surface names them."""
    text = inspect.cleandoc(doc or "")
    out: List[str] = []
    skipping = False
    skip_indent = 0
    for line in text.splitlines():
        stripped = line.lstrip()
        indent = len(line) - len(stripped)
        if re.match(r"_[A-Za-z0-9_]* *[:(]", stripped):
            skipping, skip_indent = True, indent
            continue
        if skipping:
            if stripped and indent <= skip_indent:
                skipping = False
            else:
                continue
        out.append(line)
    text = "\n".join(out).strip()
    for old, new in _DOC_NAMES:
        text = text.replace(old, new)
    return text


def _with_doc(doc: str) -> Callable[[Callable], Callable]:
    def _set(fn: Callable) -> Callable:
        fn.__doc__ = doc
        return fn

    return _set


# ---------------------------------------------------------------------------
# What a session may write
# ---------------------------------------------------------------------------

#: What ``can_store=False`` withholds, as the JSON tools it removes
#: (``_store_only_tools`` in the actor).
_STORE_ONLY = frozenset(
    {
        "functions.add",
        "functions.delete",
        "functions.reconcile_dependencies",
        "functions.patch",
        "functions.retire",
        "guidance.reconcile_dependencies",
        "guidance.patch",
    },
)
#: What ``UNIFY_STORE_ADMISSION`` withholds besides (``_admission_withheld_tools``).
_ADMISSION_WITHHELD = _STORE_ONLY | {
    "guidance.add",
    "guidance.update",
    "guidance.delete",
}


@dataclasses.dataclass(frozen=True)
class WritePolicy:
    """What a session may write, fixed when it starts."""

    can_store: bool = True
    admission_gated: bool = False
    inline_mode: str = ""
    #: ``UNIFY_REVIEW_FORK_CORE``: the policy of a forked storage review's
    #: sandbox, which stores and edits the libraries but runs no stored
    #: function and binds none in its namespace (the task is over).
    review: bool = False
    #: ``(method, why)`` pairs refused before anything else (a lessons-only
    #: review's function writes).
    withheld: Tuple[Tuple[str, str], ...] = ()

    def refusal(self, method: str) -> Optional[str]:
        """Why ``method`` (``functions.add``, ...) is refused here; ``None`` if allowed."""
        from unify.settings import SETTINGS

        for name, why in self.withheld:
            if name == method:
                return f"{method} is not available in this review: {why}"
        if self.admission_gated and method in _ADMISSION_WITHHELD:
            return (
                f"{method} is not available in this session: the libraries are "
                "read-only while UNIFY_STORE_ADMISSION decides, after the "
                "session, whether a review may add to them"
            )
        if not self.can_store and method in _STORE_ONLY:
            return (
                f"{method} is not available in this session: it may not write "
                "to the libraries (can_store is off)"
            )
        if method in ("functions.patch", "guidance.patch") and not (
            SETTINGS.UNIFY_FUNCTION_PATCH
        ):
            return f"{method} is off (UNIFY_FUNCTION_PATCH); store a corrected " + (
                "version with functions.add(..., overwrite=True)"
                if method == "functions.patch"
                else "entry with guidance.update(...)"
            )
        if method == "functions.retire" and not SETTINGS.UNIFY_FUNCTION_CASES:
            return (
                "functions.retire is off: no cases are recorded (UNIFY_FUNCTION_CASES)"
            )
        return None

    def writes(self, family: str) -> List[str]:
        """The write methods of ``family`` this session may call."""
        names = {
            FUNCTIONS: ["add", "patch", "delete", "retire"],
            GUIDANCE: ["add", "update", "patch", "delete"],
        }[family]
        return [n for n in names if self.refusal(f"{family}.{n}") is None]


REVIEW_RUN_REFUSAL = (
    "the storage review stores and edits the libraries; it does not run "
    "stored functions (the task is over)"
)


def _refuse(policy: WritePolicy, method: str) -> None:
    reason = policy.refusal(method)
    if reason is not None:
        raise PermissionError(reason)


# ---------------------------------------------------------------------------
# The sandbox objects
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class _Run:
    """One recorded call of a stored function, between its begin and its end."""

    name: str
    mode: str
    func_data: dict
    observer: Any = None
    arguments: Any = None
    recorder: Any = None
    pending: Any = None
    steering_seen: int = 0
    publish: Optional[Callable[..., Any]] = None


class FunctionLibrary:
    """The function library: reusable functions stored from earlier work.

    Every method is awaited (``await functions.search("parse dates")``).
    Reads: ``search``, ``filter``, ``list``, ``get``. ``run`` calls a stored
    function by name; a stored function a search, filter, list or get returned
    is also callable by name from the next cell. Writes: ``add``, ``patch``,
    ``delete``, ``retire``, ``reconcile_dependencies``; one this session may
    not make raises ``PermissionError`` saying why. ``help(functions.<method>)``
    prints a method's full contract.
    """

    def __init__(self, actor: Any, policy: WritePolicy) -> None:
        self._actor = actor
        self._fm = actor.function_manager
        self._policy = policy
        self._ids = itertools.count(1)
        self._runs: Dict[int, _Run] = {}

    def __repr__(self) -> str:
        return "<functions: the function library (help(functions) for its methods)>"

    # -- reads ---------------------------------------------------------------

    def _load(self, method: str, *, _sandbox: Any = None, **kwargs: Any) -> Any:
        """A read that also binds the functions it returns in the session, as
        the actor's JSON search tools do (in *_sandbox*, else the running
        cell's). A review's read binds nothing, as the review's JSON search
        tools do not."""
        from unify.actor.execution import _CURRENT_SANDBOX

        if self._policy.review:
            return getattr(self._fm, method)(**kwargs)
        sb = _sandbox if _sandbox is not None else _CURRENT_SANDBOX.get(None)
        namespace = sb.global_state if sb is not None else {}
        before = set(namespace)
        result = getattr(self._fm, method)(
            **kwargs,
            _return_callable=True,
            _namespace=namespace,
            _also_return_metadata=True,
        )
        new = set(namespace) - before
        if new and sb is not None:
            self._actor._session_executor.register_fm_globals(
                {k: namespace[k] for k in new},
            )
        return result["metadata"]

    def _bind_names(
        self,
        names: List[str],
        *,
        sandbox: Any = None,
    ) -> Dict[str, bool]:
        """Bind the stored functions *names* as ``get`` binds one; ``{name: is_async}``.

        ``UNIFY_CORE_BIND_LISTED`` (the shortlist's functions, at task start)
        and ``UNIFY_GUIDANCE_LINKED_NAMES`` (the functions a guidance read
        names). One filter read, as ``get`` makes, so no search hit is
        counted; a name the read does not load (deleted, quarantined,
        unloadable) is left out of the result.
        """
        from unify.actor.execution import _CURRENT_SANDBOX

        wanted = list(dict.fromkeys(str(n) for n in names if n))
        if not wanted:
            return {}
        sb = sandbox if sandbox is not None else _CURRENT_SANDBOX.get(None)
        if sb is None:
            return {}
        quoted = ", ".join("'" + n.replace("'", "''") + "'" for n in wanted)
        rows = self._load(
            "filter_functions",
            _sandbox=sb,
            filter=f"name IN ({quoted})",
            offset=0,
            limit=len(wanted),
            include_implementations=False,
        )
        loaded = {
            str(row.get("name"))
            for row in (rows if isinstance(rows, list) else [])
            if isinstance(row, Mapping) and row.get("name")
        }
        namespace = sb.global_state
        return {
            name: _is_async_function(namespace.get(name))
            for name in wanted
            if name in loaded and name in namespace
        }

    async def search(
        self,
        query: str = "",
        n: int = 5,
        include_implementations: bool = True,
    ) -> List[Dict[str, Any]]:
        return self._load(
            "search_functions",
            query=query,
            n=n,
            include_implementations=include_implementations,
        )

    async def filter(
        self,
        filter: Optional[str] = None,
        offset: int = 0,
        limit: int = 100,
        include_implementations: bool = True,
    ) -> List[Dict[str, Any]]:
        return self._load(
            "filter_functions",
            filter=filter,
            offset=offset,
            limit=limit,
            include_implementations=include_implementations,
        )

    async def list(self, include_implementations: bool = False) -> Any:
        return self._load(
            "list_functions",
            include_implementations=include_implementations,
        )

    async def get(self, name: str) -> Optional[Dict[str, Any]]:
        """One stored function by exact name, with its implementation, or ``None``.

        The function is then callable by name from the next cell.
        """
        quoted = str(name).replace("'", "''")
        rows = self._load(
            "filter_functions",
            filter=f"name = '{quoted}'",
            offset=0,
            limit=1,
            include_implementations=True,
        )
        return rows[0] if isinstance(rows, list) and rows else None

    async def run(
        self,
        name: str,
        /,
        *,
        state: str = "stateless",
        **kwargs: Any,
    ) -> Any:
        """Call a stored function by name, with keyword arguments, and return what it returns.

        ``state`` is where it runs: ``"stateless"`` (the default) in fresh
        globals (the sandbox's objects and the stored functions found so far,
        none of the cells' variables); ``"stateful"`` in this session's
        namespace, where it stays defined; ``"read_only"`` in a copy of this
        session's namespace, discarded afterwards. The stored functions it
        calls are defined with it, and its and their declared dependencies
        are installed first. What it raises is raised here. A dotted
        ``primitives.*`` name, or a function defined in this session, is
        called as it is. A stored function can also be called by name, once
        found, like any other function.

        Example: ``total = await functions.run("sum_invoice_lines", invoice_id=7)``
        """
        raise RuntimeError(
            "functions.run runs stored code in the sandboxed Python worker, "
            "and this cell is not running in one",
        )

    # -- writes --------------------------------------------------------------

    async def _write(self, method: str, fn: Callable[..., Any], **kwargs: Any):
        _refuse(self._policy, method)
        return await asyncio.to_thread(fn, **kwargs)

    def _add_fn(self) -> Callable[..., Any]:
        from unify.function_manager import inline_curation

        fn = self._fm.add_functions
        if self._policy.inline_mode:
            fn = inline_curation.guard_add_functions(fn)
        return fn

    async def add(
        self,
        implementations: Union[str, List[str]],
        *,
        preconditions: Optional[Dict[str, Dict]] = None,
        overwrite: bool = False,
        raise_on_error: bool = True,
        dependencies: Optional[List[str]] = None,
    ) -> Dict[str, str]:
        return await self._write(
            "functions.add",
            self._add_fn(),
            implementations=implementations,
            preconditions=preconditions,
            overwrite=overwrite,
            raise_on_error=raise_on_error,
            dependencies=dependencies,
        )

    async def patch(
        self,
        name: str,
        old: Optional[str] = None,
        new: Optional[str] = None,
        *,
        why: str,
        edits: Optional[List[Dict[str, Any]]] = None,
        replace_all: bool = False,
    ) -> Dict[str, Any]:
        _refuse(self._policy, "functions.patch")
        from unify.function_manager import inline_curation

        fn = self._fm.patch_function
        if self._policy.inline_mode:
            fn = inline_curation.guard_patch_function(fn)
        return await self._write(
            "functions.patch",
            fn,
            name=name,
            old=old,
            new=new,
            why=why,
            edits=edits,
            replace_all=replace_all,
        )

    async def delete(
        self,
        function_id: int,
        *,
        delete_dependents: bool = True,
    ) -> Dict[str, str]:
        return await self._write(
            "functions.delete",
            self._fm.delete_function,
            function_id=function_id,
            delete_dependents=delete_dependents,
        )

    async def retire(
        self,
        function_name: str,
        case_id: int,
        why: str,
    ) -> Dict[str, Any]:
        _refuse(self._policy, "functions.retire")
        return await self._write(
            "functions.retire",
            self._fm.retire_case,
            function_name=function_name,
            case_id=case_id,
            why=why,
        )

    async def reconcile_dependencies(
        self,
        function_ids: Optional[List[int]] = None,
    ) -> Dict[str, Any]:
        return await self._write(
            "functions.reconcile_dependencies",
            self._fm.reconcile_dependencies,
            function_ids=function_ids,
        )

    # -- recording calls that run in the worker --------------------------------

    def _lookup(self, name: str) -> Optional[dict]:
        """The stored function ``name``, as the ``execute_function`` tool resolves it."""
        fm = self._fm
        get = getattr(fm, "_get_function_data_by_name", None)
        data = get(name=name) if callable(get) else None
        if data is None:
            get_primitive = getattr(fm, "_get_stored_primitive_data_by_name", None)
            if callable(get_primitive):
                data = get_primitive(name=name)
        return dict(data) if isinstance(data, Mapping) else None

    def _note_use(self, func_data: dict) -> None:
        note_use = getattr(self._fm, "_note_function_use", None)
        if callable(note_use):
            try:
                note_use(func_data)
            except Exception:  # noqa: BLE001 - never break a call
                pass

    @staticmethod
    def _steering_seen() -> int:
        from unify.function_manager.steering import active_session

        steering = active_session()
        return len(getattr(steering, "messages", None) or []) if steering else 0

    def _case_pending(self, token: Any) -> Any:
        """The case a running stored function records into, for its environment calls."""
        run = self._runs.get(int(token)) if isinstance(token, int) else None
        return run.pending if run is not None else None

    async def _begin(
        self,
        *,
        name: str,
        mode: str,
        args: List[Any],
        kwargs: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Start recording a stored-function call the worker is about to run.

        ``mode`` is ``"run"`` (``functions.run``: as the ``execute_function``
        tool -- usage, trust, case, dependencies installed, events) or
        ``"call"`` (a direct call: as the boundary wrapper of an in-process
        session -- usage, trust, case). Returns ``{"found": False}`` for a
        name the library does not hold, ``{"found": True, "primitive": True}``
        for a primitive, and otherwise the token the call's environment calls
        and its end carry, with the source ``functions.run`` defines.
        """
        from unify import environment
        from unify.function_manager import store_cases
        from unify.function_manager.source_labels import function_source_filename
        from unify.function_manager.store_trust import CallObserver, source_signature

        if self._policy.review:
            raise PermissionError(REVIEW_RUN_REFUSAL)
        func_data = self._lookup(name)
        if func_data is None:
            return {"found": False}
        impl = func_data.get("implementation")
        if (
            func_data.get("is_primitive")
            or not isinstance(impl, str)
            or not impl.strip()
        ):
            if mode == "run":
                self._note_use(func_data)
            return {"found": True, "primitive": True}

        run = _Run(name=name, mode=mode, func_data=func_data)
        helpers = self._helpers(func_data) if mode == "run" else []
        if mode == "call":
            # The boundary wrapper's order: usage, trust, case.
            self._note_use(func_data)
        bind_to = None
        if mode == "call":
            signature = source_signature(impl, name)
            if signature is not None:

                def bind_to(*a: Any, **k: Any) -> None:  # noqa: ARG001
                    return None

                bind_to.__signature__ = signature  # type: ignore[attr-defined]
        run.observer = CallObserver.for_function(self._fm, func_data)
        if run.observer is not None:
            run.arguments = run.observer.before(bind_to, tuple(args), kwargs)
        run.recorder = store_cases.CaseRecorder.for_function(func_data)
        if run.recorder is not None:
            run.pending = run.recorder.begin(tuple(args), kwargs)
        if mode == "run":
            # A helper's packages are installed with the entry point's, as a
            # read that loads the entry point installs them.
            deps = list(
                dict.fromkeys(
                    spec
                    for data in (func_data, *helpers)
                    for spec in (data.get("dependencies") or [])
                ),
            )
            if deps:
                try:
                    await asyncio.to_thread(environment.ensure, deps)
                except Exception as exc:
                    # The function's own install failed, whatever the
                    # arguments: a plain dict carries no caller fault.
                    if run.observer is not None:
                        run.observer.after(dict(run.arguments), exc)
                    if run.pending is not None:
                        run.pending.trace.closed = True
                    raise
            self._note_use(func_data)
            run.publish = await _run_events(name)
        run.steering_seen = self._steering_seen()
        token = next(self._ids)
        self._runs[token] = run
        entry = _entry_name(impl) or name
        return {
            "found": True,
            "token": token,
            "fn_name": entry,
            "source": impl if mode == "run" else None,
            "filename": function_source_filename(entry),
            "helpers": [
                {
                    "name": data["name"],
                    "source": data["implementation"],
                    "filename": function_source_filename(data["name"]),
                }
                for data in helpers
            ],
        }

    def _helpers(self, func_data: dict) -> List[dict]:
        """The stored functions ``func_data`` calls, transitively, in the order
        found: what ``functions.run`` defines beside it in the worker, as a
        read that loads a function injects its ``depends_on`` (each name once,
        so recursion and cycles end; primitives, dotted names and names the
        library does not hold are left out, and fail as the call reaches them)."""
        seen = {str(func_data.get("name"))}
        queue = list(func_data.get("depends_on") or [])
        found: List[dict] = []
        while queue:
            dep = queue.pop(0)
            if not isinstance(dep, str) or not dep or "." in dep or dep in seen:
                continue
            seen.add(dep)
            data = self._lookup(dep)
            impl = data.get("implementation") if data else None
            if (
                data is None
                or data.get("is_primitive")
                or not (isinstance(impl, str) and impl.strip())
            ):
                continue
            found.append({**data, "name": data.get("name") or dep})
            queue.extend(data.get("depends_on") or [])
        return found

    async def _end(
        self,
        *,
        token: Any,
        result: Any = None,
        error: Optional[str] = None,
        abandoned: bool = False,
    ) -> Dict[str, Any]:
        """Record how a call the worker ran ended; ``{"note": ...}`` when a
        failure is not held against the function."""
        run = self._runs.pop(token, None) if isinstance(token, int) else None
        if run is None:
            return {}
        reply: Dict[str, Any] = {}
        steered = abandoned or self._steering_seen() > run.steering_seen
        if run.pending is not None and steered:
            run.pending.trace.closed = True
        if not steered:
            if run.recorder is not None:
                run.recorder.end(run.pending, result=result, error=error)
            if run.observer is not None:
                run.observer.after(run.arguments, error)
                fault = getattr(run.arguments, "caller_fault", None)
                if error and fault:
                    reply["note"] = (
                        f"Not counted against the stored function `{run.name}`: "
                        f"{fault}."
                    )
        if run.publish is not None:
            await run.publish(error)
        return reply


def _entry_name(source: str) -> Optional[str]:
    import ast

    try:
        tree = ast.parse(source)
    except SyntaxError:
        return None
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            return node.name
    return None


async def _run_events(name: str) -> Callable[..., Any]:
    """Publish a ``functions.run`` call's incoming event; returns the
    coroutine function that publishes its outgoing one."""
    from secrets import token_hex

    from unify.common._async_tool.loop_config import TOOL_LOOP_LINEAGE
    from unify.common.hierarchical_logger import log_boundary_event
    from unify.events.manager_event_logging import (
        new_call_id,
        publish_manager_method_event,
    )

    call_id = new_call_id()
    parent = TOOL_LOOP_LINEAGE.get([])
    hierarchy = [
        *(list(parent) if isinstance(parent, list) else []),
        f"functions.run({name})({token_hex(2)})",
    ]

    async def publish(**payload: Any) -> None:
        try:
            await publish_manager_method_event(
                call_id,
                "CodeActActor",
                "functions.run",
                hierarchy=hierarchy,
                display_label=f"Running: {name}",
                **payload,
            )
        except Exception as exc:  # noqa: BLE001 - events never break a call
            log_boundary_event(
                "->".join(hierarchy),
                f"Warning: failed to publish event: {type(exc).__name__}: {exc}",
                icon="⚠️",
                level="warning",
            )

    await publish(phase="incoming")
    log_boundary_event(
        "->".join(hierarchy),
        f"Running stored function {name}...",
        icon="🛠️",
    )

    async def outgoing(error: Optional[str]) -> None:
        if error:
            await publish(phase="outgoing", status="error", error=str(error)[:2000])
        else:
            await publish(phase="outgoing", status="ok")

    return outgoing


class GuidanceLibrary:
    """The guidance library: stored procedures, walkthroughs and composition strategies.

    Every method is awaited (``await guidance.search("deploy a release")``).
    Reads: ``search``, ``filter``, ``get``. Writes: ``add``, ``update``,
    ``patch``, ``delete``, ``reconcile_dependencies``; one this session may not
    make raises ``PermissionError`` saying why. Search and filter results carry
    a preview of long entries: read one in full with ``get`` before following
    it. ``help(guidance.<method>)`` prints a method's full contract.
    """

    def __init__(
        self,
        guidance_manager: Any,
        policy: WritePolicy,
        functions: Optional["FunctionLibrary"] = None,
    ) -> None:
        self._gm = guidance_manager
        self._policy = policy
        self._functions = functions

    def __repr__(self) -> str:
        return "<guidance: the guidance library (help(guidance) for its methods)>"

    def _bind_linked(self, read: Any) -> Any:
        """``UNIFY_GUIDANCE_LINKED_NAMES``: bind the functions *read* names.

        A guidance read then shows each linked function's name and signature
        (``linked_functions``); binding them, as a ``functions.get`` would,
        makes those names callable from the next cell. Off: *read* as is.
        """
        from unify.settings import SETTINGS

        if not SETTINGS.UNIFY_GUIDANCE_LINKED_NAMES or self._functions is None:
            return read
        entries = read if isinstance(read, list) else [read]
        names: List[str] = []
        for entry in entries:
            for text in getattr(entry, "linked_functions", None) or []:
                names.append(str(text).split("(", 1)[0].strip())
        if names:
            try:
                self._functions._bind_names(names)
            except Exception as exc:  # noqa: BLE001 - the read stands without it
                logger.warning(
                    "could not load the functions guidance links: %s: %s",
                    type(exc).__name__,
                    exc,
                )
        return read

    async def search(
        self,
        references: Union[str, Dict[str, str], None] = None,
        k: int = 10,
    ) -> List[Any]:
        if isinstance(references, str):
            references = {"content": references} if references.strip() else None
        return self._bind_linked(
            await asyncio.to_thread(self._gm.search, references=references, k=k),
        )

    async def filter(
        self,
        filter: Optional[str] = None,
        offset: int = 0,
        limit: int = 100,
    ) -> List[Any]:
        return self._bind_linked(
            await asyncio.to_thread(
                self._gm.filter,
                filter=filter,
                offset=offset,
                limit=limit,
            ),
        )

    async def get(self, guidance_id: int) -> Any:
        return self._bind_linked(
            await asyncio.to_thread(self._gm.get_guidance, guidance_id=guidance_id),
        )

    async def _write(self, method: str, fn: Callable[..., Any], **kwargs: Any):
        _refuse(self._policy, method)
        return await asyncio.to_thread(fn, **kwargs)

    async def add(
        self,
        title: Optional[str] = None,
        content: Optional[str] = None,
        function_ids: Optional[List[int]] = None,
    ) -> Any:
        return await self._write(
            "guidance.add",
            self._gm.add_guidance,
            title=title,
            content=content,
            function_ids=function_ids,
        )

    async def update(
        self,
        guidance_id: int,
        *,
        title: Optional[str] = None,
        content: Optional[str] = None,
        function_ids: Optional[List[int]] = None,
    ) -> Any:
        return await self._write(
            "guidance.update",
            self._gm.update_guidance,
            guidance_id=guidance_id,
            title=title,
            content=content,
            function_ids=function_ids,
        )

    async def patch(
        self,
        id_or_title: Union[int, str],
        old: Optional[str] = None,
        new: Optional[str] = None,
        *,
        why: str,
        edits: Optional[List[Dict[str, Any]]] = None,
        replace_all: bool = False,
    ) -> Any:
        _refuse(self._policy, "guidance.patch")
        return await self._write(
            "guidance.patch",
            self._gm.patch_guidance,
            id_or_title=id_or_title,
            old=old,
            new=new,
            why=why,
            edits=edits,
            replace_all=replace_all,
        )

    async def delete(self, guidance_id: int) -> Any:
        return await self._write(
            "guidance.delete",
            self._gm.delete_guidance,
            guidance_id=guidance_id,
        )

    async def reconcile_dependencies(
        self,
        guidance_ids: Optional[List[int]] = None,
    ) -> Any:
        return await self._write(
            "guidance.reconcile_dependencies",
            self._gm.reconcile_dependencies,
            guidance_ids=guidance_ids,
        )


_INSTALL_DOC = """Install packages into the workspace environment.

The environment is one persistent venv shared by every task and session, so
a package installed once stays importable in later cells; try the import
first. Never install from a cell (pip, uv or a subprocess): the sandbox has
no network for it, and an install there would bypass the managed
environment. When a stored function needs a package, record the specifier as
one of its ``dependencies`` so the install repeats wherever it runs.

``packages`` is a pip/uv specifier or a list of them (``"pandas"``,
``"pandas==2.1.0"``, ``"pandas[sql]"``, a git URL). Returns ``success``, the
installer's ``stdout``/``stderr`` (on failure, read ``stderr`` and adjust the
specifiers) and the requested ``packages``. A package that conflicts with the
runtime's own dependencies keeps the runtime's version.

Example: ``await install(["pandas>=2"])``
"""


async def install(packages: Union[str, List[str]]) -> Dict[str, Any]:
    from unify import environment

    specs = [packages] if isinstance(packages, str) else list(packages)
    return await asyncio.to_thread(environment.install, specs)


install.__doc__ = _INSTALL_DOC


def _file_tools() -> Dict[str, Callable[..., Any]]:
    """``read_file`` and ``grep``: what the JSON tools of the same names do
    under ``UNIFY_WORKSPACE=sandboxed`` (unify/actor/workspace_tools.py), as
    awaitables a cell calls."""
    from unify.actor.workspace_tools import workspace_tools

    async def _never(*_: Any, **__: Any) -> Any:  # execute_code is not used
        raise RuntimeError("unreachable")

    tools = workspace_tools(_never)
    out: Dict[str, Callable[..., Any]] = {}
    for name in ("read_file", "grep"):
        fn = tools[name].fn
        out[name] = fn
    return out


def _document() -> None:
    """Give the library methods the managers' own contracts as their docs."""
    from unify.function_manager.base import BaseFunctionManager
    from unify.function_manager.function_manager import FunctionManager
    from unify.guidance_manager.base import BaseGuidanceManager
    from unify.guidance_manager.guidance_manager import GuidanceManager

    pairs = (
        (FunctionLibrary.search, BaseFunctionManager.search_functions),
        (FunctionLibrary.filter, BaseFunctionManager.filter_functions),
        (FunctionLibrary.list, BaseFunctionManager.list_functions),
        (FunctionLibrary.add, BaseFunctionManager.add_functions),
        (FunctionLibrary.patch, FunctionManager.patch_function),
        (FunctionLibrary.delete, BaseFunctionManager.delete_function),
        (FunctionLibrary.retire, FunctionManager.retire_case),
        (
            FunctionLibrary.reconcile_dependencies,
            BaseFunctionManager.reconcile_dependencies,
        ),
        (GuidanceLibrary.filter, BaseGuidanceManager.filter),
        (GuidanceLibrary.get, BaseGuidanceManager.get_guidance),
        (GuidanceLibrary.add, BaseGuidanceManager.add_guidance),
        (GuidanceLibrary.update, BaseGuidanceManager.update_guidance),
        (GuidanceLibrary.patch, GuidanceManager.patch_guidance),
        (GuidanceLibrary.delete, BaseGuidanceManager.delete_guidance),
        (
            GuidanceLibrary.reconcile_dependencies,
            BaseGuidanceManager.reconcile_dependencies,
        ),
    )
    for method, source in pairs:
        method.__doc__ = public_doc(source.__doc__)
    GuidanceLibrary.search.__doc__ = (
        public_doc(BaseGuidanceManager.search.__doc__)
        + "\n\nA plain string is compared with the entries' ``content``; pass a "
        "mapping to choose the fields."
    )


_DOCUMENTED = False


def sandbox_objects(
    actor: Any,
    *,
    policy: WritePolicy,
) -> Dict[str, Any]:
    """The objects a core-surface session's sandbox holds, by global name."""
    global _DOCUMENTED
    if not _DOCUMENTED:
        _document()
        _DOCUMENTED = True
    objects: Dict[str, Any] = {INSTALL: install, **_file_tools()}
    if getattr(actor, "function_manager", None) is not None:
        objects[FUNCTIONS] = FunctionLibrary(actor, policy)
    if getattr(actor, "guidance_manager", None) is not None:
        objects[GUIDANCE] = GuidanceLibrary(
            actor.guidance_manager,
            policy,
            objects.get(FUNCTIONS),
        )
    return objects


def _is_async_function(value: Any) -> bool:
    """Whether the bound stored function *value* is ``async def`` (unwrapped)."""
    try:
        return inspect.iscoroutinefunction(inspect.unwrap(value))
    except Exception:  # noqa: BLE001 - a wrapper that does not unwrap
        return False


# ---------------------------------------------------------------------------
# UNIFY_REVIEW_FORK_CORE: the forked storage review's sandbox
# ---------------------------------------------------------------------------


def review_fork_enabled() -> bool:
    """Whether ``UNIFY_REVIEW_FORK_CORE`` lets a core session's review fork."""
    from unify.settings import SETTINGS

    return bool(getattr(SETTINGS, "UNIFY_REVIEW_FORK_CORE", False))


def review_fork_refusal(tool_names: List[str]) -> Optional[str]:
    """Why a core session's review cannot fork, or ``None`` when it can.

    *tool_names* are those of the session's last request: the fork reuses
    that list, and its ``execute_code`` is what the review stores through.
    """
    from unify.actor.execution import worker as worker_mod
    from unify.function_manager import store_verify

    if "execute_code" not in tool_names:
        return "the session's tool list has no execute_code for the review's cells"
    if not worker_mod.enabled():
        return (
            "UNIFY_REVIEW_FORK_CORE runs the review's cells in the sandboxed "
            "worker, which needs UNIFY_WORKSPACE=sandboxed and "
            "UNIFY_WORKSPACE_PYTHON=worker"
        )
    if store_verify.enabled():
        return (
            "UNIFY_STORE_VERIFY checks a function before it is stored, and the "
            "review's sandbox has no method for that check"
        )
    return None


# The JSON tool names the storage rulebook uses, as the sandbox names them;
# longer names first, so no name is replaced inside another. The function
# check (UNIFY_STORE_VERIFY) has no sandbox method and a core review does not
# fork with it on, so a lessons-only review's list of refused writes drops it.
_REVIEW_NAMES = (
    ("`FunctionManager_check_function`, ", ""),
    ("FunctionManager_reconcile_dependencies", "functions.reconcile_dependencies"),
    ("GuidanceManager_reconcile_dependencies", "guidance.reconcile_dependencies"),
    ("FunctionManager_search_functions", "functions.search"),
    ("FunctionManager_filter_functions", "functions.filter"),
    ("FunctionManager_list_functions", "functions.list"),
    ("FunctionManager_add_functions", "functions.add"),
    ("FunctionManager_delete_function", "functions.delete"),
    ("FunctionManager_patch_function", "functions.patch"),
    ("FunctionManager_retire_case", "functions.retire"),
    ("GuidanceManager_update_guidance", "guidance.update"),
    ("GuidanceManager_delete_guidance", "guidance.delete"),
    ("GuidanceManager_patch_guidance", "guidance.patch"),
    ("GuidanceManager_add_guidance", "guidance.add"),
    ("GuidanceManager_get_guidance", "guidance.get"),
    ("GuidanceManager_search", "guidance.search"),
    ("GuidanceManager_filter", "guidance.filter"),
    ("install_python_packages", "install"),
)


def python_names(text: str) -> str:
    """*text* with the libraries' JSON tool names as the sandbox names them."""
    for old, new in _REVIEW_NAMES:
        text = text.replace(old, new)
    return text


def review_policy(
    *,
    lesson_refusals: Optional[Mapping[str, str]] = None,
) -> WritePolicy:
    """The writes of a forked review's sandbox: every library write the
    session's switches allow, less a lessons-only review's (*lesson_refusals*:
    JSON tool name -> why), which are refused saying why."""
    withheld = tuple(
        (python_names(name), why)
        for name, why in (lesson_refusals or {}).items()
        if python_names(name) != name
    )
    return WritePolicy(review=True, withheld=withheld)


class ReviewSandbox:
    """Where a forked storage review's ``execute_code`` cells run.

    A sandbox of its own, in the confined worker, holding only ``functions``
    and ``guidance`` with the review's writes: the session's sandbox (its
    environment, files and variables) is closed when the task ends, and the
    review is not given another. The worker starts with the first cell; the
    owner closes it when the review ends.
    """

    def __init__(self, actor: Any, policy: WritePolicy) -> None:
        self._actor = actor
        self._policy = policy
        self._session: Any = None

    def _sandbox(self) -> Any:
        if self._session is None:
            from unify.actor.execution.session import PythonExecutionSession

            objects = {
                name: obj
                for name, obj in sandbox_objects(
                    self._actor,
                    policy=self._policy,
                ).items()
                if name in (FUNCTIONS, GUIDANCE)
            }
            session = PythonExecutionSession(environments={})
            session.global_state.update(objects)
            session.core_globals = dict(objects)
            self._session = session
        return self._session

    async def execute_code(
        self,
        thought: str = "",
        code: Optional[str] = None,
        *,
        language: str = "python",
        state_mode: Optional[str] = None,
        session_id: Optional[int] = None,
        session_name: Optional[str] = None,
    ) -> Any:
        """Run a Python cell in the review's sandbox (``functions`` and
        ``guidance`` only). Every cell runs in the same namespace."""
        from unify.actor.execution import _CURRENT_SANDBOX
        from unify.actor.execution.types import ExecutionResult

        _ = (thought, state_mode, session_id, session_name)
        if language != "python":
            return {
                "error": (
                    f"The storage review runs Python cells only, not {language!r}: "
                    "its sandbox holds the function and guidance libraries."
                ),
            }
        if code is None or not code.strip():
            return {"stdout": "", "stderr": "", "result": None, "error": None}
        sandbox = self._sandbox()
        token = _CURRENT_SANDBOX.set(sandbox)
        try:
            out = await sandbox.execute(code)
        finally:
            _CURRENT_SANDBOX.reset(token)
        if isinstance(out.get("stdout"), list):
            return ExecutionResult(**out)
        return out

    async def close(self) -> None:
        session, self._session = self._session, None
        if session is not None:
            await session.close()


# ---------------------------------------------------------------------------
# help(): the documentation of a harness object, as text
# ---------------------------------------------------------------------------


class _Hidden:
    """A default value help() does not show (a credential-named parameter's)."""

    def __repr__(self) -> str:
        return "<hidden>"


def _signature(obj: Any) -> str:
    try:
        sig = inspect.signature(obj, eval_str=True)
    except Exception:  # noqa: BLE001 - an annotation that does not evaluate
        try:
            sig = inspect.signature(obj)
        except (TypeError, ValueError):
            return "(...)"
    from unify import sandbox

    params = [
        (
            p.replace(default=_Hidden())
            if p.default is not p.empty and sandbox.is_secret_name(p.name)
            else p
        )
        for p in sig.parameters.values()
        if not p.name.startswith("_")
    ]
    try:
        return str(sig.replace(parameters=params, return_annotation=sig.empty))
    except ValueError:
        return "(...)"


#: The harness objects a core-surface sandbox holds, in the order help() lists them.
SANDBOX_NAMES = (
    FUNCTIONS,
    GUIDANCE,
    INSTALL,
    "read_file",
    "grep",
    "request_clarification",
)


def index_help(namespace: Mapping[str, Any]) -> str:
    """What ``help()`` with no argument prints in a core-surface cell."""
    lines = ["Objects the harness provides in this sandbox (all awaited):", ""]
    for name in SANDBOX_NAMES:
        if name not in namespace:
            continue
        doc = public_doc(inspect.getdoc(namespace[name]) or "").split("\n\n")[0]
        lines.append(f"  {name}: " + " ".join(doc.split()))
    lines += [
        "",
        "help(obj) prints the full documentation of one of them or of a method.",
    ]
    return "\n".join(lines) + "\n"


def help_text(obj: Any, label: str) -> str:
    """What ``help(obj)`` prints in a cell for a harness object named ``label``.

    Signatures and docstrings only: never the value of an attribute, so a
    help call shows nothing a cell could not otherwise call.
    """
    lines: List[str] = []
    is_async = inspect.iscoroutinefunction(obj) or inspect.iscoroutinefunction(
        getattr(obj, "__call__", None),
    )
    if callable(obj) and not inspect.isclass(obj):
        target = obj if inspect.isroutine(obj) else getattr(obj, "__call__", obj)
        prefix = "await " if is_async else ""
        lines.append(f"{prefix}{label}{_signature(target)}")
        doc = inspect.getdoc(obj) or ""
        if doc:
            lines += ["", textwrap.indent(public_doc(doc), "    ")]
        return "\n".join(lines).rstrip() + "\n"
    lines.append(label)
    doc = inspect.getdoc(obj) or ""
    if doc:
        lines += ["", textwrap.indent(public_doc(doc), "    ")]
    methods: List[str] = []
    for attr in sorted(n for n in dir(obj) if not n.startswith("_")):
        try:
            member = getattr(type(obj), attr, None)
            if member is None or isinstance(member, property):
                continue
            value = getattr(obj, attr)
        except Exception:  # noqa: BLE001 - an unreadable member is left out
            continue
        if not callable(value):
            continue
        first = (public_doc(inspect.getdoc(value) or "").split("\n\n")[0]).strip()
        first = " ".join(first.split())
        prefix = "await " if inspect.iscoroutinefunction(value) else ""
        methods.append(f"  {prefix}{label}.{attr}{_signature(value)}")
        if first:
            methods.append(textwrap.indent(textwrap.fill(first, 72), "      "))
    if methods:
        lines += ["", "Methods:", *methods]
    return "\n".join(lines).rstrip() + "\n"


# ---------------------------------------------------------------------------
# The execute_code tool
# ---------------------------------------------------------------------------

_SINGLE_CALL_RULE = re.compile(
    r"\n[ \t]*\*\*IMPORTANT — single-call rule\*\*:.*?\n[ \t]*\n",
    re.DOTALL,
)
_SESSION_TOOLS = re.compile(
    r";?\s*choose via\s*``list_sessions\(\)`` / ``inspect_state\(\)``\.",
)
_STEERING_DOC = re.compile(
    r"\n[ \t]*Steering while the block runs\n[ \t]*-+\n.*?(?=\n[ \t]*\n[ \t]*\S[^\n]*\n[ \t]*-{3,}|\Z)",
    re.DOTALL,
)


def execute_code_doc(doc: str, *, steering: bool) -> str:
    """``execute_code``'s description for the core surface: no
    ``execute_function`` to prefer, no session tools to choose a session
    with, and the steering section only where the steering tools exist."""
    doc = _SINGLE_CALL_RULE.sub("\n", doc, count=1)
    doc = _SESSION_TOOLS.sub(".", doc, count=1)
    doc = doc.replace("FunctionManager-discovered", "library", 1)
    if not steering:
        doc = _STEERING_DOC.sub("", doc, count=1)
    else:
        # The stop_* tools this names were replaced by ``steer``.
        doc = doc.replace(
            "``stop_execute_code_<call_id>`` abandons",
            '``steer(call_id=<id>, action="stop")`` abandons',
            1,
        )
    return doc.rstrip() + "\n"


def _copy_function(fn: Callable[..., Any], doc: str) -> Callable[..., Any]:
    """*fn* with another docstring: same code, defaults, closure and attributes."""
    copied = types.FunctionType(
        fn.__code__,
        fn.__globals__,
        fn.__name__,
        fn.__defaults__,
        fn.__closure__,
    )
    copied.__kwdefaults__ = dict(fn.__kwdefaults__ or {}) or None
    copied.__annotations__ = dict(getattr(fn, "__annotations__", {}) or {})
    copied.__dict__.update(getattr(fn, "__dict__", {}))
    copied.__qualname__ = fn.__qualname__
    copied.__module__ = fn.__module__
    copied.__doc__ = doc
    return copied


def core_tools(tools: Mapping[str, Any], *, steering: bool) -> Dict[str, Any]:
    """The session's JSON tools: ``execute_code`` alone, described for this surface."""
    from unify.common.tool_spec import ToolSpec

    tool = tools.get("execute_code")
    if tool is None:
        return {}
    fn = tool.fn if isinstance(tool, ToolSpec) else tool
    copied = _copy_function(fn, execute_code_doc(fn.__doc__ or "", steering=steering))
    if isinstance(tool, ToolSpec):
        return {"execute_code": dataclasses.replace(tool, fn=copied)}
    return {"execute_code": copied}


# ---------------------------------------------------------------------------
# The prompt
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class PromptSurface:
    """What a core-surface session's prompt says it has, fixed at session start."""

    functions: bool = True
    guidance: bool = True
    clarification: bool = False
    steering: bool = False
    policy: WritePolicy = WritePolicy()
    #: A response format is set: the answer is a ``final_response`` call.
    structured: bool = False
    #: The turn that compresses the context also offers ``store_skills``.
    store_skills_on_compression: bool = False
    #: A persistent session's trajectory is reviewed after each turn.
    turn_reviews: bool = False
    #: ``UNIFY_CORE_CALL_EXAMPLE``: the ``functions`` line shows a call.
    call_example: bool = False

    def tools_section(self) -> str:
        answer = (
            "When the request is addressed, answer by calling "
            "`final_response` with the answer in the required format."
            if self.structured
            else "When the request is addressed, answer with a reply that "
            "calls no tool."
        )
        loop = (
            " `wait`, `steer` and `ask_about_completed_tool` manage calls "
            "while they run."
            if self.steering
            else ""
        )
        return "### Tools\n\n" + _fill(
            "`execute_code` runs cells of Python (or bash, with "
            '`language="bash"`) in a persistent sandbox. Everything else '
            "the harness provides is a Python object in that sandbox, called "
            f"from code and awaited, and listed below.{loop} {answer}",
        )

    def index(self) -> str:
        """The 5-10 line index of the sandbox's objects."""
        lines = ["### Sandbox Objects", ""]
        if self.functions:
            writes = self.policy.writes(FUNCTIONS)
            text = (
                "- `functions`: stored functions. `search(query)`, "
                "`filter(where)`, `list()`, `get(name)`, "
                '`run(name, state="stateless", **kwargs)`'
            )
            text += (
                "; writes " + ", ".join(f"`{w}(...)`" for w in writes)
                if writes
                else "; read-only in this session"
            )
            text += ". A function found by a read is callable by name in later cells."
            if self.call_example:
                # UNIFY_CORE_CALL_EXAMPLE
                text += (
                    ' Example: `total = await functions.run("sum_invoice_lines", '
                    "invoice_id=7)`, or once found, `sum_invoice_lines(invoice_id=7)`; "
                    "a stored function that does a step saves rewriting it."
                )
            lines.append(text)
        if self.guidance:
            writes = self.policy.writes(GUIDANCE)
            text = (
                "- `guidance`: stored procedures. `search(text)`, `filter(where)`, "
                "`get(guidance_id)`"
            )
            text += (
                "; writes " + ", ".join(f"`{w}(...)`" for w in writes)
                if writes
                else "; read-only in this session"
            )
            lines.append(text + ".")
        lines.append(
            "- `install(packages)`: installs packages into the persistent "
            "workspace environment; never pip from a cell.",
        )
        lines.append(
            "- `read_file(path, start=1, end=None)` and `grep(pattern, "
            'path=".")`: numbered lines of a file, and matching lines.',
        )
        if self.clarification:
            lines.append(
                "- `request_clarification(question)`: asks the requester and "
                "returns the answer.",
            )
        lines.append(
            "- `help(obj)` prints the full documentation of any of these, or "
            "of one method.",
        )
        return "\n".join(_fill(line) for line in lines)

    def library_section(self, *, inline_curation: str = "") -> str:
        lines = [
            "### Function & Guidance Library",
            "",
            "The libraries hold functions (the *what*) and procedures (the "
            "*how*) stored from earlier work. Search them when the task may "
            "match stored work, and use what fits: call a relevant function, "
            "follow relevant guidance. Prefer healthy matches (empty "
            "`stale_reasons`). Results show long guidance as a preview: read "
            "an entry you will follow with `guidance.get(id)`.",
        ]
        if not (self.policy.writes(FUNCTIONS) or self.policy.writes(GUIDANCE)):
            return "\n".join(_fill(line) for line in lines)
        lines += ["", "#### Writing to the libraries", ""]
        if self.guidance and "add" in self.policy.writes(GUIDANCE):
            lines.append(
                "- **Guidance**: a procedure the user asks to remember goes to "
                "`guidance.add(...)`. Guidance is the canonical home of a "
                "shared rule stored functions apply: when the rule changes, "
                "update that entry first (its `function_ids` are the affected "
                "functions), then each linked function; never add a second copy.",
            )
        if self.functions and "add" in self.policy.writes(FUNCTIONS):
            patch = (
                "`functions.patch(...)`"
                if "patch" in self.policy.writes(FUNCTIONS)
                else "`functions.add(..., overwrite=True)`"
            )
            if inline_curation:
                lines.append(
                    "- **Functions, during the task**: once a reusable unit ran "
                    "and worked, store it with `functions.add(source)`; when a "
                    f"stored function fails, fix it with {patch}, keeping its "
                    "behaviour on the inputs it already handled (a change of "
                    "behaviour is a new function with a new name). Store only "
                    "code that ran, under a name that says what it does "
                    "(snake_case, a verb and its object). A write that fails a "
                    "check is refused, saying why.",
                )
            else:
                lines.append(
                    "- **Functions**: a user's request to add, update or delete "
                    f"a function uses `functions.add(...)` (`overwrite=True` to "
                    f"update), {patch} or `functions.delete(...)`.",
                )
        return "\n".join(_fill(line) for line in lines)

    def storage_notice(self, *, persist: bool, inline_curation: str) -> str:
        if inline_curation == "only":
            text = (
                "Nothing reviews this trajectory for the libraries after the "
                "task: what is worth keeping, you store during it."
            )
        elif persist:
            text = (
                "A review reads this session's trajectory "
                + ("after each completed turn and " if self.turn_reviews else "")
                + "when the session ends, and stores reusable functions and "
                "guidance from it; store directly only what the user asks "
                "you to store."
            )
        else:
            text = (
                "After you answer, a review reads the trajectory and stores "
                "reusable functions and guidance from it; store directly only "
                "what the user asks you to store."
            )
        out = "### Skill Storage\n\n" + _fill(text)
        if self.store_skills_on_compression:
            out += "\n\n" + _fill(
                "When the context window nears capacity, `compress_context` "
                "and `store_skills` become the only tools. Call `store_skills` "
                "first (with a specific request) if the trajectory holds "
                "unstored skills worth preserving; otherwise go straight to "
                "`compress_context`.",
            )
        return out

    def read_only_notice(self) -> str:
        return "### Library Writes\n\n" + _fill(
            "In this session the function and guidance libraries are "
            "read-only: their writes raise `PermissionError`. When the session "
            "ends, a review may add to the libraries, but only if an external "
            "check of the session's outcome admits it.",
        )

    def python_first(self) -> str:
        return textwrap.dedent("""
            ### Python First

            Prefer Python packages over shell CLI tools: `await install([...])`
            installs into the one persistent workspace environment, where they
            stay for every later task. When a task genuinely needs a CLI, run
            it in a bash cell or from Python with `subprocess`, and work with
            its output as data.
        """).strip()

    def tool_selection(self, shipped: str) -> str:
        """Handles and the steering checkpoint, from the shipped Tool
        Selection section, where the steering tools exist; nothing otherwise
        (the rest of that section is about choosing between JSON tools)."""
        marker = "### Responding to a steering checkpoint"
        if not self.steering or marker not in shipped:
            return ""
        handles = "### Handles\n\n" + "\n".join(
            _fill(line)
            for line in (
                "- **Handle adoption:** a steerable handle a cell returns as "
                "its **last expression** is adopted by the outer loop for "
                "steering (ask, stop, pause, resume) -- never consume a handle "
                "inside a cell (print it, await-and-discard it) when the loop "
                "needs steering.",
                "- **Handle lifetime:** an adopted handle is steerable while "
                "its work runs, and its completion is the outcome to report "
                "-- never pause a handle or relaunch finished work to keep it "
                "open for corrections that have not arrived.",
            )
        )
        return handles + "\n\n" + marker + shipped.split(marker, 1)[1]


def _fill(text: str) -> str:
    if text.startswith("#") or not text:
        return text
    indent = "  " if text.startswith("- ") else ""
    return textwrap.fill(
        text,
        width=76,
        subsequent_indent=indent,
        break_long_words=False,
        break_on_hyphens=False,
    )


# Rule 1's pointer to the session tools, and rule 4 (notifications), which
# name JSON tools this surface does not have.
_RULE_SESSIONS = (
    "   `list_sessions()` / `inspect_state()` rediscover live sessions\n"
    "   and names — variables survive context compression, since state\n"
    "   lives in the sandbox, not the transcript."
)
_RULE_SESSIONS_CORE = (
    "   Variables survive context compression, since state lives in the\n"
    "   sandbox, not the transcript."
)
_RULE_NOTIFICATIONS = re.compile(
    r"\n4\. \*\*Notifications\*\*:.*?(?=\n5\. )",
    re.DOTALL,
)


def execution_rules(text: str) -> str:
    """The shipped Execution Rules without the tools this surface does not have."""
    text = text.replace(_RULE_SESSIONS, _RULE_SESSIONS_CORE, 1)
    if _RULE_NOTIFICATIONS.search(text):
        text = _RULE_NOTIFICATIONS.sub("", text, count=1)
        head, sep, tail = text.partition("\n5. **")
        tail = re.sub(
            r"\n(\d+)\. \*\*",
            lambda m: f"\n{int(m.group(1)) - 1}. **",
            sep + tail,
        )
        text = head + tail
    return text


# ---------------------------------------------------------------------------
# One act() call
# ---------------------------------------------------------------------------

_KEEP = object()

#: How ``execute_code`` binds the sandbox's ``request_clarification`` in the
#: running act(): ``_KEEP`` (not a core session: as shipped), ``None`` (the
#: session cannot ask: no such name), or a factory of the bound function.
_CLARIFICATION: "contextvars.ContextVar[Any]" = contextvars.ContextVar(
    "unify_core_clarification",
    default=_KEEP,
)


def _clarification_factory(
    caller_queues: Optional[tuple],
    on_request: Optional[Callable[[str], Any]],
    on_answer: Optional[Callable[[str], Any]],
) -> Callable[..., Callable[..., Any]]:
    """Builds, per ``execute_code`` call, the sandbox's ``request_clarification``:
    the JSON tool of the same name, as the loop would build it."""
    from unify.common.llm_helpers import make_request_clarification_tool

    up, down = caller_queues if caller_queues else (None, None)
    tool = make_request_clarification_tool(
        up,
        down,
        on_request=on_request,
        on_answer=on_answer,
    )

    def bind(call_up: Any, call_down: Any) -> Callable[..., Any]:
        async def request_clarification(question: str) -> str:
            return await tool(
                question,
                _clarification_up_q=call_up,
                _clarification_down_q=call_down,
            )

        request_clarification.__doc__ = public_doc(tool.__doc__)
        return request_clarification

    return bind


def bind_clarification(
    namespace: Dict[str, Any],
    up_q: Any,
    down_q: Any,
) -> Optional[Callable[[], None]]:
    """In a core session, set the cell's ``request_clarification`` for one
    ``execute_code`` call; returns what restores the namespace, or ``None``
    outside a core session (nothing changed)."""
    factory = _CLARIFICATION.get()
    if factory is _KEEP:
        return None
    missing = object()
    previous = namespace.get("request_clarification", missing)
    if factory is None:
        namespace.pop("request_clarification", None)
    else:
        namespace["request_clarification"] = factory(up_q, down_q)

    def restore() -> None:
        if previous is missing:
            namespace.pop("request_clarification", None)
        else:
            namespace["request_clarification"] = previous

    return restore


@dataclasses.dataclass
class Session:
    """What one core-surface act() call runs with."""

    tools: Dict[str, Any]
    prompt: PromptSurface
    steering: bool
    objects: Dict[str, Any]
    clarification: Any

    def enter(self) -> Any:
        """Set this session's clarification binding; returns the reset token."""
        return _CLARIFICATION.set(self.clarification)

    def listed_binder(
        self,
        sandbox: Any,
    ) -> Optional[Callable[[List[str]], Dict[str, bool]]]:
        """``UNIFY_CORE_BIND_LISTED``: what binds the shortlist's functions in *sandbox*.

        ``None`` with the switch off, or without a function library: the
        shortlist is then written as shipped.
        """
        from unify.settings import SETTINGS

        library = self.objects.get(FUNCTIONS)
        if not SETTINGS.UNIFY_CORE_BIND_LISTED or library is None:
            return None
        return lambda names: library._bind_names(names, sandbox=sandbox)

    @staticmethod
    def leave(token: Any) -> None:
        try:
            _CLARIFICATION.reset(token)
        except ValueError:  # entered in another context
            pass


def start_session(
    actor: Any,
    *,
    sandbox: Any,
    environments: Mapping[str, Any],
    tools: Mapping[str, Any],
    policy: WritePolicy,
    store_skills: bool,
    clarification_enabled: bool,
    caller_queues: Optional[tuple],
    on_clarification_request: Optional[Callable[[str], Any]],
    on_clarification_answer: Optional[Callable[[str], Any]],
    structured: bool,
    turn_reviews: bool,
) -> Session:
    """The tools, sandbox objects and prompt of one core-surface act() call.

    ``tools`` are the session's JSON tools as shipped (filtered for the
    session); the result keeps ``execute_code`` (and ``store_skills``, which
    the loop offers only on the turn that compresses, when ``store_skills``)
    and puts the rest in *sandbox* as Python objects.
    """
    from unify.settings import SETTINGS

    steering = offers_steering(environments)
    session_tools = core_tools(tools, steering=steering)
    if store_skills and "store_skills" in tools:
        session_tools["store_skills"] = tools["store_skills"]
    objects = sandbox_objects(actor, policy=policy)
    sandbox.global_state.update(objects)
    sandbox.core_globals = dict(objects)
    clarification = (
        _clarification_factory(
            caller_queues,
            on_clarification_request,
            on_clarification_answer,
        )
        if clarification_enabled
        else None
    )
    prompt = PromptSurface(
        functions=FUNCTIONS in objects,
        guidance=GUIDANCE in objects,
        clarification=clarification_enabled,
        steering=steering,
        policy=policy,
        structured=structured,
        store_skills_on_compression="store_skills" in session_tools,
        turn_reviews=turn_reviews,
        call_example=bool(getattr(SETTINGS, "UNIFY_CORE_CALL_EXAMPLE", False)),
    )
    return Session(
        tools=session_tools,
        prompt=prompt,
        steering=steering,
        objects=objects,
        clarification=clarification,
    )
