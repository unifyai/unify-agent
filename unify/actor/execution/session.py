"""Execution sessions and the unified SessionExecutor.

Provides PythonExecutionSession (in-process stateful sandbox),
SessionExecutor (the orchestrator over the in-process sessions), and the
validation of execution parameters.
"""

from __future__ import annotations

import asyncio
import ast
import contextvars
import logging
import sys
import time
import traceback
import types
import uuid
from datetime import datetime, timezone
from typing import (
    Any,
    Callable,
    Dict,
    Literal,
    NoReturn,
    Optional,
    TYPE_CHECKING,
)

from unify.common._async_tool import time_context
from unify.common.hierarchical_logger import DEFAULT_ICON
from unify.common.tool_errors import ToolInputError

from .capture import _stdout_parts, capture_sandbox_output
from unify.function_manager.steering import (
    DEFAULT_TOOL_NAMESPACES,
    ExecutionStopped,
    MemoisedDispatch,
    active_session,
    bind_session,
    instrument,
    restore_session,
    run_with_steering,
)
from .types import TextPart

if TYPE_CHECKING:
    from unify.actor.environments.base import BaseEnvironment

    from .shell import BashSession
    from .worker import PythonWorker

# Handles spawned by manager primitives during an in-process sandbox
# ``execute`` call.  When the LLM fire-and-forgets a steerable handle
# (awaits the call that *returns* the handle but never ``await
# handle.result()`` and does not return the handle as the last
# expression), the outer loop never adopts it and the mutation is
# cancelled when ``execute_code`` returns.  Draining this bucket at the
# end of ``execute`` awaits those orphans so side effects land.
_SANDBOX_SPAWNED_HANDLES: contextvars.ContextVar[list[Any] | None] = (
    contextvars.ContextVar("_SANDBOX_SPAWNED_HANDLES", default=None)
)


def register_sandbox_spawned_handle(handle: Any) -> None:
    """Record a steerable handle created inside an active sandbox execute."""
    bucket = _SANDBOX_SPAWNED_HANDLES.get()
    if bucket is not None:
        bucket.append(handle)


def _collect_handles(obj: Any) -> list[Any]:
    """Collect ``SteerableToolHandle`` instances reachable from *obj*."""
    from unify.common.async_tool_loop import SteerableToolHandle

    found: list[Any] = []

    def _walk(node: Any) -> None:
        if isinstance(node, SteerableToolHandle):
            found.append(node)
            return
        if isinstance(node, dict):
            for v in node.values():
                _walk(v)
            return
        if isinstance(node, (list, tuple, set)):
            for v in node:
                _walk(v)
            return
        try:
            from pydantic import BaseModel

            if isinstance(node, BaseModel):
                for field_name in node.model_fields:
                    _walk(getattr(node, field_name))
        except Exception:
            return

    _walk(obj)
    return found


async def _await_orphan_sandbox_handles(
    *,
    spawned: list[Any],
    result: Any,
) -> None:
    """Await steerable handles spawned in-sandbox that were not returned.

    Membership is by object identity: spawned handles need not be hashable
    (logging wrappers and test doubles often are not), and ``in`` on a set
    would raise ``TypeError`` for those.
    """
    if not spawned:
        return
    returned_ids = {id(h) for h in _collect_handles(result)}
    for handle in spawned:
        if id(handle) in returned_ids:
            continue
        try:
            done = handle.done()
            if asyncio.iscoroutine(done) or asyncio.isfuture(done):
                done = await done
            if done:
                continue
        except Exception:
            pass
        await handle.result()


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------
StateMode = Literal["stateful", "read_only", "stateless"]
SessionKey = int  # session_id


# ---------------------------------------------------------------------------
# ContextVars
# ---------------------------------------------------------------------------
_CURRENT_SANDBOX: contextvars.ContextVar["PythonExecutionSession"] = (
    contextvars.ContextVar(
        "code_act_current_sandbox",
    )
)

_PARENT_CHAT_CONTEXT: contextvars.ContextVar[list | None] = contextvars.ContextVar(
    "code_act_parent_chat_context",
    default=None,
)

_CURRENT_ENVIRONMENTS: contextvars.ContextVar[dict] = contextvars.ContextVar(
    "code_act_current_environments",
    default={},
)

# Whether the actor running the current sandbox can ask its caller
# (``clarification_enabled``); ``None`` outside any actor. A sub-actor it
# starts inherits it under UNIFY_PROMPT_ACCURACY.
_CAN_CLARIFY: contextvars.ContextVar[bool | None] = contextvars.ContextVar(
    "code_act_can_clarify",
    default=None,
)


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------
def _refuse(
    *,
    message: str,
    suggestion: str,
    state_mode: str,
    session_id: int | None,
    session_name: str | None,
) -> NoReturn:
    """Refuse the call, naming the change that would make it work.

    Raising rather than returning is what lets the tool loop see the refusal.
    A returned description is indistinguishable from a successful call, so the
    same rejected request could arrive forever without anything noticing.
    """
    raise ToolInputError(
        message,
        suggestion=suggestion,
        received={
            "state_mode": state_mode,
            "session_id": session_id,
            "session_name": session_name,
        },
    )


def _validate_execution_params(
    *,
    state_mode: str,
    session_id: int | None,
    session_name: str | None,
    # Name resolution/lookup is actor-owned, so validation accepts callables.
    resolve_session_name: Optional[Callable[[str], Optional[SessionKey]]] = None,
    get_session_name_for_id: Optional[Callable[[int], Optional[str]]] = None,
    session_exists: Optional[Callable[[int], bool]] = None,
    max_sessions_total: Optional[int] = None,
    active_session_count: Optional[int] = None,
) -> None:
    """
    Validate state_mode + session selection rules for execute_code.

    Raises:
        ToolInputError: if the parameters cannot be executed as given, carrying
            the change that would make them valid.

    Notes:
        Refusals are raised rather than returned so the tool loop can see them.
        The caller still reads the message verbatim — the loop surfaces a
        refusal as its own text, not as a traceback — so self-correction is
        unaffected, while a request that keeps arriving unchanged is now
        something the loop can notice and stop.
    """
    if state_mode not in ("stateful", "read_only", "stateless"):
        _refuse(
            message=f"Unsupported state_mode: {state_mode!r}",
            suggestion="Use one of: 'stateful', 'read_only', 'stateless'",
            state_mode=state_mode,
            session_id=session_id,
            session_name=session_name,
        )

    # Stateless must not reference sessions.
    if state_mode == "stateless" and (
        session_id is not None or session_name is not None
    ):
        _refuse(
            message="Cannot use state_mode='stateless' with a session.",
            suggestion="Remove session_id/session_name or switch to state_mode='stateful' or 'read_only'.",
            state_mode=state_mode,
            session_id=session_id,
            session_name=session_name,
        )

    # Read-only requires an existing session.
    if state_mode == "read_only" and (session_id is None and session_name is None):
        _refuse(
            message="Cannot use state_mode='read_only' without specifying a session.",
            suggestion="Provide session_id or session_name (must refer to an existing session), or use state_mode='stateless'.",
            state_mode=state_mode,
            session_id=session_id,
            session_name=session_name,
        )

    # If both are present, ensure they match.
    if session_id is not None and session_name is not None:
        if resolve_session_name is None:
            _refuse(
                message="Cannot validate session_name against session_id (no resolver configured).",
                suggestion="Specify only one of session_id or session_name.",
                state_mode=state_mode,
                session_id=session_id,
                session_name=session_name,
            )
        key = resolve_session_name(session_name)
        if key is None:
            # For stateful, the caller may choose to create+bind; for read_only it must exist.
            if state_mode == "read_only":
                _refuse(
                    message=f"Session name {session_name!r} not found for read_only execution.",
                    suggestion="Use an existing session_name (see list_sessions) or specify an existing session_id.",
                    state_mode=state_mode,
                    session_id=session_id,
                    session_name=session_name,
                )
        else:
            if key != session_id:
                _refuse(
                    message=(
                        f"session_id and session_name refer to different sessions. "
                        f"{session_name!r} resolves to {key} "
                        f"but received {session_id}."
                    ),
                    suggestion="Specify only one of session_id or session_name, or make them consistent.",
                    state_mode=state_mode,
                    session_id=session_id,
                    session_name=session_name,
                )

    # If session_name is provided alone:
    if session_name is not None and session_id is None:
        if resolve_session_name is None:
            _refuse(
                message="Cannot resolve session_name (no resolver configured).",
                suggestion="Specify session_id instead, or configure a session registry.",
                state_mode=state_mode,
                session_id=session_id,
                session_name=session_name,
            )
        key = resolve_session_name(session_name)
        if key is None and state_mode == "read_only":
            _refuse(
                message=f"Session name {session_name!r} not found for read_only execution.",
                suggestion="Use an existing session_name (see list_sessions) or specify an existing session_id.",
                state_mode=state_mode,
                session_id=session_id,
                session_name=session_name,
            )

    # Optional: enforce a global session cap (actor-owned pools, so this is per-actor).
    # Note: `stateful` with no session specified means "use the default session" (session_id=0),
    # so it does NOT create a new session and should not be rejected by the global cap here.
    #
    # We only apply the cap when the call would CREATE a new stateful session:
    # - session_name is provided but does not resolve (new session would be allocated)
    if (
        max_sessions_total is not None
        and active_session_count is not None
        and state_mode == "stateful"
        and session_id is None
        and session_name is not None
        and resolve_session_name is not None
        and resolve_session_name(session_name) is None
        and active_session_count >= max_sessions_total
    ):
        _refuse(
            message=f"Session limit exceeded: {active_session_count} active sessions (max {max_sessions_total}).",
            suggestion="Close an existing session (close_session) or reuse an existing session_id/session_name.",
            state_mode=state_mode,
            session_id=session_id,
            session_name=session_name,
        )

    # If an explicit session_id is provided for stateful execution, enforce limits when it would
    # CREATE a new session (important for in-process Python sessions where pool-level limits may not apply).
    if (
        max_sessions_total is not None
        and active_session_count is not None
        and state_mode == "stateful"
        and session_id is not None
        and session_exists is not None
        and active_session_count >= max_sessions_total
    ):
        try:
            exists = bool(session_exists(session_id))
        except Exception:
            exists = False
        if not exists:
            _refuse(
                message=f"Session limit exceeded: {active_session_count} active sessions (max {max_sessions_total}).",
                suggestion="Close an existing session (close_session) or reuse an existing session_id/session_name.",
                state_mode=state_mode,
                session_id=session_id,
                session_name=session_name,
            )

    # If session_id is provided and we can validate existence for read_only, do so.
    if (
        state_mode == "read_only"
        and session_id is not None
        and session_exists is not None
    ):
        if not session_exists(session_id):
            name_hint = None
            if get_session_name_for_id is not None:
                name_hint = get_session_name_for_id(session_id)
            hint = f" (known name: {name_hint!r})" if name_hint else ""
            _refuse(
                message=f"Session {session_id} does not exist for read_only execution{hint}.",
                suggestion="Use list_sessions to find an existing session, or switch to state_mode='stateful' to create a new session.",
                state_mode=state_mode,
                session_id=session_id,
                session_name=session_name,
            )

    return None


# ---------------------------------------------------------------------------
# PythonExecutionSession
# ---------------------------------------------------------------------------
class PythonExecutionSession:
    """
    A stateful execution environment for running Python code asynchronously.

    This class maintains a persistent global state across multiple executions,
    capturing stdout, stderr, return values, and exceptions in a structured format.
    """

    def __init__(
        self,
        environments: Optional[Dict[str, "BaseEnvironment"]] = None,
    ):
        """
        Initializes the execution environment.

        Args:
            environments: Optional mapping of environment namespaces to environments. If
                provided, each environment instance is injected into globals.
        """
        from unify.function_manager.execution_env import create_execution_globals

        self.id: str = str(uuid.uuid4())
        self._module_name: str = f"__sandbox_{self.id}__"

        # Cells sharing this session mutate one module dict, one ``display``
        # hook, and one ``primitives`` binding, so they must run one at a
        # time; ``execute`` serializes on this lock.
        self._execution_lock = asyncio.Lock()

        # Register the sandbox globals as a proper module in sys.modules.
        _mod = types.ModuleType(self._module_name)
        _initial = create_execution_globals()
        _mod.__dict__.update(_initial)
        sys.modules[self._module_name] = _mod
        self.global_state: Dict[str, Any] = _mod.__dict__
        self.global_state["__name__"] = self._module_name

        # Expose sandbox metadata to user code (best-effort; callers may ignore).
        self.global_state["__sandbox_id__"] = self.id

        # UNIFY_WORKSPACE_PYTHON=worker: cells run in this child process and
        # ``global_state`` holds only what the harness provides them.
        self._worker: Optional["PythonWorker"] = None

        # UNIFY_TOOL_SURFACE=core: the harness objects of the act() this
        # sandbox belongs to (``functions``, ``guidance``, ``install``, ...),
        # which its other sessions get too (SessionExecutor._inject_fm_globals).
        self.core_globals: Dict[str, Any] = {}

        if environments:
            for namespace, env in environments.items():
                try:
                    # Use get_sandbox_instance() if available (for filtered primitives),
                    # otherwise fall back to get_instance()
                    if hasattr(env, "get_sandbox_instance"):
                        instance = env.get_sandbox_instance()
                    else:
                        instance = env.get_instance()
                    self.global_state[namespace] = instance
                except Exception:
                    # Keep sandbox usable even if a non-critical environment fails to inject.
                    continue

    async def close(self) -> None:
        """
        Best-effort cleanup for an ephemeral sandbox instance.

        Notes
        -----
        - This method is safe to call multiple times.
        """
        try:
            if self._worker is not None:
                worker, self._worker = self._worker, None
                await worker.close()
            sys.modules.pop(self._module_name, None)
            self.global_state.clear()
        except Exception as e:
            try:
                logger.warning(
                    f"{DEFAULT_ICON} PythonExecutionSession.close() failed: {e}",
                    exc_info=True,
                )
            except Exception:
                pass

    def _python_worker(self) -> Optional["PythonWorker"]:
        """This session's sandboxed worker, when ``UNIFY_WORKSPACE_PYTHON=worker``."""
        from unify import sandbox

        from . import worker as worker_mod

        if not worker_mod.enabled():
            return None
        sandbox.require_bwrap()
        if self._worker is None:
            self._worker = worker_mod.PythonWorker()
        return self._worker

    async def worker_variables(self) -> Optional[Dict[str, str]]:
        """The worker's variables (name -> short repr); None when cells run in
        process. Waits for a running cell, like another cell would."""
        from . import worker as worker_mod

        if not worker_mod.enabled():
            return None
        if self._worker is None:
            return {}
        async with self._execution_lock:
            return await self._worker.variables()

    async def execute(
        self,
        code: str,
        *,
        timeout: float | None = None,
        scratch: bool = False,
    ) -> dict:
        """
        Executes a string of Python code within the sandbox's stateful environment.

        Concurrent calls are serialized: cells share this session's module
        dict, ``display`` hook, and ``primitives`` binding, so exactly one
        cell runs at a time and queued cells run in arrival order.

        Returns a dict with:
            stdout: list[OutputPart] - structured output parts (TextPart, ImagePart)
            stderr: list[OutputPart] - structured error output parts
            result: Any - return value of the last expression
            error: str | None - traceback if an exception occurred

        ``scratch`` (worker sessions only) runs the cell against a shallow copy
        of the worker's namespace and discards it, which is what
        ``state_mode="read_only"`` means there.
        """
        queued_at = time.perf_counter()
        async with self._execution_lock:
            lock_wait_ms = (time.perf_counter() - queued_at) * 1000
            if lock_wait_ms >= 1:
                logger.debug(
                    "sandbox %s: cell waited %.0fms for the session execution lock",
                    self.id,
                    lock_wait_ms,
                )
            return await self._execute_exclusively(
                code,
                timeout=timeout,
                scratch=scratch,
            )

    async def _execute_exclusively(
        self,
        code: str,
        *,
        timeout: float | None,
        scratch: bool = False,
    ) -> dict:
        """Run one cell in the sandbox; the caller holds ``_execution_lock``."""
        from .worker import KILLED_NOTE, WorkerCellError

        result = None
        error = None
        worker = self._python_worker()

        with capture_sandbox_output() as (stdout_parts, stderr_parts, display_fn):
            # Inject display function into globals
            self.global_state["display"] = display_fn

            try:
                # Guardrails: prevent agent code from accidentally shadowing critical
                # injected environment globals (common failure mode in LLM-generated code).
                #
                # We do this via an AST rewrite (not brittle string heuristics):
                # rewrite any assignment targets named `primitives`
                # to `_primitives_local`.
                #
                # This preserves the injected globals for the rest of the session.
                try:

                    class _ShadowingGuard(ast.NodeTransformer):
                        _REMAP = {
                            "primitives": "_primitives_local",
                        }

                        def visit_Name(self, node: ast.Name) -> ast.AST:  # noqa: N802
                            # Only rewrite *assignments* (Store context). Loads are preserved.
                            if (
                                isinstance(node.ctx, ast.Store)
                                and node.id in self._REMAP
                            ):
                                return ast.copy_location(
                                    ast.Name(id=self._REMAP[node.id], ctx=node.ctx),
                                    node,
                                )
                            return node

                        def visit_Global(
                            self,
                            node: ast.Global,
                        ) -> ast.AST:  # noqa: N802
                            node.names = [self._REMAP.get(n, n) for n in node.names]
                            return node

                        def visit_Nonlocal(
                            self,
                            node: ast.Nonlocal,
                        ) -> ast.AST:  # noqa: N802
                            node.names = [self._REMAP.get(n, n) for n in node.names]
                            return node

                    tree = ast.parse(code)
                    tree = _ShadowingGuard().visit(tree)  # type: ignore[assignment]
                    ast.fix_missing_locations(tree)
                    code = ast.unparse(tree)
                except Exception:
                    # Best-effort only; if rewriting fails, proceed with original code.
                    pass

                is_empty_or_comment_only = all(
                    line.strip() == "" or line.strip().startswith("#")
                    for line in code.splitlines()
                )
                if is_empty_or_comment_only:
                    code += "\npass"

                tree = ast.parse(code)
                top_level_assign_targets = set()
                for node in tree.body:
                    if isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
                        targets = []
                        if isinstance(node, ast.Assign):
                            targets.extend(node.targets)
                        else:
                            targets.append(node.target)

                        for target in targets:
                            if isinstance(target, ast.Name):
                                top_level_assign_targets.add(target.id)
                            elif isinstance(target, ast.Tuple):
                                for elt in target.elts:
                                    if isinstance(elt, ast.Name):
                                        top_level_assign_targets.add(elt.id)

                    elif isinstance(node, (ast.Import, ast.ImportFrom)):
                        for alias in node.names:
                            top_level_assign_targets.add(
                                alias.asname or alias.name.split(".")[0],
                            )

                    elif isinstance(
                        node,
                        (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef),
                    ):
                        top_level_assign_targets.add(node.name)

                # REPL semantics: implicitly return the last expression's value
                # so callers can read it from the `result` field.
                if tree.body and isinstance(tree.body[-1], ast.Expr):
                    tree.body[-1] = ast.Return(value=tree.body[-1].value)
                    ast.fix_missing_locations(tree)
                    code = ast.unparse(tree)

                # Steering only exists while a call is in flight. The session
                # arrives by context rather than on this object, because the
                # sandbox that runs a block is often not the one that was
                # current when the tool started — stateless mode builds a
                # fresh one per call. Binding it here means every
                # in-process Python path picks it up, whichever sandbox that
                # turns out to be.
                steering = active_session()
                # Probes only. Memoising `primitives` is done further down,
                # where it can be composed in the right order with the
                # context-forwarding proxy.
                steering_token = (
                    bind_session(self.global_state, steering, tool_namespaces=[])
                    if steering is not None
                    else None
                )

                def _wrap_for_execution(body: str) -> str:
                    """Indent a block into the async wrapper it runs inside.

                    Rebuilt per attempt, because a correction rewrites the body
                    and the wrapper has to be regenerated around the new text.
                    """
                    header = "async def __exec_wrapper():\n"
                    if top_level_assign_targets:
                        header += (
                            "    global "
                            f"{', '.join(sorted(top_level_assign_targets))}\n"
                        )
                    return header + "".join(f"    {ln}\n" for ln in body.splitlines())

                async_code = _wrap_for_execution(code)

                # Inject a custom print function that writes directly to our capture
                # list via ContextVar, bypassing sys.stdout entirely. This is
                # necessary because pytest's live logging feature can replace
                # sys.stdout during LOGGER.info() calls, breaking our StreamRouter.
                _gs_builtins = self.global_state.get("__builtins__", {})
                if isinstance(_gs_builtins, dict):
                    _original_print = _gs_builtins.get("print")

                    def _sandbox_print(
                        *args,
                        sep=" ",
                        end="\n",
                        file=None,
                        flush=False,
                    ):
                        # If file is explicitly specified, use the original print
                        if file is not None:
                            if _original_print:
                                return _original_print(
                                    *args,
                                    sep=sep,
                                    end=end,
                                    file=file,
                                    flush=flush,
                                )
                            return
                        # Otherwise, write directly to our capture list via ContextVar
                        try:
                            parts = _stdout_parts.get()
                        except LookupError:
                            # No capture context - fall back to original print
                            if _original_print:
                                return _original_print(
                                    *args,
                                    sep=sep,
                                    end=end,
                                    flush=flush,
                                )
                            return
                        # Format the output like standard print
                        output = sep.join(str(arg) for arg in args) + end
                        # Merge consecutive text writes into a single TextPart
                        if parts and isinstance(parts[-1], TextPart):
                            last = parts[-1]
                            parts[-1] = TextPart(text=last.text + output)
                        else:
                            parts.append(TextPart(text=output))

                    self.global_state["__builtins__"]["print"] = _sandbox_print

                # Wrap primitives with ContextForwardingProxy if parent
                # chat context is available, so inner tool loops receive it.
                _pcc = _PARENT_CHAT_CONTEXT.get()
                _orig_prims = self.global_state.get("primitives")
                _wrapped_prims = _orig_prims
                if _pcc is not None and _wrapped_prims is not None:
                    from unify.function_manager.primitives.context_proxy import (
                        ContextForwardingProxy,
                    )

                    _wrapped_prims = ContextForwardingProxy(
                        _wrapped_prims,
                        _parent_chat_context=_pcc,
                    )
                # Outermost, so memoisation sees the call before context
                # forwarding does. This is what covers dispatches the statement
                # probes cannot reach, such as a primitive inside a
                # comprehension or gathered concurrently.
                if steering is not None and _wrapped_prims is not None:
                    _wrapped_prims = MemoisedDispatch(_wrapped_prims, steering)
                if _wrapped_prims is not _orig_prims:
                    self.global_state["primitives"] = _wrapped_prims

                spawned_handles: list[Any] = []
                spawned_token = _SANDBOX_SPAWNED_HANDLES.set(spawned_handles)

                async def _exec_wrapped(wrapped: str) -> Any:
                    """Define ``__exec_wrapper`` from *wrapped* and await it."""
                    if worker is not None:
                        return await worker.run_cell(
                            wrapped,
                            self.global_state,
                            timeout=timeout,
                            scratch=scratch,
                            stdout=stdout_parts,
                            stderr=stderr_parts,
                            display=display_fn,
                        )
                    exec(wrapped, self.global_state)
                    execution = self.global_state["__exec_wrapper"]()
                    if timeout is None:
                        return await execution
                    return await asyncio.wait_for(execution, timeout=timeout)

                async def _run_once(body: str) -> Any:
                    """Compile and run one attempt at this block."""
                    source = body
                    if steering is not None:
                        source = ast.unparse(
                            instrument(
                                ast.parse(body),
                                tool_namespaces=set(DEFAULT_TOOL_NAMESPACES),
                            ),
                        )
                    return await _exec_wrapped(_wrap_for_execution(source))

                try:
                    if steering is None:
                        result = await _exec_wrapped(async_code)
                    else:
                        # A correction rewrites `code` and re-runs; already
                        # completed dispatches replay from the session cache
                        # rather than happening twice.
                        steering.bind_source(code)
                        result = await run_with_steering(
                            code,
                            _run_once,
                            session=steering,
                        )
                    await _await_orphan_sandbox_handles(
                        spawned=spawned_handles,
                        result=result,
                    )
                finally:
                    _SANDBOX_SPAWNED_HANDLES.reset(spawned_token)
                    if _orig_prims is not None:
                        self.global_state["primitives"] = _orig_prims
                    if steering_token is not None:
                        restore_session(self.global_state, steering_token)

            except ExecutionStopped as stopped:
                result = stopped.outcome
            except asyncio.TimeoutError:
                error = f"Python execution timed out after {timeout}s"
                if worker is not None:
                    error += KILLED_NOTE
            except WorkerCellError as failed:
                error = failed.traceback
            except Exception:
                error = traceback.format_exc()
            finally:
                if "__exec_wrapper" in self.global_state:
                    del self.global_state["__exec_wrapper"]

        return {
            "stdout": stdout_parts,
            "stderr": stderr_parts,
            "result": result,
            "error": error,
        }


# ---------------------------------------------------------------------------
# SessionExecutor
# ---------------------------------------------------------------------------
class SessionExecutor:
    """
    Unified execution engine for multi-session CodeAct execution.

    Notes
    -----
    - Sessions are backed by persistent PythonExecutionSession instances,
      keyed by session id.
    """

    def __init__(
        self,
        *,
        environments: Optional[Dict[str, "BaseEnvironment"]] = None,
        timeout: Optional[float] = None,
    ) -> None:
        self._environments = environments or {}
        self._timeout = timeout

        self._python_sessions: Dict[int, PythonExecutionSession] = {}
        self._python_session_meta: Dict[int, dict[str, str]] = {}
        # Persistent bash sessions (UNIFY_WORKSPACE=sandboxed), keyed like the
        # Python ones.
        self._shell_sessions: Dict[int, "BashSession"] = {}

        self._fm_globals: Dict[str, Any] = {}

    def register_fm_globals(self, globals_dict: Dict[str, Any]) -> None:
        self._fm_globals.update(globals_dict)

    def _inject_fm_globals(self, sb: PythonExecutionSession) -> None:
        if self._fm_globals:
            sb.global_state.update(self._fm_globals)
        # UNIFY_TOOL_SURFACE=core: a stateless or named session holds the
        # running act()'s sandbox objects as its own session 0 does.
        bound = _CURRENT_SANDBOX.get(None)
        if bound is not None and bound is not sb and bound.core_globals:
            sb.global_state.update(bound.core_globals)

    def _new_session(self) -> PythonExecutionSession:
        return PythonExecutionSession(environments=self._environments)

    @staticmethod
    def _bound_sandbox(session_id: int | None) -> PythonExecutionSession | None:
        """The running ``act()`` call's sandbox, if ``session_id`` names it.

        Session 0 names that sandbox in every state mode whenever one is
        bound. With none bound, session 0 is an ordinary executor-managed
        session.
        """
        return _CURRENT_SANDBOX.get(None) if session_id == 0 else None

    def python_session(self, *, session_id: int) -> PythonExecutionSession | None:
        """The session ``session_id`` names, bound or executor-managed, if any."""
        bound = self._bound_sandbox(session_id)
        return bound if bound is not None else self._python_sessions.get(session_id)

    def has_python_session(self, *, session_id: int) -> bool:
        return self.python_session(session_id=session_id) is not None

    def list_in_process_python_sessions(self) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for session_id, sb in list(self._python_sessions.items()):
            meta = self._python_session_meta.get(session_id) or {}
            out.append(
                {
                    "session_id": int(session_id),
                    "created_at": meta.get("created_at"),
                    "last_used": meta.get("last_used"),
                    "state_summary": "active",
                },
            )
        return out

    async def close_in_process_python_session(self, *, session_id: int) -> bool:
        key = int(session_id)
        sb = self._python_sessions.pop(key, None)
        self._python_session_meta.pop(key, None)
        shell = self._shell_sessions.pop(key, None)
        if shell is not None:
            await shell.close()
        if sb is None:
            return shell is not None
        try:
            await sb.close()
        except Exception:
            pass
        return True

    async def close(self) -> None:
        for sb in list(self._python_sessions.values()):
            try:
                await sb.close()
            except Exception:
                pass
        self._python_sessions.clear()
        self._python_session_meta.clear()
        for shell in list(self._shell_sessions.values()):
            await shell.close()
        self._shell_sessions.clear()

    async def _execute_shell(
        self,
        *,
        code: str,
        state_mode: StateMode,
        session_id: int | None,
    ) -> Dict[str, Any]:
        """Run a bash cell inside the workspace sandbox."""
        from unify import sandbox
        from .shell import DEFAULT_SHELL_TIMEOUT_S, BashSession

        if not sandbox.enabled():
            raise ToolInputError(
                "language='bash' needs UNIFY_WORKSPACE=sandboxed.",
                suggestion="Run shell commands from Python with subprocess.",
                received={"language": "bash"},
            )
        if state_mode == "read_only":
            _refuse(
                message="state_mode='read_only' applies to Python sessions only.",
                suggestion="Use state_mode='stateful' or 'stateless' for bash.",
                state_mode=state_mode,
                session_id=session_id,
                session_name=None,
            )
        sandbox.require_bwrap()
        policy = sandbox.build_policy()
        timeout = self._timeout or DEFAULT_SHELL_TIMEOUT_S
        started = time_context.perf_counter()
        created = False
        if state_mode == "stateless":
            shell = BashSession(policy=policy)
            try:
                res = await shell.execute(code, timeout=timeout)
            finally:
                await shell.close()
            session_id = None
        else:
            key = int(session_id or 0)
            shell = self._shell_sessions.get(key)
            if shell is None:
                shell = self._shell_sessions[key] = BashSession(policy=policy)
                created = True
            res = await shell.execute(code, timeout=timeout)
            session_id = key
        output, error = res["output"], res["error"]
        note = sandbox.annotate_refusals(output, policy)
        if note and error:
            error = f"{error}\n{note}"
        elif note:
            output = f"{output}\n[{note}]"
        return {
            "stdout": [TextPart(text=output)] if output else [],
            "stderr": [],
            "result": res["exit_code"],
            "error": error,
            "state_mode": state_mode,
            "session_id": session_id,
            "session_created": created,
            "duration_ms": int((time_context.perf_counter() - started) * 1000),
        }

    async def execute(
        self,
        *,
        code: str,
        state_mode: StateMode,
        session_id: int | None,
        language: str = "python",
        prepare: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> Dict[str, Any]:
        """Run ``code`` in the session ``state_mode`` and ``session_id`` name.

        ``prepare`` (``UNIFY_FUNCTION_HELPERS``), when given, is called with
        the namespace the cell runs in before it runs, to define names there:
        they stay where the state mode keeps what a cell defines.
        """
        if language == "bash":
            return await self._execute_shell(
                code=code,
                state_mode=state_mode,
                session_id=session_id,
            )
        import time as _se_time
        import logging as _se_logging

        _se_t0 = _se_time.perf_counter()
        _se_log = _se_logging.getLogger("unify")

        def _se_ms():
            return f"{(_se_time.perf_counter() - _se_t0) * 1000:.0f}ms"

        _se_log.debug(
            f"⏱️ [SessionExecutor.execute +{_se_ms()}] entered "
            f"(state_mode={state_mode}, session_id={session_id})",
        )

        # The loop's monotonic clock, the one tool-call timings read.
        t0 = time_context.perf_counter()

        def _duration_ms() -> int:
            return int((time_context.perf_counter() - t0) * 1000)

        async def _execute_in_python_session(
            sb: PythonExecutionSession,
            scratch: bool = False,
        ) -> Dict[str, Any]:
            from unify import sandbox

            if not sandbox.enabled():
                return await sb.execute(code, timeout=self._timeout)
            # In process, the cell runs here and what it starts through
            # subprocess, os.system or asyncio runs in the sandbox; in a worker
            # the cell is confined whole and this confines what the harness
            # objects it calls start.
            with sandbox.confined_subprocesses(sandbox.build_policy()):
                if scratch:
                    return await sb.execute(
                        code,
                        timeout=self._timeout,
                        scratch=True,
                    )
                return await sb.execute(code, timeout=self._timeout)

        bound = self._bound_sandbox(session_id)
        if state_mode == "stateful" and bound is not None:
            self._inject_fm_globals(bound)
            if prepare is not None:
                prepare(bound.global_state)
            _se_log.debug(
                f"⏱️ [SessionExecutor.execute +{_se_ms()}] bound sandbox (session 0), executing",
            )
            res = await _execute_in_python_session(bound)
            _se_log.debug(
                f"⏱️ [SessionExecutor.execute +{_se_ms()}] bound sandbox done",
            )
            return {
                **res,
                "state_mode": state_mode,
                "session_id": 0,
                "session_created": False,
                "duration_ms": _duration_ms(),
            }
        # Stateless: fresh in-process sandbox per call.
        if state_mode == "stateless":
            _se_log.debug(
                f"⏱️ [SessionExecutor.execute +{_se_ms()}] creating stateless sandbox",
            )
            sb = self._new_session()
            _se_log.debug(
                f"⏱️ [SessionExecutor.execute +{_se_ms()}] sandbox created, injecting globals",
            )
            self._inject_fm_globals(sb)
            if prepare is not None:
                prepare(sb.global_state)
            _se_log.debug(
                f"⏱️ [SessionExecutor.execute +{_se_ms()}] globals injected, executing code",
            )
            try:
                res = await _execute_in_python_session(sb)
                _se_log.debug(
                    f"⏱️ [SessionExecutor.execute +{_se_ms()}] code execution done",
                )
            finally:
                try:
                    await sb.close()
                except Exception:
                    pass
            return {
                **res,
                "state_mode": state_mode,
                "session_id": None,
                "session_created": False,
                "duration_ms": _duration_ms(),
            }

        # Persistent sessions.
        if session_id is None:
            raise ValueError(
                "session_id is required for in-process stateful/read_only execution",
            )

        key = int(session_id)
        if state_mode == "stateful":
            created = False
            if key not in self._python_sessions:
                self._python_sessions[key] = self._new_session()
                created = True
                now = datetime.now(timezone.utc).isoformat()
                self._python_session_meta[key] = {
                    "created_at": now,
                    "last_used": now,
                }
            sb = self._python_sessions[key]
            self._inject_fm_globals(sb)
            if prepare is not None:
                prepare(sb.global_state)
            res = await _execute_in_python_session(sb)
            meta = self._python_session_meta.get(key)
            if meta is not None:
                meta["last_used"] = datetime.now(timezone.utc).isoformat()
            return {
                **res,
                "state_mode": state_mode,
                "session_id": session_id,
                "session_created": created,
                "duration_ms": _duration_ms(),
            }

        if state_mode == "read_only":
            # Create a throwaway sandbox seeded with current state.
            base = self.python_session(session_id=key)
            if base is None:
                raise ValueError(
                    f"Session {key} not found for read_only execution",
                )
            from .worker import enabled as _worker_enabled

            if _worker_enabled():
                # The session's variables live in its worker: the cell runs
                # there against a copy of its namespace.
                self._inject_fm_globals(base)
                if prepare is None:
                    res = await _execute_in_python_session(base, scratch=True)
                else:
                    # What prepare defines is the cell's, not the session's:
                    # the session's own bindings come back afterwards.
                    before = dict(base.global_state)
                    prepare(base.global_state)
                    defined = {
                        name: value
                        for name, value in base.global_state.items()
                        if name not in before or before[name] is not value
                    }
                    try:
                        res = await _execute_in_python_session(base, scratch=True)
                    finally:
                        for name, value in defined.items():
                            if base.global_state.get(name) is not value:
                                continue
                            if name in before:
                                base.global_state[name] = before[name]
                            else:
                                del base.global_state[name]
                return {
                    **res,
                    "state_mode": state_mode,
                    "session_id": session_id,
                    "session_created": False,
                    "duration_ms": _duration_ms(),
                }
            sb = self._new_session()
            try:
                # Shallow copy globals to allow read access while avoiding persistence.
                sb.global_state.update(dict(base.global_state))
                self._inject_fm_globals(sb)
                if prepare is not None:
                    prepare(sb.global_state)
                res = await _execute_in_python_session(sb)
            finally:
                try:
                    await sb.close()
                except Exception:
                    pass
            return {
                **res,
                "state_mode": state_mode,
                "session_id": session_id,
                "session_created": False,
                "duration_ms": _duration_ms(),
            }

        raise ValueError(
            f"Unsupported state_mode for in-process execution: {state_mode}",
        )
