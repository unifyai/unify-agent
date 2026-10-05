"""Python cells in a sandboxed child process (``UNIFY_WORKSPACE_PYTHON=worker``).

With ``UNIFY_WORKSPACE=sandboxed`` and ``UNIFY_WORKSPACE_PYTHON=worker`` each
Python session owns one :class:`PythonWorker`: a persistent ``python -I -S``
started inside the workspace sandbox (unify/sandbox.py) -- scrubbed
environment, store read-only, state directory hidden, network per
``UNIFY_WORKSPACE_NETWORK``. Cells run there, so a cell's variables persist
from one cell to the next exactly as in the in-process session, but the cell
cannot read the harness's environment, write the store or open a socket the
sandbox does not allow.

What the cell needs from the harness it reaches through a proxy served here.
Every name the session's namespace holds beyond the base globals is installed
in the worker as one of:

* a **remote** object (``primitives``, environment namespaces, the globals a
  registered environment binds (``apis``), ``request_clarification``,
  ``query_llm``, steering probes): attribute access
  asks the harness what the attribute is, and a call is sent over the channel,
  run here against the session's *current* binding of that name -- so the
  memoising and context-forwarding wrappers a cell installs apply -- and its
  result sent back;
* a **stored function**: its source is defined in the worker, so the
  model-written body runs confined and only its primitive calls come back;
* a **value** (plain data), sent once;
* a **worker-local** stand-in (``display``, ``run_coro_sync``,
  ``_around_cp``, ``SteerableToolHandle``);
* or **refused**, naming why, when none of these applies.

Values cross as tagged JSON (worker_child.py explains why not pickle). A result
that is not data comes back as a reference the worker can call and pass back;
a worker object that is not data is refused as an argument, naming it, and
comes back as a cell result only as its repr. Exceptions raised here are
re-raised in the worker under their own type (builtins) or a class of the same
name and module.

A cell that times out or is stopped kills the worker (its whole sandbox); the
next cell starts a fresh one and its output says so.
"""

from __future__ import annotations

import asyncio
import functools
import inspect
import itertools
import json
import linecache
import logging
import os
import sys
import types
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from unify import sandbox

from . import worker_child as child
from .worker_child import TAG, BoundaryRefusal, decode, encode, short_repr

logger = logging.getLogger(__name__)

__all__ = [
    "BoundaryRefusal",
    "PythonWorker",
    "WorkerCellError",
    "WorkerValue",
    "enabled",
]

#: Largest single message either side may send.
MAX_MESSAGE_BYTES = 64 * 1024 * 1024
START_TIMEOUT_S = 60.0

#: Appended to a timed-out cell's error.
KILLED_NOTE = (
    ". The sandboxed Python worker was killed; the next cell starts a fresh one "
    "(variables, imports and definitions are reset)."
)

_BOOTSTRAP = (
    "import runpy, sys; runpy.run_path(sys.argv[1], run_name='__unify_worker__')"
)

#: Session globals that never cross: interpreter plumbing and the harness's
#: own per-call state (clarification queues, the steering session object).
_SKIP = frozenset(
    {
        "__builtins__",
        "__name__",
        "__doc__",
        "__loader__",
        "__spec__",
        "__package__",
        "__exec_wrapper",
    },
)
_DUNDER_SENT = frozenset({"__sandbox_id__"})


def enabled() -> bool:
    """``UNIFY_WORKSPACE=sandboxed`` and ``UNIFY_WORKSPACE_PYTHON=worker``."""
    from unify.settings import SETTINGS

    return (
        sandbox.enabled()
        and getattr(SETTINGS, "UNIFY_WORKSPACE_PYTHON", "") == "worker"
    )


class WorkerCellError(Exception):
    """The cell failed in the worker; ``traceback`` is the worker's text."""

    def __init__(self, traceback_text: str) -> None:
        super().__init__(traceback_text)
        self.traceback = traceback_text


class WorkerDied(Exception):
    pass


class WorkerValue:
    """A cell result that exists only in the worker, seen as its repr."""

    def __init__(self, type_name: str, repr_text: str) -> None:
        self.type_name = type_name
        self.repr_text = repr_text

    def __repr__(self) -> str:
        return self.repr_text

    __str__ = __repr__


def _importable(name: str, value: Any) -> Optional[list]:
    """How the worker can build a base global itself, if it can."""
    import importlib

    if isinstance(value, types.ModuleType):
        root = value.__name__.split(".")[0]
        return None if root in ("unify", "unillm") else ["module", value.__name__]
    module = getattr(value, "__module__", None)
    if not isinstance(module, str) or module.split(".")[0] in ("unify", "unillm"):
        return None
    try:
        if getattr(importlib.import_module(module), name, None) is value:
            return ["attr", module, name]
    except Exception:  # noqa: BLE001
        return None
    return None


def _stored_function_code(value: Any) -> Optional[types.CodeType]:
    """The code of the stored function *value* wraps, if it wraps one.

    Stored functions are compiled under ``<function:NAME>``
    (unify/function_manager/source_labels.py); the function manager's
    lineage wrapper keeps the raw function as ``__wrapped__``.
    """
    from unify.function_manager.source_labels import function_source_filename

    prefix = function_source_filename("")[:-1]
    fn = value
    for _ in range(8):
        if isinstance(fn, types.FunctionType):
            code = fn.__code__
            return code if code.co_filename.startswith(prefix) else None
        fn = getattr(fn, "__wrapped__", None)
        if fn is None:
            return None
    return None


def _refuse_unreachable(obj: Any, label: str) -> None:
    """Modules and the environment never cross, whatever holds them.

    Without this, any exposed object with a module among its public attributes
    (``unillm.os``) would hand the worker ``os.environ.get``.
    """
    if isinstance(obj, types.ModuleType):
        raise BoundaryRefusal(
            f"{label} is a harness module ({obj.__name__}); harness modules do not "
            "cross the worker boundary (import the module in the cell instead)",
        )
    if obj is os.environ or obj is getattr(os, "environb", None):
        raise BoundaryRefusal(
            f"{label} is the harness's environment (rule `env-scrub`: "
            f"{sandbox.RULES['env-scrub']}); the cell has its own os.environ",
        )


def _model_written(value: Any) -> bool:
    """Whether *value* was defined by cell code run in a session's namespace."""
    module = getattr(value, "__module__", None)
    if not isinstance(module, str):
        module = getattr(type(value), "__module__", None)
    return isinstance(module, str) and module.startswith("__sandbox_")


class PythonWorker:
    """One persistent sandboxed Python process and the proxy that serves it."""

    def __init__(self) -> None:
        self._proc: Optional[asyncio.subprocess.Process] = None
        self._send_lock = asyncio.Lock()
        self._stderr_tail = bytearray()
        self._stderr_task: Optional[asyncio.Task] = None
        self._ids = itertools.count(1)
        self._refs: Dict[int, Any] = {}
        self._ref_ids: Dict[int, int] = {}
        self._synced: Dict[str, tuple] = {}
        self._base: Dict[str, Any] = {}
        self._restart_note: Optional[str] = None
        self.pid: Optional[int] = None

    # -- lifecycle -------------------------------------------------------------
    @property
    def is_running(self) -> bool:
        return self._proc is not None and self._proc.returncode is None

    def _init_message(self) -> dict:
        from unify.function_manager.execution_env import create_base_globals

        from unify import environment

        base = create_base_globals()
        builtins_names = sorted(base.pop("__builtins__", {}).keys())
        specs: Dict[str, list] = {}
        self._base = {}
        for name, value in base.items():
            spec = _importable(name, value)
            if spec is not None:
                specs[name] = spec
                self._base[name] = value
        paths = [p for p in sys.path if p]
        # Packages the harness installs later land here; visible once it exists.
        packages = str(environment.site_packages())
        if packages not in paths:
            paths.append(packages)
        msg = {
            "op": "init",
            "sys_path": paths,
            "builtins": builtins_names,
            "globals": specs,
        }
        if _core_surface():
            msg["help"] = True
        return msg

    async def _start(self) -> None:
        from unify import environment

        # The workspace venv is mounted (read-only) only if it exists when the
        # sandbox starts, and the harness creates it on the first install --
        # a stored function's dependency, while this worker runs. Created
        # empty first, it is mounted now and the installer fills it in place.
        venv = environment.environment_dir()
        created = not venv.exists()
        venv.mkdir(parents=True, exist_ok=True)
        # The policy as it is now: a restarted worker sees the current one.
        policy = sandbox.build_policy(fresh=created)
        argv = [sys.executable, "-I", "-S", "-c", _BOOTSTRAP, str(Path(child.__file__))]
        env = sandbox.sandbox_env(policy)
        workspace = str(policy.workspace)
        wrapped = sandbox.wrap_argv(argv, policy, cwd=workspace)
        with sandbox.unconfined():  # already wrapped; never wrap twice
            self._proc = await asyncio.create_subprocess_exec(
                *wrapped,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=env,
                cwd=workspace,
                start_new_session=True,
                limit=MAX_MESSAGE_BYTES,
            )
        self._stderr_tail.clear()
        self._stderr_task = asyncio.create_task(self._drain_stderr(self._proc))
        self._synced = {}
        self._refs.clear()
        self._ref_ids.clear()
        try:
            await self._send(self._init_message())
            ready = await asyncio.wait_for(self._read(), timeout=START_TIMEOUT_S)
        except (asyncio.TimeoutError, WorkerDied) as exc:
            tail = self._stderr_tail.decode("utf-8", "replace").strip()
            await self.close()
            raise RuntimeError(
                f"The sandboxed Python worker did not start ({type(exc).__name__})"
                + (f": {tail[-2000:]}" if tail else ""),
            ) from None
        if ready.get("op") != "ready":
            await self.close()
            raise RuntimeError(f"The sandboxed Python worker sent {ready!r} first")
        self.pid = self._proc.pid
        if ready.get("missing"):
            logger.warning(
                "sandboxed Python worker could not build base globals: %s",
                ready["missing"],
            )

    async def _drain_stderr(self, proc: asyncio.subprocess.Process) -> None:
        assert proc.stderr is not None
        while True:
            chunk = await proc.stderr.read(4096)
            if not chunk:
                return
            self._stderr_tail.extend(chunk)
            del self._stderr_tail[:-8192]

    async def close(self) -> None:
        """Kill the worker and its sandbox and wait until it is gone."""
        from .shell import _kill_group

        proc, self._proc = self._proc, None
        self._refs.clear()
        self._ref_ids.clear()
        self._synced = {}
        if proc is None:
            return
        if proc.returncode is None:
            _kill_group(proc)
        try:
            await asyncio.wait_for(proc.wait(), timeout=10)
        except asyncio.TimeoutError:
            logger.warning("sandboxed Python worker %s did not exit", proc.pid)
        if self._stderr_task is not None:
            self._stderr_task.cancel()
            self._stderr_task = None

    async def _kill(self, why: str) -> None:
        await self.close()
        self._restart_note = why

    # -- channel ---------------------------------------------------------------
    async def _send(self, msg: dict) -> None:
        proc = self._proc
        if proc is None or proc.stdin is None:
            raise WorkerDied("the worker is not running")
        data = (json.dumps(msg, separators=(",", ":")) + "\n").encode()
        if len(data) > MAX_MESSAGE_BYTES:
            raise BoundaryRefusal(
                f"a message of {len(data)} bytes is larger than the worker "
                f"channel allows ({MAX_MESSAGE_BYTES})",
            )
        async with self._send_lock:
            try:
                proc.stdin.write(data)
                await proc.stdin.drain()
            except (BrokenPipeError, ConnectionResetError) as exc:
                raise WorkerDied(str(exc)) from None

    async def _read(self) -> dict:
        proc = self._proc
        if proc is None or proc.stdout is None:
            raise WorkerDied("the worker is not running")
        try:
            line = await proc.stdout.readline()
        except (ValueError, asyncio.LimitOverrunError):
            raise WorkerDied("the worker sent a message larger than allowed")
        if not line:
            raise WorkerDied("the worker closed its channel")
        try:
            msg = json.loads(line)
        except ValueError:
            raise WorkerDied("the worker sent something that is not JSON")
        if not isinstance(msg, dict):
            raise WorkerDied("the worker sent something that is not a message")
        return msg

    # -- codec -----------------------------------------------------------------
    def _ref(self, obj: Any) -> dict:
        from unify.common.async_tool_loop import SteerableToolHandle

        _refuse_unreachable(obj, f"a {child.type_name(obj)} returned by the harness")
        rid = self._ref_ids.get(id(obj))
        if rid is None or self._refs.get(rid) is not obj:
            rid = next(self._ids)
            self._refs[rid] = obj
            self._ref_ids[id(obj)] = rid
        return {
            TAG: "ref",
            "id": rid,
            "type": child.type_name(obj),
            "repr": short_repr(obj),
            "callable": callable(obj),
            "async": _async_callable(obj),
            "handle": isinstance(obj, SteerableToolHandle),
        }

    def _pre(self, value: Any) -> Any:
        try:
            from pydantic import BaseModel
        except ImportError:  # pragma: no cover - pydantic is a dependency
            return None
        if isinstance(value, BaseModel):
            try:
                data = value.model_dump(mode="python")
            except Exception:  # noqa: BLE001 - send it by reference instead
                return None
            return {
                TAG: "model",
                "type": type(value).__name__,
                "data": encode(data, pre=self._pre, unknown=self._unknown),
                "ref": self._ref(value)["id"],
            }
        return None

    def _unknown(self, value: Any, where: str) -> Any:
        return self._ref(value)

    def encode(self, value: Any) -> Any:
        return encode(value, pre=self._pre, unknown=self._unknown)

    def decode(self, value: Any, shadow: Dict[str, Any]) -> Any:
        def tagged(tag: str, payload: dict) -> Any:
            if tag == "ref":
                rid = int(payload.get("id", -1))
                if rid not in self._refs:
                    raise BoundaryRefusal(
                        f"harness reference {rid} is unknown or belongs to a worker "
                        "that has been restarted",
                    )
                return self._refs[rid]
            if tag == "target":
                return self._resolve(shadow, payload.get("target") or {})
            if tag == "opaque":
                return WorkerValue(str(payload.get("type")), str(payload.get("repr")))
            raise BoundaryRefusal(f"the worker sent an unknown value form {tag!r}")

        return decode(value, tagged=tagged)

    # -- namespace sync --------------------------------------------------------
    def _describe_global(self, name: str, value: Any, record: bool = False) -> dict:
        from unify.common.async_tool_loop import SteerableToolHandle
        from unify.common.asyncio_compat import run_coro_sync
        from unify.function_manager.steering import AROUND_CP_FN

        if name == "display" and callable(value):
            return {"kind": "local", "local": "display"}
        if name == AROUND_CP_FN:
            return {"kind": "local", "local": "around_cp"}
        if value is run_coro_sync:
            return {"kind": "local", "local": "run_coro_sync"}
        if value is SteerableToolHandle:
            return {"kind": "local", "local": "handle_class"}
        if isinstance(value, types.ModuleType):
            spec = _importable(name, value)
            if spec is not None:
                return {"kind": "import", "spec": spec}
        if _environment_global(name, value):
            # A global the registered environment binds (AppWorld's ``apis``)
            # holds the harness's connection to the environment: a copy
            # imported in the worker would dial it from inside the sandbox,
            # which does not reach it, so it is served from here.
            return self._remote(name, value)
        code = _stored_function_code(value)
        if code is not None:
            entry = linecache.cache.get(code.co_filename)
            if entry is None:
                return {
                    "kind": "refused",
                    "reason": "the stored function's source is not available to "
                    "define it in the worker",
                }
            desc = {
                "kind": "function",
                "source": "".join(entry[2]),
                "filename": code.co_filename,
            }
            if record:
                # UNIFY_TOOL_SURFACE=core: its calls are recorded here.
                desc["record"] = True
            return desc
        if _model_written(value):
            # Defined by model code in this namespace; it runs only in the
            # worker, which defines its own copy when it runs that code.
            return {
                "kind": "model",
                "reason": "it was defined by code that ran in the harness, and "
                "model-written code runs only in the worker",
            }
        spec = _importable(name, value)
        if spec is not None:
            return {"kind": "import", "spec": spec}
        if not callable(value):
            try:
                data = encode(value, unknown=_no_refs)
            except BoundaryRefusal:
                data = None
            else:
                return {"kind": "value", "value": data}
        return self._remote(name, value)

    @staticmethod
    def _remote(name: str, value: Any) -> dict:
        from unify.function_manager.steering import RUNTIME_GLOBAL, SteeringRuntime

        desc = {
            "kind": "remote",
            "callable": callable(value),
            "async": _async_callable(value),
            "repr": short_repr(value, 200),
            "runtime": name == RUNTIME_GLOBAL and isinstance(value, SteeringRuntime),
        }
        if _function_library(value) is not None:
            # UNIFY_TOOL_SURFACE=core: ``functions.run`` runs in the worker.
            desc["library"] = "functions"
        return desc

    def _manifest(self, shadow: Dict[str, Any]) -> dict:
        want: Dict[str, dict] = {}
        record = _function_library(shadow.get("functions")) is not None
        for name, value in list(shadow.items()):
            if not isinstance(name, str) or name in _SKIP:
                continue
            if name.startswith("__") and name not in _DUNDER_SENT:
                continue
            if name in self._base and self._base[name] is value:
                continue
            want[name] = self._describe_global(name, value, record=record)
        changed: Dict[str, dict] = {}
        for name, desc in want.items():
            # A root rebinds every cell (steering wraps ``primitives`` anew);
            # the worker resolves it by name, so only its kind matters.
            token = tuple(
                sorted((k, json.dumps(v)) for k, v in desc.items() if k != "repr"),
            )
            if self._synced.get(name) != token:
                changed[name] = desc
                self._synced[name] = token
        removed = [n for n in self._synced if n not in want]
        for name in removed:
            del self._synced[name]
        return {"set": changed, "remove": removed}

    def _exposed_remote(self, name: str) -> bool:
        token = self._synced.get(name)
        return token is not None and ("kind", json.dumps("remote")) in token

    def _resolve(self, shadow: Dict[str, Any], target: dict) -> Any:
        if "root" in target:
            name = str(target["root"])
            if not self._exposed_remote(name) or name not in shadow:
                raise BoundaryRefusal(
                    f"{name!r} is not a harness object exposed to the worker",
                )
            obj = shadow[name]
            label = name
        else:
            rid = int(target.get("ref", -1))
            if rid not in self._refs:
                raise BoundaryRefusal(
                    f"harness reference {rid} is unknown or belongs to a worker "
                    "that has been restarted",
                )
            obj = self._refs[rid]
            label = f"<{child.type_name(obj)}>"
        for attr in target.get("path") or []:
            if not isinstance(attr, str) or attr.startswith("_"):
                raise BoundaryRefusal(
                    f"{label}.{attr}: attributes starting with '_' of harness "
                    "objects do not cross the worker boundary",
                )
            obj = getattr(obj, attr)
            label = f"{label}.{attr}"
            _refuse_unreachable(obj, label)
        return obj

    def _describe(self, obj: Any, name: str) -> dict:
        from unify.function_manager.steering import _PASSTHROUGH_TYPES

        if callable(obj):
            return {"kind": "callable", "async": _async_callable(obj)}
        if isinstance(obj, _PASSTHROUGH_TYPES):
            if sandbox.is_secret_name(name):
                raise BoundaryRefusal(
                    f"{name!r} looks like a credential (rule `env-scrub`: "
                    f"{sandbox.RULES['env-scrub']}); its value stays in the harness",
                )
            return {"kind": "value", "value": self.encode(obj)}
        return {"kind": "namespace", "repr": short_repr(obj, 200)}

    # -- serving ---------------------------------------------------------------
    def _error(self, exc: BaseException) -> dict:
        try:
            args = encode(list(exc.args), unknown=_no_refs)
        except BoundaryRefusal:
            args = None
        return {
            "type": type(exc).__name__,
            "module": type(exc).__module__,
            "message": str(exc),
            "args": args,
        }

    async def _serve(self, msg: dict, shadow: Dict[str, Any]) -> None:
        op = msg.get("op")
        reply: Dict[str, Any] = {"op": "reply", "id": msg.get("id")}
        try:
            with self._recording(msg.get("cases"), shadow):
                await self._answer(op, msg, shadow, reply)
        except asyncio.CancelledError:
            reply["error"] = {
                "type": "CancelledError",
                "module": "asyncio.exceptions",
                "message": "the cell ended before this harness call finished",
                "args": None,
            }
            try:
                await self._send(reply)
            except Exception:  # noqa: BLE001
                pass
            raise
        except Exception as exc:  # noqa: BLE001 - every failure goes back
            reply.pop("value", None)
            reply["error"] = self._error(exc)
        try:
            await self._send(reply)
        except WorkerDied:
            pass

    @staticmethod
    def _recording(cases: Any, shadow: Dict[str, Any]) -> Any:
        """UNIFY_TOOL_SURFACE=core: the environment calls served inside add
        to the cases of the stored-function calls the request came from."""
        import contextlib

        stack = contextlib.ExitStack()
        library = _function_library(shadow.get("functions"))
        if not cases or library is None or not isinstance(cases, list):
            return stack
        from unify.function_manager import store_cases

        for token in cases:
            stack.enter_context(store_cases.tracing(library._case_pending(token)))
        return stack

    def _library(self, shadow: Dict[str, Any]) -> Any:
        library = _function_library(shadow.get("functions"))
        if library is None or not self._exposed_remote("functions"):
            raise BoundaryRefusal(
                "stored functions are recorded only where the session's "
                "`functions` library is exposed (UNIFY_TOOL_SURFACE=core)",
            )
        return library

    async def _answer(
        self,
        op: Any,
        msg: dict,
        shadow: Dict[str, Any],
        reply: Dict[str, Any],
    ) -> None:
        if op == "fn_begin":
            reply["value"] = self.encode(
                await self._library(shadow)._begin(
                    name=str(msg.get("name")),
                    mode=str(msg.get("mode")),
                    args=list(self.decode(msg.get("args") or [], shadow)),
                    kwargs=dict(self.decode(msg.get("kwargs") or {}, shadow)),
                ),
            )
            return
        if op == "fn_end":
            reply["value"] = self.encode(
                await self._library(shadow)._end(
                    token=msg.get("token"),
                    result=self.decode(msg.get("result"), shadow),
                    error=msg.get("error"),
                    abandoned=bool(msg.get("abandoned")),
                ),
            )
            return
        if op == "doc" and _core_surface():
            from unify.actor import core_surface

            target = msg.get("target")
            if not target:
                reply["value"] = core_surface.index_help(shadow)
                return
            reply["value"] = core_surface.help_text(
                self._resolve(shadow, target),
                str(msg.get("label") or "object"),
            )
            return
        target = msg.get("target") or {}
        obj = self._resolve(shadow, target)
        if op == "describe":
            path = target.get("path") or [target.get("root", "")]
            reply["value"] = self._describe(obj, str(path[-1]))
        elif op == "dir":
            reply["value"] = sorted(n for n in dir(obj) if not n.startswith("_"))
        elif op == "call":
            args = self.decode(msg.get("args") or [], shadow)
            kwargs = self.decode(msg.get("kwargs") or {}, shadow)
            out = obj(*args, **kwargs)
            awaited = inspect.isawaitable(out)
            if awaited:
                out = await out
            reply["value"] = self.encode(out)
            reply["coroutine"] = awaited
        else:
            raise BoundaryRefusal(f"unknown request {op!r}")

    def _apply_note(self, msg: dict, shadow: Dict[str, Any]) -> None:
        from unify.function_manager.steering import SteeringRuntime

        event = msg.get("event")
        try:
            if event == "runtime":
                runtime = shadow.get("runtime")
                method = str(msg.get("method"))
                if (
                    isinstance(runtime, SteeringRuntime)
                    and method in child.RUNTIME_NOTES
                ):
                    getattr(runtime, method)(*list(msg.get("args") or [])[:1])
            elif event == "function_used":
                fn = shadow.get(str(msg.get("name")))
                on_call = getattr(fn, "_on_call", None) if fn is not None else None
                if callable(on_call):
                    on_call()
        except Exception:  # noqa: BLE001 - notes are best effort by design
            logger.debug("worker note %r failed", event, exc_info=True)

    async def _serve_until_done(self, cid: int, shadow: Dict[str, Any]) -> dict:
        tasks: set[asyncio.Task] = set()
        try:
            while True:
                msg = await self._read()
                op = msg.get("op")
                if op in _SERVED:
                    task = asyncio.create_task(self._serve(msg, shadow))
                    tasks.add(task)
                    task.add_done_callback(tasks.discard)
                elif op == "note":
                    self._apply_note(msg, shadow)
                elif op == "done" and msg.get("id") == cid:
                    return msg
        finally:
            for task in list(tasks):
                task.cancel()
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)

    # -- cells -----------------------------------------------------------------
    async def run_cell(
        self,
        source: str,
        shadow: Dict[str, Any],
        *,
        timeout: Optional[float],
        scratch: bool = False,
        stdout: List[Any],
        stderr: List[Any],
        display: Optional[Callable[[Any], None]] = None,
    ) -> Any:
        """Run one wrapped cell (``async def __exec_wrapper(): ...``) in the worker.

        Output parts are appended to *stdout* / *stderr*. Returns the cell's
        result; raises :class:`WorkerCellError` with the worker's traceback,
        ``ControlledInterruption`` when a steering probe interrupted the cell,
        or ``asyncio.TimeoutError`` after killing the worker.
        """
        from unify.function_manager.steering import ControlledInterruption

        from .types import TextPart

        if not self.is_running:
            note, self._restart_note = self._restart_note, None
            await self._start()
            if note:
                _append_text(
                    stdout,
                    f"[A fresh Python worker started: the previous one was killed "
                    f"({note}); variables, imports and definitions from earlier "
                    "cells are gone.]\n",
                    TextPart,
                )
        cid = next(self._ids)
        try:
            await self._send(
                {
                    "op": "exec",
                    "id": cid,
                    "source": source,
                    "sync": self._manifest(shadow),
                    "scratch": scratch,
                },
            )
            if timeout is None:
                done = await self._serve_until_done(cid, shadow)
            else:
                done = await asyncio.wait_for(
                    self._serve_until_done(cid, shadow),
                    timeout=timeout,
                )
        except asyncio.TimeoutError:
            await self._kill(f"the cell timed out after {timeout}s")
            raise
        except asyncio.CancelledError:
            await self._kill("the cell was stopped")
            raise
        except WorkerDied as exc:
            code = self._proc.returncode if self._proc is not None else None
            await self._kill(f"it exited: {exc}")
            raise WorkerCellError(
                f"The sandboxed Python worker exited during the cell ({exc}; "
                f"status {code}). The next cell starts a fresh worker: "
                "variables, imports and definitions are reset.",
            ) from None

        self._collect_parts(done.get("stdout") or [], stdout, display, TextPart)
        self._collect_parts(done.get("stderr") or [], stderr, None, TextPart)
        if done.get("error"):
            if (
                done.get("remote")
                and done.get("error_type") == "ControlledInterruption"
            ):
                raise ControlledInterruption(str(done.get("message") or "steered"))
            raise WorkerCellError(str(done["error"]))
        return self.decode(done.get("result"), shadow)

    @staticmethod
    def _collect_parts(
        parts: list,
        into: List[Any],
        display: Optional[Callable[[Any], None]],
        text_part: type,
    ) -> None:
        from .types import ImagePart

        for part in parts:
            if not isinstance(part, dict):
                continue
            if part.get("type") == "text":
                _append_text(into, str(part.get("text") or ""), text_part)
            elif part.get("type") == "image":
                data = str(part.get("data") or "")
                image = _decode_png(data) if display is not None else None
                if image is not None:
                    display(image)  # scaled for the model like an in-process image
                else:
                    into.append(ImagePart(mime="image/png", data=data))

    async def variables(self) -> Dict[str, str]:
        """The worker's own variables (name -> short repr), for ``inspect_state``."""
        if not self.is_running:
            return {}
        cid = next(self._ids)
        await self._send({"op": "variables", "id": cid})
        done = await asyncio.wait_for(self._serve_until_done(cid, {}), timeout=30)
        return dict(done.get("variables") or {})


#: What the worker asks of the harness, each served in a task of its own.
_SERVED = frozenset({"call", "describe", "dir", "fn_begin", "fn_end", "doc"})


_PLAIN_TYPES = (
    types.FunctionType,
    types.MethodType,
    types.BuiltinFunctionType,
    types.BuiltinMethodType,
    functools.partial,
)


def _plain(obj: Any) -> bool:
    """Whether reading an attribute of *obj* only looks it up: no ``__getattr__``
    and no ``__getattribute__`` of its class's own that could run code."""
    if isinstance(obj, _PLAIN_TYPES):
        return True
    cls = type(obj)
    return (
        inspect.getattr_static(cls, "__getattr__", None) is None
        and cls.__getattribute__ is object.__getattribute__
    )


def _async_callable(fn: Any) -> bool:
    """``steering._is_async_callable`` without running an object's dynamic
    attribute lookup.

    The harness describes the objects it serves the worker by probing them
    (``_is_coroutine_marker``, ``__wrapped__``). An environment object that
    answers any attribute name -- AppWorld's ``apis`` turns ``apis.<name>``
    into an app lookup and raises for unknown names -- would take such a probe
    for a request. For those objects only what the instance or its class
    holds is read (``inspect.getattr_static``); every other object is probed
    as before.
    """
    from unify.function_manager.steering import _is_async_callable

    seen: set[int] = set()
    while fn is not None and id(fn) not in seen:
        seen.add(id(fn))
        if _plain(fn):
            return _is_async_callable(fn)
        marker = getattr(inspect, "_is_coroutine_mark", None)
        if marker is not None and (
            inspect.getattr_static(fn, "_is_coroutine_marker", None) is marker
        ):
            return True
        fn = inspect.getattr_static(fn, "__wrapped__", None)
    return False


def _environment_global(name: str, value: Any) -> bool:
    """Whether *value* is the global ``name`` a registered environment binds
    (``UNIFY_ENV_NAMESPACES``); modules are imported as they are."""
    if isinstance(value, types.ModuleType):
        return False
    from unify.function_manager.primitives.environment import environment_globals

    try:
        bound = environment_globals()
    except Exception:  # noqa: BLE001 - an environment that fails to load binds nothing
        return False
    return name in bound and bound[name] is value


def _core_surface() -> bool:
    from unify.actor import core_surface

    return core_surface.enabled()


def _function_library(value: Any) -> Any:
    """*value* when it is a core-surface ``functions`` library, else None."""
    if value is None or not _core_surface():
        return None
    from unify.actor.core_surface import FunctionLibrary

    return value if isinstance(value, FunctionLibrary) else None


def _no_refs(value: Any, where: str) -> Any:
    raise BoundaryRefusal(f"{where} is not plain data")


def _append_text(into: List[Any], text: str, text_part: type) -> None:
    if not text:
        return
    if into and isinstance(into[-1], text_part):
        into[-1] = text_part(text=into[-1].text + text)
    else:
        into.append(text_part(text=text))


def _decode_png(data: str) -> Any:
    try:
        import base64
        import io

        from PIL import Image
    except ImportError:
        return None
    try:
        image = Image.open(io.BytesIO(base64.b64decode(data)))
        image.load()
        return image
    except Exception:  # noqa: BLE001 - passed through as sent instead
        return None
