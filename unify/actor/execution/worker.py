"""Python cells in a sandboxed child process.

The actor's and each teammate's interpreter (unify/common/box_interpreter.py)
is one :class:`PythonWorker`: a persistent ``python -I -S`` started inside the
workspace sandbox (unify/sandbox.py) -- scrubbed environment, state directory
hidden, network per ``UNIFY_WORKSPACE_NETWORK``. Cells run there, so a cell's
variables persist from one cell to the next, but the cell cannot read the
harness's environment or open a socket the sandbox does not allow.

What the cell needs from the harness it reaches through a proxy served here.
Every name the request's namespace holds beyond the base globals is installed
in the worker as one of:

* a **remote** object (``primitives``, environment namespaces, the globals a
  registered environment binds (``apis``), ``query_llm``): attribute access
  asks the harness what the attribute is, and a call is sent over the channel,
  run here against the namespace's *current* binding of that name, and its
  result sent back;
* a **value** (plain data), sent once;
* a **worker-local** stand-in (``display``, ``run_coro_sync``,
  ``ModelRequestError``, ``ModelCallFailed``);
* or **refused**, naming why, when none of these applies.

Values cross as tagged JSON (worker_child.py explains why not pickle). A result
that is not data comes back as a reference the worker can call and pass back;
a worker object that is not data is refused as an argument, naming it, and
comes back as a cell result only as its repr. Exceptions raised here are
re-raised in the worker under their own type (builtins) or a class of the same
name and module.

A cell that is stopped (a cancel) is interrupted where it is and the worker,
with its variables, is kept; a cell that times out, or does not stop within
``INTERRUPT_GRACE_S`` of its interrupt, kills the worker (its whole sandbox);
the next cell starts a fresh one and its output says so.
"""

from __future__ import annotations

import asyncio
import functools
import inspect
import itertools
import json
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
]

#: Largest single message either side may send.
MAX_MESSAGE_BYTES = 64 * 1024 * 1024
START_TIMEOUT_S = 60.0
#: How long a cancelled cell has to stop after the worker is asked to interrupt it (inside the tool loop's own
#: two seconds for a cancelled call) before the worker is killed instead.
INTERRUPT_GRACE_S = 1.5


_BOOTSTRAP = (
    "import runpy, sys; runpy.run_path(sys.argv[1], run_name='__unify_worker__')"
)

#: Namespace globals that never cross: interpreter plumbing.
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


def _subscriptable(obj: Any) -> bool:
    """Whether a cell's ``proxy[key]`` may run ``obj[key]`` in the harness.

    Only a plain list, tuple or dict (whose ``__getitem__`` runs no harness
    code), or an object whose class declares that its ``__getitem__`` is as
    safe to reach from a cell as its public attributes, with the class
    attribute ``boundary_subscript = True``. Checked before ``obj[key]`` runs.
    """
    if type(obj) in (list, tuple, dict):
        return True
    return getattr(type(obj), "boundary_subscript", False) is True


def _credential_inside(value: Any) -> Optional[str]:
    """What in a value about to cross by value looks like a credential, or None.

    Walks nested lists, tuples, sets and dicts (each container once in each
    context). A text leaf (``str``, or ``bytes``/``bytearray`` read as latin-1) anywhere,
    key or value, is a credential when it is key-shaped or a URL with a
    password (:func:`unify.sandbox._looks_like_key`). Everything under a
    credential-named dict key (:func:`unify.sandbox._is_secret_env`, the
    environment scrub's own test) is in a secret context: any non-empty text
    leaf there, however deep, is a credential, while numbers, flags, None and
    empty text cross (``max_tokens: 5``, ``password: ""``). Objects of other
    types are not looked into: they cross by reference.
    """
    stack: list = [(value, None)]
    seen: set = set()
    while stack:
        item, secret = stack.pop()
        if isinstance(item, (str, bytes, bytearray)):
            text = item if isinstance(item, str) else bytes(item).decode("latin-1")
            if secret is not None and len(text) > 0:
                return f"the key {secret!r}"
            if sandbox._looks_like_key(text):
                return "a key-shaped string"
            continue
        if not isinstance(item, (list, tuple, set, frozenset, dict)):
            continue
        if (id(item), secret) in seen:
            continue
        seen.add((id(item), secret))
        if isinstance(item, dict):
            for key, inner in item.items():
                named = (
                    isinstance(key, str)
                    and sandbox._is_secret_env(key, inner if isinstance(inner, str) else "")
                )
                stack.append((key, secret))
                stack.append((inner, key if named else secret))
        else:
            stack.extend((inner, secret) for inner in item)
    return None


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
        # The policy and binds the running sandbox was started with: the
        # paths sent to the child are those it shows (sandbox.box_path).
        self._policy: Optional[sandbox.SandboxPolicy] = None
        self._binds: list = []
        # UNIFY_MEMORY=in_episode: the library copy's pin generation this worker's modules were loaded at (None:
        # nothing loaded yet), and a re-pin's notice not yet shown (its cell ended before it could be)
        self._memory_gen: Optional[int] = None
        self._memory_notice: Optional[str] = None
        self._memory_paths: List[str] = []
        # The persistent interpreter (unify/common/box_interpreter.py) sets these before each start: the tool
        # channel's two box ends (pipes, inherited under the same numbers), the box environment entry naming them,
        # and a callback run once the sandbox has started (the channel closes the host's copies of the box ends).
        self.box_fds: tuple = ()
        self.box_env: Dict[str, str] = {}
        self.on_spawn: Optional[Callable[[], None]] = None

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
        from unify.memory.integration import hooks as _mv2

        # UNIFY_MEMORY=on: the request's memory export is imported first.
        memory_paths = _mv2.worker_paths()
        if memory_paths:
            # the library as ``memory`` in every cell, imported on first use
            specs.update(_mv2.worker_globals())
        paths = [*memory_paths, *(p for p in sys.path if p)]
        # Packages the harness installs later land here; visible once it exists.
        packages = str(environment.site_packages())
        if packages not in paths:
            paths.append(packages)
        # Each as the sandbox shows it: the workspace, the environment and
        # the memory library copy are at fixed paths there (sandbox.wrap_argv).
        policy = self._policy
        if policy is not None:
            mounts = sandbox.neutral_mounts(policy, self._binds)
            paths = [sandbox.box_path(p, policy, mounts=mounts) for p in paths]
        # UNIFY_MEMORY=in_episode: the memory paths as sent, which a re-pin replaces (Worker.repin_modules)
        self._memory_paths = paths[: len(memory_paths)]
        msg = {
            "op": "init",
            "sys_path": paths,
            "builtins": builtins_names,
            "globals": specs,
        }
        msg["help"] = True
        if memory_paths:
            # UNIFY_MEMORY=on: the child writes no bytecode (the child runs under -I, so
            # PYTHONDONTWRITEBYTECODE would be ignored); otherwise no key at all.
            msg["no_bytecode"] = True
        # UNIFY_MEMORY=on, while the request's work tree is captured: the
        # child installs the audit hook (spec §3a); otherwise no key at all.
        audit = _mv2.worker_audit()
        if audit is not None:
            msg["audit"] = audit
        return msg

    async def _start(self) -> None:
        from unify import environment

        # a fresh process loads the library copy as it is now (UNIFY_MEMORY=in_episode)
        self._memory_gen = None
        self._memory_paths = []

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
        # The sandbox's whole environment (wrap_argv clears the inherited one):
        # anything the child needs must be in it before wrap_argv is called.
        env = sandbox.sandbox_env(policy)
        workspace = str(policy.workspace)
        from unify.memory.integration import hooks as _mv2

        # UNIFY_MEMORY=on: the request's memory export, read-write (v2) ...
        mounts = _mv2.worker_mounts()
        # ... or read-only (spec v2.1 §6)
        late_ro = _mv2.worker_readonly_mounts()
        env = {**env, **self.box_env}
        wrapped = sandbox.wrap_argv(
            argv,
            policy,
            cwd=workspace,
            writable=mounts,
            late_readonly=late_ro,
            env=env,
            neutral=True,
        )
        self._policy = policy
        self._binds = [*mounts, *late_ro]
        self._proc = await asyncio.create_subprocess_exec(
            *wrapped,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=env,
            cwd=workspace,
            start_new_session=True,
            limit=MAX_MESSAGE_BYTES,
            # close_fds stays on: the box inherits only the standard streams and these pipes
            pass_fds=tuple(self.box_fds),
        )
        if self.on_spawn is not None:
            try:
                self.on_spawn()
            except BaseException:
                await self.close()
                raise
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
        if ready.get("audit", "on") != "on":
            logger.warning("memory v2: no audit hook in the worker: %r", ready["audit"])

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
        from unify.common import process_tree

        proc, self._proc = self._proc, None
        self._refs.clear()
        self._ref_ids.clear()
        self._synced = {}
        if proc is None:
            return
        if proc.returncode is None:
            # SIGKILL to the process group and every process below it, verified gone
            process_tree.kill_tree(proc.pid)
        try:
            await asyncio.wait_for(proc.wait(), timeout=10)
        except asyncio.TimeoutError:
            logger.warning("sandboxed Python worker %s did not exit", proc.pid)
        if self._stderr_task is not None:
            self._stderr_task.cancel()
            self._stderr_task = None

    async def _interrupt(self, cid: int, shadow: Dict[str, Any]) -> bool:
        """A cancelled cell: ask the worker to interrupt cell *cid* where it is, and serve it until it ends. True
        when it ends within :data:`INTERRUPT_GRACE_S` (the worker and its variables are kept); False otherwise
        (a cell that ignores the interrupt, code blocked where it cannot be interrupted, a broken channel), and
        the caller kills the worker."""
        try:
            await self._send({"op": "interrupt", "id": cid})
        except (WorkerDied, BoundaryRefusal):
            return False
        serving = asyncio.ensure_future(self._serve_until_done(cid, shadow))
        finished, _ = await asyncio.wait({serving}, timeout=INTERRUPT_GRACE_S)
        if not finished:
            serving.cancel()
            await asyncio.gather(serving, return_exceptions=True)
            return False
        if serving.exception() is not None:
            return False
        from unify.memory.integration import hooks as _mv2

        # UNIFY_MEMORY=on: the interrupted cell's audit records, as for any cell that ends
        _mv2.worker_cell_done(serving.result().get("audit"))
        return True

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
    def _describe_global(self, name: str, value: Any) -> dict:
        from unify.common.asyncio_compat import run_coro_sync
        from unify.common.cell_models import ModelCallFailed, ModelRequestError

        if name == "display" and callable(value):
            return {"kind": "local", "local": "display"}
        if value is run_coro_sync:
            return {"kind": "local", "local": "run_coro_sync"}
        if value is ModelRequestError:
            # The worker's own class, which the harness's error arrives as.
            return {"kind": "local", "local": "model_request_error"}
        if value is ModelCallFailed:
            return {"kind": "local", "local": "model_call_failed"}
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
        return {
            "kind": "remote",
            "callable": callable(value),
            "async": _async_callable(value),
            "repr": short_repr(value, 200),
        }

    def _manifest(self, shadow: Dict[str, Any]) -> dict:
        want: Dict[str, dict] = {}
        for name, value in list(shadow.items()):
            if not isinstance(name, str) or name in _SKIP:
                continue
            if name.startswith("__"):
                continue
            if name in self._base and self._base[name] is value:
                continue
            want[name] = self._describe_global(name, value)
        changed: Dict[str, dict] = {}
        for name, desc in want.items():
            # The worker resolves a root by name, so only its kind matters.
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
            if isinstance(attr, dict):
                # A subscript (``proxy[key]``): a str or int key, never one
                # starting with "_", applied with the same checks as getattr.
                key = attr.get("item")
                if (
                    set(attr) != {"item"}
                    or type(key) not in (str, int)
                    or (isinstance(key, str) and key.startswith("_"))
                ):
                    raise BoundaryRefusal(
                        f"{label}[{key!r}]: only str and int subscripts not "
                        "starting with '_' of harness objects cross the worker "
                        "boundary",
                    )
                if not _subscriptable(obj):
                    raise BoundaryRefusal(
                        f"{label}[{key!r}]: a {child.type_name(obj)} cannot be "
                        "subscripted across the worker boundary; only a list, "
                        "tuple or dict, or an object whose class sets "
                        "`boundary_subscript = True`, can",
                    )
                obj = obj[key]
                label = f"{label}[{key!r}]"
                _refuse_unreachable(obj, label)
                continue
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
        if callable(obj):
            return {"kind": "callable", "async": _async_callable(obj)}
        if isinstance(obj, _PASSTHROUGH_TYPES):
            if sandbox.is_secret_name(name):
                raise BoundaryRefusal(
                    f"{name!r} looks like a credential (rule `env-scrub`: "
                    f"{sandbox.RULES['env-scrub']}); its value stays in the harness",
                )
            found = _credential_inside(obj)
            if found is not None:
                raise BoundaryRefusal(
                    f"{name!r} holds {found}, which looks like a credential (rule "
                    f"`env-scrub`: {sandbox.RULES['env-scrub']}); its value stays "
                    "in the harness",
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


    async def _answer(
        self,
        op: Any,
        msg: dict,
        shadow: Dict[str, Any],
        reply: Dict[str, Any],
    ) -> None:
        if op == "doc":
            from unify.actor import core_surface

            target = msg.get("target")
            if not target:
                reply["value"] = core_surface.INDEX_HELP
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
            last = path[-1]
            if isinstance(last, dict):
                # A subscript's key names the value as an attribute does, so a
                # credential-named key's value stays in the harness.
                last = last.get("item", "")
            reply["value"] = self._describe(obj, str(last))
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


    async def _serve_until_done(self, cid: int, shadow: Dict[str, Any]) -> dict:
        tasks: set[asyncio.Task] = set()
        try:
            while True:
                msg = await self._read()
                op = msg.get("op")
                if op in _SERVED:
                    # Each request runs in a copy of the running cell's
                    # context, so what the harness scoped to the cell with a
                    # ContextVar (an environment observer, a case recording)
                    # is in force for it, as for an in-process cell.
                    task = asyncio.create_task(self._serve(msg, shadow))
                    tasks.add(task)
                    task.add_done_callback(tasks.discard)
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
        stdout: List[Any],
        stderr: List[Any],
    ) -> Any:
        """Run one wrapped cell (``async def __exec_wrapper(): ...``) in the worker.

        Output parts are appended to *stdout* / *stderr*. Returns the cell's
        result; raises :class:`WorkerCellError` with the worker's traceback,
        or ``asyncio.TimeoutError`` after killing the worker.
        """
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
        from unify.memory.integration import hooks as _mv2

        # UNIFY_MEMORY=in_episode: at this cell boundary the library copy may be re-pinned to a commit the writer
        # landed while the episode runs; this worker then drops the library's modules before the cell (None: off)
        step = await asyncio.to_thread(_mv2.before_cell) if _mv2.in_episode_active() else None
        try:
            repin = self._memory_step(step)
            await self._send(
                {
                    "op": "exec",
                    "id": cid,
                    "source": source,
                    "sync": self._manifest(shadow),
                    **({"repin": repin} if repin is not None else {}),
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
            kept = False
            try:
                kept = await self._interrupt(cid, shadow)
            finally:
                if not kept:
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
        finally:
            if step is not None:
                _mv2.after_cell()

        # UNIFY_MEMORY=on: the cell's audit records, stamped on this clock.
        _mv2.worker_cell_done(done.get("audit"))
        self._collect_parts(done.get("stdout") or [], stdout, TextPart)
        self._collect_parts(done.get("stderr") or [], stderr, TextPart)
        if self._memory_notice:
            # UNIFY_MEMORY=in_episode: the cell's result ends with what the re-pin changed (one line)
            notice, self._memory_notice = self._memory_notice, None
            ends_open = bool(stdout) and isinstance(stdout[-1], TextPart) and not stdout[-1].text.endswith("\n")
            _append_text(stdout, ("\n" if ends_open else "") + notice + "\n", TextPart)
        if done.get("error"):
            # The model reads the traceback without the harness's own frames (the tools stub, its channel
            # client, the library package's module and contract); the full one is logged.
            error = str(done["error"])
            shown = done.get("shown_error")
            if isinstance(shown, str) and shown:
                from unify.common.box_interpreter import bounded_for_log

                logger.info("cell error, full traceback:\n%s", bounded_for_log(error))
                error = shown
            raise WorkerCellError(error)
        return self.decode(done.get("result"), shadow)

    def _memory_step(self, step: Optional[dict]) -> Optional[dict]:
        """This worker's part of a cell boundary under ``UNIFY_MEMORY=in_episode`` (*step*: ``hooks.before_cell``):
        when the copy was re-pinned since this worker loaded it, what the child does before the cell
        (``repin_modules``: the copy's roots as the sandbox shows them, under which every loaded module is dropped,
        the import paths of the old pin to remove and the new pin's to put first); None otherwise. The notice is kept
        for this cell's result."""
        if not isinstance(step, dict):
            return None
        gen = step.get("generation")
        stale = self._memory_gen is not None and gen != self._memory_gen
        self._memory_gen = gen
        if isinstance(step.get("notice"), str) and step["notice"]:
            self._memory_notice = step["notice"]
        if not stale:
            return None
        from unify.memory.integration import hooks as _mv2

        roots, paths = _mv2.repin_roots(), _mv2.worker_paths()
        policy = self._policy
        if policy is not None:
            mounts = sandbox.neutral_mounts(policy, self._binds)
            roots = [sandbox.box_path(r, policy, mounts=mounts) for r in roots]
            paths = [sandbox.box_path(p, policy, mounts=mounts) for p in paths]
        old, self._memory_paths = self._memory_paths, list(paths)
        return {"roots": roots, "drop_paths": old, "paths": list(paths)}

    @staticmethod
    def _collect_parts(parts: list, into: List[Any], text_part: type) -> None:
        from .types import ImagePart

        for part in parts:
            if not isinstance(part, dict):
                continue
            if part.get("type") == "text":
                _append_text(into, str(part.get("text") or ""), text_part)
            elif part.get("type") == "image":
                into.append(ImagePart(mime="image/png", data=str(part.get("data") or "")))


#: What the worker asks of the harness, each served in a task of its own.
_SERVED = frozenset({"call", "describe", "dir", "doc"})


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


#: Values the harness hands back as data, never by reference.
_PASSTHROUGH_TYPES = (
    str,
    bytes,
    bytearray,
    int,
    float,
    bool,
    complex,
    type(None),
    list,
    tuple,
    dict,
    set,
    frozenset,
)


def _is_async_callable(fn: Any) -> bool:
    """Whether *fn* ultimately dispatches a coroutine: ``iscoroutinefunction``,
    following ``__wrapped__`` through synchronous wrappers."""
    seen: set[int] = set()
    while fn is not None and id(fn) not in seen:
        if asyncio.iscoroutinefunction(fn):
            return True
        seen.add(id(fn))
        fn = getattr(fn, "__wrapped__", None)
    return False


def _async_callable(fn: Any) -> bool:
    """:func:`_is_async_callable` without running an object's dynamic
    attribute lookup.

    The harness describes the objects it serves the worker by probing them
    (``_is_coroutine_marker``, ``__wrapped__``). An environment object that
    answers any attribute name -- AppWorld's ``apis`` turns ``apis.<name>``
    into an app lookup and raises for unknown names -- would take such a probe
    for a request. For those objects only what the instance or its class
    holds is read (``inspect.getattr_static``); every other object is probed
    as before.
    """
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


def _no_refs(value: Any, where: str) -> Any:
    raise BoundaryRefusal(f"{where} is not plain data")


def _append_text(into: List[Any], text: str, text_part: type) -> None:
    if not text:
        return
    if into and isinstance(into[-1], text_part):
        into[-1] = text_part(text=into[-1].text + text)
    else:
        into.append(text_part(text=text))
