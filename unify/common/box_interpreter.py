"""One persistent confined Python interpreter per agent: per pass for the writer, the curator and each helper, per
request for the actor and each teammate (``UNIFY_CELL_STATE=persistent``; :func:`cell_state`).

The interpreter is the actor's worker child (``unify/actor/execution/worker_child.py``: ``python -I -S``, it imports
nothing from the harness). Cells go to it as ``exec`` messages on its own protocol pipes and come back as ``done``
messages, so the host always knows whether a cell is running, and the code never appears in an argument vector.

* Memory agents: the child runs in :func:`unify.memory.sandbox_run.start_confined`'s box, the same box as every
  memory cell (``--unshare-all``, ``--die-with-parent``, ``--new-session``, ``--clearenv``, the seccomp filter, the
  rlimits; only pipe descriptors passed in). No harness object is proxied to it: every ``call``, ``describe``,
  ``dir``, ``doc`` or recording request it sends is refused. The worker's own observe-only audit
  (``worker_child._install_audit``, loading ``unify/memory/integration/adapters/audit.py``) returns each cell's
  records with its ``done`` message; they fill the open-log fields and ``box-opens.jsonl`` rows that the
  ``sitecustomize`` hook fills for a fresh cell (:func:`open_log`), stamped here with agent, pass and cell. Like the
  hook it is a measurement, not a control.
* Actor and teammates: their existing :class:`~unify.actor.execution.worker.PythonWorker`, one per agent, with the
  tool channel's two box ends added to the descriptors its sandbox inherits.

The tool channel (:mod:`unify.common.tool_channel`) passes two pipe ends into the box, named by ``TOOLS_CHANNEL``;
the in-box ``tools`` module is staged read-only (created new, never through a link, mode 0444) and imported when
the interpreter starts.

Restart rules. Python cannot interrupt a running cell safely, so there is no soft cancel: a cell past its time limit
gets the whole box killed with the verified kill (SIGKILL to the process group and every descendant, verified gone,
raise otherwise), and the model is told the interpreter was restarted. The same happens when the ``/memory`` quota
is broken, when the channel closes (a malformed frame, or a frame between cells), when the interpreter dies, and
when anything a cell started (a process in the box or a thread of the worker) is still alive after the cell says it
ended: ``done`` is only the cell's own claim, so the host checks the box's process tree and the worker's thread
count before the next cell. An interpreter left idle for :attr:`MemoryProfile.idle_restart_s` is stopped to bound
resident memory (up to 17 run at once in a pass); the next cell starts a fresh one and is told so. The fresh box
starts with the next cell (a stopped box holds nothing that can write ``/memory``; :meth:`BoxInterpreter.box_alive`).
"""

from __future__ import annotations

import asyncio
import builtins
import functools
import itertools
import json
import logging
import os
import posixpath
import re
import shutil
import signal
import subprocess
import tempfile
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Awaitable, Callable, Mapping, Optional

logger = logging.getLogger(__name__)

__all__ = [
    "ActorProfile",
    "Binds",
    "BoxInterpreter",
    "CellResult",
    "MemoryProfile",
    "cell_state",
    "open_log",
    "persistent",
    "stage_read_only",
    "wrap_cell",
]

# --- the switch ------------------------------------------------------------------------------------------------

#: The setting that selects the persistent interpreter (``persistent``) or a fresh box per cell (``fresh``).
CELL_STATE_SETTING = "UNIFY_CELL_STATE"
CELL_STATES = ("persistent", "fresh")


def cell_state() -> str:
    """``persistent`` (the default) or ``fresh``: the settings field when the settings define it, else the process
    environment. Any other value is refused."""
    value: Any = None
    try:
        from unify.settings import SETTINGS

        value = getattr(SETTINGS, CELL_STATE_SETTING, None)
    except Exception:  # noqa: BLE001 - the environment below is the same source the settings read
        value = None
    if value is None:
        value = os.environ.get(CELL_STATE_SETTING)
    text = str(value if value is not None else "persistent").strip().lower() or "persistent"
    if text not in CELL_STATES:
        raise ValueError(f"{CELL_STATE_SETTING} must be one of {', '.join(CELL_STATES)}, not {text!r}")
    return text


def persistent() -> bool:
    return cell_state() == "persistent"


# --- what the model is told -----------------------------------------------------------------------------------

_LOST = "variables are gone, files remain"


def _timed_out_note(limit_s: float) -> str:
    return f"(the cell timed out after {limit_s:g} s; the interpreter was restarted: {_LOST})"


def _restart_note(why: str) -> str:
    return f"({why}; the interpreter was restarted: {_LOST})"


# --- in-box layout --------------------------------------------------------------------------------------------

#: Read-only directory holding the worker child and its audit module (not on the import path).
BOX_INTERPRETER_DIR = "/interpreter"
#: Read-only directory holding the generated ``tools`` module (first on the import path).
BOX_TOOLS_DIR = "/tools"
TOOLS_MODULE = "tools"
#: The tool channel's environment variable in the box (its value comes from the channel).
TOOLS_CHANNEL_ENV = "TOOLS_CHANNEL"

_UNIFY = Path(__file__).resolve().parents[1]
WORKER_CHILD_SOURCE = _UNIFY / "actor" / "execution" / "worker_child.py"
AUDIT_SOURCE = _UNIFY / "memory" / "integration" / "adapters" / "audit.py"
_AUDIT_NAME = "cell_audit.py"

# The same start-up line as the actor's worker (worker.py ``_BOOTSTRAP``): the child file runs by path.
BOOTSTRAP = "import runpy, sys; runpy.run_path(sys.argv[1], run_name='__unify_worker__')"

#: Largest single protocol message either side sends (the actor worker's limit).
MAX_MESSAGE_BYTES = 64 * 1024 * 1024
START_TIMEOUT_S = 60.0
#: How long the host waits for a thread or process a cell just ended to disappear before calling it left running.
SETTLE_S = 1.0
_SETTLE_STEP_S = 0.05
_CLOCK_TICK_S = 0.1
#: Requests a memory agent's worker may send that would reach a harness object; each is refused.
_PROXY_OPS = frozenset({"call", "describe", "dir", "fn_begin", "fn_end", "doc"})
_NO_PROXY = (
    "no harness object is reachable from this interpreter; the agent's tools are the functions of the "
    "`tools` module"
)
_WRAPPER_FRAME = re.compile(r'^  File "<string>", line \d+, in __exec_wrapper\n', re.M)


def stage_read_only(folder: Path, name: str, data: bytes) -> Path:
    """Write *data* as ``<folder>/<name>``: created new (``O_EXCL``), never through a link (``O_NOFOLLOW``), mode
    0444, as :func:`unify.memory.sandbox_run.stage_open_hook` stages the audit hook."""
    path = Path(folder) / name
    fd = os.open(
        path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC,
        0o444,
    )
    with os.fdopen(fd, "wb") as fh:
        fh.write(data)
    return path


def wrap_cell(code: str, filename: str, *, last_value: bool = False) -> str:
    """The ``exec`` message's source for *code*: an ``async def __exec_wrapper()`` (what the worker child runs)
    that compiles the code under *filename* and runs it at the namespace's top level, so every name it binds
    persists exactly as in a script, and registers its lines so a traceback shows them. The standard ``builtins``
    module's ``exec`` and ``compile`` are used, since a namespace may hold a reduced builtins mapping (the actor's);
    the code itself still runs with the namespace's own builtins. *last_value*: a final expression statement is
    evaluated apart and its value is the cell's result (the actor's cells)."""
    lines = [
        "async def __exec_wrapper():",
        "    __b = __import__('builtins')",
        f"    __s = {code!r}",
        f"    __import__('linecache').cache[{filename!r}] = (__b.len(__s), None, __s.splitlines(True), {filename!r})",
    ]
    if last_value:
        lines += [
            "    __a = __import__('ast')",
            f"    __t = __b.compile(__s, {filename!r}, 'exec', __a.PyCF_ONLY_AST, dont_inherit=True)",
            "    __e = __t.body.pop() if __t.body and __b.isinstance(__t.body[-1], __a.Expr) else None",
            f"    __b.exec(__b.compile(__t, {filename!r}, 'exec', dont_inherit=True), __b.globals())",
            "    if __e is not None:",
            f"        return __b.eval(__b.compile(__a.Expression(__e.value), {filename!r}, 'eval', dont_inherit=True),"
            " __b.globals())",
        ]
    else:
        lines.append(f"    __b.exec(__b.compile(__s, {filename!r}, 'exec', dont_inherit=True), __b.globals())")
    return "\n".join(lines) + "\n"


def tools_bootstrap(stub_source: str) -> str:
    """The ``exec`` message that makes *stub_source* the module ``tools`` (in ``sys.modules`` and in the
    namespace), for a box with no read-only directory of its own to stage it in (the actor's). The module runs with
    the standard builtins (the stub itself uses ``exec`` and ``compile``, which the actor's namespace leaves out),
    except ``print``, which is the namespace's own, so what ``tools.list()`` and ``tools.help()`` print is the cell's
    output. A narrower builtins mapping is a convenience for cells, not a boundary: the box is."""
    return (
        "async def __exec_wrapper():\n"
        "    __b = __import__('builtins')\n"
        "    __m = __import__('types').ModuleType('tools')\n"
        "    __m.__file__ = '<tools>'\n"
        "    __own = __b.globals().get('__builtins__')\n"
        "    __nb = __b.dict(__b.vars(__b))\n"
        "    if __b.isinstance(__own, __b.dict) and __own.get('print') is not None:\n"
        "        __nb['print'] = __own['print']\n"
        "    __m.__dict__['__builtins__'] = __nb\n"
        f"    __b.exec(__b.compile({stub_source!r}, '<tools>', 'exec', dont_inherit=True), __m.__dict__)\n"
        "    __import__('sys').modules['tools'] = __m\n"
        "    __b.globals()['tools'] = __m\n"
    )


# --- the open log -----------------------------------------------------------------------------------------------

_OPEN_MODES = frozenset({"r", "w", "a", "x"})


def _open_mode(mode: Any) -> Optional[str]:
    """The open log's one-letter mode for an audit record's mode (the hook's rule, ``box_audit_hook._mode``)."""
    if not isinstance(mode, str):
        return None
    for ch in ("w", "a", "x"):
        if ch in mode:
            return ch
    return "w" if "+" in mode else "r"


def open_log(audit: Any, audit_state: Optional[str], areas: tuple[str, ...] = ("/inputs", "/memory")) -> dict:
    """The open log of one cell from the worker's audit records, in the shape of
    :meth:`unify.memory.sandbox_run.BoxOpenLog.finish` (``rows``, ``dropped``, ``dead``, ``reason``), so
    :func:`unify.memory.sandbox_run.box_open_fields` gives the tool-event fields unchanged.

    Rows are ``{"event": "open", "path", "mode"}`` for opens under *areas* only, the hook's areas. ``dead`` is True
    when the audit was not installed (``reason`` ``no_start``) or the cell's records did not arrive
    (``reader_failed``); ``dropped`` counts the events the audit lost to its caps and the open records that are
    not well formed. The records are the cell's own reports: a cell can forge or suppress them, as it can the hook's.
    """
    if audit_state != "on":
        return {"rows": [], "dropped": 0, "dead": True, "reason": "no_start"}
    if not isinstance(audit, dict) or not isinstance(audit.get("records"), list):
        return {"rows": [], "dropped": 0, "dead": True, "reason": "reader_failed"}
    rows: list[dict] = []
    dropped = audit.get("dropped")
    dropped = dropped if isinstance(dropped, int) and not isinstance(dropped, bool) and dropped > 0 else 0
    for rec in audit["records"]:
        if not isinstance(rec, dict) or rec.get("event") != "open":
            continue
        path, mode = rec.get("path"), _open_mode(rec.get("mode"))
        if not isinstance(path, str) or mode not in _OPEN_MODES or len(path) > 4096:
            dropped += 1
            continue
        if posixpath.normpath(path) != path:
            dropped += 1
            continue
        if not any(path == a or path.startswith(a + "/") for a in areas):
            continue
        rows.append({"event": "open", "path": path, "mode": mode})
    return {"rows": rows, "dropped": dropped, "dead": False, "reason": "ok"}


# --- profiles ---------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Binds:
    """Host path -> box path, as :func:`unify.memory.sandbox_run.run_confined` takes them."""

    ro: Mapping[Path, str] = field(default_factory=dict)
    rw: Mapping[Path, str] = field(default_factory=dict)
    late_ro: Mapping[Path, str] = field(default_factory=dict)


@dataclass(frozen=True)
class MemoryProfile:
    """A memory agent's interpreter box (writer, curator, helper).

    *sys_path*: box paths first on the import path after the ``tools`` directory (today's cells use
    ``PYTHONPATH=/inputs:/memory``). *fsize_bytes*: the file-size limit lowered for the interpreter (today's per-cell
    ``prlimit --fsize``). *measure*: the ``/memory`` quota measure (``writer_pass._measure`` of the box), run after
    every cell and from :meth:`BoxInterpreter.measure_quota`; a breach text kills the box. *opens_path*: the pass's
    ``box-opens.jsonl``. *idle_restart_s*: an interpreter idle this long is stopped (None: never).
    """

    pass_id: Optional[str] = None
    cwd: str = "/memory"
    sys_path: tuple[str, ...] = ("/inputs", "/memory")
    fsize_bytes: Optional[int] = None
    audit_roots: tuple[str, ...] = ("/inputs", "/memory")
    open_areas: tuple[str, ...] = ("/inputs", "/memory")
    opens_path: Optional[Path] = None
    measure: Optional[Callable[[], Optional[str]]] = None
    idle_restart_s: Optional[float] = 600.0


@dataclass
class ActorProfile:
    """The actor's or a teammate's interpreter: its existing worker and the host-side namespace the worker's proxy
    serves (``PythonWorker.run_cell``'s *shadow*). *worker* None: a new :class:`PythonWorker`."""

    shadow: dict = field(default_factory=dict)
    worker: Any = None
    idle_restart_s: Optional[float] = 3600.0
    #: a final expression's value is the cell's result (:attr:`CellResult.result`)
    last_value: bool = True


# --- the result -------------------------------------------------------------------------------------------------


@dataclass
class CellResult:
    """One cell's outcome. *output*: what it printed (stdout, then stderr), *error*: its traceback text or None.
    *timed_out*: it passed its limit. *restarted*: the interpreter's state was dropped before this cell ran or when
    it ended (the notes say why). *opens*: the cell's open log (memory agents; :func:`open_log`), None when no
    measurement applies. *quota*: the ``/memory`` quota breach found after the cell, if any.
    """

    output: str
    error: Optional[str] = None
    timed_out: bool = False
    restarted: bool = False
    opens: Optional[dict] = None
    quota: Optional[str] = None
    cell: int = 0
    notes_before: list[str] = field(default_factory=list)
    notes_after: list[str] = field(default_factory=list)
    result: Any = None
    parts: list = field(default_factory=list)

    @property
    def note(self) -> str:
        return "\n".join([*self.notes_before, *self.notes_after])

    def text(self) -> str:
        """The cell's output as the model reads it: the notes about an earlier restart, what it printed and its
        traceback (``(no output)`` when there is none), then the notes about how it ended."""
        body = self.output
        if self.error:
            body = f"{body}\n{self.error}" if body else self.error
        lines = [*self.notes_before, body.rstrip("\n") or "(no output)", *self.notes_after]
        return "\n".join(lines)


# --- process checks and the verified kill ---------------------------------------------------------------------


def _tree(pid: int) -> frozenset:
    from unify.memory import sandbox_run as _sr

    return frozenset(_sr._descendants(pid))


def _worker_pid(pid: int, tree: frozenset) -> Optional[int]:
    """The interpreter's host pid: the descendant of the box's first process with no child of its own."""
    from unify.memory import sandbox_run as _sr

    parents = set()
    for p, _ in tree:
        st = _sr._proc_stat(p)
        if st is not None:
            parents.add(st[0])
    leaves = [p for p, _ in tree if p not in parents]
    return leaves[0] if len(leaves) == 1 else None


def _threads(pid: Optional[int]) -> Optional[int]:
    if pid is None:
        return None
    try:
        return len(os.listdir(f"/proc/{pid}/task"))
    except OSError:
        return None


async def kill_verified(pid: int, exited: Callable[[], bool], wait: Callable[[], Awaitable[Any]]) -> None:
    """SIGKILL the process group *pid* leads and every descendant; return once all are gone, raise otherwise
    (:func:`unify.memory.sandbox_run._kill_box` for a process the event loop owns). *exited* tells whether *pid*
    itself has been reaped; *wait* reaps it."""
    from unify.memory import sandbox_run as _sr

    victims = _sr._descendants(pid)
    try:
        os.killpg(pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        pass
    for p, start in victims:
        if _sr._alive(p, start):
            try:
                os.kill(p, signal.SIGKILL)
            except ProcessLookupError:
                pass
    grace = float(_sr._KILL_GRACE_S)
    try:
        await asyncio.wait_for(wait(), timeout=grace)
    except asyncio.TimeoutError:
        pass
    deadline = time.monotonic() + grace
    while True:
        survivors = [p for p, start in victims if _sr._alive(p, start)]
        if exited() and not survivors:
            return
        if time.monotonic() > deadline:
            raise RuntimeError(f"interpreter processes survived SIGKILL: {pid}, others {survivors}")
        await asyncio.sleep(0.1)


# --- backends ---------------------------------------------------------------------------------------------------


class _Died(Exception):
    pass


class _MemoryBackend:
    """``worker_child.py`` in a memory agent's box, spoken to over its stdin and stdout pipes."""

    def __init__(self, owner: "BoxInterpreter", profile: MemoryProfile, binds: Binds, env: dict) -> None:
        self.owner, self.profile, self.binds, self.env = owner, profile, binds, dict(env)
        self.box: Any = None
        self.stage: Optional[Path] = None
        self.queue: Optional[asyncio.Queue] = None
        self.audit_state: Optional[str] = None
        self._ids = itertools.count(1)
        self._send_lock = threading.Lock()

    @property
    def host_pid(self) -> Optional[int]:
        return self.box.pid if self.box is not None else None

    def alive(self) -> bool:
        return self.box is not None and self.box.alive()

    async def start(self, channel: Any, stub_source: Optional[str]) -> None:
        from unify.memory import sandbox_run as _sr

        stage = Path(tempfile.mkdtemp(prefix="box-interpreter-"))
        self.stage = stage
        interp, tools = stage / "interpreter", stage / "tools"
        interp.mkdir()
        tools.mkdir()
        stage_read_only(interp, "worker_child.py", WORKER_CHILD_SOURCE.read_bytes())
        stage_read_only(interp, _AUDIT_NAME, AUDIT_SOURCE.read_bytes())
        if stub_source is not None:
            stage_read_only(tools, f"{TOOLS_MODULE}.py", stub_source.encode("utf-8"))
        env = dict(self.env)
        pass_fds: tuple[int, ...] = ()
        if channel is not None:
            pass_fds = tuple(channel.box_fds())
            env[TOOLS_CHANNEL_ENV] = channel.env_value()
        argv: list[str] = []
        if self.profile.fsize_bytes is not None:
            n = int(self.profile.fsize_bytes)
            argv += [_sr.PRLIMIT, f"--fsize={n}:{n}", "--"]
        argv += [str(_sr.PYTHON), "-I", "-S", "-c", BOOTSTRAP, f"{BOX_INTERPRETER_DIR}/worker_child.py"]
        ro = {**dict(self.binds.ro), interp: BOX_INTERPRETER_DIR, tools: BOX_TOOLS_DIR}
        self.box = await asyncio.to_thread(
            _sr.start_confined,
            argv,
            ro=ro,
            rw=dict(self.binds.rw),
            late_ro=dict(self.binds.late_ro) or None,
            cwd=self.profile.cwd,
            env=env,
            pass_fds=pass_fds,
            on_spawn=channel.after_spawn if channel is not None else None,
        )
        loop = asyncio.get_running_loop()
        self.queue = asyncio.Queue()
        threading.Thread(
            target=self._reader,
            args=(self.box.stdout, loop, self.queue),
            name="box-interpreter-reader",
            daemon=True,
        ).start()
        init = {
            "op": "init",
            "sys_path": [BOX_TOOLS_DIR, *self.profile.sys_path, *box_python_paths()],
            "builtins": sorted(dir(builtins)),
            "globals": {TOOLS_MODULE: ["module", TOOLS_MODULE]} if stub_source is not None else {},
            "help": False,
            "no_bytecode": True,
            "audit": {
                "roots": list(self.profile.audit_roots),
                "path": f"{BOX_INTERPRETER_DIR}/{_AUDIT_NAME}",
            },
        }
        await self.send(init)
        ready = await self._next(START_TIMEOUT_S)
        if not isinstance(ready, dict) or ready.get("op") != "ready":
            raise RuntimeError(
                "the interpreter did not start" + (f": {self.box.stderr_tail(1.0)[-2000:]}" if self.box else ""),
            )
        missing = ready.get("missing") or {}
        if missing:
            raise RuntimeError(f"the interpreter could not import {sorted(missing)}: {missing}")
        self.audit_state = ready.get("audit") if isinstance(ready.get("audit"), str) else None

    @staticmethod
    def _reader(stream: Any, loop: asyncio.AbstractEventLoop, queue: asyncio.Queue) -> None:
        def post(item: Any) -> None:
            try:
                loop.call_soon_threadsafe(queue.put_nowait, item)
            except RuntimeError:  # the loop is closed: nobody is waiting
                pass

        try:
            while True:
                line = stream.readline(MAX_MESSAGE_BYTES + 1)
                if not line:
                    break
                if len(line) > MAX_MESSAGE_BYTES or not line.endswith(b"\n"):
                    post(_Died("the interpreter sent a message larger than allowed"))
                    return
                try:
                    msg = json.loads(line)
                except ValueError:
                    post(_Died("the interpreter sent something that is not JSON"))
                    return
                post(msg if isinstance(msg, dict) else _Died("the interpreter sent something that is not a message"))
        except (OSError, ValueError):
            pass
        post(_Died("the interpreter closed its channel"))

    def _write(self, data: bytes) -> None:
        with self._send_lock:
            stream = self.box.stdin
            stream.write(data)
            stream.flush()

    async def send(self, msg: dict) -> None:
        data = (json.dumps(msg, separators=(",", ":")) + "\n").encode()
        if len(data) > MAX_MESSAGE_BYTES:
            raise ValueError(f"a cell of {len(data)} bytes is larger than the interpreter takes ({MAX_MESSAGE_BYTES})")
        try:
            await asyncio.to_thread(self._write, data)
        except (BrokenPipeError, ConnectionResetError, ValueError, OSError) as exc:
            raise _Died(f"the interpreter does not take input: {type(exc).__name__}") from None

    async def _next(self, timeout: Optional[float]) -> Any:
        assert self.queue is not None
        item = await asyncio.wait_for(self.queue.get(), timeout=timeout)
        if isinstance(item, _Died):
            raise item
        return item

    async def begin(self, cid: int, source: str) -> None:
        await self.send({"op": "exec", "id": cid, "source": source, "sync": {}, "scratch": False})

    async def poll(self, cid: int, timeout: float) -> Optional[dict]:
        """The cell's ``done`` message, or None when nothing ends it within *timeout*; refuses every proxy request
        on the way."""
        try:
            msg = await self._next(timeout)
        except asyncio.TimeoutError:
            return None
        op = msg.get("op")
        if op == "done" and msg.get("id") == cid:
            return msg
        if op in _PROXY_OPS:
            await self.send(
                {
                    "op": "reply",
                    "id": msg.get("id"),
                    "error": {"type": "BoundaryRefusal", "module": "sandbox", "message": _NO_PROXY, "args": None},
                },
            )
        return {}

    async def kill(self) -> None:
        box, self.box = self.box, None
        try:
            if box is not None:
                await asyncio.to_thread(box.kill)
        finally:
            stage, self.stage = self.stage, None
            if stage is not None:
                shutil.rmtree(stage, ignore_errors=True)


class _ActorBackend:
    """The actor's or a teammate's :class:`PythonWorker`, with the tool channel's ends in its sandbox."""

    def __init__(self, owner: "BoxInterpreter", profile: ActorProfile) -> None:
        self.owner, self.profile = owner, profile
        if profile.worker is None:
            from unify.actor.execution.worker import PythonWorker

            profile.worker = PythonWorker()
        self.worker = profile.worker
        self.stage: Optional[Path] = None
        self.task: Optional[asyncio.Task] = None
        self.proc: Any = None
        self.audit_state: Optional[str] = None
        self.stdout: list = []
        self.stderr: list = []

    @property
    def host_pid(self) -> Optional[int]:
        return self.proc.pid if self.proc is not None else None

    def alive(self) -> bool:
        proc = self.proc
        if proc is None:
            return False
        return proc.returncode is None or bool(_tree(proc.pid))

    async def start(self, channel: Any, stub_source: Optional[str]) -> None:
        w = self.worker
        w.box_fds = tuple(channel.box_fds()) if channel is not None else ()
        w.box_env = {TOOLS_CHANNEL_ENV: channel.env_value()} if channel is not None else {}
        w.on_spawn = channel.after_spawn if channel is not None else None
        w._restart_note = None  # this interpreter tells the model about restarts itself
        await w._start()
        self.proc = w._proc
        if stub_source is not None:
            # the actor's sandbox admits no extra read-only bind (sandbox.wrap_argv), so the module is built from
            # its source in the box; a cell can change it either way, and the host trusts nothing it sends
            await w.run_cell(
                tools_bootstrap(stub_source),
                self.profile.shadow,
                timeout=START_TIMEOUT_S,
                stdout=[],
                stderr=[],
            )

    async def begin(self, cid: int, source: str) -> None:
        self.stdout, self.stderr = [], []
        self.task = asyncio.create_task(
            self.worker.run_cell(
                source,
                self.profile.shadow,
                timeout=None,
                stdout=self.stdout,
                stderr=self.stderr,
            ),
        )

    async def poll(self, cid: int, timeout: float) -> Optional[dict]:
        assert self.task is not None
        done, _ = await asyncio.wait({self.task}, timeout=timeout)
        if not done:
            return None
        from unify.actor.execution.worker import WorkerCellError

        try:
            value = self.task.result()
        except WorkerCellError as exc:
            if "exited during the cell" in exc.traceback:
                raise _Died(exc.traceback) from None
            return {"op": "done", "error": exc.traceback, "value": None}
        except asyncio.CancelledError:
            raise _Died("the cell was cancelled") from None
        except Exception as exc:  # noqa: BLE001 - a reply or a steering interruption: re-raised after the checks
            return {"op": "done", "error": None, "value": None, "raise": exc}
        except BaseException as exc:  # a reply from a cell is a BaseException (worker_child.CellReply)
            return {"op": "done", "error": None, "value": None, "raise": exc}
        return {"op": "done", "error": None, "value": value}

    async def kill(self) -> None:
        proc, self.proc = self.proc, None
        task, self.task = self.task, None
        try:
            if proc is not None:
                await kill_verified(proc.pid, lambda: proc.returncode is not None, proc.wait)
        finally:
            if task is not None:
                if not task.done():
                    task.cancel()
                await asyncio.gather(task, return_exceptions=True)  # its outcome is the kill's, already handled
            await self.worker.close()
            stage, self.stage = self.stage, None
            if stage is not None:
                shutil.rmtree(stage, ignore_errors=True)


@functools.lru_cache(maxsize=None)
def box_python_paths() -> tuple[str, ...]:
    """The box interpreter's own import path (its standard library and site-packages), as ``python -I`` sets it;
    only entries inside the interpreter roots the box mounts (nothing else exists there)."""
    from unify.memory import sandbox_run as _sr

    out = subprocess.run(
        [str(_sr.PYTHON), "-I", "-c", "import json, sys; print(json.dumps(sys.path))"],
        capture_output=True,
        text=True,
        timeout=60,
        env={"PATH": "/usr/bin:/bin"},
        check=True,
    ).stdout
    roots = [Path(os.path.abspath(r)) for r in _sr._python_roots()]
    paths: list[str] = []
    for p in json.loads(out):
        if isinstance(p, str) and os.path.isabs(p) and any(_sr._within(Path(p), r) for r in roots):
            paths.append(p)
    return tuple(paths)


# --- the interpreter ----------------------------------------------------------------------------------------------


class BoxInterpreter:
    """One agent's persistent interpreter. :meth:`start` it, :meth:`run_cell` each cell, :meth:`stop` it (verified
    kill). *channel* is a :class:`~unify.common.tool_channel.Channel` for the first box, or a callable returning a new
    one for each box (every restart needs a new channel, since the host closes its copies of the box ends after the
    spawn); None: no tools.
    """

    def __init__(
        self,
        *,
        agent: str,
        profile: Any,
        binds: Any,
        env: dict,
        channel: Any,
        stub_source: Optional[str],
        cell_limit_s: float,
    ) -> None:
        self.agent = agent
        self.profile = profile
        self.binds = binds if isinstance(binds, Binds) else Binds(**dict(binds or {}))
        self.env = dict(env or {})
        self.stub_source = stub_source
        self.cell_limit_s = float(cell_limit_s)
        self._channel_source = channel
        self._channel_used = False
        self.channel: Any = None
        self._serve: Optional[asyncio.Task] = None
        self._backend: Any = None
        self._baseline: frozenset = frozenset()
        self._worker_pid: Optional[int] = None
        self._baseline_threads: Optional[int] = None
        self._cells = itertools.count(1)
        self._lock = asyncio.Lock()
        self._kill_lock = asyncio.Lock()
        self._pending: list[str] = []
        self._last_used = time.monotonic()
        self._in_cell = False
        self._idle: Optional[asyncio.Task] = None
        self._quota_breach: Optional[str] = None
        self._stopped = False
        self._started = False
        #: boxes started after the first one
        self.restarts = 0

    # -- lifecycle -------------------------------------------------------------------------------------------------
    @classmethod
    async def start(
        cls,
        *,
        agent: str,
        profile: Any,
        binds: Any = None,
        env: Optional[dict] = None,
        channel: Any = None,
        stub_source: Optional[str] = None,
        cell_limit_s: float,
    ) -> "BoxInterpreter":
        self = cls(
            agent=agent,
            profile=profile,
            binds=binds,
            env=env or {},
            channel=channel,
            stub_source=stub_source,
            cell_limit_s=cell_limit_s,
        )
        await self._start_box()
        idle = getattr(profile, "idle_restart_s", None)
        if idle:
            self._idle = asyncio.create_task(self._idle_watch(float(idle)))
        return self

    @property
    def memory(self) -> bool:
        return isinstance(self.profile, MemoryProfile)

    def _next_channel(self) -> Any:
        src = self._channel_source
        if src is None:
            return None
        if callable(src) and not hasattr(src, "box_fds"):
            return src()
        if self._channel_used:
            raise RuntimeError("a restarted interpreter needs a new tool channel: pass a callable that opens one")
        self._channel_used = True
        return src

    async def _start_box(self) -> None:
        channel = self._next_channel()
        backend = (
            _MemoryBackend(self, self.profile, self.binds, self.env)
            if self.memory
            else _ActorBackend(self, self.profile)
        )
        self.channel, self._backend = channel, backend
        if self._started:
            self.restarts += 1
        self._started = True
        try:
            await backend.start(channel, self.stub_source)
        except BaseException:
            await self._kill_box("start failed")
            raise
        if channel is not None:
            self._serve = asyncio.create_task(channel.serve())
        pid = backend.host_pid
        self._baseline = _tree(pid) if pid is not None else frozenset()
        self._worker_pid = _worker_pid(pid, self._baseline) if pid is not None else None
        self._baseline_threads = _threads(self._worker_pid)
        self._last_used = time.monotonic()

    async def _kill_box(self, reason: str) -> None:
        """The verified kill of the current box and the close of its channel; the next cell starts a new box."""
        async with self._kill_lock:
            backend, self._backend = self._backend, None
            channel, self.channel = self.channel, None
            serve, self._serve = self._serve, None
            try:
                if backend is not None:
                    await backend.kill()
            finally:
                if channel is not None:
                    try:
                        await channel.close(reason)
                    except Exception:  # noqa: BLE001 - the box is gone either way
                        logger.debug("closing the tool channel failed", exc_info=True)
                # a handler of the channel itself may ask for the kill (a quota breach): its own task is left to end
                if serve is not None and serve is not asyncio.current_task():
                    if not serve.done():
                        serve.cancel()
                    await asyncio.gather(serve, return_exceptions=True)

    async def restart(self, reason: str) -> None:
        """Kill the box (verified) and start a fresh one; the next cell is told its state is gone."""
        async with self._lock:
            await self._kill_box(reason)
            self._pending.append(f"(the interpreter was restarted ({reason}): {_LOST})")
            await self._start_box()

    async def stop(self) -> None:
        """Kill the box (verified; raises if a process survives) and stop the idle watch. Safe to repeat."""
        self._stopped = True
        idle, self._idle = self._idle, None
        if idle is not None and idle is not asyncio.current_task():
            idle.cancel()
            await asyncio.gather(idle, return_exceptions=True)
        await self._kill_box("stopped")

    def box_alive(self) -> bool:
        """Whether a box of this interpreter (bubblewrap or any process below it) is alive: while it is, a memory
        agent's box can still write ``/memory``."""
        backend = self._backend
        return backend is not None and backend.alive()

    async def measure_quota(self) -> Optional[str]:
        """Measure the ``/memory`` quota now (after a ``write``, ``edit`` or ``fixture`` call as after every cell);
        a breach kills the box, and the running cell's result reports it. Returns the breach text or None."""
        measure = getattr(self.profile, "measure", None)
        if measure is None:
            return None
        breach = await asyncio.to_thread(measure)
        if breach is not None:
            self._quota_breach = breach
            await self._kill_box("quota")
        return breach

    async def _idle_watch(self, idle_s: float) -> None:
        step = max(0.05, min(idle_s / 4, 30.0))
        while not self._stopped:
            await asyncio.sleep(step)
            if self._in_cell or self._backend is None or self._lock.locked():
                continue
            if time.monotonic() - self._last_used < idle_s:
                continue
            async with self._lock:  # no cell starts while the idle box is stopped
                if self._backend is None or time.monotonic() - self._last_used < idle_s:
                    continue
                try:
                    await self._kill_box("idle")
                except Exception:  # noqa: BLE001 - logged here; the next cell's start fails if it cannot replace it
                    logger.exception("stopping an idle interpreter failed")
                    raise
                self._pending.append(
                    f"(the interpreter was idle for more than {idle_s:g} s and was restarted: {_LOST})",
                )

    # -- checks between cells ----------------------------------------------------------------------------------------
    def _channel_closed(self) -> Optional[str]:
        ch = self.channel
        if ch is None:
            return None
        if getattr(ch, "closed", False) or (self._serve is not None and self._serve.done()):
            reason = getattr(ch, "close_reason", None)
            return str(reason) if reason else "closed"
        return None

    async def _left_running(self) -> Optional[str]:
        """What a cell left running in the box (a process or a worker thread), after a short settle; None when the
        box is as it was when the interpreter started."""
        backend = self._backend
        pid = backend.host_pid if backend is not None else None
        if pid is None:
            return None
        found = "a process"
        deadline = time.monotonic() + SETTLE_S
        while True:
            tree = _tree(pid)
            extra = tree - self._baseline
            threads = _threads(self._worker_pid)
            if not extra and (
                self._baseline_threads is None or threads is None or threads <= self._baseline_threads
            ):
                return None
            found = "a process" if extra else "a thread"
            if time.monotonic() >= deadline:
                return found
            await asyncio.sleep(_SETTLE_STEP_S)

    def _busy(self) -> bool:
        """A tool call of this cell is in flight on the host: the cell's clock does not run meanwhile."""
        ch = self.channel
        return bool(ch is not None and getattr(ch, "in_flight", False))

    # -- cells -------------------------------------------------------------------------------------------------------
    async def run_cell(self, code: str, *, call: Optional[int] = None) -> CellResult:
        """Run *code* at the interpreter's top level and return its :class:`CellResult`. *call*, when given, is the
        tool event's call number, added to the cell's ``box-opens.jsonl`` rows."""
        async with self._lock:
            self._in_cell = True
            try:
                return await self._run(code, call)
            finally:
                self._in_cell = False
                self._last_used = time.monotonic()

    async def _run(self, code: str, call: Optional[int]) -> CellResult:
        if self._stopped:
            raise RuntimeError("the interpreter was stopped")
        before: list[str] = []
        if self._backend is not None:
            closed = self._channel_closed()
            if closed is not None:
                await self._kill_box(f"channel {closed}")
                before.append(_restart_note(f"tool channel closed: {closed}"))
            elif not self._backend.alive():
                await self._kill_box("exited")
                before.append(_restart_note("the interpreter had exited"))
        if self._pending:
            before = [*self._pending, *before]
            self._pending = []
        if self._backend is None:
            await self._start_box()
        restarted = bool(before)
        cell = next(self._cells)
        self._quota_breach = None
        backend = self._backend
        source = wrap_cell(code, f"<cell {cell}>", last_value=bool(getattr(self.profile, "last_value", False)))
        res = CellResult(output="", cell=cell, restarted=restarted, notes_before=before)
        if self.channel is not None:
            self.channel.cell_started(cell)
        died: Optional[str] = None
        done: Optional[dict] = None
        try:
            await backend.begin(cell, source)
            used, last = 0.0, time.monotonic()
            while True:
                now = time.monotonic()
                if not self._busy():
                    used += now - last
                last = now
                left = self.cell_limit_s - used
                if left <= 0:
                    break
                got = await backend.poll(cell, min(_CLOCK_TICK_S, left))
                if got:
                    done = got
                    break
        except _Died as exc:
            died = str(exc)
        except asyncio.CancelledError:
            await self._kill_box("cancelled")
            self._pending.append(_restart_note("the previous cell was stopped"))
            raise
        finally:
            if self.channel is not None:
                try:
                    self.channel.cell_ended(cell)
                except Exception:  # noqa: BLE001 - the checks below close a broken channel
                    logger.debug("cell_ended failed", exc_info=True)
        if done is None and died is None:
            res.timed_out = True
            await self._kill_box("timeout")
            res.restarted = True
            res.notes_after.append(_timed_out_note(self.cell_limit_s))
            res.opens = self._opens(None, cell, call, dead_reason="killed") if self.memory else None
            return await self._after(res, measure=True)
        if died is not None:
            breach = self._quota_breach
            await self._kill_box("exited")
            res.restarted = True
            if breach is not None:
                res.quota = breach
                res.notes_after.append(_restart_note(breach))
            else:
                res.notes_after.append(_restart_note("the interpreter exited during the cell"))
            res.opens = self._opens(None, cell, call, dead_reason="killed") if self.memory else None
            return await self._after(res, measure=breach is None)
        self._fill(res, done)
        if self.memory:
            res.opens = self._opens(done.get("audit"), cell, call)
        closed = self._channel_closed()
        if closed is not None:
            await self._kill_box(f"channel {closed}")
            res.restarted = True
            res.notes_after.append(_restart_note(f"tool channel closed: {closed}"))
        else:
            left = await self._left_running()
            if left is not None:
                await self._kill_box("left running")
                res.restarted = True
                res.notes_after.append(
                    _restart_note(f"{left} this cell started was still running after it ended"),
                )
        res = await self._after(res, measure=True)
        raised = done.get("raise")
        if raised is not None:
            # the cell ended by a control exception (a reply, a steering interruption): the caller handles it, and
            # what the checks found is told with the next cell
            self._pending.extend(res.notes_after)
            raise raised
        return res

    async def _after(self, res: CellResult, *, measure: bool) -> CellResult:
        if measure and self.memory and res.quota is None:
            breach = await self.measure_quota()
            if breach is not None:
                res.quota = breach
                res.restarted = True
                res.notes_after.append(_restart_note(breach))
        return res

    def _fill(self, res: CellResult, done: dict) -> None:
        error = done.get("error")
        if isinstance(error, str) and error:
            res.error = _WRAPPER_FRAME.sub("", error).rstrip("\n")
        if self.memory:
            out = _parts_text(done.get("stdout"))
            err = _parts_text(done.get("stderr"))
            out, err = _tail(out), _tail(err)
            res.output = out + ("\n" + err if err else "")
        else:
            from unify.actor.execution.types import parts_to_text

            backend = self._backend
            res.parts = list(backend.stdout) if backend is not None else []
            out = parts_to_text(res.parts) if res.parts else ""
            err = parts_to_text(backend.stderr) if backend is not None and backend.stderr else ""
            res.output = out + ("\n" + err if err else "")
            res.result = done.get("value")

    def _opens(self, audit: Any, cell: int, call: Optional[int], dead_reason: Optional[str] = None) -> dict:
        profile: MemoryProfile = self.profile
        if dead_reason is not None:
            log = {"rows": [], "dropped": 0, "dead": True, "reason": dead_reason}
        else:
            state = self._backend.audit_state if self._backend is not None else None
            log = open_log(audit, state, tuple(profile.open_areas))
        if log["rows"] and profile.opens_path is not None:
            from unify.memory import sandbox_run as _sr

            stamped = [
                {
                    "agent": self.agent,
                    "pass_id": profile.pass_id,
                    "cell": cell,
                    **({"call": call} if call is not None else {}),
                    **row,
                }
                for row in log["rows"]
            ]
            _sr.append_open_rows(Path(profile.opens_path), stamped)
        return log


def _parts_text(parts: Any) -> str:
    out: list[str] = []
    for part in parts if isinstance(parts, list) else []:
        if not isinstance(part, dict):
            continue
        if part.get("type") == "text":
            out.append(str(part.get("text") or ""))
        elif part.get("type") == "image":
            out.append("[an image was displayed; images are not shown here]\n")
    return "".join(out)


def _tail(text: str) -> str:
    """The last :data:`unify.memory.sandbox_run.CAPTURE_MAX_BYTES` of *text*, marked as a fresh cell's capture is."""
    from unify.memory.sandbox_run import CAPTURE_MAX_BYTES

    data = text.encode("utf-8", errors="replace")
    if len(data) <= CAPTURE_MAX_BYTES:
        return text
    dropped = len(data) - CAPTURE_MAX_BYTES
    return f"[{dropped} earlier bytes dropped]\n" + data[-CAPTURE_MAX_BYTES:].decode("utf-8", errors="replace")
