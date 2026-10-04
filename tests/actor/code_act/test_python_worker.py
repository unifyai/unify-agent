"""Symbolic: ``UNIFY_WORKSPACE_PYTHON=worker`` runs Python cells in a sandboxed child.

Nothing here reaches a model. Cells run through the real ``SessionExecutor``,
the worker is a real child process inside the real bubblewrap policy, and the
harness objects it calls are fakes that record where they ran. Tests that need
bubblewrap are skipped, saying so, where it is missing. The world is described
in ``sandbox_world.py``.
"""

from __future__ import annotations

import asyncio
import os
import sqlite3

import pytest
from pydantic import BaseModel

from tests.helpers import _handle_project
from tests.actor.code_act.sandbox_world import (  # noqa: F401 (fixture)
    STATE_SECRET,
    TOKEN_VALUE,
    needs_bwrap,
    serve,
    world,
)
from unify.actor.execution.session import SessionExecutor
from unify.actor.execution.types import parts_to_text
from unify.actor.execution import worker as worker_mod
from unify.settings import ProductionSettings, SETTINGS


def process_id() -> str:
    """This process: its pid and the pid namespace it is counted in.

    A pid alone does not tell two processes apart across namespaces: under
    the test sandbox (tests/_test_sandbox.py) the harness and a worker in its
    own sandbox are both pid 2.
    """
    return f"{os.getpid()}@{os.readlink('/proc/self/ns/pid')}"


# The same, as an expression for cell code.
PROCESS_ID = "f\"{os.getpid()}@{os.readlink('/proc/self/ns/pid')}\""
HARNESS_PID = process_id()
# Cell code sees the restricted builtins, which leave out ``globals``.
HAS_X = "try:\n    x\n    found = True\nexcept NameError:\n    found = False\nfound"


# ── fakes ────────────────────────────────────────────────────────────────────


class Item(BaseModel):
    name: str
    size: int


class Opaque:
    """A harness object that is not data: it crosses by reference."""

    def __init__(self, label: str) -> None:
        self.label = label

    async def result(self) -> str:
        return f"finished {self.label} in {process_id()}"

    def __repr__(self) -> str:
        return f"Opaque({self.label!r})"


class HarnessOnlyError(Exception):
    pass


class FakeFiles:
    api_token = "tok-attr-never-sent"  # pragma: allowlist secret
    root = "/data"
    tools = os  # a module among the public attributes

    def __init__(self, calls: list) -> None:
        self.calls = calls
        self._private = "hidden"

    async def search(self, query, limit=3):
        self.calls.append(("search", query, limit, process_id()))
        return {"query": query, "hits": [f"{query}-{i}" for i in range(limit)]}

    def count(self, words):
        self.calls.append(("count", tuple(words), process_id()))
        return len(words)

    async def item(self) -> Item:
        return Item(name="a", size=3)

    async def describe(self, value):
        return f"{type(value).__name__}:{getattr(value, 'name', value)}"

    async def handle(self, label: str) -> Opaque:
        return Opaque(label)

    async def missing(self, key: str):
        return {}[key]

    async def broken(self):
        raise HarnessOnlyError("only the harness knows this type")

    async def environment(self):
        return os.environ


class FakePrimitives:
    def __init__(self, calls: list) -> None:
        self.files = FakeFiles(calls)


class FakeEnvironment:
    def __init__(self, calls: list) -> None:
        self._instance = FakePrimitives(calls)

    def get_instance(self):
        return self._instance


def executor_with_fakes(timeout=None):
    calls: list = []
    ex = SessionExecutor(
        environments={"primitives": FakeEnvironment(calls)},
        timeout=timeout,
    )
    return ex, calls


@pytest.fixture
def worker_world(world, monkeypatch):
    monkeypatch.setattr(SETTINGS, "UNIFY_WORKSPACE_PYTHON", "worker")
    return world


async def run(ex, code, *, mode="stateful", session_id=0):
    res = await ex.execute(
        code=code,
        state_mode=mode,
        session_id=session_id if mode != "stateless" else None,
    )
    return parts_to_text(res["stdout"]), res


# ── settings and the switch off ─────────────────────────────────────────────


def test_setting_defaults_off_and_rejects_unknown_values():
    assert ProductionSettings().UNIFY_WORKSPACE_PYTHON == ""
    assert (
        ProductionSettings(UNIFY_WORKSPACE_PYTHON="Worker").UNIFY_WORKSPACE_PYTHON
        == "worker"
    )
    with pytest.raises(ValueError):
        ProductionSettings(UNIFY_WORKSPACE_PYTHON="subprocess")


@pytest.mark.asyncio
@pytest.mark.parametrize("workspace", ["", "sandboxed"])
async def test_without_the_worker_switch_cells_run_in_process(
    workspace,
    monkeypatch,
    world,
):
    # UNIFY_WORKSPACE unset, or sandboxed with the sub-switch unset: the
    # cell is exec'd in this process exactly as before.
    monkeypatch.setattr(SETTINGS, "UNIFY_WORKSPACE", workspace)
    monkeypatch.setattr(SETTINGS, "UNIFY_WORKSPACE_PYTHON", "")
    # The sub-switch alone does nothing either.
    if not workspace:
        monkeypatch.setattr(SETTINGS, "UNIFY_WORKSPACE_PYTHON", "worker")
    ex, calls = executor_with_fakes()
    try:
        out, res = await run(
            ex,
            "import os\nx = 41\nr = await primitives.files.search('q', limit=1)\n"
            f"print({PROCESS_ID}, r['hits'])\ntype(primitives.files).__name__",
        )
        assert res["error"] is None
        assert out.split(maxsplit=1) == [str(HARNESS_PID), "['q-0']\n"]
        assert res["result"] == "FakeFiles"
        out, res = await run(ex, "x + 1")
        assert res["result"] == 42
        assert ex.python_session(session_id=0)._worker is None
        assert calls == [("search", "q", 1, HARNESS_PID)]
    finally:
        await ex.close()


# ── the worker ───────────────────────────────────────────────────────────────


@needs_bwrap
@pytest.mark.asyncio
async def test_state_persists_across_cells_in_one_worker_per_session(worker_world):
    ex, _ = executor_with_fakes()
    try:
        out, res = await run(
            ex,
            "import os, math\nx = 41\ndef f(y):\n    return y * 2\n" + PROCESS_ID,
        )
        assert res["error"] is None, res["error"]
        pid = res["result"]
        assert isinstance(pid, str) and pid != HARNESS_PID
        out, res = await run(ex, "print(f(x), math.floor(2.5))\n" + PROCESS_ID)
        assert out == "82 2\n" and res["result"] == pid
        # Another session and a stateless cell have workers of their own.
        _, res = await run(ex, HAS_X, session_id=1)
        assert res["result"] is False
        _, res = await run(ex, HAS_X, mode="stateless")
        assert res["result"] is False
        # read_only sees the session's state, and what it changes is dropped.
        res = await ex.execute(
            code="x = x - 41\nx",
            state_mode="read_only",
            session_id=0,
        )
        assert res["result"] == 0, res["error"]
        _, res = await run(ex, "x")
        assert res["result"] == 41
    finally:
        await ex.close()


@needs_bwrap
@pytest.mark.asyncio
async def test_a_cell_cannot_read_secrets_write_the_store_or_reach_the_network(
    worker_world,
):
    srv, port = serve(b"hello-from-host\n")
    store = worker_world["state"] / "store.sqlite"
    ex, _ = executor_with_fakes()
    try:
        out, res = await run(
            ex,
            "import os\n"
            "print(os.environ.get('FAKE_SERVICE_TOKEN'), "
            "os.environ.get('DB_PASSWORD'), os.environ.get('UNIFY_SANDBOX_PROBE'))",
        )
        assert out == "None None visible\n", res["error"]
        # The store reads but does not write.
        out, res = await run(
            ex,
            "import sqlite3\n"
            f"con = sqlite3.connect({str(store)!r})\n"
            "print(con.execute('select name from functions').fetchall())\n"
            "try:\n"
            "    con.execute(\"insert into functions values ('evil')\")\n"
            "    con.commit()\n"
            "    print('wrote')\n"
            "except sqlite3.OperationalError as e:\n"
            "    print('refused:', e)",
        )
        assert out.startswith("[('f',)]\nrefused:"), (out, res["error"])
        con = sqlite3.connect(store)
        assert con.execute("select name from functions").fetchall() == [("f",)]
        con.close()
        # The rest of the state directory is hidden.
        out, _ = await run(
            ex,
            f"open({str(worker_world['state'] / 'logs' / 'unify.log')!r}).read()",
        )
        assert STATE_SECRET not in out
        _, res = await run(
            ex,
            f"open({str(worker_world['state'] / 'logs' / 'unify.log')!r}).read()",
        )
        assert STATE_SECRET not in str(res["result"]) and STATE_SECRET not in str(
            res["error"],
        )
        # No socket reaches the host, not even its loopback.
        out, res = await run(
            ex,
            "import socket\n"
            "try:\n"
            f"    socket.create_connection(('127.0.0.1', {port}), timeout=3)\n"
            "    print('connected')\n"
            "except OSError as e:\n"
            "    print('refused:', type(e).__name__)",
        )
        assert out.startswith("refused:"), (out, res["error"])
    finally:
        srv.close()
        await ex.close()


@needs_bwrap
@pytest.mark.asyncio
async def test_proxy_mode_gives_the_worker_only_the_proxy_port(
    worker_world,
    monkeypatch,
):
    proxy, proxy_port = serve(b"from proxy\n")
    other, other_port = serve(b"other service\n")
    monkeypatch.setattr(SETTINGS, "UNIFY_WORKSPACE_NETWORK", "proxy")
    monkeypatch.setattr(SETTINGS, "UNIFY_WORKSPACE_PROXY_PORT", proxy_port)
    connect = (
        "import socket\n"
        "try:\n"
        "    s = socket.create_connection(('127.0.0.1', {port}), timeout=5)\n"
        "    print('got', s.recv(64).decode().strip())\n"
        "except OSError as e:\n"
        "    print('refused:', type(e).__name__)"
    )
    ex, _ = executor_with_fakes()
    try:
        out, res = await run(ex, connect.format(port=proxy_port))
        assert out == "got from proxy\n", res["error"]
        out, _ = await run(ex, connect.format(port=other_port))
        assert out.startswith("refused:")
        # The channel to the harness still works through the forwarder.
        _, res = await run(ex, "(await primitives.files.search('p', limit=1))['hits']")
        assert res["result"] == ["p-0"]
    finally:
        proxy.close()
        other.close()
        await ex.close()


@needs_bwrap
@pytest.mark.asyncio
async def test_primitives_run_in_the_harness_through_the_proxy(worker_world):
    ex, calls = executor_with_fakes()
    try:
        out, res = await run(
            ex,
            "import os\n"
            "r = await primitives.files.search('cat', limit=2)\n"
            "n = primitives.files.count(['a', 'b', 'c'])\n"
            "both = await asyncio.gather(primitives.files.search('x', limit=1), "
            "primitives.files.search('y', limit=1))\n"
            "print(r['hits'], n, [b['hits'] for b in both], primitives.files.root)\n"
            + PROCESS_ID,
        )
        assert res["error"] is None, res["error"]
        worker_pid = res["result"]
        assert out == "['cat-0', 'cat-1'] 3 [['x-0'], ['y-0']] /data\n"
        # Every call ran in the harness, not the worker.
        assert {c[-1] for c in calls} == {HARNESS_PID} and worker_pid != HARNESS_PID
        assert calls[0] == ("search", "cat", 2, HARNESS_PID)
        # A model comes by value and goes back as the original; an object
        # that is not data comes as a reference that calls back.
        out, res = await run(
            ex,
            "it = await primitives.files.item()\n"
            "print(it.name, it['size'], repr(it))\n"
            "print(await primitives.files.describe(it))\n"
            "h = await primitives.files.handle('job')\n"
            "print(repr(h), await h.result())\n"
            "print(sorted(n for n in dir(primitives.files) if not n.startswith('_'))[:3])\n"
            "h",
        )
        assert res["error"] is None, res["error"]
        lines = out.splitlines()
        assert lines[0] == "a 3 Item(name='a', size=3)"
        assert lines[1] == "Item:a"
        assert lines[2] == f"Opaque('job') finished job in {HARNESS_PID}"
        assert lines[3] == "['api_token', 'broken', 'calls']"
        # The handle returned as the result is the harness's own object.
        assert isinstance(res["result"], Opaque) and res["result"].label == "job"
    finally:
        await ex.close()


@needs_bwrap
@pytest.mark.asyncio
async def test_harness_exceptions_are_reraised_in_the_worker(worker_world):
    ex, _ = executor_with_fakes()
    try:
        out, res = await run(
            ex,
            "try:\n"
            "    await primitives.files.missing('k')\n"
            "except KeyError as e:\n"
            "    print('KeyError', e)\n"
            "try:\n"
            "    await primitives.files.broken()\n"
            "except Exception as e:\n"
            "    print(type(e).__name__, type(e).__module__, e)",
        )
        assert res["error"] is None, res["error"]
        assert out.splitlines() == [
            "KeyError 'k'",
            f"HarnessOnlyError {__name__} only the harness knows this type",
        ]
        # Uncaught, it is the cell's error, named.
        _, res = await run(ex, "await primitives.files.broken()")
        assert "HarnessOnlyError: only the harness knows this type" in res["error"]
        # A cell's own exception is reported as a traceback, as in process.
        _, res = await run(ex, "1 / 0")
        assert "ZeroDivisionError" in res["error"]
    finally:
        await ex.close()


@needs_bwrap
@pytest.mark.asyncio
async def test_what_cannot_cross_is_refused_by_name(worker_world):
    ex, calls = executor_with_fakes()
    try:
        _, res = await run(ex, "await primitives.files.search(lambda: 1)")
        assert "BoundaryRefusal" in res["error"]
        assert "argument 1 of primitives.files.search is a function" in res["error"]
        _, res = await run(
            ex,
            "class Mine:\n    pass\nawait primitives.files.search('q', limit=Mine())",
        )
        assert "argument 'limit' of primitives.files.search is a Mine" in (res["error"])
        assert calls == []  # refused before anything was sent
        _, res = await run(ex, "primitives.files._private")
        assert "attributes starting with '_'" in res["error"]
        _, res = await run(ex, "primitives.files.api_token")
        assert "looks like a credential" in res["error"]
        assert "tok-attr-never-sent" not in res["error"]
        # Modules and the harness's environment never cross, however reached.
        _, res = await run(
            ex,
            "primitives.files.tools.environ.get('FAKE_SERVICE_TOKEN')",
        )
        assert "primitives.files.tools is a harness module (os)" in res["error"]
        _, res = await run(
            ex,
            "(await primitives.files.environment()).get('DB_PASSWORD')",
        )
        assert "harness's environment" in res["error"]
        assert TOKEN_VALUE not in res["error"]
        # A worker object returned as the result comes back as its repr.
        _, res = await run(
            ex,
            "class Box:\n    def __repr__(self): return 'Box!'\nBox()",
        )
        assert res["error"] is None and repr(res["result"]) == "Box!"
        assert isinstance(res["result"], worker_mod.WorkerValue)
    finally:
        await ex.close()


@needs_bwrap
@pytest.mark.asyncio
async def test_stored_functions_run_in_the_worker_and_call_back_for_primitives(
    worker_world,
):
    from unify.function_manager.function_manager import _LineageTrackedFunction
    from unify.function_manager.source_labels import compile_function_source

    source = (
        "async def tally(word):\n"
        "    import os\n"
        "    r = await primitives.files.search(word, limit=2)\n"
        "    return {'hits': r['hits'], 'ran_in': " + PROCESS_ID + "}\n"
    )
    ns: dict = {}
    exec(compile_function_source("tally", source), ns)
    used: list = []
    stored = _LineageTrackedFunction(
        ns["tally"],
        "tally",
        on_call=lambda: used.append(1),
    )
    ex, calls = executor_with_fakes()
    ex.register_fm_globals({"tally": stored})
    try:
        out, res = await run(
            ex,
            f"import os\nr = await tally('dog')\n(r, {PROCESS_ID})",
        )
        assert res["error"] is None, res["error"]
        value, worker_pid = res["result"]
        assert value["hits"] == ["dog-0", "dog-1"]
        # The body ran in the worker; only its primitive call came back.
        assert value["ran_in"] == worker_pid != HARNESS_PID
        assert calls == [("search", "dog", 2, HARNESS_PID)]
        assert used == [1]
    finally:
        await ex.close()


@needs_bwrap
@pytest.mark.asyncio
async def test_a_timeout_kills_the_worker_and_the_next_cell_starts_fresh(
    worker_world,
):
    ex, _ = executor_with_fakes(timeout=3)
    try:
        _, res = await run(ex, "x = 1")
        tree = worker_tree(ex)
        assert len(tree) >= 2  # bwrap and the worker inside it
        _, res = await run(ex, "while True:\n    pass")
        assert "timed out after 3" in res["error"] and "killed" in res["error"]
        assert await all_gone(tree)
        out, res = await run(ex, HAS_X)
        assert res["result"] is False
        assert "fresh Python worker" in out and "timed out" in out
    finally:
        await ex.close()


@needs_bwrap
@pytest.mark.asyncio
async def test_a_stopped_cell_kills_the_worker(worker_world):
    ex, _ = executor_with_fakes()
    try:
        await run(ex, "x = 1")
        pid = worker_pid(ex)
        tree = worker_tree(ex)
        task = asyncio.create_task(run(ex, "await asyncio.sleep(60)"))
        await asyncio.sleep(1.0)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert await all_gone(tree)
        out, res = await run(ex, HAS_X)
        assert res["result"] is False and "was stopped" in out
        assert worker_pid(ex) != pid
        # A worker that exits by itself says so too.
        _, res = await run(ex, "import os\nos._exit(3)")
        assert "exited during the cell" in res["error"]
        _, res = await run(ex, "1")
        assert res["result"] == 1
    finally:
        await ex.close()


@needs_bwrap
@pytest.mark.asyncio
async def test_output_display_and_clarification_cross(worker_world):
    async def request_clarification(question: str) -> str:
        return f"answer to {question}"

    ex, _ = executor_with_fakes()
    ex.register_fm_globals({"request_clarification": request_clarification})
    try:
        out, res = await run(
            ex,
            "import subprocess, sys\n"
            "print('printed')\n"
            "sys.stdout.write('written\\n')\n"
            "display({'a': 1})\n"
            "print(await request_clarification('why?'))\n"
            "subprocess.run(['echo', 'from-subprocess'])\n"
            "print('err', file=sys.stderr)",
        )
        assert res["error"] is None, res["error"]
        assert out == ("printed\nwritten\n{'a': 1}\nanswer to why?\nfrom-subprocess\n")
        assert parts_to_text(res["stderr"]) == "err\n"
    finally:
        await ex.close()


@needs_bwrap
@pytest.mark.asyncio
async def test_steering_probes_and_memoisation_reach_the_worker(worker_world):
    from unify.function_manager.steering import (
        InterruptionRequest,
        Patch,
        SteeringSession,
        use_session,
    )

    ex, calls = executor_with_fakes()
    try:
        steering = SteeringSession()
        with use_session(steering):
            _, res = await run(
                ex,
                "out = []\n"
                "for w in ['a', 'a']:\n"
                "    out.append((await primitives.files.search(w, limit=1))['hits'])\n"
                "out",
            )
        assert res["error"] is None, res["error"]
        assert res["result"] == [["a-0"], ["a-0"]]
        assert steering.runtime.action_counter > 0
        assert steering.cache.misses == 2
        # A stop request fires at the next checkpoint in the worker.
        stopping = SteeringSession()
        stopping.interruption = InterruptionRequest(reason="enough", stop=True)
        with use_session(stopping):
            _, res = await run(ex, "await primitives.files.search('z')")
        assert res["result"] == {"status": "stopped", "reason": "enough"}
        assert not any(c[1] == "z" for c in calls)
        # A correction raised by a probe in the worker unwinds the cell there,
        # is spliced here, and the patched cell runs again in the worker.
        patching = SteeringSession()
        patching.interruption = InterruptionRequest(
            reason="use the new step",
            patches=[Patch("step", "async def step():\n    return 'patched'\n")],
        )
        with use_session(patching):
            _, res = await run(
                ex,
                "async def step():\n    return 'original'\nawait step()",
            )
        assert res["error"] is None, res["error"]
        assert res["result"] == "patched" and patching.retries == 1
    finally:
        await ex.close()


@needs_bwrap
@pytest.mark.asyncio
@_handle_project
async def test_inspect_state_reads_the_workers_variables(worker_world):
    from unify.actor.code_act_actor import CodeActActor
    from unify.actor.execution import _CURRENT_SANDBOX, PythonExecutionSession

    actor = CodeActActor(environments=[])
    tools = actor.get_tools("act")
    sandbox_session = PythonExecutionSession(environments={})
    token = _CURRENT_SANDBOX.set(sandbox_session)
    try:
        # Before any cell there is no worker and nothing to list.
        empty = await tools["inspect_state"]()
        assert empty["state"]["variables"] == []
        seeded = await tools["execute_code"](
            thought="Seed the session.",
            code="import os\ncolour = 'teal'\ntotal = 8\n" + PROCESS_ID,
        )
        assert seeded.error is None and seeded.result != HARNESS_PID
        names = await tools["inspect_state"]()
        assert names["state"]["variables"] == ["colour", "total"]
        full = await tools["inspect_state"](detail="full")
        assert full["state"]["variables"] == {"colour": "'teal'", "total": "8"}
    finally:
        _CURRENT_SANDBOX.reset(token)
        await sandbox_session.close()
        await actor.close()


@pytest.mark.asyncio
async def test_inspect_state_without_the_worker_reads_the_namespace(monkeypatch):
    monkeypatch.setattr(SETTINGS, "UNIFY_WORKSPACE_PYTHON", "")
    ex, _ = executor_with_fakes()
    try:
        await run(ex, "colour = 'teal'")
        sb = ex.python_session(session_id=0)
        assert await sb.worker_variables() is None
        assert sb.global_state["colour"] == "teal"
    finally:
        await ex.close()


def worker_pid(ex) -> int:
    """The host pid of session 0's worker sandbox."""
    return ex.python_session(session_id=0)._worker.pid


def _parent(pid: int) -> int | None:
    try:
        with open(f"/proc/{pid}/stat") as fh:
            return int(fh.read().rsplit(")", 1)[1].split()[1])
    except (OSError, ValueError, IndexError):
        return None


def worker_tree(ex) -> list[int]:
    """The worker's bwrap and every process under it."""
    root = worker_pid(ex)
    children: dict[int, list[int]] = {}
    for entry in os.listdir("/proc"):
        if entry.isdigit():
            parent = _parent(int(entry))
            if parent is not None:
                children.setdefault(parent, []).append(int(entry))
    tree, stack = [root], [root]
    while stack:
        for child in children.get(stack.pop(), []):
            tree.append(child)
            stack.append(child)
    return tree


def _running(pid: int) -> bool:
    try:
        with open(f"/proc/{pid}/stat") as fh:
            state = fh.read().rsplit(")", 1)[1].split()[0]
    except (OSError, IndexError):
        return False
    return state not in ("Z", "X")


async def all_gone(pids: list[int], within: float = 10.0) -> bool:
    loop = asyncio.get_running_loop()
    deadline = loop.time() + within
    while any(_running(p) for p in pids):
        if loop.time() > deadline:
            return False
        await asyncio.sleep(0.1)
    return True
