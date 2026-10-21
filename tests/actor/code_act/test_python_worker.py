"""Symbolic: Python cells run in a sandboxed child (the worker).

Nothing here reaches a model. Cells run in the actor's real interpreter
(``sandbox_world.actor_interpreter``), the worker is a real child process inside
the real bubblewrap policy, and the harness objects it calls are fakes that
record where they ran. Tests that need
bubblewrap are skipped, saying so, where it is missing. The world is described
in ``sandbox_world.py``.
"""

from __future__ import annotations

import asyncio
import os
import sqlite3

import pytest
from pydantic import BaseModel

from tests.actor.code_act.sandbox_world import (  # noqa: F401 (fixture)
    STATE_SECRET,
    TOKEN_VALUE,
    actor_interpreter,
    needs_bwrap,
    serve,
    world,
)
from unify.actor.execution import worker as worker_mod
from unify.function_manager.execution_env import create_execution_globals
from unify.settings import SETTINGS


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


async def executor_with_fakes(timeout: float = 60.0):
    """The actor's interpreter with ``primitives`` bound to the fakes, and the list their calls are recorded in."""
    calls: list = []
    shadow = create_execution_globals()
    shadow["primitives"] = FakePrimitives(calls)
    return await actor_interpreter(shadow=shadow, limit=timeout), calls


@pytest.fixture
def worker_world(world, monkeypatch):
    return world


async def run(ex, code):
    res = await ex.run_cell(code)
    return res.output, res


# ── the worker ───────────────────────────────────────────────────────────────


@needs_bwrap
@pytest.mark.asyncio
async def test_state_persists_across_cells_in_one_worker(worker_world):
    ex, _ = await executor_with_fakes()
    try:
        out, res = await run(
            ex,
            "import os, math\nx = 41\ndef f(y):\n    return y * 2\n" + PROCESS_ID,
        )
        assert res.error is None, res.error
        pid = res.result
        assert isinstance(pid, str) and pid != HARNESS_PID
        out, res = await run(ex, "print(f(x), math.floor(2.5))\n" + PROCESS_ID)
        assert out == "82 2\n" and res.result == pid
    finally:
        await ex.stop()


# A cell's model whose annotations name what the cell imported: pydantic
# resolves them through ``sys.modules[cls.__module__]``.
TYPED_MODEL = (
    "from typing import List, Optional\n"
    "from pydantic import BaseModel\n"
    "class Tag(BaseModel):\n"
    "    label: str\n"
    "class Record(BaseModel):\n"
    "    ids: List[int]\n"
    "    tags: Optional[List[Tag]] = None\n"
)
USE_MODEL = "Record(ids=[1, 2], tags=[Tag(label='a')]).model_dump_json()"


@needs_bwrap
@pytest.mark.asyncio
async def test_a_cells_typed_model_resolves_against_the_session_module(
    worker_world,
):
    ex, _ = await executor_with_fakes(timeout=3)
    expected = '{"ids":[1,2],"tags":[{"label":"a"}]}'
    try:
        _, res = await run(ex, TYPED_MODEL + USE_MODEL)
        assert res.error is None, res.error
        assert res.result == expected
        _, res = await run(ex, "Record.__module__")
        assert res.result == "__sandbox_worker__"
        # A fresh worker registers its module again.
        _, res = await run(ex, "while True:\n    pass")
        assert res.timed_out
        _, res = await run(ex, TYPED_MODEL + USE_MODEL)
        assert res.result == expected, res.error
    finally:
        await ex.stop()


@pytest.fixture
def answers(monkeypatch) -> dict:
    """``query_llm`` as a fake: records each ``response_format`` it receives
    and answers ``answers["next"]`` (as query_llm does for a schema dict)."""
    import unify.common.reasoning as reasoning

    state: dict = {"sent": [], "next": None}

    async def fake_query_llm(prompt, *, response_format=None, **_kwargs):
        state["sent"].append(response_format)
        return state["next"]

    monkeypatch.setattr(reasoning, "query_llm", fake_query_llm)
    return state


@needs_bwrap
@pytest.mark.asyncio
async def test_a_cells_response_model_crosses_as_its_schema(worker_world, answers):
    ex, _ = await executor_with_fakes()
    try:
        answers["next"] = {"ids": [1, 2], "tags": [{"label": "a"}]}
        _, res = await run(
            ex,
            TYPED_MODEL + "r = await query_llm('Extract.', response_format=Record)\n"
            "f'{type(r) is Record}:{r.tags[0].label}'",
        )
        assert res.error is None, res.error
        assert res.result == "True:a"
        (sent,) = answers["sent"]
        assert sent["type"] == "json_schema" and sent["json_schema"]["name"] == "Record"
        assert set(sent["json_schema"]["schema"]["properties"]) == {"ids", "tags"}
        # An answer that does not fit the cell's model raises pydantic's error
        # in the cell, as in-process.
        answers["next"] = {"ids": "not a list"}
        _, res = await run(
            ex,
            "await query_llm('Extract.', response_format=Record)",
        )
        assert "ValidationError" in res.error and "ids" in res.error
        # A schema too large to send is refused before anything crosses.
        _, res = await run(
            ex,
            "fields = {f'field_{i}': int for i in range(4000)}\n"
            "Wide = type('Wide', (BaseModel,), "
            "{'__module__': __name__, '__annotations__': fields})\n"
            "await query_llm('Extract.', response_format=Wide)",
        )
        assert "BoundaryRefusal" in res.error and "JSON schema is" in res.error
        assert len(answers["sent"]) == 2
    finally:
        await ex.stop()


OPUS = "anthropic/claude-opus-4.6@openrouter"
LUNA = "openai/gpt-6-luna@openrouter"


@needs_bwrap
@pytest.mark.asyncio
async def test_a_cell_can_call_only_the_sessions_model(worker_world, monkeypatch):
    """UNIFY_CELL_LLM_MODELS at its default: a cell's query_llm and unillm are refused another model before any
    client is built in the harness; query_llm without model= uses the session's."""
    import unillm

    import unify.common.reasoning as reasoning

    monkeypatch.setattr(SETTINGS, "UNIFY_MODEL", LUNA)
    monkeypatch.setattr(SETTINGS, "UNIFY_CELL_LLM_MODELS", "")
    built: list = []

    class FakeClient:
        async def generate(self, **_kwargs):
            return "configured"

    def fake_new_llm_client(model=None, **_kwargs):
        built.append(model)
        return FakeClient()

    def no_client(*_args, **_kwargs):
        raise AssertionError("a unillm client was built in the harness")

    monkeypatch.setattr(reasoning, "new_llm_client", fake_new_llm_client)
    monkeypatch.setattr(unillm, "AsyncUnify", no_client)
    monkeypatch.setattr(unillm, "Unify", no_client)
    ex, _ = await executor_with_fakes()
    try:
        _, res = await run(
            ex,
            "seen = []\n"
            "try:\n"
            f"    await query_llm('x', model={OPUS!r})\n"
            "except Exception as e:\n"
            "    seen.append(f'{type(e).__name__}: {e}')\n"
            "try:\n"
            f"    unillm.AsyncUnify({OPUS!r})\n"
            "except Exception as e:\n"
            "    seen.append(f'{type(e).__name__}: {e}')\n"
            "try:\n"
            "    unillm.SETTINGS\n"
            "except Exception as e:\n"
            "    seen.append(type(e).__name__)\n"
            "seen.append(await query_llm('x'))\n"
            "seen",
        )
        assert res.error is None, res.error
        refused_query, refused_unillm, refused_attr, answer = res.result
        message = (
            f"can only use the session's model {LUNA} here (UNIFY_CELL_LLM_MODELS)"
        )
        assert refused_query == (
            f"CellModelRefused: query_llm {message}; omit model= to use it"
        )
        assert refused_unillm.startswith("CellModelRefused: unillm.AsyncUnify() ")
        assert message in refused_unillm
        assert refused_attr == "CellAttributeRefused"
        # Only the call without model= built a client: the configured model.
        assert answer == "configured" and built == [None]
    finally:
        await ex.stop()


@needs_bwrap
@pytest.mark.asyncio
async def test_a_cell_cannot_read_secrets_write_the_state_or_reach_the_network(
    worker_world,
):
    srv, port = serve(b"hello-from-host\n")
    store = worker_world["state"] / "store.sqlite"
    ex, _ = await executor_with_fakes()
    try:
        out, res = await run(
            ex,
            "import os\n"
            "print(os.environ.get('FAKE_SERVICE_TOKEN'), "
            "os.environ.get('DB_PASSWORD'), os.environ.get('UNIFY_SANDBOX_PROBE'))",
        )
        assert out == "None None visible\n", res.error
        # A file in the state directory is not mounted: neither read nor written.
        out, res = await run(
            ex,
            "import sqlite3\n"
            "try:\n"
            f"    con = sqlite3.connect('file:{store}?mode=ro', uri=True)\n"
            "    print(con.execute('select name from functions').fetchall())\n"
            "except sqlite3.OperationalError as e:\n"
            "    print('refused:', e)\n"
            "try:\n"
            f"    con = sqlite3.connect({str(store)!r})\n"
            "    con.execute(\"insert into functions values ('evil')\")\n"
            "    con.commit()\n"
            "    print('wrote')\n"
            "except sqlite3.OperationalError as e:\n"
            "    print('refused:', e)",
        )
        assert out.startswith("refused: unable to open database file\n"), (
            out,
            res.error,
        )
        assert "('f',)" not in out
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
        assert STATE_SECRET not in str(res.result) and STATE_SECRET not in str(
            res.error,
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
        assert out.startswith("refused:"), (out, res.error)
    finally:
        srv.close()
        await ex.stop()


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
    ex, _ = await executor_with_fakes()
    try:
        out, res = await run(ex, connect.format(port=proxy_port))
        assert out == "got from proxy\n", res.error
        out, _ = await run(ex, connect.format(port=other_port))
        assert out.startswith("refused:")
        # The channel to the harness still works through the forwarder.
        _, res = await run(ex, "(await primitives.files.search('p', limit=1))['hits']")
        assert res.result == ["p-0"]
    finally:
        proxy.close()
        other.close()
        await ex.stop()


@needs_bwrap
@pytest.mark.asyncio
async def test_primitives_run_in_the_harness_through_the_proxy(worker_world):
    ex, calls = await executor_with_fakes()
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
        assert res.error is None, res.error
        worker_pid = res.result
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
        assert res.error is None, res.error
        lines = out.splitlines()
        assert lines[0] == "a 3 Item(name='a', size=3)"
        assert lines[1] == "Item:a"
        assert lines[2] == f"Opaque('job') finished job in {HARNESS_PID}"
        assert lines[3] == "['api_token', 'broken', 'calls']"
        # The handle returned as the result is the harness's own object.
        assert isinstance(res.result, Opaque) and res.result.label == "job"
    finally:
        await ex.stop()


@needs_bwrap
@pytest.mark.asyncio
async def test_harness_exceptions_are_reraised_in_the_worker(worker_world):
    ex, _ = await executor_with_fakes()
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
        assert res.error is None, res.error
        assert out.splitlines() == [
            "KeyError 'k'",
            f"HarnessOnlyError {__name__} only the harness knows this type",
        ]
        # Uncaught, it is the cell's error, named.
        _, res = await run(ex, "await primitives.files.broken()")
        assert "HarnessOnlyError: only the harness knows this type" in res.error
        # A cell's own exception is reported as a traceback, as in process.
        _, res = await run(ex, "1 / 0")
        assert "ZeroDivisionError" in res.error
    finally:
        await ex.stop()


@needs_bwrap
@pytest.mark.asyncio
async def test_what_cannot_cross_is_refused_by_name(worker_world):
    ex, calls = await executor_with_fakes()
    try:
        _, res = await run(ex, "await primitives.files.search(lambda: 1)")
        assert "BoundaryRefusal" in res.error
        assert "argument 1 of primitives.files.search is a function" in res.error
        _, res = await run(
            ex,
            "class Mine:\n    pass\nawait primitives.files.search('q', limit=Mine())",
        )
        assert "argument 'limit' of primitives.files.search is a Mine" in (res.error)
        assert calls == []  # refused before anything was sent
        _, res = await run(ex, "primitives.files._private")
        assert "attributes starting with '_'" in res.error
        _, res = await run(ex, "primitives.files.api_token")
        assert "looks like a credential" in res.error
        assert "tok-attr-never-sent" not in res.error
        # Modules and the harness's environment never cross, however reached.
        _, res = await run(
            ex,
            "primitives.files.tools.environ.get('FAKE_SERVICE_TOKEN')",
        )
        assert "primitives.files.tools is a harness module (os)" in res.error
        _, res = await run(
            ex,
            "(await primitives.files.environment()).get('DB_PASSWORD')",
        )
        assert "harness's environment" in res.error
        assert TOKEN_VALUE not in res.error
        # A worker object returned as the result comes back as its repr.
        _, res = await run(
            ex,
            "class Box:\n    def __repr__(self): return 'Box!'\nBox()",
        )
        assert res.error is None and repr(res.result) == "Box!"
        assert isinstance(res.result, worker_mod.WorkerValue)
    finally:
        await ex.stop()


@needs_bwrap
@pytest.mark.asyncio
async def test_a_timeout_kills_the_worker_and_the_next_cell_starts_fresh(
    worker_world,
):
    ex, _ = await executor_with_fakes(timeout=3)
    try:
        _, res = await run(ex, "x = 1")
        tree = worker_tree(ex)
        assert len(tree) >= 2  # bwrap and the worker inside it
        _, res = await run(ex, "while True:\n    pass")
        assert res.timed_out and "timed out after 3 s" in res.note
        assert await all_gone(tree)
        out, res = await run(ex, HAS_X)
        assert res.result is False
    finally:
        await ex.stop()


@needs_bwrap
@pytest.mark.asyncio
async def test_a_stopped_cell_kills_the_interpreter_and_the_next_cell_is_told(worker_world):
    ex, _ = await executor_with_fakes()
    try:
        await run(ex, "x = 1")
        tree = worker_tree(ex)
        task = asyncio.create_task(run(ex, "import time\ntime.sleep(60)"))
        await asyncio.sleep(1.0)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert await all_gone(tree)
        _, res = await run(ex, HAS_X)
        assert res.result is False and "the previous cell was stopped" in res.note
        # A worker that exits by itself says so too.
        _, res = await run(ex, "import os\nos._exit(3)")
        assert res.restarted and "exited during the cell" in res.note
        _, res = await run(ex, "1")
        assert res.result == 1
    finally:
        await ex.stop()


@needs_bwrap
@pytest.mark.asyncio
async def test_output_and_display_cross_in_order(worker_world):
    ex, _ = await executor_with_fakes()
    try:
        out, res = await run(
            ex,
            "import subprocess, sys\n"
            "print('printed')\n"
            "sys.stdout.write('written\\n')\n"
            "display({'a': 1})\n"
            "subprocess.run(['echo', 'from-subprocess'])\n"
            "print('err', file=sys.stderr)",
        )
        assert res.error is None, res.error
        assert out.startswith("printed\nwritten\n{'a': 1}\nfrom-subprocess\n")
        assert out.rstrip().endswith("err")
    finally:
        await ex.stop()


def worker_pid(ex) -> int:
    """The host pid of the interpreter's sandbox (its bwrap)."""
    return ex._backend.host_pid


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
