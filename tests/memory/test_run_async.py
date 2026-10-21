"""``memory.run`` of an ``async def`` procedure gives what it returns, never the coroutine's text (rev 2).

A captured procedure runs in its own child process (``library_helper._CAPTURE_RUNNER``): an ``async def`` ``run``,
``read_input``, ``applies`` or ``holds`` is awaited there. A library function runs in the cell's process: it is
awaited when no event loop runs there, and otherwise refused with :class:`library_helper.AsyncItemError` (the
coroutine closed). No model, network or key.
"""

from __future__ import annotations

import asyncio
import json
import subprocess
import sys
import warnings

import pytest

from unify.memory import library_helper as lh

ASYNC_PROC = '''async def read_input(request):
    return int(request.text)


async def applies(value):
    return value > 0


async def run(value):
    return value * 2


def holds(value, result):
    return result == 2 * value
'''


def _child(tmp_path, payload):
    proc = tmp_path / "procedure.py"
    proc.write_text(ASYNC_PROC)
    done = subprocess.run([sys.executable, "-I", "-c", lh._CAPTURE_RUNNER, str(proc), "MARK:"], input=json.dumps(payload),
                          capture_output=True, text=True, timeout=60, env={}, cwd=tmp_path)  # fmt: skip
    return json.loads(done.stdout.split("MARK:")[-1])


def test_an_async_captured_procedure_is_awaited_in_its_child(tmp_path):
    got = _child(tmp_path, {"args": ["21"], "kwargs": {}})
    assert got["outcome"] == "returned" and got["result"] == 42 and "result_is_text" not in got
    assert got["applies"] is True and got["holds"] is True and got["repeatable"] is True
    once = _child(tmp_path, {"args": [5], "kwargs": {}, "once": True})
    assert once["result"] == 10 and "coroutine" not in json.dumps(once)


def test_an_async_library_function_is_awaited_or_refused_never_its_text():
    async def double(x):
        return 2 * x

    assert lh._settled(double(4)) == 8  # no loop running here: run to completion
    assert lh._settled(7) == 7

    async def inside_a_running_loop():
        coro = double(4)
        with pytest.raises(lh.AsyncItemError, match="await"):
            lh._settled(coro)
        return coro

    with warnings.catch_warnings():
        warnings.simplefilter("error")  # a coroutine left un-awaited would warn: it is closed instead
        coro = asyncio.run(inside_a_running_loop())
    assert coro.cr_frame is None  # closed
