"""The tool loop's memory-v2 result hook (``tools_data._memory_v2_result_hook``).

With ``UNIFY_MEMORY_V2`` off (or no request run) the hook resolves to None and the loop never enters
it. With one, it is handed each finished call's raw value, or the exception of a call that raised, and
nothing it does (raising included) changes the tool message or ``completed_results``. Driven through
the real dispatch and result path (``schedule_base_tool_call``, the task, ``process_completed_task``)
without a model, as ``test_unknown_arguments`` does.
"""

from __future__ import annotations

import pytest

from tests.async_tool_loop.test_unknown_arguments import _Loop
from unify.common._async_tool import tools_data
from unify.memory_v2.integration import hooks
from unify.settings import SETTINGS

PLAIN = {
    "stdout": "",
    "stderr": "",
    "result": None,
    "error": None,
    "state_mode": "stateful",
    "session_id": None,
    "session_name": None,
    "session_created": False,
    "duration_ms": 0,
}


def _tools() -> dict:
    def execute_code(code: str) -> dict:
        if code == "raise":
            raise RuntimeError("the cell's tool failed")
        return dict(PLAIN)

    return {"execute_code": execute_code}


async def _run(code: str) -> tuple[str, object]:
    loop = _Loop(_tools())
    content = await loop.call("execute_code", {"code": code})
    return content, loop._tools_data.completed_results["call_1"]


@pytest.mark.asyncio
async def test_with_memory_v2_off_the_hook_is_none_and_never_entered(monkeypatch):
    monkeypatch.setattr(SETTINGS, "UNIFY_MEMORY_V2", "")
    entered: list = []
    monkeypatch.setattr(hooks, "tool_result", lambda *a, **k: entered.append(a))
    assert tools_data._memory_v2_result_hook() is None
    await _run("")
    await _run("raise")
    assert entered == []


@pytest.mark.asyncio
async def test_the_hook_gets_the_raw_value_or_the_exception(monkeypatch):
    seen: list = []

    def note(name, call_id, raw, *, raised=None):
        seen.append((name, call_id, raw, type(raised).__name__ if raised else None))

    monkeypatch.setattr(tools_data, "_memory_v2_result_hook", lambda: note)
    await _run("")
    await _run("raise")
    assert seen == [
        ("execute_code", "call_1", PLAIN, None),
        ("execute_code", "call_1", None, "RuntimeError"),
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize("code", ["", "raise"], ids=["returned", "raised"])
async def test_a_raising_hook_leaves_the_tool_message_and_results_unchanged(
    monkeypatch,
    code,
):
    monkeypatch.setattr(tools_data, "_memory_v2_result_hook", lambda: None)
    without = await _run(code)

    def broken(*a, **k):
        raise RuntimeError("the hook failed")

    monkeypatch.setattr(tools_data, "_memory_v2_result_hook", lambda: broken)
    with_broken = await _run(code)
    assert with_broken == without
