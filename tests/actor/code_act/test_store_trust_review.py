"""Symbolic: with ``UNIFY_STORE_TRUST=ramp`` the next storage review is told which functions need repair.

A quarantined function is left out of the session's searches, so nothing
repairs it unless the review hears of it. With the switch on and at least one
function quarantined, the post-run review's prompt lists each one with its
failure count and last failure, just before the trajectory (the volatile
tail, so the cached prefix is unchanged). With nothing quarantined, or with
the switch off even while records say otherwise, both review prompts are
exactly as shipped. No model is called: the review loop is mocked.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

import unify.actor.code_act_actor as code_act_actor
from tests.helpers import _handle_project
from unify.actor.code_act_actor import (
    _start_proactive_storage_loop,
    _start_storage_check_loop,
    _storage_needs_repair_note,
)
from unify.function_manager import store_trust
from unify.function_manager.function_manager import FunctionManager
from unify.settings import SETTINGS

DIVIDE = "def divide(a: int, b: int) -> float:\n    return a / b\n"
DOUBLE = "def double(x: int) -> int:\n    return x * 2\n"


def _review_prompts() -> list[str]:
    """The post-run and the proactive review's system prompts."""
    actor = MagicMock()
    prompts = []
    with (
        patch.object(code_act_actor, "_build_storage_tools", return_value=({}, [], [])),
        patch.object(code_act_actor, "new_llm_client") as client,
        patch.object(code_act_actor, "start_async_tool_loop"),
    ):
        trajectory = [{"role": "user", "content": "do it"}]
        _start_storage_check_loop(
            trajectory=trajectory,
            ask_tools={},
            actor=actor,
            original_result="done",
        )
        prompts.append(client.return_value.set_system_message.call_args[0][0])
        _start_proactive_storage_loop(
            trajectory=trajectory,
            ask_tools={},
            actor=actor,
            request="store it",
        )
        prompts.append(client.return_value.set_system_message.call_args[0][0])
    return prompts


def _quarantine(fm: FunctionManager) -> None:
    fm.add_functions(implementations=[DIVIDE, DOUBLE])
    namespace: dict = {}
    fm.list_functions(_return_callable=True, _namespace=namespace)
    namespace["double"](1)
    namespace["divide"](1, 1)
    with pytest.raises(ZeroDivisionError):
        namespace["divide"](1, 0)


@_handle_project
def test_the_review_lists_quarantined_functions_only_while_on(monkeypatch):
    monkeypatch.setattr(SETTINGS, "UNIFY_STORE_TRUST", "")
    shipped = _review_prompts()
    monkeypatch.setattr(SETTINGS, "UNIFY_STORE_TRUST", "ramp")
    assert _storage_needs_repair_note() == ""
    assert _review_prompts() == shipped  # on, but nothing needs repair

    fm = FunctionManager(include_primitives=False)
    _quarantine(fm)
    note = _storage_needs_repair_note()
    assert note == (
        "## Needs Repair\n\n"
        "These stored functions raised when they were last reused, so the "
        "session's searches leave them out until they change. Repair one when "
        "the trajectory or its failure shows the fix (patch or overwrite it; a "
        "changed function starts again on probation), delete it if it cannot "
        "work, or leave it:\n"
        "- `divide`: 1 failure(s) after 1 pass(es); last failure: "
        "ZeroDivisionError: division by zero\n\n"
    )
    post_run, proactive = _review_prompts()
    header = "## Completed Trajectory\n\n"
    assert post_run == shipped[0].replace(header, note + header, 1)
    assert proactive == shipped[1]

    # off again: records are ignored and the prompts are as shipped
    monkeypatch.setattr(SETTINGS, "UNIFY_STORE_TRUST", "")
    assert _storage_needs_repair_note() == ""
    assert _review_prompts() == shipped


@_handle_project
def test_a_repaired_function_leaves_the_list(monkeypatch):
    monkeypatch.setattr(SETTINGS, "UNIFY_STORE_TRUST", "ramp")
    fm = FunctionManager(include_primitives=False)
    _quarantine(fm)
    assert [t.name for t in store_trust.needs_repair()] == ["divide"]
    fixed = DIVIDE.replace("return a / b", "return a / b if b else 0.0")
    fm.add_functions(implementations=[fixed], overwrite=True)
    assert store_trust.needs_repair() == []
    assert _storage_needs_repair_note() == ""


@_handle_project
def test_the_list_is_capped(monkeypatch):
    monkeypatch.setattr(SETTINGS, "UNIFY_STORE_TRUST", "ramp")
    fm = FunctionManager(include_primitives=False)
    count = store_trust.MAX_REPAIR_LISTED + 2
    fm.add_functions(
        implementations=[
            f"def broken_{i}() -> None:\n    raise ValueError('{i}')\n"
            for i in range(count)
        ],
    )
    namespace: dict = {}
    fm.list_functions(_return_callable=True, _namespace=namespace)
    for i in range(count):
        with pytest.raises(ValueError):
            namespace[f"broken_{i}"]()
    note = _storage_needs_repair_note()
    assert note.count("\n- `broken_") == store_trust.MAX_REPAIR_LISTED
    assert note.endswith("- and 2 more\n\n")
