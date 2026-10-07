"""Symbolic: ``UNIFY_FUNCTION_PATCH`` offers the patch tools, and only where writes are allowed.

With the switch on, ``FunctionManager_patch_function`` and
``GuidanceManager_patch_guidance`` join the actor's tools and the storage
review's tools, and the review's prompt gains the update-before-add order.
The note describes batching an entry's changes as `edits` in one call.
They are store-only tools: an actor that cannot store, or whose writes are
withheld until the environment admits the run, does not get them. Off, the
tool sets and both review prompts are exactly as shipped. No model is called:
the tool loop and the review loop are mocked.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

import unify.actor.code_act_actor as code_act_actor
from unify.actor.code_act_actor import (
    _start_proactive_storage_loop,
    _start_storage_check_loop,
    _storage_update_first_note,
)
from unify.settings import SETTINGS

PATCH_TOOLS = {"FunctionManager_patch_function", "GuidanceManager_patch_guidance"}


@pytest.fixture
def patch_on(monkeypatch):
    monkeypatch.setattr(SETTINGS, "UNIFY_FUNCTION_PATCH", True)


# --------------------------------------------------------------------------- #
#  The storage review                                                          #
# --------------------------------------------------------------------------- #


def _storage_tool_names() -> set:
    from unify.function_manager.function_manager import FunctionManager
    from unify.guidance_manager.guidance_manager import GuidanceManager

    actor = SimpleNamespace(
        function_manager=FunctionManager(),
        guidance_manager=GuidanceManager(),
    )
    tools, _, _ = code_act_actor._build_storage_tools(actor=actor, ask_tools={})
    return set(tools)


def test_the_review_gets_the_patch_tools_only_while_on(monkeypatch):
    monkeypatch.setattr(SETTINGS, "UNIFY_FUNCTION_PATCH", False)
    off = _storage_tool_names()
    assert not PATCH_TOOLS & off
    monkeypatch.setattr(SETTINGS, "UNIFY_FUNCTION_PATCH", True)
    on = _storage_tool_names()
    assert on - off == PATCH_TOOLS
    assert off <= on


def _review_prompts() -> list[str]:
    actor = MagicMock()
    prompts = []
    with (
        patch.object(
            code_act_actor,
            "_build_storage_tools",
            return_value=({}, [], []),
        ),
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


def test_the_update_first_order_is_in_both_review_prompts_only_while_on(
    monkeypatch,
):
    monkeypatch.setattr(SETTINGS, "UNIFY_FUNCTION_PATCH", False)
    assert _storage_update_first_note() == ""
    off = _review_prompts()
    monkeypatch.setattr(SETTINGS, "UNIFY_FUNCTION_PATCH", True)
    note = _storage_update_first_note()
    assert note.startswith("### Update before you add")
    for tool in PATCH_TOOLS:
        assert f"`{tool}`" in note
    # Several changes to one entry go in one call, applied all or none.
    assert "in one call as `edits`" in note
    assert "all or none" in note
    on = _review_prompts()
    for before, after in zip(off, on):
        assert "Update before you add" not in before
        assert after == before.replace(
            code_act_actor._STORAGE_TWO_STORES,
            code_act_actor._STORAGE_TWO_STORES + note,
        )
    first, broader, add = (
        note.index("(1) patch the entry the trajectory used"),
        note.index("(2) otherwise patch a broader existing entry"),
        note.index("(3) only then add a new one"),
    )
    assert first < broader < add
