"""Symbolic: ``UNIFY_FUNCTION_PATCH`` offers the patch tools, and only where writes are allowed.

With the switch on, ``FunctionManager_patch_function`` and
``GuidanceManager_patch_guidance`` join the actor's tools and the storage
review's tools, and the review's prompt gains the update-before-add order.
They are store-only tools: an actor that cannot store, or whose writes are
withheld until the environment admits the run, does not get them. Off, the
tool sets and both review prompts are exactly as shipped. No model is called:
the tool loop and the review loop are mocked.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import unify.actor.code_act_actor as code_act_actor
from unify.actor.code_act_actor import (
    CodeActActor,
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


# --------------------------------------------------------------------------- #
#  The actor                                                                   #
# --------------------------------------------------------------------------- #


async def _act_tools(monkeypatch, **act_kwargs) -> set:
    """Run ``act()`` up to its tool loop and return the tools the loop got."""
    captured: dict = {}

    def fake_loop(client, message, tools, **kwargs):
        captured["tools"] = dict(tools)
        captured["policy"] = kwargs.get("tool_policy")
        handle = MagicMock()
        handle.result = AsyncMock(return_value="done")
        handle.next_notification = AsyncMock(
            side_effect=lambda: asyncio.Event().wait(),
        )
        handle._client = MagicMock(messages=[])
        return handle

    monkeypatch.setattr(code_act_actor, "start_async_tool_loop", fake_loop)
    monkeypatch.setattr(code_act_actor, "_start_storage_check_loop", lambda **kw: None)
    monkeypatch.setattr(code_act_actor, "publish_manager_method_event", AsyncMock())
    actor = CodeActActor(timeout=30)
    try:
        await actor.act("Do something", persist=False, **act_kwargs)
        captured["registered"] = set(actor.get_tools("act"))
    finally:
        try:
            await actor.close()
        except Exception:
            pass
    return captured


@pytest.mark.asyncio
@pytest.mark.timeout(120)
async def test_the_actor_gets_the_patch_tools_only_while_on(monkeypatch):
    monkeypatch.setattr(SETTINGS, "UNIFY_STORE_ADMISSION", "")
    monkeypatch.setattr(SETTINGS, "UNIFY_FUNCTION_PATCH", False)
    off = await _act_tools(monkeypatch, can_store=True)
    assert not PATCH_TOOLS & off["registered"]
    monkeypatch.setattr(SETTINGS, "UNIFY_FUNCTION_PATCH", True)
    on = await _act_tools(monkeypatch, can_store=True)
    assert on["registered"] - off["registered"] == PATCH_TOOLS
    assert set(on["tools"]) - set(off["tools"]) == PATCH_TOOLS
    assert set(off["tools"]) <= set(on["tools"])


@pytest.mark.asyncio
@pytest.mark.timeout(120)
async def test_an_actor_that_cannot_store_cannot_patch(patch_on, monkeypatch):
    monkeypatch.setattr(SETTINGS, "UNIFY_STORE_ADMISSION", "")
    captured = await _act_tools(monkeypatch, can_store=False)
    assert PATCH_TOOLS <= captured["registered"]
    assert not PATCH_TOOLS & set(captured["tools"])


@pytest.mark.asyncio
@pytest.mark.timeout(120)
async def test_an_admission_gated_actor_cannot_patch(patch_on, monkeypatch, tmp_path):
    monkeypatch.setattr(SETTINGS, "UNIFY_STORE_ADMISSION", str(tmp_path / "v.json"))
    captured = await _act_tools(monkeypatch, can_store=True)
    assert PATCH_TOOLS <= captured["registered"]
    assert not PATCH_TOOLS & set(captured["tools"])
    # The review that runs once the run is admitted still gets them.
    assert PATCH_TOOLS <= _storage_tool_names()


@pytest.mark.asyncio
@pytest.mark.timeout(120)
async def test_under_cache_discipline_a_gated_actor_lists_the_patch_tools_masked(
    patch_on,
    monkeypatch,
    tmp_path,
):
    """The session's one tool list keeps them; admission refuses a call by rule."""
    monkeypatch.setattr(SETTINGS, "UNIFY_STORE_ADMISSION", str(tmp_path / "v.json"))
    monkeypatch.setattr(SETTINGS, "UNIFY_CACHE_DISCIPLINE", True)
    captured = await _act_tools(monkeypatch, can_store=True)
    assert PATCH_TOOLS <= set(captured["tools"])
    searched = ["FunctionManager_search_functions", "GuidanceManager_search"]
    _mode, visible, opts = captured["policy"](5, dict(captured["tools"]), searched)
    assert not PATCH_TOOLS & set(visible)
    for name in PATCH_TOOLS:
        assert opts["mask_rules"][name] == code_act_actor._ADMISSION_MASK_RULE
