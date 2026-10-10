"""Symbolic: the storage review has the patch tools and the update-before-add order.

``FunctionManager_patch_function`` and ``GuidanceManager_patch_guidance`` are
in the storage review's tools, and both review prompts carry the
update-before-add order, which describes batching an entry's changes as
`edits` in one call. No model is called: the review loop is mocked.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import unify.actor.code_act_actor as code_act_actor
from unify.actor.code_act_actor import (
    _start_proactive_storage_loop,
    _start_storage_check_loop,
    _storage_update_first_note,
)

PATCH_TOOLS = {"FunctionManager_patch_function", "GuidanceManager_patch_guidance"}


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
    tools = code_act_actor._build_storage_tools(actor=actor)
    return set(tools)


def test_the_review_gets_the_patch_tools():
    assert PATCH_TOOLS <= _storage_tool_names()


def _review_prompts() -> list[str]:
    actor = MagicMock()
    prompts = []
    with (
        patch.object(
            code_act_actor,
            "_build_storage_tools",
            return_value={},
        ),
        patch.object(code_act_actor, "new_llm_client") as client,
        patch.object(code_act_actor, "start_async_tool_loop"),
    ):
        trajectory = [{"role": "user", "content": "do it"}]
        _start_storage_check_loop(
            trajectory=trajectory,
            actor=actor,
            original_result="done",
        )
        prompts.append(client.return_value.set_system_message.call_args[0][0])
        _start_proactive_storage_loop(
            trajectory=trajectory,
            actor=actor,
            request="store it",
        )
        prompts.append(client.return_value.set_system_message.call_args[0][0])
    return prompts


def test_the_update_first_order_is_in_both_review_prompts():
    note = _storage_update_first_note()
    assert note.startswith("### Update before you add")
    for tool in PATCH_TOOLS:
        assert f"`{tool}`" in note
    # Several changes to one entry go in one call, applied all or none.
    assert "in one call as `edits`" in note
    assert "all or none" in note
    for prompt in _review_prompts():
        assert prompt.count(note) == 1
    # Patch the entry the trajectory used, else add a focused new one; a fix
    # never moves into a broader entry.
    assert note.index("(1) patch the entry the trajectory used") < note.index(
        "(2) otherwise add a new focused entry",
    )
    assert "Do not move a fix into a broader entry" in note
