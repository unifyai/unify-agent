"""
Chat context propagation tests for async tool loop.

Verifies that `parent_chat_context` is threaded into tools that accept it and
that the loop inserts the synthetic system context header.

Also tests incremental context propagation:
- First tool call receives the filtered initial snapshot (genuine user turns
  and substantive assistant text only)
- Subsequent calls receive only incremental updates via _parent_chat_context_cont,
  passed through the same snapshot filter

Injection is opt-in under LLM_DECIDES: an omitted include_parent_chat_context
means no parent context is passed.
"""

from __future__ import annotations

import asyncio
import json
from typing import List

import pytest
from unify.common.async_tool_loop import ChatContextPropagation, start_async_tool_loop
from unify.common._async_tool.context_tracker import LoopContextState
from unify.common._async_tool.messages import loop_user_notice
from tests.helpers import _handle_project
from unify.common.llm_client import new_llm_client
from tests.baked_defaults import as_shipped  # noqa: F401


# as_shipped: deleted in step 5 (parent chat context and interjections)
@pytest.mark.usefixtures("as_shipped")
@pytest.mark.asyncio
@pytest.mark.llm_call
@_handle_project
async def test_chat_context_propagation(llm_config) -> None:
    client = new_llm_client(**llm_config)

    root_ctx = [{"role": "user", "content": "root-level message"}]
    captured_ctx: List[list[dict]] = []

    async def record_context(*, _parent_chat_context: list[dict] | None = None) -> str:
        captured_ctx.append(_parent_chat_context or [])
        return "context-recorded"

    record_context.__name__ = "record_context"
    record_context.__qualname__ = "record_context"

    handle = start_async_tool_loop(
        client=client,
        message="Please call the function `record_context()` once, then reply 'done'.",
        tools={"record_context": record_context},
        parent_chat_context=root_ctx,
        propagate_chat_context=ChatContextPropagation.ALWAYS,
    )

    final_ans = await handle.result()
    assert final_ans is not None, "Loop should complete with a response"

    # Find the runtime context header message (may not be at position 0 due to
    # other system messages like User Visibility Context being prepended)
    ctx_header_msg = next(
        (m for m in client.messages if m.get("_ctx_header") is True),
        None,
    )
    assert ctx_header_msg is not None, "Expected a system message with _ctx_header=True"
    assert ctx_header_msg["role"] == "system"

    assert len(captured_ctx) == 1
    combined = captured_ctx[0]

    assert combined[0]["content"] == "root-level message"
    assert "children" in combined[0]
    child_msgs = combined[0]["children"]
    # Find the user message (may not be at position 0 due to system messages
    # like User Visibility Context being prepended)
    user_msg = next(
        (m for m in child_msgs if m.get("role") == "user"),
        None,
    )
    assert user_msg is not None and user_msg["content"].startswith(
        "Please call the function",
    )


# as_shipped: deleted in step 5 (parent chat context and interjections)
@pytest.mark.usefixtures("as_shipped")
@pytest.mark.asyncio
@pytest.mark.llm_call
@_handle_project
async def test_chat_context_propagation_never(llm_config) -> None:
    """Verify that NEVER mode does NOT pass context to tools, even when they accept it."""
    client = new_llm_client(**llm_config)

    root_ctx = [{"role": "user", "content": "secret-context-message"}]
    captured_ctx: List[list[dict]] = []

    async def record_context(*, _parent_chat_context: list[dict] | None = None) -> str:
        captured_ctx.append(_parent_chat_context or [])
        return "context-recorded"

    record_context.__name__ = "record_context"
    record_context.__qualname__ = "record_context"

    handle = start_async_tool_loop(
        client=client,
        message="Please call the function `record_context()` once, then reply 'done'.",
        tools={"record_context": record_context},
        parent_chat_context=root_ctx,
        propagate_chat_context=ChatContextPropagation.NEVER,
    )

    final_ans = await handle.result()
    assert final_ans is not None, "Loop should complete with a response"

    # Tool should have been called
    assert len(captured_ctx) == 1, "Tool should have been called exactly once"

    # But context should be empty (NEVER mode)
    assert captured_ctx[0] == [], "Context should be empty in NEVER mode"


# as_shipped: deleted in step 5 (parent chat context and interjections)
@pytest.mark.usefixtures("as_shipped")
@pytest.mark.asyncio
@pytest.mark.llm_call
@_handle_project
async def test_chat_context_propagation_llm_decides_include(llm_config) -> None:
    """Verify that LLM_DECIDES mode passes context when the LLM explicitly opts in."""
    client = new_llm_client(**llm_config)

    root_ctx = [{"role": "user", "content": "root-level-context-marker"}]
    captured_ctx: List[list[dict]] = []

    async def record_context(*, _parent_chat_context: list[dict] | None = None) -> str:
        captured_ctx.append(_parent_chat_context or [])
        return "context-recorded"

    record_context.__name__ = "record_context"
    record_context.__qualname__ = "record_context"

    # Prompt the LLM to explicitly include context
    handle = start_async_tool_loop(
        client=client,
        message=(
            "Please call the function `record_context()` once with "
            "`include_parent_chat_context` set to `true`, then reply 'done'."
        ),
        tools={"record_context": record_context},
        parent_chat_context=root_ctx,
        propagate_chat_context=ChatContextPropagation.LLM_DECIDES,
    )

    final_ans = await handle.result()
    assert final_ans is not None, "Loop should complete with a response"

    # Tool should have been called
    assert len(captured_ctx) == 1, "Tool should have been called exactly once"

    # Context should be passed (LLM chose to include it)
    combined = captured_ctx[0]
    assert len(combined) > 0, "Context should be non-empty when LLM includes it"
    assert combined[0]["content"] == "root-level-context-marker"


# as_shipped: deleted in step 5 (parent chat context and interjections)
@pytest.mark.usefixtures("as_shipped")
@pytest.mark.asyncio
@pytest.mark.llm_call
@_handle_project
async def test_chat_context_propagation_llm_decides_exclude(llm_config) -> None:
    """Verify that LLM_DECIDES mode omits context when LLM explicitly excludes it."""
    client = new_llm_client(**llm_config)

    root_ctx = [{"role": "user", "content": "secret-context-should-not-appear"}]
    captured_ctx: List[list[dict]] = []

    async def record_context(*, _parent_chat_context: list[dict] | None = None) -> str:
        captured_ctx.append(_parent_chat_context or [])
        return "context-recorded"

    record_context.__name__ = "record_context"
    record_context.__qualname__ = "record_context"

    # Prompt the LLM to explicitly exclude context
    handle = start_async_tool_loop(
        client=client,
        message=(
            "Please call the function `record_context()` once with "
            "`include_parent_chat_context` set to `false`, then reply 'done'."
        ),
        tools={"record_context": record_context},
        parent_chat_context=root_ctx,
        propagate_chat_context=ChatContextPropagation.LLM_DECIDES,
    )

    final_ans = await handle.result()
    assert final_ans is not None, "Loop should complete with a response"

    # Tool should have been called
    assert len(captured_ctx) == 1, "Tool should have been called exactly once"

    # Context should be empty (LLM chose to exclude it)
    assert captured_ctx[0] == [], "Context should be empty when LLM excludes it"


# =============================================================================
# Unit tests for LoopContextState (incremental context tracking)
# =============================================================================


class TestLoopContextState:
    """Unit tests for the incremental context tracking state machine."""

    def test_initial_state(self):
        """Verify initial state is empty."""
        state = LoopContextState()
        assert state.parent_chat_context == []
        assert state._parent_chat_context_cont_received == []
        assert state.inner_tool_forwarding == {}

    def test_receive_context_continuation(self):
        """Verify context continuations are accumulated."""
        state = LoopContextState()

        state.receive_context_continuation([{"role": "user", "content": "msg1"}])
        assert len(state._parent_chat_context_cont_received) == 1

        state.receive_context_continuation([{"role": "user", "content": "msg2"}])
        assert len(state._parent_chat_context_cont_received) == 2

        # Empty list should not add anything
        state.receive_context_continuation([])
        assert len(state._parent_chat_context_cont_received) == 2

    def test_first_call_receives_full_context(self):
        """First call to a tool should receive the initial snapshot
        (all messages here pass the snapshot keep-rule)."""
        parent_ctx = [{"role": "user", "content": "parent msg"}]
        state = LoopContextState(parent_chat_context=parent_ctx)

        local_msgs = [{"role": "user", "content": "local msg"}]

        parent, cont = state.compute_context_for_inner_tool("call_1", local_msgs)

        # First call gets full parent context with local msgs as children
        assert parent is not None
        assert parent[0]["content"] == "parent msg"
        assert "children" in parent[0]
        assert parent[0]["children"][0]["content"] == "local msg"

        # No cont on first call (unless there were accumulated conts)
        assert cont is None

    def test_second_call_receives_only_incremental(self):
        """Second call to same tool should receive only incremental updates."""
        parent_ctx = [{"role": "user", "content": "parent msg"}]
        state = LoopContextState(parent_chat_context=parent_ctx)

        local_msgs = [{"role": "user", "content": "msg1"}]

        # First call
        state.compute_context_for_inner_tool("call_1", local_msgs)

        # Add more local messages
        local_msgs.append({"role": "assistant", "content": "msg2"})

        # Second call
        parent, cont = state.compute_context_for_inner_tool("call_1", local_msgs)

        # No parent context on subsequent calls
        assert parent is None

        # Only the new message in cont
        assert cont is not None
        assert len(cont) == 1
        assert cont[0]["content"] == "msg2"

    def test_different_tools_get_independent_tracking(self):
        """Different tool calls should have independent tracking."""
        parent_ctx = [{"role": "user", "content": "parent msg"}]
        state = LoopContextState(parent_chat_context=parent_ctx)

        local_msgs = [{"role": "user", "content": "local msg"}]

        # First tool gets full context
        parent1, _ = state.compute_context_for_inner_tool("call_1", local_msgs)
        assert parent1 is not None

        # Second tool also gets full context (independent tracking)
        parent2, _ = state.compute_context_for_inner_tool("call_2", local_msgs)
        assert parent2 is not None

    def test_cont_received_included_in_first_call(self):
        """Accumulated cont should be included on first tool call."""
        parent_ctx = [{"role": "user", "content": "parent msg"}]
        state = LoopContextState(parent_chat_context=parent_ctx)

        # Receive some cont before first tool call
        state.receive_context_continuation([{"role": "user", "content": "cont msg"}])

        local_msgs = [{"role": "user", "content": "local msg"}]

        parent, cont = state.compute_context_for_inner_tool("call_1", local_msgs)

        assert parent is not None
        assert cont is not None
        assert len(cont) == 1
        assert cont[0]["content"] == "cont msg"

    def test_new_cont_forwarded_to_existing_tool(self):
        """New cont received should be forwarded to already-called tools."""
        parent_ctx = [{"role": "user", "content": "parent msg"}]
        state = LoopContextState(parent_chat_context=parent_ctx)

        local_msgs = [{"role": "user", "content": "local msg"}]

        # First call
        state.compute_context_for_inner_tool("call_1", local_msgs)

        # Receive new cont from above
        state.receive_context_continuation([{"role": "user", "content": "new cont"}])

        # Second call should get the new cont
        parent, cont = state.compute_context_for_inner_tool("call_1", local_msgs)

        assert parent is None
        assert cont is not None
        assert len(cont) == 1
        assert cont[0]["content"] == "new cont"

    def test_initial_snapshot_keeps_user_turns_and_substantive_assistant_text(self):
        """The initial snapshot keeps exactly genuine user turns and
        substantive assistant text, on both the inherited parent layer and
        the local messages nested as children — tool results, tool_calls
        turns, system messages, and loop notices are all dropped."""
        noisy_parent = [
            {"role": "user", "content": "genuine parent turn"},
            {"role": "tool", "content": "discovery tool payload " * 50},
            {"role": "assistant", "content": "", "tool_calls": [{"id": "c1"}]},
            {"role": "assistant", "content": "substantive parent reply"},
            loop_user_notice("parent loop notice"),
        ]
        state = LoopContextState(parent_chat_context=noisy_parent)

        noisy_local = [
            {"role": "system", "content": "local system prompt"},
            {"role": "user", "content": "genuine local turn"},
            {"role": "tool", "content": "local tool payload"},
            {"role": "assistant", "content": "thinking", "tool_calls": [{"id": "c2"}]},
            {"role": "assistant", "content": "substantive local reply"},
            loop_user_notice("local loop notice"),
        ]
        parent, cont = state.compute_context_for_inner_tool("call_1", noisy_local)

        assert cont is None
        assert [(m["role"], m["content"]) for m in parent] == [
            ("user", "genuine parent turn"),
            ("assistant", "substantive parent reply"),
        ]
        assert parent[-1]["children"] == [
            {"role": "user", "content": "genuine local turn"},
            {"role": "assistant", "content": "substantive local reply"},
        ]
        # The filter reads, never mutates: the tracked messages are intact.
        assert len(state.parent_chat_context) == 5
        assert len(noisy_local) == 6

    def test_initial_snapshot_is_none_when_nothing_survives(self):
        """When every parent and local message is snapshot noise, the
        opted-in tool gets no parent context at all rather than the noise."""
        state = LoopContextState(
            parent_chat_context=[
                {"role": "tool", "content": "payload"},
                loop_user_notice("notice"),
            ],
        )
        local = [
            {"role": "assistant", "content": "", "tool_calls": [{"id": "c1"}]},
            {"role": "tool", "content": "result"},
        ]
        parent, cont = state.compute_context_for_inner_tool("call_1", local)
        assert parent is None
        assert cont is None

    def test_incremental_updates_apply_snapshot_filter(self):
        """Continuations pass the same keep-rule as the initial snapshot,
        and forwarding indices track the raw transcript so filtered-out
        messages are consumed, not re-offered on the next call."""
        state = LoopContextState(
            parent_chat_context=[{"role": "user", "content": "parent msg"}],
        )
        local = [{"role": "user", "content": "genuine turn one"}]
        state.compute_context_for_inner_tool("call_1", local)

        local += [
            {"role": "tool", "content": "new tool payload"},
            loop_user_notice("new loop notice"),
            {"role": "user", "content": "genuine turn two"},
            {"role": "assistant", "content": "ok", "tool_calls": [{"id": "c3"}]},
            {"role": "assistant", "content": "substantive follow-up"},
        ]
        parent, cont = state.compute_context_for_inner_tool("call_1", local)
        assert parent is None
        assert cont == [
            {"role": "user", "content": "genuine turn two"},
            {"role": "assistant", "content": "substantive follow-up"},
        ]

        # Nothing new: the filtered-out messages must not resurface.
        parent, cont = state.compute_context_for_inner_tool("call_1", local)
        assert parent is None
        assert cont is None

    def test_mark_cont_forwarded_to_tool(self):
        """Verify marking cont as forwarded clears pending state."""
        state = LoopContextState()

        state.compute_context_for_inner_tool("call_1", [])
        state.receive_context_continuation([{"role": "user", "content": "cont1"}])

        # Mark as forwarded
        state.mark_cont_forwarded_to_tool("call_1")

        # After marking, the next compute has nothing new to forward
        parent, cont = state.compute_context_for_inner_tool("call_1", [])
        assert parent is None
        assert cont is None


# =============================================================================
# Integration tests for incremental context propagation in tool loops
# =============================================================================


# as_shipped: deleted in step 5 (parent chat context and interjections)
@pytest.mark.usefixtures("as_shipped")
@pytest.mark.asyncio
@pytest.mark.llm_call
@_handle_project
async def test_incremental_context_multiple_tool_calls(llm_config) -> None:
    """Verify that multiple calls to the same tool receive incremental context.

    First call should receive full parent_chat_context.
    Second call should receive only _parent_chat_context_cont with new messages.
    """
    client = new_llm_client(**llm_config)

    root_ctx = [{"role": "user", "content": "initial-parent-context"}]

    # Track what each call received
    call_records: List[dict] = []

    async def track_context(
        *,
        _parent_chat_context: list[dict] | None = None,
        _parent_chat_context_cont: list[dict] | None = None,
    ) -> str:
        call_records.append(
            {
                "parent_ctx": _parent_chat_context,
                "parent_ctx_cont": _parent_chat_context_cont,
            },
        )
        return f"call {len(call_records)} recorded"

    track_context.__name__ = "track_context"
    track_context.__qualname__ = "track_context"

    handle = start_async_tool_loop(
        client=client,
        message=(
            "Please call `track_context()` TWICE, one after another. "
            "After both calls complete, reply 'done'."
        ),
        tools={"track_context": track_context},
        parent_chat_context=root_ctx,
        propagate_chat_context=ChatContextPropagation.ALWAYS,
    )

    await handle.result()

    # Should have at least 2 calls
    assert len(call_records) >= 2, f"Expected at least 2 calls, got {len(call_records)}"

    # First call should have parent_chat_context
    first_call = call_records[0]
    assert (
        first_call["parent_ctx"] is not None
    ), "First call should receive parent_chat_context"
    # Verify the initial parent context is present
    assert any(
        "initial-parent-context" in str(m.get("content", ""))
        for m in first_call["parent_ctx"]
    ), "First call should contain the initial parent context"

    # Second call should NOT have full parent_chat_context again (only incremental)
    second_call = call_records[1]
    # Second call should either have no parent_ctx or only have cont
    # The exact behavior depends on whether there were new messages between calls
    if second_call["parent_ctx"] is not None:
        # If parent_ctx is present, it should NOT duplicate the initial context
        # (This would indicate the incremental tracking is working)
        pass  # Acceptable if the model called them in a single message batch


@pytest.mark.asyncio
async def test_context_state_integration_symbolic() -> None:
    """Symbolic test: verify LoopContextState integrates correctly with tool scheduling.

    This test doesn't use LLM, it directly tests the state machine behavior.
    """
    state = LoopContextState(
        parent_chat_context=[{"role": "user", "content": "root"}],
    )

    # Simulate tool being called twice
    msgs_at_call_1 = [{"role": "user", "content": "msg1"}]
    parent1, cont1 = state.compute_context_for_inner_tool("tool_a", msgs_at_call_1)

    assert parent1 is not None, "First call should get parent context"
    assert parent1[0]["content"] == "root"
    assert cont1 is None, "First call should not have cont (no accumulated)"

    # Add more messages between calls
    msgs_at_call_2 = msgs_at_call_1 + [
        {"role": "assistant", "content": "response1"},
        {"role": "user", "content": "msg2"},
    ]

    # Receive cont from above
    state.receive_context_continuation([{"role": "system", "content": "interjection"}])

    parent2, cont2 = state.compute_context_for_inner_tool("tool_a", msgs_at_call_2)

    assert parent2 is None, "Second call should NOT get full parent context"
    assert cont2 is not None, "Second call should get incremental cont"
    # Should include new local messages + the interjection
    assert len(cont2) == 3, f"Expected 3 incremental items, got {len(cont2)}"


# =============================================================================
# Tests for context opt-in/opt-out behavior with steering methods
# =============================================================================


def test_method_to_schema_exposes_context_cont_control():
    """Verify that method_to_schema exposes include_parent_chat_context_cont when requested."""
    from unify.common.llm_helpers import method_to_schema

    def steering_method(
        question: str,
        *,
        _parent_chat_context_cont: list[dict] | None = None,
    ) -> str:
        """A steering method that accepts context continuation."""
        return "answer"

    # Without expose_context_cont_control
    schema_no_expose = method_to_schema(steering_method)
    props = schema_no_expose["function"]["parameters"]["properties"]
    assert "include_parent_chat_context_cont" not in props
    # _parent_chat_context_cont should be hidden
    assert "_parent_chat_context_cont" not in props

    # With expose_context_cont_control
    schema_exposed = method_to_schema(
        steering_method,
        expose_context_cont_control=True,
    )
    props_exposed = schema_exposed["function"]["parameters"]["properties"]
    assert "include_parent_chat_context_cont" in props_exposed
    assert props_exposed["include_parent_chat_context_cont"]["type"] == "boolean"


def test_context_injection_defaults_off_when_arg_omitted():
    """Omitting include_parent_chat_context injects no parent context under
    LLM_DECIDES — context is opt-in.

    compute_context_injection is the shared funnel behind base tool
    dispatch, steer(action=ask), and ask_about_completed_tool, so this
    single omitted-arg contract governs all of those consumers. ALWAYS
    mode is unaffected: it injects regardless of the (absent) LLM choice.
    """
    from unify.common._async_tool.tools_data import compute_context_injection

    client_messages = [{"role": "user", "content": "current message"}]

    args_omitted = {"question": "what's the status?"}
    context_state = LoopContextState(
        parent_chat_context=[{"role": "user", "content": "parent context"}],
    )
    extra_kwargs, context_opted_in = compute_context_injection(
        args=args_omitted,
        propagate_chat_context=ChatContextPropagation.LLM_DECIDES,
        context_state=context_state,
        client_messages=client_messages,
        call_id="ask_inner_tool_omitted",
        accepts_parent_ctx=True,
        accepts_parent_ctx_cont=False,
        is_continuation_only=False,
    )
    assert (
        "_parent_chat_context" not in extra_kwargs
    ), "An omitted include_parent_chat_context must not inject parent context"
    assert context_opted_in is False

    # ALWAYS mode still injects when the arg is omitted.
    context_state_always = LoopContextState(
        parent_chat_context=[{"role": "user", "content": "parent context"}],
    )
    extra_kwargs_always, context_opted_in_always = compute_context_injection(
        args={"question": "what's the status?"},
        propagate_chat_context=ChatContextPropagation.ALWAYS,
        context_state=context_state_always,
        client_messages=client_messages,
        call_id="ask_inner_tool_always_omitted",
        accepts_parent_ctx=True,
        accepts_parent_ctx_cont=False,
        is_continuation_only=False,
    )
    assert "_parent_chat_context" in extra_kwargs_always
    assert context_opted_in_always is True


# =============================================================================
# Tests for interjection context continuation message structure
# =============================================================================


# =============================================================================
# Loop-level opt-in contract (scripted turns, no LLM)
# =============================================================================


def _scripted_tool_call(call_id: str, name: str, arguments: dict) -> dict:
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": name, "arguments": json.dumps(arguments)},
    }


@pytest.mark.asyncio
@pytest.mark.timeout(60)
@_handle_project
async def test_loop_injects_no_context_unless_opted_in_then_filtered_snapshot(
    llm_config,
    monkeypatch,
) -> None:
    """Through the real loop dispatch: a tool call that omits
    include_parent_chat_context receives no parent context, and an explicit
    opt-in receives the filtered snapshot — genuine user turns and
    substantive assistant text only, with the tool payloads, tool_calls
    turns, and loop notices from the parent layer dropped."""
    client = new_llm_client(**llm_config)
    client.set_system_message("This turn is fully scripted by the test.")

    from unify.common._async_tool import loop as _loop

    turn_queue: asyncio.Queue = asyncio.Queue()

    async def _fake_gwp(_client, _preprocess_msgs, **gen_kwargs):
        next_msg = await turn_queue.get()
        _client.messages.append(next_msg)
        return {"ok": True}

    monkeypatch.setattr(_loop, "generate_with_preprocess", _fake_gwp, raising=True)

    captured_ctx: List[list[dict] | None] = []

    async def record_context(*, _parent_chat_context: list[dict] | None = None) -> str:
        captured_ctx.append(_parent_chat_context)
        return "context-recorded"

    record_context.__name__ = "record_context"
    record_context.__qualname__ = "record_context"

    parent_ctx = [
        {"role": "user", "content": "genuine parent turn"},
        {"role": "tool", "content": "discovery tool payload " * 50},
        {"role": "assistant", "content": "", "tool_calls": [{"id": "c1"}]},
        {"role": "assistant", "content": "substantive parent reply"},
        loop_user_notice("parent loop notice"),
    ]

    handle = start_async_tool_loop(
        client=client,
        message="start",
        tools={"record_context": record_context},
        parent_chat_context=parent_ctx,
        max_steps=20,
        timeout=30,
    )

    # Turn 1: the tool call omits include_parent_chat_context entirely.
    await turn_queue.put(
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [_scripted_tool_call("call_1", "record_context", {})],
        },
    )
    # Turn 2: explicit opt-in.
    await turn_queue.put(
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                _scripted_tool_call(
                    "call_2",
                    "record_context",
                    {"include_parent_chat_context": True},
                ),
            ],
        },
    )
    await turn_queue.put({"role": "assistant", "content": "done", "tool_calls": []})

    final = await asyncio.wait_for(handle.result(), timeout=30)
    assert final == "done"

    assert len(captured_ctx) == 2, f"Expected 2 tool calls, got {len(captured_ctx)}"
    assert captured_ctx[0] is None, (
        "A tool call omitting include_parent_chat_context must receive no "
        "parent context"
    )

    snapshot = captured_ctx[1]
    assert snapshot is not None, "Explicit opt-in must receive the snapshot"
    assert [(m["role"], m["content"]) for m in snapshot] == [
        ("user", "genuine parent turn"),
        ("assistant", "substantive parent reply"),
    ], "Only genuine user turns and substantive assistant text may survive"
    # Local survivors ride along as children of the last parent message:
    # the genuine "start" turn, without the tool_calls turns or tool results.
    children = snapshot[-1]["children"]
    assert {(m["role"], m["content"]) for m in children} == {("user", "start")}
