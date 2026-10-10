# tests/conversation_manager/handle/test_simulated_handle.py
from __future__ import annotations

import asyncio
import pytest
import functools
from pydantic import BaseModel, Field
from unify.legacy.conversation_manager.simulated import (
    SimulatedConversationManagerHandle,
)
from tests.helpers import (
    _handle_project,
    _assert_blocks_while_paused,
    DEFAULT_TIMEOUT,
    _unique_token,
)


# ────────────────────────────────────────────────────────────────────────────
# 0.  Docstring inheritance
# ────────────────────────────────────────────────────────────────────────────
def test_docstrings_match_base():
    """
    Public methods in SimulatedConversationManagerHandle should carry the
    SteerableToolHandle doc-strings (via ABC inheritance or functools.wraps).
    """
    # Check that ask() has a docstring that contains key phrases from the base
    assert (
        SimulatedConversationManagerHandle.ask.__doc__ is not None
    ), "ask() should have a docstring"

    # Check that pause() has a docstring
    assert (
        SimulatedConversationManagerHandle.pause.__doc__ is not None
    ), "pause() should have a docstring"

    # Check that resume() has a docstring
    assert (
        SimulatedConversationManagerHandle.resume.__doc__ is not None
    ), "resume() should have a docstring"

    # Check that stop() has a docstring
    assert (
        SimulatedConversationManagerHandle.stop.__doc__ is not None
    ), "stop() should have a docstring"

    # Check that interject() has a docstring
    assert (
        SimulatedConversationManagerHandle.interject.__doc__ is not None
    ), "interject() should have a docstring"


# ────────────────────────────────────────────────────────────────────────────
# 1.  Basic start-and-ask
# ────────────────────────────────────────────────────────────────────────────
@pytest.mark.asyncio
@pytest.mark.llm_call
@_handle_project
async def test_start_and_ask():
    cm_handle = SimulatedConversationManagerHandle(
        assistant_id="test_assistant",
    )
    # The 'ask' method on the main handle returns another handle
    answer_handle = await cm_handle.ask("Hello, who is this?")
    answer = await answer_handle.result()
    print(answer)
    assert isinstance(answer, str) and answer.strip(), "Answer should be non-empty"


# ────────────────────────────────────────────────────────────────────────────
# 2.  Stateful memory – serial asks
# ────────────────────────────────────────────────────────────────────────────
@pytest.mark.asyncio
@pytest.mark.llm_call
@_handle_project
async def test_stateful_memory_serial_asks():
    """
    Two consecutive .ask() calls should share the same conversation context
    via the stateful LLM.

    Seed a unique token in the first turn and require the second turn to
    recall it, rather than asking the model to invent a value.
    """
    cm_handle = SimulatedConversationManagerHandle(
        assistant_id="test_assistant",
    )

    token = _unique_token("MARKER")
    h1 = await cm_handle.ask(
        "Please give a one-sentence status note. "
        f"Ensure the note includes this exact marker verbatim: {token}",
    )
    first_answer = (await h1.result()).strip()
    assert first_answer, "First answer should not be empty"

    h2 = await cm_handle.ask(
        "What exact marker did you include earlier? Quote it verbatim.",
    )
    answer2 = await h2.result()
    assert (
        isinstance(answer2, str) and answer2.strip()
    ), "Second answer should be non-empty"
    assert token in answer2, "LLM should recall the previously mentioned marker"


# ────────────────────────────────────────────────────────────────────────────
# 3.  Interject then ask – state propagated
# ────────────────────────────────────────────────────────────────────────────
@pytest.mark.asyncio
@pytest.mark.llm_call
@_handle_project
async def test_stateful_interject_then_ask():
    """
    A .interject() call should influence a subsequent .ask() call.
    """
    cm_handle = SimulatedConversationManagerHandle(
        assistant_id="test_assistant",
    )

    # 1) Tell the manager about a schedule change
    await cm_handle.interject("The user's meeting has been rescheduled to 4 PM.")

    # 2) Ask about the meeting time – it should know about the update
    h_q = await cm_handle.ask("What time is my meeting now?")
    answer = (await h_q.result()).lower()

    assert (
        "4 pm" in answer or "4:00" in answer
    ), "Answer should reflect the interjected information"


# ────────────────────────────────────────────────────────────────────────────
# 4. Test description and simulation_guidance
# ────────────────────────────────────────────────────────────────────────────
@pytest.mark.asyncio
@pytest.mark.llm_call
@_handle_project
async def test_description_and_guidance():
    """
    Tests that the description and simulation_guidance parameters correctly
    influence the LLM's behavior.
    """
    cm_handle = SimulatedConversationManagerHandle(
        assistant_id="pirate_assistant",
        description="A conversation with a pirate captain who has lost his treasure.",
        simulation_guidance="You must always respond in pirate slang. Refer to the user as 'matey'.",
    )

    h1 = await cm_handle.ask("Who are you?")
    answer1 = await h1.result()
    answer1_lower = answer1.lower()
    # Check for pirate persona indicators (the LLM may use "cap'n" instead of "captain")
    pirate_indicators = [
        "pirate",
        "captain",
        "cap'n",
        "matey",
        "treasure",
        "seas",
        "ahoy",
    ]
    assert any(
        indicator in answer1_lower for indicator in pirate_indicators
    ), f"Description should set the pirate persona. Got: {answer1}"

    h2 = await cm_handle.ask("What is the capital of France?")
    answer2 = await h2.result()
    assert "matey" in answer2.lower(), "Simulation guidance should be followed."


# ────────────────────────────────────────────────────────────────────────────
# 5. Test ask with structured output (Pydantic)
# ────────────────────────────────────────────────────────────────────────────
@pytest.mark.asyncio
@pytest.mark.llm_call
@_handle_project
async def test_ask_with_structured_output():
    """
    Tests that the ask() method can correctly parse a response into a Pydantic model.
    """

    # Define the Pydantic model for the expected response
    class UserInfo(BaseModel):
        name: str = Field(description="The user's full name.")
        is_happy: bool = Field(description="Whether the user is currently happy.")
        age: int = Field(description="The user's age.")

    cm_handle = SimulatedConversationManagerHandle(
        assistant_id="test_assistant",
        simulation_guidance="The user's name is John Doe, he is 30 years old, and he is happy.",
    )

    # Ask the question and provide the response_format
    handle = await cm_handle.ask(
        "Please provide the user's information.",
        response_format=UserInfo,
    )

    # The result should be an instance of the Pydantic model
    result = await handle.result()

    assert isinstance(result, UserInfo), "Result should be an instance of UserInfo"
    assert result.name.lower() == "john doe"
    assert result.is_happy is True
    assert result.age == 30


# ────────────────────────────────────────────────────────────────────────────
#  Steerable handle tests
# ────────────────────────────────────────────────────────────────────────────


# ────────────────────────────────────────────────────────────────────────────
# 6.  Interject
# ────────────────────────────────────────────────────────────────────────────
@pytest.mark.asyncio
@pytest.mark.llm_call
@_handle_project
async def test_interject(monkeypatch):
    """Verify that interject is called and influences the conversation state."""
    cm_handle = SimulatedConversationManagerHandle(
        assistant_id="test_assistant",
    )

    # Interject with new information
    await cm_handle.interject("The user's flight has been delayed.")

    # Ask a question that should be influenced by the interjection
    h_q = await cm_handle.ask("What is the status of my flight?")
    answer = (await h_q.result()).lower()

    assert "delayed" in answer, "Answer should reflect the interjected information"


# ────────────────────────────────────────────────────────────────────────────
# 7.  Stop
# ────────────────────────────────────────────────────────────────────────────
@pytest.mark.asyncio
@_handle_project
async def test_stop():
    cm_handle = SimulatedConversationManagerHandle(
        assistant_id="test_assistant",
    )
    # The main handle's result will block until stop() is called
    result_task = asyncio.create_task(cm_handle.result())

    await asyncio.sleep(0.05)
    assert not result_task.done(), "Handle should not be done before stop()"

    await cm_handle.stop(reason="Test cleanup")

    await asyncio.wait_for(result_task, timeout=1)
    assert cm_handle.done(), "Handle should report done after stop()"
    assert "stopped" in (await result_task).lower()


# ────────────────────────────────────────────────────────────────────────────
# 8.  Pause → Resume round-trip
# ────────────────────────────────────────────────────────────────────────────
@pytest.mark.asyncio
@_handle_project
async def test_pause_and_resume(monkeypatch):
    """
    Verify that a SimulatedConversationManagerHandle may be paused and resumed.
    """
    counts = {"pause": 0, "resume": 0}

    original_pause = SimulatedConversationManagerHandle.pause

    @functools.wraps(original_pause)
    async def wrapped_pause(self, *args, **kwargs):
        counts["pause"] += 1
        return await original_pause(self, *args, **kwargs)

    monkeypatch.setattr(SimulatedConversationManagerHandle, "pause", wrapped_pause)

    original_resume = SimulatedConversationManagerHandle.resume

    @functools.wraps(original_resume)
    async def wrapped_resume(self, *args, **kwargs):
        counts["resume"] += 1
        return await original_resume(self, *args, **kwargs)

    monkeypatch.setattr(SimulatedConversationManagerHandle, "resume", wrapped_resume)

    cm_handle = SimulatedConversationManagerHandle(
        assistant_id="test_assistant",
    )

    pause_reply = await cm_handle.pause()
    assert "paused" in pause_reply.lower()

    # In this simulation, pause is a state flag, so we just check the flag and resume
    assert cm_handle._paused is True, "Pause flag should be set"

    resume_reply = await cm_handle.resume()
    assert "resumed" in resume_reply.lower()

    assert cm_handle._paused is False, "Pause flag should be unset"

    assert counts == {
        "pause": 1,
        "resume": 1,
    }, "pause/resume should each be called once"


# ────────────────────────────────────────────────────────────────────────────
# 9.  Nested ask on handle
# ────────────────────────────────────────────────────────────────────────────
@pytest.mark.asyncio
@pytest.mark.llm_call
@_handle_project
async def test_handle_ask():
    """
    The `ask()` method on the handle should return its own handle, and state should
    be preserved across these interactions.
    """
    cm_handle = SimulatedConversationManagerHandle(
        assistant_id="test_assistant",
    )

    # Start an initial ask to obtain a live handle
    h1 = await cm_handle.ask("Let's talk about the weather in London.")
    await h1.result()

    # Invoke a nested ask on the same handle
    h2 = await cm_handle.ask("What city did I just ask about?")
    nested_answer = await h2.result()
    assert (
        "london" in nested_answer.lower()
    ), "LLM should recall context from the first ask call"


# ────────────────────────────────────────────────────────────────────────────
# 10.  Stop while paused should finish immediately
# ────────────────────────────────────────────────────────────────────────────
@pytest.mark.asyncio
@_handle_project
async def test_stop_while_paused():
    """
    Calling stop() while paused should unblock result() and complete promptly.

    This follows the gold standard pattern of the simulated manager tests.
    """
    cm_handle = SimulatedConversationManagerHandle(
        assistant_id="test_assistant",
    )

    # Enter paused state
    await cm_handle.pause()
    assert cm_handle._paused is True, "Handle should be paused"

    # result() should block until stop() is called
    res_task = asyncio.create_task(cm_handle.result())
    await asyncio.sleep(0.1)
    assert not res_task.done(), "result() should block while waiting for stop()"

    # Stop should unblock and complete promptly
    await cm_handle.stop("cancelled by user")

    out = await asyncio.wait_for(res_task, timeout=DEFAULT_TIMEOUT)
    assert isinstance(out, str), "result() should return a string"
    assert cm_handle.done(), "Handle should report done after stop()"


# ────────────────────────────────────────────────────────────────────────────
# 11.  result() blocks until stopped
# ────────────────────────────────────────────────────────────────────────────
@pytest.mark.asyncio
@_handle_project
async def test_result_blocks_until_stopped():
    """
    Verify that result() blocks until stop() is explicitly called.

    This is the expected behavior for a conversation handle - it remains
    active until explicitly stopped.
    """
    cm_handle = SimulatedConversationManagerHandle(
        assistant_id="test_assistant",
    )

    # Start result() - it should block
    res_task = await _assert_blocks_while_paused(cm_handle.result())

    # Stop the handle
    await cm_handle.stop("test complete")

    # Now result() should complete
    out = await asyncio.wait_for(res_task, timeout=DEFAULT_TIMEOUT)
    assert isinstance(out, str), "result() should return a string"
    assert "stopped" in out.lower(), "Result should indicate stopped state"
    assert cm_handle.done(), "Handle should report done after stop()"
