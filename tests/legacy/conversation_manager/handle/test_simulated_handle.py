"""Symbolic: the simulated conversation manager handle keeps one stateful
conversation with its model.

The model is the scripted transport (``tests/scripted_model.py``). The
handle's model calls carry no tools, so each is a ``query_llm`` request. The
tests script what the simulated user says and check what the handle sends
and returns: the scenario in every system message, each question and
interjection, the earlier exchanges carried into the next request, and a
structured answer parsed into the requested model.
"""

from __future__ import annotations

import asyncio
import json

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
from tests.scripted_model import ScriptedModel, reply, scripted


def _text(content) -> str:
    """A message's text, whether a string or a list of content parts."""
    if isinstance(content, list):
        return "\n".join(
            str(part.get("text", "")) if isinstance(part, dict) else str(part)
            for part in content
        )
    return "" if content is None else str(content)


def _prompt(call) -> str:
    """The request's last message: what this call asked."""
    return _text(call.messages[-1]["content"])


def _system(call) -> str:
    """The request's system message."""
    return "\n".join(
        _text(m["content"]) for m in call.messages if m["role"] == "system"
    )


def _carried(call, text: str, role: str) -> bool:
    """Whether a *role* message before the request's last one contains *text*."""
    return any(
        m["role"] == role and text in _text(m["content"]) for m in call.messages[:-1]
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
@_handle_project
async def test_start_and_ask():
    said = "Hi, this is Sam. I'm the one who asked you to sort out my travel."
    model = ScriptedModel(query_llm=[reply(said)])
    with scripted(model):
        cm_handle = SimulatedConversationManagerHandle(
            assistant_id="test_assistant",
        )
        # The 'ask' method on the main handle returns another handle
        answer_handle = await cm_handle.ask("Hello, who is this?")
        # The question goes to the model when its answer is awaited.
        assert model.calls == [] and not answer_handle.done()
        answer = await answer_handle.result()
    assert answer == said, "the handle answers with what the simulated user said"
    assert answer_handle.done()

    (call,) = model.of("query_llm")
    assert '"Hello, who is this?"' in _prompt(call)
    assert "test_assistant" in _system(call)
    model.assert_used_up()


# ────────────────────────────────────────────────────────────────────────────
# 2.  Stateful memory – serial asks
# ────────────────────────────────────────────────────────────────────────────
@pytest.mark.asyncio
@_handle_project
async def test_stateful_memory_serial_asks():
    """
    Two consecutive .ask() calls should share the same conversation context
    via the stateful LLM.

    A unique token seeded in the first turn must reach the model again with
    the second question, inside the first exchange the handle carries.
    """
    token = _unique_token("MARKER")
    model = ScriptedModel(
        query_llm=[
            reply(f"All on track for today. Status marker: {token}."),
            reply(f"The marker was {token}."),
        ],
    )
    with scripted(model):
        cm_handle = SimulatedConversationManagerHandle(
            assistant_id="test_assistant",
        )

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
    assert token in answer2

    first, second = model.of("query_llm")
    assert token in _prompt(first)
    assert "Quote it verbatim" in _prompt(second)
    # The stateful client sends the whole first exchange with the second ask.
    assert _carried(second, token, "user"), "the first question is carried"
    assert _carried(second, first_answer, "assistant"), "the first answer is carried"
    model.assert_used_up()


# ────────────────────────────────────────────────────────────────────────────
# 3.  Interject then ask – state propagated
# ────────────────────────────────────────────────────────────────────────────
@pytest.mark.asyncio
@_handle_project
async def test_stateful_interject_then_ask():
    """
    A .interject() call should influence a subsequent .ask() call: the
    interjection reaches the model, and the next question carries it.
    """
    model = ScriptedModel(
        query_llm=[
            reply("Noted: the meeting is now at 4 PM."),
            reply("Your meeting is now at 4 PM."),
        ],
    )
    with scripted(model):
        cm_handle = SimulatedConversationManagerHandle(
            assistant_id="test_assistant",
        )

        # 1) Tell the manager about a schedule change
        interjection_id = await cm_handle.interject(
            "The user's meeting has been rescheduled to 4 PM.",
        )
        assert isinstance(interjection_id, str) and interjection_id

        # 2) Ask about the meeting time – it should know about the update
        h_q = await cm_handle.ask("What time is my meeting now?")
        answer = (await h_q.result()).lower()

    assert "4 pm" in answer, "the simulated user's answer is returned"

    interjection, question = model.of("query_llm")
    assert "rescheduled to 4 PM" in _prompt(interjection)
    assert "What time is my meeting now?" in _prompt(question)
    assert _carried(question, "rescheduled to 4 PM", "user")
    model.assert_used_up()


# ────────────────────────────────────────────────────────────────────────────
# 4. Test description and simulation_guidance
# ────────────────────────────────────────────────────────────────────────────
@pytest.mark.asyncio
@_handle_project
async def test_description_and_guidance():
    """
    The description and simulation_guidance parameters reach the model in
    the system message of every request the handle makes.
    """
    description = "A conversation with a pirate captain who has lost his treasure."
    guidance = "You must always respond in pirate slang. Refer to the user as 'matey'."
    model = ScriptedModel(
        query_llm=[
            reply("Ahoy, matey! I be the cap'n, and me treasure be lost."),
            reply("That be Paris, matey."),
        ],
    )
    with scripted(model):
        cm_handle = SimulatedConversationManagerHandle(
            assistant_id="pirate_assistant",
            description=description,
            simulation_guidance=guidance,
        )

        h1 = await cm_handle.ask("Who are you?")
        answer1 = await h1.result()
        h2 = await cm_handle.ask("What is the capital of France?")
        answer2 = await h2.result()

    assert answer1 == "Ahoy, matey! I be the cap'n, and me treasure be lost."
    assert answer2 == "That be Paris, matey."
    calls = model.of("query_llm")
    assert len(calls) == 2
    for call in calls:
        system = _system(call)
        assert description in system
        assert guidance in system
        assert "pirate_assistant" in system
    model.assert_used_up()


# ────────────────────────────────────────────────────────────────────────────
# 5. Test ask with structured output (Pydantic)
# ────────────────────────────────────────────────────────────────────────────
@pytest.mark.asyncio
@_handle_project
async def test_ask_with_structured_output():
    """
    The ask() method asks for the response format and parses the answer
    into the Pydantic model.
    """

    # Define the Pydantic model for the expected response
    class UserInfo(BaseModel):
        name: str = Field(description="The user's full name.")
        is_happy: bool = Field(description="Whether the user is currently happy.")
        age: int = Field(description="The user's age.")

    model = ScriptedModel(
        query_llm=[reply('{"name": "John Doe", "is_happy": true, "age": 30}')],
    )
    with scripted(model):
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

    (call,) = model.of("query_llm")
    assert "FORMAT INSTRUCTIONS" in _prompt(call)
    assert "UserInfo" in json.dumps(call.request["response_format"])
    assert "John Doe" in _system(call)
    model.assert_used_up()


# ────────────────────────────────────────────────────────────────────────────
#  Steerable handle tests
# ────────────────────────────────────────────────────────────────────────────


# ────────────────────────────────────────────────────────────────────────────
# 6.  Interject
# ────────────────────────────────────────────────────────────────────────────
@pytest.mark.asyncio
@_handle_project
async def test_interject(monkeypatch):
    """Verify that interject is called and influences the conversation state."""
    model = ScriptedModel(
        query_llm=[
            reply("Understood: the flight is delayed."),
            reply("Your flight has been delayed."),
        ],
    )
    with scripted(model):
        cm_handle = SimulatedConversationManagerHandle(
            assistant_id="test_assistant",
        )

        # Interject with new information
        await cm_handle.interject("The user's flight has been delayed.")

        # Ask a question that should be influenced by the interjection
        h_q = await cm_handle.ask("What is the status of my flight?")
        answer = (await h_q.result()).lower()

    assert "delayed" in answer, "the simulated user's answer is returned"

    interjection, question = model.of("query_llm")
    assert "flight has been delayed" in _prompt(interjection)
    assert _carried(question, "flight has been delayed", "user")
    model.assert_used_up()


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
@_handle_project
async def test_handle_ask():
    """
    The `ask()` method on the handle should return its own handle, and state should
    be preserved across these interactions.
    """
    model = ScriptedModel(
        query_llm=[
            reply("Sure, London's been grey and drizzly all week."),
            reply("You asked about London."),
        ],
    )
    with scripted(model):
        cm_handle = SimulatedConversationManagerHandle(
            assistant_id="test_assistant",
        )

        # Start an initial ask to obtain a live handle
        h1 = await cm_handle.ask("Let's talk about the weather in London.")
        await h1.result()
        # A nested ask on an answer handle is that same handle.
        assert await h1.ask("Anything else?") is h1

        # Invoke a nested ask on the same handle
        h2 = await cm_handle.ask("What city did I just ask about?")
        nested_answer = await h2.result()
    assert "london" in nested_answer.lower()

    first, second = model.of("query_llm")
    assert "weather in London" in _prompt(first)
    # The second ask carries the first exchange.
    assert _carried(second, "weather in London", "user")
    assert _carried(second, "grey and drizzly", "assistant")
    model.assert_used_up()


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
