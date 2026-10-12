from __future__ import annotations

import pytest
from pydantic import BaseModel, Field

from unify.common.async_tool_loop import start_async_tool_loop
from tests.helpers import _handle_project
from unify.common.llm_client import new_llm_client
from tests.scripted_model import ScriptedModel, reply, scripted


def _final_response(answer: dict):
    """A scripted reply that submits *answer* through ``final_response``."""
    return reply(calls=[("final_response", {"answer": answer})])


class SimpleGreeting(BaseModel):
    """Minimal schema used to validate structured-output support."""

    greeting: str = Field(..., description="A friendly greeting message.")
    lucky_number: int = Field(..., description="Any integer chosen by the model.")


@pytest.mark.asyncio
@_handle_project
async def test_structured_output_response_format(llm_config) -> None:
    """The async-tool loop should honour *response_format* and return JSON that
    validates against the supplied Pydantic schema.

    Flow:
      1. Run the loop **without** any tools of its own.
      2. The loop offers its ``final_response`` tool, whose argument is the
         `SimpleGreeting` schema, and forces a tool call.
      3. The model's call is validated, and ``result()`` returns the model.
    """

    model = ScriptedModel(
        actor=[
            _final_response({"greeting": "Hello, I am a test.", "lucky_number": 7}),
        ],
    )
    with scripted(model):
        client = new_llm_client(**llm_config)

        client.set_system_message(
            "When asked, respond with a JSON object that contains exactly two keys: "
            "'greeting' (a friendly greeting) and 'lucky_number' (an integer). Do not "
            "include any extra keys or commentary.",
        )

        handle = start_async_tool_loop(  # type: ignore[attr-defined]
            client,
            message="Please introduce yourself and pick a lucky number.",
            tools={},  # no tools needed
            response_format=SimpleGreeting,
            max_steps=8,
            timeout=120,
        )

        final_reply = await handle.result()

        # result() returns a Pydantic model instance when response_format is set.
        assert isinstance(
            final_reply,
            SimpleGreeting,
        ), f"Expected SimpleGreeting instance, got {type(final_reply).__name__}"
        assert final_reply.greeting.strip(), "Greeting must be non-empty"
        assert isinstance(final_reply.lucky_number, int)
    (request,) = model.of("actor")
    assert "final_response" in request.tool_names
    assert request.request["tool_choice"] == "required"
    assert model.kinds() == ["actor"]


# ----------------------------------------------------------------------------
# Ensure trivial response formats are satisfied in the *first* LLM
#  reply so that no follow-up re-formatting step is scheduled.
# ----------------------------------------------------------------------------


class SimpleEcho(BaseModel):
    """Extremely simple schema – just a single string field."""

    text: str = Field(..., description="Echo text sent by the user.")


@pytest.mark.asyncio
@_handle_project
async def test_no_additional_formatting_roundtrip(llm_config) -> None:  # noqa: D401
    """Verify that the loop skips the re-formatting turn when the first
    assistant reply already conforms to *response_format*.

    We check this by asserting that the synthetic follow-up prompt asking the
    model to re-emit its answer ("Please output your previous answer again…")
    is **absent** from the final chat transcript, and that the model was
    asked once.
    """

    model = ScriptedModel(actor=[_final_response({"text": "hi"})])
    with scripted(model):
        client = new_llm_client(**llm_config)

        # Prompt the model so that it can satisfy the schema in one go.
        client.set_system_message(
            "Return a JSON object with a single key 'text' containing any greeting "
            "string. Do NOT add any extra keys, numbers, or commentary.",
        )

        handle = start_async_tool_loop(  # type: ignore[attr-defined]
            client,
            message="Say hi!",
            tools={},
            response_format=SimpleEcho,
            max_steps=6,
            timeout=60,
        )

        final_reply = await handle.result()

        # result() returns a Pydantic model instance when response_format is set.
        assert isinstance(
            final_reply,
            SimpleEcho,
        ), f"Expected SimpleEcho instance, got {type(final_reply).__name__}"
        assert final_reply.text.strip(), "Text must be non-empty"

        # Ensure *no* follow-up formatting prompt was injected.
        assert not any(
            m.get("role") == "user"
            and "Please output your previous answer again" in m.get("content", "")
            for m in client.messages
        ), "Unexpected additional formatting round triggered."
    assert model.kinds() == ["actor"]


# ----------------------------------------------------------------------------
# result() returns a Pydantic model (not a raw JSON string) when
# response_format is supplied.  Tests that the handle-level parsing is
# working end-to-end.
# ----------------------------------------------------------------------------


class MathAnswer(BaseModel):
    """Schema for a simple math calculation result."""

    expression: str = Field(..., description="The mathematical expression evaluated.")
    result: int = Field(..., description="The integer result of the expression.")


@pytest.mark.asyncio
@_handle_project
async def test_result_returns_pydantic_model(llm_config) -> None:
    """handle.result() should return a validated Pydantic model instance, not a
    raw JSON string, when *response_format* is supplied.
    """
    model = ScriptedModel(
        actor=[_final_response({"expression": "7 * 6", "result": 42})],
    )
    with scripted(model):
        client = new_llm_client(**llm_config)
        client.set_system_message(
            "You are a calculator. When asked a math question, respond with a JSON "
            "object containing 'expression' (the expression as a string) and 'result' "
            "(the integer answer). Do not add extra keys or commentary.",
        )

        handle = start_async_tool_loop(
            client,
            message="What is 7 * 6?",
            tools={},
            response_format=MathAnswer,
            max_steps=8,
            timeout=120,
        )

        answer = await handle.result()

        # The core assertion: result() delivers a model, not a string.
        assert isinstance(
            answer,
            MathAnswer,
        ), f"Expected MathAnswer instance, got {type(answer).__name__}: {answer!r}"
        assert answer.result == 42
    assert model.kinds() == ["actor"]


@pytest.mark.asyncio
@_handle_project
async def test_result_returns_string_without_response_format(llm_config) -> None:
    """When *response_format* is not supplied, handle.result() should return a
    plain string as before (backward compatibility).
    """
    model = ScriptedModel(actor=[reply("hello")])
    with scripted(model):
        client = new_llm_client(**llm_config)
        client.set_system_message("Reply with exactly one word: 'hello'.")

        handle = start_async_tool_loop(
            client,
            message="Say the word.",
            tools={},
            max_steps=4,
            timeout=60,
        )

        answer = await handle.result()

        assert isinstance(
            answer,
            str,
        ), f"Expected str, got {type(answer).__name__}: {answer!r}"
    assert answer == "hello"
    assert model.kinds() == ["actor"]
