"""Symbolic: ``handle.result()`` returns a validated model when the loop was given a response format, and the text
otherwise. The model is the scripted transport; a loop with no tools of its own is a ``query_llm`` request to it.
"""

from __future__ import annotations

import json

import pytest
from pydantic import BaseModel, Field

from unify.common.async_tool_loop import start_async_tool_loop
from unify.common.llm_client import new_llm_client
from tests.scripted_model import ScriptedModel, reply, scripted


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
async def test_result_returns_pydantic_model(llm_config) -> None:
    """handle.result() should return a validated Pydantic model instance, not a
    raw JSON string, when *response_format* is supplied.
    """
    model = ScriptedModel(
        query_llm=[reply(json.dumps({"expression": "7 * 6", "result": 42}))],
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
    assert model.kinds() == ["query_llm"]


@pytest.mark.asyncio
async def test_result_returns_string_without_response_format(llm_config) -> None:
    """When *response_format* is not supplied, handle.result() should return a
    plain string as before (backward compatibility).
    """
    model = ScriptedModel(query_llm=[reply("hello")])
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
    assert model.kinds() == ["query_llm"]
