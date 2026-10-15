"""Symbolic: a reply the provider withheld (``finish_reason: "content_filter"``, no text and no tool call).

A live run's streamed replies through OpenRouter sometimes end this way: HTTP 200, usage all zero, nothing charged,
and a fixed stand-in text from the provider (or no text). The loop drops such a reply and sends the same turn again, at most twice with the
provider-error backoff, counting each retry apart; once both retries are spent the turn ends with a text that says
the reply was withheld, never an empty answer (unify/common/_async_tool/loop.py, ``_completion_withheld``). A reply
counts as withheld when its blank text, its zero output tokens or its all-zero usage shows it is a stand-in; one that
finished with ``content_filter`` with text and output tokens is the model's turn, as before. The counts are in the actor's
run stats only when there were any.

The reply is built by LiteLLM's own response conversion from an OpenRouter-shaped body; the transport is scripted
(tests/cache_discipline_helpers.py), so nothing leaves the process.
"""

from __future__ import annotations

import asyncio

import pytest

from tests import cache_discipline_helpers as h
from tests.actor.code_act.core_world import core_world, new_actor, world  # noqa: F401 (fixtures)
from tests.helpers import _handle_project
from unify.common._async_tool import loop


#: A stand-in text (made up for the test; the harness matches no provider's wording).
STAND_IN = "This reply was held back by a filter."


def withheld(content=None, completion_tokens=0, prompt_tokens=1200):
    """The reply LiteLLM hands unillm for a body that finished with content_filter."""
    from litellm.litellm_core_utils.llm_response_utils.convert_dict_to_response import (
        convert_to_model_response_object,
    )
    from litellm.types.utils import ModelResponse

    raw = {
        "id": "gen-withheld",
        "object": "chat.completion",
        "created": 1792100000,
        "model": "openai/gpt-6-luna",
        "provider": "OpenAI",
        "choices": [
            {
                "index": 0,
                "finish_reason": "content_filter",
                "native_finish_reason": "content_filter",
                "logprobs": None,
                "message": {"role": "assistant", "content": content, "refusal": None},
            },
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }
    return convert_to_model_response_object(response_object=raw, model_response_object=ModelResponse())


def test_the_withheld_reply_is_told_from_a_reply_with_text():
    reply = withheld()
    assert reply.choices[0].finish_reason == "content_filter"
    assert not reply.choices[0].message.content and not reply.choices[0].message.tool_calls
    assert loop._completion_withheld(reply)
    # a stand-in text: all-zero usage, or no output tokens
    assert loop._completion_withheld(withheld(STAND_IN, completion_tokens=0, prompt_tokens=0))
    assert loop._completion_withheld(withheld(STAND_IN, completion_tokens=0))
    # the model's own filtered text, with output tokens: its turn
    assert not loop._completion_withheld(withheld("Some text.", completion_tokens=12))
    assert not loop._completion_withheld(h.completion(content="done"))
    assert not loop._completion_withheld(h.completion(calls=[("execute_code", {"code": "print(1)"})]))


async def _act(*replies) -> tuple[str, dict, int]:
    actor = new_actor(can_store=False)
    try:
        with h.scripted(replies) as provider:
            handle = await actor.act("Do the task.", persist=False)
            result = await asyncio.wait_for(handle.result(), 120)
            # The CLI reads the run stats from the request's handle.
            stats = dict(handle.run_stats)
    finally:
        await actor.close()
    return result, stats, len(provider.requests)


@pytest.mark.asyncio
@pytest.mark.timeout(120)
@_handle_project
async def test_a_withheld_reply_is_sent_again_and_counted(core_world):  # noqa: F811
    result, stats, requests = await _act(withheld, withheld, lambda: h.completion(content="done"))
    assert result == "done" and requests == 3
    assert stats.get("content_filter_retries") == 2 and "content_filter_withheld" not in stats


@pytest.mark.asyncio
@pytest.mark.timeout(120)
@_handle_project
async def test_once_the_retries_are_spent_the_turn_says_the_reply_was_withheld(core_world):  # noqa: F811
    result, stats, requests = await _act(withheld, withheld, withheld)
    assert result == loop.WITHHELD_REPLY.format(attempts=3) and requests == 3
    assert stats.get("content_filter_retries") == 2 and stats.get("content_filter_withheld") == 1


@pytest.mark.asyncio
@pytest.mark.timeout(120)
@_handle_project
async def test_a_content_filter_reply_with_text_and_tokens_is_the_models_turn(core_world):  # noqa: F811
    result, stats, requests = await _act(lambda: withheld("Partial answer.", completion_tokens=5))
    assert result == "Partial answer." and requests == 1
    assert "content_filter_retries" not in stats and "content_filter_withheld" not in stats


@pytest.mark.asyncio
@pytest.mark.timeout(120)
@_handle_project
async def test_a_stand_in_text_with_zero_usage_is_sent_again_and_never_reaches_the_answer(core_world):  # noqa: F811
    def stand_in():
        return withheld(STAND_IN, completion_tokens=0, prompt_tokens=0)

    result, stats, requests = await _act(stand_in, stand_in, lambda: h.completion(content="done"))
    assert result == "done" and requests == 3
    assert stats.get("content_filter_retries") == 2 and "content_filter_withheld" not in stats
    result, stats, requests = await _act(stand_in, stand_in, stand_in)
    assert result == loop.WITHHELD_REPLY.format(attempts=3) and STAND_IN not in result
    assert stats.get("content_filter_withheld") == 1
