"""Symbolic: UNIFY_REASONING_DELIVERY puts a provider's reasoning items where it reads them.

LiteLLM files every non-standard key of a parsed reply under
``provider_specific_fields``, so OpenRouter's ``reasoning_details`` (encrypted
reasoning items and summaries) come back nested, and Unify sent them back
nested, where OpenRouter ignores them: on 20 recorded ARC prefixes the input
tokens were identical with the items removed (replay probe, 6 Oct). With
``top_level`` each assistant message a dispatch adds also carries them as the
top-level field, the one OpenRouter decrypts, as prime-agent sends them.
Messages already sent are never touched. No LLM calls.
"""

from __future__ import annotations

import copy

import pytest

from unify.common._async_tool.messages import generate_with_preprocess
from unify.settings import ProductionSettings, SETTINGS

ITEMS = [
    {
        "type": "reasoning.summary",
        "summary": "Look at the grid.",
        "format": "openai-responses-v1",
        "index": 0,
    },
    {
        "type": "reasoning.encrypted",
        "data": "gAAAAopaque",  # pragma: allowlist secret
        "id": "rs_1",
        "format": "openai-responses-v1",
        "index": 1,
    },
]


def _reply(text: str) -> dict:
    """An assistant message as UniLLM stores a parsed OpenRouter reply."""
    return {
        "role": "assistant",
        "content": text,
        "tool_calls": None,
        "provider_specific_fields": {
            "reasoning_details": copy.deepcopy(ITEMS),
            "reasoning": "Look at the grid.",
        },
    }


class _Client:
    def __init__(self, messages):
        self.messages = list(messages)
        self.sent: list[list[dict]] = []

    async def generate(self, **kwargs):
        self.sent.append(copy.deepcopy(self.messages))
        self.messages.append(_reply(f"reply {len(self.sent)}"))
        return "ok"


@pytest.fixture
def delivery(monkeypatch):
    def set_value(value: str) -> None:
        monkeypatch.setattr(SETTINGS, "UNIFY_REASONING_DELIVERY", value)

    return set_value


def test_the_setting_defaults_off_and_rejects_unknown_values():
    assert ProductionSettings.model_fields["UNIFY_REASONING_DELIVERY"].default == ""
    assert ProductionSettings.parse_reasoning_delivery(" TOP_LEVEL ") == "top_level"
    with pytest.raises(ValueError):
        ProductionSettings.parse_reasoning_delivery("nested")


@pytest.mark.asyncio
async def test_off_the_items_stay_nested_as_shipped(delivery):
    delivery("")
    client = _Client([{"role": "user", "content": "hi"}])
    await generate_with_preprocess(client, None)
    assert "reasoning_details" not in client.messages[-1]
    assert client.messages[-1]["provider_specific_fields"]["reasoning_details"] == ITEMS


@pytest.mark.asyncio
@pytest.mark.parametrize("preprocess", [None, lambda msgs: msgs])
async def test_on_the_next_request_carries_the_items_at_top_level(
    delivery,
    preprocess,
):
    delivery("top_level")
    client = _Client([{"role": "user", "content": "hi"}])
    await generate_with_preprocess(client, preprocess)
    client.messages.append({"role": "user", "content": "next"})
    await generate_with_preprocess(client, preprocess)

    # What the second request carried: the first reply, items at top level,
    # the nested copy untouched.
    first_reply = client.sent[1][1]
    assert first_reply["reasoning_details"] == ITEMS
    assert first_reply["provider_specific_fields"]["reasoning_details"] == ITEMS


@pytest.mark.asyncio
async def test_on_messages_already_sent_are_never_rewritten(delivery):
    """A history recorded before the switch keeps its bytes: only the reply a
    dispatch adds gains the field, so the cached prefix stays stable."""
    old = _reply("from before")
    delivery("top_level")
    client = _Client([{"role": "user", "content": "hi"}, old])
    before = copy.deepcopy(client.messages[:2])
    await generate_with_preprocess(client, None)
    assert client.messages[:2] == before
    assert "reasoning_details" not in client.messages[1]
    assert client.messages[2]["reasoning_details"] == ITEMS


@pytest.mark.asyncio
async def test_on_a_top_level_field_already_there_is_kept_and_others_ignored(
    delivery,
):
    delivery("top_level")
    client = _Client([{"role": "user", "content": "hi"}])

    async def generate(**kwargs):
        own = _reply("has its own")
        own["reasoning_details"] = [{"type": "reasoning.text", "text": "kept"}]
        empty = _reply("nothing to deliver")
        empty["provider_specific_fields"]["reasoning_details"] = []
        client.messages += [own, empty, {"role": "tool", "content": "x"}]
        return "ok"

    client.generate = generate
    await generate_with_preprocess(client, None)
    own, empty, tool = client.messages[1:]
    assert own["reasoning_details"] == [{"type": "reasoning.text", "text": "kept"}]
    assert "reasoning_details" not in empty
    assert "reasoning_details" not in tool


def test_openrouter_request_transform_passes_the_top_level_field_through():
    """LiteLLM's OpenRouter request transform forwards a top-level
    ``reasoning_details`` unchanged, so the field reaches the provider."""
    from litellm.llms.openrouter.chat.transformation import OpenrouterConfig

    message = _reply("a reply")
    message["reasoning_details"] = copy.deepcopy(ITEMS)
    body = OpenrouterConfig().transform_request(
        model="openai/gpt-6-luna",
        messages=[{"role": "user", "content": "hi"}, message],
        optional_params={},
        litellm_params={},
        headers={},
    )
    assert body["messages"][1]["reasoning_details"] == ITEMS
