# tests/async_tool_loop/test_image_blocks.py
#
# Regression tests for the image-promotion logic in
# `unify.common.llm_helpers`. The model is scripted (tests/scripted_model.py):
# the tests are about the blocks the loop builds and sends, not about what the
# model makes of the image.

from __future__ import annotations

import asyncio
import base64
from pathlib import Path

import pytest
from unify.common.async_tool_loop import start_async_tool_loop
from tests.helpers import _handle_project
from tests.scripted_model import ScriptedModel, reply, scripted
from unify.common.llm_client import new_llm_client

# --------------------------------------------------------------------------- #
#  CONSTANTS                                                                  #
# --------------------------------------------------------------------------- #

_JPEG_PREFIX = "data:image/jpeg;base64,"

# Load cat image and convert to base64
with open(Path(__file__).parent / "cat.jpg", "rb") as f:
    CAT_IMG = base64.b64encode(f.read()).decode("utf-8")


def _has_jpeg_block(content) -> bool:
    """Whether *content* (a message's content) holds a promoted JPEG
    ``image_url`` block."""
    return isinstance(content, list) and any(
        isinstance(blk, dict)
        and blk.get("type") == "image_url"
        and blk["image_url"]["url"].startswith(_JPEG_PREFIX)
        for blk in content
    )


# --------------------------------------------------------------------------- #
#  1️⃣  User-supplied image is promoted & understood                           #
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
@_handle_project
async def test_initial_user_image_is_promoted(llm_config) -> None:
    """
    The first user turn contains an inline JPEG. We expect the `image_url`
    block in the first user message, and in the request sent to the model.
    """
    model = ScriptedModel(actor=[reply("cat")])

    img_block = {
        "type": "image_url",
        "image_url": {"url": f"data:image/jpeg;base64,{CAT_IMG}"},
    }
    txt_block = {
        "type": "text",
        "text": "What animal is shown? ONE word only.",
    }

    with scripted(model):
        client = new_llm_client(**llm_config)
        client.set_system_message(
            "You will receive an image. Answer with ONE three-letter word naming the animal.",
        )
        handle = start_async_tool_loop(
            client,
            message=[img_block, txt_block],  # ready-made content blocks
            tools={},
        )
        assert await asyncio.wait_for(handle.result(), 60) == "cat"

    # ---- 1. image block exists in the first user message ------------------
    user_msg = next(m for m in client.messages if m["role"] == "user")
    assert isinstance(user_msg["content"], list), "User content must be block array"
    assert _has_jpeg_block(
        user_msg["content"],
    ), "No promoted image_url block found in user message"

    # ---- 2. the request the model received carries the block --------------
    (request,) = model.of("actor")
    assert any(
        m["role"] == "user" and _has_jpeg_block(m.get("content"))
        for m in request.messages
    ), "The request sent to the model carries no image_url block"
    assert model.kinds() == ["actor"]


# --------------------------------------------------------------------------- #
#  2️⃣  Tool-returned image is promoted & understood                           #
# --------------------------------------------------------------------------- #


async def image_tool() -> dict:
    """Return a JSON payload that **includes** a base-64 image."""
    await asyncio.sleep(0.05)
    return {"status": "ok", "image": CAT_IMG}


@pytest.mark.asyncio
@_handle_project
async def test_tool_result_image_is_promoted_and_reasoned_over(llm_config) -> None:
    """
    Flow:
      1. Assistant calls `image_tool`.
      2. Tool returns a dict containing `"image": <b64>`.
      3. Loop promotes the image to `image_url` in the tool message.
      4. The next request sent to the model carries the promoted block.
    """
    model = ScriptedModel(
        actor=[reply(calls=[("image_tool", {})]), reply("cat")],
    )
    with scripted(model):
        client = new_llm_client(**llm_config)
        client.set_system_message(
            "Call image_tool exactly once. The tool will return a base64-encoded image of a domestic cat. After the tool finishes, respond with exactly 'cat' (lowercase, no punctuation). Do not output anything else.",
        )

        primary_handle = start_async_tool_loop(
            client,
            message="go",
            tools={"image_tool": image_tool},
            timeout=120,
        )
        assert await asyncio.wait_for(primary_handle.result(), 60) == "cat"

    # Locate the tool-result message
    tool_msg = next(
        (
            m
            for m in client.messages
            if m["role"] == "tool" and m["name"] == "image_tool"
        ),
        None,
    )
    assert tool_msg is not None, "Tool result message missing"

    content_blocks = tool_msg["content"]
    assert isinstance(content_blocks, list), "Tool content must be block array"
    assert _has_jpeg_block(
        content_blocks,
    ), "Promoted image_url block not found in tool result"

    # The request after the tool ran carries the promoted block.
    first, second = model.of("actor")
    assert not any(_has_jpeg_block(m.get("content")) for m in first.messages)
    assert any(
        _has_jpeg_block(m.get("content")) for m in second.messages
    ), "The request after the tool ran carries no image_url block"
    assert model.kinds() == ["actor", "actor"]
