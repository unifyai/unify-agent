"""Evals of a live model (moved from tests/async_tool_loop/test_context_compression.py): each needs a real provider key and is run by hand
(model_evals/README.md). The test suite never collects or runs them."""

from __future__ import annotations

import base64
import copy
import inspect
from pathlib import Path
import pytest
import json
from tests.helpers import _handle_project
from tests.scripted_model import reply
import unify.common._async_tool.context_compression as _cc_mod
from unify.common._async_tool.context_compression import (
    COMPRESSION_MULTI_PASS_ADDENDUM,
    COMPRESSION_PROMPT,
    CompressedMessage,
    CompressedMessages,
    CompressionState,
    compress_and_rebuild,
    compress_context,
    compress_messages,
    tag_images_in_messages,
    _eval_transformation,
    _make_update_tool,
    _make_archive_lookup_tool,
    _scan_surviving_image_ids,
)

from tests.async_tool_loop.test_context_compression import (  # noqa: F401 - the original module's helpers
    _IMG_PNG_BLOCK,
    _accumulate_entries,
    _build_image_conversation,
    _mp_pass1,
    _mp_pass2,
    _mp_pass3,
)


@pytest.mark.eval
@pytest.mark.asyncio
@pytest.mark.llm_call
@_handle_project
async def test_multi_pass_double_further_compresses_prior(llm_config):
    """Pass 2 further compresses verbose entries that survived pass 1.

    Scenario: pass 1 compresses a verbose transcript (big tool results, traceback).
    Pass 2 receives pass 1's output as prior_entries alongside new messages and
    must not expand prior entries — and ideally compresses them further given the
    new context.
    """
    pass1_msgs = _mp_pass1()
    pass2_msgs = _mp_pass2()

    # --- Pass 1 ---
    result1 = await compress_messages(pass1_msgs, llm_config["model"])
    assert len(result1.messages) == len(pass1_msgs)

    original_tool_len = len(json.dumps(pass1_msgs[2], default=str))
    assert len(result1.messages[2].content) < original_tool_len * 0.7

    # --- Pass 2 ---
    prior_entries = [(i, msg.content) for i, msg in enumerate(result1.messages)]
    pass2_base = len(pass1_msgs)
    pass2_indices = list(range(pass2_base, pass2_base + len(pass2_msgs)))

    result2 = await compress_messages(
        pass2_msgs,
        llm_config["model"],
        prior_entries=prior_entries,
        raw_archives=[pass1_msgs],
        new_indices=pass2_indices,
    )

    assert len(result2.messages) == len(prior_entries) + len(pass2_msgs)

    for i, (_, pass1_content) in enumerate(prior_entries):
        pass2_content = result2.messages[i].content
        assert len(pass2_content) <= len(pass1_content) * 1.5, (
            f"Prior entry {i} expanded significantly: "
            f"{len(pass1_content)} → {len(pass2_content)}"
        )


@pytest.mark.eval
@pytest.mark.asyncio
@pytest.mark.llm_call
@_handle_project
async def test_multi_pass_double_with_raw_archives(llm_config):
    """Pass 2 with raw_archives can inspect originals via get_raw.

    The LLM receives both prior compressed entries and the raw_archives that
    enable the get_raw tool. The overall compressed output should be
    significantly smaller than the raw original content.
    """
    pass1_msgs = _mp_pass1()
    pass2_msgs = _mp_pass2()

    # --- Pass 1 ---
    result1 = await compress_messages(pass1_msgs, llm_config["model"])
    prior_entries = [(i, msg.content) for i, msg in enumerate(result1.messages)]
    pass2_base = len(pass1_msgs)
    pass2_indices = list(range(pass2_base, pass2_base + len(pass2_msgs)))

    # --- Pass 2 with raw_archives ---
    result2 = await compress_messages(
        pass2_msgs,
        llm_config["model"],
        prior_entries=prior_entries,
        raw_archives=[pass1_msgs],
        new_indices=pass2_indices,
    )

    total_expected = len(prior_entries) + len(pass2_msgs)
    assert len(result2.messages) == total_expected

    original_total = sum(
        len(json.dumps(m, default=str)) for m in pass1_msgs + pass2_msgs
    )
    compressed_total = sum(len(m.content) for m in result2.messages)
    assert (
        compressed_total < original_total * 0.7
    ), f"Expected significant compression: {original_total} → {compressed_total}"


@pytest.mark.eval
@pytest.mark.asyncio
@pytest.mark.llm_call
@_handle_project
async def test_multi_pass_triple_accumulates_correctly(llm_config):
    """Three sequential passes accumulate and compress all entries.

    Pass 1 compresses the initial verbose transcript. Pass 2 adds new messages
    and re-compresses with pass 1's output as prior. Pass 3 adds more messages
    and re-compresses with the accumulated prior from passes 1+2. The final
    result must contain exactly as many entries as there are original messages
    across all three passes, with significant overall compression.
    """
    pass1_msgs = _mp_pass1()
    pass2_msgs = _mp_pass2()
    pass3_msgs = _mp_pass3()

    # --- Pass 1 ---
    result1 = await compress_messages(pass1_msgs, llm_config["model"])
    assert len(result1.messages) == len(pass1_msgs)

    prior_1 = [(i, msg.content) for i, msg in enumerate(result1.messages)]
    raw_archives = [pass1_msgs]

    # --- Pass 2 ---
    pass2_base = len(pass1_msgs)
    pass2_indices = list(range(pass2_base, pass2_base + len(pass2_msgs)))

    result2 = await compress_messages(
        pass2_msgs,
        llm_config["model"],
        prior_entries=prior_1,
        raw_archives=raw_archives,
        new_indices=pass2_indices,
    )
    assert len(result2.messages) == len(prior_1) + len(pass2_msgs)

    prior_2 = _accumulate_entries(result2, pass2_indices, prior_1)
    raw_archives = [pass1_msgs, pass2_msgs]

    # --- Pass 3 ---
    pass3_base = pass2_base + len(pass2_msgs)
    pass3_indices = list(range(pass3_base, pass3_base + len(pass3_msgs)))

    result3 = await compress_messages(
        pass3_msgs,
        llm_config["model"],
        prior_entries=prior_2,
        raw_archives=raw_archives,
        new_indices=pass3_indices,
    )

    total_all_msgs = len(pass1_msgs) + len(pass2_msgs) + len(pass3_msgs)
    assert len(result3.messages) == total_all_msgs

    original_total = sum(
        len(json.dumps(m, default=str)) for m in pass1_msgs + pass2_msgs + pass3_msgs
    )
    compressed_total = sum(len(m.content) for m in result3.messages)
    assert compressed_total < original_total * 0.7, (
        f"Expected significant compression across 3 passes: "
        f"{original_total} → {compressed_total}"
    )


@pytest.mark.eval
@pytest.mark.asyncio
@pytest.mark.llm_call
@_handle_project
async def test_image_aware_compression_keeps_relevant(llm_config):
    """The compression LLM should keep [img:0] (still relevant) and may remove [img:1] (bug fixed)."""
    msgs, registry = _build_image_conversation()
    result = await compress_messages(
        msgs,
        llm_config["model"],
        image_blocks=registry,
    )
    assert len(result.messages) == len(msgs)
    assert (
        0 in result.surviving_image_ids
    ), "img:0 (dashboard) should survive — it's still actively referenced"


@pytest.mark.eval
@pytest.mark.asyncio
@pytest.mark.llm_call
@_handle_project
async def test_image_aware_multi_pass_accumulates(llm_config):
    """Multi-pass image compression: pass 2 sees surviving images from pass 1 plus new ones."""
    pass1_msgs, pass1_registry = _build_image_conversation()

    result1 = await compress_messages(
        pass1_msgs,
        llm_config["model"],
        image_blocks=pass1_registry,
    )
    assert len(result1.messages) == len(pass1_msgs)

    prior_entries = [(i, msg.content) for i, msg in enumerate(result1.messages)]

    pass2_msgs = [
        {
            "role": "user",
            "content": "Here is an updated dashboard screenshot. Compare it to the old one.",
        },
        {
            "role": "assistant",
            "content": (
                "Comparing the updated dashboard [img:2] with the original [img:0]: "
                "revenue in [img:2] is significantly higher. I need both [img:0] and "
                "[img:2] to complete the comparison you requested."
            ),
        },
    ]
    pass2_registry = {2: _IMG_PNG_BLOCK}

    all_live = {
        iid: block
        for iid, block in {**pass1_registry, **pass2_registry}.items()
        if iid in result1.surviving_image_ids or iid in pass2_registry
    }

    pass2_base = len(pass1_msgs)
    pass2_indices = list(range(pass2_base, pass2_base + len(pass2_msgs)))

    result2 = await compress_messages(
        pass2_msgs,
        llm_config["model"],
        image_blocks=all_live,
        prior_entries=prior_entries,
        raw_archives=[pass1_msgs],
        new_indices=pass2_indices,
    )

    assert len(result2.messages) == len(prior_entries) + len(pass2_msgs)
    assert (
        2 in result2.surviving_image_ids
    ), "img:2 (new dashboard) should survive — it's actively referenced"
