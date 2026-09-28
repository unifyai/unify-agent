from __future__ import annotations

import base64
import copy
import inspect
from pathlib import Path

import pytest

import json

from tests.helpers import _handle_project
import unify.common._async_tool.context_compression as _cc_mod
from unify.common._async_tool.context_compression import (
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

# Vision-aware compression is handed the real image bytes, so a placeholder
# data URL is rejected by the provider before the compressor ever runs.
with open(Path(__file__).parent / "cat.jpg", "rb") as _f:
    _CAT_IMG = base64.b64encode(_f.read()).decode("utf-8")


class TestCompressContextTool:
    def test_no_parameters(self):
        sig = inspect.signature(compress_context)
        assert len(sig.parameters) == 0

    def test_has_docstring(self):
        assert compress_context.__doc__ is not None
        assert len(compress_context.__doc__) > 20

    def test_returns_string(self):
        result = compress_context()
        assert isinstance(result, str)


class TestEvalTransformation:
    def test_overwrite(self):
        result = _eval_transformation('x = "compressed"', "original long content")
        assert result == "compressed"

    def test_surgical_replace(self):
        content = "Hello verbose world with extra stuff"
        result = _eval_transformation(
            'x = x.replace("verbose world with extra stuff", "world")',
            content,
        )
        assert result == "Hello world"

    def test_bare_expression_auto_assigned(self):
        content = "Hello verbose world with extra stuff"
        result = _eval_transformation(
            'x.replace("verbose world with extra stuff", "world")',
            content,
        )
        assert result == "Hello world"

    def test_keep_first_line(self):
        content = "first line\nsecond line\nthird line"
        result = _eval_transformation('x = x.split("\\n")[0]', content)
        assert result == "first line"

    def test_truncate(self):
        content = "a" * 500
        result = _eval_transformation("x = x[:100]", content)
        assert len(result) == 100

    def test_regex_via_re(self):
        content = "data Traceback (most recent call last):\n  File... end"
        result = _eval_transformation(
            r'x = re.sub(r"Traceback[\s\S]*", "traceback omitted", x)',
            content,
        )
        assert "traceback omitted" in result
        assert "most recent" not in result

    def test_multiline_code(self):
        content = "keep\nERROR: bad\nkeep too"
        # Mirrors what arrives after JSON decode: \\n in strings becomes \n
        # (which exec interprets as the newline escape), and actual newlines
        # separate code lines.
        code = "lines = x.split('\\n')\nx = '\\n'.join(l for l in lines if 'ERROR' not in l)"
        result = _eval_transformation(code, content)
        assert result == "keep\nkeep too"

    def test_non_string_result_coerced(self):
        result = _eval_transformation("x = len(x)", "hello")
        assert result == "5"

    def test_runtime_error_raises(self):
        with pytest.raises(ZeroDivisionError):
            _eval_transformation("x = 1 // 0", "content")


class TestMakeUpdateTool:
    _ENDPOINT = "openai/gpt-4o@openrouter"

    def test_overwrite_entry(self):
        entries = {
            0: json.dumps({"role": "user", "content": "original"}),
            1: json.dumps({"role": "assistant", "content": "keep this"}),
        }
        update = _make_update_tool(entries, self._ENDPOINT)
        result = update(0, 'x = "replaced"')
        assert "replaced" in result
        assert entries[0] == "replaced"
        assert "keep this" in entries[1]

    def test_response_includes_token_usage(self):
        entries = {0: json.dumps({"role": "user", "content": "some content here"})}
        update = _make_update_tool(entries, self._ENDPOINT)
        result = update(0, 'x = "short"')
        assert "tokens" in result
        assert "%" in result

    def test_surgical_replace(self):
        entries = {
            0: json.dumps(
                {"role": "tool", "content": "verbose error with traceback details"},
            ),
        }
        update = _make_update_tool(entries, self._ENDPOINT)
        result = update(
            0,
            'x = x.replace("verbose error with traceback details", "error")',
        )
        assert "error" in result
        assert "verbose" not in entries[0]

    def test_not_found_returns_error(self):
        entries = {0: json.dumps({"role": "user", "content": "only one"})}
        update = _make_update_tool(entries, self._ENDPOINT)
        result = update(5, 'x = "y"')
        assert "Error" in result
        assert "not found" in result

    def test_invalid_code_returns_error(self):
        entries = {0: json.dumps({"role": "user", "content": "content"})}
        original = entries[0]
        update = _make_update_tool(entries, self._ENDPOINT)
        result = update(0, "def def def")
        assert "Error" in result
        assert entries[0] == original

    def test_empty_result_becomes_marker(self):
        entries = {0: json.dumps({"role": "user", "content": "something"})}
        update = _make_update_tool(entries, self._ENDPOINT)
        result = update(0, 'x = ""')
        assert "(empty)" in result
        assert entries[0] == "(empty)"

    def test_multiple_updates(self):
        entries = {
            0: json.dumps({"role": "user", "content": "first"}),
            1: json.dumps({"role": "assistant", "content": "second"}),
            2: json.dumps({"role": "tool", "content": "third"}),
        }
        update = _make_update_tool(entries, self._ENDPOINT)
        update(0, 'x = "1"')
        update(2, 'x = "3"')
        assert entries[0] == "1"
        assert "second" in entries[1]
        assert entries[2] == "3"

    def test_bare_expression(self):
        entries = {0: json.dumps({"role": "user", "content": "hello world"})}
        update = _make_update_tool(entries, self._ENDPOINT)
        result = update(0, 'x.replace("hello", "hi")')
        assert "hi world" in result
        assert "hi world" in entries[0]

    def test_structured_transformation(self):
        entries = {
            0: json.dumps({"role": "tool", "content": "line1\nERROR line2\nline3"}),
        }
        update = _make_update_tool(entries, self._ENDPOINT)
        code = (
            "msg = json.loads(x)\n"
            "lines = msg['content'].split('\\n')\n"
            "msg['content'] = '\\n'.join(l for l in lines if 'ERROR' not in l)\n"
            "x = json.dumps(msg)"
        )
        result = update(0, code)
        assert "line1" in result
        parsed = json.loads(entries[0])
        assert "ERROR" not in parsed["content"]

    def test_non_contiguous_indices(self):
        entries = {
            3: json.dumps({"role": "user", "content": "at index 3"}),
            7: json.dumps({"role": "tool", "content": "at index 7"}),
        }
        update = _make_update_tool(entries, self._ENDPOINT)
        result = update(3, 'x = "short"')
        assert "short" in result
        assert entries[3] == "short"
        result = update(5, 'x = "nope"')
        assert "Error" in result
        assert "not found" in result


class TestMakeArchiveLookupTool:
    def test_returns_raw_content(self):
        archives = [
            [
                {"role": "user", "content": "hello"},
                {"role": "assistant", "content": "world"},
            ],
        ]
        get_raw = _make_archive_lookup_tool(archives, for_compression=True)
        result = json.loads(get_raw(0))
        assert len(result) == 1
        assert result[0]["content"] == "hello"

    def test_out_of_range_returns_error(self):
        archives = [[{"role": "user", "content": "only one"}]]
        get_raw = _make_archive_lookup_tool(archives, for_compression=True)
        result = json.loads(get_raw(5))
        assert "error" in result
        assert "out of range" in result["error"].lower()

    def test_multiple_archives(self):
        archives = [
            [
                {"role": "user", "content": "first_archive_msg0"},
                {"role": "assistant", "content": "first_archive_msg1"},
            ],
            [
                {"role": "user", "content": "second_archive_msg0"},
                {"role": "tool", "content": "second_archive_msg1"},
                {"role": "assistant", "content": "second_archive_msg2"},
            ],
        ]
        get_raw = _make_archive_lookup_tool(archives, for_compression=True)
        r0 = json.loads(get_raw(0))
        assert r0[0]["content"] == "first_archive_msg0"
        r2 = json.loads(get_raw(2))
        assert r2[0]["content"] == "second_archive_msg0"
        r4 = json.loads(get_raw(4))
        assert r4[0]["content"] == "second_archive_msg2"

    def test_n_consecutive(self):
        archives = [
            [
                {"role": "user", "content": "msg0"},
                {"role": "assistant", "content": "msg1"},
                {"role": "user", "content": "msg2"},
                {"role": "assistant", "content": "msg3"},
            ],
        ]
        get_raw = _make_archive_lookup_tool(archives, for_compression=True)
        result = json.loads(get_raw(1, n=2))
        assert len(result) == 2
        assert result[0]["content"] == "msg1"
        assert result[1]["content"] == "msg2"

    def test_compression_mode_sets_name(self):
        tool = _make_archive_lookup_tool([[]], for_compression=True)
        assert tool.__name__ == "get_raw"

    def test_unpack_mode_sets_name(self):
        tool = _make_archive_lookup_tool([[]], for_compression=False)
        assert tool.__name__ == "unpack_messages"


# ── compress_and_rebuild tests ────────────────────────────────────────────────


async def _mock_compress_for_rebuild(messages, endpoint, **kwargs):
    """Mock that returns predictable compressed entries for each input message.

    Handles prior_entries correctly: returns len(prior) + len(messages)
    entries, echoing back a prefix of each entry's content for traceability.
    """
    prior = kwargs.get("prior_entries") or []
    surviving = (
        kwargs.get("image_blocks", {}).keys() if kwargs.get("image_blocks") else set()
    )
    results = []
    for _, content in prior:
        results.append(CompressedMessage(content=f"recompressed:{content[:30]}"))
    for msg in messages:
        raw = json.dumps(msg, default=str)
        results.append(CompressedMessage(content=f"compressed:{raw[:40]}"))
    return CompressedMessages(
        messages=results,
        surviving_image_ids=set(surviving),
    )


class TestCompressAndRebuild:
    """Symbolic tests for compress_and_rebuild data transformation."""

    @pytest.mark.asyncio
    async def test_first_pass_archives_and_compresses(self, monkeypatch):
        monkeypatch.setattr(_cc_mod, "compress_messages", _mock_compress_for_rebuild)

        state = CompressionState()
        messages = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "world"},
            {"role": "user", "content": "thanks"},
        ]
        original_tools = {"my_tool": lambda: "ok"}

        result = await compress_and_rebuild(
            state,
            messages,
            "openai/gpt-4o@openrouter",
            original_tools,
        )

        assert len(state.raw_archives) == 1
        assert len(state.raw_archives[0]) == 3
        assert state.count == 1
        assert len(state.entries) == 3
        assert [idx for idx, _ in state.entries] == [0, 1, 2]

        compressed_sys = result.system_msgs[-1]
        assert compressed_sys["_compressed_message"] is True
        assert "## Compressed Prior Context" in compressed_sys["content"]
        assert "[0]" in compressed_sys["content"]
        assert "[1]" in compressed_sys["content"]
        assert "[2]" in compressed_sys["content"]

        assert "my_tool" in result.tools
        assert "unpack_messages" in result.tools

        unpacked = json.loads(result.tools["unpack_messages"](0))
        assert unpacked[0]["content"] == "hello"

    @pytest.mark.asyncio
    async def test_second_pass_skips_compressed_message_marker(self, monkeypatch):
        call_log = {}

        async def _tracking_mock(messages, endpoint, **kwargs):
            call_log["messages"] = messages
            call_log["kwargs"] = kwargs
            return await _mock_compress_for_rebuild(messages, endpoint, **kwargs)

        monkeypatch.setattr(_cc_mod, "compress_messages", _tracking_mock)

        state = CompressionState()
        pass1_messages = [
            {"role": "user", "content": "first"},
            {"role": "assistant", "content": "response"},
        ]
        await compress_and_rebuild(
            state,
            pass1_messages,
            "openai/gpt-4o@openrouter",
            {},
        )

        assert state.count == 1
        assert len(state.entries) == 2
        pass1_archive_len = len(state.raw_archives[0])

        pass2_messages = [
            {
                "role": "system",
                "_compressed_message": True,
                "content": "prior context...",
            },
            {"role": "user", "content": "follow-up"},
            {"role": "assistant", "content": "answer"},
        ]
        await compress_and_rebuild(
            state,
            pass2_messages,
            "openai/gpt-4o@openrouter",
            {},
        )

        assert state.count == 2
        mock_received = call_log["messages"]
        for msg in mock_received:
            assert not msg.get(
                "_compressed_message",
            ), "_compressed_message marker should be filtered before compress_messages"

        assert call_log["kwargs"].get("prior_entries") is not None
        assert len(call_log["kwargs"]["prior_entries"]) == 2

        new_conv_entries = [
            (idx, c) for idx, c in state.entries if idx >= pass1_archive_len
        ]
        assert len(new_conv_entries) == 2

    @pytest.mark.asyncio
    async def test_system_messages_extracted_into_system_msgs(self, monkeypatch):
        async def _sys_aware_mock(messages, endpoint, **kwargs):
            results = []
            prior = kwargs.get("prior_entries") or []
            for _, content in prior:
                results.append(
                    CompressedMessage(content=f"recompressed:{content[:30]}"),
                )
            for msg in messages:
                if msg.get("role") == "system":
                    results.append(
                        CompressedMessage(
                            content=json.dumps(
                                {"content": f"compressed-sys:{msg['content'][:20]}"},
                            ),
                        ),
                    )
                else:
                    results.append(
                        CompressedMessage(
                            content=f"compressed:{msg.get('content', '')[:20]}",
                        ),
                    )
            return CompressedMessages(messages=results)

        monkeypatch.setattr(_cc_mod, "compress_messages", _sys_aware_mock)

        state = CompressionState()
        messages = [
            {"role": "system", "content": "You are helpful"},
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi there"},
        ]
        result = await compress_and_rebuild(
            state,
            messages,
            "openai/gpt-4o@openrouter",
            {},
        )

        non_compressed_sys = [
            m for m in result.system_msgs if not m.get("_compressed_message")
        ]
        assert len(non_compressed_sys) == 1
        assert non_compressed_sys[0]["role"] == "system"
        assert "compressed-sys:" in non_compressed_sys[0]["content"]

        entry_indices = {idx for idx, _ in state.entries}
        sys_msg_idx = 0
        assert (
            sys_msg_idx not in entry_indices
        ), "System messages should be segregated into system_msgs, not entries"
        assert len(state.entries) == 2

    @pytest.mark.asyncio
    async def test_image_tagging_accumulates_across_passes(self, monkeypatch):
        monkeypatch.setattr(_cc_mod, "compress_messages", _mock_compress_for_rebuild)

        state = CompressionState()

        pass1_messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "look at this"},
                    _IMG_PNG_BLOCK,
                ],
            },
            {"role": "assistant", "content": "I see a red pixel."},
        ]
        result1 = await compress_and_rebuild(
            state,
            pass1_messages,
            "openai/gpt-4o@openrouter",
            {},
        )

        assert 0 in state.image_registry
        assert 0 in state.live_image_ids
        assert state.next_image_id == 1

        compressed_content = result1.system_msgs[-1]["content"]
        assert isinstance(compressed_content, list), "Should be multimodal with images"
        img_labels = [
            b["text"]
            for b in compressed_content
            if isinstance(b, dict)
            and b.get("type") == "text"
            and "[img:" in b.get("text", "")
        ]
        assert any("[img:0]" in label for label in img_labels)

        pass2_messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "now this one"},
                    _IMG_PNG_BLOCK_2,
                ],
            },
            {"role": "assistant", "content": "A blue pixel."},
        ]

        async def _pass2_mock(messages, endpoint, **kwargs):
            """Return surviving_image_ids with only the pass-2 image."""
            result = await _mock_compress_for_rebuild(messages, endpoint, **kwargs)
            return CompressedMessages(
                messages=result.messages,
                surviving_image_ids={1},
            )

        monkeypatch.setattr(_cc_mod, "compress_messages", _pass2_mock)
        await compress_and_rebuild(
            state,
            pass2_messages,
            "openai/gpt-4o@openrouter",
            {},
        )

        assert state.next_image_id == 2
        assert 0 in state.image_registry and 1 in state.image_registry
        assert state.live_image_ids == {1}

    @pytest.mark.asyncio
    async def test_unpack_messages_tool_retrieves_archived_content(self, monkeypatch):
        monkeypatch.setattr(_cc_mod, "compress_messages", _mock_compress_for_rebuild)

        state = CompressionState()
        messages = [
            {"role": "user", "content": "msg-zero"},
            {"role": "assistant", "content": "msg-one"},
            {"role": "user", "content": "msg-two"},
        ]
        result = await compress_and_rebuild(
            state,
            messages,
            "openai/gpt-4o@openrouter",
            {},
        )

        unpack = result.tools["unpack_messages"]
        single = json.loads(unpack(0))
        assert len(single) == 1
        assert single[0]["content"] == "msg-zero"

        pair = json.loads(unpack(0, n=2))
        assert len(pair) == 2
        assert pair[0]["content"] == "msg-zero"
        assert pair[1]["content"] == "msg-one"

        last = json.loads(unpack(2))
        assert last[0]["content"] == "msg-two"

        err = json.loads(unpack(99))
        assert "error" in err


@pytest.mark.asyncio
@pytest.mark.llm_call
@_handle_project
async def test_compress_returns_compressed_messages(llm_config):
    messages = [
        {
            "role": "user",
            "content": "Please can you find the contact named John, once you find them list their phone number & email",
        },
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": "search",
                        "arguments": '{"name": "John"}',
                    },
                },
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "call_1",
            "name": "search",
            "content": '{"contact_id": 0, "name": "John", "email": "example@email.com", "phone": "+1234567890", "city": "Giza", "country": "Egypt", "gender": "male"}',
        },
        {
            "role": "assistant",
            "content": "I found the contact named John, their email is example@email.com and their phone number is +1234567890, would you like me to help with anything else?",
        },
    ]
    result = await compress_messages(messages, llm_config["model"])
    assert isinstance(result, CompressedMessages)
    assert len(result.messages) == len(messages)


@pytest.mark.asyncio
@pytest.mark.llm_call
@_handle_project
async def test_compress_output_length_matches_input(llm_config):
    messages = [
        {"role": "user", "content": "Hello, how are you doing today?"},
        {
            "role": "assistant",
            "content": "I'm doing great, thank you for asking! How can I help you today?",
        },
        {"role": "user", "content": "What is the weather like?"},
        {
            "role": "assistant",
            "content": "I'm sorry, I don't have access to weather information. Would you like me to help with something else?",
        },
    ]
    result = await compress_messages(messages, llm_config["model"])
    assert len(result.messages) == 4


@pytest.mark.asyncio
@pytest.mark.llm_call
@_handle_project
async def test_compress_compacts_tool_call_messages(llm_config):
    messages = [
        {"role": "user", "content": "Find John"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": "search",
                        "arguments": '{"name": "John"}',
                    },
                },
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "call_1",
            "name": "search",
            "content": '{"name": "John", "email": "j@test.com"}',
        },
    ]
    result = await compress_messages(messages, llm_config["model"])
    assert len(result.messages) == 3
    tool_call_msg = result.messages[1]
    assert "search" in tool_call_msg.content.lower()


@pytest.mark.asyncio
@pytest.mark.llm_call
@_handle_project
async def test_compress_preserves_image_placeholders(llm_config):
    """An image the conversation still depends on keeps its ``[img:N]`` tag.

    ``compress_messages`` requires images to already be tagged and reports the
    tags that survived rather than describing them in prose, so the tagging step
    is part of the exercise and the image has to be a real one -- it is delivered
    inline for the compressor to look at. The trailing question is what makes
    preservation the correct outcome: the compressor is told to drop images the
    conversation has moved past, and only an open question about the picture
    keeps this one live.
    """
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "What breed is this cat?"},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{_CAT_IMG}"},
                },
            ],
        },
        {
            "role": "assistant",
            "content": "It looks like a domestic shorthair tabby.",
        },
        {
            "role": "user",
            "content": "What colour are its eyes in that photo?",
        },
    ]
    tagged, image_blocks, _ = tag_images_in_messages(messages)
    result = await compress_messages(
        tagged,
        llm_config["model"],
        image_blocks=image_blocks,
    )
    assert len(result.messages) == len(messages)
    assert result.surviving_image_ids == {0}


_VERBOSE_TRACEBACK = (
    "Traceback (most recent call last):\n"
    '  File "/app/services/contact_service.py", line 287, in search_contacts\n'
    "    results = await self._database.query(params)\n"
    '  File "/app/database/postgres.py", line 154, in query\n'
    "    validated = self._validate_params(params)\n"
    '  File "/app/database/postgres.py", line 89, in _validate_params\n'
    "    for key, value in params.items():\n"
    '  File "/app/database/validators.py", line 42, in validate_field\n'
    "    raise ValueError(\n"
    "ValueError: invalid search parameter 'email'. "
    "Supported parameters are: 'name', 'id', 'phone'. "
    "The 'email' field is not indexed and cannot be used as a search key. "
    "Please use one of the supported parameters listed above."
)


@pytest.mark.asyncio
@pytest.mark.llm_call
@_handle_project
async def test_compress_compacts_verbose_errors(llm_config):
    error_content = json.dumps({"error": _VERBOSE_TRACEBACK})
    messages = [
        {"role": "user", "content": "Search for John"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": "search",
                        "arguments": '{"email": "john@test.com"}',
                    },
                },
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "call_1",
            "name": "search",
            "content": error_content,
        },
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_2",
                    "type": "function",
                    "function": {
                        "name": "search",
                        "arguments": '{"name": "John"}',
                    },
                },
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "call_2",
            "name": "search",
            "content": '{"name": "John", "email": "john@test.com"}',
        },
        {
            "role": "assistant",
            "content": "Found John, email: john@test.com",
        },
    ]
    result = await compress_messages(messages, llm_config["model"])
    assert len(result.messages) == 6
    original_json = json.dumps(messages[2], default=str)
    error_msg = result.messages[2]
    assert len(error_msg.content) < len(original_json) * 0.5


class TestMultiPassCompression:
    """Symbolic tests for the multi-pass compression interface."""

    @pytest.mark.llm_call
    def test_result_count_includes_prior_and_new(self):
        """compress_messages with prior_entries returns prior + new entries."""
        import asyncio

        prior = [
            (0, "compressed msg 0"),
            (1, "compressed msg 1"),
        ]
        new_messages = [
            {"role": "user", "content": "new message"},
        ]
        archives = [
            [
                {"role": "user", "content": "raw 0"},
                {"role": "assistant", "content": "raw 1"},
            ],
        ]

        loop = asyncio.new_event_loop()
        try:
            result = loop.run_until_complete(
                compress_messages(
                    new_messages,
                    "openai/gpt-4o@openrouter",
                    prior_entries=prior,
                    raw_archives=archives,
                    new_indices=[2],
                ),
            )
        except Exception:
            pytest.skip("LLM call required for full integration")
        finally:
            loop.close()

        assert len(result.messages) == 3

    @pytest.mark.asyncio
    async def test_new_indices_validation(self):
        """new_indices must match messages length.

        The mismatch is rejected before any LLM call, so awaiting the coroutine
        is enough and no event loop has to be managed by hand.
        """
        with pytest.raises(ValueError, match="new_indices length"):
            await compress_messages(
                [{"role": "user", "content": "a"}],
                "openai/gpt-4o@openrouter",
                new_indices=[0, 1],
            )


_VERBOSE_PRIOR_TOOL_RESULT = json.dumps(
    {
        "role": "tool",
        "tool_call_id": "call_99",
        "name": "get_contacts",
        "content": json.dumps(
            {
                "contacts": [
                    {
                        "id": 1,
                        "name": "Alice Smith",
                        "email": "alice@example.com",
                        "phone": "+15551234567",
                        "address": "123 Oak Street, Springfield, IL 62704",
                        "notes": "Prefers morning meetings. Has a dog named Rex.",
                        "created_at": "2024-01-15T10:30:00Z",
                        "updated_at": "2024-06-20T14:22:00Z",
                        "tags": ["vip", "engineering", "remote"],
                        "company": "Acme Corp",
                        "title": "Senior Engineer",
                    },
                    {
                        "id": 2,
                        "name": "Bob Jones",
                        "email": "bob@example.com",
                        "phone": "+15559876543",
                        "address": "456 Maple Avenue, Portland, OR 97201",
                        "notes": "Referred by Alice. Working on Project Phoenix.",
                        "created_at": "2024-03-01T09:00:00Z",
                        "updated_at": "2024-07-10T11:45:00Z",
                        "tags": ["engineering", "onsite"],
                        "company": "Acme Corp",
                        "title": "Junior Engineer",
                    },
                ],
                "total": 2,
                "page": 1,
                "per_page": 50,
            },
        ),
    },
)


@pytest.mark.asyncio
@pytest.mark.llm_call
@_handle_project
async def test_compress_multi_pass_recompresses_prior(llm_config):
    """Multi-pass compression can further compress verbose prior entries."""
    prior_entries = [
        (0, _VERBOSE_PRIOR_TOOL_RESULT),
        (1, json.dumps({"role": "assistant", "content": "Found Alice and Bob."})),
    ]
    new_messages = [
        {"role": "user", "content": "What is Alice's email?"},
        {"role": "assistant", "content": "Alice's email is alice@example.com."},
    ]
    raw_archives = [
        [
            {
                "role": "tool",
                "tool_call_id": "call_99",
                "name": "get_contacts",
                "content": "original raw content",
            },
            {"role": "assistant", "content": "Found Alice and Bob."},
        ],
    ]
    result = await compress_messages(
        new_messages,
        llm_config["model"],
        prior_entries=prior_entries,
        raw_archives=raw_archives,
        new_indices=[2, 3],
    )
    assert len(result.messages) == 4
    recompressed_tool_result = result.messages[0]
    assert len(recompressed_tool_result.content) < len(_VERBOSE_PRIOR_TOOL_RESULT) * 0.7


# ── End-to-end multi-pass compression tests ──────────────────────────────────
#
# These eval tests exercise sequential compression passes through the real LLM,
# verifying that:
#   - Pass 2 can further compress pass 1's output
#   - raw_archives enables the get_raw tool for peeking at originals
#   - Triple-pass chaining accumulates entries correctly across all passes

_MULTI_PASS_CONTACT_LIST = json.dumps(
    {
        "contacts": [
            {
                "id": i,
                "name": name,
                "email": f"{name.lower().replace(' ', '.')}@example.com",
                "phone": f"+1555{i:07d}",
                "address": addr,
                "notes": notes,
                "tags": tags,
                "company": company,
                "title": title,
            }
            for i, (name, addr, notes, tags, company, title) in enumerate(
                [
                    (
                        "Alice Smith",
                        "123 Oak St, Springfield IL 62704",
                        "Prefers morning meetings. Has a dog named Rex.",
                        ["vip", "engineering"],
                        "Acme Corp",
                        "Senior Engineer",
                    ),
                    (
                        "Bob Jones",
                        "456 Maple Ave, Portland OR 97201",
                        "Referred by Alice. Working on Project Phoenix.",
                        ["engineering", "onsite"],
                        "Acme Corp",
                        "Junior Engineer",
                    ),
                    (
                        "Carol White",
                        "789 Pine Rd, Austin TX 73301",
                        "Expert in ML pipelines. Ex-colleague from TechStart.",
                        ["data-science"],
                        "DataFlow Inc",
                        "Lead Data Scientist",
                    ),
                    (
                        "David Brown",
                        "321 Elm Blvd, Seattle WA 98101",
                        "Met at PyCon 2024. Interested in OSS collaboration.",
                        ["open-source", "python"],
                        "IndieCode LLC",
                        "Founder",
                    ),
                    (
                        "Eve Davis",
                        "654 Cedar Ln, Denver CO 80201",
                        "Client contact for Q3 deliverable. Responsive on Slack.",
                        ["client", "priority"],
                        "BigClient Co",
                        "Product Manager",
                    ),
                ],
            )
        ],
        "total": 5,
        "page": 1,
        "per_page": 50,
    },
)

_MULTI_PASS_TRACEBACK = (
    "Traceback (most recent call last):\n"
    '  File "/app/services/contact_service.py", line 287, in search_contacts\n'
    "    results = await self._database.query(params)\n"
    '  File "/app/database/postgres.py", line 154, in query\n'
    "    validated = self._validate_params(params)\n"
    '  File "/app/database/postgres.py", line 89, in _validate_params\n'
    "    for key, value in params.items():\n"
    '  File "/app/database/validators.py", line 42, in validate_field\n'
    "    raise ValueError(\n"
    "ValueError: invalid search parameter 'status'. "
    "Supported parameters are: 'name', 'id', 'phone', 'email', 'company'. "
    "Please use one of the supported parameters listed above.\n"
    "\n"
    "During handling of the above exception, another exception occurred:\n"
    "\n"
    "Traceback (most recent call last):\n"
    '  File "/app/api/endpoints/contacts.py", line 45, in handle_request\n'
    "    return await service.process(request_data)\n"
    '  File "/app/services/contact_service.py", line 300, in process\n'
    "    return self._format_error_response(e, request_id=req.id)\n"
    "ServiceError: Failed to process contact search request"
)


def _mp_pass1() -> list[dict]:
    """Verbose conversation: contact lookup + failed search with traceback."""
    return [
        {
            "role": "user",
            "content": "List all contacts at Acme Corp and tell me their emails.",
        },
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": "get_contacts",
                        "arguments": '{"company": "Acme Corp"}',
                    },
                },
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "call_1",
            "name": "get_contacts",
            "content": _MULTI_PASS_CONTACT_LIST,
        },
        {
            "role": "assistant",
            "content": (
                "I found 5 contacts. The Acme Corp employees are:\n"
                "- Alice Smith: alice.smith@example.com\n"
                "- Bob Jones: bob.jones@example.com\n\n"
                "The other contacts (Carol, David, Eve) are at different companies."
            ),
        },
        {"role": "user", "content": "Search for contacts with status=active"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_2",
                    "type": "function",
                    "function": {
                        "name": "search_contacts",
                        "arguments": '{"status": "active"}',
                    },
                },
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "call_2",
            "name": "search_contacts",
            "content": json.dumps({"error": _MULTI_PASS_TRACEBACK}),
        },
        {
            "role": "assistant",
            "content": (
                "The 'status' field is not a supported search parameter. "
                "I can search by name, id, phone, email, or company. "
                "Would you like me to try a different search?"
            ),
        },
    ]


def _mp_pass2() -> list[dict]:
    """Follow-up: user asks for Alice's phone number."""
    return [
        {"role": "user", "content": "OK, just find Alice's phone number."},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_3",
                    "type": "function",
                    "function": {
                        "name": "get_contact",
                        "arguments": '{"name": "Alice Smith"}',
                    },
                },
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "call_3",
            "name": "get_contact",
            "content": json.dumps(
                {
                    "id": 0,
                    "name": "Alice Smith",
                    "email": "alice.smith@example.com",
                    "phone": "+15550000000",
                    "company": "Acme Corp",
                },
            ),
        },
        {
            "role": "assistant",
            "content": "Alice Smith's phone number is +15550000000.",
        },
    ]


def _mp_pass3() -> list[dict]:
    """Another follow-up: user asks for Bob's email."""
    return [
        {"role": "user", "content": "Now find Bob's email too."},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_4",
                    "type": "function",
                    "function": {
                        "name": "get_contact",
                        "arguments": '{"name": "Bob Jones"}',
                    },
                },
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "call_4",
            "name": "get_contact",
            "content": json.dumps(
                {
                    "id": 1,
                    "name": "Bob Jones",
                    "email": "bob.jones@example.com",
                    "phone": "+15550000001",
                    "company": "Acme Corp",
                },
            ),
        },
        {
            "role": "assistant",
            "content": "Bob Jones' email is bob.jones@example.com.",
        },
    ]


def _accumulate_entries(
    result: CompressedMessages,
    new_indices: list[int],
    prior_entries: list[tuple[int, str]] | None = None,
) -> list[tuple[int, str]]:
    """Extract accumulated (index, content) entries from a compression result.

    Mirrors the entry accumulation logic in _restart_with_compressed_context:
    re-compressed prior entries keep their original global indices, new entries
    get their global indices from new_indices.
    """
    entries: list[tuple[int, str]] = []
    n_prior = len(prior_entries) if prior_entries else 0

    if prior_entries:
        for (orig_idx, _), comp in zip(prior_entries, result.messages[:n_prior]):
            entries.append((orig_idx, comp.content))

    for global_idx, comp in zip(new_indices, result.messages[n_prior:]):
        entries.append((global_idx, comp.content))

    return entries


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


# ── Image-aware compression tests ────────────────────────────────────────────


# Valid 1x1 pixel PNGs (red / blue) accepted by all multimodal LLM APIs.
_IMG_PNG_BLOCK = {
    "type": "image_url",
    "image_url": {
        "url": (
            "data:image/png;base64,"
            "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAIAAACQd1PeAAAADElEQVR4"
            "nGP4z8AAAAMBAQDJ/pLvAAAAAElFTkSuQmCC"
        ),
    },
}
_IMG_PNG_BLOCK_2 = {
    "type": "image_url",
    "image_url": {
        "url": (
            "data:image/png;base64,"
            "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAIAAACQd1PeAAAADElEQVR4"
            "nGNgYPgPAAEDAQAIicLsAAAAAElFTkSuQmCC"
        ),
    },
}


class TestTagImagesInMessages:
    def test_single_image_tagged(self):
        msgs = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What is this?"},
                    _IMG_PNG_BLOCK,
                ],
            },
        ]
        tagged, registry, next_id = tag_images_in_messages(msgs)
        assert next_id == 1
        assert 0 in registry
        assert registry[0] is _IMG_PNG_BLOCK
        content = tagged[0]["content"]
        assert any("[img:0]" in b.get("text", "") for b in content)
        assert all(b.get("type") != "image_url" for b in content)

    def test_multiple_images_across_messages(self):
        msgs = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Compare these:"},
                    _IMG_PNG_BLOCK,
                    _IMG_PNG_BLOCK_2,
                ],
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "And this one:"},
                    _IMG_PNG_BLOCK,
                ],
            },
        ]
        tagged, registry, next_id = tag_images_in_messages(msgs)
        assert next_id == 3
        assert len(registry) == 3
        assert set(registry.keys()) == {0, 1, 2}

    def test_start_id_offset(self):
        msgs = [
            {
                "role": "user",
                "content": [{"type": "text", "text": "hi"}, _IMG_PNG_BLOCK],
            },
        ]
        tagged, registry, next_id = tag_images_in_messages(msgs, start_id=10)
        assert next_id == 11
        assert 10 in registry
        content = tagged[0]["content"]
        assert any("[img:10]" in b.get("text", "") for b in content)

    def test_preserves_text_only_messages(self):
        msgs = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there"},
        ]
        tagged, registry, next_id = tag_images_in_messages(msgs)
        assert next_id == 0
        assert len(registry) == 0
        assert tagged[0]["content"] == "Hello"
        assert tagged[1]["content"] == "Hi there"

    def test_preserves_thinking_blocks(self):
        msgs = [
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "thinking",
                        "thinking": "reasoning...",
                        "signature": "sig==",
                    },
                    {"type": "text", "text": "answer"},
                    _IMG_PNG_BLOCK,
                ],
            },
        ]
        tagged, registry, next_id = tag_images_in_messages(msgs)
        assert next_id == 1
        types = [b.get("type") for b in tagged[0]["content"]]
        assert "thinking" in types
        assert "image_url" not in types

    def test_does_not_mutate_input(self):
        original = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "look"},
                    copy.deepcopy(_IMG_PNG_BLOCK),
                ],
            },
        ]
        snapshot = copy.deepcopy(original)
        tag_images_in_messages(original)
        assert original == snapshot


class TestScanSurvivingImageIds:
    def test_finds_tags(self):
        entries = {
            0: '{"role":"user","content":"see [img:0] and [img:3]"}',
            1: '{"role":"assistant","content":"noted [img:0]"}',
        }
        assert _scan_surviving_image_ids(entries) == {0, 3}

    def test_no_tags(self):
        entries = {0: '{"role":"user","content":"no images here"}'}
        assert _scan_surviving_image_ids(entries) == set()

    def test_empty_entries(self):
        assert _scan_surviving_image_ids({}) == set()

    def test_tag_removed_not_in_result(self):
        entries = {
            0: '{"content":"kept [img:1]"}',
            1: '{"content":"this had img:2 but tag was stripped"}',
        }
        assert _scan_surviving_image_ids(entries) == {1}


# ── Eval tests: image-aware compress_messages ─────────────────────────────────


def _build_image_conversation() -> tuple[list[dict], dict[int, dict]]:
    """Build a tagged conversation with images and its registry."""
    msgs = [
        {"role": "user", "content": "What do you see in these screenshots?"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": "analyze_image",
                        "arguments": '{"image": "[img:0]"}',
                    },
                },
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "call_1",
            "name": "analyze_image",
            "content": (
                "The image [img:0] shows a detailed dashboard with multiple "
                "charts, KPI metrics, and a navigation sidebar. The main chart "
                "displays revenue over time with a clear upward trend. There are "
                "also pie charts for market segmentation and a table of top "
                "customers sorted by lifetime value."
            ),
        },
        {
            "role": "assistant",
            "content": (
                "I can see a business dashboard in [img:0]. It shows revenue "
                "trending upward with market segmentation data."
            ),
        },
        {"role": "user", "content": "OK, now look at this error screenshot."},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_2",
                    "type": "function",
                    "function": {
                        "name": "analyze_image",
                        "arguments": '{"image": "[img:1]"}',
                    },
                },
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "call_2",
            "name": "analyze_image",
            "content": (
                "The image [img:1] shows a browser console with a JavaScript "
                "TypeError: Cannot read properties of undefined (reading 'map'). "
                "The stack trace points to Dashboard.tsx line 142."
            ),
        },
        {
            "role": "assistant",
            "content": (
                "The error in [img:1] is a TypeError in Dashboard.tsx line 142 "
                "where it tries to call .map() on an undefined value. This is "
                "likely because the data array hasn't loaded yet."
            ),
        },
        {
            "role": "user",
            "content": "Great, I fixed the bug. Now just focus on the revenue data from the dashboard.",
        },
        {
            "role": "assistant",
            "content": (
                "Based on the dashboard in [img:0], the revenue shows a clear "
                "upward trend. The error screenshot [img:1] is no longer "
                "relevant since you've fixed the bug."
            ),
        },
    ]
    registry = {
        0: _IMG_PNG_BLOCK,
        1: _IMG_PNG_BLOCK_2,
    }
    return msgs, registry


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
