"""
tests/async_tool_loop/test_inspection_digest.py
================================================

Tests for digest-first inspection of COMPLETED handles: ``digest()``, its
``read_child_message(idx)`` drill-down tool, and the branch in
``AsyncToolLoopHandle.ask()`` that picks digest-vs-live-snapshot based on
``done()`` at invocation time.

A still-running handle keeps the pre-existing full-transcript snapshot
behavior — those tests live in ``test_ask_inflight_status.py`` and
``test_context_propagation.py``. This file covers only the new completed-
handle path.
"""

from __future__ import annotations

import asyncio
import json

import pytest

from unify.common import async_tool_loop as atl
from unify.common._async_tool.transcript_ops import _DIGEST_MAX_TURNS
from unify.common.async_tool_loop import AsyncToolLoopHandle, start_async_tool_loop
from unify.common.llm_client import new_llm_client
from unify.common.token_utils import count_tokens
from unify.common.tool_spec import llm_soft_required
from tests.baked_defaults import as_shipped  # noqa: F401


class _DummyClient:
    endpoint = "openai/gpt-4o-mini@openrouter"

    def __init__(self, messages: list[dict]) -> None:
        self.messages = messages


def _make_completed_handle(
    messages: list[dict],
    *,
    request: str = "start",
    task_result: str = "done",
) -> AsyncToolLoopHandle:
    """Build a handle whose task is already done() — the digest branch.

    ``task_result`` is what ``self._task.result()`` returns — the
    authoritative source ``digest()`` reads for ``final_result``.
    """
    task: asyncio.Future = asyncio.Future()
    task.set_result(task_result)
    return AsyncToolLoopHandle(
        task=task,
        interject_queue=asyncio.Queue(),
        cancel_event=asyncio.Event(),
        stop_event=asyncio.Event(),
        client=_DummyClient(messages),
        loop_id="TestLoop",
        initial_user_message=request,
    )


_SAMPLE_MESSAGES = [
    {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": "call_1",
                "type": "function",
                "function": {
                    "name": "search_web",
                    "arguments": json.dumps(
                        {
                            "thought": "look up sea level data",
                            "query": "sea level rise",
                        },
                    ),
                },
            },
        ],
    },
    {
        "role": "tool",
        "tool_call_id": "call_1",
        "name": "search_web",
        "content": ("Found at https://noaa.example.com/report — rate is 3.4mm/year."),
    },
    {
        "role": "assistant",
        "content": "Sea level is rising at 3.4mm/year.",
        "tool_calls": [],
    },
]


# ---------------------------------------------------------------------------
# digest() — mechanical structure, no LLM call
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_digest_mechanical_structure():
    handle = _make_completed_handle(
        _SAMPLE_MESSAGES,
        request="What is the sea level rise rate?",
        # Matches the transcript's own final assistant message — a plain
        # (non-structured-output) loop's task result equals that text.
        task_result="Sea level is rising at 3.4mm/year.",
    )

    digest_obj = json.loads(handle.digest())

    assert digest_obj["request"] == "What is the sea level rise rate?"
    assert digest_obj["final_result"] == "Sea level is rising at 3.4mm/year."
    assert digest_obj["sources"] == ["https://noaa.example.com/report"]

    assert len(digest_obj["turns"]) == 1
    turn = digest_obj["turns"][0]
    assert turn["tool"] == "search_web"
    assert turn["thought"] == "look up sea level data"
    assert "3.4mm/year" in turn["result_head"]
    assert turn["idx"] == 1  # position of the tool message in the transcript
    expected_bytes = len(
        "Found at https://noaa.example.com/report — rate is 3.4mm/year.".encode(
            "utf-8",
        ),
    )
    assert turn["result_bytes"] == expected_bytes


@pytest.mark.asyncio
async def test_digest_is_cached_and_byte_stable():
    handle = _make_completed_handle(_SAMPLE_MESSAGES)

    first = handle.digest()
    second = handle.digest()

    # Same object — proves it was memoized, not rebuilt.
    assert first is second


@pytest.mark.asyncio
async def test_digest_on_unfinished_handle_raises():
    """digest() must refuse rather than mislabel: reading self._task.result()
    on a still-running task raises its own InvalidStateError, which the
    generic error handler below the done() guard would otherwise catch and
    render as a false "this run ended with an error" outcome for a run that
    neither errored nor finished."""
    task: asyncio.Future = asyncio.Future()  # deliberately left unresolved
    handle = AsyncToolLoopHandle(
        task=task,
        interject_queue=asyncio.Queue(),
        cancel_event=asyncio.Event(),
        stop_event=asyncio.Event(),
        client=_DummyClient(_SAMPLE_MESSAGES),
        loop_id="TestLoop",
    )

    with pytest.raises(asyncio.InvalidStateError):
        handle.digest()


@pytest.mark.asyncio
async def test_digest_final_result_uses_task_result_not_narration_heuristic():
    """final_result must come from the task's actual result,
    not a heuristic scan that picks up pre-answer narration on structured-
    output loops — there the real answer lands in a tool message (e.g. via
    final_response), never a bare assistant message, so the old "last
    assistant message with content and no tool_calls" heuristic silently
    picked up whatever narration preceded it instead."""
    messages = [
        {
            "role": "assistant",
            "content": "Working on it, one moment.",
            "tool_calls": [],
        },
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_fr",
                    "type": "function",
                    "function": {
                        "name": "final_response",
                        "arguments": json.dumps({"answer": "THE REAL ANSWER"}),
                    },
                },
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "call_fr",
            "name": "final_response",
            "content": "submitted",
        },
    ]
    handle = _make_completed_handle(messages, task_result="THE REAL ANSWER")

    digest_obj = json.loads(handle.digest())

    assert digest_obj["final_result"] == "THE REAL ANSWER"
    assert digest_obj["final_result"] != "Working on it, one moment."


@pytest.mark.asyncio
async def test_digest_final_result_records_task_error_truthfully():
    """A task that errored gets a truthfully-labeled error outcome as its
    final_result, not a silent fallback to unrelated narration."""
    task: asyncio.Future = asyncio.Future()
    task.set_exception(RuntimeError("boom"))
    handle = AsyncToolLoopHandle(
        task=task,
        interject_queue=asyncio.Queue(),
        cancel_event=asyncio.Event(),
        stop_event=asyncio.Event(),
        client=_DummyClient(
            [{"role": "assistant", "content": "unrelated narration", "tool_calls": []}],
        ),
        loop_id="TestLoop",
    )

    digest_obj = json.loads(handle.digest())

    assert "RuntimeError" in digest_obj["final_result"]
    assert "boom" in digest_obj["final_result"]


# ---------------------------------------------------------------------------
# read_child_message(idx) — drill-down tool
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_read_child_message_returns_verbatim_message():
    handle = _make_completed_handle(_SAMPLE_MESSAGES)
    handle.digest()  # populates _digest_messages
    read_child_message = handle._make_read_child_message_tool()

    result = await read_child_message(1)
    parsed = json.loads(result)

    assert parsed["role"] == "tool"
    assert "noaa.example.com" in parsed["content"]


@pytest.mark.asyncio
async def test_read_child_message_out_of_range():
    handle = _make_completed_handle(_SAMPLE_MESSAGES)
    handle.digest()
    read_child_message = handle._make_read_child_message_tool()

    result = await read_child_message(999)

    assert "No message at idx=999" in result


@pytest.mark.asyncio
async def test_read_child_message_caps_at_32kb():
    big_content = "A" * 50_000
    handle = _make_completed_handle(
        [{"role": "tool", "tool_call_id": "x", "name": "dump", "content": big_content}],
    )
    handle.digest()
    read_child_message = handle._make_read_child_message_tool()

    result = await read_child_message(0)

    assert len(result) < 50_000
    assert "omitted" in result


@pytest.mark.asyncio
async def test_digest_and_read_child_message_redact_image_payloads():
    """The completed-handle path (digest + drill-down) must redact image
    blobs, same as the live-snapshot path — both go through
    make_messages_safe_for_context_dump. See test_context_propagation.py's
    test_ask_inspection_prompt_redacts_image_payloads for the live-path
    sibling of this test."""
    big_b64 = "A" * 4000
    raw_data_url = f"data:image/png;base64,{big_b64}"
    messages = [
        {
            "role": "tool",
            "tool_call_id": "call_1",
            "name": "execute_code",
            "content": [
                {"type": "text", "text": "screenshot captured"},
                {"type": "image_url", "image_url": {"url": raw_data_url}},
            ],
        },
    ]
    handle = _make_completed_handle(messages)

    digest_text = handle.digest()
    assert raw_data_url not in digest_text

    read_child_message = handle._make_read_child_message_tool()
    child_msg = await read_child_message(0)
    assert raw_data_url not in child_msg
    assert "data:image/png;base64,<omitted>" in child_msg


# ---------------------------------------------------------------------------
# ask() branch selection: completed -> digest, running -> live snapshot
# ---------------------------------------------------------------------------


class _DummyInspectionHandle:
    async def result(self):
        return "ok"


def _capture_start_async_tool_loop(monkeypatch, sink: list[dict]):
    def _fake(*args, **kwargs):
        sink.append({"system_message": args[0].system_message, "tools": args[2]})
        return _DummyInspectionHandle()

    monkeypatch.setattr(atl, "start_async_tool_loop", _fake)


@pytest.mark.asyncio
async def test_completed_ask_uses_digest_and_offers_read_child_message(monkeypatch):
    captured: list[dict] = []
    _capture_start_async_tool_loop(monkeypatch, captured)

    handle = _make_completed_handle(_SAMPLE_MESSAGES)
    helper = await handle.ask("What sources were used?")
    await helper.result()

    sys_msg = captured[0]["system_message"]
    assert sys_msg.startswith("You are inspecting a COMPLETED tool-use conversation")
    assert "search_web" in sys_msg  # digest embedded
    assert "read_child_message" in captured[0]["tools"]


@pytest.mark.asyncio
async def test_repeat_completed_ask_is_byte_identical(monkeypatch):
    """Acceptance: repeat ask on the same handle must cache-hit — same bytes."""
    captured: list[dict] = []
    _capture_start_async_tool_loop(monkeypatch, captured)

    handle = _make_completed_handle(_SAMPLE_MESSAGES)

    h1 = await handle.ask("What sources were consulted?")
    await h1.result()
    h2 = await handle.ask("What was the final result?")
    await h2.result()

    assert len(captured) == 2
    assert captured[0]["system_message"] == captured[1]["system_message"], (
        "the inspection system prompt for a completed handle must be byte-"
        "identical across asks so the provider prefix-cache hits"
    )


@pytest.mark.asyncio
async def test_ask_on_running_handle_keeps_live_snapshot(monkeypatch):
    """A still-running handle must NOT get the digest — done() is False."""
    captured: list[dict] = []
    _capture_start_async_tool_loop(monkeypatch, captured)

    # Deliberately unresolved: done() must read False.
    task: asyncio.Future = asyncio.Future()
    handle = AsyncToolLoopHandle(
        task=task,
        interject_queue=asyncio.Queue(),
        cancel_event=asyncio.Event(),
        stop_event=asyncio.Event(),
        client=_DummyClient(_SAMPLE_MESSAGES),
        loop_id="RunningLoop",
    )

    helper = await handle.ask("What is happening?")
    await helper.result()

    sys_msg = captured[0]["system_message"]
    assert "COMPLETED tool-use conversation" not in sys_msg
    assert "Inspected Loop Transcript" in sys_msg
    assert "read_child_message" not in captured[0]["tools"]


# ---------------------------------------------------------------------------
# Acceptance #3 proxy: inspection requests stay bounded regardless of the
# underlying transcript's size (was unbounded — up to 204k tokens/request).
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_digest_stays_bounded_regardless_of_transcript_size(monkeypatch):
    captured: list[dict] = []
    _capture_start_async_tool_loop(monkeypatch, captured)

    # ~240KB raw transcript (60 tool turns x ~4KB each) — would be tens of
    # thousands of tokens if dumped in full, as the pre-digest ask() did.
    messages: list[dict] = []
    for i in range(60):
        messages.append(
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": f"call_{i}",
                        "type": "function",
                        "function": {
                            "name": "step",
                            "arguments": json.dumps({"thought": f"step {i}", "n": i}),
                        },
                    },
                ],
            },
        )
        messages.append(
            {
                "role": "tool",
                "tool_call_id": f"call_{i}",
                "name": "step",
                "content": f"result {i}: " + ("x" * 4000),
            },
        )
    messages.append({"role": "assistant", "content": "done", "tool_calls": []})

    handle = _make_completed_handle(messages)
    helper = await handle.ask("Summarize what happened.")
    await helper.result()

    tokens = count_tokens(captured[0]["system_message"])
    assert tokens < 20_000, (
        f"digest-based inspection prompt should stay well under 20k tokens "
        f"regardless of transcript size, got {tokens}"
    )


@pytest.mark.asyncio
async def test_digest_caps_turn_count_with_elision_marker():
    """Turn *count* (not just transcript size) must stay bounded — an
    uncapped digest grows ~45 tokens/turn, blowing past the ~2k-token design
    budget for digests well before 100 turns. A 400-turn run should still produce a
    compact digest with a head+tail window and one explicit elision marker
    whose idx_range keeps the elided turns reachable via read_child_message."""
    n_turns = 400
    messages: list[dict] = []
    for i in range(n_turns):
        # Realistic-length thought/result text (not a bare number) — this is
        # what makes the token math representative of a real trajectory
        # rather than an artificially tiny one.
        messages.append(
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": f"call_{i}",
                        "type": "function",
                        "function": {
                            "name": "step",
                            "arguments": json.dumps(
                                {
                                    "thought": f"process batch {i} of the incoming queue and validate the records",
                                    "n": i,
                                },
                            ),
                        },
                    },
                ],
            },
        )
        messages.append(
            {
                "role": "tool",
                "tool_call_id": f"call_{i}",
                "name": "step",
                "content": f"step {i} completed: processed {i} records from the batch, no errors encountered.",
            },
        )
    messages.append({"role": "assistant", "content": "done", "tool_calls": []})

    handle = _make_completed_handle(messages, task_result="done")
    digest_text = handle.digest()
    digest_obj = json.loads(digest_text)

    turns = digest_obj["turns"]
    max_turns = _DIGEST_MAX_TURNS
    assert len(turns) == max_turns + 1, "head + one elision marker + tail"

    markers = [t for t in turns if isinstance(t, dict) and t.get("elided")]
    assert len(markers) == 1
    marker = markers[0]
    assert marker["count"] == n_turns - max_turns
    assert len(marker["idx_range"]) == 2
    assert marker["idx_range"][0] < marker["idx_range"][1]

    # Elided idx values are still individually retrievable.
    handle.digest()  # no-op (cached) — populates _digest_messages already
    read_child_message = handle._make_read_child_message_tool()
    elided_idx = marker["idx_range"][0]
    result = await read_child_message(elided_idx)
    assert json.loads(result)["role"] == "tool"

    tokens = count_tokens(digest_text)
    assert 1000 < tokens < 3000, (
        f"a capped digest should land in the ~2k-3k token design budget regardless "
        f"of trajectory length, got {tokens}"
    )


# ---------------------------------------------------------------------------
# End-to-end: digest answers the five librarian question shapes, with
# drill-down fetching a detail the digest's head-only preview omits.
# ---------------------------------------------------------------------------


# as_shipped: deleted in step 5 (asking completed steerable handles)
@pytest.mark.usefixtures("as_shipped")
@pytest.mark.asyncio
@pytest.mark.llm_call
async def test_digest_ask_answers_librarian_question_shapes(llm_config):
    """Approximates the five librarian question shapes a storage-review pass
    asks about a completed sub-agent: sources, result, procedure, rationale,
    and a buried detail that forces read_child_message drill-down (the
    dataset code below sits on line 4 of a multi-line tool result, so the
    digest's first-line-only preview never carries it)."""

    @llm_soft_required(thought="")
    async def search_web(thought: str, query: str) -> str:
        """Search the web for information relevant to the query.

        Parameters
        ----------
        thought : str
            One-sentence rationale for why this search is being made.
        query : str
            The search query.
        """
        return (
            "Search results for sea level rise:\n"
            "1. NOAA Tide Gauge Network summary — https://noaa.example.com/report-2024\n"
            "2. Coverage: 12 Atlantic coast stations, 1993-2024 continuous record\n"
            "3. Headline figure: 3.4mm/year average rise, +/-0.2mm/year uncertainty\n"
            "4. Cross-validated against satellite altimetry (dataset ref: SLR-ALT-7742)\n"
            "5. No station-level anomalies detected in the review window."
        )

    @llm_soft_required(thought="")
    async def compute_summary(thought: str, data: str) -> str:
        """Compute a one-line summary from gathered research data.

        Parameters
        ----------
        thought : str
            One-sentence rationale for this computation.
        data : str
            The research data to summarize.
        """
        return "Confirmed rate: 3.4mm/year, consistent across all stations."

    client = new_llm_client(**llm_config)
    client.set_system_message(
        "You are a research agent. Perform these steps in order:\n"
        "1. Call `search_web` with a `thought` and a query about sea level rise.\n"
        "2. Call `compute_summary` with a `thought`, passing the data you found.\n"
        "3. Reply with **only** the final summary sentence returned by `compute_summary`.",
    )

    handle = start_async_tool_loop(
        client=client,
        message="start",
        tools={"search_web": search_web, "compute_summary": compute_summary},
        max_steps=20,
        timeout=180,
    )

    final_result = await handle.result()
    assert final_result is not None, "Inner run should complete"
    assert handle.done(), "Handle must be completed for the digest branch"

    questions = {
        "sources": "What sources or URLs were consulted during this run?",
        "result": "What is the final conclusion, and how confident is it?",
        "steps": "What steps (tools) were taken, in order, to reach the conclusion?",
        "rationale": "Why was `search_web` called — what was the reasoning behind it?",
        "detail": (
            "What is the exact dataset reference code used to cross-validate "
            "the sea level data? Reply with just the code."
        ),
    }

    answers: dict[str, str] = {}
    for key, question in questions.items():
        helper = await handle.ask(question)
        answers[key] = await helper.result()

    assert "noaa.example.com" in answers["sources"].lower(), answers["sources"]
    assert "3.4" in answers["result"], answers["result"]
    assert any(
        term in answers["steps"].lower()
        for term in ("search_web", "search", "compute_summary")
    ), answers["steps"]
    assert answers["rationale"].strip(), "rationale answer should not be empty"
    assert "SLR-ALT-7742" in answers["detail"], (
        "the dataset code sits past the digest's first-line preview — answering "
        f"it correctly requires a read_child_message drill-down, got: {answers['detail']!r}"
    )
