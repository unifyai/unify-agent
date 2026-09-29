"""A scripted LLM transport and the request scenarios the cache-discipline tests share.

The clients are real ``unillm`` clients; only their transport
(``litellm.acompletion`` behind unillm's retry wrapper) is replaced, so every
request is recorded exactly as unillm would send it and nothing leaves the
process. The scenarios use only APIs that exist upstream, so the requests a
scenario sends with every switch off can be recorded on the upstream commit
and replayed here as the equivalence baseline (``cache_discipline_golden.json``,
written by ``python -m tests.cache_discipline_helpers --record``).
"""

from __future__ import annotations

import asyncio
import contextlib
import copy
import itertools
import json
import os
from pathlib import Path
from typing import Any, Callable, Iterator, Optional

from openai.types.chat import ChatCompletion

MODEL = "openai/gpt-5.6-sol@openrouter"
_CALL_SEQ = itertools.count()
GOLDEN = Path(__file__).with_name("cache_discipline_golden.json")
_TRANSPORT = "unillm.clients.uni_llm._acompletion_with_transient_retry"


def completion(
    content: Optional[str] = None,
    calls: list[tuple[str, dict]] = (),
    *,
    prompt_tokens: int = 100,
    cached_tokens: Optional[int] = None,
    call_ids: Optional[list[str]] = None,
    reasoning: Optional[str] = None,
) -> ChatCompletion:
    """One provider reply: text, tool calls, and usage."""
    tool_calls = [
        {
            "id": (call_ids[i] if call_ids else f"call_{next(_CALL_SEQ)}"),
            "type": "function",
            "function": {"name": name, "arguments": json.dumps(args)},
        }
        for i, (name, args) in enumerate(calls)
    ]
    usage: dict[str, Any] = {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": 5,
        "total_tokens": prompt_tokens + 5,
    }
    if cached_tokens is not None:
        usage["prompt_tokens_details"] = {"cached_tokens": cached_tokens}
    message: dict[str, Any] = {
        "role": "assistant",
        "content": content,
        "tool_calls": tool_calls or None,
    }
    if reasoning is not None:
        message["reasoning_content"] = reasoning
        message["reasoning_details"] = [
            {"type": "reasoning.encrypted", "data": reasoning * 20},
        ]
        message["provider_specific_fields"] = {"reasoning_signature": reasoning}
    return ChatCompletion.model_validate(
        {
            "id": "cmpl-test",
            "object": "chat.completion",
            "created": 0,
            "model": "openai/gpt-5.6-sol",
            "choices": [
                {
                    "index": 0,
                    "finish_reason": "tool_calls" if tool_calls else "stop",
                    "message": message,
                },
            ],
            "usage": usage,
        },
    )


class Provider:
    """Scripted transport: records each request, plays the replies in order."""

    def __init__(self, replies=()):
        self.replies = list(replies)
        self.requests: list[dict] = []

    async def __call__(self, *, shared_session=None, client=None, **kw):
        self.requests.append(
            copy.deepcopy(
                {
                    k: kw.get(k)
                    for k in ("messages", "tools", "tool_choice", "reasoning_effort")
                },
            ),
        )
        if not self.replies:
            raise AssertionError(
                f"no scripted reply left for request {len(self.requests)}",
            )
        reply = self.replies.pop(0)
        return reply() if callable(reply) else reply


@contextlib.contextmanager
def scripted(replies) -> Iterator[Provider]:
    """Install a :class:`Provider` as unillm's transport for the block."""
    import unillm.clients.uni_llm as uni_llm
    from unillm.settings import SETTINGS as unillm_settings

    global _CALL_SEQ
    _CALL_SEQ = itertools.count()
    provider = Provider(replies)
    original = uni_llm._acompletion_with_transient_retry
    old_cache = os.environ.get("UNILLM_CACHE")
    old_default = unillm_settings.UNILLM_CACHE
    uni_llm._acompletion_with_transient_retry = provider
    # Clients built inside the scenario (a compactor loop, say) take their
    # cache mode from unillm's settings; none of them may read a recording.
    os.environ["UNILLM_CACHE"] = "false"
    unillm_settings.UNILLM_CACHE = False
    try:
        yield provider
    finally:
        uni_llm._acompletion_with_transient_retry = original
        unillm_settings.UNILLM_CACHE = old_default
        if old_cache is None:
            os.environ.pop("UNILLM_CACHE", None)
        else:
            os.environ["UNILLM_CACHE"] = old_cache


def _canonical_messages(messages: list) -> list:
    """Messages with unillm's own retry notices put in a stable order.

    When a model calls a tool outside the request's schema, unillm retries
    with a tool error listing the available tools from a ``set``, whose
    order changes with the hash seed. That list is sorted here; nothing
    else is touched.
    """
    out = []
    for message in messages or []:
        content = message.get("content") if isinstance(message, dict) else None
        if message.get("role") == "tool" and isinstance(content, str):
            try:
                parsed = json.loads(content)
            except ValueError:
                parsed = None
            error = parsed.get("error") if isinstance(parsed, dict) else None
            if isinstance(error, dict) and isinstance(
                error.get("available_tools"),
                list,
            ):
                error["available_tools"] = sorted(error["available_tools"])
                message = {**message, "content": json.dumps(parsed)}
        out.append(message)
    return out


def request_bytes(request: dict) -> dict:
    """A request as the JSON strings compared byte for byte."""
    return {
        "messages": json.dumps(_canonical_messages(request["messages"]), default=str),
        "tools": json.dumps(request["tools"], default=str),
        "tool_choice": json.dumps(request["tool_choice"], default=str),
    }


def new_client(system: str = "You are a scripted test agent."):
    from unify.common.llm_client import new_llm_client

    client = new_llm_client(MODEL, cache=False, reasoning_effort="low")
    client.set_system_message(system)
    return client


# ── tools ────────────────────────────────────────────────────────────────


def make_tools(counter: dict) -> dict[str, Callable]:
    """Two library searches and a code runner; each call is counted."""

    async def FunctionManager_search_functions(query: str) -> str:
        """Search stored functions by meaning.

        Args:
            query: What the function should do.
        """
        counter["FunctionManager_search_functions"] = (
            counter.get("FunctionManager_search_functions", 0) + 1
        )
        return "[]"

    async def GuidanceManager_search(k: int) -> str:
        """Search stored guidance.

        Args:
            k: How many entries to return.
        """
        counter["GuidanceManager_search"] = counter.get("GuidanceManager_search", 0) + 1
        return "[]"

    async def execute_code(code: str) -> str:
        """Run Python code in the sandbox.

        Args:
            code: The code to run.
        """
        counter["execute_code"] = counter.get("execute_code", 0) + 1
        return f"ran {code}"

    return {
        "FunctionManager_search_functions": FunctionManager_search_functions,
        "GuidanceManager_search": GuidanceManager_search,
        "execute_code": execute_code,
    }


def gate_policy(step, tools, called_tools):
    """A discovery-first gate like the actor's: both searches before anything else."""
    fm = any(t.startswith("FunctionManager_") for t in called_tools)
    gm = any(t.startswith("GuidanceManager_") for t in called_tools)
    if fm and gm:
        return "auto", tools
    gated = {
        n: f
        for n, f in tools.items()
        if (n == "FunctionManager_search_functions" and not fm)
        or (n == "GuidanceManager_search" and not gm)
    }
    return (
        "required",
        gated,
        {"eager": True, "mask_rule": "the libraries are searched first"},
    )


# ── scenarios ────────────────────────────────────────────────────────────


async def _run(client, tools, message, **loop_kwargs) -> str:
    from unify.common.async_tool_loop import start_async_tool_loop

    handle = start_async_tool_loop(
        client,
        message,
        tools,
        log_steps=False,
        timeout=60,
        max_steps=30,
        **loop_kwargs,
    )
    return await asyncio.wait_for(handle.result(), timeout=60)


GATE_REPLIES = (
    lambda: completion(calls=[("execute_code", {"code": "early"})]),
    lambda: completion(
        calls=[
            ("FunctionManager_search_functions", {"query": "q"}),
            ("GuidanceManager_search", {"k": 3}),
        ],
    ),
    lambda: completion(calls=[("execute_code", {"code": "late"})]),
    lambda: completion(content="done"),
)


async def scenario_gate() -> tuple[str, dict, list[dict]]:
    """A discovery gate, a call it masks, the searches, then the code call."""
    counter: dict = {}
    with scripted(GATE_REPLIES) as provider:
        result = await _run(
            new_client(),
            make_tools(counter),
            "Do the task.",
            tool_policy=gate_policy,
            interrupt_llm_with_interjections=False,
        )
    return result, counter, provider.requests


THRESHOLD_REPLIES = (
    lambda: completion(
        calls=[("execute_code", {"code": "big"})],
        prompt_tokens=900_000,
    ),
    lambda: completion(calls=[("execute_code", {"code": "again"})]),
    lambda: completion(content="stopping"),
)


async def scenario_threshold() -> tuple[str, dict, list[dict]]:
    """A context-full turn, where only compress_context may run."""
    counter: dict = {}
    tools = make_tools(counter)
    with scripted(THRESHOLD_REPLIES) as provider:
        result = await _run(
            new_client(),
            {"execute_code": tools["execute_code"]},
            "Do the task.",
            interrupt_llm_with_interjections=False,
        )
    return result, counter, provider.requests


INTERRUPT_REPLIES = (
    lambda: completion(calls=[("execute_code", {"code": "one"})]),
    lambda: completion(content="done"),
)


async def scenario_interrupt() -> tuple[str, dict, list[dict]]:
    """The pre-emptive dispatch path: one code call, then the answer."""
    counter: dict = {}
    tools = make_tools(counter)
    with scripted(INTERRUPT_REPLIES) as provider:
        result = await _run(
            new_client(),
            {"execute_code": tools["execute_code"]},
            "Do the task.",
        )
    return result, counter, provider.requests


PERSIST_REPLIES = (
    lambda: completion(
        calls=[("execute_code", {"code": "x" * 1200})],
        reasoning="thinking about the first request",
    ),
    lambda: completion(content="first done", reasoning="wrapping up"),
    lambda: completion(content="second done"),
)


async def _next_response(handle) -> dict:
    while True:
        notification = await asyncio.wait_for(handle.next_notification(), 30)
        if isinstance(notification, dict) and notification.get("type") == "response":
            return notification


async def scenario_persist() -> tuple[str, dict, list[dict]]:
    """A persistent session: a turn, a storage-review compaction note, a turn.

    Between the turns the session parks (where reasoning payloads used to be
    shed) and receives the ``_compact_transcript`` sentinel a completed turn
    review sends (which used to compact the reviewed span in place).
    """
    from unify.common.async_tool_loop import start_async_tool_loop

    counter: dict = {}
    tools = make_tools(counter)
    client = new_client()
    with scripted(PERSIST_REPLIES) as provider:
        handle = start_async_tool_loop(
            client,
            "First request.",
            {"execute_code": tools["execute_code"]},
            log_steps=False,
            timeout=60,
            persist=True,
        )
        first = await _next_response(handle)
        handle._queue.put_nowait(
            {"_compact_transcript": {"reviewed_messages": len(client.messages)}},
        )
        await handle.interject("Second request.")
        second = await _next_response(handle)
        await handle.stop()
        await asyncio.wait_for(handle.result(), 30)
    return f"{first['content']}|{second['content']}", counter, provider.requests


SUMMARY = "Summary: ran the big code call; next, answer done."

COMPRESS_REPLIES = (
    lambda: completion(
        calls=[("execute_code", {"code": "big"})],
        prompt_tokens=900_000,
    ),
    lambda: completion(calls=[("compress_context", {})]),
    # the summary (a fork) or the compactor's closing reply (as shipped)
    lambda: completion(content=SUMMARY),
    lambda: completion(content="done"),
)


async def scenario_compress(replies=COMPRESS_REPLIES) -> tuple[str, dict, list[dict]]:
    """A context-full turn that compresses, then the answer after the restart."""
    counter: dict = {}
    tools = make_tools(counter)
    with scripted(replies) as provider:
        result = await _run(
            new_client(),
            {"execute_code": tools["execute_code"]},
            "Do the task.",
            interrupt_llm_with_interjections=False,
        )
    return result, counter, provider.requests


REVIEW_REPLIES = (
    # the session
    lambda: completion(calls=[("FunctionManager_list_functions", {})]),
    lambda: completion(content="Listed the stored functions; there are none."),
    # its storage review
    lambda: completion(content="Nothing worth storing."),
)


def session_tools(actor) -> dict[str, Callable]:
    """The library tools the actor and its review share, and a code runner.

    The library tools are the managers' own methods, named as the actor and
    the review name them, so the session's tool list carries the exact
    schemas the review would build.
    """
    from unify.common.llm_helpers import methods_to_tool_dict

    fm, gm = actor.function_manager, actor.guidance_manager
    tools = methods_to_tool_dict(
        fm.list_functions,
        fm.add_functions,
        gm.filter,
        gm.add_guidance,
        include_class_name=True,
    )
    tools["execute_code"] = make_tools({})["execute_code"]
    return tools


async def scenario_review(replies=REVIEW_REPLIES, *, actor=None, tools=None):
    """A session on the actor's own tools, then the storage review after it.

    The session runs through ``_StorageCheckHandle`` exactly as ``act``
    wraps it, so the review starts from the session's real client and
    transcript. Returns the review's summary as the result.
    """
    from unify.actor.code_act_actor import CodeActActor, _StorageCheckHandle
    from unify.common.async_tool_loop import start_async_tool_loop

    own_actor = actor is None
    actor = actor or CodeActActor()
    try:
        tools = tools if tools is not None else session_tools(actor)
        with scripted(replies) as provider:
            inner = start_async_tool_loop(
                new_client("You are a scripted actor."),
                "List the stored functions.",
                tools,
                loop_id="CodeActActor.act",
                log_steps=False,
                timeout=60,
                interrupt_llm_with_interjections=False,
            )
            handle = _StorageCheckHandle(inner=inner, actor=actor)
            summaries = []
            await asyncio.wait_for(handle.result(), 60)
            while True:
                notification = await asyncio.wait_for(handle.next_notification(), 60)
                if notification.get("type") in (
                    "storage_review_complete",
                    "storage_review_skipped",
                ):
                    summaries.append(notification.get("message"))
                    break
            await asyncio.wait_for(handle._lifecycle_task, 60)
    finally:
        if own_actor:
            await actor.close()
    return (summaries[0] if summaries else None), {}, provider.requests


SCENARIOS = {
    "gate": scenario_gate,
    "threshold": scenario_threshold,
    "interrupt": scenario_interrupt,
    "persist": scenario_persist,
    "compress": scenario_compress,
    "review": scenario_review,
}

# Scenarios whose requests all belong to one conversation.
ONE_SESSION = ("gate", "threshold", "interrupt", "persist", "compress")


async def record_all() -> dict:
    out = {}
    for name, scenario in SCENARIOS.items():
        _result, _counter, requests = await scenario()
        out[name] = [request_bytes(r) for r in requests]
    return out


if __name__ == "__main__":  # pragma: no cover - maintenance entry point
    import sys

    if "--record" in sys.argv:
        GOLDEN.write_text(json.dumps(asyncio.run(record_all()), indent=1) + "\n")
        print(f"wrote {GOLDEN}")
