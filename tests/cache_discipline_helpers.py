"""A scripted LLM transport and the request scenarios the cache-discipline tests share.

The clients are real ``unillm`` clients; only their transport
(``litellm.acompletion`` behind unillm's retry wrapper) is replaced, so every
request is recorded exactly as unillm would send it and nothing leaves the
process. The actor's first request under the default settings is pinned by
``tests/actor/code_act/test_baked_prompt_golden.py``.
"""

from __future__ import annotations

import asyncio
import contextlib
import copy
import itertools
import json
import os
import sys
from typing import Any, Callable, Iterator, Optional

from openai.types.chat import ChatCompletion

MODEL = "openai/gpt-5.6-sol@openrouter"
_CALL_SEQ = itertools.count()
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


_ACTIVE_PROVIDER: list[Optional[Provider]] = [None]


@contextlib.contextmanager
def scripted(replies) -> Iterator[Provider]:
    """Install a :class:`Provider` as unillm's transport for the block."""
    import unillm.clients.uni_llm as uni_llm
    from unillm.settings import SETTINGS as unillm_settings

    global _CALL_SEQ
    _CALL_SEQ = itertools.count()
    provider = Provider(replies)
    _ACTIVE_PROVIDER[0] = provider
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
        _ACTIVE_PROVIDER[0] = None
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


def _affinity_owner():
    """The class in unillm's async client that defines the affinity API, if any."""
    import unillm

    for cls in unillm.AsyncUnify.__mro__:
        if "set_cache_affinity" in cls.__dict__:
            return cls
    return None


def _called_from_unillm(depth: int = 2) -> bool:
    caller = sys._getframe(depth).f_globals.get("__name__", "")
    return caller == "unillm" or caller.startswith("unillm.")


def install_affinity_api(monkeypatch) -> list[tuple[str, int]]:
    """Record the ``cache_affinity`` keys unify sets on unillm's async client.

    The ``harness-cache`` branch of unillm has the API (``set_cache_affinity``
    and a ``cache_affinity`` property, stored as ``_cache_affinity`` and sent
    with every request); ``main`` does not, and is given the same API here.
    Where unillm has it, its own setter still runs, so the key reaches the
    request exactly as unillm sends it.

    Returns a list that each ``set_cache_affinity`` call made from outside
    unillm appends its key to, with the number of requests the scripted
    provider had then received (``-1`` outside :func:`scripted`), so a test
    can tell a key set before the first request from one set later. unillm's
    own calls (its constructor resets the key, ``copy()`` passes it on) are
    not unify setting a key and are not recorded.
    """
    import unillm

    sets: list[tuple[str, int]] = []
    owner = _affinity_owner()
    original = owner.__dict__["set_cache_affinity"] if owner is not None else None

    def set_cache_affinity(self, value):
        if not _called_from_unillm():
            provider = _ACTIVE_PROVIDER[0]
            sets.append((value, len(provider.requests) if provider else -1))
        if original is not None:
            return original(self, value)
        self._cache_affinity = value
        return self

    if owner is not None:
        monkeypatch.setattr(owner, "set_cache_affinity", set_cache_affinity)
        return sets
    monkeypatch.setattr(
        unillm.AsyncUnify,
        "set_cache_affinity",
        set_cache_affinity,
        raising=False,
    )
    monkeypatch.setattr(
        unillm.AsyncUnify,
        "cache_affinity",
        property(lambda self: getattr(self, "_cache_affinity", None)),
        raising=False,
    )
    return sets


_AFFINITY_API = ("set_cache_affinity", "cache_affinity")


def hide_affinity_api(monkeypatch) -> None:
    """Make unillm's async client look like ``main``'s: no ``cache_affinity`` API.

    Outside unillm the two names raise ``AttributeError``, as on a unillm
    without them; unillm's own code (its constructor, ``copy()``) still
    reaches them, so the client works and never holds a key.
    """
    import unillm

    if _affinity_owner() is None:
        return
    lookup = unillm.AsyncUnify.__getattribute__

    def __getattribute__(self, name):
        if name in _AFFINITY_API and not _called_from_unillm():
            raise AttributeError(name)
        return lookup(self, name)

    monkeypatch.setattr(unillm.AsyncUnify, "__getattribute__", __getattribute__)


# ── tools ────────────────────────────────────────────────────────────────


def make_tools(counter: dict) -> dict[str, Callable]:
    """A code runner; each call is counted."""

    async def execute_code(code: str) -> str:
        """Run Python code in the sandbox.

        Args:
            code: The code to run.
        """
        counter["execute_code"] = counter.get("execute_code", 0) + 1
        return f"ran {code}"

    return {"execute_code": execute_code}


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


INTERRUPT_REPLIES = (
    lambda: completion(calls=[("execute_code", {"code": "one"})]),
    lambda: completion(content="done"),
)


async def scenario_interrupt() -> tuple[str, dict, list[dict]]:
    """One code call, then the answer."""
    counter: dict = {}
    tools = make_tools(counter)
    with scripted(INTERRUPT_REPLIES) as provider:
        result = await _run(new_client(), tools, "Do the task.")
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
    """A persistent session: a turn, the session parks, then a second request and its turn."""
    from unify.common.async_tool_loop import start_async_tool_loop

    counter: dict = {}
    tools = make_tools(counter)
    with scripted(PERSIST_REPLIES) as provider:
        handle = start_async_tool_loop(
            new_client(),
            "First request.",
            tools,
            log_steps=False,
            timeout=60,
            persist=True,
        )
        first = await _next_response(handle)
        await handle.submit("Second request.")
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
    # the summary (a fork of the last request), or the fallback compressor's reply
    lambda: completion(content=SUMMARY),
    lambda: completion(content="done"),
)


async def scenario_compress(replies=COMPRESS_REPLIES) -> tuple[str, dict, list[dict]]:
    """A context-full turn of the actor's kind of loop, which compacts, then the answer after the restart."""
    counter: dict = {}
    tools = make_tools(counter)
    with scripted(replies) as provider:
        result = await _run(new_client(), tools, "Do the task.", tools_in_code=True)
    return result, counter, provider.requests


ACTOR_REPLIES = (
    lambda: completion(calls=[("execute_code", {"code": "import os\nos.listdir('.')"})]),
    lambda: completion(content="done"),
)


async def scenario_actor(replies=ACTOR_REPLIES):
    """``CodeActActor.act`` on a fresh actor, as a caller runs it."""
    from unify.actor.code_act_actor import CodeActActor

    actor = CodeActActor()
    with scripted(replies) as provider:
        handle = await actor.act("List the files in the workspace.", persist=False)
        result = await asyncio.wait_for(handle.result(), 60)
    return result, {}, provider.requests


def session_requests(requests: list[dict]) -> list[dict]:
    """The requests of the conversation that sent the first one."""
    system = requests[0]["messages"][0]
    return [r for r in requests if r["messages"][:1] == [system]]


SCENARIOS = {
    "interrupt": scenario_interrupt,
    "persist": scenario_persist,
    "compress": scenario_compress,
}

# Scenarios whose requests all belong to one conversation.
ONE_SESSION = ("interrupt", "persist", "compress")
