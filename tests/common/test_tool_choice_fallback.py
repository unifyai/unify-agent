"""Symbolic: ``UNIFY_TOOL_CHOICE_FALLBACK`` retries a refused forced tool choice as ``auto``.

Some models refuse every forced tool choice while they reason, with an HTTP 400
("tool_choice: type "tool" and "any" are not supported for this model"), which
ends a Unify session on its first, forced turn. The clients here are real
``unillm`` clients whose transport (``litellm.acompletion`` behind unillm's
retry wrapper) is replaced by a script: every request is recorded, nothing
leaves the process, and the refusal is the one OpenRouter returned on 29
September 2026 for ``anthropic/claude-opus-5.5`` with reasoning.
"""

from __future__ import annotations

import asyncio
import copy
import json

import litellm
import pytest
import unillm
from openai.types.chat import ChatCompletion

from unify.common import tool_choice_fallback as tcf
from unify.common._async_tool.messages import generate_with_preprocess
from unify.common.llm_client import new_llm_client
from unify.settings import SETTINGS

MODEL = "anthropic/claude-opus-5.5@openrouter"

# The provider's message, as OpenRouter relayed it (abridged to two hosts).
REFUSAL = (
    'OpenrouterException - {"error":{"message":"Provider returned error","code":400,'
    '"metadata":{"raw":"{\\"type\\":\\"error\\",\\"error\\":{\\"type\\":'
    '\\"invalid_request_error\\",\\"message\\":\\"tool_choice: type \\\\\\"tool\\\\\\" '
    'and \\\\\\"any\\\\\\" are not supported for this model.\\"}}",'
    '"provider_name":"Azure","previous_errors":[{"code":400,"provider_name":"Anthropic",'
    '"raw":"{\\"message\\":\\"tool_choice: type \\\\\\"tool\\\\\\" and \\\\\\"any\\\\\\" '
    'are not supported for this model.\\"}"}]}}'
)

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "FunctionManager_search_functions",
            "description": "search functions",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "GuidanceManager_search",
            "description": "search guidance",
            "parameters": {"type": "object", "properties": {"k": {"type": "integer"}}},
        },
    },
]


def refusal() -> litellm.BadRequestError:
    return litellm.BadRequestError(
        message=REFUSAL,
        model="openrouter/anthropic/claude-opus-5.5",
        llm_provider="openrouter",
    )


def completion(
    content: str | None = None,
    calls: list[tuple[str, dict]] = (),
) -> ChatCompletion:
    tool_calls = [
        {
            "id": f"call_{i}",
            "type": "function",
            "function": {"name": name, "arguments": json.dumps(args)},
        }
        for i, (name, args) in enumerate(calls)
    ]
    return ChatCompletion.model_validate(
        {
            "id": "cmpl-test",
            "object": "chat.completion",
            "created": 0,
            "model": "anthropic/claude-opus-5.5",
            "choices": [
                {
                    "index": 0,
                    "finish_reason": "tool_calls" if tool_calls else "stop",
                    "message": {
                        "role": "assistant",
                        "content": content,
                        "tool_calls": tool_calls or None,
                    },
                },
            ],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        },
    )


class Provider:
    """Scripted transport: refuses forced tool choices, then plays replies in order."""

    def __init__(self, replies=(), *, refuse_forced: bool = True, error=None):
        self.replies = list(replies)
        self.refuse_forced = refuse_forced
        self.error = error
        self.requests: list[dict] = []

    async def __call__(self, *, shared_session=None, client=None, **kw):
        self.requests.append(
            copy.deepcopy({k: kw.get(k) for k in ("messages", "tool_choice", "tools")}),
        )
        if self.error is not None:
            raise self.error
        if self.refuse_forced and tcf.is_forced_tool_choice(kw.get("tool_choice")):
            raise refusal()
        return self.replies.pop(0)


@pytest.fixture
def provider(monkeypatch):
    def install(*args, **kwargs) -> Provider:
        p = Provider(*args, **kwargs)
        monkeypatch.setattr(
            "unillm.clients.uni_llm._acompletion_with_transient_retry",
            p,
        )
        return p

    return install


@pytest.fixture(autouse=True)
def _fresh(monkeypatch):
    monkeypatch.setenv("UNILLM_CACHE", "false")
    tcf.reset_for_tests()
    yield
    tcf.reset_for_tests()


def client(monkeypatch, *, on: bool):
    monkeypatch.setattr(SETTINGS, "UNIFY_TOOL_CHOICE_FALLBACK", on)
    c = new_llm_client(MODEL, stateful=True, cache=False, reasoning_effort="low")
    c.set_system_message("You are a test.")
    return c


# ── the error test ──────────────────────────────────────────────────────────


def test_only_the_unsupported_tool_choice_400_is_recognised():
    assert tcf.is_unsupported_tool_choice_error(refusal())
    # As the tool loop re-raises it.
    try:
        try:
            raise refusal()
        except Exception as inner:
            raise Exception(
                f"LLM call failed: {type(inner).__name__}: {inner}",
            ) from inner
    except Exception as wrapped:
        assert tcf.is_unsupported_tool_choice_error(wrapped)
    assert not tcf.is_unsupported_tool_choice_error(
        litellm.BadRequestError(
            message="prompt is too long",
            model="m",
            llm_provider="openrouter",
        ),
    )
    assert not tcf.is_unsupported_tool_choice_error(
        litellm.InternalServerError(
            message="tool_choice: type any is not supported",
            model="m",
            llm_provider="openrouter",
        ),
    )
    assert not tcf.is_unsupported_tool_choice_error(RuntimeError("timeout"))
    assert tcf.is_forced_tool_choice("required")
    assert tcf.is_forced_tool_choice({"type": "function", "function": {"name": "x"}})
    assert not tcf.is_forced_tool_choice("auto")
    assert not tcf.is_forced_tool_choice(None)


# ── off: exactly as shipped ─────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_off_the_client_is_untouched_and_the_error_propagates(
    monkeypatch,
    provider,
):
    p = provider()
    c = client(monkeypatch, on=False)
    assert not getattr(c, tcf.TOOL_CHOICE_FALLBACK_MARK, False)
    assert c.generate.__func__ is unillm.AsyncUnify.generate
    c._messages = [{"role": "user", "content": "task"}]
    with pytest.raises(litellm.BadRequestError) as info:
        await c.generate(
            tools=TOOLS,
            tool_choice="required",
            return_full_completion=True,
        )
    assert "are not supported for this model" in str(info.value)
    assert len(p.requests) == 1
    assert tcf._AUTO_ONLY_ENDPOINTS == set()


# ── on ─────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_on_a_refused_forced_call_is_retried_once_as_auto(monkeypatch, provider):
    reply = completion(calls=[("FunctionManager_search_functions", {"query": "q"})])
    p = provider([reply])
    c = client(monkeypatch, on=True)
    c._messages = [{"role": "user", "content": "task"}]
    result = await c.generate(
        tools=TOOLS,
        tool_choice="required",
        return_full_completion=True,
    )

    assert (
        result.choices[0].message.tool_calls[0].function.name
        == "FunctionManager_search_functions"
    )
    assert [r["tool_choice"] for r in p.requests] == ["required", "auto"]
    retry = p.requests[1]["messages"]
    assert retry[-1]["role"] == "user"
    assert "This turn requires a tool call" in retry[-1]["content"]
    assert "`FunctionManager_search_functions`" in retry[-1]["content"]
    assert "`GuidanceManager_search`" in retry[-1]["content"]
    # The instruction is request-only: the transcript keeps the task and the reply.
    assert [m["role"] for m in c.messages] == ["system", "user", "assistant"]
    assert not any("This turn requires" in str(m.get("content")) for m in c.messages)
    assert tcf._AUTO_ONLY_ENDPOINTS == {MODEL}


@pytest.mark.asyncio
async def test_on_the_model_is_remembered_so_later_forced_calls_go_straight_to_auto(
    monkeypatch,
    provider,
):
    p = provider(
        [
            completion(calls=[("GuidanceManager_search", {"k": 5})]),
            completion(calls=[("GuidanceManager_search", {"k": 3})]),
        ],
    )
    first = client(monkeypatch, on=True)
    first._messages = [{"role": "user", "content": "task"}]
    await first.generate(
        tools=TOOLS,
        tool_choice="required",
        return_full_completion=True,
    )
    assert len(p.requests) == 2

    # A new client for the same model in the same process: no refused attempt.
    second = client(monkeypatch, on=True)
    second._messages = [{"role": "user", "content": "task two"}]
    await second.generate(
        tools=TOOLS,
        tool_choice="required",
        return_full_completion=True,
    )
    assert [r["tool_choice"] for r in p.requests] == ["required", "auto", "auto"]
    assert "This turn requires a tool call" in p.requests[2]["messages"][-1]["content"]


@pytest.mark.asyncio
async def test_on_other_errors_and_unforced_calls_propagate_unchanged(
    monkeypatch,
    provider,
):
    other = litellm.BadRequestError(
        message="prompt is too long",
        model="m",
        llm_provider="openrouter",
    )
    p = provider(error=other)
    c = client(monkeypatch, on=True)
    c._messages = [{"role": "user", "content": "task"}]
    with pytest.raises(litellm.BadRequestError) as info:
        await c.generate(
            tools=TOOLS,
            tool_choice="required",
            return_full_completion=True,
        )
    assert info.value is other
    assert len(p.requests) == 1

    # The refusal itself on a call that forced nothing is not this module's to handle.
    p2 = provider(error=refusal())
    with pytest.raises(litellm.BadRequestError):
        await c.generate(tools=TOOLS, tool_choice="auto", return_full_completion=True)
    assert len(p2.requests) == 1
    assert tcf._AUTO_ONLY_ENDPOINTS == set()


@pytest.mark.asyncio
async def test_on_a_reply_without_the_required_call_is_reprompted_once(
    monkeypatch,
    provider,
):
    p = provider(
        [
            completion(content="I will look this up."),
            completion(calls=[("FunctionManager_search_functions", {"query": "q"})]),
        ],
    )
    c = client(monkeypatch, on=True)
    c._messages = [{"role": "user", "content": "task"}]
    result = await c.generate(
        tools=TOOLS,
        tool_choice="required",
        return_full_completion=True,
    )

    assert result.choices[0].message.tool_calls
    assert [r["tool_choice"] for r in p.requests] == ["required", "auto", "auto"]
    nudge = p.requests[2]["messages"][-1]["content"]
    assert "made no required tool call" in nudge
    assert "I will look this up." in nudge
    # The rejected text reply was dropped; only the accepted one remains.
    assert [m["role"] for m in c.messages] == ["system", "user", "assistant"]
    assert c.messages[-1]["tool_calls"]


@pytest.mark.asyncio
async def test_on_a_second_text_reply_is_returned_as_it_is(monkeypatch, provider):
    p = provider([completion(content="no"), completion(content="still no")])
    c = client(monkeypatch, on=True)
    c._messages = [{"role": "user", "content": "task"}]
    result = await c.generate(
        tools=TOOLS,
        tool_choice="required",
        return_full_completion=True,
    )
    assert result.choices[0].message.content == "still no"
    assert len(p.requests) == 3
    assert [m["role"] for m in c.messages] == ["system", "user", "assistant"]


@pytest.mark.asyncio
async def test_on_a_named_tool_choice_asks_for_that_tool(monkeypatch, provider):
    choice = {"type": "function", "function": {"name": "GuidanceManager_search"}}
    p = provider(
        [
            completion(calls=[("FunctionManager_search_functions", {"query": "q"})]),
            completion(calls=[("GuidanceManager_search", {"k": 5})]),
        ],
    )
    c = client(monkeypatch, on=True)
    c._messages = [{"role": "user", "content": "task"}]
    result = await c.generate(
        tools=TOOLS,
        tool_choice=choice,
        return_full_completion=True,
    )
    assert (
        result.choices[0].message.tool_calls[0].function.name
        == "GuidanceManager_search"
    )
    assert (
        "call to the tool `GuidanceManager_search`"
        in p.requests[1]["messages"][-1]["content"]
    )
    assert len(p.requests) == 3  # the wrong tool counted as no required call


@pytest.mark.asyncio
async def test_on_through_the_tool_loop_dispatch_the_transcript_stays_clean(
    monkeypatch,
    provider,
):
    """``generate_with_preprocess`` (both tool-loop dispatch sites) copies back only the reply."""
    p = provider([completion(calls=[("GuidanceManager_search", {"k": 5})])])
    c = client(monkeypatch, on=True)
    c._messages = [{"role": "user", "content": "task"}]
    await generate_with_preprocess(
        c,
        lambda msgs: msgs,
        return_full_completion=True,
        tools=TOOLS,
        tool_choice="required",
        stateful=True,
    )
    # The canonical log holds no system prompt on this path; the reply is appended.
    assert [m["role"] for m in c.messages] == ["user", "assistant"]
    assert c.messages[-1]["tool_calls"]
    assert "This turn requires a tool call" in p.requests[1]["messages"][-1]["content"]
    assert len(p.requests) == 2


def test_the_discovery_mutator_completes_a_fallback_turn():
    """The discovery-first mutator adds the missing family on a forced turn sent as auto."""
    from unillm.clients.completion_mutator import CompletionMutatorContext

    from unify.actor.code_act_actor import _build_discovery_parallel_mutator

    mutator = _build_discovery_parallel_mutator()
    ctx = CompletionMutatorContext(
        provider="openrouter",
        original_tool_choice="auto",
        request_kw={"tools": TOOLS},
    )
    partial = completion(calls=[("FunctionManager_search_functions", {"query": "q"})])
    untouched = mutator(copy.deepcopy(partial), ctx)
    assert len(untouched.choices[0].message.tool_calls) == 1

    token = tcf._FORCED_TOOL_CHOICE.set("required")
    try:
        filled = mutator(copy.deepcopy(partial), ctx)
    finally:
        tcf._FORCED_TOOL_CHOICE.reset(token)
    names = [
        (tc["function"]["name"] if isinstance(tc, dict) else tc.function.name)
        for tc in filled.choices[0].message.tool_calls
    ]
    assert names == ["FunctionManager_search_functions", "GuidanceManager_search"]


# ── end to end: the discovery-first shape of the failed TravelPlanner cells ──


async def _loop(monkeypatch, provider, *, on: bool):
    from unify.common.async_tool_loop import start_async_tool_loop

    ran: list[str] = []

    async def search_things(query: str) -> str:
        """Search things."""
        ran.append(query)
        return "found: nothing"

    async def act(code: str) -> str:
        """Run code."""
        ran.append(code)
        return "ok"

    def policy(step, tools, called=()):
        if "search_things" not in called:
            return "required", {"search_things": tools["search_things"]}
        return "auto", tools

    p = provider(
        [
            completion(calls=[("search_things", {"query": "relevant"})]),
            completion(content="done"),
        ],
    )
    c = client(monkeypatch, on=on)
    handle = start_async_tool_loop(
        client=c,
        message="Do the task.",
        tools={"search_things": search_things, "act": act},
        tool_policy=policy,
        max_steps=40,
        timeout=30,
        log_steps=False,
    )
    return handle, p, ran, c


@pytest.mark.asyncio
@pytest.mark.timeout(60)
async def test_loop_off_the_forced_first_turn_ends_the_run(monkeypatch, provider):
    handle, p, ran, _ = await _loop(monkeypatch, provider, on=False)
    with pytest.raises(Exception) as info:
        await asyncio.wait_for(handle.result(), 30)
    assert "LLM call failed: BadRequestError" in str(info.value)
    assert "are not supported for this model" in str(info.value)
    assert ran == []
    assert [r["tool_choice"] for r in p.requests] == ["required"]


@pytest.mark.asyncio
@pytest.mark.timeout(60)
async def test_loop_on_the_forced_first_search_still_happens_and_the_run_finishes(
    monkeypatch,
    provider,
):
    handle, p, ran, c = await _loop(monkeypatch, provider, on=True)
    result = await asyncio.wait_for(handle.result(), 30)
    assert result == "done"
    assert ran == ["relevant"]
    assert [r["tool_choice"] for r in p.requests] == ["required", "auto", "auto"]
    assert "`search_things`" in p.requests[1]["messages"][-1]["content"]
    # Later unforced turns carry no instruction.
    assert "This turn requires" not in str(p.requests[2]["messages"][-1].get("content"))
    assert not any("This turn requires" in str(m.get("content")) for m in c.messages)


def test_the_switch_is_off_by_default_and_parses_as_a_boolean():
    from unify.settings import ProductionSettings

    assert ProductionSettings().UNIFY_TOOL_CHOICE_FALLBACK is False
    assert (
        ProductionSettings(UNIFY_TOOL_CHOICE_FALLBACK="1").UNIFY_TOOL_CHOICE_FALLBACK
        is True
    )
    assert (
        ProductionSettings(
            UNIFY_TOOL_CHOICE_FALLBACK="false",
        ).UNIFY_TOOL_CHOICE_FALLBACK
        is False
    )
