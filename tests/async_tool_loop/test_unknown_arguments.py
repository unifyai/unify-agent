"""A tool call carrying an argument its tool has no parameter for.

Dropped, such an argument would leave the tool to run on its defaults and
return a plausible result for a request the model never made, with nothing to
tell the model its argument was ignored. The call fails instead: the tool does
not run, and the tool message the model reads names the arguments that were
wrong and the parameters the tool takes. The tolerance the loop applies on
purpose still holds: nested ``kwargs`` expansion, empty ``a``/``kw`` noise
keys, single-parameter aliases, string coercion and the context-control keys
the loop pops itself. Steering the loop forwards to a handle still drops what
the handle's method does not take.

Most tests drive the real dispatch and result path (``schedule_base_tool_call``,
the task, ``process_completed_task``) without a model and read the tool message
the next request would carry. The transcript container, message dispatcher and
logger are the only doubles. The last test runs the whole loop on a live model.
"""

from __future__ import annotations

import asyncio
import json

import pytest

from tests.async_helpers import any_tool_message_content_contains
from tests.helpers import _handle_project
from unify.common._async_tool.context_tracker import LoopContextState
from unify.common._async_tool.loop import ToolLoopRuntimeState, _LoopToolFailureTracker
from unify.common._async_tool.messages import forward_handle_call
from unify.common._async_tool.tools_data import ToolsData
from unify.common.async_tool_loop import ChatContextPropagation, start_async_tool_loop
from unify.common.llm_client import new_llm_client


class _Client:
    def __init__(self):
        self.messages: list[dict] = []

    def append_messages(self, msgs):
        self.messages += msgs


class _Dispatcher:
    def __init__(self, client):
        self._client = client

    async def append_msgs(self, msgs, origin=None, *, skip_event_bus=False, kind=None):
        self._client.append_messages(msgs)

    async def publish_to_event_bus(self, msgs, origin=None, kind=None):
        pass


class _Logger:
    log_steps = False

    def debug(self, *a, **k): ...
    def info(self, *a, **k): ...
    def error(self, *a, **k): ...


class _Loop:
    """One loop's tool state, driven one model-written call at a time."""

    def __init__(self, tools: dict):
        self.client = _Client()
        self._dispatcher = _Dispatcher(self.client)
        self._tools_data = ToolsData(tools, client=self.client, logger=_Logger())
        self.tracker = _LoopToolFailureTracker(3, ToolLoopRuntimeState())
        self._calls = 0

    async def call(self, name: str, args: dict) -> str:
        """Dispatch a call with *args* as the model wrote them, and return the
        tool message the model reads back for it."""
        self._calls += 1
        call_id = f"call_{self._calls}"
        args_json = json.dumps(args)
        asst_msg = {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": call_id,
                    "type": "function",
                    "function": {"name": name, "arguments": args_json},
                },
            ],
        }
        self.client.messages.append(asst_msg)
        assistant_meta: dict = {}
        await self._tools_data.schedule_base_tool_call(
            asst_msg,
            name=name,
            args_json=args_json,
            call_id=call_id,
            call_idx=0,
            context_state=LoopContextState(),
            propagate_chat_context=ChatContextPropagation.NEVER,
            assistant_meta=assistant_meta,
            msg_dispatcher=self._dispatcher,
        )
        (task,) = self._tools_data.pending
        await asyncio.wait([task])
        await self._tools_data.process_completed_task(
            task,
            self.tracker,
            [None],
            assistant_meta,
            self._dispatcher,
        )
        (reply,) = [
            m
            for m in self.client.messages
            if m.get("role") == "tool" and m.get("tool_call_id") == call_id
        ]
        return reply["content"]


def _search_tool(received: list, *, is_async: bool = False):
    """A guidance-search-shaped tool that records every call it runs."""

    def search(references: dict | None = None, k: int = 10) -> str:
        received.append({"references": references, "k": k})
        return "ok"

    async def async_search(references: dict | None = None, k: int = 10) -> str:
        return search(references, k)

    return async_search if is_async else search


# ── a name the tool does not take fails the call ──────────────────────────


@pytest.mark.asyncio
@pytest.mark.parametrize("is_async", [False, True], ids=["sync", "async"])
async def test_a_misnamed_argument_fails_the_call_naming_the_real_parameters(
    is_async,
):
    """Guidance search called with the function search's parameter names, as
    the actor's discovery mutator once did, reads back a refusal, not rows."""
    received: list = []
    loop = _Loop({"search": _search_tool(received, is_async=is_async)})

    reply = await loop.call("search", {"query": "relevant guidance", "n": 5})

    assert reply == (
        "search was not run: it has no parameter named 'query' or 'n'.\n"
        "Suggestion: Call search again using only its own parameters: references, k."
    )
    assert received == []


@pytest.mark.asyncio
async def test_one_misnamed_argument_among_valid_ones_still_fails_the_call():
    """Running on the valid argument alone would still ignore the misnamed one."""
    received: list = []
    loop = _Loop({"search": _search_tool(received)})

    reply = await loop.call("search", {"references": {"content": "deploys"}, "n": 5})

    assert reply.startswith("search was not run: it has no parameter named 'n'.\n")
    assert received == []


@pytest.mark.asyncio
async def test_a_tool_without_parameters_says_to_call_it_with_none():
    ran: list = []

    def ping() -> str:
        ran.append(True)
        return "pong"

    loop = _Loop({"ping": ping})

    reply = await loop.call("ping", {"verbose": True})

    assert reply == (
        "ping was not run: it has no parameter named 'verbose'.\n"
        "Suggestion: Call ping again with no arguments."
    )
    assert ran == []


# ── it counts as a refusal, so only repetition ends the loop ──────────────


@pytest.mark.asyncio
async def test_a_misnamed_call_is_a_refusal_not_a_failure():
    """Correcting an argument name is converging on the argspec, so it must not
    spend the budget kept for unexpected exceptions."""
    loop = _Loop({"search": _search_tool([])})

    for attempt in range(loop.tracker.max_failures + 2):
        reply = await loop.call("search", {"query": f"attempt {attempt}"})
        assert reply.startswith("search was not run:"), reply

    assert loop.tracker.current_failures == 0
    assert loop.tracker.stop_reason() is None


@pytest.mark.asyncio
async def test_resending_the_same_misnamed_call_stops_the_loop():
    loop = _Loop({"search": _search_tool([])})
    args = {"query": "relevant guidance", "n": 5}

    for _ in range(_LoopToolFailureTracker.IDENTICAL_CALL_LIMIT - 1):
        await loop.call("search", args)
    with pytest.raises(RuntimeError, match="same arguments"):
        await loop.call("search", args)


@pytest.mark.asyncio
async def test_misnaming_the_same_argument_while_varying_its_value_stops_the_loop():
    """Calls that differ only in a misnamed argument's value are different
    calls, but the complaint is the same, so repeating it still ends the loop."""
    loop = _Loop({"search": _search_tool([])})

    for attempt in range(_LoopToolFailureTracker.SAME_COMPLAINT_LIMIT - 1):
        await loop.call("search", {"query": f"attempt {attempt}"})
    assert loop.tracker.stop_reason() is None

    with pytest.raises(RuntimeError, match="same reason"):
        await loop.call("search", {"query": "one more"})


# ── the tolerance that exists on purpose is kept ──────────────────────────


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "args, expected",
    [
        pytest.param(
            {"kwargs": {"references": {"content": "deploys"}, "k": 3}},
            {"references": {"content": "deploys"}, "k": 3},
            id="nested-kwargs",
        ),
        pytest.param(
            {"references": {"content": "deploys"}, "a": "", "kw": None},
            {"references": {"content": "deploys"}, "k": 10},
            id="empty-noise-keys",
        ),
        pytest.param(
            {"references": {"content": "deploys"}, "k": "3"},
            {"references": {"content": "deploys"}, "k": 3},
            id="string-coercion",
        ),
        pytest.param(
            {"references": {"content": "deploys"}, "include_parent_chat_context": True},
            {"references": {"content": "deploys"}, "k": 10},
            id="loop-popped-context-key",
        ),
    ],
)
async def test_tolerated_argument_shapes_still_run_the_tool(args, expected):
    received: list = []
    loop = _Loop({"search": _search_tool(received)})

    reply = await loop.call("search", args)

    assert reply == "ok"
    assert received == [expected]


@pytest.mark.asyncio
async def test_a_single_parameter_tool_still_takes_a_common_alias():
    received: list = []

    def ask(question: str) -> str:
        received.append(question)
        return "ok"

    loop = _Loop({"ask": ask})

    reply = await loop.call("ask", {"query": "why did the deploy fail?"})

    assert reply == "ok"
    assert received == ["why did the deploy fail?"]


@pytest.mark.asyncio
async def test_a_tool_taking_arbitrary_keywords_receives_them():
    received: list = []

    def tag(**labels) -> str:
        # **kwargs also takes the loop's own underscored plumbing.
        received.append({k: v for k, v in labels.items() if not k.startswith("_")})
        return "ok"

    loop = _Loop({"tag": tag})

    reply = await loop.call("tag", {"colour": "red", "size": "large"})

    assert reply == "ok"
    assert received == [{"colour": "red", "size": "large"}]


# ── steering forwarded to a handle keeps dropping what it does not take ───


class _Handle:
    def __init__(self):
        self.calls: list = []

    async def stop(self):
        self.calls.append(("stop",))

    async def interject(self, message: str):
        self.calls.append(("interject", message))


@pytest.mark.asyncio
async def test_forwarded_steering_drops_what_the_handles_method_does_not_take():
    """The loop writes these kwargs to the base steering contract, not a model,
    so a handle whose methods take fewer must still be stopped and steered."""
    handle = _Handle()

    await forward_handle_call(
        handle,
        "stop",
        {"reason": "the user cancelled"},
        fallback_positional_keys=["reason"],
    )
    await forward_handle_call(
        handle,
        "interject",
        {
            "content": "use the staging database",
            "_parent_chat_context_cont": [{"role": "user", "content": "hi"}],
        },
        fallback_positional_keys=["content", "message"],
    )

    assert handle.calls == [("stop",), ("interject", "use the staging database")]


# ── through the whole loop, the model reads the refusal and recovers ──────


@pytest.mark.llm_call
@pytest.mark.eval
@pytest.mark.asyncio
@_handle_project
async def test_the_model_reissues_a_refused_call_with_the_parameters_named(
    llm_config,
):
    """The model's first search is rewritten to the function search's names,
    the arguments the actor's discovery mutator once appended. The model must
    read the refusal and search again with the names it gives."""
    received: list = []

    def search_guidance(references: dict | None = None, k: int = 10) -> list[str]:
        """Search stored procedures by meaning.

        Parameters
        ----------
        references : dict | None
            Mapping of field name (``title`` or ``content``) to the text that
            field is compared with by meaning.
        k : int
            Maximum number of results to return.
        """
        received.append({"references": references, "k": k})
        return ["Deploying the web app: build it, run the migrations, restart it."]

    def misname_first_turn_searches(completion, context):
        # Only the first turn: once a tool message is in the request, the
        # model's own calls go through untouched.
        messages = context.request_kw.get("messages") or []
        if any(m.get("role") == "tool" for m in messages):
            return completion
        misnamed = json.dumps({"query": "relevant guidance", "n": 5})
        for call in completion.choices[0].message.tool_calls or []:
            if isinstance(call, dict):
                if call["function"]["name"] == "search_guidance":
                    call["function"]["arguments"] = misnamed
            elif call.function.name == "search_guidance":
                call.function.arguments = misnamed
        return completion

    client = new_llm_client(**llm_config)
    generate = client.generate

    def generate_with_misnaming(*args, **kwargs):
        kwargs.setdefault("completion_mutator", misname_first_turn_searches)
        return generate(*args, **kwargs)

    client.generate = generate_with_misnaming

    answer = await start_async_tool_loop(
        client,
        message=(
            "Use search_guidance to find the procedure for deploying the web "
            "app, then list its steps."
        ),
        tools={"search_guidance": search_guidance},
        max_consecutive_failures=2,
    ).result()

    assert any_tool_message_content_contains(
        client.messages,
        "search_guidance was not run: it has no parameter named 'query' or 'n'.\n"
        "Suggestion: Call search_guidance again using only its own parameters: "
        "references, k.",
    )
    assert received, "the model never reissued the search"
    assert all(call["references"] for call in received), received
    assert "migration" in answer.lower()
