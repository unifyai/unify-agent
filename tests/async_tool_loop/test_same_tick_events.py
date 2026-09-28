"""Events that land in the same tick of the tool loop's waits.

The loop waits on tool completions, interjections, clarifications and
notifications in two places: section A while no LLM step is in flight, and
section D, which races them against the in-flight step. Several can land in
one wake-up, and a cached replay, where a hit resolves within milliseconds,
makes such ties common.

When the step has already answered, the stateful client has inserted the reply
into the transcript, so the reply must be processed: the other events are
handled as if they arrived just after it, never in its place. When an event
supersedes the step instead, or section A observes the tick, nothing that
landed in it is dropped, and the next request is the same whichever of the two
sections observed it.

Each step is scripted in place of the provider behind ``client.generate``, and
each reply is inserted at the transcript index captured at dispatch, as the
stateful client does. An event meant to land during a step is raised from
inside it, so the loop's waiter has taken it before the loop wakes. No LLM
calls.
"""

from __future__ import annotations

import asyncio
import copy
import json

import pytest

from unify.common._async_tool.loop import _requeue_at_front
from unify.common.async_tool_loop import start_async_tool_loop
from unify.common.llm_client import new_llm_client
from tests.async_helpers import (
    _wait_for_condition,
    _wait_for_tool_request,
    make_gated_async_tool,
)


def _tool_call(call_id: str, name: str, arguments: dict) -> dict:
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": name, "arguments": json.dumps(arguments)},
    }


def _calls(*calls: dict) -> dict:
    return {"role": "assistant", "content": None, "tool_calls": list(calls)}


def _text(content: str) -> dict:
    return {"role": "assistant", "content": content, "tool_calls": None}


async def _unexpected_step() -> dict:
    return _text("unexpected extra step")


class _ScriptedSteps:
    """Stands in for the provider behind ``client.generate``.

    Step *n* awaits ``steps[n]()`` for its reply and inserts it where the
    stateful client would: at the transcript index captured when the step was
    dispatched. The prompt each step was sent is kept in ``prompts``.
    """

    def __init__(self, client, steps):
        self._client = client
        self._steps = steps
        self.prompts: list[list[dict]] = []

    def __call__(self, **_gen_kwargs):
        index = len(self._client.messages)
        self.prompts.append(copy.deepcopy(self._client.messages))
        n = len(self.prompts) - 1
        step = self._steps[n] if n < len(self._steps) else _unexpected_step
        return self._reply(index, step)

    async def _reply(self, index: int, step):
        self._client.messages.insert(index, await step())
        return None


def _index_of(prompt: list[dict], predicate) -> int:
    return next(i for i, m in enumerate(prompt) if predicate(m))


def _unanswered_calls(prompt: list[dict]) -> list[str]:
    answered = {m.get("tool_call_id") for m in prompt if m.get("role") == "tool"}
    return [
        call["id"]
        for m in prompt
        if m.get("role") == "assistant"
        for call in m.get("tool_calls") or []
        if call["id"] not in answered
    ]


async def _answer_as_a_tool_finishes(llm_config, **loop_kwargs):
    """Run a loop whose second step answers in text in the same wake-up as
    ``slow_tool`` finishes, then ends with "final answer" if asked again."""
    client = new_llm_client(**llm_config)
    client.set_system_message("This turn is fully scripted by the test.")

    gate, gated_tool = make_gated_async_tool("slow-done")
    running: dict[str, asyncio.Task] = {}

    async def slow_tool() -> str:
        running["slow"] = asyncio.current_task()
        return await gated_tool()

    async def call_slow_tool() -> dict:
        return _calls(_tool_call("call_slow", "slow_tool", {}))

    async def answer_as_slow_tool_finishes() -> dict:
        # The loop's wait registered on the tool's task before this step ran,
        # so it is woken first and resumes only after this reply is in.
        gate.set()
        await running["slow"]
        return _text("first answer")

    async def answer_again() -> dict:
        return _text("final answer")

    steps = _ScriptedSteps(
        client,
        [call_slow_tool, answer_as_slow_tool_finishes, answer_again],
    )
    client.generate = steps

    handle = start_async_tool_loop(
        client=client,
        message="start",
        tools={"slow_tool": slow_tool},
        max_steps=20,
        timeout=30,
        **loop_kwargs,
    )
    await _wait_for_tool_request(client, "slow_tool")
    # With slow_tool pending the loop waits; the interjection grants the step
    # that races it.
    await handle.interject("continue")

    result = await asyncio.wait_for(handle.result(), timeout=30)
    return result, steps, handle


@pytest.mark.asyncio
@pytest.mark.timeout(60)
async def test_text_reply_is_the_answer_when_a_tool_finishes_alongside_it(
    llm_config,
) -> None:
    """A final answer that lands with a tool result ends the loop with that
    answer, rather than being stranded while the loop asks again."""
    result, steps, handle = await _answer_as_a_tool_finishes(llm_config)

    assert result == "first answer"
    assert len(steps.prompts) == 2, "the answer must not be followed by another step"
    assert handle._runtime_state.step_index == 2


@pytest.mark.asyncio
@pytest.mark.timeout(60)
async def test_patient_mode_still_shows_the_model_a_result_landing_with_its_reply(
    llm_config,
) -> None:
    """Patient mode lets a step finish over a tool result and owes the model
    one more turn with that result, which holds when the two land together."""
    result, steps, _ = await _answer_as_a_tool_finishes(
        llm_config,
        interrupt_llm_on_tool_completion=False,
    )

    assert result == "final answer"
    assert len(steps.prompts) == 3
    last_prompt = steps.prompts[2]
    reply_at = _index_of(last_prompt, lambda m: m.get("content") == "first answer")
    result_at = _index_of(last_prompt, lambda m: "slow-done" in str(m.get("content")))
    assert reply_at < result_at


_SAME_TICK_EVIDENCE = {
    "tool_result": "slow-done",
    "interjection": "also check the logs",
    "notification": "halfway there",
    "clarification": "Which environment?",
}


@pytest.mark.asyncio
@pytest.mark.timeout(60)
@pytest.mark.parametrize("event", sorted(_SAME_TICK_EVIDENCE))
async def test_reply_runs_before_an_event_that_lands_alongside_it(
    llm_config,
    event: str,
) -> None:
    """A reply's tool calls run before the event that landed with it is
    handled, and the event still reaches the next step, after the reply."""
    client = new_llm_client(**llm_config)
    client.set_system_message("This turn is fully scripted by the test.")

    gate, gated_tool = make_gated_async_tool("slow-done")
    running: dict[str, asyncio.Task] = {}
    channels: dict[str, asyncio.Queue] = {}
    received: list = []

    async def slow_tool() -> str:
        running["slow"] = asyncio.current_task()
        return await gated_tool()

    async def listener(
        _interject_queue: asyncio.Queue | None = None,
        _notification_up_q: asyncio.Queue | None = None,
        _clarification_up_q: asyncio.Queue | None = None,
        _clarification_down_q: asyncio.Queue | None = None,
    ) -> str:
        channels["notification"] = _notification_up_q
        channels["clarification"] = _clarification_up_q
        received.append(await _interject_queue.get())
        return "listened"

    async def raise_event() -> None:
        if event == "tool_result":
            gate.set()
            await running["slow"]
        elif event == "interjection":
            await handle.interject(_SAME_TICK_EVIDENCE[event])
        elif event == "notification":
            channels[event].put_nowait({"message": _SAME_TICK_EVIDENCE[event]})
        else:
            channels[event].put_nowait(_SAME_TICK_EVIDENCE[event])

    async def start_both() -> dict:
        return _calls(
            _tool_call("call_slow", "slow_tool", {}),
            _tool_call("call_listener", "listener", {}),
        )

    async def steer_listener_as_event_lands() -> dict:
        # Raised from inside the step, the event is taken by the loop's
        # waiter before the loop wakes to the finished step.
        await raise_event()
        return _calls(
            _tool_call(
                "call_steer",
                "steer",
                {"call_id": "call_listener", "action": "interject", "payload": "focus"},
            ),
        )

    async def finish() -> dict:
        return _text("done")

    steps = _ScriptedSteps(
        client,
        [start_both, steer_listener_as_event_lands, finish],
    )
    client.generate = steps

    handle = start_async_tool_loop(
        client=client,
        message="start",
        tools={"slow_tool": slow_tool, "listener": listener},
        max_steps=20,
        timeout=30,
    )
    await _wait_for_tool_request(client, "listener")

    async def _listener_started() -> bool:
        return "notification" in channels

    await _wait_for_condition(_listener_started, poll=0.01, timeout=10.0)
    await handle.interject("continue")

    result = await asyncio.wait_for(handle.result(), timeout=30)

    assert result == "done"
    assert received == ["focus"], "the reply's steer must reach the listener"
    assert len(steps.prompts) == 3

    last_prompt = steps.prompts[2]
    reply_at = _index_of(
        last_prompt,
        lambda m: any(c["id"] == "call_steer" for c in m.get("tool_calls") or []),
    )
    event_at = _index_of(
        last_prompt,
        lambda m: _SAME_TICK_EVIDENCE[event] in str(m.get("content")),
    )
    assert reply_at < event_at, "the event must be handled after the reply"
    assert _unanswered_calls(last_prompt) == []


async def _request_after_tick(
    llm_config,
    events: tuple[str, str],
    *,
    during_step: bool,
) -> list[dict]:
    """Land both *events* in one tick and return the next answered request.

    With *during_step* the tick lands while section D races a step, which the
    events supersede; otherwise it lands while section A waits on the tools.
    The interjection that grants the raced step is left out of the returned
    request, so the two runs compare message for message.
    """
    client = new_llm_client(**llm_config)
    client.set_system_message("This turn is fully scripted by the test.")

    gate, gated_tool = make_gated_async_tool("slow-done")
    running: dict[str, asyncio.Task] = {}
    channels: dict[str, asyncio.Queue] = {}

    async def slow_tool() -> str:
        running["slow"] = asyncio.current_task()
        return await gated_tool()

    async def listener(
        _interject_queue: asyncio.Queue | None = None,
        _notification_up_q: asyncio.Queue | None = None,
        _clarification_up_q: asyncio.Queue | None = None,
        _clarification_down_q: asyncio.Queue | None = None,
    ) -> str:
        channels["notification"] = _notification_up_q
        channels["clarification"] = _clarification_up_q
        return await _interject_queue.get()

    async def land_tick() -> None:
        # Queued events are raised before the tool is released, so each
        # waiter has taken its event by the time the tool's completion wakes
        # the loop.
        for event in events:
            if event == "interjection":
                await handle.interject(_SAME_TICK_EVIDENCE[event])
            elif event == "notification":
                channels[event].put_nowait({"message": _SAME_TICK_EVIDENCE[event]})
            elif event == "clarification":
                channels[event].put_nowait(_SAME_TICK_EVIDENCE[event])
        if "tool_result" in events:
            gate.set()
            await running["slow"]

    async def start_both() -> dict:
        return _calls(
            _tool_call("call_slow", "slow_tool", {}),
            _tool_call("call_listener", "listener", {}),
        )

    async def land_tick_and_hang() -> dict:
        await land_tick()
        await asyncio.Event().wait()  # the tick supersedes this step
        raise AssertionError("a superseded step is cancelled")

    async def finish() -> dict:
        return _text("done")

    steps = _ScriptedSteps(
        client,
        (
            [start_both, land_tick_and_hang, finish]
            if during_step
            else [start_both, finish]
        ),
    )
    client.generate = steps

    handle = start_async_tool_loop(
        client=client,
        message="start",
        tools={"slow_tool": slow_tool, "listener": listener},
        max_steps=20,
        timeout=30,
    )
    await _wait_for_tool_request(client, "listener")

    async def _listener_started() -> bool:
        return "notification" in channels

    # Nothing yields between scheduling the tools and section A's wait, so a
    # tool's first step runs only once the loop is waiting there.
    await _wait_for_condition(_listener_started, poll=0.01, timeout=10.0)
    if during_step:
        await handle.interject("continue")
    else:
        await land_tick()

    assert await asyncio.wait_for(handle.result(), timeout=30) == "done"
    assert len(steps.prompts) == (3 if during_step else 2)
    return [
        m
        for m in steps.prompts[-1]
        if not (m.get("_interjection") and m.get("content") == "continue")
    ]


@pytest.mark.asyncio
@pytest.mark.timeout(60)
@pytest.mark.parametrize(
    "events",
    [
        ("tool_result", "notification"),
        ("tool_result", "interjection"),
        ("tool_result", "clarification"),
        ("interjection", "notification"),
        ("interjection", "clarification"),
        ("clarification", "notification"),
    ],
    ids="+".join,
)
async def test_a_tick_is_taken_in_whole_and_the_same_by_either_wait(
    llm_config,
    events: tuple[str, str],
) -> None:
    """Both events of a tick reach the next request, which is the same
    whether section A observed the tick or section D, superseding a step."""
    seen_by_tool_wait = await _request_after_tick(llm_config, events, during_step=False)
    seen_by_llm_race = await _request_after_tick(llm_config, events, during_step=True)

    for event in events:
        assert any(
            _SAME_TICK_EVIDENCE[event] in str(m.get("content"))
            for m in seen_by_tool_wait
        ), f"the {event} was dropped"
    assert seen_by_llm_race == seen_by_tool_wait


def test_requeue_at_front_keeps_the_queue_order() -> None:
    queue: asyncio.Queue = asyncio.Queue()
    for item in ("first", "second", "third"):
        queue.put_nowait(item)

    _requeue_at_front(queue, queue.get_nowait())

    assert [queue.get_nowait() for _ in range(queue.qsize())] == [
        "first",
        "second",
        "third",
    ]
