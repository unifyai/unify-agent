"""Symbolic: ``UNIFY_BUDGET_FOOTER`` tells the model its steps left near ``max_steps``.

The step limit stops a request without warning: the model learns of it only
from the stop. With the switch, once the request is within the last tenth
of its cap, each tool result ends with one line giving the steps left,
counted as the limit counts them (per request under
``UNIFY_STEP_CAP_REPLY``). It informs only. The line is part of the new
result, so every request still extends the one before it byte for byte.
The transport is scripted, so nothing leaves the process.
"""

from __future__ import annotations

import asyncio
import re

import pytest

from tests import cache_discipline_helpers as h
from unify.settings import SETTINGS

TASK = "Find the answer and reply with it."
CONTINUE = "Please continue and give your answer."
FINAL = "The answer is 42."
MAX_STEPS = 20  # the footer shows from 2 steps left
BOUND = 5
FOOTER = re.compile(
    r"\n\n\[step budget\] (\d+) of 20 steps left before the step limit stops "
    r"this request \(each message is a step: a tool call and its result take "
    r"two\)\.$",
)


def _last_request(messages: list) -> str:
    for message in reversed(messages):
        if message.get("role") == "user" and not message.get("_loop_authored"):
            return str(message.get("content") or "")
    return ""


def _request_start(messages: list) -> int:
    for i in range(len(messages) - 1, -1, -1):
        if messages[i].get("role") == "user" and messages[i].get("content") == CONTINUE:
            return i
    return 0


class _Model:
    def __init__(self):
        self.requests: list[list[dict]] = []

    async def __call__(self, *, shared_session=None, client=None, **kw):
        messages = kw.get("messages") or []
        self.requests.append(messages)
        if _request_start(messages):
            # Two more looks in the next request, then the answer.
            after = messages[_request_start(messages) :]
            if sum(m.get("role") == "tool" for m in after) < 2:
                return h.completion(content="Looking.", calls=[("look", {})])
            return h.completion(content=FINAL)
        return h.completion(content="Looking.", calls=[("look", {})])


async def look() -> str:
    """Look again."""
    return "Nothing new."


def _tool_results(messages: list) -> list[str]:
    return [str(m.get("content")) for m in messages if m.get("role") == "tool"]


async def _run(monkeypatch, *, footer: bool, cap_reply: str = "draft"):
    monkeypatch.setattr(SETTINGS, "UNIFY_BUDGET_FOOTER", footer)
    monkeypatch.setattr(SETTINGS, "UNIFY_STEP_CAP_REPLY", cap_reply)
    from unify.common.async_tool_loop import start_async_tool_loop

    model = _Model()
    with h.scripted(()):
        import unillm.clients.uni_llm as uni_llm

        uni_llm._acompletion_with_transient_retry = model
        handle = start_async_tool_loop(
            h.new_client(),
            TASK,
            {"look": look},
            log_steps=False,
            timeout=30,
            persist=True,
            max_steps=MAX_STEPS,
        )
        capped = (await asyncio.wait_for(h._next_response(handle), BOUND))["content"]
        first_request_calls = len(model.requests)
        await handle.interject(CONTINUE)
        await asyncio.wait_for(h._next_response(handle), BOUND)
        await handle.stop()
        await asyncio.wait_for(handle.result(), BOUND)
    return model, capped, first_request_calls


@pytest.mark.asyncio
async def test_off_no_result_carries_a_footer(monkeypatch):
    model, _capped, _ = await _run(monkeypatch, footer=False)
    for request in model.requests:
        assert not any("[step budget]" in r for r in _tool_results(request))


@pytest.mark.asyncio
@pytest.mark.parametrize("discipline", [False, True], ids=["plain", "discipline"])
async def test_on_the_last_tenth_of_the_cap_is_announced(monkeypatch, discipline):
    monkeypatch.setattr(SETTINGS, "UNIFY_CACHE_DISCIPLINE", discipline)
    model, capped, first_calls = await _run(monkeypatch, footer=True)
    assert capped.startswith("🔚 Stopped at the step limit")
    results = _tool_results(model.requests[first_calls - 1])
    left = [FOOTER.search(r) for r in results]
    shown = [int(m.group(1)) for m in left if m]
    # Shown only within the last tenth (2 of 20), counting down.
    assert shown and all(n <= 2 for n in shown), results
    assert shown[-1] <= 1
    assert shown == sorted(shown, reverse=True)
    # Before that, results are as shipped.
    first_shown = next(i for i, m in enumerate(left) if m)
    assert first_shown > 0
    assert all(r == "Nothing new." for r in results[:first_shown])
    # Results end with the footer; nothing else changes, so each request
    # extends the one before it.
    for earlier, later in zip(model.requests, model.requests[1:]):
        assert later[: len(earlier)] == earlier


@pytest.mark.asyncio
async def test_on_each_request_counts_its_own_steps(monkeypatch):
    """Per-request counting (UNIFY_STEP_CAP_REPLY): the next request starts
    with its full budget, so its first results carry no footer."""
    model, _capped, first_calls = await _run(monkeypatch, footer=True)
    last = model.requests[-1]
    assert _request_start(last) > 0
    results = _tool_results(last[_request_start(last) :])
    assert results == ["Nothing new.", "Nothing new."]


def test_the_switch_parses_as_a_bool(monkeypatch):
    from unify.settings import ProductionSettings

    monkeypatch.setenv("UNIFY_BUDGET_FOOTER", "1")
    assert ProductionSettings(_env_file=None).UNIFY_BUDGET_FOOTER is True
    monkeypatch.setenv("UNIFY_BUDGET_FOOTER", "")
    assert ProductionSettings(_env_file=None).UNIFY_BUDGET_FOOTER is False
