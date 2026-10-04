"""Symbolic: ``UNIFY_REVIEW_GATE``: a yes/no call decides whether the storage review runs.

The storage review after a session ran on every session, solved or not: 4.1
to 4.5 calls and 10-16% of USD per ARC LOW episode on 4 Oct, on episodes
that mostly left nothing to store. With the gate, one tool-free call on the
end of the trajectory decides first (Prime's refine gate); a failed or
unreadable gate runs the review as shipped.
"""

from __future__ import annotations

import asyncio
import json
import re

import pytest

from tests import cache_discipline_helpers as h
from unify.actor import review_gate
from unify.settings import SETTINGS

_LIBRARIAN = "You are a skill librarian."


# ── the decision ──────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "raw, review, reason",
    [
        ('{"review": true, "reason": "new parser ran"}', True, "new parser ran"),
        ('{"review": false, "reason": "one-off answer"}', False, "one-off answer"),
        ('Thinking...\n{"review": false, "reason": "x"}', False, "x"),
        ('{"review": false}', False, ""),
    ],
)
def test_a_stated_decision_is_read(raw, review, reason):
    decision = review_gate.parse_decision(raw)
    assert decision == review_gate.GateDecision(review, reason, True)


@pytest.mark.parametrize(
    "raw",
    ["", "yes", '{"review": "no"}', "{not json}", '["review", false]', None],
)
def test_a_reply_without_a_decision_is_none(raw):
    assert review_gate.parse_decision(raw) is None


class _Client:
    def __init__(self, reply=None, error=None):
        self.reply, self.error, self.calls = reply, error, []

    async def generate(self, **kwargs):
        self.calls.append(kwargs)
        if self.error:
            raise self.error
        return self.reply


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "client",
    [_Client(error=RuntimeError("provider down")), _Client(reply="I think so")],
)
async def test_a_failed_or_unreadable_gate_runs_the_review(client):
    decision = await review_gate.decide(
        client_factory=lambda: client,
        trajectory=[],
        final_result="done",
    )
    assert decision.review is True and decision.decided is False


@pytest.mark.asyncio
async def test_the_gate_sends_one_tool_free_request():
    client = _Client(reply='{"review": false, "reason": "nothing reusable"}')
    decision = await review_gate.decide(
        client_factory=lambda: client,
        trajectory=[{"role": "user", "content": "Add 2 and 3."}],
        final_result="5",
        outcome_note="## Checked Outcome\n\n- Solved: yes",
    )
    assert decision == review_gate.GateDecision(False, "nothing reusable", True)
    (call,) = client.calls
    assert set(call) == {"user_message", "system_message"}
    assert call["system_message"] == review_gate.GATE_SYSTEM_PROMPT
    for part in ("Add 2 and 3.", "## Checked Outcome", "## Final reply\n\n5"):
        assert part in call["user_message"]


def test_the_rendered_trajectory_keeps_its_end_and_skips_system_messages():
    messages = [
        {"role": "system", "content": "SYSTEM PROMPT"},
        {"role": "user", "content": "first " + "x" * 60_000},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {"function": {"name": "execute_code", "arguments": '{"code": "1+1"}'}},
            ],
        },
        {"role": "tool", "name": "execute_code", "content": "2"},
        {"role": "assistant", "content": "the end"},
    ]
    text = review_gate.render_trajectory(messages)
    assert "SYSTEM PROMPT" not in text
    assert len(text) <= review_gate.GATE_TAIL_CHARS + 100
    assert text.endswith("[assistant]\nthe end")
    assert '-> execute_code({"code": "1+1"})' in text
    assert "[tool result (execute_code)]\n2" in text


def test_the_gate_names_no_benchmark_and_asks_for_no_example_checks():
    words = set(re.findall(r"[a-z]+", review_gate.GATE_SYSTEM_PROMPT.lower()))
    for word in (
        "arc",
        "appworld",
        "scienceworld",
        "crafter",
        "grid",
        "demo",
        "example",
    ):
        assert word not in words


# ── the session's review ────────────────────────────────────────────────


async def _session(
    monkeypatch,
    *,
    gate: bool,
    gate_reply: str,
    counts: tuple = (1, 0),
) -> list[dict]:
    """One session and its review; *counts* is the library's (functions,
    guidance) size the actor reads, non-empty by default so the gate is asked."""
    from unify.actor import code_act_actor as caa
    from unify.actor.code_act_actor import CodeActActor

    monkeypatch.setattr(SETTINGS, "UNIFY_REVIEW_GATE", gate)
    monkeypatch.setattr(caa, "_library_counts", lambda *_a, **_k: counts)
    actor = CodeActActor()
    try:
        with h.scripted([]) as provider:

            def _reply():
                request = provider.requests[-1]
                system = request["messages"][0]["content"]
                if system == review_gate.GATE_SYSTEM_PROMPT:
                    return h.completion(content=gate_reply)
                if len(provider.requests) == 1:
                    return h.completion(
                        calls=[
                            ("FunctionManager_search_functions", {"query": "files"}),
                            ("GuidanceManager_search", {"k": 3}),
                        ],
                    )
                return h.completion(content="done")

            provider.replies = [_reply] * 30
            handle = await actor.act("List the files in the workspace.", persist=False)
            await asyncio.wait_for(handle.result(), 60)
            # The review runs after the result; wait for the lifecycle.
            await asyncio.wait_for(handle._completion_event.wait(), 60)
    finally:
        await actor.close()
    return provider.requests


def _gate_requests(requests):
    return [
        r
        for r in requests
        if r["messages"][0]["content"] == review_gate.GATE_SYSTEM_PROMPT
    ]


def _review_requests(requests):
    return [r for r in requests if r["messages"][0]["content"].startswith(_LIBRARIAN)]


@pytest.mark.asyncio
@pytest.mark.timeout(180)
async def test_a_no_skips_the_review(monkeypatch):
    requests = await _session(
        monkeypatch,
        gate=True,
        gate_reply='{"review": false, "reason": "nothing reusable"}',
    )
    (gate_request,) = _gate_requests(requests)
    assert _review_requests(requests) == []
    assert gate_request["tools"] in (None, [])
    assert gate_request["reasoning_effort"] == review_gate.GATE_EFFORT
    assert "List the files in the workspace." in gate_request["messages"][1]["content"]


@pytest.mark.asyncio
@pytest.mark.timeout(180)
async def test_a_yes_runs_the_review_after_it(monkeypatch):
    requests = await _session(
        monkeypatch,
        gate=True,
        gate_reply='{"review": true, "reason": "working code"}',
    )
    (gate_request,) = _gate_requests(requests)
    reviews = _review_requests(requests)
    assert reviews
    assert requests.index(gate_request) < requests.index(reviews[0])


@pytest.mark.asyncio
@pytest.mark.timeout(180)
async def test_an_unreadable_gate_runs_the_review(monkeypatch):
    requests = await _session(monkeypatch, gate=True, gate_reply="hmm")
    assert len(_gate_requests(requests)) == 1
    assert _review_requests(requests)


@pytest.mark.asyncio
@pytest.mark.timeout(180)
async def test_off_no_gate_and_the_review_runs(monkeypatch):
    requests = await _session(monkeypatch, gate=False, gate_reply="")
    assert _gate_requests(requests) == []
    assert _review_requests(requests)


@pytest.mark.asyncio
@pytest.mark.timeout(180)
async def test_the_gate_takes_the_review_effort_when_set(monkeypatch):
    monkeypatch.setattr(SETTINGS, "UNIFY_REVIEW_REASONING_EFFORT", "medium")
    requests = await _session(
        monkeypatch,
        gate=True,
        gate_reply='{"review": false, "reason": "x"}',
    )
    (gate_request,) = _gate_requests(requests)
    assert gate_request["reasoning_effort"] == "medium"


# ── an empty library ────────────────────────────────────────────────


@pytest.mark.parametrize(
    "counts, empty",
    [
        ((0, 0), True),
        ((1, 0), False),
        ((0, 1), False),
        ((None, 0), False),
        ((0, None), False),
        ((None, None), False),
    ],
)
def test_only_known_zero_counts_are_an_empty_library(counts, empty):
    assert review_gate.library_is_empty(counts) is empty


@pytest.mark.asyncio
@pytest.mark.timeout(180)
async def test_an_empty_library_is_reviewed_without_asking_the_gate(monkeypatch):
    # The first sessions seed the library: the gate would have said no here.
    requests = await _session(
        monkeypatch,
        gate=True,
        gate_reply='{"review": false, "reason": "nothing reusable"}',
        counts=(0, 0),
    )
    assert _gate_requests(requests) == []
    assert _review_requests(requests)


@pytest.mark.asyncio
@pytest.mark.timeout(180)
async def test_an_unknown_library_size_asks_the_gate(monkeypatch):
    requests = await _session(
        monkeypatch,
        gate=True,
        gate_reply='{"review": false, "reason": "nothing reusable"}',
        counts=(None, None),
    )
    assert len(_gate_requests(requests)) == 1
    assert _review_requests(requests) == []


@pytest.mark.asyncio
@pytest.mark.timeout(180)
async def test_off_an_empty_library_changes_nothing(monkeypatch):
    requests = await _session(monkeypatch, gate=False, gate_reply="", counts=(0, 0))
    assert _gate_requests(requests) == []
    assert _review_requests(requests)


@pytest.mark.parametrize("value, expected", [("1", True), ("0", False), ("", False)])
def test_the_setting_parses_booleans(value, expected):
    from unify.settings import ProductionSettings

    assert ProductionSettings(UNIFY_REVIEW_GATE=value).UNIFY_REVIEW_GATE is expected


def test_the_default_is_off():
    from unify.settings import ProductionSettings

    assert ProductionSettings.model_fields["UNIFY_REVIEW_GATE"].default is False


def test_the_system_prompt_is_one_fixed_text():
    # One static request prefix for every gate call.
    assert json.loads(json.dumps(review_gate.GATE_SYSTEM_PROMPT)) == (
        review_gate.GATE_SYSTEM_PROMPT
    )
