"""Symbolic: ``UNIFY_REVIEW_GATE_FORK`` asks the review gate in a fork of the session.

The standalone gate (``UNIFY_REVIEW_GATE``) is a new prompt -- its own system
message and the transcript re-rendered as text -- so none of it is in the
provider's cache: on the 4-5 Oct ARC LOW lean-all runs its 24 calls per run
(about 17k tokens each) read 0 cached tokens, 30-35% of the arm's cache
writes. Forked, the gate's request is the session's last request as sent,
the session's reply, and one appended question, so the session's cached
prefix serves all but the question.

Requests are captured at unillm's transport (``tests/cache_discipline_helpers.py``);
nothing leaves the process.
"""

from __future__ import annotations

import json

import pytest

from tests import cache_discipline_helpers as h
from unify.actor import code_act_actor as caa
from unify.actor import review_gate
from unify.common._async_tool import cache_discipline as cd
from unify.settings import ProductionSettings, SETTINGS

_LIBRARIAN = "You are a skill librarian."
_SESSION_SYSTEM = "You are a scripted actor."
_NO = '{"review": false, "reason": "nothing reusable"}'
_YES = '{"review": true, "reason": "working code"}'


@pytest.fixture
def switches(monkeypatch):
    def set_(
        *,
        discipline: bool = True,
        gate: bool = True,
        gate_fork: bool = True,
        review_fork: bool = False,
    ) -> None:
        monkeypatch.setattr(SETTINGS, "UNIFY_CACHE_DISCIPLINE", discipline)
        monkeypatch.setattr(SETTINGS, "UNIFY_REVIEW_GATE", gate)
        monkeypatch.setattr(SETTINGS, "UNIFY_REVIEW_GATE_FORK", gate_fork)
        monkeypatch.setattr(SETTINGS, "UNIFY_REVIEW_FORK", review_fork)
        # A library that is not empty, so the gate is asked.
        monkeypatch.setattr(caa, "_library_counts", lambda *_a, **_k: (1, 0))

    return set_


@pytest.fixture
def info_lines(monkeypatch):
    lines: list[str] = []
    original = caa.logger.info

    def capture(msg, *args, **kwargs):
        lines.append(str(msg))
        return original(msg, *args, **kwargs)

    monkeypatch.setattr(caa.logger, "info", capture)
    return lines


def _dumps(messages: list[dict]) -> list[str]:
    return [json.dumps(m, default=str) for m in messages]


def _replies(gate_reply):
    """The session (a library call, then its answer), the gate, then a review."""
    gate = (
        gate_reply
        if callable(gate_reply)
        else (lambda: h.completion(content=gate_reply))
    )
    return (
        h.REVIEW_REPLIES[0],
        h.REVIEW_REPLIES[1],
        gate,
        lambda: h.completion(content="Nothing worth storing."),
    )


def _is_gate_fork(request: dict) -> bool:
    last = request["messages"][-1]
    return (
        last.get("role") == "user"
        and isinstance(last.get("content"), str)
        and last["content"].startswith("## Library Review Gate")
    )


def _is_standalone_gate(request: dict) -> bool:
    return request["messages"][0]["content"] == review_gate.GATE_SYSTEM_PROMPT


# ── the fork ─────────────────────────────────────────────────────────────


@pytest.mark.asyncio
@pytest.mark.timeout(120)
async def test_the_gate_request_continues_the_sessions_last_request_byte_for_byte(
    switches,
):
    switches()
    summary, _, requests = await h.scenario_review(_replies(_NO))
    assert summary == "review gate: nothing reusable"
    assert len(requests) == 3
    last, gate = requests[1], requests[2]
    assert _is_gate_fork(gate)

    # The actor's last request, byte for byte: system prompt, messages,
    # tools and tool choice -- then the actor's reply, then one question.
    n = len(last["messages"])
    assert _dumps(gate["messages"])[:n] == _dumps(last["messages"])
    assert json.dumps(gate["messages"][:n]) == json.dumps(last["messages"])
    assert h.request_bytes(gate)["tools"] == h.request_bytes(last)["tools"]
    assert gate["tool_choice"] == last["tool_choice"]
    assert gate["reasoning_effort"] == last["reasoning_effort"]
    assert len(gate["messages"]) == n + 2
    reply, question = gate["messages"][n], gate["messages"][n + 1]
    assert reply["role"] == "assistant"
    assert reply["content"] == "Listed the stored functions; there are none."
    assert question["role"] == "user"
    assert question["content"].startswith(review_gate.GATE_FORK_PROMPT)
    assert "## Final reply\n\nListed the stored functions" in question["content"]
    # No standalone gate prompt and no re-rendered transcript.
    assert gate["messages"][0] == {"role": "system", "content": _SESSION_SYSTEM}
    assert review_gate.GATE_SYSTEM_PROMPT not in json.dumps(gate["messages"])
    assert "## Session (its transcript" not in json.dumps(gate["messages"])


@pytest.mark.asyncio
@pytest.mark.timeout(120)
async def test_a_forked_yes_runs_the_review(switches):
    switches()
    summary, _, requests = await h.scenario_review(_replies(_YES))
    assert summary == "Nothing worth storing."
    assert [_is_gate_fork(r) for r in requests] == [False, False, True, False]
    assert requests[3]["messages"][0]["content"].startswith(_LIBRARIAN)


@pytest.mark.asyncio
@pytest.mark.timeout(120)
async def test_a_forked_review_after_a_forked_gate_starts_from_the_session_not_the_gate(
    switches,
):
    """The review forks the session's conversation; the gate's exchange is
    not in it, so both forks share the session's cached prefix."""
    switches(review_fork=True)
    summary, _, requests = await h.scenario_review(_replies(_YES))
    assert summary == "Nothing worth storing."
    last, gate, review = requests[1], requests[2], requests[3]
    n = len(last["messages"])
    assert _dumps(review["messages"])[: n + 1] == _dumps(gate["messages"])[: n + 1]
    assert len(review["messages"]) == n + 2
    assert review["messages"][-1]["content"].startswith("## Storage Review")
    assert not any(_is_gate_fork({"messages": [m]}) for m in review["messages"])


@pytest.mark.asyncio
@pytest.mark.timeout(120)
async def test_a_tool_call_in_the_gate_states_no_decision_and_runs_nothing(
    switches,
):
    switches()
    gate_reply = lambda: h.completion(  # noqa: E731
        calls=[("execute_code", {"code": "rerun the task"})],
    )
    summary, _, requests = await h.scenario_review(_replies(gate_reply))
    # Failed open: the review ran, and the gate's tool call was never run
    # (the session's tools are the managers' own; the review answered next).
    assert summary == "Nothing worth storing."
    assert _is_gate_fork(requests[2])
    assert requests[3]["messages"][0]["content"].startswith(_LIBRARIAN)
    assert "rerun the task" not in json.dumps(requests[3]["messages"])


@pytest.mark.asyncio
@pytest.mark.timeout(120)
async def test_the_gate_fork_shares_the_sessions_affinity_key(monkeypatch, switches):
    switches()
    sets = h.install_affinity_api(monkeypatch)
    forks: list = []
    real_fork = caa.fork_llm_client

    def spy(parent, **kwargs):
        client = real_fork(parent, **kwargs)
        forks.append((parent, kwargs, client))
        return client

    monkeypatch.setattr(caa, "fork_llm_client", spy)
    await h.scenario_review(_replies(_NO))
    ((parent, kwargs, client),) = forks
    assert kwargs["origin"] == review_gate.ORIGIN
    assert kwargs["purpose"] == "planning"
    assert client.cache_affinity == parent.cache_affinity is not None
    assert all(key == parent.cache_affinity for key, _n in sets)


@pytest.mark.asyncio
@pytest.mark.timeout(120)
async def test_the_review_effort_setting_sets_the_forked_gates(monkeypatch, switches):
    switches()
    monkeypatch.setattr(SETTINGS, "UNIFY_REVIEW_REASONING_EFFORT", "medium")
    _summary, _, requests = await h.scenario_review(_replies(_NO))
    assert requests[1]["reasoning_effort"] == "low"
    assert requests[2]["reasoning_effort"] == "medium"
    assert _is_gate_fork(requests[2])


def test_a_forced_tool_choice_is_sent_as_auto():
    """The answer is text: a forced choice would make the model call a tool."""

    class _Client:
        def __init__(self):
            self.calls = []

        async def generate(self, **kwargs):
            self.calls.append(kwargs)
            return h.completion(content=_NO)

    import asyncio

    client = _Client()
    tools = [{"type": "function", "function": {"name": "t"}}]
    decision = asyncio.run(
        review_gate.decide_in_fork(
            client_factory=lambda: client,
            fork_source={
                "sent_messages": [{"role": "system", "content": "s"}],
                "tools": tools,
                "tool_choice": "required",
            },
            final_result="5",
        ),
    )
    assert decision == review_gate.GateDecision(False, "nothing reusable", True)
    (call,) = client.calls
    assert call["tool_choice"] == "auto"
    assert call["tools"] == tools
    assert call["messages"][0] == {"role": "system", "content": "s"}


@pytest.mark.parametrize(
    "error",
    [RuntimeError("provider down"), None],
)
def test_a_failed_or_unreadable_forked_gate_runs_the_review(error):
    import asyncio

    class _Client:
        async def generate(self, **_kwargs):
            if error is not None:
                raise error
            return h.completion(content="I think so")

    decision = asyncio.run(
        review_gate.decide_in_fork(
            client_factory=_Client,
            fork_source={"sent_messages": [], "tools": None, "tool_choice": None},
            final_result="done",
        ),
    )
    assert decision.review is True and decision.decided is False


# ── the text ─────────────────────────────────────────────────────────────


def test_the_fork_asks_the_standalone_gates_criteria():
    criteria = review_gate.GATE_SYSTEM_PROMPT.split(" A review costs", 1)[1]
    assert criteria in review_gate.GATE_FORK_PROMPT
    assert review_gate.GATE_FORK_PROMPT.startswith("## Library Review Gate\n\n")
    assert "do not call any tool" in review_gate.GATE_FORK_PROMPT
    message = review_gate.build_fork_message(
        final_result="5",
        outcome_note="## Checked Outcome\n\n- Solved: yes",
    )
    assert message.index("## Checked Outcome") < message.index("## Final reply\n\n5")
    assert message.endswith("Should this session's work be reviewed for the library?")


def test_the_fork_names_no_benchmark_and_asks_for_no_example_checks():
    import re

    words = set(re.findall(r"[a-z]+", review_gate.GATE_FORK_PROMPT.lower()))
    for word in ("arc", "appworld", "scienceworld", "crafter", "grid", "example"):
        assert word not in words


# ── fallbacks and the switch ───────────────────────────────────────────────


@pytest.mark.asyncio
@pytest.mark.timeout(120)
async def test_without_cache_discipline_the_gate_is_asked_standalone_and_says_why(
    switches,
    info_lines,
):
    switches(discipline=False)
    _summary, _, requests = await h.scenario_review(_replies(_NO))
    assert _is_standalone_gate(requests[2])
    assert not any(_is_gate_fork(r) for r in requests)
    assert any(
        "StorageCheck gate fork skipped: UNIFY_REVIEW_GATE_FORK needs "
        "UNIFY_CACHE_DISCIPLINE" in line
        for line in info_lines
    )


@pytest.mark.asyncio
@pytest.mark.timeout(120)
async def test_off_the_gate_is_standalone_and_nothing_is_logged(switches, info_lines):
    switches(gate_fork=False)
    _summary, _, requests = await h.scenario_review(_replies(_NO))
    assert _is_standalone_gate(requests[2])
    assert requests[2]["tools"] in (None, [])
    assert not any("gate fork skipped" in line for line in info_lines)


@pytest.mark.asyncio
@pytest.mark.timeout(120)
async def test_without_the_gate_the_fork_switch_does_nothing(switches):
    switches(gate=False)
    summary, _, requests = await h.scenario_review(_replies("unused"))
    # No gate: the review runs right after the session (the third reply
    # answers it).
    assert summary == "unused"
    assert not any(_is_gate_fork(r) or _is_standalone_gate(r) for r in requests)


def test_the_gate_forks_under_the_core_surface(monkeypatch, switches):
    """The gate runs no tool, so a session's core tool list does not stop it."""
    switches()
    monkeypatch.setattr(SETTINGS, "UNIFY_TOOL_SURFACE", "core")
    client = h.new_client(_SESSION_SYSTEM)
    client._messages.append({"role": "user", "content": "task"})
    cd.record_sent_request(
        client,
        list(client.messages),
        {
            "tools": [{"type": "function", "function": {"name": "execute_code"}}],
            "tool_choice": "auto",
        },
    )
    client._messages.append({"role": "assistant", "content": "done"})
    from types import SimpleNamespace

    inner = SimpleNamespace(_client=client, _compression=SimpleNamespace(count=0))
    actor = SimpleNamespace(_preprocess_msgs=None)
    source, reason = caa._gate_fork_source(inner, actor)
    assert reason is None
    assert source["sent_messages"] == [
        {"role": "system", "content": _SESSION_SYSTEM},
        {"role": "user", "content": "task"},
        {"role": "assistant", "content": "done"},
    ]
    review_source, review_reason = caa._review_fork_source(inner, actor)
    assert review_source is None and review_reason is None  # UNIFY_REVIEW_FORK off


@pytest.mark.parametrize("value, expected", [("1", True), ("0", False), ("", False)])
def test_the_setting_parses_booleans(value, expected):
    settings = ProductionSettings(UNIFY_REVIEW_GATE_FORK=value)
    assert settings.UNIFY_REVIEW_GATE_FORK is expected


def test_the_default_is_off():
    assert ProductionSettings.model_fields["UNIFY_REVIEW_GATE_FORK"].default is False
