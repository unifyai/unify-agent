"""Symbolic: ``UNIFY_REVIEW_FORK`` runs the storage review as a fork of the session.

The storage review was 72.5% of AppWorld cost, about 40% of TravelPlanner
and 12% of ScienceWorld on the pre-rebase build, and its first call got 0%
of its input from the cache: it was a fresh conversation, with its own
system prompt holding a JSON dump of the trajectory, its own tools, and a
new client. Forked, its first request is the session's own conversation
plus one user message with the rulebook, so the prefix the session cached
serves it.

Requests are captured at unillm's transport (``tests/cache_discipline_helpers.py``);
with the switch off, or when the fork falls back, the review's requests are
the upstream bytes.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from tests import cache_discipline_helpers as h
from unify.actor import code_act_actor as caa
from unify.common._async_tool import cache_discipline as cd
from unify.settings import SETTINGS


@pytest.fixture
def switches(monkeypatch):
    def set_(*, discipline: bool, fork: bool) -> None:
        monkeypatch.setattr(SETTINGS, "UNIFY_CACHE_DISCIPLINE", discipline)
        monkeypatch.setattr(SETTINGS, "UNIFY_REVIEW_FORK", fork)

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


FORK_REPLIES = (
    h.REVIEW_REPLIES[0],
    h.REVIEW_REPLIES[1],
    # the review: a task tool (refused) and a library read (its own)
    lambda: h.completion(
        calls=[
            ("execute_code", {"code": "rerun the task"}),
            ("FunctionManager_list_functions", {}),
        ],
        call_ids=["review_code", "review_list"],
    ),
    lambda: h.completion(content="Nothing worth storing."),
)


async def _forked_review(monkeypatch):
    from unify.actor.code_act_actor import CodeActActor

    forks: list[dict] = []
    real_fork = caa.fork_llm_client

    def spy(parent, **kwargs):
        client = real_fork(parent, **kwargs)
        forks.append({"parent": parent, "kwargs": kwargs, "client": client})
        return client

    monkeypatch.setattr(caa, "fork_llm_client", spy)
    counter: dict = {}
    actor = CodeActActor()
    try:
        tools = h.session_tools(actor)
        tools["execute_code"] = h.make_tools(counter)["execute_code"]
        summary, _, requests = await h.scenario_review(
            FORK_REPLIES,
            actor=actor,
            tools=tools,
        )
    finally:
        await actor.close()
    return summary, requests, forks, counter


# ── the fork ─────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_the_review_request_continues_the_sessions_conversation(
    monkeypatch,
    switches,
):
    switches(discipline=True, fork=True)
    summary, requests, _forks, _counter = await _forked_review(monkeypatch)
    assert summary == "Nothing worth storing."
    assert len(requests) == 4
    last, review = requests[1], requests[2]

    # The session's last request, byte for byte, then the session's own reply
    # to it, then exactly one appended user message.
    n = len(last["messages"])
    assert _dumps(review["messages"])[:n] == _dumps(last["messages"])
    assert len(review["messages"]) == n + 2
    reply, appended = review["messages"][n], review["messages"][n + 1]
    assert reply["role"] == "assistant"
    assert reply["content"] == "Listed the stored functions; there are none."
    assert appended["role"] == "user"
    assert appended["content"].startswith(
        "## Storage Review\n\nThe task above is over.",
    )
    assert "## Final Result\n\nListed the stored functions" in appended["content"]
    # No trajectory dump and no new system prompt: the conversation is it.
    assert "## Completed Trajectory" not in json.dumps(review["messages"])
    assert review["messages"][0] == {
        "role": "system",
        "content": "You are a scripted actor.",
    }
    # The session's tools and tool choice, as it last sent them.
    assert h.request_bytes(review)["tools"] == h.request_bytes(last)["tools"]
    assert review["tool_choice"] == last["tool_choice"]


@pytest.mark.asyncio
async def test_the_fork_keeps_the_sessions_effort_under_the_review_origin(
    monkeypatch,
    switches,
):
    switches(discipline=True, fork=True)
    _summary, requests, forks, _counter = await _forked_review(monkeypatch)
    assert len(forks) == 1
    fork = forks[0]
    assert fork["kwargs"]["origin"] == "StorageCheck"
    assert fork["kwargs"]["purpose"] == "planning"
    assert fork["client"].reasoning_effort == fork["parent"].reasoning_effort
    assert requests[2]["reasoning_effort"] == requests[1]["reasoning_effort"] == "low"
    # Like the standalone review, the running loop then names it by loop id.
    assert fork["client"].origin == "StorageCheck(CodeActActor.act)"
    assert fork["client"] is not fork["parent"]


@pytest.mark.asyncio
async def test_a_task_tool_is_refused_in_the_review_and_the_list_stays(
    monkeypatch,
    switches,
):
    switches(discipline=True, fork=True)
    _summary, requests, _forks, counter = await _forked_review(monkeypatch)
    assert counter == {}  # the review never ran the task's code runner
    follow_up = requests[3]
    replies = {
        m["tool_call_id"]: m["content"]
        for m in follow_up["messages"]
        if m.get("role") == "tool"
    }
    assert caa._REVIEW_FORK_MASK_RULE in replies["review_code"]
    assert replies["review_list"] == "{}"
    assert h.request_bytes(follow_up)["tools"] == h.request_bytes(requests[1])["tools"]
    sent = _dumps(requests[2]["messages"])
    assert _dumps(follow_up["messages"])[: len(sent)] == sent


@pytest.mark.asyncio
async def test_the_fork_carries_the_update_first_and_needs_repair_notes(
    monkeypatch,
    switches,
):
    """The two notes the standalone review gets from their switches."""
    switches(discipline=True, fork=True)
    monkeypatch.setattr(SETTINGS, "UNIFY_FUNCTION_PATCH", True)
    repair = (
        "## Needs Repair\n\n- `broken_fn`: 1 failure(s) after 0 pass(es); "
        "last failure: boom\n\n"
    )
    monkeypatch.setattr(caa, "_storage_needs_repair_note", lambda: repair)
    _summary, requests, _forks, _counter = await _forked_review(monkeypatch)
    appended = requests[2]["messages"][-1]["content"]
    update_first = caa._storage_update_first_note()
    assert update_first.startswith("### Update before you add")
    assert update_first in appended
    assert repair in appended
    assert (
        appended.index(update_first)
        < appended.index(repair)
        < appended.index("## Final Result")
    )


# ── fallbacks ────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_without_cache_discipline_the_review_runs_as_shipped(
    switches,
    info_lines,
):
    switches(discipline=False, fork=True)
    summary, _, requests = await h.scenario_review()
    assert summary == "Nothing worth storing."
    golden = json.loads(h.GOLDEN.read_text())["review"]
    assert [h.request_bytes(r) for r in requests] == golden
    assert any(
        "StorageCheck fork skipped: UNIFY_REVIEW_FORK needs UNIFY_CACHE_DISCIPLINE"
        in line
        for line in info_lines
    )


@pytest.mark.asyncio
async def test_off_the_review_is_upstreams_and_nothing_is_logged(
    switches,
    info_lines,
):
    switches(discipline=False, fork=False)
    _summary, _, requests = await h.scenario_review()
    golden = json.loads(h.GOLDEN.read_text())["review"]
    assert [h.request_bytes(r) for r in requests] == golden
    assert not any("fork skipped" in line for line in info_lines)


def _recorded_session(extra_messages=()):
    client = h.new_client("You are a scripted actor.")
    client._messages.extend(
        [
            {"role": "user", "content": "task"},
            *extra_messages,
        ],
    )
    cd.record_sent_request(
        client,
        list(client.messages),
        {
            "tools": [{"type": "function", "function": {"name": "t"}}],
            "tool_choice": "auto",
        },
    )
    client._messages.append({"role": "assistant", "content": "done"})
    inner = SimpleNamespace(_client=client, _compression=SimpleNamespace(count=0))
    actor = SimpleNamespace(_preprocess_msgs=None)
    return client, inner, actor


def test_the_fork_source_is_the_raw_history_and_the_last_tools(switches):
    switches(discipline=True, fork=True)
    client, inner, actor = _recorded_session()
    source, reason = caa._review_fork_source(inner, actor)
    assert reason is None
    assert source["client"] is client
    assert source["tools"] == [{"type": "function", "function": {"name": "t"}}]
    assert source["tool_choice"] == "auto"
    assert source["messages"] == client.messages
    client._messages.append({"role": "user", "content": "later"})
    assert source["messages"][-1] == {"role": "assistant", "content": "done"}


@pytest.mark.parametrize(
    ("case", "reason"),
    [
        ("off", None),
        ("no discipline", "needs UNIFY_CACHE_DISCIPLINE"),
        ("compressed", "history was compressed"),
        ("unrecorded", "recorded no request"),
        ("rewritten", "history changed after its last request"),
        ("unanswered", "unanswered tool calls"),
    ],
)
def test_the_review_falls_back_and_says_why(switches, case, reason):
    switches(discipline=case != "no discipline", fork=case != "off")
    extra = ()
    if case == "unanswered":
        extra = (
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "open_call",
                        "type": "function",
                        "function": {"name": "t", "arguments": "{}"},
                    },
                ],
            },
        )
    client, inner, actor = _recorded_session(extra)
    if case == "compressed":
        inner._compression.count = 1
    if case == "unrecorded":
        cd.restore_sent_request(client, None)
    if case == "rewritten":
        client._messages[1]["content"] = "task, edited after it was sent"
    source, why = caa._review_fork_source(inner, actor)
    assert source is None
    if reason is None:
        assert why is None
    else:
        assert reason in why


def test_the_outcome_hook_adds_nothing_yet():
    assert caa._storage_review_outcome_note() == ""
