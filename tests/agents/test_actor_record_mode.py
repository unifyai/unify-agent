"""Symbolic: in record mode the actor offers no steering surface, binds record and
agents, and sees new entries only between its steps."""

import asyncio
import json

import pytest

from tests import cache_discipline_helpers as h
from unify.actor.environments import actor as actor_env
from unify.settings import SETTINGS

_DONE = [lambda: h.completion(content="done")]
_STEERING = {
    "wait",
    "steer",
    "ask_about_completed_tool",
    "request_clarification",
    "send_notification",
}


async def _run(replies, monkeypatch, tmp_path, mode="record"):
    from unify.actor.code_act_actor import CodeActActor
    from unify.agents import binding

    monkeypatch.setattr(SETTINGS, "UNIFY_AGENTS", mode)
    monkeypatch.setattr(binding, "records_dir", lambda: tmp_path / "records")
    actor = CodeActActor(
        environments=actor_env.top_level_environments(),
        tool_policy=None,
    )
    try:
        with h.scripted(replies) as provider:
            handle = await actor.act(
                "Answer the request.",
                persist=False,
                can_store=False,
                clarification_enabled=False,
            )
            result = await asyncio.wait_for(handle.result(), 60)
    finally:
        await actor.close()
    return provider.requests, handle, result


def _pool():
    from unify.agents.binding import current_root_pool

    return current_root_pool()


@pytest.mark.asyncio
@pytest.mark.timeout(120)
async def test_record_mode_offers_no_steering_tools_and_no_sub_actor(
    monkeypatch,
    tmp_path,
):
    requests, handle, _ = await _run(_DONE, monkeypatch, tmp_path)
    tools = {t["function"]["name"] for t in requests[0]["tools"] or []}
    assert not (tools & _STEERING)
    assert "primitives.actor" not in json.dumps(requests[0])
    assert "### Team record" in json.dumps(requests[0]["messages"])
    assert handle.agents_pool.record.entries[0].author == "user"


@pytest.mark.asyncio
@pytest.mark.timeout(120)
async def test_record_and_agents_are_in_the_sandbox(monkeypatch, tmp_path):
    replies = [
        lambda: h.completion(
            calls=[
                ("execute_code", {"thought": "t", "code": "record.post('@user hi')"}),
            ],
        ),
        *_DONE,
    ]
    requests, handle, _ = await _run(replies, monkeypatch, tmp_path)
    texts = [e.text for e in handle.agents_pool.record.entries]
    assert texts == ["Answer the request.", "@user hi", "done"]  # request, post, reply
    assert "nobody reads @user in this run" in json.dumps(requests[1]["messages"])


@pytest.mark.asyncio
@pytest.mark.timeout(120)
async def test_switch_off_binds_nothing(monkeypatch, tmp_path):
    requests, handle, _ = await _run(_DONE, monkeypatch, tmp_path, mode="")
    assert getattr(handle, "agents_pool", None) is None
    assert "### Team record" not in json.dumps(requests[0]["messages"])


@pytest.mark.asyncio
@pytest.mark.timeout(120)
async def test_posts_during_a_call_and_a_cell_arrive_together_after_the_tool_result(
    monkeypatch,
    tmp_path,
):
    def first():
        _pool().record.append("user", "posted during the model call")
        return h.completion(
            calls=[
                (
                    "execute_code",
                    {
                        "thought": "t",
                        "code": "from unify.agents.binding import current_root_pool\n"
                        "current_root_pool().record.append('user', "
                        "'posted during the cell')\n'ok'",
                    },
                ),
            ],
        )

    requests, _, result = await _run([first, *_DONE], monkeypatch, tmp_path)
    assert result == "done" and len(requests) == 2  # no extra or cancelled call
    assert "posted during" not in json.dumps(requests[0])
    msgs = requests[1]["messages"]
    tool_i = max(i for i, m in enumerate(msgs) if m["role"] == "tool")
    assert tool_i == len(msgs) - 2 and msgs[-1]["role"] == "user"
    block = json.dumps(msgs[-1]["content"])
    assert "posted during the model call" in block and "posted during the cell" in block
    first_msgs = requests[0]["messages"]
    assert msgs[: len(first_msgs)] == first_msgs


async def _run_slow_cell(monkeypatch, tmp_path, mode):
    from unify.actor.code_act_actor import CodeActActor
    from unify.agents import binding

    monkeypatch.setattr(SETTINGS, "UNIFY_AGENTS", mode)
    monkeypatch.setattr(binding, "records_dir", lambda: tmp_path / "records")
    actor = CodeActActor(
        environments=actor_env.top_level_environments(),
        tool_policy=None,
    )
    # A cell long enough for the code-step heartbeat to fire.
    actor._active_work_heartbeat_interval_s = 0.2
    actor._active_work_fallback_initial_delay_s = 0.3
    replies = [
        lambda: h.completion(
            calls=[
                (
                    "execute_code",
                    {
                        "thought": "t",
                        "code": "import asyncio\nawait asyncio.sleep(2)\n'ok'",
                    },
                ),
            ],
        ),
        *_DONE * 3,
    ]
    try:
        with h.scripted(replies) as provider:
            handle = await actor.act(
                "Answer the request.",
                persist=False,
                can_store=False,
                clarification_enabled=False,
            )
            result = await asyncio.wait_for(handle.result(), 60)
    finally:
        await actor.close()
    return provider.requests, result


@pytest.mark.asyncio
@pytest.mark.timeout(120)
async def test_a_long_cell_never_wakes_the_model_in_record_mode(monkeypatch, tmp_path):
    requests, result = await _run_slow_cell(monkeypatch, tmp_path, "record")
    assert result == "done"
    assert len(requests) == 2  # the cell's call, then one call after the cell ended
    assert "Still working on the code step" not in json.dumps(requests)
    msgs = requests[1]["messages"]
    assert msgs[-1]["role"] == "tool" and "ok" in json.dumps(msgs[-1])
