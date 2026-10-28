"""Symbolic: the actor on the agent record.

Its only model-API tool is ``execute_code``; the record starts with the request, its rules are one help topic away
(``tools.help("record")``), the handle carries the record's pool, a stopped actor stops its helpers and posts no
reply, and an answer from a cell ends the request with no further model call. Cells run in the real sandboxed worker;
the model is the scripted transport.
"""

import asyncio
import json

import pytest

from tests import cache_discipline_helpers as h
from tests.actor.code_act.core_world import core_world  # noqa: F401 (fixture)
from tests.actor.code_act.sandbox_world import needs_bwrap, world  # noqa: F401 (fixture)
from unify.common import tool_registry as R

pytestmark = [needs_bwrap, pytest.mark.asyncio, pytest.mark.timeout(180)]

_DONE = [lambda: h.completion(content="done")]


def _cell(code: str):
    return lambda: h.completion(calls=[("execute_code", {"thought": "t", "code": code})])


def _actor(monkeypatch, tmp_path):
    from unify.actor.code_act_actor import CodeActActor
    from unify.agents import binding

    monkeypatch.setattr(binding, "_run_home", lambda: tmp_path / "home")
    return CodeActActor()


async def _run(replies, monkeypatch, tmp_path):
    actor = _actor(monkeypatch, tmp_path)
    with h.scripted(replies) as provider:
        handle = await actor.act("Answer the request.", persist=False, clarification_enabled=False)
        result = await asyncio.wait_for(handle.result(), 60)
    return provider.requests, handle, result


def _tool_texts(request: dict) -> str:
    return json.dumps([m.get("content") for m in request["messages"] if m.get("role") == "tool"])


async def test_the_actor_has_one_tool_and_the_record_starts_with_the_request(
    core_world,  # noqa: F811
    monkeypatch,
    tmp_path,
):
    requests, handle, result = await _run(_DONE, monkeypatch, tmp_path)
    assert result == "done"
    assert [t["function"]["name"] for t in requests[0]["tools"]] == ["execute_code"]
    assert "primitives.actor" not in json.dumps(requests[0])
    first = handle.agents_pool.record.entries[0]
    assert (first.author, first.text) == ("user", "Answer the request.")


async def test_the_record_rules_are_one_help_topic_away(core_world, monkeypatch, tmp_path):  # noqa: F811
    requests, _, _ = await _run([_cell('print(tools.help("record"))'), *_DONE], monkeypatch, tmp_path)
    # rev 2: the record's rules (COMMON_SECTION) come after the cache-preserving system prompt, in the first user
    # message, and tools.help("record") gives them again
    # the run's own section: this suite runs with UNIFY_MEMORY=off, so the one that names no memory agent
    rules = R.common_section()
    messages = requests[0]["messages"]
    assert rules not in str(messages[0].get("content"))  # the system message
    first_user = next(m for m in messages if m.get("role") == "user")
    assert rules in str(first_user.get("content"))
    assert json.dumps(rules)[1:-1] in _tool_texts(requests[1])  # the tool texts are JSON-encoded


async def test_a_stopped_main_agent_stops_its_helpers_and_records_no_reply(
    core_world,  # noqa: F811
    monkeypatch,
    tmp_path,
):
    from unify.agents.binding import current_root_pool

    actor = _actor(monkeypatch, tmp_path)
    helper = {}

    def first():
        pool = current_root_pool()
        pool.record.add_agent("h9", spawner=pool.record.root)
        helper["task"] = pool._tasks["h9"] = asyncio.get_running_loop().create_task(asyncio.sleep(3600))
        return h.completion(calls=[("execute_code", {"thought": "t", "code": "import time\ntime.sleep(5)"})])

    with h.scripted([first, *_DONE * 3]):
        handle = await actor.act("Answer the request.", persist=False, clarification_enabled=False)
        await asyncio.sleep(1.0)
        await handle.stop("the user stopped it")
        await asyncio.wait_for(handle.result(), 60)
    pool = current_root_pool()
    assert helper["task"].done()
    kinds = [(e.author, e.kind) for e in pool.record.entries]
    assert (pool.record.root, "reply") not in kinds
    notice = [e.text for e in pool.record.entries if e.kind == "system"]
    assert notice and "stopped" in notice[-1]


async def test_an_answer_from_a_cell_ends_the_request_with_no_block_and_no_model_call(
    core_world,  # noqa: F811
    monkeypatch,
    tmp_path,
):
    from unify.agents.binding import current_root_pool

    actor = _actor(monkeypatch, tmp_path)

    def first():
        current_root_pool().record.append("user", "posted during the last call")
        return h.completion(calls=[("execute_code", {"thought": "t", "code": "tools.answer('final answer')"})])

    with h.scripted([first]) as provider:
        handle = await actor.act("Answer the request.", persist=False, clarification_enabled=False)
        result = await asyncio.wait_for(handle.result(), 60)
    assert result == "final answer" and len(provider.requests) == 1
    assert "posted during the last call" not in json.dumps(provider.requests)
