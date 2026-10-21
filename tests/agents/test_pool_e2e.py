"""Symbolic: the actor and its teammates in their own sandboxed interpreters, all through the agent record.

The model is the scripted transport (``tests/scripted_model.py``): the actor's requests and its teammates' (the
``helper`` kind) are answered from their own scripts. Cells run in the real sandboxed worker and call
``tools.delegate``, ``tools.wait`` and ``tools.answer`` through the tool channel.
"""

import asyncio
import json

import pytest

from tests.actor.code_act.core_world import core_world  # noqa: F401 (fixture)
from tests.actor.code_act.sandbox_world import needs_bwrap, world  # noqa: F401 (fixture)
from tests.scripted_model import Always, ScriptedModel, reply, scripted

pytestmark = [needs_bwrap, pytest.mark.asyncio, pytest.mark.timeout(300)]


def _code(text: str):
    return reply(calls=[("execute_code", {"code": text})])


def _task(call) -> str:
    """The first user message of a request: the task the agent was given."""
    return next(json.dumps(m.get("content")) for m in call.messages if m.get("role") == "user")


async def _run(model: ScriptedModel, monkeypatch, tmp_path):
    from unify.actor.code_act_actor import CodeActActor
    from unify.agents import binding

    monkeypatch.setattr(binding, "_run_home", lambda: tmp_path / "home")
    with scripted(model):
        handle = await CodeActActor().act(
            "Count the rows of two tables.",
            persist=False,
            clarification_enabled=False,
        )
        pool = handle.agents_pool
        try:
            result = await asyncio.wait_for(handle.result(), 240)
        finally:
            await pool.close("test over")
    return result, pool


def _spawners(pool) -> dict[str, str]:
    """Each started agent's name and the agent that started it."""
    return {
        name: p.spawner
        for name, p in pool.record.participants.items()
        if name != pool.record.root and p.spawner is not None
    }


FAN_OUT = (
    "names = [tools.delegate('Count the rows of the first table, please.'),\n"
    "         tools.delegate('Count the rows of the second table, please.')]\n"
    "done = []\n"
    "while len(done) < 2:\n"
    "    done += [line for line in tools.wait(timeout=120).splitlines() if '(reply)' in line]\n"
    "tools.answer(' / '.join(sorted(line.split(': ', 1)[1] for line in done)))\n"
)


async def test_two_teammates_work_at_once_and_the_actor_waits_for_both(core_world, monkeypatch, tmp_path):  # noqa: F811
    def mate(call):
        return _code("tools.answer('3 rows')" if "the first table" in _task(call) else "tools.answer('5 rows')")

    model = ScriptedModel(actor=[_code(FAN_OUT)], helper=Always(mate))
    result, pool = await _run(model, monkeypatch, tmp_path)
    assert result == "3 rows / 5 rows"
    assert model.kinds().count("actor") == 1  # waiting cost no model call
    assert set(_spawners(pool).values()) == {pool.record.root}
    assert len(_spawners(pool)) == 2


async def test_a_teammate_starts_a_teammate_of_its_own(core_world, monkeypatch, tmp_path):  # noqa: F811
    def mate(call):
        if "another helper for this request" in _task(call):
            return reply("nested done")
        if any(m.get("role") == "tool" for m in call.messages):
            return reply("started one")
        return _code("print('started', tools.delegate('another helper for this request, please.'))")

    model = ScriptedModel(
        actor=[
            _code("print(tools.delegate('Start a teammate of your own, please.'))\nprint(tools.wait(timeout=120))"),
            reply("done"),
        ],
        helper=Always(mate),
    )
    result, pool = await _run(model, monkeypatch, tmp_path)
    assert result == "done"
    spawners = _spawners(pool)
    (first,) = [name for name, by in spawners.items() if by == pool.record.root]
    # any agent may start agents: the record shows the teammate started the second
    assert [name for name, by in spawners.items() if by == first], spawners


async def test_agents_waiting_on_each_other_are_released_by_their_timeouts(
    core_world,  # noqa: F811
    monkeypatch,
    tmp_path,
):
    model = ScriptedModel(
        actor=[
            _code("tools.delegate('Wait for the main agent to say go, then reply.')\nprint(tools.wait(timeout=2))"),
            reply("gave up waiting"),
        ],
        helper=[_code("print(tools.wait(timeout=2))"), reply("no go")],
    )
    result, pool = await _run(model, monkeypatch, tmp_path)
    assert result == "gave up waiting"
    assert all(task.done() for task in pool._tasks.values())
