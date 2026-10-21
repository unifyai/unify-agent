"""Symbolic: the scripted model (``tests/scripted_model.py``) routes each
request to its kind's script, and its fixed fragments still match the code.

No model is reached. The act at the end runs a cell in the sandboxed Python
worker (bubblewrap), the default workspace.
"""

from __future__ import annotations

import asyncio
import inspect

import pytest

from tests.scripted_model import (
    Always,
    HELPER_FRAGMENT,
    ScriptedModel,
    cell,
    kind_of,
    reply,
    scripted,
    scripted_model,  # noqa: F401 (fixture)
)

# ── the fixed fragments still match the code that writes them ───────────────


def test_each_fixed_fragment_is_in_the_code_that_writes_it():
    from unify.agents import binding

    assert HELPER_FRAGMENT in inspect.getsource(binding)


# ── classification of each kind, from the request alone ────────────────────


def _tool(name: str) -> dict:
    return {"type": "function", "function": {"name": name, "parameters": {}}}


def _request(system="You solve tasks.", users=("Do it.",), tools=("execute_code",)):
    messages = [{"role": "system", "content": system}]
    messages += [{"role": "user", "content": u} for u in users]
    return {"messages": messages, "tools": [_tool(t) for t in tools] or None}


def test_each_kind_is_recognised():
    from unify.common._async_tool import cache_discipline, context_compression

    cases = {
        "actor": _request(),
        "compression_fork": _request(
            users=("Do it.", cache_discipline.COMPRESSION_FORK_INSTRUCTION),
        ),
        "compactor": _request(
            system=context_compression.COMPRESSION_PROMPT,
            tools=("update",),
        ),
        "helper": _request(users=("You are `h1`" + HELPER_FRAGMENT + "...",)),
        "query_llm": _request(system="You are a focused subroutine.", tools=()),
    }
    assert {kind: kind_of(req) for kind, req in cases.items()} == {
        kind: kind for kind in cases
    }


# ── the script ───────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_each_kind_plays_its_own_script_in_order():
    model = ScriptedModel(
        actor=[reply("a1"), reply("a2")],
        query_llm=Always(reply("q")),
    )
    actor_request = _request()
    query = _request(system="You are a focused subroutine.", tools=())
    answers = [
        await model(**actor_request),
        await model(**query),
        await model(**query),
        await model(**actor_request),
    ]
    contents = [a.choices[0].message.content for a in answers]
    assert contents == ["a1", "q", "q", "a2"]
    assert model.kinds() == ["actor", "query_llm", "query_llm", "actor"]
    model.assert_used_up()


@pytest.mark.asyncio
async def test_an_unscripted_kind_fails_naming_it_unless_allowed():
    query = _request(system="You are a focused subroutine.", tools=())
    model = ScriptedModel(actor=[reply("a1")])
    with pytest.raises(AssertionError, match="'query_llm' request"):
        await model(**query)
    allowing = ScriptedModel(allow={"query_llm"})
    answer = await allowing(**query)
    assert answer.choices[0].message.content == ""


@pytest.mark.asyncio
async def test_an_entry_may_raise_or_answer_from_the_request():
    error = RuntimeError("provider down")
    model = ScriptedModel(
        actor=[error, lambda call: reply(f"saw {len(call.messages)} messages")],
    )
    with pytest.raises(RuntimeError, match="provider down"):
        await model(**_request())
    answer = await model(**_request())
    assert answer.choices[0].message.content == "saw 2 messages"


def test_unknown_kinds_are_refused():
    with pytest.raises(ValueError, match="unknown kinds"):
        ScriptedModel(actr=[reply("typo")])


def test_recording_is_off_inside_the_block_and_restored_after():
    from unillm.settings import SETTINGS as unillm_settings

    before = unillm_settings.UNILLM_CACHE
    with scripted(ScriptedModel()):
        assert unillm_settings.UNILLM_CACHE is False
    assert unillm_settings.UNILLM_CACHE == before


# ── through a real act ───────────────────────────────────────────────────────


@pytest.mark.asyncio
@pytest.mark.timeout(120)
async def test_an_act_runs_its_cell_and_replies(scripted_model):
    from unify.actor.code_act_actor import CodeActActor

    model = scripted_model(
        actor=[reply(calls=[cell("x = 41\nprint(x + 1)")]), reply("42")],
    )
    handle = await CodeActActor().act("Compute.")
    assert await asyncio.wait_for(handle.result(), 60) == "42"
    assert model.kinds() == ["actor", "actor"]
    _, second = model.of("actor")
    results = [m for m in second.messages if m.get("role") == "tool"]
    assert len(results) == 1 and "42" in str(results[0]["content"])
    model.assert_used_up()
