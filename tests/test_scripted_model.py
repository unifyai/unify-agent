"""Symbolic: the scripted model (``tests/scripted_model.py``) routes each
request to its kind's script, and its fixed fragments still match the code.

No model is reached. The two acts at the end run cells in the sandboxed
Python worker (bubblewrap), the default workspace.
"""

from __future__ import annotations

import asyncio
import inspect

import pytest

from tests.scripted_model import (
    Always,
    COMPRESS_TOOLS,
    HELPER_FRAGMENT,
    LAST_WORD_FRAGMENT,
    PROACTIVE_STORAGE_PREFIX,
    STANDALONE_REVIEW_PREFIX,
    ScriptedModel,
    cell,
    kind_of,
    reply,
    scripted,
    scripted_model,  # noqa: F401 (fixture)
)

# ── the fixed fragments still match the code that writes them ───────────────


def _source(module) -> str:
    return inspect.getsource(module)


def test_each_fixed_fragment_is_in_the_code_that_writes_it():
    from unify.actor import code_act_actor
    from unify.agents import binding
    from unify.common._async_tool import loop

    assert HELPER_FRAGMENT in _source(binding)
    assert LAST_WORD_FRAGMENT in _source(loop)
    actor_source = _source(code_act_actor)
    assert STANDALONE_REVIEW_PREFIX in actor_source
    assert PROACTIVE_STORAGE_PREFIX in actor_source
    # The forced turn's tools are functions; their names are the tool names.
    assert loop.compress_context.__name__ in COMPRESS_TOOLS
    assert "async def store_skills(" in actor_source
    assert "store_skills" in COMPRESS_TOOLS


# ── classification of each kind, from the request alone ────────────────────


def _tool(name: str) -> dict:
    return {"type": "function", "function": {"name": name, "parameters": {}}}


def _request(system="You solve tasks.", users=("Do it.",), tools=("execute_code",)):
    messages = [{"role": "system", "content": system}]
    messages += [{"role": "user", "content": u} for u in users]
    return {"messages": messages, "tools": [_tool(t) for t in tools] or None}


def test_each_kind_is_recognised():
    from unify.actor import code_act_actor, review_gate
    from unify.common._async_tool import cache_discipline, context_compression

    review_role = code_act_actor._REVIEW_FORK_ROLE_UNIFIED
    cases = {
        "actor": _request(),
        "review": _request(users=("Do it.", review_role)),
        "gate": _request(system=review_gate.GATE_SYSTEM_PROMPT, tools=()),
        "compression_fork": _request(
            users=("Do it.", cache_discipline.COMPRESSION_FORK_INSTRUCTION),
        ),
        "compress_turn": _request(tools=("compress_context", "store_skills")),
        "compactor": _request(
            system=context_compression.COMPRESSION_PROMPT,
            tools=("update",),
        ),
        "standalone_review": _request(
            system=STANDALONE_REVIEW_PREFIX + " ...",
            tools=("FunctionManager_add_functions",),
        ),
        "proactive_storage": _request(
            system=PROACTIVE_STORAGE_PREFIX + " to store skills.",
            tools=("FunctionManager_add_functions",),
        ),
        "last_word": _request(users=(LAST_WORD_FRAGMENT + " (steps): ...",), tools=()),
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
    model = ScriptedModel(actor=[reply("a1")])
    with pytest.raises(AssertionError, match="'gate' request"):
        await model(**_request(system="x " + _gate_prompt(), tools=()))
    allowing = ScriptedModel(allow={"gate"})
    answer = await allowing(**_request(system=_gate_prompt(), tools=()))
    assert answer.choices[0].message.content == ""


def _gate_prompt() -> str:
    from unify.actor import review_gate

    return review_gate.GATE_SYSTEM_PROMPT


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
    actor = CodeActActor()
    try:
        handle = await actor.act("Compute.", can_store=False)
        assert await asyncio.wait_for(handle.result(), 60) == "42"
    finally:
        await actor.close()
    assert model.kinds() == ["actor", "actor"]
    _, second = model.of("actor")
    results = [m for m in second.messages if m.get("role") == "tool"]
    assert len(results) == 1 and "42" in str(results[0]["content"])
    model.assert_used_up()


@pytest.mark.asyncio
@pytest.mark.timeout(120)
async def test_the_review_after_an_act_does_not_take_the_actors_replies(
    scripted_model,
):
    """The storage review forks the session (same system prompt, same tools):
    it is routed to its own script, so the actor's script is exactly the
    actor's turns. With the library empty the gate is not asked and the
    review runs."""
    from unify.actor.code_act_actor import CodeActActor

    model = scripted_model(
        actor=[reply(calls=[cell("print('done')")]), reply("Finished.")],
        review=Always(reply("Nothing to store.")),
    )
    actor = CodeActActor()
    try:
        handle = await actor.act("Do the task.", can_store=True)
        assert await asyncio.wait_for(handle.result(), 60) == "Finished."
        await asyncio.wait_for(handle._lifecycle_task, 60)
    finally:
        await actor.close()
    assert model.kinds()[:2] == ["actor", "actor"]
    assert model.kinds()[2:] and set(model.kinds()[2:]) == {"review"}
    # The fork carries the session's own system prompt and tools.
    first_review, *_ = model.of("review")
    _, last_actor = model.of("actor")
    assert first_review.tool_names == last_actor.tool_names
    model.assert_used_up()
