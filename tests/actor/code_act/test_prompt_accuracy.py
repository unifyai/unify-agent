"""Symbolic: ``UNIFY_PROMPT_ACCURACY``: the actor is told only what its session has.

Captured requests of the 4 Oct ARC LOW runs carry statements that are false
for the session that receives them. A persistent session is told a review
runs "after each completed turn" and that its results "arrive in the
conversation as bracketed background notes", though turn reviews have been
off by default since 22 Sep (``UNIFY_TURN_STORAGE_REVIEWS``) and the review
at the session's end reports only to the CLI. Each fix is tested on the
prompt the builder renders and on the request a scripted ``act()`` sends.
"""

from __future__ import annotations

import asyncio
import json

import pytest

from tests import cache_discipline_helpers as h
from unify.actor import prompt_builders as pb
from unify.settings import SETTINGS

_PER_TURN = "**after each completed turn**"
_NOTES = "bracketed background notes"
_SESSION_END = "**when the session ends**"


def _flat(text: str) -> str:
    """*text* with every run of whitespace one space, so wrapping does not matter."""
    return " ".join(text.split())


def _prompt(*, persist: bool, turn_reviews: bool) -> str:
    from unify.actor.code_act_actor import CodeActActor

    actor = CodeActActor()
    return pb.build_code_act_prompt(
        environments={},
        tools=dict(actor.get_tools("act")),
        can_store=True,
        persist=persist,
        turn_reviews=turn_reviews,
    )


async def _first_request(*, persist: bool, **act_kwargs) -> dict:
    """The first request a scripted ``act()`` sends."""
    from unify.actor.code_act_actor import CodeActActor

    actor = CodeActActor()
    try:
        with h.scripted(h.ACTOR_REPLIES) as provider:
            handle = await actor.act(
                "List the files in the workspace.",
                persist=persist,
                **act_kwargs,
            )
            if persist:
                for _ in range(200):
                    if provider.requests:
                        break
                    await asyncio.sleep(0.05)
                await handle.stop("done")
            await asyncio.wait_for(handle.result(), 60)
    finally:
        await actor.close()
    return provider.requests[0]


def _system_text(request: dict) -> str:
    return "\n\n".join(
        m["content"]
        for m in request["messages"]
        if m["role"] == "system" and isinstance(m["content"], str)
    )


# ── the storage schedule (D14) ──────────────────────────────────────────


@pytest.mark.parametrize("framing", ["", "unified"])
def test_on_a_persistent_session_without_turn_reviews_is_told_the_session_end(
    monkeypatch,
    framing,
):
    monkeypatch.setattr(SETTINGS, "UNIFY_PROMPT_ACCURACY", True)
    monkeypatch.setattr(SETTINGS, "UNIFY_REVIEW_FRAMING", framing)
    prompt = _flat(_prompt(persist=True, turn_reviews=False))
    assert _SESSION_END in prompt
    assert _PER_TURN not in prompt
    assert _NOTES not in prompt
    assert "result is added to this conversation" in prompt
    # The rest of the notice is kept.
    assert "**Before compression**" in prompt
    assert "**Direct writes vs trajectory storage**" in prompt


@pytest.mark.parametrize("framing", ["", "unified"])
def test_on_a_session_with_turn_reviews_keeps_the_per_turn_notice(
    monkeypatch,
    framing,
):
    monkeypatch.setattr(SETTINGS, "UNIFY_REVIEW_FRAMING", framing)
    monkeypatch.setattr(SETTINGS, "UNIFY_PROMPT_ACCURACY", False)
    shipped = _prompt(persist=True, turn_reviews=True)
    monkeypatch.setattr(SETTINGS, "UNIFY_PROMPT_ACCURACY", True)
    assert _prompt(persist=True, turn_reviews=True) == shipped
    assert _PER_TURN in shipped


def test_on_a_one_shot_session_keeps_its_notice(monkeypatch):
    monkeypatch.setattr(SETTINGS, "UNIFY_PROMPT_ACCURACY", False)
    shipped = _prompt(persist=False, turn_reviews=False)
    monkeypatch.setattr(SETTINGS, "UNIFY_PROMPT_ACCURACY", True)
    assert _prompt(persist=False, turn_reviews=False) == shipped


def test_off_the_persistent_notice_is_as_shipped(monkeypatch):
    monkeypatch.setattr(SETTINGS, "UNIFY_PROMPT_ACCURACY", False)
    prompt = _prompt(persist=True, turn_reviews=False)
    assert pb._STORAGE_SESSION_NOTICE in prompt


@pytest.mark.asyncio
@pytest.mark.timeout(180)
@pytest.mark.parametrize("turn_reviews", [False, True])
async def test_act_describes_the_schedule_the_session_gets(monkeypatch, turn_reviews):
    monkeypatch.setattr(SETTINGS, "UNIFY_PROMPT_ACCURACY", True)
    monkeypatch.setattr(SETTINGS, "UNIFY_TURN_STORAGE_REVIEWS", turn_reviews)
    system = _flat(_system_text(await _first_request(persist=True)))
    assert (_PER_TURN in system) is turn_reviews
    assert (_SESSION_END in system) is not turn_reviews


# ── the steering docs name `steer` (D20) ────────────────────────────────


def _tool_descriptions() -> dict[str, str]:
    from unify.actor.code_act_actor import CodeActActor
    from unify.common.llm_helpers import method_to_schema

    actor = CodeActActor()
    out = {}
    for name in ("execute_code", "execute_function"):
        tool = actor.get_tools("act")[name]
        fn = getattr(tool, "fn", tool)
        out[name] = method_to_schema(fn, name)["function"]["description"]
    return out


def test_on_the_steering_docs_name_the_steer_tool(monkeypatch):
    monkeypatch.setattr(SETTINGS, "UNIFY_PROMPT_ACCURACY", True)
    for name, text in _tool_descriptions().items():
        assert "stop_execute_" not in text, name
        assert 'steer(call_id=<id>, action="stop")' in text, name
        assert 'action="interject"' in text, name


def test_off_the_steering_docs_are_as_shipped(monkeypatch):
    monkeypatch.setattr(SETTINGS, "UNIFY_PROMPT_ACCURACY", False)
    docs = _tool_descriptions()
    assert "``stop_execute_code_<call_id>``" in docs["execute_code"]
    assert "``stop_execute_function_<call_id>``" in docs["execute_function"]


def test_a_corrected_actor_leaves_the_next_actors_docs_alone(monkeypatch):
    monkeypatch.setattr(SETTINGS, "UNIFY_PROMPT_ACCURACY", True)
    _tool_descriptions()
    monkeypatch.setattr(SETTINGS, "UNIFY_PROMPT_ACCURACY", False)
    assert "``stop_execute_code_<call_id>``" in _tool_descriptions()["execute_code"]


# ── no parent conversation for a loop without a parent (D23) ───────────────

_PARENT = "## Parent Chat Context"


async def _loop_request(*, lineage=None, parent_chat_context=None) -> dict:
    """The first request of a scripted loop, started under *lineage*."""
    from unify.common._async_tool.loop_config import TOOL_LOOP_LINEAGE
    from unify.common.async_tool_loop import start_async_tool_loop

    token = TOOL_LOOP_LINEAGE.set(list(lineage or []))
    try:
        with h.scripted([h.completion(content="done")]) as provider:
            handle = start_async_tool_loop(
                h.new_client(),
                "Say done.",
                {},
                log_steps=False,
                timeout=30,
                parent_chat_context=parent_chat_context,
            )
            await asyncio.wait_for(handle.result(), 30)
    finally:
        TOOL_LOOP_LINEAGE.reset(token)
    return provider.requests[0]


@pytest.mark.asyncio
@pytest.mark.timeout(180)
async def test_on_a_top_level_actor_is_not_told_of_a_parent(monkeypatch):
    monkeypatch.setattr(SETTINGS, "UNIFY_PROMPT_ACCURACY", True)
    request = await _first_request(persist=False)
    assert _PARENT not in _system_text(request)
    assert "outer_user" not in json.dumps(request["messages"])


@pytest.mark.asyncio
@pytest.mark.timeout(180)
async def test_off_a_top_level_actor_gets_the_shipped_section(monkeypatch):
    monkeypatch.setattr(SETTINGS, "UNIFY_PROMPT_ACCURACY", False)
    assert _PARENT in _system_text(await _first_request(persist=False))


@pytest.mark.asyncio
@pytest.mark.timeout(60)
@pytest.mark.parametrize(
    "lineage, context, expected",
    [
        (None, None, False),  # no parent loop, no context: top level
        (["Outer.act(ab12)"], None, True),  # started inside another loop
        (None, [], True),  # a caller handed it (empty) parent context
        (None, [{"role": "user", "content": "hi"}], True),
    ],
)
async def test_on_the_section_follows_whether_a_parent_exists(
    monkeypatch,
    lineage,
    context,
    expected,
):
    monkeypatch.setattr(SETTINGS, "UNIFY_PROMPT_ACCURACY", True)
    request = await _loop_request(lineage=lineage, parent_chat_context=context)
    assert (_PARENT in _system_text(request)) is expected


@pytest.mark.asyncio
@pytest.mark.timeout(60)
async def test_off_a_loop_without_a_parent_still_gets_the_section(monkeypatch):
    monkeypatch.setattr(SETTINGS, "UNIFY_PROMPT_ACCURACY", False)
    assert _PARENT in _system_text(await _loop_request())


# ── clarification only where it exists (D33, D31) ───────────────────────


def _clarify_prompt(*, can_clarify: bool) -> str:
    from unify.actor.code_act_actor import CodeActActor

    actor = CodeActActor()
    return pb.build_code_act_prompt(
        environments={},
        tools=dict(actor.get_tools("act")),
        can_store=True,
        can_clarify=can_clarify,
    )


def test_on_without_the_tool_the_rules_never_mention_clarification(monkeypatch):
    monkeypatch.setattr(SETTINGS, "UNIFY_PROMPT_ACCURACY", True)
    prompt = _flat(_clarify_prompt(can_clarify=False))
    assert "request_clarification" not in prompt
    assert "request clarification" not in prompt
    assert "Proactive clarification" not in prompt
    # What the rules say about evidence and batches is kept.
    assert "If the evidence contradicts the result, fix and re-run." in prompt
    assert "process a 5–10 item batch, and review it before scaling;" in prompt
    assert "7. **Data provenance" in prompt


def test_on_with_the_tool_the_rules_are_as_shipped(monkeypatch):
    monkeypatch.setattr(SETTINGS, "UNIFY_PROMPT_ACCURACY", False)
    shipped = _clarify_prompt(can_clarify=True)
    monkeypatch.setattr(SETTINGS, "UNIFY_PROMPT_ACCURACY", True)
    assert _clarify_prompt(can_clarify=True) == shipped


def test_off_the_rules_mention_clarification_as_shipped(monkeypatch):
    monkeypatch.setattr(SETTINGS, "UNIFY_PROMPT_ACCURACY", False)
    prompt = _clarify_prompt(can_clarify=False)
    assert pb._EXECUTION_RULES in prompt
    assert pb._INCREMENTAL_EXECUTION in prompt


@pytest.mark.asyncio
@pytest.mark.timeout(180)
async def test_act_without_clarification_sends_no_clarification_text(monkeypatch):
    monkeypatch.setattr(SETTINGS, "UNIFY_PROMPT_ACCURACY", True)
    request = await _first_request(persist=False, clarification_enabled=False)
    system = _flat(_system_text(request))
    assert "request_clarification" not in system
    assert "request clarification" not in system
    names = [t["function"]["name"] for t in request["tools"] or []]
    assert "request_clarification" not in names


class _FakeInnerActor:
    def __init__(self):
        self.kwargs = None

    async def act(self, request, **kwargs):
        self.kwargs = kwargs

        class _Handle:
            async def result(self):
                return "done"

        return _Handle()

    async def close(self):
        pass


async def _sub_actor_clarification(monkeypatch, parent_can_clarify) -> bool:
    from unify.actor.environments import actor as actor_env
    from unify.actor.execution import _CAN_CLARIFY

    fake = _FakeInnerActor()
    monkeypatch.setattr(
        actor_env,
        "_build_inner_actor",
        lambda **_kw: (fake, None),
    )
    token = _CAN_CLARIFY.set(parent_can_clarify)
    try:
        handle = await actor_env._ActorRunner().act("a sub-task")
        await handle.result()
    finally:
        _CAN_CLARIFY.reset(token)
    return fake.kwargs["clarification_enabled"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "accuracy, parent, expected",
    [
        (True, False, False),  # the parent cannot ask: neither can its sub-actor
        (True, True, True),
        (True, None, True),  # outside any actor: as shipped
        (False, False, True),  # off: always offered, as shipped
    ],
)
async def test_a_sub_actor_asks_only_when_its_parent_can(
    monkeypatch,
    accuracy,
    parent,
    expected,
):
    monkeypatch.setattr(SETTINGS, "UNIFY_PROMPT_ACCURACY", accuracy)
    assert await _sub_actor_clarification(monkeypatch, parent) is expected


@pytest.mark.asyncio
@pytest.mark.timeout(180)
@pytest.mark.parametrize("clarify", [False, True])
async def test_act_tells_its_sandbox_whether_it_can_ask(monkeypatch, clarify):
    """The value a sub-actor started from this actor's sandbox inherits."""
    from unify.actor.code_act_actor import CodeActActor
    from unify.actor.execution import _CAN_CLARIFY

    monkeypatch.setattr(SETTINGS, "UNIFY_PROMPT_ACCURACY", True)
    actor = CodeActActor()
    try:
        with h.scripted(h.ACTOR_REPLIES):
            handle = await actor.act(
                "List the files in the workspace.",
                persist=False,
                can_store=False,
                clarification_enabled=clarify,
            )
            seen = _CAN_CLARIFY.get(None)
            await asyncio.wait_for(handle.result(), 60)
    finally:
        await actor.close()
    assert seen is clarify


@pytest.mark.parametrize("value, expected", [("1", True), ("0", False), ("", False)])
def test_the_setting_parses_booleans(value, expected):
    from unify.settings import ProductionSettings

    assert (
        ProductionSettings(UNIFY_PROMPT_ACCURACY=value).UNIFY_PROMPT_ACCURACY
        is expected
    )


def test_the_default_is_off():
    from unify.settings import ProductionSettings

    assert ProductionSettings.model_fields["UNIFY_PROMPT_ACCURACY"].default is False


def test_no_fixed_text_names_a_benchmark():
    import re

    text = json.dumps(
        [pb._STORAGE_SESSION_END_NOTICE, pb._STORAGE_SESSION_END_NOTICE_UNIFIED],
    ).lower()
    words = set(re.findall(r"[a-z]+", text))
    for word in ("arc", "appworld", "scienceworld", "crafter", "grid", "benchmark"):
        assert word not in words
