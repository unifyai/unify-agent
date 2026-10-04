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
