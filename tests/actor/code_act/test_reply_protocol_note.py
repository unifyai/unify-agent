"""Symbolic: ``UNIFY_REPLY_PROTOCOL_NOTE``: reply-format actions are the actor's own reply.

On ARC LOW (first 97 episodes, 2 Oct) the actor, told to request a
demonstration by replying ``{"action": "request_demos"}``, delegated the
request to sub-actors, which called functions that do not exist
(``request_demonstration``, ``arc_request_demo``, ...) and reported them
unavailable; about half the time the actor then submitted a guess. 35, 10
and 13 episodes (baseline, A, B) carried such calls. The note states, for
any requester-defined reply protocol, that such an action is only ever the
actor's final reply. Sub-actors are CodeActActors and build the same prompt.
"""

from __future__ import annotations

import re

import pytest

from unify.actor import prompt_builders as pb
from unify.settings import SETTINGS


def _prompt() -> str:
    from unify.actor.code_act_actor import CodeActActor

    actor = CodeActActor()
    return pb.build_code_act_prompt(
        environments={},
        tools=dict(actor.get_tools("act")),
        can_store=True,
    )


def test_on_the_note_follows_the_execution_rules(monkeypatch):
    monkeypatch.setattr(SETTINGS, "UNIFY_REPLY_PROTOCOL_NOTE", True)
    prompt = _prompt()
    assert pb._REPLY_PROTOCOL_NOTE in prompt
    assert (
        prompt.index("### Execution Rules")
        < prompt.index("### Actions Taken By Replying")
        < prompt.index(pb._INCREMENTAL_EXECUTION[:40])
    )
    # Static text: the cached prefix is the same for every session.
    assert _prompt().split("### Current Time")[0] == prompt.split("### Current Time")[0]


def test_off_the_prompt_is_as_shipped(monkeypatch):
    monkeypatch.setattr(SETTINGS, "UNIFY_REPLY_PROTOCOL_NOTE", False)
    prompt = _prompt()
    assert "### Actions Taken By Replying" not in prompt


def test_the_note_names_no_benchmark_or_action():
    words = set(re.findall(r"[a-z]+", pb._REPLY_PROTOCOL_NOTE.lower()))
    for word in ("arc", "demo", "demos", "grid", "submit", "appworld", "scienceworld"):
        assert word not in words


@pytest.mark.parametrize("value, expected", [("1", True), ("0", False), ("", False)])
def test_the_setting_parses_booleans(value, expected):
    from unify.settings import ProductionSettings

    assert (
        ProductionSettings(UNIFY_REPLY_PROTOCOL_NOTE=value).UNIFY_REPLY_PROTOCOL_NOTE
        is expected
    )
