"""Symbolic: ``UNIFY_REPLY_WORDING`` words the reply rule so a reply may carry reasoning.

On the long lean-all Continual-ARC LOW run (``arc-pm-lean8958-h-low-ws0``,
first attempt, 3,803 calls) 394 of 754 ``execute_code`` cells (52%) did
nothing (``pass`` 190, comment-only 88, ``print('')`` 61, empty 32) and 335
of them carried a ``thought`` announcing the next protocol action, which the
model then took in text on the next call: two calls per decision, 10.3% of
the run's USD. Unify @ main had 7-8% such cells. The jump tracks the lean
profile's "Your answer is your final reply: a message without a tool call"
(controls carrying only the reply-protocol note stay at 16-19%), and design
B's cell description produced more of them than the legacy tool on the same
replayed requests (11 against 5).

``reason`` states, once, where the prompt states the reply rule, that a reply
may carry reasoning before its answer or action and that thinking needs no
cell; the sentences around it are made consistent with it, and so are the
reply-protocol note and design B's cell description. ``action_last`` changes
only the note. Off, the prompt is as shipped (and the switches-off golden
pins it).
"""

from __future__ import annotations

import pytest

from unify.actor import core_surface, notebook_cells
from unify.actor import prompt_builders as pb
from unify.settings import SETTINGS

REASON = (
    "You may reason in your reply before its final answer or action; you do "
    "not need a cell to think or to announce a step."
)
NOTE_ACTION_LAST = (
    "Requester-defined actions can only be taken by your reply, not by code, "
    "search or sub-agents. End your turn with a reply whose last line is the "
    "action; you may reason before it."
)
SHIPPED_NOTE_PASSAGE = "to take one, end your turn with exactly that reply."

PROFILES = ("", "lean")
SURFACES = ("json", "core")


def _flat(text: str) -> str:
    return " ".join(text.split())


@pytest.fixture
def actor_tools():
    from unify.actor.code_act_actor import CodeActActor

    actor = CodeActActor()
    return actor, dict(actor.get_tools("act"))


def _prompt(actor_tools, surface: str) -> str:
    actor, tools = actor_tools
    if surface == "core":
        return pb.build_code_act_prompt(
            environments=actor.environments,
            tools={"execute_code": tools["execute_code"]},
            can_store=True,
            core=core_surface.PromptSurface(),
        )
    return pb.build_code_act_prompt(
        environments=actor.environments,
        tools=tools,
        can_store=True,
    )


@pytest.fixture
def wording(monkeypatch):
    def _set(value: str, *, profile: str = "", note: bool = False) -> None:
        monkeypatch.setattr(SETTINGS, "UNIFY_REPLY_WORDING", value)
        monkeypatch.setattr(SETTINGS, "UNIFY_PROMPT_PROFILE", profile)
        monkeypatch.setattr(SETTINGS, "UNIFY_REPLY_PROTOCOL_NOTE", note)

    return _set


def test_the_switch_is_validated():
    from unify.settings import ProductionSettings

    for value in ("", "reason", "action_last"):
        assert ProductionSettings(UNIFY_REPLY_WORDING=value).UNIFY_REPLY_WORDING == (
            value
        )
    with pytest.raises(ValueError):
        ProductionSettings(UNIFY_REPLY_WORDING="think")


@pytest.mark.parametrize("profile", PROFILES)
@pytest.mark.parametrize("surface", SURFACES)
@pytest.mark.parametrize("note", [False, True])
def test_off_nothing_new_is_said(actor_tools, wording, profile, surface, note):
    wording("", profile=profile, note=note)
    prompt = _flat(_prompt(actor_tools, surface))
    assert REASON not in prompt
    assert NOTE_ACTION_LAST not in prompt
    if note:
        assert SHIPPED_NOTE_PASSAGE in prompt


@pytest.mark.parametrize("profile", PROFILES)
@pytest.mark.parametrize("surface", SURFACES)
def test_reason_is_said_once_where_the_reply_rule_is(
    actor_tools,
    wording,
    profile,
    surface,
):
    wording("reason", profile=profile)
    prompt = _flat(_prompt(actor_tools, surface))
    assert prompt.count(REASON) == 1
    if profile == "lean":
        role = _flat(prompt[: prompt.index("### Tools")])
        assert (
            "Your answer is your final reply: a message without a tool call. " + REASON
        ) in role
        # The requester's format governs the answer or action, not the reasoning.
        assert "the answer or action in each reply follows that format exactly" in role
        assert "each reply follows that format exactly" not in role.replace(
            "the answer or action in each reply follows that format exactly",
            "",
        )
    else:
        assert (
            "you **MUST** provide the final answer as a tool-less assistant "
            "message — never via a tool call. " + REASON
        ) in prompt
        assert "the final answer directly as a tool-less" not in prompt


@pytest.mark.parametrize("profile", PROFILES)
@pytest.mark.parametrize("surface", SURFACES)
@pytest.mark.parametrize("value", ["reason", "action_last"])
def test_the_note_lets_a_reply_reason_before_its_action(
    actor_tools,
    wording,
    profile,
    surface,
    value,
):
    wording(value, profile=profile, note=True)
    prompt = _flat(_prompt(actor_tools, surface))
    assert prompt.count(NOTE_ACTION_LAST) == 1
    assert "exactly that reply" not in prompt
    assert "### Actions Taken By Replying" in prompt


@pytest.mark.parametrize("profile", PROFILES)
@pytest.mark.parametrize("surface", SURFACES)
def test_action_last_changes_only_the_note(actor_tools, wording, profile, surface):
    wording("", profile=profile, note=True)
    shipped = _prompt(actor_tools, surface)
    wording("action_last", profile=profile, note=True)
    changed = _prompt(actor_tools, surface)
    note = pb._REPLY_PROTOCOL_NOTE
    assert note in shipped and note not in changed
    head, tail = shipped.split(note)
    assert changed.startswith(head) and changed.endswith(tail)
    # Without the note it is inert.
    wording("", profile=profile)
    off = _prompt(actor_tools, surface)
    wording("action_last", profile=profile)
    assert _prompt(actor_tools, surface) == off


@pytest.mark.parametrize("structured", [False, True])
def test_the_notebook_cell_is_not_for_thinking(monkeypatch, structured):
    caps = notebook_cells.Capabilities()
    monkeypatch.setattr(SETTINGS, "UNIFY_REPLY_WORDING", "")
    shipped = notebook_cells.describe(caps, steering=False, structured=structured)
    monkeypatch.setattr(SETTINGS, "UNIFY_REPLY_WORDING", "reason")
    reason = notebook_cells.describe(caps, steering=False, structured=structured)
    head = shipped.rsplit("\n\n", 1)[0]
    assert reason.startswith(head + "\n\n")
    closing = reason.rsplit("\n\n", 1)[1]
    assert closing.startswith(
        "A cell is for computing, not for thinking or announcing a step: you "
        "can reason in your reply. ",
    )
    assert closing.endswith(
        (
            "You answer by calling `final_response`."
            if structured
            else "You answer, and take any action the requester defines, by replying."
        ),
    )
    # action_last changes only the note.
    monkeypatch.setattr(SETTINGS, "UNIFY_REPLY_WORDING", "action_last")
    assert notebook_cells.describe(caps, steering=False, structured=structured) == (
        shipped
    )
