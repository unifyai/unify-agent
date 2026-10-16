"""``UNIFY_HANDBACK_DECISION=required``: the held answer waits for a decision (unify/actor/drafts.py).

With the decision required, the question's turn goes on through the model's own cells (looking the tool up first) until
``tools.decide`` or a text reply, at most 3 model turns with cells per ask. A turn that ended without deciding (a text
reply, or its cells used up) gets the question once more, with one added line; after the second ask's turn the answer
goes back unchanged, recorded as ``none``. A decision at either ask releases the answer then. With ``on`` nothing changes: one
cell, one ask, and the rows carry no ``asks``.

The model is scripted (no key, no network). The end-to-end tests run the minimal arm's setting (memory off, the library
off, the tools called from code, a persistent interpreter), cells in the real sandboxed worker.
"""

from __future__ import annotations

import json
import os

import pytest

from tests.actor.code_act.core_world import core_world, world  # noqa: F401 (fixtures)
from tests.actor.code_act.sandbox_world import needs_bwrap
from tests.actor.code_act.test_in_code_tools_wiring import _act, _Collect, _why, code
from tests.scripted_model import ScriptedModel, reply
from unify.actor import adaptive_policy, drafts, in_code_tools
from unify.settings import SETTINGS

DEFINE_AND_ANSWER = "def triple(v):\n    return v * 3\ntools.answer('done')\n"


class _Events(list):
    def append(self, row):  # noqa: D401 - the EventFile's one method
        super().append(dict(row))


class _Settings:
    def __init__(self, **values):
        self.__dict__.update(values)


def _session(tmp_path, required=True, events=None):
    root = tmp_path / "workspace"
    root.mkdir(parents=True, exist_ok=True)
    fd = os.open(root, os.O_RDONLY | os.O_DIRECTORY)
    session = drafts.SessionDrafts(root_fd=fd, events=events if events is not None else _Events(), started=1_760_000_000.0)
    session.required = required
    return session, fd


# --- the setting -----------------------------------------------------------------------------------------------------


def test_the_decision_takes_on_or_required_and_the_policy_refuses_either():
    from unify.settings import ProductionSettings

    assert [drafts.parse_decision(v) for v in (None, "", "off", " ON ", "Required")] == ["", "", "", "on", "required"]
    with pytest.raises(ValueError, match="'on' or 'required'"):
        drafts.parse_decision("always")
    assert ProductionSettings(UNIFY_HANDBACK_DECISION="required").UNIFY_HANDBACK_DECISION == "required"
    with pytest.raises(ValueError, match="UNIFY_HANDBACK_DECISION"):
        ProductionSettings(UNIFY_HANDBACK_DECISION="cells")
    assert drafts.decision_enabled(_Settings(UNIFY_HANDBACK_DECISION="required"))
    assert drafts.refusal(_Settings(UNIFY_DRAFTS="on", UNIFY_HANDBACK_DECISION="required", UNIFY_TOOLS_IN_CODE="on")) is None
    assert "UNIFY_DRAFTS" in (drafts.refusal(_Settings(UNIFY_DRAFTS="", UNIFY_HANDBACK_DECISION="required",
                                                      UNIFY_TOOLS_IN_CODE="on")) or "")  # fmt: skip
    policy = _Settings(UNIFY_ADAPTIVE_POLICY="on", UNIFY_TOOLS_IN_CODE="on", UNIFY_MEMORY="", UNIFY_DRAFTS="",
                       UNIFY_HANDBACK_DECISION="required", UNIFY_CONSEQUENCES="")  # fmt: skip
    assert "UNIFY_HANDBACK_DECISION" in (adaptive_policy.refusal(policy) or "")


# --- the session -----------------------------------------------------------------------------------------------------


def _use_up_cells(session):
    """Three model turns with cells and no decision: the third ends the turn (the bound)."""
    assert [session.turn_continues() for _ in range(drafts.MAX_CELLS_PER_ASK)] == [True, True, False]


def test_a_turn_goes_on_through_its_cells_then_the_question_is_asked_again_then_closes_as_none(tmp_path):
    events = _Events()
    session, fd = _session(tmp_path, events=events)
    try:
        session.take({"defs": {"triple": "def triple(v):\n    return v * 3"}, "imports": []})
        first = session.question(["triple"])
        _use_up_cells(session)
        again = session.after_question()
        assert again == first + "\n" + drafts.REASK_LINE
        assert drafts.REASK_LINE == "A keep or skip is needed before your answer goes back."
        _use_up_cells(session)  # the second ask gets its own three
        assert session.after_question() is None  # the asks used up: the answer goes back
        assert session.after_question() is None  # nothing open any more
        rows = [r for r in events if r["type"] in ("decision", "decision_asked_again")]
        assert rows == [
            {"type": "decision_asked_again", "asked": ["triple"], "asks": 2},
            {"type": "decision", "choice": "none", "names": [], "why": "", "asked": ["triple"], "asks": 2},
        ]
    finally:
        os.close(fd)


def test_a_text_reply_without_a_decision_is_asked_again_then_released(tmp_path):
    events = _Events()
    session, fd = _session(tmp_path, events=events)
    try:
        session.take({"defs": {"triple": "def triple(v):\n    return v * 3"}, "imports": []})
        first = session.question(["triple"])
        assert session.turn_continues() is True  # one cell (a help lookup, say), then a text reply
        assert session.after_question() == first + "\n" + drafts.REASK_LINE
        assert session.after_question() is None  # a second text reply: the answer goes back
        assert [r for r in events if r["type"] in ("decision", "decision_asked_again")] == [
            {"type": "decision_asked_again", "asked": ["triple"], "asks": 2},
            {"type": "decision", "choice": "none", "names": [], "why": "", "asked": ["triple"], "asks": 2},
        ]
    finally:
        os.close(fd)


def test_a_decision_ends_the_turn_at_either_ask_and_records_its_asks(tmp_path):
    events = _Events()
    session, fd = _session(tmp_path, events=events)
    try:
        session.take({"defs": {"triple": "def triple(v):\n    return v * 3"}, "imports": []})
        session.question(["triple"])
        _use_up_cells(session)
        assert session.after_question() is not None
        assert session.turn_continues() is True  # the second ask: a help cell
        assert session.decide("skip", None, None)[0] is None
        assert session.turn_continues() is False  # decided: the turn ends
        assert session.after_question() is None
        decisions = [r for r in events if r["type"] == "decision"]
        assert decisions == [{"type": "decision", "choice": "skip", "names": [], "why": "", "asked": ["triple"], "asks": 2}]
    finally:
        os.close(fd)


def test_a_cells_file_question_is_asked_again_with_its_own_text(tmp_path):
    events = _Events()
    root = tmp_path / "workspace"
    root.mkdir(parents=True)
    fd = os.open(root, os.O_RDONLY | os.O_DIRECTORY)
    session = drafts.SessionDrafts(root_fd=fd, events=events, mode=drafts.CELLS, started=1_760_000_000.0)
    session.required = True
    try:
        session.note_cell(0, "rows = [1, 2]", True)
        rel = session.take_cells()
        first = session.question_file(rel)
        _use_up_cells(session)
        assert session.after_question() == first + "\n" + drafts.REASK_LINE
        assert session.decide("keep", None, "the rows")[0] is None
        assert session.after_question() is None
        assert [r["asks"] for r in events if r["type"] == "decision"] == [2]
    finally:
        os.close(fd)


def test_with_on_a_question_is_asked_once_and_the_rows_are_unchanged(tmp_path):
    events = _Events()
    session, fd = _session(tmp_path, required=False, events=events)
    try:
        session.take({"defs": {"triple": "def triple(v):\n    return v * 3"}, "imports": []})
        session.question(["triple"])
        assert session.turn_continues() is False  # one cell only, as before
        assert session.after_question() is None
        assert [r for r in events if r["type"] == "decision"] == [
            {"type": "decision", "choice": "none", "names": [], "why": "", "asked": ["triple"]},
        ]
    finally:
        os.close(fd)


# --- end to end ------------------------------------------------------------------------------------------------------


@pytest.fixture
def required(core_world, monkeypatch):  # noqa: F811
    """The minimal arm's setting with the drafts on and the decision required; every in-code session opened."""
    monkeypatch.setattr(SETTINGS, "UNIFY_MEMORY", "")
    monkeypatch.setattr(SETTINGS, "UNIFY_LIBRARY", "off")
    monkeypatch.setattr(SETTINGS, "UNIFY_TOOLS_IN_CODE", "on")
    monkeypatch.setattr(SETTINGS, "UNIFY_CELL_STATE", "persistent")
    monkeypatch.setattr(SETTINGS, "UNIFY_DRAFTS", "on")
    monkeypatch.setattr(SETTINGS, "UNIFY_HANDBACK_DECISION", "required")
    collect = _Collect()
    monkeypatch.setattr(in_code_tools, "SESSIONS", collect)
    yield {"workspace": core_world["workspace"], "opened": collect.opened}


def _rows(session) -> list[dict]:
    path = in_code_tools.request_folder(session.request_id) / "tool-events.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines()]


def _decisions(session) -> list[dict]:
    return [r for r in _rows(session) if r.get("type") in ("decision", "decision_asked_again")]


HELP = 'print(tools.help("decide"))\n'


@needs_bwrap
@pytest.mark.asyncio
@pytest.mark.timeout(400)
async def test_a_help_lookup_then_a_decision_within_the_bound(required):
    model = ScriptedModel(
        actor=[reply(calls=[code(DEFINE_AND_ANSWER)]), reply(calls=[code(HELP)]), reply(calls=[code('tools.decide("skip")\n')])],
    )
    assert await _act(model, "Define and answer.") == "done", _why(model)
    assert model.kinds() == ["actor", "actor", "actor"]
    rows = _decisions(required["opened"][0])
    assert [(r["type"], r["choice"], r["asks"]) for r in rows] == [("decision", "skip", 1)]


@needs_bwrap
@pytest.mark.asyncio
@pytest.mark.timeout(400)
async def test_three_cells_without_a_decision_then_the_question_again_then_the_answer_goes_back(required):
    model = ScriptedModel(actor=[reply(calls=[code(DEFINE_AND_ANSWER)]), *(reply(calls=[code(HELP)]) for _ in range(6))])
    assert await _act(model, "Define and answer.") == "done", _why(model)
    assert model.kinds() == ["actor"] * 7
    assert "A keep or skip is needed before your answer goes back." in json.dumps(model.calls[4].messages, default=str)
    assert "A keep or skip is needed" not in json.dumps(model.calls[3].messages, default=str)
    rows = _decisions(required["opened"][0])
    assert [(r["type"], r.get("choice"), r["asks"]) for r in rows] == [
        ("decision_asked_again", None, 2), ("decision", "none", 2),
    ]


@needs_bwrap
@pytest.mark.asyncio
@pytest.mark.timeout(400)
async def test_a_text_reply_is_asked_again_then_the_answer_goes_back(required):
    model = ScriptedModel(actor=[reply(calls=[code(DEFINE_AND_ANSWER)]), reply("Keep triple."), reply("Still: keep it.")])
    assert await _act(model, "Define and answer.") == "done", _why(model)
    assert model.kinds() == ["actor", "actor", "actor"]
    assert "A keep or skip is needed before your answer goes back." in json.dumps(model.calls[2].messages, default=str)
    rows = _decisions(required["opened"][0])
    assert [(r["type"], r.get("choice"), r["asks"]) for r in rows] == [
        ("decision_asked_again", None, 2), ("decision", "none", 2),
    ]


@needs_bwrap
@pytest.mark.asyncio
@pytest.mark.timeout(400)
async def test_a_decision_after_the_question_again(required):
    keep = 'print(tools.decide("keep", names=["triple"], why="times three comes up"))\n'
    model = ScriptedModel(
        actor=[reply(calls=[code(DEFINE_AND_ANSWER)]), *(reply(calls=[code(HELP)]) for _ in range(3)), reply(calls=[code(keep)])],
    )
    assert await _act(model, "Define and answer.") == "done", _why(model)
    assert model.kinds() == ["actor"] * 5
    rows = _decisions(required["opened"][0])
    assert [(r["type"], r.get("choice"), r["asks"]) for r in rows] == [
        ("decision_asked_again", None, 2), ("decision", "keep", 2),
    ]


@needs_bwrap
@pytest.mark.asyncio
@pytest.mark.timeout(400)
async def test_a_decision_at_the_first_ask_needs_no_second(required):
    keep = 'print(tools.decide("keep", names=["triple"], why="times three comes up"))\n'
    model = ScriptedModel(actor=[reply(calls=[code(DEFINE_AND_ANSWER)]), reply(calls=[code(keep)])])
    assert await _act(model, "Define and answer.") == "done", _why(model)
    assert model.kinds() == ["actor", "actor"]
    rows = _decisions(required["opened"][0])
    assert [(r["type"], r["choice"], r["asks"]) for r in rows] == [("decision", "keep", 1)]
