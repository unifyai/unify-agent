"""``UNIFY_DRAFTS`` (drafts by default) and ``UNIFY_HANDBACK_DECISION`` (a decision at hand-back),
unify/actor/drafts.py.

With the drafts on, every hand-back writes the functions and classes the actor's cells defined that are new or changed,
unchanged and with the cells' imports, to one file per session under the workspace's ``drafts/``; a later session
imports it in one line, and the import is recorded as a ``draft_use`` row. With the decision on as well, a hand-back
that drafted something gets one more turn with the keep-or-skip question, answered with ``tools.decide``; the answer is
handed back unchanged whatever that turn does, and the decision is a ``decision`` row. Off, nothing of it exists: no
file, no extra turn, no ``decide`` tool.

The model is scripted (no key, no network). The end-to-end tests run the minimal arm's setting (memory off, the
library off, the tools called from code, a persistent interpreter), cells in the real sandboxed worker.
"""

from __future__ import annotations

import json
import os
from types import SimpleNamespace

import pytest

from tests.actor.code_act.core_world import core_world, world  # noqa: F401 (fixtures)
from tests.actor.code_act.sandbox_world import needs_bwrap
from tests.actor.code_act.test_in_code_tools_wiring import _act, _Collect, _system_texts, _tool_results, _why, code
from tests.scripted_model import ScriptedModel, reply
from unify import cli
from unify.actor import drafts, in_code_tools
from unify.common import tool_registry as R
from unify.settings import SETTINGS

DEFINE = (
    "import math\n"
    "def double(v):\n"
    "    return v * 2\n"
    "class Box:\n"
    "    def __init__(self, v):\n"
    "        self.v = math.floor(v)\n"
    "print(double(4), Box(2.5).v)\n"
)
DEFINE_AND_ANSWER = "def triple(v):\n    return v * 3\ntools.answer('done')\n"


class _Events(list):
    def append(self, row):  # noqa: D401 - the EventFile's one method
        super().append(dict(row))


def _session(tmp_path, events=None):
    root = tmp_path / "workspace"
    root.mkdir(exist_ok=True)
    fd = os.open(root, os.O_RDONLY | os.O_DIRECTORY)
    return drafts.SessionDrafts(root_fd=fd, events=events if events is not None else _Events()), root, fd


# --- the settings ----------------------------------------------------------------------------------------------------


def test_the_switches_are_off_by_default_and_the_decision_needs_the_drafts_and_the_tools_in_code():
    assert [drafts.parse(v) for v in (None, "", "off", " ON ")] == ["", "", "", "on"]
    with pytest.raises(ValueError):
        drafts.parse("yes")
    on = {"UNIFY_DRAFTS": "on", "UNIFY_HANDBACK_DECISION": "on", "UNIFY_TOOLS_IN_CODE": "on"}
    assert drafts.refusal(SimpleNamespace(**on)) is None
    assert "UNIFY_DRAFTS" in drafts.refusal(SimpleNamespace(**{**on, "UNIFY_DRAFTS": ""}))
    assert "UNIFY_TOOLS_IN_CODE" in drafts.refusal(SimpleNamespace(**{**on, "UNIFY_TOOLS_IN_CODE": "off"}))
    assert drafts.refusal(SimpleNamespace(UNIFY_DRAFTS="on", UNIFY_HANDBACK_DECISION="")) is None


def test_the_cli_refuses_the_decision_without_the_drafts(monkeypatch, capsys):
    monkeypatch.setattr(SETTINGS, "UNIFY_HANDBACK_DECISION", "on")
    monkeypatch.setattr(SETTINGS, "UNIFY_DRAFTS", "")
    monkeypatch.setattr(cli, "_configure_environment", lambda args: None)

    async def _never(args):
        raise AssertionError("the run started")

    monkeypatch.setattr(cli, "_run_act", _never)
    assert cli.main(["act", "Do the task."]) == 2
    assert "UNIFY_HANDBACK_DECISION" in capsys.readouterr().err


def test_decide_is_on_the_actors_allow_list_only_with_the_decision(monkeypatch):
    assert "decide" not in R.allow_list("actor", frozenset({"helper"}))
    assert "decide" in R.allow_list("actor", frozenset({"helper", R.DECISION_FLAG}))
    assert "decide" not in R.allow_list("teammate", frozenset({"helper", R.DECISION_FLAG}))
    from unify.actor import tool_handlers as H

    monkeypatch.setattr(SETTINGS, "UNIFY_HANDBACK_DECISION", "")
    assert R.DECISION_FLAG not in H.actor_flags("actor", False, False)
    monkeypatch.setattr(SETTINGS, "UNIFY_HANDBACK_DECISION", "on")
    assert R.DECISION_FLAG in H.actor_flags("actor", False, False)
    assert R.DECISION_FLAG not in H.actor_flags("teammate", False, False)


# --- the draft file and the decision, without a box ------------------------------------------------------------------


def test_a_draft_holds_the_definitions_and_imports_unchanged_under_a_header_with_no_request_text(tmp_path):
    session, root, fd = _session(tmp_path)
    try:
        probed = {"defs": {"double": "def double(v):\n    return v * 2", "Box": "class Box:\n    pass"},
                  "imports": ["import math"]}  # fmt: skip
        assert session.take(probed) == ["double", "Box"]
        text = (root / session.rel).read_text()
        lines = text.splitlines()
        assert lines[0].startswith("# Drafted by the harness at ") and lines[0].endswith(
            " from definitions made in this session, unchanged.",
        )
        assert lines[1] == f"# Reuse: from drafts.{session.module} import <name>"
        assert "import math" in lines and "def double(v):" in lines and "class Box:" in lines
        assert "Request" not in text
        assert session.take(probed) == []  # nothing new: no rewrite, no row
        probed["defs"]["double"] = "def double(v):\n    return v + v"
        assert session.take(probed) == ["double"]
        assert "return v + v" in (root / session.rel).read_text()
        rows = session.events
        assert [r["type"] for r in rows] == ["draft", "draft"] and rows[1]["names"] == ["double"]
    finally:
        os.close(fd)


def test_two_sessions_in_the_same_second_get_two_files(tmp_path):
    first, root, fd = _session(tmp_path)
    second = drafts.SessionDrafts(root_fd=fd, events=_Events(), started=None)
    second.module = first.module  # the same start second
    try:
        first.take({"defs": {"f": "def f():\n    return 1"}, "imports": []})
        second.take({"defs": {"g": "def g():\n    return 2"}, "imports": []})
        assert first.rel != second.rel and second.module == f"{first.module}_2"
        assert "def f()" in (root / first.rel).read_text() and "def g()" in (root / second.rel).read_text()
    finally:
        os.close(fd)


def test_keep_marks_the_named_ones_and_drops_the_others_skip_drops_them_and_none_is_recorded(tmp_path):
    session, root, fd = _session(tmp_path)
    try:
        session.take({"defs": {"a": "def a():\n    pass", "b": "def b():\n    pass"}, "imports": []})
        assert session.decide("keep", ["a"], "why")[0] == "no_question"  # nothing asked yet
        question = session.question(["a", "b"])
        assert question.startswith("Before your answer is handed back: this request defined `a`, `b`.")
        assert 'tools.decide("skip")' in question and "goes back unchanged" in question
        assert session.decide("maybe", None, None)[0] == "bad_choice"
        assert session.decide("keep", [], None)[0] == "no_names"
        assert session.decide("keep", ["c"], None) == ("not_asked", "c")
        assert session.decide("keep", ["a"], "reads any  grid\ntext")[0] is None
        text = (root / session.rel).read_text()
        assert "# Kept: reads any grid text\ndef a():" in text and "def b()" not in text
        assert session.decide("skip", None, None)[0] == "no_question"  # answered once
        session.after_question()
        session.take({"defs": {"c": "def c():\n    pass"}, "imports": []})
        session.question(["c"])
        assert session.decide("skip", None, None)[0] is None
        assert "def c()" not in (root / session.rel).read_text() and "def a()" in (root / session.rel).read_text()
        session.take({"defs": {"d": "def d():\n    pass"}, "imports": []})
        session.question(["d"])
        session.after_question()  # the turn ended without an answer
        decisions = [r for r in session.events if r["type"] == "decision"]
        assert [(r["choice"], r["names"]) for r in decisions] == [("keep", ["a"]), ("skip", []), ("none", [])]
    finally:
        os.close(fd)


def test_a_skip_that_leaves_nothing_removes_the_file(tmp_path):
    session, root, fd = _session(tmp_path)
    try:
        session.take({"defs": {"only": "def only():\n    pass"}, "imports": []})
        assert (root / session.rel).exists()
        session.question(["only"])
        assert session.decide("skip", None, None)[0] is None
        assert not (root / session.rel).exists()
    finally:
        os.close(fd)


def test_an_import_from_a_draft_is_recorded_with_its_names_and_whether_it_is_this_sessions(tmp_path):
    session, _root, fd = _session(tmp_path)
    try:
        session.note_use(
            "from drafts.d20261015T210300Z import double, Box\nimport drafts.d20261016T000000Z\n"
            f"from drafts import {session.module}\nimport json\nfrom .drafts import x\n",
        )
        session.note_use("def broken(:\n")
        rows = {r["file"]: r for r in session.events}
        assert rows["drafts/d20261015T210300Z.py"]["names"] == ["Box", "double"]
        assert rows["drafts/d20261016T000000Z.py"]["names"] == []
        assert len(rows) == 3 and all(r["same_session"] is False for r in rows.values())  # no file of its own yet
        session.events.clear()
        session.take({"defs": {"f": "def f():\n    pass"}, "imports": []})
        session.events.clear()
        session.note_use(f"from drafts import {session.module}\n")
        assert session.events[0]["file"] == session.rel and session.events[0]["same_session"] is True
    finally:
        os.close(fd)


# --- end to end ------------------------------------------------------------------------------------------------------


@pytest.fixture
def minimal(core_world, monkeypatch):  # noqa: F811
    """The minimal arm's setting with the drafts on; every in-code session the test opens."""
    monkeypatch.setattr(SETTINGS, "UNIFY_MEMORY", "")
    monkeypatch.setattr(SETTINGS, "UNIFY_LIBRARY", "off")
    monkeypatch.setattr(SETTINGS, "UNIFY_TOOLS_IN_CODE", "on")
    monkeypatch.setattr(SETTINGS, "UNIFY_CELL_STATE", "persistent")
    monkeypatch.setattr(SETTINGS, "UNIFY_DRAFTS", "on")
    collect = _Collect()
    monkeypatch.setattr(in_code_tools, "SESSIONS", collect)
    yield {"workspace": core_world["workspace"], "opened": collect.opened}


def _rows(session) -> list[dict]:
    path = in_code_tools.request_folder(session.request_id) / "tool-events.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines()]


def _drafts_in(workspace) -> list:
    folder = workspace / "drafts"
    return sorted(folder.glob("d*.py")) if folder.is_dir() else []


@needs_bwrap
@pytest.mark.asyncio
@pytest.mark.timeout(400)
async def test_a_session_drafts_what_it_defined_and_the_next_session_imports_it(minimal):
    model = ScriptedModel(actor=[reply(calls=[code(DEFINE + "tools.answer('ok')\n")])])
    assert await _act(model, "Build two helpers.") == "ok", _why(model)
    (draft,) = _drafts_in(minimal["workspace"])
    text = draft.read_text()
    assert "import math" in text and "def double(v):" in text and "class Box:" in text, text
    assert "Build two helpers" not in text
    assert [r for r in _rows(minimal["opened"][0]) if r.get("type") == "draft"][0]["names"] == ["double", "Box"]
    reuse = f"from drafts.{draft.stem} import double, Box\nprint('REUSED', double(21), Box(3.7).v)\n"
    model2 = ScriptedModel(actor=[reply(calls=[code(reuse)]), reply("again")])
    assert await _act(model2, "Use the helpers.") == "again", _why(model2)
    assert "REUSED 42 3" in _tool_results(model2.calls[1]), _why(model2)
    uses = [r for r in _rows(minimal["opened"][1]) if r.get("type") == "draft_use"]
    assert uses == [{"type": "draft_use", "file": f"drafts/{draft.stem}.py", "names": ["Box", "double"],
                     "same_session": False}]  # fmt: skip


@needs_bwrap
@pytest.mark.asyncio
@pytest.mark.timeout(400)
async def test_with_the_drafts_alone_there_is_no_extra_turn_and_nothing_in_the_prompt(minimal, monkeypatch):
    model = ScriptedModel(actor=[reply(calls=[code(DEFINE_AND_ANSWER)])])
    assert await _act(model, "Define and answer.") == "done", _why(model)
    assert model.kinds() == ["actor"]
    system = "\n".join(_system_texts(model.calls[0]))
    # nothing in the prompt or the tools: the same as with the drafts off (below)
    monkeypatch.setattr(SETTINGS, "UNIFY_DRAFTS", "")
    off = ScriptedModel(actor=[reply(calls=[code(DEFINE_AND_ANSWER)])])
    assert await _act(off, "Define and answer.") == "done", _why(off)
    assert "\n".join(_system_texts(off.calls[0])) == system
    assert off.calls[0].request["tools"] == model.calls[0].request["tools"]
    assert len(_drafts_in(minimal["workspace"])) == 1  # the off run wrote none


@pytest.fixture
def deciding(minimal, monkeypatch):
    monkeypatch.setattr(SETTINGS, "UNIFY_HANDBACK_DECISION", "on")
    yield minimal


@needs_bwrap
@pytest.mark.asyncio
@pytest.mark.timeout(400)
async def test_keep_gets_one_extra_turn_and_the_answer_goes_back_unchanged(deciding):
    keep = 'print(tools.decide("keep", names=["triple"], why="times three comes up"))\n'
    model = ScriptedModel(actor=[reply(calls=[code(DEFINE_AND_ANSWER)]), reply(calls=[code(keep)])])
    assert await _act(model, "Define and answer.") == "done", _why(model)
    assert model.kinds() == ["actor", "actor"]
    asked = json.dumps(model.calls[1].messages, default=str)
    assert "Before your answer is handed back: this request defined `triple`." in asked
    (draft,) = _drafts_in(deciding["workspace"])
    assert "# Kept: times three comes up\ndef triple(v):" in draft.read_text()
    rows = [r for r in _rows(deciding["opened"][0]) if r.get("type") == "decision"]
    assert rows == [{"type": "decision", "choice": "keep", "names": ["triple"], "why": "times three comes up",
                     "asked": ["triple"]}]  # fmt: skip


@needs_bwrap
@pytest.mark.asyncio
@pytest.mark.timeout(400)
async def test_skip_drops_the_draft_and_an_unanswered_question_is_none(deciding):
    model = ScriptedModel(actor=[reply(calls=[code(DEFINE_AND_ANSWER)]), reply(calls=[code('tools.decide("skip")\n')])])
    assert await _act(model, "Define and answer.") == "done", _why(model)
    assert _drafts_in(deciding["workspace"]) == []
    model2 = ScriptedModel(actor=[reply(calls=[code(DEFINE_AND_ANSWER)]), reply("No need to keep anything.")])
    assert await _act(model2, "Define and answer.") == "done", _why(model2)
    rows = [r for r in _rows(deciding["opened"][1]) if r.get("type") == "decision"]
    assert [r["choice"] for r in rows] == ["none"]


@needs_bwrap
@pytest.mark.asyncio
@pytest.mark.timeout(400)
async def test_no_question_when_nothing_was_defined(deciding):
    model = ScriptedModel(actor=[reply(calls=[code("tools.answer('plain')\n")])])
    assert await _act(model, "Answer.") == "plain", _why(model)
    assert model.kinds() == ["actor"]
    assert _drafts_in(deciding["workspace"]) == []
