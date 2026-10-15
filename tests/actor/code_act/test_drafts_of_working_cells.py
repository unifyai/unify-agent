"""``UNIFY_DRAFTS=cells`` and ``both``: a request's working cells, drafted at each hand-back (unify/actor/drafts.py).

At every hand-back the cells of the request that ran without error, a repeated source once, reduced to what the last
one that binds a name depends on, go to ``drafts/d<session start>-r<k>.cells.py``: each under a ``# cell N`` line,
unchanged, below a header that says when the harness wrote it and that it is for reading. A request whose working
cells only print or query writes nothing. With the decision on, a hand-back that wrote such a file and drafted no
definitions asks about the file: keep marks it, skip removes it. A later cell that names its path is a ``draft_use``
row. ``both`` writes the definitions module too. The policy refuses any drafts mode set by hand.

The model is scripted (no key, no network). The end-to-end tests run the minimal arm's setting (memory off, the library
off, the tools called from code, a persistent interpreter), cells in the real sandboxed worker.
"""

from __future__ import annotations

import json
import os
import re

import pytest

from tests.actor.code_act.core_world import core_world, world  # noqa: F401 (fixtures)
from tests.actor.code_act.sandbox_world import needs_bwrap
from tests.actor.code_act.test_in_code_tools_wiring import _act, _Collect, _tool_results, _why, code
from tests.scripted_model import ScriptedModel, reply
from unify import sandbox
from unify.actor import adaptive_policy, consequences, drafts, in_code_tools
from unify.settings import SETTINGS


class _Events(list):
    def append(self, row):  # noqa: D401 - the EventFile's one method
        super().append(dict(row))


class _Settings:
    def __init__(self, **values):
        self.__dict__.update(values)


def _session(tmp_path, mode=drafts.CELLS, events=None, record=None, started=None):
    root = tmp_path / "workspace"
    root.mkdir(parents=True, exist_ok=True)
    fd = os.open(root, os.O_RDONLY | os.O_DIRECTORY)
    session = drafts.SessionDrafts(
        root_fd=fd,
        events=events if events is not None else _Events(),
        mode=mode,
        record=record,
        started=started,
    )
    return session, root, fd


def _cells_files(root) -> list:
    folder = root / "drafts"
    return sorted(folder.glob("*.cells.py")) if folder.is_dir() else []


# --- the setting -----------------------------------------------------------------------------------------------------


def test_the_mode_takes_on_cells_or_both_and_refuses_anything_else():
    assert [drafts.parse_mode(v) for v in (None, "", "off", " ON ", "Cells", "both")] == ["", "", "", "on", "cells", "both"]
    with pytest.raises(ValueError, match="'on', 'cells' or 'both'"):
        drafts.parse_mode("functions")
    assert drafts.mode(_Settings(UNIFY_DRAFTS="cells")) == "cells"
    assert drafts.enabled(_Settings(UNIFY_DRAFTS="both")) and not drafts.enabled(_Settings(UNIFY_DRAFTS=""))
    # the other switches still take on or off only
    with pytest.raises(ValueError):
        drafts.parse("cells")


def test_the_settings_take_the_modes_and_the_policy_refuses_any_of_them():
    from unify.settings import ProductionSettings

    assert ProductionSettings(UNIFY_DRAFTS="cells").UNIFY_DRAFTS == "cells"
    assert ProductionSettings(UNIFY_DRAFTS=" Both ").UNIFY_DRAFTS == "both"
    with pytest.raises(ValueError, match="UNIFY_DRAFTS"):
        ProductionSettings(UNIFY_DRAFTS="all")
    with pytest.raises(ValueError, match="UNIFY_HANDBACK_DECISION"):
        ProductionSettings(UNIFY_HANDBACK_DECISION="cells")
    policy = _Settings(UNIFY_ADAPTIVE_POLICY="on", UNIFY_TOOLS_IN_CODE="on", UNIFY_MEMORY="", UNIFY_DRAFTS="cells",
                       UNIFY_HANDBACK_DECISION="", UNIFY_CONSEQUENCES="")  # fmt: skip
    assert "UNIFY_DRAFTS" in (adaptive_policy.refusal(policy) or "")
    # the decision asks about drafts of any mode
    assert drafts.refusal(_Settings(UNIFY_DRAFTS="cells", UNIFY_HANDBACK_DECISION="on", UNIFY_TOOLS_IN_CODE="on")) is None


# --- which cells -----------------------------------------------------------------------------------------------------


def test_the_slice_keeps_what_the_last_binding_cell_depends_on():
    cells = [
        (0, "import json\nraw = '[1, 2, 3]'", True),
        (1, "print(raw)", True),  # inspection: binds nothing, read by nothing kept
        (2, "values = json.loads(raw)\nunused = 7", True),
        (3, "total = sum(values) / 0", False),  # an error: never a working cell
        (4, "print(values)", True),
        (5, "total = sum(values)", True),
        (6, "print(total)", True),  # after the last binding cell: not kept
    ]
    assert drafts.working_slice(cells) == [(0, cells[0][1]), (2, cells[2][1]), (5, cells[5][1])]


def test_a_repeated_source_is_kept_once_at_its_last_run_and_prints_alone_give_nothing():
    cells = [(0, "x = 1", True), (1, "y = x + 1", True), (2, "x = 1", True), (3, "z = x * 3", True)]
    assert drafts.working_slice(cells) == [(2, "x = 1"), (3, "z = x * 3")]
    assert drafts.working_slice([(0, "print(memory.ask('which rule?'))", True), (1, "print(1)", True)]) == []
    assert drafts.working_slice([(0, "x = 1", False)]) == []
    assert drafts.working_slice([]) == []


# --- the file --------------------------------------------------------------------------------------------------------


def test_each_hand_back_writes_its_own_file_with_a_reading_header_and_no_request_text(tmp_path):
    events = _Events()
    session, root, fd = _session(tmp_path, events=events, started=1_760_000_000.0)
    try:
        session.note_cell(0, "rows = [3, 1, 2]", True)
        session.note_cell(1, "print(rows)", True)
        session.note_cell(2, "rows.sort()\nbest = rows[-1]", True)
        first = session.take_cells()
        assert first == "drafts/d20251009T085320Z-r1.cells.py"
        text = (root / first).read_text()
        lines = text.splitlines()
        assert re.fullmatch(r"# Drafted by the harness at \S+Z from cells of one request in this session that ran "
                            r"without error\.", lines[0]), lines[0]  # fmt: skip
        assert lines[1] == "# Reuse: read it and copy what applies; running it repeats those steps."
        assert text.endswith("\n# cell 0\nrows = [3, 1, 2]\n\n# cell 2\nrows.sort()\nbest = rows[-1]\n"), text
        assert "print(rows)" not in text
        # its name is not a module name, so no import can run it
        assert not root.joinpath(first).name[: -len(".py")].isidentifier()
        assert [r for r in events if r["type"] == "draft"] == [
            {"type": "draft", "mode": "cells", "file": first, "cells": [0, 2], "lines": 3, "bytes": len(text.encode())},
        ]
        # a request whose cells only print writes nothing; the next one gets the next hand-back's number
        session.note_cell(3, "print(best)", True)
        assert session.take_cells() is None
        session.note_cell(4, "best = best * 2", True)
        assert session.take_cells() == "drafts/d20251009T085320Z-r3.cells.py"
        assert [p.name for p in _cells_files(root)] == ["d20251009T085320Z-r1.cells.py", "d20251009T085320Z-r3.cells.py"]
    finally:
        os.close(fd)


def test_cells_mode_drafts_no_definitions_and_both_mode_drafts_both(tmp_path):
    probed = {"defs": {"double": "def double(v):\n    return v * 2"}, "imports": []}
    cells_only, root, fd = _session(tmp_path, mode=drafts.CELLS, started=1_760_000_000.0)
    try:
        assert cells_only.take(probed) == []
        assert not (root / "drafts" / "d20251009T085320Z.py").exists()
    finally:
        os.close(fd)
    both, root2, fd2 = _session(tmp_path / "b", mode=drafts.BOTH, started=1_760_000_100.0)
    try:
        both.note_cell(0, "def double(v):\n    return v * 2\nanswer = double(4)", True)
        assert both.take(probed) == ["double"]
        assert both.take_cells() == "drafts/d20251009T085500Z-r1.cells.py"
        assert sorted(p.name for p in (root2 / "drafts").iterdir()) == ["d20251009T085500Z-r1.cells.py",
                                                                        "d20251009T085500Z.py"]  # fmt: skip
    finally:
        os.close(fd2)


def test_a_session_in_the_same_second_as_an_earlier_cells_file_takes_another_name(tmp_path):
    earlier, root, fd = _session(tmp_path, started=1_760_000_000.0)
    later, _root, fd2 = _session(tmp_path, started=1_760_000_000.0)
    try:
        earlier.note_cell(0, "a = 1", True)
        assert earlier.take_cells() == "drafts/d20251009T085320Z-r1.cells.py"
        later.note_cell(0, "b = 2", True)
        assert later.take_cells() == "drafts/d20251009T085320Z_2-r1.cells.py"
    finally:
        os.close(fd)
        os.close(fd2)


# --- the decision about a file ---------------------------------------------------------------------------------------


def test_keep_marks_the_file_skip_removes_it_and_an_unanswered_question_is_none(tmp_path):
    events = _Events()
    record = consequences.DraftRecord(tmp_path / "drafts-record.json")
    session, root, fd = _session(tmp_path, events=events, record=record, started=1_760_000_000.0)
    try:
        session.note_cell(0, "table = {1: 'a'}", True)
        rel = session.take_cells()
        asked = session.question_file(rel)
        assert asked == drafts.CELLS_QUESTION.format(path=f"`{rel}`")
        assert session.decide("keep", None, "  builds  the table ") == (None, f"recorded: kept {rel}")
        assert "# Kept: builds the table\n\n# cell 0\ntable = {1: 'a'}\n" in (root / rel).read_text()
        assert session.decide("skip", None, "") == ("no_question", "")
        assert record.load()[rel[len("drafts/"):-3]]["kept"] == ["cells"]
        session.after_question()
        session.note_cell(1, "table[2] = 'b'", True)
        rel2 = session.take_cells()
        session.question_file(rel2)
        assert session.decide("skip", None, None) == (None, f"recorded: skipped; removed {rel2}")
        assert not (root / rel2).exists() and (root / rel).exists()
        session.after_question()
        session.note_cell(2, "table[3] = 'c'", True)
        session.question_file(session.take_cells())
        session.after_question()
        rows = [r for r in events if r["type"] == "decision"]
        assert [(r["choice"], r["mode"], r["asked"]) for r in rows] == [
            ("keep", "cells", [rel]), ("skip", "cells", [rel2]), ("none", "cells", ["drafts/d20251009T085320Z-r3.cells.py"]),
        ]
    finally:
        os.close(fd)


# --- a later read ----------------------------------------------------------------------------------------------------


def test_a_cell_naming_a_cells_file_is_a_draft_use_and_the_mirror_counts_it(tmp_path):
    events = _Events()
    record = consequences.DraftRecord(tmp_path / "drafts-record.json")
    writer, root, fd = _session(tmp_path, events=_Events(), record=record, started=1_760_000_000.0)
    reader, _root, fd2 = _session(tmp_path, events=events, record=record, started=1_760_000_500.0)
    try:
        writer.note_cell(0, "grid = [[1]]", True)
        rel = writer.take_cells()
        reader.note_use(f"print(open('/workspace/{rel}').read())\n")
        assert events == [{"type": "draft_use", "file": rel, "names": [], "same_session": False, "mode": "cells"}]
        key = rel[len("drafts/"):-3]
        assert reader.imported_now == [key] and record.load()[key]["imports"] == 1
        line = consequences.Mirror().line(None, reader.imported_now, record)
        assert f"{rel}: imported in 1 request; no keep or skip recorded" in line, line
        # the writer's own read: same session, and a request that read it like any other
        writer.note_use(f"exec(open('{rel}').read())")
        assert writer.imported_now == [key] and record.load()[key]["imports"] == 2
    finally:
        os.close(fd)
        os.close(fd2)


# --- end to end ------------------------------------------------------------------------------------------------------


WORK = (
    "import math\nrows = [4, 9, 16]\n",
    "print(rows)\n",
    "roots = [math.isqrt(v) for v in rows]\n",
    "1 / 0\n",
    "tools.answer(str(sum(roots)))\n",
)


@pytest.fixture
def cells_mode(core_world, monkeypatch):  # noqa: F811
    """The minimal arm's setting with UNIFY_DRAFTS=cells; every in-code session the test opens."""
    monkeypatch.setattr(SETTINGS, "UNIFY_MEMORY", "")
    monkeypatch.setattr(SETTINGS, "UNIFY_LIBRARY", "off")
    monkeypatch.setattr(SETTINGS, "UNIFY_TOOLS_IN_CODE", "on")
    monkeypatch.setattr(SETTINGS, "UNIFY_CELL_STATE", "persistent")
    monkeypatch.setattr(SETTINGS, "UNIFY_DRAFTS", "cells")
    collect = _Collect()
    monkeypatch.setattr(in_code_tools, "SESSIONS", collect)
    yield {"workspace": core_world["workspace"], "opened": collect.opened}


def _rows(session) -> list[dict]:
    path = in_code_tools.request_folder(session.request_id) / "tool-events.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines()]


@needs_bwrap
@pytest.mark.asyncio
@pytest.mark.timeout(400)
async def test_a_request_drafts_its_working_cells_and_a_later_request_reads_them(cells_mode):
    model = ScriptedModel(actor=[reply(calls=[code(c)]) for c in WORK])
    assert await _act(model, "Sum the square roots.") == "9", _why(model)
    (cells_file,) = _cells_files(cells_mode["workspace"])
    text = cells_file.read_text()
    assert "import math\nrows = [4, 9, 16]" in text and "roots = [math.isqrt(v) for v in rows]" in text, text
    assert "print(rows)" not in text and "1 / 0" not in text and "tools.answer" not in text, text
    assert "Sum the square roots" not in text
    assert re.findall(r"^# cell (\d+)$", text, flags=re.M) == ["0", "2"], text
    drafted = [r for r in _rows(cells_mode["opened"][0]) if r.get("type") == "draft"]
    assert [(r["mode"], r["cells"]) for r in drafted] == [("cells", [0, 2])]
    read = f"print('READ', len(open('{sandbox.WORKSPACE_MOUNT}/drafts/{cells_file.name}').read()) > 0)\n"
    model2 = ScriptedModel(actor=[reply(calls=[code(read)]), reply("read it")])
    assert await _act(model2, "Look at earlier work.") == "read it", _why(model2)
    assert "READ True" in _tool_results(model2.calls[1]), _why(model2)
    uses = [r for r in _rows(cells_mode["opened"][1]) if r.get("type") == "draft_use"]
    assert uses == [{"type": "draft_use", "file": f"drafts/{cells_file.name}", "names": [], "same_session": False,
                     "mode": "cells"}]  # fmt: skip


@needs_bwrap
@pytest.mark.asyncio
@pytest.mark.timeout(400)
async def test_with_the_decision_a_written_file_is_asked_about_and_the_answer_goes_back_unchanged(cells_mode, monkeypatch):
    monkeypatch.setattr(SETTINGS, "UNIFY_HANDBACK_DECISION", "on")
    keep = 'print(tools.decide("keep", why="the roots table"))\n'
    model = ScriptedModel(actor=[*(reply(calls=[code(c)]) for c in WORK), reply(calls=[code(keep)])])
    assert await _act(model, "Sum the square roots.") == "9", _why(model)
    asked = json.dumps(model.calls[-1].messages, default=str)
    assert "this request's working cells are saved as a draft in `drafts/" in asked, _why(model)
    (cells_file,) = _cells_files(cells_mode["workspace"])
    assert "# Kept: the roots table\n" in cells_file.read_text()
    rows = [r for r in _rows(cells_mode["opened"][0]) if r.get("type") == "decision"]
    assert [(r["choice"], r["mode"]) for r in rows] == [("keep", "cells")]
    # a request whose cells only print writes nothing and asks nothing
    plain = ScriptedModel(actor=[reply(calls=[code("print(2)\n")]), reply(calls=[code("tools.answer('two')\n")])])
    assert await _act(plain, "Print.") == "two", _why(plain)
    assert plain.kinds() == ["actor", "actor"] and len(_cells_files(cells_mode["workspace"])) == 1
