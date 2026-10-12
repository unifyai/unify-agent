"""The writer's read, grep and read_episode tools, head-first cell output and the read log (a measure: finish never
waits on reading). Built on test_writer_pass's own helpers (``_call``, ``Turns``, ``_writer``).
"""

import asyncio


from unify.memory_v2 import batch_map as bm
from unify.memory_v2 import views
from unify.memory_v2.episodes import Cell, Episode
from unify.memory_v2.writer_pass import _head_marked, writer_tools
from unify.memory_v2.trigger import PassRequest
from tests.memory_v2.test_writer_pass import Turns, _call, _writer, needs_bwrap


def _episode(eid, cells=(), request=("req",)):
    return Episode(
        episode_id=eid,
        started_at="2026-10-09T00:00:00Z",
        ended_at="2026-10-09T00:01:00Z",
        build="b",
        model="m",
        effort="low",
        regime="none",
        memory_main="abc",
        worktree_before=None,
        worktree_after=None,
        request=list(request),
        transcript=[],
        cells=list(cells),
        actions=[],
    )


LONG_CELL = Cell(0, "def f():\n    return 1\n" + "# pad\n" * 2000, "")
EPS = {
    "e1": _episode("e1", [LONG_CELL], ("req", "same", "same")),
    "e2": _episode("e2"),
}


def _pass(tmp_path, turns, eids=("e1", "e2"), **cfg):
    model = Turns(turns)
    _, _, writer = _writer(tmp_path, model, library_mode=True, **cfg)
    writer.load = EPS.__getitem__
    out = asyncio.run(
        writer.run(PassRequest("incremental", "svc", list(eids), False), "p1"),
    )
    return out, model


def test_tools_present_only_with_library_mode():
    names = lambda ts: {t["function"]["name"] for t in ts}  # noqa: E731
    assert names(writer_tools()) == {"execute_code", "check", "finish"}
    assert names(writer_tools(library_mode=True)) == {
        "execute_code",
        "check",
        "finish",
        "read",
        "grep",
        "overview",
        "read_episode",
        "fixture",
        "replay",
    }
    assert "pool" in names(writer_tools(library_mode=True, pool=True))
    desc = next(
        t
        for t in writer_tools(library_mode=True)
        if t["function"]["name"] == "read_episode"
    )
    assert "observation:<i>" in desc["function"]["description"]


def test_cell_output_is_head_first_and_marked():
    text = "A" * 9000 + "TAIL"
    out, full = _head_marked(text)
    assert (
        out.startswith("AAAA")
        and "[… shown bytes 0–8000 of 9004; next: offset=8000]" in out
    )
    assert full == text


def test_finish_never_waits_on_reading(tmp_path):
    out, model = _pass(tmp_path, [], max_calls=5)  # Turns finishes on every turn
    assert not any("neither covered" in str(c) for c in model.outputs.values())
    assert out.coverage["unread"] == ["e1", "e2"] and out.coverage["opened"] == []


def test_read_episode_pages_and_logs_what_was_shown(tmp_path):
    ep = EPS["e1"]
    cell = bm.part_text(ep, "cell:0").encode()
    _, _, b = views.view_range(cell, 0)
    assert b < len(cell)  # the cell needs two pages
    turns = [
        _call("r1", "read_episode", {"episode": "e1", "part": "request"}),
        _call("r3", "read_episode", {"episode": "e1", "part": "cell:0"}),
        _call("r4", "read_episode", {"episode": "e1", "part": "cell:0", "offset": b}),
        _call("f1", "finish", {"summary": "done"}),
    ]
    out, model = _pass(tmp_path, turns, max_calls=20)
    assert model.outputs["r1"] == '== request ==\n"req"'
    assert model.outputs["r3"].endswith(f"next: offset={b}]")
    s = out.coverage
    assert s["opened"] == ["e1"] and s["unread"] == ["e2"]
    assert s["parts_read"] == {"e1": ["cell:0", "request"]}
    assert s["bytes_read"] == len(b'"req"') + len(cell)


def test_read_and_grep_tools_are_confined_and_bounded(tmp_path, monkeypatch):
    monkeypatch.setattr(views, "GREP_TIMEOUT_S", 1.0)
    slow = _episode("e3", request=("a" * 30 + "!",))
    EPS["e3"] = slow
    try:
        turns = [
            _call("esc", "read", {"path": "/inputs/../../etc/passwd"}),
            _call("ok", "read", {"path": "/inputs/batch_map.json"}),
            _call("bad", "grep", {"pattern": "("}),
            _call("slow", "grep", {"pattern": "(a+)+$", "path": "/inputs/episodes"}),
            _call("cell", "execute_code", {"code": "print(1)"}),
        ]
        out, model = _pass(tmp_path, turns, eids=("e3",), max_calls=10)
    finally:
        del EPS["e3"]
    assert model.outputs["esc"] == (
        "refused: /inputs/../../etc/passwd is outside /inputs, /memory and /outputs"
    )
    assert '"episode_id": "e3"' in model.outputs["ok"]
    assert model.outputs["bad"].startswith("refused: invalid pattern:")
    assert (
        model.outputs["slow"]
        == "[grep timed out after 1 s: narrow the pattern or the path]"
    )
    assert out.coverage["opened"] == []  # read, grep and cells are not episode reads


@needs_bwrap
def test_long_cell_output_is_head_first_with_the_full_output_readable(tmp_path):
    turns = [
        _call("c1", "execute_code", {"code": "print('A' * 9000 + 'TAIL')"}),
        _call("r1", "read", {"path": "/outputs/cell-0.txt", "offset": 8000}),
    ]
    _, model = _pass(tmp_path, turns, eids=("e2",), max_calls=10)
    assert model.outputs["c1"].startswith("AAAA")
    assert "[… shown bytes 0–8000 of 9005; next: offset=8000]" in model.outputs["c1"]
    assert "(full output: /outputs/cell-0.txt)" in model.outputs["c1"]
    assert model.outputs["r1"].rstrip().endswith("TAIL")


def test_library_tools_are_unknown_when_off(tmp_path):
    model = Turns([_call("r", "read", {"path": "/inputs"})])
    _, _, writer = _writer(tmp_path, model)
    out = asyncio.run(writer.run(PassRequest("incremental", "venmo", [], False), "p0"))
    assert (
        model.outputs["r"]
        == "refused: unknown tool 'read'; your tools: execute_code, check, finish"
    )
    assert out.coverage is None


def _calls(*calls):
    """One assistant turn carrying several tool calls."""
    return {"role": "assistant", "tool_calls": [c["tool_calls"][0] for c in calls]}


def test_reader_calls_use_their_own_budget_not_max_calls(tmp_path):
    read = lambda cid: _call(
        cid,
        "read_episode",
        {"episode": "e2", "part": "request"},
    )  # noqa: E731
    turns = [read("a"), read("b"), read("c")]
    out, model = _pass(tmp_path, turns, max_calls=6, max_reads=2)
    assert model.outputs["a"] == '== request ==\n"req"'
    assert model.outputs["b"] == '== request ==\n"req"'
    assert model.outputs["c"] == "not run: the pass's reader-call limit (2) is reached"
    assert out.reads == 2


def test_at_most_16_reader_calls_per_turn(tmp_path):
    many = [_call(f"r{i}", "read", {"path": "/inputs"}) for i in range(17)]
    _, model = _pass(tmp_path, [_calls(*many)], max_calls=3)
    assert model.outputs["r15"].startswith("batch_map.json")
    assert model.outputs["r16"] == (
        "not run: at most 16 reader calls run per model turn; send it again"
    )


def test_read_episode_parts_share_one_page(tmp_path):
    ep = EPS["e1"]
    cell = bm.part_text(ep, "cell:0").encode()
    turns = [
        _call(
            "p",
            "read_episode",
            {"episode": "e1", "parts": ["request", "cell:0", "observation:0"]},
        ),
        _call("many", "read_episode", {"episode": "e1", "parts": ["request"] * 9}),
    ]
    out, model = _pass(tmp_path, turns, max_calls=5)
    got = model.outputs["p"]
    request, rest = got.split("\n\n== cell:0 ==\n", 1)
    assert request == '== request ==\n"req"'
    shown = views.VIEW_BYTES - len(b'"req"')
    assert f"[… shown bytes 0–{shown} of {len(cell)}; next: offset={shown}]" in rest
    assert rest.endswith(
        '== observation:0 ==\nnot shown (page full): read again with parts=["observation:0"]',
    )
    # m4k: more than 8 parts page instead of refusing; the rest is named for the next call
    assert model.outputs["many"].endswith('next: parts=["request"]]')
    s = out.coverage
    assert s["parts_read"] == {"e1": ["cell:0", "request"]}


def test_read_episode_bad_offsets_refuse_the_whole_call(tmp_path):
    turns = [
        _call(
            "bogus",
            "read_episode",
            {"episode": "e1", "parts": ["request", "bogus"]},
        ),
        _call(
            "neg",
            "read_episode",
            {"episode": "e1", "parts": ["request", {"part": "cell:0", "offset": -1}]},
        ),
        _call(
            "txt",
            "read_episode",
            {"episode": "e1", "parts": [{"part": "request", "offset": "x"}]},
        ),
    ]
    out, model = _pass(tmp_path, turns, max_calls=5)
    # an unknown name is answered with the valid names, while the valid parts are shown
    assert model.outputs["bogus"].startswith(
        "unknown part(s) in e1: 'bogus'; valid: request, observation:0..1, cell:0..0",
    )
    assert '== request ==\n"req"' in model.outputs["bogus"]
    assert model.outputs["neg"].startswith("refused: offset must be an integer >= 0")
    assert model.outputs["txt"].startswith("refused: offset must be an integer >= 0")
    assert out.coverage["parts_read"] == {"e1": ["request"]}


def test_read_episode_says_its_page_is_one_8000_byte_page():
    desc = next(
        t
        for t in writer_tools(library_mode=True)
        if t["function"]["name"] == "read_episode"
    )
    assert "on one page of 8000 bytes" in desc["function"]["description"]
    assert "parts" in desc["function"]["parameters"]["properties"]
