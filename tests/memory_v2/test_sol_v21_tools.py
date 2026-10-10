"""P1 Task 5 (spec v2.1 §7.3–7.4, P1, P5): the writer's read, grep, read_episode and dismiss tools, head-first cell
output, and coverage at finish. Built on test_sol_pass's own helpers (``_call``, ``Turns``, ``_sol``): the plan's
``_pass``/``_scripted`` sketches do not exist in that file."""

import asyncio


from unify.memory_v2 import batch_map as bm
from unify.memory_v2 import views
from unify.memory_v2.episodes import Cell, Episode
from unify.memory_v2.sol_pass import CODE_PASS_CAP, _head_marked, sol_tools
from unify.memory_v2.trigger import PassRequest
from tests.memory_v2.test_sol_pass import Turns, _call, _sol, needs_bwrap


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
    _, _, sol = _sol(tmp_path, model, v21=True, **cfg)
    sol.load = EPS.__getitem__
    out = asyncio.run(
        sol.run(PassRequest("incremental", "svc", list(eids), False), "p1"),
    )
    return out, model


def test_tools_present_only_with_v21():
    names = lambda ts: {t["function"]["name"] for t in ts}  # noqa: E731
    assert names(sol_tools()) == {"execute_code", "check", "finish"}
    assert names(sol_tools(v21=True)) == {
        "execute_code",
        "check",
        "finish",
        "read",
        "grep",
        "read_episode",
        "dismiss",
        "fixture",
    }
    desc = next(
        t for t in sol_tools(v21=True) if t["function"]["name"] == "read_episode"
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


def test_finish_refused_until_covered_then_cap(tmp_path):
    out, model = _pass(tmp_path, [], max_calls=5)  # Turns finishes on every turn
    assert out.coverage["missing"] == ["e1", "e2"]
    assert any(
        c.startswith("not finished: 2 episode(s)") for c in model.outputs.values()
    )
    assert not out.passed and CODE_PASS_CAP in out.codes


def test_read_episode_credits_shown_ranges_and_finish_succeeds(tmp_path):
    ep = EPS["e1"]
    cell = bm.part_text(ep, "cell:0").encode()
    _, _, b = views.view_range(cell, 0)
    assert b < len(cell)  # the cell needs two pages
    turns = [
        _call("r1", "read_episode", {"episode": "e1", "part": "request"}),
        _call(
            "r2",
            "read_episode",
            {"episode": "e1", "part": "observation:1"},
        ),  # a copy credits observation:0
        _call("r3", "read_episode", {"episode": "e1", "part": "cell:0"}),
        _call("r4", "read_episode", {"episode": "e1", "part": "cell:0", "offset": b}),
        _call("d1", "dismiss", {"episode": "e2", "reason": "no reusable work"}),
        _call("f1", "finish", {"summary": "done"}),
    ]
    out, model = _pass(tmp_path, turns, max_calls=20)
    assert model.outputs["r3"].endswith(f"next: offset={b}]")
    assert (
        out.summary == "done"
    )  # finish accepted (Turns sees no reply after the last turn)
    s = out.coverage
    assert (
        s["missing"] == []
        and s["covered"] == 1
        and s["dismissed"] == {"e2": "no reusable work"}
    )
    dismissed = len(
        bm.part_text(EPS["e2"], "request").encode(),
    )  # e2 was dismissed, not read
    assert s["bytes_read"] == s["bytes_required"] - dismissed


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
    assert model.outputs["esc"] == "refused: outside the readable roots"
    assert '"episode_id": "e3"' in model.outputs["ok"]
    assert model.outputs["bad"].startswith("refused: invalid pattern:")
    assert (
        model.outputs["slow"]
        == "[grep timed out after 1 s: narrow the pattern or the path]"
    )
    assert out.coverage["parts_read"] == 0  # read, grep and cells earn no credit


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


def test_v21_tools_are_unknown_when_off(tmp_path):
    model = Turns([_call("r", "read", {"path": "/inputs"})])
    _, _, sol = _sol(tmp_path, model)
    out = asyncio.run(sol.run(PassRequest("incremental", "venmo", [], False), "p0"))
    assert (
        model.outputs["r"] == "unknown tool 'read'; use execute_code, check or finish"
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
    turns = [
        read("a"),
        read("b"),
        _calls(
            read("c"),
            _call("d", "dismiss", {"episode": "e1", "reason": "nothing reusable"}),
        ),
    ]
    out, model = _pass(tmp_path, turns, max_calls=6, max_reads=2)
    assert model.outputs["a"] == '"req"' and model.outputs["b"] == '"req"'
    assert model.outputs["c"] == "not run: reader budget reached"
    assert (
        model.outputs["d"] == "ok"
    )  # dismiss still runs past the budget, so the pass can finish
    # model turns only, reader calls counted apart; finish is refused twice for its missing manifest (r2 §3)
    assert out.reads == 3 and out.calls == 6
    assert out.coverage["missing"] == [] and out.summary == "done"


def test_at_most_16_reader_calls_per_turn(tmp_path):
    many = [_call(f"r{i}", "read", {"path": "/inputs"}) for i in range(17)]
    _, model = _pass(tmp_path, [_calls(*many)], max_calls=3)
    assert model.outputs["r15"].startswith("batch_map.json")
    assert model.outputs["r16"] == "not run: at most 16 reader calls per turn"


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
    assert model.outputs["many"] == "refused: at most 8 parts per call"
    s = out.coverage
    assert s["bytes_read"] == len(b'"req"') + shown and s["parts_read"] == 1


def test_read_episode_parts_refuses_the_whole_call_before_any_credit(tmp_path):
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
    assert model.outputs["bogus"].startswith("refused: unknown part 'bogus'")
    assert model.outputs["neg"].startswith("refused: offset must be an integer >= 0")
    assert model.outputs["txt"].startswith("refused: offset must be an integer >= 0")
    assert out.coverage["bytes_read"] == 0 and out.coverage["parts_read"] == 0


def test_read_episode_says_its_page_is_content_bytes():
    desc = next(
        t for t in sol_tools(v21=True) if t["function"]["name"] == "read_episode"
    )
    assert "8000 bytes of content in all" in desc["function"]["description"]
