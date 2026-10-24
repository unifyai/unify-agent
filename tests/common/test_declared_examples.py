"""The examples a request is given as data, and the check before an answer (rev 2.3).

A world adapter may send example pairs beside a message (``"examples": [{"input": ..., "output": ...}]`` on the CLI's
stdin): they are kept in the order received, each distinct pair once, and ``tools.examples()`` returns them (``[]`` in
a world that declares none; examples written only in a message's text are never parsed out). ``tools.answer_from``
first checks the function on each pair with the box-local check: those are check runs, never uses. When one fails it
answers nothing and gives the report back (a withhold), which is no refusal, no bad use and no re-ask event. The
opening's examples line stays one line, the same bytes whether or not a world declares any.
"""

from __future__ import annotations

import json

import pytest

from tests.common.test_answer_from import _calls, double, load  # noqa: F401
from tests.memory.integration.test_request import _abort, _begin, mv2  # noqa: F401
from unify.common import declared_input
from unify.common import tool_registry as R

FULL = frozenset({"pool", "similar", "helper", "edit", "store"})
PAIRS = [{"input": {"prices": [1]}, "output": {"count": 2}}, {"input": {"prices": [1, 2]}, "output": {"count": 4}}]


@pytest.fixture
def clean_declared():
    declared_input.clear()
    yield
    declared_input.clear()


def test_pairs_are_kept_in_order_each_once_and_a_malformed_list_is_refused(clean_declared):
    assert declared_input.examples() == []
    assert declared_input.add_examples(PAIRS[:1])
    assert declared_input.add_examples(PAIRS)  # the first again, then a new one
    assert declared_input.examples() == PAIRS
    for bad in ({"input": 1, "output": 2}, [{"input": 1}], [{"input": 1, "output": 2, "x": 3}], [[1, 2]]):
        assert not declared_input.add_examples(bad), bad
    assert declared_input.examples() == PAIRS


def test_examples_returns_the_pairs_as_data(load):
    tools = load()
    assert tools.examples() == []
    tools._client["EXAMPLES"].extend(PAIRS)
    assert tools.examples() == PAIRS
    report = tools.check_examples(double, tools.examples())
    assert report.ok


def test_answer_from_checks_on_the_pairs_first_and_answers_when_all_pass(load):
    tools = load()
    tools._client["EXAMPLES"].extend(PAIRS)
    tools.answer_from(double, input={"prices": [5, 6, 7]})
    calls = _calls(tools)
    assert calls[0] == ("answer", {"answer": {"count": 6}})
    assert calls[1][1]["checked"] == {"passed": 2, "total": 2} and calls[1][1]["withheld"] is False


def test_a_failing_pair_withholds_the_answer_and_gives_the_report(load):
    tools = load()
    tools._client["EXAMPLES"].extend(PAIRS + [{"input": {"prices": []}, "output": {"count": 1}}])
    reply = tools.answer_from(double, input={"prices": [5]})
    assert reply.startswith("not answered: fn did not pass this request's examples (passed 2 of 3)")
    assert "passed 2 of 3" in reply.split("\n\n", 1)[1]
    calls = _calls(tools)
    assert [c[0] for c in calls] == ["answered_from"]  # nothing was answered
    row = calls[0][1]
    assert row["withheld"] is True and row["answered"] is False and row["checked"] == {"passed": 2, "total": 3}


def test_the_cli_keeps_the_pairs_declared_beside_a_message(monkeypatch, clean_declared):
    from tests.memory.integration import test_request as T

    async def submit(self, line):
        pass

    monkeypatch.setattr(T._Handle, "submit", submit, raising=False)
    line = json.dumps({"message": "m", "examples": PAIRS, "input": {"prices": [3]}}).encode() + b'\n{"quit": true}\n'
    code, *_ = T._drive(monkeypatch, ["act", "--persist", "--jsonl", "--no-clarify", "--quiet", "Say hi"], line, None)
    assert code == 0 and declared_input.examples() == PAIRS
    assert declared_input.current()["value"] == {"prices": [3]}


def test_the_channel_gives_the_pairs(clean_declared):
    from unify.actor import tool_handlers as H

    assert json.loads(H.declared_examples(None, {}).text) == []
    declared_input.add_examples(PAIRS)
    assert json.loads(H.declared_examples(None, {}).text) == PAIRS


def test_the_line_is_one_line_and_its_bytes_do_not_depend_on_any_declaration(clean_declared):
    def opening():
        return R.preamble("actor", {"environment": "Your box: /workspace is writable.", "flags": FULL})

    before = opening()
    declared_input.add_examples(PAIRS)
    declared_input.declare({"prices": [3]})
    assert opening() == before and before.count(R.EXAMPLES_ANSWER_LINE) == 1
    assert "\n" not in R.EXAMPLES_ANSWER_LINE and "`tools.examples()`" in R.EXAMPLES_ANSWER_LINE


def test_a_withhold_is_no_bad_use_and_no_reask_event(mv2):  # noqa: F811
    run = _begin(mv2, "the request")
    try:
        run.note_answered_from({"item": "capture:00112233445566aa", "source": "given", "input_digest": "a",
                                "output_digest": "", "input": None, "output": None, "kept": False, "answered": False,
                                "checked": {"passed": 0, "total": 1}, "withheld": True})  # fmt: skip
        run.take_checker({"label": "fail"})
        assert run.bad_uses() == {}
        assert run.reask_events("the request") == ["outcome"]  # the verdict, not the withhold
    finally:
        _abort(mv2, run)


# --- a stored item under answer_from: its checks on the examples are check runs, never uses ----------------------


def test_a_stored_item_whose_examples_fail_is_withheld_with_no_use_and_no_failure_recorded(tmp_path):
    from tests.memory.test_in_request_bad_use import CHECK_PRELUDE
    from tests.memory.test_library_items_api import EVENTS, PAINT, _json, _library
    from unify.common import box_answer_source

    pairs = [{"input": [[1, 2]], "output": [[2, 1]]}, {"input": "not a grid", "output": []}]
    root = _library(tmp_path)
    got = _json(
        root,
        EVENTS
        + CHECK_PRELUDE
        + f"calls = []\n"
        f"def _call(tool, args):\n"
        f"    calls.append(tool)\n"
        f"    return json.dumps({pairs!r}) if tool == 'declared_examples' else 'recorded'\n"
        f"def _answer(value):\n"
        f"    calls.append('answer')\n"
        f"    return 'answered'\n"
        f"af = {{'_call': _call, '_answer': _answer, '_check_examples': check_examples, '_contract': None}}\n"
        f"exec({box_answer_source.SOURCE!r}, af)\n"
        f"item = memory.get({PAINT!r})\n"
        "start = len(seen)\n"
        "reply = af['answer_from'](item, input=[[3, 4]])\n"
        "during = [e for e in seen[start:] if e[1][0] != 'memory:get']\n"
        "print(json.dumps([reply, during, calls, memory.failed_here()]))\n",
    )
    reply, during, calls, failed = got
    assert reply.startswith("not answered: fn did not pass this request's examples (passed 1 of 2)"), reply
    # no tracked call, no result, no failure: the item's check runs leave no use (no item_use row, nothing metered)
    assert during == [] and failed == []
    assert calls == ["declared_examples", "answered_from"]  # read the pairs, recorded the withhold, answered nothing


# --- the first request's declaration (--declared): its first message is the command line, never a stdin line --------


def test_a_declared_file_declares_both_or_nothing(tmp_path, clean_declared):
    good = tmp_path / "d.json"
    good.write_text(json.dumps({"input": {"prices": [3]}, "examples": PAIRS}))
    assert declared_input.load_file(good) is None
    assert declared_input.current()["value"] == {"prices": [3]} and declared_input.examples() == PAIRS
    declared_input.clear()
    for doc, why in (
        ([1], "an object"),
        ({}, "an object"),
        ({"input": 1, "other": 2}, "an object"),
        ({"examples": [{"input": 1}]}, '"examples"'),
        ({"examples": PAIRS, "input": "y" * (declared_input.MAX_BYTES + 1)}, '"input"'),
    ):
        bad = tmp_path / "bad.json"
        bad.write_text(json.dumps(doc))
        got = declared_input.load_file(bad)
        assert got is not None and why in got, (doc, got)
        assert declared_input.current() is None and declared_input.examples() == [], doc  # all or nothing
    assert "cannot be read" in declared_input.load_file(tmp_path / "missing.json")


def test_the_cli_reads_the_declared_file_before_the_actor_starts(monkeypatch, tmp_path, clean_declared):
    from tests.memory.integration import test_request as T

    path = tmp_path / "d.json"
    path.write_text(json.dumps({"input": {"prices": [3]}, "examples": PAIRS}))
    argv = ["act", "--persist", "--jsonl", "--no-clarify", "--quiet", "--declared", str(path), "Say hi"]
    code, _out, log, _handle = T._drive(monkeypatch, argv, b'{"quit": true}\n', None)
    assert code == 0 and [e[0] for e in log][:2] == ["begin", "act"]
    assert declared_input.current()["value"] == {"prices": [3]} and declared_input.examples() == PAIRS


def test_without_the_flag_nothing_is_declared_and_a_bad_file_stops_the_run(monkeypatch, tmp_path, clean_declared):
    from tests.memory.integration import test_request as T

    declared_input.declare({"left": "over"})  # a request starts with nothing declared
    code, *_ = T._drive(monkeypatch, ["act", "--persist", "--jsonl", "--no-clarify", "--quiet", "Say hi"],
                        b'{"quit": true}\n', None)  # fmt: skip
    assert code == 0 and declared_input.current() is None and declared_input.examples() == []
    bad = tmp_path / "bad.json"
    bad.write_text("not json")
    code, _out, log, _handle = T._drive(
        monkeypatch, ["act", "--persist", "--jsonl", "--no-clarify", "--quiet", "--declared", str(bad), "Say hi"],
        b'{"quit": true}\n', None,
    )
    assert code == 2 and log == []  # the actor never started


# --- a closing turn sent as a session's first message: --final --------------------------------------------------


def test_with_final_the_first_request_asks_the_librarian_nothing(monkeypatch, clean_declared):
    import asyncio
    from types import SimpleNamespace

    from unify.memory import librarian as L

    asked: list = []

    async def ask(ctx, args):
        asked.append(args)
        return SimpleNamespace(status="ok", text='{"fits": []}')

    from unify.actor import tool_handlers as H

    monkeypatch.setattr(H, "ask_library", ask)
    monkeypatch.setattr(L, "library_items", lambda copy: (["capture:00112233445566aa"], set(), {}))
    monkeypatch.setattr("unify.memory.integration.request.current", lambda: SimpleNamespace(copy_root="/x"))
    state = SimpleNamespace(request_text="The episode is over.", open_librarian=object())
    declared_input.declare_first_final()
    assert asyncio.run(L.request_start_block(state)) == "" and asked == []
    declared_input.clear()
    assert asyncio.run(L.request_start_block(state)) != "" and len(asked) == 1


def test_the_cli_marks_the_first_request_final_only_with_the_flag(monkeypatch, clean_declared):
    from tests.memory.integration import test_request as T

    seen: list = []
    code, *_ = T._drive(monkeypatch, ["act", "--persist", "--jsonl", "--no-clarify", "--quiet", "--final", "Over."],
                        b'{"quit": true}\n', None)  # fmt: skip
    seen.append(declared_input.first_final())
    code2, *_ = T._drive(monkeypatch, ["act", "--persist", "--jsonl", "--no-clarify", "--quiet", "Say hi"],
                         b'{"quit": true}\n', None)  # fmt: skip
    seen.append(declared_input.first_final())
    assert (code, code2) == (0, 0) and seen == [True, False]


# --- declarations belong to their request ----------------------------------------------------------------------


def test_a_new_request_starts_with_nothing_declared_and_its_own_declarations_are_kept(clean_declared):
    from unify.cli import Act, _parse_args

    act = Act(_parse_args(["act", "--persist", "--jsonl", "--no-clarify", "--quiet", "Say hi"]))
    act._progress = lambda text: None
    act._take_declarations({"message": "m1", "input": [1], "examples": PAIRS[:1]})
    act._take_declarations({"message": "m2", "examples": PAIRS[1:]})  # the same request: added to
    assert declared_input.current()["value"] == [1] and declared_input.examples() == PAIRS
    act._turn_answered = True  # the session answered: the next message starts a new request
    act._take_declarations({"message": "next request"})
    assert declared_input.current() is None and declared_input.examples() == []
    act._turn_answered = True
    act._take_declarations({"message": "and another", "input": [2]})
    assert declared_input.current()["value"] == [2] and act._turn_answered is False
