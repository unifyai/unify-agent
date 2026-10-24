"""Answering from a function: ``tools.answer_from(fn, input=None)`` and the request's declared input (rev 2.3).

``answer_from`` runs a function of the agent's session once, in its box, on the request's input, and answers with what
it returns through ``tools.answer`` (the same path and the same response-format check). The input is the value the
world declared beside a message (``{"message": ..., "input": ...}`` on the CLI's stdin, kept as data, never parsed from
text), or the ``input=`` given; with neither the call raises and answers nothing. What it answered from is recorded on
the channel. The guide's examples line stays one line and names it. In the memory run, the declared input tells a use
on the request's real input from a use on any other value, and an answer made from an item names that item when the
world judges it.
"""

from __future__ import annotations

import asyncio
import json
import types

import pytest

from tests.memory.integration.test_request import _abort, _begin, mv2  # noqa: F401
from unify.common import box_answer_source, declared_input
from unify.common import tool_registry as R

FULL = frozenset({"pool", "similar", "helper", "edit", "store"})

#: A channel client that records every call and answers from a table: the declared input, and the answer's reply.
CLIENT = '''
class Result(str):
    paths = ()
    next = None
class Refused(Exception):
    pass
class NotRun(Refused):
    pass
class ToolError(Exception):
    pass
class ChannelClosed(Exception):
    pass
CALLS = []
DECLARED = {"declared": False}
EXAMPLES = []
REFUSE_ANSWER = []
def _call(tool, args):
    import json
    if tool == "declared_examples":
        return Result(json.dumps(EXAMPLES))  # read, never recorded among the calls
    CALLS.append((tool, args))
    if tool == "declared_input":
        return Result(json.dumps(DECLARED))
    if tool == "answer" and REFUSE_ANSWER:
        raise Refused("not answered: the answer does not match the required response format: x")
    return Result("%s answered" % tool)
'''
SHELL = "class ShellResult(str):\n    pass\n\n\ndef sh(command, timeout=None):\n    return command\n"


@pytest.fixture
def load(monkeypatch):
    monkeypatch.setattr(R, "client_source", lambda: CLIENT)
    monkeypatch.setattr(R, "shell_source", lambda: SHELL)

    def _load(agent: str = "actor") -> types.ModuleType:
        mod = types.ModuleType("tools")
        exec(compile(R.render_stub(R.allow_list(agent, FULL), agent), "<tools>", "exec"), mod.__dict__)
        return mod

    return _load


def _calls(tools) -> list:
    return tools._client["CALLS"]


def double(order):
    return {"count": 2 * len(order["prices"])}


# --- who has it --------------------------------------------------------------------------------------------------


def test_the_actor_and_a_teammate_have_it_in_their_box_and_no_memory_agent_does():
    spec = R.registry()["answer_from"]
    assert spec.kind == "local" and not spec.handler
    for agent in ("actor", "teammate"):
        assert "answer_from" in R.allow_list(agent, FULL), agent
    for agent in ("writer", "helper", "librarian"):
        assert "answer_from" not in R.allow_list(agent, FULL), agent
    # its channel tools are for the box's code only: on the channel, never in tools.list() or the stub
    assert {"declared_input", "answered_from"} <= set(R.channel_only("actor"))
    stub = R.render_stub(R.allow_list("actor", FULL), "actor")
    assert repr(box_answer_source.SOURCE) in stub
    assert "def declared_input(" not in stub and "def answered_from(" not in stub


def test_the_examples_line_stays_one_line_and_names_answer_from():
    assert "\n" not in R.EXAMPLES_ANSWER_LINE
    assert "`tools.answer_from`" in R.EXAMPLES_ANSWER_LINE and "`input=`" in R.EXAMPLES_ANSWER_LINE
    assert R.EXAMPLES_ANSWER_LINE.startswith(R.EXAMPLES_LINE[:-1])


# --- the call ----------------------------------------------------------------------------------------------------


def test_with_no_declared_input_and_none_given_it_raises_and_answers_nothing(load):
    tools = load()
    with pytest.raises(TypeError) as got:
        tools.answer_from(double)
    assert type(got.value).__name__ == "WrongCall" and "declares no input: pass input=" in str(got.value)
    assert [c[0] for c in _calls(tools)] == ["declared_input"]


def test_with_input_given_it_answers_the_functions_output(load):
    tools = load()
    reply = tools.answer_from(double, input={"prices": [2, 3]})
    assert reply == "answer answered"
    calls = _calls(tools)
    assert calls[0] == ("answer", {"answer": {"count": 4}})
    tool, row = calls[1]
    assert tool == "answered_from" and row["answered"] is True and row["source"] == "given"
    assert row["input"] == {"prices": [2, 3]} and row["output"] == {"count": 4} and row["item"] is None
    assert row["output_digest"] == declared_input.digest(declared_input.canonical({"count": 4}))
    assert row["checked"] is None and row["withheld"] is False  # no examples: nothing checked


def test_with_a_declared_input_it_answers_on_that_input(load):
    tools = load()
    tools._client["DECLARED"].update({"declared": True, "value": {"prices": [1]}})
    tools.answer_from(double)
    calls = _calls(tools)
    assert calls[0] == ("declared_input", {}) and calls[1] == ("answer", {"answer": {"count": 2}})
    assert calls[2][1]["source"] == "declared"


def test_what_fn_raises_reaches_the_caller_and_nothing_is_answered(load):
    tools = load()

    def broken(x):
        raise KeyError("prices")

    with pytest.raises(KeyError):
        tools.answer_from(broken, input={})
    assert _calls(tools) == []


def test_a_refused_answer_is_recorded_as_not_answered_and_raises(load):
    tools = load()
    tools._client["REFUSE_ANSWER"].append(True)
    with pytest.raises(tools.Refused):
        tools.answer_from(double, input={"prices": []})
    assert _calls(tools)[-1][0] == "answered_from" and _calls(tools)[-1][1]["answered"] is False


def test_an_item_of_the_library_is_named_in_the_record(load):
    tools = load()

    class Item:
        __memory_item__ = "memory.tables.count:count_rows"

        def __call__(self, rows):
            return len(rows)

    tools.answer_from(Item(), input=[1, 2, 3])
    assert _calls(tools)[-1][1]["item"] == "memory.tables.count:count_rows"


def test_a_large_value_is_recorded_by_its_digest_only(load):
    tools = load()
    big = "x" * (box_answer_source.KEPT_CHARS + 10)
    tools.answer_from(lambda s: s[:3], input=big)
    row = _calls(tools)[-1][1]
    assert row["input"] is None and row["kept"] is False and row["output"] == "xxx"
    assert row["input_digest"] == declared_input.digest(declared_input.canonical(big))


def test_a_library_item_carries_its_id_as_the_marker():
    from unify.memory.library_helper import Item

    item = Item("memory.a.b:f", "function", "s", "", "unchecked", "on-trial", "", "", (), ())
    assert item.__memory_item__ == "memory.a.b:f"


# --- the declared input -------------------------------------------------------------------------------------------


@pytest.fixture
def clean_declared():
    declared_input.clear()
    yield
    declared_input.clear()


def test_the_latest_declaration_is_kept_and_an_unkeepable_one_is_refused(clean_declared):
    assert declared_input.current() is None
    assert declared_input.declare({"b": (1, 2), "a": "x"})
    assert declared_input.current() == {"value": {"a": "x", "b": [1, 2]},
                                        "digest": declared_input.digest('{"a":"x","b":[1,2]}')}  # fmt: skip
    assert not declared_input.declare("y" * (declared_input.MAX_BYTES + 1))
    assert declared_input.current()["value"] == {"a": "x", "b": [1, 2]}


def test_the_channel_gives_the_declared_input(clean_declared):
    from unify.actor import tool_handlers as H

    assert json.loads(H.declared_input(None, {}).text) == {"declared": False}
    declared_input.declare([[1, 2]])
    assert json.loads(H.declared_input(None, {}).text) == {"declared": True, "value": [[1, 2]]}


def test_the_cli_keeps_the_input_declared_beside_a_message(monkeypatch, clean_declared):
    from tests.memory.integration import test_request as T

    submitted: list = []

    async def submit(self, line):
        submitted.append(line)

    monkeypatch.setattr(T._Handle, "submit", submit, raising=False)
    code, _out, _log, _handle = T._drive(
        monkeypatch,
        ["act", "--persist", "--jsonl", "--no-clarify", "--quiet", "Say hi"],
        b'{"message": "the next one", "input": {"rows": [[1, 2]]}}\n{"quit": true}\n',
        None,
    )
    assert code == 0 and submitted == ["the next one"]
    assert declared_input.current()["value"] == {"rows": [[1, 2]]}


# --- the memory run: the real input, and the item an answer was made from ----------------------------------------


A = "capture:00112233445566aa"


def _d(value) -> str:
    return declared_input.digest(declared_input.canonical(value))


def test_with_a_declared_input_only_calls_on_it_count(mv2, clean_declared):  # noqa: F811
    run = _begin(mv2, "the request")
    try:
        declared_input.declare({"rows": [1]})
        results = [
            {"item": A, "outcome": "ok", "digest": _d(5), "input": _d({"rows": [9]})},  # another value
            {"item": A, "outcome": "ok", "digest": _d(6), "input": _d({"rows": [1]})},  # the declared input
            {"item": A, "outcome": "ok", "digest": _d(7), "input": ""},  # no arguments: the request itself
            {"item": A, "outcome": "ok", "digest": _d(8), "input": _d("the request")},  # the request's text
            {"item": "memory.x.y:f", "outcome": "error", "digest": "", "input": _d({"rows": [9]})},
        ]
        run.note_calls({"calls": {"counts": {}, "results": results}})
        assert run.returned_here == {A: {_d(6), _d(7), _d(8)}}
        assert run.bad_uses() == {}  # the failure was on another value
    finally:
        _abort(mv2, run)


def test_without_a_declared_input_every_tracked_call_counts(mv2, clean_declared):  # noqa: F811
    run = _begin(mv2, "the request")
    try:
        run.note_calls({"calls": {"results": [{"item": A, "outcome": "refused", "digest": "", "input": _d(3)}]}})
        assert list(run.bad_uses()) == [A]
    finally:
        _abort(mv2, run)


def test_a_rejected_answer_made_from_an_item_is_a_bad_use_of_that_item(mv2, monkeypatch):  # noqa: F811
    run = _begin(mv2, "the request")
    try:
        monkeypatch.setattr(run, "_observations_now", lambda: 1)
        out = _d([[1, 2], [3, 4]])
        # the item's tracked call on the real input returned the answered value: the row names it with that proof
        run.note_calls({"calls": {"results": [{"item": A, "outcome": "ok", "digest": out, "input": ""}]}})
        run.note_answered_from({"item": A, "source": "declared", "input_digest": "a", "output_digest": out,
                                "input": None, "output": None, "kept": False, "answered": True})  # fmt: skip
        run.take_checker({"label": "fail"})
        assert run.answer_attributions[-1] == {"label": "fail", "obs": 1, "items": [A], "short_items": [],
                                               "how": "answered_from"}  # fmt: skip
        assert run.bad_here[A]["how"] == "answered_from"
    finally:
        _abort(mv2, run)


def test_a_row_naming_an_item_no_real_call_returned_its_value_proves_nothing(mv2, monkeypatch):  # noqa: F811
    # a forged row (a cell can reach the channel): no tracked call of A on the real input returned that value
    run = _begin(mv2, "the request")
    try:
        monkeypatch.setattr(run, "_observations_now", lambda: 1)
        run.note_answered_from({"item": A, "source": "declared", "input_digest": "a", "output_digest": _d([[9]]),
                                "input": None, "output": None, "kept": False, "answered": True})  # fmt: skip
        run.take_checker({"label": "fail"})
        assert run.bad_uses() == {} and all(r.get("how") != "answered_from" for r in run.answer_attributions)
    finally:
        _abort(mv2, run)


def _ctx(answer_text):
    return types.SimpleNamespace(agent="actor", state=types.SimpleNamespace(
        reply_slot=types.SimpleNamespace(text=answer_text)))  # fmt: skip


def test_the_answered_from_channel_records_only_a_row_whose_output_is_the_answer_received(mv2, monkeypatch):  # noqa: F811
    from unify.actor import tool_handlers as H
    from unify.memory.integration import request as RQ

    run = _begin(mv2, "the request")
    try:
        monkeypatch.setattr(RQ, "current", lambda: run)
        row = {"item": None, "fn": "f", "source": "given", "input_digest": "a", "output_digest": _d(2), "input": 1,
               "output": 2, "kept": True, "answered": True, "checked": None, "withheld": False}  # fmt: skip
        assert H.answered_from(_ctx("2"), row).text == "recorded"  # answered as its JSON text
        assert run.answered_from[-1]["output"] == 2
        assert H.answered_from(_ctx("two"), dict(row, output_digest=_d("two"))).text == "recorded"  # a text answer
        # forged: not the answer this request received, or no answer received at all
        for ctx in (_ctx("3"), _ctx(None), types.SimpleNamespace(agent="actor")):
            got = H.answered_from(ctx, row)
            assert getattr(got, "status", "") == "refused" and "not the answer" in got.text, got.text
        n = len(run.answered_from)
        # a withhold answered nothing, so nothing is compared (and it never counts for anything)
        H.answered_from(_ctx(None), dict(row, answered=False, withheld=True))
        assert len(run.answered_from) == n + 1
        assert getattr(H.answered_from(_ctx("2"), {"item": None}), "status", "") == "refused"
    finally:
        _abort(mv2, run)


def test_the_search_profile_reads_the_declared_input_first():
    from unify.memory import library_search as S
    from unify.memory.captures import shape

    value = {"rows": [[1, 2]]}
    assert S.request_profile("prose with no JSON", value) == S.shape_profile(shape(value))
    assert S.request_profile("prose with no JSON") != S.shape_profile(shape(value))


def test_a_result_event_with_another_shape_is_malformed():
    from unify.memory.integration.adapters import audit

    cell = audit.CellAudit.__new__(audit.CellAudit)
    cell._reset_calls()
    cell._id = lambda item: item
    cell._call_result((A, "ok", "0123456789abcdef", ""))
    cell._call_result((A, "error", "", "-"))
    for bad in [(A, "ok", "", ""), (A, "error", "0123", ""), (A, "maybe", "", ""), (A, "ok", "0a", "x y"), (A, "ok")]:
        cell._call_result(bad)
    assert [r["outcome"] for r in cell.result_records] == ["ok", "error"]
    assert cell.calls_malformed == 5


def test_the_box_reads_a_failure_as_on_the_real_input_by_the_same_rule(monkeypatch):
    from unify.memory import library_helper as LH

    monkeypatch.setattr(LH, "_declared_digest", lambda: None)
    assert LH._on_real_input(_d(9))
    monkeypatch.setattr(LH, "_declared_digest", lambda: _d({"rows": [1]}))
    monkeypatch.setattr(LH, "_current_request", lambda: ("the request", None))
    assert LH._on_real_input("") and LH._on_real_input(_d({"rows": [1]})) and LH._on_real_input(_d("the request"))
    assert not LH._on_real_input(_d({"rows": [9]})) and not LH._on_real_input("-")
    assert LH.input_digest((), {}) == "" and LH.input_digest(({"rows": [1]},), {}) == _d({"rows": [1]})
    assert LH.input_digest((1, 2), {}) == "-" and LH.input_digest((1,), {"k": 2}) == "-"
