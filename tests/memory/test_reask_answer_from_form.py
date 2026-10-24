"""A recalled procedure, without a blind answer (rev 2.3).

At a request's start the library's answer gives the value call, never an answer to the world. Only in the re-ask line,
only when the request holds at least one example pair given as data, and only for a recorded procedure that reads the
request's declared input (one call), the renderer writes the primary's call as the answer from it,
``tools.answer_from(memory.get('<id>'))``, which checks it on those examples before it answers. The librarian's own
fits keep the value call: a fit whose call goes through ``tools`` is still refused.
"""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import pytest

from tests.memory.test_reask_on_new_evidence import FIT_A, A, _ask_stub, _run
from unify.common import declared_input
from unify.memory import librarian as L

CALL = f"tools.answer_from(memory.get({A!r}))"


@pytest.fixture
def copy(tmp_path):
    d = tmp_path / "copy" / ".memory" / "captures" / A[8:]
    d.mkdir(parents=True)
    (d / "record.json").write_text(json.dumps({"reads": "declared_input"}))
    declared_input.clear()
    yield tmp_path / "copy"
    declared_input.clear()


def _reask(run) -> str:
    return asyncio.run(L.request_reask_line(run, "Your answer was wrong."))


def _gate_open(run):
    run.checker_seen.append({"label": "fail"})
    return run


def test_with_examples_the_reask_line_writes_the_answer_from_form(monkeypatch, copy):
    _ask_stub(monkeypatch, FIT_A, [])
    declared_input.add_examples([{"input": 1, "output": 2}])
    run = _gate_open(_run(copy_root=copy, reask_state=SimpleNamespace(request_text="r", open_librarian=object())))
    line = _reask(run)
    assert f"call: {CALL}" in line and run.reasks[-1]["answer_from"] == [A]


def test_without_examples_the_reask_line_keeps_the_value_call(monkeypatch, copy):
    _ask_stub(monkeypatch, FIT_A, [])
    run = _gate_open(_run(copy_root=copy, reask_state=SimpleNamespace(request_text="r", open_librarian=object())))
    line = _reask(run)
    assert "tools.answer_from" not in line and FIT_A["fits"][0]["call"] in line


def test_a_procedure_that_does_not_read_the_declared_input_keeps_the_value_call(monkeypatch, copy):
    (copy / ".memory" / "captures" / A[8:] / "record.json").write_text(json.dumps({"entry_point": {"form": "value"}}))
    _ask_stub(monkeypatch, FIT_A, [])
    declared_input.add_examples([{"input": 1, "output": 2}])
    run = _gate_open(_run(copy_root=copy, reask_state=SimpleNamespace(request_text="r", open_librarian=object())))
    assert "tools.answer_from" not in _reask(run)


def test_the_start_block_never_writes_the_answer_from_form(copy):
    declared_input.add_examples([{"input": 1, "output": 2}])
    assert "answer_from" not in L.start_block(FIT_A)


def test_a_librarian_fit_through_tools_is_still_refused():
    got, why = L.check_answer({"fits": [{"id": A, "call": CALL, "why": "w"}]}, [A])
    assert got is None and "is not through memory" in why
