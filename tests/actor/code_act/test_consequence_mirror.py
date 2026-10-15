"""``UNIFY_CONSEQUENCES``: each cell result ends with one line of measured facts (unify/actor/consequences.py).

The line holds the current request's model calls and USD so far (calls with no reported cost counted, never priced)
and, for each draft a cell of the request imported, the requests that imported it and its keep or skip record from
the deployment's draft record. Off, nothing of it exists: no line, the same prompt and tools.

The model is scripted (no key, no network); the end-to-end test runs the minimal arm's setting, cells in the real
sandboxed worker.
"""

from __future__ import annotations

import re
from decimal import Decimal
from types import SimpleNamespace

import pytest

from tests.actor.code_act.core_world import core_world, world  # noqa: F401 (fixtures)
from tests.actor.code_act.sandbox_world import needs_bwrap
from tests.actor.code_act.test_in_code_tools_wiring import _act, _Collect, _system_texts, _tool_results, _why, code
from tests.scripted_model import ScriptedModel, reply
from unify.actor import consequences, in_code_tools
from unify.common.llm_meter import RunMeter
from unify.settings import SETTINGS

FACTS = re.compile(r"\(so far in this request: (\d+) model calls?, ([0-9.]+) USD")


def test_the_mirror_is_off_by_default_and_needs_the_tools_in_code():
    assert not consequences.enabled(SimpleNamespace(UNIFY_CONSEQUENCES=""))
    assert consequences.enabled(SimpleNamespace(UNIFY_CONSEQUENCES=" on "))
    assert consequences.refusal(SimpleNamespace(UNIFY_CONSEQUENCES="on", UNIFY_TOOLS_IN_CODE="on")) is None
    refusal = consequences.refusal(SimpleNamespace(UNIFY_CONSEQUENCES="on", UNIFY_TOOLS_IN_CODE="off"))
    assert "UNIFY_CONSEQUENCES" in refusal and "UNIFY_TOOLS_IN_CODE" in refusal


def test_the_line_counts_the_requests_calls_and_cost_from_its_start():
    meter, mirror = RunMeter(), consequences.Mirror()
    meter.add("planning", prompt_tokens=10, completion_tokens=2, cost="0.0125")
    meter.add("planning", prompt_tokens=10, completion_tokens=2, cost=0.0005)
    assert mirror.line(meter, [], None) == "(so far in this request: 2 model calls, 0.013 USD)"
    mirror.request_ended(meter)
    assert mirror.line(meter, [], None) == "(so far in this request: 0 model calls, 0 USD)"
    meter.add("planning", prompt_tokens=10, completion_tokens=2, cost=None)
    assert mirror.line(meter, [], None) == (
        "(so far in this request: 1 model call, 0 USD (1 call with no reported cost))"
    )
    assert mirror.line(None, [], None) == ""


def test_the_draft_record_keeps_imports_and_decisions_and_the_line_shows_them(tmp_path):
    record = consequences.DraftRecord(tmp_path / "home" / consequences.RECORD)
    assert record.load() == {}
    record.imported("d1")
    record.imported("d1")
    record.decided("d1", ["a"], ["b"])
    record.decided("d1", ["b"], [])
    assert record.load()["d1"] == {"imports": 2, "kept": ["a", "b"], "skipped": []}
    line = consequences.Mirror().line(None, ["d1", "d2"], record)
    assert line == (
        "(drafts/d1.py: imported in 2 requests; kept: a, b; drafts/d2.py: imported in 0 requests; "
        "no keep or skip recorded)"
    )
    (tmp_path / "home" / consequences.RECORD).write_text("not json")
    assert record.load() == {}


@pytest.fixture
def minimal(core_world, monkeypatch):  # noqa: F811
    monkeypatch.setattr(SETTINGS, "UNIFY_MEMORY", "")
    monkeypatch.setattr(SETTINGS, "UNIFY_LIBRARY", "off")
    monkeypatch.setattr(SETTINGS, "UNIFY_TOOLS_IN_CODE", "on")
    monkeypatch.setattr(SETTINGS, "UNIFY_CELL_STATE", "persistent")
    collect = _Collect()
    monkeypatch.setattr(in_code_tools, "SESSIONS", collect)
    yield {"workspace": core_world["workspace"], "opened": collect.opened}


@needs_bwrap
@pytest.mark.asyncio
@pytest.mark.timeout(400)
async def test_on_each_cell_result_ends_with_the_facts_and_off_there_is_none(minimal, monkeypatch):
    monkeypatch.setattr(SETTINGS, "UNIFY_CONSEQUENCES", "on")
    model = ScriptedModel(actor=[reply(calls=[code("print('one')\n")]), reply(calls=[code("print('two')\n")]),
                                 reply("done")])  # fmt: skip
    assert await _act(model, "Print twice.") == "done", _why(model)
    first, second = _tool_results(model.calls[1]), _tool_results(model.calls[2])
    calls = [int(m.group(1)) for m in FACTS.finditer(second)]
    assert calls == [1, 2], second  # the first cell's line, then the second's
    assert "one" in first and FACTS.search(first)
    system = "\n".join(_system_texts(model.calls[0]))
    monkeypatch.setattr(SETTINGS, "UNIFY_CONSEQUENCES", "")
    off = ScriptedModel(actor=[reply(calls=[code("print('one')\n")]), reply("done")])
    assert await _act(off, "Print once.") == "done", _why(off)
    assert "so far in this request" not in _tool_results(off.calls[1])
    assert "\n".join(_system_texts(off.calls[0])) == system
    assert off.calls[0].request["tools"] == model.calls[0].request["tools"]


@needs_bwrap
@pytest.mark.asyncio
@pytest.mark.timeout(400)
async def test_an_imported_draft_shows_its_record(minimal, monkeypatch):
    monkeypatch.setattr(SETTINGS, "UNIFY_DRAFTS", "on")
    monkeypatch.setattr(SETTINGS, "UNIFY_CONSEQUENCES", "on")
    build = ScriptedModel(actor=[reply(calls=[code("def double(v):\n    return v * 2\ntools.answer('ok')\n")])])
    assert await _act(build, "Build.") == "ok", _why(build)
    (draft,) = sorted((minimal["workspace"] / "drafts").glob("d*.py"))
    use = ScriptedModel(actor=[reply(calls=[code(f"from drafts.{draft.stem} import double\nprint(double(2))\n")]),
                               reply("done")])  # fmt: skip
    assert await _act(use, "Use.") == "done", _why(use)
    shown = _tool_results(use.calls[1])
    assert f"drafts/{draft.stem}.py: imported in 1 request; no keep or skip recorded" in shown, shown
    assert Decimal(FACTS.search(shown).group(2)) >= 0
