"""Symbolic: with ``UNIFY_TOOLS_IN_CODE=on`` a cell's row in ``tool-events.jsonl`` counts the cell's model calls the
provider refused for the request's own content (``model_request_errors``), and the cell goes on.

The model is the scripted transport below unillm (tests/scripted_model.py): no key, no network. Cells run in the
real sandboxed worker (the fixtures of tests/actor/code_act/test_in_code_tools_wiring.py).
"""

from __future__ import annotations

import json

import pytest

from tests.actor.code_act.core_world import core_world  # noqa: F401 (fixture)
from tests.actor.code_act.sandbox_world import needs_bwrap, world  # noqa: F401 (fixture)
from tests.actor.code_act.test_in_code_tools_wiring import (  # noqa: F401 (fixture)
    _act,
    _assert_ended,
    _rows,
    _tool_results,
    _why,
    code,
    in_code,
)
from tests.common.test_cell_model_request_errors import PROVIDER_MESSAGE, SCHEMA, bad_schema
from tests.scripted_model import ScriptedModel, reply

REFUSED_CELL = (
    f"schema = {SCHEMA!r}\n"
    "for attempt in range(2):\n"
    "    try:\n"
    "        run_coro_sync(query_llm('Tag this.', response_format=schema))\n"
    "    except ModelRequestError as exc:\n"
    "        print('CAUGHT', isinstance(exc, ValueError), exc)\n"
    "print('AFTER')\n"
)
FIXED_CELL = (
    "schema['json_schema']['schema']['properties']['tags']['items'] = {'type': 'string'}\n"
    "print('TAGS', run_coro_sync(query_llm('Tag this.', response_format=schema)))\n"
    "tools.answer('tagged')\n"
)


@needs_bwrap
@pytest.mark.asyncio
@pytest.mark.timeout(300)
async def test_a_cells_request_errors_are_counted_on_its_row(in_code):
    model = ScriptedModel(
        actor=[reply(calls=[code(REFUSED_CELL)]), reply(calls=[code(FIXED_CELL)]), reply("not finished")],
        query_llm=[bad_schema(), bad_schema(), reply('{"tags": ["a"]}')],
    )
    assert await _act(model, "Tag it.") == "tagged", _why(model)
    assert model.kinds() == ["actor", "query_llm", "query_llm", "actor", "query_llm"]
    # the second actor turn read the first cell's output: both refusals caught, and the cell went on
    shown = _tool_results(model.calls[3])
    assert shown.count(f"CAUGHT True {PROVIDER_MESSAGE}") == 2, shown[-2000:]
    assert "AFTER" in shown, shown[-2000:]

    (session,) = in_code["opened"]
    cells = sorted((r for r in _rows(session) if r["tool"] == "execute_code"), key=lambda r: r["call"])
    assert [r["model_request_errors"] for r in cells] == [2, 0]
    assert [r["error"] for r in cells] == [False, False]
    # a count only: the provider's text is not in the evidence
    assert PROVIDER_MESSAGE not in json.dumps(_rows(session))
    _assert_ended(in_code["opened"])


FAILED_CELL = (
    "for attempt in range(2):\n"
    "    try:\n"
    "        run_coro_sync(query_llm('Tag this.'))\n"
    "    except ModelCallFailed as exc:\n"
    "        print('FAILED', isinstance(exc, RuntimeError), exc)\n"
    "print('AFTER')\n"
)
UNCAUGHT_CELL = "run_coro_sync(query_llm('Tag this.'))\n"


@needs_bwrap
@pytest.mark.asyncio
@pytest.mark.timeout(300)
async def test_an_unclassified_provider_error_reaches_the_cell_as_a_fixed_text_only(in_code, monkeypatch):
    import litellm

    from tests.common.test_cell_model_request_errors import PLANTED, assert_nothing_planted, planted
    from unify.actor.execution.worker import PythonWorker

    # every frame the harness sends the worker
    sent: list[str] = []
    send = PythonWorker._send

    async def recording_send(self, msg):
        sent.append(json.dumps(msg))
        return await send(self, msg)

    monkeypatch.setattr(PythonWorker, "_send", recording_send)
    answer = "tools.answer('failed')\n"
    model = ScriptedModel(
        actor=[
            reply(calls=[code(FAILED_CELL)]),
            reply(calls=[code(UNCAUGHT_CELL)]),
            reply(calls=[code(answer)]),
            reply("not finished"),
        ],
        query_llm=[planted(), planted(litellm.RateLimitError), planted(litellm.InternalServerError)],
    )
    assert await _act(model, "Tag it.") == "failed", _why(model)
    assert model.kinds() == ["actor", "query_llm", "query_llm", "actor", "query_llm", "actor"]
    shown = _tool_results(model.calls[3])
    assert "FAILED True the model call failed (status 400, BadRequestError)" in shown, shown[-2000:]
    assert "FAILED True the model call failed (status 429, RateLimitError)" in shown, shown[-2000:]
    assert "AFTER" in shown, shown[-2000:]
    uncaught = _tool_results(model.calls[5])
    assert "ModelCallFailed" in uncaught and "status 500, InternalServerError" in uncaught, uncaught[-2000:]

    # the worker got the fixed text, and nothing of the provider's reply reached a frame, the transcript or the rows
    assert any("ModelCallFailed" in frame for frame in sent)
    for frame in sent:
        assert_nothing_planted(frame)
    for call in model.calls:
        assert_nothing_planted(json.dumps(call.request, default=str))
    (session,) = in_code["opened"]
    assert_nothing_planted(json.dumps(_rows(session)))
    assert PLANTED["fingerprint"] not in json.dumps([c.messages for c in model.calls], default=str)
    _assert_ended(in_code["opened"])
