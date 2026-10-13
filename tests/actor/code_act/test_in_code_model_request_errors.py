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
    "tools.finish('tagged')\n"
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
