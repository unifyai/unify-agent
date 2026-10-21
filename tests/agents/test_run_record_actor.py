"""The actor on the run record (UNIFY_AGENTS_RUN_RECORD=on, tools in code): its cells reach the record through the
comms tools, ``who`` shows the roster, a report to the writer lands on the record while the writer service is idle,
and the cached prefix (system message and tools) is the same across requests whatever the roster holds (the
redesign's A1 test 11 and its actor -> writer acceptance exchange).

Cells run in the real sandboxed worker; the model is scripted.
"""

from __future__ import annotations

import asyncio
import json

import pytest

from tests import cache_discipline_helpers as h
from tests.actor.code_act.core_world import (  # noqa: F401 (fixtures)
    core_world,
    new_actor,
    world,
)
from tests.actor.code_act.sandbox_world import needs_bwrap
from tests.helpers import _handle_project
from unify.agents import run_record as rr
from unify.common import tool_registry as R
from unify.db import store_home
from unify.settings import SETTINGS

REPORT = "@writer the served item item:notes/totals gave a wrong sum here: action:0"


def _text(message: dict) -> str:
    """A message's text (a string, or its text blocks joined)."""
    content = message.get("content")
    if isinstance(content, list):
        return "\n".join(b.get("text", "") for b in content if isinstance(b, dict))
    return content if isinstance(content, str) else ""


def _cell(code: str):
    return lambda: h.completion(
        calls=[("execute_code", {"thought": "t", "code": code})]
    )


async def _act(replies) -> tuple[list[dict], object]:
    actor = new_actor()
    with h.scripted(tuple(replies)) as provider:
        handle = await actor.act("Do the task.", persist=False)
        assert await asyncio.wait_for(handle.result(), 120) == "done"
    return provider.requests, handle


@pytest.fixture
def on_record(core_world, monkeypatch):  # noqa: F811
    monkeypatch.setattr(SETTINGS, "UNIFY_AGENTS_RUN_RECORD", "on")
    home = store_home()
    rec = rr.join_run_record(home)
    rec.join(
        "writer", "writer", state="idle"
    )  # the writer service, idle between passes
    return home, rec


@needs_bwrap
@pytest.mark.asyncio
@pytest.mark.timeout(240)
@_handle_project
async def test_the_actor_reports_to_the_writer_and_reads_the_roster_through_its_tools(
    on_record,
):
    home, _rec = on_record
    requests, handle = await _act(
        [
            _cell(f"print(tools.who())\nprint(tools.send({REPORT!r}))"),
            lambda: h.completion(content="done"),
        ],
    )
    name = handle.agents_pool.record.root
    assert name.startswith("actor-")
    rows = rr.read_entries(home)
    report = next(r for r in rows if r["author"] == name)
    assert report["mentions"] == ["writer"] and report["role"] == "actor"
    assert "item:notes/totals" in report["refs"]
    shown = json.dumps(requests[1]["messages"])
    assert "writer (writer, idle)" in shown and "sent to writer" in shown
    # the first message carried the common section and the actor's roster lines
    assert any(R.COMMON_SECTION in _text(m) for m in requests[0]["messages"] if m.get("role") == "user")
    # the actor left the record at its request's end
    roster = {r["name"]: r["state"] for r in rr.read_roster(home)["agents"]}
    assert roster[name] == "ended" and roster["writer"] == "idle"


@needs_bwrap
@pytest.mark.asyncio
@pytest.mark.timeout(240)
@_handle_project
async def test_the_cached_prefix_is_the_same_across_requests_whatever_the_roster_holds(
    on_record,
):
    """A1 test 11: the roster is never in the system message; two requests with different rosters send the same
    cached prefix (system message and tools)."""
    home, rec = on_record
    first, _ = await _act([lambda: h.completion(content="done")])
    rec.join("h9", "helper")  # someone new is on the record
    rec.join("librarian-0001", "librarian", episode="ep-x")
    second, _ = await _act([lambda: h.completion(content="done")])
    assert first[0]["tools"] == second[0]["tools"] == R.json_tools("actor")
    assert first[0]["messages"][0] == second[0]["messages"][0]
    assert first[0]["messages"][0]["role"] == "system"
    assert "h9" not in _text(second[0]["messages"][0])
    assert R.COMMON_SECTION not in _text(first[0]["messages"][0])
