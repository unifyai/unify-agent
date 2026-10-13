"""Symbolic: with ``UNIFY_TOOLS_IN_CODE=on`` the actor and its teammates carry one model-API tool,
``execute_code(code)``; their cells run in one persistent interpreter per agent and call every other tool as
``tools.<name>(...)`` through the tool channel to the host handlers (unify/actor/in_code_tools.py).

The model is the scripted transport (tests/scripted_model.py): no key, no network. Cells run in the real
sandboxed worker. Each request's ``tool-events.jsonl`` (in its own folder under the harness's state directory)
has one row per cell and one per in-code call. When the request ends, no interpreter box and no handler thread
is left. With the switch off, nothing of this is built.
"""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path

import pytest

from tests.actor.code_act.core_world import core_world  # noqa: F401 (fixture)
from tests.actor.code_act.sandbox_world import ENV_SECRET, TOKEN_VALUE, needs_bwrap, world  # noqa: F401 (fixture)
from tests.scripted_model import ScriptedModel, reply, scripted
from unify.actor import in_code_tools
from unify.common import tool_registry as R
from unify.settings import SETTINGS

ANSWER = "the answer"

FIRST_CELL = (
    "page = tools.read('/workspace/data.txt')\n"
    "print('PAGE', page.splitlines()[:3])\n"
    "print('HITS', tools.grep('line 3'))\n"
    "print('WHO', tools.who())\n"
    "kept = 41\n"
)
SECOND_CELL = (
    "for path in ('/etc/passwd', '/workspace/.env'):\n"
    "    try:\n"
    "        tools.read(path)\n"
    "        print('READ', path)\n"
    "    except tools.Refused as exc:\n"
    "        print('REFUSED', exc.form, str(exc)[:200])\n"
    "print('KEPT', kept + 1)\n"
    f"tools.finish({ANSWER!r})\n"
)


def code(text: str):
    """An ``execute_code`` call with its one argument."""
    return ("execute_code", {"code": text})


class _Collect:
    """Stands in for :data:`in_code_tools.SESSIONS`: keeps every session a test opens."""

    def __init__(self) -> None:
        self.opened: list = []

    def add(self, session) -> None:
        self.opened.append(session)


@pytest.fixture
def in_code(core_world, monkeypatch):  # noqa: F811
    """The switch on, a persistent interpreter, a memory run (no stored-function library), a ``.env`` file in the
    workspace, and every in-code session the test opens."""
    monkeypatch.setattr(SETTINGS, "UNIFY_TOOLS_IN_CODE", "on")
    monkeypatch.setattr(SETTINGS, "UNIFY_CELL_STATE", "persistent")
    monkeypatch.setattr(SETTINGS, "UNIFY_MEMORY", "on")
    (core_world["workspace"] / ".env").write_text(f"KEY={ENV_SECRET}\n")
    collect = _Collect()
    monkeypatch.setattr(in_code_tools, "SESSIONS", collect)
    yield {"world": core_world, "opened": collect.opened}


async def _act(model: ScriptedModel, request: str) -> str:
    from unify.actor.code_act_actor import CodeActActor

    actor = CodeActActor()
    try:
        with scripted(model):
            handle = await actor.act(request, persist=False)
            return await asyncio.wait_for(handle.result(), 240)
    finally:
        await actor.close()


def _rows(session) -> list[dict]:
    path = in_code_tools.request_folder(session.request_id) / "tool-events.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines()]


def _channel_rows(rows: list[dict]) -> list[dict]:
    return sorted((r for r in rows if r.get("via") == "channel"), key=lambda r: r["call"])


def _text(content) -> str:
    if isinstance(content, list):
        return "\n".join(str(p.get("text", "")) if isinstance(p, dict) else str(p) for p in content)
    return "" if content is None else str(content)


def _tool_results(call) -> str:
    return "\n".join(_text(m.get("content")) for m in call.messages if m.get("role") == "tool")


def _why(model: ScriptedModel) -> str:
    """The last request's tool results, line by line (a failing assertion shows each line)."""
    return "\n" + "\n".join(_tool_results(model.calls[-1]).splitlines()[-60:]) if model.calls else ""


def _assert_one_tool(call) -> None:
    assert call.request["tools"] == R.json_tools("actor")


OPENING_HEADING = "## How you work here"
KEPT_SENTENCE = "keeps its variables between cells"


def _system_texts(call) -> list[str]:
    return [_text(m.get("content")) for m in call.messages if m.get("role") == "system"]


def _assert_opening_first(call, session, world: dict) -> str:
    """The request's system message starts with the agent's shared opening, then its own prompt; the opening holds
    box paths only. Returns the system message."""
    systems = _system_texts(call)
    assert systems, "no system message"
    system = systems[0]
    opening = session.opening()
    assert system.startswith(opening + "\n\n"), system[:600]
    assert opening.startswith(OPENING_HEADING + "\n")
    assert "`tools.list()` prints one line per tool you have" in opening
    assert "\n".join(systems).count(OPENING_HEADING) == 1
    names = R.tool_names_in(opening) - {"list", "help", "Refused", "NotRun"}
    assert names <= set(session.allow), sorted(names - set(session.allow))
    for value in (ENV_SECRET, TOKEN_VALUE, os.path.expanduser("~")):
        assert value not in opening
    for root in ("home", "state", "workspace"):
        for path in {str(world[root]), os.path.realpath(world[root])}:
            assert path not in opening, root
    return system


def _assert_ended(opened: list) -> None:
    assert opened
    for session in opened:
        assert session.closed
        assert not session.box_alive()
        assert not session.threads_alive()
        assert session.interp is None


def _assert_no_host_text(model: ScriptedModel, home: Path) -> None:
    for call in model.calls:
        sent = json.dumps(call.request)
        assert "stored function" not in sent.lower()
        for root in {str(home), os.path.realpath(home)}:
            assert root not in sent
        assert ENV_SECRET not in sent


@needs_bwrap
@pytest.mark.asyncio
@pytest.mark.timeout(300)
async def test_the_actor_calls_its_tools_from_cells_and_finishes_with_an_answer(in_code):
    # the last entry ends the request if finish did not, so a failure shows the cells' output
    model = ScriptedModel(
        actor=[reply(calls=[code(FIRST_CELL)]), reply(calls=[code(SECOND_CELL)]), reply("not finished")],
    )
    result = await _act(model, "Look at data.txt and answer.")

    assert result == ANSWER, _why(model)
    # tools.finish ended the request: no model call after the second cell
    assert model.unused() == {"actor": 1}
    for call in model.calls:
        _assert_one_tool(call)
    shown = _tool_results(model.calls[1])
    assert "PAGE" in shown and "line 1" in shown, shown[:2000]
    assert "/workspace/data.txt:3" in shown, shown[:2000]
    assert "agents on the record" in shown
    _assert_no_host_text(model, in_code["world"]["state"])

    (session,) = in_code["opened"]
    assert session.agent == "actor"
    assert "delegate" in session.allow and "ask" not in session.allow
    # every request of the actor opens with the shared opening, the same bytes each time
    system = _assert_opening_first(model.calls[0], session, in_code["world"])
    assert KEPT_SENTENCE in session.opening()
    assert all(_system_texts(c)[0] == system for c in model.calls)
    rows = _rows(session)
    cells = sorted((r for r in rows if r["tool"] == "execute_code"), key=lambda r: r["call"])
    calls = _channel_rows(rows)
    assert [r["via"] for r in cells] == ["model", "model"]
    assert sorted(r["call"] for r in rows) == list(range(len(rows)))
    assert [r["tool"] for r in calls] == ["read", "grep", "who", "read", "read", "finish"]
    assert {r["agent"] for r in rows} == {"actor"}
    assert {r["request_id"] for r in rows} == {session.request_id}
    first, second = (c["call"] for c in cells)
    assert [r["cell"] for r in calls] == [first] * 3 + [second] * 3
    # a refused call raised tools.Refused in the cell, with its form; the cell went on to finish
    refused = [r for r in calls if r["status"] == "refused"]
    assert [r["refusal"]["form"] for r in refused] == ["outside_roots", "unreadable"]
    assert calls[-1]["status"] == "ok"
    assert all(len(r["args_sha256"]) == 64 for r in calls)
    assert ENV_SECRET not in json.dumps(rows)
    _assert_ended(in_code["opened"])


@needs_bwrap
@pytest.mark.asyncio
@pytest.mark.timeout(300)
async def test_a_refused_call_raises_in_the_cell(in_code):
    probe = (
        "for path in ('/etc/passwd', '/workspace/.env'):\n"
        "    try:\n"
        "        tools.read(path)\n"
        "        print('READ', path)\n"
        "    except tools.Refused as exc:\n"
        "        print('REFUSED', exc.form, str(exc)[:200])\n"
        "tools.read('/etc/passwd')\n"
    )
    model = ScriptedModel(actor=[reply(calls=[code(probe)]), reply("done")])
    assert await _act(model, "Read two files.") == "done"
    shown = _tool_results(model.calls[1])
    assert "REFUSED outside_roots" in shown and "/etc/passwd is outside /workspace" in shown, shown[:2000]
    assert "REFUSED unreadable" in shown and "READ /workspace/.env" not in shown
    # uncaught, the refusal is the cell's error, with its text
    assert "tools.Refused: refused: /etc/passwd is outside /workspace" in shown, _why(model)
    assert ENV_SECRET not in shown
    _assert_ended(in_code["opened"])


TEAMMATE_CELL = (
    "print('TEAM', tools.read('/workspace/data.txt').splitlines()[2])\n"
    "print('WHO', tools.who())\n"
    "tools.finish('the third line is line 3')\n"
)
ACTOR_START = "print(tools.delegate('Read /workspace/data.txt and tell me its third line, please.'))\n"
ACTOR_WAIT = (
    "got = tools.wait(timeout=120)\n"
    "print('GOT', got)\n"
    "done = [line for line in got.splitlines() if '(done)' in line]\n"
    "tools.finish(done[0].split(': ', 1)[1])\n"
)


@needs_bwrap
@pytest.mark.asyncio
@pytest.mark.timeout(300)
async def test_a_teammate_gets_the_same_wiring_with_its_own_allow_list(in_code):
    model = ScriptedModel(
        actor=[reply(calls=[code(ACTOR_START)]), reply(calls=[code(ACTOR_WAIT)]), reply("not finished")],
        helper=[reply(calls=[code(TEAMMATE_CELL)]), reply("not finished")],
    )
    result = await _act(model, "Ask a teammate for the third line of data.txt.")

    assert result == "the third line is line 3", _why(model)
    assert sorted(model.kinds()) == ["actor", "actor", "helper"]
    assert model.unused() == {"actor": 1, "helper": 1}
    for call in model.calls:
        _assert_one_tool(call)
    _assert_no_host_text(model, in_code["world"]["state"])

    by_agent = {s.agent: s for s in in_code["opened"]}
    assert set(by_agent) == {"actor", "teammate"}
    mate = by_agent["teammate"]
    # the teammate's first request opens with its own opening (its allow-list, its end), the actor's with the actor's
    mate_system = _assert_opening_first(model.of("helper")[0], mate, in_code["world"])
    actor_system = _assert_opening_first(model.of("actor")[0], by_agent["actor"], in_code["world"])
    assert mate.opening() != by_agent["actor"].opening()
    assert "answers whoever started you" in mate_system and "answers whoever started you" not in actor_system
    assert mate.allow == R.allow_list("teammate", frozenset({"helper"}))
    assert "delegate" in mate.allow and "ask" in mate.allow  # any agent may start agents
    assert mate.state.spawner == by_agent["actor"].binding.name
    rows = _rows(mate)
    assert {r["agent"] for r in rows} == {"teammate"}
    calls = _channel_rows(rows)
    assert [r["tool"] for r in calls] == ["read", "who", "finish"]
    assert [r["tool"] for r in _channel_rows(_rows(by_agent["actor"]))] == ["delegate", "wait", "finish"]
    _assert_ended(in_code["opened"])


@needs_bwrap
@pytest.mark.asyncio
@pytest.mark.timeout(300)
async def test_off_the_actor_is_as_before(core_world, monkeypatch):  # noqa: F811
    monkeypatch.setattr(SETTINGS, "UNIFY_TOOLS_IN_CODE", "off")
    monkeypatch.setattr(SETTINGS, "UNIFY_MEMORY", "on")
    collect = _Collect()
    monkeypatch.setattr(in_code_tools, "SESSIONS", collect)
    cell = ("execute_code", {"thought": "Counting.", "code": "print(sum(range(4)))"})
    model = ScriptedModel(actor=[reply(calls=[cell]), reply("6")])
    result = await _act(model, "Sum 0..3.")

    assert result == "6" and collect.opened == []
    assert not any(OPENING_HEADING in t for c in model.calls for t in _system_texts(c))
    (tool,) = model.calls[0].request["tools"]
    params = tool["function"]["parameters"]["properties"]
    assert "thought" in params and "code" in params
    assert model.calls[0].request["tools"] != R.json_tools("actor")
    assert "6" in _tool_results(model.calls[1])
    assert not in_code_tools.request_folder("x").parent.exists()


# ── the parts, without a sandbox ─────────────────────────────────────────────


def test_the_channel_table_takes_the_actor_handlers_and_schemas():
    allow = R.allow_list("actor", frozenset({"helper"}))
    specs = in_code_tools.channel_specs("actor", allow)
    assert "sh" in allow and "sh" not in specs  # run in the box by the stub
    for name, spec in specs.items():
        handler, blocking = R.handler_for("actor", name)
        assert (spec.handler, spec.blocking) == (handler, blocking)
        assert spec.parameters == R.strict_parameters("actor", name)
    assert specs["read"].handler == "unify.actor.tool_handlers:read"
    assert set(specs["finish"].parameters["properties"]) == {"answer"}


def test_the_event_file_is_created_new_and_never_through_a_link(tmp_path):
    target = tmp_path / "elsewhere.jsonl"
    target.write_text("")
    folder = tmp_path / "requests" / "r1"
    folder.mkdir(parents=True)
    (folder / "tool-events.jsonl").symlink_to(target)
    linked = in_code_tools.EventFile(folder)
    linked.append({"event": "tool_call", "call": 0})
    assert target.read_text() == ""

    fresh = in_code_tools.EventFile(tmp_path / "requests" / "r2")
    fresh.append({"event": "tool_call", "call": 0})
    fresh.append({"event": "tool_call", "call": 1})
    lines = (tmp_path / "requests" / "r2" / "tool-events.jsonl").read_text().splitlines()
    assert [json.loads(x)["call"] for x in lines] == [0, 1]


@needs_bwrap
@pytest.mark.asyncio
@pytest.mark.timeout(300)
async def test_fresh_cell_state_starts_every_cell_in_a_new_interpreter(in_code, monkeypatch):
    monkeypatch.setattr(SETTINGS, "UNIFY_CELL_STATE", "fresh")
    model = ScriptedModel(
        actor=[
            reply(calls=[code("kept = 1\nprint('SET')")]),
            reply(calls=[code("print('HAS', 'kept' in dir())")]),
            reply("done"),
        ],
    )
    assert await _act(model, "Twice.") == "done"
    assert "HAS False" in _tool_results(model.calls[2]), _why(model)
    _assert_ended(in_code["opened"])


def _session(tmp_path, **kw):
    from types import SimpleNamespace

    return in_code_tools.InCodeTools(
        agent="actor",
        name="root",
        record=SimpleNamespace(),
        workspace_root=str(tmp_path),
        shadow={},
        cell_limit_s=None,
        events_folder=tmp_path / "events",
        **kw,
    )


@pytest.mark.asyncio
async def test_the_default_cell_state_is_the_one_the_opening_states(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from unify.common import box_interpreter
    from unify.settings import ProductionSettings

    default = ProductionSettings.model_fields["UNIFY_CELL_STATE"].default
    monkeypatch.setattr(SETTINGS, "UNIFY_CELL_STATE", default)
    monkeypatch.setattr(SETTINGS, "UNIFY_MEMORY", "off")
    assert box_interpreter.cell_state() == "persistent" and box_interpreter.persistent()
    planted = tmp_path / "planted-run-root"
    monkeypatch.setenv("UNIFY_HOME", str(planted))
    monkeypatch.setenv("PLANTED_VALUE_FOR_THE_OPENING_TEST", "planted-value-for-the-opening-test")
    actor = _session(tmp_path)
    mate = in_code_tools.InCodeTools(
        agent="teammate",
        name="mate",
        record=SimpleNamespace(),
        workspace_root=str(tmp_path),
        shadow={},
        cell_limit_s=30.0,
        events_folder=tmp_path / "mate-events",
    )
    try:
        for session in (actor, mate):
            opening = session.opening()
            assert opening.startswith(OPENING_HEADING + "\n") and KEPT_SENTENCE in opening
            assert "fresh interpreter" not in opening
            for value in (str(tmp_path), str(planted), "planted-value-for-the-opening-test"):
                assert value not in opening
            assert "/workspace" in opening
            assert opening == session.opening()  # constant for the run: a stable prefix
        assert "A cell runs for at most 30 s." in mate.opening()
        assert "A cell runs for at most" not in actor.opening()  # no cell limit set
        assert "The library is importable" not in actor.opening()
        monkeypatch.setattr(SETTINGS, "UNIFY_CELL_STATE", "fresh")
        assert "each call runs a Python cell in a fresh interpreter" in actor.opening()
    finally:
        await actor.close(completed=True)
        await mate.close(completed=True)


@pytest.mark.asyncio
async def test_the_end_awaits_work_a_call_started_or_cancels_it(tmp_path):
    done = _session(tmp_path)
    finished = asyncio.ensure_future(asyncio.sleep(0.05, result="stored"))
    done.state.host_tasks.add(finished)
    await done.close(completed=True)
    assert finished.done() and finished.result() == "stored"

    stopped = _session(tmp_path)
    forever = asyncio.ensure_future(asyncio.sleep(3600))
    stopped.state.host_tasks.add(forever)
    await stopped.close(completed=False)
    assert forever.cancelled()
    assert stopped.closed and not stopped.threads_alive()
