"""Symbolic: ``UNIFY_BIND_REQUEST=on``: the request is a read-only ``request``.

In 309 accepted Continual-ARC answers only 39% were backed by code that
printed them, and that code held the request's input as a pasted literal in
about 88% of cases: the model retyped data the request already held. With
the switch on, model code in a cell reads the current request instead:
``request.text`` is the requester's latest message as the model reads it,
and ``request.data`` the JSON objects and arrays found in that text, in order
of appearance, parsed with the standard ``json`` module. Nothing else is
parsed and nothing keys on what the request is about.

Each cell gets a fresh object, so what a cell does to it never reaches a
later cell or turn. Off, the sandbox has no ``request`` and the prompt and
tools are as shipped.

The transport is scripted (``tests/cache_discipline_helpers.py``): nothing
leaves the process. Each test that drives a session bounds every wait.
"""

from __future__ import annotations

import asyncio
import base64
import json
import math
import os
import re
import sys
import time
from types import SimpleNamespace

import pytest

from tests import cache_discipline_helpers as h
from tests.actor.code_act.core_world import core_world  # noqa: F401
from tests.actor.code_act.sandbox_world import needs_bwrap, world  # noqa: F401
from unify.actor import core_surface, notebook_cells
from unify.actor import prompt_builders as pb
from unify.actor.execution.session import SessionExecutor
from unify.actor.execution.types import parts_to_text
from unify.settings import SETTINGS

SENTENCE = (
    "The current request is available in cells as `request`: `request.text` "
    "is its text and `request.data` the JSON values it contains, in order."
)
# Every session a test drives ends within this.
SESSION_BOUND_S = 2.0

FIRST = (
    'Sum the values in [1, 2, 3] scaled by {"scale": 2}.\n'
    "```json\n"
    '{"label": "first", "ids": ["7", 8]}\n'
    "```\n"
    "Reply with the total."
)
FIRST_DATA = [[1, 2, 3], {"scale": 2}, {"label": "first", "ids": ["7", 8]}]
SECOND = 'Now [[4, 5], [6]] with {"scale": "3"}; reply with the total.'
SECOND_DATA = [[[4, 5], [6]], {"scale": "3"}]

# A cell that reports what it sees, safe from any escaping on the way back.
REPORT_CELL = (
    "import base64, json\n"
    "try:\n"
    "    _seen = {'bound': True, 'text': request.text, 'data': request.data}\n"
    "except NameError:\n"
    "    _seen = {'bound': False}\n"
    "print('SEEN', base64.b64encode(json.dumps(_seen).encode()).decode())\n"
)


def _seen(text: str) -> list[dict]:
    return [
        json.loads(base64.b64decode(m.group(1)))
        for m in re.finditer(r"SEEN ([A-Za-z0-9+/=]+)", text)
    ]


def _flat(text: str) -> str:
    return " ".join(text.split())


@pytest.fixture
def bound(monkeypatch):
    monkeypatch.setattr(SETTINGS, "UNIFY_BIND_REQUEST", "on")
    monkeypatch.setattr(SETTINGS, "UNIFY_PROMPT_PROFILE", "")
    monkeypatch.setattr(SETTINGS, "UNIFY_TOOL_SURFACE", "")


# ── the switch ───────────────────────────────────────────────────────────


def test_the_switch_is_validated():
    from unify.settings import ProductionSettings

    assert ProductionSettings().UNIFY_BIND_REQUEST == ""
    assert ProductionSettings(UNIFY_BIND_REQUEST="off").UNIFY_BIND_REQUEST == ""
    assert ProductionSettings(UNIFY_BIND_REQUEST=" On ").UNIFY_BIND_REQUEST == "on"
    with pytest.raises(ValueError):
        ProductionSettings(UNIFY_BIND_REQUEST="yes")


# ── the parser ───────────────────────────────────────────────────────────


def _values(text: str) -> list:
    from unify.actor.execution.worker_child import json_values

    return json_values(text)


@pytest.mark.parametrize(
    "text, expected",
    [
        # JSON in prose, in order of appearance.
        ('Use {"a": 1} then [2, 3].', [{"a": 1}, [2, 3]]),
        # A fenced block, with or without its language tag.
        ('Data:\n```json\n{"k": [1, {"x": null}]}\n```\n', [{"k": [1, {"x": None}]}]),
        ("```\n[true, false]\n```", [[True, False]]),
        # A nested array is one value: the outermost one.
        ("[[0, 1], [1, 0]]", [[[0, 1], [1, 0]]]),
        ("[[0, 1], [1, 0]] and [[2]]", [[[0, 1], [1, 0]], [[2]]]),
        # Numbers stay numbers and strings stay strings.
        ('[1, "1", 1.5, -2e3, "x"]', [[1, "1", 1.5, -2000.0, "x"]]),
        # Brackets inside a JSON string are part of the string.
        ('{"note": "a ] or } here", "n": [1]}', [{"note": "a ] or } here", "n": [1]}]),
        # Invalid JSON is ignored; the scan goes on after it.
        ("{a: 1} [1, 2,] {'b': 2} [x] [3]", [[3]]),
        ('{"open": [1, 2', []),
        # Invalid JSON is ignored up to where it fails, values inside it too;
        # a value in non-JSON text before that point is still found.
        ('{"a": [1, 2], oops} [4]', [[4]]),
        ("{'a': [1, 2]}", [[1, 2]]),
        # Top-level scalars are not collected; nor are Markdown links.
        ('42 "text" true null [link](https://example.com)', []),
        ("", []),
        ("no data here", []),
        ("empty [] and {}", [[], {}]),
    ],
)
def test_json_values(text, expected):
    assert _values(text) == expected


def test_json_values_types_and_extremes():
    (value,) = _values('[1, 1.0, "1", 10000000000000000000000]')
    assert [type(v) for v in value] == [int, float, str, int]
    # The standard json module's own reading, NaN included.
    (value,) = _values("[NaN]")
    assert math.isnan(value[0])
    # Nesting too deep for the parser ends the scan, without an error.
    assert _values("[1] " + "[" * 100_000 + " [2]") == [[1]]
    assert _values("[1] " + "[1," * 100_000) == [[1]]
    deep = "[" * 50 + "]" * 50
    assert _values(deep) == [json.loads(deep)]


# ── the object ───────────────────────────────────────────────────────────


def test_the_request_object_is_read_only():
    from unify.actor.execution.worker_child import Request

    req = Request(FIRST)
    assert req.text == FIRST
    assert req.data == FIRST_DATA
    for name in ("text", "data", "other"):
        with pytest.raises(AttributeError, match="read-only"):
            setattr(req, name, "x")
    with pytest.raises(AttributeError, match="read-only"):
        del req.text
    # A renewed copy shares nothing with the old one.
    req.data[0].append(99)
    req.data.append("x")
    fresh = req.renewed()
    assert fresh.data == FIRST_DATA and fresh.text == FIRST
    assert "request" in repr(fresh) and "3 JSON values" in repr(fresh)


def test_the_requester_text_of_a_message():
    from unify.common._async_tool import bound_request

    assert bound_request.text_of("hi [1]") == "hi [1]"
    assert bound_request.text_of({"role": "user", "content": "x"}) == "x"
    assert bound_request.text_of({"role": "system", "content": "x"}) is None
    blocks = [{"type": "text", "text": "a [1]"}, {"type": "image_url"}]
    assert bound_request.text_of(blocks) == "a [1]"
    assert bound_request.text_of({"role": "user", "content": blocks}) == "a [1]"
    batch = [{"role": "user", "content": "old"}, "new [2]"]
    assert bound_request.text_of(batch) == "new [2]"


@pytest.mark.asyncio
@pytest.mark.timeout(10)
async def test_the_handle_keeps_the_request_for_a_restart(bound):
    """A loop restarted after context compression starts with a
    loop-authored message; the handle's slot keeps the request."""
    from unify.common._async_tool import bound_request
    from unify.common.async_tool_loop import start_async_tool_loop

    with h.scripted([h.completion(content="done")]):
        handle = start_async_tool_loop(
            h.new_client(),
            FIRST,
            {},
            log_steps=False,
            bind_request=True,
        )
        assert await asyncio.wait_for(handle.result(), SESSION_BOUND_S) == "done"
    slot = handle._loop_config["bind_request"]
    assert isinstance(slot, bound_request.RequestSlot)
    assert slot.text == FIRST
    # What the restart does with the handle's slot.
    token = bound_request.bind(slot)
    try:
        assert bound_request.current() is slot
        bound_request.record_first(
            slot,
            "Context was compressed. Continue from where you left off.",
        )
        assert slot.text == FIRST
        # A later requester message still replaces it.
        bound_request.record(slot, SECOND)
        assert slot.text == SECOND
    finally:
        bound_request.unbind(token)


def test_off_a_handle_passes_the_flag_unchanged(monkeypatch):
    from unify.common._async_tool import bound_request

    monkeypatch.setattr(SETTINGS, "UNIFY_BIND_REQUEST", "")
    assert bound_request.new_slot(True) is True
    assert bound_request.new_slot(False) is False
    assert bound_request.bind(True) is None


# ── the prompt ───────────────────────────────────────────────────────────


@pytest.fixture
def actor_tools():
    from unify.actor.code_act_actor import CodeActActor

    actor = CodeActActor()
    return actor, dict(actor.get_tools("act"))


def _prompt(actor_tools, surface: str) -> str:
    actor, tools = actor_tools
    if surface == "core":
        return pb.build_code_act_prompt(
            environments=actor.environments,
            tools={"execute_code": tools["execute_code"]},
            can_store=True,
            core=core_surface.PromptSurface(),
        )
    return pb.build_code_act_prompt(
        environments=actor.environments,
        tools=tools,
        can_store=True,
    )


@pytest.mark.parametrize("profile", ["", "lean"])
@pytest.mark.parametrize("surface", ["json", "core"])
@pytest.mark.parametrize("projection", ["", "notebook"])
def test_the_prompt_says_it_once_where_the_sandbox_is_described(
    monkeypatch,
    actor_tools,
    profile,
    surface,
    projection,
):
    monkeypatch.setattr(SETTINGS, "UNIFY_PROMPT_PROFILE", profile)

    def render() -> str:
        text = _prompt(actor_tools, surface)
        return notebook_cells.rewrite_prompt(text) if projection else text

    monkeypatch.setattr(SETTINGS, "UNIFY_BIND_REQUEST", "")
    off = render()
    monkeypatch.setattr(SETTINGS, "UNIFY_BIND_REQUEST", "on")
    on = render()
    assert "request.data" not in off
    assert _flat(on).count(SENTENCE) == 1
    # In the sandbox section, before its table of globals.
    section = on[on.index("### Sandbox Environment") :]
    assert _flat(section).index(SENTENCE) < _flat(section).index("| Global |")
    assert on.replace(_sentence_as_written(on) + "\n\n", "", 1) == off


def _sentence_as_written(prompt: str) -> str:
    start = prompt.index("The current request is available")
    return prompt[start : prompt.index("in order.", start) + len("in order.")]


def test_the_tools_are_unchanged_by_the_switch(monkeypatch):
    from unify.actor.code_act_actor import CodeActActor
    from unify.common.llm_helpers import method_to_schema

    def schemas() -> dict:
        tools = CodeActActor().get_tools("act")
        return {
            name: method_to_schema(getattr(tool, "fn", tool), name)
            for name, tool in tools.items()
        }

    monkeypatch.setattr(SETTINGS, "UNIFY_BIND_REQUEST", "")
    off = schemas()
    monkeypatch.setattr(SETTINGS, "UNIFY_BIND_REQUEST", "on")
    assert schemas() == off


# ── the cell, in process and in the worker ───────────────────────────────


async def _run(ex: SessionExecutor, code: str) -> tuple[str, dict]:
    res = await asyncio.wait_for(
        ex.execute(code=code, state_mode="stateful", session_id=0),
        SESSION_BOUND_S,
    )
    return parts_to_text(res["stdout"]), res


async def _cell_checks(ex: SessionExecutor) -> None:
    """What a cell sees of the request, wherever the cell runs."""
    from unify.common._async_tool import bound_request

    token = bound_request.bind(True)
    try:
        slot = bound_request.current()
        # No request yet: no ``request``.
        out, res = await _run(ex, REPORT_CELL)
        assert _seen(out) == [{"bound": False}], res
        slot.text = FIRST
        out, res = await _run(ex, REPORT_CELL)
        assert _seen(out) == [{"bound": True, "text": FIRST, "data": FIRST_DATA}]
        # Read-only attributes; a cell's changes stay in that cell.
        out, res = await _run(ex, "request.text = 'changed'")
        assert "read-only" in str(res["error"])
        out, res = await _run(ex, "request.data = []")
        assert "read-only" in str(res["error"])
        out, res = await _run(
            ex,
            "kept = request.data\n"
            "request.data[0].append(99)\n"
            "request.data.append('x')\n"
            "print(len(request.data))\n",
        )
        assert res["error"] is None and out == "4\n"
        out, res = await _run(
            ex,
            "print(kept is request.data, len(kept))\n" + REPORT_CELL,
        )
        assert out.startswith("False 4\n")
        assert _seen(out)[0]["data"] == FIRST_DATA
        # The next request replaces the first.
        slot.text = SECOND
        out, res = await _run(ex, REPORT_CELL)
        assert _seen(out) == [{"bound": True, "text": SECOND, "data": SECOND_DATA}]
        slot.text = FIRST
        # A variable of the model's own named ``request`` (HTTP code's
        # ``request = {...}``) is left alone, in later cells and after the
        # next request arrives.
        out, res = await _run(ex, "request = {'url': 'x'}")
        assert res["error"] is None
        out, res = await _run(ex, "print(request)")
        assert out == "{'url': 'x'}\n"
        slot.text = SECOND
        out, res = await _run(ex, "print(request)")
        assert out == "{'url': 'x'}\n"
        # Once the model deletes it, the next cell has the current request.
        # (A cell of only ``del name`` for an earlier cell's name fails as
        # shipped: the cell's wrapper declares ``global`` only for names the
        # cell assigns, so the assignment makes ``del`` reach the global.)
        out, res = await _run(ex, "request = None\ndel request")
        assert res["error"] is None
        out, res = await _run(ex, REPORT_CELL)
        assert _seen(out) == [{"bound": True, "text": SECOND, "data": SECOND_DATA}]
        # A cell's change to request.data still does not reach the next cell.
        out, res = await _run(ex, "request.data.append('x')\nprint(len(request.data))")
        assert out == "3\n"
        out, res = await _run(ex, REPORT_CELL)
        assert _seen(out)[0]["data"] == SECOND_DATA
    finally:
        bound_request.unbind(token)
    # A loop that answers no requester (the storage review) has none.
    token = bound_request.bind(False)
    try:
        out, res = await _run(ex, REPORT_CELL)
        assert _seen(out) == [{"bound": False}]
    finally:
        bound_request.unbind(token)


@needs_bwrap
@pytest.mark.asyncio
@pytest.mark.timeout(20)
async def test_a_cell_reads_the_request_in_the_worker(bound, world, monkeypatch):
    ex = SessionExecutor()
    try:
        await _run(ex, "1")  # starts the worker
        started = time.monotonic()
        await _cell_checks(ex)
        assert time.monotonic() - started < SESSION_BOUND_S
        assert ex.python_session(session_id=0)._worker is not None
    finally:
        await ex.close()


@pytest.mark.asyncio
@pytest.mark.timeout(10)
async def test_off_a_cell_has_no_request(monkeypatch):
    from unify.common._async_tool import bound_request

    monkeypatch.setattr(SETTINGS, "UNIFY_BIND_REQUEST", "")
    assert bound_request.bind(True) is None
    ex = SessionExecutor()
    try:
        out, _ = await _run(ex, REPORT_CELL)
        assert _seen(out) == [{"bound": False}]
    finally:
        await ex.close()


# ── one-shot act ─────────────────────────────────────────────────────────


async def _act_once(actor) -> tuple[str, list[dict], float]:
    """One request: a cell that reports what it sees, then a text reply."""
    replies = [
        h.completion(calls=[("execute_code", {"code": REPORT_CELL})]),
        h.completion(content="12"),
    ]
    started = time.monotonic()
    with h.scripted(replies) as provider:
        handle = await actor.act(
            FIRST,
            persist=False,
            can_store=False,
            clarification_enabled=False,
        )
        result = await asyncio.wait_for(handle.result(), SESSION_BOUND_S)
    elapsed = time.monotonic() - started
    assert result == "12"
    (tool,) = [m for m in handle._client.messages if m.get("role") == "tool"]
    return str(tool["content"]), provider.requests, elapsed


@pytest.mark.asyncio
@pytest.mark.timeout(15)
@pytest.mark.parametrize("projection", ["", "notebook"])
@pytest.mark.parametrize("switch", ["on", ""])
async def test_a_one_shot_act_binds_its_request(
    bound,
    monkeypatch,
    projection,
    switch,
):
    from unify.actor.code_act_actor import CodeActActor

    monkeypatch.setattr(SETTINGS, "UNIFY_BIND_REQUEST", switch)
    monkeypatch.setattr(SETTINGS, "UNIFY_CODE_PROJECTION", projection)
    actor = CodeActActor()
    try:
        tool, requests, elapsed = await _act_once(actor)
    finally:
        await actor.close()

    assert elapsed < SESSION_BOUND_S
    system = _flat(str(requests[0]["messages"][0]["content"]))
    if switch:
        assert _seen(tool) == [{"bound": True, "text": FIRST, "data": FIRST_DATA}]
        assert system.count(SENTENCE) == 1
    else:
        assert _seen(tool) == [{"bound": False}]
        assert "request.data" not in system


@needs_bwrap
@pytest.mark.asyncio
@pytest.mark.timeout(30)
@pytest.mark.parametrize("projection", ["", "notebook"])
async def test_a_one_shot_act_binds_its_request_on_the_core_surface(
    bound,
    core_world,
    monkeypatch,
    projection,
):
    from tests.actor.code_act.core_world import new_actor

    monkeypatch.setattr(SETTINGS, "UNIFY_CODE_PROJECTION", projection)
    actor = new_actor()
    try:
        tool, requests, elapsed = await _act_once(actor)
    finally:
        await actor.close()

    assert elapsed < SESSION_BOUND_S
    assert _seen(tool) == [{"bound": True, "text": FIRST, "data": FIRST_DATA}]
    system = _flat(str(requests[0]["messages"][0]["content"]))
    assert system.count(SENTENCE) == 1


# ── unify act --persist --jsonl ──────────────────────────────────────────


def _is_review(messages: list) -> bool:
    text = json.dumps(messages, default=str)
    # The storage review as shipped, or framed as the agent's own curation
    # step (UNIFY_REVIEW_FRAMING=unified, the default since the code freeze).
    return (
        "## Storage Review" in text
        or "You are a skill librarian" in text
        or "This is the curation step that follows" in text
    )


def _since_request(messages: list) -> list[dict]:
    """The messages after the latest requester message (the loop's own notices
    are user messages too, and their markers are not sent)."""
    for index in range(len(messages) - 1, -1, -1):
        message = messages[index]
        if message.get("role") == "user" and message.get("content") in (
            FIRST,
            SECOND,
        ):
            return messages[index + 1 :]
    return messages


class _Model:
    """Each request: one cell that reports what it sees, then a text reply."""

    def __init__(self) -> None:
        self.requests: list[list[dict]] = []

    async def __call__(self, *, shared_session=None, client=None, **kw):
        messages = kw.get("messages") or []
        self.requests.append(messages)
        if _is_review(messages):
            return h.completion(content="Nothing worth storing.")
        if not any(m.get("role") == "tool" for m in _since_request(messages)):
            return h.completion(calls=[("execute_code", {"code": REPORT_CELL})])
        return h.completion(content="done")


class _Actor:
    """The actor ``unify act`` starts: its own execute_code on a persistent loop."""

    def __init__(self) -> None:
        from unify.actor.code_act_actor import CodeActActor

        self._actor = CodeActActor()

    async def act(self, request: str, *, persist: bool, **_kwargs):
        from unify.actor.code_act_actor import _StorageCheckHandle
        from unify.common.async_tool_loop import start_async_tool_loop

        inner = start_async_tool_loop(
            h.new_client("You are a scripted test agent."),
            request,
            {"execute_code": self._actor.get_tools("act")["execute_code"]},
            loop_id="CodeActActor.act",
            log_steps=False,
            timeout=300,
            persist=persist,
            bind_request=True,
        )
        return _StorageCheckHandle(inner=inner, actor=self._actor, persist=persist)

    async def close(self) -> None:
        await self._actor.close()


@pytest.fixture
def jsonl_session(monkeypatch, bound):
    """``unify act --persist --jsonl`` on the scripted actor; stdin is a pipe."""
    from unify.cli import Act

    read_fd, write_fd = os.pipe()
    monkeypatch.setattr(sys, "stdin", os.fdopen(read_fd, "r"))

    async def start(self) -> None:
        self._actor = _Actor()

    monkeypatch.setattr(Act, "start", start)
    session = Act(
        SimpleNamespace(
            persist=True,
            jsonl=True,
            quiet=True,
            no_clarify=True,
            no_compose=False,
            no_store=False,
            timeout=None,
        ),
    )
    lines: list[dict] = []
    session._emit = lambda **payload: lines.append(payload)

    def send(payload: dict) -> None:
        os.write(write_fd, (json.dumps(payload) + "\n").encode())

    yield session, lines, send
    os.close(write_fd)


async def _until(predicate, timeout: float = SESSION_BOUND_S) -> None:
    async def poll():
        while not predicate():
            await asyncio.sleep(0.01)

    await asyncio.wait_for(poll(), timeout)


def _responses(lines: list[dict]) -> list[dict]:
    return [line for line in lines if line["type"] == "response"]


@pytest.mark.asyncio
@pytest.mark.timeout(10)
async def test_a_persistent_session_binds_each_request(jsonl_session):
    session, lines, send = jsonl_session
    model = _Model()
    started = time.monotonic()
    with h.scripted(()):
        import unillm.clients.uni_llm as uni_llm

        uni_llm._acompletion_with_transient_retry = model
        run = asyncio.create_task(session.run(FIRST))
        await _until(lambda: len(_responses(lines)) == 1)
        send({"message": SECOND})
        await _until(lambda: len(_responses(lines)) == 2)
        send({"quit": True})
        code = await asyncio.wait_for(run, SESSION_BOUND_S)
    assert time.monotonic() - started < 2 * SESSION_BOUND_S

    assert code == 0
    assert _responses(lines) == [
        {"type": "response", "content": "done"},
        {"type": "response", "content": "done"},
    ]
    last = [r for r in model.requests if not _is_review(r)][-1]
    tools = [
        m for m in last if m.get("role") == "tool" and "SEEN" in str(m.get("content"))
    ]
    seen = [_seen(str(m["content"]))[0] for m in tools]
    # The second request's data replaces the first's.
    assert seen == [
        {"bound": True, "text": FIRST, "data": FIRST_DATA},
        {"bound": True, "text": SECOND, "data": SECOND_DATA},
    ]
