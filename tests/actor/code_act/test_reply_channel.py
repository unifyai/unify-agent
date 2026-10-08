"""Symbolic: ``UNIFY_REPLY_CHANNEL=code+text``: a cell sends the turn's reply.

On the long lean-all Continual-ARC LOW run (``arc-pm-lean8958-h-low-ws0``,
first attempt, 3,803 calls) 394 of 754 ``execute_code`` cells (52%) did
nothing, and 206 of them were followed at once by a text reply carrying the
protocol action the cell's ``thought`` had announced: two calls per
decision, 10.3% of the run's USD. On AppWorld, where the action is code, no
cell is a no-op. With the switch on a cell can take the action itself:
``reply(text)`` ends the cell and the turn ends with exactly that text as
the reply, without another model call, as if the model had replied with it.
Plain text replies keep working. Which kind of reply ended a turn is
recorded, out of the model's sight, for analysis.

The transport is scripted (``tests/cache_discipline_helpers.py``): nothing
leaves the process. Each test that drives a session bounds every wait.
"""

from __future__ import annotations

from unify.actor.core_surface import PromptSurface
import asyncio
import json
import os
import sys
import time
from types import SimpleNamespace

import pytest

from tests import cache_discipline_helpers as h
from tests.actor.code_act.sandbox_world import needs_bwrap, world  # noqa: F401
from unify.actor import core_surface, notebook_cells
from unify.actor import prompt_builders as pb
from unify.actor.execution.session import SessionExecutor
from unify.actor.execution.types import parts_to_text
from unify.common._async_tool import cell_reply
from unify.function_manager.source_labels import compile_function_source
from unify.settings import SETTINGS

FROM_CELL = (
    "You can also reply from a cell with `reply(text)`, for example "
    "`reply(answer)` when the answer is in a variable; it ends your turn."
)
NOTE_FROM_CELL = (
    "A cell's `reply(text)` is your reply too: `reply(action)` takes the "
    "action and ends your turn."
)
# Leading and trailing whitespace, a newline and non-ASCII text, which the
# reply must carry unchanged.
ANSWER = '  first line\n{"action": "move", "to": [1, 2]} é ✓\n'
CELL = (
    "answer = '  first line\\n' + json.dumps({'action': 'move', 'to': [1, 2]})"
    " + ' é ✓\\n'\n"
    "print('computed')\n"
    "reply(answer)\n"
    "print('not reached')\n"
)
# Every session a test drives ends within this.
SESSION_BOUND_S = 2.0


def _flat(text: str) -> str:
    return " ".join(text.split())


@pytest.fixture
def channel(monkeypatch):
    monkeypatch.setattr(SETTINGS, "UNIFY_REPLY_CHANNEL", "code+text")


# ── the switch and the prompt ────────────────────────────────────────────


def test_the_switch_is_validated():
    from unify.settings import ProductionSettings

    assert ProductionSettings().UNIFY_REPLY_CHANNEL == ""
    assert ProductionSettings(UNIFY_REPLY_CHANNEL="text").UNIFY_REPLY_CHANNEL == ""
    assert (
        ProductionSettings(UNIFY_REPLY_CHANNEL="Code+Text").UNIFY_REPLY_CHANNEL
        == "code+text"
    )
    with pytest.raises(ValueError):
        ProductionSettings(UNIFY_REPLY_CHANNEL="code")


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
            can_store=True,
            core=core_surface.PromptSurface(),
        )
    return pb.build_code_act_prompt(
        environments=actor.environments,
        can_store=True,
        core=PromptSurface(),
    )


@pytest.mark.parametrize("profile", ["", "lean"])
@pytest.mark.parametrize("surface", ["json", "core"])
def test_the_prompt_says_it_once_where_the_reply_rule_is(
    monkeypatch,
    actor_tools,
    profile,
    surface,
):
    monkeypatch.setattr(SETTINGS, "UNIFY_REPLY_CHANNEL", "")
    off = _flat(_prompt(actor_tools, surface))
    monkeypatch.setattr(SETTINGS, "UNIFY_REPLY_CHANNEL", "code+text")
    on = _flat(_prompt(actor_tools, surface))
    assert FROM_CELL not in off and NOTE_FROM_CELL not in off
    assert on.count(FROM_CELL) == 1
    # Beside the reply rule.
    rule = (
        "Your answer is your final reply: a message without a tool call."
        if profile == "lean"
        else "never via a tool call."
    )
    assert f"{rule} {FROM_CELL}" in on
    # The reply-protocol note is gone.
    assert NOTE_FROM_CELL not in on
    assert on.replace(FROM_CELL + " ", "").replace(NOTE_FROM_CELL + " ", "") == off


def test_the_code_tool_mentions_reply_only_under_the_switch(monkeypatch):
    from unify.actor.code_act_actor import CodeActActor
    from unify.common.llm_helpers import method_to_schema

    def described() -> tuple[str, str]:
        tool = CodeActActor().get_tools("act")["execute_code"]
        fn = getattr(tool, "fn", tool)
        core = core_surface.core_tools({"execute_code": tool})
        core_fn = getattr(core["execute_code"], "fn", core["execute_code"])
        return tuple(
            method_to_schema(f, "execute_code")["function"]["description"]
            for f in (fn, core_fn)
        )

    monkeypatch.setattr(SETTINGS, "UNIFY_REPLY_CHANNEL", "")
    for text in described():
        assert "reply(" not in text
    monkeypatch.setattr(SETTINGS, "UNIFY_REPLY_CHANNEL", "code+text")
    for text in described():
        assert _flat(text).endswith(
            "``reply(text)`` sends ``text`` (a str) as your reply and ends your "
            "turn, as replying with that text would; the cell stops there. For "
            "example ``reply(answer)`` when the answer is in a variable.",
        )
    caps = notebook_cells.Capabilities()
    assert notebook_cells.describe(caps, structured=False).endswith(
        "You answer, and take any action the requester defines, by replying, or "
        "from a cell with `reply(text)`, which ends your turn.",
    )
    # A request answered by final_response has no reply to send.
    assert notebook_cells.describe(caps, structured=True).endswith(
        "You answer by calling `final_response`.",
    )


# ── provenance ───────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "call, literal",
    [
        ('reply("done")', True),
        ("reply('a' 'b')", True),
        ('reply(f"done")', True),
        ('reply("a" + "b")', True),
        ("reply(answer)", False),
        ('reply(f"{answer}")', False),
        ("reply(json.dumps(x))", False),
        ('reply("x" + answer)', False),
    ],
)
def test_a_reply_of_a_literal_is_marked(call, literal):
    marked = cell_reply.mark_literal_replies(f"x = 1\n{call}\n")
    assert ("_literal_reply" in marked) is literal


# ── the cell, in process and in the worker ───────────────────────────────


async def _run(ex: SessionExecutor, code: str, **kwargs) -> tuple[str, dict]:
    res = await asyncio.wait_for(
        ex.execute(code=code, state_mode="stateful", session_id=0, **kwargs),
        SESSION_BOUND_S,
    )
    return parts_to_text(res["stdout"]), res


def _define_stored(namespace: dict) -> None:
    """A stored function that calls reply(), as the library compiles one."""
    exec(
        compile_function_source(
            "answer_for",
            "def answer_for(text):\n    reply(text)\n",
        ),
        namespace,
    )


async def _cell_checks(ex: SessionExecutor) -> None:
    """What a cell's reply() does, wherever the cell runs."""
    token = cell_reply.bind(True)
    try:
        slot = cell_reply.current()
        out, res = await _run(ex, "import json\n" + CELL)
        assert res["error"] is None
        assert out == "computed\n"
        assert (slot.text, slot.from_value) == (ANSWER, True)
        # A second reply in the same turn is refused.
        out, res = await _run(ex, "reply('again')")
        assert cell_reply.ALREADY_REPLIED in res["error"]
        assert slot.text == ANSWER
        slot.clear()
        # A literal is recorded as one.
        await _run(ex, "reply('a literal')")
        assert (slot.text, slot.from_value) == ("a literal", False)
        slot.clear()
        # Only a str is taken.
        out, res = await _run(ex, "reply({'action': 'move'})")
        assert "reply() takes a str, not dict" in res["error"]
        assert slot.text is None
        # Not from inside a stored function.
        out, res = await _run(ex, "answer_for('x')", prepare=_define_stored)
        assert "reply() cannot be called inside a stored function (answer_for)" in (
            res["error"]
        )
        assert slot.text is None
        # Nor in a call execute_function makes.
        with cell_reply.refused("no reply from here"):
            out, res = await _run(ex, "reply('x')")
        assert "no reply from here" in res["error"]
        assert slot.text is None
    finally:
        cell_reply.unbind(token)
    # A loop that takes no cell reply (the storage review) refuses it.
    token = cell_reply.bind(False)
    try:
        out, res = await _run(ex, "reply('x')")
        assert cell_reply.NOT_AVAILABLE in res["error"]
    finally:
        cell_reply.unbind(token)


@needs_bwrap
@pytest.mark.asyncio
@pytest.mark.timeout(20)
async def test_a_cell_replies_in_the_worker(channel, world, monkeypatch):
    ex = SessionExecutor()
    try:
        await _run(ex, "1")  # starts the worker
        started = time.monotonic()
        await _cell_checks(ex)
        assert time.monotonic() - started < SESSION_BOUND_S
        assert ex.python_session(session_id=0)._worker is not None
    finally:
        await ex.close()


def test_off_the_sandbox_has_no_reply(monkeypatch):
    from unify.actor.execution.session import PythonExecutionSession

    monkeypatch.setattr(SETTINGS, "UNIFY_REPLY_CHANNEL", "")
    assert "reply" not in PythonExecutionSession().global_state
    monkeypatch.setattr(SETTINGS, "UNIFY_REPLY_CHANNEL", "code+text")
    assert "reply" in PythonExecutionSession().global_state


# ── one-shot act ─────────────────────────────────────────────────────────


@needs_bwrap
@pytest.mark.asyncio
@pytest.mark.timeout(30)
async def test_a_one_shot_act_answers_with_the_cells_reply(channel, world, monkeypatch):
    from unify.actor.code_act_actor import CodeActActor

    # The core surface runs cells in the sandboxed worker.

    actor = CodeActActor()
    try:
        replies = [
            h.completion(calls=[("execute_code", {"code": "import json\n" + CELL})]),
        ]
        started = time.monotonic()
        with h.scripted(replies) as provider:
            handle = await actor.act(
                "Move the piece.",
                persist=False,
                can_store=False,
                clarification_enabled=False,
            )
            result = await asyncio.wait_for(handle.result(), SESSION_BOUND_S)
        elapsed = time.monotonic() - started
        messages = list(handle._client.messages)
        state = handle._runtime_state
    finally:
        await actor.close()

    assert result == ANSWER
    assert elapsed < SESSION_BOUND_S
    # No model call after the reply: the one scripted reply was the only one.
    assert len(provider.requests) == 1
    final = messages[-1]
    assert final["role"] == "assistant" and final["content"] == ANSWER
    assert final[cell_reply.SOURCE_KEY] == "cell"
    assert final[cell_reply.FROM_VALUE_KEY] is True
    (tool,) = [m for m in messages if m.get("role") == "tool"]
    assert "computed" in str(tool["content"])
    assert "not reached" not in str(tool["content"])
    assert (state.replies_from_cell, state.replies_from_value) == (1, 1)


@needs_bwrap
@pytest.mark.asyncio
@pytest.mark.timeout(30)
async def test_a_text_reply_still_answers_with_the_switch_on(
    channel,
    world,
    monkeypatch,
):
    from unify.actor.code_act_actor import CodeActActor

    # The core surface runs cells in the sandboxed worker.

    actor = CodeActActor()
    try:
        with h.scripted([h.completion(content=ANSWER.strip())]) as provider:
            handle = await actor.act(
                "Move the piece.",
                persist=False,
                can_store=False,
                clarification_enabled=False,
            )
            result = await asyncio.wait_for(handle.result(), SESSION_BOUND_S)
        final = handle._client.messages[-1]
        state = handle._runtime_state
    finally:
        await actor.close()
    # UniLLM strips the whitespace around a model's text; a cell's reply is
    # sent as given.
    assert result == ANSWER.strip()
    assert len(provider.requests) == 1
    assert final[cell_reply.SOURCE_KEY] == "text"
    assert final[cell_reply.FROM_VALUE_KEY] is False
    assert (state.replies_from_cell, state.replies_from_value) == (0, 0)


# ── unify act --persist --jsonl ──────────────────────────────────────────

TASK = 'Move the piece. Reply with one action as a JSON object: {"action": ...}.'
FOLLOW_UP = "Feedback: moved. Reply with your next action."
FINAL = '{"action": "stay"}'
SUMMARY = "Nothing worth storing."


def _is_review(messages: list) -> bool:
    text = json.dumps(messages, default=str)
    # The storage review as shipped, or framed as the agent's own curation
    # step (UNIFY_REVIEW_FRAMING=unified, the default since the code freeze).
    return (
        "## Storage Review" in text
        or "You are a skill librarian" in text
        or "This is the curation step that follows" in text
    )


def _last_request_text(messages: list) -> str:
    for message in reversed(messages):
        if message.get("role") == "user" and not message.get("_loop_authored"):
            return str(message.get("content") or "")
    return ""


class _Model:
    """The task's first call runs the cell; the follow-up is answered in
    text; any other call is one too many."""

    def __init__(self) -> None:
        self.requests: list[list[dict]] = []
        self.extra = 0

    async def __call__(self, *, shared_session=None, client=None, **kw):
        messages = kw.get("messages") or []
        self.requests.append(messages)
        if _is_review(messages):
            return h.completion(content=SUMMARY)
        if FOLLOW_UP in _last_request_text(messages):
            return h.completion(content=FINAL)
        if not any(m.get("role") == "tool" for m in messages):
            return h.completion(
                calls=[("execute_code", {"code": "import json\n" + CELL})],
            )
        self.extra += 1
        return h.completion(content="a call after the cell's reply")


class _Actor:
    """The actor ``unify act`` starts: its own execute_code on a persistent loop."""

    def __init__(self) -> None:
        from unify.actor.code_act_actor import CodeActActor

        self._actor = CodeActActor()

    async def act(self, request: str, *, persist: bool, **_kwargs):
        from unify.actor.code_act_actor import _StorageCheckHandle
        from unify.common.async_tool_loop import start_async_tool_loop

        inner = start_async_tool_loop(
            h.new_client(TASK),
            request,
            {"execute_code": self._actor.get_tools("act")["execute_code"]},
            loop_id="CodeActActor.act",
            log_steps=False,
            timeout=300,
            persist=persist,
            reply_channel=True,
        )
        return _StorageCheckHandle(inner=inner, actor=self._actor, persist=persist)

    async def close(self) -> None:
        await self._actor.close()


@pytest.fixture
def jsonl_session(monkeypatch, channel):
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
async def test_the_cli_response_line_carries_the_cells_reply(jsonl_session):
    session, lines, send = jsonl_session
    model = _Model()
    started = time.monotonic()
    with h.scripted(()):
        import unillm.clients.uni_llm as uni_llm

        uni_llm._acompletion_with_transient_retry = model
        run = asyncio.create_task(session.run(TASK))
        await _until(lambda: len(_responses(lines)) == 1)
        # The turn ended on the reply: no model call followed it.
        assert len(model.requests) == 1
        send({"message": FOLLOW_UP})
        await _until(lambda: len(_responses(lines)) == 2)
        send({"quit": True})
        code = await asyncio.wait_for(run, SESSION_BOUND_S)
    assert time.monotonic() - started < 2 * SESSION_BOUND_S

    assert code == 0
    assert model.extra == 0
    first, second = _responses(lines)
    # Exactly the bytes the cell passed, as a text reply's line would carry.
    assert first == {"type": "response", "content": ANSWER}
    # A text reply still answers a request.
    assert second == {"type": "response", "content": FINAL}
    (result,) = [line for line in lines if line["type"] == "result"]
    assert result["run_stats"] == {"replies_from_cell": 1, "replies_from_value": 1}
    # The follow-up's request holds the reply as the assistant's message, and
    # none of the reply's records reach the model.
    (follow_up,) = [r for r in model.requests if _last_request_text(r) == FOLLOW_UP]
    assistant = [m for m in follow_up if m.get("role") == "assistant"]
    assert assistant[-1]["content"] == ANSWER
    assert not any(
        key in m
        for m in follow_up
        for key in (cell_reply.SOURCE_KEY, cell_reply.FROM_VALUE_KEY)
    )
