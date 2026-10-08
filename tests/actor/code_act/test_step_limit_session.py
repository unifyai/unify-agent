"""Symbolic: a persistent ``unify act --jsonl`` session that reaches its step limit.

In the Continual-ARC runs of 5 October (``arc-pm2-up592-h-low-ws0``, instance
8, and ``arc-pm2-up592-h0-low-fresh0``) a persistent session's task loop
reached ``max_steps`` (300, a count of every message in the session) and
ended: the CLI wrote its stop notice as the ``result`` line, the host took it
as the agent's turn and sent its next message, and the CLI, still reading
stdin, handed that message to a handle whose task loop was gone. Nothing was
ever written back, so the host waited until its idle timeout (300 s) and
restarted the session.

As shipped now, a session whose task loop ends on its own says so: the CLI
stops taking follow-ups, finishes the storage review and writes ``ended``.
With ``UNIFY_STEP_CAP_REPLY`` the step limit ends only the request: the
session answers it with a reply that says it stopped at the step limit and
quotes its latest draft, and the next message is handled as usual.

The transport is scripted (``tests/cache_discipline_helpers.py``): nothing
leaves the process.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from types import SimpleNamespace

import pytest

from tests import cache_discipline_helpers as h
from unify.settings import SETTINGS

PROTOCOL = (
    "You are solving a grid puzzle. Reply with one action as a JSON object on "
    'the last line: {"action": "submit", "grid": [[...]]}.'
)
TASK = "Solve the puzzle. " + PROTOCOL
DRAFT = 'Still checking. Best so far: {"action": "submit", "grid": [[4]]}'
FOLLOW_UP = (
    "Feedback since your last action:\nNo action JSON object was found in your "
    "reply. Reply with one action."
)
FINAL = '{"action": "submit", "grid": [[4]]}'
SUMMARY = "Nothing worth storing."
MAX_STEPS = 6


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
    """Scripted replies: the session keeps looking until it gets the
    follow-up, which it answers with an action; the review summarises."""

    def __init__(self) -> None:
        self.requests: list[list[dict]] = []

    async def __call__(self, *, shared_session=None, client=None, **kw):
        messages = kw.get("messages") or []
        self.requests.append(messages)
        if _is_review(messages):
            return h.completion(content=SUMMARY)
        if FOLLOW_UP in _last_request_text(messages):
            return h.completion(content=FINAL)
        return h.completion(content=DRAFT, calls=[("look", {})])


async def look() -> str:
    """Look at the grid again."""
    return "Nothing new."


class _Actor:
    """The actor ``unify act`` starts, on a scripted persistent loop."""

    def __init__(self) -> None:
        from unify.actor.code_act_actor import CodeActActor

        self._actor = CodeActActor()

    async def act(self, request: str, *, persist: bool, **_kwargs):
        from unify.actor.code_act_actor import _StorageCheckHandle
        from unify.common.async_tool_loop import start_async_tool_loop

        inner = start_async_tool_loop(
            h.new_client(PROTOCOL),
            request,
            {"look": look},
            loop_id="CodeActActor.act",
            log_steps=False,
            timeout=30,
            persist=persist,
            max_steps=MAX_STEPS,
        )
        return _StorageCheckHandle(inner=inner, actor=self._actor, persist=persist)

    async def close(self) -> None:
        await self._actor.close()


@pytest.fixture
def jsonl_session(monkeypatch):
    """``unify act --persist --jsonl`` on the scripted actor; stdin is a pipe."""
    from unify.cli import Act

    read_fd, write_fd = os.pipe()
    monkeypatch.setattr(sys, "stdin", os.fdopen(read_fd, "r"))

    # Built before any test's clock starts: the actor's build (its own
    # function and guidance managers) and the first import of the bridge the
    # session attaches are not the session's time.
    import unify.agents.cli_bridge  # noqa: F401

    actor = _Actor()

    async def start(self) -> None:
        self._actor = actor

    monkeypatch.setattr(Act, "start", start)
    session = Act(
        SimpleNamespace(
            persist=True,
            jsonl=True,
            quiet=True,
            no_clarify=False,
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


async def _until(predicate, timeout: float = 20) -> None:
    async def poll():
        while not predicate():
            await asyncio.sleep(0.05)

    await asyncio.wait_for(poll(), timeout)


def _types(lines: list[dict]) -> list[str]:
    return [line["type"] for line in lines if line["type"] != "storage"]


@pytest.mark.asyncio
async def test_a_session_ended_by_its_step_limit_says_so(jsonl_session, monkeypatch):
    """The recorded hang: the host's next message after the stop notice."""
    monkeypatch.setattr(SETTINGS, "UNIFY_STEP_CAP_REPLY", False)
    session, lines, send = jsonl_session
    model = _Model()
    with h.scripted(()):
        import unillm.clients.uni_llm as uni_llm

        uni_llm._acompletion_with_transient_retry = model
        run = asyncio.create_task(session.run(TASK))
        await _until(lambda: "result" in _types(lines))
        send({"message": FOLLOW_UP})
        # As shipped the session never answers this message and run() waits
        # on stdin for ever; now it ends the session and says so.
        code = await asyncio.wait_for(run, 20)

    assert code == 0
    assert _types(lines) == ["result", "ended"]
    assert (
        lines[0]["content"] == f"🔚 Terminating early: max_steps ({MAX_STEPS}) exceeded"
    )
    # The message reached neither the session nor its review.
    assert not any(FOLLOW_UP in json.dumps(r, default=str) for r in model.requests)
    assert any(_is_review(r) for r in model.requests)


@pytest.mark.asyncio
async def test_on_the_step_limit_ends_the_request_and_the_session_goes_on(
    jsonl_session,
    monkeypatch,
):
    monkeypatch.setattr(SETTINGS, "UNIFY_STEP_CAP_REPLY", True)
    session, lines, send = jsonl_session
    model = _Model()
    with h.scripted(()):
        import unillm.clients.uni_llm as uni_llm

        uni_llm._acompletion_with_transient_retry = model
        run = asyncio.create_task(session.run(TASK))
        await _until(lambda: "response" in _types(lines))
        send({"message": FOLLOW_UP})
        await _until(lambda: _types(lines).count("response") == 2)
        send({"quit": True})
        code = await asyncio.wait_for(run, 20)

    assert code == 0
    assert _types(lines) == ["response", "response", "result", "ended"]
    capped, answered = (line["content"] for line in lines if line["type"] == "response")
    assert capped.startswith(
        f"🔚 Stopped at the step limit: max_steps ({MAX_STEPS}) exceeded",
    )
    assert capped.endswith("Best current answer:\n" + DRAFT)
    assert answered == FINAL
    # The follow-up reached the session, after the reply that informed it.
    (turn,) = [r for r in model.requests if _last_request_text(r) == FOLLOW_UP]
    assert any(
        m.get("role") == "assistant" and m.get("content") == capped for m in turn
    )


RESTART = "Context was compressed. Continue from where you left off."


def _is_compaction(messages: list) -> bool:
    """The compactor's request, or the fork summary's
    (the cache discipline, baked in at the code freeze)."""
    from unify.common._async_tool import cache_discipline

    return any(
        m.get("role") == "system"
        and "You are a context compactor" in str(m.get("content"))
        for m in messages
    ) or (
        bool(messages)
        and cache_discipline.COMPRESSION_FORK_INSTRUCTION
        in str(messages[-1].get("content"))
    )


class _CompactingModel(_Model):
    """As _Model; the compactor returns at once, and after a compaction
    the session replies with its draft."""

    async def __call__(self, *, shared_session=None, client=None, **kw):
        messages = kw.get("messages") or []
        if _is_compaction(messages):
            self.requests.append(messages)
            return h.completion(content="Compacted.")
        restarted = any(
            m.get("role") == "user" and RESTART in str(m.get("content"))
            for m in messages
        )
        if (
            restarted
            and not _is_review(messages)
            and FOLLOW_UP not in _last_request_text(messages)
        ):
            self.requests.append(messages)
            return h.completion(content=DRAFT)
        return await super().__call__(shared_session=shared_session, **kw)


@pytest.mark.asyncio
async def test_compact_at_the_step_limit_and_the_session_goes_on(
    jsonl_session,
    monkeypatch,
):
    """UNIFY_STEP_CAP_COMPACT=on, the step limit counting the whole session:
    the session compacts its conversation at the limit and answers, so the
    host sees no stop notice and has nothing to restart."""
    monkeypatch.setattr(SETTINGS, "UNIFY_STEP_CAP_REPLY", False)
    monkeypatch.setattr(SETTINGS, "UNIFY_STEP_CAP_COMPACT", "on")
    # Room for the system messages a compacted conversation starts with.
    monkeypatch.setattr(sys.modules[__name__], "MAX_STEPS", 17)
    session, lines, send = jsonl_session
    model = _CompactingModel()
    with h.scripted(()):
        import unillm.clients.uni_llm as uni_llm

        uni_llm._acompletion_with_transient_retry = model
        run = asyncio.create_task(session.run(TASK))
        await _until(lambda: "response" in _types(lines), 2)
        send({"message": FOLLOW_UP})
        await _until(lambda: _types(lines).count("response") == 2, 2)
        send({"quit": True})
        code = await asyncio.wait_for(run, 20)

    assert code == 0
    assert _types(lines) == ["response", "response", "result", "ended"]
    first, second = (line["content"] for line in lines if line["type"] == "response")
    assert (first, second) == (DRAFT, FINAL)
    assert not any("🔚" in json.dumps(line, ensure_ascii=False) for line in lines)
    compactions = [r for r in model.requests if _is_compaction(r)]
    assert len(compactions) == 1
    # The follow-up was answered from the compacted conversation.
    (turn,) = [r for r in model.requests if _last_request_text(r) == FOLLOW_UP]
    assert any(RESTART in str(m.get("content")) for m in turn)
