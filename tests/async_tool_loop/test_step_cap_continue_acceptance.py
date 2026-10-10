"""Symbolic: acceptance tests for ``UNIFY_STEP_CAP_COMPACT=continue``.

Written from the design only (``step-cap-500-cache-compaction.md`` §2-§4 of
the research repo, 8 Oct 2026), as an independent black-box check of the
implementation. In ``continue`` mode the actor's task loop (the loop that
answers a requester):

* counts its step limit per request, as ``UNIFY_STEP_CAP_REPLY``'s
  ``timer.start_request()`` path does (the limit counts messages, a turn
  with one call adds about two);
* at the per-request limit compacts the context and goes on with a fresh
  budget, as often as needed; the rebuilt context keeps the system prompt,
  the tools, the session's first user message and the current request's
  requester messages byte for byte, and appends the summary after them as a
  loop-authored message;
* ends a request through the step-limit reply path, with draft semantics
  (the latest draft is quoted, no tool-less last-word call) even when
  ``UNIFY_STEP_CAP_REPLY`` is empty, at every terminal stop: LOOP_STOP, the
  loop's timeout, a failed compaction (no restart), and a second
  ineffective compaction in a row; a persistent session then takes the
  next request;
* keeps LOOP_STOP's no-progress count across a compaction (the summary is
  loop-authored).

Off and ``on`` behave exactly as before; the off and ``on`` cases here pass
on the base without the implementation, and the requests a scripted session
sends under them are pinned to the bytes the base sends.

The transport is scripted (``tests/scripted_model.py``): nothing leaves the
process.
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import io
import json
import re
from dataclasses import dataclass
from typing import Any, Optional

import pytest

import unify.common._async_tool.context_compression as _cc
from tests import cache_discipline_helpers as h
from tests.scripted_model import Always, ScriptedModel, reply, scripted
from unify.settings import SETTINGS, ProductionSettings

TASK = "Find the answer and reply with it."
CONTINUE = "Please continue and give your answer."
LOOP_AGAIN = "Look into it once more."
INTERJECT = "Also check the second source."
REQUESTS = (TASK, CONTINUE, LOOP_AGAIN, INTERJECT) + tuple(
    f"Request {i}." for i in range(2, 12)
)
DRAFT = "Checking again; best so far: 42."
FINAL = "The answer is 42."
LAST_WORD = "My best answer: 42."
SUMMARY = "Summary: looked several times; best so far 42."
TERMINATED = "🔚 Terminating early: max_steps ({}) exceeded"
STOP_MARK = "🔚"
WAIT = 5  # seconds any one wait of a session may take


# ── tools ────────────────────────────────────────────────────────────────


async def look() -> str:
    """Look again."""
    return "Nothing new."


async def execute_code(thought: str = "", code: str = "") -> str:
    """Run Python code.

    Args:
        thought: Why the code is run.
        code: The code to run.
    """
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        exec(code, {})  # the scripted model's own constant cells
    return out.getvalue()


async def wait_long() -> str:
    """Wait for the slow service."""
    await asyncio.sleep(30)
    return "late"


# ── reading requests ─────────────────────────────────────────────────────


def _content(message: dict) -> str:
    content = message.get("content")
    if isinstance(content, list):
        return "\n".join(
            str(p.get("text", "")) if isinstance(p, dict) else str(p) for p in content
        )
    return "" if content is None else str(content)


def _requester_index(messages: list) -> tuple[Optional[str], int]:
    """The latest requester message (one of REQUESTS, maybe after a clock
    prefix) and its index; ``(None, -1)`` when none is in the request."""
    for i in range(len(messages) - 1, -1, -1):
        m = messages[i]
        if m.get("role") != "user":
            continue
        text = _content(m)
        for r in REQUESTS:
            if text == r or text.endswith(r):
                return r, i
    return None, -1


def _tool_turns_since_request(messages: list) -> int:
    _, i = _requester_index(messages)
    return sum(
        1
        for m in messages[i + 1 :]
        if m.get("role") == "assistant" and m.get("tool_calls")
    )


def _forks(model: ScriptedModel) -> int:
    return len(model.of("compression_fork"))


def _tool_turns(model: ScriptedModel) -> int:
    """Actor replies that called a tool."""
    return sum(
        1
        for c in model.of("actor")
        if c.answer is not None
        and not isinstance(c.answer, BaseException)
        and c.answer.choices[0].message.tool_calls
    )


def _last_words(model: ScriptedModel) -> list:
    """Tool-less calls (the last word, or LOOP_STOP's, classified query_llm)."""
    return model.of("last_word") + model.of("query_llm")


# ── the session ──────────────────────────────────────────────────────────


@pytest.fixture
def switches(monkeypatch):
    def set_(
        *,
        compact: str,
        max_steps: int = 300,
        cap_reply: str = "",
        loop_stop: str = "",
        k: int = 10,
        keep_prefix: str = "",
    ) -> None:
        monkeypatch.setattr(SETTINGS, "UNIFY_STEP_CAP_COMPACT", compact)
        # The prefix-preserving rebuild of the design is its own switch.
        monkeypatch.setattr(SETTINGS, "UNIFY_COMPACTION_KEEP_PREFIX", keep_prefix)
        monkeypatch.setattr(SETTINGS, "UNIFY_STEP_CAP_REPLY", cap_reply)
        monkeypatch.setattr(SETTINGS, "UNIFY_LOOP_STOP", loop_stop)
        monkeypatch.setattr(SETTINGS, "UNIFY_LOOP_STOP_K", k)
        monkeypatch.setattr(SETTINGS, "UNIFY_MAX_TOOL_LOOP_STEPS", max_steps)

    return set_


@dataclass
class Session:
    handle: Any
    driver: asyncio.Future

    async def next(self, wait: float = WAIT) -> tuple[str, Any]:
        """``("response", text)`` for the next response, or ``("ended",
        result)`` when the loop ended first."""
        response = asyncio.ensure_future(h._next_response(self.handle))
        done, _ = await asyncio.wait(
            {response, self.driver},
            timeout=wait,
            return_when=asyncio.FIRST_COMPLETED,
        )
        if response in done:
            return "response", response.result()["content"]
        response.cancel()
        await asyncio.gather(response, return_exceptions=True)
        if self.driver in done:
            return "ended", self.driver.result()
        raise AssertionError(f"no response and no end within {wait}s")

    async def ask(self, text: str, wait: float = WAIT) -> tuple[str, Any]:
        await self.handle.submit(text)
        return await self.next(wait)

    async def close(self) -> None:
        if not self.driver.done():
            await self.handle.stop()
        await asyncio.wait_for(asyncio.shield(self.driver), WAIT)


def _start(tools=None, *, timeout: float = 30, **kwargs) -> Session:
    """The actor's task loop: persistent, it answers a requester, and no
    other loop started it. Its result is awaited in the background, as
    ``unify act --persist`` does: the handle restarts a compacted loop from
    its ``result()``. The step limit comes from UNIFY_MAX_TOOL_LOOP_STEPS."""
    from unify.common.async_tool_loop import start_async_tool_loop

    kwargs.setdefault("persist", True)
    handle = start_async_tool_loop(
        h.new_client(),
        TASK,
        tools or {"look": look},
        loop_id="CodeActActor.act",
        log_steps=False,
        timeout=timeout,
        reply_channel=True,
        bind_request=True,
        **kwargs,
    )
    return Session(handle, asyncio.ensure_future(handle.result()))


# ── the switch ───────────────────────────────────────────────────────────


def test_continue_is_a_value_of_the_switch():
    assert (
        ProductionSettings(UNIFY_STEP_CAP_COMPACT="continue").UNIFY_STEP_CAP_COMPACT
        == "continue"
    )
    assert (
        ProductionSettings(UNIFY_STEP_CAP_COMPACT=" Continue ").UNIFY_STEP_CAP_COMPACT
        == "continue"
    )
    # The other values keep their meaning.
    assert ProductionSettings().UNIFY_STEP_CAP_COMPACT == ""
    assert ProductionSettings(UNIFY_STEP_CAP_COMPACT="off").UNIFY_STEP_CAP_COMPACT == ""
    assert (
        ProductionSettings(UNIFY_STEP_CAP_COMPACT="on").UNIFY_STEP_CAP_COMPACT == "on"
    )
    with pytest.raises(ValueError):
        ProductionSettings(UNIFY_STEP_CAP_COMPACT="yes")


# ── (a) the step budget counts per request ───────────────────────────────

BUDGET_STEPS = 10  # messages; one request below adds 4 (request, call, result, reply)
BUDGET_REQUESTS = [TASK] + [f"Request {i}." for i in range(2, 8)]


def _two_calls_per_request(call):
    """One look, then the answer: two model calls per request."""
    request, _ = _requester_index(call.messages)
    if _tool_turns_since_request(call.messages) >= 1:
        return reply(f"done: {request}")
    return reply(DRAFT, calls=[("look", {})])


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["continue", ""])
async def test_a_persistent_session_has_a_budget_per_request(switches, mode):
    """Seven requests of two calls each, against a 10-message limit: under
    ``continue`` no request reaches it, with no compaction; off, the
    conversation-wide count ends the session at the third request."""
    switches(compact=mode, max_steps=BUDGET_STEPS)
    model = ScriptedModel(actor=Always(_two_calls_per_request))
    outcomes = []
    with scripted(model):
        session = _start()
        outcomes.append(await session.next())
        for request in BUDGET_REQUESTS[1:]:
            if outcomes[-1][0] == "ended":
                break
            outcomes.append(await session.ask(request))
        await session.close()

    if mode == "continue":
        assert outcomes == [("response", f"done: {r}") for r in BUDGET_REQUESTS]
        assert _forks(model) == 0
        assert set(model.kinds()) == {"actor"}
    else:
        assert outcomes[:2] == [("response", f"done: {r}") for r in BUDGET_REQUESTS[:2]]
        assert outcomes[2] == ("ended", TERMINATED.format(BUDGET_STEPS))
        assert len(outcomes) == 3


# ── (c) compactions per request are unbounded ────────────────────────────

LONG_STEPS = 17  # room for a compacted context and a few calls after it


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["continue", "on"])
async def test_one_long_request_is_compacted_as_often_as_it_needs(switches, mode):
    """The request needs three compactions. ``continue`` compacts three
    times and answers; ``on`` compacts twice and stops at its third limit,
    as shipped."""
    switches(compact=mode, max_steps=LONG_STEPS)
    holder: list[ScriptedModel] = []

    def actor(call):
        if _forks(holder[0]) >= 3:
            return reply(FINAL)
        return reply(DRAFT, calls=[("look", {})])

    model = ScriptedModel(actor=Always(actor), compression_fork=Always(reply(SUMMARY)))
    holder.append(model)
    with scripted(model):
        session = _start()
        outcome = await session.next()
        await session.close()

    if mode == "continue":
        assert outcome == ("response", FINAL)
        assert _forks(model) == 3
        # Every compaction carried on the same request.
        assert not any(
            STOP_MARK in _content(m) for c in model.calls for m in c.messages
        )
    else:
        assert outcome == ("ended", TERMINATED.format(LONG_STEPS))
        assert _forks(model) == 2
    assert model.of("compactor") == []


@pytest.mark.asyncio
async def test_the_rebuild_keeps_the_prefix_byte_for_byte(switches):
    """The first request after a compaction starts with the system prompt,
    the session's first user message and the current request's requester
    messages, byte for byte as the request before the compaction sent them,
    with the same tools; the summary follows them."""
    switches(compact="continue", max_steps=LONG_STEPS, keep_prefix="on")
    holder: list = []
    state = {"interjected": False}

    async def actor(call):
        request, _ = _requester_index(call.messages)
        if request == TASK:
            if _tool_turns_since_request(call.messages) >= 1:
                return reply(f"done: {TASK}")
            return reply(DRAFT, calls=[("look", {})])
        if _forks(holder[0]) >= 1:
            return reply(FINAL)
        if (
            request == LOOP_AGAIN
            and not state["interjected"]
            and _tool_turns_since_request(call.messages) >= 1
        ):
            # The requester adds to the running request.
            state["interjected"] = True
            await holder[1].handle.submit(INTERJECT)
        return reply(DRAFT, calls=[("look", {})])

    model = ScriptedModel(actor=Always(actor), compression_fork=Always(reply(SUMMARY)))
    holder.append(model)
    with scripted(model):
        session = _start()
        holder.append(session)
        first = await session.next()
        second = await session.ask(LOOP_AGAIN)
        await session.close()

    assert first == ("response", f"done: {TASK}")
    assert second == ("response", FINAL)
    assert state["interjected"] and _forks(model) == 1
    calls = model.calls
    fork_at = next(i for i, c in enumerate(calls) if c.kind == "compression_fork")
    before = next(c for c in reversed(calls[:fork_at]) if c.kind == "actor")
    after = next(c for c in calls[fork_at + 1 :] if c.kind == "actor")

    def canon(m: dict) -> str:
        return json.dumps(m, sort_keys=True, default=str)

    def is_requester(m: dict) -> bool:
        text = _content(m)
        return m.get("role") == "user" and any(
            text == r or text.endswith(r) for r in REQUESTS
        )

    sent = before.messages
    assert sent[0].get("role") == "system"
    prefix = [sent[0]] + [m for m in sent[1:] if is_requester(m)]
    # The session's first user message and both messages of the request.
    assert [r for m in prefix[1:] for r in REQUESTS if _content(m).endswith(r)] == [
        TASK,
        LOOP_AGAIN,
        INTERJECT,
    ]
    assert [canon(m) for m in after.messages[: len(prefix)]] == [
        canon(m) for m in prefix
    ]
    assert SUMMARY in _content(after.messages[len(prefix)])
    assert json.dumps(after.request["tools"], sort_keys=True) == json.dumps(
        before.request["tools"],
        sort_keys=True,
    )


# ── (c, case 4) an ineffective compaction ────────────────────────────────

MAX_INPUT = 4000  # tokens; the context threshold is 0.7 of it
OVER = 5000  # prompt tokens: over the threshold
HUGE_SUMMARY = "Summary: " + " ".join(f"fact{i}" for i in range(6000))


@pytest.mark.asyncio
async def test_two_ineffective_compactions_in_a_row_end_the_request_with_a_reply(
    switches,
    monkeypatch,
):
    """Every summary is larger than the context threshold, so no compaction
    brings the context under it. The second such compaction in a row ends
    the request with a reply (no third compaction), and the session takes
    the next request."""
    import unillm

    switches(compact="continue", keep_prefix="on")
    monkeypatch.setattr(unillm, "get_max_input_tokens", lambda *a, **k: MAX_INPUT)
    holder: list[ScriptedModel] = []
    phase = {"first_done": False, "forks_in_first": None}

    def actor(call):
        request, _ = _requester_index(call.messages)
        looked = _tool_turns_since_request(call.messages) >= 1
        if request == CONTINUE:
            # The context is still over its threshold when the next request
            # arrives: compress once (the forced turn), then answer. (After the
            # shipped fallback the oversized summary is the session's first
            # message, which a keep-prefix rebuild keeps; this call reports it
            # as fitting.)
            if not looked and _forks(holder[0]) == phase["forks_in_first"]:
                return reply(calls=[("compress_context", {})])
            return reply(f"done: {CONTINUE}")
        # Bounded without the implementation: the threshold path compacts
        # without limit there.
        if _forks(holder[0]) >= 4:
            return reply(FINAL)
        # The turn the threshold forces keeps the session's tools, so it is
        # an actor call too: look once, then compress.
        if looked:
            return reply(calls=[("compress_context", {})], prompt_tokens=OVER)
        return reply(DRAFT, calls=[("look", {})], prompt_tokens=OVER)

    def fork(call):
        return reply(SUMMARY if phase["first_done"] else HUGE_SUMMARY)

    model = ScriptedModel(
        actor=Always(actor),
        compress_turn=Always(
            reply(calls=[("compress_context", {})], prompt_tokens=OVER),
        ),
        compression_fork=Always(fork),
        last_word=Always(reply(LAST_WORD)),
        query_llm=Always(reply(LAST_WORD)),
    )
    holder.append(model)
    with scripted(model):
        session = _start()
        first = await session.next()
        forks_in_first = _forks(model)
        phase["first_done"], phase["forks_in_first"] = True, forks_in_first
        second = await session.ask(CONTINUE)
        await session.close()

    assert first[0] == "response" and first[1].startswith(STOP_MARK), first
    assert forks_in_first == 2
    assert second == ("response", f"done: {CONTINUE}")


# ── (e) terminal stops reply, with the draft ─────────────────────────────


def _loop_stop_headline(k: int) -> str:
    return f"🔚 Stopped: the last {k} tool calls made no progress"


@pytest.mark.asyncio
async def test_loop_stop_fires_at_k_across_a_compaction(switches):
    """K=6; the step limit (11 messages) is reached after five no-op calls,
    the request is compacted, and the sixth no-op call after it ends the
    request: the count carried across the compaction."""
    k = 6
    switches(compact="continue", max_steps=11, loop_stop="on", k=k)

    def actor(call):
        if sum(1 for c in model.of("actor")) > 30:  # bounded without it
            return reply(FINAL)
        return reply(
            DRAFT,
            calls=[("execute_code", {"thought": "Again.", "code": "print('')"})],
        )

    model = ScriptedModel(
        actor=Always(actor),
        compression_fork=Always(reply(SUMMARY)),
        last_word=Always(reply(LAST_WORD)),
        query_llm=Always(reply(LAST_WORD)),
    )
    with scripted(model):
        session = _start({"execute_code": execute_code})
        outcome = await session.next()
        await session.close()

    assert outcome[0] == "response", outcome
    assert outcome[1].startswith(_loop_stop_headline(k)), outcome
    assert _forks(model) == 1
    assert _tool_turns(model) == k


@pytest.mark.asyncio
@pytest.mark.parametrize("stop", ["loop_stop", "timeout"])
async def test_a_terminal_stop_replies_with_the_draft_and_the_session_goes_on(
    switches,
    stop,
):
    """LOOP_STOP and the loop's timeout end the request through the reply
    path with draft semantics, although UNIFY_STEP_CAP_REPLY is empty: the
    reply quotes the latest draft and no tool-less call is made. The next
    request is answered in the same session."""
    k = 4
    switches(compact="continue", loop_stop="on" if stop == "loop_stop" else "", k=k)

    def actor(call):
        request, _ = _requester_index(call.messages)
        if request == CONTINUE:
            return reply(f"done: {CONTINUE}")
        if stop == "timeout":
            return reply(DRAFT, calls=[("wait_long", {})])
        return reply(
            DRAFT,
            calls=[("execute_code", {"thought": "Again.", "code": "pass"})],
        )

    model = ScriptedModel(
        actor=Always(actor),
        last_word=Always(reply(LAST_WORD)),
        query_llm=Always(reply(LAST_WORD)),
    )
    tools = {"execute_code": execute_code, "wait_long": wait_long}
    with scripted(model):
        session = _start(tools, timeout=1 if stop == "timeout" else 30)
        first = await session.next()
        second = await session.ask(CONTINUE) if first[0] == "response" else None
        await session.close()

    assert first[0] == "response", first
    assert first[1].startswith(STOP_MARK)
    if stop == "loop_stop":
        assert first[1].startswith(_loop_stop_headline(k))
    assert first[1].endswith(f"Best current answer:\n{DRAFT}")
    assert _last_words(model) == []
    assert second == ("response", f"done: {CONTINUE}")


@pytest.mark.asyncio
async def test_a_failed_compaction_ends_the_request_with_a_reply(
    switches,
    monkeypatch,
):
    """The fork fails, and so does the compactor it falls back to: the
    request ends with a reply quoting the draft, the session is not
    restarted, and the next request is answered."""
    switches(compact="continue", max_steps=LONG_STEPS)

    async def fail(*args, **kwargs):
        raise RuntimeError("provider unavailable")

    monkeypatch.setattr(_cc, "compress_messages", fail)

    def actor(call):
        request, _ = _requester_index(call.messages)
        if request == CONTINUE:
            return reply(f"done: {CONTINUE}")
        return reply(DRAFT, calls=[("look", {})])

    model = ScriptedModel(
        actor=Always(actor),
        compression_fork=Always(RuntimeError("provider unavailable")),
        last_word=Always(reply(LAST_WORD)),
        query_llm=Always(reply(LAST_WORD)),
    )
    with scripted(model):
        session = _start()
        first = await session.next()
        second = await session.ask(CONTINUE) if first[0] == "response" else None
        await session.close()

    assert first[0] == "response", first
    assert first[1].startswith(STOP_MARK)
    assert first[1].endswith(f"Best current answer:\n{DRAFT}")
    assert _forks(model) == 1
    assert second == ("response", f"done: {CONTINUE}")


# ── off and on: the bytes sent are as before ─────────────────────────────

# sha256 of the canonical requests of the scripted session below, recorded
# from a run on the parent of the mode's merge (unify-agent 682cfc992, which
# has no ``continue``): the assertion message prints the digest. The same
# digests at a later commit mean off and ``on`` send what they sent before.
PINNED = {
    "off": "2e3a3c1dd8d224a14f5e32a1563f925b3cc562e87c6895204a225b1dbf2a8adf",  # pragma: allowlist secret
    "on": "f2baa9edfb6aa34dfa71e27a387c2a7cb4eb5f338c1521c233638ea162ff665c",  # pragma: allowlist secret
    "off+draft": "881c091fc18d8ae47c2eca5aad10f15b4dd9789f0a7d9ceb8680d765476af90d",  # pragma: allowlist secret
}
PINNED_STEPS = 9


def _canonical(calls) -> str:
    """Every request sent, with the scripted call ids numbered in order."""
    text = json.dumps(
        [{"kind": c.kind, **c.request} for c in calls],
        sort_keys=True,
        default=str,
    )
    ids: dict[str, str] = {}

    def number(match: re.Match) -> str:
        return ids.setdefault(match.group(0), f"call_#{len(ids)}")

    return re.sub(r"call_\d+", number, text)


@pytest.mark.asyncio
@pytest.mark.parametrize("case", sorted(PINNED))
async def test_off_and_on_send_the_bytes_they_sent_before(switches, case):
    compact, cap_reply = {
        "off": ("", ""),
        "on": ("on", ""),
        "off+draft": ("", "draft"),
    }[case]
    switches(compact=compact, cap_reply=cap_reply, max_steps=PINNED_STEPS)

    def actor(call):
        request, _ = _requester_index(call.messages)
        if request == CONTINUE:
            return reply(f"done: {CONTINUE}")
        return reply(DRAFT, calls=[("look", {})])

    model = ScriptedModel(
        actor=Always(actor),
        compression_fork=Always(reply(SUMMARY)),
    )
    outcomes = []
    with scripted(model):
        session = _start()
        outcomes.append(await session.next())
        if outcomes[-1][0] == "response":
            outcomes.append(await session.ask(CONTINUE))
        await session.close()

    canonical = _canonical(model.calls)
    digest = hashlib.sha256(canonical.encode()).hexdigest()
    assert digest == PINNED[case], (
        case,
        digest,
        outcomes,
        model.kinds(),
        canonical[:4000],
    )
