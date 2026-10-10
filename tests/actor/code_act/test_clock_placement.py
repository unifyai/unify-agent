"""Symbolic: ``UNIFY_CLOCK_PLACEMENT`` moves the host clock out of the system prompt.

As shipped, the system prompt carries a "Current Time" section near its end
(``prompt_builders._build_clock_context``) that names the host's time to the
minute and tells the model to treat it as authoritative. Two sessions that
start in different minutes then send different system prompts: everything
after the clock misses the cross-session prompt cache, and the cache
affinity key (``cache_discipline.prefix_affinity_key``, a hash of the model,
the whole system prompt and the tools) differs too (office-v2 defect note
P2c; the step-cap design's "Cache-breaking behaviour" item 3).

With ``first_message`` the system prompt has no clock section, so it and the
affinity key are the same for every session of one configuration, and the
session's first user message opens with one line stating the host's reading
as non-authoritative. A persistent session's later requests do not repeat
it, and the session's forks (the storage review, the compression summary)
still start with the session's own system prompt and first user message.

Requests are captured at unillm's transport, so nothing leaves the process.
"""

from __future__ import annotations

import asyncio
import json
import re
from datetime import datetime, timezone

import pytest

from tests import cache_discipline_helpers as h
from tests.actor.code_act.test_baked_prompt_golden import (
    BAKED_GOLDEN,
    baked_recording,
)
from tests.scripted_model import (  # noqa: F401 (fixture)
    Always,
    cell,
    reply,
    scripted_model,
)
from unify.settings import SETTINGS

CLOCK_LINE = (
    "The host clock reads {now}. Dates stated in the request or in the files "
    "and records you work with take precedence."
)
CLOCK_PREFIX = "The host clock reads "
SECTION = "### Current Time"
EARLY = datetime(2026, 7, 1, 9, 0, tzinfo=timezone.utc)
LATE = datetime(2026, 7, 2, 17, 45, tzinfo=timezone.utc)
SUMMARY = "Summary: listed the workspace; nothing else is left to do."


def _clock_at(moment: datetime):
    """A ``prompt_helpers.now`` that reads *moment*, in its rendering."""

    def now(time_only: bool = False, as_string: bool = True):
        if not as_string:
            return moment
        if time_only:
            return moment.strftime("%I:%M %p UTC")
        return moment.strftime("%A, %B %d, %Y at %I:%M %p UTC")

    return now


def _set_clock(monkeypatch, moment: datetime) -> str:
    """Make the prompt clock read *moment*; return the line it gives."""
    from unify.common import prompt_helpers

    now = _clock_at(moment)
    monkeypatch.setattr(prompt_helpers, "now", now)
    return CLOCK_LINE.format(now=now())


def _golden() -> dict:
    return json.loads(BAKED_GOLDEN.read_text())


def _without_clock_section(system_prompt: str) -> str:
    """*system_prompt* less its "Current Time" section."""
    stripped = re.sub(
        r"\n\n### Current Time\n\n.*?(?=\n\n### )",
        "",
        system_prompt,
        count=1,
        flags=re.DOTALL,
    )
    assert stripped != system_prompt, "the golden has no clock section"
    return stripped


def _text(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(
            str(part.get("text", "")) if isinstance(part, dict) else str(part)
            for part in content
        )
    return "" if content is None else str(content)


def _first_user(messages: list[dict]) -> str:
    return next(_text(m["content"]) for m in messages if m["role"] == "user")


def _clock_lines(messages: list[dict]) -> list[str]:
    """Every line of every message that states the host clock."""
    return [
        line
        for m in messages
        for line in _text(m.get("content")).splitlines()
        if line.startswith(CLOCK_PREFIX)
    ]


def _prefix_through_first_user(messages: list[dict]) -> list[str]:
    """The messages up to and including the first user message, as sent."""
    end = next(i for i, m in enumerate(messages) if m["role"] == "user")
    return [json.dumps(m, sort_keys=True, default=str) for m in messages[: end + 1]]


async def _first_request_and_key(monkeypatch, moment: datetime):
    """A fresh actor's first request (as the golden records it) and the cache
    affinity key its session was given, with the clock at *moment*."""
    _set_clock(monkeypatch, moment)
    sets = h.install_affinity_api(monkeypatch)
    _result, _, requests = await h.scenario_actor(
        [lambda: h.completion(content="done")] * 8,
    )
    assert sets, "the session was given no cache affinity key"
    return baked_recording(requests), sets[0][0]


# ── (a) off: as shipped ────────────────────────────────────────────────────


@pytest.mark.asyncio
@pytest.mark.timeout(180)
async def test_off_the_first_request_is_the_golden(monkeypatch):
    monkeypatch.setattr(SETTINGS, "UNIFY_CLOCK_PLACEMENT", "")
    _result, _, requests = await h.scenario_actor(
        [lambda: h.completion(content="done")] * 8,
    )
    recorded = baked_recording(requests)
    golden = _golden()
    assert recorded == golden
    # The clock is in the system prompt, and only there.
    assert recorded["system_prompt"].count(SECTION) == 1
    assert "The current date and time is **" in recorded["system_prompt"]
    assert CLOCK_PREFIX not in recorded["first_user_message"]
    assert SECTION not in recorded["first_user_message"]


@pytest.mark.parametrize("value", ["", "system", "SYSTEM", None, "first_message"])
def test_the_switch_takes_its_values(value):
    from unify.settings import ProductionSettings

    expected = "first_message" if value == "first_message" else ""
    assert ProductionSettings(UNIFY_CLOCK_PLACEMENT=value).UNIFY_CLOCK_PLACEMENT == (
        expected
    )


def test_the_switch_refuses_other_values():
    from unify.settings import ProductionSettings

    with pytest.raises(ValueError, match="UNIFY_CLOCK_PLACEMENT"):
        ProductionSettings(UNIFY_CLOCK_PLACEMENT="user")


# ── (b) first_message: one system prompt for every session ────────────────


@pytest.mark.asyncio
@pytest.mark.timeout(240)
async def test_first_message_sessions_started_at_different_times_share_the_prefix(
    monkeypatch,
):
    monkeypatch.setattr(SETTINGS, "UNIFY_CLOCK_PLACEMENT", "first_message")
    early, early_key = await _first_request_and_key(monkeypatch, EARLY)
    late, late_key = await _first_request_and_key(monkeypatch, LATE)

    assert SECTION not in early["system_prompt"]
    assert CLOCK_PREFIX not in early["system_prompt"]
    assert early["system_prompt"] == late["system_prompt"]
    assert early["tools"] == late["tools"]
    assert early_key == late_key
    # The first user messages differ only in the time they state.
    assert early["first_user_message"] != late["first_user_message"]


@pytest.mark.asyncio
@pytest.mark.timeout(240)
async def test_off_sessions_started_at_different_times_do_not_share_it(
    monkeypatch,
):
    """The control: as shipped, the clock makes the two prefixes differ."""
    monkeypatch.setattr(SETTINGS, "UNIFY_CLOCK_PLACEMENT", "")
    early, early_key = await _first_request_and_key(monkeypatch, EARLY)
    late, late_key = await _first_request_and_key(monkeypatch, LATE)

    assert early["system_prompt"] != late["system_prompt"]
    assert early_key != late_key
    assert early["first_user_message"] == late["first_user_message"]


# ── (c) first_message: the line that opens the first user message ─────────


@pytest.mark.asyncio
@pytest.mark.timeout(180)
async def test_first_message_opens_the_first_user_message_with_the_clock(
    monkeypatch,
):
    monkeypatch.setattr(SETTINGS, "UNIFY_CLOCK_PLACEMENT", "first_message")
    line = _set_clock(monkeypatch, LATE)
    assert line == (
        "The host clock reads Thursday, July 02, 2026 at 05:45 PM UTC. Dates "
        "stated in the request or in the files and records you work with take "
        "precedence."
    )
    _result, _, requests = await h.scenario_actor(
        [lambda: h.completion(content="done")] * 8,
    )
    recorded = baked_recording(requests)
    golden = _golden()
    first = h.session_requests(requests)[0]

    # The line opens the first user message, once, ahead of the context the
    # golden records; nothing else in that message changes.
    user = recorded["first_user_message"]
    assert user.splitlines()[0] == line
    assert _clock_lines(first["messages"]) == [line]
    assert user == f"{line}\n\n{golden['first_user_message']}"
    # The system prompt is the golden less its clock section; the tools and
    # the tool choice are the golden's.
    assert recorded["system_prompt"] == _without_clock_section(
        golden["system_prompt"],
    )
    assert recorded["tools"] == golden["tools"]
    assert recorded["tool_choice"] == golden["tool_choice"]


# ── (d) a persistent session states the clock once ────────────────────────


@pytest.mark.asyncio
@pytest.mark.timeout(180)
async def test_first_message_a_persistent_sessions_next_request_has_no_clock(
    monkeypatch,
    scripted_model,
):
    from unify.actor.code_act_actor import CodeActActor

    monkeypatch.setattr(SETTINGS, "UNIFY_CLOCK_PLACEMENT", "first_message")
    line = _set_clock(monkeypatch, EARLY)
    model = scripted_model(actor=[reply("First answer."), reply("Second answer.")])
    actor = CodeActActor()
    try:
        handle = await actor.act("First request.", persist=True, can_store=False)
        first = await h._next_response(handle)
        # The clock moves on before the next request.
        _set_clock(monkeypatch, LATE)
        await handle.submit("Second request.")
        second = await h._next_response(handle)
        await handle.stop()
        await asyncio.wait_for(handle.result(), 60)
    finally:
        await actor.close()
    assert (first["content"], second["content"]) == (
        "First answer.",
        "Second answer.",
    )
    assert model.kinds() == ["actor", "actor"]
    one, two = model.of("actor")
    assert _first_user(one.messages).splitlines()[0] == line
    # The second request extends the first and states the clock nowhere new.
    assert _prefix_through_first_user(two.messages) == _prefix_through_first_user(
        one.messages,
    )
    assert _clock_lines(two.messages) == [line]
    later = [
        _text(m["content"])
        for m in two.messages[len(one.messages) :]
        if m["role"] == "user"
    ]
    assert later and any("Second request." in text for text in later)
    assert not any(CLOCK_PREFIX in text for text in later)
    model.assert_used_up()


# ── (e) the session's forks keep its prefix ───────────────────────────────


@pytest.mark.asyncio
@pytest.mark.timeout(180)
async def test_first_message_the_storage_review_fork_reuses_the_session_prefix(
    monkeypatch,
    scripted_model,
):
    from unify.actor.code_act_actor import CodeActActor

    monkeypatch.setattr(SETTINGS, "UNIFY_CLOCK_PLACEMENT", "first_message")
    line = _set_clock(monkeypatch, EARLY)
    model = scripted_model(
        actor=[reply(calls=[cell("print('done')")]), reply("Finished.")],
        review=Always(reply("Nothing to store.")),
    )
    actor = CodeActActor()
    try:
        handle = await actor.act("Do the task.", can_store=True)
        assert await asyncio.wait_for(handle.result(), 60) == "Finished."
        await asyncio.wait_for(handle._lifecycle_task, 60)
    finally:
        await actor.close()
    first_actor, last_actor = model.of("actor")
    reviews = model.of("review")
    assert reviews, f"no storage review ran: {model.kinds()}"
    session_prefix = _prefix_through_first_user(first_actor.messages)
    assert _first_user(first_actor.messages).splitlines()[0] == line
    assert SECTION not in _text(first_actor.messages[0]["content"])
    for review in reviews:
        assert _prefix_through_first_user(review.messages) == session_prefix
        assert _clock_lines(review.messages) == [line]
    # The first review request is the session's last request, then its reply,
    # then the review's own message.
    n = len(last_actor.messages)
    assert reviews[0].messages[:n] == last_actor.messages
    assert reviews[0].tool_names == last_actor.tool_names


@pytest.mark.asyncio
@pytest.mark.timeout(180)
async def test_first_message_the_compression_fork_reuses_the_session_prefix(
    monkeypatch,
    scripted_model,
):
    from unify.actor.code_act_actor import CodeActActor

    monkeypatch.setattr(SETTINGS, "UNIFY_CLOCK_PLACEMENT", "first_message")
    line = _set_clock(monkeypatch, EARLY)
    model = scripted_model(
        actor=[
            # A context-full turn: the next one must compress.
            reply(calls=[cell("print('big')")], prompt_tokens=900_000),
            reply("Finished."),
        ],
        compress_turn=[reply(calls=[("compress_context", {})])],
        compression_fork=[reply(SUMMARY)],
    )
    actor = CodeActActor()
    try:
        handle = await actor.act("Do the task.", can_store=False)
        # The clock moves on before the session restarts from the summary.
        _set_clock(monkeypatch, LATE)
        assert await asyncio.wait_for(handle.result(), 60) == "Finished."
    finally:
        await actor.close()
    assert model.kinds() == ["actor", "compress_turn", "compression_fork", "actor"]
    first_actor, restarted = model.of("actor")
    (turn,) = model.of("compress_turn")
    (fork,) = model.of("compression_fork")
    session_prefix = _prefix_through_first_user(first_actor.messages)
    assert _first_user(first_actor.messages).splitlines()[0] == line
    # The forced turn and the summary fork extend the session's conversation.
    for request in (turn, fork):
        assert _prefix_through_first_user(request.messages) == session_prefix
        assert _clock_lines(request.messages) == [line]
    assert fork.messages[: len(turn.messages)] == turn.messages
    # The restarted session keeps the system prompt and states the time the
    # session started with, once.
    assert restarted.messages[0] == first_actor.messages[0]
    assert _clock_lines(restarted.messages) == [line]
    assert SUMMARY in json.dumps(restarted.messages)
    model.assert_used_up()
