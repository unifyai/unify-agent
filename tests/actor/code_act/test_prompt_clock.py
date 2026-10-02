"""Symbolic: ``UNIFY_PROMPT_CLOCK`` and ``UNIFY_CACHE_AFFINITY_SCOPE=static``.

The actor's system prompt ends with the clock (minute resolution) and the
filesystem context (the workspace's paths). The ``prefix`` affinity key
hashes the whole system prompt, so it changed every minute and workspace:
on ARC A 253 sessions of one configuration were sent under 199 keys, each a
sticky route to a replica that had not cached the prefix. Provider guidance
is to put timestamps at the end of the prompt or in later messages.

``UNIFY_PROMPT_CLOCK=message`` moves both sections, sampled once per
session, to the start of the first user message, so the system prompt is
the same for every session of one configuration. ``static`` keys a session
on its system prompt without those sections, so sessions share a key
whichever of the two places they are in. Requests are captured at unillm's
transport; the clock is frozen by ``static_now`` and set per session here.
"""

from __future__ import annotations

import asyncio
import re

import pytest

from tests import cache_discipline_helpers as h
from unify.actor import prompt_builders as pb
from unify.common._async_tool import cache_discipline as cd
from unify.settings import SETTINGS

TASK = "List the files in the workspace."
CLOCK = "### Current Time"
FILES = "### Filesystem Context"


@pytest.fixture
def switches(monkeypatch):
    def set_(*, clock: str = "", scope: str = "prefix", discipline: bool = False):
        monkeypatch.setattr(SETTINGS, "UNIFY_PROMPT_CLOCK", clock)
        monkeypatch.setattr(SETTINGS, "UNIFY_CACHE_AFFINITY_SCOPE", scope)
        monkeypatch.setattr(SETTINGS, "UNIFY_CACHE_DISCIPLINE", discipline)

    return set_


@pytest.fixture
def session_at(monkeypatch, tmp_path):
    """Put the next session at a given minute and workspace."""

    def set_(minute: int, workspace: str) -> None:
        from unify.common import prompt_helpers

        stamp = f"Friday, June 13, 2025 at 12:{minute:02d} PM UTC"
        monkeypatch.setattr(prompt_helpers, "now", lambda *a, **k: stamp)
        root = tmp_path / workspace
        root.mkdir(exist_ok=True)
        monkeypatch.setattr("unify.workspace.get_local_root", lambda: str(root))

    return set_


async def _act() -> list[dict]:
    from unify.actor.code_act_actor import CodeActActor

    actor = CodeActActor()
    try:
        with h.scripted(h.ACTOR_REPLIES) as provider:
            handle = await actor.act(TASK, persist=False)
            await asyncio.wait_for(handle.result(), 60)
    finally:
        await actor.close()
    return h.session_requests(provider.requests)


def _first_user(request: dict) -> str:
    return next(m["content"] for m in request["messages"] if m["role"] == "user")


# ── the clock's place ────────────────────────────────────────────────────


@pytest.mark.asyncio
@pytest.mark.timeout(120)
async def test_on_the_clock_and_paths_open_the_first_user_message(
    switches,
    static_now,
):
    switches(clock="")
    off = await _act()
    switches(clock="message")
    on = await _act()
    system_off, system_on = off[0]["messages"][0], on[0]["messages"][0]
    assert system_on["role"] == system_off["role"] == "system"
    assert CLOCK in system_off["content"] and FILES in system_off["content"]
    assert CLOCK not in system_on["content"] and FILES not in system_on["content"]
    stamp = static_now.strftime("%A, %B %d, %Y at %I:%M %p UTC")
    first = _first_user(on[0])
    assert first.startswith(f"{CLOCK}\n\nThe current date and time is **{stamp}**")
    assert first.endswith(f"\n\n---\n\n{TASK}")
    # The same sections, moved: the system prompt as shipped is the one sent
    # with them put back where they were.
    sections = first.split("\n\n---\n\n")[0]
    assert CLOCK in sections and FILES in sections
    head, tail = system_off["content"].split(sections)
    assert head.rstrip("\n") + tail == system_on["content"]
    # Sent once, and never edited: every later request starts the same way.
    for request in on[1:]:
        assert request["messages"][0] == system_on
        assert _first_user(request) == first
    assert h.request_bytes(on[0])["tools"] == h.request_bytes(off[0])["tools"]


@pytest.mark.asyncio
@pytest.mark.timeout(120)
async def test_with_the_library_snapshot_the_clock_comes_first(switches, monkeypatch):
    switches(clock="message")
    monkeypatch.setattr(SETTINGS, "UNIFY_LIBRARY_SNAPSHOT", True)
    on = await _act()
    context, task = _first_user(on[0]).split("\n\n---\n\n")
    assert task == TASK
    sections, snapshot = context.rsplit("\n\n", 1)
    assert sections.startswith(CLOCK) and FILES in sections
    assert re.fullmatch(
        r"Library at task start: \d+ stored functions?, \d+ guidance entr(y|ies)\."
        r"( An empty library is not searched first\.)?",
        snapshot,
    )
    assert CLOCK not in on[0]["messages"][0]["content"]


@pytest.mark.asyncio
@pytest.mark.timeout(120)
async def test_on_the_system_prompt_is_the_same_across_minutes_and_workspaces(
    switches,
    session_at,
):
    switches(clock="message")
    session_at(1, "a")
    first = await _act()
    session_at(2, "b")
    second = await _act()
    assert first[0]["messages"][0] == second[0]["messages"][0]
    assert _first_user(first[0]) != _first_user(second[0])
    assert "12:01 PM" in _first_user(first[0]) and "/a`" in _first_user(first[0])


@pytest.mark.asyncio
@pytest.mark.timeout(120)
async def test_off_the_first_user_message_is_the_task(switches):
    switches(clock="")
    off = await _act()
    assert _first_user(off[0]) == TASK


@pytest.mark.asyncio
async def test_a_compressed_session_restarts_with_the_sections_again(monkeypatch):
    """The restart's first message is a new one; it carries the context too."""
    context = pb.build_session_context({"execute_code": object()})
    counter: dict = {}
    tools = h.make_tools(counter)
    with h.scripted(h.COMPRESS_REPLIES) as provider:
        result = await h._run(
            h.new_client(),
            {"execute_code": tools["execute_code"]},
            "Do the task.",
            interrupt_llm_with_interjections=False,
            first_message_context=context,
        )
    assert result == "done"
    final = provider.requests[-1]["messages"]
    users = [m["content"] for m in final if m["role"] == "user"]
    assert users[0].startswith(context + "\n\n---\n\n")
    assert users[0] != f"{context}\n\n---\n\nDo the task."  # the restart's message
    first = provider.requests[0]["messages"]
    assert _first_user({"messages": first}) == f"{context}\n\n---\n\nDo the task."


# ── the static affinity key ──────────────────────────────────────────────


async def _keys(switches, session_at, monkeypatch, *, clock: str, scope: str):
    """The key each actor session set, before its first request."""
    switches(clock=clock, scope=scope, discipline=True)
    sets = h.install_affinity_api(monkeypatch)
    keys = []
    for minute, workspace in ((1, "a"), (2, "b")):
        session_at(minute, workspace)
        start = len(sets)
        await _act()
        # The session's own key comes first; the storage review after it
        # is another loop, with its own prompt and key.
        key, sent = sets[start]
        assert sent == 0
        keys.append(key)
    return keys


@pytest.mark.asyncio
@pytest.mark.timeout(240)
async def test_static_sessions_at_other_minutes_and_paths_share_one_key(
    switches,
    session_at,
    monkeypatch,
):
    first, second = await _keys(
        switches,
        session_at,
        monkeypatch,
        clock="",
        scope="static",
    )
    assert first == second and len(first) == 32


@pytest.mark.asyncio
@pytest.mark.timeout(240)
async def test_prefix_sessions_at_other_minutes_and_paths_get_other_keys(
    switches,
    session_at,
    monkeypatch,
):
    first, second = await _keys(
        switches,
        session_at,
        monkeypatch,
        clock="",
        scope="prefix",
    )
    assert first != second


@pytest.mark.asyncio
@pytest.mark.timeout(240)
async def test_static_and_prefix_agree_once_the_clock_is_in_the_message(
    switches,
    session_at,
    monkeypatch,
):
    static = await _keys(
        switches,
        session_at,
        monkeypatch,
        clock="message",
        scope="static",
    )
    prefix = await _keys(
        switches,
        session_at,
        monkeypatch,
        clock="message",
        scope="prefix",
    )
    assert static[0] == static[1] == prefix[0] == prefix[1]


def test_static_falls_back_to_the_sent_prompt_without_a_recorded_one(monkeypatch):
    monkeypatch.setattr(SETTINGS, "UNIFY_CACHE_AFFINITY_SCOPE", "static")

    class Client:
        endpoint = h.MODEL
        system_message = "You are a scripted test agent."
        cache_affinity = None

        def set_cache_affinity(self, value):
            self.cache_affinity = value

    plain, recorded = Client(), Client()
    cd.set_static_system_message(recorded, "the static part")
    assert cd.ensure_cache_affinity(plain, []) == cd.prefix_affinity_key(
        h.MODEL,
        "You are a scripted test agent.",
        [],
    )
    assert cd.ensure_cache_affinity(recorded, []) == cd.prefix_affinity_key(
        h.MODEL,
        "the static part",
        [],
    )


def test_without_execute_code_there_are_no_session_sections():
    assert pb.build_session_context({}) == ""
    assert pb.build_session_context({"execute_code": object()}).startswith(CLOCK)


@pytest.mark.parametrize(
    "raw, parsed",
    [("", ""), (None, ""), (" Message ", "message")],
)
def test_the_clock_setting_parses(raw, parsed):
    from unify.settings import ProductionSettings

    assert ProductionSettings.parse_prompt_clock(raw) == parsed


def test_the_clock_setting_refuses_anything_else():
    from unify.settings import ProductionSettings

    with pytest.raises(ValueError):
        ProductionSettings.parse_prompt_clock("system")


def test_the_scope_setting_takes_static():
    from unify.settings import ProductionSettings

    assert ProductionSettings.parse_cache_affinity_scope("STATIC") == "static"
