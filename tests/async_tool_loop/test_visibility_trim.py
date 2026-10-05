"""Symbolic: ``UNIFY_PROMPT_TRIM`` and the "User Visibility Context" message.

The loop appends the message the first time an interjection, a progress
notification or a clarification reaches the model; it says what the user
sees (their messages, notifications, clarification requests, the final
reply) and that "[progress ...]", "[clarification ...]", "[steerable ...]"
and "[askable ...]" messages are not the user's. A lean-all ARC session has
no send_notification and no clarification request, and its lifecycle
announcements are off, yet every one of its 36 sessions got the message on
the requester's second message. With the switch on an interjection appends
it only when the model has a channel to the user besides its final reply,
and whatever appends it, it names only the channels and messages the loop
has. Off, it is as shipped.

The transport is scripted (tests/cache_discipline_helpers.py), as in
test_lean_loop.py, whose loop runner these tests use.
"""

from __future__ import annotations

import asyncio

import pytest

from tests.async_tool_loop.test_lean_loop import LEAN, TOOLS, _batch, _done, _run
from unify.common._async_tool import tools_data as td
from unify.settings import SETTINGS

HEADING = "User Visibility Context"


def _visibility(requests: list[dict]) -> list[str]:
    seen = []
    for r in requests:
        for m in r["messages"]:
            text = str(m.get("content") or "")
            if m.get("role") == "system" and HEADING in text and text not in seen:
                seen.append(text)
    return seen


async def _interject(handle):
    await asyncio.sleep(0.3)
    await handle.interject("ANOTHER_REQUEST")


async def notifying_tool(_notification_up_q: asyncio.Queue | None = None) -> str:
    """Report progress once, then return."""
    if _notification_up_q is not None:
        await _notification_up_q.put({"message": "halfway"})
    await asyncio.sleep(0.6)
    return "NOTIFY_RESULT"


def test_the_trimmed_text_keeps_the_shipped_wording():
    full = td.trimmed_visibility_guidance(notify=True, clarify=True, lifecycle=True)
    assert full == td.USER_VISIBILITY_GUIDANCE
    bare = td.trimmed_visibility_guidance(notify=False, clarify=False, lifecycle=False)
    assert "notifications you emit" not in bare
    assert "clarification requests you send" not in bare
    assert "[steerable" not in bare and "ask_about_completed_tool" not in bare
    assert "1. Their original request" in bare and "2. Your FINAL plain-text" in bare
    assert "[progress <call_id>]" in bare and "[clarification <call_id>]" in bare
    assert bare.rstrip() == bare and bare in td.USER_VISIBILITY_GUIDANCE.replace(
        "2. Any notifications you emit (status updates, progress indicators, etc.)\n"
        "3. Any clarification requests you send asking for more information\n"
        "4. Your FINAL",
        "2. Your FINAL",
    )
    only_notify = td.trimmed_visibility_guidance(
        notify=True,
        clarify=False,
        lifecycle=False,
    )
    assert "2. Any notifications you emit" in only_notify
    assert "3. Your FINAL plain-text" in only_notify


@pytest.mark.asyncio
async def test_off_an_interjection_appends_the_shipped_message(monkeypatch):
    requests, _, _ = await _run(
        monkeypatch,
        [_batch("fast_tool", "slow_tool"), *[_batch("wait")] * 3, *_done()],
        switches={**LEAN, "UNIFY_PROMPT_TRIM": False},
        during=_interject,
    )
    assert _visibility(requests) == [td.USER_VISIBILITY_GUIDANCE]


@pytest.mark.asyncio
async def test_on_an_interjection_without_a_user_channel_appends_nothing(monkeypatch):
    requests, _, _ = await _run(
        monkeypatch,
        [_batch("fast_tool", "slow_tool"), *[_batch("wait")] * 3, *_done()],
        switches={**LEAN, "UNIFY_PROMPT_TRIM": True},
        during=_interject,
    )
    assert any(
        "ANOTHER_REQUEST" in str(m.get("content")) for m in requests[-1]["messages"]
    )
    assert _visibility(requests) == []


@pytest.mark.asyncio
async def test_on_an_interjection_with_notifications_appends_what_applies(monkeypatch):
    requests, _, _ = await _run(
        monkeypatch,
        [_batch("fast_tool", "slow_tool"), *[_batch("wait")] * 3, *_done()],
        switches={**LEAN, "UNIFY_PROMPT_TRIM": True},
        during=_interject,
        on_notify=lambda _text: None,
    )
    assert _visibility(requests) == [
        td.trimmed_visibility_guidance(notify=True, clarify=False, lifecycle=False),
    ]


@pytest.mark.asyncio
async def test_on_a_progress_message_still_appends_it(monkeypatch):
    """A "[progress ...]" message could be taken for the user's: the
    guidance comes with it, naming no channel the loop lacks."""
    requests, _, _ = await _run(
        monkeypatch,
        [_batch("notifying_tool"), *[_batch("wait")] * 3, *_done()],
        switches={**LEAN, "UNIFY_PROMPT_TRIM": True},
        tools={**TOOLS, "notifying_tool": notifying_tool},
    )
    assert any(
        str(m.get("content") or "").startswith("[progress ")
        for r in requests
        for m in r["messages"]
    )
    assert _visibility(requests) == [
        td.trimmed_visibility_guidance(notify=False, clarify=False, lifecycle=False),
    ]


@pytest.mark.asyncio
async def test_on_with_lifecycle_notices_the_announcements_are_explained(monkeypatch):
    requests, _, _ = await _run(
        monkeypatch,
        [_batch("fast_tool", "medium_tool"), *[_batch("wait")] * 3, *_done()],
        switches={**LEAN, "UNIFY_LIFECYCLE_NOTICES": True, "UNIFY_PROMPT_TRIM": True},
    )
    assert _visibility(requests) == [
        td.trimmed_visibility_guidance(notify=False, clarify=False, lifecycle=True),
    ]


def test_the_switch_is_read_once_per_loop(monkeypatch):
    monkeypatch.setattr(SETTINGS, "UNIFY_PROMPT_TRIM", True)
    data = td.ToolsData({}, client=None, logger=None)
    monkeypatch.setattr(SETTINGS, "UNIFY_PROMPT_TRIM", False)
    assert data._prompt_trim is True
