"""Symbolic: under ``UNIFY_CACHE_DISCIPLINE`` a session lists the tools its mode can ever call.

The fixed tool list used to hold every library write an admission-gated
session withholds, refused by rule on every call. On AppWorld frozen dev
that was 8 tools and about 2.4k tokens a call (+$0.0006 of the +$0.00107
per task the switch added), and no V1 cell ever refused one: in a frozen
library nothing can call them. They stay listed, masked, only where
something sending the session's list may call them later -- the review
that forks the session after an admitted outcome. Masking is kept for what
becomes available later in the same list (the discovery gate, that fork).

``act()`` runs as a caller runs it; requests are captured at unillm's
transport (``tests/cache_discipline_helpers.py``), so nothing leaves the
process.
"""

from __future__ import annotations

import asyncio
import json

import pytest

from tests import cache_discipline_helpers as h
from unify.actor import code_act_actor as caa
from unify.settings import SETTINGS

# The library writes admission withholds that a default actor registers.
WRITES = {
    "store_skills",
    "FunctionManager_add_functions",
    "FunctionManager_delete_function",
    "FunctionManager_reconcile_dependencies",
    "GuidanceManager_add_guidance",
    "GuidanceManager_update_guidance",
    "GuidanceManager_delete_guidance",
    "GuidanceManager_reconcile_dependencies",
}
LESSON = {"title": "A lesson", "content": "What the task taught."}


@pytest.fixture
def modes(monkeypatch):
    def set_(*, admission: str, fork: bool, discipline: bool = True) -> None:
        monkeypatch.setattr(SETTINGS, "UNIFY_CACHE_DISCIPLINE", discipline)
        monkeypatch.setattr(SETTINGS, "UNIFY_REVIEW_FORK", fork)
        monkeypatch.setattr(SETTINGS, "UNIFY_STORE_ADMISSION", admission)
        monkeypatch.setattr(SETTINGS, "UNIFY_FUNCTION_PATCH", False)

    return set_


@pytest.fixture
def info_lines(monkeypatch):
    lines: list[str] = []
    original = caa.logger.info

    def capture(msg, *args, **kwargs):
        lines.append(str(msg))
        return original(msg, *args, **kwargs)

    monkeypatch.setattr(caa.logger, "info", capture)
    return lines


def _is_review(request: dict) -> bool:
    return any(
        m.get("role") == "user"
        and isinstance(m.get("content"), str)
        and m["content"].startswith("## Storage Review")
        for m in request["messages"]
    )


def _calls(message: dict) -> list[dict]:
    """A message's tool calls as dicts (the transcript may hold SDK objects)."""
    if message.get("role") != "assistant":
        return []
    return [
        call.model_dump() if hasattr(call, "model_dump") else call
        for call in (message.get("tool_calls") or [])
    ]


def _called(request: dict) -> list[str]:
    return [call["function"]["name"] for m in request["messages"] for call in _calls(m)]


def _tool_replies(request: dict) -> dict[str, str]:
    """Each tool reply in *request*, by the name of the call it answers."""
    names = {
        call["id"]: call["function"]["name"]
        for m in request["messages"]
        for call in _calls(m)
    }
    return {
        names.get(m.get("tool_call_id"), "?"): str(m.get("content"))
        for m in request["messages"]
        if m.get("role") == "tool"
    }


def _respond(provider: h.Provider, *, search: bool):
    """A scripted model that reads the request it answers.

    The session searches both libraries (when *search*), then tries to add
    guidance, then answers; a review adds the guidance, then summarises.
    """

    def reply():
        request = provider.requests[-1]
        called = _called(request)
        if _is_review(request):
            start = max(
                i
                for i, m in enumerate(request["messages"])
                if _is_review({"messages": [m]})
            )
            if not _called({"messages": request["messages"][start:]}):
                return h.completion(
                    calls=[("GuidanceManager_add_guidance", LESSON)],
                )
            return h.completion(content="Stored one lesson.")
        if search and "FunctionManager_search_functions" not in called:
            return h.completion(
                calls=[
                    ("FunctionManager_search_functions", {"query": "files"}),
                    ("GuidanceManager_search", {"k": 3}),
                ],
            )
        if "GuidanceManager_add_guidance" not in called:
            return h.completion(calls=[("GuidanceManager_add_guidance", LESSON)])
        return h.completion(content="done")

    return reply


async def _act(*, gate: bool = True) -> tuple[list[dict], list[dict]]:
    """Run ``act()`` and its storage review; ``(session requests, review requests)``.

    ``gate=False`` runs the actor without the discovery gate, so no search
    is still running when the session ends: a search that finishes after
    the next turn leaves a placeholder reply, and the review then does not
    fork (``_review_fork_source``: "unanswered tool calls").
    """
    from unify.actor.code_act_actor import CodeActActor

    actor = CodeActActor() if gate else CodeActActor(tool_policy=None)
    try:
        with h.scripted(()) as provider:
            provider.replies = [_respond(provider, search=gate)] * 20
            handle = await actor.act("List the files in the workspace.", persist=False)
            await asyncio.wait_for(handle.result(), 60)
            lifecycle = getattr(handle, "_lifecycle_task", None)
            if lifecycle is not None:
                await asyncio.wait_for(lifecycle, 60)
    finally:
        await actor.close()
    requests = provider.requests
    session = [r for r in requests if not _is_review(r)]
    review = [r for r in requests if _is_review(r)]
    return h.session_requests(session), review


def _names(request: dict) -> set[str]:
    return {t["function"]["name"] for t in request["tools"]}


def _assert_one_list(requests: list[dict]) -> str:
    """Every request sends the same tool list, byte for byte; returns it."""
    lists = {h.request_bytes(r)["tools"] for r in requests}
    assert len(lists) == 1, [sorted(json.loads(t)) for t in lists]
    return lists.pop()


# ── frozen library: the writes are never listed ──────────────────────────


@pytest.mark.asyncio
@pytest.mark.timeout(120)
async def test_a_frozen_session_never_lists_the_writes_and_keeps_one_list(
    modes,
    info_lines,
):
    modes(admission="never", fork=True)
    session, review = await _act()
    assert len(session) >= 3
    _assert_one_list(session)
    names = _names(session[0])
    assert not WRITES & names
    assert {
        "FunctionManager_search_functions",
        "GuidanceManager_search",
        "execute_code",
    } <= names
    # A write the model calls anyway is not in the list: refused, not run.
    refusal = _tool_replies(session[-1])["GuidanceManager_add_guidance"]
    assert "not in the current tool schema" in refusal
    # No review runs, and no verdict file is looked for.
    assert review == []
    assert any(caa._STORE_ADMISSION_NEVER_REASON in line for line in info_lines)
    assert not any("no admission verdict" in line for line in info_lines)


@pytest.mark.asyncio
@pytest.mark.timeout(120)
async def test_a_gated_session_whose_review_does_not_fork_never_lists_them(
    modes,
    tmp_path,
):
    verdict = tmp_path / "verdict.json"
    verdict.write_text(json.dumps({"admit": False, "reason": "not now"}))
    modes(admission=str(verdict), fork=False)
    session, _review = await _act()
    _assert_one_list(session)
    assert not WRITES & _names(session[0])


# ── a fork will reuse the list: listed from the start, refused until then ──


@pytest.mark.asyncio
@pytest.mark.timeout(120)
async def test_writes_a_fork_will_use_are_listed_from_the_start_and_refused_until_it(
    modes,
    info_lines,
    tmp_path,
):
    verdict = tmp_path / "verdict.json"
    verdict.write_text(json.dumps({"admit": True, "reason": "check passed"}))
    modes(admission=str(verdict), fork=True)
    session, review = await _act(gate=False)
    listed = _assert_one_list(session)
    assert WRITES <= _names(session[0])  # on the very first call
    refusal = _tool_replies(session[-1])["GuidanceManager_add_guidance"]
    assert caa._ADMISSION_MASK_RULE in refusal
    # The review is the fork: the session's exact list, and its last request
    # is a byte prefix of the review's first.
    assert review, [line for line in info_lines if "StorageCheck" in line]
    assert _assert_one_list(review) == listed
    sent = [json.dumps(m, default=str) for m in session[-1]["messages"]]
    first = [json.dumps(m, default=str) for m in review[0]["messages"]]
    assert first[: len(sent)] == sent
    # There the write is available and runs, once.
    assert len(review) == 2
    stored = _tool_replies(review[-1])["GuidanceManager_add_guidance"]
    assert "guidance created successfully" in stored


# ── with the switch off nothing changes ─────────────────────────────────


@pytest.mark.asyncio
@pytest.mark.timeout(120)
async def test_off_a_frozen_session_sends_the_shipped_lists(modes, info_lines):
    modes(admission="never", fork=False, discipline=False)
    session, review = await _act()
    assert not any(WRITES & _names(r) for r in session)
    assert review == []
    assert any(caa._STORE_ADMISSION_NEVER_REASON in line for line in info_lines)


@pytest.mark.parametrize(
    ("value", "never"),
    [("never", True), (" NEVER ", True), ("", False), ("/tmp/never", False)],
)
def test_only_the_word_never_freezes_the_library(value, never):
    assert caa._store_admission_never(value) is never
