"""Symbolic: ``UNIFY_STORE_ADMISSION`` gates the review that runs when a session ends.

The setting names a JSON file written by an external check of the session's
outcome. Phase 2 of ``_StorageCheckHandle`` reads it once the session has
ended and runs the review only for ``{"admit": true}``; a missing, malformed
or non-admitting file skips the review (fail-closed) with a
``storage_review_skipped`` notification naming why. While the setting is set,
``act()`` withholds the session's own library write tools, does not describe
in-session storage, offers no storage tool before compression, and turns
turn-level reviews off, so the libraries change only through an admitted
review. Unset, nothing is read and the review runs as shipped. No model is
called: the inner handle and the review loop are mocked.
"""

from __future__ import annotations

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import unify.actor.code_act_actor as code_act_actor
from unify.actor.code_act_actor import (
    CodeActActor,
    _read_store_admission,
    _StorageCheckHandle,
)
from unify.actor.prompt_builders import build_code_act_prompt
from unify.settings import SETTINGS

_WRITE_TOOLS = {
    "store_skills",
    "FunctionManager_add_functions",
    "FunctionManager_delete_function",
    "FunctionManager_reconcile_dependencies",
    "GuidanceManager_add_guidance",
    "GuidanceManager_update_guidance",
    "GuidanceManager_delete_guidance",
    "GuidanceManager_reconcile_dependencies",
}


def _inner_handle(result_future: "asyncio.Future[str]") -> MagicMock:
    inner = MagicMock()

    async def _result():
        return await result_future

    inner.result = _result
    inner.next_notification = AsyncMock(
        side_effect=lambda: asyncio.Event().wait(),
    )
    mock_client = MagicMock()
    mock_client.messages = [{"role": "user", "content": "do something"}]
    inner._client = mock_client
    mock_task = MagicMock()
    mock_task.get_ask_tools = MagicMock(return_value={})
    mock_task.get_completed_tool_metadata = MagicMock(return_value={})
    inner._task = mock_task
    return inner


async def _run_session_end() -> tuple[MagicMock, list[dict]]:
    """End a mocked session and return the review-loop mock and the
    notifications the handle emitted."""
    result_future: asyncio.Future[str] = asyncio.get_event_loop().create_future()
    inner = _inner_handle(result_future)
    actor = MagicMock()
    actor.function_manager = None
    actor.guidance_manager = None
    with (
        patch("unify.actor.code_act_actor._start_storage_check_loop") as mock_loop,
        patch(
            "unify.actor.code_act_actor.publish_manager_method_event",
            new_callable=AsyncMock,
        ),
    ):
        mock_loop.return_value = None
        handle = _StorageCheckHandle(inner=inner, actor=actor)
        result_future.set_result("done")
        for _ in range(500):
            if handle.done():
                break
            await asyncio.sleep(0.01)
        assert handle.done()
    notes = []
    while not handle._notification_q.empty():
        notes.append(handle._notification_q.get_nowait())
    return mock_loop, notes


# ---------------------------------------------------------------------------
# Reading the verdict
# ---------------------------------------------------------------------------


def test_verdict_admit_true(tmp_path):
    path = tmp_path / "verdict.json"
    path.write_text(json.dumps({"admit": True, "reason": "check passed"}))
    admitted, reason = _read_store_admission(str(path))
    assert admitted is True
    assert "check passed" in reason


@pytest.mark.parametrize(
    "content",
    [
        json.dumps({"admit": False, "reason": "check failed"}),
        json.dumps({"admit": "true"}),
        json.dumps({"admit": 1}),
        json.dumps({"reason": "no admit key"}),
        json.dumps([{"admit": True}]),
        json.dumps(True),
        "{not json",
        "",
        "\xff\xfe",
    ],
)
def test_verdict_anything_else_does_not_admit(tmp_path, content):
    path = tmp_path / "verdict.json"
    path.write_bytes(content.encode("latin-1"))
    admitted, reason = _read_store_admission(str(path))
    assert admitted is False
    assert reason


def test_verdict_missing_does_not_admit(tmp_path):
    admitted, reason = _read_store_admission(str(tmp_path / "absent.json"))
    assert admitted is False
    assert "no admission verdict" in reason


def test_verdict_unreadable_does_not_admit(tmp_path):
    admitted, reason = _read_store_admission(str(tmp_path))  # a directory
    assert admitted is False
    assert "unreadable" in reason


def test_verdict_too_large_does_not_admit(tmp_path):
    path = tmp_path / "verdict.json"
    path.write_text(json.dumps({"admit": True, "pad": "x" * 70000}))
    admitted, _ = _read_store_admission(str(path))
    assert admitted is False


def test_admission_is_off_by_default():
    assert SETTINGS.UNIFY_STORE_ADMISSION == ""


# ---------------------------------------------------------------------------
# Phase 2 of _StorageCheckHandle
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.timeout(30)
async def test_admitted_session_is_reviewed(tmp_path, monkeypatch):
    path = tmp_path / "verdict.json"
    path.write_text(json.dumps({"admit": True}))
    monkeypatch.setattr(SETTINGS, "UNIFY_STORE_ADMISSION", str(path))
    mock_loop, notes = await _run_session_end()
    mock_loop.assert_called_once()
    assert not [n for n in notes if n.get("type") == "storage_review_skipped"]


@pytest.mark.asyncio
@pytest.mark.timeout(30)
@pytest.mark.parametrize(
    "content, expected",
    [
        (json.dumps({"admit": False, "reason": "check failed"}), "check failed"),
        ("{not json", "not JSON"),
        (None, "no admission verdict"),
    ],
)
async def test_unadmitted_session_is_not_reviewed(
    tmp_path,
    monkeypatch,
    content,
    expected,
):
    path = tmp_path / "verdict.json"
    if content is not None:
        path.write_text(content)
    monkeypatch.setattr(SETTINGS, "UNIFY_STORE_ADMISSION", str(path))
    mock_loop, notes = await _run_session_end()
    mock_loop.assert_not_called()
    skipped = [n for n in notes if n.get("type") == "storage_review_skipped"]
    assert len(skipped) == 1
    assert expected in skipped[0]["message"]


@pytest.mark.asyncio
@pytest.mark.timeout(30)
async def test_unset_reviews_without_reading(tmp_path, monkeypatch):
    """Unset: the review runs as shipped, whatever a verdict file says."""
    path = tmp_path / "verdict.json"
    path.write_text(json.dumps({"admit": False}))
    monkeypatch.setattr(SETTINGS, "UNIFY_STORE_ADMISSION", "")
    reads = []
    monkeypatch.setattr(
        code_act_actor,
        "_read_store_admission",
        lambda p: reads.append(p) or (False, "read"),
    )
    mock_loop, notes = await _run_session_end()
    mock_loop.assert_called_once()
    assert reads == []
    assert not [n for n in notes if n.get("type") == "storage_review_skipped"]


# ---------------------------------------------------------------------------
# act(): tools, prompt and turn reviews of an admission-gated session
# ---------------------------------------------------------------------------


async def _start_act(monkeypatch, *, persist: bool) -> dict:
    """Run ``act()`` up to its tool loop and capture what the loop was given."""
    captured: dict = {}

    def fake_loop(client, message, tools, **kwargs):
        captured["tools"] = dict(tools)
        captured["policy"] = kwargs.get("tool_policy")
        captured["extra_compression_tools"] = kwargs.get("extra_compression_tools")
        handle = MagicMock()
        handle.result = AsyncMock(return_value="done")
        handle.next_notification = AsyncMock(
            side_effect=lambda: asyncio.Event().wait(),
        )
        handle._client = MagicMock(messages=[])
        return handle

    real_builder = build_code_act_prompt

    def spy_builder(**kwargs):
        captured["prompt_kwargs"] = dict(kwargs)
        captured["prompt"] = real_builder(**kwargs)
        return captured["prompt"]

    monkeypatch.setattr(code_act_actor, "start_async_tool_loop", fake_loop)
    monkeypatch.setattr(code_act_actor, "build_code_act_prompt", spy_builder)
    monkeypatch.setattr(code_act_actor, "_start_storage_check_loop", lambda **kw: None)
    monkeypatch.setattr(
        code_act_actor,
        "publish_manager_method_event",
        AsyncMock(),
    )
    actor = CodeActActor(timeout=30)
    try:
        handle = await actor.act("Do something", persist=persist, can_store=True)
        captured["handle"] = handle
        # The default discovery-first policy: once both libraries have been
        # searched, the full (statically filtered) tool set is visible.
        searched = ["FunctionManager_search_functions", "GuidanceManager_search"]
        decision = captured["policy"](5, dict(captured["tools"]), searched)
        captured["visible_later"] = set(decision[1])
    finally:
        try:
            await actor.close()
        except Exception:
            pass
    return captured


@pytest.mark.asyncio
@pytest.mark.timeout(60)
async def test_gated_session_withholds_write_tools(tmp_path, monkeypatch):
    monkeypatch.setattr(
        SETTINGS,
        "UNIFY_STORE_ADMISSION",
        str(tmp_path / "verdict.json"),
    )
    captured = await _start_act(monkeypatch, persist=False)
    assert not _WRITE_TOOLS & set(captured["tools"])
    assert "execute_code" in captured["visible_later"]
    assert not _WRITE_TOOLS & captured["visible_later"]
    # Reads stay.
    assert "FunctionManager_search_functions" in captured["tools"]
    assert "GuidanceManager_search" in captured["tools"]
    assert captured["extra_compression_tools"] is None
    assert captured["prompt_kwargs"]["can_store"] is False
    assert captured["prompt_kwargs"]["library_read_only"] is True
    assert "### Library Writes" in captured["prompt"]
    assert "### Skill Storage" not in captured["prompt"]
    # The session is still wrapped, so an admitted review can run at its end.
    assert isinstance(captured["handle"], _StorageCheckHandle)


@pytest.mark.asyncio
@pytest.mark.timeout(60)
async def test_gated_session_has_no_turn_reviews(tmp_path, monkeypatch):
    monkeypatch.setattr(SETTINGS, "UNIFY_TURN_STORAGE_REVIEWS", True)
    monkeypatch.setattr(
        SETTINGS,
        "UNIFY_STORE_ADMISSION",
        str(tmp_path / "verdict.json"),
    )
    captured = await _start_act(monkeypatch, persist=True)
    assert isinstance(captured["handle"], _StorageCheckHandle)
    assert captured["handle"]._turn_reviews_enabled is False


@pytest.mark.asyncio
@pytest.mark.timeout(60)
async def test_unset_session_is_unchanged(monkeypatch):
    monkeypatch.setattr(SETTINGS, "UNIFY_TURN_STORAGE_REVIEWS", True)
    monkeypatch.setattr(SETTINGS, "UNIFY_STORE_ADMISSION", "")
    captured = await _start_act(monkeypatch, persist=True)
    assert {
        "store_skills",
        "FunctionManager_add_functions",
        "GuidanceManager_add_guidance",
    } <= set(captured["tools"])
    assert captured["extra_compression_tools"] == ["store_skills"]
    assert captured["prompt_kwargs"]["can_store"] is True
    assert "library_read_only" not in captured["prompt_kwargs"]
    assert "### Library Writes" not in captured["prompt"]
    assert captured["handle"]._turn_reviews_enabled is True


def test_prompt_default_has_no_read_only_notice():
    """The builder's default leaves the prompt as shipped."""
    tools = {"execute_code": None, "FunctionManager_search_functions": None}
    for can_store in (True, False):
        for persist in (True, False):
            prompt = build_code_act_prompt(
                environments={},
                tools=tools,
                can_store=can_store,
                persist=persist,
            )
            assert "### Library Writes" not in prompt
            assert prompt == build_code_act_prompt(
                environments={},
                tools=tools,
                can_store=can_store,
                persist=persist,
                library_read_only=False,
            )
