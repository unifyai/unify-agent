"""Symbolic: ``UNIFY_OUTCOME`` gives the storage review the environment's checked outcome.

The review decided what to keep from the agent's own account of its work,
which is wrong often enough to matter: 31 of Unify's 42 failed AppWorld runs
ended with the agent calling the task a success. The environment posts its
checker's verdict into the harness process (``unify.outcome.post``, or an
``{"outcome": ...}`` line on ``unify act --jsonl``'s stdin), and the review,
forked or standalone, reads it in its own section. Its "Final Result" is
the agent's last reply before the outcome arrived, not the stop notice every
persistent session used to end on. ``UNIFY_REVIEW_FAILED=lessons`` reviews
a failed run with function writes refused and guidance writes allowed.

Requests are captured at unillm's transport (``tests/cache_discipline_helpers.py``);
nothing leaves the process.
"""

from __future__ import annotations

import asyncio
import json
import os
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from tests import cache_discipline_helpers as h
from unify import outcome as outcome_mod
from unify.actor import code_act_actor as caa
from unify.settings import SETTINGS

SESSION_REPLY = "All done: the email was sent to Kim."
CLOSING_REPLY = "Understood."
CLOSING = "The task is over."
REVIEW_SUMMARY = "Nothing worth storing."

FAILED = {
    "solved": False,
    "score": 0.5,
    "source": "grader",
    "checks": [
        {"name": "email_sent", "passed": False, "reason": "no email to Kim"},
        {"name": "no_side_effects", "passed": True, "reason": ""},
    ],
}


@pytest.fixture
def switches(monkeypatch):
    def set_(
        *,
        outcome: bool = False,
        review_failed: str = "",
        discipline: bool = False,
        fork: bool = False,
        admission: str = "",
    ) -> None:
        monkeypatch.setattr(SETTINGS, "UNIFY_OUTCOME", outcome)
        monkeypatch.setattr(SETTINGS, "UNIFY_REVIEW_FAILED", review_failed)
        monkeypatch.setattr(SETTINGS, "UNIFY_CACHE_DISCIPLINE", discipline)
        monkeypatch.setattr(SETTINGS, "UNIFY_REVIEW_FORK", fork)
        monkeypatch.setattr(SETTINGS, "UNIFY_STORE_ADMISSION", admission)

    set_()
    return set_


# ── the schema ───────────────────────────────────────────────────────────


def test_an_outcome_is_normalized_and_bounded():
    checks = [
        {"name": f"c{i}", "passed": i % 3 != 0, "reason": "r" * 1000} for i in range(30)
    ]
    out = outcome_mod.normalize(
        {"solved": False, "score": 3, "checks": checks, "source": "grader", "x": 1},
    )
    assert out["solved"] is False and out["score"] == 3
    assert out["checks_total"] == 30
    assert out["checks_passed"] == 20
    assert len(out["checks"]) == outcome_mod.MAX_CHECKS
    # the ten failed checks come first, then passed ones in order
    assert [c["name"] for c in out["checks"][:10]] == [f"c{i}" for i in range(0, 30, 3)]
    assert all(c["passed"] is True for c in out["checks"][10:])
    assert all(len(c["reason"]) <= outcome_mod.MAX_REASON for c in out["checks"])
    assert "x" not in out


def test_an_empty_outcome_is_all_unknown():
    assert outcome_mod.normalize({}) == {
        "solved": None,
        "score": None,
        "checks": [],
        "checks_total": 0,
        "checks_passed": 0,
        "source": "unspecified",
    }


@pytest.mark.parametrize(
    "raw",
    [
        [],
        "solved",
        {"solved": "yes"},
        {"score": True},
        {"score": float("nan")},
        {"checks": {"a": 1}},
        {"checks": ["a"]},
        {"checks": [{"name": "a", "passed": "no"}]},
        {"source": 3},
    ],
)
def test_a_malformed_outcome_is_refused(raw):
    with pytest.raises(outcome_mod.OutcomeError):
        outcome_mod.normalize(raw)


class _Receiver:
    """Stands in for a session's handle (registered weakly, as a handle is)."""

    def __init__(self, session_id: str | None = None) -> None:
        self.got: list[dict] = []
        self.outcome_session_id = session_id
        self.stopped = None
        self.interjections: list[str] = []

    def receive_outcome(self, outcome: dict) -> None:
        self.got.append(outcome)

    async def stop(self, reason=None) -> None:
        self.stopped = reason

    async def interject(self, message: str) -> None:
        self.interjections.append(message)


def test_post_needs_the_switch_and_a_live_session(switches):
    receiver = _Receiver()
    outcome_mod.register("s1", receiver)
    with pytest.raises(outcome_mod.OutcomeError, match="UNIFY_OUTCOME is off"):
        outcome_mod.post("s1", FAILED)
    switches(outcome=True)
    with pytest.raises(outcome_mod.OutcomeError, match="no session"):
        outcome_mod.post("s2", FAILED)
    assert outcome_mod.post("s1", FAILED)["solved"] is False
    assert receiver.got[0]["checks"][0]["name"] == "email_sent"


def test_the_section_says_whose_verdict_it_is():
    text = outcome_mod.render(outcome_mod.normalize(FAILED))
    assert text.startswith(outcome_mod.OUTCOME_HEADER + "\n")
    assert "- Solved: no" in text
    assert "- Score: 0.5" in text
    assert "- Checks: 1 of 2 passed" in text
    assert "FAILED `email_sent`: no email to Kim" in text
    assert "passed `no_side_effects`" in text
    assert "not from the agent" in text
    assert outcome_mod.render(None) == ""
    assert caa._storage_review_outcome_note() == ""


# ── scenarios: a persistent session, its outcome, its review ────────────


async def _next(handle, kinds) -> dict:
    while True:
        note = await asyncio.wait_for(handle.next_notification(), 30)
        if isinstance(note, dict) and note.get("type") in kinds:
            return note


def _replies(review):
    return (
        lambda: h.completion(calls=[("FunctionManager_list_functions", {})]),
        lambda: h.completion(content=SESSION_REPLY),
        lambda: h.completion(content=CLOSING_REPLY),
        *review,
    )


async def _persistent_review(
    review=(lambda: h.completion(content=REVIEW_SUMMARY),),
    *,
    outcome=None,
    closing=True,
):
    """A persistent session: one turn, the outcome, a closing turn, its end."""
    from unify.actor.code_act_actor import (
        SESSION_ENDED,
        CodeActActor,
        _StorageCheckHandle,
    )
    from unify.common.async_tool_loop import start_async_tool_loop

    actor = CodeActActor()
    posted = None
    try:
        with h.scripted(_replies(review)) as provider:
            inner = start_async_tool_loop(
                h.new_client("You are a scripted actor."),
                "Send the email to Kim.",
                h.session_tools(actor),
                loop_id="CodeActActor.act",
                log_steps=False,
                timeout=60,
                persist=True,
            )
            handle = _StorageCheckHandle(inner=inner, actor=actor)
            await _next(handle, ("response",))
            if outcome is not None:
                posted = outcome_mod.post(handle.outcome_session_id, outcome)
            if closing:
                await handle.interject(CLOSING)
                await _next(handle, ("response",))
            else:
                provider.replies.pop(0)
            await handle.stop(SESSION_ENDED)
            note = await _next(
                handle,
                ("storage_review_complete", "storage_review_skipped"),
            )
            await asyncio.wait_for(handle._lifecycle_task, 30)
    finally:
        await actor.close()
    return note, provider.requests, handle, posted


def _review_text(request: dict) -> str:
    """The review's rulebook: its system prompt, or the fork's appended message."""
    first, last = request["messages"][0], request["messages"][-1]
    if "## Storage Review" in str(last.get("content")):
        return last["content"]
    return first["content"]


@pytest.mark.asyncio
async def test_the_outcome_reaches_the_standalone_review(switches):
    switches(outcome=True)
    note, requests, _handle, _ = await _persistent_review(outcome=FAILED)
    assert note["message"] == REVIEW_SUMMARY
    text = _review_text(requests[3])
    assert "## Completed Trajectory" in text
    section = text.index(outcome_mod.OUTCOME_HEADER)
    final = text.index("## Final Result\n\n")
    assert text.index("## Completed Trajectory") < section < final
    assert "FAILED `email_sent`: no email to Kim" in text
    # the final result is the reply the task ended on, not the closing reply
    # and not the loop's stop notice
    assert text[final:] == f"## Final Result\n\n{SESSION_REPLY}"


@pytest.mark.asyncio
async def test_the_outcome_reaches_the_forked_review(switches):
    switches(outcome=True, discipline=True, fork=True)
    note, requests, _handle, _ = await _persistent_review(outcome=FAILED)
    assert note["message"] == REVIEW_SUMMARY
    review = requests[3]
    text = review["messages"][-1]["content"]
    assert text.startswith("## Storage Review")
    assert "## Completed Trajectory" not in text
    assert outcome_mod.OUTCOME_HEADER in text
    assert "- Solved: no" in text
    assert text.endswith(f"## Final Result\n\n{SESSION_REPLY}")
    # still a fork: the session's own requests are its prefix
    sent = json.dumps(requests[2]["messages"], default=str)
    assert (
        json.dumps(review["messages"][: len(requests[2]["messages"])], default=str)
        == sent
    )


@pytest.mark.asyncio
async def test_without_an_outcome_there_is_no_section_but_the_reply_is_final(switches):
    switches(outcome=True)
    _note, requests, _handle, _ = await _persistent_review(closing=False)
    text = _review_text(requests[2])
    assert outcome_mod.OUTCOME_HEADER not in text
    assert text.endswith(f"## Final Result\n\n{SESSION_REPLY}")


@pytest.mark.asyncio
async def test_off_the_final_result_is_the_stop_notice_and_nothing_is_posted(switches):
    _note, requests, handle, _ = await _persistent_review()
    assert handle.outcome_session_id is None
    text = _review_text(requests[3])
    assert outcome_mod.OUTCOME_HEADER not in text
    assert text.endswith(f"## Final Result\n\n{caa._STOPPED_NOTICE}")


@pytest.mark.asyncio
async def test_off_the_review_requests_are_upstreams(switches):
    """The equivalence baseline recorded on the upstream commit."""
    _summary, _, requests = await h.scenario_review()
    golden = json.loads(h.GOLDEN.read_text())["review"]
    assert [h.request_bytes(r) for r in requests] == golden


@pytest.mark.asyncio
async def test_on_without_an_outcome_a_finished_task_reviews_as_upstream(switches):
    """A task that ends by itself has a real result; with no outcome posted
    the switch leaves its review byte-identical."""
    switches(outcome=True)
    _summary, _, requests = await h.scenario_review()
    golden = json.loads(h.GOLDEN.read_text())["review"]
    assert [h.request_bytes(r) for r in requests] == golden


@pytest.mark.asyncio
async def test_an_outcome_after_the_session_ended_is_refused(switches):
    switches(outcome=True)
    _note, _requests, handle, _ = await _persistent_review()
    with pytest.raises(outcome_mod.OutcomeError, match="already ended"):
        outcome_mod.post(handle.outcome_session_id, FAILED)


# ── failure lessons ──────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_lessons_standalone_offers_no_function_writes_and_keeps_guidance(
    switches,
):
    switches(outcome=True, review_failed="lessons")
    captured: dict = {}
    real = caa.start_async_tool_loop

    def spy(*args, **kwargs):
        if kwargs.get("loop_id") == "StorageCheck(CodeActActor.act)":
            captured["tools"] = sorted(kwargs["tools"])
        return real(*args, **kwargs)

    with patch.object(caa, "start_async_tool_loop", spy):
        _note, requests, _handle, _ = await _persistent_review(outcome=FAILED)
    assert "GuidanceManager_add_guidance" in captured["tools"]
    assert "GuidanceManager_update_guidance" in captured["tools"]
    for name in outcome_mod.LESSON_REFUSED_TOOLS:
        assert name not in captured["tools"]
    names = {t["function"]["name"] for t in requests[3]["tools"]}
    assert "FunctionManager_add_functions" not in names
    assert "GuidanceManager_add_guidance" in names
    text = _review_text(requests[3])
    assert outcome_mod.LESSONS_HEADER in text


LESSON_FORK_REVIEW = (
    lambda: h.completion(
        calls=[
            (
                "FunctionManager_add_functions",
                {"implementations": ["def f():\n    return 1\n"]},
            ),
            (
                "GuidanceManager_add_guidance",
                {"title": "Check the recipient", "content": "Confirm Kim's address."},
            ),
        ],
        call_ids=["write_fn", "write_note"],
    ),
    lambda: h.completion(content="Recorded one lesson."),
)


@pytest.mark.asyncio
async def test_lessons_fork_refuses_function_writes_and_runs_guidance_writes(
    switches,
):
    switches(outcome=True, review_failed="lessons", discipline=True, fork=True)
    note, requests, _handle, _ = await _persistent_review(
        review=LESSON_FORK_REVIEW,
        outcome=FAILED,
    )
    assert note["message"] == "Recorded one lesson."
    replies = {
        m["tool_call_id"]: m["content"]
        for m in requests[4]["messages"]
        if m.get("role") == "tool"
    }
    assert outcome_mod.LESSON_MASK_RULE in replies["write_fn"]
    assert outcome_mod.LESSON_MASK_RULE not in replies["write_note"]
    assert "refused" not in replies["write_note"].lower()
    # the fixed tool list is unchanged: refused by rule, not removed
    assert (
        h.request_bytes(requests[3])["tools"] == h.request_bytes(requests[2])["tools"]
    )


@pytest.mark.asyncio
async def test_lessons_need_a_failed_outcome(switches):
    switches(outcome=True, review_failed="lessons")
    solved = {**FAILED, "solved": True}
    _note, requests, _handle, _ = await _persistent_review(outcome=solved)
    names = {t["function"]["name"] for t in requests[3]["tools"]}
    assert "FunctionManager_add_functions" in names
    assert outcome_mod.LESSONS_HEADER not in _review_text(requests[3])


# ── lessons and the admission gate (mocked review loop) ──────────────────


def _inner_handle(result_future: "asyncio.Future[str]") -> MagicMock:
    inner = MagicMock()

    async def _result():
        return await result_future

    inner.result = _result
    inner.next_notification = AsyncMock(side_effect=lambda: asyncio.Event().wait())
    inner._client = MagicMock(messages=[{"role": "user", "content": "do something"}])
    inner._task = MagicMock()
    inner._task.get_ask_tools = MagicMock(return_value={})
    inner._task.get_completed_tool_metadata = MagicMock(return_value={})
    return inner


async def _session_end(post=None) -> tuple:
    result_future: asyncio.Future[str] = asyncio.get_event_loop().create_future()
    inner = _inner_handle(result_future)
    actor = SimpleNamespace(function_manager=None, guidance_manager=None)
    with (
        patch.object(caa, "_start_storage_check_loop") as mock_loop,
        patch.object(caa, "publish_manager_method_event", new_callable=AsyncMock),
    ):
        mock_loop.return_value = None
        handle = caa._StorageCheckHandle(inner=inner, actor=actor)
        if post is not None:
            outcome_mod.post(handle.outcome_session_id, post)
        result_future.set_result("done")
        for _ in range(500):
            if handle.done():
                break
            await asyncio.sleep(0.01)
    notes = []
    while not handle._notification_q.empty():
        notes.append(handle._notification_q.get_nowait())
    return mock_loop, notes


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("verdict", "outcome", "expected"),
    [
        # admission on: only a "lessons" verdict reviews a failed run, and
        # never with function writes
        ({"admit": "lessons", "reason": "failed train task"}, FAILED, "lessons"),
        ({"admit": "lessons"}, None, "lessons"),
        ({"admit": False, "reason": "frozen library"}, FAILED, "skipped"),
        ({"admit": True}, FAILED, "lessons"),
        ({"admit": True}, {**FAILED, "solved": True}, "full"),
        # admission off: the outcome decides
        (None, FAILED, "lessons"),
        (None, {**FAILED, "solved": None}, "full"),
        (None, None, "full"),
    ],
)
async def test_lessons_obey_the_admission_gate(
    switches,
    tmp_path,
    verdict,
    outcome,
    expected,
):
    path = ""
    if verdict is not None:
        path = str(tmp_path / "verdict.json")
        with open(path, "w") as fh:
            json.dump(verdict, fh)
    switches(outcome=True, review_failed="lessons", admission=path)
    mock_loop, notes = await _session_end(post=outcome)
    if expected == "skipped":
        mock_loop.assert_not_called()
        assert [n for n in notes if n.get("type") == "storage_review_skipped"]
        return
    mock_loop.assert_called_once()
    assert mock_loop.call_args.kwargs["lessons"] is (expected == "lessons")


@pytest.mark.asyncio
async def test_off_a_lessons_verdict_does_not_admit(switches, tmp_path):
    path = tmp_path / "verdict.json"
    path.write_text(json.dumps({"admit": "lessons"}))
    switches(admission=str(path))
    mock_loop, notes = await _session_end()
    mock_loop.assert_not_called()
    skipped = [n for n in notes if n.get("type") == "storage_review_skipped"]
    assert skipped[0]["message"] == "not admitted"


# ── the transport ────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_the_outcome_never_reaches_the_disk_or_the_environment(switches):
    """The outcome lives on the session's handle only: not under UNIFY_HOME
    (state, store, workspace), not in the working directory, not in the
    environment the workspace sandbox's commands inherit."""
    from unify.db import store_home
    from unify.workspace import get_local_root

    switches(outcome=True)
    marker = "outcome-marker-7f3e9c"
    outcome = {**FAILED, "summary": marker}
    _note, requests, handle, _ = await _persistent_review(outcome=outcome)
    assert marker in _review_text(requests[3])
    assert handle._outcome["summary"] == marker
    roots = {str(store_home()), str(get_local_root())}
    assert roots
    for root in roots:
        for dirpath, _dirs, files in os.walk(root):
            for name in files:
                full = os.path.join(dirpath, name)
                try:
                    if os.path.getsize(full) > 5_000_000:
                        continue
                    with open(full, "rb") as fh:
                        assert marker.encode() not in fh.read(), full
                except OSError:
                    continue
    assert not any(marker in v for v in os.environ.values())


@pytest.mark.asyncio
async def test_the_jsonl_control_line_posts_and_is_answered(
    switches,
    monkeypatch,
    capsys,
):
    import sys

    from unify.cli import Act

    switches(outcome=True)
    handle = _Receiver("cli-session")
    received = handle.got
    outcome_mod.register("cli-session", handle)
    read_fd, write_fd = os.pipe()
    monkeypatch.setattr(sys, "stdin", os.fdopen(read_fd, "r"))
    session = Act(SimpleNamespace(persist=True, quiet=True, jsonl=True))
    session._handle = handle
    reader = asyncio.create_task(session._read_lines())
    os.write(write_fd, (json.dumps({"outcome": FAILED}) + "\n").encode())
    os.write(write_fd, (json.dumps({"outcome": {"solved": "maybe"}}) + "\n").encode())
    os.write(write_fd, b'{"quit": true}\n')
    os.close(write_fd)
    await asyncio.wait_for(reader, timeout=5)
    lines = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert lines[0] == {
        "type": "outcome",
        "accepted": True,
        "solved": False,
        "checks": 2,
    }
    assert lines[1]["accepted"] is False and "solved" in lines[1]["reason"]
    assert received[0]["checks"][0]["reason"] == "no email to Kim"
    assert handle.stopped == caa.SESSION_ENDED


@pytest.mark.asyncio
async def test_off_the_jsonl_control_line_is_ignored(switches, monkeypatch, capsys):
    import sys

    from unify.cli import Act

    handle = _Receiver()
    read_fd, write_fd = os.pipe()
    monkeypatch.setattr(sys, "stdin", os.fdopen(read_fd, "r"))
    session = Act(SimpleNamespace(persist=True, quiet=True, jsonl=True))
    session._handle = handle
    reader = asyncio.create_task(session._read_lines())
    os.write(write_fd, (json.dumps({"outcome": FAILED}) + "\n").encode())
    os.write(write_fd, b'{"quit": true}\n')
    os.close(write_fd)
    await asyncio.wait_for(reader, timeout=5)
    assert capsys.readouterr().out == ""
    assert handle.interjections == []
    assert handle.stopped == caa.SESSION_ENDED
