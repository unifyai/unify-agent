"""The ``unify act`` entry point: one actor, no conversation loop."""

from __future__ import annotations

import asyncio
import os
import sys
from types import SimpleNamespace

import pytest

from unify.actor.code_act_actor import SESSION_ENDED
from unify.cli import Act, _parse_args

pytestmark = pytest.mark.no_unify_context


def test_no_command_is_chat():
    assert _parse_args([]).command == "chat"
    assert _parse_args(["--debug"]).debug is True


@pytest.mark.parametrize(
    "argv",
    [
        ["--home", "/elsewhere", "--debug", "act", "hi"],
        ["act", "--home", "/elsewhere", "--debug", "hi"],
        ["--home", "/elsewhere", "--debug", "chat"],
        ["chat", "--home", "/elsewhere", "--debug"],
    ],
)
def test_common_options_parse_before_or_after_the_subcommand(argv):
    args = _parse_args(argv)
    assert (args.home, args.debug) == ("/elsewhere", True)


@pytest.mark.parametrize("argv", [["chat"], ["act", "hi"]])
def test_common_options_default_under_a_subcommand(argv):
    args = _parse_args(argv)
    assert (args.home, args.debug) == (None, False)


def test_act_flags_parse():
    args = _parse_args(
        [
            "act",
            "count the rows",
            "--persist",
            "--no-store",
            "--timeout",
            "12",
            "--json",
        ],
    )
    assert args.command == "act"
    assert args.request == "count the rows"
    assert args.persist and args.no_store and args.json
    assert args.timeout == 12.0
    assert args.no_compose is False and args.no_clarify is False


class _FakeHandle:
    def __init__(self) -> None:
        self.answers: list[tuple[str, str]] = []
        self.interjections: list[str] = []
        self.stopped: str | None = None

    async def answer_clarification(self, call_id: str, answer: str) -> None:
        self.answers.append((call_id, answer))

    async def interject(self, message: str) -> None:
        self.interjections.append(message)

    async def stop(self, reason: str | None = None) -> None:
        self.stopped = reason


@pytest.mark.asyncio
async def test_typed_lines_answer_pending_questions_else_steer(monkeypatch):
    """A line answers the pending question when there is one, otherwise it
    is an interjection into the running actor, and /quit stops it."""
    read_fd, write_fd = os.pipe()
    monkeypatch.setattr(sys, "stdin", os.fdopen(read_fd, "r"))
    session = Act(SimpleNamespace(persist=True, quiet=True, jsonl=False))
    handle = _FakeHandle()
    session._handle = handle
    await session._pending_clarifications.put({"call_id": "c1", "question": "which?"})

    reader = asyncio.create_task(session._read_lines())
    os.write(write_fd, b"the second one\n")
    os.write(write_fd, b"also skip the header row\n")
    os.write(write_fd, b"/quit\n")
    os.close(write_fd)
    await asyncio.wait_for(reader, timeout=5)

    assert handle.answers == [("c1", "the second one")]
    assert handle.interjections == ["also skip the header row"]
    assert handle.stopped == SESSION_ENDED


@pytest.mark.llm_call
@pytest.mark.asyncio
async def test_act_runs_one_request_and_prints_the_result(capsys):
    """The direct mode answers a request without any conversation loop."""
    args = _parse_args(
        ["act", "--no-store", "--quiet", "Reply with exactly the single word: pong"],
    )
    session = Act(args)
    try:
        code = await asyncio.wait_for(session.run(args.request), timeout=120)
    finally:
        await session.close()
    assert code == 0
    assert capsys.readouterr().out.strip().lower().rstrip(".") == "pong"
