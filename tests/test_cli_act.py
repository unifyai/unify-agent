"""The ``unify act`` entry point: one actor, no conversation loop."""

from __future__ import annotations

import asyncio
import io
import os
import pty
import sys
from types import SimpleNamespace

import pytest

from tests.scripted_model import ScriptedModel, reply, scripted
from unify.actor.code_act_actor import SESSION_ENDED
from unify.cli import Act, _parse_args

pytestmark = pytest.mark.no_unify_context


def test_no_command_is_act_reading_stdin():
    args = _parse_args([])
    assert (args.command, args.request) == ("act", None)
    args = _parse_args(["--debug"])
    assert (args.command, args.debug) == ("act", True)


@pytest.mark.parametrize(
    "argv",
    [
        ["--home", "/elsewhere", "--debug", "act", "hi"],
        ["act", "--home", "/elsewhere", "--debug", "hi"],
    ],
)
def test_common_options_parse_before_or_after_the_subcommand(argv):
    args = _parse_args(argv)
    assert (args.home, args.debug) == ("/elsewhere", True)


@pytest.mark.parametrize("argv", [["act"], ["act", "hi"]])
def test_common_options_default_under_a_subcommand(argv):
    args = _parse_args(argv)
    assert (args.home, args.debug) == (None, False)


def test_act_flags_parse():
    args = _parse_args(
        [
            "act",
            "count the rows",
            "--persist",
            "--timeout",
            "12",
            "--json",
        ],
    )
    assert args.command == "act"
    assert args.request == "count the rows"
    assert args.persist and args.json
    assert args.timeout == 12.0
    assert args.no_clarify is False
    assert not hasattr(args, "no_compose")


def test_no_clarify_parses_and_turns_clarification_off(monkeypatch):
    """``--no-clarify`` (passed by unattended launchers): even at a terminal,
    nobody reads the record's @user posts, so the actor is started with
    clarification off."""
    args = _parse_args(["act", "--no-clarify", "--quiet", "hi"])
    assert args.no_clarify is True

    seen: dict = {}

    class _Actor:
        async def act(self, request, **kwargs):
            seen.update(kwargs)
            raise RuntimeError("stop here")

    class _Tty(io.StringIO):
        def isatty(self) -> bool:
            return True

    monkeypatch.setattr(sys, "stdin", _Tty())
    session = Act(args)

    async def _start() -> None:
        session._actor = _Actor()

    monkeypatch.setattr(session, "start", _start)
    with pytest.raises(RuntimeError, match="stop here"):
        asyncio.run(session.run("hi"))  # Act.run starts the actor itself (it was _run)
    assert seen["clarification_enabled"] is False


class _FakeHandle:
    def __init__(self) -> None:
        self.interjections: list[str] = []
        self.interjected = asyncio.Event()
        self.stopped: str | None = None

    async def submit(self, message: str) -> None:
        self.interjections.append(message)
        self.interjected.set()

    async def stop(self, reason: str | None = None) -> None:
        self.stopped = reason


@pytest.mark.asyncio
async def test_typed_lines_are_the_next_messages_and_quit_stops_the_session(monkeypatch):
    """Each typed line is the next message to the running actor, and /quit
    stops it."""
    read_fd, write_fd = os.pipe()
    monkeypatch.setattr(sys, "stdin", os.fdopen(read_fd, "r"))
    session = Act(SimpleNamespace(persist=True, quiet=True, jsonl=False))
    handle = _FakeHandle()
    session._handle = handle

    reader = asyncio.create_task(session._read_lines())
    os.write(write_fd, b"the second one\n")
    os.write(write_fd, b"also skip the header row\n")
    os.write(write_fd, b"/quit\n")
    os.close(write_fd)
    await asyncio.wait_for(reader, timeout=5)

    assert handle.interjections == ["the second one", "also skip the header row"]
    assert handle.stopped == SESSION_ENDED


@pytest.mark.asyncio
async def test_reading_a_terminal_leaves_it_blocking(monkeypatch):
    """A terminal's stdin, stdout and stderr are one open file. Were reading
    stdin to make it non-blocking, any write the terminal could not take at
    once would fail with BlockingIOError instead of waiting for it."""
    controller, terminal = pty.openpty()
    monkeypatch.setattr(sys, "stdin", os.fdopen(terminal, "r"))
    session = Act(SimpleNamespace(persist=True, quiet=True, jsonl=False))
    handle = _FakeHandle()
    session._handle = handle

    reader = asyncio.create_task(session._read_lines())
    os.write(controller, b"a follow-up\n")
    await asyncio.wait_for(handle.interjected.wait(), timeout=5)
    assert os.get_blocking(terminal)

    os.write(controller, b"/quit\n")
    await asyncio.wait_for(reader, timeout=5)
    os.close(controller)


@pytest.mark.asyncio
async def test_act_runs_one_request_and_prints_the_result(capsys, monkeypatch):
    """The direct mode answers a request without any conversation loop: one
    actor turn (the scripted model's), its reply printed, exit code 0."""
    # Whether stdin is a terminal decides whether the actor may ask
    # questions, and so the tools the model is offered: pin it rather than
    # inherit the runner's.
    monkeypatch.setattr(sys, "stdin", io.StringIO())
    args = _parse_args(
        ["act", "--quiet", "Reply with exactly the single word: pong"],
    )
    model = ScriptedModel(actor=[reply("pong")])
    with scripted(model):
        session = Act(args)
        try:
            code = await asyncio.wait_for(session.run(args.request), timeout=120)
        finally:
            await session.close()
    assert code == 0
    assert capsys.readouterr().out.strip() == "pong"
    assert model.kinds() == ["actor"]
    (turn,) = model.of("actor")
    assert any(
        m.get("role") == "user" and "the single word: pong" in str(m.get("content"))
        for m in turn.messages
    )
    model.assert_used_up()
