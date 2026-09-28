"""The terminal front ends: chat with the assistant, or drive the actor alone.

``unify`` (or ``python -m unify``) starts the slow brain in-process, wires the
terminal to the in-app chat, and renders what the assistant sends back.
Every line typed is an inbound ``UnifyMessageReceived`` event; every reply is
the ``UnifyMessageSent`` event the brain publishes, so the terminal is one
front end over the same loop any other client would drive.

``unify act "request"`` bypasses the conversation loop: one ``CodeActActor``
takes the request, its progress streams to the terminal, any question it asks
is answered from the terminal, and the result is printed. This is the unit to
compare against single-loop harnesses, and the shape a benchmark runner wants.

Runtime logs go to ``<UNIFY_HOME>/logs`` and stay off the terminal unless
``--debug`` is given.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import asyncio
import os
import shutil
import sys
import uuid
from collections.abc import Iterator
from pathlib import Path

from dotenv import load_dotenv

MAX_ATTACHMENT_BYTES = 25 * 1024 * 1024
BOOT_TIMEOUT_SECONDS = 300.0

HELP = """\
Type a message and press Enter. The assistant keeps working on anything you
asked for while you keep typing; follow-up messages steer it.

  /attach <path>   attach a file to your next message
  /attach          list queued attachments
  /detach          clear queued attachments
  /help            show this help
  /quit            exit (Ctrl-D and Ctrl-C work too)
"""


ACT_HELP = """\
Drive one actor directly, without the conversation loop.

The request is taken from the command line, or from stdin when omitted or
given as "-". Progress lines stream to stderr while the actor works; the
result goes to stdout. When the actor asks a question, type the answer and
press Enter. With --persist the actor stays alive after answering: each
further line is a follow-up in the same sandbox, /quit ends the session.
With --jsonl the session speaks newline-delimited JSON instead, for a
program driving the actor: each stdin line is {"message": "..."} (a
follow-up, which may span lines) or {"quit": true}; each stdout line is
{"type": "result" | "response" | "question" | "storage" | "ended", ...}.
With --persist every turn ends in one "response" line as the actor starts
waiting, its content empty when the turn produced no text. Progress still
goes to stderr.
"""


def _add_common_options(
    parser: argparse.ArgumentParser,
    *,
    subcommand: bool = False,
) -> None:
    """Add the options accepted both before and after the subcommand.

    argparse copies everything a subcommand's parser holds over what the
    top-level parser parsed, defaults included, so a subcommand's copies
    default to SUPPRESS: they set nothing unless given after the subcommand.
    """
    parser.add_argument(
        "--home",
        metavar="DIR",
        default=argparse.SUPPRESS if subcommand else None,
        help="where the store, workspace and logs live "
        "(default: UNIFY_HOME or ~/.unify)",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        default=argparse.SUPPRESS if subcommand else False,
        help="stream runtime logs to the terminal as well as the log files",
    )


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="unify",
        description="Chat with the local assistant, or drive its actor directly.",
    )
    _add_common_options(parser)
    commands = parser.add_subparsers(dest="command")

    chat = commands.add_parser("chat", help="chat with the assistant (the default)")
    _add_common_options(chat, subcommand=True)

    act = commands.add_parser(
        "act",
        help="run one request through the actor, bypassing the conversation loop",
        description=ACT_HELP,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    _add_common_options(act, subcommand=True)
    act.add_argument(
        "request",
        nargs="?",
        help='the request; "-" or omitted reads stdin',
    )
    act.add_argument(
        "--persist",
        action="store_true",
        help="keep the actor alive after the result; further lines are follow-ups",
    )
    act.add_argument(
        "--no-store",
        action="store_true",
        help="skip the storage review that distils the run into functions and guidance",
    )
    act.add_argument(
        "--no-compose",
        action="store_true",
        help="forbid execute_code: the actor may only call stored functions",
    )
    act.add_argument(
        "--no-clarify",
        action="store_true",
        help="disable request_clarification; the actor must decide on its own",
    )
    act.add_argument(
        "--timeout",
        type=float,
        metavar="SECONDS",
        help="give up on the request after this long",
    )
    act.add_argument(
        "--json",
        action="store_true",
        help="print the result as a JSON object with the run statistics",
    )
    act.add_argument(
        "--quiet",
        action="store_true",
        help="do not stream progress lines",
    )
    act.add_argument(
        "--jsonl",
        action="store_true",
        help="newline-delimited JSON on stdin and stdout, for a program driving "
        "the actor (see the description)",
    )

    args = parser.parse_args(argv)
    if args.command is None:
        args.command = "chat"
    return args


def _configure_environment(args: argparse.Namespace) -> Path:
    """Point the runtime at its home directory and route logs there."""
    load_dotenv()
    if args.home:
        os.environ["UNIFY_HOME"] = str(Path(args.home).expanduser())
    home = Path(os.environ.get("UNIFY_HOME", "").strip() or "~/.unify").expanduser()
    home.mkdir(parents=True, exist_ok=True)

    from unify.logger import LOGGER, configure_log_dir

    configure_log_dir(os.environ.get("UNIFY_LOG_DIR", "").strip() or str(home / "logs"))
    if not args.debug:
        for handler in list(LOGGER.handlers):
            if getattr(handler, "_unify_terminal", False):
                LOGGER.removeHandler(handler)
    return home


def _stage_attachment(source: Path) -> str:
    """Copy a local file into the workspace and return its workspace path."""
    from unify.workspace import get_local_root

    attachments_dir = Path(get_local_root()) / "Attachments"
    attachments_dir.mkdir(parents=True, exist_ok=True)
    target_name = f"{uuid.uuid4().hex[:8]}_{source.name}"
    shutil.copy2(source, attachments_dir / target_name)
    return f"Attachments/{target_name}"


def _assistant_name() -> str:
    from unify.session_details import PLACEHOLDER_ASSISTANT_FIRST_NAME, SESSION_DETAILS

    return SESSION_DETAILS.assistant.first_name or PLACEHOLDER_ASSISTANT_FIRST_NAME


class Chat:
    """One terminal session over a running ConversationManager."""

    def __init__(self) -> None:
        self._cm = None
        self._ready = asyncio.Event()
        self._closing = asyncio.Event()
        self._pending_attachments: list[Path] = []

    # ── lifecycle ────────────────────────────────────────────────────────

    async def start(self) -> None:
        from unify import db
        from unify.conversation_manager.main import run_conversation_manager
        from unify.session_details import SESSION_DETAILS

        SESSION_DETAILS.populate_from_env()
        db.connect()

        self._cm = await run_conversation_manager()
        self._listener = asyncio.create_task(self._listen())

    async def close(self) -> None:
        self._closing.set()
        if self._cm is not None:
            self._cm.stop.set()
            try:
                await asyncio.wait_for(self._cm.cleanup(), timeout=15.0)
            except asyncio.TimeoutError:
                pass
        self._listener.cancel()

    # ── outbound ─────────────────────────────────────────────────────────

    async def _listen(self) -> None:
        from unify.conversation_manager.events import (
            ActorClarificationRequest,
            ActorNotification,
            ActorResult,
            DirectMessageEvent,
            Error,
            Event,
            InitializationComplete,
            UnifyMessageSent,
        )

        async with self._cm.event_broker.pubsub() as pubsub:
            await pubsub.psubscribe("app:comms:*", "app:actor:*")
            while not self._closing.is_set():
                msg = await pubsub.get_message(
                    timeout=1.0,
                    ignore_subscribe_messages=True,
                )
                if not msg:
                    continue
                event = Event.from_json(msg["data"])
                if isinstance(event, InitializationComplete):
                    self._ready.set()
                elif isinstance(event, (UnifyMessageSent, DirectMessageEvent)):
                    self._say(_assistant_name(), event.content)
                    for attachment in getattr(event, "attachments", []):
                        self._status(f"attached {attachment}")
                elif isinstance(event, ActorNotification):
                    prefix = "done" if event.completed else "working"
                    self._status(f"{prefix}: {event.response}")
                elif isinstance(event, ActorResult):
                    if not event.success:
                        self._status(f"action failed: {event.error}")
                elif isinstance(event, ActorClarificationRequest):
                    self._status(f"the assistant is asking: {event.query}")
                elif isinstance(event, Error):
                    self._status(f"error: {event.message}")

    def _say(self, who: str, text: str) -> None:
        print(f"\n{who}> {text}\n", flush=True)

    def _status(self, text: str) -> None:
        print(f"  · {text}", flush=True)

    # ── inbound ──────────────────────────────────────────────────────────

    async def send(self, text: str) -> None:
        from unify.conversation_manager.events import UnifyMessageReceived

        attachments = [_stage_attachment(p) for p in self._pending_attachments]
        self._pending_attachments.clear()
        event = UnifyMessageReceived(content=text, attachments=attachments)
        await self._cm.event_broker.publish(UnifyMessageReceived.topic, event.to_json())

    def attach(self, raw_path: str) -> str:
        path = Path(raw_path).expanduser()
        if not path.is_file():
            return f"no such file: {raw_path}"
        if path.stat().st_size > MAX_ATTACHMENT_BYTES:
            return f"too large to attach (limit 25MB): {raw_path}"
        self._pending_attachments.append(path)
        return f"queued {path.name} for your next message"

    def queued_attachments(self) -> str:
        if not self._pending_attachments:
            return "no attachments queued"
        return "queued: " + ", ".join(p.name for p in self._pending_attachments)

    def detach(self) -> str:
        count = len(self._pending_attachments)
        self._pending_attachments.clear()
        return f"cleared {count} queued attachment(s)"

    # ── loop ─────────────────────────────────────────────────────────────

    async def run(self) -> None:
        print("starting the assistant ...", flush=True)
        await self.start()
        try:
            await asyncio.wait_for(self._ready.wait(), timeout=BOOT_TIMEOUT_SECONDS)
        except asyncio.TimeoutError:
            print("the assistant did not finish starting; check the logs", flush=True)
            return
        print("ready. /help for commands, /quit to exit.\n", flush=True)
        while True:
            try:
                raw = await asyncio.to_thread(input, "> ")
            except (EOFError, KeyboardInterrupt):
                print()
                return
            line = raw.strip()
            if not line:
                continue
            if line.startswith("/"):
                if await self._command(line):
                    return
                continue
            await self.send(line)

    async def _command(self, line: str) -> bool:
        """Run a slash command; return True when the session should end."""
        name, _, arg = line[1:].partition(" ")
        name = name.lower()
        arg = arg.strip()
        if name in {"quit", "exit", "q"}:
            return True
        if name in {"help", "h", "?"}:
            print(HELP)
        elif name == "attach":
            print(self.attach(arg) if arg else self.queued_attachments())
        elif name == "detach":
            print(self.detach())
        else:
            print(f"unknown command: /{name} (try /help)")
        return False


@contextlib.contextmanager
def _stdin_reader() -> Iterator[asyncio.StreamReader]:
    """Stream stdin into a reader on the running loop, leaving stdin blocking.

    A pipe transport would switch stdin to non-blocking, and a terminal's
    stdin, stdout and stderr are one open file: every write the terminal
    could not take at once would then fail with BlockingIOError. Reading
    only once the descriptor is readable never blocks the loop.
    """
    loop = asyncio.get_running_loop()
    fd = sys.stdin.fileno()
    reader = asyncio.StreamReader()

    def feed() -> None:
        data = os.read(fd, 65536)
        if data:
            reader.feed_data(data)
        else:
            loop.remove_reader(fd)
            reader.feed_eof()

    loop.add_reader(fd, feed)
    try:
        yield reader
    finally:
        loop.remove_reader(fd)


class Act:
    """One actor driven from the terminal, with no conversation loop above it."""

    def __init__(self, args: argparse.Namespace) -> None:
        self._args = args
        self._actor = None
        self._handle = None
        self._pending_clarifications: asyncio.Queue[dict] = asyncio.Queue()
        self._closing = asyncio.Event()

    # ── lifecycle ────────────────────────────────────────────────────────

    async def start(self) -> None:
        import unify
        from unify.actor.environments import ActorEnvironment
        from unify.manager_registry import ManagerRegistry
        from unify.session_details import SESSION_DETAILS
        from unify.workspace import get_local_root

        SESSION_DETAILS.populate_from_env()
        unify.init()
        # Relative paths in the actor's code resolve against the workspace,
        # exactly as they do under the conversation loop.
        local_root = Path(get_local_root())
        local_root.mkdir(parents=True, exist_ok=True)
        os.chdir(local_root)
        self._actor = ManagerRegistry.get_actor(
            description="direct actor session",
            environments=[ActorEnvironment()],
        )

    async def close(self) -> None:
        self._closing.set()
        if self._handle is not None and not self._handle.done():
            try:
                await self._handle.stop("session closed")
            except Exception:
                pass
        if self._actor is not None:
            try:
                await self._actor.close()
            except Exception:
                pass

    # ── output ───────────────────────────────────────────────────────────

    def _progress(self, text: str) -> None:
        if not self._args.quiet:
            print(f"  · {text}", file=sys.stderr, flush=True)

    def _emit(self, **payload: object) -> None:
        """One JSON line on stdout (``--jsonl``)."""
        print(json.dumps(payload, default=str), flush=True)

    async def _watch_notifications(self) -> None:
        while not self._closing.is_set():
            notif = await self._handle.next_notification()
            if not isinstance(notif, dict):
                self._progress(str(notif))
                continue
            kind = notif.get("type", "")
            if kind == "response":
                # A persist-mode turn finished; its answer is the result of the
                # follow-up the user typed.
                if self._args.jsonl:
                    self._emit(type="response", content=notif.get("content", ""))
                else:
                    print(f"\n{notif.get('content', '')}\n", flush=True)
            elif kind in ("storage_review_complete", "turn_storage_review_complete"):
                verdict = "stored" if notif.get("success") else "storage review failed"
                if self._args.jsonl:
                    self._emit(
                        type="storage",
                        success=bool(notif.get("success")),
                        message=notif.get("message", ""),
                    )
                else:
                    self._progress(f"{verdict}: {notif.get('message', '')}")
            else:
                text = notif.get("message") or notif.get("result_summary") or kind
                self._progress(str(text))

    async def _watch_clarifications(self) -> None:
        while not self._closing.is_set():
            clar = await self._handle.next_clarification()
            if not clar:
                continue
            if self._args.jsonl:
                self._emit(
                    type="question",
                    call_id=str(clar.get("call_id") or ""),
                    question=clar.get("question", ""),
                )
            else:
                print(f"\nactor asks> {clar.get('question', '')}", flush=True)
            await self._pending_clarifications.put(clar)

    # ── input ────────────────────────────────────────────────────────────

    async def _read_lines(self) -> None:
        """Route typed lines: answer a pending question, else steer the actor."""
        with _stdin_reader() as reader:
            while not self._closing.is_set():
                raw = await reader.readline()
                if not raw:
                    if self._args.persist:
                        from unify.actor.code_act_actor import SESSION_ENDED

                        await self._handle.stop(SESSION_ENDED)
                    return
                line = raw.decode(errors="replace").strip()
                if not line:
                    continue
                if self._args.jsonl:
                    try:
                        item = json.loads(line)
                    except ValueError:
                        self._progress(
                            f"ignored a stdin line that is not JSON: {line[:80]!r}",
                        )
                        continue
                    if not isinstance(item, dict):
                        continue
                    if item.get("quit"):
                        line = "/quit"
                    else:
                        message = item.get("message")
                        if not isinstance(message, str) or not message:
                            continue
                        line = message
                if line in {"/quit", "/exit", "/q"}:
                    from unify.actor.code_act_actor import SESSION_ENDED

                    self._progress("session ended; reviewing the work for storage")
                    await self._handle.stop(SESSION_ENDED)
                    return
                if not self._pending_clarifications.empty():
                    clar = await self._pending_clarifications.get()
                    await self._handle.answer_clarification(
                        str(clar.get("call_id") or ""),
                        line,
                    )
                    continue
                await self._handle.interject(line)

    # ── run ──────────────────────────────────────────────────────────────

    async def run(self, request: str) -> int:
        await self.start()
        args = self._args
        interactive = sys.stdin.isatty()
        clarify = not args.no_clarify and interactive
        if not args.no_clarify and not interactive:
            self._progress("stdin is not a terminal; the actor cannot ask questions")
        self._handle = await self._actor.act(
            request,
            persist=args.persist,
            can_compose=not args.no_compose,
            can_store=not args.no_store,
            clarification_enabled=clarify,
        )
        watchers = [
            asyncio.create_task(self._watch_notifications()),
            asyncio.create_task(self._watch_clarifications()),
        ]
        reader = None
        if (interactive and (clarify or args.persist)) or args.jsonl:
            reader = asyncio.create_task(self._read_lines())
        try:
            result = await asyncio.wait_for(self._handle.result(), timeout=args.timeout)
        except asyncio.TimeoutError:
            print(f"timed out after {args.timeout:g}s", file=sys.stderr, flush=True)
            return 1
        except Exception as exc:
            print(f"error: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
            return 1
        finally:
            if reader is not None and not args.persist:
                reader.cancel()

        self._print_result(result)

        if args.persist:
            self._progress("actor is waiting; type a follow-up, /quit to end")
            if reader is not None:
                await reader
        while not self._handle.done():
            await asyncio.sleep(0.2)
        for task in watchers:
            task.cancel()
        if args.jsonl:
            self._emit(type="ended")
        return 0

    def _print_result(self, result: object) -> None:
        if self._args.jsonl:
            self._emit(
                type="result",
                content=result if isinstance(result, str) else str(result),
                run_stats=getattr(self._handle, "run_stats", {}) or {},
            )
            return
        if self._args.json:
            payload = {
                "result": (
                    result if isinstance(result, (str, dict, list)) else str(result)
                ),
                # The storage-aware handle meters tokens; a bare loop handle does not.
                "run_stats": getattr(self._handle, "run_stats", {}) or {},
            }
            print(json.dumps(payload, indent=2, default=str), flush=True)
        else:
            print(result, flush=True)


def _read_request(args: argparse.Namespace) -> str:
    if args.request and args.request != "-":
        return args.request
    text = sys.stdin.read().strip()
    if not text:
        raise SystemExit(
            "unify act: no request given (pass it as an argument or on stdin)",
        )
    return text


async def _run_chat(args: argparse.Namespace) -> int:
    chat = Chat()
    try:
        await chat.run()
    finally:
        await chat.close()
    return 0


async def _run_act(args: argparse.Namespace) -> int:
    session = Act(args)
    try:
        return await session.run(_read_request(args))
    finally:
        await session.close()


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    _configure_environment(args)
    runner = _run_act if args.command == "act" else _run_chat
    try:
        return asyncio.run(runner(args))
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    sys.exit(main())
