"""The terminal front end: drive the actor.

``unify act "request"`` (and ``unify`` or ``python -m unify`` with no
subcommand, which runs ``act`` on a request read from stdin): one
``CodeActActor`` takes the request, its progress streams to the terminal, a
post it addresses to ``@user`` is shown and answered from the terminal, and
the result is printed.

Runtime logs go to ``<UNIFY_HOME>/logs`` and stay off the terminal unless
``--debug`` is given.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import asyncio
import os
import select
import sys
import threading
from collections.abc import Iterator
from pathlib import Path

from dotenv import load_dotenv


ACT_HELP = """\
Drive one actor directly. `unify` with no subcommand runs this command.

The request is taken from the command line, or from stdin when omitted or
given as "-". Progress lines stream to stderr while the actor works; the
result goes to stdout. A post the actor addresses to @user is shown; type
the reply and press Enter (with --no-clarify, or when stdin is not a
terminal, nobody reads @user posts). With --persist the actor stays alive after answering: each
further line is a follow-up in the same sandbox, /quit ends the session.
With --jsonl the session speaks newline-delimited JSON instead, for a
program driving the actor: each stdin line is {"message": "..."} (a
follow-up, which may span lines), {"cancel": true} or {"quit": true}
(with memory on, "consolidate": false on any line, e.g. with the
stream's last request or with quit, makes the session record its episode but
start no due consolidation pass whenever it ends, also at a step or time
limit: a driver ending its stream uses it, since no pass may be tied to the
stream's end); each
stdout line is {"type": "result" | "response" | "record" | "ended", ...}.
With --persist every turn ends in one "response" line as the
actor starts waiting, its content empty when the turn produced no text.
{"cancel": true} ends the running turn at once and keeps the session: the
model call and the tool calls still running are cancelled, and the turn's
"response" line says "cancelled": true, its content the text the turn had
drafted; with no turn running it is ignored. Without --persist it ends the
session as {"quit": true} does. Progress still goes to stderr. With memory
on, a stdin line {"outcome": {...}} gives the request its checked outcome
(see unify/outcome.py) and is answered with {"type": "outcome",
"accepted": ...}. A message is posted to the run's shared agent
record and reaches the actor at its next step instead of interrupting it,
and each record entry that mentions @user is written out as
{"type": "record", ...} (a "record>" line without --jsonl).
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
        help="where the workspace, transcripts, memory and logs live "
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
        description="Drive the actor (`unify act`, the default).",
    )
    _add_common_options(parser)
    commands = parser.add_subparsers(dest="command")

    act = commands.add_parser(
        "act",
        help="run a request through the actor (the default)",
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
        "--no-clarify",
        action="store_true",
        help="no one answers questions: the record's @user posts get no reply "
        "(for unattended runs)",
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
        help="print the result as a JSON object",
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

    argv = list(sys.argv[1:] if argv is None else argv)
    args = parser.parse_args(argv)
    if args.command is None:
        # No subcommand runs `act`, its request read from stdin.
        args = parser.parse_args([*argv, "act"])
    return args


def _configure_environment(args: argparse.Namespace) -> Path:
    """Point the runtime at its home directory and route logs there."""
    load_dotenv()
    # Settings were read before .env: a Sol route value only there is refused
    # (it would leave Sol on the actor's route) and its token leaves the
    # environment.
    from unify.memory.integration.switch import settle_writer_route_env
    from unify.settings import SETTINGS

    refusal = settle_writer_route_env(os.environ, SETTINGS)
    if refusal is not None:
        print(f"memory v2: {refusal}", file=sys.stderr)
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


@contextlib.contextmanager
def _channel_pump(
    loop: asyncio.AbstractEventLoop,
    fd: int,
    reader: asyncio.StreamReader,
) -> Iterator[None]:
    """Read *fd* on a thread into *reader* while the block runs.

    The thread owns no descriptor: it stops (within 0.1 s) before the
    caller closes *fd*, so it never reads a number the process reused.
    """
    stop = threading.Event()

    def deliver(callback, *args) -> None:
        try:
            loop.call_soon_threadsafe(callback, *args)
        except RuntimeError:  # the loop is closed
            stop.set()

    def pump() -> None:
        while not stop.is_set():
            try:
                ready, _, _ = select.select([fd], [], [], 0.1)
                if not ready:
                    continue
                data = os.read(fd, 65536)
            except OSError:
                data = b""
            if not data:
                deliver(reader.feed_eof)
                return
            deliver(reader.feed_data, data)

    thread = threading.Thread(target=pump, name="act-stdin", daemon=True)
    thread.start()
    try:
        yield
    finally:
        stop.set()
        thread.join(5)


@contextlib.contextmanager
def _stdin_reader() -> Iterator[asyncio.StreamReader]:
    """Stream stdin into a reader on the running loop, leaving stdin blocking.

    A pipe transport would switch stdin to non-blocking, and a terminal's
    stdin, stdout and stderr are one open file: every write the terminal
    could not take at once would then fail with BlockingIOError. Reading
    only once the descriptor is readable never blocks the loop.

    When stdin is not a terminal it is the driver's message channel, and
    nothing else in the process may read it. The channel is read from a
    private copy while descriptor 0 becomes ``/dev/null``, so a subprocess
    or thread that model code starts reads end of input, as in the
    sandboxed worker, instead of waiting on, or taking, the driver's lines.

    That channel is read on its own thread, which hands the bytes to the
    loop in order.
    """
    loop = asyncio.get_running_loop()
    fd = sys.stdin.fileno()
    moved_from = None
    if not os.isatty(fd):
        channel = os.dup(fd)
        null = os.open(os.devnull, os.O_RDONLY)
        os.dup2(null, fd)
        os.close(null)
        moved_from, fd = fd, channel
    reader = asyncio.StreamReader()
    if moved_from is not None:
        try:
            with _channel_pump(loop, fd, reader):
                yield reader
        finally:
            os.dup2(fd, moved_from)
            os.close(fd)
        return

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


def _checker_lines_on() -> bool:
    """Whether the CLI takes ``{"checker": ...}`` lines: whenever the run has memory. Without memory such a line is
    no control line at all."""
    from unify.memory.integration.switch import memory_on
    from unify.settings import SETTINGS

    return memory_on(SETTINGS)


class Act:
    """One actor driven from the terminal, with no conversation loop above it."""

    def __init__(self, args: argparse.Namespace) -> None:
        self._args = args
        self._actor = None
        self._handle = None
        # Routes lines into the shared agent record.
        self._bridge = None
        self._closing = asyncio.Event()
        # Set once the session is asked to end (/quit, {"quit": true}, end
        # of input); a persistent session whose result arrives without it
        # ended on its own.
        self._stop_requested = False
        # UNIFY_MEMORY: this request's memory run (None while the switch is off).
        self._mv2 = None
        # "consolidate": false on any stdin line: when the session ends, record the episode, start no pass.
        self._mv2_consolidate = True
        # Why the request stopped unfinished, for its memory run's journal: "timeout", or "error" with the
        # exception's class name (never its text); None when nothing stopped it.
        self._mv2_stop: dict | None = None

    # ── lifecycle ────────────────────────────────────────────────────────

    async def start(self) -> None:
        import unify
        from unify.actor.code_act_actor import CodeActActor
        from unify.actor.environments import registered_environments
        from unify.workspace import get_local_root

        unify.init()
        # Relative paths in the actor's code resolve against the workspace.
        local_root = Path(get_local_root())
        local_root.mkdir(parents=True, exist_ok=True)
        os.chdir(local_root)
        self._actor = CodeActActor(environments=registered_environments())

    async def close(self) -> None:
        self._closing.set()
        if self._handle is not None and not self._handle.done():
            try:
                await self._handle.stop("session closed")
            except Exception:
                pass
        if self._mv2 is not None:
            # A run not finished (a timeout, an error) records nothing now; its journal keeps the turns, and the
            # next start records them as an incomplete episode with why it stopped.
            self._mv2.abort(**({"stopped": self._mv2_stop} if self._mv2_stop else {}))

    # ── output ───────────────────────────────────────────────────────────

    def _progress(self, text: str) -> None:
        if not self._args.quiet:
            print(f"  · {text}", file=sys.stderr, flush=True)

    def _emit(self, **payload: object) -> None:
        """One JSON line on stdout (``--jsonl``)."""
        print(json.dumps(payload, default=str), flush=True)

    def _emit_record(self, **entry: object) -> None:
        """A record entry that mentions @user."""
        if self._args.jsonl:
            self._emit(**entry)
        else:
            print(
                f"\nrecord> #{entry.get('seq')} {entry.get('author')}: "
                f"{entry.get('text')}",
                flush=True,
            )

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
                    extra = {"cancelled": True} if notif.get("cancelled") else {}
                    self._emit(
                        type="response",
                        content=notif.get("content", ""),
                        **extra,
                    )
                else:
                    print(f"\n{notif.get('content', '')}\n", flush=True)
                if self._bridge is not None:
                    await self._bridge.root_replied(str(notif.get("content", "")))
            else:
                text = notif.get("message") or notif.get("result_summary") or kind
                self._progress(str(text))

    # ── input ────────────────────────────────────────────────────────────

    async def _read_lines(self) -> None:
        """Route typed lines: each is the next message (read at the session's
        next turn boundary)."""
        with _stdin_reader() as reader:
            while not self._closing.is_set():
                raw = await reader.readline()
                if not raw:
                    if self._args.persist:
                        from unify.actor.code_act_actor import SESSION_ENDED

                        self._stop_requested = True
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
                    if item.get("consolidate") is False:
                        # UNIFY_MEMORY: on any line (with the stream's last request, or with
                        # quit): whenever this session ends, record the episode, start no pass.
                        self._mv2_consolidate = False
                    if "outcome" in item:
                        self._post_outcome(item.get("outcome"))
                        continue
                    if set(item) == {"checker"} and _checker_lines_on():
                        # a verdict the environment showed the actor; never a message. Only a checker-only line:
                        # any other line, including one that also names message, quit or cancel, is handled as such.
                        self._post_checker(item.get("checker"))
                        continue
                    if (
                        "checker" in item
                        and set(item) <= {"checker", "episode", "about"}
                        and _checker_lines_on()
                    ):
                        # a verdict about an action or item, or on an earlier episode named by its id (never by
                        # position)
                        self._post_checker(
                            item.get("checker"),
                            episode=item.get("episode"),
                            about=item.get("about"),
                        )
                        continue
                    if item.get("quit"):
                        line = "/quit"
                    elif item.get("cancel"):
                        if self._bridge is not None:
                            self._bridge.cancel_posted()
                        # A one-shot session's request is the session.
                        if not self._args.persist:
                            line = "/quit"
                        else:
                            if not await self._handle.cancel_request():
                                self._progress(
                                    "cancel ignored: the session takes no more requests",
                                )
                            continue
                    else:
                        message = item.get("message")
                        if not isinstance(message, str) or not message:
                            continue
                        if "input" in item:
                            # rev 2.3: the request's input as the world declares it beside the message (data, never
                            # parsed from text); tools.answer_from answers on it
                            from unify.common import declared_input

                            if not declared_input.declare(item.get("input")):
                                self._progress("the declared input was not kept: it has no JSON form or is too large")
                        line = message
                if line in {"/quit", "/exit", "/q"}:
                    from unify.actor.code_act_actor import SESSION_ENDED

                    self._progress("session ended")
                    self._stop_requested = True
                    await self._handle.stop(SESSION_ENDED)
                    return
                if self._mv2 is not None:
                    # UNIFY_MEMORY (rev 2.3): the librarian asked again on new evidence; its answer goes above
                    from unify.memory.integration import hooks as _mv2

                    above = await _mv2.reask_line(self._mv2, line)
                    if above:
                        line = f"{above}\n\n{line}"
                if self._bridge is not None:
                    # A record post, read at the next boundary.
                    await self._bridge.user_message(line)
                    continue
                await self._handle.submit(line)

    def _post_checker(self, raw: object, **where: object) -> None:
        """Hand a ``{"checker": ...}`` line to the memory run and answer it (memory v2.1 P9: a verdict the
        bed showed the actor, kept as an agent-visible checker signal). Without a memory run it is refused.
        *where*: the line's ``episode`` and ``about`` (a run with memory).
        """
        if self._mv2 is None:
            answer = {
                "type": "checker",
                "accepted": False,
                "reason": "this session records no memory",
            }
        else:
            answer = self._mv2.take_checker(raw, **where)
        if not answer["accepted"]:
            self._progress(f"checker line refused: {answer['reason']}")
        self._emit(**answer)

    def _post_outcome(self, raw: object) -> None:
        """Hand an ``{"outcome": ...}`` line to the request's memory run and answer it (unify/outcome.py): the
        answer names whether it was taken and, if not, why. Without a memory run it is refused."""
        if self._mv2 is None:
            answer = {
                "type": "outcome",
                "accepted": False,
                "reason": "this session records no memory",
            }
        else:
            answer = self._mv2.take_outcome(raw)
        if not answer["accepted"]:
            self._progress(f"outcome refused: {answer['reason']}")
        self._emit(**answer)

    # ── run ──────────────────────────────────────────────────────────────

    async def run(self, request: str) -> int:
        await self.start()
        args = self._args
        from unify.memory.integration import hooks as _mv2

        # UNIFY_MEMORY=on: export memory and open the request's record
        # before the actor builds its prompt.
        # rev 2.3: a request starts with no input or examples declared; its messages may declare them
        from unify.common import declared_input

        declared_input.clear()
        self._mv2 = _mv2.begin_request(request)
        if args.jsonl and self._mv2 is not None:
            # the episode's id, so a verdict that arrives after it ends can name it
            self._emit(type="episode", episode_id=self._mv2.episode_id)
        interactive = sys.stdin.isatty()
        # Someone can answer only at a terminal: there the actor's posts to
        # @user are read and answered; otherwise the record says nobody reads
        # them.
        clarify = interactive and not args.no_clarify
        if not interactive:
            self._progress("stdin is not a terminal; the actor cannot ask questions")
        elif not clarify:
            self._progress("--no-clarify: nobody reads the actor's @user posts")
        self._handle = await self._actor.act(
            request,
            persist=args.persist,
            clarification_enabled=clarify,
        )
        # UNIFY_MEMORY=in_episode: a segment fork continues this session at a checkpoint (inert otherwise)
        _mv2.attach_handle(self._handle)
        from unify.agents.cli_bridge import attach_bridge

        self._bridge = attach_bridge(self._handle, self._emit_record)
        watchers = [asyncio.create_task(self._watch_notifications())]
        reader = None
        if (interactive and (clarify or args.persist)) or args.jsonl:
            reader = asyncio.create_task(self._read_lines())
        try:
            result = await asyncio.wait_for(self._handle.result(), timeout=args.timeout)
        except asyncio.TimeoutError:
            print(f"timed out after {args.timeout:g}s", file=sys.stderr, flush=True)
            self._mv2_stop = {"reason": "timeout"}
            return 1
        except Exception as exc:
            print(f"error: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
            self._mv2_stop = {"reason": "error", "error": type(exc).__name__}
            return 1
        finally:
            if reader is not None and not args.persist:
                reader.cancel()

        # A persistent session's result arrives when the session ends. Unless
        # it was asked to end, its task loop ended on its own (at a step or
        # time limit): there is no session left to take a follow-up, so input
        # stops here and the session ends as after /quit, with its review and
        # an "ended" line, rather than reading messages it never answers.
        ended_on_its_own = args.persist and not self._stop_requested
        if ended_on_its_own and reader is not None:
            reader.cancel()
            await asyncio.gather(reader, return_exceptions=True)
            reader = None

        self._print_result(result)

        if ended_on_its_own:
            self._progress("the actor stopped on its own and takes no follow-ups")
        elif args.persist:
            self._progress("actor is waiting; type a follow-up, /quit to end")
            if reader is not None:
                await reader
        while not self._handle.done():
            await asyncio.sleep(0.2)
        for task in watchers:
            task.cancel()
        if self._mv2 is not None:
            # Record the episode and run the due consolidation passes, blocking.
            await self._mv2.finish(
                self._handle,
                consolidate=self._mv2_consolidate,
                progress=self._progress,
                emit=(lambda event: self._emit(**event)) if args.jsonl else None,
            )
        if args.jsonl:
            self._emit(type="ended")
        return 0

    def _print_result(self, result: object) -> None:
        if self._args.jsonl:
            self._emit(
                type="result",
                content=result if isinstance(result, str) else str(result),
            )
            return
        if self._args.json:
            payload = {
                "result": (
                    result if isinstance(result, (str, dict, list)) else str(result)
                ),
            }
            print(json.dumps(payload, indent=2, default=str), flush=True)
        else:
            print(result, flush=True)


def _read_request(args: argparse.Namespace) -> str:
    if args.request and args.request != "-":
        return args.request
    if sys.stdin.isatty():
        print("type the request; end with Ctrl-D", file=sys.stderr, flush=True)
    text = sys.stdin.read().strip()
    if not text:
        raise SystemExit(
            "unify act: no request given (pass it as an argument or on stdin)",
        )
    return text


async def _run_act(args: argparse.Namespace) -> int:
    session = Act(args)
    try:
        return await session.run(_read_request(args))
    finally:
        await session.close()


def _memory_refusal() -> str | None:
    """Why this run may not start under its memory settings (named, never a value), or None."""
    from unify.memory.integration.switch import memory_on, writer_refusal
    from unify.settings import SETTINGS

    return writer_refusal(SETTINGS) if memory_on(SETTINGS) else None


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    home = _configure_environment(args)
    refusal = _memory_refusal()
    if refusal is not None:
        # fail closed: nothing starts, no model is called
        print(f"unify: {refusal}", file=sys.stderr)
        return 2
    # memory's model lanes: with memory on and no lane given, this process starts the local lane relay and runs the
    # command as its child, with each lane's route (unify/lanes/launch.py); None: run it here
    from unify.lanes.launch import run_controller

    code = run_controller(list(sys.argv[1:] if argv is None else argv), home)
    if code is not None:
        return code
    try:
        return asyncio.run(_run_act(args))
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    sys.exit(main())
