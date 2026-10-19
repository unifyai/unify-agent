"""The fork's relay to the run record (``UNIFY_AGENTS_RUN_RECORD=on``): a host-side process between the fork's box and
the record.

The fork's worker (:mod:`unify.memory.integration.fork_worker`) runs in a box that binds no state directory, so it
cannot open the record, and it may outlive the request that started it. This process outlives neither: the request
starts it in place of the box, it starts the box, and it ends when the box ends. It hosts the fork's place on the
record (the request put the fork on the roster under its name; :meth:`unify.agents.run_record.RunRecord.attach`), and
answers the fork's comms calls (``send``, ``messages``, ``wait``, ``who``) and its turn-boundary blocks, one JSON line
each way over the worker's stdout and stdin, with the same texts every agent's comms tools give
(:mod:`unify.agents.run_record_tools`). When the box ends, the fork leaves the record.

Input: one JSON document on stdin (never argv, the environment or a file): ``home``, ``name``, ``argv`` (the box),
``doc`` (the fork's request, its route token in the headers), ``lock_fd`` (the episode's staging lock, inherited and
handed on to the box) and ``log`` (the worker's log). The route token is registered as a secret of this process, so
nothing it writes to the record can carry it.
"""

from __future__ import annotations

import asyncio
import json
import os
import signal
import sys
import time
from typing import Any

#: The longest line the box may send (a comms call), and the box's own wall bound plus a margin as this relay's.
LINE_LIMIT = 1 << 20
_MARGIN_S = 120.0


def _register_token(doc: dict) -> None:
    from unify.process_secrets import register_secret

    headers = doc.get("headers") if isinstance(doc, dict) else None
    auth = (headers or {}).get("Authorization") if isinstance(headers, dict) else None
    if isinstance(auth, str) and auth.startswith("Bearer "):
        register_secret("UNIFY_MEMORY_LIBRARY_FORK_TOKEN", auth[len("Bearer ") :])


async def answer(view: Any, op: Any, args: Any, deadline: float) -> str:
    """The text one comms call of the fork gets (the same texts as every agent's tools)."""
    from unify.agents import run_record_tools as T
    from unify.agents.record import MAX_WAIT_S
    from unify.common.tool_registry import refusal

    args = args if isinstance(args, dict) else {}
    if op == "block":
        return await asyncio.to_thread(view.take_block) or ""
    if op == "send":
        text, _form = await asyncio.to_thread(T.send_text, view, args.get("text"))
        return text
    if op == "messages":
        return await asyncio.to_thread(
            T.messages_text, view, all_entries=args.get("all") is True
        )
    if op == "who":
        return await asyncio.to_thread(T.who_text, view)
    if op == "wait":
        timeout = args.get("timeout")
        if timeout is not None and (
            isinstance(timeout, bool)
            or not isinstance(timeout, (int, float))
            or not 0 < timeout <= MAX_WAIT_S
        ):
            return str(refusal("wait", "bad_timeout"))
        return await T.wait_text(view, timeout, deadline=deadline)
    return "not run: only tools.send, tools.messages, tools.wait and tools.who run here"


async def relay(spec: dict) -> int:
    from unify.agents.run_record import RunRecordView, join_run_record
    from unify.memory.integration.fork_worker import WALL_GUARD_S

    doc = spec["doc"]
    _register_token(doc)
    name = str(spec["name"])
    record = join_run_record(spec["home"])
    record.attach(name)
    view = RunRecordView(record, name, "fork")
    started = time.monotonic()
    deadline = started + WALL_GUARD_S
    lock_fd = spec.get("lock_fd")
    fds = (
        (int(lock_fd),)
        if isinstance(lock_fd, int) and not isinstance(lock_fd, bool)
        else ()
    )
    log = open(spec["log"], "ab") if spec.get("log") else None
    try:
        proc = await asyncio.create_subprocess_exec(
            *spec["argv"],
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=log if log is not None else asyncio.subprocess.DEVNULL,
            limit=LINE_LIMIT,
            pass_fds=fds,
            close_fds=True,
            env={"PATH": "/usr/bin:/bin"},
        )
    finally:
        for fd in fds:
            try:
                os.close(
                    fd
                )  # the box holds the lock now: it is held exactly while the box lives
            except OSError:
                pass
        if log is not None:
            log.close()
    loop = asyncio.get_running_loop()
    loop.add_signal_handler(signal.SIGTERM, _stop, proc)
    try:
        proc.stdin.write((json.dumps(doc, default=str) + "\n").encode("utf-8"))
        await proc.stdin.drain()
        while True:
            left = deadline + _MARGIN_S - time.monotonic()
            if left <= 0:
                _stop(proc)
                break
            try:
                line = await asyncio.wait_for(proc.stdout.readline(), timeout=left)
            except asyncio.TimeoutError:
                _stop(proc)
                break
            except (ValueError, asyncio.LimitOverrunError):
                _stop(
                    proc
                )  # a line past the limit: the box is not speaking the protocol
                break
            if not line:
                break
            try:
                call = json.loads(line)
            except ValueError:
                continue
            if not isinstance(call, dict):
                continue
            try:
                text = await answer(view, call.get("op"), call.get("args"), deadline)
            except (
                Exception
            ) as exc:  # noqa: BLE001 - a failed call is answered, never ends the fork
                text = f"not run: the record could not answer ({type(exc).__name__})"
            try:
                proc.stdin.write(
                    (json.dumps({"id": call.get("id"), "text": text}) + "\n").encode(
                        "utf-8"
                    )
                )
                await proc.stdin.drain()
            except (BrokenPipeError, ConnectionResetError):
                break
        return await proc.wait()
    finally:
        try:
            record.leave(name)
        except Exception:  # noqa: BLE001 - leaving never changes how the fork ended
            pass


def _stop(proc: Any) -> None:
    try:
        proc.terminate()
    except ProcessLookupError:
        pass


def main() -> int:
    spec = json.loads(sys.stdin.buffer.read().decode("utf-8"))
    os.dup2(os.open(os.devnull, os.O_RDONLY), 0)
    try:
        return int(asyncio.run(relay(spec)) or 0)
    except (
        Exception
    ) as exc:  # noqa: BLE001 - the class only: a message could carry recorded text
        print(f"fork relay: {type(exc).__name__}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
