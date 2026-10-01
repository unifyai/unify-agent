"""A persistent bash session for ``execute_code(language="bash")``.

One long-lived ``bash --norc --noprofile`` process per session, so the working
directory, variables and functions carry over from one cell to the next. Each
cell is sent as one line -- the command base64-encoded and ``eval``'d with its
stdin on ``/dev/null``, so a command that reads stdin cannot swallow the
protocol -- followed by a sentinel carrying the exit status. Output is stdout
and stderr interleaved, up to the sentinel.

A cell that outruns its timeout kills the whole session (its process group,
or the sandbox's PID namespace) and says so; the next cell starts a fresh
session. Under ``UNIFY_WORKSPACE=sandboxed`` the process runs inside the
workspace sandbox (unify/sandbox.py).
"""

from __future__ import annotations

import asyncio
import base64
import os
import re
import signal
import uuid
from typing import Any, Optional

from unify import sandbox

__all__ = ["BashSession", "DEFAULT_SHELL_TIMEOUT_S", "MAX_OUTPUT_BYTES"]

DEFAULT_SHELL_TIMEOUT_S = 600.0
# Output beyond this keeps its head and tail; the middle is dropped and counted.
MAX_OUTPUT_BYTES = 256 * 1024
_READ_CHUNK = 65536


class BashSession:
    """One persistent bash process, optionally inside the workspace sandbox."""

    def __init__(
        self,
        *,
        policy: Optional[sandbox.SandboxPolicy] = None,
        cwd: Optional[str] = None,
    ) -> None:
        self._policy = policy
        self._cwd = cwd or (str(policy.workspace) if policy is not None else None)
        self._proc: Optional[asyncio.subprocess.Process] = None
        self._lock = asyncio.Lock()

    @property
    def is_running(self) -> bool:
        return self._proc is not None and self._proc.returncode is None

    @property
    def pid(self) -> Optional[int]:
        return self._proc.pid if self._proc is not None else None

    async def _start(self) -> None:
        argv = ["/bin/bash", "--norc", "--noprofile"]
        if self._policy is not None:
            env = sandbox.sandbox_env(self._policy)
            argv = sandbox.wrap_argv(argv, self._policy, cwd=self._cwd)
        else:
            env = dict(os.environ)
        with sandbox.unconfined():  # already wrapped; never wrap twice
            self._proc = await asyncio.create_subprocess_exec(
                *argv,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
                env=env,
                cwd=self._cwd,
                start_new_session=True,
            )

    async def execute(self, command: str, *, timeout: Optional[float]) -> dict:
        """Run *command*; returns ``output``, ``exit_code`` and ``error``."""
        async with self._lock:
            if not self.is_running:
                await self._start()
                # Whatever the shell (or the sandbox) prints while starting --
                # a locale warning, a mount failure -- belongs to no cell.
                warm = await self._run(":", timeout=30)
                if warm["exit_code"] != 0:
                    await self.close()
                    warm["error"] = (
                        f"The bash session failed to start. {warm['error'] or ''}"
                    ).strip()
                    return warm
            return await self._run(command, timeout=timeout)

    async def _run(self, command: str, *, timeout: Optional[float]) -> dict[str, Any]:
        assert self._proc is not None and self._proc.stdin is not None
        # A fresh sentinel per command: output a killed command left behind
        # can never end a later one.
        marker = f"__UNIFY_DONE_{uuid.uuid4().hex}__"
        done_re = re.compile(rb"\n" + re.escape(marker.encode()) + rb" (-?\d+)\n")
        encoded = base64.b64encode(command.encode()).decode()
        line = (
            f"eval \"$(printf '%s' '{encoded}' | base64 -d)\" </dev/null 2>&1; "
            f"printf '\\n%s %d\\n' '{marker}' \"$?\"\n"
        )
        buf = bytearray()
        dropped = 0
        try:
            self._proc.stdin.write(line.encode())
            await self._proc.stdin.drain()
        except (BrokenPipeError, ConnectionResetError) as exc:
            await self.close()
            return _result(
                b"",
                exit_code=None,
                error=f"The bash session could not take input: {exc}",
            )

        async def read_until_done() -> Optional[int]:
            nonlocal dropped
            while True:
                match = done_re.search(buf)
                if match:
                    status = int(match.group(1))
                    del buf[match.start() :]
                    return status
                chunk = await self._proc.stdout.read(_READ_CHUNK)
                if not chunk:
                    return None
                buf.extend(chunk)
                if len(buf) > 4 * MAX_OUTPUT_BYTES:
                    # Keep memory bounded on runaway output: the head, and a
                    # tail long enough to hold the sentinel.
                    half = MAX_OUTPUT_BYTES // 2
                    cut = len(buf) - half - (half + _READ_CHUNK)
                    dropped += cut
                    del buf[half : half + cut]

        try:
            status = await asyncio.wait_for(read_until_done(), timeout=timeout)
        except asyncio.CancelledError:
            # The cell was stopped: its command must not keep running.
            proc, self._proc = self._proc, None
            if proc is not None and proc.returncode is None:
                _kill_group(proc)
            raise
        except asyncio.TimeoutError:
            await self.close()
            return _result(
                bytes(buf),
                dropped=dropped,
                exit_code=None,
                error=f"The command timed out after {timeout}s. The bash session was "
                "killed; the next command starts a fresh one (working directory "
                "and variables are reset).",
            )
        if status is None:
            code = await self._reap()
            return _result(
                bytes(buf),
                dropped=dropped,
                exit_code=code,
                error=f"The bash session exited (status {code}); the next command "
                "starts a fresh one.",
            )
        return _result(
            bytes(buf),
            dropped=dropped,
            exit_code=status,
            error=None if status == 0 else f"The command exited with status {status}.",
        )

    async def _reap(self) -> Optional[int]:
        proc, self._proc = self._proc, None
        if proc is None:
            return None
        try:
            return await asyncio.wait_for(proc.wait(), timeout=5)
        except asyncio.TimeoutError:
            _kill_group(proc)
            return await proc.wait()

    async def close(self) -> None:
        """Kill the session and wait until it is gone. Safe to repeat."""
        proc, self._proc = self._proc, None
        if proc is None:
            return
        if proc.returncode is None:
            _kill_group(proc)
        try:
            await asyncio.wait_for(proc.wait(), timeout=10)
        except asyncio.TimeoutError:
            pass


def _kill_group(proc: asyncio.subprocess.Process) -> None:
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        try:
            proc.kill()
        except ProcessLookupError:
            pass


def _result(
    raw: bytes,
    *,
    exit_code: Optional[int],
    error: Optional[str],
    dropped: int = 0,
) -> dict:
    half = MAX_OUTPUT_BYTES // 2
    if len(raw) > MAX_OUTPUT_BYTES or dropped:
        dropped += max(0, len(raw) - 2 * half)
        head, tail = raw[:half], raw[half:][-half:]
        raw = head + f"\n[... {dropped} bytes of output omitted ...]\n".encode() + tail
    return {
        "output": raw.decode("utf-8", errors="replace"),
        "exit_code": exit_code,
        "error": error,
    }
