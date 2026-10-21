"""The source text of ``sh(command, timeout=None)``, the box-local shell helper the in-box ``tools`` module embeds.

:data:`SOURCE` is standard-library-only Python, self-contained, that the generated ``tools`` module includes
verbatim (it defines ``ShellResult`` and ``sh`` and private names starting ``_sh_``). It runs inside the agent's
interpreter box, so its commands get the same mounts, seccomp filter and rlimits as the cell; it never touches the
tool channel or the host.

Each call runs ``bash --norc --noprofile`` as a child of the interpreter with the sentinel protocol of the actor's
bash session (``unify/actor/execution/shell.py``): the command is sent base64-encoded and ``eval``'d with its stdin
on ``/dev/null``, stdout and stderr interleaved, followed by a fresh sentinel line carrying the exit status. After
the sentinel the shell reports its working directory and exported variables, which the next call starts from, so
``cd`` and ``export`` carry over from one call to the next and from one cell to the next (shell functions and
unexported variables do not). The bash process ends with its call: nothing the helper starts outlives the call
unless the command itself left a background job, which the host finds between cells like any process a cell leaves
running (the interpreter is then restarted).

Output keeps at most :data:`MAX_OUTPUT_BYTES` (its head and tail, the middle counted), as the actor's bash session
does. ``timeout`` (seconds) kills the command's process group and says so; without one the cell's own limit bounds
it.
"""

from __future__ import annotations


SOURCE = r'''
# --- sh(): one bash call per command; the working directory and exported variables carry over ---------------
import base64 as _sh_base64
import os as _sh_os
import re as _sh_re
import select as _sh_select
import signal as _sh_signal
import subprocess as _sh_subprocess
import threading as _sh_threading
import time as _sh_time
import uuid as _sh_uuid

_SH_MAX_OUTPUT_BYTES = 256 * 1024
_SH_READ_CHUNK = 65536
_SH_BASH = ("/bin/bash", "--norc", "--noprofile")
# never handed to the shell: the tool channel's descriptor names belong to this interpreter only
_SH_PRIVATE_ENV = ("TOOLS_CHANNEL",)
# set by bash itself in every process; carried over they would only describe the previous call's shell
_SH_SHELL_OWN = ("_", "SHLVL", "BASH_EXECUTION_STRING")
_sh_lock = _sh_threading.Lock()
_sh_state = {"cwd": None, "env": None}


class ShellResult(str):
    """The command's output (stdout and stderr interleaved) with ``.returncode``: the exit status, or None when
    the command timed out or its shell ended before reporting one."""

    returncode = None
    paths = ()
    next = None

    def __new__(cls, text, returncode):
        self = str.__new__(cls, text)
        self.returncode = returncode
        return self


def _sh_start_state():
    if _sh_state["env"] is None:
        env = {k: v for k, v in _sh_os.environ.items() if k not in _SH_PRIVATE_ENV}
        _sh_state["env"] = env
    if _sh_state["cwd"] is None:
        try:
            _sh_state["cwd"] = _sh_os.getcwd()
        except OSError:
            _sh_state["cwd"] = "/"
    return _sh_state["cwd"], dict(_sh_state["env"])


def _sh_capped(raw, dropped):
    half = _SH_MAX_OUTPUT_BYTES // 2
    if len(raw) > _SH_MAX_OUTPUT_BYTES or dropped:
        dropped += max(0, len(raw) - 2 * half)
        head, tail = raw[:half], raw[half:][-half:]
        raw = head + ("\n[... %d bytes of output omitted ...]\n" % dropped).encode() + tail
    return raw.decode("utf-8", errors="replace")


def _sh_take_state(raw):
    """The working directory and exported variables the shell reported after the sentinel, if complete."""
    parts = raw.split(b"\0")
    if len(parts) < 2 or parts[-1] != b"":
        return
    cwd = parts[0].decode("utf-8", errors="surrogateescape")
    env = {}
    for item in parts[1:-1]:
        key, sep, value = item.partition(b"=")
        if not sep or not key:
            continue
        name = key.decode("utf-8", errors="surrogateescape")
        if name in _SH_PRIVATE_ENV or name in _SH_SHELL_OWN:
            continue
        env[name] = value.decode("utf-8", errors="surrogateescape")
    if cwd.startswith("/"):
        _sh_state["cwd"] = cwd
    _sh_state["env"] = env


def sh(command, timeout=None):
    """Run *command* with bash in this box and return its output (stdout and stderr interleaved) as a string
    with ``.returncode``. The working directory and exported variables carry over to the next call; stdin is
    empty. *timeout* (seconds) kills the command; the cell's own time limit always applies.
    """
    if not isinstance(command, str):
        raise TypeError("sh() takes the command as a string")
    if timeout is not None and (not isinstance(timeout, (int, float)) or timeout <= 0):
        raise ValueError("sh() timeout must be a positive number of seconds or None")
    with _sh_lock:
        return _sh_run(command, timeout)


def _sh_run(command, timeout):
    cwd, env = _sh_start_state()
    if not _sh_os.path.isdir(cwd):
        cwd = "/"
    marker = "__SH_DONE_%s__" % _sh_uuid.uuid4().hex
    done_re = _sh_re.compile(b"\n" + _sh_re.escape(marker.encode()) + b" (-?\\d+)\n")
    encoded = _sh_base64.b64encode(command.encode("utf-8", errors="surrogateescape")).decode()
    line = (
        "eval \"$(printf '%%s' '%s' | base64 -d)\" </dev/null 2>&1; "
        "printf '\\n%%s %%d\\n' '%s' \"$?\"; "
        "printf '%%s\\0' \"$PWD\"; exec /usr/bin/env -0\n"
    ) % (encoded, marker)
    proc = _sh_subprocess.Popen(
        _SH_BASH,
        stdin=_sh_subprocess.PIPE,
        stdout=_sh_subprocess.PIPE,
        stderr=_sh_subprocess.STDOUT,
        cwd=cwd,
        env=env,
        start_new_session=True,
    )
    buf = bytearray()
    dropped = 0
    status = None
    timed_out = False
    try:
        try:
            proc.stdin.write(line.encode())
            proc.stdin.close()
        except OSError:
            pass
        fd = proc.stdout.fileno()
        deadline = None if timeout is None else _sh_time.monotonic() + float(timeout)
        while True:
            tick = 0.1
            if deadline is not None:
                tick = min(tick, max(0.0, deadline - _sh_time.monotonic()))
            ready = _sh_select.select([fd], [], [], tick)[0]
            if not ready:
                if proc.poll() is not None:
                    # the shell ended and nothing more is waiting (a background job may still hold the pipe)
                    break
                if deadline is not None and _sh_time.monotonic() >= deadline:
                    timed_out = True
                    break
                continue
            chunk = _sh_os.read(fd, _SH_READ_CHUNK)
            if not chunk:
                break
            buf.extend(chunk)
            if done_re.search(buf) is None and len(buf) > 4 * _SH_MAX_OUTPUT_BYTES:
                # bounded memory on runaway output: the head, and a tail long enough to hold the sentinel
                half = _SH_MAX_OUTPUT_BYTES // 2
                cut = len(buf) - half - (half + _SH_READ_CHUNK)
                dropped += cut
                del buf[half : half + cut]
    finally:
        if timed_out or proc.poll() is None:
            try:
                _sh_os.killpg(proc.pid, _sh_signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                pass
        proc.wait()
        proc.stdout.close()
    match = done_re.search(buf)
    state = b""
    if match is not None:
        status = int(match.group(1))
        state = bytes(buf[match.end() :])
        del buf[match.start() :]
    if not timed_out and match is not None:
        _sh_take_state(state)
    text = _sh_capped(bytes(buf), dropped)
    if timed_out:
        text += "\n[the command timed out after %s s and was killed]" % (timeout,)
        status = None
    elif match is None:
        # the shell ended before its sentinel (``exit`` in the command): its status is the command's
        status = proc.returncode
    return ShellResult(text, status)
'''


def source() -> str:
    """:data:`SOURCE`, the text the ``tools`` module embeds."""
    return SOURCE
