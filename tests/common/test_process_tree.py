"""Symbolic: a sandbox's kill is verified: every process of it is gone when the kill returns.

``process_tree.kill_tree`` scans ``/proc`` again every round, so a process one scan could not link to the tree is
found by a later one: through the leader's group while that group lives, and through the box's pid namespace
(recorded when the box starts) whatever the process's parent, group or session.
"""

from __future__ import annotations

import os
import shutil
import signal
import subprocess
import sys
import uuid
from pathlib import Path

import pytest

from unify.common import process_tree
from unify.memory import sandbox_run

needs_bwrap = pytest.mark.skipif(
    shutil.which("bwrap") is None or not sys.platform.startswith("linux"),
    reason="bubblewrap (bwrap) is not installed; the box cannot run here",
)


def _sleep_argument() -> str:
    """A unique ``sleep`` duration (60-90 s) that names its process."""
    return f"{60 + uuid.uuid4().int % 30}.{uuid.uuid4().int % 10**6:06d}"


def _alive_with(argument: str) -> list[int]:
    """The live processes whose command line holds *argument* (a zombie's is empty)."""
    pids = []
    for proc in Path("/proc").iterdir():
        if not proc.name.isdigit():
            continue
        try:
            argv = (proc / "cmdline").read_bytes().split(b"\0")
        except OSError:
            continue
        if argument.encode() in argv:
            pids.append(int(proc.name))
    return pids


def _sweep(argument: str) -> None:
    """Kill whatever a failed test left behind, so it never outlives the test."""
    for pid in _alive_with(argument):
        try:
            os.kill(pid, signal.SIGKILL)
        except OSError:
            pass


def test_a_grandchild_its_exiting_parent_left_behind_is_killed_through_the_group():
    argument = _sleep_argument()
    # the leader's child starts the sleep and exits at once: the sleep is reparented out of the leader's tree but
    # stays in its group
    code = (
        "import os, subprocess, sys, time\n"
        "if os.fork() == 0:\n"
        f"    subprocess.Popen(['/bin/sleep', {argument!r}])\n"
        "    os._exit(0)\n"
        "os.wait()\n"
        "sys.stdout.write('ready\\n'); sys.stdout.flush()\n"
        "time.sleep(120)\n"
    )
    leader = subprocess.Popen(
        [sys.executable, "-I", "-c", code],
        stdout=subprocess.PIPE,
        start_new_session=True,
    )
    try:
        assert leader.stdout.readline() == b"ready\n"
        assert _alive_with(argument), "the sleep did not start"
        process_tree.kill_tree(leader.pid, exited=lambda: leader.poll() is not None, grace_s=10)
        assert _alive_with(argument) == []
    finally:
        _sweep(argument)
        if leader.poll() is None:
            leader.kill()
            leader.wait()


def test_a_plain_process_has_no_box_namespace():
    leader = subprocess.Popen([sys.executable, "-I", "-c", "import time; time.sleep(30)"])
    try:
        assert process_tree.box_namespace(leader.pid, timeout_s=0.2) is None
    finally:
        leader.kill()
        leader.wait()


def _box_that_detaches_a_sleep(argument: str) -> "sandbox_run.ConfinedBox":
    """A box whose Python starts a ``sleep`` in a session of its own (out of the box's group and tree), says it is
    ready and exits at once, just before the host kills the box."""
    code = (
        "import subprocess, sys\n"
        f"subprocess.Popen(['/bin/sleep', {argument!r}], start_new_session=True,\n"
        "                 stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)\n"
        "sys.stdout.write('ready\\n'); sys.stdout.flush()\n"
    )
    return sandbox_run.start_confined([str(sandbox_run.PYTHON), "-I", "-c", code])


@needs_bwrap
def test_a_box_records_its_own_pid_namespace():
    argument = _sleep_argument()
    box = _box_that_detaches_a_sleep(argument)
    try:
        ns = box.proc.box_pid_ns
        assert ns is not None and ns.startswith("pid:[")
        assert ns != process_tree.pid_namespace("self")
    finally:
        box.kill()
        _sweep(argument)


#: Busy processes beside the kills: enough to load the box's scheduling, few enough that a suite running in parallel
#: keeps its pace (one per CPU starved the other test processes).
BURNERS = 2


def _burners() -> list[subprocess.Popen]:
    """A few busy processes, so the kills below run under load."""
    return [subprocess.Popen([sys.executable, "-I", "-S", "-c", "while True: pass"]) for _ in range(BURNERS)]


@needs_bwrap
@pytest.mark.timeout(900)
def test_a_detached_sleep_never_outlives_the_box_kill_under_load():
    burners = _burners()
    arguments: list[str] = []
    try:
        for _ in range(200):
            argument = _sleep_argument()
            arguments.append(argument)
            box = _box_that_detaches_a_sleep(argument)
            try:
                assert box.stdout.readline() == b"ready\n"
            finally:
                box.kill()
            assert _alive_with(argument) == [], f"run {len(arguments)}: the detached sleep outlived the kill"
    finally:
        for burner in burners:
            burner.kill()
            burner.wait()
        for argument in arguments:
            _sweep(argument)


def _box_that_keeps_a_sleep(argument: str) -> "sandbox_run.ConfinedBox":
    """A box whose Python waits on a ``sleep``: it stays alive until it is killed."""
    code = f"import subprocess\nsubprocess.Popen(['/bin/sleep', {argument!r}]).wait()\n"
    return sandbox_run.start_confined([str(sandbox_run.PYTHON), "-I", "-c", code])


@needs_bwrap
@pytest.mark.timeout(300)
def test_a_box_that_ended_never_has_a_later_box_killed_in_its_place():
    # box A ends by itself; its namespace stays pinned (held open since its start) until its kill releases it, so no
    # box started meanwhile can take its number, and A's kill finds nothing of theirs
    a = sandbox_run.start_confined([str(sandbox_run.PYTHON), "-I", "-c", "pass"])
    name_a, fd_a = a.proc.box_pid_ns, a.proc.box_pid_ns_fd
    assert name_a is not None and fd_a is not None
    a.proc.wait(timeout=60)
    later: list = []
    arguments: list[str] = []
    try:
        for _ in range(20):
            arguments.append(_sleep_argument())
            later.append(_box_that_keeps_a_sleep(arguments[-1]))
            assert later[-1].proc.box_pid_ns != name_a  # A's number is still held
        for argument in arguments:
            assert _alive_with(argument)
        a.kill()  # A's cleanup: its namespace, held, matched by name
        assert a.proc.box_pid_ns_fd is None  # released after the kill
        for argument in arguments:
            assert _alive_with(argument), "a later box was killed by an earlier box's cleanup"
        # a name without its open fd is never trusted: a kill handed only box B's name leaves B alone
        dummy = subprocess.Popen([sys.executable, "-I", "-c", "import time; time.sleep(60)"], start_new_session=True)
        try:
            process_tree.kill_tree(
                dummy.pid,
                exited=lambda: dummy.poll() is not None,
                grace_s=10,
                pid_ns=later[0].proc.box_pid_ns,
            )
        finally:
            if dummy.poll() is None:
                dummy.kill()
                dummy.wait()
        assert _alive_with(arguments[0])
    finally:
        for box in later:
            box.kill()
        for argument in arguments:
            _sweep(argument)
