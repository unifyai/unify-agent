"""Verified termination of a sandbox: a process group and every process below it.

Shared by every sandbox the harness starts (the actor's Python worker and bash
sessions, ``unify/actor/execution``; the memory agents' boxes,
``unify/memory/sandbox_run.py``). Stdlib only.
"""

from __future__ import annotations

import os
import signal
import time
from collections.abc import Callable
from pathlib import Path
from typing import Optional

__all__ = ["KILL_GRACE_S", "alive", "descendants", "kill_tree", "proc_stat"]

KILL_GRACE_S = 10.0


def proc_stat(pid: int) -> tuple[int, str, str] | None:
    """(ppid, state, starttime) of *pid*, or None when it no longer exists."""
    try:
        raw = Path(f"/proc/{pid}/stat").read_text()
    except OSError:
        return None
    fields = raw[raw.rfind(")") + 2 :].split()
    return int(fields[1]), fields[0], fields[19]


def descendants(pid: int) -> list[tuple[int, str]]:
    """Every live descendant of *pid* as (pid, starttime), so a reused pid is never mistaken for it."""
    children: dict[int, list[tuple[int, str]]] = {}
    for entry in os.listdir("/proc"):
        if not entry.isdigit():
            continue
        st = proc_stat(int(entry))
        if st is not None:
            children.setdefault(st[0], []).append((int(entry), st[2]))
    found: list[tuple[int, str]] = []
    todo = [pid]
    while todo:
        for child in children.get(todo.pop(), []):
            found.append(child)
            todo.append(child[0])
    return found


def alive(pid: int, starttime: str) -> bool:
    """Whether *pid* still runs as the process started at *starttime* (a zombie has ended)."""
    st = proc_stat(pid)
    return st is not None and st[2] == starttime and st[1] not in ("Z", "X")


def kill_tree(
    pid: int,
    *,
    exited: Optional[Callable[[], bool]] = None,
    grace_s: Optional[float] = None,
) -> None:
    """SIGKILL the process group led by *pid* and every process below *pid*; raise unless all are gone.

    *exited* says whether the leader has ended (and reaps it, as ``Popen.poll`` does); without it the
    leader counts as ended once it is gone or a zombie. Raises ``RuntimeError`` naming the survivors when
    any process outlives *grace_s* seconds after the signal.
    """
    leader = proc_stat(pid)
    victims = descendants(pid)
    try:
        os.killpg(pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        pass
    for child, start in victims:
        if alive(child, start):
            try:
                os.kill(child, signal.SIGKILL)
            except ProcessLookupError:
                pass
    if leader is not None and alive(pid, leader[2]):
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass

    def leader_gone() -> bool:
        if exited is not None:
            return exited()
        return leader is None or not alive(pid, leader[2])

    deadline = time.monotonic() + (KILL_GRACE_S if grace_s is None else grace_s)
    while True:
        survivors = [child for child, start in victims if alive(child, start)]
        if leader_gone() and not survivors:
            return
        if time.monotonic() > deadline:
            raise RuntimeError(
                f"sandbox processes survived SIGKILL: leader {pid}, others {survivors}",
            )
        time.sleep(0.05)
