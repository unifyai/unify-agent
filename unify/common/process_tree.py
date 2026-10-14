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

__all__ = [
    "KILL_GRACE_S",
    "alive",
    "box_namespace",
    "descendants",
    "kill_tree",
    "pid_namespace",
    "proc_stat",
]

KILL_GRACE_S = 10.0
#: The states of a process that has ended (a zombie, or dead).
_ENDED = ("Z", "X")


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


def pid_namespace(pid: int | str) -> str | None:
    """The pid namespace *pid* runs in (``/proc/<pid>/ns/pid``, e.g. ``pid:[4026532201]``); None when it cannot be
    read (the process is gone, or it is not ours to inspect)."""
    try:
        return os.readlink(f"/proc/{pid}/ns/pid")
    except OSError:
        return None


def box_namespace(pid: int, timeout_s: float = 2.0) -> str | None:
    """The pid namespace of the box that *pid* started (bubblewrap with ``--unshare-pid``): the namespace of the
    first process below *pid* that is not in this process's own, waiting up to *timeout_s* for the box to make it.
    None when no process below *pid* has a namespace of its own by then."""
    own = pid_namespace("self")
    deadline = time.monotonic() + timeout_s
    while True:
        for child, _ in descendants(pid):
            ns = pid_namespace(child)
            if ns is not None and ns != own:
                return ns
        st = proc_stat(pid)
        if time.monotonic() >= deadline or st is None or st[1] in _ENDED:
            return None
        time.sleep(0.005)


def _table() -> dict[int, tuple[int, str, int, str]]:
    """One pass over ``/proc``: pid -> (ppid, state, process group, starttime)."""
    out: dict[int, tuple[int, str, int, str]] = {}
    for entry in os.listdir("/proc"):
        if not entry.isdigit():
            continue
        try:
            raw = Path(f"/proc/{entry}/stat").read_text()
        except OSError:
            continue
        fields = raw[raw.rfind(")") + 2 :].split()
        out[int(entry)] = (int(fields[1]), fields[0], int(fields[2]), fields[19])
    return out



def kill_tree(
    pid: int,
    *,
    exited: Optional[Callable[[], bool]] = None,
    grace_s: Optional[float] = None,
    pid_ns: Optional[str] = None,
) -> None:
    """SIGKILL the process group led by *pid*, every process below *pid* and every process in the box's pid
    namespace; return only once all are verified gone, raise otherwise.

    Every round scans ``/proc`` afresh, so a process one scan could not link (forked after it, or the child of a
    parent that was exiting during it) is found by a later one:

    * the tree, walked from *pid* while the leader still holds its pid and from every targeted process still
      running;
    * the group: a process whose group is *pid*'s, while that group has never been empty since the kill began
      (so its id cannot have been reused);
    * the namespace: a process in the box's pid namespace, *pid_ns* (recorded when the box started:
      :func:`box_namespace`) or any namespace other than this process's own that a targeted process runs in,
      whatever its parent, group or session (one that left by ``setsid`` or was reparented included).

    Every process found gets SIGKILL. The kill returns once the leader has ended and two consecutive scans find
    nothing alive; ``RuntimeError`` names the survivors when any outlives *grace_s* seconds.

    *exited* says whether the leader has ended (and reaps it, as ``Popen.poll`` does); without it the leader counts
    as ended once it is gone or a zombie.
    """
    me = os.getpid()
    own_ns = pid_namespace("self")
    watched = {pid_ns} if pid_ns and pid_ns != own_ns else set()
    leader = proc_stat(pid)
    leader_start = leader[2] if leader is not None else None
    targets: dict[int, str] = {}
    group_held = leader is not None
    leader_ended = leader is None

    def leader_gone() -> bool:
        if exited is not None:
            return exited()
        return leader is None or not alive(pid, leader[2])

    def scan() -> dict[int, str]:
        nonlocal group_held
        table = _table()
        found = {p: start for p, start in targets.items() if alive(p, start)}
        if not leader_ended and leader_start is not None and alive(pid, leader_start):
            for child, start in descendants(pid):
                if child != me:
                    found.setdefault(child, start)
        roots = list(found)
        children: dict[int, list[int]] = {}
        for p, row in table.items():
            children.setdefault(row[0], []).append(p)
        todo = roots
        while todo:
            for child in children.get(todo.pop(), []):
                row = table[child]
                if child in found or child in (me, pid) or row[1] in _ENDED:
                    continue
                found[child] = row[3]
                todo.append(child)
        if group_held:
            members = [p for p, row in table.items() if row[2] == pid]
            if not members:
                group_held = False
            for p in members:
                row = table[p]
                if p not in (me, pid) and row[1] not in _ENDED:
                    found.setdefault(p, row[3])
        for p in list(found):
            ns = pid_namespace(p)
            if ns is not None and ns != own_ns:
                watched.add(ns)
        if watched:
            for p, row in table.items():
                if p in found or p in (me, pid) or row[1] in _ENDED:
                    continue
                if pid_namespace(p) in watched:
                    found[p] = row[3]
        return found

    if leader is not None:
        try:
            os.killpg(pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass
        if alive(pid, leader[2]):
            try:
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass

    deadline = time.monotonic() + (KILL_GRACE_S if grace_s is None else grace_s)
    empty = 0
    while True:
        found = scan()
        for p, start in found.items():
            targets[p] = start
            try:
                os.kill(p, signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                pass
        if leader_gone():
            leader_ended = True
        empty = empty + 1 if leader_ended and not found else 0
        if empty >= 2:
            return
        if time.monotonic() > deadline:
            raise RuntimeError(
                f"sandbox processes survived SIGKILL: leader {pid}, others {sorted(found)}",
            )
        time.sleep(0.05)
