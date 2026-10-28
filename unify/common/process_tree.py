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
    "pin_box_namespace",
    "pin_namespace",
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


def pin_namespace(pid: int | str, starttime: str | None = None) -> tuple[str, int] | None:
    """The pid namespace *pid* runs in, held open: ``(name, fd)``. While the fd is open the namespace's inode stays
    allocated, so no namespace created later can take its number (``pid:[N]``) while it is matched by name; the
    caller closes the fd. None when it cannot be opened, or the namespace opened is not the one named, or, given
    the *starttime* the caller found *pid* with, *pid* no longer runs as that process once the namespace is open
    (it ended and its pid was taken by another process, whose namespace this would be)."""
    name = pid_namespace(pid)
    if name is None:
        return None
    try:
        fd = os.open(f"/proc/{pid}/ns/pid", os.O_RDONLY | os.O_CLOEXEC)
    except OSError:
        return None
    try:
        same = name == f"pid:[{os.fstat(fd).st_ino}]"
    except OSError:
        same = False
    if same and starttime is not None:
        st = proc_stat(int(pid))
        same = st is not None and st[2] == starttime
    if not same:
        os.close(fd)
        return None
    return name, fd


def pin_box_namespace(pid: int, timeout_s: float = 2.0) -> tuple[str, int] | None:
    """The pid namespace of the box that *pid* started (bubblewrap with ``--unshare-pid``), held open
    (:func:`pin_namespace`): ``(name, fd)`` of the namespace of the first process below *pid* that is not in this
    process's own, waiting up to *timeout_s* for the box to make it. None when no process below *pid* has a namespace
    of its own by then."""
    own = pid_namespace("self")
    deadline = time.monotonic() + timeout_s
    while True:
        for child, start in descendants(pid):
            ns = pid_namespace(child)
            if ns is not None and ns != own:
                pinned = pin_namespace(child, start)
                if pinned is not None and pinned[0] == ns:
                    return pinned
                if pinned is not None:
                    os.close(pinned[1])
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
    pid_ns_fd: Optional[int] = None,
) -> None:
    """SIGKILL the process group led by *pid*, every process below *pid* and every process in the box's pid
    namespace; return only once all are verified gone, raise otherwise.

    Every round scans ``/proc`` afresh, so a process one scan could not link (forked after it, or the child of a
    parent that was exiting during it) is found by a later one:

    * the tree, walked from *pid* while the leader still holds its pid and from every targeted process still
      running;
    * the group: a process whose group is *pid*'s, while that group has never been empty since the kill began
      (so its id cannot have been reused);
    * the namespace: a process in the box's pid namespace, *pid_ns* held open by *pid_ns_fd* (pinned when the box
      started: :func:`pin_box_namespace`; a name without its open fd is not trusted), or any namespace other than
      this process's own that a targeted process runs in, pinned as it is found, whatever the process's parent,
      group or session (one that left by ``setsid`` or was reparented included). A namespace is matched by name
      only while it is held open, so a namespace created after it ended can never take its number and be killed
      in its place; the fds the kill opens are closed before it returns or raises.

    Every process found gets SIGKILL. The kill returns once the leader has ended and two consecutive scans find
    nothing alive; ``RuntimeError`` names the survivors when any outlives *grace_s* seconds.

    *exited* says whether the leader has ended (and reaps it, as ``Popen.poll`` does); without it the leader counts
    as ended once it is gone or a zombie.
    """
    me = os.getpid()
    own_ns = pid_namespace("self")
    watched: set[str] = set()
    held: list[int] = []  # the fds this kill opened on the namespaces it watches
    if pid_ns and pid_ns != own_ns and pid_ns_fd is not None:
        try:
            if pid_ns == f"pid:[{os.fstat(pid_ns_fd).st_ino}]":
                watched.add(pid_ns)
        except OSError:
            pass
    leader = proc_stat(pid)
    leader_start = leader[2] if leader is not None else None
    born = int(leader_start) if leader_start is not None and leader_start.isdigit() else None
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
        roots = list(found)
        # the tree from the leader while it still holds its pid, walked in this round's table (one /proc read a round)
        if not leader_ended and leader_start is not None and pid in table and table[pid][3] == leader_start:
            roots.append(pid)
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
            if ns is not None and ns != own_ns and ns not in watched:
                pinned = pin_namespace(p, found[p])
                if pinned is not None:
                    held.append(pinned[1])
                    if pinned[0] == ns:
                        watched.add(ns)
        if watched:
            for p, row in table.items():
                if p in found or p in (me, pid) or row[1] in _ENDED:
                    continue
                # a process started before the box cannot be in its namespace: its namespace is not read
                if born is not None and row[3].isdigit() and int(row[3]) < born:
                    continue
                if pid_namespace(p) in watched:
                    found[p] = row[3]
        return found

    try:
        # the tree as it stands before the signal: a process the signal leaves running is waited on even once the
        # leader's pid no longer leads to it
        targets.update((child, start) for child, start in descendants(pid) if child != me)
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
            if found:
                # wait for the processes just signalled to end, checking only them (a full scan only after)
                settle = min(deadline, time.monotonic() + 0.05)
                while time.monotonic() < settle and any(alive(p, start) for p, start in found.items()):
                    time.sleep(0.002)
            else:
                # an empty scan is confirmed by the next one at once (the leader being reaped meanwhile)
                time.sleep(0.001)
    finally:
        for fd in held:
            try:
                os.close(fd)
            except OSError:
                pass
