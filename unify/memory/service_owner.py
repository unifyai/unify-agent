"""The runner's side of a runner-owned writer service (``UNIFY_MEMORY_WRITER_SERVICE_OWNER=runner``): the
environment to start it with, and its stop at the cell's end, before the key proxy.

The runner starts one service per cell, ``<venv python> -m unify.memory.writer_service --home H --foreground``, as the
main process of a sandbox it owns that outlives every session (the controller's sandbox ends with each session and would
take a service inside it down mid-call). It keeps the ``Popen`` of that sandbox, and at the cell's end calls
:func:`stop_owned` with it: the stop marker, the in-flight pass's deadline plus a grace, then SIGTERM, then SIGKILL, and
the stop is verified only when the process has exited and the service's lock is free. Only then does the runner stop
the key proxy.

Standard library only, and no import of the ``unify`` package: the runner may load this file by path (``python -I
<this file> stop --home H --pid P``, or ``runpy``) without the settings, the logger or the model client, which read
``.env`` files and fetch keys.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import re
import signal
import sys
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Optional

SERVICE_DIR = Path("memory-state") / "memory-service"
LOCK_FILE = "service.lock"
STATE_FILE = "service.json"
STOP_FILE = "service.stop"
#: The runner-owned service's own folder: its empty HOME and its current folder (no ``.env`` there).
OWNED_DIR = "owned"
#: Why the service is stopped (the service's own ``STOP_BY``): ``drain`` at the run's end, ``suspend`` at the end of one
#: stage of a run that goes on (the entries still owed stay owed for the next stage's service).
STOP_BY = ("drain", "suspend", "operator", "cap", "lease")
#: After the in-flight pass's deadline, this long before SIGTERM; then this long before SIGKILL; then this long for
#: the process to be gone.
STOP_GRACE_S = 30.0
TERM_S = 10.0
SETTLE_S = 5.0
#: The service's environment is an allowlist, never the runner's environment stripped: the locale and time zone,
#: and the names the build's settings module declares (``unify/settings.py``, read as text: no import) that are not
#: secret by name, the writer's and the helpers' route base URLs and token descriptor numbers among them (a route
#: token reaches it only on an inherited pipe; a base URL is a route to the key proxy, never a key). A name the
#: runner's environment holds for anything else (an audit's redaction list, a key) never reaches it.
BASICS = ("LANG", "LC_ALL", "LC_CTYPE", "TZ")
SECRET_NAME = re.compile(r"KEY|SECRET|PASSWORD|PASSWD|CREDENTIAL|COOKIE|AUTH|_TOKEN$|^TOKEN", re.IGNORECASE)
ROUTE_NAMES = (
    "UNIFY_MEMORY_WRITER_BASE_URL",
    "UNIFY_MEMORY_WRITER_TOKEN_FD",
    "UNIFY_MEMORY_LIBRARY_HELPER_BASE_URL",
    "UNIFY_MEMORY_LIBRARY_HELPER_TOKEN_FD",
)
OWNER_NAME = "UNIFY_MEMORY_WRITER_SERVICE_OWNER"
SETTINGS_FILE = Path(__file__).resolve().parents[1] / "settings.py"
#: A heartbeat older than this is not a live service (the service's own takeover age).
STALE_S = 90.0


def service_dir(home: Path | str) -> Path:
    return Path(home) / SERVICE_DIR


def _stamp(at: datetime) -> str:
    return at.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


def _parse(text: Any) -> Optional[datetime]:
    if not isinstance(text, str) or not text.endswith("Z"):
        return None
    try:
        return datetime.strptime(text[:-1], "%Y-%m-%dT%H:%M:%S.%f").replace(tzinfo=timezone.utc)
    except ValueError:
        try:
            return datetime.strptime(text[:-1], "%Y-%m-%dT%H:%M:%S").replace(tzinfo=timezone.utc)
        except ValueError:
            return None


def secret_name(name: str) -> bool:
    """Whether *name* reads as one that carries a credential (a key, a secret, a password, a token). A token's
    descriptor number (``*_TOKEN_FD``) and a count or price per token (``*_TOKENS``, ``*_PER_TOKEN``) do not."""
    up = name.upper()
    if up.endswith(("_TOKEN_FD", "_TOKENS", "_PER_TOKEN")):
        return False
    return bool(SECRET_NAME.search(up))


def _allowed(name: str) -> bool:
    return not secret_name(name)


def setting_names(path: Path = SETTINGS_FILE) -> frozenset[str]:
    """The ``UNIFY_*`` names the build's settings module declares (its class fields), read from its text."""
    import ast

    names: set[str] = set()
    for node in ast.walk(ast.parse(Path(path).read_text(encoding="utf-8"))):
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            # a field the module declares a secret (SecretStr) is never passed on, whatever its name
            if node.target.id.startswith("UNIFY_") and "SecretStr" not in ast.unparse(node.annotation):
                names.add(node.target.id)
    return frozenset(names)


def owned_env(
    home: Path | str, base: Optional[Mapping[str, str]] = None, *, extra: tuple[str, ...] = ()
) -> dict[str, str]:
    """The environment to start the runner-owned service with, an allowlist: a fixed ``PATH``, a fresh empty ``HOME``
    under the service's own folder, :data:`BASICS`, and of *base* (default this process's) only the settings the
    build declares (:func:`setting_names`) and *extra* names the runner adds, each not secret by name (descriptor
    numbers kept: the route tokens come on their pipes), with ``UNIFY_MEMORY_WRITER_SERVICE_OWNER=runner``. It holds
    no key, so the model client reads none, and its HOME holds no ``.env`` or cloud automation config to read one
    from; its ``TMPDIR`` is its own too (a session's temporary folder may go at that session's end). Creates the
    folders; :func:`owned_cwd` is the folder to start it in."""
    base = os.environ if base is None else base
    home_dir = owned_cwd(home) / "home"
    home_dir.mkdir(parents=True, exist_ok=True)
    allowed = (setting_names() | set(ROUTE_NAMES) | set(extra)) - {OWNER_NAME}
    env = {k: base[k] for k in BASICS if base.get(k)}
    for name in sorted(allowed):
        if name in base and _allowed(name):
            env[name] = base[name]
    tmp_dir = owned_cwd(home) / "tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    env.update({"PATH": "/usr/bin:/bin", "HOME": str(home_dir), "TMPDIR": str(tmp_dir), OWNER_NAME: "runner"})
    return env


def owned_cwd(home: Path | str) -> Path:
    folder = service_dir(home) / OWNED_DIR
    folder.mkdir(parents=True, exist_ok=True)
    return folder


def write_stop(home: Path | str, by: str, reason: str) -> None:
    """The stop marker: the service stops at it once its pass in flight has ended, and no request starts it again."""
    if by not in STOP_BY:
        raise ValueError(f"by is one of {', '.join(STOP_BY)}, not {by!r}")
    path = service_dir(home) / STOP_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    doc = {"v": 1, "by": by, "at": _stamp(datetime.now(timezone.utc)), "reason": str(reason)[:500]}
    fd, tmp = tempfile.mkstemp(prefix=".service.stop.", dir=path.parent)
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        json.dump(doc, fh)
    os.replace(tmp, path)


def lock_free(home: Path | str) -> bool:
    """Whether no process holds the service's lock (the service holds it its whole life; a flock is seen across pid
    namespaces). A missing lock file is free."""
    path = service_dir(home) / LOCK_FILE
    try:
        fd = os.open(path, os.O_RDWR)
    except FileNotFoundError:
        return True
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        return False
    finally:
        os.close(fd)  # closing drops a lock taken just now
    return True


def _proc_start(pid: Any) -> Optional[str]:
    try:
        rest = Path(f"/proc/{int(pid)}/stat").read_text().rsplit(")", 1)[-1].split()
    except (OSError, ValueError, TypeError):
        return None
    return None if len(rest) < 20 or rest[0] in ("Z", "X") else rest[19]


def _live_doc(home: Path | str) -> bool:
    """Whether ``service.json`` names a live process (its pid with its ``/proc`` start time) in this pid namespace."""
    doc = _state(home) or {}
    start = doc.get("proc_start")
    return bool(start) and doc.get("state") != "stopped" and _proc_start(doc.get("pid")) == str(start)


def owned_ready(home: Path | str, wait_s: float = 30.0, *, sleep: Callable[[float], Any] = time.sleep) -> bool:
    """Whether the runner-owned service is up, within *wait_s*: ``service.json`` names a runner-owned service that is
    running or idle, its heartbeat is fresh, and its lock is held. The runner calls it before the first session; a
    stage that finds it not ready refuses (no session runs without the service)."""
    limit = time.monotonic() + max(0.0, wait_s)
    while True:
        doc = _state(home) or {}
        beat = _parse(doc.get("heartbeat_at"))
        fresh = beat is not None and (datetime.now(timezone.utc) - beat).total_seconds() <= STALE_S
        if doc.get("owner") == "runner" and doc.get("state") in ("idle", "running") and fresh and not lock_free(home):
            return True
        if time.monotonic() >= limit:
            return False
        sleep(0.1)


def _state(home: Path | str) -> Optional[dict]:
    try:
        doc = json.loads((service_dir(home) / STATE_FILE).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return doc if isinstance(doc, dict) else None


@dataclass(frozen=True)
class Stopped:
    """How :func:`stop_owned` ended: ``verified`` (the process exited and the lock is free), ``how`` (``clean``: it
    stopped at the marker; ``term``; ``kill``), the pid still running if not verified."""

    verified: bool
    how: str
    leftover: tuple[int, ...] = ()


class _Pid:
    """A process by pid and ``/proc`` start time (for a runner that does not hold its ``Popen``)."""

    def __init__(self, pid: int):
        self.pid = int(pid)
        self._start = self._read_start()

    def _read_start(self) -> Optional[str]:
        try:
            rest = Path(f"/proc/{self.pid}/stat").read_text().rsplit(")", 1)[-1].split()
        except OSError:
            return None
        return None if len(rest) < 20 or rest[0] in ("Z", "X") else rest[19]

    def poll(self) -> Optional[int]:
        return None if self._start is not None and self._read_start() == self._start else 0

    def terminate(self) -> None:
        if self.poll() is None:
            os.kill(self.pid, signal.SIGTERM)

    def kill(self) -> None:
        if self.poll() is None:
            os.kill(self.pid, signal.SIGKILL)


def stop_owned(
    home: Path | str,
    proc: Any,
    *,
    by: str = "drain",
    reason: str = "the cell ended",
    grace_s: float = STOP_GRACE_S,
    term_s: float = TERM_S,
    settle_s: float = SETTLE_S,
    sleep: Callable[[float], Any] = time.sleep,
) -> Stopped:
    """Stop the runner-owned service whose sandbox *proc* is (a ``Popen``, or a pid): the stop marker, and nothing else
    until the in-flight pass's deadline (``service.json`` ``pass_deadline``) plus *grace_s* (the service stops at the
    marker once its pass has ended); then SIGTERM, *term_s*; then SIGKILL, *settle_s* (signals reach the sandbox's own
    process, the service ending with it: in effect a hard stop). Verified only when the process has exited, the
    service's lock is free and ``service.json`` names no live process. The runner stops the key proxy only after this
    returns verified."""
    if isinstance(proc, int):
        proc = _Pid(proc)
    write_stop(home, by, reason)

    def gone() -> bool:
        return proc.poll() is not None

    def until(limit: float) -> bool:
        while not gone() and time.monotonic() < limit:
            sleep(0.1)
        return gone()

    def pass_end() -> Optional[datetime]:
        return _parse((_state(home) or {}).get("pass_deadline"))

    end = pass_end()
    limit = time.monotonic() + max(0.0, (end - datetime.now(timezone.utc)).total_seconds() if end else 0.0) + grace_s
    while not gone() and time.monotonic() < limit:
        sleep(0.1)
        new_end = pass_end()
        if new_end is not None and new_end != end:  # a pass that started before the marker was seen
            end = new_end
            limit = time.monotonic() + max(0.0, (new_end - datetime.now(timezone.utc)).total_seconds()) + grace_s
    how = "clean"
    if not gone():
        how = "term"
        try:
            proc.terminate()
        except ProcessLookupError:
            pass
        until(time.monotonic() + term_s)
    if not gone():
        how = "kill"
        try:
            proc.kill()
        except ProcessLookupError:
            pass
        until(time.monotonic() + settle_s)
    exited = gone()
    free = lock_free(home)
    if exited and not free:  # its last descriptor on the lock closes with it: give it a moment
        limit = time.monotonic() + settle_s
        while not free and time.monotonic() < limit:
            sleep(0.05)
            free = lock_free(home)
    # verified: the sandbox exited, the service's lock is free, and service.json names no live process here
    live = _live_doc(home)
    return Stopped(exited and free and not live, how, () if exited else (int(getattr(proc, "pid", 0) or 0),))


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(prog="service_owner")
    sub = p.add_subparsers(dest="cmd", required=True)
    stop = sub.add_parser("stop", help="stop the runner-owned service (by its sandbox's pid) and verify")
    stop.add_argument("--home", required=True)
    stop.add_argument("--pid", type=int, required=True)
    stop.add_argument("--by", default="drain", choices=STOP_BY)
    stop.add_argument("--reason", default="the cell ended")
    args = p.parse_args(argv)
    got = stop_owned(args.home, args.pid, by=args.by, reason=args.reason)
    print(json.dumps({"verified": got.verified, "how": got.how, "leftover": list(got.leftover)}))
    return 0 if got.verified else 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
