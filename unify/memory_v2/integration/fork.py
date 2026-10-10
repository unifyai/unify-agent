"""The episode-end fork (memory v2.1 design r5 §2-§3; ``UNIFY_MEMORY_V21_FORK``, off by default).

After the reply has been returned and the episode recorded, the actor's own conversation is continued once more by a
separate process: its last request's messages, tools and tool choice unchanged, so the provider serves the prefix
from cache, plus one fixed instruction. The process writes what the reply stages into
``<state>/memory-staging/<episode>/`` and nothing else.

* **F1, the boundary:**
  - :mod:`.fork_worker` runs by path with the system python (``/usr/bin/python3 -I /s/fork_worker.py``). It is
    standard library only.
  - The request reaches it on stdin, a fresh pipe, never argv, the environment or a file.
  - Its environment is PATH and PYTHONDONTWRITEBYTECODE, nothing else.
  - It calls the actor's own proxy route (``UNILLM_LLM_GATEWAY_URL`` with its placeholder key). Without a local
    proxy and a placeholder key there is no fork, so it never reaches a provider directly and never holds a key.
* **F2, mounts:** a bubblewrap box with a read-only root (``/usr``) and a private ``/tmp``. Home, the state dir and
  the library are not mounted. The worker file is read-only at ``/s``. The only writable bind is the episode's
  fresh staging dir.
  - It keeps only the network of the process that starts it (``--share-net``). Under the runner that is the
    controller's netbridge relay to the proxy, so the actor's own route.
* **F3:** it starts after the episode is recorded, from bytes serialised once. Nothing flows back to the reply,
  the transcript or the record.
* **F6:** with the switch off, :func:`start` returns at once: no import of the actor, no process, no staging dir.

**Lifetime.** The fork runs in its own session. With ``UNIFY_MEMORY_V21_WAIT_SLOT=on`` (hosts whose sandbox ends
every process of the controller with it), :func:`wait` holds the CLI after the reply until the fork ends. That wait
is bounded by :data:`WAIT_GUARD_S`, an operational guard reported in ``fork.json``, not a design limit.

**Spend.** The calls carry ``x-unify-call-kind: actor_fork``, journaled as ``other`` by the pinned proxy. They also
carry ``x-unify-session: fork.<episode>``, which is what separates their spend, and ``x-unify-parent: <episode>``.
There is no token cap; the run-level proxy guards bound the money.
"""

from __future__ import annotations

import ipaddress
import json
import os
import shutil
import subprocess
import threading
from pathlib import Path
from typing import Any, Callable
from urllib.parse import urlparse

INSTRUCTION = (
    "Your work on this request is finished and your reply has been sent; nothing you write now changes it. "
    "Write down what would help with similar requests later, as files. For code you wrote here that worked and "
    "could be reused, write it as a general function with a docstring in `candidates/<name>.py`, and the inputs "
    "and outputs it had in this conversation in `cases/<name>.json`. For each thing that failed and why, and for "
    "any other lesson, write a short note in `notes.md`, stated in general terms so it applies beyond this "
    "request, and point to the part of this conversation it comes from. Write each file as a fenced block whose "
    "first line is `path: <relative path>`. If nothing here is worth keeping, write `notes.md` saying so."
)
WORKER_FILE = Path(__file__).resolve().parent / "fork_worker.py"
BOX_WORKER = "/s/fork_worker.py"
#: The live audit's mark for the fork worker in a process's command line (runner token audits).
FORK_MARK = BOX_WORKER
STAGING_IN_BOX = "/staging"
#: The actor's proxy placeholders (continual_arc_baselines.proxy.PLACEHOLDERS): the only keys the fork may send.
PLACEHOLDERS = ("arc-proxy", "ARC_LLM_API_KEY")
#: Operational guard on the CLI's wait for the fork (run safety, reported; not a design limit).
WAIT_GUARD_S = 1800.0
WRITE_GUARD_S = 120.0


def staging_dir(paths: Any, eid: str) -> Path:
    return Path(paths.state_dir) / "memory-staging" / eid


def route(environ: dict) -> tuple[tuple[str, str] | None, str | None]:
    """The actor's proxy route ``(chat completions URL, placeholder key)``, or why the fork has none."""
    base = (environ.get("UNILLM_LLM_GATEWAY_URL") or "").strip().rstrip("/")
    key = (environ.get("UNILLM_LLM_GATEWAY_KEY") or "").strip()
    if not base or not key:
        return None, "no proxy route"
    u = urlparse(base)
    host = (u.hostname or "").lower()
    try:
        local = host == "localhost" or ipaddress.ip_address(host).is_loopback
    except ValueError:
        local = False
    if u.scheme not in ("http", "https") or not local:
        return None, "the gateway is not a local proxy"
    if key not in PLACEHOLDERS:
        return None, "the gateway key is not a proxy placeholder"
    return (base + "/chat/completions", key), None


def _model(client: Any) -> str | None:
    endpoint = str(
        getattr(client, "endpoint", "") or getattr(client, "model", "") or "",
    )
    return endpoint.rsplit("@", 1)[0] or None


def request_doc(
    handle: Any,
    eid: str,
    url: str,
    key: str,
) -> tuple[dict | None, str | None]:
    """The fork's request, from the session's last request as sent, or why the session cannot be forked."""
    from unify.actor.code_act_actor import _session_fork_source

    inner = getattr(handle, "_inner", None) or handle
    source, why = _session_fork_source(inner, getattr(handle, "_actor", None))
    if source is None:
        return None, why
    client = source["client"]
    model = _model(client)
    if not model:
        return None, "the session's client names no model"
    body: dict = {
        "model": model,
        "messages": [
            *source["sent_messages"],
            {"role": "user", "content": INSTRUCTION},
        ],
        "tools": source["tools"],
    }
    if source.get("tool_choice") is not None:
        body["tool_choice"] = source["tool_choice"]
    effort = getattr(client, "reasoning_effort", None)
    if isinstance(effort, str) and effort:
        # as the actor's own transport sends it: LiteLLM's OpenRouter route passes reasoning_effort at the top level
        body["reasoning_effort"] = "xhigh" if effort == "max" else effort
    headers = {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
        "x-unify-call-kind": "actor_fork",
        "x-unify-session": f"fork.{eid}",
        "x-unify-parent": eid,
    }
    return {"episode": eid, "url": url, "headers": headers, "body": body}, None


def _record(staging: Path, eid: str, status: str) -> None:
    doc = {
        "episode": eid,
        "status": status,
        "model": None,
        "turns": 0,
        "usage": {},
        "usd": None,
        "files": [],
        "refused_files": [],
    }
    (staging / "fork.json").write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n")


def box_argv(staging: Path, *, die_with_parent: bool) -> list[str]:
    """bubblewrap around the system python: read-only /usr, private /tmp, the worker file read-only, the staging
    dir the only writable bind, the starting process's network only."""
    bwrap, prlimit = shutil.which("bwrap"), shutil.which("prlimit")
    if not bwrap or not prlimit:
        raise RuntimeError(
            "bubblewrap and prlimit are required: the fork never runs outside a box",
        )
    args = [bwrap, "--unshare-all", "--share-net", "--new-session"]
    if die_with_parent:
        args.append("--die-with-parent")
    args += ["--ro-bind", "/usr", "/usr"]
    for link, target in (
        ("/lib", "usr/lib"),
        ("/lib64", "usr/lib64"),
        ("/bin", "usr/bin"),
        ("/sbin", "usr/sbin"),
    ):
        if os.path.islink(link):
            args += ["--symlink", target, link]
    for path in ("/etc/hosts", "/etc/ssl", "/etc/ca-certificates"):
        if os.path.exists(path):
            args += ["--ro-bind", path, path]
    args += [
        "--proc", "/proc", "--dev", "/dev", "--tmpfs", "/tmp",
        "--ro-bind", str(WORKER_FILE), BOX_WORKER,
        "--bind", str(staging), STAGING_IN_BOX,
        "--chdir", "/tmp", "--clearenv", "--setenv", "PATH", "/usr/bin", "--setenv", "PYTHONDONTWRITEBYTECODE", "1",
        prlimit, "--nproc=64:64", "--nofile=256:256", "--as=2147483648:2147483648", "--fsize=8388608:8388608",
        "--", "/usr/bin/python3", "-I", BOX_WORKER, "--staging", STAGING_IN_BOX,
    ]  # fmt: skip
    return args


class Fork:
    """A started fork: its process, the thread writing its request, and its staging dir."""

    def __init__(self, proc: Any, writer: threading.Thread, staging: Path, eid: str):
        self.proc, self.writer, self.staging, self.eid = proc, writer, staging, eid


def start(
    paths: Any,
    handle: Any,
    eid: str,
    settings: Any,
    *,
    environ: dict | None = None,
    popen: Callable[..., Any] = subprocess.Popen,
    progress: Callable[[str], None] = lambda _m: None,
) -> Fork | None:
    """Start the episode's fork, or record why not; never raises. Off: returns None and touches nothing."""
    from . import switch

    if not (switch.v21_enabled(settings) and switch.v21_fork(settings)):
        return None
    staging = staging_dir(paths, eid)
    try:
        staging.mkdir(parents=True, exist_ok=False)
    except OSError as exc:
        progress(f"memory v2.1: fork not started: staging dir: {type(exc).__name__}")
        return None
    try:
        r, why = route(dict(os.environ if environ is None else environ))
        doc = None
        if r is not None:
            doc, why = request_doc(handle, eid, *r)
        if doc is None:
            _record(staging, eid, f"refused: {why}")
            progress(f"memory v2.1: fork refused: {why}")
            return None
        data = json.dumps(doc, default=str).encode("utf-8")
        log = open(Path(paths.state_dir) / "fork-worker.log", "ab")
        try:
            proc = popen(
                box_argv(staging, die_with_parent=switch.v21_wait_slot(settings)),
                stdin=subprocess.PIPE,
                stdout=log,
                stderr=log,
                start_new_session=True,
                close_fds=True,
                env={"PATH": "/usr/bin:/bin"},
            )
        finally:
            log.close()

        def feed() -> None:
            try:
                proc.stdin.write(data)
            except (BrokenPipeError, OSError):
                pass
            finally:
                try:
                    proc.stdin.close()
                except OSError:
                    pass

        writer = threading.Thread(target=feed, name=f"fork-request-{eid}", daemon=True)
        writer.start()
        progress("memory v2.1: fork started")
        return Fork(proc, writer, staging, eid)
    except (
        Exception
    ) as exc:  # noqa: BLE001 - the request is already answered and recorded
        _record(staging, eid, f"error: {type(exc).__name__}")
        progress(f"memory v2.1: fork failed to start: {type(exc).__name__}")
        return None


def finish(
    fork: Fork | None,
    settings: Any,
    progress: Callable[[str], None] = lambda _m: None,
) -> None:
    """After the reply: hand over the request; with ``UNIFY_MEMORY_V21_WAIT_SLOT=on``, wait for the fork (bounded by
    :data:`WAIT_GUARD_S`, operational). A fork that leaves no ``fork.json`` gets one saying why.
    """
    from . import switch

    if fork is None:
        return
    fork.writer.join(WRITE_GUARD_S)
    if not switch.v21_wait_slot(settings):
        return
    try:
        fork.proc.wait(timeout=WAIT_GUARD_S)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(fork.proc.pid, 9)
        except OSError:
            pass
        fork.proc.wait()
        if not (fork.staging / "fork.json").exists():
            _record(
                fork.staging,
                fork.eid,
                f"error: operational wait guard ({WAIT_GUARD_S:g} s) reached",
            )
    if not (fork.staging / "fork.json").exists():
        _record(fork.staging, fork.eid, f"error: worker exit {fork.proc.returncode}")
    progress("memory v2.1: fork ended")
