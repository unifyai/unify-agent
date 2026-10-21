"""The world the workspace-sandbox tests run in: a home and ``UNIFY_HOME`` to hide.

Both are placed outside ``/tmp`` on purpose: the sandbox's ``/tmp`` is private,
so a secret under the host's ``/tmp`` would be invisible whether or not its
mask worked.

Cells run as the actor's do: in its persistent interpreter
(unify/common/box_interpreter.py), the sandboxed Python worker
(:func:`actor_interpreter`); a shell command runs as a cell's subprocess
(:func:`sh`), confined like the cell.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import socket
import sqlite3
import sys
import threading
from pathlib import Path

import pytest

from unify import sandbox
from unify.settings import SETTINGS

needs_bwrap = pytest.mark.skipif(
    shutil.which("bwrap") is None or not sys.platform.startswith("linux"),
    reason="bubblewrap (bwrap) is not installed; the sandbox cannot run here",
)

SSH_SECRET = "PRIVATE-KEY-MATERIAL-5c1e"  # pragma: allowlist secret
ENV_SECRET = "env-file-value-77d2a9"  # pragma: allowlist secret
TOKEN_VALUE = "tok-sandbox-3141592653"  # pragma: allowlist secret
STATE_SECRET = "unify-state-only-8e8e"  # pragma: allowlist secret


@pytest.fixture
def world(unify_home, monkeypatch, request):
    """A home and ``UNIFY_HOME`` outside /tmp, seeded with things to hide."""
    root_dir = Path(__file__).resolve().parents[3] / "logs" / "sandbox_worlds"
    digest = hashlib.md5(request.node.nodeid.encode()).hexdigest()[:12]
    root = root_dir / f"{digest}-{os.getpid()}"
    shutil.rmtree(root, ignore_errors=True)
    home = root / "home"
    state = root / "unify"
    (home / ".ssh").mkdir(parents=True)
    (home / ".ssh" / "id_rsa").write_text(SSH_SECRET + "\n")
    (home / ".env").write_text(f"SOME_SETTING={ENV_SECRET}\n")
    (home / "project").mkdir()
    (home / "project" / ".env").write_text(f"OTHER={ENV_SECRET}\n")
    (home / "project" / "notes.txt").write_text("alpha\nneedle one\nbeta\n")
    (home / "plain.txt").write_text("readable outside the workspace\n")
    (state / "logs").mkdir(parents=True)
    (state / "logs" / "unify.log").write_text(STATE_SECRET + "\n")
    (state / "embeddings.sqlite").write_text(STATE_SECRET)
    (state / "transcripts").mkdir()
    (state / "transcripts" / "s.jsonl").write_text('{"seq": 0}\n')
    (state / "workspace").mkdir()
    (state / "workspace" / "data.txt").write_text(
        "".join(f"line {i}\n" for i in range(1, 11)),
    )
    con = sqlite3.connect(state / "store.sqlite")
    con.execute("create table functions (name text)")
    con.execute("insert into functions values ('f')")
    con.commit()
    con.close()

    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("UNIFY_HOME", str(state))
    monkeypatch.setenv("FAKE_SERVICE_TOKEN", TOKEN_VALUE)
    monkeypatch.setenv("DB_PASSWORD", TOKEN_VALUE)
    monkeypatch.setenv("UNIFY_SANDBOX_PROBE", "visible")
    # A cell gets an allow-list of the harness's environment: the probe is
    # declared, and so are the two credentials, which the second filter (a
    # credential's name) still keeps out.
    monkeypatch.setattr(
        SETTINGS,
        "UNIFY_CELL_ENV_ALLOW",
        "UNIFY_SANDBOX_PROBE,FAKE_SERVICE_*,DB_PASSWORD",
    )
    monkeypatch.setattr(SETTINGS, "UNIFY_LOCAL_ROOT", "")
    monkeypatch.setattr(SETTINGS, "UNIFY_WORKSPACE_NETWORK", "")
    monkeypatch.setattr(SETTINGS, "UNIFY_WORKSPACE_PROXY_PORT", 0)
    monkeypatch.setattr(sandbox, "_POLICY_CACHE", None)
    yield {"home": home, "state": state, "workspace": state / "workspace"}
    shutil.rmtree(root, ignore_errors=True)


async def actor_interpreter(*, shadow=None, limit: float = 60.0):
    """The actor's persistent interpreter with no tool channel: the sandboxed worker, its cells' namespace served
    from *shadow* (the actor's base globals when None). Stop it with ``await interp.stop()``."""
    from unify.actor.in_code_tools import display
    from unify.common.box_interpreter import ActorProfile, BoxInterpreter
    from unify.function_manager.execution_env import create_execution_globals

    shadow = create_execution_globals() if shadow is None else shadow
    # as the actor's: the name marks the worker's own display
    shadow.setdefault("display", display)
    return await BoxInterpreter.start(
        agent="actor",
        profile=ActorProfile(
            shadow=shadow,
            idle_restart_s=None,
        ),
        binds=None,
        env={},
        channel=None,
        stub_source=None,
        cell_limit_s=limit,
    )


async def sh(interp, command: str, *, timeout: float | None = None) -> tuple[str, int]:
    """Run *command* with ``bash -c`` as a subprocess of a cell of *interp*: its output (stdout and stderr in
    order) and its exit status."""
    limit = f", timeout={timeout!r}" if timeout is not None else ""
    code = (
        "import subprocess as _sp\n"
        f"_r = _sp.run(['bash', '-c', {command!r}], stdout=_sp.PIPE, stderr=_sp.STDOUT, text=True,"
        f" stdin=_sp.DEVNULL{limit})\n"
        "print(_r.stdout, end='')\n"
        "print('\\n__status__', _r.returncode)\n"
    )
    res = await interp.run_cell(code)
    assert res.error is None, res.text()
    out, _, status = res.output.rpartition("\n__status__ ")
    return out, int(status.strip())


def serve(reply: bytes) -> tuple[socket.socket, int]:
    """A host TCP listener on loopback that answers every connection."""
    srv = socket.socket()
    srv.bind(("127.0.0.1", 0))
    srv.listen(8)

    def loop():
        while True:
            try:
                conn, _ = srv.accept()
            except OSError:
                return
            conn.sendall(reply)
            conn.close()

    threading.Thread(target=loop, daemon=True).start()
    return srv, srv.getsockname()[1]


CONNECT = (
    "python3 -c \"import socket,sys; s=socket.create_connection(('{host}', {port}), "
    "timeout=5); print('got', s.recv(64).decode().strip())\" 2>&1 | tail -1"
)
