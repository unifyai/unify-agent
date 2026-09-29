"""The world the workspace-sandbox tests run in: a home and ``UNIFY_HOME`` to hide.

Both are placed outside ``/tmp`` on purpose: the sandbox's ``/tmp`` is private,
so a secret under the host's ``/tmp`` would be invisible whether or not its
mask worked.
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
from unify.actor.execution.types import parts_to_text
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
    monkeypatch.delenv("UNIFY_STORE_PATH", raising=False)
    monkeypatch.setenv("FAKE_SERVICE_TOKEN", TOKEN_VALUE)
    monkeypatch.setenv("DB_PASSWORD", TOKEN_VALUE)
    monkeypatch.setenv("UNIFY_SANDBOX_PROBE", "visible")
    monkeypatch.setattr(SETTINGS, "UNIFY_LOCAL_ROOT", "")
    monkeypatch.setattr(SETTINGS, "UNIFY_WORKSPACE", "sandboxed")
    monkeypatch.setattr(SETTINGS, "UNIFY_WORKSPACE_NETWORK", "")
    monkeypatch.setattr(SETTINGS, "UNIFY_WORKSPACE_PROXY_PORT", 0)
    monkeypatch.setattr(sandbox, "_POLICY_CACHE", None)
    yield {"home": home, "state": state, "workspace": state / "workspace"}
    shutil.rmtree(root, ignore_errors=True)


async def bash(executor, code, *, mode="stateful", session_id=0):
    res = await executor.execute(
        code=code,
        state_mode=mode,
        session_id=session_id if mode != "stateless" else None,
        language="bash",
    )
    return parts_to_text(res["stdout"]), res


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
