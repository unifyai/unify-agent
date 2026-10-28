"""Symbolic: the workspace sandbox confines a cell and every subprocess it starts.

Nothing here reaches a model. Cells run in the actor's real interpreter (the
sandboxed worker, ``sandbox_world.actor_interpreter``) and the confinement is
the real bubblewrap; tests that need bubblewrap are skipped, saying so, where it
is missing. The world they run in is described in ``sandbox_world.py``.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.actor.code_act.sandbox_world import (  # noqa: F401 (fixture)
    CONNECT,
    ENV_SECRET,
    SSH_SECRET,
    STATE_SECRET,
    TOKEN_VALUE,
    actor_interpreter,
    needs_bwrap,
    serve,
    sh,
    world,
)
from unify import sandbox
from unify.common.tool_errors import ToolInputError
from unify.settings import ProductionSettings, SETTINGS

# ── settings ────────────────────────────────────────────────────────────────


def test_settings_default_no_network_and_reject_unknown_values():
    s = ProductionSettings()
    assert s.UNIFY_WORKSPACE_NETWORK == ""
    assert s.UNIFY_WORKSPACE_PROXY_PORT == 0
    with pytest.raises(ValueError):
        ProductionSettings(UNIFY_WORKSPACE_NETWORK="host")


# ── confinement ──────────────────────────────────────────────────────────────


@needs_bwrap
@pytest.mark.asyncio
async def test_writes_land_only_in_the_workspace_and_private_tmp(world):
    interp = await actor_interpreter()
    try:
        # The sandbox shows the workspace at a fixed path and starts there.
        out, _ = await sh(interp, "pwd")
        assert out.strip() == "/workspace"
        out, _ = await sh(interp, "echo made > made.txt && cat made.txt")
        assert out.strip() == "made"
        assert (world["workspace"] / "made.txt").read_text() == "made\n"
        out, status = await sh(interp, f"touch {world['home']}/escape.txt")
        assert not (world["home"] / "escape.txt").exists()
        assert status != 0
        # A file in the state directory is not mounted: nothing a cell does
        # reaches the host's file.
        store = world["state"] / "store.sqlite"
        out, _ = await sh(
            interp,
            "python3 -c \"import sqlite3; c=sqlite3.connect('file:"
            f"{store}?mode=ro', uri=True); "
            "print(c.execute('select name from functions').fetchall())\" 2>&1",
        )
        assert "('f',)" not in out and "unable to open database file" in out, out
        before = store.read_bytes()
        await sh(
            interp,
            f"python3 -c \"import sqlite3; c=sqlite3.connect('{store}'); "
            "c.execute('create table evil (x)'); c.commit()\"",
        )
        assert store.read_bytes() == before
        # The transcripts are readable, never writable (at their fixed path).
        out, _ = await sh(interp, "echo x >> /transcripts/s.jsonl")
        assert "Read-only file system" in out
        assert (world["state"] / "transcripts" / "s.jsonl").read_text() == (
            '{"seq": 0}\n'
        )
        out, _ = await sh(interp, "cat /transcripts/s.jsonl")
        assert out.strip() == '{"seq": 0}'
        # /tmp is writable and private.
        out, _ = await sh(interp, "echo t > /tmp/private.txt && cat /tmp/private.txt")
        assert out.strip() == "t" and not Path("/tmp/private.txt").exists()
    finally:
        await interp.stop()


@needs_bwrap
@pytest.mark.asyncio
async def test_transcripts_written_after_the_interpreter_started_are_searchable(world):
    """The pointer a compaction leaves (a session transcript) resolves inside the sandbox."""
    import shutil

    shutil.rmtree(world["state"] / "transcripts")
    interp = await actor_interpreter()
    try:
        await sh(interp, "true")
        later = world["state"] / "transcripts" / "later.jsonl"
        later.write_text('{"type": "message", "content": "the fact is 7"}\n')
        out, _ = await sh(interp, "grep -h 'fact is' /transcripts/*.jsonl")
        assert "the fact is 7" in out
    finally:
        await interp.stop()


@needs_bwrap
@pytest.mark.asyncio
async def test_masked_paths_read_as_a_notice_naming_the_rule(world):
    interp = await actor_interpreter()
    home, state = world["home"], world["state"]
    try:
        out, _ = await sh(interp, f"cat {home}/.ssh/id_rsa 2>&1; ls -A {home}/.ssh")
        assert SSH_SECRET not in out
        assert out.split()[-1] == sandbox.MASK_NOTICE_NAME
        out, _ = await sh(interp, f"cat {home}/.ssh/{sandbox.MASK_NOTICE_NAME}")
        assert "rule mask-credentials" in out
        # Outside the allowlisted root nothing of the home exists: not its
        # .env files, not its plain files.
        out, _ = await sh(interp, f"cat {home}/.env {home}/project/.env 2>&1")
        assert ENV_SECRET not in out and out.count("No such file") == 2
        out, _ = await sh(
            interp,
            f"cat {home}/plain.txt {home}/project/notes.txt 2>&1",
        )
        assert "readable outside the workspace" not in out
        assert "needle one" not in out and out.count("No such file") == 2
        out, _ = await sh(interp, f"cat {state}/logs/unify.log 2>&1")
        assert STATE_SECRET not in out
        # What may be seen of the state directory is shown at fixed paths,
        # never at its host path.
        out, _ = await sh(interp, f"ls -A {state}")
        assert set(out.split()) == {sandbox.MASK_NOTICE_NAME}
        out, _ = await sh(interp, f"cat {state}/{sandbox.MASK_NOTICE_NAME}")
        assert "rule mask-unify-state" in out
        out, _ = await sh(interp, f"grep -r {STATE_SECRET} {state} 2>/dev/null | wc -l")
        assert out.strip() == "0"
    finally:
        await interp.stop()


@needs_bwrap
@pytest.mark.asyncio
async def test_credential_variables_are_removed(world):
    interp = await actor_interpreter()
    try:
        out, _ = await sh(interp, "env")
        names = {line.split("=", 1)[0] for line in out.splitlines() if "=" in line}
        assert "UNIFY_SANDBOX_PROBE" in names
        assert not {"FAKE_SERVICE_TOKEN", "DB_PASSWORD"} & names
        assert TOKEN_VALUE not in out
        assert not any(sandbox.is_secret_name(n) for n in names)
        # The cell's own environment is the same scrubbed one.
        res = await interp.run_cell("__import__('os').environ.get('FAKE_SERVICE_TOKEN')")
        assert res.error is None and res.result is None
    finally:
        await interp.stop()


@needs_bwrap
@pytest.mark.asyncio
async def test_there_is_no_network_not_even_the_hosts_loopback(world):
    srv, port = serve(b"host service\n")
    interp = await actor_interpreter()
    try:
        out, _ = await sh(interp, CONNECT.format(host="127.0.0.1", port=port))
        assert "got host service" not in out and "ConnectionRefused" in out
        out, _ = await sh(
            interp,
            "python3 -c \"import socket; socket.create_connection(('192.0.2.1', 80), "
            'timeout=5)"',
        )
        assert "Network is unreachable" in out
    finally:
        srv.close()
        await interp.stop()


@needs_bwrap
@pytest.mark.asyncio
async def test_proxy_mode_reaches_only_the_proxy_port(world, monkeypatch):
    proxy, proxy_port = serve(b"from proxy\n")
    other, other_port = serve(b"other service\n")
    monkeypatch.setattr(SETTINGS, "UNIFY_WORKSPACE_NETWORK", "proxy")
    monkeypatch.setattr(SETTINGS, "UNIFY_WORKSPACE_PROXY_PORT", proxy_port)
    interp = await actor_interpreter()
    try:
        out, _ = await sh(interp, CONNECT.format(host="127.0.0.1", port=proxy_port))
        assert out.strip() == "got from proxy"
        out, _ = await sh(interp, CONNECT.format(host="127.0.0.1", port=other_port))
        assert "other service" not in out and "ConnectionRefused" in out
        out, _ = await sh(interp, 'echo "$HTTPS_PROXY"')
        assert out.strip() == f"http://127.0.0.1:{proxy_port}"
    finally:
        proxy.close()
        other.close()
        await interp.stop()


@needs_bwrap
@pytest.mark.asyncio
async def test_python_cell_subprocesses_run_in_the_sandbox(world):
    interp = await actor_interpreter()
    escape = world["home"] / "escape-from-python.txt"
    code = f"""
import asyncio, os, subprocess
r = subprocess.run(['touch', {str(escape)!r}], capture_output=True, text=True)
shell = subprocess.run('echo "[$FAKE_SERVICE_TOKEN]"; cat {world['home']}/.ssh/id_rsa',
                       shell=True, capture_output=True, text=True)
p = await asyncio.create_subprocess_exec('cat', {str(world['home'] / '.env')!r},
                                         stdout=asyncio.subprocess.PIPE)
env_out = (await p.communicate())[0].decode()
status = os.system('touch /workspace/inside-workspace.txt')
direct = os.environ.get('FAKE_SERVICE_TOKEN')
(r.returncode, r.stderr, shell.stdout, env_out, status, direct)
"""
    try:
        res = await interp.run_cell(code)
        assert res.error is None, res.text()
        code_rc, stderr, shell_out, env_out, status, direct = res.result
        assert code_rc != 0 and "Read-only file system" in stderr
        assert not escape.exists()
        assert shell_out.startswith("[]") and SSH_SECRET not in shell_out
        # The home's .env is outside the allowlisted root: absent.
        assert ENV_SECRET not in env_out and env_out == ""
        assert status == 0 and (world["workspace"] / "inside-workspace.txt").exists()
        assert direct is None
    finally:
        await interp.stop()


@pytest.mark.asyncio
async def test_without_bubblewrap_nothing_runs_unconfined(world, monkeypatch):
    monkeypatch.setattr(sandbox, "bwrap_path", lambda: None)
    with pytest.raises(sandbox.SandboxRefusal) as refused:
        await actor_interpreter()
    assert refused.value.rule == "sandbox-required"
    assert "rule `sandbox-required`" in str(refused.value)


def test_refusals_are_tool_input_errors_that_name_the_rule():
    err = sandbox.SandboxRefusal("network-off", "no route")
    assert isinstance(err, ToolInputError)
    assert err.as_tool_result().startswith(
        "Refused by workspace sandbox rule `network-off`",
    )
    assert set(sandbox.RULES) >= {
        "workspace-write",
        "mask-unify-state",
        "mask-credentials",
        "mask-env-file",
        "network-off",
        "network-proxy-only",
        "sandbox-required",
    }
