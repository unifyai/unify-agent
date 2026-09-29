"""Symbolic: ``UNIFY_WORKSPACE=sandboxed`` runs shell cells and cell subprocesses confined.

Nothing here reaches a model. Cells run through the real ``SessionExecutor``
and the confinement is the real bubblewrap; tests that need bubblewrap are
skipped, saying so, where it is missing. The world they run in is described in
``sandbox_world.py``.
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

import pytest

from tests.actor.code_act.sandbox_world import (  # noqa: F401 (fixture)
    CONNECT,
    ENV_SECRET,
    SSH_SECRET,
    STATE_SECRET,
    TOKEN_VALUE,
    bash,
    needs_bwrap,
    serve,
    world,
)
from unify import sandbox
from unify.actor.execution.session import SessionExecutor
from unify.common.tool_errors import ToolInputError
from unify.settings import ProductionSettings, SETTINGS

# ── settings ────────────────────────────────────────────────────────────────


def test_settings_default_off_and_reject_unknown_values():
    s = ProductionSettings()
    assert s.UNIFY_WORKSPACE == "" and s.UNIFY_WORKSPACE_NETWORK == ""
    assert s.UNIFY_WORKSPACE_PROXY_PORT == 0
    assert ProductionSettings(UNIFY_WORKSPACE="Sandboxed").UNIFY_WORKSPACE == (
        "sandboxed"
    )
    with pytest.raises(ValueError):
        ProductionSettings(UNIFY_WORKSPACE="yolo")
    with pytest.raises(ValueError):
        ProductionSettings(UNIFY_WORKSPACE_NETWORK="host")


# ── off: exactly as shipped ─────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_off_python_cell_subprocesses_are_not_confined(monkeypatch, tmp_path):
    monkeypatch.setattr(SETTINGS, "UNIFY_WORKSPACE", "")
    # Even with the Popen wrapper installed by an earlier sandboxed cell, a
    # subprocess started outside a sandboxed cell runs as shipped.
    sandbox._install_popen_patch()
    outside = tmp_path / "written-by-subprocess"
    executor = SessionExecutor()
    res = await executor.execute(
        code=(
            "import subprocess, os\n"
            f"subprocess.run(['touch', {str(outside)!r}], check=True)\n"
            "os.environ.get('FAKE_SERVICE_TOKEN')"
        ),
        state_mode="stateless",
        session_id=None,
    )
    assert res["error"] is None and outside.exists()
    needs_switch = "UNIFY_WORKSPACE=sandboxed"  # pragma: allowlist secret
    with pytest.raises(ToolInputError, match=needs_switch):
        await executor.execute(
            code="echo hi",
            state_mode="stateful",
            session_id=0,
            language="bash",
        )
    await executor.close()


# ── bash sessions ────────────────────────────────────────────────────────────


@needs_bwrap
@pytest.mark.asyncio
async def test_bash_session_keeps_directory_variables_and_functions(world):
    ex = SessionExecutor()
    try:
        out, res = await bash(ex, "pwd")
        assert out.strip() == str(world["workspace"]) and res["session_created"]
        await bash(
            ex,
            "mkdir -p sub && cd sub && export COLOUR=teal && f() { echo f:$1; }",
        )
        out, res = await bash(ex, 'pwd; echo "$COLOUR"; f x')
        assert out.split() == [str(world["workspace"] / "sub"), "teal", "f:x"]
        assert res["result"] == 0 and res["error"] is None
        assert not res["session_created"]
        # Another session, and a stateless cell, share none of it.
        out, _ = await bash(ex, 'echo "[$COLOUR]"', session_id=1)
        assert out.strip() == "[]"
        out, _ = await bash(ex, 'echo "[$COLOUR]"; pwd', mode="stateless")
        assert out.split() == ["[]", str(world["workspace"])]
        # stdout and stderr interleave; a failing command reports its status;
        # a command reading stdin gets EOF instead of the protocol.
        out, res = await bash(ex, "echo out; echo err >&2; cat; false")
        assert out.split() == ["out", "err"]
        assert res["result"] == 1 and "status 1" in res["error"]
        out, _ = await bash(ex, "printf 'no newline'")
        assert out == "no newline"
        out, _ = await bash(ex, 'echo "$COLOUR"')
        assert out.strip() == "teal"
    finally:
        await ex.close()


@needs_bwrap
@pytest.mark.asyncio
async def test_a_timed_out_cell_kills_the_session_and_the_next_starts_fresh(world):
    ex = SessionExecutor(timeout=1)
    try:
        await bash(ex, "export KEEP=1")
        pid = ex._shell_sessions[0].pid
        out, res = await bash(ex, "echo started; sleep 30")
        assert "timed out after 1" in res["error"] and "started" in out
        assert ex._shell_sessions[0].pid is None
        with pytest.raises(ProcessLookupError):
            os.kill(pid, 0)
        out, res = await bash(ex, 'echo "[$KEEP]"')
        assert out.strip() == "[]" and res["error"] is None
        out, res = await bash(ex, "exit 3")
        assert res["result"] == 3 and "exited" in res["error"]
        out, res = await bash(ex, "echo back")
        assert out.strip() == "back"
    finally:
        await ex.close()


@needs_bwrap
@pytest.mark.asyncio
async def test_a_stopped_cell_kills_its_command(world):
    from tests.async_helpers import _wait_for_condition

    ex = SessionExecutor()
    marker = world["workspace"] / "started"
    try:
        await bash(ex, "export KEEP=1")
        pid = ex._shell_sessions[0].pid
        cell = asyncio.create_task(bash(ex, f"touch {marker}; sleep 30; echo late"))

        async def started():
            return marker.exists()

        async def gone():
            return not _alive(pid)

        await _wait_for_condition(started, poll=0.02, timeout=30)
        cell.cancel()
        with pytest.raises(asyncio.CancelledError):
            await cell
        await _wait_for_condition(gone, poll=0.02, timeout=10)
        out, res = await bash(ex, 'echo "[$KEEP]"')
        assert out.strip() == "[]" and res["error"] is None
    finally:
        await ex.close()


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    # A killed child not yet reaped by the event loop is a zombie.
    try:
        with open(f"/proc/{pid}/stat") as fh:
            return fh.read().split()[2] != "Z"
    except FileNotFoundError:
        return False


# ── confinement ──────────────────────────────────────────────────────────────


@needs_bwrap
@pytest.mark.asyncio
async def test_writes_land_only_in_the_workspace_and_private_tmp(world):
    ex = SessionExecutor()
    try:
        out, res = await bash(ex, "echo made > made.txt && cat made.txt")
        assert out.strip() == "made"
        assert (world["workspace"] / "made.txt").read_text() == "made\n"
        out, res = await bash(ex, f"touch {world['home']}/escape.txt")
        assert not (world["home"] / "escape.txt").exists()
        assert "Read-only file system" in out
        assert "rule `workspace-write`" in res["error"]
        # The store file and the transcripts are readable, never writable.
        store = world["state"] / "store.sqlite"
        out, res = await bash(
            ex,
            "python3 -c \"import sqlite3; c=sqlite3.connect('file:"
            f"{store}?mode=ro', uri=True); "
            "print(c.execute('select name from functions').fetchall())\"",
        )
        assert out.strip() == "[('f',)]"
        before = store.read_bytes()
        out, res = await bash(
            ex,
            f"python3 -c \"import sqlite3; c=sqlite3.connect('{store}'); "
            "c.execute('create table evil (x)'); c.commit()\"",
        )
        assert res["result"] != 0 and store.read_bytes() == before
        out, res = await bash(ex, f"echo x >> {world['state']}/transcripts/s.jsonl")
        assert "Read-only file system" in out
        assert (world["state"] / "transcripts" / "s.jsonl").read_text() == (
            '{"seq": 0}\n'
        )
        out, _ = await bash(ex, f"cat {world['state']}/transcripts/s.jsonl")
        assert out.strip() == '{"seq": 0}'
        # /tmp is writable and private.
        out, _ = await bash(ex, "echo t > /tmp/private.txt && cat /tmp/private.txt")
        assert out.strip() == "t" and not Path("/tmp/private.txt").exists()
    finally:
        await ex.close()


@needs_bwrap
@pytest.mark.asyncio
async def test_transcripts_written_after_the_session_started_are_searchable(
    world,
    monkeypatch,
):
    """The pointer a compaction leaves (UNIFY_TRANSCRIPTS) resolves inside the sandbox."""
    import shutil

    shutil.rmtree(world["state"] / "transcripts")
    monkeypatch.setattr(SETTINGS, "UNIFY_TRANSCRIPTS", True)
    ex = SessionExecutor()
    try:
        await bash(ex, "true")
        later = world["state"] / "transcripts" / "later.jsonl"
        later.write_text('{"type": "message", "content": "the fact is 7"}\n')
        out, _ = await bash(ex, f"grep -h 'fact is' {later.parent}/*.jsonl")
        assert "the fact is 7" in out
    finally:
        await ex.close()


@needs_bwrap
@pytest.mark.asyncio
async def test_masked_paths_read_as_a_notice_naming_the_rule(world):
    ex = SessionExecutor()
    home, state = world["home"], world["state"]
    try:
        out, _ = await bash(ex, f"cat {home}/.ssh/id_rsa 2>&1; ls -A {home}/.ssh")
        assert SSH_SECRET not in out
        assert out.split()[-1] == sandbox.MASK_NOTICE_NAME
        out, _ = await bash(ex, f"cat {home}/.ssh/{sandbox.MASK_NOTICE_NAME}")
        assert "rule mask-credentials" in out
        out, _ = await bash(ex, f"cat {home}/.env {home}/project/.env")
        assert ENV_SECRET not in out and out.count("rule mask-env-file") == 2
        out, _ = await bash(ex, f"cat {home}/plain.txt {home}/project/notes.txt")
        assert "readable outside the workspace" in out and "needle one" in out
        out, _ = await bash(ex, f"cat {state}/logs/unify.log 2>&1")
        assert STATE_SECRET not in out
        out, _ = await bash(ex, f"ls -A {state}")
        assert set(out.split()) == {
            sandbox.MASK_NOTICE_NAME,
            "store.sqlite",
            "transcripts",
            "workspace",
        }
        out, _ = await bash(ex, f"cat {state}/{sandbox.MASK_NOTICE_NAME}")
        assert "rule mask-unify-state" in out
        out, _ = await bash(ex, f"grep -r {STATE_SECRET} {state} 2>/dev/null | wc -l")
        assert out.strip() == "0"
    finally:
        await ex.close()


@needs_bwrap
@pytest.mark.asyncio
async def test_credential_variables_are_removed(world):
    ex = SessionExecutor()
    try:
        out, _ = await bash(ex, "env")
        names = {line.split("=", 1)[0] for line in out.splitlines() if "=" in line}
        assert "UNIFY_SANDBOX_PROBE" in names
        assert not {"FAKE_SERVICE_TOKEN", "DB_PASSWORD"} & names
        assert TOKEN_VALUE not in out
        assert not any(sandbox.is_secret_name(n) for n in names)
    finally:
        await ex.close()


@needs_bwrap
@pytest.mark.asyncio
async def test_there_is_no_network_not_even_the_hosts_loopback(world):
    srv, port = serve(b"host service\n")
    ex = SessionExecutor()
    try:
        out, _ = await bash(ex, CONNECT.format(host="127.0.0.1", port=port))
        assert "got host service" not in out and "ConnectionRefused" in out
        out, res = await bash(
            ex,
            "python3 -c \"import socket; socket.create_connection(('192.0.2.1', 80), "
            'timeout=5)"',
        )
        assert "Network is unreachable" in out
        assert "rule `network-off`" in res["error"]
    finally:
        srv.close()
        await ex.close()


@needs_bwrap
@pytest.mark.asyncio
async def test_proxy_mode_reaches_only_the_proxy_port(world, monkeypatch):
    proxy, proxy_port = serve(b"from proxy\n")
    other, other_port = serve(b"other service\n")
    monkeypatch.setattr(SETTINGS, "UNIFY_WORKSPACE_NETWORK", "proxy")
    monkeypatch.setattr(SETTINGS, "UNIFY_WORKSPACE_PROXY_PORT", proxy_port)
    ex = SessionExecutor()
    try:
        out, _ = await bash(ex, CONNECT.format(host="127.0.0.1", port=proxy_port))
        assert out.strip() == "got from proxy"
        out, _ = await bash(ex, CONNECT.format(host="127.0.0.1", port=other_port))
        assert "other service" not in out and "ConnectionRefused" in out
        out, _ = await bash(ex, 'echo "$HTTPS_PROXY"')
        assert out.strip() == f"http://127.0.0.1:{proxy_port}"
    finally:
        proxy.close()
        other.close()
        await ex.close()


@needs_bwrap
@pytest.mark.asyncio
async def test_python_cell_subprocesses_run_in_the_sandbox(world):
    ex = SessionExecutor()
    escape = world["home"] / "escape-from-python.txt"
    code = f"""
import asyncio, os, subprocess
r = subprocess.run(['touch', {str(escape)!r}], capture_output=True, text=True)
shell = subprocess.run('echo "[$FAKE_SERVICE_TOKEN]"; cat {world['home']}/.ssh/id_rsa',
                       shell=True, capture_output=True, text=True)
p = await asyncio.create_subprocess_exec('cat', {str(world['home'] / '.env')!r},
                                         stdout=asyncio.subprocess.PIPE)
env_out = (await p.communicate())[0].decode()
status = os.system('touch {world['workspace']}/inside-workspace.txt')
direct = os.environ.get('FAKE_SERVICE_TOKEN')
(r.returncode, r.stderr, shell.stdout, env_out, status, direct)
"""
    try:
        res = await ex.execute(code=code, state_mode="stateless", session_id=None)
        assert res["error"] is None, res["error"]
        code_rc, stderr, shell_out, env_out, status, direct = res["result"]
        assert code_rc != 0 and "Read-only file system" in stderr
        assert not escape.exists()
        assert shell_out.startswith("[]") and SSH_SECRET not in shell_out
        assert ENV_SECRET not in env_out and "mask-env-file" in env_out
        assert status == 0 and (world["workspace"] / "inside-workspace.txt").exists()
        # The documented gap: the cell itself runs in the harness process.
        assert direct == TOKEN_VALUE
    finally:
        await ex.close()


@pytest.mark.asyncio
async def test_without_bubblewrap_nothing_runs_unconfined(world, monkeypatch, tmp_path):
    monkeypatch.setattr(sandbox, "bwrap_path", lambda: None)
    ex = SessionExecutor()
    try:
        with pytest.raises(sandbox.SandboxRefusal) as refused:
            await bash(ex, "echo hi")
        assert refused.value.rule == "sandbox-required"
        assert "rule `sandbox-required`" in str(refused.value)
        outside = tmp_path / "must-not-exist"
        res = await ex.execute(
            code=f"import subprocess\nsubprocess.run(['touch', {str(outside)!r}])",
            state_mode="stateless",
            session_id=None,
        )
        assert "sandbox-required" in res["error"] and not outside.exists()
    finally:
        await ex.close()


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


def test_the_harness_installs_packages_unconfined(world, monkeypatch, tmp_path):
    """Dependency installs started from inside a sandboxed cell are the harness's own."""
    import subprocess

    from unify import environment

    monkeypatch.setattr(sys, "path", list(sys.path))
    confinement_seen = []

    def fake_run(argv, **kwargs):
        confinement_seen.append(sandbox._CONFINE.get())
        return subprocess.CompletedProcess(argv, 0, "", "")

    policy = sandbox.build_policy(fresh=True)
    outside = tmp_path / "unconfined-touch"
    with sandbox.confined_subprocesses(policy):
        with sandbox.unconfined():
            subprocess.run(["touch", str(outside)], check=True)
        monkeypatch.setattr(environment.subprocess, "run", fake_run)
        environment.install(["humanize"])
    assert outside.exists()
    # ``uv venv`` then ``uv pip install``, neither under the sandbox.
    assert confinement_seen == [None, None]
