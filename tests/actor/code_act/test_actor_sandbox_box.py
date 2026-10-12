"""Symbolic: the actor's sandbox shows fixed paths, runs under resource limits
with an explicit environment, and is killed verifiably.

Nothing here reaches a model. The command lines are the real ``wrap_argv``'s;
the end-to-end test starts the real sandboxed worker (bubblewrap), runs a
cell, installs a local wheel offline with the real installer and calls the
cell's ``read_file`` and ``grep``.
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import io
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

from tests.actor.code_act.sandbox_world import (  # noqa: F401 (fixture)
    TOKEN_VALUE,
    needs_bwrap,
    world,
)
from unify import sandbox
from unify.common import process_tree
from unify.settings import ProductionSettings, SETTINGS

GIB = 1024**3


def _pairs(argv: list[str], option: str, width: int) -> list[list[str]]:
    end = argv.index("--")
    return [argv[i + 1 : i + 1 + width] for i in range(end) if argv[i] == option]


# ── the command line ─────────────────────────────────────────────────────────


@needs_bwrap
def test_a_neutral_sandbox_binds_the_workspace_at_a_fixed_path_and_starts_there(
    world,
):
    policy = sandbox.build_policy(fresh=True)
    workspace = str(Path(os.path.realpath(world["workspace"])))
    env = sandbox.sandbox_env(policy)
    argv = sandbox.wrap_argv(
        ["true"],
        policy,
        cwd=str(world["workspace"]),
        env=env,
        neutral=True,
    )
    binds = _pairs(argv, "--bind", 2)
    assert [workspace, "/workspace"] in binds
    # The host path is a bind source only, never a place in the sandbox.
    destinations = [
        b[1]
        for opt in ("--bind", "--ro-bind")
        for b in _pairs(argv, opt, 2)
    ]
    assert not any(d == workspace or d.startswith(workspace + "/") for d in destinations)
    transcripts = str(Path(os.path.realpath(world["state"] / "transcripts")))
    assert [transcripts, "/transcripts"] in _pairs(argv, "--ro-bind", 2)
    # The command starts in the workspace, as the sandbox shows it.
    assert argv[argv.index("--chdir") + 1] == "/workspace"
    assert ["PWD", "/workspace"] in _pairs(argv, "--setenv", 2)
    # Unchanged without neutral: the workspace at its own path.
    plain = sandbox.wrap_argv(["true"], policy, cwd=str(world["workspace"]))
    assert [workspace, workspace] in _pairs(plain, "--bind", 2)
    assert "--clearenv" not in plain[: plain.index("--")]


@needs_bwrap
def test_the_sandbox_gets_an_explicit_environment_and_resource_limits(
    world,
    monkeypatch,
):
    policy = sandbox.build_policy(fresh=True)
    env = sandbox.sandbox_env(policy)
    argv = sandbox.wrap_argv(["true"], policy, env=env, neutral=True)
    options = argv[: argv.index("--")]
    assert "--clearenv" in options
    setenv = dict(_pairs(argv, "--setenv", 2))
    assert setenv.pop("PWD") == "/workspace"
    assert setenv == env
    # The allow-list and its second filter: a declared probe passes, the
    # declared credentials do not, by name or by value.
    assert setenv["UNIFY_SANDBOX_PROBE"] == "visible"
    assert "FAKE_SERVICE_TOKEN" not in argv and "DB_PASSWORD" not in argv
    assert TOKEN_VALUE not in argv
    assert "UNIFY_HOME" not in setenv
    # A credential handed in explicitly is still dropped from the command line.
    planted = {**env, "PLANTED_API_KEY": "planted-key-for-the-env-test"}
    again = sandbox.wrap_argv(["true"], policy, env=planted, neutral=True)
    assert "PLANTED_API_KEY" not in again
    assert "planted-key-for-the-env-test" not in again
    # Inside, prlimit sets the limits before the command runs.
    inside = argv[argv.index("--") + 1 :]
    assert inside == [
        sandbox.PRLIMIT,
        f"--as={16 * GIB}:{16 * GIB}",
        "--core=0:0",
        f"--fsize={512 * 1024**2}:{512 * 1024**2}",
        "--nofile=1024:1024",
        "--nproc=256:256",
        "--",
        "true",
    ]
    monkeypatch.setattr(SETTINGS, "UNIFY_SANDBOX_ADDRESS_SPACE_BYTES", 8 * GIB)
    argv = sandbox.wrap_argv(["true"], policy, env=env, neutral=True)
    assert f"--as={8 * GIB}:{8 * GIB}" in argv


def test_the_address_space_limit_is_a_validated_setting():
    assert ProductionSettings().UNIFY_SANDBOX_ADDRESS_SPACE_BYTES == 16 * GIB
    assert (
        ProductionSettings(
            UNIFY_SANDBOX_ADDRESS_SPACE_BYTES="4294967296",
        ).UNIFY_SANDBOX_ADDRESS_SPACE_BYTES
        == 4 * GIB
    )
    for bad in ("0", "-1", "1.5", "4G", "", True):
        with pytest.raises(ValueError, match="positive whole number"):
            ProductionSettings(UNIFY_SANDBOX_ADDRESS_SPACE_BYTES=bad)


def test_paths_map_between_the_sandbox_and_the_host(world):
    policy = sandbox.build_policy(fresh=True)
    workspace = Path(os.path.realpath(world["workspace"]))
    assert sandbox.host_path("/workspace/data.txt", policy) == str(
        workspace / "data.txt",
    )
    assert sandbox.host_path("/workspace", policy) == str(workspace)
    # Leaving the mount leaves the mapping; the caller's checks then apply.
    assert sandbox.host_path("/workspace/../etc/hosts", policy) == "/etc/hosts"
    assert sandbox.host_path("data.txt", policy) == "data.txt"
    assert sandbox.box_path(workspace / "a" / "b.txt", policy) == "/workspace/a/b.txt"
    assert sandbox.box_path("/usr/bin/env", policy) == "/usr/bin/env"
    text = f"at {workspace}/x and {world['state']}/logs/unify.log"
    neutral = sandbox.neutral_text(text, policy)
    assert str(world["state"]) not in neutral and "/workspace/x" in neutral


# ── verified termination ─────────────────────────────────────────────────────


def test_the_actor_and_the_memory_boxes_share_one_kill():
    from unify.actor.execution import shell
    from unify.memory import sandbox_run

    assert sandbox_run._descendants is process_tree.descendants
    assert sandbox_run._alive is process_tree.alive
    assert not hasattr(shell, "_kill_group")


@pytest.mark.asyncio
async def test_the_kill_ends_a_process_group_and_every_descendant():
    proc = await asyncio.create_subprocess_exec(
        "/bin/sh",
        "-c",
        "sleep 60 & sleep 60",
        start_new_session=True,
    )
    await asyncio.sleep(0.3)
    children = process_tree.descendants(proc.pid)
    assert children
    from unify.actor.execution.shell import kill_sandbox

    kill_sandbox(proc)
    assert not any(process_tree.alive(pid, start) for pid, start in children)
    await asyncio.wait_for(proc.wait(), 10)


@pytest.mark.asyncio
async def test_a_survivor_makes_the_kill_raise(monkeypatch):
    from unify.actor.execution.shell import kill_sandbox

    proc = await asyncio.create_subprocess_exec(
        "sleep",
        "60",
        start_new_session=True,
    )
    survivor = (2_000_000_000, "stubbed")
    real_alive = process_tree.alive
    monkeypatch.setattr(process_tree, "KILL_GRACE_S", 0.3)
    monkeypatch.setattr(
        process_tree,
        "descendants",
        lambda pid: [survivor],
    )
    monkeypatch.setattr(
        process_tree,
        "alive",
        lambda pid, start: (pid, start) == survivor or real_alive(pid, start),
    )
    try:
        with pytest.raises(RuntimeError, match="survived SIGKILL"):
            kill_sandbox(proc)
    finally:
        try:
            os.killpg(proc.pid, 9)
        except ProcessLookupError:
            pass
        await asyncio.wait_for(proc.wait(), 10)


# ── the whole sandbox, end to end ────────────────────────────────────────────


def _wheel(directory: Path, name: str, version: str) -> Path:
    """A pure-Python wheel of one module, built by hand (no build step)."""
    files = {
        f"{name}/__init__.py": f"VALUE = {name!r}\n",
        f"{name}-{version}.dist-info/METADATA": (
            f"Metadata-Version: 2.1\nName: {name}\nVersion: {version}\n"
        ),
        f"{name}-{version}.dist-info/WHEEL": (
            "Wheel-Version: 1.0\nGenerator: hand\nRoot-Is-Purelib: true\n"
            "Tag: py3-none-any\n"
        ),
    }
    record = []
    for path, text in files.items():
        digest = hashlib.sha256(text.encode()).digest()
        b64 = base64.urlsafe_b64encode(digest).rstrip(b"=").decode()
        record.append(f"{path},sha256={b64},{len(text.encode())}")
    record.append(f"{name}-{version}.dist-info/RECORD,,")
    files[f"{name}-{version}.dist-info/RECORD"] = "\n".join(record) + "\n"
    target = directory / f"{name}-{version}-py3-none-any.whl"
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        for path, text in files.items():
            z.writestr(path, text)
    target.write_bytes(buf.getvalue())
    return target


_CELL = """
import importlib, os, sys
seen = {}
seen["cwd"] = os.getcwd()
seen["env"] = sorted(os.environ)
seen["values"] = sorted(os.environ.values())
outcome = await install(["/workspace/neutralprobe-0.1-py3-none-any.whl"])
seen["install"] = outcome
for key in [k for k, v in sys.path_importer_cache.items() if v is None]:
    del sys.path_importer_cache[key]
importlib.invalidate_caches()
import neutralprobe
seen["imported"] = [neutralprobe.VALUE, neutralprobe.__file__]
read = await read_file("/workspace/data.txt", 2, 2)
seen["read"] = [read["path"], read["content"]]
found = await grep("line 3", "/workspace")
seen["grep"] = found["hits"]
seen
"""


@needs_bwrap
@pytest.mark.asyncio
@pytest.mark.timeout(300)
async def test_the_actors_sandbox_runs_a_cell_installs_and_uses_its_tools(
    world,
    monkeypatch,
):
    """The real worker under the allow-list: a cell sees /workspace, only the
    allow-listed variables, installs a local wheel offline (the installer of
    test_install_confinement.py, UV_OFFLINE) and calls read_file and grep;
    no reply names the host's state directory."""
    if shutil.which("uv") is None:
        pytest.skip("uv is not installed")
    from unify import db
    from unify.actor import core_surface
    from unify.actor.code_act_actor import CodeActActor
    from unify.actor.execution import _CURRENT_SANDBOX, PythonExecutionSession
    from unify.actor.execution import worker

    assert worker.enabled(), "cells must run in the sandboxed worker"
    monkeypatch.setenv("UV_OFFLINE", "1")
    (world["state"] / "store.sqlite").unlink()
    db.reset_store()
    _wheel(world["workspace"], "neutralprobe", "0.1")
    actor = CodeActActor(environments=[])
    session = PythonExecutionSession(environments={})
    objects = core_surface.sandbox_objects(actor, policy=core_surface.WritePolicy())
    session.global_state.update(objects)
    session.core_globals = objects
    token = _CURRENT_SANDBOX.set(session)
    try:
        out = await actor.get_tools("act")["execute_code"].fn(
            thought="Probe the sandbox.",
            code=_CELL,
        )
    finally:
        _CURRENT_SANDBOX.reset(token)
        await session.close()
        await actor.close()
        db.reset_store()
    assert out.error is None, out.error
    seen = out.result
    assert seen["cwd"] == "/workspace"
    allowed = set(sandbox.CELL_ENV_NAMES) | {
        "PWD",
        "TMPDIR",
        "USER",
        "LOGNAME",
        "UNIFY_SANDBOX_PROBE",
        *sandbox.TRUSTED_ENV,
    }
    assert set(seen["env"]) <= allowed | {
        n for n in seen["env"] if n.startswith("LC_")
    }, seen["env"]
    assert "UNIFY_HOME" not in seen["env"] and "FAKE_SERVICE_TOKEN" not in seen["env"]
    assert not any(str(world["state"]) in v for v in seen["values"])
    assert seen["install"]["success"] is True, seen["install"]
    installer_text = seen["install"]["stdout"] + seen["install"]["stderr"]
    assert str(world["state"]) not in installer_text, installer_text
    name, location = seen["imported"]
    assert name == "neutralprobe" and location.startswith("/venv/"), location
    assert seen["read"] == ["/workspace/data.txt", "     2\tline 2\n"]
    assert seen["grep"] == ["/workspace/data.txt:3:line 3"]
    assert str(world["state"]) not in repr(seen)


def test_the_teammates_request_names_the_record_by_its_sandbox_path(tmp_path):
    from unify.agents.binding import helper_request

    path = tmp_path / "run-label-7q" / "records" / "x" / "record.jsonl"
    text = helper_request("h1", "root", "Count the rows of a.csv please.", 3, path)
    assert "/record/record.jsonl" in text
    assert "run-label-7q" not in text and str(tmp_path) not in text


@pytest.mark.skipif(sys.platform != "linux", reason="needs /proc")
def test_a_plain_process_is_killed_and_verified(tmp_path):
    proc = subprocess.Popen(["sleep", "60"], start_new_session=True)
    process_tree.kill_tree(proc.pid, exited=lambda: proc.poll() is not None)
    assert proc.poll() is not None
