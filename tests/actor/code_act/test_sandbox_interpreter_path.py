"""Symbolic: ``python`` and ``python3`` in a cell's subprocess are the cell's interpreter.

A cell that ran ``subprocess.run(["python", ...])`` failed with
``FileNotFoundError: 'python'`` on a host whose ``PATH`` has no ``python``
(w129: only ``~/.local/bin/python3``), because the sandbox passed the host's
``PATH`` on as it was. ``sandbox_env`` now puts the directory of the
interpreter cells run with first on ``PATH``. That directory is already
mounted (the interpreter chain), so no mount changes.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

from tests.actor.code_act.sandbox_world import (  # noqa: F401 (fixture)
    needs_bwrap,
    world,
)
from unify import sandbox
from unify.actor.execution.session import SessionExecutor


def _host_path_without_interpreter() -> str:
    dirs = set(sandbox.interpreter_bin_dirs())
    return os.pathsep.join(
        entry
        for entry in os.environ.get("PATH", "").split(os.pathsep)
        if entry and entry not in dirs
    )


def test_the_interpreter_directory_comes_first_and_has_both_names():
    dirs = sandbox.interpreter_bin_dirs()
    assert dirs and dirs[0] == os.path.dirname(os.path.abspath(sys.executable))
    for name in ("python", "python3"):
        assert any(os.access(os.path.join(d, name), os.X_OK) for d in dirs), name
    policy = sandbox.SandboxPolicy(workspace=Path("/w"), state_dir=Path("/s"))
    env = sandbox.sandbox_env(policy, None)
    entries = env["PATH"].split(os.pathsep)
    assert entries[: len(dirs)] == dirs
    # Each entry once, the host's after them in their order.
    assert len(entries) == len(set(entries))
    host = [e for e in os.environ.get("PATH", "").split(os.pathsep) if e]
    assert [e for e in entries if e not in dirs] == [
        e for e in dict.fromkeys(host) if e not in dirs
    ]


def test_an_explicit_env_keeps_its_path():
    policy = sandbox.SandboxPolicy(workspace=Path("/w"), state_dir=Path("/s"))
    assert sandbox.sandbox_env(policy, {"PATH": "/opt/x"})["PATH"] == "/opt/x"
    assert "PATH" not in sandbox.sandbox_env(policy, {"OTHER": "1"})


@needs_bwrap
def test_the_mounts_are_the_same_with_or_without_the_path_change(world, monkeypatch):
    """The PATH change is in the environment only: bubblewrap's command line,
    and so every mount, is what it was without it, the environment differs in
    PATH alone, and the directories it adds are inside the root already shown."""
    dirs = sandbox.interpreter_bin_dirs()
    policy = sandbox.build_policy(fresh=True)
    argv = ["python", "-c", "print(1)"]
    after = (sandbox.wrap_argv(argv, policy), sandbox.sandbox_env(policy))
    # As before the change: nothing put first on PATH.
    monkeypatch.setattr(sandbox, "interpreter_bin_dirs", lambda: [])
    before = (sandbox.wrap_argv(argv, policy), sandbox.sandbox_env(policy))
    assert after[0] == before[0]
    changed = {
        k
        for k in after[1].keys() | before[1].keys()
        if after[1].get(k) != before[1].get(k)
    }
    assert changed <= {"PATH"}
    assert after[1]["PATH"].split(os.pathsep)[: len(dirs)] == dirs
    for d in dirs:
        real = Path(os.path.realpath(d))
        assert any(real.is_relative_to(v) for v in policy.root_visible), d


CELL = """
import shutil, subprocess
runs = {
    name: subprocess.run(
        [name, "-c", "import sys; print(sys.version_info[:2])"],
        capture_output=True,
        text=True,
    )
    for name in ("python", "python3")
}
out = {
    name: [r.returncode, r.stdout.strip(), shutil.which(name)]
    for name, r in runs.items()
}
out
"""


@needs_bwrap
@pytest.mark.asyncio
async def test_python_and_python3_run_in_a_cell_without_them_on_the_host_path(
    world,
    monkeypatch,
):
    monkeypatch.setenv("PATH", _host_path_without_interpreter())
    ex = SessionExecutor()
    try:
        res = await ex.execute(code=CELL, state_mode="stateless", session_id=None)
    finally:
        await ex.close()
    assert res["error"] is None, res["error"]
    policy = sandbox.build_policy()
    version = str(tuple(sys.version_info[:2]))
    dirs = sandbox.interpreter_bin_dirs()
    for name in ("python", "python3"):
        code, stdout, found = res["result"][name]
        assert (code, stdout) == (0, version), res["result"]
        # Found in the interpreter's directory, which an existing mount shows.
        assert os.path.dirname(found) in dirs, found
        real = Path(os.path.realpath(found))
        assert any(real.is_relative_to(v) for v in policy.root_visible), found
