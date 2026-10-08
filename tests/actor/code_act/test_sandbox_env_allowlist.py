"""A sandboxed command's environment is built from an allow-list.

On 7 Oct 2026 a model-written cell enumerated ``os.environ`` and printed the
harness's provider key. Cells now run in the sandboxed worker, whose
environment was the harness's minus a deny-list of credential names; a key
under a name no marker catches (``OR_PROVIDER``) still reached the cell. The
environment is now built from an allow-list (``unify/sandbox.py``,
``CELL_ENV_NAMES``, ``CELL_ENV_PATTERNS`` and ``UNIFY_CELL_ENV_ALLOW``), with
the deny-list (a credential's name, a key-shaped value, a URL with a password)
kept as a second filter. Every value here is fake.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from tests.actor.code_act.sandbox_world import (  # noqa: F401 (fixture)
    bash,
    needs_bwrap,
    world,
)
from unify import sandbox
from unify.actor.execution.session import SessionExecutor
from unify.settings import SETTINGS, ProductionSettings

FAKE_KEY = "fake-provider-key-0c4d9e7b1a"  # pragma: allowlist secret
FAKE_SK = "sk-or-v1-FAKE00000000000000000000"  # pragma: allowlist secret
FAKE_GOOGLE = "AIzaFAKE000000000000000000000000000"  # pragma: allowlist secret


HARNESS_ENV = {
    "OPENROUTER_API_KEY": FAKE_KEY,
    # No credential marker in the name: only the allow-list keeps it out.
    "OR_PROVIDER": FAKE_KEY,
    "GOOGLE_APPLICATION_CREDENTIALS": "/home/someone/fake-service-account.json",
    "LITELLM_MASTER": FAKE_SK,
    "PATH": "/usr/bin:/bin",
    "LANG": "C.UTF-8",
    "APPWORLD_RELAY_SOCKET": "/tmp/cwaw-x/relay.sock",
    "UNIFY_CELL_ENV_ALLOW": "FOO_*",
    "FOO_BAR": "1",
    "OTHER_SETTING": "not declared",
}


def _policy() -> sandbox.SandboxPolicy:
    return sandbox.SandboxPolicy(workspace=Path("/w"), state_dir=Path("/s"))


def _generated() -> set[str]:
    return {"TMPDIR"} | ({"USER", "LOGNAME"} if sandbox._account_name() else set())


def _on_the_list(name: str) -> bool:
    patterns = sandbox._cell_env_patterns()
    return name in sandbox.CELL_ENV_NAMES or any(
        sandbox._env_pattern_matches(p, name) for p in patterns
    )


def test_the_allow_list_keeps_exactly_the_allowed_variables(monkeypatch):
    monkeypatch.setattr(SETTINGS, "UNIFY_CELL_ENV_ALLOW", "FOO_*")
    assert sandbox.allowed_env(HARNESS_ENV) == {
        "PATH": "/usr/bin:/bin",
        "LANG": "C.UTF-8",
        "APPWORLD_RELAY_SOCKET": "/tmp/cwaw-x/relay.sock",
        "FOO_BAR": "1",
    }


def test_a_sandboxed_command_gets_no_harness_secret(monkeypatch):
    monkeypatch.setattr(SETTINGS, "UNIFY_CELL_ENV_ALLOW", "FOO_*")
    for name, value in HARNESS_ENV.items():
        monkeypatch.setenv(name, value)
    env = sandbox.sandbox_env(_policy())
    assert {"LANG", "APPWORLD_RELAY_SOCKET", "FOO_BAR"} <= set(env)
    assert env["PATH"].split(os.pathsep)[-2:] == ["/usr/bin", "/bin"]
    for name in (
        "OPENROUTER_API_KEY",
        "OR_PROVIDER",
        "GOOGLE_APPLICATION_CREDENTIALS",
        "LITELLM_MASTER",
        "UNIFY_CELL_ENV_ALLOW",
        "OTHER_SETTING",
    ):
        assert name not in env, name
    joined = "\n".join(f"{k}={v}" for k, v in env.items())
    assert FAKE_KEY not in joined and FAKE_SK not in joined
    assert "fake-service-account" not in joined
    # Whatever else the harness's environment holds, only the list passes.
    assert [
        n
        for n in env
        if not _on_the_list(n) and n not in _generated() | set(sandbox.TRUSTED_ENV)
    ] == []


def test_the_default_allow_list_keeps_the_runtime_and_relay_variables(monkeypatch):
    monkeypatch.setattr(SETTINGS, "UNIFY_CELL_ENV_ALLOW", "")
    kept = {
        "PATH": "/usr/bin",
        "HOME": "/home/someone",
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
        "TERM": "xterm",
        "PYTHONPATH": "/opt/client",
        "PYTHONUNBUFFERED": "1",
        "PYTHONIOENCODING": "utf-8",
        "PYTHONDONTWRITEBYTECODE": "1",
        "SCIENCEWORLD_RELAY_SOCKET": "/r/relay.sock",
        "APPWORLD_VERIFY_SOCKET": "/v/verify.sock",
        "APPWORLD_RELAY_TIMEOUT": "180",
        "UNIFY_ENV_NAMESPACES": "appworld_unify_env:surface",
    }
    assert (
        sandbox.allowed_env({**kept, "VIRTUAL_ENV": "/v", "UNIFY_HOME": "/u"}) == kept
    )
    # The fake clock's variables come from the harness as it has them.
    monkeypatch.setenv("TZ", "UTC")
    monkeypatch.setenv("FAKETIME", "@2024-01-02 03:04:05")
    env = sandbox.sandbox_env(_policy())
    assert env["TZ"] == "UTC" and env["FAKETIME"] == "@2024-01-02 03:04:05"
    assert env["TMPDIR"] == "/tmp"


def test_an_allow_listed_credential_is_still_left_out(monkeypatch):
    monkeypatch.setattr(SETTINGS, "UNIFY_CELL_ENV_ALLOW", "FOO_*,ANTHROPIC_API_KEY")
    env = {
        "FOO_TOKEN": "x",
        "FOO_AUTH": "x",
        "FOO_DB": "postgres://user:pw@db.internal/app",  # pragma: allowlist secret
        "FOO_VALUE": FAKE_SK,
        "FOO_GOOGLE": FAKE_GOOGLE,
        "FOO_PEM": "-----BEGIN RSA PRIVATE KEY-----\nAAAA",  # pragma: allowlist secret
        "ANTHROPIC_API_KEY": FAKE_KEY,
        "FOO_OK": "plain",
        "FOO_PATHLIKE": "/opt/sk-tools/bin",
    }
    assert set(sandbox.allowed_env(env)) == {"FOO_OK", "FOO_PATHLIKE"}


@pytest.mark.parametrize(
    "name",
    [
        "OPENROUTER_API_KEY",
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
        "GEMINI_API_KEY",
        "GOOGLE_API_KEY",
        "AZURE_OPENAI_API_KEY",
        "HF_TOKEN",
        "VAST_API_KEY",
        "UNIFY_KEY",
        "AWS_SECRET_ACCESS_KEY",
        "GITHUB_TOKEN",
    ],
)
def test_common_provider_key_names_never_pass_even_when_declared(monkeypatch, name):
    monkeypatch.setattr(SETTINGS, "UNIFY_CELL_ENV_ALLOW", name)
    monkeypatch.setenv(name, FAKE_KEY)
    assert sandbox.allowed_env({name: FAKE_KEY}) == {}
    assert name not in sandbox.sandbox_env(_policy())


def test_a_cells_explicit_env_is_kept_without_credentials():
    """A cell's own ``env=`` (already inside the sandbox) is unchanged: kept as
    given, without credential names."""
    env = sandbox.sandbox_env(
        _policy(),
        {"PATH": "/opt/x", "MY_SETTING": "1", "OPENAI_API_KEY": FAKE_KEY},
    )
    assert env["PATH"] == "/opt/x" and env["MY_SETTING"] == "1"
    assert "OPENAI_API_KEY" not in env


def test_the_setting_takes_names_and_prefix_patterns_only():
    assert ProductionSettings.model_fields["UNIFY_CELL_ENV_ALLOW"].default == ""
    s = ProductionSettings(UNIFY_CELL_ENV_ALLOW=" FOO_*, BAR ,")
    assert s.UNIFY_CELL_ENV_ALLOW == "FOO_*,BAR"
    for bad in ("*", "FOO BAR", "A-B", "*_KEY", "FOO*BAR"):
        with pytest.raises(ValueError):
            ProductionSettings(UNIFY_CELL_ENV_ALLOW=bad)


# ── in the sandbox ───────────────────────────────────────────────────────────

# Everything a cell can enumerate of process environments: its own, through
# os.environ and /proc/self/environ, pid 1's, and every /proc/<pid>/environ it
# can read (its parent's among them). The fake key is not in the cell's code.
PROBE = r"""
import os
def read(path):
    try:
        with open(path, "rb") as fh:
            return fh.read().decode("utf-8", "replace")
    except OSError as exc:
        return "<" + type(exc).__name__ + ">"
pids = sorted(int(d) for d in os.listdir("/proc") if d.isdigit())
dump = {
    "os.environ": "\n".join(f"{k}={v}" for k, v in os.environ.items()),
    "/proc/self/environ": read("/proc/self/environ"),
    "/proc/1/environ": read("/proc/1/environ"),
}
for pid in pids:
    dump[f"/proc/{pid}/environ"] = read(f"/proc/{pid}/environ")
(pids, dump, sorted(os.environ), read("/proc/1/cmdline").split("\0")[0], os.getppid())
"""


@needs_bwrap
@pytest.mark.asyncio
async def test_no_environment_a_cell_can_read_holds_the_harnesss_key(
    world,
    monkeypatch,
):
    for name in ("OPENROUTER_API_KEY", "ANTHROPIC_API_KEY", "OR_PROVIDER"):
        monkeypatch.setenv(name, FAKE_KEY)
    ex = SessionExecutor()
    try:
        res = await ex.execute(code=PROBE, state_mode="stateless", session_id=None)
        assert res["error"] is None, res["error"]
        pids, dump, names, pid1, ppid = res["result"]
        assert FAKE_KEY not in repr(res)
        for where, text in dump.items():
            assert FAKE_KEY not in text, where
        # The walk read real environments: the cell's own has PATH.
        assert "PATH=" in dump["/proc/self/environ"]
        # Only the allow-list (and what the sandbox sets) is there.
        allowed = set(sandbox.TRUSTED_ENV) | _generated()
        assert [n for n in names if not _on_the_list(n) and n not in allowed] == []
        assert "UNIFY_SANDBOX_PROBE" in names  # declared by the world
        # A fresh pid namespace: pid 1 is bubblewrap's init, the cell's parent
        # is inside it, and only the sandbox's few processes are listed (the
        # harness, a host pid, is not).
        assert 1 in pids and ppid in pids and len(pids) <= 10, pids
        assert all(p < 100 for p in pids), pids
        assert os.path.basename(pid1) == "bwrap", pid1
        # The bash cell: env, both environ files and the process list.
        out, _ = await bash(
            ex,
            "env; echo ---; tr '\\0' '\\n' < /proc/self/environ; echo ---; "
            "tr '\\0' '\\n' < /proc/1/environ 2>&1; echo ---; "
            "for f in /proc/[0-9]*/environ; do tr '\\0' '\\n' < $f 2>/dev/null; done; "
            "echo ---; ls /proc | grep -E '^[0-9]+$' | tr '\\n' ' '",
        )
        assert FAKE_KEY not in out
        assert "PATH=" in out
        bash_pids = [int(p) for p in out.rsplit("---", 1)[1].split()]
        assert 1 in bash_pids and all(p < 100 for p in bash_pids), bash_pids
    finally:
        await ex.close()
