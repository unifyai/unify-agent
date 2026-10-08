"""A package install never runs with the harness's credentials or outside the sandbox.

A stored function's ``dependencies``, ``install(...)`` from a cell and
``%pip install`` all name packages the model chose, and the harness installs
them with ``uv`` into the workspace environment. Installs are binary-only,
so a package's build steps (an sdist's ``setup.py`` or build backend) never
run. The installer runs inside bubblewrap like a shell cell: credentials
hidden, ``/`` read-only, and only the environment and the installer's own
cache writable. Its only network is the harness's allow-listing proxy
(test_install_egress.py). It gets an explicit, minimal environment, never
the harness's.

No model is called and nothing is fetched: the first tests stub the
subprocess launcher and read the command line and environment the installer
gets; the last installs a local source package offline, whose build backend
would record what it can see and touch, and finds it never ran.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from tests.actor.code_act.sandbox_world import (  # noqa: F401 (fixture)
    ENV_SECRET,
    SSH_SECRET,
    STATE_SECRET,
    TOKEN_VALUE,
    needs_bwrap,
    world,
)
from unify import environment, sandbox

# Fake values with the names the harness's credentials have.
_CREDENTIALS = {
    "OPENROUTER_API_KEY": "sk-or-fake-0c7d",  # pragma: allowlist secret
    "ANTHROPIC_API_KEY": "sk-ant-fake-91aa",  # pragma: allowlist secret
    "GITHUB_TOKEN": "ghp-fake-5e21",  # pragma: allowlist secret
    "AWS_SECRET_ACCESS_KEY": "aws-fake-77b0",  # pragma: allowlist secret
}


def _secret_named(env: dict) -> list[str]:
    return sorted(k for k in env if sandbox.is_secret_name(k))


@pytest.fixture
def credentials(monkeypatch):
    for name, value in _CREDENTIALS.items():
        monkeypatch.setenv(name, value)
    monkeypatch.setattr(sys, "path", list(sys.path))
    return _CREDENTIALS


@pytest.fixture
def launched(monkeypatch):
    """Every command the environment module starts: ``(argv, env, cwd)``.

    ``uv venv`` is simulated by creating the interpreter path it would.
    """
    calls: list[tuple[list[str], dict | None, str | None]] = []

    def fake_run(argv, **kwargs):
        calls.append((list(argv), kwargs.get("env"), kwargs.get("cwd")))
        if "venv" in argv and "pip" not in argv:
            python = environment.environment_python()
            python.parent.mkdir(parents=True, exist_ok=True)
            python.touch()
        return subprocess.CompletedProcess(argv, 0, "", "")

    monkeypatch.setattr(environment.subprocess, "run", fake_run)
    return calls


def _install_call(calls):
    (call,) = [c for c in calls if "pip" in c[0]]
    return call


@needs_bwrap
def test_the_installer_gets_no_credentials_and_runs_in_the_sandbox(
    world,
    credentials,
    launched,
):
    environment.install(["humanize"])
    assert len(launched) == 2, launched  # uv venv, then uv pip install

    for argv, env, _ in launched:
        assert env is not None, f"{argv[:3]} inherited the harness's environment"
        assert _secret_named(env) == [], argv[:3]
        values = set(env.values())
        assert not values & set(credentials.values())
        assert TOKEN_VALUE not in values
        # Only what an install needs, not the harness's settings.
        assert "UNIFY_SANDBOX_PROBE" not in env
        assert "UNIFY_HOME" not in env

    argv, env, _ = _install_call(launched)
    venv = str(environment.environment_dir())
    policy = sandbox.build_policy()
    # /bin/sh opens the seccomp program for bwrap, then execs it.
    assert argv[0] == "/bin/sh" and Path(argv[4]).name == "bwrap", argv[:5]
    command = argv[argv.index("--") + 1 :]
    # The forwarder to the installer's proxy runs uv as its child.
    assert command[:4] == [sys.executable, "-I", "-S", "-c"], command[:4]
    assert command[7:] == [
        "uv",
        "pip",
        "install",
        "--no-config",
        "--only-binary",
        ":all:",
        "--python",
        str(environment.environment_python()),
        "humanize",
    ]
    options = argv[: argv.index("--")]
    # The state directory is hidden and the environment put back writable.
    assert ["--tmpfs", str(policy.state_dir)] == options[
        options.index("--tmpfs", options.index("--unshare-all")) :
    ][:2]
    binds = [options[i : i + 3] for i, a in enumerate(options) if a == "--bind"]
    cache = env["UV_CACHE_DIR"]
    assert sorted(b[1] for b in binds) == sorted(
        [str(policy.workspace), venv, cache],
    )
    assert all(b[1] == b[2] for b in binds)
    assert Path(cache).parent == policy.state_dir
    # Credential directories are masked as for a shell cell.
    for path, _rule in policy.masked_dirs:
        assert ["--tmpfs", str(path)] in [
            options[i : i + 2] for i, a in enumerate(options) if a == "--tmpfs"
        ]
    # No network of its own: only the proxy, through its forwarder.
    assert "--share-net" not in options
    assert env["HTTPS_PROXY"] == f"http://127.0.0.1:{sandbox.INSTALLER_PROXY_PORT}"
    assert "NO_PROXY" not in env and "no_proxy" not in env
    assert env["TMPDIR"] == "/tmp"


def test_an_install_takes_packages_never_installer_options(
    unify_home,
    launched,
):
    for specs in (
        ["--index-url", "https://example.invalid/simple", "humanize"],
        ["-r", "/etc/passwd"],
        ["humanize", "--target=/home"],
    ):
        with pytest.raises(ValueError, match="not an installer option"):
            environment.install(specs)
    assert launched == []


_BACKEND = """
import base64, hashlib, json, os, zipfile

REPORT = {report!r}
OUTSIDE = {outside!r}
SECRETS = {secrets!r}


def _probe():
    seen = {{"env": sorted(os.environ)}}
    try:
        with open(OUTSIDE, "w") as f:
            f.write("written by a build step")
        seen["wrote_outside"] = True
    except OSError as exc:
        seen["wrote_outside"] = False
    readable = []
    for path in SECRETS:
        try:
            with open(path) as f:
                readable.append(path + ": " + f.read(80))
        except OSError:
            pass
    seen["read_secrets"] = readable
    with open(REPORT, "w") as f:
        json.dump(seen, f)


def _line(name, data):
    digest = base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=")
    return f"{{name}},sha256={{digest.decode()}},{{len(data)}}"


def get_requires_for_build_wheel(config_settings=None):
    _probe()
    return []


def prepare_metadata_for_build_wheel(metadata_directory, config_settings=None):
    _probe()
    raise RuntimeError("probe: metadata hook ran")


def build_wheel(wheel_directory, config_settings=None, metadata_directory=None):
    _probe()
    name = "probe_pkg-0.1-py3-none-any.whl"
    files = {{
        "probe_pkg/__init__.py": b"VALUE = 1\\n",
        "probe_pkg-0.1.dist-info/METADATA": (
            b"Metadata-Version: 2.1\\nName: probe-pkg\\nVersion: 0.1\\n"
        ),
        "probe_pkg-0.1.dist-info/WHEEL": (
            b"Wheel-Version: 1.0\\nGenerator: probe\\nRoot-Is-Purelib: true\\n"
            b"Tag: py3-none-any\\n"
        ),
    }}
    record = [_line(n, d) for n, d in files.items()]
    record.append("probe_pkg-0.1.dist-info/RECORD,,")
    files["probe_pkg-0.1.dist-info/RECORD"] = ("\\n".join(record) + "\\n").encode()
    with zipfile.ZipFile(os.path.join(wheel_directory, name), "w") as z:
        for n, d in files.items():
            z.writestr(n, d)
    return name
"""


@needs_bwrap
@pytest.mark.timeout(180)
def test_a_source_package_is_refused_before_any_build_step_runs(
    world,
    credentials,
    monkeypatch,
):
    """A real offline install of a local source package whose build backend
    probes its world (its version is dynamic, so its metadata needs a hook).

    Installs are wheels only: uv refuses with its own message, and none of
    the backend's hooks runs. What does run at install time, uv itself, is
    confined (test_install_egress.py probes it from inside).
    """
    import shutil

    if shutil.which("uv") is None:
        pytest.skip("uv is not installed")
    monkeypatch.setenv("UV_OFFLINE", "1")
    source = world["workspace"] / "probe_pkg"
    source.mkdir()
    report = world["workspace"] / "build-report.json"
    outside = world["home"] / "written-by-a-build-step"
    secrets = [
        str(world["home"] / ".ssh" / "id_rsa"),
        str(world["home"] / ".env"),
        str(world["state"] / "logs" / "unify.log"),
    ]
    (source / "pyproject.toml").write_text(
        textwrap.dedent(
            """\
            [build-system]
            requires = []
            build-backend = "backend"
            backend-path = ["."]

            [project]
            name = "probe-pkg"
            dynamic = ["version"]
            """,
        ),
    )
    (source / "backend.py").write_text(
        _BACKEND.format(report=str(report), outside=str(outside), secrets=secrets),
    )

    for spec in (str(source), f"probe-pkg @ file://{source}"):
        outcome = environment.install([spec])
        assert not outcome["success"], outcome
        # uv's own message, which the model reads.
        assert "Building source distributions" in outcome["stderr"], outcome
        assert "disabled" in outcome["stderr"], outcome
    assert not report.exists(), report.read_text()
    assert not outside.exists()
    assert environment.missing(["probe-pkg"]) == ["probe-pkg"]
