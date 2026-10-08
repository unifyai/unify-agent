"""Symbolic: the sandbox's root is an allowlist, and sockets are limited to three families.

``unify/sandbox.py`` used to mount the host's ``/`` read-only and hide a
deny-list (credential directories, ``.env`` files found by a scan). A
deny-list over arbitrary host files cannot be complete: ``/workspaces/envs``,
``/root`` or ``/var/log`` stayed readable. Now only ``_ROOT_ALLOWLIST`` and
the paths derived from the interpreter exist, and a seccomp filter refuses
every socket family but AF_UNIX, AF_INET and AF_INET6 (AF_VSOCK reaches the
Windows host from WSL2 whatever the network namespace).

These tests check access and socket creation only: they never read a real
credential file (the secret-like files are ones the test creates) and never
connect to a host service.
"""

from __future__ import annotations

import errno
import os
import struct
from pathlib import Path

import pytest

from tests.actor.code_act.sandbox_world import (  # noqa: F401 (fixture)
    ENV_SECRET,
    bash,
    needs_bwrap,
    world,
)
from unify import sandbox
from unify.actor.execution.session import SessionExecutor
from unify.settings import SETTINGS

BIND_OPTIONS = ("--bind", "--ro-bind", "--dev-bind", "--bind-try", "--ro-bind-try")


def _binds(argv: list[str]) -> list[tuple[str, str, str]]:
    """Every ``(option, source, destination)`` bind before the command."""
    end = argv.index("--")
    return [
        (argv[i], argv[i + 1], argv[i + 2])
        for i in range(end - 2)
        if argv[i] in BIND_OPTIONS
    ]


def _homes() -> list[Path]:
    import pwd

    out = [Path("/home"), Path(pwd.getpwuid(os.getuid()).pw_dir), Path.home()]
    return [
        Path(os.path.realpath(p))
        for p in out
        if not Path(os.path.realpath(p)).is_relative_to("/tmp")
    ]


def _assert_no_broad_bind(argv: list[str]) -> None:
    assert " ".join(argv).find("--ro-bind / /") == -1, argv
    homes = _homes()
    for option, src, dst in _binds(argv):
        for p in (src, dst):
            real = Path(os.path.realpath(p))
            assert real != Path("/") and Path(p) != Path("/"), (option, src, dst)
            for home in homes:
                assert not home.is_relative_to(real), (option, src, dst, home)
                assert not home.is_relative_to(Path(p)), (option, src, dst, home)


# ── the guard: never / and never a whole home ───────────────────────────────


@needs_bwrap
def test_no_command_line_ever_binds_root_or_a_whole_home(world, monkeypatch):
    from unify import environment

    policy = sandbox.build_policy(fresh=True)
    _assert_no_broad_bind(sandbox.wrap_argv(["true"], policy))
    # The installer's command line: writable venv and cache, uv read-only,
    # the host's network.
    _assert_no_broad_bind(
        sandbox.wrap_argv(
            ["uv", "--version"],
            policy,
            writable=[environment.environment_dir(), environment.installer_cache()],
            readonly=[Path("/usr/bin/env")],
            share_network=True,
        ),
    )
    # A derived root that would show a whole home is left out, not mounted:
    # here PYTHONPATH names the account's home itself.
    import sys

    home = str(_homes()[1])
    monkeypatch.setenv("PYTHONPATH", home)
    monkeypatch.setattr(sys, "path", [*sys.path, home])
    policy = sandbox.build_policy(fresh=True)
    argv = sandbox.wrap_argv(["true"], policy)
    _assert_no_broad_bind(argv)
    assert home not in [dst for _, _, dst in _binds(argv)]


@needs_bwrap
def test_a_workspace_that_is_a_whole_home_is_refused(world, monkeypatch):
    monkeypatch.setattr(SETTINGS, "UNIFY_LOCAL_ROOT", str(world["home"]))
    policy = sandbox.build_policy(fresh=True)
    with pytest.raises(sandbox.SandboxRefusal) as raised:
        sandbox.wrap_argv(["true"], policy)
    assert raised.value.rule == "root-allowlist"


def test_the_allowlist_names_a_reason_for_every_entry_and_never_all_of_etc():
    paths = [p for p, _ in sandbox._ROOT_ALLOWLIST]
    assert len(paths) == len(set(paths))
    assert all(reason.strip() for _, reason in sandbox._ROOT_ALLOWLIST)
    assert "/" not in paths and "/etc" not in paths and "/home" not in paths
    for forbidden in ("/root", "/var", "/opt", "/mnt", "/workspaces", "/srv", "/run"):
        assert not any(Path(p).is_relative_to(forbidden) for p in paths), forbidden
    assert {"/etc/shadow", "/etc/ssh", "/etc/docker", "/etc/pip.conf"}.isdisjoint(
        paths,
    )


# ── access: what is outside the allowlist does not exist ────────────────────


@needs_bwrap
@pytest.mark.asyncio
async def test_a_credential_like_tree_outside_the_allowlist_is_absent(world):
    """A stand-in for ``/workspaces/envs/.env.*``: created by the test, outside
    the home and every mount, beside the world."""
    envs = world["home"].parent / "workspaces" / "envs"
    envs.mkdir(parents=True)
    (envs / ".env.prod").write_text(f"KEY={ENV_SECRET}\n")
    (envs / "service-account.json").write_text(f'{{"k": "{ENV_SECRET}"}}\n')
    policy = sandbox.build_policy(fresh=True)
    for name in (".env.prod", "service-account.json"):
        assert policy.readable_violation(envs / name) is not None
    assert policy.readable_violation(envs / "service-account.json")[0] == (
        "root-allowlist"
    )
    ex = SessionExecutor()
    try:
        out, _ = await bash(
            ex,
            f"for p in {envs} {envs}/.env.prod {envs}/service-account.json; do "
            'test -e "$p" && echo "seen $p" || echo absent; done; '
            f"cat {envs}/.env.prod {envs}/service-account.json 2>&1 | wc -c",
        )
        assert ENV_SECRET not in out and "seen" not in out
        assert out.split().count("absent") == 3
    finally:
        await ex.close()


@needs_bwrap
@pytest.mark.asyncio
async def test_host_directories_outside_the_allowlist_do_not_exist(world):
    """Existence only, inside the sandbox: nothing on the host is read."""
    dirs = ("/root", "/var", "/var/log", "/opt", "/mnt", "/mnt/c", "/workspaces")
    dirs += ("/srv", "/media", "/run", "/snap", "/boot", "/init")
    ex = SessionExecutor()
    try:
        out, _ = await bash(
            ex,
            "for p in " + " ".join(dirs) + "; do "
            'test -e "$p" && echo "seen $p"; done; echo done',
        )
        assert out.strip() == "done", out
        # The system directories are there, read-only.
        out, _ = await bash(
            ex,
            "test -x /usr/bin/env && test -e /bin/sh && test -f /etc/ld.so.cache "
            "&& echo system",
        )
        assert out.strip() == "system"
        out, _ = await bash(ex, "test -e /etc/shadow || echo no-shadow")
        assert out.strip() == "no-shadow"
    finally:
        await ex.close()


@needs_bwrap
@pytest.mark.asyncio
async def test_etc_passwd_lists_only_root_this_account_and_nobody(world):
    import pwd

    ex = SessionExecutor()
    try:
        out, _ = await bash(ex, "cut -d: -f1 /etc/passwd; whoami")
        names = out.split()
        me = pwd.getpwuid(os.getuid()).pw_name
        assert names[-1] == me
        assert set(names[:-1]) <= {"root", me, "nobody"}
    finally:
        await ex.close()


@needs_bwrap
def test_the_harness_file_tools_refuse_what_the_root_does_not_show(world):
    policy = sandbox.build_policy(fresh=True)
    for path in ("/root/x", "/var/log/syslog", "/opt/x", "/workspaces/envs/.env.a"):
        violation = policy.readable_violation(Path(path))
        assert violation is not None and violation[0] in (
            "root-allowlist",
            "mask-env-file",
        ), (path, violation)
    assert policy.readable_violation(Path("/usr/bin/env")) is None
    assert policy.readable_violation(world["workspace"] / "data.txt") is None


# ── the interpreter and the fake clock ──────────────────────────────────────


@needs_bwrap
def test_the_interpreter_chain_is_mounted_from_sys_prefix(world):
    import sys

    policy = sandbox.build_policy(fresh=True)
    argv = sandbox.wrap_argv(["true"], policy)
    dests = [Path(dst) for _, _, dst in _binds(argv)]

    def shown(p: Path) -> bool:
        return any(p == d or p.is_relative_to(d) for d in dests) or any(
            p.is_relative_to(Path(os.path.realpath(s)))
            for s in sandbox._SYSTEM_DIRS
            if os.path.isdir(s)
        )

    for p in (sys.prefix, sys.base_prefix, os.path.realpath(sys.executable)):
        assert shown(Path(p)), p
    assert shown(Path(sandbox.__file__).resolve().parent)


@needs_bwrap
def test_the_fake_clock_comes_only_from_the_harness(world, monkeypatch):
    """LD_PRELOAD, FAKETIME, FAKETIME_SHARED and TZ: the harness's values, the
    preloaded library mounted read-only; never what a cell passes."""
    lib = world["home"].parent / "libfaketime" / "libfaketimeMT.so.1"
    lib.parent.mkdir(parents=True)
    lib.write_bytes(b"\x7fELF-not-loaded-here")
    monkeypatch.setenv("LD_PRELOAD", str(lib))
    monkeypatch.setenv("FAKETIME", "@2024-01-02 03:04:05")
    monkeypatch.setenv("TZ", "UTC")
    monkeypatch.delenv("FAKETIME_SHARED", raising=False)
    policy = sandbox.build_policy(fresh=True)
    argv = sandbox.wrap_argv(["true"], policy)
    assert ("--ro-bind", str(lib), str(lib)) in _binds(argv)
    assert not any(Path(dst) == lib.parent for _, _, dst in _binds(argv))
    cell_env = {
        "LD_PRELOAD": "/tmp/evil.so",
        "FAKETIME": "+100d",
        "FAKETIME_SHARED": "x y",
        "TZ": "Asia/Tokyo",
        "OTHER": "kept",
    }
    env = sandbox.sandbox_env(policy, cell_env)
    assert env["LD_PRELOAD"] == str(lib) and env["FAKETIME"] == "@2024-01-02 03:04:05"
    assert env["TZ"] == "UTC" and "FAKETIME_SHARED" not in env
    assert env["OTHER"] == "kept"
    monkeypatch.delenv("LD_PRELOAD")
    policy = sandbox.build_policy(fresh=True)
    assert "LD_PRELOAD" not in sandbox.sandbox_env(policy, cell_env)
    assert str(lib) not in sandbox.wrap_argv(["true"], policy)


# ── the socket-family filter ────────────────────────────────────────────────

SOCKETS = """
import errno, os, socket
out = {}
for name, fam, typ in (("vsock", 40, socket.SOCK_STREAM), ("netlink", 16, socket.SOCK_RAW),
                       ("packet", 17, socket.SOCK_RAW), ("bluetooth", 31, socket.SOCK_STREAM),
                       ("unix", 1, socket.SOCK_STREAM), ("inet", 2, socket.SOCK_STREAM),
                       ("inet6", 10, socket.SOCK_STREAM)):
    try:
        socket.socket(fam, typ).close()
        out[name] = "ok"
    except OSError as e:
        out[name] = errno.errorcode.get(e.errno, str(e.errno))
try:
    a, b = socket.socketpair(socket.AF_UNIX)
    a.sendall(b"x"); out["unix-pair"] = b.recv(1).decode()
except OSError as e:
    out["unix-pair"] = errno.errorcode.get(e.errno, str(e.errno))
try:
    os.unshare(os.CLONE_NEWUSER)
    out["userns"] = "ok"
except OSError as e:
    out["userns"] = errno.errorcode.get(e.errno, str(e.errno))
"""

EXPECTED = {
    "vsock": "EAFNOSUPPORT",
    "netlink": "EAFNOSUPPORT",
    "packet": "EAFNOSUPPORT",
    "bluetooth": "EAFNOSUPPORT",
    "unix": "ok",
    "inet": "ok",
    "inet6": "ok",
    "unix-pair": "x",
    "userns": "EPERM",
}


@needs_bwrap
@pytest.mark.asyncio
async def test_a_python_cell_creates_only_unix_and_inet_sockets(world):
    """In the sandboxed worker (the default): creation only, nothing connects."""
    ex = SessionExecutor()
    try:
        res = await ex.execute(
            code=SOCKETS + "out",
            state_mode="stateless",
            session_id=None,
        )
        assert res["error"] is None, res["error"]
        assert res["result"] == EXPECTED
    finally:
        await ex.close()


@needs_bwrap
@pytest.mark.asyncio
async def test_a_shell_cell_and_its_children_are_filtered_too(world):
    import json
    import sys

    code = SOCKETS + "import json; print(json.dumps(out))"
    path = world["workspace"] / "sockets.py"
    path.write_text(code)
    ex = SessionExecutor()
    try:
        # The harness's interpreter (os.unshare needs 3.12), run from a shell.
        out, _ = await bash(
            ex,
            f"{sys.executable} {path} 2>&1 && (unshare -U true; echo $?)",
        )
        lines = out.strip().splitlines()
        assert json.loads(lines[0]) == EXPECTED
        assert lines[-1].strip() != "0"
    finally:
        await ex.close()


def test_the_program_checks_the_architecture_and_refuses_others():
    for arch in ("x86_64", "aarch64"):
        prog = sandbox.seccomp_program(arch)
        assert len(prog) % 8 == 0 and len(prog) // 8 < 64
        code, jt, jf, k = struct.unpack("=HBBI", prog[:8])
        assert (code, k) == (0x20, 4)  # load the audit arch first
        code, jt, jf, k = struct.unpack("=HBBI", prog[8:16])
        assert code == 0x15 and k == sandbox._SECCOMP_ARCHES[arch][0]
        # The last instruction kills: a mismatched arch jumps there.
        assert struct.unpack("=HBBI", prog[-8:]) == (0x06, 0, 0, 0x80000000)
    assert sandbox.seccomp_program("x86_64") != sandbox.seccomp_program("aarch64")
    with pytest.raises(sandbox.SandboxRefusal):
        sandbox.seccomp_arch("riscv64")
    assert errno.EAFNOSUPPORT == sandbox._EAFNOSUPPORT


@needs_bwrap
def test_bwrap_gets_the_program_on_a_descriptor_it_closes(world):
    policy = sandbox.build_policy(fresh=True)
    argv = sandbox.wrap_argv(["true"], policy)
    assert argv[:3] == ["/bin/sh", "-c", sandbox._SECCOMP_EXEC]
    assert Path(argv[3]).read_bytes() == sandbox.seccomp_program()
    options = argv[: argv.index("--")]
    assert options[options.index("--seccomp") + 1] == str(sandbox._SECCOMP_FD)
    assert options[-4:] == ["--remount-ro", "/", "--chdir", options[-1]]
