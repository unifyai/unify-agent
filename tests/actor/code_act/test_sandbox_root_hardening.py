"""Symbolic: the derived roots of the sandbox's allowlisted root are enumerated, not guessed.

A static review of the allowlisted root found that the roots taken from the
harness process could show far more than the interpreter and the packages:
``~/.local`` accepted as an "interpreter" (keyrings, uv credentials, every
benchmark's data), a ``PYTHONPATH`` of ``/mnt``, an editable checkout with its
``.git`` and tests, a link's name bound beside its target without the
target's masks, ``.env`` files below the first level, and any file named in
``LD_PRELOAD``. Each test below checks access only, on files it creates
itself, and nothing connects anywhere.
"""

from __future__ import annotations

import io
import os
import struct
import subprocess
import sys
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

BIND_OPTIONS = ("--bind", "--ro-bind", "--dev-bind", "--bind-try", "--ro-bind-try")


def _binds(argv: list[str]) -> list[tuple[str, str, str]]:
    end = argv.index("--")
    return [
        (argv[i], argv[i + 1], argv[i + 2])
        for i in range(end - 2)
        if argv[i] in BIND_OPTIONS
    ]


def _shown(argv: list[str]) -> list[Path]:
    """Every destination a bind, --dir or --symlink creates."""
    end = argv.index("--")
    out = [Path(dst) for _, _, dst in _binds(argv)]
    out += [Path(argv[i + 1]) for i in range(end - 1) if argv[i] == "--dir"]
    out += [Path(argv[i + 2]) for i in range(end - 2) if argv[i] == "--symlink"]
    return out


def _on_path(monkeypatch, *paths: Path) -> None:
    monkeypatch.setattr(sys, "path", [*sys.path, *map(str, paths)])


def _pth(monkeypatch, base: Path, *entries: Path) -> Path:
    """A site-packages directory on ``sys.path`` with one ``.pth`` naming *entries*."""
    site = base / "fake-venv" / "site-packages"
    site.mkdir(parents=True, exist_ok=True)
    (site / "_fake_editable.pth").write_text("".join(f"{e}\n" for e in entries))
    _on_path(monkeypatch, site, *entries)
    return site


def _candidates() -> dict[Path, sandbox.DerivedRoot]:
    return {d.path: d for d in sandbox._derived_candidates([])}


# ── the deny set ────────────────────────────────────────────────────────────


def test_the_deny_set_refuses_broad_and_private_paths(world):
    import pwd

    account = Path(os.path.realpath(pwd.getpwuid(os.getuid()).pw_dir))
    refused = [
        "/",
        "/mnt",
        "/var",
        "/home",
        "/usr",
        "/mnt/c",
        "/media/usb",
        "/home/someone-else/project",
        str(account),
        str(account / ".local"),
        str(account / ".local" / "share"),
        str(account / ".config"),
        str(account / ".cache"),
        str(world["home"]),
        str(world["home"] / ".local"),
    ]
    for path in refused:
        assert sandbox._root_refusal(Path(path)) is not None, path
    # The state directory and the log directories, and what contains them.
    state = world["state"]
    assert sandbox._root_refusal(state, [state]) is not None
    assert sandbox._root_refusal(state.parent, [state]) is not None
    # Inside the private directories, and inside the account's home: fine.
    for path in (
        account / ".local" / "share" / "uv" / "python",
        account / "unify-agent-worktrees",
        state / "transcripts",
    ):
        assert sandbox._root_refusal(path, [Path("/nonexistent")]) is None, path


@needs_bwrap
def test_a_pth_entry_or_pythonpath_of_root_or_mnt_is_left_out(world, monkeypatch):
    _pth(monkeypatch, world["home"].parent, Path("/"), Path("/mnt"))
    monkeypatch.setenv("PYTHONPATH", os.pathsep.join(["/", "/mnt"]))
    candidates = sandbox._derived_candidates([])
    for d in candidates:
        if d.path in (Path("/"), Path("/mnt")):
            assert d.refusal is not None, d
    policy = sandbox.build_policy(fresh=True)
    argv = sandbox.wrap_argv(["true"], policy)
    for dst in _shown(argv):
        assert dst not in (Path("/"), Path("/mnt")), dst


# ── ~/.local is never an interpreter, nor any other derived root ────────────


def _fake_local(home: Path) -> dict[str, Path]:
    """A fake home's ``.local`` with data, a keyring and a python link."""
    local = home / ".local"
    (local / "share" / "x").mkdir(parents=True)
    (local / "share" / "x" / "secret.txt").write_text(ENV_SECRET + "\n")
    (local / "share" / "keyrings").mkdir()
    (local / "share" / "keyrings" / "login.keyring").write_text(ENV_SECRET + "\n")
    # What made the old heuristic call ~/.local a Python installation.
    (local / "lib" / "python3.12" / "site-packages").mkdir(parents=True)
    (local / "bin").mkdir()
    (local / "bin" / "python3").symlink_to(os.path.realpath(sys.executable))
    return {
        "local": local,
        "share": local / "share",
        "secret": local / "share" / "x" / "secret.txt",
        "keyring": local / "share" / "keyrings" / "login.keyring",
        "python": local / "bin" / "python3",
    }


@needs_bwrap
def test_an_interpreter_link_in_dot_local_does_not_mount_it(world, monkeypatch):
    fake = _fake_local(world["home"])
    monkeypatch.setattr(sys, "executable", str(fake["python"]))
    by_path = _candidates()
    assert by_path[fake["local"]].kind == "interpreter"
    assert by_path[fake["local"]].refusal is not None
    accepted = [d.path for d in by_path.values() if d.refusal is None]
    assert not any(
        p == fake["local"] or fake["local"].is_relative_to(p) for p in accepted
    )
    args, visible = sandbox._root_mounts(sandbox._notices_dir())
    assert not any(
        v == fake["local"] or v.is_relative_to(fake["local"]) for v in visible
    )
    assert str(fake["local"]) not in args


@needs_bwrap
@pytest.mark.asyncio
async def test_no_pth_or_pythonpath_shows_a_fake_dot_local(world, monkeypatch):
    fake = _fake_local(world["home"])
    entries = (fake["local"], fake["share"], world["home"])
    _pth(monkeypatch, world["home"].parent, *entries)
    monkeypatch.setenv("PYTHONPATH", os.pathsep.join(map(str, entries)))
    by_path = _candidates()
    for entry in entries:
        assert by_path[entry].refusal is not None, by_path[entry]
    policy = sandbox.build_policy(fresh=True)
    argv = sandbox.wrap_argv(["true"], policy)
    for dst in _shown(argv):
        assert not (dst == fake["local"] or dst.is_relative_to(fake["local"])), dst
    for name in ("secret", "keyring"):
        assert policy.readable_violation(fake[name]) is not None
    ex = SessionExecutor()
    try:
        out, _ = await bash(
            ex,
            f"for p in {fake['secret']} {fake['keyring']} {fake['share']}; do "
            'test -e "$p" && echo "seen $p" || echo absent; done; '
            f"cat {fake['secret']} {fake['keyring']} 2>&1 | wc -c",
        )
        assert ENV_SECRET not in out and "seen" not in out, out
        assert out.split().count("absent") == 3
    finally:
        await ex.close()


# ── an editable checkout shows its packages, not itself ─────────────────────


@needs_bwrap
@pytest.mark.asyncio
async def test_an_editable_worktree_shows_only_its_packages(world, monkeypatch):
    root = world["home"].parent / "editable-checkout"
    (root / "pkg").mkdir(parents=True)
    (root / "pkg" / "__init__.py").write_text("VALUE = 42\n")
    (root / "single_module.py").write_text("NAME = 'module'\n")
    (root / ".env").write_text(f"KEY={ENV_SECRET}\n")
    (root / ".git").mkdir()
    (root / ".git" / "config").write_text("[core]\n")
    (root / "tests").mkdir()
    (root / "tests" / "x.py").write_text("SECRET = 1\n")
    (root / "logs").mkdir()
    _pth(monkeypatch, world["home"].parent, root)
    by_path = _candidates()
    assert by_path[root].kind == "editable-root" and by_path[root].refusal is None
    assert by_path[root / "pkg"].kind == "editable-package"
    policy = sandbox.build_policy(fresh=True)
    argv = sandbox.wrap_argv(["true"], policy)
    assert ("--ro-bind", str(root), str(root)) not in _binds(argv)
    ex = SessionExecutor()
    try:
        res = await ex.execute(
            code="import pkg, single_module\n(pkg.VALUE, single_module.NAME)",
            state_mode="stateless",
            session_id=None,
        )
        assert res["error"] is None, res["error"]
        assert tuple(res["result"]) == (42, "module")
        out, _ = await bash(
            ex,
            f"cd {root} && for p in .env .git tests logs pkg; do "
            'test -e "$p" && echo "seen $p" || echo "absent $p"; done',
        )
        assert out.split() == [
            "absent",
            ".env",
            "absent",
            ".git",
            "absent",
            "tests",
            "absent",
            "logs",
            "seen",
            "pkg",
        ], out
    finally:
        await ex.close()


# ── a link's name reaches the one mount of its target, masks included ──────


@needs_bwrap
@pytest.mark.asyncio
async def test_a_symlinked_root_keeps_its_masks_through_the_link(world, monkeypatch):
    real = world["home"].parent / "client-real"
    real.mkdir()
    (real / "relay_client.py").write_text("X = 1\n")
    (real / ".env").write_text(f"KEY={ENV_SECRET}\n")
    link = world["home"].parent / "client-link"
    link.symlink_to(real)
    _on_path(monkeypatch, link)
    monkeypatch.setenv("PYTHONPATH", str(link))
    policy = sandbox.build_policy(fresh=True)
    argv = sandbox.wrap_argv(["true"], policy)
    assert str(link) not in [dst for _, _, dst in _binds(argv)]
    end = argv.index("--")
    links = [
        (argv[i + 1], argv[i + 2]) for i in range(end - 2) if argv[i] == "--symlink"
    ]
    assert (str(real), str(link)) in links
    ex = SessionExecutor()
    try:
        out, _ = await bash(
            ex,
            f"cat {link}/.env {real}/.env 2>&1; test -f {link}/relay_client.py "
            "&& echo client",
        )
        assert ENV_SECRET not in out, out
        assert "client" in out
    finally:
        await ex.close()


@needs_bwrap
@pytest.mark.asyncio
async def test_secret_files_deep_in_a_derived_root_are_masked(world, monkeypatch):
    root = world["home"].parent / "relay-client"
    deep = root / "a" / "b" / "c"
    deep.mkdir(parents=True)
    (deep / ".env").write_text(f"KEY={ENV_SECRET}\n")
    (root / "a" / ".envrc").write_text(f"export KEY={ENV_SECRET}\n")
    (root / "a" / "b" / "service-key.json").write_text(f'{{"k": "{ENV_SECRET}"}}\n')
    (root / "a" / "id.pem").write_text(ENV_SECRET + "\n")
    (root / "a" / "plain.txt").write_text("plain\n")
    _on_path(monkeypatch, root)
    monkeypatch.setenv("PYTHONPATH", str(root))
    policy = sandbox.build_policy(fresh=True)
    masked = {p for p, _ in policy.masked_files}
    secrets = [
        deep / ".env",
        root / "a" / ".envrc",
        root / "a" / "b" / "service-key.json",
        root / "a" / "id.pem",
    ]
    assert set(secrets) <= masked
    ex = SessionExecutor()
    try:
        out, _ = await bash(
            ex,
            "cat " + " ".join(map(str, secrets)) + f" 2>&1; cat {root}/a/plain.txt",
        )
        assert ENV_SECRET not in out, out
        assert "plain" in out
    finally:
        await ex.close()


# ── LD_PRELOAD: shared libraries only ───────────────────────────────────────


@needs_bwrap
def test_an_ld_preload_of_a_non_library_is_not_bound(world, monkeypatch):
    notes = world["home"].parent / "preload" / "notes.txt"
    notes.parent.mkdir(parents=True)
    notes.write_text(ENV_SECRET + "\n")
    lib = notes.parent / "libfaketimeMT.so.1"
    lib.write_bytes(b"\x7fELF-not-loaded-here")
    monkeypatch.setenv("LD_PRELOAD", f"{notes}:{lib}")
    by_path = _candidates()
    assert by_path[notes].refusal is not None
    assert by_path[lib].refusal is None
    policy = sandbox.build_policy(fresh=True)
    argv = sandbox.wrap_argv(["true"], policy)
    dests = [dst for _, _, dst in _binds(argv)]
    assert str(notes) not in dests and str(lib) in dests


# ── the environment ─────────────────────────────────────────────────────────


def test_scrubbed_env_drops_credential_words_and_urls_with_passwords():
    env = {
        "GITHUB_PAT": "x",
        "NPM_AUTH": "x",
        "MYSQL_PASSWD": "x",
        "REDIS_PASS": "x",
        "DATABASE_URL": "postgres://user:pw@db.internal:5432/app",  # pragma: allowlist secret
        "PATH": "/usr/bin",
        "GIT_AUTHOR_NAME": "someone",
        "DOCS_URL": "https://example.org/docs",
        "PASSES": "3",
    }
    out = sandbox.scrubbed_env(env)
    assert set(out) == {"PATH", "GIT_AUTHOR_NAME", "DOCS_URL", "PASSES"}


# ── the dry run ─────────────────────────────────────────────────────────────


def test_print_roots_lists_paths_kinds_and_reasons_only(world, monkeypatch):
    monkeypatch.setenv("PYTHONPATH", "/mnt")
    _on_path(monkeypatch, Path("/mnt"))
    monkeypatch.setenv("UNIFY_SANDBOX_PROBE_VALUE", ENV_SECRET)
    buf = io.StringIO()
    sandbox.print_roots(buf)
    lines = buf.getvalue().splitlines()
    rows = [line.split("\t") for line in lines]
    assert rows and all(len(r) == 4 for r in rows), lines
    assert all(r[0] in ("accepted", "refused") for r in rows)
    assert ["accepted", "unify-package"] in [r[:2] for r in rows]
    assert any(r[:3] == ["refused", "pythonpath", "/mnt"] for r in rows), rows
    assert ENV_SECRET not in buf.getvalue()


def test_the_module_prints_the_roots_and_exits_zero(world):
    """As the launcher runs it, ``python -I -m unify.sandbox --print-roots``;
    ``-I`` drops the checkout from ``sys.path``, so this one is put back first
    (the venv's own editable install may name another checkout)."""
    checkout = str(Path(sandbox.__file__).resolve().parents[1])
    run = (
        "import runpy, sys; sys.path.insert(0, sys.argv.pop(1)); "
        "runpy.run_module('unify.sandbox', run_name='__main__', alter_sys=True)"
    )
    proc = subprocess.run(
        [sys.executable, "-I", "-c", run, checkout, "--print-roots"],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert proc.returncode == 0, proc.stderr[-2000:]
    assert "accepted\tunify-package\t" in proc.stdout


# ── the seccomp program, run by a BPF interpreter ───────────────────────────


def _run_bpf(prog: bytes, nr: int, arch: int, arg0: int) -> int:
    """Classic BPF over ``struct seccomp_data`` (the instructions the program uses)."""
    ins = [struct.unpack("=HBBI", prog[i : i + 8]) for i in range(0, len(prog), 8)]
    data = struct.pack(
        "<iIQ6Q",
        nr,
        arch,
        0,
        arg0 & 0xFFFFFFFFFFFFFFFF,
        0,
        0,
        0,
        0,
        0,
    )
    pc, acc = 0, 0
    for _ in range(len(ins) + 1):
        code, jt, jf, k = ins[pc]
        if code == 0x20:  # BPF_LD | BPF_W | BPF_ABS
            acc = struct.unpack_from("<I", data, k)[0]
            pc += 1
        elif code == 0x06:  # BPF_RET | BPF_K
            return k
        elif code == 0x15:  # BPF_JMP | BPF_JEQ | BPF_K
            pc += 1 + (jt if acc == k else jf)
        elif code == 0x35:  # BPF_JMP | BPF_JGE | BPF_K
            pc += 1 + (jt if acc >= k else jf)
        elif code == 0x45:  # BPF_JMP | BPF_JSET | BPF_K
            pc += 1 + (jt if acc & k else jf)
        else:
            raise AssertionError(f"unexpected BPF instruction {code:#x}")
    raise AssertionError("the program did not return")


ALLOW = 0x7FFF0000
KILL = 0x80000000
EPERM = 0x00050000 | 1
ENOSYS = 0x00050000 | 38
EAFNOSUPPORT = 0x00050000 | 97


@pytest.mark.parametrize("arch", ["x86_64", "aarch64"])
def test_the_seccomp_program_decides_each_sample_call(arch):
    audit, x32, nr = sandbox._SECCOMP_ARCHES[arch]
    prog = sandbox.seccomp_program(arch)
    read_nr = 0 if arch == "x86_64" else 63
    cases = [
        ("socket AF_UNIX", nr["socket"], audit, 1, ALLOW),
        ("socket AF_INET", nr["socket"], audit, 2, ALLOW),
        ("socket AF_INET6", nr["socket"], audit, 10, ALLOW),
        ("socket AF_VSOCK", nr["socket"], audit, 40, EAFNOSUPPORT),
        ("socket AF_NETLINK", nr["socket"], audit, 16, EAFNOSUPPORT),
        ("socket AF_PACKET", nr["socket"], audit, 17, EAFNOSUPPORT),
        ("socketpair AF_UNIX", nr["socketpair"], audit, 1, ALLOW),
        ("socketpair AF_VSOCK", nr["socketpair"], audit, 40, EAFNOSUPPORT),
        ("clone, a thread", nr["clone"], audit, 0x3D0F00, ALLOW),
        ("clone CLONE_NEWUSER", nr["clone"], audit, 0x10000011, EPERM),
        ("unshare CLONE_NEWNET", nr["unshare"], audit, 0x40000000, ALLOW),
        ("unshare CLONE_NEWUSER", nr["unshare"], audit, 0x10000000, EPERM),
        ("clone3", nr["clone3"], audit, 0, ENOSYS),
        ("io_uring_setup", nr["io_uring_setup"], audit, 0, ENOSYS),
        ("io_uring_enter", nr["io_uring_enter"], audit, 0, ENOSYS),
        ("io_uring_register", nr["io_uring_register"], audit, 0, ENOSYS),
        ("read", read_nr, audit, 0, ALLOW),
        # i386 (int 0x80) on x86_64, 32-bit ARM on aarch64: a foreign arch.
        ("a foreign arch", 1, 0x40000003, 0, KILL),
        ("a foreign arch's socket", nr["socket"], 0x40000028, 40, KILL),
    ]
    if x32:
        cases.append(("x32 socket", 0x40000000 | nr["socket"], audit, 40, ENOSYS))
        cases.append(("x32 read", 0x40000000, audit, 0, ENOSYS))
    for name, number, audit_arch, arg0, expected in cases:
        got = _run_bpf(prog, number, audit_arch, arg0)
        assert got == expected, (arch, name, hex(got), hex(expected))
