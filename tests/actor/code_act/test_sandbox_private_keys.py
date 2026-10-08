"""Symbolic: private keys in a mounted root are masked by their name, and in the
workspace by their armour too.

A bare ``id_rsa``, a ``server.key`` or a keystore in the workspace is the same
class of secret as a ``.env`` file: the secret scan masks it in cells and the
harness's file tools refuse it. In the workspace, a small file of any name
whose first bytes hold private-key armour (a PEM or PGP private key) is masked
as well; public halves (``*.pub``, public-key armour), ``known_hosts`` and
``authorized_keys`` stay readable. Every key here is fake (the armour lines
around a ``FAKE`` body) and made by the test; nothing connects anywhere.
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

FAKE_BODY = "FAKE-KEY-BODY-4b1d"  # pragma: allowlist secret

NAMED_KEYS = (
    "id_rsa",
    "id_dsa",
    "id_ecdsa",
    "id_ed25519",
    "id_ecdsa_sk",
    "id_ed25519_sk",
    "id_rsa_backup",
    "id_ed25519-work",
    "server.key",
    "SERVER.KEY",
    "client.p12",
    "client.pfx",
    "app.jks",
    "release.keystore",
    "putty.ppk",
    "vault.kdbx",
    ".git-credentials",
    ".netrc",
    "_netrc",
)
PUBLIC_NAMES = (
    "id_rsa.pub",
    "id_ed25519.pub",
    "id_ecdsa_sk.pub",
    "id_rsa-cert.pub",
    "server.key.pub",
    "known_hosts",
    "authorized_keys",
    "pubring.asc",
    "notes.txt",
)
PRIVATE_LABELS = (
    "PRIVATE KEY",
    "OPENSSH PRIVATE KEY",
    "RSA PRIVATE KEY",
    "EC PRIVATE KEY",
    "DSA PRIVATE KEY",
    "ENCRYPTED PRIVATE KEY",
    "PGP PRIVATE KEY BLOCK",
)


def _armour(label: str) -> str:
    return f"-----BEGIN {label}-----\n{FAKE_BODY}\n-----END {label}-----\n"


def _policy(workspace: Path, state: Path) -> sandbox.SandboxPolicy:
    workspace.mkdir(parents=True, exist_ok=True)
    state.mkdir(parents=True, exist_ok=True)
    ws = Path(os.path.realpath(workspace))
    return sandbox.SandboxPolicy(
        workspace=ws,
        state_dir=Path(os.path.realpath(state)),
        root_visible=[ws],
    )


# ── by name ─────────────────────────────────────────────────────────────────


def test_private_keys_are_masked_by_name_and_public_halves_are_not():
    for name in NAMED_KEYS:
        assert sandbox._secret_rule(name) == "mask-credentials", name
    for name in PUBLIC_NAMES:
        assert sandbox._secret_rule(name) is None, name


def test_the_workspace_and_root_scans_mask_named_keys(tmp_path, monkeypatch):
    workspace = tmp_path / "ws"
    (workspace / "deploy").mkdir(parents=True)
    for name in NAMED_KEYS:
        (workspace / "deploy" / name).write_text(_armour("OPENSSH PRIVATE KEY"))
    for name in PUBLIC_NAMES:
        (workspace / "deploy" / name).write_text("ssh-ed25519 FAKE\n")
    monkeypatch.setattr(sandbox, "_WORKSPACE_SCANS", {})
    scan = sandbox._scan_workspace(workspace, [])
    expected = {(workspace / "deploy" / n, "mask-credentials") for n in NAMED_KEYS}
    assert set(scan.files) == expected
    # The interpreter roots' walk matches the same names, without reading.
    files, dirs = sandbox._walk_secret_files(workspace, [])
    assert set(files) == expected and dirs == []


def test_the_name_rules_are_in_the_scan_cache_key(monkeypatch):
    before = sandbox._rules_digest()
    assert sandbox._ROOT_SCAN_RULES_VERSION >= 2
    monkeypatch.setattr(
        sandbox,
        "_SECRET_SUFFIXES",
        (*sandbox._SECRET_SUFFIXES, ".other"),
    )
    assert sandbox._rules_digest() != before


# ── by armour, in the workspace only ────────────────────────────────────────


def test_a_workspace_file_holding_private_key_armour_is_masked(tmp_path, monkeypatch):
    workspace = tmp_path / "ws"
    workspace.mkdir()
    keys = []
    for i, label in enumerate(PRIVATE_LABELS):
        path = workspace / f"notes{i}.txt"
        path.write_text(_armour(label))
        keys.append(path)
    # A service-account JSON (no "key" in its name) holds PKCS#8 armour.
    sa = workspace / "account.json"
    sa.write_text(
        '{"type": "service_account", "private_key": "'
        + _armour("PRIVATE KEY").replace("\n", "\\n")
        + '"}\n',
    )
    keys.append(sa)
    (workspace / "x.txt").write_text(_armour("PUBLIC KEY"))
    (workspace / "cert.txt").write_text(_armour("CERTIFICATE"))
    (workspace / "pub.asc").write_text(_armour("PGP PUBLIC KEY BLOCK"))
    monkeypatch.setattr(sandbox, "_WORKSPACE_SCANS", {})
    scan = sandbox._scan_workspace(workspace, [])
    assert set(scan.files) == {(p, "mask-credentials") for p in keys}
    # Each file read counts against the scan's cap.
    assert scan.entries == 2 * len(list(workspace.iterdir()))
    # The interpreter roots' walk never reads content.
    assert sandbox._walk_secret_files(workspace, []) == ([], [])


def test_large_tiny_linked_and_special_files_are_not_read(tmp_path, monkeypatch):
    workspace = tmp_path / "ws"
    workspace.mkdir()
    small = workspace / "small.txt"
    small.write_text(_armour("RSA PRIVATE KEY"))
    big = workspace / "big.txt"
    big.write_text(_armour("RSA PRIVATE KEY") + "A" * (65 * 1024))
    (workspace / "tiny.txt").write_text("x\n")
    (workspace / "link.txt").symlink_to(small)
    os.mkfifo(workspace / "pipe")
    read: list[str] = []
    real = sandbox._holds_private_key

    def spy(path):
        read.append(Path(path).name)
        return real(path)

    monkeypatch.setattr(sandbox, "_holds_private_key", spy)
    monkeypatch.setattr(sandbox, "_WORKSPACE_SCANS", {})
    scan = sandbox._scan_workspace(workspace, [])
    assert scan.files == [(small, "mask-credentials")]
    assert read == ["small.txt"]
    # Directly: a file over the size limit is not opened, a FIFO never blocks.
    assert real(big) is False
    assert real(workspace / "pipe") is False
    assert real(workspace / "link.txt") is False


def test_the_harness_file_tools_refuse_named_and_armoured_keys(tmp_path):
    policy = _policy(tmp_path / "ws", tmp_path / "state")
    ws = policy.workspace
    (ws / "id_ed25519").write_text("FAKE\n")
    (ws / "id_ed25519.pub").write_text("ssh-ed25519 FAKE\n")
    (ws / "known_hosts").write_text("host ssh-ed25519 FAKE\n")
    (ws / "notes.txt").write_text(_armour("OPENSSH PRIVATE KEY"))
    (ws / "x.txt").write_text(_armour("PUBLIC KEY"))
    (ws / "big.txt").write_text(_armour("PRIVATE KEY") + "A" * (65 * 1024))
    (ws / "alias.txt").symlink_to(ws / "notes.txt")
    for name in ("id_ed25519", "notes.txt", "alias.txt"):
        violation = policy.readable_violation(ws / name)
        assert violation is not None and violation[0] == "mask-credentials", name
    for name in ("id_ed25519.pub", "known_hosts", "x.txt", "big.txt"):
        assert policy.readable_violation(ws / name) is None, name
    # Read again on every request: armour written into a file seen before.
    plain = ws / "plain.txt"
    plain.write_text("nothing here yet, padded to be long enough\n")
    assert policy.readable_violation(plain) is None
    plain.write_text(_armour("EC PRIVATE KEY"))
    assert policy.readable_violation(plain) is not None
    with pytest.raises(sandbox.SandboxRefusal) as refused:
        sandbox.check_readable("notes.txt", policy)
    assert refused.value.rule == "mask-credentials"


# ── in a cell ───────────────────────────────────────────────────────────────


@needs_bwrap
@pytest.mark.asyncio
async def test_workspace_private_keys_are_masked_in_a_cell(world):
    ws = world["workspace"]
    (ws / "keys").mkdir()
    named = ws / "id_rsa"
    named.write_text(_armour("OPENSSH PRIVATE KEY"))
    armoured = ws / "keys" / "notes.txt"
    armoured.write_text(_armour("RSA PRIVATE KEY"))
    public = ws / "id_rsa.pub"
    public.write_text("ssh-rsa FAKE-PUBLIC-HALF\n")
    public_pem = ws / "keys" / "x.txt"
    public_pem.write_text(_armour("PUBLIC KEY").replace(FAKE_BODY, "FAKE-PUBLIC"))
    policy = sandbox.build_policy(fresh=True)
    masked = dict(policy.workspace_masked)
    assert masked.get(named) == "mask-credentials"
    assert masked.get(armoured) == "mask-credentials"
    assert public not in masked and public_pem not in masked
    ex = SessionExecutor()
    try:
        out, _ = await bash(ex, f"cat {ws}/id_rsa {ws}/keys/notes.txt 2>&1")
        assert FAKE_BODY not in out, out
        assert out.count("rule mask-credentials") == 2, out
        out, _ = await bash(ex, f"cat {ws}/id_rsa.pub {ws}/keys/x.txt")
        assert "FAKE-PUBLIC-HALF" in out and "BEGIN PUBLIC KEY" in out, out
        out, _ = await bash(ex, f"grep -r {FAKE_BODY} {ws} 2>/dev/null | wc -l")
        assert out.strip() == "0"
    finally:
        await ex.close()
    # The host's files are untouched.
    assert FAKE_BODY in named.read_text() and FAKE_BODY in armoured.read_text()
