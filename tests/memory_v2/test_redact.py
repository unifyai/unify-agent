from unify.memory_v2.redact import Redactor

FAKE_KEY = "sk-or-v1-" + "ab" * 32  # pragma: allowlist secret


def test_secret_value_becomes_label():
    r = Redactor({"OPENROUTER_API_KEY": "s3cr3t-value-123"})
    assert r.text("x s3cr3t-value-123 y") == "x <secret:OPENROUTER_API_KEY> y"


def test_key_shaped_replaced_everywhere():
    r = Redactor()
    out = r.obj({"a": [FAKE_KEY, {"b": "pre" + FAKE_KEY}], "c": 3})
    assert FAKE_KEY not in repr(out)
    assert out["c"] == 3 and r.hits == 2


def test_from_environ_takes_only_secret_names_and_long_values():
    r = Redactor.from_environ(
        {
            "MY_API_KEY": "abcdefgh12",  # pragma: allowlist secret
            "PATH": "/usr/bin",
            "SHORT_TOKEN": "abc",
        },
    )
    assert r.text("abcdefgh12") == "<secret:MY_API_KEY>"
    assert r.text("/usr/bin abc") == "/usr/bin abc"


def test_pem_private_key_blocks_are_redacted_whole_or_cut():
    from unify.memory_v2.redact import KEY_SHAPED

    full = "a\n-----BEGIN OPENSSH PRIVATE KEY-----\nAAAAB3NzaC1\nmore\n-----END OPENSSH PRIVATE KEY-----\nz"  # pragma: allowlist secret
    cut = "head\n-----BEGIN RSA PRIVATE KEY-----\nMIIEpAIBAAKCAQEA\npart"  # pragma: allowlist secret
    public = "-----BEGIN PUBLIC KEY-----\nMIIBIjAN\n-----END PUBLIC KEY-----"
    assert "AAAAB3NzaC1" not in KEY_SHAPED.sub("<k>", full) and KEY_SHAPED.sub(
        "<k>",
        full,
    ).endswith("\nz")
    assert "MIIEpAIBAAKCAQEA" not in KEY_SHAPED.sub("<k>", cut)
    assert KEY_SHAPED.search(public) is None


def test_redact_imports_where_unify_is_absent(tmp_path):
    # memlab copies this module into Sol's box, which has no unify package
    import shutil
    import subprocess
    import sys
    from pathlib import Path

    import unify.memory_v2.redact as red

    pkg = tmp_path / "memlab"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("")
    shutil.copy(Path(red.__file__), pkg / "redact.py")
    code = (
        "import sys\n"
        f"sys.path.insert(0, {str(tmp_path)!r})\n"
        "sys.modules['unify'] = None  # as in the box: unify cannot be imported\n"
        "import memlab.redact as r\n"
        "print(r.registered_secrets(), r.Redactor().text('x'))\n"
    )
    out = subprocess.run(
        [sys.executable, "-I", "-c", code],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert out.returncode == 0, out.stderr[-2000:]
    assert out.stdout.strip() == "() x"
