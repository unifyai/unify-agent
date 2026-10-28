from unify.memory.redact import Redactor

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
    from unify.memory.redact import KEY_SHAPED

    full = "a\n-----BEGIN OPENSSH PRIVATE KEY-----\nAAAAB3NzaC1\nmore\n-----END OPENSSH PRIVATE KEY-----\nz"  # pragma: allowlist secret
    cut = "head\n-----BEGIN RSA PRIVATE KEY-----\nMIIEpAIBAAKCAQEA\npart"  # pragma: allowlist secret
    public = "-----BEGIN PUBLIC KEY-----\nMIIBIjAN\n-----END PUBLIC KEY-----"
    assert "AAAAB3NzaC1" not in KEY_SHAPED.sub("<k>", full) and KEY_SHAPED.sub(
        "<k>",
        full,
    ).endswith("\nz")
    assert "MIIEpAIBAAKCAQEA" not in KEY_SHAPED.sub("<k>", cut)
    assert KEY_SHAPED.search(public) is None


#: A provider credential held in the environment that has no key shape (only its value names it).
ENV_NAME, ENV_VALUE = "X_PROVIDER_API_KEY", "uk_live_" + "Q" * 30  # pragma: allowlist secret
QUERY_VALUE = "AIza" + "x" * 35  # pragma: allowlist secret


def test_an_error_drops_an_environment_credential_by_value(monkeypatch):
    from unify.memory.redact import redact_error

    monkeypatch.setenv(ENV_NAME, ENV_VALUE)
    out = redact_error(f"boom {ENV_VALUE} at the provider")
    assert ENV_VALUE not in out and f"<secret:{ENV_NAME}>" in out, out


def test_an_error_drops_a_credential_in_a_url_query():
    from unify.memory.redact import redact_error

    out = redact_error(f"GET https://h.example/v1/x?key={QUERY_VALUE}&alt=json failed")
    assert QUERY_VALUE not in out and "?key=<redacted>&alt=json" in out, out
    out = redact_error(f"https://h.example/a?b=1&token={QUERY_VALUE}#frag")
    assert QUERY_VALUE not in out and "&token=<redacted>#frag" in out, out


def test_a_request_error_keeps_the_credential_out_of_errors_jsonl_and_the_fallback_log(monkeypatch, tmp_path, caplog):
    import logging

    from unify.memory.integration import request as request_mod
    from unify.memory.integration.paths import Paths

    monkeypatch.setenv(ENV_NAME, ENV_VALUE)
    paths = Paths.under(tmp_path / "state")
    run = request_mod.RequestRun("r", paths)
    exc = RuntimeError(f"the fork failed with {ENV_VALUE} and https://h.example/?key={QUERY_VALUE}")
    # written: errors.jsonl holds the redacted row
    run._error("fork", exc, lambda _msg: None)
    written = paths.errors.read_text()
    assert ENV_VALUE not in written and QUERY_VALUE not in written and "fork" in written, written
    # unwritable (a directory in its place): the fallback log line is redacted the same way
    paths.errors.unlink()
    paths.errors.mkdir()
    # the package's logger does not propagate to the root, so the capture handler is attached to it directly
    request_mod.logger.addHandler(caplog.handler)
    try:
        with caplog.at_level(logging.WARNING, logger=request_mod.logger.name):
            run._error("fork", exc, lambda _msg: None)
    finally:
        request_mod.logger.removeHandler(caplog.handler)
    assert "fork failed" in caplog.text
    assert ENV_VALUE not in caplog.text and QUERY_VALUE not in caplog.text, caplog.text
