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
