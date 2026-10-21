"""Signature inspection must not run proposed Python source."""

import inspect

import pytest

from unify.function_manager.function_manager import FunctionManager


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("def f():\n    pass", "()"),
        ("def f(a, b=1, c=None):\n    pass", "(a, b=1, c=None)"),
        ("def f(a, /, b, *, c, d=2):\n    pass", "(a, /, b, *, c, d=2)"),
        (
            "def f(a, *items, option=True, **options):\n    pass",
            "(a, *items, option=True, **options)",
        ),
        (
            "async def f(value: int) -> str:\n    return str(value)",
            "(value: int) -> str",
        ),
        (
            "def f(value: 'Unknown') -> 'Result':\n    pass",
            "(value: 'Unknown') -> 'Result'",
        ),
        (
            "def f(value: int = 1, *, names: list[str] = None) -> bool:\n    pass",
            "(value: int = 1, *, names: list[str] = None) -> bool",
        ),
        (
            "def f(value={'a': [1, -2]}, z=1+2j):\n    pass",
            "(value={'a': [1, -2]}, z=(1+2j))",
        ),
    ],
    ids=[
        "empty",
        "literal-defaults",
        "positional-keyword-only",
        "variadics",
        "async",
        "forward-annotations",
        "annotated-defaults",
        "container-defaults",
    ],
)
def test_declared_signature(source, expected):
    assert FunctionManager._signature_of(source, "f") == expected


@pytest.mark.parametrize(
    "source",
    [
        "def other(x):\n    pass",
        "def f(:\n    pass",
        "if True:\n    def f(x):\n        pass",
        "def f(x):\n    pass\nf = replacement",
        "def f(x):\n    pass\ndef f(y):\n    pass",
        "@decorator\ndef f(x):\n    pass",
        "def f[T](value: T):\n    pass",
        "def f(x, x):\n    pass",
        "def f(value=missing()):\n    pass",
        "def f(value=default_value):\n    pass",
        "def f(value: annotation()) -> result_type():\n    pass",
        "def f(value: [x for x in values]):\n    pass",
        "def f(value={'a', 'b'}):\n    pass",
        "def f(value=set()):\n    pass",
    ],
    ids=[
        "name-mismatch",
        "syntax-error",
        "conditional-definition",
        "rebound",
        "duplicate-definition",
        "decorated",
        "type-parameters",
        "duplicate-parameter",
        "dynamic-default",
        "named-default",
        "dynamic-annotation",
        "annotation-comprehension",
        "unordered-default",
        "set-call-default",
    ],
)
def test_unsupported_signature_is_unknown(source):
    assert FunctionManager._signature_of(source, "f") == "(...)"


@pytest.mark.parametrize(
    "position",
    ["default", "annotation", "return", "decorator", "module", "body"],
)
def test_signature_does_not_execute_source(tmp_path, position):
    marker = tmp_path / "executed"
    expression = f"open({str(marker)!r}, 'w').write('executed')"
    sources = {
        "default": f"def f(value={expression}):\n    pass",
        "annotation": f"def f(value: {expression}):\n    pass",
        "return": f"def f() -> {expression}:\n    pass",
        "decorator": f"@{expression}\ndef f():\n    pass",
        "module": f"{expression}\ndef f():\n    pass",
        "body": f"def f():\n    {expression}",
    }
    signature = FunctionManager._signature_of(sources[position], "f")
    assert not marker.exists()
    assert signature == ("()" if position == "body" else "(...)")


def test_literal_signature_matches_inspect():
    def f(a, /, b=1, *items, enabled=True, **options):
        pass

    source = "def f(a, /, b=1, *items, enabled=True, **options):\n    pass"
    assert FunctionManager._signature_of(source, "f") == str(inspect.signature(f))


def test_candidate_construction_does_not_load_source(tmp_path):
    marker = tmp_path / "candidate-loaded"
    source = f"def f(value=open({str(marker)!r}, 'w').write('executed')):\n    pass"
    manager = object.__new__(FunctionManager)
    candidate = manager._verify_candidate(name="f", source=source, depends_on=[])
    assert candidate.name == "f"
    assert candidate.source == source
    assert candidate.signature == "(...)"
    assert callable(candidate.loader)
    assert not marker.exists()
