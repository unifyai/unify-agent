"""Rev 2.1: ``memory.find``'s argument shapes are read statically from an item's own files, never by running it.

The shapes the actor's ``memory.find`` matches came from the gate's runs of a function's tests; the gate runs no code
now, so they are read from the function's recorded-inputs fixture (the values its tests are parametrised over) and its
definition: the parameter is named from the signature's AST when the function takes exactly one. Where no shape can be
read, none is made up, and the function is not listed by ``find``.
"""

from __future__ import annotations

import json

import pytest

from unify.memory import sandbox_run
from unify.memory.shape_rows import sole_param, static_shapes

INIT = '"""Accounts."""\n'
MODULE = '''"""Account readers."""

raise SystemExit("importing this module would end the test run")


def user_id(response):
    return response["user_id"]


def pair(a, b):
    return a, b


def plain(text):
    return text


def nothing(response):
    return response
'''


@pytest.fixture
def no_box(monkeypatch):
    def refuse(*a, **k):
        raise AssertionError("a confined run was started")

    monkeypatch.setattr(sandbox_run, "_spawn_box", refuse)


def _tree(tmp_path):
    root = tmp_path / "lib"
    (root / "memory/acct/tests/data").mkdir(parents=True)
    (root / "memory/acct/__init__.py").write_text(INIT)
    (root / "memory/acct/users.py").write_text(MODULE)
    data = root / "memory/acct/tests/data"

    def lines(name, rows):
        (data / f"users.{name}.inputs.jsonl").write_text("".join(r + "\n" for r in rows))

    src = {"episode": "e1", "source": "action:0"}
    lines("user_id", [json.dumps({"input": {"user_id": "u1", "name": "Ada"}, "source": src}),
                      json.dumps({"input": {"user_id": "u2", "name": "Bo", "zone": "UTC"}, "source": src}),
                      "not json", json.dumps(["no input key"]), json.dumps({"source": src})])  # fmt: skip
    lines("pair", [json.dumps({"input": {"a": 1}, "source": src})])
    lines("plain", [json.dumps({"input": "just some words", "source": src})])
    return root


def test_shapes_come_from_the_recorded_inputs_fixture_with_the_sole_parameter(tmp_path, no_box):
    got = static_shapes(_tree(tmp_path))
    digest, shapes = got["memory.acct.users:user_id"]
    assert len(digest) == 64 and len(shapes) == 2
    assert all(d["param"] == "response" for d in shapes)
    assert all(d.get("kind") for d in shapes), shapes  # each is a value-shape descriptor
    # two parameters: the shape is known, the parameter it is passed as is not, so none is named
    (pair,) = got["memory.acct.users:pair"][1]
    assert "param" not in pair
    # plain text has no structure, and a function without a fixture has nothing: neither gets a made-up shape
    assert "memory.acct.users:plain" not in got and "memory.acct.users:nothing" not in got


def test_the_same_files_give_the_same_shapes(tmp_path, no_box):
    root = _tree(tmp_path)
    assert static_shapes(root) == static_shapes(root)


@pytest.mark.parametrize(
    "body, name",
    [
        ("def f(response):\n    return 1\n", "response"),
        ("def f(self, value):\n    return 1\n", "value"),
        ("async def f(*, rows):\n    return 1\n", "rows"),
        ("def f(a, b):\n    return 1\n", None),
        ("def f():\n    return 1\n", None),
        ("def f(*args):\n    return 1\n", None),
        ("def f(x, **kw):\n    return 1\n", None),
        ("def f(:\n", None),
        ("x = 1\n", None),
    ],
)
def test_the_sole_parameter_is_read_from_the_definition(body, name):
    assert sole_param(body) == name
