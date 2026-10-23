"""J12: every test the design doc (REV2.2-DESIGN.md) names exists in the tree: each named test file somewhere under
``tests/``, each named test function (or a test file of that name). The names were extracted from the doc into
``design_named_tests.json`` (with the doc's sha256); re-extract when the doc changes.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest

HERE = Path(__file__).parent
TESTS = HERE.parent
NAMED = json.loads((HERE / "design_named_tests.json").read_text())


def _defined() -> tuple[set[str], set[str]]:
    files, funcs = set(), set()
    for p in TESTS.rglob("test_*.py"):
        files.add(p.name)
        try:
            tree = ast.parse(p.read_text())
        except SyntaxError:
            continue
        funcs.update(n.name for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)))
    return files, funcs


FILES, FUNCS = _defined()


@pytest.mark.parametrize("name", NAMED["files"])
def test_the_design_named_test_file_exists(name):
    assert name in FILES, f"the design names {name}; no such file under tests/"


@pytest.mark.parametrize("name", NAMED["names"])
def test_the_design_named_test_exists(name):
    assert name in FUNCS or f"{name}.py" in FILES, f"the design names {name}; no such test under tests/"
