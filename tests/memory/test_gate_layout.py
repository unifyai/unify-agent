"""Compiled code, start-up hooks and stray root entries are refused before anything is extracted or run.

On 9deefbfd1 ``layout_allowed`` admitted every non-``.py`` path (``manifest.py:297``, a
non-``.py`` path under a tests directory, and ``manifest.py:300``, any other non-``.py`` path) and
``forbidden`` (``manifest.py:255``) names only ``sitecustomize.py``/``usercustomize.py`` and ``*.pth``. So
``Gate._early`` (``gate.py:624``) passed such a candidate, ``materialise`` extracted it and G3 ran pytest with
``/memory`` holding, say, a sourceless ``env/__init__.pyc`` (run on every ``import env.*``), a root
``sitecustomize.pyc`` (run at interpreter start) or a root ``pytest.pyc`` (run instead of pytest). A file
under ``env/<channel>/tests/`` could even be declared in ``support`` (``support_allowed``,
``manifest.py:281``, admits any name with a non-``.py`` suffix, as no module name holds a dot) and merged. Each test below states which 9deefbfd1 check admitted its path, by asserting that check (unchanged
here), and shows that the gate now refuses it with no test run at all.

Module-level imports are only names 9deefbfd1 already had, so this file also runs against that build (the
fail-before run); the tests of the new policy function import it where they use it.

The gate-level cases (a candidate refused before anything runs) are on rev 2's library layout in
``test_gate_library_safety.py``; this file keeps the policy functions' own tests.
"""

import ast
from pathlib import Path

import pytest

from unify.memory.manifest import (
    TESTKIT,
    forbidden,
    layout_allowed,
)
from tests.memory.test_gate import (
    BASE2,
    FILES,
    PROBE_BASE,
)
from tests.memory.test_kinds_gate import WT_FILES

# --- the policy function itself -------------------------------------------------------------------------


@pytest.mark.parametrize(
    "path, check, fragment",
    [
        ("x.pth", "G6", "path configuration file"),
        ("env/venmo/tests/X.PYC", "G6", "bytecode or native code file"),
        ("env/venmo/tests/a.cpython-312.pyc", "G6", "bytecode or native code file"),
        (
            "env/venmo/tests/a.pypy310-pp73-x86_64-linux-gnu.so",
            "G6",
            "bytecode or native",
        ),
        ("env/venmo/tests/a.cpython-312.py", "G6", "bytecode or native code file"),
        ("workflows/__pycache__/x.md", "G6", "bytecode cache path"),
        ("env/venmo/tests/sitecustomize/d.txt", "G6", "interpreter start-up hook"),
        ("_pytest/x.txt", "G6", "shadows the _pytest module"),
        ("pluggy/x.txt", "G6", "shadows the pluggy module"),
        ("env", "G1", "outside the layout"),
        ("workflows", "G1", "outside the layout"),
        ("README.md", "G1", "outside the layout"),
        ("conftest/x.txt", "G1", "outside the layout"),
    ],
)
def test_unsafe_path_classes(path, check, fragment):
    from unify.memory.manifest import unsafe_path

    got = unsafe_path(path)
    assert got is not None and got[0] == check and fragment in got[1] and path in got[1]


def test_root_allow_list_shadows_nothing_and_root_conftest_stays_forbidden():
    from unify.memory.manifest import ROOT_DIRS, root_shadowed

    assert not {*ROOT_DIRS, TESTKIT.removesuffix(".py")} & root_shadowed()
    assert {"json", "pytest", "_pytest", "pluggy", "site"} <= root_shadowed()
    assert forbidden("conftest.py") and forbidden("sitecustomize.py")


def _fixture_library_paths() -> tuple[set[str], int]:
    """Every string key of every library-file dict literal in tests/memory (this file excepted)."""
    here = Path(__file__)
    keys: set[str] = set()
    dicts = 0
    for f in sorted(here.parent.rglob("*.py")):
        if f == here:
            continue
        for node in ast.walk(ast.parse(f.read_text())):
            if not isinstance(node, ast.Dict):
                continue
            ks = [
                k.value
                for k in node.keys
                if isinstance(k, ast.Constant) and isinstance(k.value, str)
            ]
            if any(k.startswith(("env/", "workflows/")) or k == TESTKIT for k in ks):
                dicts += 1
                keys.update(ks)
    return keys, dicts


def test_every_library_fixture_is_still_admitted():
    """The new refusals reject nothing a library fixture of tests/memory holds that 9deefbfd1 admitted."""
    from unify.memory.manifest import unsafe_path

    named = {**FILES, **PROBE_BASE, **BASE2, **WT_FILES}
    for p in named:
        assert layout_allowed(p) and not forbidden(p) and unsafe_path(p) is None, p
    keys, dicts = _fixture_library_paths()
    admitted = {k for k in keys if layout_allowed(k) and not forbidden(k)}
    assert dicts >= 40 and len(admitted) >= 10 and set(named) <= admitted
    # the fixtures that 9deefbfd1 already refused (bad on purpose) stay refused by the older checks
    assert [p for p in sorted(admitted) if unsafe_path(p) is not None] == []
