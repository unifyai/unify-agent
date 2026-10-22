"""Compiled code, start-up hooks and stray root entries are refused before anything is extracted or run.

On 9deefbfd1 the gate's layout check admitted every non-``.py`` path (``manifest.py:297``, a
non-``.py`` path under a tests directory, and ``manifest.py:300``, any other non-``.py`` path) and
``forbidden`` (``manifest.py:255``) names only ``sitecustomize.py``/``usercustomize.py`` and ``*.pth``. So
``Gate._early`` (``gate.py:624``) passed such a candidate, ``materialise`` extracted it and G3 ran pytest with
``/memory`` holding, say, a sourceless ``env/__init__.pyc`` (run on every ``import env.*``), a root
``sitecustomize.pyc`` (run at interpreter start) or a root ``pytest.pyc`` (run instead of pytest). A file
under ``env/<channel>/tests/`` could even be declared in ``support`` (the support check,
``manifest.py:281``, admits any name with a non-``.py`` suffix, as no module name holds a dot) and merged. Each test below states which 9deefbfd1 check admitted its path, by asserting that check (unchanged
here), and shows that the gate now refuses it with no test run at all.

Module-level imports are only names 9deefbfd1 already had, so this file also runs against that build (the
fail-before run); the tests of the new policy function import it where they use it.

The gate-level cases (a candidate refused before anything runs) are on rev 2's library layout in
``test_gate_library_safety.py``; this file keeps the policy functions' own tests.
"""


import pytest

from unify.memory.manifest import (
    TESTKIT,
    forbidden,
)

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
