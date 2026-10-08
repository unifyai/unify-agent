"""What the import system could load from the library is refused (v2.1 review I4), in every mode.

The gate refuses before extraction what :func:`unify.memory_v2.manifest.unsafe_path` names (bytecode,
native code and ``__pycache__`` entries anywhere, ``.pth`` files and start-up hooks under any suffix, and
every root entry but ``env/``, ``workflows/`` and the test kit), beside the older layout and forbidden-file
refusals; :func:`unify.memory_v2.catalogue.reserved` names the root entries that could shadow ``import env``
or ``import memory``. The export never writes such a file from an older commit. The gate-level cases are in
``test_gate_layout.py``.

These refusals are a declared safety fix applied whatever the v2.1 switches say, so they must refuse nothing
a v2 library holds: :func:`test_no_path_of_a_v2_library_fixture_is_refused` checks every library path in the
v2 suite's fixtures (and the paths the v2 design records name), and
:func:`test_no_path_of_a_recorded_library_is_refused` checks real libraries when
``MEMORY_V2_LIBRARY_REPOS`` names their memory repositories (colon-separated bare repos, every commit on
``main``). Pure functions only; the gate-level and export tests live in ``test_gate_layout.py``, ``test_gate.py``
and ``integration/test_checkout.py``.
"""

import importlib
import os
import random
import re
import subprocess
import sys
from pathlib import Path

import pytest

from unify.memory_v2 import manifest as m
from unify.memory_v2.catalogue import reserved

# The probes of the re-review (I4), plus the shapes of each kind of importable artefact.
REFUSED = [
    "env/__init__.pyc",  # would turn the env namespace into a package whose code runs on every import
    "sitecustomize.pyc",  # run at interpreter start-up
    "usercustomize.pyc",
    "json.pyc",  # import json would run it
    "pytest.pyc",  # python -m pytest would run it
    "pytest/__init__.pyc",
    "pytest_plugin/data.txt",
    "_pytest.so",
    "conftest.pyc",
    "x.pth",
    "memory.abi3.so",
    "memory.cpython-312-x86_64-linux-gnu.so",
    "env.py",
    "memory/notes.md",
    "json/__init__.pyc",
    "os/data.txt",
    "README.md",
    "NOTES.md",
    "notes.txt",
    "proposals/better.md",
    ".memory/catalog.json",
    "env/venmo/__pycache__/__init__.cpython-312.pyc",
    "env/venmo/tests/__pycache__/test_me.cpython-312-pytest-8.3.4.pyc",
    "env/venmo/tests/fixture.pyc",
    "env/venmo/tests/data/rows.pyo",
    "env/venmo/tests/fast.so",
    "env/venmo/tests/sitecustomize.txt",
    "env/venmo/helper.pyd",
    "workflows/__pycache__/x.pyc",
]

# Root entries the gate also names reserved: the generated catalogue and what `import env` or
# `import memory` could resolve to.
RESERVED = (
    "memory.abi3.so",
    "memory.cpython-312-x86_64-linux-gnu.so",
    "env.py",
    "memory/notes.md",
    "README.md",
    ".memory/catalog.json",
)

# Paths the v2 design records name for real screen libraries (docs/design/memory-v2*.md in the research
# repository), beside the layout's own kinds.
RECORDED_V2_PATHS = [
    "env/env/__init__.py",
    "env/env/tests/test_demo_state.py",
    "env/env/tests/demos.json",
    "env/worktree/__init__.py",
    "env/uv/__init__.py",
    "env/worktree_workspace/tests/data/ledger.csv",
    "env/dialogue_user/NOTES.md",
    "workflows/close-the-month.md",
    m.TESTKIT,
]

FIXTURE_MODULES = (
    "tests.memory_v2.test_gate",
    "tests.memory_v2.test_gate_docstrings",
    "tests.memory_v2.test_catalogue",
    "tests.memory_v2.test_sol_pass",
    "tests.memory_v2.test_kinds_gate",
    "tests.memory_v2.test_held_out",
    "tests.memory_v2.test_memory_repo",
    "tests.memory_v2.integration.test_checkout",
    "tests.memory_v2.integration.test_cli_e2e_office",
)
_LIBRARY_PATH = re.compile(r"^(?:env|workflows)/[^\s:#]+\Z")


def _v2_layout_allowed(path: str) -> bool:
    """9deefbfd1's ``layout_allowed``, kept here to name the paths a v2 library could hold."""
    match = m.TESTS_DIR.match(path)
    if match is not None:
        if any(m._stem(p) in m._RESERVED_STEMS for p in match.group("rest").split("/")):
            return False
        if m.TEST_PATH.match(path):
            return True
        if path.endswith(".py"):
            return m.support_allowed(path)
        return True
    if path.endswith(".py"):
        return bool(m.MODULE_PATH.match(path)) or path == m.TESTKIT
    return True


def _declarable(path: str) -> bool:
    """A path a v2 manifest could declare: under env/ or workflows/, or the test kit, and v2-admitted."""
    name = path.rsplit("/", 1)[-1]
    return (
        (path == m.TESTKIT or bool(_LIBRARY_PATH.match(path)))
        and _v2_layout_allowed(path)
        and name not in m.FORBIDDEN_NAMES
        and not name.endswith(".pth")
    )


def _strings(value, out: set[str], depth: int = 0) -> None:
    if depth > 6:
        return
    if isinstance(value, str):
        out.add(value)
    elif isinstance(value, dict):
        for k, v in value.items():
            _strings(k, out, depth + 1)
            _strings(v, out, depth + 1)
    elif isinstance(value, (list, tuple, set, frozenset)):
        for v in value:
            _strings(v, out, depth + 1)


def _fixture_paths() -> list[str]:
    found: set[str] = set()
    for name in FIXTURE_MODULES:
        module = importlib.import_module(name)
        for key, value in vars(module).items():
            if not key.startswith("__") and isinstance(value, (dict, list, tuple)):
                _strings(value, found)
    return sorted(p for p in found if p == m.TESTKIT or _LIBRARY_PATH.match(p))


def _refused_early(path: str) -> bool:
    """Refused by the gate before anything is extracted (``Gate._early``): a forbidden file (G6), outside
    the layout, :func:`~unify.memory_v2.manifest.unsafe_path`, or a reserved path."""
    return (
        not m.layout_allowed(path)
        or m.forbidden(path)
        or m.unsafe_path(path) is not None
        or reserved(path)
    )


@pytest.mark.parametrize("path", REFUSED)
def test_importable_artefacts_and_foreign_root_entries_are_refused(path):
    assert _refused_early(path), path


@pytest.mark.parametrize("path", RESERVED)
def test_the_gate_names_env_and_memory_root_entries_reserved(path):
    assert reserved(path) and _refused_early(path), path


@pytest.mark.parametrize(
    "path",
    [
        p
        for p in REFUSED
        if p not in RESERVED and m.layout_allowed(p) and not m.forbidden(p)
    ],
)
def test_unsafe_path_names_every_other_artefact(path):
    """What the older layout and forbidden checks admit, unsafe_path refuses; reserved is only for names."""
    assert m.unsafe_path(path) is not None and not reserved(path), path


def _admitted(path: str) -> bool:
    return not _refused_early(path)


def test_no_path_of_a_v2_library_fixture_is_refused():
    paths = [p for p in _fixture_paths() + RECORDED_V2_PATHS if _declarable(p)]
    assert len(paths) >= 25, paths  # the fixtures were found
    assert [p for p in paths if not _admitted(p)] == []


@pytest.mark.skipif(
    not os.environ.get("MEMORY_V2_LIBRARY_REPOS"),
    reason="set MEMORY_V2_LIBRARY_REPOS to the recorded libraries' memory repositories",
)
def test_no_path_of_a_recorded_library_is_refused():
    seen: set[str] = set()
    for repo in filter(None, os.environ["MEMORY_V2_LIBRARY_REPOS"].split(":")):
        shas = _git(repo, "rev-list", "refs/heads/main", "--")
        for sha in shas.decode("ascii").split():
            names = _git(repo, "ls-tree", "-r", "--name-only", "-z", sha).split(b"\0")
            seen |= {n.decode("utf-8") for n in names if n}
    assert seen
    assert sorted(p for p in seen if not _admitted(p)) == []


# Classes of git failure, named in the assertion (stderr itself is not echoed: it can carry paths).
_GIT_FAILURES = (
    ("dubious ownership", "safe.directory ownership check"),
    ("not a git repository", "not a git repository"),
    ("unknown revision", "no main branch"),
    ("ambiguous argument", "no main branch"),
    ("permission denied", "permission denied"),
    ("no such file or directory", "missing path"),
)


def _git(repo: str, *args: str) -> bytes:
    """Run git read-only on the bare *repo*, independent of the caller's git environment and config.

    The variables a runner may set (``GIT_DIR``, ``GIT_INDEX_FILE``, ``GIT_CONFIG_*``…) are dropped, the
    global and system config are not read, and ``safe.directory=*`` is set for this call only, so a
    library repository owned by another user (a copied recording) is read rather than refused.
    """
    env = {k: v for k, v in os.environ.items() if not k.startswith("GIT_")}
    env.update(GIT_CONFIG_GLOBAL=os.devnull, GIT_CONFIG_NOSYSTEM="1")
    path = str(Path(repo).expanduser().resolve())
    proc = subprocess.run(
        ["git", "-c", "safe.directory=*", "--git-dir", path, *args],
        capture_output=True,
        env=env,
        check=False,
    )
    if proc.returncode != 0:
        err = proc.stderr.decode("utf-8", "replace").lower()
        cls = next((c for k, c in _GIT_FAILURES if k in err), "other")
        pytest.fail(f"git {args[0]} on {path} exited {proc.returncode}: {cls}")
    return proc.stdout


def test_the_allowed_root_entries_shadow_no_module():
    """``env`` is the library itself; ``workflows`` and the test kit must not be importable names."""
    names = m.shadowed_module_names() | set(sys.stdlib_module_names) | m.root_shadowed()
    assert "workflows" not in names
    assert m.TESTKIT.removesuffix(".py") not in names - m._RESERVED_STEMS


def test_random_root_names_and_suffixes_are_refused():
    """Seeded: any root entry other than env/, workflows/ and the test kit is refused, and any bytecode or
    extension suffix anywhere, whatever its name, by unsafe_path's bytecode reason."""
    rng = random.Random(20261008)
    stdlib = sorted(sys.stdlib_module_names)
    stems = set(rng.sample(stdlib, 40)) | {
        "sitecustomize",
        "usercustomize",
        "pytest",
        "_pytest",
        "pluggy",
        "pytest_cov",
        "conftest",
        "env",
        "memory",
        "json",
        "site",
        "doctest",
        "readme",
        "notes",
        "helpers",
        "x1",
    }
    stems = sorted(stems)
    suffixes = [
        "",
        ".py",
        ".pyc",
        ".pyo",
        ".so",
        ".abi3.so",
        ".cpython-312-x86_64-linux-gnu.so",
        ".pyd",
        ".dll",
        ".pth",
        ".txt",
        ".json",
        ".md",
        "/__init__.py",
        "/__init__.pyc",
        "/data.txt",
        "/x.pyc",
        "/__pycache__/x.cpython-312.pyc",
    ]
    compiled = [
        ".pyc",
        ".pyo",
        ".so",
        ".abi3.so",
        ".pyd",
        ".dll",
        ".dylib",
        ".cpython-311-x86_64-linux-gnu.so",
    ]
    for _ in range(600):
        path = rng.choice(stems) + rng.choice(suffixes)
        if path.split("/", 1)[0] in m.ROOT_DIRS and "/" in path:
            # the library's own directories: only artefacts are newly refused there, and the layout is v2's
            assert (m.unsafe_path(path) is not None) == _artefact(path), path
            assert m.layout_allowed(path) == _v2_layout_allowed(path), path
            continue
        assert _refused_early(path), path
    for _ in range(400):
        depth = rng.randint(0, 3)
        parts = ["env", rng.choice(["venmo", "shell_uv", "env"]), "tests"][
            : rng.randint(1, 3)
        ]
        parts += [rng.choice(stems) for _ in range(depth)]
        path = "/".join(parts + [rng.choice(stems) + rng.choice(compiled)])
        assert m.unsafe_path(path) == (
            "G6",
            f"bytecode or native code file {path}",
        ), path


def _artefact(path: str) -> bool:
    """Bytecode, a native extension or a ``__pycache__`` entry, by the names the review lists."""
    parts = path.split("/")
    return "__pycache__" in parts or parts[-1].lower().endswith(
        (".pyc", ".pyo", ".so", ".pyd", ".dll", ".dylib"),
    )


def test_startup_hook_names_are_refused_with_any_suffix_anywhere():
    rng = random.Random(7)
    for _ in range(100):
        stem = rng.choice(["sitecustomize", "usercustomize"])
        suffix = rng.choice(["", ".py", ".pyc", ".txt", ".so", ".json"])
        where = rng.choice(
            ["", "env/venmo/", "env/venmo/tests/", "env/venmo/tests/data/"],
        )
        path = f"{where}{stem}{suffix}"
        got = m.unsafe_path(path)
        assert got is not None and got[0] == "G6" and path in got[1], path
        assert got[1].startswith(
            ("interpreter start-up hook", "bytecode or native code file"),
        ), path


def test_sol_and_the_gate_run_pytest_without_bytecode():
    """The gate's confined pytest runs with PYTHONDONTWRITEBYTECODE=1 (set by run_confined) and
    ``-p no:cacheprovider`` (run_pytest); Sol is told the same command."""
    from unify.memory_v2 import sandbox_run, sol_pass

    src = Path(sandbox_run.__file__).read_text()
    confined = src[src.index("def run_confined(") : src.index("def run_pytest(")]
    assert re.search(r'"--setenv",\s*"PYTHONDONTWRITEBYTECODE",\s*"1"', confined)
    assert re.search(r'"-p",\s*"no:cacheprovider"', src[src.index("def run_pytest(") :])
    assert (
        "no:cacheprovider" in sol_pass.SOL_SYSTEM
        and '"PYTHONDONTWRITEBYTECODE": "1"' in sol_pass.SOL_SYSTEM
    )
