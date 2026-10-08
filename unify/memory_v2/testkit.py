"""The library test kit (memory v2.1 stage 5): one versioned ``memlab`` wherever a library's tests run.

A stored library's tests may import ``memlab`` (``memlab.replay.env_from``/``RecordedEnv``, the harness's
exact-call replay, so tests never fake the environment themselves; ``memlab.inputs``; the analysis tools) and read recorded payloads by blob id (``memlab.inputs.blob``). Whether they can must never depend on a
gate switch: a library merged while a stage-5 switch was on must stay testable when every switch is off, in
another arm, in Sol's box and in the actor's export. So the kit is a property of the library, not of the
switches. :func:`stage` writes the same kit (the module set :data:`MODULES` at :data:`KIT_VERSION`, the pin
plugin, the referenced blobs) under a root directory, and every place a library's tests can run gets it:

* **the gate**: ``/inputs`` in every pytest run of a check when a stage-5 switch is on **or** a test-side file of
  the parent or the candidate uses the kit (:func:`uses_kit`: it imports ``memlab`` or the pin plugin, or names
  a recorded blob id). A library whose tests never use it is checked with exactly the calls of the screen
  build;
* **Sol's box**: ``/inputs`` (beside the pass's episodes) under the same condition, the parent library standing
  in for the candidate;
* **the actor's export**: ``<export>/.memlab``, put on the import path after the export, when the exported
  library's tests use the kit (:func:`stage_for_tree`).

Each root holds ``memlab/`` (the modules, a git stub and an ``__init__`` carrying :data:`KIT_VERSION`),
``_memv2_pin.py`` (the determinism plugin, inert unless loaded with ``-p``) and ``blobs/<id>`` (the recorded
blobs the tests name, bounded). ``memlab.inputs.blob`` finds ``blobs/`` beside the kit, so a test reads a
recorded blob the same way in every place; a test naming the ``/inputs`` path itself would not.

Library code (a module under ``env/``, not a test) must never use the kit: the actor imports the library
without it. :func:`library_uses` finds such uses and :func:`unresolved` finds a test's ``memlab`` imports the
current kit does not provide; the gate refuses both (G3, ``[qa:kit]``). Standard library only on the box side;
this module runs on the host.
"""

from __future__ import annotations

import ast
import shutil
from pathlib import Path
from typing import Callable, Iterable

from .manifest import TESTKIT, TESTS_DIR

KIT_VERSION = "1"  # bump when a module's public names change; stored tests are checked against the kit
# (replay.env_from/calls/call_of and RecordedEnv.issued/misses were added within "1": additive, and no stored
# library used the kit before them)
# Sol's toolkit before stage 5 (the order matters to nothing); the kit adds memlab.inputs.
BASE_MODULES = (
    "analysis",
    "replay.py",
    "episodes.py",
    "fingerprint.py",
    "blobs.py",
    "redact.py",
)
MODULES = BASE_MODULES + ("inputs.py",)
PIN_MODULE = "_memv2_pin"
PACKAGE = "memlab"
EXPORT_DIR = ".memlab"  # the kit's root inside the actor's export
MAX_REF_BLOBS = 1000
MAX_INPUT_FILE_BYTES = 16 * 1024**2
MAX_REF_BYTES = 64 * 1024**2
MAX_SCAN_FILES = 2000
MAX_SCAN_BYTES = 2 * 1024**2  # read of one test-side file
MAX_SCAN_TOTAL = 64 * 1024**2  # reads of every test-side file of one tree
INPUTS_PATH = "/inputs"
GITIO_STUB = '''\
"""memlab has no git in a library's test runs; this stands in for the names episodes.py imports."""


class GitError(RuntimeError):
    pass


class Repo:
    def __init__(self, *args, **kwargs):
        raise GitError("git is not available here")
'''
_SRC = Path(__file__).parent


def _init_text() -> str:
    return (
        '"""memlab: the memory library\'s test kit (replay, recorded inputs) and analysis tools."""\n\n'
        f'KIT_VERSION = "{KIT_VERSION}"\n'
    )


def stage_package(lab: Path) -> None:
    """Write the ``memlab`` package (every module of :data:`MODULES`, the git stub, the versioned init)."""
    lab.mkdir(parents=True)
    for name in MODULES:
        s = _SRC / name
        if s.is_dir():
            shutil.copytree(s, lab / name, ignore=shutil.ignore_patterns("__pycache__"))
        else:
            shutil.copyfile(s, lab / name)
    (lab / "__init__.py").write_text(_init_text())
    (lab / "gitio.py").write_text(GITIO_STUB)


def stage(
    root: Path,
    refs: Iterable[str],
    *,
    has_blob: Callable[[str], bool],
    read_blob: Callable[[str], bytes],
    blob_size: Callable[[str], int],
) -> list[str]:
    """The kit under *root*: ``memlab/``, ``_memv2_pin.py`` and the recorded blobs *refs* names in ``blobs/``
    (a blob already there is kept); the notes for blobs left out by the bounds."""
    root.mkdir(parents=True, exist_ok=True)
    stage_package(root / PACKAGE)
    shutil.copyfile(_SRC / "pin.py", root / f"{PIN_MODULE}.py")
    bdir = root / "blobs"
    bdir.mkdir(exist_ok=True)
    refs = list(refs)
    used = skipped = 0
    for sha in refs[:MAX_REF_BLOBS]:
        if not has_blob(sha):
            continue  # a 64-hex token that names no recorded blob
        size = blob_size(sha)
        if size > MAX_INPUT_FILE_BYTES or used + size > MAX_REF_BYTES:
            skipped += 1
            continue
        if not (bdir / sha).exists():
            (bdir / sha).write_bytes(read_blob(sha))
        used += size
    over = max(0, len(refs) - MAX_REF_BLOBS)
    if skipped or over:
        return [
            f"[qa:blobs] {skipped + over} referenced blob(s) not mounted at /inputs/blobs (over "
            f"{MAX_REF_BLOBS} blobs, {MAX_INPUT_FILE_BYTES} bytes each or {MAX_REF_BYTES} in all)",
        ]
    return []


# --- which files use the kit ----------------------------------------------------------------------------------


def is_test_side(path: str) -> bool:
    """Whether *path* is test-side: under ``env/<channel>/tests/`` or the root test kit."""
    return bool(TESTS_DIR.match(path)) or path == TESTKIT


def read_sources(tree: Path, paths: Iterable[str]) -> dict[str, bytes]:
    """The test-side files among *paths* under *tree*, bounded per file and in all (path -> bytes)."""
    out: dict[str, bytes] = {}
    total = 0
    for rel in sorted(paths):
        if len(out) >= MAX_SCAN_FILES or total >= MAX_SCAN_TOTAL:
            break
        if not is_test_side(rel):
            continue
        p = tree / rel
        if p.is_symlink() or not p.is_file():
            continue
        with open(p, "rb") as fh:
            data = fh.read(min(MAX_SCAN_BYTES, MAX_SCAN_TOTAL - total))
        out[rel] = data
        total += len(data)
    return out


def tree_paths(tree: Path) -> list[str]:
    """The test-side paths of a checked-out library (``env/*/tests/**`` and the root test kit)."""
    out = [TESTKIT] if (tree / TESTKIT).is_file() else []
    for p in sorted(tree.glob("env/*/tests/**/*")):
        if len(out) >= MAX_SCAN_FILES:
            break
        if p.is_file() and not p.is_symlink():
            out.append(p.relative_to(tree).as_posix())
    return out


def _parse(source: bytes) -> ast.Module | None:
    try:
        return ast.parse(source)
    except (SyntaxError, ValueError, RecursionError):
        return None


# Calls that import a module named by their first argument (``importlib.import_module``,
# ``pytest.importorskip``, ``__import__``), matched by the called name whatever it is reached through.
DYNAMIC_IMPORTS = ("import_module", "importorskip", "__import__")


def _dynamic(tree: ast.Module) -> list[tuple[int, str | None]]:
    """``(line, module)`` of each dynamic import call; ``module`` None when not a constant string."""
    out: list[tuple[int, str | None]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        f = node.func
        called = f.attr if isinstance(f, ast.Attribute) else getattr(f, "id", None)
        if called not in DYNAMIC_IMPORTS:
            continue
        arg = node.args[0] if node.args else None
        if arg is None:
            arg = next(
                (k.value for k in node.keywords if k.arg in ("name", "modname")),
                None,
            )
        ok = isinstance(arg, ast.Constant) and isinstance(arg.value, str)
        out.append((node.lineno, arg.value if ok else None))
    return out


def _imports(tree: ast.Module) -> list[tuple[int, str, list[str]]]:
    """``(line, module, names)`` of each absolute import (``names`` empty for ``import m``), dynamic imports
    with a constant absolute name included."""
    out = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            out += [(node.lineno, a.name, []) for a in node.names]
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            out.append((node.lineno, node.module, [a.name for a in node.names]))
    out += [(line, m, []) for line, m in _dynamic(tree) if m and not m.startswith(".")]
    return out


def _kit_name(dotted: str) -> bool:
    head = dotted.split(".", 1)[0]
    return head in (PACKAGE, PIN_MODULE)


def imports_kit(source: bytes) -> bool:
    """Whether a source file imports ``memlab`` or the pin plugin (statically or by a dynamic import with a
    constant name), or may (a dynamic import whose name is not a constant, which :func:`dynamic_unnamed`
    finds and the gate refuses in a test)."""
    tree = _parse(source)
    return tree is not None and (
        any(_kit_name(m) for _, m, _ in _imports(tree))
        or any(m is None for _, m in _dynamic(tree))
    )


def dynamic_unnamed(source: bytes) -> list[int]:
    """Lines of dynamic imports whose module name is not a constant string (the kit cannot be staged for
    a name known only at run time)."""
    tree = _parse(source)
    return (
        []
        if tree is None
        else sorted({line for line, m in _dynamic(tree) if m is None})
    )


def uses_kit(
    sources: Iterable[bytes],
    has_blob: Callable[[str], bool],
    refs: list[str],
) -> bool:
    """Whether test-side *sources* use the kit: a ``memlab`` or pin import (dynamic ones included; one
    with a run-time name counts), or a named recorded blob."""
    return any(imports_kit(s) for s in sources) or any(has_blob(r) for r in refs)


def library_uses(source: bytes) -> list[int]:
    """Lines where library (non-test) code uses the kit: an import of ``memlab`` or the pin plugin, the
    name of either as a string (``importlib.import_module``), or a path under ``/inputs``.
    """
    tree = _parse(source)
    if tree is None:
        return []
    lines = {line for line, m, _ in _imports(tree) if _kit_name(m)}
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            v = node.value
            if (
                _kit_name(v)
                and v.replace(".", "").replace("_", "").isalnum()
                or v == INPUTS_PATH
                or v.startswith(INPUTS_PATH + "/")
            ):
                lines.add(node.lineno)
    return sorted(lines)


def names_inputs_path(source: bytes) -> list[int]:
    """Lines of string constants naming a path under ``/inputs`` (a test that would only run in one box)."""
    tree = _parse(source)
    if tree is None:
        return []
    return sorted(
        {
            node.lineno
            for node in ast.walk(tree)
            if isinstance(node, ast.Constant)
            and isinstance(node.value, str)
            and (node.value == INPUTS_PATH or node.value.startswith(INPUTS_PATH + "/"))
        },
    )


def _provided() -> dict[str, set[str] | None]:
    """Every module the kit provides (dotted, under ``memlab``) and its top-level names (None: a package)."""
    out: dict[str, set[str] | None] = {PACKAGE: {"KIT_VERSION"}}

    def names(path: Path) -> set[str]:
        tree = _parse(path.read_bytes())
        found: set[str] = set()
        for node in tree.body if tree is not None else []:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                found.add(node.name)
            elif isinstance(node, (ast.Assign, ast.AnnAssign)):
                targets = (
                    node.targets if isinstance(node, ast.Assign) else [node.target]
                )
                found |= {t.id for t in targets if isinstance(t, ast.Name)}
            elif isinstance(node, (ast.Import, ast.ImportFrom)):
                found |= {(a.asname or a.name).split(".")[0] for a in node.names}
        return found

    for name in MODULES + ("gitio.py",):
        s = _SRC / name
        if name == "gitio.py":
            out[f"{PACKAGE}.gitio"] = {"GitError", "Repo"}
        elif s.is_dir():
            out[f"{PACKAGE}.{name}"] = None
            for sub in sorted(s.glob("*.py")):
                if sub.name != "__init__.py":
                    out[f"{PACKAGE}.{name}.{sub.stem}"] = names(sub)
        else:
            out[f"{PACKAGE}.{name[:-3]}"] = names(s)
    return out


def unresolved(source: bytes) -> list[tuple[int, str]]:
    """``(line, what)`` of each ``memlab`` import in a test file that the current kit does not provide."""
    tree = _parse(source)
    if tree is None:
        return []
    kit = _provided()
    out = []
    for line, module, names in _imports(tree):
        if module.split(".", 1)[0] != PACKAGE:
            continue
        if module not in kit:
            out.append((line, module))
            continue
        provided = kit[module]
        for n in names:
            if n == "*" or f"{module}.{n}" in kit:
                continue
            if provided is not None and n not in provided:
                out.append((line, f"{module}.{n}"))
            elif provided is None:
                out.append((line, f"{module}.{n}"))
    return out


def stage_for_tree(
    tree: Path,
    root: Path,
    *,
    has_blob: Callable[[str], bool],
    read_blob: Callable[[str], bytes],
    blob_size: Callable[[str], int],
    force: bool = False,
) -> bool:
    """Stage the kit under *root* when the library checked out at *tree* uses it (or *force*); whether it did."""
    from .qa_static import blob_refs

    sources = list(read_sources(tree, tree_paths(tree)).values())
    refs = blob_refs(sources)
    if not (force or uses_kit(sources, has_blob, refs)):
        return False
    stage(root, refs, has_blob=has_blob, read_blob=read_blob, blob_size=blob_size)
    return True
