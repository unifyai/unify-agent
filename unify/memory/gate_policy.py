"""The rev 2 gate: every check labels, and only a statically pure item's own tests run.

The rules: no capture or function is re-run to classify or test it (that fails for impure items and for worlds that
cannot be replayed); a write-time quality check labels, never rejects; confinement, security and credential
checks stay hard; the replay of a pure function's own recorded real calls stays. So, with
The gate's policy:

* **Labels only** (:func:`hard`): a finding about an item is recorded on its label, a finding about the pass is a
  note, and nothing refuses a merge, except the hard checks: G6 (safety: key-shaped strings, unreadable items), the
  pass's declared change set and manifest (G1 findings that no item owns: scope, undeclared or deleted items, the
  manifest itself), the provenance of recorded evidence (fixtures re-derived from the recording, cited episodes),
  and what breaks the library itself (syntax, imports that fail, missing paths, the gate's own kit).
* **Execution** (:func:`purity`, :func:`test_runnable`): the gate runs code only for an item that is statically pure
  (its form is a value or a plain function: at most one input besides plain values, no ``env``/``tools`` calls, no
  I/O, and it reads its input), and only the writer's own tests of it, its recipe on the cases the writer listed, or
  the replay of its own recorded calls. An impure, state-form or constant item is never run: its label says
  ``not run: <why>``.
* **Recorded replays** (off by default): the episode-cover replays, the
  held-out value reruns, the memorisation reruns (drawn and perturbed inputs) and the history probe replay recorded
  task data; they run only with that switch on, for comparisons.

Everything here reads syntax only: nothing is imported or run.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

#: Names whose use makes code impure (I/O or code from text); module roots whose use does too.
_IMPURE_CALLS = frozenset({"open", "exec", "eval", "compile", "__import__", "input", "breakpoint"})
_IMPURE_MODULES = frozenset({
    "os", "sys", "subprocess", "socket", "shutil", "pathlib", "io", "requests", "urllib", "http", "tempfile", "glob",
    "time", "random", "asyncio", "threading", "multiprocessing", "ctypes", "signal", "sqlite3",
})  # fmt: skip
#: Parameter names of the state forms (``run(state)``, ``run(state, env)``) and of a tool namespace.
_STATE_NAMES = frozenset({"state", "env", "tools"})
NOT_RUN_REPLAYS = "not run: recorded replays are off (rev 2)"


def _function(source: str) -> ast.FunctionDef | ast.AsyncFunctionDef | None:
    try:
        tree = ast.parse(source)
    except (SyntaxError, ValueError, RecursionError, MemoryError):
        return None
    fns = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
    return fns[0] if len(fns) == 1 else None


def purity(source: Any) -> tuple[bool, str]:
    """Whether the one function *source* defines (a library function's definition, as the gate reads it) is
    statically pure (module docstring), and why not."""
    from .fixed_answer import NEVER_READS, fixed_answer

    if not isinstance(source, str):
        return False, "its code cannot be read"
    fn = _function(source)
    if fn is None:
        return False, "its code is not one function"
    if isinstance(fn, ast.AsyncFunctionDef):
        return False, "it is async"
    params = [a.arg for a in (*fn.args.posonlyargs, *fn.args.args, *fn.args.kwonlyargs)]
    if _STATE_NAMES & set(params):
        return False, "it takes the state or a tool namespace (a state form)"
    for node in ast.walk(fn):
        if isinstance(node, ast.Name) and node.id in (_STATE_NAMES | {"memory"}) and isinstance(node.ctx, ast.Load):
            return False, f"it uses {node.id}"
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in _IMPURE_CALLS:
            return False, f"it calls {node.func.id}"
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            roots = [(a.name or "").split(".")[0] for a in node.names] if isinstance(node, ast.Import) else [
                (node.module or "").split(".")[0]
            ]
            bad = sorted(set(roots) & _IMPURE_MODULES)
            if bad:
                return False, f"it imports {bad[0]}"
        if isinstance(node, (ast.Global, ast.Nonlocal)):
            return False, "it writes a name outside itself"
        if isinstance(node, (ast.Yield, ast.YieldFrom, ast.Await)):
            return False, "it is a generator or async"
    fixed = fixed_answer(source)
    if fixed is not None and fixed.get("why") == NEVER_READS:
        return False, "it never reads its input (a constant)"
    return True, ""


def test_subjects(source: Any) -> set[str] | None:
    """The library functions a test module uses (``memory.<pkg>.<mod>:<name>``), from its imports: ``from
    memory.x.y import f`` gives ``memory.x.y:f``; a module imported whole gives the names used on it. None when it
    cannot be told from the syntax (it does not parse, or imports with ``*``)."""
    if not isinstance(source, str):
        return None
    try:
        tree = ast.parse(source)
    except (SyntaxError, ValueError, RecursionError, MemoryError):
        return None
    out: set[str] = set()
    modules: dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and (node.module or "").startswith("memory."):
            for a in node.names:
                if a.name == "*":
                    return None
                out.add(f"{node.module}:{a.name}")
        elif isinstance(node, ast.Import):
            for a in node.names:
                if a.name.startswith("memory.") and a.asname:
                    modules[a.asname] = a.name
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id in modules:
            out.add(f"{modules[node.value.id]}:{node.attr}")
    return out


@dataclass
class Policy:
    """One gate run's execution policy (module docstring): *replays* (recorded task data replayed) is always off;
    *bodies* the candidate's function sources by item id (the gate's ``c_bodies`` sources)."""

    replays: bool = False
    bodies: dict[str, str] = field(default_factory=dict)
    _pure: dict[str, tuple[bool, str]] = field(default_factory=dict)

    def item(self, item: str) -> tuple[bool, str]:
        """Whether *item* may be run (:func:`purity` of its candidate source), and why not."""
        if item not in self._pure:
            self._pure[item] = purity(self.bodies.get(item))
        return self._pure[item]

    def test(self, tree: Path, rel: str) -> tuple[bool, str]:
        """Whether the test module *rel* of *tree* may be run: every library function it uses is pure."""
        try:
            text = (Path(tree) / rel).read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            return False, f"{rel} cannot be read"
        subjects = test_subjects(text)
        if subjects is None:
            return False, f"what {rel} uses cannot be read from its code"
        for s in sorted(subjects):
            if s not in self.bodies:
                continue  # not a function of the library (a module, a class): judged where it is defined
            ok, why = self.item(s)
            if not ok:
                return False, f"{s} is not run: {why}"
        return True, ""

    def suite(self, tree: Path, rel: str) -> tuple[bool, str]:
        """Whether the test directory *rel* of *tree* may be run: each test module in it may (:meth:`test`)."""
        root = Path(tree) / rel
        if not root.is_dir():
            return True, ""
        for p in sorted(root.rglob("test_*.py")):
            ok, why = self.test(tree, str(p.relative_to(tree)))
            if not ok:
                return False, why
        return True, ""


def hard(check: str, reason: str, owners: Iterable[str]) -> bool:
    """Whether a finding stays a refusal under labels only (module docstring)."""
    from .gate import _HARD_MARKS

    text = str(reason)
    if check == "G6" or any(m in text for m in _HARD_MARKS):
        return True
    if check == "G1":
        if not list(owners):
            return True  # the pass's declared change set and its manifest
        return any(m in text for m in ("fixture provenance", "source episode", "cites unknown episode", "file "))
    return False
