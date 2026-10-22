"""The rev 2.1 gate: every check labels, and the gate runs no code at all.

The rules: no capture or function is executed or re-run to classify, dedupe, rank or test it (that fails for impure
items and for worlds that cannot be replayed, and a statically pure one is no exception); a write-time quality check
labels, never rejects; confinement, security and credential checks stay hard. So:

* **Labels only** (:func:`hard`): a finding about an item is recorded on its label, a finding about the pass is a
  note, and nothing refuses a merge, except the hard checks: G6 (safety: key-shaped strings, unreadable items), the
  pass's declared change set and manifest (G1 findings that no item owns: scope, undeclared or deleted items, the
  manifest itself), the provenance of recorded evidence (fixtures re-derived from the recording, cited episodes),
  and what breaks the library itself (syntax, imports that fail, missing paths, the gate's own kit).
* **Execution**: none. An item's tests, its recipe, its recorded calls and its package's suites are not run by the
  gate, and its label says ``not run: <why>`` (:data:`NOT_RUN_GATE`, :func:`not_run_reason`). Running code is a test
  the writer chooses: it states what it expects and runs it with ``tools.test`` during its pass. :func:`purity` says
  whether an item is statically pure, as a fact on its label, never as a licence to run it.

Everything here reads syntax only: nothing is imported or run.
"""

from __future__ import annotations

import ast
from typing import Any, Iterable

#: Names whose use makes code impure (I/O or code from text); module roots whose use does too.
_IMPURE_CALLS = frozenset({"open", "exec", "eval", "compile", "__import__", "input", "breakpoint"})
_IMPURE_MODULES = frozenset({
    "os", "sys", "subprocess", "socket", "shutil", "pathlib", "io", "requests", "urllib", "http", "tempfile", "glob",
    "time", "random", "asyncio", "threading", "multiprocessing", "ctypes", "signal", "sqlite3",
})  # fmt: skip
#: Parameter names of the state forms (``run(state)``, ``run(state, env)``) and of a tool namespace.
_STATE_NAMES = frozenset({"state", "env", "tools"})
#: Why the gate runs no item, test or recipe (rev 2.1).
NOT_RUN_GATE = "the gate runs no code (rev 2.1); the writer's tests are the ones it states and runs with tools.test"


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
    from .fixed_answer import reads_input

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
    if reads_input(source) is False:
        return False, "it never reads its input (a constant)"
    return True, ""


def not_run_reason(source: Any) -> str:
    """Why the gate did not run an item (:data:`NOT_RUN_GATE`), with whether its *source* is statically pure
    (:func:`purity`) as a fact for its label."""
    pure, why = purity(source)
    return NOT_RUN_GATE + ("" if pure else f"; it is not pure: {why}")


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
