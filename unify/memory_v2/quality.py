"""Static test-quality checks of memory v2.1 (spec §9.1, §13.1–13.2, §13.4). Pure AST; nothing runs.

* :func:`exact_assertions` finds every ``assert <call of the function> == <expected>`` (either side) in each test
  function, and reports three things:
  - whether the function's input comes from a recorded fixture (``on_fixture``);
  - whether the expected value has trust 1–3 (``trusted``): it derives from a recorded fixture, or it is an
    informative literal equal to a recorded value (:func:`recorded_literals`);
  - whether the expected value recomputes the function's own expression (``oracle``: a self-reimplementing
    oracle, §13.4). That is an expression of at least :data:`ORACLE_MIN_NODES` nodes equal to one of the
    function's, with names renamed in order and builtins and constants kept.
* :func:`inventory`, :func:`test_changes` and :func:`count_items` give a tree's tests and items, and which tests a
  candidate deletes or weakens (fewer assertions, or fewer exact ones).

"Derives from a recorded fixture" is data flow within the test module:
- a string constant naming a fixture file (its path, or a '/'-bounded suffix of it) taints its expression;
- the taint is carried by assignments, ``for``, ``with`` and comprehension targets, ``pytest.mark.parametrize``
  argument names, and pytest fixtures returning a tainted value.

Known limits:
- data flow through helpers in other modules is not followed (such an assertion is not counted; the test can name
  the fixture directly);
- an oracle that changes a constant (``r["amount"]`` for ``r[key]``) is not matched.
"""

from __future__ import annotations

import ast
import builtins
import copy
import json
import re
from dataclasses import dataclass
from typing import Any, Callable, Iterable

from .episodes import Episode
from .fixtures import FixtureError, canonical, source_bytes, sources_of

ORACLE_MIN_NODES = 4
MAX_RECORDED = 500_000  # recorded literals kept per check (a compute bound; past it fewer assertions count)
TEST_FILE = re.compile(
    r"^memory/(?:[a-z_][a-z0-9_]*/)+tests/(?:[^/]+/)*test_[^/]*\.py\Z",
)
_BUILTINS = frozenset(dir(builtins))
_TOKEN = re.compile(r"[^\s\"'`,;()\[\]{}]+")
_NUMBER = re.compile(r"-?\d+(?:\.\d+)?\Z")
_TRIVIAL = frozenset(canonical(v) for v in (None, True, False, 0, 1, -1, "", [], {}))


@dataclass(frozen=True)
class Exact:
    test: str
    line: int
    on_fixture: bool
    trusted: bool
    oracle: bool


@dataclass(frozen=True)
class TestInfo:
    __test__ = False  # not a pytest class
    asserts: int
    exact: int


def _parse(source: bytes) -> ast.Module | None:
    try:
        return ast.parse(source)
    except (SyntaxError, ValueError, RecursionError):
        return None


def _names_fixture(node: ast.AST, fixtures: list[str]) -> bool:
    if not (
        isinstance(node, ast.Constant)
        and isinstance(node.value, str)
        and node.value.strip("/")
    ):
        return False
    tail = node.value.lstrip("/")
    return any(f == tail or f.endswith("/" + tail) for f in fixtures)


def _tainted(expr: ast.AST, names: set[str], fixtures: list[str]) -> bool:
    return any(
        (isinstance(n, ast.Name) and n.id in names) or _names_fixture(n, fixtures)
        for n in ast.walk(expr)
    )


def _flow(stmts: list[ast.stmt], seed: set[str], fixtures: list[str]) -> set[str]:
    names = set(seed)
    nodes = [n for s in stmts for n in ast.walk(s)]
    changed = True
    while changed:
        changed = False
        for n in nodes:
            pairs: list[tuple[ast.AST, ast.AST]] = []
            if isinstance(n, ast.Assign):
                pairs = [(t, n.value) for t in n.targets]
            elif (
                isinstance(n, (ast.AnnAssign, ast.AugAssign, ast.NamedExpr))
                and n.value is not None
            ):
                pairs = [(n.target, n.value)]
            elif isinstance(n, (ast.For, ast.AsyncFor, ast.comprehension)):
                pairs = [(n.target, n.iter)]
            elif isinstance(n, ast.withitem) and n.optional_vars is not None:
                pairs = [(n.optional_vars, n.context_expr)]
            for target, value in pairs:
                if _tainted(value, names, fixtures):
                    for t in ast.walk(target):
                        if isinstance(t, ast.Name) and t.id not in names:
                            names.add(t.id)
                            changed = True
    return names


def _is_fixture_decorator(d: ast.expr) -> bool:
    target = d.func if isinstance(d, ast.Call) else d
    return (isinstance(target, ast.Attribute) and target.attr == "fixture") or (
        isinstance(target, ast.Name) and target.id == "fixture"
    )


def _fixture_functions(
    mod: ast.Module,
    names: set[str],
    fixtures: list[str],
) -> set[str]:
    out = set()
    for s in mod.body:
        if isinstance(s, (ast.FunctionDef, ast.AsyncFunctionDef)) and any(
            _is_fixture_decorator(d) for d in s.decorator_list
        ):
            inner = _flow(s.body, names, fixtures)
            if any(
                isinstance(r, ast.Return)
                and r.value is not None
                and _tainted(r.value, inner, fixtures)
                for r in ast.walk(s)
            ):
                out.add(s.name)
    return out


def _param_names(fn: ast.AST, names: set[str], fixtures: list[str]) -> set[str]:
    out: set[str] = set()
    for d in getattr(fn, "decorator_list", []):
        if not (
            isinstance(d, ast.Call)
            and isinstance(d.func, ast.Attribute)
            and d.func.attr == "parametrize"
        ):
            continue
        if len(d.args) < 2 or not _tainted(d.args[1], names, fixtures):
            continue
        a = d.args[0]
        if isinstance(a, ast.Constant) and isinstance(a.value, str):
            out |= {x.strip() for x in a.value.split(",") if x.strip()}
        elif isinstance(a, (ast.List, ast.Tuple)):
            out |= {
                e.value
                for e in a.elts
                if isinstance(e, ast.Constant) and isinstance(e.value, str)
            }
    return out


def _tests(mod: ast.Module) -> list[tuple[str, ast.FunctionDef | ast.AsyncFunctionDef]]:
    out = []
    for s in mod.body:
        if isinstance(s, (ast.FunctionDef, ast.AsyncFunctionDef)) and s.name.startswith(
            "test",
        ):
            out.append((s.name, s))
        elif isinstance(s, ast.ClassDef) and s.name.startswith("Test"):
            out += [
                (f"{s.name}::{m.name}", m)
                for m in s.body
                if isinstance(m, (ast.FunctionDef, ast.AsyncFunctionDef))
                and m.name.startswith("test")
            ]
    return out


def _assigned(fn: ast.AST) -> dict[str, list[ast.expr]]:
    out: dict[str, list[ast.expr]] = {}
    for n in ast.walk(fn):
        if (
            isinstance(n, ast.Assign)
            and len(n.targets) == 1
            and isinstance(n.targets[0], ast.Name)
        ):
            out.setdefault(n.targets[0].id, []).append(n.value)
        elif (
            isinstance(n, ast.AnnAssign)
            and isinstance(n.target, ast.Name)
            and n.value is not None
        ):
            out.setdefault(n.target.id, []).append(n.value)
    return out


def _resolve(expr: ast.expr, assigned: dict[str, list[ast.expr]]) -> list[ast.expr]:
    """*expr* and the values assigned (once) to each name it reads."""
    out = [expr]
    for n in ast.walk(expr):
        if isinstance(n, ast.Name):
            out += assigned.get(n.id, [])
    return out


def _calls(expr: ast.AST, function: str) -> list[ast.Call]:
    return [
        n
        for n in ast.walk(expr)
        if isinstance(n, ast.Call)
        and (
            (isinstance(n.func, ast.Name) and n.func.id == function)
            or (isinstance(n.func, ast.Attribute) and n.func.attr == function)
        )
    ]


def _recorded(expr: ast.expr, recorded: set[str]) -> bool:
    try:
        value = ast.literal_eval(expr)
    except (ValueError, TypeError, SyntaxError, RecursionError, MemoryError):
        return False
    if isinstance(value, (set, frozenset, bytes, complex)):
        return False
    if isinstance(value, tuple):
        value = list(value)
    key = canonical(value)
    if key in _TRIVIAL or (isinstance(value, str) and len(value) < 2):
        return False
    return key in recorded


def _size(expr: ast.AST) -> int:
    return sum(1 for x in ast.walk(expr) if isinstance(x, ast.expr))


class _Rename(ast.NodeTransformer):
    def __init__(self) -> None:
        self.map: dict[str, str] = {}

    def visit_Name(self, node: ast.Name) -> ast.Name:
        if node.id in _BUILTINS:
            return ast.Name(id=node.id, ctx=ast.Load())
        return ast.Name(
            id=self.map.setdefault(node.id, f"_v{len(self.map)}"),
            ctx=ast.Load(),
        )

    def visit_arg(self, node: ast.arg) -> ast.arg:
        return ast.arg(arg=self.map.setdefault(node.arg, f"_v{len(self.map)}"))


def _norm(expr: ast.AST) -> str:
    return ast.dump(
        _Rename().visit(copy.deepcopy(expr)),
        annotate_fields=False,
        include_attributes=False,
    )


def _significant(function_source: bytes, function: str) -> set[str]:
    mod = _parse(function_source)
    fn = None
    for n in mod.body if mod is not None else []:
        if (
            isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
            and n.name == function
        ):
            fn = n
    if fn is None:
        return set()
    return {
        _norm(e)
        for stmt in fn.body
        for e in ast.walk(stmt)
        if isinstance(e, ast.expr) and _size(e) >= ORACLE_MIN_NODES
    }


def exact_assertions(
    test_source: bytes,
    function_source: bytes,
    function: str,
    fixtures: Iterable[str],
    recorded: set[str],
    path: str = "",
) -> list[Exact]:
    mod = _parse(test_source)
    if mod is None:
        return []
    fixtures = sorted(fixtures)
    top = [
        s
        for s in mod.body
        if not isinstance(s, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
    ]
    module_names = _flow(top, set(), fixtures)
    fixture_fns = _fixture_functions(mod, module_names, fixtures)
    sig = _significant(function_source, function)
    out: list[Exact] = []
    for qual, fn in _tests(mod):
        seed = set(module_names) | _param_names(fn, module_names, fixtures)
        seed |= {a.arg for a in fn.args.args if a.arg in fixture_fns}
        names = _flow(fn.body, seed, fixtures)
        assigned = _assigned(fn)
        for node in ast.walk(fn):
            if not (
                isinstance(node, ast.Assert)
                and isinstance(node.test, ast.Compare)
                and len(node.test.ops) == 1
                and isinstance(node.test.ops[0], ast.Eq)
            ):
                continue
            left, right = node.test.left, node.test.comparators[0]
            for got, want in ((left, right), (right, left)):
                calls = [
                    c for e in _resolve(got, assigned) for c in _calls(e, function)
                ]
                wants = _resolve(want, assigned)
                if not calls or any(_calls(w, function) for w in wants):
                    continue
                on_fixture = any(
                    _tainted(arg, names, fixtures)
                    for c in calls
                    for arg in [*c.args, *(k.value for k in c.keywords)]
                )
                trusted = any(
                    _tainted(w, names, fixtures) or _recorded(w, recorded)
                    for w in wants
                )
                oracle = any(
                    _norm(sub) in sig
                    for w in wants
                    for sub in ast.walk(w)
                    if isinstance(sub, ast.expr)
                    and _size(sub) >= ORACLE_MIN_NODES
                    and not any(_names_fixture(x, fixtures) for x in ast.walk(sub))
                )
                out.append(
                    Exact(
                        f"{path}::{qual}" if path else qual,
                        node.lineno,
                        on_fixture,
                        trusted,
                        oracle,
                    ),
                )
                break
    return out


def _number(tok: str) -> int | float | None:
    if not _NUMBER.match(tok):
        return None
    return float(tok) if "." in tok else int(tok)


def recorded_literals(
    episodes: Iterable[Episode],
    blob: Callable[[str], bytes],
) -> set[str]:
    """Canonical JSON of every recorded value a literal may equal: each source's JSON sub-values, its lines, and
    the tokens (and numbers) of its strings."""
    out: set[str] = set()

    def add(v: Any) -> None:
        if len(out) < MAX_RECORDED:
            out.add(canonical(v))

    def walk(v: Any, depth: int = 0) -> None:
        add(v)
        if depth >= 16:
            return
        if isinstance(v, dict):
            for x in v.values():
                walk(x, depth + 1)
        elif isinstance(v, list):
            for x in v:
                walk(x, depth + 1)
        elif isinstance(v, str):
            for tok in _TOKEN.findall(v)[:10_000]:
                add(tok)
                num = _number(tok)
                if num is not None:
                    add(num)

    for ep in episodes:
        for s in sources_of(ep):
            try:
                text = source_bytes(ep, s, blob).decode("utf-8")
            except (FixtureError, UnicodeDecodeError):
                continue
            try:
                walk(json.loads(text))
            except ValueError:
                walk(text)
            for line in text.splitlines():
                if line.strip():
                    add(line.strip())
    return out


def _strength(fn: ast.AST) -> TestInfo:
    asserts = exact = 0
    for n in ast.walk(fn):
        if isinstance(n, ast.Assert):
            asserts += 1
            if (
                isinstance(n.test, ast.Compare)
                and len(n.test.ops) == 1
                and isinstance(n.test.ops[0], ast.Eq)
            ):
                exact += 1
        elif (
            isinstance(n, ast.withitem)
            and isinstance(n.context_expr, ast.Call)
            and isinstance(n.context_expr.func, ast.Attribute)
            and n.context_expr.func.attr == "raises"
        ):
            asserts += 1
    return TestInfo(asserts, exact)


def inventory(files: dict[str, bytes]) -> dict[str, TestInfo]:
    """``path::[Class::]test`` -> its assertion counts, for every test of the test files *files*."""
    out: dict[str, TestInfo] = {}
    for path in sorted(files):
        mod = _parse(files[path])
        if mod is None:
            out[f"{path}::<unparsable>"] = TestInfo(0, 0)
            continue
        for qual, fn in _tests(mod):
            out[f"{path}::{qual}"] = _strength(fn)
    return out


def test_changes(
    before: dict[str, TestInfo],
    after: dict[str, TestInfo],
) -> tuple[list[str], list[str]]:
    """(deleted tests, weakened tests: fewer assertions or fewer exact ones)."""
    deleted = sorted(t for t in before if t not in after)
    weakened = sorted(
        t
        for t in before
        if t in after
        and (after[t].asserts < before[t].asserts or after[t].exact < before[t].exact)
    )
    return deleted, weakened


test_changes.__test__ = False  # not a pytest test


def count_items(files: dict[str, bytes]) -> int:
    """Public top-level functions of the library modules (not tests, not the reserved helper), plus notes."""
    n = 0
    for p, data in files.items():
        if p.startswith("notes/") and p.endswith(".md"):
            n += 1
        elif (
            p.startswith("memory/")
            and p.endswith(".py")
            and "/tests/" not in p
            and p != "memory/__init__.py"
        ):
            mod = _parse(data)
            n += sum(
                1
                for s in (mod.body if mod is not None else [])
                if isinstance(s, (ast.FunctionDef, ast.AsyncFunctionDef))
                and not s.name.startswith("_")
            )
    return n
