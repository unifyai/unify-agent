"""The static half of the gate's stage-5 test checks: replay fidelity, cuts of truncated recordings, blob refs.

Nothing here runs model code: test sources are parsed (:mod:`ast`) and recorded actions read.

**Replay fidelity** (:func:`stand_ins`). A function that takes the environment (declared input ``env``) and
whose channel has recorded calls is tested through :class:`~.replay.RecordedEnv` (exact-call replay). A test
that passes it a stand-in of its own is refused. Each call of such a function in a test file is resolved
structurally, by where its first argument comes from (assignments in the calling function and at module level,
pytest fixtures named by the test's parameters, and the return values of helper functions, through the test
kit ``unify_memory_testkit``, to a depth of :data:`MAX_DEPTH`):

* ``replay``: a call of the kit's replay, ``env_from``, ``RecordedEnv`` or ``RecordedEnv.from_jsonl``, imported
  from ``memlab.replay`` (directly, or re-exported or returned by a root ``unify_memory_testkit``);
* ``stand-in``: an instance of a class the tests or the test kit define (a subclass of ``RecordedEnv``
  included), a lambda, a literal container, ``type(...)`` with three arguments, or anything imported from
  ``unittest.mock`` or ``types.SimpleNamespace``;
* ``unknown``: anything else (a parameter of a helper, an attribute, a subscript). Never refused.

A call inside ``with pytest.raises(...)`` is a negative test (the function must refuse what it is given) and is
never refused, whatever it passes.

Symbols are resolved through the files' own imports and definitions, never by the words in their names.

**Cuts** (:func:`cuts`, :func:`asserts_on_cut`). A recording the recorder truncated carries its marker: a tool
response replaced by ``{"__truncated__": {"bytes", "shape", "preview"}}`` (the preview is cut at its end), a
dialogue observation whose middle is elided (``\\n[... N chars, middle elided ...]\\n``), and a shell output
tail of the recorder's full tail size (its first line is cut). A test of a function whose cover is truncated is
refused when an ``assert`` statement holds a string constant that contains a marker, or that occurs in the
truncated text only where it touches the cut: ending at a preview's end, overlapping or adjoining the elision
marker, or inside a cut tail's first line. Such a test learns the recorder's cut, not the environment.

**Blob references** (:func:`blob_refs`): the 64-hex tokens in test-side files, the blob ids the gate mounts at
``/inputs/blobs`` for the runs, so fixtures can reference recorded payloads by id instead of copying them.
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass
from typing import Any, Iterable

from .integration.adapters.shell import MAX_TAIL_BYTES
from .integration.adapters.tool import TRUNCATED

MAX_DEPTH = 3
MIN_ASSERTED_CHARS = 3
MAX_CUT_TEXT = (
    256 * 1024
)  # characters of a truncated recording searched for asserted constants
_BLOB_TOKEN = re.compile(rb"(?<![0-9a-f])[0-9a-f]{64}(?![0-9a-f])")
# The dialogue recorder's elision marker (adapters.dialogue.cap_text); a test keeps the two in step.
ELIDED = re.compile(r"\n\[\.\.\. \d+ chars, middle elided \.\.\.\]\n")
_REPLAY = "memlab.replay.RecordedEnv"
# the kit's replay constructors (memlab.replay, :mod:`.replay`): exact-call replay of recorded rows
REPLAY_ORIGINS = frozenset(
    (_REPLAY, _REPLAY + ".from_jsonl", "memlab.replay.env_from"),
)
_TESTKIT_MODULE = "unify_memory_testkit"


# --- replay fidelity -----------------------------------------------------------------------------------------


class _Scope:
    """One source file's imports and definitions."""

    def __init__(self, tree: ast.Module, name: str) -> None:
        self.name = name
        self.imports: dict[str, str] = {}  # local name -> dotted origin
        self.classes: set[str] = set()
        self.functions: dict[str, ast.FunctionDef | ast.AsyncFunctionDef] = {}
        self.assigns: dict[str, list[ast.expr]] = {}  # module-level bindings
        for node in tree.body:
            if isinstance(node, ast.Import):
                for a in node.names:
                    if a.asname:
                        self.imports[a.asname] = a.name
                    else:
                        head = a.name.split(".", 1)[0]
                        self.imports[head] = head
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                for a in node.names:
                    if a.name != "*":
                        self.imports[a.asname or a.name] = f"{node.module}.{a.name}"
            elif isinstance(node, ast.ClassDef):
                self.classes.add(node.name)
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                self.functions[node.name] = node
            elif isinstance(node, (ast.Assign, ast.AnnAssign)):
                targets = (
                    node.targets if isinstance(node, ast.Assign) else [node.target]
                )
                for t in targets:
                    if isinstance(t, ast.Name) and node.value is not None:
                        self.assigns.setdefault(t.id, []).append(node.value)


class _Resolver:
    def __init__(self, test: _Scope, kit: _Scope | None) -> None:
        self.test, self.kit = test, kit

    def origin(self, expr: ast.expr, scope: _Scope) -> tuple[str, _Scope | None]:
        """A dotted origin for *expr*: ``local-class:N``/``local-func:N`` (in the returned scope), an
        imported module path, ``builtins.N``, or ``""``."""
        if isinstance(expr, ast.Name):
            if expr.id in scope.classes:
                return f"local-class:{expr.id}", scope
            if expr.id in scope.functions:
                return f"local-func:{expr.id}", scope
            if expr.id in scope.imports:
                dotted = scope.imports[expr.id]
                return self._through_kit(dotted)
            return f"builtins.{expr.id}", None
        if isinstance(expr, ast.Attribute):
            base, where = self.origin(expr.value, scope)
            if base and not base.startswith(("local-", "builtins.")):
                return self._through_kit(f"{base}.{expr.attr}")
            if base.startswith("local-class:") and where is not None:
                return (
                    base,
                    where,
                )  # a classmethod of a local class still builds a stand-in
            return "", None
        return "", None

    def _through_kit(self, dotted: str) -> tuple[str, _Scope | None]:
        head, _, rest = dotted.partition(".")
        if head == _TESTKIT_MODULE and self.kit is not None and rest:
            name = rest.split(".", 1)[0]
            if name in self.kit.classes:
                return f"local-class:{name}", self.kit
            if name in self.kit.functions:
                return f"local-func:{name}", self.kit
            if name in self.kit.imports:
                return self.kit.imports[name] + rest[len(name) :], None
        return dotted, None

    def kind(
        self,
        expr: ast.expr | None,
        scope: _Scope,
        fn: ast.FunctionDef | ast.AsyncFunctionDef | None,
        depth: int = 0,
        fixtures: bool = True,
    ) -> str:
        """``replay``, ``stand-in`` or ``unknown`` for *expr* in *fn* (a test, a fixture or a helper).

        *fixtures*: whether *fn*'s parameters are pytest fixtures (a test or a fixture; not a helper).
        """
        if expr is None or depth > MAX_DEPTH:
            return "unknown"
        if isinstance(expr, ast.Lambda):
            return "stand-in"
        if isinstance(expr, (ast.Dict, ast.List, ast.Tuple, ast.Set)):
            return "stand-in"
        if isinstance(expr, ast.Call):
            origin, where = self.origin(expr.func, scope)
            if origin in REPLAY_ORIGINS:
                return "replay"
            if origin.startswith("local-class:"):
                return "stand-in"
            if origin.startswith("local-func:") and where is not None:
                helper = where.functions[origin.split(":", 1)[1]]
                fixture = _fixture(where, helper.name) is helper
                return self._returns(helper, where, depth + 1, fixture)
            if origin.startswith("unittest.mock.") or origin == "types.SimpleNamespace":
                return "stand-in"
            if origin == "builtins.type" and len(expr.args) == 3:
                return "stand-in"
            return "unknown"
        if isinstance(expr, ast.Name):
            values = _bindings(fn, expr.id) if fn is not None else []
            if not values and fn is not None and expr.id in _params(fn):
                if not fixtures:
                    return "unknown"  # a helper's parameter: whatever its caller passed
                fixture = _fixture(scope, expr.id) or _fixture(self.kit, expr.id)
                if fixture is not None:
                    owner = scope if fixture in scope.functions.values() else self.kit
                    return self._returns(fixture, owner, depth + 1, True)
                return "unknown"
            if not values:
                values = scope.assigns.get(expr.id, [])
                fn = None
            kinds = {self.kind(v, scope, fn, depth + 1, fixtures) for v in values}
            if "stand-in" in kinds:
                return "stand-in"
            return "replay" if kinds == {"replay"} else "unknown"
        return "unknown"

    def _returns(self, fn, scope, depth: int, fixtures: bool) -> str:
        values = [
            n.value
            for n in ast.walk(fn)
            if isinstance(n, (ast.Return, ast.Yield)) and n.value is not None
        ]
        kinds = {self.kind(v, scope, fn, depth, fixtures) for v in values}
        if "stand-in" in kinds:
            return "stand-in"
        return "replay" if kinds == {"replay"} else "unknown"


def _params(fn) -> set[str]:
    a = fn.args
    return {p.arg for p in a.posonlyargs + a.args + a.kwonlyargs}


def _bindings(fn, name: str) -> list[ast.expr]:
    out: list[ast.expr] = []
    for node in ast.walk(fn):
        if isinstance(node, ast.Assign):
            if any(isinstance(t, ast.Name) and t.id == name for t in node.targets):
                out.append(node.value)
        elif isinstance(node, ast.AnnAssign) and node.value is not None:
            if isinstance(node.target, ast.Name) and node.target.id == name:
                out.append(node.value)
        elif isinstance(node, (ast.With, ast.AsyncWith)):
            for it in node.items:
                if (
                    isinstance(it.optional_vars, ast.Name)
                    and it.optional_vars.id == name
                ):
                    out.append(it.context_expr)
    return out


def _fixture(scope: _Scope | None, name: str):
    """The pytest fixture *name* defined in *scope* (decorated with ``pytest.fixture``), or None."""
    if scope is None or name not in scope.functions:
        return None
    fn = scope.functions[name]
    for dec in fn.decorator_list:
        target = dec.func if isinstance(dec, ast.Call) else dec
        dotted = ""
        if isinstance(target, ast.Name):
            dotted = scope.imports.get(target.id, "")
        elif isinstance(target, ast.Attribute) and isinstance(target.value, ast.Name):
            dotted = scope.imports.get(target.value.id, "") + "." + target.attr
        if dotted in ("pytest.fixture", "_pytest.fixtures.fixture"):
            return fn
    return None


def _parse(source: bytes | None, name: str) -> _Scope | None:
    if source is None:
        return None
    try:
        return _Scope(ast.parse(source), name)
    except (SyntaxError, ValueError, RecursionError):
        return None


def _item_of(dotted: str) -> str | None:
    parts = dotted.split(".")
    if len(parts) == 3 and parts[0] == "env":
        return f"env/{parts[1]}:{parts[2]}"
    return None


def stand_ins(
    test_source: bytes,
    kit_source: bytes | None,
    env_items: dict[str, str],
) -> list[tuple[int, str]]:
    """``(line, item)`` of each call in *test_source* that passes a stand-in environment to an item.

    *env_items* maps each environment-taking item with recorded calls to its first parameter's name.
    """
    test = _parse(test_source, "test")
    if test is None:
        return []
    resolver = _Resolver(test, _parse(kit_source, _TESTKIT_MODULE))
    tree = ast.parse(test_source)
    out: list[tuple[int, str]] = []

    # calls inside ``with pytest.raises(...)``: negative tests, never a stand-in for the environment
    negative: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.With, ast.AsyncWith)) and any(
            isinstance(w.context_expr, ast.Call)
            and resolver.origin(w.context_expr.func, test)[0]
            in ("pytest.raises", "_pytest.python_api.raises")
            for w in node.items
        ):
            for stmt in node.body:
                negative.update(id(n) for n in ast.walk(stmt))

    def visit(root: ast.AST, fn, fixtures: bool) -> None:
        for node in ast.walk(root):
            if not isinstance(node, ast.Call) or id(node) in negative:
                continue
            origin, _ = resolver.origin(node.func, test)
            item = _item_of(origin)
            if item is None or item not in env_items:
                continue
            arg: ast.expr | None = None
            if node.args and not isinstance(node.args[0], ast.Starred):
                arg = node.args[0]
            else:
                arg = next(
                    (k.value for k in node.keywords if k.arg == env_items[item]),
                    None,
                )
            if resolver.kind(arg, test, fn, 0, fixtures) == "stand-in":
                out.append((int(node.lineno), item))

    # a function's parameters are fixtures when pytest calls it: a test (pytest's default collection rule,
    # the gate runs with ``-c /dev/null``) or a fixture; a helper's are whatever its caller passes
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            visit(
                node,
                node,
                node.name.startswith("test") or _fixture(test, node.name) is node,
            )
        elif isinstance(node, ast.ClassDef):
            for sub in node.body:
                if isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    visit(
                        sub,
                        sub,
                        node.name.startswith("Test") and sub.name.startswith("test"),
                    )
        else:
            visit(node, None, False)
    return sorted(set(out))


# --- cuts of truncated recordings ----------------------------------------------------------------------------


@dataclass(frozen=True)
class Cut:
    """A truncated recording's text and where the recorder cut it.

    ``where``: ``end`` (the text stops at the cut), ``middle`` (``[start, end)`` is the elision marker) or
    ``start`` (the text begins at the cut; its first line, up to ``end``, is partial).
    """

    text: str
    start: int
    end: int
    where: str
    marker: str = ""


def _texts(response: Any) -> list[str]:
    if isinstance(response, str):
        return [response]
    if isinstance(response, dict):
        return [
            v for k, v in response.items() if isinstance(v, str) and k == "observation"
        ]
    return []


def cuts(action: Any) -> list[Cut]:
    """The cuts in *action*'s recording ([] when the recorder kept it whole)."""
    kind = getattr(action, "kind", "tool")
    r = getattr(action, "response", None)
    out: list[Cut] = []
    if kind == "tool" and isinstance(r, dict) and set(r) == {TRUNCATED}:
        marker = r[TRUNCATED]
        preview = marker.get("preview") if isinstance(marker, dict) else None
        if isinstance(preview, str):
            out.append(Cut(preview, len(preview), len(preview), "end", TRUNCATED))
    if kind == "dialogue":
        for text in _texts(r):
            for m in ELIDED.finditer(text):
                out.append(Cut(text, m.start(), m.end(), "middle", m.group()))
    if kind == "shell" and isinstance(r, dict):
        for key in ("tail", "stdout_tail", "stderr_tail"):
            text = r.get(key)
            if (
                isinstance(text, str)
                and len(text.encode("utf-8", "replace")) >= MAX_TAIL_BYTES - 3
            ):
                nl = text.find("\n")
                out.append(Cut(text, 0, nl if nl >= 0 else len(text), "start"))
    return out


def truncated(action: Any) -> bool:
    return bool(cuts(action))


def asserted_strings(test_source: bytes) -> list[tuple[int, str]]:
    """``(line, constant)`` of each string constant inside an ``assert`` statement's test."""
    try:
        tree = ast.parse(test_source)
    except (SyntaxError, ValueError, RecursionError):
        return []
    out: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Assert):
            for sub in ast.walk(node.test):
                if isinstance(sub, ast.Constant) and isinstance(sub.value, str):
                    out.append((int(node.lineno), sub.value))
    return out


def _touches(cut: Cut, s: int, e: int) -> bool:
    if cut.where == "end":
        return e == cut.end
    if cut.where == "middle":
        return s < cut.end and e > cut.start or e == cut.start or s == cut.end
    return s < cut.end  # start: inside the partial first line


def on_cut(constant: str, cut: Cut) -> bool:
    """Whether asserting *constant* can only have learned *cut*."""
    if len(constant) < MIN_ASSERTED_CHARS:
        return False
    if (
        TRUNCATED in constant
        or ELIDED.search(constant)
        or (cut.marker and cut.marker in constant)
    ):
        return True
    text = cut.text[:MAX_CUT_TEXT]
    spans: list[tuple[int, int]] = []
    at = text.find(constant)
    while at >= 0 and len(spans) < 1000:
        spans.append((at, at + len(constant)))
        at = text.find(constant, at + 1)
    return bool(spans) and all(_touches(cut, s, e) for s, e in spans)


def asserts_on_cut(test_source: bytes, actions: Iterable[Any]) -> list[int]:
    """Lines of *test_source*'s ``assert`` statements that hold a constant learned from a cut of *actions*."""
    all_cuts = [c for a in actions for c in cuts(a)]
    if not all_cuts:
        return []
    return sorted(
        {
            line
            for line, const in asserted_strings(test_source)
            if any(on_cut(const, c) for c in all_cuts)
        },
    )


# --- blob references -----------------------------------------------------------------------------------------


def blob_refs(sources: Iterable[bytes]) -> list[str]:
    """The distinct 64-hex tokens in *sources*, in first-seen order (candidate blob ids)."""
    seen: dict[str, None] = {}
    for data in sources:
        for m in _BLOB_TOKEN.finditer(data):
            seen.setdefault(m.group().decode(), None)
    return list(seen)
