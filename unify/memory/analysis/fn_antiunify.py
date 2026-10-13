"""Anti-unification of function bodies (memory hygiene, stage 4): the most specific generalisation of n functions.

Structure only, never words: functions are compared as Python ASTs, after a normalisation that removes what does
not change behaviour and what only names things locally:

* docstrings are stripped (from the function and from every nested function or class) and annotations dropped
  (parameter and return annotations removed, an annotated assignment's annotation replaced by ``None``);
* names are resolved **per scope**, as Python does: the function, each nested function and lambda, each
  comprehension (whose first iterable belongs to the enclosing scope), each class body (skipped by the scopes
  nested in it); defaults and decorators belong to the enclosing scope, a walrus target in a comprehension to the
  enclosing function, ``global`` and ``nonlocal`` are honoured. Every *variable* (a scope and a name bound in it)
  is renamed canonically, ``_l0``, ``_l1``, ..., in order of first appearance, so ``def f(obs): total = obs + 1``
  and ``def g(x): z = x + 1`` normalise to the same tree while ``lambda len: len`` in one function never makes
  the builtin ``len`` read elsewhere a local. Free names (builtins, module globals, sibling functions,
  ``MemoryInputError``), attributes and constants are never renamed; nor are class-body names (they are
  attributes) or the name a dotted ``import a.b`` binds. A keyword at a call of a nested function follows that
  function's renamed parameter. **Names a caller can observe are kept as written**: a nested class's or type
  alias's name; a nested function's name and parameters when it escapes (read other than as a direct callee,
  called with ``**``, rebound or decorated: something may pass its parameters by keyword or read its
  ``__name__``); a lambda's parameters unless it is only called directly, with no keywords; and every name in a
  function that calls ``locals``, ``vars``, ``eval``, ``exec`` or ``dir``. The root function's own parameter
  names are interface too: :attr:`Generalisation.same_signature` compares them;
* optionally (:func:`antiunify_source` does it by default), an expression-statement call of a sibling library
  function, ``parse_submit_feedback(observation)``, is inlined first, so a function that validates through a
  sibling is compared on what it actually checks. The sibling's final ``return <expr>`` is kept as an expression
  statement (its calls and raises still happen) unless it is a bare name or constant. Inlining is refused when
  it could change what a name means (a sibling's global read that the caller binds, a caller name the sibling's
  inner scopes rebind, a fresh name already in use) or what the code does (a default that is not a constant or
  bare name, which would become fresh per call; a call of ``locals``, ``vars``, ``eval``, ``exec``, ``dir`` or
  ``super``, which would see the caller's frame), and stops, with ``inline_capped``, before the inlined
  function would pass :data:`MAX_NODES`; ``inline_depth`` levels (default 1), never re-entering a function
  already being inlined.

The generalisation is Plotkin's least general generalisation, extended to statement and argument lists:

* identical subtrees are kept whole; nodes of the same kind (same label: type plus identifiers, operators and
  constants) are kept and their children generalised; anything else becomes a numbered hole ``__holeN__``;
* list fields (bodies, arguments, operands) are aligned first (a dynamic-programming alignment that pairs only
  nodes of the same label, preferring identical and structurally closer ones); unpaired runs become one hole
  per gap (a statement ``__holeN__``, a ``*__holeN__`` argument, a ``**__holeN__`` keyword) bound to the run,
  possibly empty, of each input;
* the same disagreement (the same subtrees in every input) reuses the same hole;
* a node that cannot hold a hole (a keyword, comprehension, handler, pattern) passes the difference up to the
  nearest expression or statement; local-name pairings made inside a branch that ends as a hole are undone;
* locals that differ only by name are paired consistently across the inputs (``_mK``; :attr:`Generalisation.pairs`
  maps each back), a conflicting pairing is a hole.

The score is the **kept share**: the generalisation's non-hole nodes over the mean input size (nodes of the
normalised trees; operators and contexts are part of their node's label, not counted). A high kept share with
few holes marks a merge candidate, and the generalisation, holes as parameters, is the proposed merged body.
With inlining, kept nodes are split by owner: caller code in every input, inlined helper code in every input
(``helper_kept``; a pair whose kept share is mostly this is ``helper_driven``: both call the same helper, a
signal to extract, not to merge) or mixed (one input contains what another calls). ``owned_share`` scores only
caller-owned code and ``kept_share_uninlined`` is the share without inlining. A pair where one input calls
another is a ``wrapper`` (:attr:`Generalisation.calls`): keep or inline the wrapper, not a merge of peers. Each
hole carries its data flow in each input (:mod:`.fn_dataflow`, over-approximate): the parameters that reach it,
the return and raise sites it reaches and whether it has effects the analysis does not follow, so a merged
body's holes are typed as values, ``nonlocal`` (untracked effects, treated as values), guards or local steps.

Bounds: a function over :data:`MAX_NODES` nodes or :data:`MAX_DEPTH` deep is refused (``None``); literals are
keyed by type, length and digest, never by ``repr``; list alignments over :data:`DP_CELLS` cells pair only a
common prefix and suffix; the whole generalisation spends at most a work *budget* and, once spent, turns what is
left into holes (``bounded=True``). Every traversal is iterative, except the generalisation itself, which recurses
at most :data:`MAX_DEPTH` levels. Deterministic: no hashing that varies per process, no set iteration order.

Standard library only.
"""

from __future__ import annotations

import ast
import hashlib
import struct
from collections import Counter
from dataclasses import dataclass, field, replace
from typing import Any, Iterable, Mapping, Sequence

from .fn_dataflow import FlowSummary, RegionFlow
from .fn_dataflow import flows as _flows

MAX_NODES = 50_000  # AST nodes per function (operators and contexts excluded), inlined code included
MAX_DEPTH = 120  # AST nesting per function
WORK_BUDGET = 500_000  # generalisation steps and alignment cells per anti-unification
DP_CELLS = 40_000  # largest list alignment done exactly
MAX_LOCALS = 64  # distinct local names tracked per subtree for renaming-invariant keys
INLINE_DEPTH = 1  # levels of sibling inlining by default
MAX_INLINE_DEPTH = 4
SHORT_LITERAL = 64  # strings and bytes up to this length are labelled by value, longer ones by digest

_FN = (ast.FunctionDef, ast.AsyncFunctionDef)
_SCOPES = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Module)
_COMPS = (ast.ListComp, ast.SetComp, ast.GeneratorExp, ast.DictComp)
# builtins that read or change the calling frame's variables by name: no renaming in a function that calls one
_INTROSPECT = frozenset({"locals", "vars", "eval", "exec", "dir"})
# Singleton node kinds folded into their parent's label.
_FOLDED = (ast.expr_context, ast.boolop, ast.operator, ast.unaryop, ast.cmpop)
# List fields of these kinds correspond position by position (or are kept whole).
_POSITIONAL = {
    "Dict",
    "Compare",
    "arguments",
    "MatchMapping",
    "MatchClass",
    "JoinedStr",
    "TemplateStr",
}
# Generalised only as a whole: equal, or a hole.
_WHOLE = {"JoinedStr", "TemplateStr", "FormattedValue", "Interpolation"}
# Fields that bind a local name, by node type.
_BINDING_FIELDS = {
    "Name": "id",
    "arg": "arg",
    "ExceptHandler": "name",
    "FunctionDef": "name",
    "AsyncFunctionDef": "name",
    "ClassDef": "name",
    "MatchAs": "name",
    "MatchStar": "name",
    "MatchMapping": "rest",
    "alias": "asname",
}
_STARRED_GAPS = {("Call", "args"), ("List", "elts"), ("Tuple", "elts"), ("Set", "elts")}
_SLOT = "\x00local"
_LOCATION = {"lineno": 1, "col_offset": 0, "end_lineno": 1, "end_col_offset": 0}


class _Bounds(Exception):
    """A function over the node or depth cap."""


def _digest(parts: Any) -> bytes:
    """A digest of *parts*, which hold only bounded scalars (see :func:`_scalar`), so ``repr`` is safe here."""
    return hashlib.blake2b(
        repr(parts).encode("utf-8", "backslashreplace"),
        digest_size=16,
    ).digest()


def _blob(data: bytes) -> str:
    return hashlib.blake2b(data, digest_size=16).hexdigest()


# --- parsing and copying ------------------------------------------------------------------------------------


def function_defs(source: str) -> dict[str, ast.FunctionDef | ast.AsyncFunctionDef]:
    """The module's top-level functions by name; ``{}`` if *source* does not parse within the interpreter's limits."""
    try:
        tree = ast.parse(source)
    except (SyntaxError, ValueError, RecursionError, MemoryError):
        return {}
    return {n.name: n for n in tree.body if isinstance(n, _FN)}


def _children(node: ast.AST) -> Iterable[ast.AST]:
    for name in node._fields:
        value = getattr(node, name, None)
        if isinstance(value, ast.AST):
            yield value
        elif isinstance(value, list):
            for v in value:
                if isinstance(v, ast.AST):
                    yield v


def _preorder(root: ast.AST) -> Iterable[ast.AST]:
    stack = [root]
    while stack:
        node = stack.pop()
        yield node
        stack.extend(reversed(list(_children(node))))


def _is_docstring(stmt: ast.AST) -> bool:
    return (
        isinstance(stmt, ast.Expr)
        and isinstance(stmt.value, ast.Constant)
        and isinstance(stmt.value.value, str)
    )


def _copy(root: ast.AST, edit) -> tuple[ast.AST, dict[int, ast.AST]]:
    """An iterative deep copy of *root*; ``edit(original, copy)`` adjusts each copied node's own fields.

    Folded singletons (contexts, operators) are shared, not copied. Returns the copy and a map from each copied
    node's ``id`` to its original. Raises :class:`_Bounds` past :data:`MAX_NODES` or :data:`MAX_DEPTH`.
    """
    origin: dict[int, ast.AST] = {}
    count = 0
    # post-order: (original, depth, visited)
    stack: list[tuple[ast.AST, int, bool]] = [(root, 0, False)]
    built: dict[int, ast.AST] = {}
    while stack:
        node, depth, visited = stack.pop()
        if not visited:
            count += 1
            if count > MAX_NODES or depth > MAX_DEPTH:
                raise _Bounds()
            stack.append((node, depth, True))
            for child in reversed(list(_children(node))):
                if not isinstance(child, _FOLDED):
                    stack.append((child, depth + 1, False))
            continue
        fields = {}
        for name in node._fields:
            value = getattr(node, name, None)
            if isinstance(value, ast.AST):
                fields[name] = (
                    value if isinstance(value, _FOLDED) else built.pop(id(value))
                )
            elif isinstance(value, list):
                fields[name] = [
                    (
                        built.pop(id(v))
                        if isinstance(v, ast.AST) and not isinstance(v, _FOLDED)
                        else v
                    )
                    for v in value
                ]
            else:
                fields[name] = value
        new = type(node)(**fields)
        for attr in ("lineno", "col_offset", "end_lineno", "end_col_offset"):
            if hasattr(node, attr):
                setattr(new, attr, getattr(node, attr))
        edit(node, new)
        origin[id(new)] = node
        built[id(node)] = new
    return built.pop(id(root)), origin


# --- scopes and names ----------------------------------------------------------------------------------------


class _Scope:
    __slots__ = ("kind", "parent", "index", "bound", "globals", "nonlocals", "pinned")

    def __init__(self, kind: str, parent: "_Scope | None", index: int) -> None:
        self.kind = kind  # "module", "function", "lambda", "comp" or "class"
        self.parent = parent
        self.index = index
        self.bound: set[str] = set()
        self.globals: set[str] = set()
        self.nonlocals: set[str] = set()
        self.pinned: set[str] = (
            set()
        )  # bound by a dotted import with no alias: kept as written


_Var = tuple[int, str]  # (scope index, name)


@dataclass
class _Resolution:
    """Every name occurrence of a function resolved to its variable (see the module docstring)."""

    walk: list[tuple[ast.AST, _Scope]]  # pre-order, annotations skipped
    root: _Scope  # the function's own scope
    inner: list[_Scope]  # scopes nested in it
    occ: dict[
        int,
        list[tuple[str, int, _Var]],
    ]  # id(node) -> (field, list position or -1, variable)
    order: list[_Var]  # renamable variables, by first appearance
    fixed: set[
        str
    ]  # names kept as written: free, global, class-level, dotted imports, pinned
    free: set[str]  # names resolving outside the function (module globals, builtins)
    names: set[str]  # every name the function binds or reads
    # variables whose names are observable, kept as written (see _pinned)
    pinned: set[_Var] = field(default_factory=set)

    @property
    def root_names(self) -> set[str]:
        r = self.root
        return r.bound - r.globals - r.nonlocals

    @property
    def inner_bound(self) -> set[str]:
        out: set[str] = set()
        for s in self.inner:
            out |= s.bound
        return out


def _import_binding(a: ast.alias) -> str | None:
    if a.asname:
        return a.asname
    return None if a.name == "*" else a.name.split(".")[0]


def _resolve_in(name: str, scope: _Scope) -> _Scope | None:
    """The scope whose variable *name* is when used in *scope*; ``None`` when it resolves outside the function."""
    s: _Scope | None = scope
    first = True
    while s is not None and s.kind != "module":
        if name in s.globals:
            return None
        if name in s.nonlocals:
            s, first = s.parent, False
            continue
        if (
            s.kind == "class" and not first
        ):  # class bodies are not visible to the scopes nested in them
            s = s.parent
            continue
        if name in s.bound:
            return s
        s, first = s.parent, False
    return None


def _scoped_walk(
    fn: ast.AST,
) -> tuple[list[tuple[ast.AST, _Scope]], list[_Scope], dict[int, _Scope]]:
    """Pre-order (node, scope it is evaluated in), with every scope; raises :class:`_Bounds` past the caps."""
    module = _Scope("module", None, 0)
    scopes = [module]
    def_scope: dict[int, _Scope] = {}
    first_iter: dict[int, _Scope] = {}
    out: list[tuple[ast.AST, _Scope]] = []
    stack: list[tuple[ast.AST, _Scope, int]] = [(fn, module, 0)]

    def new(kind: str, parent: _Scope) -> _Scope:
        s = _Scope(kind, parent, len(scopes))
        scopes.append(s)
        return s

    while stack:
        node, scope, depth = stack.pop()
        if len(out) >= MAX_NODES or depth > MAX_DEPTH:
            raise _Bounds()
        out.append((node, scope))
        kids: list[tuple[Any, _Scope]]
        if isinstance(node, _FN):
            f = new("function", scope)
            def_scope[id(node)] = f
            kids = [
                (node.args, f),
                *((s, f) for s in node.body),
                *((d, scope) for d in node.decorator_list),
                *((t, scope) for t in getattr(node, "type_params", ()) or ()),
            ]
        elif isinstance(node, ast.Lambda):
            f = new("lambda", scope)
            def_scope[id(node)] = f
            kids = [(node.args, f), (node.body, f)]
        elif isinstance(
            node,
            ast.arguments,
        ):  # visited in its function's scope; defaults belong outside
            outer = scope.parent or scope
            kids = []
            for name in node._fields:
                value = getattr(node, name, None)
                where = outer if name in ("defaults", "kw_defaults") else scope
                for v in value if isinstance(value, list) else [value]:
                    if isinstance(v, ast.AST):
                        kids.append((v, where))
        elif isinstance(node, ast.arg):
            kids = []  # its annotation is dropped
        elif isinstance(node, ast.ClassDef):
            c = new("class", scope)
            kids = [
                *((b, scope) for b in node.bases),
                *((k, scope) for k in node.keywords),
                *((s, c) for s in node.body),
                *((d, scope) for d in node.decorator_list),
                *((t, scope) for t in getattr(node, "type_params", ()) or ()),
            ]
        elif isinstance(node, _COMPS):
            q = new("comp", scope)
            kids = []
            for name in node._fields:
                value = getattr(node, name)
                if name == "generators":
                    for i, g in enumerate(value):
                        if i == 0:
                            first_iter[id(g)] = scope
                        kids.append((g, q))
                elif isinstance(value, ast.AST):
                    kids.append((value, q))
        elif isinstance(node, ast.comprehension):
            outer = first_iter.get(id(node), scope)
            kids = [
                (node.target, scope),
                (node.iter, outer),
                *((x, scope) for x in node.ifs),
            ]
        elif isinstance(node, ast.AnnAssign):
            kids = [(node.target, scope)]
            if node.value is not None:
                kids.append((node.value, scope))
        elif isinstance(node, ast.NamedExpr):
            t = scope
            while t.kind == "comp" and t.parent is not None:
                t = t.parent
            kids = [(node.target, t), (node.value, scope)]
        else:
            kids = [(c, scope) for c in _children(node) if not isinstance(c, _FOLDED)]
        for child, sc in reversed(kids):
            stack.append((child, sc, depth + 1))
    return out, scopes, def_scope


def _resolve(fn: ast.FunctionDef | ast.AsyncFunctionDef) -> _Resolution:
    walk, scopes, def_scope = _scoped_walk(fn)
    root = def_scope[id(fn)]
    # pass 1: what each scope binds or declares
    for node, scope in walk:
        if node is fn:
            continue
        if isinstance(node, ast.Name):
            if not isinstance(node.ctx, ast.Load):
                scope.bound.add(node.id)
        elif isinstance(node, ast.arg):
            scope.bound.add(node.arg)
        elif isinstance(node, (*_FN, ast.ClassDef)):
            scope.bound.add(node.name)
        elif isinstance(node, (ast.ExceptHandler, ast.MatchAs, ast.MatchStar)):
            if node.name:
                scope.bound.add(node.name)
        elif isinstance(node, ast.MatchMapping):
            if node.rest:
                scope.bound.add(node.rest)
        elif isinstance(node, ast.alias):
            b = _import_binding(node)
            if b:
                scope.bound.add(b)
                if node.asname is None and "." in node.name:
                    scope.pinned.add(b)
        elif isinstance(node, ast.Global):
            scope.globals.update(node.names)
        elif isinstance(node, ast.Nonlocal):
            scope.nonlocals.update(node.names)
    # pass 2: each occurrence's variable, numbered by first appearance
    occ: dict[int, list[tuple[str, int, _Var]]] = {}
    order: list[_Var] = []
    seen: set[_Var] = set()
    fixed: set[str] = set()
    free: set[str] = set()
    names: set[str] = set()
    nbind: Counter = Counter()
    def_node: dict[_Var, ast.AST] = {}
    calls: list[tuple[ast.Call, _Scope]] = []
    callee: dict[int, ast.Call] = {
        id(n.func): n for n, _ in walk if isinstance(n, ast.Call)
    }
    escapes: set[_Var] = set()  # read other than as a direct callee, or called with **
    kw_called: set[_Var] = set()  # called with keywords
    introspects = False

    def see(
        node: ast.AST,
        fld: str,
        pos: int,
        name: str,
        scope: _Scope,
        binding: bool,
    ) -> _Var | None:
        names.add(name)
        owner = _resolve_in(name, scope)
        if owner is None:
            fixed.add(name)
            free.add(name)
            return None
        if owner.kind == "class" or name in owner.pinned:
            fixed.add(name)
            return None
        var = (owner.index, name)
        if var not in seen:
            seen.add(var)
            order.append(var)
        occ.setdefault(id(node), []).append((fld, pos, var))
        if binding:
            nbind[var] += 1
            def_node[var] = node
        return var

    for node, scope in walk:
        if node is fn:
            continue
        if isinstance(node, ast.Name):
            load = isinstance(node.ctx, ast.Load)
            var = see(node, "id", -1, node.id, scope, not load)
            call = callee.get(id(node))
            if var is not None and load:
                if call is None or any(k.arg is None for k in call.keywords):
                    escapes.add(var)
                elif call.keywords:
                    kw_called.add(var)
            if var is None and load and call is not None and node.id in _INTROSPECT:
                introspects = True
        elif isinstance(node, ast.arg):
            see(node, "arg", -1, node.arg, scope, True)
        elif isinstance(node, (*_FN, ast.ClassDef)):
            see(node, "name", -1, node.name, scope, True)
        elif isinstance(node, (ast.ExceptHandler, ast.MatchAs, ast.MatchStar)):
            if node.name:
                see(node, "name", -1, node.name, scope, True)
        elif isinstance(node, ast.MatchMapping):
            if node.rest:
                see(node, "rest", -1, node.rest, scope, True)
        elif isinstance(node, ast.alias):
            b = _import_binding(node)
            if b:
                see(node, "asname", -1, b, scope, True)
        elif isinstance(node, ast.Nonlocal):
            for i, n in enumerate(node.names):
                see(node, "names", i, n, scope, False)
        elif isinstance(node, ast.Global):
            fixed.update(node.names)
            free.update(node.names)
            names.update(node.names)
        elif (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.keywords
        ):
            calls.append((node, scope))
    # a keyword at a call of a nested function follows that function's renamed parameter
    for call, scope in calls:
        owner = _resolve_in(call.func.id, scope)
        if owner is None or owner.kind == "class":
            continue
        var = (owner.index, call.func.id)
        d = def_node.get(var)
        if nbind[var] != 1 or not isinstance(d, _FN):
            continue
        f = def_scope[id(d)]
        params = {p.arg for p in (*d.args.args, *d.args.kwonlyargs)}
        for kw in call.keywords:
            if kw.arg in params and (f.index, kw.arg) in seen:
                occ.setdefault(id(kw), []).append(("arg", -1, (f.index, kw.arg)))
    inner = [s for s in scopes if s is not root and s.kind != "module"]
    res = _Resolution(walk, root, inner, occ, order, fixed, free, names)
    res.pinned = (
        set(order)
        if introspects
        else _pinned(fn, res, def_scope, nbind, escapes, kw_called)
    )
    fixed.update(name for _, name in res.pinned)
    return res


def _pinned(
    fn: ast.AST,
    res: _Resolution,
    def_scope: Mapping[int, _Scope],
    nbind: Mapping[_Var, int],
    escapes: set[_Var],
    kw_called: set[_Var],
) -> set[_Var]:
    """Variables whose names are observable, so renaming them could make different functions look equal.

    * a nested class's name and a type alias's name (``__name__``, ``__qualname__``, reprs);
    * a nested function's name and parameters when the function escapes (its variable is read other than as a
      direct callee, called with ``**``, bound more than once or decorated): a caller elsewhere may pass its
      parameters by keyword or read its name. A directly called nested function keeps both renamable; keywords
      at those direct calls follow the renamed parameters;
    * a lambda's parameters unless it is called directly, or bound once to a name only called directly, with
      no keywords either way.
    """
    pinned: set[_Var] = set()

    def var_of(node: ast.AST, fld: str) -> _Var | None:
        for f, _, var in res.occ.get(id(node), ()):
            if f == fld:
                return var
        return None

    def params(node: ast.AST) -> list[_Var]:
        f = def_scope[id(node)]
        a = node.args
        return [(f.index, p.arg) for p in (*a.posonlyargs, *a.args, *a.kwonlyargs)]

    lambda_var: dict[int, _Var] = {}
    direct: set[int] = set()  # lambdas called in place with no keywords
    alias = getattr(ast, "TypeAlias", None)
    for node, _ in res.walk:
        if node is fn:
            continue
        if isinstance(node, ast.ClassDef):
            v = var_of(node, "name")
            if v is not None:
                pinned.add(v)
        elif alias is not None and isinstance(node, alias):
            v = var_of(node.name, "id") if isinstance(node.name, ast.Name) else None
            if v is not None:
                pinned.add(v)
        elif (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and isinstance(node.value, ast.Lambda)
        ):
            v = var_of(node.targets[0], "id")
            if v is not None:
                lambda_var[id(node.value)] = v
        elif (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Lambda)
            and not node.keywords
        ):
            direct.add(id(node.func))
    for node, _ in res.walk:
        if node is fn:
            continue
        if isinstance(node, _FN):
            v = var_of(node, "name")
            if v is None or (
                nbind.get(v, 0) != 1 or v in escapes or node.decorator_list
            ):
                if v is not None:
                    pinned.add(v)
                pinned.update(params(node))
        elif isinstance(node, ast.Lambda):
            v = lambda_var.get(id(node))
            safe = id(node) in direct or (
                v is not None
                and nbind.get(v, 0) == 1
                and v not in escapes
                and v not in kw_called
            )
            if not safe:
                pinned.update(params(node))
    return pinned


def _binds(node: ast.AST, fld: str) -> bool:
    """Whether the occurrence in *node*'s field *fld* binds its name (rather than reading or declaring it)."""
    if fld == "id":
        return not isinstance(node.ctx, ast.Load)
    if fld == "arg":
        return isinstance(node, ast.arg)
    return fld in ("name", "rest", "asname")


def _renamer(occ: Mapping[int, list], names: Mapping[_Var, str]):
    """An ``edit`` for :func:`_copy` giving each resolved occurrence its variable's new name (if it has one)."""

    def edit(orig: ast.AST, new: ast.AST) -> None:
        for fld, pos, var in occ.get(id(orig), ()):
            name = names.get(var)
            if name is None:
                continue
            if fld == "names":
                lst = list(new.names)
                lst[pos] = name
                new.names = lst
            else:
                setattr(new, fld, name)

    return edit


def _free_prefix(base: str, taken: Iterable[str]) -> str:
    """*base*, with leading underscores added until no name in *taken* is it followed by digits."""
    taken = list(taken)
    prefix = base
    while any(t.startswith(prefix) and t[len(prefix) :].isdigit() for t in taken):
        prefix = "_" + prefix
    return prefix


# --- inlining sibling calls ----------------------------------------------------------------------------------


@dataclass
class _Plan:
    """A sibling ready to inline: its parameters, statements and the names it binds and reads."""

    params: list[str]
    defaults: dict[str, ast.expr]
    body: list[ast.stmt]  # without the docstring and the final return
    final: ast.expr | None  # a final return's value worth keeping as a statement
    res: _Resolution
    size: int
    capped: bool  # its own inlining stopped at the node cap
    root_names: frozenset[str] = frozenset()  # names local to its own scope
    inner_bound: frozenset[str] = frozenset()  # names its nested scopes bind


def _constant_default(node: ast.AST) -> bool:
    """A default that is the same value however often it is evaluated: a constant (or a negated one), a tuple of
    those, or a bare name (a module global or builtin, checked against the caller's names separately).
    """
    if isinstance(node, (ast.Constant, ast.Name)):
        return True
    if isinstance(node, ast.UnaryOp) and isinstance(node.operand, ast.Constant):
        return True
    return isinstance(node, ast.Tuple) and all(_constant_default(e) for e in node.elts)


def _plan(callee: ast.AST) -> _Plan | None:
    if not isinstance(callee, ast.FunctionDef) or callee.decorator_list:
        return None
    a = callee.args
    if a.vararg or a.kwarg or a.kwonlyargs or a.posonlyargs:
        return None
    if not all(_constant_default(d) for d in a.defaults):
        return None  # a mutable default (acc=[]) is shared across calls; inlined it would be fresh per call
    try:
        res = _resolve(callee)
    except _Bounds:
        return None
    root = res.root
    if root.globals or root.nonlocals or root.pinned:
        return None
    for node, _ in res.walk:
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id in _INTROSPECT | {"super"}
            and node.func.id in res.free
        ):
            return None  # it reads its own frame: inlined, it would see the caller's
    body = list(callee.body)
    if body and _is_docstring(body[0]):
        body = body[1:]
    final_stmt = body[-1] if body and isinstance(body[-1], ast.Return) else None
    final = None
    if final_stmt is not None:
        body = body[:-1]
        value = final_stmt.value
        if value is not None and not isinstance(value, (ast.Name, ast.Constant)):
            final = value
    params = [p.arg for p in a.args]
    pvars = {(root.index, p) for p in params}
    own_args = {id(p) for p in a.args}
    for node, scope in res.walk:
        if node is final_stmt:
            continue
        if scope is root and isinstance(
            node,
            (ast.Return, ast.Yield, ast.YieldFrom, ast.Await),
        ):
            return (
                None  # an early return or a generator cannot become straight-line code
            )
        if id(node) not in own_args and any(
            var in pvars and _binds(node, fld)
            for fld, _, var in res.occ.get(id(node), ())
        ):
            return None  # a rebound parameter (assignment, loop, except, import, def ...) rebinds the caller's name
    defaults = dict(
        zip(params[len(params) - len(a.defaults) :], a.defaults, strict=True),
    )
    return _Plan(
        params,
        defaults,
        body,
        final,
        res,
        len(res.walk),
        False,
        frozenset(res.root_names),
        frozenset(res.inner_bound),
    )


def _cached_plan(
    name: str,
    library: Mapping[str, ast.AST],
    depth: int,
    stack: tuple[str, ...],
    cache: dict,
) -> _Plan | None:
    key = (name, depth, frozenset(stack) if depth > 1 else None)
    if key in cache:
        return cache[key]
    callee = library[name]
    plan = None
    if isinstance(callee, ast.FunctionDef):
        if depth > 1:
            try:
                deeper, _, capped = _inline(
                    callee,
                    library,
                    depth - 1,
                    stack + (name,),
                    cache,
                )
            except _Bounds:
                deeper, capped = None, False
            plan = _plan(deeper) if deeper is not None else None
            if plan is not None:
                plan.capped = capped
        else:
            plan = _plan(callee)
    cache[key] = plan
    return plan


def _inline(
    fn: ast.FunctionDef | ast.AsyncFunctionDef,
    library: Mapping[str, ast.AST],
    depth: int,
    stack: tuple[str, ...],
    cache: dict,
) -> tuple[ast.FunctionDef | ast.AsyncFunctionDef, dict[int, str], bool]:
    """*fn* with its sibling call statements inlined; the inlined nodes' helper; whether the node cap stopped any."""
    res = _resolve(fn)
    new, _ = _copy(fn, lambda o, n: None)
    root_locals = res.root_names
    stack = stack + (fn.name,)
    total = len(res.walk)
    owner: dict[int, str] = {}
    capped = False
    body: list[ast.stmt] = []
    for k, stmt in enumerate(new.body):
        call = stmt.value if isinstance(stmt, ast.Expr) else None
        name = (
            call.func.id
            if isinstance(call, ast.Call) and isinstance(call.func, ast.Name)
            else None
        )
        plan = None
        if (
            depth >= 1
            and name is not None
            and name in library
            and name
            not in root_locals  # the caller's own local of that name is not the sibling
            and name not in stack  # no recursion, direct or mutual
            and not call.keywords
            and all(isinstance(x, ast.Name) for x in call.args)
        ):
            plan = _cached_plan(name, library, depth, stack, cache)
        args = [x.id for x in call.args] if plan is not None else []
        if plan is not None and not (
            len(plan.params) - len(plan.defaults) <= len(args) <= len(plan.params)
        ):
            plan = None
        if plan is not None and total + plan.size > MAX_NODES:
            capped, plan = True, None  # checked first: it is O(1), the rest is not
        fresh: dict[str, str] = {}
        if plan is not None:
            passed = set(plan.params[: len(args)])
            fresh = {
                n: f"_inl{k}_{n}" for n in sorted(plan.root_names) if n not in passed
            }
            if (
                not root_locals.isdisjoint(
                    plan.res.free,
                )  # a sibling's global read the caller binds
                or not plan.inner_bound.isdisjoint(
                    args,
                )  # a caller name the sibling's inner scopes rebind
                or any(f in res.names or f in plan.res.names for f in fresh.values())
            ):
                plan = None
        if plan is None:
            body.append(stmt)
            continue
        total += plan.size
        capped = capped or plan.capped
        root = plan.res.root.index
        names: dict[_Var, str] = {}
        for i, p in enumerate(plan.params):
            names[(root, p)] = args[i] if i < len(args) else fresh[p]
        for n, f in fresh.items():
            names[(root, n)] = f
        edit = _renamer(plan.res.occ, names)
        pre: list[ast.stmt] = []
        for p in plan.params[len(args) :]:
            value, _ = _copy(plan.defaults[p], edit)
            pre.append(
                ast.Assign(
                    targets=[ast.Name(id=fresh[p], ctx=ast.Store(), **_LOCATION)],
                    value=value,
                    **_LOCATION,
                ),
            )
        for s in plan.body:
            copied, _ = _copy(s, edit)
            pre.append(copied)
        if plan.final is not None:
            value, _ = _copy(plan.final, edit)
            pre.append(ast.Expr(value=value, **_LOCATION))
        for s in pre:
            for node in _preorder(s):
                owner[id(node)] = name
        body.extend(pre)
    new.body = body or [ast.Pass()]
    return new, owner, capped


def inline_library_calls(
    fn: ast.FunctionDef | ast.AsyncFunctionDef,
    library: Mapping[str, ast.AST],
    *,
    depth: int = INLINE_DEPTH,
) -> ast.FunctionDef | ast.AsyncFunctionDef:
    """A copy of *fn* with each top-level ``sibling(name, ...)`` call statement replaced by the sibling's body.

    Inlined only when the call is a whole statement (its value unused), its arguments are plain names, the
    sibling is a plain function with no early return, no yield, no rebound parameter, no default that is not a
    constant or a bare name (a mutable default is shared across calls) and no call of ``locals``, ``vars``,
    ``eval``, ``exec``, ``dir`` or ``super`` (they read the frame), and no name changes meaning (module
    docstring). Parameters become the caller's argument names (defaults for the rest); the
    sibling's other locals get fresh names; a final ``return <expr>`` stays as ``<expr>`` unless it is a bare
    name or constant. *depth* levels, never re-entering a function on the way. Raises ``_Bounds`` when *fn*
    itself is over the caps.
    """
    new, _, _ = _inline(fn, library, max(0, min(depth, MAX_INLINE_DEPTH)), (), {})
    return new


# --- terms ---------------------------------------------------------------------------------------------------


class _Term:
    """A normalised AST node with its label, children, size and digests."""

    __slots__ = (
        "node",
        "orig",
        "cls",
        "label",
        "kids",
        "size",
        "digest",
        "alpha",
        "locs",
        "start",
        "kid_digests",
        "alabel",
        "own",
        "inl",
        "isize",
    )

    def __init__(self, node: ast.AST, orig: ast.AST | None) -> None:
        self.node = node
        self.orig = orig
        self.cls = type(node).__name__
        self.label: tuple = ()
        # (field, kind, value): kind "one" (term or None), "list", "plist"
        self.kids: tuple = ()
        self.size = 1
        self.digest = b""
        self.alpha: bytes | None = None
        self.locs: tuple[str, ...] | None = ()
        self.start = 0
        self.kid_digests: Counter = Counter()
        self.alabel: tuple = ()  # the label with local names as slots
        self.own: tuple[
            str,
            ...,
        ] = ()  # the local name this node binds or reads in its own fields
        self.inl: str | None = None  # the sibling this node was inlined from
        self.isize = 0  # inlined nodes in the subtree

    @property
    def category(self) -> str:
        if isinstance(self.node, ast.stmt):
            return "stmt"
        if isinstance(self.node, ast.expr):
            return "expr"
        if isinstance(self.node, ast.keyword):
            return "keyword"
        return "other"

    def children(self) -> list["_Term"]:
        out = []
        for _, kind, value in self.kids:
            if kind == "one":
                if value is not None:
                    out.append(value)
            else:
                out.extend(v for v in value if v is not None)
        return out


def _scalar(value: Any) -> Any:
    """A bounded structural key for a non-node field: type and value for small literals, else type, length and
    digest. Never ``repr`` of an arbitrary literal (a 4,000-digit integer's ``repr`` raises).
    """
    if isinstance(value, ast.AST):  # a folded singleton
        return type(value).__name__
    if isinstance(value, list):
        return tuple(_scalar(v) for v in value)
    if value is None or value is ...:
        return (type(value).__name__,)
    if isinstance(value, bool):
        return ("bool", value)
    if isinstance(value, int):
        if -(2**63) <= value < 2**63:
            return ("int", value)
        n = value.bit_length()
        return ("int", n, _blob(value.to_bytes(n // 8 + 1, "big", signed=True)))
    if isinstance(value, float):
        return ("float", struct.pack(">d", value).hex())
    if isinstance(value, complex):
        return ("complex", struct.pack(">dd", value.real, value.imag).hex())
    if isinstance(value, str):
        if len(value) <= SHORT_LITERAL:
            return ("str", value)
        return ("str", len(value), _blob(value.encode("utf-8", "surrogatepass")))
    if isinstance(value, (bytes, bytearray)):
        if len(value) <= SHORT_LITERAL:
            return ("bytes", bytes(value).hex())
        return ("bytes", len(value), _blob(bytes(value)))
    return (type(value).__name__,)


def _build_terms(
    root: ast.AST,
    origin: Mapping[int, ast.AST],
    local: set[str],
    owner: Mapping[int, str] | None = None,
) -> _Term:
    """Terms for the normalised tree *root*, bottom-up and iterative."""
    owner = owner or {}
    stack: list[tuple[ast.AST, bool]] = [(root, False)]
    made: dict[int, _Term] = {}
    while stack:
        node, visited = stack.pop()
        if not visited:
            stack.append((node, True))
            for child in reversed(list(_children(node))):
                if not isinstance(child, _FOLDED):
                    stack.append((child, False))
            continue
        t = _Term(node, origin.get(id(node)))
        if t.orig is not None:
            t.inl = owner.get(id(t.orig))
        positional = t.cls in _POSITIONAL
        label: list[Any] = [t.cls]
        alabel: list[Any] = [t.cls]
        own: list[str] = []
        kids: list[tuple[str, str, Any]] = []
        bind_field = _BINDING_FIELDS.get(t.cls)
        for name in node._fields:
            if name in ("type_comment", "kind"):
                continue
            value = getattr(node, name, None)
            if isinstance(value, ast.AST) and not isinstance(value, _FOLDED):
                kids.append((name, "one", made.pop(id(value))))
            elif isinstance(value, list) and any(
                isinstance(v, ast.AST) and not isinstance(v, _FOLDED) for v in value
            ):
                terms = tuple(
                    made.pop(id(v)) if isinstance(v, ast.AST) else None for v in value
                )
                if positional or any(v is None for v in terms):
                    label.append((name, len(terms)))
                    alabel.append((name, len(terms)))
                    kids.append((name, "plist", terms))
                else:
                    kids.append((name, "list", terms))
            elif isinstance(value, list) and not value:
                kids.append((name, "list", ()))
            elif value is None and name in _child_fields(node):
                kids.append((name, "one", None))
            else:
                s = _scalar(value)
                label.append((name, s))
                if name == bind_field and isinstance(value, str) and value in local:
                    alabel.append((name, _SLOT))
                    own.append(value)
                else:
                    alabel.append((name, s))
        t.label = tuple(label)
        t.alabel = tuple(alabel)
        t.own = tuple(own)
        t.kids = tuple(kids)
        children = t.children()
        t.size = 1 + sum(c.size for c in children)
        t.isize = (1 if t.inl is not None else 0) + sum(c.isize for c in children)
        t.digest = _digest(
            (
                t.label,
                tuple(
                    (
                        f,
                        k,
                        (
                            (v.digest if v is not None else None)
                            if k == "one"
                            else tuple(x.digest if x is not None else None for x in v)
                        ),
                    )
                    for f, k, v in kids
                ),
            ),
        )
        t.kid_digests = Counter(c.digest for c in children)
        # the renaming-invariant key: locals become slots numbered by first appearance within the subtree
        locs: list[str] = []
        seen_locs: set[str] = set()
        ok = True
        for group in (own, *(c.locs for c in children)):
            if group is None or len(locs) > MAX_LOCALS:  # stop merging past the cap
                ok = False
                break
            for n in group:
                if n not in seen_locs:
                    seen_locs.add(n)
                    locs.append(n)
        if ok and len(locs) <= MAX_LOCALS:
            index = {n: i for i, n in enumerate(locs)}
            parts = []
            for f, k, v in kids:
                vs = [v] if k == "one" else list(v)
                parts.append(
                    (
                        f,
                        k,
                        tuple(
                            (
                                None
                                if x is None
                                else (x.alpha, tuple(index[n] for n in x.locs))
                            )
                            for x in vs
                        ),
                    ),
                )
            t.locs = tuple(locs)
            t.alpha = _digest(
                ("A", tuple(alabel), tuple(index[n] for n in own), tuple(parts)),
            )
        else:
            t.locs, t.alpha = None, None
        made[id(node)] = t
    root_term = made.pop(id(root))
    # pre-order positions: a subtree is the interval [start, start + size)
    order = 0
    stack2 = [root_term]
    while stack2:
        t = stack2.pop()
        t.start = order
        order += 1
        stack2.extend(reversed(t.children()))
    return root_term


def _term_preorder(t: _Term) -> Iterable[_Term]:
    stack = [t]
    while stack:
        cur = stack.pop()
        yield cur
        stack.extend(reversed(cur.children()))


def _child_fields(node: ast.AST) -> set[str]:
    """Fields of *node*'s type that hold a child node when set (so ``None`` there is an absent child)."""
    return _OPTIONAL_CHILD.get(type(node).__name__, set())


_OPTIONAL_CHILD: dict[str, set[str]] = {
    "Return": {"value"},
    "AnnAssign": {"value"},
    "Raise": {"exc", "cause"},
    "Assert": {"msg"},
    "ExceptHandler": {"type"},
    "Yield": {"value"},
    "Slice": {"lower", "upper", "step"},
    "withitem": {"optional_vars"},
    "comprehension": set(),
    "arguments": {"vararg", "kwarg"},
    "arg": {"annotation"},
    "FunctionDef": {"returns"},
    "AsyncFunctionDef": {"returns"},
    "FormattedValue": {"format_spec"},
    "match_case": {"guard"},
    "MatchAs": {"pattern"},
    "TypeVar": {"bound", "default_value"},
    "ParamSpec": {"default_value"},
    "TypeVarTuple": {"default_value"},
}


# --- normalisation -------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Normalised:
    """A function after normalisation: the tree, its size in counted nodes, and canonical -> original names."""

    node: ast.FunctionDef | ast.AsyncFunctionDef
    size: int
    renames: dict[str, str]
    term: _Term
    fixed: frozenset[str] = (
        frozenset()
    )  # names kept as written (free, global, class-level, dotted imports)
    inlined: tuple[str, ...] = ()  # siblings inlined into it
    owned: int = 0  # nodes that are the function's own code (not inlined)
    inline_capped: bool = False  # the node cap stopped an inlining


def normalise(
    fn: ast.FunctionDef | ast.AsyncFunctionDef,
    *,
    library: Mapping[str, ast.AST] | None = None,
    inline_depth: int = INLINE_DEPTH,
    cache: dict | None = None,
) -> Normalised | None:
    """*fn* normalised (module docstring); with *library*, sibling call statements inlined first.

    ``None`` when the function is over :data:`MAX_NODES` or :data:`MAX_DEPTH` (or a literal or the parser's
    own limits make it unreadable).
    """
    try:
        depth = max(0, min(inline_depth, MAX_INLINE_DEPTH))
        if library and depth:
            src, owner, capped = _inline(
                fn,
                library,
                depth,
                (),
                {} if cache is None else cache,
            )
        else:
            src, owner, capped = fn, {}, False
        res = _resolve(src)
        prefix = _free_prefix("_l", res.fixed)
        renamed = [var for var in res.order if var not in res.pinned]
        names = {var: f"{prefix}{i}" for i, var in enumerate(renamed)}
        rename = _renamer(res.occ, names)

        def edit(orig: ast.AST, new: ast.AST) -> None:
            rename(orig, new)
            if isinstance(new, _SCOPES) and new.body and _is_docstring(new.body[0]):
                new.body = new.body[1:] or [ast.Pass()]
            if isinstance(new, ast.arg):
                new.annotation = None
            elif isinstance(new, _FN):
                new.returns = None
            elif isinstance(new, ast.AnnAssign):
                new.annotation = ast.Constant(value=None)

        node, origin = _copy(src, edit)
        node.name = "_fn"
        term = _build_terms(node, origin, set(names.values()), owner)
    except (_Bounds, ValueError, RecursionError, MemoryError):
        return None
    return Normalised(
        node,
        term.size,
        {names[var]: var[1] for var in renamed},
        term,
        frozenset(res.fixed),
        tuple(sorted(set(owner.values()))),
        term.size - term.isize,
        capped,
    )


# --- the generalisation --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Hole:
    """Hole ``__hole<index>__``: what each input has there (canonical names; ``""`` for nothing) and its size.

    *flows* is, per input, the hole's place in that input's data flow (``None`` where the input has nothing
    there; a truncated analysis gives the conservative answer, kind ``nonlocal``).
    """

    index: int
    bindings: tuple[str, ...]
    sizes: tuple[int, ...]
    name: str = ""
    flows: tuple[RegionFlow | None, ...] = ()

    @property
    def kind(self) -> str:
        """``value``, ``nonlocal``, ``guard`` or ``local`` (:attr:`.fn_dataflow.RegionFlow.kind`) when every input
        with something here agrees; else ``mixed`` (``empty`` when no input has anything, ``unknown`` when no
        flow is known). Only ``local`` (in every input) marks a hole whose difference cannot reach a return or
        raise; ``nonlocal`` counts as value-determining."""
        present = [k for k, s in enumerate(self.sizes) if s]
        if not present:
            return "empty"
        kinds = {
            self.flows[k].kind for k in present if k < len(self.flows) and self.flows[k]
        }
        if not kinds:
            return "unknown"
        return kinds.pop() if len(kinds) == 1 else "mixed"


@dataclass(frozen=True)
class Generalisation:
    source: str | None  # the generalised function (``None`` if it cannot be rendered)
    node: ast.AST
    holes: tuple[Hole, ...]
    kept: int  # nodes of the generalisation that are not holes
    sizes: tuple[int, ...]  # each normalised input's size
    kept_share: float  # kept / mean(sizes)
    bounded: bool  # the work budget ran out and the rest became holes
    renames: tuple[
        dict[str, str],
        ...,
    ]  # per input: canonical name -> the input's own name
    # the generalisation's paired names -> each input's canonical name
    pairs: dict[str, tuple[str, ...]] = field(default_factory=dict)
    owned_share: float = 0.0  # kept caller-owned nodes / mean caller-owned size
    helper_kept: int = 0  # kept nodes that are inlined sibling code in every input
    mixed_kept: int = (
        0  # kept nodes inlined in some inputs and the caller's own in others
    )
    shared_helpers: tuple[str, ...] = ()  # siblings inlined into every input
    kept_share_uninlined: float | None = (
        None  # the kept share without inlining (when something was inlined)
    )
    inline_capped: bool = False
    flows: tuple[
        FlowSummary | None,
        ...,
    ] = ()  # per input: parameters -> return and raise sites
    # (i, j): input i calls input j by name (a free name, as written, before inlining)
    calls: tuple[tuple[int, int], ...] = ()
    # each input's own parameter list as written ("/", "*" and "**" markers kept): the call interface
    signatures: tuple[tuple[str, ...], ...] = ()

    @property
    def per_input_share(self) -> tuple[float, ...]:
        return tuple(self.kept / s if s else 0.0 for s in self.sizes)

    @property
    def helper_driven(self) -> bool:
        """At least half of what is kept is a shared helper's inlined code: extract or reuse, not merge."""
        return self.kept > 0 and 2 * self.helper_kept >= self.kept

    @property
    def wrapper(self) -> bool:
        """One input calls another: a wrapper and what it wraps (keep or inline the wrapper), not two peers to
        merge. Its kept share mostly counts the callee's code inlined into the caller (``mixed_kept``).
        """
        return bool(self.calls)

    @property
    def same_signature(self) -> bool:
        """Every input takes the same parameters under the same names (callers pass recorded keywords by name,
        so a renamed root parameter is an interface difference even when the bodies generalise fully).
        """
        return len(set(self.signatures)) <= 1


_MISMATCH = object()


class _State:
    def __init__(self, n: int, budget: int, mprefix: str = "_m") -> None:
        self.n = n
        self.budget = budget
        self.work = 0
        self.bounded = False
        self.mprefix = mprefix
        self.holes: dict[str, tuple[tuple[_Term, ...], ...]] = {}
        # consistent renaming: a tuple of local names (one per input) -> the generalisation's name
        self.var: dict[tuple[str, ...], str] = {}
        self.used: list[dict[str, tuple[str, ...]]] = [{} for _ in range(n)]
        self.log: list[tuple[str, ...]] = (
            []
        )  # pairings in the order made, to undo an abandoned branch
        self.fresh = 0
        # id(generalised or kept node) -> (node, the input terms it stands for, kept whole)
        self.made: dict[int, tuple[ast.AST, Sequence[_Term], bool]] = {}

    def mark(self) -> int:
        return len(self.log)

    def undo(self, mark: int) -> None:
        while len(self.log) > mark:
            names = self.log.pop()
            del self.var[names]
            for k, name in enumerate(names):
                if self.used[k].get(name) == names:
                    del self.used[k][name]

    def pair(self, names: tuple[str, ...]) -> str | None:
        """The generalisation's name for these inputs' locals, or None if one is already paired otherwise."""
        if names in self.var:
            return self.var[names]
        if any(name in self.used[k] for k, name in enumerate(names)):
            return None
        if len(set(names)) == 1:
            var = names[0]
        else:
            var = f"{self.mprefix}{self.fresh}"
            self.fresh += 1
        for k, name in enumerate(names):
            self.used[k][name] = names
        self.var[names] = var
        self.log.append(names)
        return var

    def pair_whole(self, t: _Term) -> bool:
        """Whether the identical subtree *t* can be kept whole: each of its locals pairs with itself."""
        if t.locs is None:
            return False
        for name in t.locs:
            same = (name,) * self.n
            if any(self.used[k].get(name, same) != same for k in range(self.n)):
                return False
        for name in t.locs:
            self.pair((name,) * self.n)
        return True

    def spend(self, units: int) -> bool:
        self.work += units
        if self.work > self.budget:
            self.bounded = True
            return False
        return True

    def hole(self, runs: Sequence[Sequence[_Term]]) -> ast.Name:
        token = f"\x00hole{len(self.holes)}"
        self.holes[token] = tuple(tuple(r) for r in runs)
        return ast.Name(id=token, ctx=ast.Load())

    def note(self, node: ast.AST, ts: Sequence[_Term], whole: bool) -> None:
        self.made[id(node)] = (node, ts, whole)


def _wrap(category: str, name: ast.Name, starred: bool = False) -> Any:
    if category == "stmt":
        return ast.Expr(value=name)
    if category == "keyword":
        return ast.keyword(arg=None, value=name)
    if category == "expr":
        return ast.Starred(value=name, ctx=ast.Load()) if starred else name
    return _MISMATCH


def _hole_or_mismatch(ts: Sequence[_Term], st: _State) -> Any:
    if (
        ts[0].cls == "arguments"
    ):  # different parameter lists: ``*__holeN__`` stands for them
        token = st.hole([(t,) for t in ts]).id
        return (
            ast.arguments(
                posonlyargs=[],
                args=[],
                vararg=ast.arg(arg=token),
                kwonlyargs=[],
                kw_defaults=[],
                kwarg=None,
                defaults=[],
            ),
            0,
        )
    category = ts[0].category
    if category not in ("stmt", "expr", "keyword"):
        return _MISMATCH
    return (_wrap(category, st.hole([(t,) for t in ts])), 0)


def _score(x: _Term, y: _Term) -> int:
    if x.alabel != y.alabel:
        return 0
    if x.digest == y.digest:
        return 3 * x.size
    shared = x.kid_digests & y.kid_digests
    return 1 + sum(shared.values())


def _align(
    xs: Sequence[_Term],
    ys: Sequence[_Term],
    st: _State,
) -> list[tuple[int, int]]:
    n, m = len(xs), len(ys)
    if not n or not m:
        return []
    if n * m > DP_CELLS or not st.spend(n * m):
        pairs = []
        i = 0
        while i < min(n, m) and xs[i].alabel == ys[i].alabel:
            pairs.append((i, i))
            i += 1
        tail = []
        j = 1
        while j <= min(n, m) - i and xs[n - j].alabel == ys[m - j].alabel:
            tail.append((n - j, m - j))
            j += 1
        return pairs + tail[::-1]
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    score = [[0] * m for _ in range(n)]
    for i in range(n - 1, -1, -1):
        row, below = dp[i], dp[i + 1]
        for j in range(m - 1, -1, -1):
            s = _score(xs[i], ys[j])
            score[i][j] = s
            best = max(below[j], row[j + 1])
            if s and below[j + 1] + s > best:
                best = below[j + 1] + s
            row[j] = best
    pairs, i, j = [], 0, 0
    while i < n and j < m:
        s = score[i][j]
        if s and dp[i][j] == dp[i + 1][j + 1] + s:
            pairs.append((i, j))
            i += 1
            j += 1
        elif dp[i + 1][j] == dp[i][j]:
            i += 1
        else:
            j += 1
    return pairs


def _gen(ts: Sequence[_Term], st: _State, depth: int) -> Any:
    """The generalisation of *ts* as ``(node, kept)``, or ``_MISMATCH`` for the parent to absorb."""
    first = ts[0]
    if all(t.digest == first.digest for t in ts) and st.pair_whole(first):
        st.note(first.node, ts, True)
        return (first.node, first.size)
    if depth > MAX_DEPTH + 2 or not st.spend(1):
        return _hole_or_mismatch(ts, st)
    if first.cls in _WHOLE or any(t.alabel != first.alabel for t in ts):
        return _hole_or_mismatch(ts, st)
    mark = st.mark()
    renamed: dict[str, str] = {}
    if first.own:  # a local name here: paired consistently across the inputs, or a hole
        var = st.pair(tuple(t.own[0] for t in ts))
        if var is None:
            return _hole_or_mismatch(ts, st)
        renamed[_BINDING_FIELDS[first.cls]] = var
    # equal labels: the same type with the same field layout (list lengths of positional fields are in the label)
    fields: dict[str, Any] = {}
    kept = 1
    for idx, (name, kind, _) in enumerate(first.kids):
        values = [t.kids[idx][2] for t in ts]
        if kind == "one":
            r = _gen_one(values, st, depth)
        elif kind == "plist":
            r = _gen_positional(values, st, depth)
        else:
            r = _gen_list(first.cls, name, values, st, depth)
        if r is _MISMATCH:
            st.undo(
                mark,
            )  # the pairings made below this node belong to the hole's bindings now
            return _hole_or_mismatch(ts, st)
        fields[name], k = r
        kept += k
    for name in first.node._fields:
        if name not in fields:
            fields[name] = renamed.get(name, getattr(first.node, name, None))
    node = type(first.node)(**fields)
    # locations: ast.unparse reads a definition's line
    for attr in first.node._attributes:
        if hasattr(first.node, attr):
            setattr(node, attr, getattr(first.node, attr))
    st.note(node, ts, False)
    return (node, kept)


def _gen_one(values: Sequence[_Term | None], st: _State, depth: int) -> Any:
    present = [v for v in values if v is not None]
    if not present:
        return (None, 0)
    if len(present) < len(values):
        if present[0].category != "expr":
            return _MISMATCH
        return (st.hole([() if v is None else (v,) for v in values]), 0)
    return _gen(values, st, depth + 1)


def _gen_positional(values: Sequence[tuple], st: _State, depth: int) -> Any:
    out, kept = [], 0
    for column in zip(*values, strict=True):  # equal lengths: they are in the label
        r = _gen_one(column, st, depth)
        if r is _MISMATCH:
            return _MISMATCH
        out.append(r[0])
        kept += r[1]
    return (out, kept)


def _gen_list(
    cls: str,
    field: str,
    lists: Sequence[tuple],
    st: _State,
    depth: int,
) -> Any:
    """Align the lists, generalise each aligned column, and make each unaligned run a gap."""
    n = len(lists)
    cols: list[tuple[int, ...]] = [(i,) for i in range(len(lists[0]))]
    for k in range(1, n):
        reps = [lists[0][c[0]] for c in cols]
        cols = [cols[i] + (j,) for i, j in _align(reps, lists[k], st)]
    end = tuple(len(lst) for lst in lists)
    out: list[Any] = []
    kept = 0
    prev = (-1,) * n
    for idx in range(len(cols) + 1):
        c = cols[idx] if idx < len(cols) else end
        runs = [lists[k][prev[k] + 1 : c[k]] for k in range(n)]
        if any(runs):
            r = _gen_gap(cls, field, runs, st, depth)
            if r is _MISMATCH:
                return _MISMATCH
            out.extend(r[0])
            kept += r[1]
        if idx < len(cols):
            r = _gen([lists[k][c[k]] for k in range(n)], st, depth + 1)
            if r is _MISMATCH:
                return _MISMATCH
            out.append(r[0])
            kept += r[1]
        prev = c
    return (out, kept)


def _gen_gap(
    cls: str,
    field: str,
    runs: Sequence[tuple],
    st: _State,
    depth: int,
) -> Any:
    """A run of unaligned elements between two aligned columns (or an end), one run per input."""
    if (
        len({len(r) for r in runs}) == 1
    ):  # the same count everywhere: generalise position by position
        mark = st.mark()
        zipped, kept = [], 0
        for column in zip(*runs, strict=True):
            r = _gen(column, st, depth + 1)
            if r is _MISMATCH:
                st.undo(mark)
                break
            zipped.append(r[0])
            kept += r[1]
        else:
            return (zipped, kept)
    sample = next(t for r in runs for t in r)
    wrapped = _wrap(sample.category, st.hole(runs), (cls, field) in _STARRED_GAPS)
    if wrapped is _MISMATCH:
        return _MISMATCH
    return ([wrapped], 0)


def _render(node: ast.AST) -> str | None:
    try:
        return ast.unparse(node)
    except (RecursionError, ValueError, TypeError, AttributeError):
        return None


def _number_holes(
    root: ast.AST,
    st: _State,
    hole_prefix: str = "__",
) -> tuple[tuple[Hole, ...], dict[str, int]]:
    """Number the holes in pre-order (the same disagreement, the same number); token -> number as well."""
    numbers: dict[tuple, int] = {}
    tokens: dict[str, int] = {}
    holes: list[Hole] = []
    for node in _preorder(root):
        fld = (
            "id"
            if isinstance(node, ast.Name)
            else "arg" if isinstance(node, ast.arg) else None
        )
        token = getattr(node, fld) if fld else None
        if token not in st.holes:
            continue
        runs = st.holes[token]
        key = tuple(tuple(t.digest for t in r) for r in runs)
        if key not in numbers:
            numbers[key] = len(numbers)
            bindings = []
            for r in runs:
                sep = "\n" if r and r[0].category == "stmt" else ", "
                parts = [_render(t.node) for t in r]
                bindings.append(
                    sep.join(p if p is not None else "<deep>" for p in parts),
                )
            holes.append(
                Hole(
                    numbers[key],
                    tuple(bindings),
                    tuple(sum(t.size for t in r) for r in runs),
                    f"{hole_prefix}hole{numbers[key]}__",
                ),
            )
        tokens[token] = numbers[key]
        setattr(node, fld, f"{hole_prefix}hole{numbers[key]}__")
    return tuple(holes), tokens


def _ownership(root: ast.AST, st: _State) -> tuple[int, int, int]:
    """Kept nodes that are (caller-owned in every input, inlined in every input, mixed)."""
    owned = inlined = mixed = 0
    stack = [root]
    while stack:
        node = stack.pop()
        entry = st.made.get(id(node))
        if entry is not None and entry[0] is node:
            _, ts, whole = entry
            groups = (
                zip(*(_term_preorder(t) for t in ts), strict=True) if whole else [ts]
            )
            for group in groups:
                flags = [t.inl is not None for t in group]
                if all(flags):
                    inlined += 1
                elif any(flags):
                    mixed += 1
                else:
                    owned += 1
            if whole:
                continue
        stack.extend(_children(node))
    return owned, inlined, mixed


def _signature(fn: ast.FunctionDef | ast.AsyncFunctionDef) -> tuple[str, ...]:
    a = fn.args
    out = [p.arg for p in a.posonlyargs]
    if a.posonlyargs:
        out.append("/")
    out += [p.arg for p in a.args]
    if a.vararg:
        out.append("*" + a.vararg.arg)
    elif a.kwonlyargs:
        out.append("*")
    out += [p.arg for p in a.kwonlyargs]
    if a.kwarg:
        out.append("**" + a.kwarg.arg)
    return tuple(out)


def _calls_between(fns: Sequence[ast.AST]) -> tuple[tuple[int, int], ...]:
    """(i, j) where input i calls input j by a name it does not bind itself."""
    out = []
    for i, f in enumerate(fns):
        called: set[str] = set()
        bound: set[str] = set()
        for n in ast.walk(f):
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Name):
                called.add(n.func.id)
            elif isinstance(n, ast.Name) and not isinstance(n.ctx, ast.Load):
                bound.add(n.id)
            elif isinstance(n, ast.arg):
                bound.add(n.arg)
            elif isinstance(n, (*_FN, ast.ClassDef)) and n is not f:
                bound.add(n.name)
        for j, g in enumerate(fns):
            name = getattr(g, "name", None)
            if (
                j != i
                and name != getattr(f, "name", None)
                and name in called
                and name not in bound
            ):
                out.append((i, j))
    return tuple(out)


def antiunify(
    fns: Sequence[ast.FunctionDef | ast.AsyncFunctionDef],
    *,
    library: Mapping[str, ast.AST] | None = None,
    budget: int = WORK_BUDGET,
    inline_depth: int = INLINE_DEPTH,
) -> Generalisation | None:
    """The most specific generalisation of *fns* (two or more); ``None`` if any is over the bounds.

    With *library* (name -> function), sibling call statements are inlined before comparing (*inline_depth*
    levels), and the kept share without inlining is reported beside it.
    """
    if len(fns) < 2:
        return None
    cache: dict = {}
    norms = [
        normalise(f, library=library, inline_depth=inline_depth, cache=cache)
        for f in fns
    ]
    if any(n is None for n in norms):
        return None
    taken: set[str] = set()
    for n in norms:
        taken |= n.fixed
    st = _State(len(norms), budget, _free_prefix("_m", taken))
    r = _gen([n.term for n in norms], st, 0)
    node, kept = r  # a function is a statement, so never a mismatch
    if (
        isinstance(node, ast.Name) and node.id in st.holes
    ):  # the whole function is a hole
        node, kept = ast.Expr(value=node), 0
    hole_prefix = "__"
    while any(t.startswith(hole_prefix + "hole") for t in taken):
        hole_prefix = "_" + hole_prefix
    holes, tokens = _number_holes(node, st, hole_prefix)
    # each hole's data flow in each input
    regions: list[dict[int, list[ast.AST]]] = [{} for _ in norms]
    for token, number in tokens.items():
        for k, run in enumerate(st.holes[token]):
            if run:
                regions[k].setdefault(number, []).extend(t.node for t in run)
    summaries = tuple(_flows(n.node, regions[k]) for k, n in enumerate(norms))
    holes = tuple(
        replace(h, flows=tuple(s.regions.get(h.index) for s in summaries))
        for h in holes
    )
    sizes = tuple(n.size for n in norms)
    mean = sum(sizes) / len(sizes)
    owned, helper, mixed = _ownership(node, st)
    owned_mean = sum(n.owned for n in norms) / len(norms)
    shared = set(norms[0].inlined)
    for n in norms[1:]:
        shared &= set(n.inlined)
    uninlined = None
    if library and any(n.inlined for n in norms):
        raw = antiunify(fns, budget=budget)
        uninlined = raw.kept_share if raw is not None else None
    return Generalisation(
        source=_render(node),
        node=node,
        holes=holes,
        kept=kept,
        sizes=sizes,
        kept_share=kept / mean if mean else 0.0,
        bounded=st.bounded,
        renames=tuple(n.renames for n in norms),
        pairs={v: names for names, v in st.var.items() if len(set(names)) > 1},
        owned_share=owned / owned_mean if owned_mean else 0.0,
        helper_kept=helper,
        mixed_kept=mixed,
        shared_helpers=tuple(sorted(shared)),
        kept_share_uninlined=uninlined,
        inline_capped=any(n.inline_capped for n in norms),
        flows=summaries,
        calls=_calls_between(fns),
        signatures=tuple(_signature(f) for f in fns),
    )


def antiunify_source(
    source: str,
    names: Sequence[str],
    *,
    inline: bool = True,
    budget: int = WORK_BUDGET,
    inline_depth: int = INLINE_DEPTH,
) -> Generalisation | None:
    """Anti-unify the top-level functions *names* of a module's *source* (sibling calls inlined with *inline*)."""
    defs = function_defs(source)
    if any(n not in defs for n in names):
        return None
    return antiunify(
        [defs[n] for n in names],
        library=defs if inline else None,
        budget=budget,
        inline_depth=inline_depth,
    )
