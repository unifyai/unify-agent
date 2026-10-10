"""Intra-function data flow (memory hygiene, stage 4): which parameters reach which return and raise sites.

**Direction: over-approximate, sound for flagging.** The analysis may report flows that cannot happen; it must
not miss one that can, under the assumptions below. A region is called ``local`` only when nothing it does can
reach a return or raise site; whatever the analysis does not follow makes it ``nonlocal`` (treated as
value-determining), never ``local``. So ``local``, the one label that could justify ignoring a hole, is the
conservative one.

A def-use graph over one function's AST (normalised by :mod:`.fn_antiunify`, so every scope's locals have
distinct names), flow-insensitive:

* a **definition** binds a name: an assignment, augmented or annotated assignment target; the base name of a
  subscript or attribute store (``x[i] = v`` changes ``x``); a ``for``, ``with``, ``except``, ``match``,
  comprehension, lambda or walrus target; an import; a nested ``def`` or ``class``. It depends on every name the
  statement reads and on the statement's **control context**;
* **calls mutate**: every name passed to a call (any argument, at any depth of the expression) and the receiver
  of a method call (``v`` in ``v.m(...)``) are defined by the statement, unless the callee is one of a few
  builtins that only read their arguments (``len``, ``isinstance``, ...); iterating a name (``for``, a
  comprehension) defines it too (an iterator is consumed);
* **aliases are two-way**: a statement that binds names from an expression that may return one of the names it
  reads (``y = x``, ``y = x[0]``, ``y = f(x)``, ``for y in xs``, ``with x as y``; not ``y = x + 1``) makes those
  names depend on the bound ones, so mutating ``y`` later changes ``x``;
* the **control context** of a statement is what governs whether it runs: ``if`` and ``while`` tests, ``for``
  iterables, ``with`` items, ``match`` subjects, earlier cases and guards, ``except`` types and everything the
  ``try`` body reads (any of it may raise into the handler); and, for the statements **after** a compound
  statement that can ``return``, ``break`` or ``continue``, every test inside it (an early exit decides whether
  they run). A statement that can only ``raise`` does not govern what follows: the value returned, when one is,
  does not depend on it (that is what makes a region a ``guard``);
* a **site** is one of the function's own ``return`` statements (its value) or ``raise``/``assert`` statements,
  with the same dependencies (a nested function's sites are not the function's);
* **regions**, caller-supplied root nodes (the subtrees an anti-unification hole binds), are pseudo-names: a
  region depends on what it reads and on its context, and whatever contains it or reads what it defines depends
  on it. A region at the parameter list stands for every parameter.

One bit-set fixpoint gives every name, region and site the parameters and regions that reach it. A region's flow
is then the parameters that reach it, the return and raise sites it reaches, and whether it has **untracked**
effects: a call (other than the read-only builtins, or the exception a ``raise`` constructs), an attribute or
subscript store or delete, a ``del``, a ``global``/``nonlocal`` declaration or a binding of such a name, a
``yield`` or ``await``, an import, a ``with``, a ``break``/``continue``, an ``async for``, a class or a decorated
function. Its *kind* is ``value`` (it reaches a return), else ``nonlocal`` (untracked effects: treated as
value-determining), else ``guard`` (it reaches only raises), else ``local``.

**Assumptions** (stated, not checked): operators, comparisons and attribute or subscript *reads* run no code
with effects and return fresh values (``x + y`` aliases neither); the read-only builtins are the builtins when
the function does not bind their names.

**Bounds**, all checked before or during the work, never silent: a function over :data:`MAX_NODES` nodes is not
analysed; the graph build and fixpoint stop past :data:`MAX_WORK` steps (nodes visited plus dependency edges),
the fixpoint past :data:`MAX_STEPS` updates, and everything past a wall-clock budget (:data:`MAX_SECONDS`). Each
gives ``truncated=True`` (with the limit named) and every region the conservative answer: every parameter may
reach it, kind ``nonlocal``. Control contexts and statements are graph nodes of their own, so a binder adds one
edge, not one per name in its context: the build is linear in the function's size. Pure and deterministic
(the time budget aside). Standard library only.
"""

from __future__ import annotations

import ast
import time
from collections import deque
from dataclasses import dataclass
from typing import Hashable, Iterable, Mapping

# AST nodes per function as ast.walk counts them (contexts and operators too), checked first
MAX_NODES = 20_000
# nodes visited plus dependency edges added or evaluated, per function
MAX_WORK = 2_000_000
MAX_STEPS = 400_000  # fixpoint updates, per function
MAX_SECONDS = 1.0  # wall clock, per function
_CLOCK_EVERY = 4096  # work steps between clock reads

_FOLDED = (ast.expr_context, ast.boolop, ast.operator, ast.unaryop, ast.cmpop)
_NESTED = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
# expressions whose value is fresh: they alias none of the names they read (see Assumptions)
_FRESH = (
    ast.BinOp,
    ast.UnaryOp,
    ast.Compare,
    ast.JoinedStr,
    ast.FormattedValue,
    ast.Constant,
)
# builtins that only read their arguments and return a fresh or immutable value
_READ_ONLY = frozenset(
    {
        "abs",
        "ascii",
        "bin",
        "bool",
        "callable",
        "chr",
        "complex",
        "divmod",
        "float",
        "format",
        "hash",
        "hex",
        "id",
        "int",
        "isinstance",
        "issubclass",
        "len",
        "oct",
        "ord",
        "pow",
        "repr",
        "round",
        "str",
        "type",
    },
)
# builtins whose only effect is on their arguments (they may consume an iterator): tracked, not untracked,
# when called with no keywords and no lambda
_ARGS_ONLY = frozenset(
    {
        "all",
        "any",
        "dict",
        "enumerate",
        "frozenset",
        "list",
        "max",
        "min",
        "reversed",
        "set",
        "sorted",
        "sum",
        "tuple",
        "zip",
    },
)
_UNTRACKED = (
    ast.Delete,
    ast.Global,
    ast.Nonlocal,
    ast.Yield,
    ast.YieldFrom,
    ast.Await,
    ast.Import,
    ast.ImportFrom,
    ast.Break,
    ast.Continue,
    ast.With,
    ast.AsyncWith,
    ast.AsyncFor,
    ast.ClassDef,
)


class _Over(Exception):
    """A bound was reached; ``args[0]`` names it."""


@dataclass(frozen=True)
class RegionFlow:
    """How one region sits in its function's data flow."""

    params: tuple[
        int,
        ...,
    ]  # positions of the parameters that reach the region (data or control)
    returns: int  # return sites the region reaches
    raises: int  # raise and assert sites the region reaches
    # it has effects the graph does not follow (module docstring)
    untracked: bool = False
    # the analysis was cut short: this is the conservative answer
    truncated: bool = False

    @property
    def kind(self) -> str:
        """``value``, ``nonlocal``, ``guard`` or ``local`` (module docstring); ``nonlocal`` when truncated."""
        if self.truncated:
            return "nonlocal"
        if self.returns:
            return "value"
        if self.untracked:
            return "nonlocal"
        if self.raises:
            return "guard"
        return "local"

    @property
    def value_determining(self) -> bool:
        return self.kind in ("value", "nonlocal")


@dataclass(frozen=True)
class FlowSummary:
    params: tuple[str, ...]  # the function's parameter names, in order
    # per parameter: (return sites, raise sites) it reaches; None when truncated
    param_sites: tuple[tuple[int, int], ...] | None
    returns: int | None  # the function's own return sites (None when truncated)
    raises: int | None  # its own raise and assert sites (None when truncated)
    regions: dict[Hashable, RegionFlow]
    truncated: bool = False
    # the bound that cut it short: "nodes", "work", "steps", "time" or "depth"
    limit: str = ""


def _kids(node: ast.AST) -> list[ast.AST]:
    out = []
    for name in node._fields:
        value = getattr(node, name, None)
        if isinstance(value, ast.AST):
            if not isinstance(value, _FOLDED):
                out.append(value)
        elif isinstance(value, list):
            out.extend(
                v
                for v in value
                if isinstance(v, ast.AST) and not isinstance(v, _FOLDED)
            )
    return out


def _store_base(node: ast.AST) -> str | None:
    """The name whose value a subscript or attribute store changes (``x`` in ``x[i].a = v``)."""
    base = node
    while isinstance(base, (ast.Subscript, ast.Attribute, ast.Starred)):
        base = base.value
    return base.id if isinstance(base, ast.Name) else None


def _import_binding(a: ast.alias) -> str | None:
    if a.asname:
        return a.asname
    return None if a.name == "*" else a.name.split(".")[0]


def _blocks(s: ast.AST) -> list[list[ast.stmt]]:
    """The statement lists directly inside compound statement *s* (not a nested function's or class's)."""
    if isinstance(s, _NESTED):
        return []
    out = [
        getattr(s, f)
        for f in ("body", "orelse", "finalbody")
        if isinstance(getattr(s, f, None), list)
    ]
    out += [h.body for h in getattr(s, "handlers", ()) or ()]
    out += [c.body for c in getattr(s, "cases", ()) or ()]
    return out


def _scan(fn: ast.AST) -> tuple[set[str], set[str]] | None:
    """(names *fn* binds anywhere, names declared global or nonlocal), or None past :data:`MAX_NODES`."""
    bound: set[str] = set()
    declared: set[str] = set()
    count = 0
    stack = [fn]
    while stack:
        n = stack.pop()
        count += 1
        if count > MAX_NODES:
            return None
        if isinstance(n, ast.Name) and not isinstance(n.ctx, ast.Load):
            bound.add(n.id)
        elif isinstance(n, ast.arg):
            bound.add(n.arg)
        elif isinstance(n, _NESTED) and n is not fn:
            bound.add(n.name)
        elif isinstance(n, (ast.ExceptHandler, ast.MatchAs, ast.MatchStar)) and n.name:
            bound.add(n.name)
        elif isinstance(n, ast.MatchMapping) and n.rest:
            bound.add(n.rest)
        elif isinstance(n, ast.alias):
            b = _import_binding(n)
            if b:
                bound.add(b)
        elif isinstance(n, (ast.Global, ast.Nonlocal)):
            declared.update(n.names)
        stack.extend(ast.iter_child_nodes(n))
    return bound, declared


class _Graph:
    def __init__(
        self,
        roots: Mapping[int, str],
        bound: set[str],
        declared: set[str],
        deadline: float,
    ) -> None:
        self.roots = roots  # id(region root) -> region key
        self.bound = bound
        self.declared = declared
        self.deadline = deadline
        self.deps: dict[str, set[str]] = {}
        self.sites: list[tuple[str, frozenset[str]]] = []
        self.work = 0
        self.clock = _CLOCK_EVERY
        self.fresh = 0
        self.exits: set[int] = set()

    def tick(self, n: int = 1) -> None:
        self.work += n
        if self.work > MAX_WORK:
            raise _Over("work")
        if self.work >= self.clock:
            self.clock = self.work + _CLOCK_EVERY
            if time.monotonic() > self.deadline:
                raise _Over("time")

    def dep(self, name: str, on: Iterable[str]) -> None:
        on = set(on)
        self.tick(1 + len(on))
        self.deps.setdefault(name, set()).update(on)

    def node(self, on: Iterable[str]) -> str:
        """A fresh pseudo-name depending on *on* (a context, a statement's reads, a set of binders)."""
        self.fresh += 1
        key = f"\x00n{self.fresh}"
        self.dep(key, on)
        return key

    def ctx(self, on: Iterable[str], sink: str | None) -> str:
        """A control context depending on *on*, collected by *sink* (an enclosing statement that can exit)."""
        h = self.node(on)
        if sink is not None:
            self.dep(sink, (h,))
        return h

    def _builtin(self, func: ast.AST, names: frozenset[str]) -> bool:
        return (
            isinstance(func, ast.Name)
            and func.id in names
            and func.id not in self.bound
        )

    def _loads(self, nodes: Iterable[ast.AST]) -> set[str]:
        """Names loaded and region keys met under *nodes* (no side effects)."""
        out: set[str] = set()
        stack = [n for n in nodes if n is not None]
        while stack:
            n = stack.pop()
            self.tick()
            key = self.roots.get(id(n))
            if key is not None:
                out.add(key)
            if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load):
                out.add(n.id)
            stack.extend(_kids(n))
        return out

    def _alias(self, expr: ast.AST | None) -> set[str]:
        """The names whose objects *expr*'s value may be or contain (see Assumptions)."""
        out: set[str] = set()
        stack = [expr]
        while stack:
            n = stack.pop()
            if n is None:
                continue
            self.tick()
            if isinstance(n, ast.Name):
                out.add(n.id)
            elif isinstance(n, _FRESH):
                continue
            elif isinstance(n, (ast.Attribute, ast.Subscript)):
                stack.append(n.value)
            elif isinstance(n, ast.Call):
                if self._builtin(n.func, _READ_ONLY):
                    continue
                stack.extend(n.args)
                stack.extend(k.value for k in n.keywords)
                if isinstance(n.func, ast.Attribute):
                    stack.append(n.func.value)
            else:
                stack.extend(_kids(n))
        return out

    def stmt(
        self,
        nodes: Iterable[ast.AST | None],
        ctx: str | None,
        alias_from: Iterable[ast.AST] = (),
    ) -> set[str]:
        """What *nodes* read (names, regions); what they bind or mutate depends on that and on *ctx*."""
        c = {ctx} if ctx else set()
        out: set[str] = set()
        binders: set[str] = set()
        values = list(alias_from)
        stack = [n for n in nodes if n is not None]
        while stack:
            n = stack.pop()
            self.tick()
            key = self.roots.get(id(n))
            if key is not None:
                out.add(key)
                self.dep(key, self._loads([n]) | c)
            if isinstance(n, ast.Name):
                if isinstance(n.ctx, ast.Load):
                    out.add(n.id)
                else:
                    binders.add(n.id)
            elif isinstance(n, (ast.Subscript, ast.Attribute)) and not isinstance(
                n.ctx,
                ast.Load,
            ):
                base = _store_base(n)
                if base:
                    binders.add(base)
            elif isinstance(n, ast.arg):
                binders.add(n.arg)
            elif isinstance(n, (ast.ExceptHandler, ast.MatchAs, ast.MatchStar)):
                if n.name:
                    binders.add(n.name)
            elif isinstance(n, ast.MatchMapping) and n.rest:
                binders.add(n.rest)
            elif isinstance(n, ast.alias):
                bound = _import_binding(n)
                if bound:
                    binders.add(bound)
            elif isinstance(n, _NESTED):
                binders.add(n.name)
            elif isinstance(n, ast.Call) and not self._builtin(n.func, _READ_ONLY):
                for a in n.args:
                    binders |= self._alias(a)
                for k in n.keywords:
                    binders |= self._alias(k.value)
                if isinstance(n.func, ast.Attribute):
                    binders |= self._alias(n.func.value)
            elif isinstance(n, ast.comprehension):
                values.append(n.iter)
                binders |= self._alias(n.iter)  # consumed
            if (
                isinstance(n, (ast.Assign, ast.AugAssign, ast.AnnAssign, ast.NamedExpr))
                and n.value is not None
            ):
                values.append(n.value)
            elif isinstance(n, ast.withitem):
                values.append(n.context_expr)
            stack.extend(_kids(n))
        if binders:
            k = self.node(out | c)
            for b in binders:
                self.dep(b, (k,))
            aliased: set[str] = set()
            for v in values:
                aliased |= self._alias(v)
            if aliased:
                a = self.node(binders)
                for x in aliased:
                    self.dep(x, (a,))
        return out

    def _exit_statements(self, stmts: list[ast.stmt]) -> None:
        """Mark the compound statements that contain a ``return``, ``break`` or ``continue`` of this function."""
        order: list[ast.stmt] = []
        parent_of: dict[int, ast.stmt] = {}
        stack = list(stmts)
        while stack:
            s = stack.pop()
            self.tick()
            order.append(s)
            for block in _blocks(s):
                for x in block:
                    parent_of[id(x)] = s
                    stack.append(x)
        for s in order:
            if isinstance(s, (ast.Return, ast.Break, ast.Continue)):
                p = parent_of.get(id(s))
                while p is not None and id(p) not in self.exits:
                    self.exits.add(id(p))
                    p = parent_of.get(id(p))

    def body(self, stmts: list[ast.stmt]) -> None:
        self._exit_statements(stmts)
        # frames: [block, next position, context, sink of the nearest enclosing statement that can exit]
        stack: list[list] = [[list(stmts), 0, None, None]]
        while stack:
            fr = stack[-1]
            block, i, c, owner = fr
            if i >= len(block):
                stack.pop()
                continue
            fr[1] = i + 1
            s = block[i]
            self.tick()
            cs = {c} if c else set()
            sc = c
            key = self.roots.get(id(s))
            if key is not None:  # a statement region governs what it contains
                self.dep(key, self._loads([s]) | cs)
                sc = self.node(cs | {key})
            scs = {sc} if sc else set()
            sink = None
            if id(s) in self.exits:
                sink = self.node(scs)
                if owner is not None:
                    self.dep(owner, (sink,))
            inner = sink if sink is not None else owner
            nested: list[tuple[list[ast.stmt], str | None]] = []
            if isinstance(s, _NESTED):
                self.stmt([s], sc)
            elif isinstance(s, ast.Return):
                self.sites.append(("return", frozenset(self.stmt([s.value], sc) | scs)))
            elif isinstance(s, ast.Raise):
                self.sites.append(
                    ("raise", frozenset(self.stmt([s.exc, s.cause], sc) | scs)),
                )
            elif isinstance(s, ast.Assert):
                self.sites.append(
                    ("raise", frozenset(self.stmt([s.test, s.msg], sc) | scs)),
                )
            elif isinstance(s, (ast.If, ast.While)):
                h = self.ctx(self.stmt([s.test], sc) | scs, inner)
                nested.append((s.body + s.orelse, h))
            elif isinstance(s, (ast.For, ast.AsyncFor)):
                h = self.ctx(self.stmt([s.iter], sc) | scs, inner)
                self.stmt([s.target], h, alias_from=[s.iter])
                for x in self._alias(s.iter):  # consumed
                    self.dep(x, (h,))
                nested.append((s.body + s.orelse, h))
            elif isinstance(s, (ast.With, ast.AsyncWith)):
                h = self.ctx(self.stmt(s.items, sc) | scs, inner)
                nested.append((s.body, h))
            elif isinstance(s, (ast.Try, getattr(ast, "TryStar", ast.Try))):
                # any statement of the body may raise into a handler: handlers and else depend on all it reads
                hb = self.ctx(self._loads(s.body) | scs, inner)
                for handler in s.handlers:
                    hc = hb
                    hkey = self.roots.get(id(handler))
                    if hkey is not None:
                        self.dep(hkey, self._loads([handler]) | {hb})
                        hc = self.ctx({hb, hkey}, inner)
                    h = self.ctx(self.stmt([handler.type], hc) | {hc}, inner)
                    if handler.name:
                        self.dep(handler.name, (h,))
                    nested.append((handler.body, h))
                nested.append((s.body, sc))
                nested.append((s.orelse, hb))
                nested.append((s.finalbody, sc))
            elif isinstance(s, ast.Match):
                prev = self.ctx(self.stmt([s.subject], sc) | scs, inner)
                # a later case runs only if the earlier ones did not match
                for case in s.cases:
                    prev = self.ctx(
                        self.stmt([case.pattern, case.guard], prev) | {prev},
                        inner,
                    )
                    nested.append((case.body, prev))
            else:  # expressions, assignments, deletes, imports, pass, break, continue
                self.stmt([s], sc)
            # an early exit inside s decides whether what follows runs
            if sink is not None:
                fr[2] = self.node(cs | {sink})
            for blk, h in reversed(nested):
                if blk:
                    stack.append([list(blk), 0, h, inner])

    def untracked(self, nodes: Iterable[ast.AST]) -> bool:
        """Whether the subtrees *nodes* have effects the graph does not follow (module docstring)."""
        exempt: set[int] = set()
        stack = list(nodes)
        while stack:
            n = stack.pop()
            self.tick()
            if isinstance(n, _UNTRACKED):
                return True
            if (
                isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
                and n.decorator_list
            ):
                return True
            if isinstance(n, (ast.Attribute, ast.Subscript)) and not isinstance(
                n.ctx,
                ast.Load,
            ):
                return True
            if (
                isinstance(n, ast.Name)
                and not isinstance(n.ctx, ast.Load)
                and n.id in self.declared
            ):
                return True
            if isinstance(n, ast.Raise) and isinstance(n.exc, ast.Call):
                exempt.add(id(n.exc))  # constructing the exception raised
            if isinstance(n, ast.Call) and id(n) not in exempt:
                read_only = self._builtin(n.func, _READ_ONLY)
                args_only = (
                    self._builtin(n.func, _ARGS_ONLY)
                    and not n.keywords
                    and not any(isinstance(a, ast.Lambda) for a in n.args)
                )
                if not (read_only or args_only):
                    return True
            stack.extend(_kids(n))
        return False


def _params(fn: ast.FunctionDef | ast.AsyncFunctionDef) -> list[ast.arg]:
    a = fn.args
    return [
        *a.posonlyargs,
        *a.args,
        *([a.vararg] if a.vararg else []),
        *a.kwonlyargs,
        *([a.kwarg] if a.kwarg else []),
    ]


def _truncated(names: list[str], keys: list[Hashable], limit: str) -> FlowSummary:
    every = tuple(range(len(names)))
    return FlowSummary(
        params=tuple(names),
        param_sites=None,
        returns=None,
        raises=None,
        regions={
            k: RegionFlow(every, 0, 0, untracked=True, truncated=True) for k in keys
        },
        truncated=True,
        limit=limit,
    )


def flows(
    fn: ast.FunctionDef | ast.AsyncFunctionDef,
    regions: Mapping[Hashable, Iterable[ast.AST]] | None = None,
    *,
    budget_s: float = MAX_SECONDS,
) -> FlowSummary:
    """The data flow of *fn* (see the module docstring), with the flow of each region (key -> root nodes).

    Never ``None``: past a bound the summary is ``truncated`` and every region gets the conservative answer.
    """
    regions = regions or {}
    keys = sorted(regions)
    params = _params(fn)
    names = [p.arg for p in params]
    scan = _scan(fn)  # the node cap, before any other work
    if scan is None:
        return _truncated(names, keys, "nodes")
    rkey = {k: f"\x00region{i}" for i, k in enumerate(keys)}
    roots: dict[int, str] = {}
    for k in keys:
        for node in regions[k]:
            roots[id(node)] = rkey[k]
    g = _Graph(roots, *scan, deadline=time.monotonic() + budget_s)
    try:
        for p in params:
            if id(p) in roots:
                g.dep(p.arg, {roots[id(p)]})
        if id(fn.args) in roots:  # the parameter list itself is a region
            for p in params:
                g.dep(p.arg, {roots[id(fn.args)]})
        g.body(list(fn.body))
        untracked = {
            k: g.untracked([n for n in regions[k] if n is not fn.args]) for k in keys
        }
        # bits: parameters first, then regions
        own: dict[str, int] = {}
        for i, n in enumerate(names):
            own[n] = own.get(n, 0) | (1 << i)
        for j, k in enumerate(keys):
            own[rkey[k]] = 1 << (len(names) + j)
        mask: dict[str, int] = dict(own)
        rev: dict[str, list[str]] = {}
        for v in sorted(g.deps):
            g.tick(1 + len(g.deps[v]))
            for d in sorted(g.deps[v]):
                rev.setdefault(d, []).append(v)
        queue = deque(sorted(g.deps))
        queued = set(queue)
        steps = 0
        while queue:
            steps += 1
            if steps > MAX_STEPS:
                raise _Over("steps")
            v = queue.popleft()
            queued.discard(v)
            ds = g.deps[v]
            g.tick(1 + len(ds))
            m = own.get(v, 0)
            for d in ds:
                m |= mask.get(d, 0)
            if m != mask.get(v, 0):
                mask[v] = m
                for u in rev.get(v, ()):
                    if u not in queued:
                        queued.add(u)
                        queue.append(u)
        site_masks = []
        for kind, ds in g.sites:
            g.tick(1 + len(ds))
            m = 0
            for d in ds:
                m |= mask.get(d, 0)
            site_masks.append((kind, m))
    except _Over as over:
        return _truncated(names, keys, str(over.args[0]))
    except RecursionError:
        return _truncated(names, keys, "depth")
    pbits = (1 << len(names)) - 1

    def reach(bit: int) -> tuple[int, int]:
        r = sum(1 for kind, m in site_masks if kind == "return" and m >> bit & 1)
        e = sum(1 for kind, m in site_masks if kind == "raise" and m >> bit & 1)
        return r, e

    out: dict[Hashable, RegionFlow] = {}
    for j, k in enumerate(keys):
        m = mask.get(rkey[k], 0) & pbits
        r, e = reach(len(names) + j)
        out[k] = RegionFlow(
            tuple(i for i in range(len(names)) if m >> i & 1),
            r,
            e,
            untracked=untracked[k],
        )
    return FlowSummary(
        params=tuple(names),
        param_sites=tuple(reach(i) for i in range(len(names))),
        returns=sum(1 for kind, _ in site_masks if kind == "return"),
        raises=sum(1 for kind, _ in site_masks if kind == "raise"),
        regions=out,
    )
