"""Detect a function that replaces a value it computed from its input under a condition (memory v2.1, stage 7).

The lesson (office-v2 F3, 8 Oct 2026): a stored expense function computed ``pay = limit`` for an over-cap
claim, then ``if rule: pay = Decimal("0.00")`` paid every claim with a rule nothing, over-cap ones included.
Its recorded cases passed, because they record what it returned, not whether the policy was right. Such a
function encodes a decision rule, and the gate asks for independent evidence before it merges one.

**The pattern.** :func:`find_overrides` flags an *override*: on some path through one function,

1. a plain name ``v`` is bound (assignment, augmented or annotated assignment, walrus, ``for`` or ``with``
   target) from a value that depends on a parameter (``A1``);
2. later on that path, before anything *consumes* ``v``, ``v`` is bound again from a value that depends on
   none of the parameters ``A1`` depended on: a constant, a module name, or another parameter's data (``A2``);
3. ``A2`` sits under a condition ``A1`` does not (an ``if``/``elif``/``else`` branch, a ``match`` case or an
   ``except`` handler), and ``A1`` is consumed on some other path;
4. ``v`` reaches an output: a ``return`` or ``yield`` value, directly or through names and containers
   (``totals[k] += v``, ``out.append(v)``), or an object rooted at a parameter (``apis.files.write(v)``).

Any read of ``v`` consumes it except one in an ``if`` or ``while`` test or a ``match`` subject or guard (a
condition decides; it does not use the value), so ``x = f(a); x = g(x)`` is not an override, while
``pay = usd; if pay > limit: pay = 0`` is. Parameters themselves are not ``A1`` (a default-argument fill
``if limit is None: limit = 10`` is not flagged). Purely structural: no names, words or values matter.

**Approximations** (each errs toward *not* flagging, except that over-approximating ``A1``'s dependence and
``v``'s reach errs toward flagging):

* dependence on parameters is flow-insensitive: a name depends on every parameter any of its bindings or
  mutations reads (``x[k] = v`` and ``x.m(v)`` make ``x`` depend on ``v``), and it reaches an output if any
  of its uses does;
* a loop body is walked once, as if it ran zero or one time (no back edge), so a value carried to the next
  iteration and replaced there is not an override; ``continue`` ends a path, ``break`` joins the loop's exit;
* an ``except`` handler starts from the state before its ``try`` (the body raised before it bound anything);
* only the function's own statements are walked: a nested function, class or lambda is one binding of its
  name (or a read) whose body is not analysed, and helpers it calls are not followed.

**Bounds.** A function over :data:`MAX_NODES` AST nodes is not analysed, and the walk stops past
:data:`MAX_WORK` steps or Python's recursion limit; each gives ``truncated=True`` and no overrides (the caller
notes it). No wall clock, so the result is a pure function of the AST. Standard library only, so it runs in
the consolidation sandbox as ``memlab.analysis.overrides``.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass

MAX_NODES = 20_000  # AST nodes per function, counted first
# statements, expression nodes and state entries visited, per function
MAX_WORK = 1_000_000

_SCOPES = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)


@dataclass(frozen=True)
class Override:
    name: str  # the variable whose computed value is replaced
    line: int  # the replacing binding (A2), a line of the analysed source
    computed: int  # the binding it replaces (A1)


@dataclass(frozen=True)
class OverrideReport:
    overrides: tuple[Override, ...] = ()
    truncated: bool = False
    limit: str = ""  # the bound that cut it short: "nodes", "work" or "depth"

    @property
    def flagged(self) -> bool:
        return bool(self.overrides)

    @property
    def line(self) -> int:
        """The first replacing line (0 when not flagged)."""
        return min((o.line for o in self.overrides), default=0)


class _Over(Exception):
    pass


def _loads(node: ast.AST | None) -> set[str]:
    """Names read anywhere under *node* (lambdas and comprehensions included)."""
    if node is None:
        return set()
    return {
        n.id
        for n in ast.walk(node)
        if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)
    }


def _targets(target: ast.AST) -> tuple[list[str], list[str]]:
    """(plain names bound, base names mutated) by an assignment target."""
    names: list[str] = []
    bases: list[str] = []
    stack = [target]
    while stack:
        t = stack.pop()
        if isinstance(t, ast.Name):
            names.append(t.id)
        elif isinstance(t, (ast.Tuple, ast.List)):
            stack.extend(t.elts)
        elif isinstance(t, ast.Starred):
            stack.append(t.value)
        elif isinstance(t, (ast.Subscript, ast.Attribute)):
            base = t.value
            while isinstance(base, (ast.Subscript, ast.Attribute)):
                base = base.value
            if isinstance(base, ast.Name):
                bases.append(base.id)
    return names, bases


def _receiver(call: ast.Call) -> tuple[str, set[str]] | None:
    """A method call's receiver base name (``x`` in ``x[0].f().m(v)``) and the names its arguments read."""
    if not isinstance(call.func, ast.Attribute):
        return None
    base = call.func.value
    while isinstance(base, (ast.Subscript, ast.Attribute, ast.Call)):
        base = base.func if isinstance(base, ast.Call) else base.value
    if not isinstance(base, ast.Name):
        return None
    reads: set[str] = set()
    for a in call.args:
        reads |= _loads(a)
    for k in call.keywords:
        reads |= _loads(k.value)
    return base.id, reads


def _walrus(node: ast.AST | None) -> list[ast.NamedExpr]:
    if node is None:
        return []
    return [n for n in ast.walk(node) if isinstance(n, ast.NamedExpr)]


class _Events:
    """Flow-insensitive facts: which parameters each name depends on, and which names reach an output."""

    def __init__(self, params: list[str]) -> None:
        self.params = params
        # (name bound or mutated, names read)
        self.binds: list[tuple[str, set[str]]] = []
        self.sinks: set[str] = set()
        self.work = 0

    def tick(self, n: int = 1) -> None:
        self.work += n
        if self.work > MAX_WORK:
            raise _Over("work")

    def add(self, name: str, reads: set[str]) -> None:
        self.tick(1 + len(reads))
        self.binds.append((name, reads))

    def collect(self, fn: ast.AST) -> None:
        stack = list(ast.iter_child_nodes(fn))
        while stack:
            n = stack.pop()
            self.tick()
            if isinstance(n, _SCOPES):
                if not isinstance(n, ast.Lambda):
                    self.add(n.name, _loads(n))
                continue  # a nested scope's statements are not this function's
            if isinstance(n, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
                reads = _loads(n.value)
                tgts = n.targets if isinstance(n, ast.Assign) else [n.target]
                for t in tgts:
                    names, bases = _targets(t)
                    own = isinstance(n, ast.AugAssign)
                    for x in names:
                        self.add(x, reads | {x} if own else reads)
                    for b in bases:
                        self.add(b, reads | _loads(t))
            elif isinstance(n, (ast.For, ast.AsyncFor, ast.comprehension)):
                for x in _targets(n.target)[0]:
                    self.add(x, _loads(n.iter))
            elif isinstance(n, ast.withitem) and n.optional_vars is not None:
                for x in _targets(n.optional_vars)[0]:
                    self.add(x, _loads(n.context_expr))
            elif isinstance(n, ast.NamedExpr):
                self.add(n.target.id, _loads(n.value))
            elif isinstance(n, ast.Call) and (rec := _receiver(n)) is not None:
                self.add(*rec)
                # a write into the input or the environment is an output
                if rec[0] in self.params:
                    self.sinks |= rec[1]
            elif (
                isinstance(n, (ast.Return, ast.Yield, ast.YieldFrom))
                and n.value is not None
            ):
                self.sinks |= _loads(n.value)
            stack.extend(ast.iter_child_nodes(n))

    def masks(self) -> dict[str, int]:
        """Name -> bit set of the parameters it depends on (a least fixpoint, worklist-driven)."""
        mask = {p: 1 << i for i, p in enumerate(self.params)}
        readers: dict[str, list[int]] = {}
        for i, (_, reads) in enumerate(self.binds):
            for r in reads:
                readers.setdefault(r, []).append(i)
        queue = list(range(len(self.binds)))
        while queue:
            i = queue.pop()
            name, reads = self.binds[i]
            self.tick(1 + len(reads))
            m = mask.get(name, 0)
            for r in reads:
                m |= mask.get(r, 0)
            if m != mask.get(name, 0):
                mask[name] = m
                queue.extend(readers.get(name, ()))
        return mask

    def reaching(self) -> set[str]:
        """Names whose value can reach an output (a backward fixpoint over the same bindings)."""
        by_name: dict[str, list[set[str]]] = {}
        for name, reads in self.binds:
            by_name.setdefault(name, []).append(reads)
        seen = set(self.sinks)
        queue = list(seen)
        while queue:
            v = queue.pop()
            for reads in by_name.get(v, ()):
                self.tick(1 + len(reads))
                for r in reads - seen:
                    seen.add(r)
                    queue.append(r)
        return seen


# The forward walk's state: name -> the pending A1 bindings of it (input-dependent, not yet consumed);
# None for an unreachable point (after return, raise, break or continue).
_State = dict[str, frozenset[int]] | None


def _join(*states: _State) -> _State:
    live = [s for s in states if s is not None]
    if not live:
        return None
    out: dict[str, frozenset[int]] = dict(live[0])
    for s in live[1:]:
        for k, v in s.items():
            out[k] = out[k] | v if k in out else v
    return out


class _Walk:
    def __init__(self, ev: _Events, mask: dict[str, int]) -> None:
        self.ev, self.mask = ev, mask
        # (name, line, parameter mask, the conditions it sits under: if branches, match cases, handlers)
        self.defs: list[tuple[str, int, int, frozenset[int]]] = []
        self.consumed: set[int] = set()
        self.kills: list[tuple[int, int]] = []  # (A1, A2) def indices
        self.breaks: list[list[_State]] = []
        self.conds: frozenset[int] = frozenset()
        self.n_conds = 0

    def value_mask(self, reads: set[str]) -> int:
        m = 0
        for r in reads:
            m |= self.mask.get(r, 0)
        return m

    def consume(self, st: _State, reads: set[str]) -> _State:
        if st is None:
            return None
        for r in reads:
            pending = st.get(r)
            if pending:
                self.ev.tick(len(st))
                self.consumed |= pending
                st = {**st}
                del st[r]
        return st

    def bind(self, st: _State, name: str, line: int, m: int) -> _State:
        if st is None:
            return None
        self.ev.tick(1 + len(st))  # the state is copied below
        d = len(self.defs)
        self.defs.append((name, line, m, self.conds))
        for p in st.get(name, ()):
            # A2 depends on none of A1's parameters and sits under a condition A1 does not
            if self.defs[p][2] & m == 0 and not self.conds <= self.defs[p][3]:
                self.kills.append((p, d))
        st = {**st}
        if m:
            st[name] = frozenset((d,))
        else:
            st.pop(name, None)
        return st

    def expr(self, st: _State, node: ast.AST | None, consuming: bool = True) -> _State:
        """Read *node*: consume what it reads (unless a condition), then bind its walrus targets."""
        if node is None or st is None:
            return st
        self.ev.tick()
        if consuming:
            st = self.consume(st, _loads(node))
        for w in _walrus(node):
            st = self.bind(st, w.target.id, w.lineno, self.value_mask(_loads(w.value)))
        return st

    def block(self, st: _State, stmts: list[ast.stmt]) -> _State:
        for s in stmts:
            if st is None:
                break
            st = self.stmt(st, s)
        return st

    def branch(self, st: _State, stmts: list[ast.stmt], bind: tuple = ()) -> _State:
        """Walk *stmts* under a new condition (an if branch, a match case, an except handler)."""
        outer = self.conds
        self.n_conds += 1
        self.conds = outer | {self.n_conds}
        try:
            for name, line, m in bind:
                st = self.bind(st, name, line, m)
            return self.block(st, stmts)
        finally:
            self.conds = outer

    def stmt(self, st: _State, s: ast.stmt) -> _State:
        self.ev.tick()
        if isinstance(s, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            reads = _loads(s)
            st = self.consume(st, reads)
            return self.bind(st, s.name, s.lineno, self.value_mask(reads))
        if isinstance(s, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
            if s.value is None:  # a bare annotation binds nothing
                return st
            tgts = s.targets if isinstance(s, ast.Assign) else [s.target]
            reads = _loads(s.value)
            st = self.expr(st, s.value)
            for t in tgts:
                names, bases = _targets(t)
                st = self.consume(st, _loads(t) | set(bases))
                for x in names:
                    own = {x} if isinstance(s, ast.AugAssign) else set()
                    st = self.consume(st, own)
                    st = self.bind(st, x, s.lineno, self.value_mask(reads | own))
            return st
        if isinstance(s, (ast.Return, ast.Raise)):
            nodes = [s.value] if isinstance(s, ast.Return) else [s.exc, s.cause]
            for n in nodes:
                st = self.expr(st, n)
            return None
        if isinstance(s, ast.Break):
            if self.breaks:
                self.breaks[-1].append(st)
            return None
        if isinstance(s, ast.Continue):
            return None
        if isinstance(s, ast.If):
            st = self.expr(st, s.test, consuming=False)
            self.ev.tick(len(st or ()))  # the join below
            return _join(self.branch(st, s.body), self.branch(st, s.orelse))
        if isinstance(s, (ast.For, ast.AsyncFor, ast.While)):
            if isinstance(s, ast.While):
                st = self.expr(st, s.test, consuming=False)
                entry = st
            else:
                st = self.expr(st, s.iter)
                entry = st
                m = self.value_mask(_loads(s.iter))
                for x in _targets(s.target)[0]:
                    entry = self.bind(entry, x, s.lineno, m)
            self.breaks.append([])
            end = self.block(entry, s.body)
            broke = self.breaks.pop()
            after = self.block(_join(st, end), s.orelse)
            return _join(after, *broke)
        if isinstance(s, (ast.With, ast.AsyncWith)):
            for item in s.items:
                st = self.expr(st, item.context_expr)
                if item.optional_vars is not None:
                    m = self.value_mask(_loads(item.context_expr))
                    for x in _targets(item.optional_vars)[0]:
                        st = self.bind(st, x, s.lineno, m)
            return self.block(st, s.body)
        if isinstance(s, ast.Try) or type(s).__name__ == "TryStar":
            end = self.block(self.block(st, s.body), s.orelse)
            ends = [end]
            for h in s.handlers:
                hst = self.expr(st, h.type)
                named = ((h.name, h.lineno, 0),) if h.name else ()
                ends.append(self.branch(hst, h.body, named))
            return self.block(_join(*ends), s.finalbody)
        if isinstance(s, ast.Match):
            st = self.expr(st, s.subject, consuming=False)
            m = self.value_mask(_loads(s.subject))
            ends: list[_State] = [st]
            for case in s.cases:
                bound = []
                for n in ast.walk(case.pattern):
                    if isinstance(n, (ast.MatchAs, ast.MatchStar)) and n.name:
                        bound.append((n.name, case.pattern.lineno, m))
                    elif isinstance(n, ast.MatchMapping) and n.rest:
                        bound.append((n.rest, case.pattern.lineno, m))
                cst = self.expr(st, case.guard, consuming=False)
                ends.append(self.branch(cst, case.body, tuple(bound)))
            return _join(*ends)
        if isinstance(s, ast.Delete):
            for t in s.targets:
                names, bases = _targets(t)
                st = self.consume(st, _loads(t) | set(bases))
                if st is not None:
                    st = {k: v for k, v in st.items() if k not in names}
            return st
        if isinstance(s, (ast.Import, ast.ImportFrom)):
            for a in s.names:
                bound = a.asname or (None if a.name == "*" else a.name.split(".")[0])
                if bound:
                    st = self.bind(st, bound, s.lineno, 0)
            return st
        # expression statements, assert, global, nonlocal, pass, type aliases
        for child in ast.iter_child_nodes(s):
            st = self.expr(st, child)
        return st


def _params(fn: ast.FunctionDef | ast.AsyncFunctionDef) -> list[str]:
    a = fn.args
    return [
        p.arg
        for p in (
            *a.posonlyargs,
            *a.args,
            *([a.vararg] if a.vararg else []),
            *a.kwonlyargs,
            *([a.kwarg] if a.kwarg else []),
        )
    ]


def find_overrides(fn: ast.FunctionDef | ast.AsyncFunctionDef) -> OverrideReport:
    """The overrides in *fn* (module docstring), or ``truncated`` with none past a bound."""
    count = 0
    for _ in ast.walk(fn):
        count += 1
        if count > MAX_NODES:
            return OverrideReport(truncated=True, limit="nodes")
    try:
        ev = _Events(_params(fn))
        ev.collect(fn)
        mask = ev.masks()
        out = ev.reaching()
        w = _Walk(ev, mask)
        w.block({}, list(fn.body))
    except _Over as over:
        return OverrideReport(truncated=True, limit=str(over.args[0]))
    except RecursionError:
        return OverrideReport(truncated=True, limit="depth")
    found = []
    for a1, a2 in w.kills:
        name, computed, _, _ = w.defs[a1]
        if a1 in w.consumed and name in out:
            found.append(Override(name, w.defs[a2][1], computed))
    return OverrideReport(
        tuple(sorted(set(found), key=lambda o: (o.line, o.computed, o.name))),
    )
