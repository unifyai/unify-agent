"""Static analysis the harness does on the model's code: provenance, parameters, smells.

All of it is general: it looks at Python syntax, the request text and the files the code
read. Nothing here knows about any benchmark.
"""
from __future__ import annotations

import ast
import difflib
import math
import re

# ---------------------------------------------------------------- provenance (slicing)


class _DefUse(ast.NodeVisitor):
    """Global names a cell defines or mutates, and global names it reads. Locals of functions,
    lambdas and comprehensions are tracked in a scope stack and ignored."""

    def __init__(self):
        self.defs, self.uses, self.local = set(), set(), []

    def _is_local(self, name):
        return any(name in scope for scope in self.local)

    def _define(self, name):
        (self.local[-1] if self.local else self.defs).add(name)

    def visit_Name(self, node):
        if isinstance(node.ctx, (ast.Store, ast.Del)):
            self._define(node.id)
        elif not self._is_local(node.id):
            self.uses.add(node.id)

    def _mutated(self, node):  # x.append(...), x[k] = v, x.attr = v: x may change
        if isinstance(node, ast.Name) and not self._is_local(node.id):
            self.defs.add(node.id)

    def visit_Attribute(self, node):
        self._mutated(node.value)
        self.generic_visit(node)

    def visit_Subscript(self, node):
        if isinstance(node.ctx, ast.Store):
            self._mutated(node.value)
        self.generic_visit(node)

    def visit_AugAssign(self, node):
        if isinstance(node.target, ast.Name) and not self._is_local(node.target.id):
            self.uses.add(node.target.id)
        self.generic_visit(node)

    def visit_Import(self, node):
        for a in node.names:
            self._define((a.asname or a.name).split(".")[0])

    visit_ImportFrom = visit_Import

    def _function(self, node, body):
        a = node.args
        for d in getattr(node, "decorator_list", []) + a.defaults + [d for d in a.kw_defaults if d]:
            self.visit(d)
        params = {x.arg for x in a.posonlyargs + a.args + a.kwonlyargs} | {x.arg for x in (a.vararg, a.kwarg) if x}
        self.local.append(params)
        for stmt in body:
            self.visit(stmt)
        self.local.pop()

    def visit_FunctionDef(self, node):
        self._define(node.name)
        self._function(node, node.body)

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_Lambda(self, node):
        self._function(node, [node.body])

    def _comprehension(self, node):
        self.visit(node.generators[0].iter)  # evaluated in the enclosing scope
        self.local.append(set())
        for g in node.generators:
            self.visit(g.target)
        for i, g in enumerate(node.generators):
            if i:
                self.visit(g.iter)
            for cond in g.ifs:
                self.visit(cond)
        for f in ("elt", "key", "value"):
            if hasattr(node, f):
                self.visit(getattr(node, f))
        self.local.pop()

    visit_ListComp = visit_SetComp = visit_GeneratorExp = visit_DictComp = _comprehension


def defs_uses(code: str) -> tuple[set[str], set[str]]:
    """Over-approximates on purpose: an extra cell in a slice costs a little replay time;
    a missing one breaks the replay (and the replay check then refuses to store it)."""
    v = _DefUse()
    v.visit(ast.parse(code))
    return v.defs, v.uses


def kills(code: str) -> set[str]:
    """Names a cell certainly overwrites before reading them: plain top-level assignments,
    defs and imports. Earlier definitions of these names are not needed by this cell."""
    out, read = set(), set()
    for stmt in ast.parse(code).body:
        uses = defs_uses(ast.unparse(stmt))[1]
        names = set()
        if isinstance(stmt, (ast.Assign, ast.AnnAssign)):
            targets = stmt.targets if isinstance(stmt, ast.Assign) else [stmt.target]
            names = {t.id for t in targets if isinstance(t, ast.Name)}
        elif isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names = {stmt.name}
        elif isinstance(stmt, (ast.Import, ast.ImportFrom)):
            names = {(a.asname or a.name).split(".")[0] for a in stmt.names}
        out |= names - uses - read
        read |= uses
    return out


FS_CALLS = {"rename", "replace", "move", "copy", "copyfile", "copy2", "copytree", "remove", "unlink", "rmdir",
            "rmtree", "removedirs", "makedirs", "mkdir", "write_text", "write_bytes", "touch", "symlink_to",
            "symlink", "link", "chmod", "truncate", "run", "call", "check_call", "check_output", "Popen", "system"}


def touches_files(code: str) -> bool:
    """Does the cell call something that can change files without going through open()?"""
    return any(isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr in FS_CALLS
               for n in ast.walk(ast.parse(code)))


def backward_slice(cells: list[str], target: int, external: set[str] = frozenset(),
                   reads: list[set] | None = None, writes: list[set] | None = None,
                   seeds: set[int] = frozenset(), given: set[str] = frozenset()) -> tuple[list[int], set[str]]:
    """Indices of the cells needed to recompute cell `target`, oldest first, plus the names it
    needs that no cell defines (for example a value the harness injected).
    Dependencies are both names (def-use) and files: a cell that wrote a file a chosen cell
    reads is chosen too. `seeds` are cells that must be included (file-changing cells when
    the deliverable is files)."""
    info = [defs_uses(c) for c in cells]
    reads = reads or [set() for _ in cells]
    writes = writes or [set() for _ in cells]
    # `given`: names the harness provides (an API client): calling them is not a dependency on earlier cells
    info = [(d - set(given), u - set(given)) for d, u in info]
    needed, needed_files = info[target][1] - kills(cells[target]), set(reads[target])
    chosen = [target]
    for i in range(target - 1, -1, -1):
        if i in seeds or info[i][0] & needed or writes[i] & needed_files:
            chosen.append(i)
            k = kills(cells[i])  # names this cell writes before reading: its later reads see its own value
            needed = (needed - k) | (info[i][1] - k)
            needed_files |= reads[i]
    return sorted(chosen), {n for n in needed if n in external and not any(n in info[i][0] for i in chosen)}


# ---------------------------------------------------------------- parameters from the request


def _in_request(value, request: str) -> bool:
    if isinstance(value, bool) or value in (0, 1) or value is None:
        return False
    text = str(value)
    if len(text) < 2:
        return False
    return re.search(r"(?<![\w-])" + re.escape(text) + r"(?![\w-])", request, re.IGNORECASE) is not None


class _Lifter(ast.NodeTransformer):
    def __init__(self, request: str):
        self.request, self.params = request, {}

    def visit_JoinedStr(self, node):  # f-string parts must stay constants
        return node

    def visit_Constant(self, node):
        if isinstance(node.value, (str, int, float)) and _in_request(node.value, self.request):
            for name, val in self.params.items():
                if val == node.value:
                    return ast.copy_location(ast.Name(name, ast.Load()), node)
            name = f"p{len(self.params) + 1}"
            self.params[name] = node.value
            return ast.copy_location(ast.Name(name, ast.Load()), node)
        return node


def lift_parameters(code: str, request: str) -> tuple[str, dict]:
    """Turn literals that came from the request into named parameters p1, p2, ...
    'total spend on meals in 2026-03' + code filtering == "meals" -> p1 = "meals"."""
    lifter = _Lifter(request)
    tree = lifter.visit(ast.parse(code))
    return ast.unparse(ast.fix_missing_locations(tree)), lifter.params


def normalized(code: str) -> str:
    return ast.dump(ast.parse(code), annotate_fields=False)


def shape(code: str) -> str:
    """The code's structure with every constant and parameter blanked: two procedures with the
    same shape do the same thing with different values (a paraphrased return of the same job
    lifts different words into parameters but keeps the shape)."""
    class Blank(ast.NodeTransformer):
        def visit_Constant(self, node):
            return ast.copy_location(ast.Constant("_"), node)

        def visit_Name(self, node):
            if re.fullmatch(r"p\d+", node.id):
                return ast.copy_location(ast.Constant("_"), node)
            return node
    return ast.dump(Blank().visit(ast.parse(code)), annotate_fields=False)


# ---------------------------------------------------------------- smell tests


def value_smells(value) -> list[str]:
    """Results that are often wrong and cheap to recognise."""
    if isinstance(value, dict) and set(value) == {"__repr__"}:
        value = value["__repr__"]
    if value is None:
        return ["the answer is None"]
    if isinstance(value, bool):
        return []
    if isinstance(value, (int, float)):
        if isinstance(value, float) and math.isnan(value):
            return ["the answer is NaN"]
        if isinstance(value, float) and math.isinf(value):
            return ["the answer is infinite"]
        return ["the answer is zero"] if value == 0 else []
    if isinstance(value, str):
        s = value.strip().lower()
        if not s:
            return ["the answer is an empty string"]
        if s in {"nan", "none", "null", "n/a"} or re.fullmatch(r"[$€£]?\s*-?0+(\.0+)?", s):
            return [f"the answer {value!r} looks empty or zero"]
        return []
    if isinstance(value, (list, tuple, dict, set)):
        if not value:
            return [f"the answer is an empty {type(value).__name__}"]
        vals = list(value.values()) if isinstance(value, dict) else list(value)
        if all(isinstance(v, (int, float)) and not isinstance(v, bool) and v == 0 for v in vals):
            return ["every number in the answer is zero"]
    return []


def _compared_literals(code: str) -> dict[str, bool]:
    """String literals the code compares data against -> True if compared exactly (==, !=),
    False if as a substring or prefix (in, startswith, find, ...)."""
    found: dict[str, bool] = {}
    def take(node, exact):
        items = node.elts if isinstance(node, (ast.List, ast.Tuple, ast.Set)) else [node]
        for i in items:
            if isinstance(i, ast.Constant) and isinstance(i.value, str) and i.value.strip():
                found[i.value] = found.get(i.value, True) and exact
    for node in ast.walk(ast.parse(code)):
        if isinstance(node, ast.Compare):
            sides = [node.left, *node.comparators]
            # exact only when the other side is a raw field (x, x.a, x['a']), not a slice or a call
            raw = any(isinstance(p, (ast.Name, ast.Attribute)) or
                      (isinstance(p, ast.Subscript) and not isinstance(p.slice, ast.Slice)) for p in sides)
            exact = raw and all(isinstance(op, (ast.Eq, ast.NotEq)) for op in node.ops)
            for p in sides:
                take(p, exact)
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and \
                node.func.attr in {"startswith", "endswith", "count", "find", "index"}:
            for a in node.args:
                take(a, False)
    return found


def _field_values(text: str) -> set[str]:
    return {v.strip() for v in re.split(r"[,\t;|\n\r\"{}\[\]:]", text) if v.strip()}


def ungrounded_literals(code: str, files: dict[str, str]) -> list[str]:
    """Literals the code compares against that never occur as a whole value in the files it
    read. This is the 'meal' versus 'meals' check: filtering on a value the data does not
    contain is the commonest cause of a confident zero."""
    if not files:
        return []
    values = set().union(*(_field_values(t) for t in files.values()))
    lowered = {v.lower(): v for v in values}
    findings = []
    text = "\n".join(files.values())
    for lit, exact in sorted(_compared_literals(code).items()):
        if lit in values or (not exact and lit in text):
            continue
        near = [v for v in values if lit.lower() in v.lower() and len(v) <= len(lit) + 12]
        near += difflib.get_close_matches(lit, list(values), n=3, cutoff=0.75)
        if lit.lower() in lowered:
            near.insert(0, lowered[lit.lower()])
        near = sorted(set(near))[:5]
        hint = f"; the data has {', '.join(repr(n) for n in near)}" if near else ""
        findings.append(f"the code compares against {lit!r}, which is not a value in {', '.join(sorted(files))}{hint}")
    return findings


def preview(files: dict[str, str], lines: int = 3) -> list[str]:
    return [f"{name}: {len(t.splitlines())} lines; first lines: " + " | ".join(t.splitlines()[:lines])
            for name, t in sorted(files.items())]


def strip_displays(code: str) -> str:
    """Drop top-level bare expressions that only display a value (`len(rows)`, `total`).
    Calls stay: they may have effects (deliver(...), f.write(...), print is harmless)."""
    tree = ast.parse(code)
    tree.body = [s for s in tree.body if not (isinstance(s, ast.Expr) and not isinstance(s.value, ast.Call))]
    return ast.unparse(tree) if tree.body else ""


def file_smells(files: dict[str, dict]) -> list[str]:
    """Smells of delivered files: missing content, a table with no rows, a lone zero."""
    out = []
    for name, f in sorted(files.items()):
        head, size = f.get("head", ""), f.get("bytes", 0)
        lines = [ln for ln in head.splitlines() if ln.strip()]
        ext = name.rsplit(".", 1)[-1].lower() if "." in name else ""
        if size == 0 or not head.strip():
            out.append(f"{name} is empty")
        elif ext in ("csv", "tsv") and len(lines) == 1 and size <= len(head.encode()):
            out.append(f"{name} has a header and no rows")
        elif ext == "jsonl" and not lines:
            out.append(f"{name} has no rows")
        elif ext == "json" and head.strip() in ("[]", "{}", "null"):
            out.append(f"{name} holds an empty JSON value ({head.strip()})")
        elif len(lines) == 1 and len(lines[0].split()) == 1 and size <= len(head.encode()):
            out += [f"{name}: {s}" for s in value_smells(lines[0].strip())]
    return out
