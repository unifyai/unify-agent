"""J7: no dead machinery that a text or a reader relies on. Every module of ``unify/`` is reachable from an entry
point; every refusal form a tool declares has a producer; every signal source the regimes read has a producer; every
name the in-code tool table carries is a registered tool. Static: the source is parsed, nothing of it runs (the
in-code table and the registry are read through their own module).
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
#: The processes and modules the harness starts by name or path (audit H3's entry points).
ENTRY_POINTS = (
    "unify.__main__", "unify.cli", "unify.memory.integration.fork_worker", "unify.agents.relay_boot",
    "unify.lanes.relay_server", "unify.common._async_tool.transformation_child", "unify.memory.box_audit_hook",
    "unify.memory.box_state", "unify.common.tool_channel_client", "unify.actor.execution.worker_child",
    "unify.memory.writer_service", "unify.memory.integration.async_pass", "unify.memory.views", "unify.agents.views",
    "unify.memory.integration.adapters.worktree_shapes", "unify.sandbox",
)


def _modules() -> dict[str, Path]:
    mods = {}
    for p in (ROOT / "unify").rglob("*.py"):
        name = ".".join(p.relative_to(ROOT).with_suffix("").parts)
        mods[name[: -len(".__init__")] if name.endswith(".__init__") else name] = p
    return mods


def _trees() -> dict[str, ast.AST]:
    out = {}
    for m, p in _modules().items():
        try:
            out[m] = ast.parse(p.read_text())
        except SyntaxError:
            pass
    return out


def test_every_module_is_reachable_from_an_entry_point():
    mods = _modules()

    def resolve(cur: str, level: int, module: str | None) -> str:
        if level == 0:
            return module or ""
        base = cur.split(".")
        pkg = base if mods[cur].name == "__init__.py" else base[:-1]
        pkg = pkg[: len(pkg) - (level - 1)]
        return ".".join(pkg + ([module] if module else []))

    strref = re.compile(r"unify(?:\.[a-z_][a-z0-9_]*)+")
    edges: dict[str, set[str]] = {m: set() for m in mods}
    for m, tree in _trees().items():
        for n in ast.walk(tree):
            if isinstance(n, ast.Import):
                edges[m].update(a.name for a in n.names)
            elif isinstance(n, ast.ImportFrom):
                base = resolve(m, n.level, n.module)
                edges[m].add(base)
                edges[m].update(f"{base}.{a.name}" for a in n.names)
            elif isinstance(n, ast.Constant) and isinstance(n.value, str):
                edges[m].update(strref.findall(n.value))

    def norm(t: str) -> str | None:
        parts = t.split(".")
        for i in range(len(parts), 0, -1):
            if ".".join(parts[:i]) in mods:
                return ".".join(parts[:i])
        return None

    assert all(e in mods for e in ENTRY_POINTS), [e for e in ENTRY_POINTS if e not in mods]
    seen, stack = set(), list(ENTRY_POINTS)
    while stack:
        m = stack.pop()
        if m in seen:
            continue
        seen.add(m)
        parts = m.split(".")
        stack += [".".join(parts[:i]) for i in range(1, len(parts)) if ".".join(parts[:i]) in mods]
        stack += [n for n in map(norm, edges.get(m, ())) if n and n not in seen]
    assert not sorted(set(mods) - seen), sorted(set(mods) - seen)


def _literals_outside_refusal_tables(trees: dict[str, ast.AST]) -> set[str]:
    """Every string literal of ``unify/`` except the keys of refusal tables (a ``refusals=`` dict, or a module-level
    ``…REFUSALS`` dict) and docstrings: where a form is named by its producer."""
    skip: set[int] = set()
    for tree in trees.values():
        for n in ast.walk(tree):
            tables = []
            if isinstance(n, ast.keyword) and n.arg == "refusals" and isinstance(n.value, ast.Dict):
                tables.append(n.value)
            if isinstance(n, (ast.Assign, ast.AnnAssign)) and isinstance(n.value, ast.Dict):
                targets = n.targets if isinstance(n, ast.Assign) else [n.target]
                if any(isinstance(t, ast.Name) and t.id.endswith("REFUSALS") for t in targets):
                    tables.append(n.value)
            for d in tables:
                skip.update(id(k) for k in d.keys if k is not None)
            if isinstance(n, ast.Expr) and isinstance(n.value, ast.Constant):  # a docstring names, never produces
                skip.add(id(n.value))
    return {n.value for tree in trees.values() for n in ast.walk(tree)
            if isinstance(n, ast.Constant) and isinstance(n.value, str) and id(n) not in skip}


def test_every_refusal_form_has_a_producer():
    from unify.common import tool_registry as R

    literals = _literals_outside_refusal_tables(_trees())
    orphans = sorted({(tool, form) for tool, forms in R.refusal_codes().items() for form in forms
                      if form not in literals})
    assert not orphans, orphans


def test_every_signal_source_the_regimes_read_has_a_producer():
    from unify.memory import signals as SG

    produced: set[str] = set()
    for m, tree in _trees().items():
        for n in ast.walk(tree):
            if isinstance(n, ast.Call) and getattr(n.func, "id", getattr(n.func, "attr", None)) == "Signal":
                if len(n.args) > 2 and isinstance(n.args[2], ast.Constant):
                    produced.add(n.args[2].value)
                produced.update(k.value.value for k in n.keywords
                                if k.arg == "source" and isinstance(k.value, ast.Constant))
    read = set().union(*SG.REGIME_SOURCES.values())
    assert read <= produced, sorted(read - produced)


def test_every_in_code_name_is_a_registered_tool():
    from unify.common import tool_registry as R

    assert not sorted(set(R._IN_CODE) - set(R.registry())), sorted(set(R._IN_CODE) - set(R.registry()))
