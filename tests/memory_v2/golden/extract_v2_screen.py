"""Read the v2 screen build's surfacing texts out of commit 9deefbfd1's source (standard library only).

The goldens in ``v2_screen_9deefbfd1.json`` are what this prints, so they are taken from that commit's code
by ``ast``, never typed by hand: Sol's brief template and tools, Sol's first message, the export line and
index budget of the actor's prompt, and the evidence store's schema. ``python
tests/memory_v2/golden/extract_v2_screen.py > tests/memory_v2/golden/v2_screen_9deefbfd1.json`` (in a clone
holding the commit) rewrites the file; ``test_v21_switch_defaults.py`` checks it against the commit when the
clone has it. Nothing here imports the repository's packages.
"""

from __future__ import annotations

import ast
import json
import subprocess
import sys
from pathlib import Path

BASE = "9deefbfd1"
SOURCES = (
    "unify/memory_v2/sol_pass.py",
    "unify/memory_v2/integration/prompt.py",
    "unify/memory_v2/evidence.py",
    "unify/memory_v2/gate.py",
    "unify/memory_v2/index.py",
)


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), *args],
        capture_output=True,
        text=True,
        check=True,
    ).stdout


def _template(node: ast.AST) -> str:
    """An f-string's text with each replacement field written as ``{<its source>}``."""
    if isinstance(node, ast.Constant):
        return node.value
    out = []
    for v in node.values:
        out.append(
            (
                v.value
                if isinstance(v, ast.Constant)
                else "{" + ast.unparse(v.value) + "}"
            ),
        )
    return "".join(out)


class _Templates(ast.NodeTransformer):
    def visit_JoinedStr(self, node: ast.JoinedStr) -> ast.Constant:
        return ast.copy_location(ast.Constant(_template(node)), node)


def _assigned(tree: ast.Module, name: str) -> ast.expr:
    for n in tree.body:
        if isinstance(n, ast.Assign) and any(
            getattr(t, "id", None) == name for t in n.targets
        ):
            return n.value
    raise KeyError(name)


def extract(repo: Path, base: str = BASE) -> dict:
    full = _git(repo, "rev-parse", f"{base}^{{commit}}").strip()
    src = {p: _git(repo, "show", f"{full}:{p}") for p in SOURCES}
    sol = ast.parse(src["unify/memory_v2/sol_pass.py"])
    tools = ast.literal_eval(_Templates().visit(_assigned(sol, "_TOOLS")))
    first = None
    for n in ast.walk(sol):
        if isinstance(n, ast.Dict):
            keys = [getattr(k, "value", None) for k in n.keys]
            if "role" in keys and "content" in keys:
                c = n.values[keys.index("content")]
                if isinstance(c, ast.JoinedStr) and "Current index" in _template(c):
                    first = _template(c)
    prompt = ast.parse(src["unify/memory_v2/integration/prompt.py"])
    export = next(
        _template(n.body[-1].value)
        for n in ast.walk(prompt)
        if isinstance(n, ast.FunctionDef) and n.name == "export_line"
    )
    gate = src["unify/memory_v2/gate.py"]
    g4 = gate[gate.index("    def _g4(") : gate.index("    def _g5(")]
    init = ast.parse(gate)
    budget = next(
        d.value
        for n in ast.walk(init)
        if isinstance(n, ast.FunctionDef) and n.name == "__init__"
        for a, d in zip(n.args.kwonlyargs, n.args.kw_defaults)
        if a.arg == "budget_tokens"
    )
    # The commit is named by its short id and no blob ids are kept: the goldens hold no long hex strings
    # (secret scanners flag them); this function, re-run on the commit, is the provenance.
    return {
        "commit": base,
        "sol_prompt_template": ast.literal_eval(_assigned(sol, "_PROMPT")),
        "sol_tools": tools,
        "sol_first_message": first,
        "export_line": export,
        "index_budget_tokens": ast.literal_eval(
            _assigned(prompt, "INDEX_BUDGET_TOKENS"),
        ),
        "evidence_schema": ast.literal_eval(
            _assigned(ast.parse(src["unify/memory_v2/evidence.py"]), "_SCHEMA"),
        ),
        "gate_budget_tokens_default": budget,
        "gate_g4": g4,
    }


if __name__ == "__main__":
    repo = (
        Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parents[3]
    )
    json.dump(extract(repo), sys.stdout, indent=1, sort_keys=True, ensure_ascii=False)
    sys.stdout.write("\n")
