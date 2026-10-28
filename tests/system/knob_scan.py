"""Every environment variable and setting the harness reads (guard J9): a static scan of ``unify/``, no import.

A knob is: a field of the settings model (``unify/settings.py``), a name read with ``os.environ[...]``,
``os.environ.get(...)``, ``os.getenv(...)`` or ``getattr(SETTINGS, ...)``, an upper-case attribute read on
``SETTINGS``, or an ``UNIFY_*`` / ``UNILLM_*`` string literal anywhere (names held in constants and read later).
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

NAME = re.compile(r"^(UNIFY|UNILLM|MEMORY|TOOLS)_[A-Z0-9_]+$")
ENV_NAME = re.compile(r"^[A-Z][A-Z0-9_]{2,}$")


def scan(root: Path) -> dict[str, list[str]]:
    """Knob name -> the files (relative to *root*) that name it."""
    out: dict[str, set[str]] = {}

    def add(name: str, rel: str) -> None:
        out.setdefault(name, set()).add(rel)

    for p in sorted((root / "unify").rglob("*.py")):
        rel = str(p.relative_to(root))
        try:
            tree = ast.parse(p.read_text(encoding="utf-8"))
        except SyntaxError:
            continue
        for n in ast.walk(tree):
            if isinstance(n, ast.Constant) and isinstance(n.value, str) and NAME.match(n.value):
                add(n.value, rel)
            elif isinstance(n, ast.Subscript) and _is_environ(n.value):
                k = n.slice
                if isinstance(k, ast.Constant) and isinstance(k.value, str) and ENV_NAME.match(k.value):
                    add(k.value, rel)
            elif isinstance(n, ast.Call):
                f = n.func
                env_get = isinstance(f, ast.Attribute) and f.attr in ("get", "pop", "setdefault") and _is_environ(f.value)
                getenv = isinstance(f, ast.Attribute) and f.attr == "getenv"
                settings_getattr = isinstance(f, ast.Name) and f.id == "getattr" and n.args and \
                    isinstance(n.args[0], ast.Name) and n.args[0].id == "SETTINGS"
                if (env_get or getenv) and n.args and isinstance(n.args[0], ast.Constant) and \
                        isinstance(n.args[0].value, str) and ENV_NAME.match(n.args[0].value):
                    add(n.args[0].value, rel)
                if settings_getattr and len(n.args) > 1 and isinstance(n.args[1], ast.Constant) and \
                        isinstance(n.args[1].value, str) and ENV_NAME.match(n.args[1].value):
                    add(n.args[1].value, rel)
            elif isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name) and n.value.id == "SETTINGS" and \
                    ENV_NAME.match(n.attr):
                add(n.attr, rel)
            elif rel == "unify/settings.py" and isinstance(n, ast.AnnAssign) and isinstance(n.target, ast.Name) and \
                    ENV_NAME.match(n.target.id):
                add(n.target.id, rel)
    return {k: sorted(v) for k, v in sorted(out.items())}


def _is_environ(node: ast.AST) -> bool:
    return (isinstance(node, ast.Attribute) and node.attr == "environ") or (isinstance(node, ast.Name) and node.id == "environ")


if __name__ == "__main__":
    import json
    import sys

    print(json.dumps(scan(Path(sys.argv[1])), indent=1))
