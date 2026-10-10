# unify/memory_v2/memory_repo.py
"""The memory repo's layout, its items, and the harness's remove-only edits (spec §4, §9).

Limitation (v0): hiding a note item removes the whole NOTES.md file, not one section; notes are
rewritten by the next pass.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import ast
import re
from dataclasses import dataclass, field
from pathlib import Path

from .analysis.overrides import find_overrides
from .gitio import Repo

_EFFECT = re.compile(r"^\s*Effect:\s*(read|write|unknown)\s*$", re.M)
# The declared input form (a manifest ``input``, :data:`.manifest.INPUT_KINDS`): any word is read here; the
# gate checks it against the manifest and the index shows only a known form.
_INPUT = re.compile(r"^\s*Input:\s*([A-Za-z_]{1,40})\s*$", re.M)
_INPUT_LINE = re.compile(
    r"^[ \t]*Input:",
    re.M,
)  # every Input: line, well-formed or not


@dataclass
class Item:
    item_id: str
    kind: str
    path: str
    name: str
    signature: str
    doc: str
    effect: str
    listed: bool
    input: str = ""  # the docstring's ``Input:`` form, or "" without one
    input_lines: int = (
        0  # how many ``Input:`` lines the docstring holds (the gate allows one)
    )
    # an environment function that replaces a value it computed from its input under a condition
    # (:mod:`.analysis.overrides`): the first such line of its module, else 0; ``rule_unchecked`` when the
    # analysis stopped at a bound
    rule_line: int = 0
    rule_unchecked: bool = False


@dataclass
class ItemsReport:
    items: list[Item] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


def _all_node(tree: ast.Module) -> ast.Assign | ast.AnnAssign | None:
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == "__all__" for t in node.targets
        ):
            return node
        if (
            isinstance(node, ast.AnnAssign)
            and isinstance(node.target, ast.Name)
            and node.target.id == "__all__"
            and node.value is not None
        ):
            return node
    return None


def _all_names(tree: ast.Module) -> list[str] | None:
    node = _all_node(tree)
    if node is not None and isinstance(node.value, (ast.List, ast.Tuple)):
        return [
            e.value
            for e in node.value.elts
            if isinstance(e, ast.Constant) and isinstance(e.value, str)
        ]
    return None


def _signature(fn: ast.FunctionDef | ast.AsyncFunctionDef) -> str:
    ret = f" -> {ast.unparse(fn.returns)}" if fn.returns else ""
    return f"{fn.name}({ast.unparse(fn.args)}){ret}"


def _slug(heading: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", heading.lower()).strip("-")


def items(checkout: Path) -> ItemsReport:
    rep = ItemsReport()
    for mod in sorted(checkout.glob("env/*/__init__.py")):
        rel = mod.relative_to(checkout).as_posix()
        channel = mod.parent.name
        try:
            tree = ast.parse(mod.read_text())
        except SyntaxError as exc:
            rep.errors.append(f"{rel}: {exc.msg} (line {exc.lineno})")
            continue
        except ValueError as exc:  # includes UnicodeDecodeError
            rep.errors.append(f"{rel}: {exc}")
            continue
        listed = _all_names(tree)
        for node in tree.body:
            if isinstance(
                node,
                (ast.FunctionDef, ast.AsyncFunctionDef),
            ) and not node.name.startswith("_"):
                doc = ast.get_docstring(node) or ""
                m = _EFFECT.search(doc)
                form = _INPUT.search(doc)
                rule = find_overrides(node)
                rep.items.append(
                    Item(
                        f"env/{channel}:{node.name}",
                        "env_function",
                        rel,
                        node.name,
                        _signature(node),
                        doc.strip().splitlines()[0] if doc.strip() else "",
                        m.group(1) if m else "",
                        listed is None or node.name in listed,
                        form.group(1) if form else "",
                        len(_INPUT_LINE.findall(doc)),
                        rule.line,
                        rule.truncated,
                    ),
                )
    for notes in sorted(checkout.glob("env/*/NOTES.md")):
        rel = notes.relative_to(checkout).as_posix()
        for heading, body in _sections(notes.read_text()):
            rep.items.append(
                Item(
                    f"{rel}#{_slug(heading)}",
                    "env_note",
                    rel,
                    heading,
                    "",
                    (body.strip().splitlines() or [""])[0],
                    "",
                    True,
                ),
            )
    for wf in sorted(checkout.glob("workflows/*.md")):
        rel = wf.relative_to(checkout).as_posix()
        front = _front_matter(wf.read_text())
        rep.items.append(
            Item(
                rel,
                "workflow",
                rel,
                front.get("title", wf.stem),
                "",
                "",
                "",
                front.get("listed", "true") != "false",
            ),
        )
    return rep


def _sections(text: str) -> list[tuple[str, str]]:
    out, cur, buf = [], None, []
    for ln in text.splitlines():
        if ln.startswith("## "):
            if cur is not None:
                out.append((cur, "\n".join(buf)))
            cur, buf = ln[3:].strip(), []
        elif cur is not None:
            buf.append(ln)
    if cur is not None:
        out.append((cur, "\n".join(buf)))
    return out


def _front_matter(text: str) -> dict[str, str]:
    if not text.startswith("---\n"):
        return {}
    end = text.find("\n---", 4)
    pairs = [ln.split(":", 1) for ln in text[4:end].splitlines() if ":" in ln]
    return {k.strip(): v.strip() for k, v in pairs}


def _drop_from_all(source: str, name: str) -> str:
    tree = ast.parse(source)
    node = _all_node(tree)
    if node is None:
        return source
    names = [n for n in (_all_names(tree) or []) if n != name]
    lines = source.splitlines(keepends=True)
    new = f"__all__ = {names!r}\n".replace("'", '"')
    return "".join(lines[: node.lineno - 1]) + new + "".join(lines[node.end_lineno :])


def unlist_function(source: str, name: str) -> str:
    tree = ast.parse(source)
    if _all_node(tree) is not None:
        return _drop_from_all(source, name)
    public = [
        n.name
        for n in tree.body
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
        and not n.name.startswith("_")
        and n.name != name
    ]
    after = 0
    for i, n in enumerate(tree.body):
        is_doc = (
            i == 0
            and isinstance(n, ast.Expr)
            and isinstance(n.value, ast.Constant)
            and isinstance(n.value.value, str)
        )
        is_future = isinstance(n, ast.ImportFrom) and n.module == "__future__"
        if is_doc or is_future:
            after = n.end_lineno
        else:
            break
    lines = source.splitlines(keepends=True)
    new = f"__all__ = {public!r}\n".replace("'", '"')
    if after and lines[after - 1] and not lines[after - 1].endswith("\n"):
        lines[after - 1] += "\n"
    return "".join(lines[:after]) + new + "".join(lines[after:])


def remove_function(source: str, name: str) -> str:
    source = _drop_from_all(source, name)
    tree = ast.parse(source)
    lines = source.splitlines(keepends=True)
    for node in tree.body:
        if (
            isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name == name
        ):
            start = min([node.lineno] + [d.lineno for d in node.decorator_list]) - 1
            return "".join(lines[:start] + lines[node.end_lineno :])
    raise KeyError(name)


class MemoryRepo:
    def __init__(self, repo: Repo) -> None:
        self.repo = repo

    def hide(self, item_id: str, reason: str, evidence: list[str]) -> str:
        base = self.repo.head()
        with self.repo.temp_checkout(base) as wt:
            if ":" in item_id:  # env function
                rel_dir, name = item_id.split(":", 1)
                mod = wt / rel_dir / "__init__.py"
                mod.write_text(remove_function(mod.read_text(), name))
            else:
                path = wt / item_id.split("#", 1)[0]
                path.unlink()
            sha = self.repo.commit_all(
                wt,
                f"hide {item_id}",
                {"Hide": reason, "Evidence": evidence},
            )
        self.repo.fast_forward("main", sha, expected_old=base)
        return sha

    def status_commit(self, changes: Mapping[str, str], evidence: Sequence[str]) -> str:
        """memory v2.1 (spec §4.4, §10.1; D10's harness commit): one empty commit on ``main`` recording a
        consolidation's status changes, item by item (``Status: <item> <status>``) with the episodes behind them
        (``Evidence:``). The tree is unchanged: a v2.1 item leaves the index by its status, never by losing its
        code, so ``memory.show`` still reaches it. The commit gives the requests after it a new pin, so the
        prompt prefix changes only when a commit lands (cache rule). v2's :meth:`hide` is unchanged.
        """
        if not changes:
            raise ValueError("no status change to record")
        base = self.repo.head()
        with self.repo.temp_checkout(base) as wt:
            sha = self.repo.commit_all(
                wt,
                f"status: {len(changes)} item(s)",
                {
                    "Status": [f"{item} {changes[item]}" for item in sorted(changes)],
                    "Evidence": sorted(set(evidence)),
                },
                allow_empty=True,
            )
        self.repo.fast_forward("main", sha, expected_old=base)
        return sha
