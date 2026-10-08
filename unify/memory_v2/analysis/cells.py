from __future__ import annotations

import ast
import json
from dataclasses import dataclass
from typing import Any

from ..episodes import Cell


@dataclass(frozen=True)
class Hole:
    expr: str


@dataclass
class CallSite:
    cell: int
    order: int
    channel: str
    method: str
    kwargs: dict[str, Any]
    args: list[Any]


def _raw(content: Any) -> str:
    if isinstance(content, str):
        return content
    return "".join(p.get("text", "") for p in content or [] if isinstance(p, dict))


def _stdout(raw: str) -> str:
    if "\n--- stdout ---\n" in raw:
        raw = raw.split("\n--- stdout ---\n", 1)[1]
    return raw.split("\n--- stderr ---\n", 1)[0]


def cells_from_transcript(lines: list[dict]) -> list[Cell]:
    pending: dict[str, str] = {}
    cells: list[Cell] = []
    for ln in lines:
        msg = ln.get("message") if ln.get("type") == "message" else None
        if not msg:
            continue
        for tc in msg.get("tool_calls") or []:
            fn = tc.get("function", {})
            if fn.get("name") == "execute_code":
                try:
                    args = fn.get("arguments") or "{}"
                    if not isinstance(args, dict):
                        args = json.loads(args)
                    pending[tc.get("id")] = args.get("code") or ""
                except (json.JSONDecodeError, AttributeError, TypeError):
                    pending[tc.get("id")] = ""
        if msg.get("role") == "tool" and msg.get("tool_call_id") in pending:
            code = pending.pop(msg["tool_call_id"])
            raw = _raw(msg.get("content"))
            err = None
            if "\n--- stderr ---\n" in raw:
                err = raw.split("\n--- stderr ---\n", 1)[1] or None
            cells.append(Cell(len(cells), code, _stdout(raw), err))
    return cells


def _value(node: ast.AST) -> Any:
    try:
        return ast.literal_eval(node)
    except (ValueError, SyntaxError, TypeError):
        return Hole(ast.unparse(node))


def _chain(node: ast.AST) -> list[str] | None:
    names = []
    while isinstance(node, ast.Attribute):
        names.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        return [node.id] + names[::-1]
    return None


def _calls_in_eval_order(node: ast.AST):
    """Post-order: arguments are yielded before the call that consumes them."""
    for child in ast.iter_child_nodes(node):
        yield from _calls_in_eval_order(child)
    if isinstance(node, ast.Call):
        yield node


def call_sites(cells: list[Cell], roots: tuple[str, ...] = ("apis",)) -> list[CallSite]:
    sites: list[CallSite] = []
    for c in cells:
        try:
            tree = ast.parse(c.code)
        except SyntaxError:
            continue
        for n in _calls_in_eval_order(tree):
            chain = _chain(n.func)
            if chain and chain[0] in roots and len(chain) == 3:
                sites.append(
                    CallSite(
                        c.index,
                        len(sites),
                        chain[1],
                        chain[2],
                        {k.arg: _value(k.value) for k in n.keywords if k.arg},
                        [_value(a) for a in n.args],
                    ),
                )
    return sites
