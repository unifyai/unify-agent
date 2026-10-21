"""How alike two stored functions' code is, for ``UNIFY_STORE_DEDUPE=warn``.

Each source is parsed and dumped with its docstrings and annotations dropped,
the names it binds (arguments, assignment, loop, ``with`` and ``except``
targets, nested ``def``\\ s) renamed ``VAR0``, ``VAR1``, … in order of
appearance, and its own name renamed ``FUNC``; similarity is the multiset
Jaccard of the dump's word tokens. Two copies that differ only in naming,
comments, docstrings or type hints score 1.0. This is the measure the
library-health audit of past stores used, so its threshold carries over.
"""

from __future__ import annotations

import ast
import collections
import re
import textwrap
from typing import Dict, List, Optional

# In the audit of past stores (568 function pairs), 6 pairs scored >= 0.8 and
# 2 >= 0.9. The top pair, rewind_spotify_until_artist and
# rewind_appworld_spotify_until_artist (0.96), was one function stored twice;
# the pairs between 0.8 and 0.9 were distinct operations (advance vs rewind,
# set vs ensure a minimum rating). 0.9 flags copies, not siblings.
NEAR_DUPLICATE_JACCARD = 0.9

_CONTEXT_TOKENS = {"Load", "Store", "Del"}


class _Normaliser(ast.NodeTransformer):
    def __init__(self, own_name: Optional[str], bound: Dict[str, str]) -> None:
        self.own = own_name
        self.bound = bound

    def _rename(self, name: str) -> str:
        if name == self.own:
            return "FUNC"
        return self.bound.get(name, name)

    @staticmethod
    def _strip_docstring(node: ast.AST) -> ast.AST:
        body = node.body
        if (
            body
            and isinstance(body[0], ast.Expr)
            and isinstance(body[0].value, ast.Constant)
            and isinstance(body[0].value.value, str)
        ):
            node.body = body[1:] or [ast.Pass()]
        return node

    def visit_Module(self, node: ast.Module) -> ast.AST:
        self.generic_visit(node)
        return self._strip_docstring(node)

    def visit_ClassDef(self, node: ast.ClassDef) -> ast.AST:
        self.generic_visit(node)
        return self._strip_docstring(node)

    def _visit_def(self, node: ast.AST) -> ast.AST:
        self.generic_visit(node)
        self._strip_docstring(node)
        node.returns = None
        node.name = self._rename(node.name)
        return node

    visit_FunctionDef = visit_AsyncFunctionDef = _visit_def

    def visit_Name(self, node: ast.Name) -> ast.AST:
        node.id = self._rename(node.id)
        return node

    def visit_arg(self, node: ast.arg) -> ast.AST:
        node.arg = self.bound.get(node.arg, node.arg)
        node.annotation = None
        return node

    def visit_ExceptHandler(self, node: ast.ExceptHandler) -> ast.AST:
        self.generic_visit(node)
        if node.name:
            node.name = self.bound.get(node.name, node.name)
        return node


def _bound_names(tree: ast.AST, own: Optional[str]) -> List[str]:
    found = []
    for node in ast.walk(tree):
        at = (getattr(node, "lineno", 0), getattr(node, "col_offset", 0))
        if isinstance(node, ast.arg):
            found.append((at, node.arg))
        elif isinstance(node, ast.Name) and isinstance(node.ctx, (ast.Store, ast.Del)):
            found.append((at, node.id))
        elif isinstance(node, ast.ExceptHandler) and node.name:
            found.append((at, node.name))
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            found.append((at, node.name))
    order: List[str] = []
    for _, name in sorted(found, key=lambda item: item[0]):
        if name != own and name not in order:
            order.append(name)
    return order


def code_tokens(source: str) -> List[str]:
    """The word tokens of ``source``'s normalised AST dump (raw words if it does not parse)."""
    try:
        tree = ast.parse(textwrap.dedent(source))
    except (SyntaxError, ValueError):
        return re.findall(r"\w+", source.lower())
    defs = [
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    ]
    own = defs[0].name if defs else None
    bound = {name: f"VAR{i}" for i, name in enumerate(_bound_names(tree, own))}
    tree = _Normaliser(own, bound).visit(tree)
    dump = ast.dump(tree, annotate_fields=False, include_attributes=False)
    return [t for t in re.findall(r"\w+", dump) if t not in _CONTEXT_TOKENS]


def jaccard(a: List[str], b: List[str]) -> float:
    """Multiset Jaccard of two token lists (1.0 for two empty lists)."""
    ca, cb = collections.Counter(a), collections.Counter(b)
    union = sum((ca | cb).values())
    return sum((ca & cb).values()) / union if union else 1.0


def similarity(source_a: str, source_b: str) -> float:
    """How alike two functions' code is, from 0.0 to 1.0."""
    return jaccard(code_tokens(source_a), code_tokens(source_b))


__all__ = ["NEAR_DUPLICATE_JACCARD", "code_tokens", "jaccard", "similarity"]
