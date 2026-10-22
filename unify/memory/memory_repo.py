# unify/memory/memory_repo.py
"""The memory repo's harness-written commits, and the signature text of a library function.

The library's items are read by :mod:`.layout`; the only commit the harness writes on ``main`` itself is a
consolidation's status commit (:meth:`MemoryRepo.status_commit`).
"""

from __future__ import annotations

import ast
from collections.abc import Mapping, Sequence

from .gitio import Repo


def _signature(fn: ast.FunctionDef | ast.AsyncFunctionDef) -> str:
    ret = f" -> {ast.unparse(fn.returns)}" if fn.returns else ""
    return f"{fn.name}({ast.unparse(fn.args)}){ret}"


class MemoryRepo:
    def __init__(self, repo: Repo) -> None:
        self.repo = repo

    def status_commit(self, changes: Mapping[str, str], evidence: Sequence[str]) -> str:
        """In library mode: one empty commit on ``main`` recording a
        consolidation's status changes, item by item (``Status: <item> <status>``) with the episodes behind them
        (``Evidence:``). The tree is unchanged: a library item leaves the index by its status, never by losing its
        code, so ``memory.show`` still reaches it. The commit gives the requests after it a new pin, so the
        prompt prefix changes only when a commit lands (cache rule).
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
