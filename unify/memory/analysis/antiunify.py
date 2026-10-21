"""Anti-unification of environment call sequences (Plotkin-style, over aligned calls). Never of whole solutions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .cells import CallSite, Hole


@dataclass
class Template:
    steps: list[tuple[str, str, dict[str, Any]]]
    params: list[str]


def _lcs(a: list[tuple[str, str]], b: list[tuple[str, str]]) -> list[tuple[int, int]]:
    dp = [[0] * (len(b) + 1) for _ in range(len(a) + 1)]
    for i in range(len(a) - 1, -1, -1):
        for j in range(len(b) - 1, -1, -1):
            dp[i][j] = (
                dp[i + 1][j + 1] + 1
                if a[i] == b[j]
                else max(dp[i + 1][j], dp[i][j + 1])
            )
    pairs, i, j = [], 0, 0
    while i < len(a) and j < len(b):
        if a[i] == b[j]:
            pairs.append((i, j))
            i += 1
            j += 1
        elif dp[i + 1][j] >= dp[i][j + 1]:
            i += 1
        else:
            j += 1
    return pairs


def antiunify(seqs: list[list[CallSite]]) -> Template | None:
    if len(seqs) < 2:
        return None

    def keys(s):
        return [(c.channel, c.method) for c in s]

    aligned = [[c] for c in seqs[0]]
    for other in seqs[1:]:
        pairs = _lcs([(g[0].channel, g[0].method) for g in aligned], keys(other))
        aligned = [aligned[i] + [other[j]] for i, j in pairs]
    params: list[str] = []
    steps = []
    for group in aligned:
        merged: dict[str, Any] = {}
        names = set.intersection(*[set(c.kwargs) for c in group])
        for name in sorted(names):
            vals = [c.kwargs[name] for c in group]
            if (
                any(isinstance(v, Hole) for v in vals)
                or len({repr(v) for v in vals}) > 1
            ):
                p = f"$p{len(params)}"
                params.append(p)
                merged[name] = p
            else:
                merged[name] = vals[0]
        steps.append((group[0].channel, group[0].method, merged))
    return Template(steps, params)
