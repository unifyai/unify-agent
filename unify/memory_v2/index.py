# unify/memory_v2/index.py
"""The compact index Luna sees in its cached prefix (spec §6, D3).

A function's line ends with the form its first argument takes, from its docstring's ``Input:`` line, when
that names a known form (:data:`.manifest.INPUT_KINDS`): ``- `parse_load_log(data)` — Parse ... (input:
text)``. A function whose ``Effect:`` line says ``write`` is marked ``(writes)``: the gate refuses a function
that covers a recorded write call without saying so (stage 7), so every such function carries the mark.
The index is a function of the tree alone, so the same commit gives the same bytes.
"""

from __future__ import annotations

import math
from pathlib import Path

from .manifest import INPUT_KINDS
from .memory_repo import items

HEADER = (
    "Memory library: candidates to check, not authority. Import with `from env.<name> import ...`. "
    "Each function checks its inputs and raises MemoryInputError with a diagnosis when they differ from what it "
    "was built from; when that happens, do the work directly.\n"
)


class IndexOverBudget(ValueError):
    pass


def estimate_tokens(text: str) -> int:
    return math.ceil(len(text) / 4)


def build_index(
    checkout: Path,
    *,
    budget_tokens: int = 4000,
    suspect: set[str] = frozenset(),
    channels: list[str] | None = None,
) -> str:
    rep = items(checkout)
    by_channel: dict[str, list[str]] = {}
    for it in rep.items:
        if it.kind == "workflow" or not it.listed:
            continue
        ch = it.path.split("/")[1]
        if channels is not None and ch not in channels:
            continue
        line = (
            f"- `{it.signature}` — {it.doc}"
            if it.kind == "env_function"
            else f"- note: {it.name} — {it.doc}"
        )
        if it.kind == "env_function" and it.input in INPUT_KINDS:
            line += f" (input: {it.input})"
        if it.kind == "env_function" and it.effect == "write":
            line += " (writes)"
        by_channel.setdefault(ch, []).append(line)
    parts = [HEADER]
    for ch in sorted(by_channel):
        flag = (
            " (suspect: the environment changed since these were built; verify before use)"
            if ch in suspect
            else ""
        )
        parts.append(f"\n## env.{ch}{flag}\n" + "\n".join(by_channel[ch]) + "\n")
    text = "".join(parts)
    if estimate_tokens(text) > budget_tokens:
        raise IndexOverBudget(
            f"index needs {estimate_tokens(text)} tokens; budget {budget_tokens}",
        )
    return text
