# unify/memory_v2/index.py
"""The v2 compact index: one line per function (spec §6, D3).

It ends the actor's prompt under ``UNIFY_MEMORY_V2_SURFACING=index`` (the default) and is Sol's first message
there; G4 caps it unless ``UNIFY_MEMORY_V2_SOFT_BUDGET`` is on. Under ``catalogue`` the prompt ends with a
constant guide (:mod:`.integration.prompt`) and the catalogue is the export's generated README and
``memory.catalog()`` (:mod:`.catalogue`); the index is then kept for the consolidation end event's
``index_tokens`` measurement.

A function's line ends with the form its first argument takes, from its docstring's ``Input:`` line, when
that names a known form (:data:`.manifest.INPUT_KINDS`): ``- `parse_load_log(data)` — Parse ... (input:
text)``. A function that replaces a value it computed from its input under a condition
(:mod:`.analysis.overrides`) has `` (applies a rule; check it)`` after its summary. A function whose
``Effect:`` line says ``write`` is marked ``(writes)``: the gate refuses a function that covers a recorded
write call without saying so (stage 7), so every such function carries the mark. The index is a function of
the tree alone, so the same commit gives the same bytes.
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


RULE_FLAG = " (applies a rule; check it)"


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
    return index_with_names(
        checkout,
        budget_tokens=budget_tokens,
        suspect=suspect,
        channels=channels,
    )[0]


def index_with_names(
    checkout: Path,
    *,
    budget_tokens: int = 4000,
    suspect: set[str] = frozenset(),
    channels: list[str] | None = None,
) -> tuple[str, list[str], list[str]]:
    """:func:`build_index`'s text, with the item ids whose own lines it carries and the channels it heads
    (both sorted): what the index shows, for the use record (``analysis.use.record_shown``).
    """
    rep = items(checkout)
    by_channel: dict[str, list[str]] = {}
    shown: list[str] = []
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
        if it.kind == "env_function" and it.rule_line:
            line += RULE_FLAG
        if it.kind == "env_function" and it.input in INPUT_KINDS:
            line += f" (input: {it.input})"
        if it.kind == "env_function" and it.effect == "write":
            line += " (writes)"
        if it.kind == "env_function":
            shown.append(it.item_id)
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
    return text, sorted(shown), sorted(by_channel)
