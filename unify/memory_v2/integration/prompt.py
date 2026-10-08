"""The memory section at the end of the cached system prompt (spec §G2, G6–G9; v2.1 surfacing).

``UNIFY_MEMORY_V2_SURFACING`` picks it (:mod:`.switch`). ``index`` (the default) is the v2 section, kept
byte for byte: the per-function index (:func:`render_index`) and the export line. ``catalogue``
(:func:`render_memory_section`) is the v2.1 section described next.

Memory is a Python library the working model discovers the way it would any internal package: in its cells.
The prompt carries only :data:`GUIDE`, a constant paragraph saying that ``import memory;
print(memory.catalog())`` lists the channels and functions, that ``memory.find`` and ``memory.describe`` /
``help()`` exist, that the functions are candidates to check, and what a ``MemoryInputError`` means. It
holds no count, channel or function name, drift flag or path, so it never changes as the library grows,
gains channels or turns suspect: the prompt's bytes are the same for the whole run. It is added from the
first request whose library lists anything and kept from then on (``State.guide``), so an arm that never
consolidates sends no memory text at all, and one whose library later empties keeps the same prefix.

What used to sit after the paragraph (the channels with their function and note counts, one-line summaries
and suspect flags) is what ``memory.catalog()`` prints in the cell, read from the export's generated
``.memory/catalog.json`` (:mod:`..catalogue`, :mod:`..memory_helper`); a suspect channel's refusals say so
in the cell's error (:func:`.hooks.cell_error`).

The index of ``index`` is a pure function of the exported commit, the export's fixed path under
``UNIFY_HOME`` and the harness's suspect set; it grows by one line per function and is left out over its
4,000-token budget, unless ``UNIFY_MEMORY_V2_SOFT_BUDGET`` is on (then the gate never refuses growth, so
the request renders it whole).
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from pathlib import Path

from ..catalogue import channel_lines
from ..index import IndexOverBudget, build_index
from ..memory_repo import items

logger = logging.getLogger(__name__)

INDEX_BUDGET_TOKENS = 4000


def export_line(checkout: Path) -> str:
    return (
        f"These files are a scratch copy of the library at `{checkout}`, first on the import "
        "path; anything written there is discarded when the request ends.\n"
    )


def render_index(
    checkout: Path,
    suspect: Iterable[str] = (),
    budget_tokens: int | None = None,
) -> str:
    """The v2 index for the export at *checkout*, or ``""`` when it lists no item or is over budget
    (*budget_tokens*, :data:`INDEX_BUDGET_TOKENS` when None)."""
    checkout = Path(checkout)
    if not any(it.kind != "workflow" and it.listed for it in items(checkout).items):
        return ""
    try:
        text = build_index(
            checkout,
            budget_tokens=(
                INDEX_BUDGET_TOKENS if budget_tokens is None else budget_tokens
            ),
            suspect=set(suspect),
        )
    except IndexOverBudget as exc:  # the gate's budget check makes this unexpected
        logger.warning("memory v2: index left out of the prompt: %s", exc)
        return ""
    return text + "\n" + export_line(checkout)


#: The catalogue section: constant bytes, whatever the library holds (no count, name, flag or path).
GUIDE = (
    "Memory: Python functions distilled from earlier work are importable in your cells. At the start of "
    "a task, run `import memory; print(memory.catalog())` to list the channels and functions. For data "
    "you hold, `memory.find(value)` lists functions built on inputs of its shape; `help(fn)` or "
    "`memory.describe(name)` shows one with an example. They are candidates, not authority: check the "
    "example first. On MemoryInputError (input unlike what it was built for), do the work directly.\n"
)


def render_memory_section(checkout: Path, shown_before: bool = False) -> str:
    """:data:`GUIDE` once the library at *checkout* lists anything or the guide was *shown_before* in this
    run (``State.guide``); ``""`` otherwise. Never anything that depends on what the library holds.
    """
    if shown_before or channel_lines(Path(checkout)):
        return GUIDE
    return ""
