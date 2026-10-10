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
from ..index import IndexOverBudget, index_with_names
from ..memory_repo import items
from .. import prompts_v21 as _prompts_v21

logger = logging.getLogger(__name__)

#: The reviewed v2.1 actor guide (P7, spec §12.1): constant bytes, naming nothing the library holds.
GUIDE_V21 = _prompts_v21.GUIDE_V21

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
    return render_memory(checkout, suspect, budget_tokens)[0]


def render_memory(
    checkout: Path,
    suspect: Iterable[str] = (),
    budget_tokens: int | None = None,
) -> tuple[str, dict]:
    """(:func:`render_index`'s text, what it shows as :func:`..analysis.use.record_shown` records it).

    The record holds the channel and item names the text carries and the text's digest, never the text;
    the request's use record reads it, so it never has to find the section in the prompt by its wording.
    """
    from ..analysis.use import record_shown

    checkout = Path(checkout)
    text, ids, channels = "", [], []
    if any(it.kind != "workflow" and it.listed for it in items(checkout).items):
        try:
            index, ids, channels = index_with_names(
                checkout,
                budget_tokens=(
                    INDEX_BUDGET_TOKENS if budget_tokens is None else budget_tokens
                ),
                suspect=set(suspect),
            )
        except IndexOverBudget as exc:  # the gate's budget check makes this unexpected
            logger.warning("memory v2: index left out of the prompt: %s", exc)
        else:
            text = index + "\n" + export_line(checkout)
    return text, record_shown(text, channels=channels, items=ids, renderer="index")


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
    return render_catalogue(checkout, shown_before)[0]


def render_catalogue(checkout: Path, shown_before: bool = False) -> tuple[str, dict]:
    """(:func:`render_memory_section`'s text, what it shows as :func:`..analysis.use.record_shown` records it).

    The guide names no channel or function, so the record names none: it says the guide was shown (its
    digest and size, renderer ``catalogue``) and nothing per item. Per-item exposure under ``catalogue``
    comes from the cells' own ``memory.catalog()`` / ``find`` / ``describe`` / ``help`` calls and imports.
    """
    from ..analysis.use import record_shown

    checkout = Path(checkout)
    text = GUIDE if shown_before or channel_lines(checkout) else ""
    return text, record_shown(text, channels=(), items=(), renderer="catalogue")


def location_line(checkout: Path) -> str:
    return f"Library files: `{checkout}`, holding INDEX.md, links.json, memory/ and notes/.\n"


def render_memory_v21(
    checkout: Path,
    budget_tokens: int | None = None,
) -> tuple[str, dict]:
    """``UNIFY_MEMORY_V21=on`` (spec v2.1 §4.5, §6): the guide, the copy's location, then the index view
    (:func:`..library_index.index_view` of the copy's ``INDEX.md``), or ``""`` while the index lists no item.

    The hooks append it last in the system prompt, after the clock line (:func:`.hooks.system_prompt`). The
    guide and the location are the same for the whole run, and the view changes only when a consolidation
    commits or changes a status (cache rule), so each pass changes only the section's tail.
    """
    from ..analysis.use import record_shown
    from ..layout import INDEX_FILE
    from ..library_index import INDEX_VIEW_TOKENS, index_view, listed, packages_of

    checkout = Path(checkout)
    try:
        index = (checkout / INDEX_FILE).read_text(encoding="utf-8")
    except OSError:
        index = ""
    text = ""
    if listed(index):
        budget = INDEX_VIEW_TOKENS if budget_tokens is None else budget_tokens
        text = GUIDE_V21 + location_line(checkout) + "\n" + index_view(index, budget)
    # per-item exposure under v2.1 is P5's (use records); the record keeps the packages and the digest
    return text, record_shown(
        text,
        channels=packages_of(index),
        items=(),
        renderer="index_v21",
    )
