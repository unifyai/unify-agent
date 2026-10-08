"""The memory section at the end of the cached system prompt (spec §G2, G6–G9; v2.1 surfacing).

``UNIFY_MEMORY_V2_SURFACING`` picks it (:mod:`.switch`). ``index`` (the default) is the v2 section, kept
byte for byte: the per-function index (:func:`render_index`) and the export line. ``catalogue``
(:func:`render_memory_section`) is the v2.1 section described next.

Memory is a Python library the working model uses like any well-kept internal package, so the prompt
holds one short paragraph (where the library is, how to import from it, the README, ``help()``,
``memory.find`` and ``memory.describe``, "check the example before relying on it", and that edits to the
scratch copy are discarded but recorded) and then the **channel catalogue**: one line per channel with its
function and note counts, never one line per function. Per-function detail lives in the generated
``README.md`` and ``.memory/catalog.json`` of the export (:mod:`..catalogue`), read on demand.

The text is a pure function of the exported memory commit, the export's fixed path under ``UNIFY_HOME`` and
the harness's suspect set (which changes only at drift events, between requests): sorted, with no
timestamps, counters or request text. Two requests on the same ``main`` therefore send byte-identical
system prompts, and nothing request-specific ever enters the prefix. The catalogue section has no size
cut: it grows by one line per channel. The index grows by one line per function and is left out over its
4,000-token budget, unless ``UNIFY_MEMORY_V2_SOFT_BUDGET`` is on (then the gate never refuses growth, so
the request renders it whole).
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from pathlib import Path

from ..catalogue import README, channel_lines
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


GUIDE = (
    "Memory: a library of Python functions distilled from earlier work, at `{root}` (first on the "
    "import path; import with `from env.<channel> import <function>`). Its functions are candidates to "
    "check, not authority. `{root}/{readme}` lists every function with its signature, one-line summary "
    "and input form; `help(env.<channel>)` and `help(<function>)` show the documentation, each with a "
    "runnable example. For data you hold (a file path, bytes, text or a parsed value), `import memory` "
    "and call `memory.find(value)`: it lists the functions whose recorded inputs have the same shape; "
    "`memory.describe(<function>)` shows one function's documentation. Check a function's example before "
    "relying on it. A function raises MemoryInputError when its input differs from what it was built "
    "from, and the message says what it expected; then do the work directly. The library is a scratch "
    "copy: you may propose an improvement by editing a function there or adding "
    "`{root}/proposals/<name>.md`; the copy is discarded when the request ends, but what you wrote is "
    "recorded for the next consolidation.\n"
)


def render_memory_section(checkout: Path, suspect: Iterable[str] = ()) -> str:
    """The memory section for the export at *checkout*, or ``""`` when the library lists nothing."""
    checkout = Path(checkout)
    lines = channel_lines(checkout, suspect)
    if not lines:
        return ""
    return GUIDE.format(root=checkout, readme=README) + "\nChannels:\n" + lines
