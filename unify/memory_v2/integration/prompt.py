"""The memory index at the end of the cached system prompt (spec §G2, G6–G9).

The text is a pure function of the exported memory commit, the export's fixed path under ``UNIFY_HOME`` and
the harness's suspect set (which changes only at drift events, between requests): sorted, with no
timestamps, counters or request text. Two requests on the same ``main`` therefore send byte-identical
system prompts, and nothing request-specific ever enters the prefix.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from pathlib import Path

from ..index import IndexOverBudget, index_with_names
from ..memory_repo import items

logger = logging.getLogger(__name__)

INDEX_BUDGET_TOKENS = 4000


def export_line(checkout: Path) -> str:
    return (
        f"These files are a scratch copy of the library at `{checkout}`, first on the import "
        "path; anything written there is discarded when the request ends.\n"
    )


def render_index(checkout: Path, suspect: Iterable[str] = ()) -> str:
    """The index for the export at *checkout*, or ``""`` when it lists no item or is over budget."""
    return render_memory(checkout, suspect)[0]


def render_memory(checkout: Path, suspect: Iterable[str] = ()) -> tuple[str, dict]:
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
                budget_tokens=INDEX_BUDGET_TOKENS,
                suspect=set(suspect),
            )
        except IndexOverBudget as exc:  # the gate's budget check makes this unexpected
            logger.warning("memory v2: index left out of the prompt: %s", exc)
        else:
            text = index + "\n" + export_line(checkout)
    return text, record_shown(text, channels=channels, items=ids, renderer="index")
