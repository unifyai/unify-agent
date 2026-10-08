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

from ..index import IndexOverBudget, build_index
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
    checkout = Path(checkout)
    if not any(it.kind != "workflow" and it.listed for it in items(checkout).items):
        return ""
    try:
        text = build_index(
            checkout,
            budget_tokens=INDEX_BUDGET_TOKENS,
            suspect=set(suspect),
        )
    except IndexOverBudget as exc:  # the gate's budget check makes this unexpected
        logger.warning("memory v2: index left out of the prompt: %s", exc)
        return ""
    return text + "\n" + export_line(checkout)
