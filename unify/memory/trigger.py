"""What a writer pass is asked to cover (:class:`PassRequest`), and which recorded episodes a pass may take yet
(:func:`admit_settled`: an episode enters a pass only once its episode-end fork has settled).

Passes are never due by a budget or a schedule here: the writer service hands the attention ledger's entries to its
passes (:mod:`.attention`), and each pass covers the episodes those entries name.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class PassRequest:
    kind: str
    channel: str | None
    episodes: list[str]
    lift: bool
    experience_tokens: int | None = (
        None  # batched passes: the experience the pass covers
    )
    #: batched passes under admission: pending episodes whose fork has not settled; they roll to the next pass
    held: list[str] = field(default_factory=list)
    #: the covered episodes whose capture is a verdict (kind ``"verdict"``: the reason the pass is due now)
    verdicts: list[str] = field(default_factory=list)
    #: kind ``"attention"``: the attention-ledger entries handed to the pass (:class:`.attention.AttentionEntry`); its
    #: ``episodes`` are theirs (:func:`.attention.pass_episodes`)
    attention: list = field(default_factory=list)


def admit_settled(
    staging_root: Path | str,
    episode_ids: list[str],
) -> tuple[list[str], list[str]]:
    """The admission rule, as a pure function: ``(admitted, waiting)`` in the given order. An episode is settled once
    its staging dir under *staging_root* (``<state>/memory-staging``) holds its status file (``fork.json`` with any
    status, ``"timeout"`` and ``"lost"`` included). An episode with no staging dir had no fork and is admitted (with
    the fork on, the dir is reserved before the episode is recorded). It reads files only; recording ``"lost"`` or
    ``"timeout"`` for a fork that can no longer settle itself is :func:`.integration.fork.settled`'s work.
    """
    from .staging import STATUS_FILES

    admitted, waiting = [], []
    root = Path(staging_root)
    for eid in episode_ids:
        d = root / eid
        if not os.path.isdir(d) or os.path.islink(d):
            admitted.append(eid)
        elif any(os.path.lexists(d / name) for name in STATUS_FILES):
            admitted.append(eid)
        else:
            waiting.append(eid)
    return admitted, waiting


