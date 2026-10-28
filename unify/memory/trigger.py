"""What a writer pass is asked to cover (:class:`PassRequest`).

Passes are never due by a budget or a schedule here: the writer service hands the attention ledger's entries to its
passes (:mod:`.attention`), and each pass covers the episodes those entries name.
"""

from __future__ import annotations

from dataclasses import dataclass, field


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
