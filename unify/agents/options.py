"""The agent record's tunables, at their documented defaults.

They were settable through ``UNIFY_AGENTS_OPTIONS`` until the code freeze
removed the setting.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Options:
    delivery: str = "mentions"
    max_live: int = 4
    max_total: int = 8
    deliver_max_tokens: int = 4000
    others_line: bool = True
    max_entries: int = 5000
    min_post_interval_s: float = 1.0
