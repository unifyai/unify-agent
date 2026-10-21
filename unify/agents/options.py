"""The agent record's tunables, at their documented defaults."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class Options:
    delivery: str = "mentions"
    max_live: int = 4
    max_total: int = 8
    deliver_max_tokens: int = 4000
    others_line: bool = True
    #: None: no count cap (the run record, which lasts a whole run: a cap would refuse posts mid-run)
    max_entries: Optional[int] = 5000
    min_post_interval_s: float = 1.0
    #: the run record's byte backstop: past it, posts from model code are refused and the harness's entries are cut
    #: short (None: no byte backstop, as a session record)
    max_record_bytes: Optional[int] = None
