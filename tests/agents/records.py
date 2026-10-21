"""A run record for the agents tests, as every run now has one (:func:`unify.agents.run_record.join_run_record`).

:func:`run_record` opens a fresh run's record in a new home with *root* joined as the request's agent (an actor), the
posting interval off unless *opts* set it."""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

from unify.agents import run_record as rr
from unify.agents.options import Options


def run_record(*, root: str = "root", home: Path | str | None = None, user_reads: bool = False, **opts: Any) -> Any:
    opts.setdefault("min_post_interval_s", 0)
    rec = rr.join_run_record(home if home is not None else tempfile.mkdtemp(), options=Options(**opts))
    rec.root = root
    rec.user_reads = user_reads
    rec.join(root, "actor")
    return rec
