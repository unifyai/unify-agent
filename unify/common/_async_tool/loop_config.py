from secrets import token_hex
from contextvars import ContextVar

from ..llm_helpers import short_id

# Hierarchical lineage of nested async tool loops (propagates via contextvars)
TOOL_LOOP_LINEAGE: ContextVar[list[str]] = ContextVar("TOOL_LOOP_LINEAGE", default=[])

__all__ = [
    "TOOL_LOOP_LINEAGE",
    "LoopConfig",
]


class LoopConfig:
    """A loop's identity for its logs and transcript: its id, and its lineage with a random suffix on its own
    segment."""

    def __init__(self, loop_id, lineage, parent_lineage):
        self._loop_id = loop_id if loop_id is not None else short_id()
        _suffix = token_hex(2)
        # Embed the suffix into the leaf segment so the hierarchy array is
        # self-describing for unique parent-child linking (every segment
        # carries per-invocation identity).
        if lineage is not None:
            self._lineage = list(lineage)
            if self._lineage:
                self._lineage[-1] = f"{self._lineage[-1]}({_suffix})"
        else:
            self._lineage = [*parent_lineage, f"{self._loop_id}({_suffix})"]
        self._label = (
            "->".join(self._lineage) if self._lineage else f"{self._loop_id}({_suffix})"
        )

    @property
    def loop_id(self):
        return self._loop_id

    @property
    def lineage(self):
        return self._lineage

    @property
    def label(self):
        return self._label
