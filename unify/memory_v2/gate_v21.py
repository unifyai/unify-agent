"""The gate's memory v2.1 switch (P3 Amendment B; P4 Amendment E): a config object, not a bool.

``Gate(v21=None)`` is v2. ``consolidate`` passes ``V21Config()`` when ``UNIFY_MEMORY_V21`` is on; gate code tests
``self.v21 is not None and self.v21.layout``, so later sub-plans add their own fields here.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class V21Config:
    #: The v2.1 library layout (P3): covers without channel equality, and G3's behaviour scope and per-item
    #: reduction by the import graph (D42).
    layout: bool = True
