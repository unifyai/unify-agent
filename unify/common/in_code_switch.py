"""Merge-time shim: tools are always called from code.

Every agent has one model-API tool, ``execute_code``, and calls every other tool as ``tools.<name>(...)`` in its
cells. The memory side still asks; this module goes once it no longer does (CLEAN-CORE-LOG.md, merge-time shims).
"""

from __future__ import annotations


def tools_in_code() -> bool:
    """Always True."""
    return True
