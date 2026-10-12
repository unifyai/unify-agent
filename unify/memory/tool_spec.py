"""The memory tools' definitions, re-exported from the shared registry (:mod:`unify.common.tool_registry`), which
holds one definition per tool for every agent. The names here stay so that callers and the scorer's readers that
import them are unchanged: ``ToolSpec`` is the registry's ``RegisteredTool``.
"""

from __future__ import annotations

from unify.common.tool_registry import (
    GENERAL,
    GENERAL_REFUSALS,
    KEPT_STARTS,
    REFUSED,
    Refusal,
    form_of,
    refusal,
    refusal_codes,
    tool_json,
    tools_for,
    tools_section,
)
from unify.common.tool_registry import RegisteredTool as ToolSpec
from unify.common.tool_registry import registry as _registry

#: The memory pass roles the earlier model-API surface is keyed on (see agent_profiles.py).
ROLES = ("write", "curate", "helper")


def registry() -> tuple[ToolSpec, ...]:
    """Every registered tool, in registry order."""
    return tuple(_registry().values())


__all__ = [
    "GENERAL",
    "GENERAL_REFUSALS",
    "KEPT_STARTS",
    "REFUSED",
    "ROLES",
    "Refusal",
    "ToolSpec",
    "form_of",
    "refusal",
    "refusal_codes",
    "registry",
    "tool_json",
    "tools_for",
    "tools_section",
]
