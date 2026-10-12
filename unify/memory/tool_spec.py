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
)
from unify.common import tool_registry as _tool_registry
from unify.common.tool_registry import RegisteredTool as ToolSpec
from unify.common.tool_registry import registry as _registry

#: The memory pass roles the earlier model-API surface is keyed on (see agent_profiles.py).
ROLES = ("write", "curate", "helper")


def registry() -> tuple[ToolSpec, ...]:
    """Every registered tool, in registry order."""
    return tuple(_registry().values())


# Each reads this module's registry() when called, so a caller that narrows it narrows these too.


def tools_for(role: str, flags: frozenset[str] = frozenset()) -> list:
    """The tools *role* got as model-API tools with the configuration *flags* on (see the registry's
    ``tools_for``)."""
    return _tool_registry.tools_for(role, flags, registry())


def tool_json(role: str, flags: frozenset[str] = frozenset()) -> list[dict]:
    return _tool_registry.tool_json(role, flags, registry())


def tools_section(role: str, flags: frozenset[str] = frozenset()) -> str:
    """The Tools section of *role*'s instructions: one bullet per tool it actually gets."""
    return _tool_registry.tools_section(role, flags, registry())


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
