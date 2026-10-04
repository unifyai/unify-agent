"""The actor environment for the namespaces an environment registered.

``UNIFY_ENV_NAMESPACES`` (``unify/function_manager/primitives/environment.py``)
registers an environment's callable surface as ``primitives.<name>``
namespaces. This environment puts them in the actor's sandbox beside
``primitives.actor``: it shares the ``primitives`` namespace, so the actor
merges the two into one ``Primitives`` object whose scope holds both, and its
prompt context is a short listing. Method documentation stays out of the
prompt; each method is a primitive row that function search returns, and
``help(primitives.<name>.<method>)`` reads it in the sandbox.
"""

from __future__ import annotations

import asyncio
from typing import Dict, List, Optional

from unify.actor.environments.base import BaseEnvironment, ToolMetadata
from unify.function_manager.primitives.environment import (
    environment_modules,
    environment_namespaces,
    environment_surface,
)
from unify.function_manager.primitives.registry import get_registry


class EnvironmentNamespacesEnvironment(BaseEnvironment):
    """``primitives.<name>`` for every namespace the environment registered."""

    NAMESPACE = "primitives"

    def __init__(
        self,
        *,
        clarification_up_q: Optional[asyncio.Queue[str]] = None,
        clarification_down_q: Optional[asyncio.Queue[str]] = None,
    ) -> None:
        from unify.function_manager.primitives import Primitives, PrimitiveScope

        self._aliases = frozenset(environment_namespaces())
        if not self._aliases:
            raise ValueError("no environment namespace is registered")
        self._primitives = Primitives(
            primitive_scope=PrimitiveScope(scoped_managers=self._aliases),
        )
        super().__init__(
            instance=self._primitives,
            namespace=self.NAMESPACE,
            clarification_up_q=clarification_up_q,
            clarification_down_q=clarification_down_q,
        )

    @property
    def prompt_documented_names(self) -> frozenset[str]:
        """None: the listing names namespaces, so every method stays searchable."""
        return frozenset()

    def get_instance(self):
        return self._primitives

    def get_tools(self) -> Dict[str, ToolMetadata]:
        registry = get_registry()
        tools: Dict[str, ToolMetadata] = {}
        for alias, namespace in sorted(environment_namespaces().items()):
            if alias not in self._aliases:
                continue
            for method in namespace.methods:
                fq_name = f"{self.NAMESPACE}.{alias}.{method.name}"
                tools[fq_name] = ToolMetadata(
                    name=fq_name,
                    is_impure=method.effect != "read",
                    is_steerable=False,
                    signature=method.signature,
                    function_id=registry.get_function_id(alias, method.name),
                    function_context="primitive",
                )
        return tools

    def get_prompt_context(self) -> str:
        from unify.actor import core_surface

        namespaces = environment_namespaces()
        # UNIFY_TOOL_SURFACE=core: the library search is the sandbox's.
        search = (
            "`await functions.search(...)`"
            if core_surface.enabled()
            else "`FunctionManager_search_functions`"
        )
        lines = [
            "### `primitives.*` — This Environment's Namespaces\n",
            "The environment you work in registered the namespaces below. "
            "Call their methods from Python as "
            "`primitives.<namespace>.<method>(...)`. A stored function that "
            "calls them needs no import: they are recorded and injected when it "
            "runs later. Each method's signature and "
            f"documentation come back from {search} "
            "(as primitive rows) and from `help(primitives.<namespace>.<method>)`. "
            "Effects: `read` methods change nothing, `write` methods create or "
            "change state, `destructive` methods delete or overwrite it.\n",
            "| Namespace | Methods (read / write / destructive) | What it is |",
            "|---|---|---|",
        ]
        for alias in sorted(self._aliases):
            namespace = namespaces.get(alias)
            if namespace is None:
                continue
            counts = namespace.effect_counts()
            lines.append(
                f"| `primitives.{alias}` | {len(namespace.methods)} "
                f"({counts['read']} / {counts['write']} / {counts['destructive']}) "
                f"| {namespace.description or '—'} |",
            )
        surface = environment_surface()
        if surface is not None and surface.globals:
            names = ", ".join(f"`{name}`" for name in sorted(surface.globals))
            lines.append(
                f"\nThe environment also binds these sandbox globals: {names}.",
            )
        modules = environment_modules()
        if modules:
            listed = ", ".join(f"`{name}`" for name in sorted(modules))
            lines.append(
                f"\nModules the environment supplies (import them freely; never "
                f"declare them as `dependencies`): {listed}.",
            )
        return "\n".join(lines)


def registered_environments() -> List[BaseEnvironment]:
    """The actor environments for the registered namespaces: none, or one."""
    if not environment_namespaces():
        return []
    return [EnvironmentNamespacesEnvironment()]


__all__ = ["EnvironmentNamespacesEnvironment", "registered_environments"]
