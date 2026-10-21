"""The actor environment for the namespaces an environment registered.

``UNIFY_ENV_NAMESPACES`` (``unify/function_manager/primitives/environment.py``)
registers an environment's callable surface as ``primitives.<name>``
namespaces. This environment binds them in the actor's cells as one
``Primitives`` object, and its prompt context is a short listing. Method
documentation stays out of the prompt: ``help(primitives.<name>.<method>)``
reads it in a cell.
"""

from __future__ import annotations

from typing import List

from unify.actor.environments.base import BaseEnvironment
from unify.function_manager.primitives.environment import (
    environment_modules,
    environment_namespaces,
    environment_surface,
)


class EnvironmentNamespacesEnvironment(BaseEnvironment):
    """``primitives.<name>`` for every namespace the environment registered."""

    NAMESPACE = "primitives"

    def __init__(self) -> None:
        from unify.function_manager.primitives import Primitives

        self._aliases = frozenset(environment_namespaces())
        if not self._aliases:
            raise ValueError("no environment namespace is registered")
        super().__init__(instance=Primitives(), namespace=self.NAMESPACE)

    def get_prompt_context(self) -> str:
        namespaces = environment_namespaces()
        lines = [
            "### `primitives.*` — This Environment's Namespaces\n",
            "The environment you work in registered the namespaces below. "
            "Call their methods from Python as "
            "`primitives.<namespace>.<method>(...)`. Each method's "
            "signature and "
            "documentation come back from "
            "`help(primitives.<namespace>.<method>)`. "
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
