"""``primitives`` in the actor's cells: one attribute per namespace the environment registered."""

from __future__ import annotations

from typing import Any


class Primitives:
    """The namespaces the environment registered (``UNIFY_ENV_NAMESPACES``), as ``primitives.<name>``.

    Each attribute resolves to the namespace's object, which holds exactly its declared methods
    (unify/function_manager/primitives/environment.py); any other name raises ``AttributeError``
    naming the namespaces that exist.
    """

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_"):
            raise AttributeError(name)
        from unify.function_manager.primitives.environment import (
            environment_aliases,
            namespace_object,
        )

        found = namespace_object(name)
        if found is None:
            known = ", ".join(sorted(environment_aliases())) or "none"
            raise AttributeError(
                f"primitives has no namespace {name!r}; the environment registered: {known}",
            )
        return found

    def __dir__(self) -> list[str]:
        from unify.function_manager.primitives.environment import environment_aliases

        return sorted(environment_aliases())

    def __repr__(self) -> str:
        return f"<primitives: {', '.join(dir(self)) or 'no namespace'}>"
