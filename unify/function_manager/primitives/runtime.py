"""
Runtime primitives interface.

This module provides:
- `Primitives` - Scoped runtime interface for accessing primitive namespaces
- `get_primitive_callable` - Resolve stored primitive metadata to a callable

All namespace configuration (aliases, excluded methods, class paths) is
defined in `unify.function_manager.primitives.registry`. This module only
handles runtime instantiation.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Optional

from unify.function_manager.primitives.scope import (
    PrimitiveScope,
    VALID_MANAGER_ALIASES,
    default_runtime_scope,
)
from unify.function_manager.primitives.registry import (
    get_registry,
    _CLASS_PATH_TO_ALIAS,
    _MANAGER_BY_ALIAS,
)

logger = logging.getLogger(__name__)


class Primitives:
    """
    Scoped runtime interface to the primitive namespaces.

    Only aliases in the provided `primitive_scope` are accessible.
    Attempting to access an out-of-scope alias raises AttributeError.

    Each namespace object is constructed directly from the class path in
    its ``ManagerSpec`` and cached for the lifetime of this instance.

    Usage:
        primitives = Primitives()

        handle = await primitives.actor.act("Summarise the attached report")
    """

    def __init__(self, *, primitive_scope: Optional[PrimitiveScope] = None) -> None:
        """
        Initialize primitives with the given scope.

        Args:
            primitive_scope: Defines which aliases are accessible.
                           If None, uses the default runtime scope.
        """
        self._primitive_scope = primitive_scope or default_runtime_scope()
        # Lazy-initialized namespace instances
        self._managers: dict[str, Any] = {}

    @property
    def primitive_scope(self) -> PrimitiveScope:
        """The scope controlling which managers are accessible."""
        return self._primitive_scope

    def _get_manager(self, alias: str) -> Any:
        """
        Get or create a manager instance by alias.

        Raises AttributeError, saying why, if alias is not in scope.
        """
        spec = _MANAGER_BY_ALIAS.get(alias)
        if spec is None and alias not in VALID_MANAGER_ALIASES:
            # A namespace the environment registered has its spec there.
            spec = get_registry().get_manager_spec(alias)
        if spec is None:
            raise AttributeError(f"No ManagerSpec for alias: {alias}")

        if alias not in self._primitive_scope.scoped_managers:
            raise AttributeError(
                f"primitives.{alias} is not available to this actor: it was "
                f"started without {spec.domain} ({spec.description}). "
                "Complete the task with the tools this actor has.",
            )

        if alias in self._managers:
            return self._managers[alias]

        if alias not in VALID_MANAGER_ALIASES:
            # A namespace the environment registered: its methods, as declared.
            from unify.function_manager.primitives.environment import (
                namespace_object,
            )

            env_namespace = namespace_object(alias)
            if env_namespace is None:
                raise AttributeError(f"No namespace registered for alias: {alias}")
            self._managers[alias] = env_namespace
            return env_namespace

        cls = get_registry()._load_manager_class(spec.primitive_class_path)
        if cls is None:
            raise AttributeError(
                f"Could not load class for alias {alias!r}: "
                f"{spec.primitive_class_path}",
            )
        manager = cls()

        self._managers[alias] = manager
        return manager

    def __getattr__(self, name: str) -> Any:
        """Attribute access for manager retrieval."""
        if name in VALID_MANAGER_ALIASES:
            return self._get_manager(name)
        if not name.startswith("_"):
            from unify.function_manager.primitives.environment import (
                environment_aliases,
            )

            if name in environment_aliases():
                return self._get_manager(name)

        raise AttributeError(f"'Primitives' object has no attribute '{name}'")

    @property
    def actor(self) -> Any:
        """Actor delegation primitives (``act``)."""
        return self._get_manager("actor")


# =============================================================================
# Primitive Callable Resolution (for FunctionManager execution)
# =============================================================================


def get_primitive_callable(
    primitive_data: dict[str, Any],
    *,
    primitives: Optional[Primitives] = None,
) -> Optional[Callable]:
    """
    Resolve a primitive metadata dict to its actual callable.

    Uses the ``primitives`` instance (which handles all aliases uniformly)
    when available, falling back to a default-scoped Primitives instance.

    Args:
        primitive_data: Primitive metadata with primitive_class and primitive_method.
        primitives: Scoped Primitives instance for resolution.

    Returns:
        The callable method, or None if resolution fails.
    """
    class_path = primitive_data.get("primitive_class")
    method_name = primitive_data.get("primitive_method")

    if not class_path or not method_name:
        return None

    manager_alias = _CLASS_PATH_TO_ALIAS.get(class_path)
    if not manager_alias:
        from unify.function_manager.primitives.environment import (
            namespace_for_class_path,
        )

        env_namespace = namespace_for_class_path(class_path)
        if env_namespace is None:
            return None
        manager_alias = env_namespace.name

    # Use provided primitives instance, or construct a default-scoped one.
    if primitives is None:
        primitives = Primitives()

    manager = getattr(primitives, manager_alias, None)
    if manager is None:
        return None
    return getattr(manager, method_name, None)
