"""
unify/manager_registry.py
=========================

Centralized manager infrastructure for Unify's state managers.

This module is the single source of truth for:
  - Manager implementation registration (real, simulated, etc.)
  - Settings-based IMPL resolution
  - Singleton instance caching
  - Typed factory methods for obtaining manager instances

Usage:
    # Get a manager via typed method (auto-resolves IMPL, returns singleton):
    function_manager = ManagerRegistry.get_function_manager()

    # For simulated managers, pass description:
    ManagerRegistry.get_guidance_manager(description="test scenario")

    # Clear all singletons (for test isolation):
    ManagerRegistry.clear()

Available typed methods:
    - get_actor()
    - get_conversation_manager_handle()
    - get_function_manager()
    - get_guidance_manager()
"""

from __future__ import annotations

from abc import ABCMeta
from threading import Lock
from typing import TYPE_CHECKING, Any, Callable, Dict, Optional, Type

if TYPE_CHECKING:
    from .actor.base import BaseActor
    from .conversation_manager.base import BaseConversationManagerHandle
    from .function_manager.base import BaseFunctionManager
    from .guidance_manager.base import BaseGuidanceManager
    from .function_manager.primitives.scope import PrimitiveScope

__all__ = [
    "ManagerRegistry",
    "SingletonABCMeta",
]


class ManagerRegistry:
    """Central registry for manager implementations and singleton instances.

    Handles:
    - Implementation class registration (real, simulated, etc.)
    - Settings-based IMPL resolution
    - Singleton instance caching
    - Factory method for obtaining correctly-configured manager instances
    """

    # (manager_key, impl_name) -> class
    _classes: Dict[tuple[str, str], Type] = {}

    # class -> singleton instance
    _instances: Dict[Type, Any] = {}

    # manager_key -> callable that returns the settings object for that manager
    _settings_map: Dict[str, Callable[[], Any]] = {}

    _lock: Lock = Lock()

    # Flag to track if the registry has been populated
    _populated: bool = False

    # ──────────────────────────────────────────────────────────────────────────
    # Registration API
    # ──────────────────────────────────────────────────────────────────────────

    @classmethod
    def register_class(cls, manager_key: str, impl_name: str, klass: Type) -> None:
        """Register a manager implementation class.

        Parameters
        ----------
        manager_key : str
            Logical name for the manager (e.g., "functions", "guidance").
        impl_name : str
            Implementation variant (e.g., "real", "simulated").
        klass : Type
            The concrete class to register.
        """
        cls._classes[(manager_key, impl_name)] = klass

    @classmethod
    def register_settings(
        cls,
        manager_key: str,
        settings_accessor: Callable[[], Any],
    ) -> None:
        """Register a settings accessor for a manager key.

        Parameters
        ----------
        manager_key : str
            Logical name for the manager.
        settings_accessor : Callable
            A callable that returns the settings object for this manager.
            Expected to have an `IMPL` attribute.
        """
        cls._settings_map[manager_key] = settings_accessor

    # ──────────────────────────────────────────────────────────────────────────
    # Lazy Population
    # ──────────────────────────────────────────────────────────────────────────

    @classmethod
    def _ensure_populated(cls) -> None:
        """Ensure the registry is populated with all known implementations.

        This is called lazily on first access to avoid circular imports.
        The managers import SingletonABCMeta from this module, so we can't
        import them at module load time.
        """
        if cls._populated:
            return
        with cls._lock:
            if cls._populated:
                return
            _populate_registry()
            cls._populated = True

    # ──────────────────────────────────────────────────────────────────────────
    # Factory API
    # ──────────────────────────────────────────────────────────────────────────

    @classmethod
    def get(
        cls,
        manager_key: str,
        *,
        description: str | None = None,
        simulation_guidance: str | None = None,
        _force_new: bool = False,
        **kwargs: Any,
    ) -> Any:
        """Get or create the singleton instance for a manager.

        Automatically resolves the implementation based on settings (IMPL).

        Parameters
        ----------
        manager_key : str
            Logical name for the manager (e.g., "functions", "guidance").
        description : str | None
            For simulated managers, the scenario description. Ignored for real
            managers.
        simulation_guidance : str | None
            For simulated managers, additional guidance for simulation behavior.
            Ignored for real managers.
        _force_new : bool
            If True, bypass the singleton cache and create a fresh instance.
            Primarily for testing.
        **kwargs
            Additional keyword arguments passed to the manager constructor.

        Returns
        -------
        Any
            The manager instance (singleton unless _force_new=True).

        Raises
        ------
        ValueError
            If the manager_key is unknown or no implementation is registered
            for the resolved IMPL.
        """
        cls._ensure_populated()

        # 1. Resolve IMPL from settings
        impl_name = cls._resolve_impl(manager_key)

        # 2. Get the class
        klass = cls.get_class(manager_key, impl_name)

        # 3. Check singleton cache (unless forced)
        if not _force_new:
            with cls._lock:
                existing = cls._instances.get(klass)
                if existing is not None:
                    return existing

        # 4. Build constructor kwargs
        ctor_kwargs = dict(kwargs)
        # Simulated-only parameters
        if impl_name == "simulated":
            if description is not None:
                ctor_kwargs["description"] = description
            if simulation_guidance is not None:
                ctor_kwargs["simulation_guidance"] = simulation_guidance

        # 5. Create instance. A singleton class's metaclass hands back its
        # registered instance from a plain call, so a forced new instance
        # is built past it: otherwise a caller scoping its "fresh" manager
        # (a sub-actor's guidance scope) would scope everyone's.
        if _force_new and isinstance(klass, SingletonABCMeta):
            instance = ABCMeta.__call__(klass, **ctor_kwargs)
        else:
            instance = klass(**ctor_kwargs)

        # 6. Cache (unless forced)
        if not _force_new:
            with cls._lock:
                # Double-check pattern for thread safety
                existing = cls._instances.get(klass)
                if existing is not None:
                    return existing
                cls._instances[klass] = instance

        return instance

    @classmethod
    def get_class(cls, manager_key: str, impl_name: str | None = None) -> Type:
        """Get the class for a manager without instantiating.

        Parameters
        ----------
        manager_key : str
            Logical name for the manager.
        impl_name : str | None
            Implementation variant. If None, resolves from settings.

        Returns
        -------
        Type
            The manager class.

        Raises
        ------
        ValueError
            If no implementation is registered for the given key/impl.
        """
        cls._ensure_populated()

        if impl_name is None:
            impl_name = cls._resolve_impl(manager_key)

        key = (manager_key, impl_name)
        if key not in cls._classes:
            available = [k[1] for k in cls._classes if k[0] == manager_key]
            raise ValueError(
                f"Unknown implementation '{impl_name}' for manager '{manager_key}'. "
                f"Available: {available}",
            )
        return cls._classes[key]

    @classmethod
    def _resolve_impl(cls, manager_key: str) -> str:
        """Resolve the IMPL setting for a manager key.

        Checks environment variables at runtime first to support test-time
        overrides. SETTINGS is frozen at import time, so test conftests that
        set os.environ after import won't affect SETTINGS values.

        The env var name is derived from the settings object's model_config
        env_prefix (e.g., UNIFY_GUIDANCE_ -> UNIFY_GUIDANCE_IMPL).
        """
        import os

        settings_accessor = cls._settings_map.get(manager_key)
        if settings_accessor is None:
            raise ValueError(
                f"No settings registered for manager '{manager_key}'. "
                f"Available: {list(cls._settings_map.keys())}",
            )
        settings = settings_accessor()

        # Derive env var name from the settings model_config env_prefix
        env_prefix = settings.model_config.get("env_prefix", "")
        if env_prefix:
            env_var = f"{env_prefix}IMPL"
            env_value = os.environ.get(env_var, "")
            if env_value:
                return env_value

        # Fall back to SETTINGS value (frozen at import time)
        return getattr(settings, "IMPL", "real")

    # ──────────────────────────────────────────────────────────────────────────
    # Singleton Management (for direct class instantiation & test isolation)
    # ──────────────────────────────────────────────────────────────────────────

    @classmethod
    def get_instance(cls, klass: Type) -> Any | None:
        """Get a cached singleton instance by class.

        Used by SingletonABCMeta for direct class instantiation.
        """
        with cls._lock:
            return cls._instances.get(klass)

    @classmethod
    def register_instance(cls, klass: Type, instance: Any) -> None:
        """Register an instance as the singleton for a class.

        Used by SingletonABCMeta for direct class instantiation.
        """
        with cls._lock:
            cls._instances[klass] = instance

    @classmethod
    def deregister_instance(cls, klass: Type) -> None:
        """Drop one cached singleton so the next construction builds fresh.

        A retired instance must never be handed to a successor: an
        in-process reboot that receives it back gets a session whose stop
        event is already set and whose registries were already discarded.
        """
        with cls._lock:
            cls._instances.pop(klass, None)

    @classmethod
    def clear(cls) -> None:
        """Remove all cached singleton instances.

        Call this between tests to ensure isolation.
        """
        with cls._lock:
            cls._instances.clear()

    # ──────────────────────────────────────────────────────────────────────────
    # Typed Factory Methods
    # ──────────────────────────────────────────────────────────────────────────

    @classmethod
    def get_actor(
        cls,
        *,
        description: str | None = None,
        simulation_guidance: str | None = None,
        _force_new: bool = False,
        **kwargs: Any,
    ) -> "BaseActor":
        """Get the Actor singleton (respects IMPL settings)."""
        return cls.get(
            "actor",
            description=description,
            simulation_guidance=simulation_guidance,
            _force_new=_force_new,
            **kwargs,
        )

    @classmethod
    def get_conversation_manager_handle(
        cls,
        *,
        description: str | None = None,
        simulation_guidance: str | None = None,
        _force_new: bool = False,
        **kwargs: Any,
    ) -> "BaseConversationManagerHandle":
        """Get the ConversationManagerHandle singleton (respects IMPL settings)."""
        return cls.get(
            "conversation",
            description=description,
            simulation_guidance=simulation_guidance,
            _force_new=_force_new,
            **kwargs,
        )

    @classmethod
    def get_function_manager(
        cls,
        *,
        primitive_scope: Optional[PrimitiveScope] = None,
        description: str | None = None,
        simulation_guidance: str | None = None,
        _force_new: bool = False,
        **kwargs: Any,
    ) -> "BaseFunctionManager":
        """Get the FunctionManager for a given primitive scope.

        Unlike other managers, FunctionManager is NOT a singleton because each
        scope requires its own instance with scoped primitive sync/search.

        Parameters
        ----------
        primitive_scope : PrimitiveScope | None
            Defines which managers' primitives are indexed and searchable.
            If None, defaults to role-gated runtime scope.
        """
        from unify.function_manager.primitives import default_runtime_scope

        if primitive_scope is None:
            primitive_scope = default_runtime_scope()

        # FunctionManager is always created fresh per scope (not singleton)
        return cls.get(
            "functions",
            description=description,
            simulation_guidance=simulation_guidance,
            _force_new=True,  # Always create new instance per scope
            primitive_scope=primitive_scope,
            **kwargs,
        )

    @classmethod
    def get_guidance_manager(
        cls,
        *,
        description: str | None = None,
        simulation_guidance: str | None = None,
        _force_new: bool = False,
        **kwargs: Any,
    ) -> "BaseGuidanceManager":
        """Get the GuidanceManager singleton (respects IMPL settings)."""
        return cls.get(
            "guidance",
            description=description,
            simulation_guidance=simulation_guidance,
            _force_new=_force_new,
            **kwargs,
        )


class SingletonABCMeta(ABCMeta):
    """Metaclass that enforces the Singleton pattern via ManagerRegistry.

    Any concrete subclass that uses this metaclass will only ever be
    instantiated once (until ManagerRegistry.clear() is called). Subsequent
    constructor calls return the existing instance without calling __init__
    again.

    Note: This supports direct class instantiation (e.g., `FunctionManager()`).
    For settings-aware instantiation that respects IMPL, use the typed methods
    like `ManagerRegistry.get_function_manager()` instead.
    """

    def __call__(cls, *args: Any, **kwargs: Any) -> Any:
        existing = ManagerRegistry.get_instance(cls)
        if existing is not None:
            return existing

        # First instantiation – create and register
        instance = super().__call__(*args, **kwargs)
        ManagerRegistry.register_instance(cls, instance)
        return instance


# ──────────────────────────────────────────────────────────────────────────────
# Registry Population
# ──────────────────────────────────────────────────────────────────────────────


def _populate_registry() -> None:
    """Populate the registry with all known implementations and settings.

    Imports are deferred to avoid circular dependencies.
    """
    # ─────────────────────────────────────────────────────────────────────────
    # Settings mappings (manager_key -> settings accessor)
    # ─────────────────────────────────────────────────────────────────────────
    from .settings import SETTINGS

    ManagerRegistry.register_settings("actor", lambda: SETTINGS.actor)
    ManagerRegistry.register_settings("conversation", lambda: SETTINGS.conversation)
    ManagerRegistry.register_settings("guidance", lambda: SETTINGS.guidance)
    ManagerRegistry.register_settings("functions", lambda: SETTINGS.function)

    # ─────────────────────────────────────────────────────────────────────────
    # Actor implementations
    # ─────────────────────────────────────────────────────────────────────────
    from .actor.code_act_actor import CodeActActor
    from .actor.simulated import SimulatedActor

    ManagerRegistry.register_class("actor", "code_act", CodeActActor)
    ManagerRegistry.register_class("actor", "simulated", SimulatedActor)

    # ─────────────────────────────────────────────────────────────────────────
    # ConversationManager implementations
    # ─────────────────────────────────────────────────────────────────────────
    from .conversation_manager.handle import ConversationManagerHandle
    from .conversation_manager.simulated import SimulatedConversationManagerHandle

    ManagerRegistry.register_class("conversation", "real", ConversationManagerHandle)
    ManagerRegistry.register_class(
        "conversation",
        "simulated",
        SimulatedConversationManagerHandle,
    )

    # ─────────────────────────────────────────────────────────────────────────
    # GuidanceManager implementations
    # ─────────────────────────────────────────────────────────────────────────
    from .guidance_manager.guidance_manager import GuidanceManager
    from .guidance_manager.simulated import SimulatedGuidanceManager

    ManagerRegistry.register_class("guidance", "real", GuidanceManager)
    ManagerRegistry.register_class("guidance", "simulated", SimulatedGuidanceManager)

    # ─────────────────────────────────────────────────────────────────────────
    # FunctionManager implementations
    # ─────────────────────────────────────────────────────────────────────────
    from .function_manager.function_manager import FunctionManager
    from .function_manager.simulated import SimulatedFunctionManager

    ManagerRegistry.register_class("functions", "real", FunctionManager)
    ManagerRegistry.register_class("functions", "simulated", SimulatedFunctionManager)
