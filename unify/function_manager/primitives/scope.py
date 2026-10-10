"""PrimitiveScope: the single knob for controlling which primitive namespaces are exposed."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import FrozenSet

# Canonical namespace aliases - the only valid values for scoped_managers.
# This is the authoritative list; ToolSurfaceRegistry.MANAGERS must match.
# Namespaces an environment registers at start-up are valid as well
# (``valid_manager_aliases()``).
VALID_MANAGER_ALIASES: frozenset[str] = frozenset(
    {
        "actor",
    },
)


def valid_manager_aliases() -> frozenset[str]:
    """Unify's own aliases plus the namespaces the environment registered."""
    from unify.function_manager.primitives.environment import environment_aliases

    registered = environment_aliases()
    return VALID_MANAGER_ALIASES | registered if registered else VALID_MANAGER_ALIASES


@dataclass(frozen=True, slots=True)
class PrimitiveScope:
    """
    Defines which primitive namespaces a runtime exposes.

    This is the single source of truth for scoping. All downstream consumers
    (Primitives, ActorEnvironment, FunctionManager) read from this object.

    Attributes
    ----------
    scoped_managers : frozenset[str]
        Set of namespace aliases to expose, each a valid alias from
        VALID_MANAGER_ALIASES. Empty for a runtime granted no primitives,
        such as an actor that may not spawn sub-actors.

    Examples
    --------
    scope = PrimitiveScope(scoped_managers=frozenset({"actor"}))
    """

    scoped_managers: "FrozenSet[str]"

    def __post_init__(self) -> None:
        """Validate scoped_managers."""
        # Only a scope naming more than Unify's own aliases consults the
        # environment's registry (never at import, for the default scope).
        valid = (
            VALID_MANAGER_ALIASES
            if self.scoped_managers <= VALID_MANAGER_ALIASES
            else valid_manager_aliases()
        )
        invalid = self.scoped_managers - valid
        if invalid:
            raise ValueError(
                f"Invalid manager aliases: {sorted(invalid)}. "
                f"Valid aliases: {sorted(valid)}",
            )

    @property
    def scope_key(self) -> str:
        """
        Stable, deterministic key for caching and registry lookups.

        Returns a sorted comma-separated string of manager aliases.
        """
        return ",".join(sorted(self.scoped_managers))

    def includes(self, manager_alias: str) -> bool:
        """Check if a namespace alias is in scope."""
        return manager_alias in self.scoped_managers

    @classmethod
    def all_managers(cls) -> "PrimitiveScope":
        """Create a scope with every namespace exposed."""
        return cls(scoped_managers=valid_manager_aliases())

    @classmethod
    def single(cls, manager_alias: str) -> "PrimitiveScope":
        """Create a scope with a single namespace exposed."""
        return cls(scoped_managers=frozenset({manager_alias}))

    @classmethod
    def none(cls) -> "PrimitiveScope":
        """Create a scope that exposes no namespace."""
        return cls(scoped_managers=frozenset())


_DEFAULT_RUNTIME_SCOPE = PrimitiveScope(scoped_managers=VALID_MANAGER_ALIASES)


def default_runtime_scope() -> PrimitiveScope:
    """Return the default primitive scope for runtime usage.

    Every namespace the environment registered is in it, as ``actor`` is.
    """
    valid = valid_manager_aliases()
    if valid == VALID_MANAGER_ALIASES:
        return _DEFAULT_RUNTIME_SCOPE
    return PrimitiveScope(scoped_managers=valid)
