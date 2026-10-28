"""The environment's own callable surface, bound in the actor's cells as ``primitives.<name>``.

- :class:`EnvironmentSurface` / :func:`register_environment`: the namespaces an environment registers
  (``UNIFY_ENV_NAMESPACES``, unify/function_manager/primitives/environment.py);
- :class:`Primitives`: what ``primitives`` resolves to in a cell.
"""

from unify.function_manager.primitives.environment import (
    EnvironmentMethod,
    EnvironmentNamespace,
    EnvironmentSurface,
    register_environment,
)
from unify.function_manager.primitives.runtime import Primitives

__all__ = [
    "EnvironmentMethod",
    "EnvironmentNamespace",
    "EnvironmentSurface",
    "Primitives",
    "register_environment",
]
