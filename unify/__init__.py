"""
unify/__init__.py
==================

Package initialization for the unify assistant runtime.

The runtime must be explicitly initialized via init() before an actor runs:

    import unify
    unify.init()  # Validates providers, loads environment namespaces

For code that may run before or after init(), use ensure_initialised() which
is a no-op if already initialized.

Logging is configured centrally in unify.logger (imported below).
"""

# Logging is configured entirely in unify.logger — import it so that
# the module-level setup (handler, formatter, library muting) runs once.
import unify.logger  # noqa: F401

_INITIALISED = False


def init() -> None:  # noqa: D401 – imperative name
    """Initialise the runtime: validate providers, load environment namespaces."""

    global _INITIALISED
    if _INITIALISED:
        return

    from unify.settings import SETTINGS as _SETTINGS

    _SETTINGS.validate_llm_providers()

    if _SETTINGS.UNIFY_ENV_NAMESPACES.strip():
        # An environment that cannot register its namespaces stops start-up.
        from unify.function_manager.primitives.environment import (
            load_environment_namespaces,
        )

        load_environment_namespaces()

    _INITIALISED = True


# What the package exports at top-level
__all__ = ["init", "ensure_initialised"]
