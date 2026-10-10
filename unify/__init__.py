"""
unify/__init__.py
==================

Package initialization for the unify assistant runtime.

The runtime must be explicitly initialized via init() before using managers:

    import unify
    unify.init()  # Validates providers, opens the store, installs hooks

For code that may run before or after init(), use ensure_initialised() which
is a no-op if already initialized.

Logging is configured centrally in unify.logger (imported below).
"""

try:
    import onnxruntime as _ort

    _ort.set_default_logger_severity(4)  # FATAL — suppress thread affinity noise
except Exception:
    pass

from unify import db

# Logging is configured entirely in unify.logger — import it so that
# the module-level setup (handler, formatter, library muting) runs once.
import unify.logger  # noqa: F401
from unify.common.startup_timing import startup_timing
from unify.logger import LOGGER

_INITIALISED = False


def init() -> None:  # noqa: D401 – imperative name
    """Initialise the runtime: validate providers, open the store, install hooks."""

    global _INITIALISED
    if _INITIALISED:
        return

    from unify.settings import SETTINGS as _SETTINGS

    with startup_timing(LOGGER, "unify.init.validate_llm_providers"):
        _SETTINGS.validate_llm_providers()

    with startup_timing(LOGGER, "unify.init.open_store", f"path={db.store_path()}"):
        db.connect()

    if _SETTINGS.UNIFY_ENV_NAMESPACES.strip():
        # An environment that cannot register its namespaces stops start-up.
        from unify.function_manager.primitives.environment import (
            load_environment_namespaces,
        )

        with startup_timing(LOGGER, "unify.init.environment_namespaces"):
            load_environment_namespaces()

    if str(getattr(_SETTINGS, "UNIFY_STORE_VERIFY", "") or "").strip():
        # A verifier that cannot be loaded (or one set without store
        # admission) stops start-up.
        from unify.function_manager.store_verify import verifier

        with startup_timing(LOGGER, "unify.init.store_verify"):
            verifier()

    from .events.llm_event_hook import install_llm_event_hook

    with startup_timing(LOGGER, "unify.init.install_llm_event_hook"):
        install_llm_event_hook()

    _INITIALISED = True


def ensure_initialised() -> None:
    """Run :func:`init` unless it has already run in this process."""
    if not _INITIALISED:
        init()


# What the package exports at top-level
__all__ = ["db", "init", "ensure_initialised"]
