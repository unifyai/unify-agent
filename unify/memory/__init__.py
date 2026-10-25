"""Memory: what later requests learn from earlier ones (docs/memory.md describes it).

Episodes are git-tracked and append-only (``<UNIFY_HOME>/episodes.git``); the library is a git repository of tested
functions and linked notes (``<UNIFY_HOME>/memory``) that only gated writer passes extend; a SQLite evidence store
(``<UNIFY_HOME>/memory-evidence.sqlite``) indexes both. ``UNIFY_MEMORY`` switches it on (the default) or off; the
actor's side is wired in :mod:`.integration.hooks`.
"""
