"""Legacy, unused, unsupported: the conversation manager's read of a handle's
pause state, moved from ``unify/common/_async_tool/utils.py`` (8 Oct 2026),
where nothing the actor runs reaches it."""


def get_handle_paused_state(handle) -> bool | None:
    """Whether a steerable handle is paused, read from its ``_pause_event``
    (set = running, cleared = paused). A handle that tracks pause state
    another way exposes ``_pause_event`` as a proxy with ``is_set()``.
    Returns None when the handle has no such event.
    """
    try:
        pev = getattr(handle, "_pause_event", None)
        if pev is not None and hasattr(pev, "is_set"):
            return not pev.is_set()
    except Exception:
        pass
    return None
