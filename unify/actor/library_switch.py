"""``UNIFY_LIBRARY``: the actor with or without its library of stored functions and guidance.

Empty or ``on`` (the default) is today's actor. ``off`` removes memory and library surface only, and nothing else
(the tools, the sandbox, ``install``, ``read_file``, ``grep``, ``sh``, ``tools.check_examples``, the record and
teammates are as they are):

* the sandbox has no ``functions`` or ``guidance`` object, so the prompt has no library sections, no shortlist of
  stored entries and no line giving the library's size (:func:`sandbox_objects`, applied where the core surface
  binds its objects);
* the session may not store (:func:`can_store`): no ``store_skills``, no storage review at a turn or at the end,
  no library write;
* the system prompt no longer points to ``functions.search`` or says the actor has stored functions to draw on
  (:func:`system_prompt`).

Memory is a library too: a run that sets ``UNIFY_MEMORY=on`` with ``UNIFY_LIBRARY=off`` is refused before anything
runs (:func:`refusal`), so ``off`` always means no memory surface and no memory work.

The module reads the settings only when called (``unify.settings`` imports it to parse the value).
"""

from __future__ import annotations

from typing import Any

SETTING = "UNIFY_LIBRARY"
OFF = "off"
#: The sandbox objects the switch removes (the core surface's names for the two libraries).
LIBRARY_OBJECTS = ("functions", "guidance")


def parse(v: Any) -> str:
    """``UNIFY_LIBRARY``: ``""`` for empty or ``on`` (today's behaviour), ``off``; anything else is refused."""
    value = str(v if v is not None else "").strip().lower()
    if value in ("", "on"):
        return ""
    if value == OFF:
        return OFF
    raise ValueError(f"{SETTING} must be empty, 'on' or 'off', not {v!r}"[:200])


def off(settings: Any = None) -> bool:
    """Whether the run removes the library (``UNIFY_LIBRARY=off``) in *settings* (the run's when None)."""
    if settings is None:
        from unify.settings import SETTINGS as settings
    try:
        return parse(getattr(settings, SETTING, "") or "") == OFF
    except ValueError:
        return False


def refusal(settings: Any = None) -> str | None:
    """Why this run may not start under its ``UNIFY_LIBRARY`` (settings named, never a value), or None."""
    if settings is None:
        from unify.settings import SETTINGS as settings
    if not off(settings):
        return None
    if str(getattr(settings, "UNIFY_MEMORY", "") or "").strip().lower() == "on":
        return (
            f"{SETTING}=off removes memory and library surface, so UNIFY_MEMORY must be off; unset one of them, "
            "so nothing runs"
        )
    return None


def can_store(value: Any) -> Any:
    """No storage review and no library write with the library off; *value* otherwise."""
    return False if off() else value


def sandbox_objects(objects: dict) -> dict:
    """The core surface's sandbox objects without ``functions`` and ``guidance`` with the library off."""
    if not off():
        return objects
    return {k: v for k, v in objects.items() if k not in LIBRARY_OBJECTS}


def system_prompt(text: str) -> str:
    """*text* without the core prompt's pointer to ``functions.search`` and without stored functions in the role
    and the sandbox section, with the library off; *text* unchanged otherwise."""
    if not off():
        return text
    from unify.actor import prompt_builders as _prompts

    text = text.replace(_prompts._CORE_SANDBOX_SEARCH_PYTHON, _prompts._CORE_SANDBOX_HELP_ONLY)
    text = text.replace(_prompts._LEAN_ROLE_LIBRARY, _prompts._LEAN_ROLE_NO_LIBRARY)
    return text.replace(_prompts._SANDBOX_GLOBALS_INTRO, _prompts._SANDBOX_GLOBALS_INTRO_MEMORY)
