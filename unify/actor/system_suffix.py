"""``UNIFY_SYSTEM_SUFFIX``: text appended at the very end of the actor's system prompt, pinned by its sha256.

``UNIFY_SYSTEM_SUFFIX`` holds the text itself and ``UNIFY_SYSTEM_SUFFIX_SHA256`` the lowercase hex sha256 of its
UTF-8 bytes, exactly as set (no newline added or removed). Both empty is today's prompt, byte for byte. Otherwise
they are set together and must match: a run where one is missing or the text does not match is refused before
anything runs (:func:`refusal`, read by the CLI at start-up), naming the settings, never the text. The text then
ends the system prompt of every ``act()`` (the actor's and a teammate's) after one blank line (:func:`append`), after
everything else the prompt carries (the memory section included), so the prefix before it is the same for every run
that differs only in it. The sha256 is on the CLI's ``ended`` line (:func:`record`).

The module reads the settings only when called (``unify.settings`` imports it to parse the sha256).
"""

from __future__ import annotations

import hashlib
import re
from typing import Any

SETTING = "UNIFY_SYSTEM_SUFFIX"
SHA_SETTING = "UNIFY_SYSTEM_SUFFIX_SHA256"
_SHA256 = re.compile(r"[0-9a-f]{64}")


class SystemSuffixRefused(RuntimeError):
    """The suffix and its sha256 are not set together, or do not match."""


def parse_sha256(v: Any) -> str:
    """``UNIFY_SYSTEM_SUFFIX_SHA256``: empty, or 64 lowercase hex characters."""
    value = str(v if v is not None else "").strip()
    if value and not _SHA256.fullmatch(value):
        raise ValueError(f"{SHA_SETTING} must be empty or 64 lowercase hex characters (a sha256)")
    return value


def digest(text: str) -> str:
    """The lowercase hex sha256 of *text*'s UTF-8 bytes."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _settings(settings: Any) -> Any:
    if settings is None:
        from unify.settings import SETTINGS as settings
    return settings


def refusal(settings: Any = None) -> str | None:
    """Why this run may not start under its suffix settings (named, never a value), or None."""
    settings = _settings(settings)
    text = str(getattr(settings, SETTING, "") or "")
    sha = str(getattr(settings, SHA_SETTING, "") or "")
    if not text and not sha:
        return None
    if not text or not sha:
        return f"{SETTING} and {SHA_SETTING} are set together or not at all; set both, so nothing runs"
    if digest(text) != sha:
        return f"{SETTING} does not match {SHA_SETTING} (sha256 of its UTF-8 bytes); nothing runs"
    return None


def suffix(settings: Any = None) -> str:
    """The text to append (empty when none); :class:`SystemSuffixRefused` when :func:`refusal` names a reason."""
    settings = _settings(settings)
    why = refusal(settings)
    if why is not None:
        raise SystemSuffixRefused(why)
    return str(getattr(settings, SETTING, "") or "")


def append(text: str, settings: Any = None) -> str:
    """*text* with the suffix after one blank line; *text* unchanged when there is none."""
    extra = suffix(settings)
    return f"{text}\n\n{extra}" if extra else text


def record(settings: Any = None) -> dict:
    """``{"system_suffix_sha256": ...}`` when a suffix is set, else ``{}`` (a default run's lines are unchanged)."""
    settings = _settings(settings)
    sha = str(getattr(settings, SHA_SETTING, "") or "")
    return {"system_suffix_sha256": sha} if sha and getattr(settings, SETTING, "") else {}
