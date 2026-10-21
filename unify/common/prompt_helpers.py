from typing import Callable, Dict
from datetime import datetime

__all__ = [
    "now",
    "tool_name",
]


def _utc_now() -> datetime:
    from datetime import timezone as dt_timezone

    return datetime.now(dt_timezone.utc)


def now(time_only: bool = False, as_string: bool = True) -> "str | datetime":
    """Return the current time in UTC.

    Args:
        time_only: If True and as_string=True, return only the time portion.
        as_string: If True, return formatted string. If False, return datetime object.

    Returns:
        If as_string=True: "Thursday, January 15, 2026 at 02:09 PM UTC" (or time only)
        If as_string=False: datetime object

    In tests, this function is monkeypatched by tests/conftest.py to return
    fixed or incrementing datetimes for cache consistency.
    """
    utc_now = _utc_now()
    if not as_string:
        return utc_now
    if time_only:
        return utc_now.strftime("%I:%M %p UTC")
    return utc_now.strftime("%A, %B %d, %Y at %I:%M %p UTC")


def tool_name(tools: Dict[str, Callable], needle: str) -> str | None:
    """Best-effort lookup: find the first tool whose name contains ``needle``.

    Comparison is case-insensitive. Returns ``None`` if not found.
    """
    lowered = needle.lower()
    return next((name for name in tools if lowered in name.lower()), None)


