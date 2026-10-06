"""``UNIFY_AGENTS_OPTIONS``: the record's tunables, parsed strictly."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Options:
    delivery: str = "mentions"
    max_live: int = 4
    max_total: int = 8
    deliver_max_tokens: int = 4000
    others_line: bool = True
    max_entries: int = 5000
    min_post_interval_s: float = 1.0


_INTS = ("max_live", "max_total", "deliver_max_tokens", "max_entries")
_RESERVED = ("interrupt", "cancel")
_TRUE = ("1", "true", "yes", "on")
_FALSE = ("0", "false", "no", "off")


def parse_options(text: str) -> Options:
    """Parse ``key=value,…``; any unknown, reserved or malformed item is an error."""
    values: dict = {}
    for item in (part.strip() for part in (text or "").split(",")):
        if not item:
            continue
        key, sep, raw = item.partition("=")
        key, raw = key.strip(), raw.strip().lower()
        if not sep:
            raise ValueError(f"UNIFY_AGENTS_OPTIONS item {item!r} is not key=value")
        if key in _RESERVED:
            raise ValueError(
                f"UNIFY_AGENTS_OPTIONS {key!r} is reserved for a later phase",
            )
        if key == "delivery":
            if raw not in ("mentions", "all"):
                raise ValueError(
                    "UNIFY_AGENTS_OPTIONS delivery must be mentions or all",
                )
            values[key] = raw
        elif key in _INTS:
            number = int(raw)
            if number < 0:
                raise ValueError(f"UNIFY_AGENTS_OPTIONS {key} must be ≥ 0")
            values[key] = number
        elif key == "others_line":
            if raw not in _TRUE + _FALSE:
                raise ValueError("UNIFY_AGENTS_OPTIONS others_line must be 0 or 1")
            values[key] = raw in _TRUE
        elif key == "min_post_interval_s":
            number = float(raw)
            if number < 0:
                raise ValueError(
                    "UNIFY_AGENTS_OPTIONS min_post_interval_s must be ≥ 0",
                )
            values[key] = number
        else:
            raise ValueError(f"unknown UNIFY_AGENTS_OPTIONS key {key!r}")
    return Options(**values)


def current_options() -> Options:
    from unify.settings import SETTINGS

    return parse_options(SETTINGS.UNIFY_AGENTS_OPTIONS)
