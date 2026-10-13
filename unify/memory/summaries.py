"""The pointers in an episode's ``summary.md``, checked against the recorded episode: a measure only.

The episode-end fork writes ``summary.md`` in the format :data:`.library_prompts.SUMMARY_FORMAT` states: each line
under Approach, Worked, Failed and Outcome ends with a pointer, a part name in backticks and a short quote copied from
that part, as in ``(`cell:3` "ValueError: invalid literal for int()")``. A pointer *resolves* when the part exists in
the recorded episode (:func:`.writer_tools.part_names`) and the part's text holds the quote, with runs of whitespace
compared as one space. A part is read as the writer reads it (:func:`.batch_map.part_text`); a cell's or an action's
fields are also compared one by one, so a quote of text that its JSON form escapes still resolves.

Nothing reads these counts back: no gate, prompt or admission rule depends on them. They go into the pass's end event
(``staging_episodes``), so the evidence shows per episode how many of a summary's pointers do not resolve.
"""

from __future__ import annotations

import json
import re
from typing import Any

#: One pointer: a part name in backticks, then a quote in straight or curly double quotes, then the closing bracket.
POINTER = re.compile(r"`(?P<part>[^`\s]+)`\s*[\"“](?P<quote>.*?)[\"”]\s*\)")
#: Why a pointer does not resolve.
NO_SUCH_PART = "no such part"
NOT_IN_PART = "quote not in the part"
EMPTY_QUOTE = "empty quote"
#: The characters of a pointer's quote an unresolved row keeps.
QUOTE_CHARS = 120


def pointers(text: str) -> list[tuple[str, str]]:
    """Every ``(part, quote)`` pointer in a summary's *text*, in order."""
    return [(m.group("part"), m.group("quote")) for m in POINTER.finditer(text or "")]


def _squash(text: str) -> str:
    return " ".join(text.split())


def _strings(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        return [s for v in value.values() for s in _strings(v)]
    if isinstance(value, list):
        return [s for v in value for s in _strings(v)]
    return [] if value is None else [str(value)]


def _texts(ep: Any, part: str) -> list[str] | None:
    """The texts a pointer into *part* may quote, or None when *ep* has no such part."""
    from .batch_map import part_text
    from .writer_tools import part_names

    if part not in part_names(ep):
        return None
    try:
        text = part_text(ep, part)
    except (KeyError, ValueError, IndexError, StopIteration, TypeError):
        return None
    out = [text]
    try:
        out += _strings(json.loads(text))
    except ValueError:
        pass
    return out


def check(text: str, ep: Any) -> dict:
    """The pointer counts of one summary *text* against its recorded episode *ep*: ``pointers``, ``resolved``,
    ``unresolved`` (their number) and ``unresolved_pointers`` (each ``{part, quote, why}``, the quote cut at
    :data:`QUOTE_CHARS` characters)."""
    found = pointers(text)
    bad = []
    for part, quote in found:
        q = _squash(quote)
        if not q:
            bad.append((part, quote, EMPTY_QUOTE))
            continue
        texts = _texts(ep, part)
        if texts is None:
            bad.append((part, quote, NO_SUCH_PART))
        elif not any(q in _squash(t) for t in texts):
            bad.append((part, quote, NOT_IN_PART))
    return {
        "pointers": len(found),
        "resolved": len(found) - len(bad),
        "unresolved": len(bad),
        "unresolved_pointers": [
            {"part": p[:80], "quote": q[:QUOTE_CHARS], "why": why} for p, q, why in bad
        ],
    }


def _count(value: Any) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _reader_record(view: Any) -> dict:
    """The reader's own record of its spend, wall time and tokens (``fork.json``), as the end event carries them."""
    if view is None:
        return {"reader_usd": None, "reader_seconds": None, "reader_tokens": None}
    from decimal import Decimal, InvalidOperation

    rec = getattr(view, "record", None) or {}
    try:
        usd = rec.get("usd")
        usd = format(Decimal(str(usd)), "f") if usd is not None and not isinstance(usd, bool) else "unknown"
    except (InvalidOperation, ValueError):
        usd = "unknown"
    seconds = rec.get("seconds")
    usage = rec.get("usage") if isinstance(rec.get("usage"), dict) else {}
    return {
        "reader_usd": usd,
        "reader_seconds": seconds if isinstance(seconds, (int, float)) and not isinstance(seconds, bool) else None,
        "reader_tokens": {
            "prompt": _count(usage.get("prompt_tokens")),
            "cached": _count(usage.get("cached_tokens")),
            "completion": _count(usage.get("completion_tokens")),
            "reasoning": _count(usage.get("reasoning_tokens")),
        },
    }


def episode_row(eid: str, entry: dict | None, view: Any, load: Any) -> dict:
    """One ``staging_episodes`` row of a pass's end event, for a batch episode: its staging as the writer got it
    (*entry*, the batch-map row's ``staging``); which reader staged it (``reader``: the view's source, ``none``
    without staging), its recorded status, spend, wall time and tokens, and what an unfinished reader left that was
    not handed on (*view*, :class:`.staging.StagingView`; the writer's row never names the reader); and its
    summary's size and pointer counts against the recorded episode (``load(eid)``). The pointer counts are None when
    there is no summary; ``summary_error`` says why the episode could not be read for them. ``reader_usd`` is a
    Decimal string, ``unknown`` when the reader recorded none, None without a reader. Never raises."""
    entry = entry if isinstance(entry, dict) else {}
    files = getattr(view, "files", None) or {}
    summary = files.get("summary.md")
    row: dict = {
        "episode": eid,
        "reader": getattr(view, "source", None) or "none",
        "status": entry.get("status", "none"),
        "reader_status": getattr(view, "status", None),
        **_reader_record(view),
        "files": len(entry.get("files") or []),
        "candidates": len(entry.get("candidates") or []),
        "withheld": len(getattr(view, "withheld", None) or []),
        "summary_bytes": len(summary) if summary is not None else None,
        "summary_pointers": None,
        "summary_pointers_resolved": None,
        "summary_pointers_unresolved": None,
        "unresolved_pointers": [],
    }
    if summary is None:
        return row
    text = summary.decode("utf-8", "replace")
    try:
        counts = check(text, load(eid))
    except Exception as exc:  # noqa: BLE001 - a measure never stops a pass
        row["summary_pointers"] = len(pointers(text))
        row["summary_error"] = f"{type(exc).__name__}: {str(exc)[:200]}"
        return row
    row.update(
        summary_pointers=counts["pointers"],
        summary_pointers_resolved=counts["resolved"],
        summary_pointers_unresolved=counts["unresolved"],
        unresolved_pointers=counts["unresolved_pointers"],
    )
    return row
