"""Replace one exact excerpt of a stored text (``UNIFY_FUNCTION_PATCH``).

A patch names the text to replace, ``old``, exactly as it appears. It is
applied only when ``old`` occurs exactly once; otherwise nothing changes and
the refusal says how many times it occurs, with a short excerpt of where (or,
for no match, of the closest line), so the caller can copy the right text and
try again. Occurrences are counted overlapping: ``"aa"`` occurs twice in
``"aaa"``, and such a patch would be ambiguous.
"""

from __future__ import annotations

import difflib
from typing import List

# Excerpts in a refusal: lines of context either side, occurrences shown, and
# the width a line is cut to.
_CONTEXT_LINES = 1
_MAX_SHOWN = 3
_LINE_WIDTH = 120


class PatchRefused(ValueError):
    """``old`` does not occur exactly once, or the patch changes nothing."""


def occurrences(text: str, old: str) -> List[int]:
    """Every offset at which ``old`` starts in ``text``, overlapping ones included."""
    found: List[int] = []
    at = text.find(old)
    while at != -1:
        found.append(at)
        at = text.find(old, at + 1)
    return found


def _excerpt(lines: List[str], line_no: int) -> str:
    lo = max(1, line_no - _CONTEXT_LINES)
    hi = min(len(lines), line_no + _CONTEXT_LINES)
    shown = []
    for number in range(lo, hi + 1):
        line = lines[number - 1]
        if len(line) > _LINE_WIDTH:
            line = line[:_LINE_WIDTH] + " …"
        shown.append(f"{number:>4} | {line}")
    return "\n".join(shown)


def _line_of(text: str, offset: int) -> int:
    return text.count("\n", 0, offset) + 1


def apply_once(text: str, old: str, new: str, *, what: str) -> str:
    """Return ``text`` with its one occurrence of ``old`` replaced by ``new``.

    ``what`` names the text in a refusal (e.g. ``"the source of 'scale'"``).

    Raises:
        PatchRefused: ``old`` is empty, equals ``new``, or does not occur
            exactly once; the message says which, with an excerpt.
    """
    if not isinstance(old, str) or not old:
        raise PatchRefused("`old` must be the exact, non-empty text to replace")
    if not isinstance(new, str):
        raise PatchRefused("`new` must be text (use an empty string to delete)")
    if old == new:
        raise PatchRefused(
            "`old` and `new` are the same, so there is nothing to change",
        )
    found = occurrences(text, old)
    if len(found) == 1:
        at = found[0]
        return text[:at] + new + text[at + len(old) :]
    lines = text.splitlines()
    if not found:
        message = (
            f"`old` occurs 0 times in {what}, so nothing was changed. Copy it "
            f"exactly from the current text; whitespace and indentation count."
        )
        if " ".join(old.split()) in " ".join(text.split()):
            message += (
                " It does match when whitespace is ignored, so the spacing, "
                "indentation or line breaks differ."
            )
        first = next((line.strip() for line in old.splitlines() if line.strip()), "")
        stripped = [line.strip() for line in lines]
        close = (
            difflib.get_close_matches(first, stripped, n=1, cutoff=0.5) if first else []
        )
        if close and lines:
            line_no = stripped.index(close[0]) + 1
            message += f" The closest line is {line_no}:\n{_excerpt(lines, line_no)}"
        elif lines:
            message += f" It begins:\n{_excerpt(lines, 1 + _CONTEXT_LINES)}"
        raise PatchRefused(message)
    line_nos = [_line_of(text, at) for at in found]
    listed = ", ".join(str(n) for n in line_nos[:_MAX_SHOWN])
    if len(line_nos) > _MAX_SHOWN:
        listed += ", …"
    shown = "\n----\n".join(
        _excerpt(lines, n) for n in sorted(set(line_nos))[:_MAX_SHOWN]
    )
    raise PatchRefused(
        f"`old` occurs {len(found)} times in {what} (at lines {listed}), so "
        f"nothing was changed. Include enough surrounding text to pick out "
        f"one occurrence:\n{shown}",
    )


__all__ = ["PatchRefused", "apply_once", "occurrences"]
