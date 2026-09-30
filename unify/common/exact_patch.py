"""Replace excerpts of a stored text (``UNIFY_FUNCTION_PATCH``).

:func:`apply_edits` applies a batch of ``{old, new}`` edits in order to the
evolving text, all or nothing. Each ``old`` is looked for on a ladder of
matching levels, strictest first, and the first level that finds it decides:

``exact``
    Byte for byte.
``trailing_whitespace``
    Line endings (CRLF, CR) read as LF and spaces or tabs at the end of a
    line ignored.
``indentation``
    Whole lines compared after removing the common leading whitespace of
    ``old`` and of the candidate block (tabs expanded to ``TAB_SIZE``
    columns); ``new`` is re-indented to the block it replaces.
``collapsed_whitespace``
    As ``trailing_whitespace``, with every run of spaces or tabs read as one
    space.

No level joins or splits lines, so a fuzzy match always spans as many lines
as ``old``. A level that finds ``old`` more than once refuses the edit
(unless the edit sets ``replace_all``) rather than trying a looser level, so
an ambiguous match is never applied. Occurrences are counted overlapping:
``"aa"`` occurs twice in ``"aaa"``, and such an edit is ambiguous. A refusal
says how to retry: the closest block with line numbers and a character diff
when ``old`` is not found, or the line of every occurrence when it is
ambiguous.
"""

from __future__ import annotations

import ast
import difflib
import json
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

from pydantic import BaseModel, Field

# Excerpts in a refusal: lines of context either side, occurrences shown, and
# the width a line is cut to.
_CONTEXT_LINES = 1
_MAX_SHOWN = 3
_LINE_WIDTH = 120
# Occurrence line numbers listed in an ambiguity refusal, and diff lines
# shown against the closest block.
_MAX_LISTED = 20
_MAX_DIFF_LINES = 40
# Below this similarity the closest block is not worth showing.
_CLOSEST_CUTOFF = 0.4

TAB_SIZE = 4

LEVELS = ("exact", "trailing_whitespace", "indentation", "collapsed_whitespace")
_LEVEL_WHEN = {
    "trailing_whitespace": "line endings and trailing spaces are ignored",
    "indentation": "indentation is compared relative to the block",
    "collapsed_whitespace": "runs of spaces and tabs count as one",
}

_EOL = re.compile(r"\r\n|\r|\n")
_EDIT_KEYS = {"old", "new", "replace_all", "old_string", "new_string"}


class PatchRefused(ValueError):
    """``old`` does not occur exactly once, or the patch changes nothing."""


class PatchEdit(BaseModel):
    """One replacement in a batch of edits."""

    old: str = Field(
        description=(
            "The text to replace, copied from the current text with enough "
            "surrounding text to occur once."
        ),
    )
    new: str = Field(
        description="The replacement text (an empty string deletes `old`).",
    )
    replace_all: bool = Field(
        default=False,
        description="Replace every occurrence of `old` instead of exactly one.",
    )


def occurrences(text: str, old: str) -> List[int]:
    """Every offset at which ``old`` starts in ``text``, overlapping ones included."""
    found: List[int] = []
    at = text.find(old)
    while at != -1:
        found.append(at)
        at = text.find(old, at + 1)
    return found


def _cut(line: str) -> str:
    return line[:_LINE_WIDTH] + " …" if len(line) > _LINE_WIDTH else line


def _excerpt(lines: List[str], line_no: int, last: Optional[int] = None) -> str:
    lo = max(1, line_no - _CONTEXT_LINES)
    hi = min(len(lines), (last or line_no) + _CONTEXT_LINES)
    return "\n".join(
        f"{number:>4} | {_cut(lines[number - 1])}" for number in range(lo, hi + 1)
    )


def _line_of(text: str, offset: int) -> int:
    return len(_EOL.findall(text, 0, offset)) + 1


def _lines(text: str) -> List[str]:
    """The lines of ``text``, split at LF, CRLF and CR only."""
    lines = _EOL.split(text)
    if len(lines) > 1 and lines[-1] == "":
        lines.pop()
    return lines if text else []


# --------------------------------------------------------------------------- #
#  Arguments                                                                   #
# --------------------------------------------------------------------------- #


def _pick(first: Any, alias: Any, name: str, alias_name: str, where: str) -> Any:
    if first is not None and alias is not None and first != alias:
        raise PatchRefused(
            f"{where}`{name}` and `{alias_name}` differ; `{alias_name}` is "
            f"another name for `{name}`, so give only one",
        )
    return first if first is not None else alias


def _as_bool(value: Any, where: str) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    if isinstance(value, str) and value.strip().lower() in ("true", "false"):
        return value.strip().lower() == "true"
    raise PatchRefused(f"{where}`replace_all` must be true or false")


def _checked(old: Any, new: Any, replace_all: Any, where: str) -> Dict[str, Any]:
    if not isinstance(old, str) or not old:
        raise PatchRefused(f"{where}`old` must be the non-empty text to replace")
    if new is None:
        raise PatchRefused(
            f"{where}`new` is missing (use an empty string to delete `old`)",
        )
    if not isinstance(new, str):
        raise PatchRefused(f"{where}`new` must be text (an empty string deletes)")
    if old == new:
        raise PatchRefused(
            f"{where}`old` and `new` are the same, so there is nothing to change",
        )
    return {"old": old, "new": new, "replace_all": _as_bool(replace_all, where)}


def collect_edits(
    *,
    old: Optional[str] = None,
    new: Optional[str] = None,
    edits: Any = None,
    replace_all: Any = False,
    old_string: Optional[str] = None,
    new_string: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """The edits a patch call asks for, as ``[{"old", "new", "replace_all"}]``.

    A call gives either ``old``/``new`` (``old_string``/``new_string`` are
    accepted as the same arguments) with an optional ``replace_all``, or
    ``edits``: a list of ``{old, new, replace_all?}`` objects, a single such
    object, or the list as a JSON string.

    Raises:
        PatchRefused: the arguments are missing, mixed or malformed.
    """
    old = _pick(old, old_string, "old", "old_string", "")
    new = _pick(new, new_string, "new", "new_string", "")
    if edits is None or (isinstance(edits, (list, tuple, str)) and not edits):
        if old is None:
            raise PatchRefused(
                "give the text to change as `old` and `new`, or several "
                "changes as `edits`: [{old, new}, ...]",
            )
        return [_checked(old, new, replace_all, "")]
    # Schema-filling models send empty `old`/`new` beside `edits`.
    if old or new:
        raise PatchRefused(
            "give either `old` and `new` or `edits`, not both; put every "
            "change in `edits`",
        )
    if _as_bool(replace_all, ""):
        raise PatchRefused(
            "top-level `replace_all` applies to `old`/`new`; with `edits`, "
            "set `replace_all` inside the edit it applies to",
        )
    if isinstance(edits, str):
        try:
            edits = json.loads(edits)
        except json.JSONDecodeError:
            raise PatchRefused(
                "`edits` must be a list of {old, new} objects",
            ) from None
    if isinstance(edits, (dict, BaseModel)):
        edits = [edits]
    if not isinstance(edits, (list, tuple)) or not edits:
        raise PatchRefused("`edits` must be a non-empty list of {old, new} objects")
    collected = []
    for number, edit in enumerate(edits, start=1):
        where = f"edit {number} of {len(edits)}: " if len(edits) > 1 else ""
        if isinstance(edit, BaseModel):
            edit = edit.model_dump()
        if not isinstance(edit, dict):
            raise PatchRefused(f"{where}each edit must be an object {{old, new}}")
        unknown = sorted(set(edit) - _EDIT_KEYS)
        if unknown:
            raise PatchRefused(
                f"{where}unknown keys {', '.join(map(repr, unknown))}; an edit "
                f"takes `old`, `new` and optionally `replace_all`",
            )
        collected.append(
            _checked(
                _pick(
                    edit.get("old"),
                    edit.get("old_string"),
                    "old",
                    "old_string",
                    where,
                ),
                _pick(
                    edit.get("new"),
                    edit.get("new_string"),
                    "new",
                    "new_string",
                    where,
                ),
                edit.get("replace_all", False),
                where,
            ),
        )
    return collected


# --------------------------------------------------------------------------- #
#  The matching ladder                                                         #
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class _Match:
    start: int
    end: int
    new: str


def _normal_view(text: str, collapse: bool) -> Tuple[str, List[int], List[int]]:
    """``text`` with LF endings and no trailing blanks (and runs collapsed).

    Returns the view and, for each of its characters, where the text it
    stands for starts and ends in ``text``: a line ending stands for all of
    its CRLF, and a collapsed space for its whole run.
    """
    chars: List[str] = []
    starts: List[int] = []
    ends: List[int] = []
    pos, size = 0, len(text)
    while True:
        eol = _EOL.search(text, pos)
        content_end = eol.start() if eol else size
        limit = pos + len(text[pos:content_end].rstrip(" \t"))
        at = pos
        while at < limit:
            run = at + 1
            if collapse and text[at] in " \t":
                while run < limit and text[run] in " \t":
                    run += 1
                chars.append(" ")
            else:
                chars.append(text[at])
            starts.append(at)
            ends.append(run)
            at = run
        if eol is None:
            return "".join(chars), starts, ends
        chars.append("\n")
        starts.append(eol.start())
        ends.append(eol.end())
        pos = eol.end()


def _fit_newlines(span: str, new: str) -> str:
    """``new`` with the span's CRLF or CR line endings, when it uses LF."""
    if "\r" in new:
        return new
    if "\r\n" in span:
        return new.replace("\n", "\r\n")
    if "\r" in span and "\n" not in span:
        return new.replace("\n", "\r")
    return new


def _find_in_view(text: str, old: str, new: str, collapse: bool) -> List[_Match]:
    view, starts, ends = _normal_view(text, collapse)
    needle = _normal_view(old, collapse)[0]
    if not needle.strip():
        return []
    found = []
    for at in occurrences(view, needle):
        start, end = starts[at], ends[at + len(needle) - 1]
        found.append(_Match(start, end, _fit_newlines(text[start:end], new)))
    return found


def _lead(line: str) -> str:
    return line[: len(line) - len(line.lstrip(" \t"))]


def _width(lead: str) -> int:
    return len(lead.expandtabs(TAB_SIZE))


def _dedented(lines: Sequence[str]) -> Tuple[List[str], int]:
    """Lines without trailing blanks, less their common indentation, and its width."""
    rows = [line.rstrip(" \t") for line in lines]
    widths = [_width(_lead(row)) for row in rows if row]
    base = min(widths) if widths else 0
    out = []
    for row in rows:
        lead = _lead(row)
        out.append(" " * (_width(lead) - base) + row[len(lead) :] if row else "")
    return out, base


def _reindent(new: str, old_base: int, block: Sequence[str]) -> str:
    """``new`` moved from ``old``'s indentation to that of ``block``."""
    indented = [row for row in block if row.strip()]
    base_lead = min((_lead(row) for row in indented), key=_width, default="")
    base = _width(base_lead)
    tabs = any("\t" in _lead(row) for row in indented)

    def render(columns: int) -> str:
        if tabs:
            return "\t" * (columns // TAB_SIZE) + " " * (columns % TAB_SIZE)
        return " " * columns

    out = []
    for line in new.split("\n"):
        if not line.strip():
            out.append("")
            continue
        lead = _lead(line)
        shift = _width(lead) - old_base
        if shift >= 0:
            indent = base_lead + render(shift)
        else:
            indent = render(max(0, base + shift))
        out.append(indent + line[len(lead) :])
    return "\n".join(out)


def _find_by_indentation(text: str, old: str, new: str) -> List[_Match]:
    body = _EOL.sub("\n", old)
    whole_lines = body.endswith("\n")
    if whole_lines:
        body = body[:-1]
    wanted, old_base = _dedented(body.split("\n"))
    if not any(wanted):
        return []
    new = _EOL.sub("\n", new)
    # Where every line starts, its content, and the length of its ending.
    starts, contents, endings = [], [], []
    pos = 0
    for eol in _EOL.finditer(text):
        starts.append(pos)
        contents.append(text[pos : eol.start()])
        endings.append(eol.end() - eol.start())
        pos = eol.end()
    if pos < len(text):
        starts.append(pos)
        contents.append(text[pos:])
        endings.append(0)
    size = len(wanted)
    found = []
    for first in range(len(contents) - size + 1):
        block = contents[first : first + size]
        if _dedented(block)[0] != wanted:
            continue
        last = first + size - 1
        start = starts[first]
        end = starts[last] + len(contents[last])
        if whole_lines:
            end += endings[last]
        replacement = _reindent(new, old_base, block)
        found.append(_Match(start, end, _fit_newlines(text[start:end], replacement)))
    return found


def _find(text: str, old: str, new: str, level: str) -> List[_Match]:
    if level == "exact":
        return [_Match(at, at + len(old), new) for at in occurrences(text, old)]
    if level == "indentation":
        return _find_by_indentation(text, old, new)
    return _find_in_view(text, old, new, collapse=level == "collapsed_whitespace")


def _disjoint(found: List[_Match]) -> List[_Match]:
    kept: List[_Match] = []
    for match in sorted(found, key=lambda m: m.start):
        if not kept or match.start >= kept[-1].end:
            kept.append(match)
    return kept


# --------------------------------------------------------------------------- #
#  Refusals                                                                    #
# --------------------------------------------------------------------------- #


def _old_lines(old: str) -> List[str]:
    return _EOL.sub("\n", old).rstrip("\n").split("\n")


def _closest_block(lines: List[str], old: str) -> Optional[Tuple[int, int, float]]:
    """The block of ``old``'s size most like it: first and last line, similarity.

    Blocks are compared as the ladder tolerates them, less trailing blanks
    and common indentation, so the similarity reflects what still differs.
    """
    if not lines:
        return None
    wanted = _dedented(_old_lines(old))[0]
    size = min(len(wanted), len(lines))
    matcher = difflib.SequenceMatcher(autojunk=False)
    matcher.set_seq2("\n".join(wanted))
    best: Optional[Tuple[int, float]] = None
    for first in range(len(lines) - size + 1):
        matcher.set_seq1("\n".join(_dedented(lines[first : first + size])[0]))
        if best is not None and (
            matcher.real_quick_ratio() <= best[1] or matcher.quick_ratio() <= best[1]
        ):
            continue
        ratio = matcher.ratio()
        if best is None or ratio > best[1]:
            best = (first + 1, ratio)
    assert best is not None
    return best[0], best[0] + size - 1, best[1]


def _not_found(text: str, old: str, what: str, prefix: str) -> str:
    lines = _lines(text)
    message = (
        f"{prefix}`old` occurs 0 times in {what}, even with whitespace "
        f"differences ignored, so nothing was changed."
    )
    if " ".join(old.split()) in " ".join(text.split()):
        message += (
            " It does match when line breaks are ignored as well, so copy the "
            "lines as they are broken in the current text."
        )
    closest = _closest_block(lines, old)
    if closest is None:
        return f"{message} The text is empty."
    first, last, ratio = closest
    if ratio < _CLOSEST_CUTOFF:
        return f"{message} Nothing is close; it begins:\n{_excerpt(lines, 1, 3)}"
    span = f"line {first}" if first == last else f"lines {first}-{last}"
    diff = list(
        difflib.ndiff(
            [_cut(row) for row in _dedented(_old_lines(old))[0]],
            [_cut(row) for row in _dedented(lines[first - 1 : last])[0]],
        ),
    )
    if len(diff) > _MAX_DIFF_LINES:
        diff = diff[:_MAX_DIFF_LINES] + ["  …"]
    return (
        f"{message} The closest text is {span} (similarity {ratio:.2f}; line "
        f"numbers are not part of the text):\n{_excerpt(lines, first, last)}\n"
        f"How `old` (-) differs from it (+), indentation aside, `?` marking "
        f"the characters:\n" + "\n".join(diff)
    )


def _span_label(span: Tuple[int, int], count: int) -> str:
    first, last = span
    label = str(first) if first == last else f"{first}-{last}"
    return label if count == 1 else f"{label} ({count} times)"


def _ambiguous(
    text: str,
    found: List[_Match],
    what: str,
    level: str,
    prefix: str,
) -> str:
    lines = _lines(text)
    spans = []
    for match in found:
        first = _line_of(text, match.start)
        last = _line_of(text, max(match.start, match.end - 1))
        spans.append((first, last))
    listed = ", ".join(
        _span_label(span, spans.count(span))
        for span in dict.fromkeys(spans[:_MAX_LISTED])
    )
    if len(spans) > _MAX_LISTED:
        listed += ", …"
    when = "" if level == "exact" else f" when {_LEVEL_WHEN[level]}"
    shown = "\n----\n".join(
        _excerpt(lines, a, b) for a, b in sorted(set(spans))[:_MAX_SHOWN]
    )
    return (
        f"{prefix}`old` occurs {len(found)} times in {what}{when} (at lines "
        f"{listed}), so nothing was changed. Add a neighbouring line to `old` "
        f"so it picks out one occurrence, or set `replace_all` to change every "
        f"one:\n{shown}"
    )


# --------------------------------------------------------------------------- #
#  Applying a batch                                                            #
# --------------------------------------------------------------------------- #


def _apply(text: str, matches: List[_Match]) -> str:
    for match in sorted(matches, key=lambda m: m.start, reverse=True):
        text = text[: match.start] + match.new + text[match.end :]
    return text


def _prefix(number: int, total: int) -> str:
    if total == 1:
        return ""
    if number == 1:
        return f"Edit 1 of {total}: "
    before = "edit 1" if number == 2 else f"edits 1-{number - 1}"
    return (
        f"Edit {number} of {total} (matched in the text as {before} left it; "
        f"no edit was kept): "
    )


def apply_edits(
    text: str,
    edits: List[Dict[str, Any]],
    *,
    what: str,
) -> Tuple[str, List[Dict[str, Any]]]:
    """Apply ``edits`` (from :func:`collect_edits`) in order, all or nothing.

    Each edit is matched on the ladder in the text as the edits before it
    left it. Returns the new text and, per edit, ``{"match": level,
    "replaced": count}``.

    Raises:
        PatchRefused: an edit does not match exactly once (or at all, with
            ``replace_all``), or the edits leave the text unchanged. Nothing
            is applied; the message names the edit and says how to retry.
    """
    current = text
    report = []
    for number, edit in enumerate(edits, start=1):
        prefix = _prefix(number, len(edits))
        for level in LEVELS:
            found = _find(current, edit["old"], edit["new"], level)
            if not found:
                continue
            if edit.get("replace_all"):
                found = _disjoint(found)
            elif len(found) > 1:
                raise PatchRefused(_ambiguous(current, found, what, level, prefix))
            current = _apply(current, found)
            report.append({"match": level, "replaced": len(found)})
            break
        else:
            raise PatchRefused(_not_found(current, edit["old"], what, prefix))
    if current == text:
        raise PatchRefused(
            f"the edits leave {what} as it was, so there is nothing to change",
        )
    return current, report


def syntax_refusal(source: str) -> Optional[str]:
    """Why ``source`` does not parse as Python, with the line; ``None`` if it does."""
    try:
        ast.parse(source)
    except SyntaxError as exc:
        lines = _lines(source)
        line_no = exc.lineno or 0
        if 0 < line_no <= len(lines):
            return f"line {line_no}: {exc.msg}\n{_excerpt(lines, line_no)}"
        return str(exc.msg)
    return None


__all__ = [
    "LEVELS",
    "PatchEdit",
    "PatchRefused",
    "apply_edits",
    "collect_edits",
    "occurrences",
    "syntax_refusal",
]
