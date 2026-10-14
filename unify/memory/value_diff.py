"""The one comparison of an expected value with an actual one, and the difference report it gives.

Used wherever a value a function returned is checked against a value the recording shows: the gate's run of a
procedure's recipe on its cases (:mod:`.recipes`) and any later check of a candidate against shown examples.

* **Same** (:func:`same`): the two values are equal as JSON (canonical form: sorted keys, no spacing). Text and a
  JSON value meet halfway: text that parses as JSON is compared as the value it parses to, and other text is compared
  without its leading and trailing whitespace, so ``"42"`` shown in a message and ``42`` returned are the same.
* **The report** (:func:`difference`): None when they are the same; else text that names the expected value and the
  actual one and where they differ. JSON values (objects and arrays) get a structured difference, one line per place
  (``[2][1]: expected 3, got 4``; a missing or an extra key; a list of another length); text gets a unified diff of
  its lines (or one line naming both, for one-line text); values of different types say so. The report is bounded at
  :data:`REPORT_CHARS` characters, and a cut is marked with how many places were left out.

Standard library only, and pure: nothing here runs the values it is given.
"""

from __future__ import annotations

import difflib
import json
from typing import Any

#: The longest report, in characters; a cut is marked.
REPORT_CHARS = 6000
#: The longest rendering of one value inside a report line, in characters; a cut is marked.
VALUE_CHARS = 600


def canonical(value: Any) -> str:
    """*value* as canonical JSON text (sorted keys, no spacing; an unencodable leaf as its text)."""
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), default=str)


def normal(value: Any) -> Any:
    """*value* as it is compared: text that parses as JSON as the value it parses to, other text stripped of leading
    and trailing whitespace, a tuple as a list, anything else as its JSON form."""
    if isinstance(value, str):
        text = value.strip()
        try:
            return json.loads(text)
        except (ValueError, RecursionError):
            return text
    try:
        return json.loads(canonical(value))
    except (ValueError, RecursionError):
        return str(value)


def same(expected: Any, actual: Any) -> bool:
    """Whether *actual* is *expected* (module docstring)."""
    return canonical(normal(expected)) == canonical(normal(actual))


def _show(value: Any, limit: int = VALUE_CHARS) -> str:
    text = canonical(value)
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _kind(value: Any) -> str:
    if isinstance(value, dict):
        return "an object"
    if isinstance(value, list):
        return "a list"
    if isinstance(value, str):
        return "text"
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "a boolean"
    return "a number" if isinstance(value, (int, float)) else type(value).__name__


def _places(expected: Any, actual: Any, path: str, out: list[str]) -> None:
    """Each place where *actual* differs from *expected*, as a line, depth first."""
    if canonical(expected) == canonical(actual):
        return
    where = path or "the value"
    if isinstance(expected, dict) and isinstance(actual, dict):
        for key in sorted(set(expected) | set(actual), key=str):
            sub = f"{path}.{key}" if path else str(key)
            if key not in actual:
                out.append(f"{sub}: expected {_show(expected[key])}, missing")
            elif key not in expected:
                out.append(f"{sub}: not expected, got {_show(actual[key])}")
            else:
                _places(expected[key], actual[key], sub, out)
        return
    if isinstance(expected, list) and isinstance(actual, list):
        if len(expected) != len(actual):
            out.append(f"{where}: expected {len(expected)} items, got {len(actual)}")
        for i, (e, a) in enumerate(zip(expected, actual)):
            _places(e, a, f"{path}[{i}]", out)
        for i in range(len(actual), len(expected)):
            out.append(f"{path}[{i}]: expected {_show(expected[i])}, missing")
        for i in range(len(expected), len(actual)):
            out.append(f"{path}[{i}]: not expected, got {_show(actual[i])}")
        return
    if _kind(expected) != _kind(actual):
        out.append(f"{where}: expected {_kind(expected)} {_show(expected)}, got {_kind(actual)} {_show(actual)}")
        return
    out.append(f"{where}: expected {_show(expected)}, got {_show(actual)}")


def _bounded(head: list[str], lines: list[str], what: str) -> str:
    text = "\n".join(head)
    for n, line in enumerate(lines):
        if len(text) + 1 + len(line) > REPORT_CHARS:
            return text + f"\n… {len(lines) - n} more {what} not shown"
        text += "\n" + line
    return text


def difference(expected: Any, actual: Any) -> str | None:
    """None when *actual* is *expected* (:func:`same`); else the difference report (module docstring)."""
    e, a = normal(expected), normal(actual)
    if canonical(e) == canonical(a):
        return None
    head = [f"expected: {_show(e)}", f"actual:   {_show(a)}"]
    if isinstance(e, str) and isinstance(a, str) and ("\n" in e or "\n" in a):
        diff = list(
            difflib.unified_diff(e.splitlines(), a.splitlines(), "expected", "actual", lineterm="", n=2),
        )
        return _bounded([*head, "difference (unified, expected to actual):"], diff, "diff lines")
    places: list[str] = []
    _places(e, a, "", places)
    return _bounded([*head, f"difference ({len(places)} places):"], places, "places")
