"""Shape summaries of recorded shell outputs.

A shell action records ``args`` (the command or argv), and ``response = {"exit_code": n | None,
"tail": text}``: the last part of the output (its first line may be cut). :func:`output_shape` summarises a
tail structurally:

* ``empty``; ``json`` (the key tree, as :mod:`.shapes` computes it); ``table`` (a delimiter that splits
  every line into the same number of fields, or ``"whitespace"`` for aligned columns); else ``lines``;
* ``first`` and ``last``: the shapes of the first and last non-empty lines, where each run of digits is
  ``9``, each run of letters ``a`` and each run of spaces one space (``"5 passed in 0.31s"`` becomes
  ``"9 a a 9.9a"``); other characters stay.

:func:`signature` is the part a fingerprint compares (format, tree, delimiter and width; never line
shapes or counts, which vary with every run). Imports only the standard library.
"""

from __future__ import annotations

import json
import re
from typing import Any

from .shapes import sniff_delimiter, tree

_RUNS = re.compile(r"[0-9]+|[A-Za-z]+|[ \t]+")


def line_shape(line: str, limit: int = 160) -> str:
    def sub(m: re.Match) -> str:
        s = m.group(0)
        return "9" if s[0].isdigit() else "a" if s[0].isalpha() else " "

    return _RUNS.sub(sub, line.strip())[:limit]


def output_shape(tail: str) -> dict:
    lines = [ln for ln in (tail or "").splitlines() if ln.strip()]
    if not lines:
        return {"format": "empty", "lines": 0}
    out: dict[str, Any] = {
        "lines": len(lines),
        "first": line_shape(lines[0]),
        "last": line_shape(lines[-1]),
    }
    s = tail.strip()
    if s[:1] in ("{", "["):
        try:
            out.update(format="json", tree=tree(json.loads(s)))
            return out
        except (ValueError, RecursionError):
            pass
    if len(lines) >= 2:
        d = sniff_delimiter("\n".join(lines) + "\n")
        if d is not None:
            out.update(format="table", delimiter=d, width=len(lines[0].split(d)))
            return out
        widths = {len(ln.split()) for ln in lines}
        if len(widths) == 1 and next(iter(widths)) >= 2:
            out.update(format="table", delimiter="whitespace", width=widths.pop())
            return out
    out["format"] = "lines"
    return out


def signature(shape: dict) -> str:
    """The stable part of an output shape: format, JSON tree, delimiter and width."""
    keep = {k: shape[k] for k in ("format", "tree", "delimiter", "width") if k in shape}
    return json.dumps(keep, sort_keys=True, ensure_ascii=False)
