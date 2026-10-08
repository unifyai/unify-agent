"""Shape summaries of recorded shell outputs (spec §F2, §I4).

A shell action records ``args`` (the command or argv), and ``response = {"exit_code": n | None,
"tail": text}``: the last part of the output (its first line may be cut). :func:`records` collects them
from exported episodes; :func:`output_shape` summarises a tail structurally:

* ``empty``; ``json`` (the key tree, as :mod:`.shapes` computes it); ``table`` (a delimiter that splits
  every line into the same number of fields, or ``"whitespace"`` for aligned columns); else ``lines``;
* ``first`` and ``last``: the shapes of the first and last non-empty lines, where each run of digits is
  ``9``, each run of letters ``a`` and each run of spaces one space (``"5 passed in 0.31s"`` becomes
  ``"9 a a 9.9a"``); other characters stay.

:func:`signature` is the part a fingerprint compares (format, tree, delimiter and width; never line
shapes or counts, which vary with every run). :func:`exit_codes` and :func:`by_shape` group records.
Imports only the standard library, so it runs in the consolidation sandbox as
``memlab.analysis.shellout``.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Iterable

from .shapes import sniff_delimiter, tree

_RUNS = re.compile(r"[0-9]+|[A-Za-z]+|[ \t]+")


@dataclass(frozen=True)
class ShellOutput:
    episode: str
    index: int
    channel: str
    command: str
    exit_code: int | None
    tail: str
    status: str


def _get(a: Any, name: str, default: Any = None) -> Any:
    return a.get(name, default) if isinstance(a, dict) else getattr(a, name, default)


def command_of(a: Any) -> str:
    args = _get(a, "args") or []
    if len(args) == 1 and isinstance(args[0], str):
        return args[0]
    return " ".join(str(x) for x in args)


def from_actions(episode: str, actions: Iterable[Any]) -> list[ShellOutput]:
    """The shell actions of one episode that recorded an output tail, in order."""
    out = []
    for pos, a in enumerate(actions):
        if (_get(a, "kind") or "tool") != "shell":
            continue
        resp = _get(a, "response")
        if not isinstance(resp, dict) or not isinstance(resp.get("tail"), str):
            continue
        code = resp.get("exit_code")
        idx = _get(a, "index", pos)
        out.append(
            ShellOutput(
                episode,
                int(idx if idx is not None else pos),
                _get(a, "channel"),
                command_of(a),
                code if isinstance(code, int) and not isinstance(code, bool) else None,
                resp["tail"],
                _get(a, "status"),
            ),
        )
    return out


def records(episodes: Iterable[dict]) -> list[ShellOutput]:
    """Every recorded shell output of exported episode rows (``/inputs/episodes/<id>.json``)."""
    out: list[ShellOutput] = []
    for row in episodes:
        out += from_actions(row["episode_id"], row.get("actions") or [])
    return out


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


def exit_codes(recs: Iterable[ShellOutput]) -> dict[str, list[int | None]]:
    """channel -> the distinct recorded exit codes (``None``: not recorded), sorted with None last."""
    out: dict[str, set] = {}
    for r in recs:
        out.setdefault(r.channel, set()).add(r.exit_code)
    return {
        ch: sorted(v, key=lambda c: (c is None, c if c is not None else 0))
        for ch, v in out.items()
    }


def by_shape(recs: Iterable[ShellOutput]) -> dict[str, list[tuple[str, int]]]:
    """``exit=<code> <signature>`` -> the (episode, index) pairs with that exit code and output shape."""
    out: dict[str, list[tuple[str, int]]] = {}
    for r in recs:
        k = f"exit={r.exit_code} {signature(output_shape(r.tail))}"
        out.setdefault(k, []).append((r.episode, r.index))
    return out
