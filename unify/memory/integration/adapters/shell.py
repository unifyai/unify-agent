"""Adapter 2: shell actions.

Two sources become ``Action(kind="shell")`` rows:

- **Process records** from the worker's audit hook (a separate adapter): one dict
  per started process, ``{"event": "subprocess", "argv": [...], "cell": n}``.
  The channel is the basename of ``argv[0]``. An audit hook sees the start of a
  process, not its end, so a record without captured output becomes a row with
  ``status="unrecorded"`` and ``response=None``. A record that does carry
  ``exit_code`` / ``stdout`` / ``stderr`` is used as given.
- **Bash cells** run by the harness's shell-cell runner (``%%bash``,
  ``execute_code(language="bash")``): the command text, exit code and output.
  The channel is ``"bash"``. The runner at the time of writing interleaves
  stderr into stdout (``2>&1``), so ``stderr_tail`` is usually empty.

Every string that reaches a row (argv, tails, error text) passes through
``unify.memory.redact`` first, and output is kept only as a bounded tail.
``shell_fingerprint`` summarises rows per executable without any values: exit
codes, and the shape of each output tail (a line-count bucket and the
character-class structure of the first token).

Nothing here is wired into the actor yet.
"""

from __future__ import annotations

from typing import Any, Iterable, Mapping, Optional

from ...episodes import Action

__all__ = [
    "MAX_ARGV",
    "MAX_TAIL_BYTES",
    "shell_fingerprint",
]

MAX_TAIL_BYTES = 8 * 1024  # each of stdout and stderr
MAX_ARGV = 64  # argv entries kept per row
def _cell(v: Any) -> int:
    """A cell index; -1 when missing or not an integer."""
    if isinstance(v, bool):
        return -1
    try:
        return int(v)
    except (TypeError, ValueError):
        return -1


def _status(exit_code: Optional[int], error: Optional[str]) -> str:
    if exit_code is None:
        return "error" if error else "unrecorded"
    return "ok" if exit_code == 0 else "error"


# --- fingerprint -------------------------------------------------------------

_LINE_BUCKETS = ((0, "0"), (1, "1"), (9, "2-9"), (99, "10-99"), (999, "100-999"))
_MAX_TOKEN_SHAPE = 24


def _line_bucket(n: int) -> str:
    for bound, label in _LINE_BUCKETS:
        if n <= bound:
            return label
    return "1000+"


def _token_shape(tok: str) -> str:
    """Character-class structure of a token: runs of letters become ``a``,
    runs of digits ``9``, any other non-ASCII run ``u``; ASCII punctuation is
    kept. ``drwxr-xr-x`` -> ``a-a-a``, ``2026-10-08`` -> ``9-9-9``,
    ``/home/x.py`` -> ``/a/a.a``. No values survive."""
    out: list[str] = []
    for ch in tok:
        if (ch.isascii() and ch.isalpha()) or ch == "_":
            c = "a"
        elif ch.isascii() and ch.isdigit():
            c = "9"
        elif ch.isascii():
            c = ch
        else:
            c = "u"
        if c in "a9u" and out and out[-1] == c:
            continue
        out.append(c)
    s = "".join(out)
    return s if len(s) <= _MAX_TOKEN_SHAPE else s[:_MAX_TOKEN_SHAPE] + "~"


def output_shape(text: str) -> str:
    """``lines=<bucket>`` plus ``first=<token shape>`` of the first non-blank line."""
    lines = (text or "").splitlines()
    bucket = _line_bucket(len(lines))
    first = next((ln.split()[0] for ln in lines if ln.split()), None)
    return f"lines={bucket}" + (f",first={_token_shape(first)}" if first else "")


def shell_fingerprint(actions: Iterable[Action]) -> dict[str, dict[str, list[str]]]:
    """Per executable (channel) of the shell rows: the exit-code patterns seen
    (``"0"``, ``"2"``, ``"none"`` for a run with no exit code, ``"unrecorded"``)
    and the output shapes (``stdout:lines=2-9,first=a-a-a|stderr:lines=0``).
    Values never enter: tails are reduced to line-count buckets and the
    character-class structure of their first token."""
    out: dict[str, dict[str, set[str]]] = {}
    for a in actions:
        if a.kind != "shell":
            continue
        slot = out.setdefault(a.channel, {"exit_codes": set(), "shapes": set()})
        if not isinstance(a.response, Mapping):
            slot["exit_codes"].add("unrecorded")
            continue
        code = a.response.get("exit_code")
        slot["exit_codes"].add("none" if code is None else str(code))
        slot["shapes"].add(
            f"stdout:{output_shape(a.response.get('stdout_tail', ''))}"
            f"|stderr:{output_shape(a.response.get('stderr_tail', ''))}",
        )
    return {
        ch: {"exit_codes": sorted(v["exit_codes"]), "shapes": sorted(v["shapes"])}
        for ch, v in sorted(out.items())
    }
