"""Adapter 2 (spec §C2): shell actions.

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
``unify.memory_v2.redact`` first, and output is kept only as a bounded tail.
``shell_fingerprint`` summarises rows per executable without any values: exit
codes, and the shape of each output tail (a line-count bucket and the
character-class structure of the first token).

Nothing here is wired into the actor yet.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Optional

from ...episodes import Action
from ...redact import Redactor

__all__ = [
    "BASH_CHANNEL",
    "MAX_ARG_CHARS",
    "MAX_ARGV",
    "MAX_TAIL_BYTES",
    "ShellCellResult",
    "action_from_audit",
    "action_from_cell",
    "bounded_tail",
    "program_of",
    "shell_actions",
    "shell_fingerprint",
]

MAX_TAIL_BYTES = 8 * 1024  # each of stdout and stderr
MAX_ARGV = 64  # argv entries kept per row
MAX_ARG_CHARS = 4096  # characters kept per argv entry (and per bash command)
BASH_CHANNEL = "bash"
# Audit event names that denote a started process. The audit adapter emits
# "subprocess"; the raw sys.audit names are accepted too.
_SPAWN_EVENTS = frozenset(
    {
        "subprocess",
        "subprocess.Popen",
        "os.system",
        "os.posix_spawn",
        "os.exec",
        "os.spawn",
    },
)
_UNKNOWN_PROGRAM = "unknown"


@dataclass
class ShellCellResult:
    """One bash cell as the harness ran it."""

    cell: int
    command: str
    exit_code: Optional[int]
    stdout: str = ""
    stderr: str = ""
    error: Optional[str] = None  # the runner's error text (timeout, session died, ...)

    @classmethod
    def from_execute_result(
        cls,
        cell: int,
        command: str,
        out: Mapping[str, Any],
    ) -> "ShellCellResult":
        """From the dict ``SessionExecutor.execute(language="bash")`` returns:
        ``stdout``/``stderr`` (a str or a list of text parts), ``result`` (the
        exit code, or None) and ``error``."""
        code = out.get("result")
        return cls(
            cell=cell,
            command=command,
            exit_code=(
                code if isinstance(code, int) and not isinstance(code, bool) else None
            ),
            stdout=_text_of(out.get("stdout")),
            stderr=_text_of(out.get("stderr")),
            error=out.get("error") or None,
        )


def _text_of(parts: Any) -> str:
    if parts is None:
        return ""
    if isinstance(parts, str):
        return parts
    if isinstance(parts, (bytes, bytearray)):
        return bytes(parts).decode("utf-8", errors="replace")
    if isinstance(parts, (list, tuple)):
        out = []
        for p in parts:
            if isinstance(p, str):
                out.append(p)
            elif isinstance(p, Mapping):
                out.append(str(p.get("text", "")))
            else:
                out.append(str(getattr(p, "text", "")))
        return "".join(out)
    return str(parts)


def bounded_tail(text: Any, redactor: Redactor, limit: int = MAX_TAIL_BYTES) -> str:
    """The redacted last ``limit`` bytes (UTF-8) of ``text``.

    Redaction runs on the whole text before the cut, so a secret that straddles
    the cut is never left half-visible."""
    if limit <= 0:
        return ""
    s = redactor.text(_text_of(text))
    raw = s.encode("utf-8", errors="replace")
    if len(raw) <= limit:
        return s
    return raw[-limit:].decode("utf-8", errors="ignore")  # drop a split leading char


def _decode(a: Any) -> str:
    try:
        return os.fsdecode(a) if isinstance(a, (bytes, os.PathLike)) else str(a)
    except TypeError:
        return str(a)


def _arg(a: Any, redactor: Redactor) -> str:
    """One argv entry: decoded, redacted in full, and only then capped at
    ``MAX_ARG_CHARS`` with a marker, so a secret that straddles the cap is
    never left as a fragment. Redaction is linear in the entry, and the OS
    bounds an entry by ARG_MAX."""
    s = redactor.text(_decode(a))
    if len(s) <= MAX_ARG_CHARS:
        return s
    return s[:MAX_ARG_CHARS] + f"<truncated: {len(s) - MAX_ARG_CHARS} chars>"


def _argv(raw: Any, redactor: Redactor) -> list[str]:
    """argv as a list of redacted, capped strings. A string (``os.system``, or
    Popen with ``shell=True`` before expansion) runs under ``sh -c``. Entries
    past ``MAX_ARGV`` are dropped and counted in a final marker entry."""
    if raw is None:
        return []
    if isinstance(raw, (str, bytes, os.PathLike)):
        return ["sh", "-c", _arg(raw, redactor)]
    if isinstance(raw, (list, tuple)):
        out = [_arg(a, redactor) for a in raw[:MAX_ARGV]]
        if len(raw) > MAX_ARGV:
            out.append(f"<truncated: {len(raw) - MAX_ARGV} more args>")
        return out
    return [_arg(raw, redactor)]


def program_of(
    argv: list[str],
    exe: Any = None,
    redactor: Optional[Redactor] = None,
) -> str:
    """The channel key: the basename of argv[0] (else of ``exe``). The name is
    redacted in full before the basename is taken, so a secret in the program
    path never reaches the channel."""
    red = redactor or Redactor()
    for cand in (argv[0] if argv else None, exe):
        if cand:
            name = os.path.basename(red.text(_decode(cand)).rstrip("/"))
            if name:
                return name[:MAX_ARG_CHARS]
    return _UNKNOWN_PROGRAM


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


def _error_text(
    exit_code: Optional[int],
    error: Optional[str],
    redactor: Redactor,
) -> Optional[str]:
    if error:
        return bounded_tail(error, redactor, MAX_ARG_CHARS)
    if exit_code not in (None, 0):
        return f"exit status {exit_code}"
    return None


def _is_spawn(record: Mapping[str, Any]) -> bool:
    return record.get("event", "subprocess") in _SPAWN_EVENTS


def action_from_audit(
    record: Mapping[str, Any],
    redactor: Optional[Redactor] = None,
    *,
    tail_bytes: int = MAX_TAIL_BYTES,
) -> Action:
    """One process record from the audit adapter as a shell action."""
    red = redactor or Redactor()
    args = _argv(record.get("argv", record.get("args")), red)
    channel = program_of(args, record.get("exe") or record.get("executable"), red)
    exit_code = record.get("exit_code")
    if isinstance(exit_code, bool) or not isinstance(exit_code, int):
        exit_code = None
    captured = exit_code is not None or any(
        record.get(k) is not None for k in ("stdout", "stderr")
    )
    if not captured:
        return Action(
            cell=_cell(record.get("cell")),
            channel=channel,
            method="run",
            args=args,
            kwargs={},
            response=None,
            status="unrecorded",
            kind="shell",
        )
    error = record.get("error") or None
    return Action(
        cell=_cell(record.get("cell")),
        channel=channel,
        method="run",
        args=args,
        kwargs={},
        response={
            "exit_code": exit_code,
            "stdout_tail": bounded_tail(record.get("stdout"), red, tail_bytes),
            "stderr_tail": bounded_tail(record.get("stderr"), red, tail_bytes),
        },
        status=_status(exit_code, error),
        error=_error_text(exit_code, error, red),
        kind="shell",
    )


def action_from_cell(
    result: ShellCellResult,
    redactor: Optional[Redactor] = None,
    *,
    tail_bytes: int = MAX_TAIL_BYTES,
) -> Action:
    """One bash cell as a shell action (channel ``"bash"``, args ``[command]``)."""
    red = redactor or Redactor()
    return Action(
        cell=_cell(result.cell),
        channel=BASH_CHANNEL,
        method="run",
        args=[_arg(result.command, red)],
        kwargs={},
        response={
            "exit_code": result.exit_code,
            "stdout_tail": bounded_tail(result.stdout, red, tail_bytes),
            "stderr_tail": bounded_tail(result.stderr, red, tail_bytes),
        },
        status=_status(result.exit_code, result.error),
        error=_error_text(result.exit_code, result.error, red),
        kind="shell",
    )


def shell_actions(
    audit_records: Iterable[Mapping[str, Any]] = (),
    cell_results: Iterable[ShellCellResult] = (),
    *,
    redactor: Optional[Redactor] = None,
    tail_bytes: int = MAX_TAIL_BYTES,
) -> list[Action]:
    """All shell actions of a request, in cell order (stable within a cell:
    bash cells first, then process records in the order they were audited).
    Audit records of other events (file opens, listings) are skipped."""
    red = redactor or Redactor()
    rows = [action_from_cell(r, red, tail_bytes=tail_bytes) for r in cell_results]
    rows += [
        action_from_audit(r, red, tail_bytes=tail_bytes)
        for r in audit_records
        if _is_spawn(r)
    ]
    return sorted(rows, key=lambda a: a.cell)


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
