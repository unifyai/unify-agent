"""The calls a library function's tests make, as the gate's test runs record them: its examples.

A function's examples are the calls its tests make on it. The gate does not read them from the test source: an
observe-only pytest plugin, loaded in the gate's box like the skip-mark plugin, records every call of a target
function made from test-side code (``memory/<package>/tests/**``) with its arguments (digest, type, a bounded JSON
form, and the file path a string names), its outcome, and the fixture files the test module had opened. It writes
one JSON line per event to ``/calls/calls.jsonl``; the host reads that file with :func:`read_calls`.

The file is written by code running beside model-written tests, so it is untrusted input: :func:`read_calls` opens it
without following links, bounds every line, and never raises. A record is :attr:`CallRecord.complete` only when the
plugin said it loaded, an end line arrived, and nothing was dropped or unreadable; a gate judges no example from an
incomplete record.

Calls are matched across runs by :class:`CallKey` (test, parameter indices, ordinal), never by pytest's node id, which
is derived from parameter values that a perturbed run changes.
"""

from __future__ import annotations

import json
import os
import stat
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

#: How a call ended: it returned, it raised ``MemoryInputError`` (refused), or it raised anything else.
OUTCOMES = ("returned", "refused", "raised")
#: The junit suite property the plugin writes; a report without it is not a recorded run.
CALLS_LOADED = "memory_calls_loaded"
#: Where the plugin's module is mounted read-only in the box (appended to PYTHONPATH).
CALLS_DIR = "/memory-calls"
#: The plugin's module name (``-p _memory_calls``).
CALLS_MODULE = "_memory_calls"
#: Where the plugin writes its record in the box (a host temp dir mounted read-write).
CALLS_OUT = "/calls/calls.jsonl"
TARGETS_ENV = "MEMORY_CALLS_TARGETS"
OUT_ENV = "MEMORY_CALLS_OUT"
#: Bounds the plugin keeps and the host enforces.
MAX_CALL_LINES = 20000
MAX_LINE_BYTES = 262144
MAX_RECORD_BYTES = 64 * 1024 * 1024
ARG_JSON_MAX = 65536


@dataclass(frozen=True)
class ArgRecord:
    name: str
    sha: str | None
    type: str
    json: Any | None  # None also when cut
    cut: bool
    path: str | None  # tree-relative path when the value names a file under /memory


@dataclass(frozen=True, order=True)
class CallKey:
    test: str  # "memory/<pkg>/tests/test_x.py::test_name" (no parameter brackets)
    indices: tuple[tuple[str, int], ...]
    ordinal: int


@dataclass(frozen=True)
class ExampleCall:
    item: str
    key: CallKey
    node: str
    file: str
    line: int
    args: tuple[ArgRecord, ...]
    outcome: str  # one of OUTCOMES
    raised: str | None
    result_sha: str | None
    result_type: str | None
    result_json: Any | None
    opened: tuple[str, ...]


@dataclass
class CallRecord:
    loaded: bool  # the start line and the junit property were both present
    calls: list[ExampleCall] = field(default_factory=list)
    unreadable: int = 0  # lines refused by the parser's bounds or shape
    dropped: int = 0  # the plugin's own count past MAX_CALL_LINES
    unresolved: list[str] = field(
        default_factory=list,
    )  # targets the plugin could not observe
    unobservable: str | None = None
    ended: bool = False  # an end line arrived

    @property
    def complete(self) -> bool:
        return (
            self.loaded
            and self.ended
            and self.unreadable == 0
            and self.dropped == 0
            and self.unobservable is None
        )


def _str(v: Any) -> str | None:
    return v if isinstance(v, str) else None


def _arg(raw: Any) -> ArgRecord | None:
    if (
        not isinstance(raw, dict)
        or not isinstance(raw.get("name"), str)
        or not isinstance(raw.get("type"), str)
    ):
        return None
    if not isinstance(raw.get("cut", False), bool):
        return None
    return ArgRecord(
        name=raw["name"],
        sha=_str(raw.get("sha")),
        type=raw["type"],
        json=raw.get("json"),
        cut=bool(raw.get("cut", False)),
        path=_str(raw.get("path")),
    )


def _call(raw: dict) -> ExampleCall | None:
    item, test, node, file = (raw.get(k) for k in ("item", "test", "node", "file"))
    line, ordinal, outcome = raw.get("line"), raw.get("ordinal"), raw.get("outcome")
    if not all(isinstance(v, str) for v in (item, test, node, file)):
        return None
    if not all(isinstance(v, int) and not isinstance(v, bool) for v in (line, ordinal)):
        return None
    if outcome not in OUTCOMES:
        return None
    indices = raw.get("indices") or {}
    if not isinstance(indices, dict) or not all(
        isinstance(k, str) and isinstance(v, int) and not isinstance(v, bool)
        for k, v in indices.items()
    ):
        return None
    args_raw = raw.get("args")
    if not isinstance(args_raw, list):
        return None
    args = tuple(_arg(a) for a in args_raw)
    if any(a is None for a in args):
        return None
    opened = raw.get("opened") or []
    if not isinstance(opened, list) or not all(isinstance(p, str) for p in opened):
        return None
    return ExampleCall(
        item=item,
        key=CallKey(test, tuple(sorted(indices.items())), ordinal),
        node=node,
        file=file,
        line=line,
        args=args,  # type: ignore[arg-type]
        outcome=outcome,
        raised=_str(raw.get("raised")),
        result_sha=_str(raw.get("result_sha")),
        result_type=_str(raw.get("result_type")),
        result_json=raw.get("result_json"),
        opened=tuple(opened),
    )


def read_calls(path: Path, loaded_property: bool = True) -> CallRecord:
    """The record at *path*, parsed within its bounds; never raises. *loaded_property*: the junit report carried
    :data:`CALLS_LOADED` (the caller reads the report). Each line over :data:`MAX_LINE_BYTES`, not JSON, or not of a
    known shape counts as unreadable; a missing, linked or oversized file is an empty, not loaded record.
    """
    rec = CallRecord(loaded=False)
    try:
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK | os.O_CLOEXEC)
    except OSError:
        return rec
    try:
        st = os.fstat(fd)
        if not stat.S_ISREG(st.st_mode) or st.st_size > MAX_RECORD_BYTES:
            return rec
        with os.fdopen(fd, "rb", closefd=False) as fh:
            data = fh.read(MAX_RECORD_BYTES + 1)
    except OSError:
        return rec
    finally:
        os.close(fd)
    started = False
    for raw_line in data.split(b"\n"):
        if not raw_line.strip():
            continue
        if len(raw_line) > MAX_LINE_BYTES:
            rec.unreadable += 1
            continue
        try:
            row = json.loads(raw_line)
        except ValueError:
            rec.unreadable += 1
            continue
        if not isinstance(row, dict):
            rec.unreadable += 1
            continue
        event = row.get("event")
        if event == "start":
            started = True
            why = row.get("unobservable")
            rec.unobservable = why[:200] if isinstance(why, str) else None
        elif event == "open":
            continue  # opens are summarised per call in "opened"
        elif event == "call":
            call = _call(row)
            if call is None:
                rec.unreadable += 1
            else:
                rec.calls.append(call)
        elif event == "end":
            rec.ended = True
            dropped = row.get("dropped", 0)
            rec.dropped = (
                dropped
                if isinstance(dropped, int)
                and not isinstance(dropped, bool)
                and dropped >= 0
                else 1
            )
            unresolved = row.get("unresolved") or []
            if isinstance(unresolved, list):
                rec.unresolved = [u for u in unresolved if isinstance(u, str)][:100]
        else:
            rec.unreadable += 1
    rec.loaded = started and loaded_property
    return rec


#: The plugin's source (standard library only), written per run as ``_memory_calls.py``. Filled in with the gate's
#: verification; until then a run that asks for calls fails loudly rather than recording nothing.
CALLS_SOURCE = ""


def plugin_files(dest: Path) -> None:
    """Write the plugin module into *dest* (a fresh host directory mounted at :data:`CALLS_DIR`)."""
    if not CALLS_SOURCE:
        raise NotImplementedError("the call-recording plugin is not built yet")
    (Path(dest) / f"{CALLS_MODULE}.py").write_text(CALLS_SOURCE)


def calls_env(targets: list[str]) -> dict[str, str]:
    """The box environment that tells the plugin which functions to observe and where to write."""
    return {TARGETS_ENV: json.dumps(sorted(set(targets))), OUT_ENV: CALLS_OUT}
