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
                rec.unresolved = [u[:300] for u in unresolved if isinstance(u, str)][:100]
        else:
            rec.unreadable += 1
    rec.loaded = started and loaded_property
    return rec


#: The plugin's source (standard library and pytest only), written per run as ``_memory_calls.py`` into a fresh
#: host directory mounted read-only at :data:`CALLS_DIR`. The host never imports it (importing it changes nothing,
#: but its hooks belong to the box's pytest). Unresolved targets are listed in the end line as ``<item> (<what it
#: is>)``; :func:`unresolved_kind` splits them.
CALLS_SOURCE = r'''
"""Record the calls test code makes of the library's functions (the memory gate's runs; see example_calls).

Observe-only: ``sys.monitoring`` events on each target's code object and an audit hook on opens; no code under test is
wrapped or changed. Standard library and pytest only.
"""
import datetime
import decimal
import hashlib
import importlib
import inspect
import json
import os
import pathlib
import sys
import types

import pytest
from _pytest.junitxml import xml_key

LOADED = "memory_calls_loaded"
TARGETS_ENV = "MEMORY_CALLS_TARGETS"
OUT_ENV = "MEMORY_CALLS_OUT"
ROOT = "/memory"
MAX_CALL_LINES = 20000
MAX_LINE_BYTES = 262144
MAX_RECORD_BYTES = 64 * 1024 * 1024
ARG_JSON_MAX = 65536
MAX_OBJECT_DEPTH = 8
TOOL_IDS = (2, 3, 4)
END_RESERVE = 65536
MAX_UNRESOLVED = 100
MAX_OPENED = 2000
MAX_STACK = 400
_GENERATOR_FLAGS = inspect.CO_GENERATOR | inspect.CO_COROUTINE | inspect.CO_ASYNC_GENERATOR
_MISSING = object()

STATE = {
    "out": None,
    "tool": None,
    "targets": [],
    "codes": {},
    "wrappers": {},
    "pending": {},
    "node": None,
    "collecting": None,
    "opened": {},
    "open_seen": set(),
    "open_lines": 0,
    "ordinals": {},
    "infos": {},
    "kinds": {},
    "calls": 0,
    "dropped": 0,
    "outside": 0,
    "unattributed_opens": 0,
    "unresolved": [],
    "unobservable": None,
    "ended": False,
}


class Opaque(Exception):
    """A value with no canonical form."""


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), default=str)


def normalise(v, depth=0):
    """The JSON value whose canonical text digests *v*: the host's ``fixtures.normalise``, copied (the plugin is
    standalone). Raises Opaque for a value without one."""
    if depth > MAX_OBJECT_DEPTH:
        raise Opaque()
    if v is None or isinstance(v, (bool, int, float, str)):
        return v
    if isinstance(v, (list, tuple)):
        return [normalise(x, depth + 1) for x in v]
    if isinstance(v, (set, frozenset)):
        return sorted((normalise(x, depth + 1) for x in v), key=canonical)
    if isinstance(v, dict):
        out = {}
        for k, x in v.items():
            if not isinstance(k, str):
                raise Opaque()
            out[k] = normalise(x, depth + 1)
        return out
    if isinstance(v, (bytes, bytearray)):
        return {"__bytes__": hashlib.sha256(bytes(v)).hexdigest()}
    if isinstance(v, decimal.Decimal):
        return str(v)
    if isinstance(v, (datetime.date, datetime.datetime)):
        return v.isoformat()
    attrs = getattr(v, "__dict__", None)
    if isinstance(attrs, dict):
        return {"__type__": type(v).__name__, "vars": normalise(dict(attrs), depth + 1)}
    raise Opaque()


def _type_name(v):
    return type(v).__name__[:100]


def file_path(v):
    """The tree-relative path a str (or a path object) argument names, when it names a file under /memory."""
    if isinstance(v, pathlib.PurePath):
        v = str(v)
    if not isinstance(v, str) or not v or len(v) > 4096 or "\0" in v:
        return None
    p = os.path.normpath(os.path.join(ROOT, v))
    if not p.startswith(ROOT + "/"):
        return None
    try:
        if not os.path.isfile(p) or not os.path.realpath(p).startswith(ROOT + "/"):
            return None
    except Exception:
        return None
    return p[len(ROOT) + 1 :]


def describe(name, v):
    """One argument's record: digest, type, bounded JSON form and the file it names."""
    rec = {"name": name, "sha": None, "type": _type_name(v), "json": None, "cut": False, "path": file_path(v)}
    try:
        n = normalise(v)
        data = canonical(n).encode("utf-8")
    except Exception:  # no canonical form, a cycle, too deep, or text that is not UTF-8
        return rec
    rec["sha"] = hashlib.sha256(data).hexdigest()
    if len(data) <= ARG_JSON_MAX:
        rec["json"] = n
    else:
        rec["cut"] = True
    return rec


def outcome_of(exc):
    """("refused", None) for a MemoryInputError (by name, anywhere in its type's MRO), else ("raised", name)."""
    t = type(exc)
    if any(c.__name__ == "MemoryInputError" for c in t.__mro__):
        return "refused", None
    return "raised", t.__name__ if t.__module__ == "builtins" else "another exception"


def tree_kind(filename, cache={}):
    """"test" for a file under memory/<package>/tests/, "library" for another file of a package, else "other"."""
    kind = cache.get(filename)
    if kind is None:
        kind = "other"
        p = os.path.normpath(filename) if filename.startswith("/") else ""
        if p.startswith(ROOT + "/memory/"):
            parts = p[len(ROOT) + 1 :].split("/")
            if len(parts) >= 3:
                kind = "test" if parts[2] == "tests" else "library"
        cache[filename] = kind
    return kind


def _rel(filename):
    p = os.path.normpath(filename)
    return p[len(ROOT) + 1 :] if p.startswith(ROOT + "/") else p


def data_path(p):
    """The tree-relative path of a fixture under memory/<package>/tests/data/, else None."""
    if not p.startswith(ROOT + "/memory/"):
        return None
    rel = p[len(ROOT) + 1 :]
    parts = rel.split("/")
    if len(parts) >= 5 and parts[2] == "tests" and parts[3] == "data":
        return rel
    return None


def _reading(mode, flags):
    if isinstance(mode, str):
        return not any(ch in mode for ch in "wax+")
    if isinstance(flags, int):
        return not flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND)
    return True


class Out:
    """The record file: bounded lines, a bounded file, with room kept for the end line."""

    def __init__(self, path):
        self.fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC | os.O_NOFOLLOW | os.O_CLOEXEC, 0o644)
        self.size = 0

    def write(self, row, final=False):
        data = (json.dumps(row, separators=(",", ":"), default=str) + "\n").encode("ascii")
        if len(data) > MAX_LINE_BYTES:
            return False
        if self.size + len(data) > MAX_RECORD_BYTES - (0 if final else END_RESERVE):
            return False
        view = memoryview(data)
        while view:
            n = os.write(self.fd, view)
            view = view[n:]
        self.size += len(data)
        return True


def _info(item):
    info = STATE["infos"].get(id(item))
    if info is None or info["item"] is not item:
        nodeid = item.nodeid
        name, original = getattr(item, "name", ""), getattr(item, "originalname", None)
        test = nodeid
        if original and name != original and nodeid.endswith(name):
            test = nodeid[: len(nodeid) - len(name)] + original
        spec = getattr(item, "callspec", None)
        indices = dict(getattr(spec, "indices", None) or {}) if spec is not None else {}
        info = {
            "item": item,
            "node": nodeid,
            "test": test,
            "indices": {str(k): int(v) for k, v in indices.items()},
            "module": nodeid.split("::", 1)[0],
        }
        STATE["infos"][id(item)] = info
    return info


def _note_open(rel):
    node = STATE["node"]
    if node is not None:
        info = _info(node)
        module, where, line = info["module"], info["node"], {"event": "open", "phase": "test", "node": info["node"]}
    elif STATE["collecting"] is not None:
        module = where = STATE["collecting"]
        line = {"event": "open", "phase": "collect", "module": module}
    else:
        STATE["unattributed_opens"] += 1
        return
    opened = STATE["opened"].setdefault(module, [])
    if rel not in opened and len(opened) < MAX_OPENED:
        opened.append(rel)
    if (where, rel) not in STATE["open_seen"] and STATE["open_lines"] < MAX_CALL_LINES:
        STATE["open_seen"].add((where, rel))
        line["path"] = rel
        if STATE["out"].write(line):
            STATE["open_lines"] += 1


def audit(event, args):
    if event != "open":
        return
    try:
        p, mode, flags = args
        if isinstance(p, int):
            return
        if not isinstance(p, str):
            p = os.fsdecode(p)
        if not _reading(mode, flags):
            return
        if not p.startswith("/"):
            p = os.path.join(os.getcwd(), p)
        rel = data_path(os.path.normpath(p))
        if rel is not None:
            _note_open(rel)
    except Exception:  # an odd path or event: this open is not attributed
        STATE["unattributed_opens"] += 1


def _frame(code):
    f = sys._getframe(2)  # the target's frame: callback, this helper, then the frame the event is about
    for _ in range(4):
        if f is None or f.f_code is code:
            return f
        f = f.f_back
    return None


def test_site(frame, items):
    """(file, line) of the test-side frame the call came from, or None when a library frame comes first."""
    wrappers = set()
    for item in items:
        wrappers |= STATE["wrappers"].get(item, set())
    f, depth = frame.f_back, 0
    while f is not None and depth < MAX_STACK:
        kind = tree_kind(f.f_code.co_filename)
        if kind == "test":
            return _rel(f.f_code.co_filename), f.f_lineno
        if kind == "library" and f.f_code not in wrappers:
            return None
        f, depth = f.f_back, depth + 1
    return None


def arguments(frame, code):
    loc = frame.f_locals
    n = code.co_argcount + code.co_kwonlyargcount
    out = [describe(name, loc[name]) for name in code.co_varnames[:n] if name in loc]
    i = n
    if code.co_flags & inspect.CO_VARARGS:
        name = code.co_varnames[i]
        i += 1
        for j, v in enumerate(loc.get(name) or ()):
            out.append(describe(f"{name}[{j}]", v))
    if code.co_flags & inspect.CO_VARKEYWORDS:
        name = code.co_varnames[i]
        for k, v in (loc.get(name) or {}).items():
            out.append(describe(f"{name}.{k}", v))
    return out


def on_start(code, offset):
    try:
        items = STATE["codes"].get(code)
        if not items:
            return
        frame = _frame(code)
        if frame is None:
            STATE["dropped"] += 1
            return
        site = test_site(frame, items)
        if site is None:
            return
        node = STATE["node"]
        if node is None:
            STATE["outside"] += 1
            return
        STATE["pending"][id(frame)] = (frame, items, node, site, arguments(frame, code))
    except Exception:
        STATE["dropped"] += 1


def _fits(row):
    """Cut the largest JSON forms until the line is within MAX_LINE_BYTES; False when it cannot be."""
    for _ in range(len(row["args"]) + 2):
        if len(json.dumps(row, separators=(",", ":"), default=str)) + 1 <= MAX_LINE_BYTES:
            return True
        sized = [(len(json.dumps(a["json"], default=str)), a) for a in row["args"] if a["json"] is not None]
        if row["result_json"] is not None:
            sized.append((len(json.dumps(row["result_json"], default=str)), None))
        if not sized:
            return False
        _, biggest = max(sized, key=lambda s: s[0])
        if biggest is None:
            row["result_json"], row["result_cut"] = None, True
        else:
            biggest["json"], biggest["cut"] = None, True
    return len(json.dumps(row, separators=(",", ":"), default=str)) + 1 <= MAX_LINE_BYTES


def finish(code, entry, outcome, raised, retval):
    frame, items, node, (file, line), args = entry
    info = _info(node)
    generator = bool(code.co_flags & _GENERATOR_FLAGS)
    result = describe("result", retval) if outcome == "returned" and not generator else None
    for item in items:
        k = (info["node"], item)
        ordinal = STATE["ordinals"].get(k, 0)
        STATE["ordinals"][k] = ordinal + 1
        if STATE["calls"] >= MAX_CALL_LINES:
            STATE["dropped"] += 1
            continue
        row = {
            "event": "call",
            "item": item,
            "node": info["node"],
            "test": info["test"],
            "indices": info["indices"],
            "ordinal": ordinal,
            "file": file,
            "line": line,
            "args": [dict(a) for a in args],
            "outcome": outcome,
            "raised": raised,
            "result_sha": result["sha"] if result else None,
            "result_type": result["type"] if result else (_type_name(retval) if generator else None),
            "result_json": result["json"] if result else None,
            "result_cut": result["cut"] if result else False,
            "opened": list(STATE["opened"].get(info["module"], [])),
        }
        if _fits(row) and STATE["out"].write(row):
            STATE["calls"] += 1
        else:
            STATE["dropped"] += 1


def _end_of(code, outcome, raised, retval):
    if code not in STATE["codes"]:
        return
    frame = _frame(code)
    entry = STATE["pending"].pop(id(frame), None) if frame is not None else None
    if entry is None:
        return
    if entry[0] is not frame:
        STATE["dropped"] += 1
        return
    finish(code, entry, outcome, raised, retval)


def on_return(code, offset, retval):
    try:
        _end_of(code, "returned", None, retval)
    except Exception:
        STATE["dropped"] += 1


def on_unwind(code, offset, exc):
    if code not in STATE["codes"]:
        return
    try:
        outcome, raised = outcome_of(exc)
        _end_of(code, outcome, raised, None)
    except Exception:
        STATE["dropped"] += 1


def claim_tool():
    mon = getattr(sys, "monitoring", None)
    if mon is None:
        return "no sys.monitoring"
    for i in TOOL_IDS:
        if mon.get_tool(i) is None:
            try:
                mon.use_tool_id(i, "memory-calls")
            except ValueError:
                continue
            STATE["tool"] = i
            return None
    return "no monitoring tool id"


def resolve(item):
    """Observe *item*; None when it is a Python function, else what it is."""
    module_name, _, name = item.partition(":")
    if not module_name or not name:
        return "not a function id"
    module = sys.modules.get(module_name)
    if module is None:
        try:
            module = importlib.import_module(module_name)
        except BaseException:
            return "its module does not import"
    obj = getattr(module, name, _MISSING)
    if obj is _MISSING:
        return "no such name in its module"
    layers, fn = [], obj
    for _ in range(100):
        layers.append(fn)
        nxt = getattr(fn, "__wrapped__", _MISSING) if isinstance(fn, types.FunctionType) else _MISSING
        if nxt is _MISSING:
            break
        fn = nxt
    if not isinstance(fn, types.FunctionType):
        return _type_name(fn)
    STATE["codes"].setdefault(fn.__code__, []).append(item)
    STATE["wrappers"][item] = {f.__code__ for f in layers[:-1] if isinstance(f, types.FunctionType)}
    return None


def _writer(config):
    xml = config.stash.get(xml_key, None)
    if xml is None or not hasattr(xml, "add_global_property"):
        raise pytest.UsageError("_memory_calls needs pytest's junit writer (--junitxml)")
    return xml


@pytest.hookimpl(trylast=True)
def pytest_configure(config):
    xml = _writer(config)
    path = os.environ.get(OUT_ENV) or ""
    try:
        targets = json.loads(os.environ.get(TARGETS_ENV) or "")
    except ValueError:
        targets = None
    if not path or not isinstance(targets, list) or not all(isinstance(t, str) for t in targets):
        raise pytest.UsageError("_memory_calls needs MEMORY_CALLS_OUT and a JSON list in MEMORY_CALLS_TARGETS")
    STATE["targets"] = targets
    STATE["out"] = Out(path)
    STATE["unobservable"] = claim_tool()
    STATE["out"].write({"event": "start", "targets": targets, "unobservable": STATE["unobservable"]})
    sys.addaudithook(audit)
    xml.add_global_property(LOADED, "1")


@pytest.hookimpl(wrapper=True)
def pytest_make_collect_report(collector):
    before = STATE["collecting"]
    if isinstance(collector, pytest.Module):
        STATE["collecting"] = _rel(str(collector.path))
    try:
        return (yield)
    finally:
        STATE["collecting"] = before


def pytest_collection_finish(session):
    for item in STATE["targets"]:
        why = resolve(item)
        if why is not None and len(STATE["unresolved"]) < MAX_UNRESOLVED:
            STATE["unresolved"].append(f"{item} ({why})")
    tool = STATE["tool"]
    if tool is None:
        return
    mon = sys.monitoring
    ev = mon.events
    mon.register_callback(tool, ev.PY_START, on_start)
    mon.register_callback(tool, ev.PY_RETURN, on_return)
    mon.register_callback(tool, ev.PY_YIELD, on_return)
    mon.register_callback(tool, ev.PY_UNWIND, on_unwind)
    for code in STATE["codes"]:
        mon.set_local_events(tool, code, ev.PY_START | ev.PY_RETURN | ev.PY_YIELD)
    mon.set_events(tool, ev.PY_UNWIND)


@pytest.hookimpl(wrapper=True)
def pytest_runtest_protocol(item, nextitem):
    STATE["node"] = item
    try:
        return (yield)
    finally:
        STATE["node"] = None


@pytest.hookimpl(trylast=True)
def pytest_sessionfinish(session, exitstatus):
    if STATE["ended"] or STATE["out"] is None:
        return
    STATE["ended"] = True
    tool = STATE["tool"]
    if tool is not None:
        mon = sys.monitoring
        mon.set_events(tool, 0)
        for code in STATE["codes"]:
            mon.set_local_events(tool, code, 0)
        mon.free_tool_id(tool)
    STATE["dropped"] += len(STATE["pending"])
    STATE["pending"].clear()
    STATE["out"].write(
        {
            "event": "end",
            "calls": STATE["calls"],
            "dropped": STATE["dropped"],
            "unresolved": STATE["unresolved"],
            "outside": STATE["outside"],
            "unattributed_opens": STATE["unattributed_opens"],
        },
        final=True,
    )
'''


def unresolved_kind(entry: str) -> tuple[str, str]:
    """``(item, what it is)`` of one :attr:`CallRecord.unresolved` entry (``<item> (<what>)``)."""
    item, sep, rest = entry.partition(" (")
    return (item, rest[:-1]) if sep and rest.endswith(")") else (entry, "not observed")


def plugin_files(dest: Path) -> None:
    """Write the plugin module into *dest* (a fresh host directory mounted at :data:`CALLS_DIR`)."""
    (Path(dest) / f"{CALLS_MODULE}.py").write_text(CALLS_SOURCE)


def calls_env(targets: list[str]) -> dict[str, str]:
    """The box environment that tells the plugin which functions to observe and where to write."""
    return {TARGETS_ENV: json.dumps(sorted(set(targets))), OUT_ENV: CALLS_OUT}
