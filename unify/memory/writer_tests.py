"""Tests the writer states before it runs them (``tools.test``).

Running code is evidence the writer asks for, never an identity rule the harness applies. The writer says what it
expects first (:class:`Expectation`); the harness writes that to the pass's ``tests.jsonl`` with a ``test_id``
before anything runs, then runs the item confined and appends the result (:class:`TestResult`). An expectation is
never edited after the run: a second call with another expectation is another test, and both rows stay. Every result
is evidence on the item's label (:mod:`.item_label`); it is never used to group, gate or refuse.

**What runs.** The item runs on a recorded input the writer names (``on``): the episode's state (``<episode>``), its
request or an observation as the request object (``<episode>/request``, ``<episode>/observation:<n>``), an action's
recorded response (``<episode>/action:<n>``) or a cell's recorded output (``<episode>/cell:<n>``). Its result is
compared by the digest of its canonical JSON. Nothing the episode called is replayed: the item gets its input and
nothing else.

**Where it runs.** In a box like the writer's own (:func:`.sandbox_run.run_confined`: no network, an empty
environment): a snapshot of the writer's library copy read-only at ``/memory``, the case read-only at ``/case``, only
``/out`` writable. Outer bounds, not steering: :data:`TEST_S` per test, the box's rlimits and :data:`MAX_CASE_BYTES`
of recorded input per test.

**Outcomes** (:data:`OUTCOMES`), in this order: ``timeout`` (the box ran out of time), ``not_run`` (it gave no
result), then the expectation: ``as_stated``, ``not_as_stated``, or ``error`` (the item raised where the expectation
did not say it would).
"""

from __future__ import annotations

import ast
import hashlib
import json
import os
import shutil
import stat
import tempfile
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Any, Literal, get_args

from .fixtures import canonical
from .request_value import REQUEST_CLASS, request_data
from .sandbox_run import PYTHON, SandboxResult, run_confined

ExpectKind = Literal["returns", "raises", "refuses"]
Outcome = Literal["as_stated", "not_as_stated", "error", "timeout", "not_run"]
EXPECT_KINDS: tuple[str, ...] = get_args(ExpectKind)
OUTCOMES: tuple[str, ...] = get_args(Outcome)
TESTS_FILE = "tests.jsonl"
#: One test's wall bound (an outer bound, not a quota).
TEST_S = 30.0
#: The recorded input handed into one test's box, at most.
MAX_CASE_BYTES = 64 * 1024**2
MAX_RESULT_BYTES = 8 * 1024**2
#: The episode's state as the box gives it to an item (:class:`.box_state.State`), copied into each case.
BOX_STATE = Path(__file__).resolve().with_name("box_state.py")


class WriterTestRefused(ValueError):
    """A ``tools.test`` call that is not a test (nothing was recorded or run); the text says what was given, what is
    expected and one correct call."""


def value_digest(value: Any) -> str:
    return hashlib.sha256(canonical(value).encode("utf-8", "surrogatepass")).hexdigest()[:16]


@dataclass(frozen=True)
class Expectation:
    """What the writer states before a test runs."""

    kind: ExpectKind
    value: object | None = None  # for "returns": compared by canonical-JSON digest
    exception: str | None = None  # for "raises": the class name

    @classmethod
    def of(cls, raw: Any) -> "Expectation":
        if isinstance(raw, Expectation):
            return raw
        if not isinstance(raw, Mapping):
            raise WriterTestRefused(
                f"expect is a dict, given {type(raw).__name__}; for example {{'kind': 'returns', 'value': [6, 9]}}",
            )
        kind = raw.get("kind")
        if kind not in EXPECT_KINDS:
            raise WriterTestRefused(
                f"expect['kind'] is one of {', '.join(EXPECT_KINDS)}, given {kind!r}; for example "
                "{'kind': 'returns', 'value': [6, 9]}",
            )
        if kind == "returns" and "value" not in raw:
            raise WriterTestRefused("a 'returns' expectation states the value: {'kind': 'returns', 'value': ...}")
        if kind == "raises" and not (isinstance(raw.get("exception"), str) and raw["exception"].isidentifier()):
            raise WriterTestRefused("a 'raises' expectation names the class: {'kind': 'raises', 'exception': 'KeyError'}")
        return cls(
            kind,
            json.loads(json.dumps(raw.get("value"), default=str)) if kind == "returns" else None,
            raw.get("exception") if kind == "raises" else None,
        )

    def row(self) -> dict:
        return {
            "kind": self.kind,
            "value": self.value,
            "exception": self.exception,
        }


@dataclass(frozen=True)
class TestResult:
    test_id: str
    item: str
    on: str  # "<episode>/<part>" or "<episode>"
    stated: Expectation  # recorded BEFORE the run
    outcome: Outcome
    detail: str  # e.g. "returned a value with digest 3f1c..., stated 9a07..."
    seconds: str

    def row(self) -> dict:
        return {
            "test_id": self.test_id,
            "item": self.item,
            "on": self.on,
            "stated": self.stated.row(),
            "outcome": self.outcome,
            "detail": self.detail,
            "seconds": self.seconds,
        }


# --- the target and its input ---------------------------------------------------------------------------------------------


def _function_target(item: str, library: Path) -> tuple[str, str, list[str]]:
    """``(module, function, positional parameters)`` of a function item in the writer's copy, parsed (never run)."""
    from .layout import FUNCTION_ID, item_path

    m = FUNCTION_ID.match(item or "")
    if m is None:
        if isinstance(item, str) and item.startswith("notes/"):
            raise WriterTestRefused(f"{item} is a note: a note is tested by reading it, not run")
        raise WriterTestRefused(
            f"item is a function's id, memory.<package>.<module>:<name>, given {str(item)[:80]!r}; for example "
            "tools.test('memory.files.report:total', '<episode>/cell:2', {'kind': 'returns', 'value': 6})",
        )
    path = Path(library) / item_path(item)
    try:
        if not stat.S_ISREG(os.lstat(path).st_mode):
            raise OSError
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except (OSError, SyntaxError, ValueError, UnicodeDecodeError):
        raise WriterTestRefused(f"{item}: /memory/{item_path(item)} cannot be read and parsed; write it first") from None
    name = m.group(3)
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
            a = node.args
            return f"memory.{m.group(1)}.{m.group(2)}", name, [x.arg for x in (*a.posonlyargs, *a.args)]
    raise WriterTestRefused(f"{item}: /memory/{item_path(item)} defines no function {name}")


def _split_on(on: Any) -> tuple[str, str | None]:
    if not isinstance(on, str) or not on.strip():
        raise WriterTestRefused(
            "on names a recorded input: '<episode>' (its state), '<episode>/request', '<episode>/observation:<n>', "
            "'<episode>/action:<n>' or '<episode>/cell:<n>'",
        )
    eid, sep, part = on.strip().partition("/")
    return eid, (part if sep else None)


def _input_doc(ep: Any, part: str | None) -> dict:
    """The recorded input a value test hands its item (module docstring)."""
    from . import capture_trace as ct

    msgs = ct.messages_of(ep)
    if part is None:
        return {"type": "state", "messages": msgs, "data": ct.state_data(msgs)}
    if part == "request":
        text = msgs[0] if msgs else ""
        return {"type": "request", "text": text, "data": request_data(text)}
    kind, _, idx = part.partition(":")
    if not idx.isdigit():
        raise LookupError(part)
    i = int(idx)
    if kind == "observation":
        text = msgs[i + 1]
        return {"type": "request", "text": text, "data": request_data(text)}
    if kind == "action":
        return {"type": "json", "value": ep.actions[i].response}
    if kind == "cell":
        cell = next(c for c in ep.cells if int(c.index) == i)
        return {"type": "text", "value": cell.output}
    raise LookupError(part)


# --- the box ------------------------------------------------------------------------------------------------------------

_RUN = r"""
import copy, importlib, json, sys
sys.dont_write_bytecode = True
sys.path[:0] = ["/memory", "/case/lib"]
from box_state import State
""" + REQUEST_CLASS + r"""

def canon(v):
    return json.dumps(v, sort_keys=True, ensure_ascii=False, separators=(",", ":"), default=str)


with open("/case/case.json", encoding="utf-8") as f:
    case = json.load(f)
out = {}
try:
    fn = getattr(importlib.import_module(case["module"]), case["function"])
    doc = case["input"]
    if doc["type"] == "state":
        args = (State(doc["messages"], doc["data"]),)
    elif doc["type"] == "request":
        args = (Request(doc["text"], doc["data"]),)
    else:
        args = (copy.deepcopy(doc["value"]),)
    result = fn(*args)
    out["outcome"] = "returned"
    out["result"] = canon(result)
except BaseException as exc:
    out["outcome"] = "refused" if any(c.__name__ == "MemoryInputError" for c in type(exc).__mro__) else "error"
    out["error_class"] = type(exc).__name__
with open("/out/result.json", "w", encoding="utf-8") as f:
    json.dump(out, f, default=str)
"""


def _read_result(path: Path) -> dict | None:
    try:
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    except OSError:
        return None
    with os.fdopen(fd, "rb") as fh:
        if not stat.S_ISREG(os.fstat(fh.fileno()).st_mode):
            return None
        data = fh.read(MAX_RESULT_BYTES + 1)
    if len(data) > MAX_RESULT_BYTES:
        return None
    try:
        doc = json.loads(data.decode("utf-8", "replace"))
    except ValueError:
        return None
    return doc if isinstance(doc, dict) else None


def _cls(name: Any) -> str:
    return name if isinstance(name, str) and name.isidentifier() and len(name) <= 64 else "an exception"


# --- the comparison ------------------------------------------------------------------------------------------------------


def judge(expect: Expectation, run: Mapping[str, Any]) -> tuple[str, str]:
    """``(outcome, detail)`` of one test from its box's *run* (module docstring: the order of outcomes)."""
    if run.get("timed_out"):
        return "timeout", f"the test did not finish within {TEST_S:g} s"
    if run.get("outcome") not in ("returned", "refused", "error"):
        return "not_run", str(run.get("why") or "the box gave no result")
    outcome, error = run.get("outcome"), _cls(run.get("error_class")) if run.get("outcome") != "returned" else None
    kind = expect.kind
    if kind == "returns":
        if outcome == "returned":
            got = run.get("result")
            if value_digest(got) == value_digest(expect.value):
                return "as_stated", "returned the stated value"
            return "not_as_stated", f"returned a value with digest {value_digest(got)}, stated {value_digest(expect.value)}"
        return "error", f"raised {error}"
    if kind == "raises":
        if outcome == "returned":
            return "not_as_stated", "returned; stated it raises " + str(expect.exception)
        if error == expect.exception:
            return "as_stated", f"raised {error}"
        return "not_as_stated", f"raised {error}, stated {expect.exception}"
    # refuses
    if outcome == "refused":
        return "as_stated", "refused its input (MemoryInputError)"
    return "not_as_stated", "returned; stated it refuses its input" if outcome == "returned" else (
        f"raised {error}, stated a refusal"
    )


# --- one test ------------------------------------------------------------------------------------------------------------


def _rows(path: Path) -> list[dict]:
    try:
        text = Path(path).read_text(encoding="utf-8")
    except OSError:
        return []
    out = []
    for line in text.splitlines():
        try:
            row = json.loads(line)
        except ValueError:
            continue
        if isinstance(row, dict):
            out.append(row)
    return out


def _append(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = (json.dumps(row, sort_keys=True, default=str) + "\n").encode("utf-8")
    fd = os.open(path, os.O_WRONLY | os.O_APPEND | os.O_CREAT | os.O_NOFOLLOW | os.O_CLOEXEC, 0o600)
    try:
        while data:
            data = data[os.write(fd, data) :]
        os.fsync(fd)
    finally:
        os.close(fd)


def results(path: Path | str) -> list[TestResult]:
    """Every finished test of a pass's ``tests.jsonl`` (its stated row joined to its result row)."""
    stated: dict[str, dict] = {}
    out = []
    for r in _rows(Path(path)):
        if r.get("phase") == "stated" and isinstance(r.get("test_id"), str):
            stated[r["test_id"]] = r
        elif r.get("phase") == "result" and r.get("test_id") in stated:
            s = stated[r["test_id"]]
            try:
                exp = Expectation.of(s.get("stated"))
            except WriterTestRefused:
                continue
            out.append(
                TestResult(r["test_id"], str(s.get("item")), str(s.get("on")), exp, r.get("outcome"),
                           str(r.get("detail") or ""), str(r.get("seconds") or "0")),
            )  # fmt: skip
    return out


@dataclass
class TestContext:
    """What one pass's tests need: where they are recorded, the writer's library copy and the episodes."""

    pass_id: str
    tests_path: Path
    library: Path
    episode: Callable[[str], Any]
    work: Path | None = None
    runner: Callable[..., SandboxResult] = run_confined
    python: Path = PYTHON
    snapshot: Callable[[Path, Path], None] | None = None
    clock: Callable[[], float] = time.monotonic


def run_test(ctx: TestContext, item: Any, on: Any, expect: Any) -> TestResult:
    """One stated test (module docstring). Raises :class:`WriterTestRefused` (nothing recorded, nothing run) for a call
    that is not a test: a note, an item or input that cannot be found, an item that does not take one input, or an
    expectation that says nothing checkable."""
    stated = Expectation.of(expect)
    module, function, params = _function_target(str(item), ctx.library)
    eid, part = _split_on(on)
    ep = ctx.episode(eid)
    if ep is None:
        raise WriterTestRefused(f"no recorded episode {eid[:80]!r}; on names one: '<episode>' or '<episode>/<part>'")
    if len(params) != 1:
        raise WriterTestRefused(
            f"{item} takes {len(params)} positional arguments ({', '.join(params)}); a test hands it one recorded "
            "input: judge an item of another kind by reading its episodes, and say so on its label",
        )
    case: dict = {"module": module, "function": function}
    try:
        case["input"] = _input_doc(ep, part)
    except (LookupError, StopIteration, ValueError):
        from .writer_tools import valid_names

        raise WriterTestRefused(f"{on}: no such part; {eid}'s parts are {valid_names(ep)}") from None
    data = json.dumps(case, ensure_ascii=False, default=str).encode("utf-8")
    if len(data) > MAX_CASE_BYTES:
        raise WriterTestRefused(f"{on}: the recorded input is over the test bound ({MAX_CASE_BYTES} bytes)")
    # the pass's tests are numbered in the order they are stated (one stated row each)
    n = len([r for r in _rows(ctx.tests_path) if r.get("phase") == "stated"])
    test_id = hashlib.sha256(f"{ctx.pass_id}\0{n}".encode()).hexdigest()[:16]
    # the expectation is on record before anything runs
    _append(
        ctx.tests_path,
        {"v": 1, "phase": "stated", "test_id": test_id, "n": n, "pass_id": ctx.pass_id, "item": str(item),
         "on": str(on), "stated": stated.row(), "at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())},
    )  # fmt: skip
    t0 = ctx.clock()
    run = _run_box(ctx, data)
    outcome, detail = judge(stated, run)
    seconds = str(Decimal(str(max(0.0, ctx.clock() - t0))).quantize(Decimal("0.001")))
    result = TestResult(test_id, str(item), str(on), stated, outcome, detail, seconds)
    _append(
        ctx.tests_path,
        {"v": 1, "phase": "result", "test_id": test_id, "outcome": outcome, "detail": detail, "seconds": seconds,
         "at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())},
    )  # fmt: skip
    return result


def _run_box(ctx: TestContext, data: bytes) -> dict:
    work = Path(ctx.work) if ctx.work is not None else Path(tempfile.mkdtemp(prefix="writer-test-"))
    work.mkdir(parents=True, exist_ok=True)
    case = Path(tempfile.mkdtemp(prefix="case-", dir=work))
    out = Path(tempfile.mkdtemp(prefix="out-", dir=work))
    lib = Path(tempfile.mkdtemp(prefix="memory-", dir=work))
    try:
        (case / "lib").mkdir()
        shutil.copyfile(BOX_STATE, case / "lib" / "box_state.py")
        (case / "case.json").write_bytes(data)
        (case / "run.py").write_text(_RUN, encoding="utf-8")
        # the writer's copy as it is now, so a cell still writing cannot change what is tested
        (ctx.snapshot or _mirror)(ctx.library, lib)
        res = ctx.runner(
            [str(ctx.python), "-I", "/case/run.py"],
            ro={case: "/case", lib: "/memory"},
            rw={out: "/out"},
            cwd="/out",
            timeout_s=TEST_S,
            env={},
        )
        if res.timed_out:
            return {"timed_out": True}
        doc = _read_result(out / "result.json")
        if doc is None:
            return {"outcome": "not run", "why": "the box gave no result"}
    finally:
        for d in (case, out, lib):
            shutil.rmtree(d, ignore_errors=True)
        if ctx.work is None:
            shutil.rmtree(work, ignore_errors=True)
    if isinstance(doc.get("result"), str):
        try:
            doc["result"] = json.loads(doc["result"])
        except ValueError:
            pass
    return doc


def _mirror(src: Path, dest: Path) -> None:
    """A copy of the library at *src* (regular files and directories only, never through a link)."""
    from .writer_pass import _mirror as mirror

    mirror(Path(src), Path(dest))
