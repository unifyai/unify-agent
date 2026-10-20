"""Tests the writer states before it runs them (``tools.test``, ``UNIFY_MEMORY_WRITER_TESTS``).

Running code is evidence the writer asks for, never an identity rule the harness applies. The writer says what it
expects first (:class:`Expectation`); the harness writes that to the pass's ``tests.jsonl`` with a ``test_id``
before anything runs, then runs the item confined and appends the result (:class:`TestResult`). An expectation is
never edited after the run: a second call with another expectation is another test, and both rows stay. Every result
is evidence on the item's label (:mod:`.item_label`); it is never used to group, gate or refuse.

**Modes.**

* ``value``: the item runs on a recorded input the writer names (``on``): the episode's state (``<episode>``), its
  request or an observation as the request object (``<episode>/request``, ``<episode>/observation:<n>``), an action's
  recorded response (``<episode>/action:<n>``) or a cell's recorded output (``<episode>/cell:<n>``). Its result is
  compared by the digest of its canonical JSON.
* ``replay``: the item runs on the episode's state against :class:`.replay_box.ReplayEnv`, built from the episode's own
  recorded actions: every call, whatever its effect (``read``, ``write`` or ``unknown``), is answered only by an
  identical recorded call; writes and unknown-effect calls are recorded and compared, never executed; a call with no
  recording raises ``ReplayMiss``. The box holds no real channel object. Refused until the replay stand-in's
  confinement review is recorded (:data:`REPLAY_REVIEWED`).

**Where it runs.** In a box like the writer's own (:func:`.sandbox_run.run_confined`: no network, an empty
environment): a snapshot of the writer's library copy read-only at ``/memory``, the case read-only at ``/case``, only
``/out`` writable. Outer bounds, not steering: :data:`TEST_S` per test, the box's rlimits, :data:`MAX_REPLAY_CALLS`
replayed calls and :data:`MAX_CASE_BYTES` of recorded input per test.

**Outcomes** (:data:`OUTCOMES`), in this order: ``timeout`` (the box ran out of time), ``not_run`` (it gave no result),
``replay_miss`` (a replayed call the episode never made, to a channel and method it never called), then the
expectation: ``as_stated``, ``not_as_stated`` (the detail names the first difference, a differing call by its channel,
method and argument), or ``error`` (the item raised where the expectation did not say it would).
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
from collections.abc import Callable, Iterable, Mapping
from dataclasses import asdict, dataclass
from decimal import Decimal
from pathlib import Path
from typing import Any, Literal, get_args

from .fixtures import canonical
from .request_value import REQUEST_CLASS, request_data
from .sandbox_run import PYTHON, SandboxResult, run_confined

#: Whether the replay stand-in's confinement review (G2(a)) is recorded OK. Flipped only by the reviewed commit; until
#: then ``mode="replay"`` is refused with what to do instead.
REPLAY_REVIEWED = False

Mode = Literal["value", "replay"]
ExpectKind = Literal["returns", "issues_calls", "raises", "refuses"]
Outcome = Literal["as_stated", "not_as_stated", "error", "timeout", "replay_miss", "not_run"]
Effect = Literal["read", "write", "unknown"]
MODES: tuple[str, ...] = get_args(Mode)
EXPECT_KINDS: tuple[str, ...] = get_args(ExpectKind)
OUTCOMES: tuple[str, ...] = get_args(Outcome)
EFFECTS: tuple[str, ...] = get_args(Effect)
TESTS_FILE = "tests.jsonl"
#: One test's wall bound (an outer bound, not a quota).
TEST_S = 30.0
#: Calls one replay serves at most.
MAX_REPLAY_CALLS = 2000
#: The recorded input handed into one test's box, at most.
MAX_CASE_BYTES = 64 * 1024**2
MAX_RESULT_BYTES = 8 * 1024**2
#: Issued calls a result keeps (each call's arguments as their digest and clipped JSON).
MAX_LISTED_CALLS = 2000
REPLAY_BOX = Path(__file__).resolve().with_name("replay_box.py")


class WriterTestRefused(ValueError):
    """A ``tools.test`` call that is not a test (nothing was recorded or run); the text says what was given, what is
    expected and one correct call."""


def args_digest(args: Iterable[Any] = (), kwargs: Mapping[str, Any] | None = None) -> str:
    """The first 16 hex digits of the sha256 of a call's arguments as canonical JSON (``[args, kwargs]``)."""
    doc = [json.loads(json.dumps(list(args), default=str)), json.loads(json.dumps(dict(kwargs or {}), default=str))]
    return hashlib.sha256(canonical(doc).encode("utf-8", "surrogatepass")).hexdigest()[:16]


def value_digest(value: Any) -> str:
    return hashlib.sha256(canonical(value).encode("utf-8", "surrogatepass")).hexdigest()[:16]


@dataclass(frozen=True)
class CallSpec:
    """One call an item is expected to issue: its channel and method, and optionally its arguments' digest
    (:func:`args_digest`) and its effect (None: not stated, any matches)."""

    channel: str
    method: str
    args_digest: str | None = None
    effect: Effect | None = None

    @classmethod
    def of(cls, raw: Any) -> "CallSpec":
        """A CallSpec from a dict the writer gives: ``channel``, ``method`` and either ``args_digest`` or the
        arguments themselves (``args``, ``kwargs``), and optionally ``effect``."""
        if isinstance(raw, CallSpec):
            return raw
        if not isinstance(raw, Mapping):
            raise WriterTestRefused(
                f"each expected call is a dict with channel and method (and args/kwargs or args_digest), given "
                f"{type(raw).__name__}; for example "
                "{'channel': 'files', 'method': 'write', 'args': ['out/a.txt', 'x'], 'effect': 'write'}",
            )
        channel, method = raw.get("channel"), raw.get("method")
        if not (isinstance(channel, str) and channel and isinstance(method, str) and method):
            raise WriterTestRefused("an expected call names its channel and method, each a non-empty string")
        effect = raw.get("effect")
        if effect is not None and effect not in EFFECTS:
            raise WriterTestRefused(f"an expected call's effect is one of {', '.join(EFFECTS)} or left out, given {effect!r}")
        digest = raw.get("args_digest")
        if digest is None and ("args" in raw or "kwargs" in raw):
            digest = args_digest(raw.get("args") or [], raw.get("kwargs") or {})
        if digest is not None and not isinstance(digest, str):
            raise WriterTestRefused("args_digest is the 16-hex digest of the call's arguments, or left out")
        return cls(channel, method, digest, effect)


@dataclass(frozen=True)
class Expectation:
    """What the writer states before a test runs."""

    kind: ExpectKind
    value: object | None = None  # for "returns": compared by canonical-JSON digest
    calls: tuple[CallSpec, ...] = ()  # for "issues_calls": channel, method, args digest, effect
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
                "{'kind': 'issues_calls', 'calls': [{'channel': 'files', 'method': 'write', 'args': ['a.txt', 'x']}]}",
            )
        if kind == "returns" and "value" not in raw:
            raise WriterTestRefused("a 'returns' expectation states the value: {'kind': 'returns', 'value': ...}")
        if kind == "issues_calls" and not isinstance(raw.get("calls"), list):
            raise WriterTestRefused(
                "an 'issues_calls' expectation lists the calls in order: {'kind': 'issues_calls', 'calls': [...]} "
                "(an empty list states that no call is issued)",
            )
        if kind == "raises" and not (isinstance(raw.get("exception"), str) and raw["exception"].isidentifier()):
            raise WriterTestRefused("a 'raises' expectation names the class: {'kind': 'raises', 'exception': 'KeyError'}")
        return cls(
            kind,
            json.loads(json.dumps(raw.get("value"), default=str)) if kind == "returns" else None,
            tuple(CallSpec.of(c) for c in raw.get("calls") or []) if kind == "issues_calls" else (),
            raw.get("exception") if kind == "raises" else None,
        )

    def row(self) -> dict:
        return {
            "kind": self.kind,
            "value": self.value,
            "calls": [asdict(c) for c in self.calls],
            "exception": self.exception,
        }


@dataclass(frozen=True)
class TestResult:
    test_id: str
    item: str
    on: str  # "<episode>/<part>" or "<episode>"
    mode: Mode
    stated: Expectation  # recorded BEFORE the run
    outcome: Outcome
    detail: str  # e.g. "issued 6 recorded calls; 1 write differed: env.files.write arg 0"
    seconds: str

    def row(self) -> dict:
        return {
            "test_id": self.test_id,
            "item": self.item,
            "on": self.on,
            "mode": self.mode,
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
            "tools.test('memory.files.report:write_report', '<episode>', {'kind': 'issues_calls', 'calls': [...]}, "
            "'replay')",
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
from replay_box import ReplayEnv, ReplayMiss, State, TooManyCalls
""" + REQUEST_CLASS + r"""

def canon(v):
    return json.dumps(v, sort_keys=True, ensure_ascii=False, separators=(",", ":"), default=str)


with open("/case/case.json", encoding="utf-8") as f:
    case = json.load(f)
out = {}
env = None
try:
    fn = getattr(importlib.import_module(case["module"]), case["function"])
    if case["mode"] == "value":
        doc = case["input"]
        if doc["type"] == "state":
            args = (State(doc["messages"], doc["data"]),)
        elif doc["type"] == "request":
            args = (Request(doc["text"], doc["data"]),)
        else:
            args = (copy.deepcopy(doc["value"]),)
    else:
        env = ReplayEnv(case["actions"], case["max_calls"])
        state = State(case["messages"], case["data"])
        args = tuple(env if p == "env" else state for p in case["bind"])
    result = fn(*args)
    out["outcome"] = "returned"
    out["result"] = canon(result)
except BaseException as exc:
    out["outcome"] = "refused" if any(c.__name__ == "MemoryInputError" for c in type(exc).__mro__) else "error"
    out["error_class"] = type(exc).__name__
    out["replay_miss"] = isinstance(exc, ReplayMiss)
finally:
    if env is not None:
        out["calls"] = env.stats()
        out["issued"] = [
            {"channel": c["channel"], "method": c["method"], "args": c["args"], "kwargs": c["kwargs"],
             "outcome": c["outcome"], "effect": c["effect"], "row": c["row"]}
            for c in env.calls()[: case["listed"]]
        ]
        out["recorded"] = env.recorded()
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


def _first_difference(call: Mapping[str, Any], recorded: list[dict]) -> str:
    """Which argument of an issued *call* differs from the first recorded call of its channel and method."""
    same = [r for r in recorded if r.get("method") == call.get("method")]
    if not same:
        return ""
    r = same[0]
    a, b = list(call.get("args") or []), list(r.get("args") or [])
    for i in range(max(len(a), len(b))):
        if i >= len(a) or i >= len(b) or canonical(a[i]) != canonical(b[i]):
            return f" arg {i}"
    ka, kb = dict(call.get("kwargs") or {}), dict(r.get("kwargs") or {})
    for k in sorted(set(ka) | set(kb)):
        if k not in ka or k not in kb or canonical(ka[k]) != canonical(kb[k]):
            return f" keyword {k}"
    return ""


def _call_name(c: Mapping[str, Any]) -> str:
    return f"env.{c.get('channel')}.{c.get('method')}"


def judge(expect: Expectation, run: Mapping[str, Any], mode: str) -> tuple[str, str]:
    """``(outcome, detail)`` of one test from its box's *run* (module docstring: the order of outcomes)."""
    if run.get("timed_out"):
        return "timeout", f"the test did not finish within {TEST_S:g} s"
    if run.get("outcome") not in ("returned", "refused", "error"):
        return "not_run", str(run.get("why") or "the box gave no result")
    issued = [c for c in run.get("issued") or [] if isinstance(c, dict)]
    recorded = [c for c in run.get("recorded") or [] if isinstance(c, dict)]
    stats = run.get("calls") if isinstance(run.get("calls"), dict) else {}
    differed = [c for c in issued if c.get("outcome") == "differed" or (c.get("outcome") == "repeat" and c.get("effect") != "read")]
    summary = ""
    if mode == "replay":
        summary = (
            f"issued {len(issued)} call{'' if len(issued) == 1 else 's'} ({int(stats.get('matched') or 0)} answered "
            f"from the record)"
        )
        if differed:
            first = differed[0]
            what = "write" if first.get("effect") == "write" else "call"
            summary += f"; {len(differed)} differed: {_call_name(first)}{_first_difference(first, recorded)}"
            if first.get("outcome") == "repeat":
                summary += f" (a repeated {what})"
        missing = int(stats.get("missing") or 0)
        if missing:
            summary += f"; {missing} recorded call{'' if missing == 1 else 's'} not issued"
    unrecorded = [c for c in issued if c.get("outcome") == "unrecorded"]
    if unrecorded:
        return "replay_miss", f"{_call_name(unrecorded[0])} was never recorded in this episode; " + summary
    outcome, error = run.get("outcome"), _cls(run.get("error_class")) if run.get("outcome") != "returned" else None
    kind = expect.kind
    if kind == "returns":
        if outcome == "returned":
            got = run.get("result")
            if value_digest(got) == value_digest(expect.value):
                return "as_stated", (summary + "; " if summary else "") + "returned the stated value"
            return "not_as_stated", (summary + "; " if summary else "") + (
                f"returned a value with digest {value_digest(got)}, stated {value_digest(expect.value)}"
            )
        if differed:
            return "not_as_stated", summary + f"; then raised {error}"
        return "error", (summary + "; " if summary else "") + f"raised {error}"
    if kind == "issues_calls":
        got = [(c.get("channel"), c.get("method"), args_digest(c.get("args") or [], c.get("kwargs") or {}),
                c.get("effect")) for c in issued]  # fmt: skip
        for i in range(max(len(got), len(expect.calls))):
            if i >= len(got):
                return "not_as_stated", summary + f"; call {i} (env.{expect.calls[i].channel}.{expect.calls[i].method}) was not issued"
            if i >= len(expect.calls):
                return "not_as_stated", summary + f"; call {i} ({_call_name(issued[i])}) was issued and not stated"
            want, (ch, me, dig, eff) = expect.calls[i], got[i]
            if (ch, me) != (want.channel, want.method):
                return "not_as_stated", summary + f"; call {i} was env.{ch}.{me}, stated env.{want.channel}.{want.method}"
            if want.args_digest is not None and dig != want.args_digest:
                return "not_as_stated", summary + f"; call {i} ({_call_name(issued[i])}) had other arguments" + (
                    _first_difference(issued[i], recorded)
                )
            if want.effect is not None and eff != want.effect:
                return "not_as_stated", summary + f"; call {i} ({_call_name(issued[i])}) was recorded {eff}, stated {want.effect}"
        if outcome == "returned":
            return "as_stated", summary + "; every stated call issued, in order"
        return "error", summary + f"; then raised {error}"
    if kind == "raises":
        if outcome == "returned":
            return "not_as_stated", (summary + "; " if summary else "") + "returned; stated it raises " + str(expect.exception)
        if error == expect.exception:
            return "as_stated", (summary + "; " if summary else "") + f"raised {error}"
        return "not_as_stated", (summary + "; " if summary else "") + f"raised {error}, stated {expect.exception}"
    # refuses
    if outcome == "refused":
        return "as_stated", (summary + "; " if summary else "") + "refused its input (MemoryInputError)"
    return "not_as_stated", (summary + "; " if summary else "") + (
        "returned; stated it refuses its input" if outcome == "returned" else f"raised {error}, stated a refusal"
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
                TestResult(r["test_id"], str(s.get("item")), str(s.get("on")), s.get("mode"), exp, r.get("outcome"),
                           str(r.get("detail") or ""), str(r.get("seconds") or "0")),
            )  # fmt: skip
    return out


@dataclass
class TestContext:
    """What one pass's tests need: where they are recorded, the writer's library copy, the episodes and the modes the
    run offers."""

    pass_id: str
    tests_path: Path
    library: Path
    episode: Callable[[str], Any]
    modes: frozenset[str] = frozenset({"value"})
    work: Path | None = None
    runner: Callable[..., SandboxResult] = run_confined
    python: Path = PYTHON
    snapshot: Callable[[Path, Path], None] | None = None
    clock: Callable[[], float] = time.monotonic


def run_test(ctx: TestContext, item: Any, on: Any, expect: Any, mode: Any = "value") -> TestResult:
    """One stated test (module docstring). Raises :class:`WriterTestRefused` (nothing recorded, nothing run) for a call
    that is not a test: an unknown mode, a mode the run does not offer, replay before its review, a note, an item or
    input that cannot be found, or an expectation that says nothing checkable."""
    if mode not in MODES:
        raise WriterTestRefused(f"mode is 'value' or 'replay', given {mode!r}")
    if mode not in ctx.modes:
        raise WriterTestRefused(
            f"this run offers {', '.join(sorted(ctx.modes)) or 'no'} tests; mode {mode!r} is not among them",
        )
    if mode == "replay" and not REPLAY_REVIEWED:
        raise WriterTestRefused(
            "replay tests are not available yet: the replay stand-in awaits its confinement review. Test by value on "
            "a recorded input (mode='value'), or judge the item by reading its episodes; say which on its label",
        )
    stated = Expectation.of(expect)
    module, function, params = _function_target(str(item), ctx.library)
    eid, part = _split_on(on)
    ep = ctx.episode(eid)
    if ep is None:
        raise WriterTestRefused(f"no recorded episode {eid[:80]!r}; on names one: '<episode>' or '<episode>/<part>'")
    case: dict = {"module": module, "function": function, "mode": mode, "listed": MAX_LISTED_CALLS}
    if mode == "value":
        if len(params) != 1:
            raise WriterTestRefused(
                f"{item} takes {len(params)} positional arguments ({', '.join(params)}); a value test hands it one "
                "recorded input: test an item that takes the state and the environment with mode='replay'",
            )
        try:
            case["input"] = _input_doc(ep, part)
        except (LookupError, StopIteration, ValueError):
            from .writer_tools import valid_names

            raise WriterTestRefused(f"{on}: no such part; {eid}'s parts are {valid_names(ep)}") from None
    else:
        if part is not None:
            raise WriterTestRefused(f"a replay test runs on a whole episode's state and actions: on={eid!r}, not {on!r}")
        bind = _bind(params, str(item))
        from . import capture_trace as ct

        msgs = ct.messages_of(ep)
        case.update(
            bind=bind,
            messages=msgs,
            data=ct.state_data(msgs),
            actions=ct.served_actions(ep),
            max_calls=MAX_REPLAY_CALLS,
        )
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
         "on": str(on), "mode": mode, "stated": stated.row(), "at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())},
    )  # fmt: skip
    t0 = ctx.clock()
    run = _run_box(ctx, data)
    outcome, detail = judge(stated, run, mode)
    seconds = str(Decimal(str(max(0.0, ctx.clock() - t0))).quantize(Decimal("0.001")))
    result = TestResult(test_id, str(item), str(on), mode, stated, outcome, detail, seconds)
    _append(
        ctx.tests_path,
        {"v": 1, "phase": "result", "test_id": test_id, "outcome": outcome, "detail": detail, "seconds": seconds,
         "calls": run.get("calls"), "effects": issued_effects(run),
         "at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())},
    )  # fmt: skip
    return result


def issued_effects(run: Mapping[str, Any]) -> dict:
    """How many issued calls a replay answered from a recording of each effect (the label's ``seen_in_tests``)."""
    out = {"read": 0, "write": 0, "unknown": 0}
    for c in run.get("issued") or []:
        if isinstance(c, dict) and c.get("effect") in out and c.get("outcome") in ("served", "repeat"):
            out[c["effect"]] += 1
    return out


def _bind(params: list[str], item: str) -> list[str]:
    """How a replay test hands the item its arguments: by name (``state``, ``env``), else ``(state, env)`` by
    position."""
    if params and all(p in ("state", "env") for p in params):
        return list(params)
    if len(params) == 2:
        return ["state", "env"]
    raise WriterTestRefused(
        f"{item} takes ({', '.join(params)}); a replay test hands it the episode's state and its recorded tools: "
        "an item taking (state, env), or parameters named state and env",
    )


def _run_box(ctx: TestContext, data: bytes) -> dict:
    work = Path(ctx.work) if ctx.work is not None else Path(tempfile.mkdtemp(prefix="writer-test-"))
    work.mkdir(parents=True, exist_ok=True)
    case = Path(tempfile.mkdtemp(prefix="case-", dir=work))
    out = Path(tempfile.mkdtemp(prefix="out-", dir=work))
    lib = Path(tempfile.mkdtemp(prefix="memory-", dir=work))
    try:
        (case / "lib").mkdir()
        shutil.copyfile(REPLAY_BOX, case / "lib" / "replay_box.py")
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
