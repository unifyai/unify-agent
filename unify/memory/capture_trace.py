"""The capture contract under ``UNIFY_MEMORY_CAPTURE_TRACE=on``: what the episode did, a value only when there is one.

A capture records what its episode did: a pointer list into the episode's own record (its request, observations,
cells and actions with their responses; :func:`trace_of`), never a copy. The procedure the fork wrote takes one of
three forms (:data:`FORMS`, read from its code by :func:`entry_point`, never run to find out):

==============  ============================================================================================
form            contract
==============  ============================================================================================
``value``       ``read_input(request) -> run(value)``: the whole job was in the first message
``state``       ``run(state)``: :class:`.replay_box.State` holds every message so far (``.request``, ``.messages``,
                ``.last``, ``.data``)
``state_env``   ``run(state, env)``: ``env`` is the episode's tool namespace, ``env.<channel>.<method>(...)``
                as the episode's actions name them (:class:`.replay_box.ReplayEnv` in the own check)
==============  ============================================================================================

**Rev 2: no own check runs.** A capture is never executed to classify, dedupe or rank it, so the capture step records
``own_check`` as :data:`NOT_RUN` and makes no entry from it. :func:`run_box` and :func:`own_check` stay, unused by the
capture step, for what they describe:

**The own check** (:func:`run_box`, :func:`own_check`) runs the procedure confined, as the harness's other runs of
model code are (no network, an empty environment, only ``/out`` writable): on the recorded request (``value``), on the
recorded :class:`State` (``state``), or on the recorded state against a replay of the episode's own recorded actions
(``state_env``), where every call is answered only by an identical recorded call and nothing is executed. It runs twice
(determinism) and records what happened as evidence (``own_check``), never as a failure of the capture:

* ``value_match``: ``matched`` or ``differed`` against the value the episode handed back, ``not_applicable`` when it
  handed none back (absence is a state of its own), ``unknown`` when the run gave nothing to compare;
* ``calls``: ``replay`` (with how many issued calls matched, differed, were missing or unrecorded), ``value`` (a form
  with no tools) or ``not_run``.

**The value** (:func:`value_of`): ``{"present": false}``, or ``{"present": true, "digest", "shape"}``. Its outcome comes
from the signals only (``positive``, ``contrary``, ``none``), never from the own check.

Nothing here keys on a task id, a stream position, a world or a word.
"""

from __future__ import annotations

import ast
import copy
import hashlib
import json
import shutil
import stat
import tempfile
from collections.abc import Callable, Iterable, Mapping
from pathlib import Path
from typing import Any, Literal, get_args

from .fixtures import canonical
from .request_value import REQUEST_CLASS, request_data
from .sandbox_run import PYTHON, SandboxResult, run_confined

Form = Literal["value", "state", "state_env"]
Outcome = Literal["positive", "contrary", "none"]
FORMS: tuple[str, ...] = get_args(Form)
OUTCOMES: tuple[str, ...] = get_args(Outcome)
#: The record version a trace capture stamps (``record.json`` and its ``captures-events.jsonl`` row).
VERSION = 2
#: One confined own check's wall bound (both runs), as the capture step's other runs.
RUN_S = 60.0
#: Calls one replay serves at most (an outer bound, not a quota): past it the run raises ``TooManyCalls``.
MAX_REPLAY_CALLS = 2000
#: The recorded messages and actions handed into one box, at most (a bound on the box's memory): past it the own
#: check does not run, and says why.
MAX_CASE_BYTES = 64 * 1024**2
#: Issued calls a box's result lists (each call's arguments clipped), for the evidence that names a differing call.
MAX_LISTED_CALLS = 200
LISTED_ARGS_CHARS = 400
MAX_RESULT_BYTES = 8 * 1024 * 1024
REPLAY_BOX = Path(__file__).resolve().with_name("replay_box.py")
PROCEDURE_FILE = "procedure.py"
#: What a capture's ``own_check`` records under rev 2: it was not run, and why.
NOT_RUN: dict = {"ran": False, "why": "rev2: captures are never executed to classify"}


# --- the procedure's form -----------------------------------------------------------------------------------------------


def _function(tree: ast.Module, name: str) -> ast.FunctionDef | ast.AsyncFunctionDef | None:
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
            return node
    return None


def _positional(node: ast.FunctionDef | ast.AsyncFunctionDef) -> list[str]:
    a = node.args
    return [x.arg for x in (*a.posonlyargs, *a.args)]


def entry_point(source: str) -> dict | None:
    """``{"name": "run", "params": [...], "form": ...}`` from the procedure's code (parsed, never run), or None when it
    has no ``run``: two or more positional parameters are ``state_env``; one, with a ``read_input``, ``value``; else
    ``state``."""
    try:
        tree = ast.parse(source)
    except (SyntaxError, ValueError):
        return None
    run = _function(tree, "run")
    if run is None:
        return None
    params = _positional(run)
    if len(params) >= 2:
        form = "state_env"
    elif _function(tree, "read_input") is not None:
        form = "value"
    else:
        form = "state"
    return {"name": "run", "params": params, "form": form}


def safety(source: str) -> str | None:
    """Why *source* must not be captured or run under the trace contract, or None: the capture step's checks
    (a key-shaped string, a path the gate refuses, its size, code that does not parse), then the form's own: a
    ``run``, and with one parameter and a ``read_input``, both."""
    from .captures import MAX_SOURCE_BYTES
    from .manifest import forbidden, unsafe_path
    from .redact import KEY_SHAPED

    if KEY_SHAPED.search(source):
        return "key-shaped string"
    if forbidden(PROCEDURE_FILE):
        return f"forbidden path {PROCEDURE_FILE}"
    bad = unsafe_path(f"env/captured/{PROCEDURE_FILE}")
    if bad is not None:
        return bad[1]
    if len(source.encode("utf-8", "surrogatepass")) > MAX_SOURCE_BYTES:
        return "larger than the procedure bound"
    try:
        ast.parse(source)
    except (SyntaxError, ValueError) as exc:
        return f"does not parse: {type(exc).__name__}"
    if entry_point(source) is None:
        return "defines no run"
    return None


# --- what the episode did ----------------------------------------------------------------------------------------------


def _ranged(kind: str, indexes: list[int]) -> list[str]:
    """``kind:a..b`` for each run of consecutive indexes, ``kind:i`` for a lone one."""
    out: list[str] = []
    if not indexes:
        return out
    start = prev = indexes[0]
    for i in indexes[1:] + [None]:  # type: ignore[list-item]
        if i is not None and i == prev + 1:
            prev = i
            continue
        out.append(f"{kind}:{start}" if start == prev else f"{kind}:{start}..{prev}")
        if i is not None:
            start = prev = i
    return out


def trace_of(ep: Any, episode: str | None = None) -> dict:
    """Pointers into the episode's record (the writer's part names), never a copy of it."""
    n_obs = max(0, len(getattr(ep, "request", None) or []) - 1)
    cells = sorted(int(c.index) for c in getattr(ep, "cells", None) or [])
    parts = ["request"]
    parts += _ranged("observation", list(range(n_obs)))
    parts += _ranged("cell", cells)
    parts += _ranged("action", list(range(len(getattr(ep, "actions", None) or []))))
    if getattr(ep, "worktree_diff", ""):
        parts.append("diff")
    return {"episode": episode or str(getattr(ep, "episode_id", "")), "parts": parts}


def effects_seen(ep: Any) -> dict:
    """How many of the episode's actions had each recorded effect."""
    out = {"read": 0, "write": 0, "unknown": 0}
    for a in getattr(ep, "actions", None) or []:
        effect = getattr(a, "effect", "unknown")
        out[effect if effect in out else "unknown"] += 1
    return out


def messages_of(ep: Any) -> list[str]:
    """Every user or environment message of the episode, in order (the first is the request)."""
    return [m if isinstance(m, str) else json.dumps(m, default=str) for m in getattr(ep, "request", None) or []]


def state_data(messages: Iterable[str]) -> list:
    """The JSON objects and arrays the messages contain, in order (the request's ``data`` rule for each)."""
    out: list = []
    for m in messages:
        out.extend(request_data(m))
    return out


def served_actions(ep: Any) -> list[dict]:
    """The episode's actions as the replay serves them (every recorded field it compares or answers with)."""
    rows = []
    for a in getattr(ep, "actions", None) or []:
        rows.append(
            {
                "channel": a.channel,
                "method": a.method,
                "args": list(a.args or []),
                "kwargs": dict(a.kwargs or {}),
                "response": a.response,
                "status": a.status,
                "effect": a.effect,
                "error": a.error,
                "kind": getattr(a, "kind", "tool"),
            },
        )
    return rows


def digest(value: Any) -> str:
    """The first 16 hex digits of the sha256 of *value*'s canonical JSON."""
    return hashlib.sha256(canonical(value).encode("utf-8", "surrogatepass")).hexdigest()[:16]


def value_of(ep: Any, printed: frozenset[str] = frozenset()) -> tuple[dict, Any]:
    """``({"present": false}, None)``, or ``({"present": true, "digest", "shape"}, value)`` for the value the episode
    handed back (:func:`.captures.handed_back`)."""
    from .captures import handed_back, shape

    found, value = handed_back(ep)
    if not found:
        return {"present": False}, None
    return {"present": True, "digest": digest(value), "shape": shape(value, printed)}, value


# --- the confined run ----------------------------------------------------------------------------------------------------

_RUN = r"""
import copy, importlib.util, json, sys
sys.dont_write_bytecode = True
sys.path.insert(0, "/case/lib")
from replay_box import ReplayEnv, ReplayMiss, State, TooManyCalls
""" + REQUEST_CLASS + r"""

def canon(v):
    return json.dumps(v, sort_keys=True, ensure_ascii=False, separators=(",", ":"), default=str)


def clip(calls, n, chars):
    out = []
    for c in calls[:n]:
        row = dict(c)
        for k in ("args", "kwargs"):
            text = canon(row.get(k))
            row[k] = text if len(text) <= chars else text[:chars] + "..."
        out.append(row)
    return out


with open("/case/case.json", encoding="utf-8") as f:
    case = json.load(f)
form = case["form"]
out, stage = {"form": form}, "import"


def listed(env):
    return [(c["channel"], c["method"], canon(c["args"]), canon(c["kwargs"]), c["outcome"]) for c in env.calls()]


try:
    spec = importlib.util.spec_from_file_location("procedure", "/case/procedure.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    if form == "value":
        stage = "read_input"
        value = mod.read_input(Request(case["request"], case["data"]))
        out["value"] = canon(value)
        base = lambda: copy.deepcopy(value)
    else:
        state = State(case["messages"], case["data"])
        base = lambda: copy.deepcopy(state)

    def args():
        if form == "state_env":
            return (base(), ReplayEnv(case["actions"], case["max_calls"]))
        return (base(),)

    if callable(getattr(mod, "applies", None)):
        stage = "applies"
        out["applies"] = bool(mod.applies(base()))
    stage = "run"
    first_args = args()
    try:
        first = mod.run(*first_args)
    finally:
        if form == "state_env":
            env1 = first_args[1]
            out["calls"] = env1.stats()
            out["issued"] = clip(env1.calls(), case["listed"], case["chars"])
            out["calls1"] = listed(env1)
    out["result"] = canon(first)
    second_args = args()
    second = mod.run(*second_args)
    out["result2"] = canon(second)
    if form == "state_env":
        out["calls2"] = listed(second_args[1])
    if callable(getattr(mod, "holds", None)):
        stage = "holds"
        out["holds"] = bool(mod.holds(base(), copy.deepcopy(first)))
    out["outcome"] = "returned"
except BaseException as exc:
    out["outcome"] = "refused" if any(c.__name__ == "MemoryInputError" for c in type(exc).__mro__) else "error"
    out["error_class"] = type(exc).__name__
    out["stage"] = stage
with open("/out/result.json", "w", encoding="utf-8") as f:
    json.dump(out, f, default=str)
"""


def _cls(name: Any) -> str:
    return name if isinstance(name, str) and name.isidentifier() and len(name) <= 64 else "an exception"


def _read_result(path: Path) -> dict | None:
    import os

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


def case_doc(form: str, ep: Any) -> dict:
    """What the box is given for *form*: the request and its data (``value``), every message and their data
    (``state``), and the episode's recorded actions too (``state_env``)."""
    msgs = messages_of(ep)
    doc: dict = {"form": form, "listed": MAX_LISTED_CALLS, "chars": LISTED_ARGS_CHARS, "max_calls": MAX_REPLAY_CALLS}
    if form == "value":
        text = msgs[0] if msgs else ""
        doc.update(request=text, data=request_data(text))
    else:
        doc.update(messages=msgs, data=state_data(msgs))
        if form == "state_env":
            doc["actions"] = served_actions(ep)
    return doc


def run_box(
    source: str,
    form: str,
    case: Mapping[str, Any],
    *,
    work: Path,
    runner: Callable[..., SandboxResult] = run_confined,
    python: Path = PYTHON,
    timeout_s: float = RUN_S,
) -> dict:
    """Run *source* in its *form* on *case* (:func:`case_doc`) in a confined box: the procedure, the case and the
    replay stand-in read-only at ``/case``, only ``/out`` writable, no network, an empty environment. Returns
    ``{outcome, value, result, deterministic, applies, holds, error_class, stage, calls, issued}``; ``outcome`` is
    ``returned``, ``refused``, ``error`` or ``not run`` (the case was over the bound, the box timed out or gave no
    result; ``why`` says which)."""
    if form not in FORMS:
        return {"outcome": "not run", "why": f"unknown form {form!r}"}
    data = json.dumps(case, ensure_ascii=False, default=str).encode("utf-8")
    if len(data) > MAX_CASE_BYTES:
        return {"outcome": "not run", "why": "the recorded messages and actions are over the replay bound"}
    work.mkdir(parents=True, exist_ok=True)
    box = Path(tempfile.mkdtemp(prefix="case-", dir=work))
    out = Path(tempfile.mkdtemp(prefix="out-", dir=work))
    try:
        (box / "lib").mkdir()
        shutil.copyfile(REPLAY_BOX, box / "lib" / "replay_box.py")
        (box / PROCEDURE_FILE).write_text(source, encoding="utf-8")
        (box / "case.json").write_bytes(data)
        (box / "run.py").write_text(_RUN, encoding="utf-8")
        res = runner(
            [str(python), "-I", "/case/run.py"],
            ro={box: "/case"},
            rw={out: "/out"},
            cwd="/out",
            timeout_s=timeout_s,
            env={},
        )
        if res.timed_out:
            return {"outcome": "not run", "why": f"it did not finish within {timeout_s:g} s"}
        doc = _read_result(out / "result.json")
        if doc is None:
            return {"outcome": "not run", "why": "it could not be run (no result)"}
    finally:
        shutil.rmtree(box, ignore_errors=True)
        shutil.rmtree(out, ignore_errors=True)
    outcome = doc.get("outcome")
    if outcome not in ("returned", "refused", "error"):
        return {"outcome": "not run", "why": "it could not be run (no outcome)"}
    row: dict = {"outcome": outcome, "form": form}
    for key in ("value", "result", "result2"):
        if isinstance(doc.get(key), str):
            try:
                row[key] = json.loads(doc[key])
            except ValueError:
                row[key] = doc[key]
    if outcome == "returned":
        same = doc.get("result") == doc.get("result2")
        if form == "state_env":
            same = same and doc.get("calls1") == doc.get("calls2")
        row["deterministic"] = same
    row.pop("result2", None)
    for key in ("applies", "holds"):
        if isinstance(doc.get(key), bool):
            row[key] = doc[key]
    if isinstance(doc.get("calls"), dict):
        row["calls"] = {k: int(doc["calls"].get(k) or 0) for k in ("matched", "differed", "missing", "unrecorded", "repeats")}
    if isinstance(doc.get("issued"), list):
        row["issued"] = [c for c in doc["issued"] if isinstance(c, dict)][:MAX_LISTED_CALLS]
    if outcome != "returned":
        row["error_class"] = _cls(doc.get("error_class"))
        row["stage"] = doc.get("stage") if doc.get("stage") in ("import", "read_input", "applies", "run", "holds") else None
    return row


def own_check(run: Mapping[str, Any], value: Mapping[str, Any], handed: Any = None) -> dict:
    """The own check's evidence (module docstring) from a :func:`run_box` result and the episode's :func:`value_of`."""
    ran = run.get("outcome") in ("returned", "refused", "error")
    form = run.get("form")
    if not ran:
        calls = {"mode": "not_run", "matched": 0, "differed": 0, "missing": 0, "unrecorded": 0}
    elif form == "state_env":
        got = run.get("calls") if isinstance(run.get("calls"), dict) else {}
        calls = {"mode": "replay", **{k: int(got.get(k) or 0) for k in ("matched", "differed", "missing", "unrecorded")}}
    else:
        calls = {"mode": "value", "matched": 0, "differed": 0, "missing": 0, "unrecorded": 0}
    if not value.get("present"):
        match = "not_applicable"
    elif run.get("outcome") != "returned":
        match = "unknown"
    else:
        match = "matched" if digest(run.get("result")) == value.get("digest") else "differed"
    out = {
        "ran": ran,
        "error": run.get("error_class") if run.get("outcome") == "error" else None,
        "refused": run.get("outcome") == "refused",
        "deterministic": run.get("deterministic") if isinstance(run.get("deterministic"), bool) else None,
        "value_match": match,
        "calls": calls,
    }
    if not ran:
        out["why"] = str(run.get("why") or "not run")
    if run.get("stage") is not None and run.get("outcome") in ("refused", "error"):
        out["stage"] = run["stage"]
    return out


def outcome_of(judged: Mapping[str, Any]) -> str:
    """``contrary`` or ``positive`` as the episode's signals judged the value (:func:`.captures.judged_outcome`: a
    rejected verdict, a contrary end or a correction after the answer; an accepted verdict or a positive end); else
    ``none``. Signals only."""
    if judged.get("contrary"):
        return "contrary"
    if judged.get("positive"):
        return "positive"
    return "none"


def signature_input(form: str, run: Mapping[str, Any], case: Mapping[str, Any]) -> Any:
    """What a form's signature describes as its input: the read value (``value``) when a run read one, else the recorded
    data (the request's for ``value``, the state's otherwise; rev 2: nothing is run, so always the recorded data)."""
    if form == "value" and "value" in run:
        return run.get("value")
    return copy.deepcopy(case.get("data") or [])
