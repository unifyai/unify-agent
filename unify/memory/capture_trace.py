"""The capture contract: what the episode did, and a value only when there is one.

A capture records what its episode did: a pointer list into the episode's own record (its request, observations,
cells and actions with their responses; :func:`trace_of`), never a copy. The procedure the fork wrote takes one of
three forms (:data:`FORMS`, read from its code by :func:`entry_point`, never run to find out):

==============  ============================================================================================
form            contract
==============  ============================================================================================
``value``       ``read_input(request) -> run(value)``: the whole job was in the first message
``state``       ``run(state)``: :class:`.box_state.State` holds every message so far (``.request``, ``.messages``,
                ``.last``, ``.data``)
``state_env``   ``run(state, env)``: ``env`` is the episode's tool namespace, ``env.<channel>.<method>(...)``
                as the episode's actions name them
==============  ============================================================================================

**No capture is ever run.** A capture is never executed to classify, dedupe or rank it, so the capture step records
``own_check`` as :data:`NOT_RUN`.

**The value** (:func:`value_of`): ``{"present": false}``, or ``{"present": true, "digest", "shape"}``. Its outcome comes
from the signals only (``positive``, ``contrary``, ``none``).

Nothing here keys on a task id, a stream position, a world or a word.
"""

from __future__ import annotations

import ast
import copy
import hashlib
import json
from collections.abc import Iterable, Mapping
from typing import Any, Literal, get_args

from .fixtures import canonical
from .request_value import request_data

Form = Literal["value", "state", "state_env"]
Outcome = Literal["positive", "contrary", "none"]
FORMS: tuple[str, ...] = get_args(Form)
OUTCOMES: tuple[str, ...] = get_args(Outcome)
#: The record version a trace capture stamps (``record.json`` and its ``captures-events.jsonl`` row).
VERSION = 2
#: Calls one replay serves at most (an outer bound, not a quota): past it the run raises ``TooManyCalls``.
MAX_REPLAY_CALLS = 2000
#: Issued calls a box's result lists (each call's arguments clipped), for the evidence that names a differing call.
MAX_LISTED_CALLS = 200
LISTED_ARGS_CHARS = 400
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
