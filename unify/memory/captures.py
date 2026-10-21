"""The capture store: every completed attempt's working procedure, captured by the harness at once.

A *completed attempt* is an episode that handed back a result: an answered dialogue action, a final reply, or a
change to the work tree. Whether it succeeded is not part of the definition. When the episode-end fork settles, it has
written, beside its other files, ``procedure.py`` (the procedure that produced the result, from code that ran in the
attempt) and ``broke.md`` (what broke). :func:`capture` then, without waiting for a writer pass:

1. **checks safety first**: no key-shaped string, a file that parses as Python, and the gate's path rules; a file that
   fails is not captured, and the reason is recorded;
2. **runs it confined** (:func:`.sandbox_run.run_confined`: no network, an empty environment, the procedure and the
   request read-only, only ``/out`` writable): ``read_input`` on the request (its text, with ``.text`` and ``.data``,
   the JSON values it contains: :mod:`.request_value`), ``applies`` on the value, ``run`` twice and ``holds`` on the
   value and the result. Model-written code never runs on the host;
3. **compares** the result with the value the episode handed back, by the digest of their canonical JSON:
   ``reproduces``, ``differs`` or ``unknown`` (the attempt handed back no value, for example a work-tree change);
4. **writes the capture** to ``<state>/memory-captures/<capture id>/``: ``procedure.py``, ``signature.json``,
   ``record.json`` and ``broke.md``. The capture id is the first 16 hex digits of the sha256 of ``procedure.py``'s
   bytes, so two attempts that produce the same code share one capture and their outcomes join one record;
5. **recognises recurrence by content**: every existing capture whose input signature is this one's is run on this
   request, confined; each that gives the handed-back value is listed in ``same_as`` and gains a use.

**Negative knowledge.** An attempt whose episode carries a contrary signal, or whose capture ``differs``, is captured
with ``outcome: "failed"``. It is never offered to run; it reaches the writer as a ``failure`` pool entry with its
``broke.md`` lines, and the writer may land a note of what broke or a test that records the failure.

**The applicability signature** (:func:`signature`) is computed here, never written by the model, and holds no
identifier and no request text: the shapes of the read value and the result (key paths with identifier-shaped keys
replaced, leaf types, length classes), the structural relations between them, whether the stated conditions exist and
held (with their source digests), and the goal, the ``## Asked`` section of the fork's ``summary.md`` with every
identifier-shaped token replaced, which is shown and never compared.

**Trust.** ``record.json`` keeps every use (:class:`.trust.Use` rows), the track record per kind of signal, the
status (``on-trial``, ``adopted``, ``retired``) and the evidence level (``unchecked``, ``reproduces-its-case``,
``confirmed-by-outcomes``) that :mod:`.trust` gives from them.

**Where it lives.** ``<state>/memory-captures/`` is written only by the harness. It is outside every box-writable
root (the fork's box binds only its own staging dir; the writer, curator and helper boxes bind none of the state
dir), so no agent edits it, and it is not the library: the writer reads it as input (pool entries of kind
``capture`` and ``failure``, and ``/inputs/captures/``) and promotes from it by landing library items, and stays
the only agent that edits the library. The actor's copy gets a snapshot under ``.memory/captures/`` at each request
start (:func:`export_files`); captures never enter ``INDEX.md`` or the prompt.

Evidence: ``captures-events.jsonl`` holds one row per capture step (``capture_id``, ``episode``, ``safety``,
``reproduces``, ``deterministic``, ``applies``, ``holds``, ``outcome``, ``same_as``, ``seconds``,
``published_at``); an episode with a row there is done. A row whose outcome is positive and whose capture reproduced
the handed-back value is a *verdict* (:func:`verdict`): a writer pass covering its episode starts at once
(``trigger.Trigger``'s ``verdict``), without waiting for the batch.
Nothing here keys on a task id, a stream boundary or a position.
"""

from __future__ import annotations

import ast
import contextlib
import datetime as dt
import fcntl
import hashlib
import json
import os
import re
import stat
import tempfile
import time
from collections.abc import Callable, Iterable, Iterator, Mapping
from pathlib import Path
from typing import Any

from . import signatures as _signatures
from . import trust
from .fixtures import canonical
from .redact import KEY_SHAPED
from .signals import REGIME_SOURCES

STORE_DIR = "memory-captures"
EVENTS_FILE = "captures-events.jsonl"
LOCK_FILE = ".lock"
PROCEDURE_FILE = "procedure.py"
BROKE_FILE = "broke.md"
SIGNATURE_FILE = "signature.json"
RECORD_FILE = "record.json"
SUMMARY_FILE = "summary.md"
#: What a capture was saved against besides its signature (``memory.match`` hands it to its matcher): its
#: procedure's docstring and its episode's Asked and Approach sections, identifiers stripped; the actor's snapshot
#: adds what broke.
SAVED_FILE = "saved.json"
#: Written by ``land_capture`` when it lands the capture (:func:`record_landing`): the procedure item it became, its
#: reader, the item that is its stated ``applies``, the episode it was landed from, the pass that landed it, its
#: recognition record (:mod:`.recognition`) and when it was tested. The harness's own: never copied to a box. The item
#: records take the landed procedure's signature, ``applies`` and recognition from it (:func:`landed_items`).
LANDED_FILE = "landed.json"
#: The actor's snapshot of the store, under its library copy.
EXPORT_DIR = ".memory/captures"
#: In the snapshot, beside the capture folders, with ``UNIFY_MEMORY_LIBRARY_WITHDRAW_CLOSED_CAPTURES=on``: the
#: captures no longer offered (the writer superseded or retired them) and what is offered in each one's place
#: (:func:`export_files`).
WITHDRAWN_FILE = "withdrawn.json"
#: The shape index (rev 2): one row per shape address, ``{"shape", "capture_id", "at"}``, the first row of a shape
#: naming the capture its later occurrences fold into (:mod:`.capture_shape`).
SHAPES_FILE = "shapes.jsonl"
#: Each capture's occurrences: one row per episode (or segment) it was captured from, and the raw bytes of an
#: occurrence that folded into it (``occurrences/<n>/procedure.py``), kept as evidence.
OCCURRENCES_FILE = "occurrences.jsonl"
OCCURRENCES_DIR = "occurrences"
#: The key of an exported record that offers its capture as a worked example
#: (``UNIFY_MEMORY_LIBRARY_FIXED_ANSWERS_AS_EXAMPLES=on``): ``{"why", "returns"}`` (:mod:`.fixed_answer`).
EXAMPLE_KEY = "worked_example"
#: Each confined run's wall bound (the procedure runners' own, :data:`.procedures.PROCEDURE_S`).
RUN_S = 60.0
MAX_RESULT_BYTES = 8 * 1024 * 1024
MAX_SOURCE_BYTES = 1024 * 1024
GOAL_CHARS = 600
REPRODUCES, DIFFERS, UNKNOWN = "reproduces", "differs", "unknown"
FAILED = "failed"
#: A trace capture's outcome when a signal contradicted its episode (``UNIFY_MEMORY_CAPTURE_TRACE``): negative knowledge,
#: as ``failed`` is for a capture of the earlier contract (:func:`negative`).
CONTRARY = "contrary"
#: The record version of a trace capture (:mod:`.capture_trace`).
TRACE_VERSION = 2


def negative(record: Any) -> bool:
    """Whether a capture's record is negative knowledge: its attempt failed (``failed``), or, for a trace capture,
    a signal contradicted its episode (``contrary``). Never offered to run, never adopted."""
    return isinstance(record, Mapping) and record.get("outcome") in (FAILED, CONTRARY)
_ID = re.compile(r"[0-9a-f]{16}\Z")
_EPISODE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,127}\Z")
STATE_FUNCTIONS = ("read_input", "run", "applies", "holds")

# --- identifiers (the one rule: :mod:`.signatures`) ---------------------------------------------------------------

ID_MARK = _signatures.ID_MARK
is_identifier = _signatures.is_identifier
strip_identifiers = _signatures.strip_identifiers


def _is_id(key: Any, printed: frozenset[str]) -> bool:
    return str(key) in printed or is_identifier(str(key))


def _strip_keys(tree: Any, printed: frozenset[str] = frozenset()) -> Any:
    """A shape tree with every identifier key (:mod:`.signatures`, and the values the request labels as ids,
    *printed*) replaced by :data:`ID_MARK` (the first such key's subtree is kept; the shape of a mapping keyed by ids
    is the shape of one of its entries)."""
    if isinstance(tree, dict):
        out: dict = {}
        for k in sorted(tree, key=str):
            name = ID_MARK if _is_id(k, printed) else k
            if name not in out:
                out[name] = _strip_keys(tree[k], printed)
        return out
    if isinstance(tree, list):
        return [_strip_keys(t, printed) for t in tree]
    return tree


def _strip_path(path: str, printed: frozenset[str] = frozenset()) -> str:
    parts = re.split(r"(\.|\[\])", path)
    return "".join(ID_MARK if p not in (".", "[]", "") and _is_id(p, printed) else p for p in parts)


# --- the signature --------------------------------------------------------------------------------------------


def _length_class(n: int) -> str:
    return "0" if n == 0 else "1" if n == 1 else "2-9" if n < 10 else "10-99" if n < 100 else "100+"


def shape(value: Any, printed: frozenset[str] = frozenset()) -> dict:
    """The shape of a JSON value, without any value or identifier: a mapping or list as its key tree and length
    classes (:func:`.memory_helper.value_shape`, identifier-shaped keys replaced), text as its length class, any
    other value as its type."""
    from .memory_helper import value_shape

    if isinstance(value, (dict, list)):
        desc = value_shape(value)
        if desc is not None:
            out = {
                "kind": "value",
                "tree": _strip_keys(desc.get("tree"), printed),
                "lengths": {
                    _strip_path(p, printed): c for p, c in sorted((desc.get("lengths") or {}).items())
                },
            }
            if desc.get("truncated"):
                out["truncated"] = True
            return out
    if isinstance(value, str):
        return {"kind": "text", "length": _length_class(len(value))}
    if value is None:
        return {"kind": "null"}
    return {"kind": "scalar", "type": type(value).__name__}


def _flat(value: Any) -> Iterator[Any]:
    stack = [value]
    while stack:
        v = stack.pop()
        yield v
        if isinstance(v, dict):
            stack.extend(v.values())
        elif isinstance(v, list):
            stack.extend(v)


def relations(value: Any, result: Any) -> list[str]:
    """The general structural relations that hold between the read value and the result: ``same_shape``,
    ``same_outer_length``, ``output_within_input`` (the result occurs as a part of the input) and
    ``scalar_output``."""
    out = []
    if shape(value) == shape(result):
        out.append("same_shape")
    if (
        isinstance(value, (list, dict, str))
        and isinstance(result, (list, dict, str))
        and len(value) == len(result)
    ):
        out.append("same_outer_length")
    want = canonical(result)
    if any(canonical(part) == want for part in _flat(value)):
        out.append("output_within_input")
    if not isinstance(result, (list, dict)):
        out.append("scalar_output")
    return out


def function_digests(source: str) -> dict[str, str | None]:
    """The sha256 (16 hex digits) of each of :data:`STATE_FUNCTIONS`' source segments in *source*, or None when it
    does not define one. Parsed, never run."""
    try:
        tree = ast.parse(source)
    except (SyntaxError, ValueError):
        return dict.fromkeys(STATE_FUNCTIONS)
    out: dict[str, str | None] = dict.fromkeys(STATE_FUNCTIONS)
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in out:
            seg = ast.get_source_segment(source, node) or ""
            out[node.name] = hashlib.sha256(seg.encode("utf-8", "surrogatepass")).hexdigest()[:16]
    return out


def asked(summary: str, request: str = "") -> str:
    """The ``## Asked`` section of a fork's ``summary.md`` (its text up to the next heading), identifiers replaced
    and clipped; ``""`` when it has none."""
    lines, on = [], False
    for line in str(summary).splitlines():
        if line.startswith("#"):
            if on:
                break
            on = line.lstrip("#").strip().lower() == "asked"
            continue
        if on:
            lines.append(line)
    text = " ".join(" ".join(lines).split())
    text = strip_identifiers(text, request=request)
    return text if len(text) <= GOAL_CHARS else text[: GOAL_CHARS - 1].rstrip() + "…"


def signature(
    value: Any,
    result: Any,
    *,
    returned: bool,
    applies: bool | None,
    holds: bool | None,
    digests: Mapping[str, str | None],
    goal: str = "",
    request: str = "",
) -> dict:
    """The applicability signature (module docstring). *returned*: the run returned a result (else ``output`` and
    ``relations`` are empty). *request* is read only for the values it labels as identifiers
    (:func:`.signatures.printed_ids`), which are stripped with every identifier-shaped token; it is never kept."""
    printed = frozenset(_signatures.printed_ids(request))
    return {
        "input": shape(value, printed),
        "output": shape(result, printed) if returned else None,
        "relations": relations(value, result) if returned else [],
        "constraints": {
            "applies": {
                "stated": digests.get("applies") is not None,
                "held": applies,
                "sha256": digests.get("applies"),
            },
            "holds": {
                "stated": digests.get("holds") is not None,
                "held": holds,
                "sha256": digests.get("holds"),
            },
        },
        "goal": strip_identifiers(goal, request=request) if goal else "",
    }


# --- safety ---------------------------------------------------------------------------------------------------


def safety(source: str, rel: str = PROCEDURE_FILE) -> str | None:
    """Why *source* must not be captured or run, or None: a key-shaped string, a path the gate refuses (G6), code
    that does not parse, or no ``read_input`` and ``run``."""
    from .manifest import forbidden, unsafe_path

    if KEY_SHAPED.search(source):
        return "key-shaped string"
    if forbidden(rel):
        return f"forbidden path {rel}"
    bad = unsafe_path(f"env/captured/{rel}")
    if bad is not None:
        return bad[1]
    if len(source.encode("utf-8", "surrogatepass")) > MAX_SOURCE_BYTES:
        return "larger than the procedure bound"
    digests = function_digests(source)
    try:
        ast.parse(source)
    except (SyntaxError, ValueError) as exc:
        return f"does not parse: {type(exc).__name__}"
    missing = [f for f in ("read_input", "run") if digests.get(f) is None]
    if missing:
        return f"defines no {' or '.join(missing)}"
    return None


# --- the confined run -----------------------------------------------------------------------------------------

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


# --- what the episode handed back -----------------------------------------------------------------------------


def digest(value: Any) -> str:
    """The first 16 hex digits of the sha256 of *value*'s canonical JSON."""
    return hashlib.sha256(canonical(value).encode("utf-8", "surrogatepass")).hexdigest()[:16]


def answered_action(ep: Any) -> int | None:
    """The index in ``ep.actions`` of the action whose value :func:`handed_back` gives (the last answered dialogue
    action, whose arguments are the fixture source ``args:<i>``); None when the value comes from the final reply or
    the episode handed back none."""
    actions = getattr(ep, "actions", []) or []
    for i in range(len(actions) - 1, -1, -1):
        if getattr(actions[i], "kind", "tool") == "dialogue" and getattr(actions[i], "status", None) == "ok":
            return i
    return None


def handed_back(ep: Any) -> tuple[bool, Any]:
    """``(True, value)`` for the value the episode handed back: the last answered dialogue action's value, else,
    when the work tree did not change, the final reply's trailing value (its trailing JSON object, else its last
    non-empty line, read as a dialogue action is); ``(False, None)`` when it handed back no value: nothing, or a
    work-tree change with no answered dialogue action, whose outcome is the change itself (its closing reply, often a
    prose summary of what was done, is not a value to reproduce)."""
    from .integration.adapters.dialogue import action_payload, split_action
    from .procedures import action_value

    i = answered_action(ep)
    if i is not None:
        return True, action_value((getattr(ep, "actions", []) or [])[i])
    before, after = getattr(ep, "worktree_before", None), getattr(ep, "worktree_after", None)
    if after is not None and after != before:
        return False, None
    replies = [r for r in getattr(ep, "replies", []) or [] if isinstance(r, str) and r.strip()]
    if replies:
        _method, args, kwargs = split_action(action_payload(replies[-1], None))
        if kwargs:
            return True, dict(kwargs)
        return True, (args[0] if len(args) == 1 else list(args))
    return False, None


#: The implicit corrections: a reader that corrected the answer or asked again, and the same request asked again.
#: Each says the episode's answer was wrong only when it came after the answer was handed back.
IMPLICIT_CORRECTIONS = frozenset({("reader", "correct"), ("reader", "re_ask"), ("recurrence", "re_ask")})


def _instant(ts: Any) -> dt.datetime | None:
    """An ISO time as an aware datetime (``Z`` read as UTC), or None when it is not one."""
    if not isinstance(ts, str) or not ts:
        return None
    try:
        at = dt.datetime.fromisoformat(ts.replace("Z", "+00:00"))
    except ValueError:
        return None
    return at if at.tzinfo is not None else at.replace(tzinfo=dt.UTC)


def came_after(sig: Any, ep: Any) -> bool:
    """Whether *sig* came after episode *ep* handed back its value: after the episode ended (no action carries a
    time of its own, and the value is final when the episode ends). Unknown when either time is missing or not a
    time, and then False."""
    at, ended = _instant(getattr(sig, "ts", None)), _instant(getattr(ep, "ended_at", None))
    return at is not None and ended is not None and at > ended


def _given_order(sig: Any) -> tuple:
    """A checker verdict's place among the episode's: its time, then its number (``<episode>.checker.<n>``)."""
    sid = str(getattr(sig, "signal_id", ""))
    tail = sid.rsplit(".", 1)[-1]
    return (str(getattr(sig, "ts", "")), int(tail) if tail.isdigit() else -1, sid)


def final_value_signals(signals: Iterable[Any], ep: Any) -> list:
    """The signals that judge the value the episode handed back (:func:`handed_back`), so a capture of the final
    procedure is never judged by a verdict on another value.

    A checker verdict names the action it judged (``<episode>/actions/<i>``) where the pairing is certain. Those on
    the answered action (:func:`answered_action`) judge the handed-back value; one that names another action judged
    another value and is left out (an attempt that was rejected and then accepted is judged by its acceptance). When
    verdicts name actions but none names this one, no verdict judges it. When none names an action, the last verdict
    given stands for the final value. An implicit correction (:data:`IMPLICIT_CORRECTIONS`) judges it only when it
    came after the value was handed back (:func:`came_after`); one that came before was about an earlier value. Every
    other signal (provenance) is about the episode and is kept."""
    sigs = [
        s
        for s in signals
        if (getattr(s, "source", None), getattr(s, "label", None)) not in IMPLICIT_CORRECTIONS or came_after(s, ep)
    ]
    checker = [s for s in sigs if getattr(s, "source", None) == "checker"]
    if not checker:
        return sigs
    others = [s for s in sigs if getattr(s, "source", None) != "checker"]
    eid = str(getattr(ep, "episode_id", ""))
    named = [s for s in checker if str(getattr(s, "refers_to", "")).startswith(f"{eid}/actions/")]
    i = answered_action(ep)
    if named:
        kept = [s for s in named if i is not None and getattr(s, "refers_to", "") == f"{eid}/actions/{i}"]
    else:
        kept = [max(checker, key=_given_order)]
    return others + kept


def completed(ep: Any) -> bool:
    """Whether the episode is a completed attempt: it handed back a value or changed the work tree."""
    found, _ = handed_back(ep)
    before, after = getattr(ep, "worktree_before", None), getattr(ep, "worktree_after", None)
    return found or (after is not None and after != before)


def compare(run: Mapping[str, Any], found: bool, value: Any) -> str:
    """``reproduces`` when the run returned the handed-back value, ``differs`` when it returned another or failed,
    ``unknown`` when the episode handed back no value."""
    if not found:
        return UNKNOWN
    if run.get("outcome") != "returned":
        return DIFFERS
    return REPRODUCES if digest(run.get("result")) == digest(value) else DIFFERS


# --- the store ------------------------------------------------------------------------------------------------


def store_dir(state_dir: Path | str) -> Path:
    return Path(state_dir) / STORE_DIR


def capture_id(source: str) -> str:
    return hashlib.sha256(source.encode("utf-8", "surrogatepass")).hexdigest()[:16]


@contextlib.contextmanager
def _locked(store: Path) -> Iterator[None]:
    store.mkdir(parents=True, exist_ok=True)
    fd = os.open(store / LOCK_FILE, os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW | os.O_CLOEXEC, 0o600)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        yield
    finally:
        os.close(fd)


def _write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".tmp-", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(data)
        os.replace(tmp, path)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(tmp)
        raise


def _json(doc: Any) -> bytes:
    return (json.dumps(doc, sort_keys=True, indent=1, ensure_ascii=False, default=str) + "\n").encode("utf-8")


def _read_json(path: Path) -> Any:
    try:
        st = os.lstat(path)
        if not stat.S_ISREG(st.st_mode) or st.st_size > MAX_RESULT_BYTES:
            return None
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _read_text(path: Path) -> str | None:
    try:
        st = os.lstat(path)
        if not stat.S_ISREG(st.st_mode) or st.st_size > MAX_RESULT_BYTES:
            return None
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None


def empty_record(cid: str) -> dict:
    return {
        "capture_id": cid,
        "episodes": [],
        "cases": [],
        "recurrence": [],
        "reproduces": UNKNOWN,
        "deterministic": None,
        "applies": None,
        "holds": None,
        "outcome": "none",
        "uses": [],
        "track": trust.empty_track(),
        "status": trust.ON_TRIAL,
        "status_basis": None,
        "status_reason": None,
        "level": trust.UNCHECKED,
        "same_as": [],
    }


def refresh(record: dict) -> dict:
    """*record* with its derived fields recomputed from its cases and uses: ``reproduces`` (``differs`` when any
    case differs, ``reproduces`` when every case with a value reproduces, else ``unknown``), ``outcome`` (``failed``
    when any case failed), the track record, the status and the evidence level (:mod:`.trust`)."""
    cases = [c for c in record.get("cases") or [] if isinstance(c, dict)]
    if record.get("v") != TRACE_VERSION:
        # a trace record carries no reproduction: what its own check gave is evidence on each case
        known = [c.get("reproduces") for c in cases if c.get("reproduces") in (REPRODUCES, DIFFERS)]
        record["reproduces"] = (
            DIFFERS if DIFFERS in known else REPRODUCES if known else UNKNOWN
        )
    dets = [c.get("deterministic") for c in cases if isinstance(c.get("deterministic"), bool)]
    record["deterministic"] = all(dets) if dets else None
    for key in ("applies", "holds"):
        vals = [c.get(key) for c in cases if isinstance(c.get(key), bool)]
        record[key] = all(vals) if vals else None
    if any(c.get("outcome") == FAILED for c in cases):
        record["outcome"] = FAILED
    elif any(c.get("outcome") == CONTRARY for c in cases):
        # a trace capture (``v`` 2): a contrary signal, never its own check, makes it negative knowledge
        record["outcome"] = CONTRARY
    else:
        signals = [c.get("outcome") for c in cases if c.get("outcome") not in (None, "none")]
        record["outcome"] = signals[-1] if signals else "none"
    uses = [u for u in (trust.use_from_row(r) for r in record.get("uses") or []) if u is not None]
    record["track"] = trust.track(uses)
    prior = record.get("status") if record.get("status") in trust.STATUSES else None
    # a capture starts unchecked (I.2.3, I.11): whether its run reproduced its case is recorded beside it, in
    # ``reproduces``; the levels above unchecked are the gate's to give when it is landed, and outcomes move it
    # through the one decider
    base = trust.UNCHECKED
    out = trust.apply_outcome(base, prior, uses)
    if negative(record) and out.status == trust.ADOPTED:
        # a capture that failed is negative knowledge: it is never adopted, whatever later uses show
        out = trust.Outcome(trust.ON_TRIAL, base, None, None, "its attempt failed", ())
    record["status"] = out.status
    record["status_basis"] = out.basis
    record["status_rule"] = out.rule
    record["status_reason"] = out.reason
    record["status_text"] = trust.status_text(out.status, out.basis)
    record["level"] = out.level
    record["record_line"] = trust.record_line(record["track"])
    return record


def judged_outcome(signals: Iterable[Any], ep: Any, regime: str) -> dict:
    """What the episode's signals say of the value it handed back, as the capture step reads them: the signals that
    judge that value (:func:`final_value_signals`) under the episode's *regime*, ``{"regime", "verdict",
    "end_outcome", "corrected"}`` (:func:`.trust.signal_outcome`), with ``contrary`` (a rejected verdict, a contrary
    end, or a correction after the answer that the regime observes) and ``positive``. Under
    ``UNIFY_MEMORY_REGIME=inferred`` a hidden grader's verdict (never shown to the actor) judges nothing."""
    regime = regime if regime in trust.REGIMES else "none"
    signals = list(signals)
    if any(getattr(s, "timing", "") for s in signals):
        # UNIFY_MEMORY_REGIME=inferred (its signals carry their timing): a verdict the actor never saw (a hidden
        # grader's) decides nothing, so it never stands for the final value's verdict either
        signals = [
            s
            for s in signals
            if not (getattr(s, "source", None) == "checker" and getattr(s, "visible_to_actor", None) is False)
        ]
    judged = final_value_signals(signals, ep)
    sig_out = trust.signal_outcome(judged, regime)
    corrected = any(
        (getattr(s, "source", None), getattr(s, "label", None)) in IMPLICIT_CORRECTIONS
        and getattr(s, "source", None) in REGIME_SOURCES.get(regime, frozenset())
        for s in judged
    )
    contrary = sig_out["verdict"] == "rejected" or sig_out["end_outcome"] == "contrary" or corrected
    positive = sig_out["verdict"] == "accepted" or sig_out["end_outcome"] == "positive"
    return {"regime": regime, **sig_out, "contrary": contrary, "positive": positive}


def rejudge_episode(
    store: Path | str,
    eid: str,
    signals: Iterable[Any],
    ep: Any,
    regime: str
) -> list[str]:
    """Every stored capture with a case or a use in episode *eid*, judged again on the episode's signals now (one
    arrived after the capture, or the episode's regime changed: ``UNIFY_MEMORY_REGIME=inferred``): each case of that
    episode gets its outcome as the capture step gives it (:func:`judged_outcome`; a case that differs from what was
    handed back stays failed), each use row of that episode its regime, verdict, end outcome and correction, then the
    record's derived fields (:func:`refresh`). Returns the ids whose record changed, in id order."""
    root = Path(store)
    out = judged_outcome(signals, ep, regime)
    changed: list[str] = []
    try:
        names = sorted(os.listdir(root))
    except OSError:
        return changed
    with _locked(root):
        for name in names:
            d = root / name
            if not _ID.fullmatch(name) or not d.is_dir() or d.is_symlink():
                continue
            rec = _read_json(d / RECORD_FILE)
            if not isinstance(rec, dict):
                continue
            # a segment's case is judged by the signals about the segment alone (judge_segments), never by the
            # episode's verdict on its final value
            cases = [c for c in rec.get("cases") or []
                     if isinstance(c, dict) and c.get("episode") == eid and c.get("segment") is None]  # fmt: skip
            uses = [u for u in rec.get("uses") or [] if isinstance(u, dict) and u.get("episode") == eid]
            if not cases and not uses:
                continue
            before = _json(rec)
            for case in cases:
                differs = case.get("reproduces") == DIFFERS
                case["outcome"] = FAILED if (out["contrary"] or differs) else ("positive" if out["positive"] else "none")
            for use in uses:
                use.update(
                    regime=out["regime"],
                    verdict=out["verdict"],
                    end_outcome=out["end_outcome"],
                    corrected=bool(out["corrected"]),
                )
            refresh(rec)
            if _json(rec) != before:
                _write(d / RECORD_FILE, _json(rec))
                changed.append(name)
    return changed


def load(store: Path | str) -> dict[str, dict]:
    """Every capture in *store*, by id: ``{"record", "signature", "procedure", "broke"}`` (absent files are None)."""
    root = Path(store)
    out: dict[str, dict] = {}
    try:
        names = sorted(os.listdir(root))
    except OSError:
        return out
    for name in names:
        d = root / name
        if not _ID.fullmatch(name):
            continue
        try:
            if not stat.S_ISDIR(os.lstat(d).st_mode):
                continue
        except OSError:
            continue
        record = _read_json(d / RECORD_FILE)
        if not isinstance(record, dict):
            continue
        out[name] = {
            "record": record,
            "signature": _read_json(d / SIGNATURE_FILE),
            "procedure": _read_text(d / PROCEDURE_FILE),
            "broke": _read_text(d / BROKE_FILE),
            "saved": _read_json(d / SAVED_FILE),
        }
    return out


def saved_against(source: str, summary: str, request: str = "") -> dict:
    """What a capture was saved against besides its signature (:data:`SAVED_FILE`): its procedure's docstring and
    its episode's ``summary.md`` Asked and Approach sections, every identifier stripped (with the values *request*
    labels as identifiers)."""
    from .library_export import _docstring_of, _section

    return {
        "docstring": strip_identifiers(_docstring_of(source), request=request),
        "asked": asked(summary, request),
        "approach": strip_identifiers(_section(str(summary), "Approach"), request=request),
    }


def episode_rows(store: Path | str) -> dict[str, dict]:
    """The capture step's last row per episode, from ``captures-events.jsonl`` (which also marks an episode done)."""
    out: dict[str, dict] = {}
    try:
        with open(Path(store) / EVENTS_FILE, encoding="utf-8", errors="replace") as fh:
            for line in fh:
                try:
                    row = json.loads(line)
                except ValueError:
                    continue
                if isinstance(row, dict) and isinstance(row.get("episode"), str):
                    # a segment's row (``UNIFY_MEMORY_IN_EPISODE``) is kept under its staging name, beside its episode's
                    staging = row.get("staging")
                    out[staging if isinstance(staging, str) and staging else row["episode"]] = row
    except OSError:
        pass
    return out


def verdict(row: Any) -> bool:
    """Whether a capture step's row makes a writer pass due at once: its capture reproduced the value its attempt
    handed back, and nothing contradicted the attempt (outcome positive, or none where the environment gives no
    signal; never failed). A positive signal only orders what is captured first
    (:func:`.integration.consolidate.verdict_candidates`), it never gates a pass, so the passes run where no reward or
    feedback exists."""
    return isinstance(row, Mapping) and row.get("reproduces") == REPRODUCES and row.get("outcome") in ("positive", "none")


def stored(row: Any) -> bool:
    """Whether a capture step's row stored a capture (it names one), whatever its outcome or whether it reproduced
    its value: under ``UNIFY_MEMORY_LIBRARY_REVIEW_EACH_CAPTURE=on`` each such row makes a writer pass due at once
    (:class:`.trigger.Trigger`'s ``per_pass``). A step that refused the staged procedure stored nothing."""
    cid = row.get("capture_id") if isinstance(row, Mapping) else None
    return isinstance(cid, str) and _ID.fullmatch(cid) is not None


def episode_row(store: Path | str, eid: str) -> dict | None:
    """The capture step's row for episode *eid*, or None when it has none."""
    if not _EPISODE_ID.fullmatch(eid or ""):
        return None
    return episode_rows(store).get(eid)


# --- the capture step -----------------------------------------------------------------------------------------


def _event(store: Path, row: dict) -> None:
    store.mkdir(parents=True, exist_ok=True)
    with open(store / EVENTS_FILE, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, sort_keys=True, default=str) + "\n")


def _staged(staging: Path, rel: str) -> str | None:
    """A staged file's text, only a regular file reached without a link (staging is untrusted)."""
    p = staging / rel
    try:
        fd = os.open(p, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    except OSError:
        return None
    with os.fdopen(fd, "rb") as fh:
        if not stat.S_ISREG(os.fstat(fh.fileno()).st_mode):
            return None
        data = fh.read(MAX_SOURCE_BYTES + 1)
    if len(data) > MAX_SOURCE_BYTES:
        return None
    return data.decode("utf-8", "replace")


def _broke_lines(text: str) -> list[str]:
    from .pool import broke_lines

    return broke_lines(text)


# --- the trace contract (UNIFY_MEMORY_CAPTURE_TRACE) ----------------------------------------------------------


def trace_record(cid: str) -> dict:
    """An empty trace record (``v`` 2): the record of :func:`empty_record` without what the trace contract removes
    (``same_as``, ``recurrence`` and ``reproduces``: reproduction is neither a gate nor a verdict here)."""
    rec = empty_record(cid)
    for key in ("same_as", "recurrence", "reproduces"):
        rec.pop(key, None)
    return {"v": TRACE_VERSION, **rec}


def capture_trace_step(
    staging: Path | str,
    ep: Any,
    store: Path | str,
    *,
    signals: Iterable[Any] = (),
    clock: Callable[[], float] = time.time,
    segment: int | None = None,
) -> dict:
    """Capture what episode *ep* (or one segment of it) did, under the trace contract (:mod:`.capture_trace`):

    1. the safety checks, as for every capture (a key-shaped string, a refused path, code that does not parse, no
       ``run``); a file that fails is not captured, and the reason is recorded;
    2. the procedure's form, read from its code (``value``, ``state`` or ``state_env``);
    3. no own check (rev 2): a capture is never run to classify it, so ``own_check`` records
       ``{"ran": false, "why"}`` (:data:`.capture_trace.NOT_RUN`); its signature describes the recorded input and the
       value the episode handed back;
    4. the value the episode handed back, if any (``value.present``), and the outcome from its signals only;
    5. its shape address (:func:`.capture_shape.shape_of`, from its syntax alone): when the store already holds a
       capture of that shape, this occurrence folds into it (no new capture; the occurrence and its raw bytes kept
       there, :data:`OCCURRENCES_FILE`), else it is a new capture under its own id;
    6. the capture's files and its record (``v`` 2: ``trace``, ``value``, ``entry_point``, ``own_check``,
       ``effects_seen``, ``outcome``, ``shape``, ``occurrences``, and ``constant`` when its ``run`` gives a fixed
       answer).

    No recurrence step runs and nothing credits a use: the writer reads and groups. Returns the step's row, also
    appended to ``captures-events.jsonl``; never raises for what the staging holds. *runner*, *python* and *work* are
    kept for the callers and never used: nothing is run."""
    from . import capture_trace as ct
    from .capture_shape import shape_of
    from .fixed_answer import fixed_answer

    t0 = clock()
    staging, store = Path(staging), Path(store)
    name = staging.name
    eid = name.rsplit(f".s{segment}", 1)[0] if segment is not None and name.endswith(f".s{segment}") else name
    row: dict = {
        "v": TRACE_VERSION,
        "capture_id": None,
        "episode": eid,
        "segment": segment,
        "staging": name,
        "safety": "ok",
        "seconds": None,
        "published_at": None,
    }

    def finish(**fields: Any) -> dict:
        row.update(fields)
        row["seconds"] = round(clock() - t0, 3)
        if _EPISODE_ID.fullmatch(name):
            with _locked(store):
                _event(store, row)
        return row

    if not _EPISODE_ID.fullmatch(name):
        return finish(safety="no episode id")
    source = _staged(staging, PROCEDURE_FILE)
    if source is None:
        return finish(safety="no procedure")
    why = ct.safety(source)
    if why is not None:
        return finish(safety=why)
    broke = _staged(staging, BROKE_FILE)
    if broke is not None and (KEY_SHAPED.search(broke) or not _broke_lines(broke)):
        broke = None  # nothing broke ("none"), or a key-shaped string, which never enters the store
    summary = _staged(staging, SUMMARY_FILE) or ""
    request_text = ep.request[0] if getattr(ep, "request", None) else ""
    entry = ct.entry_point(source) or {"name": "run", "params": [], "form": "state"}
    form = entry["form"]
    printed = frozenset(_signatures.printed_ids(request_text))
    value_doc, value = ct.value_of(ep, printed)
    case_doc = ct.case_doc(form, ep)
    # rev 2: the procedure is never run to classify it; its own check is recorded as not run
    check = dict(ct.NOT_RUN)
    run: dict = {"outcome": "not run", "why": check["why"]}
    # the one derivation of an outcome from the episode's signals (judged_outcome), as the value-form capture's
    judged = judged_outcome(signals, ep, str(getattr(ep, "regime", "") or "none"))
    outcome = ct.outcome_of(judged)
    trace = ct.trace_of(ep, episode=name)
    effects = ct.effects_seen(ep)
    # the signature describes what was recorded: the input (the request's or the state's data) and the value the
    # episode handed back, never a run's result
    returned = bool(value_doc.get("present"))
    sig = signature(
        ct.signature_input(form, run, case_doc),
        value,
        returned=returned,
        applies=None,
        holds=None,
        digests=function_digests(source),
        goal=asked(summary, request_text),
        request=request_text,
    )
    constant = fixed_answer(source)
    raw_cid = capture_id(source)
    address = shape_of(source)
    shape_id = address["shape"] if address else None
    case = {
        "episode": eid,
        "segment": segment,
        "staging": name,
        "form": form,
        "trace": trace,
        "value": value_doc,
        "own_check": check,
        "effects_seen": effects,
        "outcome": outcome,
        "deterministic": None,
        "applies": None,
        "holds": None,
        "run": run["outcome"],
        "error_class": None,
        "source_capture_id": raw_cid,
    }
    with _locked(store):
        known = shape_index(store).get(shape_id) if shape_id else None
        cid = known or raw_cid
        folded = cid != raw_cid
        if shape_id and known is None:
            _append_row(store / SHAPES_FILE, {"shape": shape_id, "capture_id": cid, "at": clock()})
        d = store / cid
        rec = _read_json(d / RECORD_FILE)
        rec = rec if isinstance(rec, dict) and rec.get("v") == TRACE_VERSION else trace_record(cid)
        if name not in rec["episodes"]:
            rec["episodes"].append(name)
            rec["cases"].append(case)
            _occurrence(d, len(rec["episodes"]), case, source if folded else None, shape_id, clock())
        rec.update(
            entry_point=entry,
            trace=trace,
            value=value_doc,
            own_check=check,
            effects_seen=effects,
            constant=constant,
            shape=shape_id,
            occurrences=len(rec["episodes"]),
        )
        refresh(rec)
        if not (d / PROCEDURE_FILE).exists():
            _write(d / PROCEDURE_FILE, source.encode("utf-8", "surrogatepass"))
        if not (d / SIGNATURE_FILE).exists():
            _write(d / SIGNATURE_FILE, _json(sig))
        if not (d / SAVED_FILE).exists():
            _write(d / SAVED_FILE, _json(saved_against(source, summary, request_text)))
        if broke is not None and broke.strip():
            old = _read_text(d / BROKE_FILE) or ""
            if broke.strip() not in old:
                _write(d / BROKE_FILE, (old + broke.rstrip() + "\n").encode("utf-8"))
        _write(d / RECORD_FILE, _json(rec))
    return finish(
        capture_id=cid,
        source_capture_id=raw_cid,
        shape=shape_id,
        folded=folded,
        trace=trace,
        value=value_doc,
        entry_point=entry,
        own_check=check,
        effects_seen=effects,
        outcome=outcome,
        constant=constant is not None,
        run=run["outcome"],
        published_at=clock(),
    )


# --- the shape index and occurrences (rev 2) -----------------------------------------------------------------------------


def _append_row(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, sort_keys=True, default=str) + "\n")
        fh.flush()
        os.fsync(fh.fileno())


def _rows_of(path: Path) -> list[dict]:
    try:
        with open(path, encoding="utf-8", errors="replace") as fh:
            lines = fh.readlines()
    except OSError:
        return []
    out = []
    for line in lines:
        try:
            row = json.loads(line)
        except ValueError:
            continue
        if isinstance(row, dict):
            out.append(row)
    return out


def shape_index(store: Path | str) -> dict[str, str]:
    """Shape address -> the capture its occurrences fold into (the first row of each shape in :data:`SHAPES_FILE`).
    A store from before the index (a run continued from an earlier stage) is indexed once, from its procedures' syntax
    (nothing run), in the order its captures were first made; an older capture of a shape already indexed stays as it
    is (nothing is merged or deleted). The caller holds the store's lock."""
    from .capture_shape import shape_of

    store = Path(store)
    path = store / SHAPES_FILE
    if not path.exists():
        order: list[str] = []
        for row in _rows_of(store / EVENTS_FILE):
            cid = row.get("capture_id")
            if isinstance(cid, str) and _ID.fullmatch(cid) and cid not in order:
                order.append(cid)
        try:
            names = sorted(n for n in os.listdir(store) if _ID.fullmatch(n))
        except OSError:
            names = []
        order += [n for n in names if n not in order]
        rows = []
        for cid in order:
            address = shape_of(_read_text(store / cid / PROCEDURE_FILE))
            if address is not None:
                rows.append({"shape": address["shape"], "capture_id": cid, "at": None, "indexed": "existing"})
        _write(path, "".join(json.dumps(r, sort_keys=True) + "\n" for r in rows).encode("utf-8"))
    out: dict[str, str] = {}
    for row in _rows_of(path):
        shape, cid = row.get("shape"), row.get("capture_id")
        if isinstance(shape, str) and isinstance(cid, str) and _ID.fullmatch(cid):
            out.setdefault(shape, cid)
    return out


def _occurrence(d: Path, n: int, case: Mapping[str, Any], raw: str | None, shape: str | None, at: float) -> None:
    """Record occurrence *n* of the capture in *d*: its row, and with *raw* (it folded in with other bytes) its source
    as evidence under ``occurrences/<n>/procedure.py``."""
    row = {
        "n": n,
        "episode": case.get("episode"),
        "segment": case.get("segment"),
        "staging": case.get("staging"),
        "outcome": case.get("outcome"),
        "source_capture_id": case.get("source_capture_id"),
        "shape": shape,
        "folded": raw is not None,
        "at": at,
    }
    if raw is not None:
        _write(d / OCCURRENCES_DIR / str(n) / PROCEDURE_FILE, raw.encode("utf-8", "surrogatepass"))
    _append_row(d / OCCURRENCES_FILE, row)


def occurrences(store: Path | str, cid: str) -> list[dict]:
    """The occurrence rows of capture *cid* (:data:`OCCURRENCES_FILE`), in order."""
    return _rows_of(Path(store) / cid / OCCURRENCES_FILE)


# --- after a fork settles -------------------------------------------------------------------------------------


def capture_settled(
    state_dir: Path | str,
    eids: Iterable[str],
    *,
    episode: Callable[[str], Any],
    signals: Callable[[str], Iterable[Any]],
    segment_episode: Callable[[str], Any] | None = None,
) -> list[dict]:
    """Capture each episode of *eids* whose fork has settled and that has no capture row yet. *episode* reads an
    episode by id (raising when it is not recorded), *signals* its signals. Returns the new rows. Never raises: an
    episode that cannot be read is left for a later call.

    Each is captured under the trace contract (:func:`capture_trace_step`): what it did, nothing run. A segment's
    staging dir (``<episode>.s<n>``, ``UNIFY_MEMORY=in_episode``) is captured from the segment's own record, read by
    *segment_episode* (None: segments are left alone), unjudged: its episode is still running, so no signal of it is
    recorded yet, and the episode's own rules would judge it by the final value, which it did not hand back. Once its
    episode is recorded, :func:`judge_segments` gives it its outcome from the signals about the segment itself."""
    from .staging import STATUS_FILES
    from .segments import split_staging_name

    store = store_dir(state_dir)
    root = Path(state_dir) / "memory-staging"
    done = episode_rows(store)
    out = []
    for eid in eids:
        if not _EPISODE_ID.fullmatch(eid or "") or eid in done:
            continue
        d = root / eid
        try:
            if not stat.S_ISDIR(os.lstat(d).st_mode):
                continue
        except OSError:
            continue
        if not any(os.path.lexists(d / name) for name in STATUS_FILES):
            continue  # not settled yet
        base, segment = split_staging_name(eid)
        try:
            if segment is not None:
                if segment_episode is None:
                    continue
                ep = segment_episode(eid)
                if ep is None:
                    continue  # its record is not written yet: a later call captures it
                sigs = []  # judged once its episode is recorded (judge_segments), by the signals about it alone
            else:
                ep = episode(eid)
                sigs = list(signals(eid) or [])
        except Exception:  # noqa: BLE001 - not recorded yet: a later call captures it
            continue
        try:
            out.append(
                capture_trace_step(d, ep, store, signals=sigs, segment=segment),
            )
        except OSError:
            continue
    if segment_episode is not None:
        out += judge_segments(store, episode=episode, signals=signals, segment_episode=segment_episode)  # fmt: skip
    return out


#: The mark of a segment row judged once its episode was recorded (:func:`judge_segments`).
SEGMENT_JUDGED = "episode_recorded"


def segment_outcome(signals: Iterable[Any], seg_ep: Any, ep: Any, *, regime: str) -> str:
    """The outcome of one segment of the recorded episode *ep* (its record as it stood at the segment's checkpoint is
    *seg_ep*), from the signals about the segment alone, as :func:`.capture_trace.outcome_of` reads them:

    * a verdict naming an action (``<episode>/actions/<i>``) judges the segment only when it names the segment's own
      answered action (its actions are numbered from the episode's start), whenever it came;
    * an implicit correction judges it when it came after the segment's checkpoint and before its episode ended;
    * every other signal judges it only when it came while the segment's cells ran (at or before its checkpoint).

    So a mid-episode segment is never labelled by the verdict on the episode's final value."""
    import dataclasses

    from . import capture_trace as ct

    base = str(getattr(ep, "episode_id", "") or "")
    seg_end, ep_end = _instant(getattr(seg_ep, "ended_at", None)), _instant(getattr(ep, "ended_at", None))
    view = dataclasses.replace(seg_ep, episode_id=base)
    i = answered_action(view)
    mine = []
    for s in signals:
        at = _instant(getattr(s, "ts", None))
        named = str(getattr(s, "refers_to", "") or "")
        if getattr(s, "source", None) == "checker" and named.startswith(f"{base}/actions/"):
            if i is not None and named == f"{base}/actions/{i}":
                mine.append(s)
        elif (getattr(s, "source", None), getattr(s, "label", None)) in IMPLICIT_CORRECTIONS:
            if at is not None and ep_end is not None and at <= ep_end:
                mine.append(s)
        elif at is not None and seg_end is not None and at <= seg_end:
            mine.append(s)
    return ct.outcome_of(judged_outcome(mine, view, regime))


def judge_segments(
    store: Path | str,
    *,
    episode: Callable[[str], Any],
    signals: Callable[[str], Iterable[Any]],
    segment_episode: Callable[[str], Any],
) -> list[dict]:
    """Each captured segment whose episode is now recorded and that is not judged yet gets its outcome
    (:func:`segment_outcome`) on its case in its capture's record (the record's derived fields recomputed), and a
    new ``captures-events.jsonl`` row (its capture row as before, with ``outcome`` and ``judged``), once. Returns
    those rows; never raises for an episode or a segment record that cannot be read (a later call tries again)."""
    store = Path(store)
    rows = episode_rows(store)
    due = [r for name, r in rows.items() if r.get("segment") is not None and r.get("capture_id") and not r.get("judged")
           and r.get("v") == TRACE_VERSION and name == r.get("staging")]  # fmt: skip
    out: list[dict] = []
    for row in due:
        base, name = str(row.get("episode") or ""), str(row["staging"])
        try:
            ep = episode(base)  # raises while the episode is not recorded
            sigs = list(signals(base) or [])
            seg_ep = segment_episode(name)
        except Exception:  # noqa: BLE001 - not recorded yet: a later call judges it
            continue
        if seg_ep is None:
            continue
        outcome = segment_outcome(sigs, seg_ep, ep, regime=str(getattr(ep, "regime", "") or "none"))  # fmt: skip
        with _locked(store):
            d = store / str(row["capture_id"])
            rec = _read_json(d / RECORD_FILE)
            if isinstance(rec, dict):
                for case in rec.get("cases") or []:
                    if isinstance(case, dict) and case.get("staging") == name:
                        case["outcome"] = outcome
                        case["judged"] = SEGMENT_JUDGED
                refresh(rec)
                _write(d / RECORD_FILE, _json(rec))
            judged = {**row, "outcome": outcome, "judged": SEGMENT_JUDGED}
            _event(store, judged)
        out.append(judged)
    return out


# --- what the actor and the writer read -----------------------------------------------------------------------


def listed_record(record: dict, landing: Mapping[str, Any] | None) -> dict:
    """*record* as a box reads it: with ``listing`` (``{"listed": false, "why"}``) when the capture's *landing* tested
    its procedure and it is not listed (:func:`.recognition.listing`), so a captured procedure, the same code, is not
    listed either; *record* itself otherwise."""
    from .recognition import listing

    rec = landing.get("recognition") if isinstance(landing, Mapping) else None
    listed, why = listing(rec if isinstance(rec, Mapping) else None)
    return record if listed else {**record, "listing": {"listed": False, "why": why}}


def export_files(
    store: Path | str,
    *,
    examples: bool = False,
    withdrawn: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, bytes]:
    """The actor's snapshot of the store, by path under its library copy (``.memory/captures/<id>/<file>``). The
    record of a capture whose landed procedure is not listed says so (:func:`listed_record`).

    With *examples* (``UNIFY_MEMORY_LIBRARY_FIXED_ANSWERS_AS_EXAMPLES=on``) the record of a capture whose ``run``
    gives a fixed answer (:func:`.fixed_answer.fixed_answer`; a failed attempt is never offered anyway) carries
    :data:`EXAMPLE_KEY`, and the package, the catalogue, the search index and the dispatch file then offer it as a
    worked example to read, never as a procedure to call.

    With *withdrawn* (``UNIFY_MEMORY_LIBRARY_WITHDRAW_CLOSED_CAPTURES=on``: :func:`.pool.withdrawn_captures`, capture
    id -> ``{"status", "by", "reason"}``) those captures are left out of the snapshot, so nothing offers them, and
    ``.memory/captures/withdrawn.json`` names each with what is offered in its place and what it was saved against
    (its signature and ``saved.json`` texts, identifiers already stripped), so the item that superseded it is offered
    where it would have been."""
    from .fixed_answer import fixed_answer

    out: dict[str, bytes] = {}
    landed = landings(store)
    docs = load(store)
    hidden = dict(withdrawn or {})
    for cid, doc in docs.items():
        if cid in hidden:
            continue
        for name, key in (
            (PROCEDURE_FILE, "procedure"),
            (BROKE_FILE, "broke"),
        ):
            if isinstance(doc.get(key), str):
                out[f"{EXPORT_DIR}/{cid}/{name}"] = doc[key].encode("utf-8", "surrogatepass")
        record = listed_record(doc["record"], landed.get(cid))
        fixed = fixed_answer(doc.get("procedure")) if examples and not negative(record) else None
        if fixed is not None:
            record = {**record, EXAMPLE_KEY: fixed}
        out[f"{EXPORT_DIR}/{cid}/{RECORD_FILE}"] = _json(record)
        if isinstance(doc.get("signature"), dict):
            out[f"{EXPORT_DIR}/{cid}/{SIGNATURE_FILE}"] = _json(doc["signature"])
        saved = doc.get("saved") if isinstance(doc.get("saved"), dict) else {}
        broke = doc.get("broke") if isinstance(doc.get("broke"), str) else ""
        out[f"{EXPORT_DIR}/{cid}/{SAVED_FILE}"] = _json(
            {
                "docstring": str(saved.get("docstring") or ""),
                "asked": str(saved.get("asked") or ""),
                "approach": str(saved.get("approach") or ""),
                "broke": [strip_identifiers(line) for line in _broke_lines(broke)] if broke else [],
            },
        )
    if withdrawn is not None:
        rows: dict[str, dict] = {}
        for cid, why in sorted(hidden.items()):
            if not _ID.fullmatch(str(cid)) or not isinstance(why, Mapping):
                continue
            doc = docs.get(cid) or {}
            saved = doc.get("saved") if isinstance(doc.get("saved"), dict) else {}
            by = why.get("by")
            rows[cid] = {
                "status": str(why.get("status") or ""),
                "by": by if isinstance(by, str) and by else None,
                "reason": strip_identifiers(str(why.get("reason") or "")),
                "signature": doc.get("signature") if isinstance(doc.get("signature"), dict) else None,
                "docstring": str(saved.get("docstring") or ""),
                "asked": str(saved.get("asked") or ""),
                "approach": str(saved.get("approach") or ""),
            }
        out[f"{EXPORT_DIR}/{WITHDRAWN_FILE}"] = _json({"version": 1, "captures": rows})
    return out


def batch_row(store: Path | str, eid: str) -> dict | None:
    """The batch-map row's ``capture`` for episode *eid*: ``{id, reproduces, outcome, level, status, same_as,
    path}``, or None when the episode has no capture."""
    row = episode_row(store, eid)
    cid = row.get("capture_id") if isinstance(row, dict) else None
    if not isinstance(cid, str) or not _ID.fullmatch(cid):
        return None
    rec = _read_json(Path(store) / cid / RECORD_FILE)
    rec = rec if isinstance(rec, dict) else {}
    return {
        "id": cid,
        "reproduces": row.get("reproduces"),
        "outcome": row.get("outcome"),
        "level": rec.get("level"),
        "status": rec.get("status_text") or rec.get("status"),
        "same_as": list(row.get("same_as") or []),
        "path": f"/inputs/captures/{cid}",
    }


def record_landing(store: Path | str, cid: str, landing: Mapping[str, Any]) -> None:
    """Record that capture *cid* was landed as *landing* (``item``, ``reader``, ``applies``, ``episode``, ``pass``: the
    pass that landed it, ``recognition``: its recognition record, :mod:`.recognition`, ``verified``: rev 2's
    ``{"ran": false, "why"}`` (nothing was run to verify it), and ``at``: when it landed), replacing an earlier landing
    of it."""
    if not _ID.fullmatch(str(cid)):
        raise ValueError(f"not a capture id: {str(cid)[:40]!r}")
    root = Path(store)
    with _locked(root):
        keys = ("item", "reader", "applies", "episode", "pass", "recognition", "verified", "at")
        _write(root / cid / LANDED_FILE, _json({k: landing.get(k) for k in keys}))


def landings(store: Path | str) -> dict[str, dict]:
    """Every recorded landing in *store*, by capture id (:data:`LANDED_FILE`); one without an item is left out."""
    root = Path(store)
    try:
        names = sorted(os.listdir(root))
    except OSError:
        return {}
    out: dict[str, dict] = {}
    for name in names:
        doc = _read_json(root / name / LANDED_FILE) if _ID.fullmatch(name) else None
        if isinstance(doc, dict) and isinstance(doc.get("item"), str) and doc["item"]:
            out[name] = doc
    return out


def landed_items(store: Path | str | None) -> dict[str, dict]:
    """What each landed procedure item carries from its capture, by item id: ``capture`` (its id), ``reader`` (the
    reader it was landed with), ``signature`` (its applicability signature as the store holds it; None: none),
    ``applies`` (the item that is its stated ``applies``; None: it states none) and ``recognition`` (its recognition
    record, :mod:`.recognition`; None: not tested). An item several captures landed (a new case of an earlier landing)
    carries the first of them in id order, and the recognition of the one tested last (each landing tests the
    library's procedure as it then is)."""
    if store is None:
        return {}
    out: dict[str, dict] = {}
    tested: dict[str, float] = {}
    for cid, doc in landings(store).items():
        item = doc["item"]
        recognition, at = doc.get("recognition"), doc.get("at")
        at = float(at) if isinstance(at, (int, float)) and not isinstance(at, bool) else float("-inf")
        if item in out:
            if isinstance(recognition, dict) and at >= tested.get(item, float("-inf")):
                out[item]["recognition"], tested[item] = recognition, at
            continue
        sig = _read_json(Path(store) / cid / SIGNATURE_FILE)
        applies, reader = doc.get("applies"), doc.get("reader")
        out[item] = {
            "capture": cid,
            "reader": reader if isinstance(reader, str) and reader else None,
            "signature": sig if isinstance(sig, dict) else None,
            "applies": applies if isinstance(applies, str) and applies else None,
            "recognition": recognition if isinstance(recognition, dict) else None,
        }
        if isinstance(recognition, dict):
            tested[item] = at
    return out


def export_for_writer(store: Path | str, cids: Iterable[str], dest: Path) -> list[str]:
    """Copy the captures *cids* (with the captures their ``same_as`` names) to the writer's ``/inputs/captures/``
    (*dest*); returns the ids copied. The record of a capture whose landed procedure is not listed says why
    (:func:`listed_record`)."""
    docs = load(store)
    landed = landings(store)
    want = set()
    for cid in cids:
        if cid in docs:
            want.add(cid)
            want.update(c for c in docs[cid]["record"].get("same_as") or [] if c in docs)
    for cid in sorted(want):
        d = dest / cid
        d.mkdir(parents=True, exist_ok=True)
        doc = docs[cid]
        (d / RECORD_FILE).write_bytes(_json(listed_record(doc["record"], landed.get(cid))))
        if isinstance(doc.get("signature"), dict):
            (d / SIGNATURE_FILE).write_bytes(_json(doc["signature"]))
        for name, key in ((PROCEDURE_FILE, "procedure"), (BROKE_FILE, "broke")):
            if isinstance(doc.get(key), str):
                (d / name).write_text(doc[key], encoding="utf-8")
    return sorted(want)
