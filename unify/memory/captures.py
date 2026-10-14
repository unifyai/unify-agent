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
``published_at``); an episode with a row there is done.
Nothing here keys on a task id, a stream boundary or a position.
"""

from __future__ import annotations

import ast
import contextlib
import fcntl
import hashlib
import json
import os
import re
import shutil
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
from .request_value import REQUEST_CLASS, request_data
from .sandbox_run import PYTHON, SandboxResult, run_confined

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
#: The files of one capture, as the store, the pool and the actor's snapshot hold them.
CAPTURE_FILES = (PROCEDURE_FILE, SIGNATURE_FILE, RECORD_FILE, BROKE_FILE)
#: The actor's snapshot of the store, under its library copy.
EXPORT_DIR = ".memory/captures"
#: Each confined run's wall bound (the procedure runners' own, :data:`.procedures.PROCEDURE_S`).
RUN_S = 60.0
MAX_RESULT_BYTES = 8 * 1024 * 1024
MAX_SOURCE_BYTES = 1024 * 1024
GOAL_CHARS = 600
REPRODUCES, DIFFERS, UNKNOWN = "reproduces", "differs", "unknown"
FAILED = "failed"
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

_RUN = r"""
import copy, importlib.util, json, sys
sys.dont_write_bytecode = True
""" + REQUEST_CLASS + r"""

def canon(v):
    return json.dumps(v, sort_keys=True, ensure_ascii=False, separators=(",", ":"), default=str)


out, stage = {}, "import"
try:
    spec = importlib.util.spec_from_file_location("procedure", "/case/procedure.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    with open("/case/request.txt", encoding="utf-8") as f:
        text = f.read()
    with open("/case/data.json", encoding="utf-8") as f:
        data = json.load(f)
    stage = "read_input"
    value = mod.read_input(Request(text, data))
    out["value"] = canon(value)
    if callable(getattr(mod, "applies", None)):
        stage = "applies"
        out["applies"] = bool(mod.applies(copy.deepcopy(value)))
    stage = "run"
    first = mod.run(copy.deepcopy(value))
    out["result"] = canon(first)
    second = mod.run(copy.deepcopy(value))
    out["result2"] = canon(second)
    if callable(getattr(mod, "holds", None)):
        stage = "holds"
        out["holds"] = bool(mod.holds(copy.deepcopy(value), copy.deepcopy(first)))
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


def run_box(
    source: str,
    request_text: str,
    *,
    work: Path,
    runner: Callable[..., SandboxResult] = run_confined,
    python: Path = PYTHON,
    timeout_s: float = RUN_S,
) -> dict:
    """Run the procedure *source* on *request_text* (handed to ``read_input`` with its data, :mod:`.request_value`) in a
    confined box. Returns ``{outcome, value, result,
    deterministic, applies, holds, error_class, stage}``: ``outcome`` is ``returned``, ``refused``, ``error`` or
    ``not run`` (the box gave no result, or timed out); ``value`` and ``result`` are parsed JSON."""
    work.mkdir(parents=True, exist_ok=True)
    case = Path(tempfile.mkdtemp(prefix="case-", dir=work))
    out = Path(tempfile.mkdtemp(prefix="out-", dir=work))
    try:
        (case / PROCEDURE_FILE).write_text(source, encoding="utf-8")
        (case / "request.txt").write_text(request_text, encoding="utf-8")
        (case / "data.json").write_text(json.dumps(request_data(request_text), ensure_ascii=False), encoding="utf-8")
        (case / "run.py").write_text(_RUN, encoding="utf-8")
        res = runner(
            [str(python), "-I", "/case/run.py"],
            ro={case: "/case"},
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
        shutil.rmtree(case, ignore_errors=True)
        shutil.rmtree(out, ignore_errors=True)
    outcome = doc.get("outcome")
    if outcome not in ("returned", "refused", "error"):
        return {"outcome": "not run", "why": "it could not be run (no outcome)"}
    row: dict = {"outcome": outcome}
    for key in ("value", "result", "result2"):
        if isinstance(doc.get(key), str):
            try:
                row[key] = json.loads(doc[key])
            except ValueError:
                row[key] = doc[key]
    if outcome == "returned":
        row["deterministic"] = doc.get("result") == doc.get("result2")
    row.pop("result2", None)
    for key in ("applies", "holds"):
        if isinstance(doc.get(key), bool):
            row[key] = doc[key]
    if outcome != "returned":
        row["error_class"] = _cls(doc.get("error_class"))
        row["stage"] = doc.get("stage") if doc.get("stage") in ("import", *STATE_FUNCTIONS) else None
    return row


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
    """``(True, value)`` for the value the episode handed back: the last answered dialogue action's value, else
    the final reply's trailing value (its trailing JSON object, else its last non-empty line, read as a dialogue
    action is); ``(False, None)`` when it handed back no value (a work-tree change only, or nothing)."""
    from .integration.adapters.dialogue import action_payload, split_action
    from .procedures import action_value

    i = answered_action(ep)
    if i is not None:
        return True, action_value((getattr(ep, "actions", []) or [])[i])
    replies = [r for r in getattr(ep, "replies", []) or [] if isinstance(r, str) and r.strip()]
    if replies:
        _method, args, kwargs = split_action(action_payload(replies[-1], None))
        if kwargs:
            return True, dict(kwargs)
        return True, (args[0] if len(args) == 1 else list(args))
    return False, None


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
    if record["outcome"] == FAILED and out.status == trust.ADOPTED:
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
                    out[row["episode"]] = row
    except OSError:
        pass
    return out


def episode_row(store: Path | str, eid: str) -> dict | None:
    """The capture step's row for episode *eid*, or None when it has none."""
    if not _EPISODE_ID.fullmatch(eid or ""):
        return None
    return episode_rows(store).get(eid)


def record_use(store: Path | str, cid: str, use: trust.Use) -> dict | None:
    """Add *use* to capture *cid*'s record and recompute its trust; returns the record, or None when there is no such
    capture. A use of the same episode and the same kind replaces nothing: each call is one use."""
    root = Path(store)
    if not _ID.fullmatch(cid or ""):
        return None
    with _locked(root):
        rec = _read_json(root / cid / RECORD_FILE)
        if not isinstance(rec, dict):
            return None
        rec.setdefault("uses", []).append(use.row())
        refresh(rec)
        _write(root / cid / RECORD_FILE, _json(rec))
    return rec


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


def capture(
    staging: Path | str,
    ep: Any,
    store: Path | str,
    *,
    signals: Iterable[Any] = (),
    checker_visible: bool = False,
    runner: Callable[..., SandboxResult] = run_confined,
    python: Path = PYTHON,
    work: Path | None = None,
    clock: Callable[[], float] = time.time,
) -> dict:
    """Capture episode *ep*'s attempt from its settled staging dir into *store* (module docstring). Returns the
    capture step's row, also appended to ``captures-events.jsonl``. Never
    raises for what the staging holds; a step that cannot capture says why in ``safety``."""
    t0 = clock()
    staging, store = Path(staging), Path(store)
    eid = str(getattr(ep, "episode_id", ""))
    row: dict = {
        "capture_id": None,
        "episode": eid,
        "safety": "ok",
        "reproduces": None,
        "deterministic": None,
        "applies": None,
        "holds": None,
        "outcome": None,
        "same_as": [],
        "seconds": None,
        "published_at": None,
    }

    def finish(**fields: Any) -> dict:
        row.update(fields)
        row["seconds"] = round(clock() - t0, 3)
        if _EPISODE_ID.fullmatch(eid):
            with _locked(store):
                _event(store, row)
        return row

    if not _EPISODE_ID.fullmatch(eid):
        return finish(safety="no episode id")
    if not completed(ep):
        return finish(safety="not a completed attempt: nothing was handed back")
    source = _staged(staging, PROCEDURE_FILE)
    if source is None:
        return finish(safety="no procedure")
    why = safety(source)
    if why is not None:
        return finish(safety=why)
    broke = _staged(staging, BROKE_FILE)
    if broke is not None and (KEY_SHAPED.search(broke) or not _broke_lines(broke)):
        broke = None  # nothing broke ("none"), or a key-shaped string, which never enters the store
    summary = _staged(staging, SUMMARY_FILE) or ""
    request_text = ep.request[0] if getattr(ep, "request", None) else ""
    work_root = Path(work) if work is not None else Path(tempfile.mkdtemp(prefix="capture-"))
    try:
        run = run_box(source, request_text, work=work_root / "self", runner=runner, python=python)
        found, value = handed_back(ep)
        reproduces = compare(run, found, value) if run.get("outcome") != "not run" else UNKNOWN
        regime = str(getattr(ep, "regime", "") or "none")
        regime = regime if regime in trust.REGIMES else "none"
        sig_out = trust.signal_outcome(list(signals), regime, checker_visible=checker_visible)
        contrary = sig_out["verdict"] == "rejected" or sig_out["end_outcome"] == "contrary"
        positive = sig_out["verdict"] == "accepted" or sig_out["end_outcome"] == "positive"
        outcome = FAILED if (contrary or reproduces == DIFFERS) else ("positive" if positive else "none")
        returned = run.get("outcome") == "returned"
        sig = signature(
            run.get("value"),
            run.get("result"),
            returned=returned,
            applies=run.get("applies"),
            holds=run.get("holds"),
            digests=function_digests(source),
            goal=asked(summary, request_text),
            request=request_text,
        )
        cid = capture_id(source)
        agreed = None
        if returned and reproduces != UNKNOWN:
            agreed = reproduces == REPRODUCES and run.get("deterministic") is not False
        elif returned:
            agreed = run.get("deterministic")
        case = {
            "episode": eid,
            "reproduces": reproduces,
            "deterministic": run.get("deterministic"),
            "applies": run.get("applies"),
            "holds": run.get("holds"),
            "run": run.get("outcome"),
            "error_class": run.get("error_class"),
            "outcome": outcome,
        }
        use = (
            trust.Use(
                eid,
                regime,
                run["outcome"],
                answered=reproduces == REPRODUCES,
                verdict=sig_out["verdict"],
                end_outcome=sig_out["end_outcome"],
                holds=run.get("holds"),
                corrected=sig_out["corrected"],
                agreed=agreed,
            )
            if run.get("outcome") in trust.OUTCOMES
            else None
        )
        # recurrence needs this procedure's read of the request: its input signature
        same_as = _recurrence(
            store, cid, sig, request_text, found, value, eid, regime, sig_out,
            runner=runner, python=python, work=work_root,
        ) if "value" in run else []  # fmt: skip
        with _locked(store):
            d = store / cid
            rec = _read_json(d / RECORD_FILE)
            rec = rec if isinstance(rec, dict) else empty_record(cid)
            if eid not in rec["episodes"]:
                rec["episodes"].append(eid)
                rec["cases"].append(case)
                if use is not None:
                    rec["uses"].append(use.row())
            rec["same_as"] = sorted(set(rec.get("same_as") or []) | set(same_as))
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
            reproduces=reproduces,
            deterministic=run.get("deterministic"),
            applies=run.get("applies"),
            holds=run.get("holds"),
            outcome=outcome,
            same_as=same_as,
            run=run.get("outcome"),
            published_at=clock(),
        )
    finally:
        if work is None:
            shutil.rmtree(work_root, ignore_errors=True)


def _recurrence(
    store: Path,
    cid: str,
    sig: Mapping[str, Any],
    request_text: str,
    found: bool,
    value: Any,
    eid: str,
    regime: str,
    sig_out: Mapping[str, Any],
    *,
    runner: Callable[..., SandboxResult],
    python: Path,
    work: Path,
) -> list[str]:
    """Run every other capture whose input signature is *sig*'s on this request, confined; each one that gives the
    handed-back value is returned (``same_as``) and gains a use. Every run is noted in its ``recurrence`` rows. Failed
    captures are never run. Content only: no identifier and no text of the request is compared."""
    if not found:
        return []
    same: list[str] = []
    for other, doc in sorted(load(store).items()):
        if other == cid or doc.get("procedure") is None:
            continue
        rec, osig = doc["record"], doc.get("signature")
        if rec.get("outcome") == FAILED or not isinstance(osig, dict):
            continue
        if osig.get("input") != sig.get("input") or eid in (rec.get("episodes") or []):
            continue
        if safety(doc["procedure"]) is not None:
            continue
        run = run_box(doc["procedure"], request_text, work=work / other, runner=runner, python=python)
        gives = (
            "same"
            if run.get("outcome") == "returned" and digest(run.get("result")) == digest(value)
            else "different" if run.get("outcome") == "returned" else str(run.get("outcome"))
        )
        with _locked(store):
            r = _read_json(store / other / RECORD_FILE)
            if not isinstance(r, dict):
                continue
            r.setdefault("recurrence", []).append({"episode": eid, "gives": gives})
            if gives == "same":
                r.setdefault("uses", []).append(
                    trust.Use(
                        eid,
                        regime,
                        "returned",
                        answered=True,
                        verdict=sig_out.get("verdict"),
                        end_outcome=sig_out.get("end_outcome"),
                        holds=run.get("holds"),
                        corrected=bool(sig_out.get("corrected")),
                        agreed=run.get("deterministic") is not False,
                    ).row(),
                )
                r["same_as"] = sorted(set(r.get("same_as") or []) | {cid})
            refresh(r)
            _write(store / other / RECORD_FILE, _json(r))
        if gives == "same":
            same.append(other)
    return same


# --- after a fork settles -------------------------------------------------------------------------------------


def capture_settled(
    state_dir: Path | str,
    eids: Iterable[str],
    *,
    episode: Callable[[str], Any],
    signals: Callable[[str], Iterable[Any]],
    checker_visible: bool,
    runner: Callable[..., SandboxResult] = run_confined,
    python: Path = PYTHON,
) -> list[dict]:
    """Capture each episode of *eids* whose fork has settled and that has no capture row yet. *episode* reads an
    episode by id (raising when it is not recorded), *signals* its signals. Returns the new rows. Never raises: an
    episode that cannot be read is left for a later call."""
    from .staging import STATUS_FILES

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
        try:
            ep = episode(eid)
            sigs = list(signals(eid) or [])
        except Exception:  # noqa: BLE001 - not recorded yet: a later call captures it
            continue
        try:
            out.append(
                capture(d, ep, store, signals=sigs, checker_visible=checker_visible, runner=runner, python=python),
            )
        except OSError:
            continue
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


def export_files(store: Path | str) -> dict[str, bytes]:
    """The actor's snapshot of the store, by path under its library copy (``.memory/captures/<id>/<file>``). The
    record of a capture whose landed procedure is not listed says so (:func:`listed_record`)."""
    out: dict[str, bytes] = {}
    landed = landings(store)
    for cid, doc in load(store).items():
        for name, key in (
            (PROCEDURE_FILE, "procedure"),
            (BROKE_FILE, "broke"),
        ):
            if isinstance(doc.get(key), str):
                out[f"{EXPORT_DIR}/{cid}/{name}"] = doc[key].encode("utf-8", "surrogatepass")
        out[f"{EXPORT_DIR}/{cid}/{RECORD_FILE}"] = _json(listed_record(doc["record"], landed.get(cid)))
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
    pass that landed it, ``recognition``: its recognition record, :mod:`.recognition`, and ``at``: when that was
    tested), replacing an earlier landing of it."""
    if not _ID.fullmatch(str(cid)):
        raise ValueError(f"not a capture id: {str(cid)[:40]!r}")
    root = Path(store)
    with _locked(root):
        keys = ("item", "reader", "applies", "episode", "pass", "recognition", "at")
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
