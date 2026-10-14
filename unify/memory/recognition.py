"""Whether a landed procedure recognises the requests it fits, tested when it lands on the store's own recorded requests.

``memory.match`` and the librarian list a stored procedure where its reader parses the current request and its
``applies`` holds on what the reader gave. That listing is only as good as the ``applies``: one written for one case may
hold almost anywhere. It is tested here, when ``land_capture`` lands the procedure, on recorded requests only:

* **the other recorded requests** (:func:`recorded_requests`): every episode the capture step saw (its rows in
  ``captures-events.jsonl``) but the one the procedure is landed from. The landed reader, the procedure's ``applies``,
  the procedure and its ``holds`` run on each, confined. Each request the reader parsed is *tested*, and labelled:

  - a **true hold**: ``applies`` held and the procedure is right there: its own ``holds`` held, or its result
    reproduces the value that episode handed back (the digest of its canonical JSON, as :func:`.captures.compare`
    compares) and the store records a success for that episode (:func:`episode_outcomes`). Reproducing the value of
    an episode recorded as failed, or of one with no recorded outcome, is not a true hold;
  - a **false hold**: ``applies`` held and its own ``holds`` is False there;
  - **unknown**: ``applies`` held and nothing says whether the procedure is right there;
  - otherwise ``applies`` did not hold (or the procedure states none), which is no hold.

**The listing rule** (:func:`listing`): a procedure whose ``applies`` held where its ``holds`` failed (``false_holds``
at least 1) is not listed by ``memory.match`` or in the librarian's facts until the
writer fixes it; a procedure with no other recorded request to test is listed, marked untested. The landing records
the counts and the reason with it (:func:`.captures.record_landing`), the item record carries them
(``recognition``), and the catalogue card and ``memory.get`` say why an item is not listed.

Requests are picked by what the store recorded and how the runs went, never by a task id, a benchmark label or a
position in a stream; nothing compares words. Model-written code runs only in :func:`.sandbox_run.run_confined`: the
reader's value and the procedure's result stay in the box, which compares the result with the handed-back value's
digest itself, and only labels come back.
"""

from __future__ import annotations

import json
import shutil
import tempfile
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from . import captures as _captures
from .sandbox_run import PYTHON, SandboxResult, run_confined

TRUE_HOLD, FALSE_HOLD, UNKNOWN, NO_HOLD = "true-hold", "false-hold", "unknown", "no-hold"
#: Each request's own time bound in the box, in seconds (the capture runs' own, :data:`.captures.RUN_S`).
EACH_S = _captures.RUN_S
#: The box's slack beyond the requests' bounds, for starting and reporting.
SLACK_S = 30.0

#: The child that runs the landed items on each request (stdlib only). It imports them from the library at
#: ``/memory``, hands each reader the request's text, and writes only labels: whether the reader parsed, whether ``applies`` and ``holds`` held,
#: whether the result reproduces the expected digest, the stage that raised or ran past its bound.
_CHILD = r"""
import copy, hashlib, importlib, json, signal, sys
sys.path[:0] = ["/memory", "/kit"]
sys.dont_write_bytecode = True
spec = json.load(open("/case/case.json", encoding="utf-8"))
EACH = float(spec["each_s"])


class Late(BaseException):
    pass


def late(*_):
    raise Late()


def digest(v):
    text = json.dumps(v, sort_keys=True, ensure_ascii=False, separators=(",", ":"), default=str)
    return hashlib.sha256(text.encode("utf-8", "surrogatepass")).hexdigest()[:16]


def load(item):
    module, _, name = item.partition(":")
    return getattr(importlib.import_module(module), name)


out = {"rows": []}
try:
    signal.signal(signal.SIGALRM, late)
    fns = {k: (load(spec[k]) if spec.get(k) else None) for k in ("reader", "applies", "procedure", "holds")}
except BaseException as exc:
    out["load"] = type(exc).__name__
else:
    for req in spec["requests"]:
        row, stage = {}, "read"
        try:
            signal.setitimer(signal.ITIMER_REAL, EACH)
            try:
                value = fns["reader"](req["text"])
                row["read"] = "parsed"
                stage = "applies"
                if fns["applies"] is not None:
                    row["applies"] = bool(fns["applies"](copy.deepcopy(value)))
                if row.get("applies") is True:
                    stage = "run"
                    result = fns["procedure"](copy.deepcopy(value))
                    row["run"] = "returned"
                    if req.get("expect") is not None:
                        row["reproduces"] = digest(result) == req["expect"]
                    if fns["holds"] is not None:
                        stage = "holds"
                        row["holds"] = bool(fns["holds"](copy.deepcopy(value), copy.deepcopy(result)))
            finally:
                signal.setitimer(signal.ITIMER_REAL, 0)
        except Late:
            row["late"] = stage
        except BaseException as exc:
            row["raised"] = stage
            row["error_class"] = type(exc).__name__[:64]
        out["rows"].append(row)
with open("/out/result.json", "w", encoding="utf-8") as f:
    json.dump(out, f)
"""


def _label(row: Mapping[str, Any], expected: bool, success: bool) -> str | None:
    """One tested request's label (module docstring); None when the reader did not parse it (not tested)."""
    if row.get("read") != "parsed":
        return None
    if row.get("applies") is not True:
        return NO_HOLD
    if row.get("holds") is False:
        return FALSE_HOLD
    if row.get("holds") is True:
        return TRUE_HOLD
    if expected and success and row.get("reproduces") is True:
        return TRUE_HOLD
    return UNKNOWN


def run_rows(
    tree: Path,
    items: Mapping[str, str | None],
    requests: list[dict],
    *,
    python: Path = PYTHON,
    work: Path,
    runner: Callable[..., SandboxResult] = run_confined,
    each_s: float = EACH_S,
) -> tuple[list[dict], str | None]:
    """Run the landed *items* (``reader``, ``applies``, ``procedure``, ``holds``: item ids, None for one not stated)
    from the library at *tree* on each of *requests* (``{text, expect}``), in one confined box.
    Returns (one row of labels per request, why none could be run or None)."""
    from . import testkit

    if not requests:
        return [], None
    work.mkdir(parents=True, exist_ok=True)
    case = Path(tempfile.mkdtemp(prefix="case-", dir=work))
    out = Path(tempfile.mkdtemp(prefix="out-", dir=work))
    kit = Path(tempfile.mkdtemp(prefix="kit-", dir=work))
    try:
        testkit.stage_package(kit / testkit.PACKAGE)
        (case / "run.py").write_text(_CHILD, encoding="utf-8")
        spec = {**{k: items.get(k) for k in ("reader", "applies", "procedure", "holds")}, "each_s": each_s}
        spec["requests"] = requests
        (case / "case.json").write_text(json.dumps(spec, ensure_ascii=False, default=str), encoding="utf-8")
        limit = each_s * len(requests) + SLACK_S
        res = runner(
            [str(python), "-I", "/case/run.py"],
            ro={tree: "/memory", case: "/case", kit: "/kit"},
            rw={out: "/out"},
            cwd="/memory",
            timeout_s=limit,
            env={},
        )
        if res.timed_out:
            return [], f"the runs did not finish within {limit:g} s"
        doc = _captures._read_result(out / "result.json")
    finally:
        for d in (case, out, kit):
            shutil.rmtree(d, ignore_errors=True)
    if doc is None:
        return [], "the runs could not be started (no result)"
    if isinstance(doc.get("load"), str):
        return [], f"load: {_captures._cls(doc['load'])}"
    rows = doc.get("rows") if isinstance(doc.get("rows"), list) else []
    if len(rows) != len(requests):
        return [], "the runs gave a malformed result"
    return [r if isinstance(r, dict) else {} for r in rows], None


# --- the recorded requests ----------------------------------------------------------------------------------------


def episode_outcomes(store: Path | str) -> dict[str, str]:
    """The outcome the store records for each episode: ``success`` when a use or case row of any capture carries an
    accepted verdict or a positive end for it and none a rejected verdict or a contrary end, ``failure`` when one
    carries a rejected verdict or a contrary end. An episode the store records no outcome for is absent."""
    good: set[str] = set()
    bad: set[str] = set()
    for doc in _captures.load(store).values():
        rec = doc["record"]
        for row in list(rec.get("uses") or []) + list(rec.get("cases") or []):
            if not isinstance(row, dict) or not isinstance(row.get("episode"), str):
                continue
            eid = row["episode"]
            if row.get("verdict") == "rejected" or row.get("end_outcome") == "contrary":
                bad.add(eid)
            elif row.get("verdict") == "accepted" or row.get("end_outcome") == "positive" or (
                row.get("outcome") == "positive"
            ):
                good.add(eid)
    return {e: "failure" for e in bad} | {e: "success" for e in good - bad}


def _recorded_ids(store: Path) -> list[str]:
    """Every episode the store records: each the capture step saw, and each a capture record names; in id order."""
    ids = set(_captures.episode_rows(store))
    for doc in _captures.load(store).values():
        ids.update(e for e in doc["record"].get("episodes") or [] if isinstance(e, str))
    return sorted(i for i in ids if _captures._EPISODE_ID.fullmatch(i))


def _text(ep: Any) -> str | None:
    request = getattr(ep, "request", None) if ep is not None else None
    if not request:
        return None
    text = request[0]
    return text if isinstance(text, str) and text else None


def recorded_requests(store: Path | str, own: str, episode: Callable[[str], Any]) -> list[dict]:
    """Every other recorded request of the store (module docstring), as the box takes it: ``{episode, text, expect,
    success}``, ``expect`` the digest of the value the episode handed back (None: it handed back none) and
    ``success`` whether the store records a success for it. An episode that cannot be read is left out."""
    store = Path(store)
    outcomes = episode_outcomes(store)
    out: list[dict] = []
    for eid in _recorded_ids(store):
        if eid == own:
            continue
        try:
            ep = episode(eid)
        except Exception:  # noqa: BLE001 - an episode that cannot be read is not a recorded request here
            continue
        text = _text(ep)
        if text is None:
            continue
        found, value = _captures.handed_back(ep)
        out.append(
            {
                "episode": eid,
                "text": text,
                "expect": _captures.digest(value) if found else None,
                "success": outcomes.get(eid) == "success",
            },
        )
    return out


# --- the test and the listing rule ---------------------------------------------------------------------------------


def recognise_landing(
    store: Path | str,
    cid: str,
    own: str,
    own_text: str,
    items: Mapping[str, str | None],
    *,
    tree: Path,
    episode: Callable[[str], Any],
    python: Path = PYTHON,
    work: Path,
    runner: Callable[..., SandboxResult] = run_confined,
) -> dict:
    """The recognition record of a procedure landed from capture *cid*'s episode *own* (module docstring): its
    landed *items* run from the library at *tree* on the store's other recorded requests. ``{requests, tested,
    true_holds, false_holds, unknown, listed, why}``, with ``error`` when the runs could not be made (nothing is then
    counted)."""
    others = recorded_requests(store, own, episode)
    sent = [{k: r[k] for k in ("text", "expect")} for r in others]
    rows, why = run_rows(Path(tree), items, sent, python=python, work=Path(work), runner=runner)
    counts = {TRUE_HOLD: 0, FALSE_HOLD: 0, UNKNOWN: 0, NO_HOLD: 0}
    tested = 0
    if why is None:
        for req, row in zip(others, rows):
            got = _label(row, req["expect"] is not None, req["success"])
            if got is not None:
                tested += 1
                counts[got] += 1
    out: dict = {
        "requests": len(others),
        "tested": tested,
        "true_holds": counts[TRUE_HOLD],
        "false_holds": counts[FALSE_HOLD],
        "unknown": counts[UNKNOWN],
    }
    if why is not None:
        out["error"] = why
    listed, line = listing(out)
    out["listed"], out["why"] = listed, line
    return out


def listing(rec: Mapping[str, Any] | None) -> tuple[bool, str]:
    """Whether a procedure with recognition record *rec* is listed by ``memory.match`` and in the librarian's facts,
    and the line that says why (module docstring). A procedure without a record (not landed, or landed before this
    test) is listed as before, with no line."""
    if not isinstance(rec, Mapping):
        return True, ""

    def n(key: str) -> int:
        v = rec.get(key)
        return v if isinstance(v, int) and not isinstance(v, bool) and v > 0 else 0

    if n("false_holds"):
        return False, (
            f"not listed: its applies held on {n('false_holds')} of {n('tested')} tested recorded request(s) where "
            "its own holds failed; make applies hold only where the procedure is right"
        )
    if isinstance(rec.get("error"), str) and rec["error"]:
        return True, f"untested: its test could not be run ({rec['error']})"
    if not n("tested") and n("requests"):
        return True, f"untested: its reader parsed none of the {n('requests')} other recorded request(s)"
    if not n("tested"):
        return True, "untested: no other recorded request to test its applies on"
    return True, (
        f"tested on {n('tested')} recorded request(s): {n('true_holds')} true hold(s), 0 false holds, "
        f"{n('unknown')} unknown"
    )


def listed(record: Mapping[str, Any] | None) -> bool:
    """Whether the item record *record* (its ``recognition``) is listed (:func:`listing`)."""
    rec = record.get("recognition") if isinstance(record, Mapping) else None
    return listing(rec)[0]


def unlisted_captures(records: Mapping[str, Any] | None) -> set[str]:
    """The capture ids whose landed procedure the item *records* do not list: a captured procedure is the same code,
    so it is not listed either."""
    out: set[str] = set()
    for rec in (records or {}).values():
        if not isinstance(rec, Mapping) or listed(rec):
            continue
        landed = rec.get("landed_from")
        if isinstance(landed, Mapping) and isinstance(landed.get("capture"), str):
            out.add(landed["capture"])
    return out
