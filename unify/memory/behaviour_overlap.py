"""Functions that do the same job, judged by what they do: each run on the other's recorded inputs.

Two library functions overlap in behaviour when each, run on its own recorded argument lists and on the other's,
gives the same result as the other wherever both return: at least one of each function's own lists is a case where
both return, and no list where both return gives different results. A list one of them refuses, raises on, or cannot
be called with is no case either way: a narrower function is not a different one. Names, docstrings, bodies and
fixtures are never read; two functions that take the same kind of argument and read the same files but compute
different things are not an overlap.

**Recorded inputs.** A function's recorded argument lists are the calls its tests made of it that returned, as the
gate's call recorder saw them (:mod:`.example_calls`), every argument recorded whole as JSON: at most
:data:`MAX_INPUTS` distinct lists in the order of their call keys, and at most :data:`MAX_INPUT_BYTES` of them in
all (:func:`recorded_inputs`). The gate keeps them with each landed function's body digest in the evidence store
(:meth:`.evidence.EvidenceStore.write_function_inputs`), so a function's lists are those of its current definition.
A function without recorded lists (an argument too large to record, an object with no JSON form) has no case and is
never a candidate.

**The comparison** runs in one confined box (:func:`.sandbox_run.run_confined`: the library read-only at /memory, no
network, a cleared environment, the box's limits) with :mod:`.behaviour_box` run by path; each call is bounded by
:data:`CALL_TIMEOUT_S` and the box by :data:`BOX_TIMEOUT_S`. Only outcomes and result digests come back, read as
untrusted input. Only pairs that share a recorded argument shape (:func:`.overlap.shape_signatures`) are compared,
at most *max_pairs* of them in sorted order; what was cut is named in ``truncated``. Everything is deterministic for
the same library, records and bounds.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
from collections.abc import Iterable, Mapping
from itertools import combinations
from pathlib import Path
from typing import Any

#: Distinct recorded argument lists kept per function, and their total JSON size.
MAX_INPUTS = 6
MAX_INPUT_BYTES = 256 * 1024
#: Bounds of one comparison: each call, and the whole box.
CALL_TIMEOUT_S = 2.0
BOX_TIMEOUT_S = 180.0
#: Function pairs compared per call, in sorted order.
MAX_PAIRS = 2000
#: The largest result file the host reads back.
MAX_RESULT_BYTES = 32 * 1024 * 1024
RULE = "behaviour"
SAME, DIFFER, NO_CASE = "same", "differ", "no shared case"


def _canon(obj: object) -> str:
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def _replayable(value: object) -> bool:
    """Whether a recorded JSON form rebuilds the value it was recorded from closely enough to call with: an object
    recorded by its attributes or bytes recorded by their digest does not."""
    if isinstance(value, dict):
        if "__type__" in value or "__bytes__" in value:
            return False
        return all(_replayable(v) for v in value.values())
    if isinstance(value, list):
        return all(_replayable(v) for v in value)
    return True


def recorded_inputs(calls: Iterable[Any]) -> list[list[list]]:
    """A function's recorded argument lists from its example calls (:class:`.example_calls.ExampleCall`): each call
    that returned with every argument recorded whole and replayable, as ``[[name, json], ...]``; distinct lists in
    call-key order, at most :data:`MAX_INPUTS` and :data:`MAX_INPUT_BYTES` in all."""
    out: list[list[list]] = []
    seen: set[str] = set()
    size = 0
    for c in sorted(calls, key=lambda c: c.key):
        if c.outcome != "returned":
            continue
        if any(a.json is None or a.cut or not _replayable(a.json) for a in c.args):
            continue
        rec = [[a.name, a.json] for a in c.args]
        text = _canon(rec)
        if text in seen:
            continue
        if size + len(text) > MAX_INPUT_BYTES:
            break
        seen.add(text)
        size += len(text)
        out.append(rec)
        if len(out) >= MAX_INPUTS:
            break
    return out


def input_id(rec: list) -> str:
    """The id of one recorded argument list: the digest of its canonical JSON."""
    return hashlib.sha256(_canon(rec).encode("utf-8")).hexdigest()[:24]


def _box_script() -> Path:
    return Path(__file__).with_name("behaviour_box.py")


def run_box(
    tree: Path,
    functions: Mapping[str, tuple[str, str]],
    inputs: Mapping[str, list],
    runs: Iterable[tuple[str, str]],
    *,
    python: Path | None = None,
    call_timeout_s: float = CALL_TIMEOUT_S,
    timeout_s: float = BOX_TIMEOUT_S,
    confined: Any = None,
) -> tuple[dict[tuple[str, str], tuple[str, str | None]], str | None]:
    """Run each ``(function, input id)`` of *runs* in one confined box over the library at *tree*.

    *functions*: item -> (module, name); *inputs*: input id -> recorded argument list. Returns ``(results, problem)``:
    ``(item, input id) -> (outcome, result digest or None)``, and None or why the box gave no readable result (then
    the results are empty). *confined* replaces :func:`.sandbox_run.run_confined` (tests).
    """
    from . import sandbox_run

    todo = sorted({(str(f), str(i)) for f, i in runs if f in functions and i in inputs})
    if not todo:
        return {}, None
    job = {
        "functions": {f: list(functions[f]) for f in sorted({f for f, _ in todo})},
        "inputs": {i: inputs[i] for i in sorted({i for _, i in todo})},
        "runs": [list(r) for r in todo],
        "call_timeout_s": float(call_timeout_s),
    }
    run = confined or sandbox_run.run_confined
    work = Path(tempfile.mkdtemp(prefix="memory-behaviour-"))
    try:
        jobdir, outdir, script = work / "job", work / "out", work / "script"
        for d in (jobdir, outdir, script):
            d.mkdir()
        (jobdir / "job.json").write_text(json.dumps(job), encoding="utf-8")
        shutil.copyfile(_box_script(), script / "behaviour_box.py")
        r = run(
            [str(python or sandbox_run.PYTHON), "-I", "-B", "/script/behaviour_box.py"],
            ro={Path(tree): "/memory", jobdir: "/job", script: "/script"},
            rw={outdir: "/out"},
            cwd="/memory",
            timeout_s=timeout_s,
            env={},
        )
        if getattr(r, "timed_out", False):
            return {}, "the comparison box timed out"
        path = outdir / "result.json"
        try:
            if (
                path.is_symlink()
                or not path.is_file()
                or path.stat().st_size > MAX_RESULT_BYTES
            ):
                return {}, "the comparison box left no readable result"
            data = json.loads(path.read_text(encoding="utf-8", errors="replace"))
        except (OSError, ValueError):
            return {}, "the comparison box left no readable result"
    finally:
        shutil.rmtree(work, ignore_errors=True)
    wanted = set(todo)
    out: dict[tuple[str, str], tuple[str, str | None]] = {}
    rows = data.get("results") if isinstance(data, dict) else None
    for row in rows if isinstance(rows, list) else []:
        if not isinstance(row, list) or len(row) != 4:
            continue
        item, iid, outcome, digest = row
        key = (item, iid)
        if key not in wanted or not isinstance(outcome, str):
            continue
        out[key] = (
            outcome[:20],
            digest if isinstance(digest, str) and len(digest) == 64 else None,
        )
    return out, None


def compare(
    a: str,
    b: str,
    own: Mapping[str, Iterable[str]],
    results: Mapping[tuple[str, str], tuple[str, str | None]],
) -> dict:
    """The verdict on *a* and *b* from *results*: ``same`` (each has at least one own list where both return, and
    every list where both return gives equal results), ``differ`` (some list where both return gives different
    results) or ``no shared case``; with the counts behind it. *own*: item -> its input ids.
    """
    mine = {f: set(own.get(f, ())) for f in (a, b)}
    agree = differ = 0
    shared = {a: 0, b: 0}
    for iid in sorted(mine[a] | mine[b]):
        ra, rb = results.get((a, iid)), results.get((b, iid))
        if not ra or not rb or ra[0] != "returned" or rb[0] != "returned":
            continue
        if ra[1] is None or rb[1] is None:
            continue
        if ra[1] == rb[1]:
            agree += 1
        else:
            differ += 1
        for f in (a, b):
            shared[f] += iid in mine[f]
    verdict = DIFFER if differ else SAME if shared[a] and shared[b] else NO_CASE
    return {
        "verdict": verdict,
        "agree": agree,
        "differ": differ,
        "own_cases": [shared[a], shared[b]],
        "own_lists": [len(mine[a]), len(mine[b])],
    }


def shape_pairs(
    shapes: Mapping[str, Mapping],
    functions: Iterable[str],
    *,
    within: Iterable[str] | None = None,
) -> list[tuple[str, str]]:
    """Sorted pairs of *functions* that share a recorded argument shape (*shapes*: the shape rows), each pair
    sorted; with *within*, only pairs with at least one function of it."""
    from .overlap import shape_signatures

    sig = shape_signatures(shapes)
    names = sorted(set(functions))
    focus = set(within) if within is not None else None
    out = []
    for a, b in combinations(names, 2):
        if focus is not None and a not in focus and b not in focus:
            continue
        if sig.get(a, set()) & sig.get(b, set()):
            out.append((a, b))
    return out


def compare_pairs(
    tree: Path,
    pairs: list[tuple[str, str]],
    inputs: Mapping[str, list[list]],
    functions: Mapping[str, tuple[str, str]],
    *,
    max_pairs: int = MAX_PAIRS,
    python: Path | None = None,
    confined: Any = None,
) -> tuple[dict[tuple[str, str], dict], list[str]]:
    """Compare each of *pairs* (at most *max_pairs*, in order) on the library at *tree*: each function runs on its own
    and its partner's recorded lists (*inputs*: item -> lists). Returns ``(pair -> compare() row, truncated)``.
    """
    cut: list[str] = []
    if len(pairs) > max_pairs:
        cut.append(
            f"behaviour: {len(pairs) - max_pairs} of {len(pairs)} pairs not compared"
        )
        pairs = pairs[:max_pairs]
    pool: dict[str, list] = {}
    own: dict[str, list[str]] = {}
    for item, lists in inputs.items():
        own[item] = []
        for rec in lists:
            iid = input_id(rec)
            pool.setdefault(iid, rec)
            own[item].append(iid)
    runs = set()
    for a, b in pairs:
        if not own.get(a) or not own.get(b):
            continue
        for f in (a, b):
            for iid in own[a] + own[b]:
                runs.add((f, iid))
    results, problem = run_box(
        tree,
        functions,
        pool,
        runs,
        python=python,
        confined=confined,
    )
    if problem is not None:
        cut.append(f"behaviour: {problem}")
    return {(a, b): compare(a, b, own, results) for a, b in pairs}, cut


def reason(a: str, b: str, row: Mapping) -> str:
    """The reason line of a behaviour candidate."""
    na, nb = row["own_cases"]
    return (
        f"same results on every recorded argument list both return on ({row['agree']}): {na} of {a}'s "
        f"{row['own_lists'][0]} and {nb} of {b}'s {row['own_lists'][1]}"
    )


def candidates(
    tree: Path,
    shapes: Mapping[str, Mapping],
    inputs: Mapping[str, list[list]],
    *,
    max_pairs: int = MAX_PAIRS,
    python: Path | None = None,
    confined: Any = None,
) -> tuple[list[dict], list[str]]:
    """The behaviour overlap candidates of the library at *tree*: ``({"rule", "items", "reasons", "agree"}, ...)``
    for each pair of its functions that :func:`compare` calls ``same``, sorted by items, and what was cut.
    *shapes*: the commit's shape rows; *inputs*: each function's recorded argument lists.
    """
    from . import layout

    lib = layout.discover(Path(tree))
    functions = {f.item_id: (f.module, f.name) for f in lib.functions}
    pairs = shape_pairs(shapes, [f for f in functions if inputs.get(f)])
    rows, cut = compare_pairs(
        tree,
        pairs,
        {f: inputs[f] for f in functions if inputs.get(f)},
        functions,
        max_pairs=max_pairs,
        python=python,
        confined=confined,
    )
    out = [
        {
            "rule": RULE,
            "items": [a, b],
            "reasons": [reason(a, b, row)],
            "agree": row["agree"],
        }
        for (a, b), row in sorted(rows.items())
        if row["verdict"] == SAME
    ]
    return out, cut


def inputs_at(
    evidence: Any,
    digests: Mapping[str, str],
) -> dict[str, list[list]]:
    """item -> the recorded argument lists the evidence store keeps for its current definition (*digests*: item ->
    body digest, as :func:`.catalogue.body_digest`); an item with none is left out."""
    rows = evidence.function_inputs(sorted(digests))
    out = {}
    for item, digest in digests.items():
        lists = (rows.get(item) or {}).get(digest)
        if lists:
            out[item] = lists
    return out


def body_digests(tree: Path) -> dict[str, str]:
    """item -> body digest (:func:`.catalogue.body_digest`) of every function of the library at *tree*, as the gate
    keys its recorded inputs."""
    from . import layout
    from .catalogue import body_digest

    return {
        item: body_digest(body)
        for item, body in layout.function_bodies(Path(tree)).items()
    }
