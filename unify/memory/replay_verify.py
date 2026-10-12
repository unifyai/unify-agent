"""Verify a function by replaying the recorded instances it replaces (as in ASI and CRAFT).

A function item may say which pieces of the actor's recorded code it replaces: ``replaces: [{"episode", "cell",
"lines": [a, b], "call": "<statement>"}]`` in its manifest entry, where ``call`` is the statement that does the same
by calling the function (``comps = components(g, 8)``). For each instance, in a bubblewrap box with no network, the
confined runner's seccomp filter and the writer's tree read-only at /library on the import path, three runs each
start in their own pristine copy of the episode's recorded working tree (an empty one when it has none):

1. the episode's successful cells up to and including that cell, twice (the original runs);
2. the same with lines ``a..b`` of the last cell replaced by ``call`` (the function imported).

The recording is the oracle, so before anything is compared the original run must reproduce it:

- **recording check**: each replayed cell's stdout must equal the output the episode recorded for it (see
  :func:`recorded_readings` for how the recorder's own additions are read). An original run that raises where the
  recording ran cleanly, or prints something else in any cell, makes the instance
  ``unknown: oracle unsuitable: ...``;
- **determinism check**: the two original runs must end with the same values, file changes and stdout; otherwise
  the instance is ``unknown: oracle unsuitable: nondeterministic``.

An unknown instance is neither verified nor differs: it gives no test case and is counted apart
(:func:`summary`). Only then is the rewritten run compared: the instance is **verified** when its observation equals
the original's: its live-out values (every name bound before line ``a``, every name ``call`` assigns, and every name
both runs end with, scalars and empty containers included), its working tree's file deltas, and its stdout when
printing is the replaced lines' only effect. Otherwise it ``differs``. Alias groups, deleted names and values that
cannot be encoded are recorded as unknown, not compared.

Every compared instance carries ``oracle``: what the two checks found (:func:`verdict`). A cell whose output was not
recorded at all (``None``) cannot be checked; when no replayed cell has a recording the check reports
``recording: "not available"`` and the verdict rests on the determinism check alone, said so in the result rather than
taken as reproduced.

The equivalence needs no reward signal. A verified instance becomes a case of the item's harness-written test
(:func:`test_source`): the call's inputs as recorded at line ``a`` go to the item's replay case file
(:func:`case_file`, ``tests/data/<function>.replay.jsonl``, one line per instance naming its episode, cell and lines),
which the test reads, and its call and expected results stay in the test. The gate re-verifies a new or changed case
file by replaying the instances its lines name (:func:`reverify`), so the inputs a test loads from it are recorded
values. Nothing staged or written by the model runs on the host.
"""

from __future__ import annotations

import ast
import builtins
import json
import re
import shutil
from pathlib import Path
from typing import Callable

from . import fixtures as _fx
from . import mining_behaviour as _mb
from .mining import successful
from .tool_spec import REFUSED

_ITEM = re.compile(
    r"memory\.(?P<pkg>[a-z_][a-z0-9_]*)\.(?P<mod>[a-z_][a-z0-9_]*):(?P<fn>[A-Za-z_][A-Za-z0-9_]*)\Z",
)


def call_names(call: str) -> tuple[list[str], list[str]]:
    """(the names *call* reads, the names it assigns), builtins left out; ValueError when it is not one statement."""
    tree = ast.parse(call)
    if len(tree.body) != 1:
        raise ValueError("call must be one statement")
    stmt = tree.body[0]
    targets: list[str] = []
    if isinstance(stmt, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
        for t in stmt.targets if isinstance(stmt, ast.Assign) else [stmt.target]:
            targets += [n.id for n in ast.walk(t) if isinstance(n, ast.Name)]
    reads = []
    for n in ast.walk(stmt):
        if (
            isinstance(n, ast.Name)
            and isinstance(n.ctx, ast.Load)
            and n.id not in reads
            and not hasattr(builtins, n.id)
        ):
            reads.append(n.id)
    return reads, targets


def rewrite(code: str, lines: tuple[int, int], call: str) -> str:
    rows = code.splitlines()
    a, b = lines
    if not (1 <= a <= b <= len(rows)):
        raise ValueError("lines outside the cell")
    indent = rows[a - 1][: len(rows[a - 1]) - len(rows[a - 1].lstrip())]
    return "\n".join([*rows[: a - 1], indent + call.strip(), *rows[b:]]) + "\n"


def _job(ep, inst: dict, fn_import: str, fn: str) -> dict:
    cell_idx, (a, b) = int(inst["cell"]), tuple(int(x) for x in inst["lines"])
    cells = [c for c in ep.cells if successful(c) and c.index <= cell_idx]
    if not cells or cells[-1].index != cell_idx:
        raise ValueError("the cell did not run without an error")
    reads, targets = call_names(str(inst["call"]))
    return {
        "mode": "verify",
        "cells": [
            {"index": c.index, "code": c.code, "output": c.output} for c in cells
        ],
        "cell": cell_idx,
        "line": a,
        "names": [n for n in reads if n != fn],
        "targets": targets,
        "rewritten": rewrite(cells[-1].code, (a, b), str(inst["call"])),
        "imports": [fn_import],
        "path": "/library",
    }


def _other_effect(first: dict, expect: dict) -> bool:
    """Whether the replaced lines had an effect besides printing: a value the call's case expects, or a file delta."""
    return bool(expect) or bool(first.get("files"))


#: What the actor's execution tool writes in a cell's result besides its stdout: the line naming the session's
#: variables (appended last) and the placeholder for a cell that printed nothing.
VARIABLES_LINE = "[variables]"
NO_OUTPUT = "(no output)"
UNSUITABLE = "unknown: oracle unsuitable"


def recorded_readings(output) -> list[str] | None:
    """The stdout texts (whitespace-stripped) a cell's recorded *output* may stand for; None when it records none.

    The recorded output is the stdout part of the cell's result as the actor saw it. Besides the text as recorded,
    it reads as: the text without a final variables line; empty when what is left is the no-output placeholder;
    empty when what is left is one JSON object (the result's metadata block, which stands alone when the cell
    printed nothing). A run reproduces the cell when its stripped stdout equals any of these.
    """
    if not isinstance(output, str):
        return None
    text = output.strip()
    out = [text]
    rows = text.splitlines()
    if rows and rows[-1].startswith(VARIABLES_LINE):
        text = "\n".join(rows[:-1]).strip()
        out.append(text)
    if text == NO_OUTPUT:
        out.append("")
    try:
        if isinstance(json.loads(text), dict):
            out.append("")
    except ValueError:
        pass
    return out


def recording_check(job: dict, run: dict) -> dict:
    """Whether the original *run* printed, cell by cell, what the episode recorded for the cells in *job*.

    ``recording``: ``reproduced`` (every recorded cell matches), ``not reproduced`` (``differs_at`` lists the cells
    that do not) or ``not available`` (no replayed cell has a recorded output). ``unrecorded`` lists the cells that
    could not be checked.
    """
    printed = run.get("printed") or []
    differs, unrecorded, checked = [], [], 0
    for k, cell in enumerate(job.get("cells") or []):
        readings = recorded_readings(cell.get("output"))
        if readings is None:
            unrecorded.append(cell.get("index"))
            continue
        checked += 1
        got = printed[k] if k < len(printed) else None
        if got not in {_mb.printed_digest(r) for r in readings}:
            differs.append(cell.get("index"))
    state = (
        "not reproduced" if differs else "reproduced" if checked else "not available"
    )
    return {"recording": state, "differs_at": differs, "unrecorded": unrecorded}


def verdict(job: dict, got: dict) -> dict:
    """The instance's verdict from the three runs (:data:`.mining_behaviour.OBS_VERSION`), and its test case when
    verified.

    First the oracle: the original run must not raise, must reproduce the episode's recorded output
    (:func:`recording_check`) and must be deterministic (both original runs end with the same values, file deltas
    and stdout). When it fails either check the verdict is ``unknown: oracle unsuitable: <cause>``, never verified
    and never differs. Each verdict past this point carries ``oracle``: ``{"recording": "reproduced" | "not
    reproduced" | "not available", "differs_at": [cell, ...], "unrecorded": [cell, ...], "deterministic": true |
    false | null}`` (null when the determinism check was not reached).

    Then the comparison: the live-out values (each name bound before the replaced lines, each name the call assigns,
    and each name both runs end with), the working tree's file deltas, and stdout. stdout decides only when the
    replaced lines have no other effect; otherwise a stdout difference is reported as ``stdout_differs``. Alias
    groups, deleted names and values that cannot be encoded are recorded under ``unknown`` and not compared.
    """
    first, second, third = (
        got.get("first", {}),
        got.get("second", {}),
        got.get("third", {}),
    )
    if "error" in first:
        # the recording says these cells ran without an error, so this replay does not reproduce it
        return {
            "verdict": f"{UNSUITABLE}: the original raised {first['error']}",
            "oracle": {
                "recording": "not reproduced",
                "differs_at": [],
                "unrecorded": [],
                "deterministic": None,
            },
        }
    if "error" in third:
        return {"verdict": f"not replayable: rewritten {third['error']}"}
    if any(
        r.get("obs") != _mb.OBS_VERSION
        for r in (first, third, *([] if "error" in second else [second]))
    ):
        return {"verdict": "not replayable: observation version mismatch"}
    oracle = {**recording_check(job, first), "deterministic": None}
    if oracle["recording"] == "not reproduced":
        return {
            "verdict": f"{UNSUITABLE}: the original does not reproduce the recorded output",
            "oracle": oracle,
        }
    oracle["deterministic"] = "error" not in second and all(
        first.get(k) == second.get(k) for k in ("after", "files", "stdout")
    )
    if not oracle["deterministic"]:
        return {"verdict": f"{UNSUITABLE}: nondeterministic", "oracle": oracle}
    if "bound" not in first:
        return {"verdict": "not replayable: line not executed", "oracle": oracle}
    a1, a3 = first["after"], third.get("after", {})
    before = [n for n in first["bound"] if n in a1]
    must = sorted(set(before) | set(job["targets"]) | (set(a1) & set(a3)))
    unknown = {
        "original": first.get("unknown") or {},
        "rewritten": third.get("unknown") or {},
    }
    base = {"unknown": unknown, "oracle": oracle}
    if any(n not in a1 or a3.get(n) != a1[n] for n in must):
        return {"verdict": "differs", "differs_in": "values", **base}
    if first.get("files") != third.get("files"):
        return {"verdict": "differs", "differs_in": "files", **base}
    expect = {n: a1[n] for n in job["targets"]}
    expect |= {
        n: a1[n]
        for n in job["names"]
        if n in a1 and first.get("inputs", {}).get(n) != a1[n]
    }
    tags: dict = {}
    if _other_effect(first, expect):
        if first.get("stdout") != third.get("stdout"):
            tags["stdout_differs"] = True
    else:
        if not any(n for n, _sha in first.get("stdout") or []):
            return {"verdict": "not replayable: no observed effect", **base}
        if first.get("stdout") != third.get("stdout"):
            return {"verdict": "differs", "differs_in": "stdout", **base}
    return {
        "verdict": "verified",
        **tags,
        "files": first.get("files") or [],
        **base,
        "case": {
            "inputs": third.get("inputs", {}),
            "call": job["rewritten_call"],
            "expect": expect,
        },
    }


def summary(results: list[dict]) -> dict:
    """Counts of *results* (from :func:`verify`) by outcome: ``verified``, ``differs``, ``unknown`` (the oracle was
    unsuitable), ``not_replayable`` and ``refused``; ``tested`` is true when at least one instance was verified,
    which needs an instance whose original reproduced its recording. Unknown instances never count as either.
    """
    out = {"verified": 0, "differs": 0, "unknown": 0, "not_replayable": 0, "refused": 0}
    for r in results:
        v = str(r.get("verdict", ""))
        key = (
            "verified" if v == "verified"
            else "differs" if v == "differs"
            else "unknown" if v.startswith("unknown:")
            else "refused" if v.startswith(REFUSED)  # results recorded before replay stopped refusing
            else "not_replayable"
        )  # fmt: skip
        out[key] += 1
    return {**out, "tested": out["verified"] > 0}


def _world(src: Path | None, dest: Path) -> Path:
    """A pristine copy of the episode's recorded working tree (an empty one when it has none) at *dest*."""
    shutil.rmtree(dest, ignore_errors=True)
    if src is not None:
        shutil.copytree(src, dest, symlinks=True)
    else:
        dest.mkdir(parents=True)
    return dest


def verify(
    load: Callable,
    item: str,
    replaces: list,
    tree: Path,
    *,
    trees: dict[str, Path] | None = None,
    scratch: Path,
) -> list[dict]:
    """Each instance's verdict (``verified`` with its case, ``differs``, ``unknown: oracle unsuitable: <cause>`` or
    ``not replayable: <cause>``; see :func:`verdict` for ``oracle``), each stamped with the observation version
    (``obs``). A box failure is ``not replayable`` and marked ``transient``: a caller never keeps it as the
    instance's result. :func:`summary` counts the outcomes."""
    m = _ITEM.match(item)
    if not m:
        return [{"verdict": "not replayable: not a function item id"}]
    fn_import = f"from memory.{m['pkg']}.{m['mod']} import {m['fn']}"
    out = []
    for inst in replaces if isinstance(replaces, list) else []:
        at = (
            {k: inst.get(k) for k in ("episode", "cell", "lines")}
            if isinstance(inst, dict)
            else {}
        )
        at["obs"] = _mb.OBS_VERSION
        try:
            ep = load(str(inst["episode"]))
            job = _job(ep, inst, fn_import, m["fn"])
        except (
            Exception
        ) as exc:  # noqa: BLE001 - a bad entry is a verdict, never a crash
            out.append(
                {
                    **at,
                    "verdict": f"not replayable: {type(exc).__name__}: {str(exc)[:120]}",
                },
            )
            continue
        job["rewritten_call"] = str(inst["call"]).strip()
        # /lib is a link on merged-/usr hosts, so the library goes at /library
        binds = [(str(tree), "/library", False)]
        # three runs (original, original again, rewritten), each in its own pristine copy of the working tree
        src = (trees or {}).get(str(inst["episode"]))
        worlds = [_world(src, scratch / f"w-{len(out)}-{k}") for k in (1, 2, 3)]
        binds += [(str(w), f"/w{k}", True) for k, w in enumerate(worlds, 1)]
        job["cwds"] = ["/w1", "/w2", "/w3"]
        try:
            got = _mb.in_box(job, 900, binds)
        finally:
            for w in worlds:
                shutil.rmtree(w, ignore_errors=True)
        if "box_error" in got:
            out.append(
                {
                    **at,
                    "verdict": f"not replayable: {got['box_error'][:120]}",
                    "transient": True,
                },
            )
            continue
        out.append({**at, **verdict(job, got)})
    return out


#: At most this many recorded instances of new or changed replay case files are replayed again in one gate run;
#: past it the files are not judged.
MAX_REPLAY_REVERIFY = 8

_TEST = '''"""Harness-written: {item} reproduces the recorded instances it replaces."""

import json
from pathlib import Path

from memory.{pkg}.{mod} import {fn}

CASES = json.loads({cases!r})
INPUTS = Path(__file__).parent / "data" / {data!r}


def _dec(d, frozen=False):
    if isinstance(d, list):
        return [_dec(v) for v in d]
    if isinstance(d, dict):
        t, v = d.get("t"), d.get("v")
        if t == "t":
            return tuple(_dec(x, frozen) for x in v)
        if t == "s":
            s = [_dec(x, True) for x in v]
            return frozenset(s) if frozen else set(s)
        if t == "d":
            return {{_dec(k, True): _dec(x) for k, x in v}}
        if t == "f":
            return float(v)
        if t == "dec":
            import decimal
            return decimal.Decimal(v)
        if t in ("date", "dt"):
            import datetime
            return (datetime.date if t == "date" else datetime.datetime).fromisoformat(v)
    return d


def test_{fn}_reproduces_its_recorded_instances():
    lines = [json.loads(x) for x in INPUTS.read_text().splitlines() if x.strip()]
    assert len(lines) == len(CASES), "the replay case file and the test name different instances"
    for case, line in zip(CASES, lines):
        ns = {{"{fn}": {fn}}}
        ns.update({{k: _dec(v) for k, v in line["input"].items()}})
        exec(case["call"], ns)
        for name, want in case["expect"].items():
            assert ns[name] == _dec(want), (case["at"], name)
'''


def case_path(item: str) -> str:
    """The replay case file of *item*: ``memory/<package>/tests/data/<function>.replay.jsonl``."""
    m = _ITEM.match(item)
    if not m:
        raise ValueError(item)
    return f"memory/{m['pkg']}/tests/data/{m['fn']}{_fx.REPLAY_SUFFIX}"


def case_file(item: str, cases: list[dict]) -> tuple[str, bytes]:
    """(the path in the tree, the bytes) of *item*'s replay case file: per verified case (each with ``episode``,
    ``cell``, ``lines`` and ``inputs``), ``{"input": inputs, "source": {"episode", "source":
    "replay:<cell>:<a>-<b>", "slice": null}}``."""
    rows = []
    for c in cases:
        a, b = c["lines"]
        source = {"episode": str(c["episode"]), "source": f"replay:{int(c['cell'])}:{int(a)}-{int(b)}", "slice": None}
        rows.append(json.dumps({"input": c["inputs"], "source": source}, sort_keys=True, ensure_ascii=False))
    return case_path(item), "".join(r + "\n" for r in rows).encode("utf-8")


def test_source(item: str, cases: list[dict]) -> tuple[str, str]:
    """(the test's path in the tree, its source) for *item*'s verified *cases* (each with ``at``); the test reads the
    inputs from :func:`case_file`, written beside it from the same cases in the same order."""
    m = _ITEM.match(item)
    if not m:
        raise ValueError(item)
    rel = f"memory/{m['pkg']}/tests/test_{m['fn'].lower()}_replay.py"
    kept = [{k: c[k] for k in ("at", "call", "expect") if k in c} for c in cases]
    return rel, _TEST.format(
        item=item,
        pkg=m["pkg"],
        mod=m["mod"],
        fn=m["fn"],
        cases=json.dumps(kept, sort_keys=True),
        data=case_path(item).rsplit("/", 1)[1],
    )


def _instance(replaces: list, episode: str, at: tuple[int, int, int]) -> dict | None:
    cell, a, b = at
    for inst in replaces if isinstance(replaces, list) else []:
        if (
            isinstance(inst, dict)
            and str(inst.get("episode")) == episode
            and inst.get("cell") == cell
            and list(inst.get("lines") or []) == [a, b]
        ):
            return inst
    return None


def reverify(
    load: Callable,
    files: dict[str, bytes],
    replaces_of: dict[str, list],
    tree: Path,
    *,
    trees: dict[str, Path] | None = None,
    scratch: Path,
) -> tuple[dict[str, dict], list[str]]:
    """Replay again the instances that the new or changed replay case *files* (path -> bytes) name: (path -> its
    provenance entry, for each file whose every line verifies now with the inputs it holds; reasons). *replaces_of*:
    each function item's manifest ``replaces``. More than :data:`MAX_REPLAY_REVERIFY` instances in all are not
    judged (a reason). Reasons name paths, lines, episodes and cells, never a value."""
    entries: dict[str, dict] = {}
    reasons: list[str] = []
    plan = []
    for path in sorted(files):
        item = next(
            (i for i in sorted(replaces_of) if _ITEM.match(i) and case_path(i) == path),
            None,
        )
        if item is None:
            reasons.append(
                f"[examples:replay] {path}: no function item lists replaces for this case file; replay writes it "
                "from an item's replaces",
            )
            continue
        entry = _fx.assembled_entry(files[path])
        if entry is None or entry.get("kind") != "replay":
            reasons.append(f"[examples:replay] {path} is not a file of replay case lines")
            continue
        inputs = [json.loads(x)["input"] for x in files[path].decode("utf-8").splitlines() if x.strip()]
        insts = []
        for n, row in enumerate(entry["lines"], 1):
            at = _fx.replay_source(row.get("source"))
            inst = _instance(replaces_of[item], str(row.get("episode")), at) if at else None
            if inst is None:
                reasons.append(
                    f"[examples:replay] {path} line {n} names an instance ({row.get('episode')} "
                    f"{row.get('source')}) that {item}'s replaces does not list",
                )
                break
            insts.append(inst)
        else:
            plan.append((path, item, inputs, insts, entry))
    total = sum(len(p[3]) for p in plan)
    if total > MAX_REPLAY_REVERIFY:
        reasons.append(
            f"[examples:replay] {total} recorded instances are named by new or changed replay case files "
            f"({', '.join(p[0] for p in plan)}); at most {MAX_REPLAY_REVERIFY} are replayed again in one pass, so "
            "they are not judged",
        )
        return {}, reasons
    for path, item, inputs, insts, entry in plan:
        got = verify(load, item, insts, tree, trees=trees, scratch=scratch)
        bad = None
        if len(got) != len(insts):
            bad = f"[examples:replay] {path}: {len(got)} of {len(insts)} instances were replayed"
        for n, (g, want) in enumerate(zip(got, inputs), 1):
            if bad is not None:
                break
            verdict_ = str(g.get("verdict", "")).split(":", 1)[0]
            if verdict_ != "verified":
                bad = f"[examples:replay] {path} line {n}: its instance does not verify by replay now ({verdict_})"
            elif _fx.canonical((g.get("case") or {}).get("inputs")) != _fx.canonical(want):
                bad = f"[examples:replay] {path} line {n}: its inputs differ from the inputs the replay records"
        if bad is None:
            entries[path] = entry
        else:
            reasons.append(bad)
    return entries, reasons
