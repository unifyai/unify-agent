"""A procedure's recipe, and the gate's run of it on every case its episode observed.

**The recipe.** Every landed procedure carries one callable in its own module, ``apply(request) -> result`` (named
in its manifest entry's ``recipe``): it is handed the request as the harness hands it (its text, a ``str``, with
``.text`` and ``.data``, :mod:`.request_value`), reads from it every parameter it needs, calls the general functions
and returns what the job hands back. A procedure landed from a capture gets one written for it
(:func:`.land_capture.recipe_block`): its reader on the request, its ``applies`` as an optional precondition, then the
procedure.

**The cases.** A procedure's manifest entry lists its case files (``cases``), each made by the fixture tool from the
recording (:func:`.fixtures.make_case`): the source request with the value its episode handed back, and each input
with the output the request or the environment showed. The gate re-derives each from the recording
(:func:`.fixtures.read_case`), so none can be invented.

**The run** (:func:`run_case`, :func:`check_cases`). The recipe runs confined, one box per case (so no case can carry
state into the next): the library read-only at ``/memory``, the test kit at ``/kit``, the case's input read-only at
``/case``, only ``/out`` writable, no network and an empty environment; the expected output stays on the host. The box
writes its outcome label and, when the recipe returned, its result as JSON text; the host parses that as data and
compares it with the expected output through the one comparison (:mod:`.value_diff`). What leaves the check is the
outcome label, the counts and the difference report.
"""

from __future__ import annotations

import json
import shutil
from collections.abc import Callable, Iterable, Mapping
from pathlib import Path
from typing import Any

from . import testkit
from . import value_diff
from .procedures import PROCEDURE_S
from .request_value import REQUEST_CLASS, request_data
from .sandbox_run import SandboxResult, run_confined

#: Each case's wall bound, the procedure runners' own.
RECIPE_S = PROCEDURE_S
#: The largest result a box may write, in bytes (the capture runs' own bound).
MAX_RESULT_BYTES = 8 * 1024 * 1024

_RUN = r"""
import importlib, json, sys
sys.path[:0] = ["/memory", "/kit"]
sys.dont_write_bytecode = True
""" + REQUEST_CLASS + r"""

with open("/case/case.json", encoding="utf-8") as f:
    spec = json.load(f)
out = {}
try:
    module, _, name = spec["recipe"].partition(":")
    fn = getattr(importlib.import_module(module), name)
    result = fn(Request(spec["text"], spec["data"]))
    try:
        out["value"] = json.dumps(result, sort_keys=True, ensure_ascii=False, allow_nan=False)
        out["outcome"] = "returned"
    except (TypeError, ValueError, RecursionError):
        out["outcome"] = "not_json"
except BaseException as exc:
    out["outcome"] = "refused" if any(c.__name__ == "MemoryInputError" for c in type(exc).__mro__) else "raised"
    out["error_class"] = type(exc).__name__
with open("/out/result.json", "w", encoding="utf-8") as f:
    json.dump(out, f)
"""


def _cls(name: Any) -> str:
    return name if isinstance(name, str) and name.isidentifier() and len(name) <= 64 else "an exception"


def run_case(
    recipe: str,
    text: str,
    *,
    tree: Path,
    python: Path,
    work: Path,
    runner: Callable[..., SandboxResult] = run_confined,
    timeout_s: float = RECIPE_S,
) -> dict:
    """Run *recipe* (a function id of the library at *tree*) confined on the request *text*. Returns ``{"outcome"}``
    (``returned``, ``refused``: it raised ``MemoryInputError``, ``raised``, ``not_json`` or ``not run``), with
    ``value`` (the parsed result) when it returned, ``error_class`` when it raised and ``why`` when it was not run."""
    from .captures import _read_result

    case, out, kit = work / "case", work / "out", work / "kit"
    case.mkdir(parents=True)
    out.mkdir()
    try:
        testkit.stage_package(kit / testkit.PACKAGE)
        (case / "run.py").write_text(_RUN, encoding="utf-8")
        (case / "case.json").write_text(
            json.dumps({"recipe": recipe, "text": str(text), "data": request_data(str(text))}, ensure_ascii=False),
            encoding="utf-8",
        )
        res = runner(
            [str(python), "-I", "/case/run.py"],
            ro={tree: "/memory", case: "/case", kit: "/kit"},
            rw={out: "/out"},
            cwd="/memory",
            timeout_s=timeout_s,
            env={},
        )
        if res.timed_out:
            return {"outcome": "not run", "why": f"it did not finish within {timeout_s:g} s"}
        doc = _read_result(out / "result.json")
    finally:
        shutil.rmtree(work, ignore_errors=True)
    if doc is None or doc.get("outcome") not in ("returned", "refused", "raised", "not_json"):
        return {"outcome": "not run", "why": "it could not be run (no result)"}
    row: dict = {"outcome": doc["outcome"]}
    if doc["outcome"] == "returned":
        try:
            row["value"] = json.loads(doc.get("value"))
        except (TypeError, ValueError, RecursionError):
            return {"outcome": "not run", "why": "it could not be run (an unreadable result)"}
    if doc["outcome"] in ("refused", "raised"):
        row["error_class"] = _cls(doc.get("error_class"))
    return row


def verdict(run: Mapping[str, Any], expected: Any) -> str | None:
    """None when the recipe's run gave *expected*; else what went wrong: the difference report
    (:func:`.value_diff.difference`) when it returned another value, or how it ended."""
    outcome = run.get("outcome")
    if outcome == "returned":
        return value_diff.difference(expected, run.get("value"))
    if outcome == "refused":
        return "it refused the request (MemoryInputError): the recipe must apply to every case its episode observed"
    if outcome == "raised":
        return f"it raised {run.get('error_class') or 'an exception'}"
    if outcome == "not_json":
        return "its result is not a JSON value"
    return str(run.get("why") or "it could not be run")


def check_cases(
    recipe: str,
    cases: Iterable[Mapping[str, Any]],
    *,
    tree: Path,
    python: Path,
    work: Path,
    runner: Callable[..., SandboxResult] = run_confined,
    timeout_s: float = RECIPE_S,
) -> list[str | None]:
    """Per case (``{"input", "expected"}``), None when *recipe* gives its expected output on its input, else the
    report (:func:`verdict`). Each case runs in a box of its own."""
    out: list[str | None] = []
    for n, case in enumerate(cases):
        run = run_case(
            recipe,
            str(case["input"]),
            tree=tree,
            python=python,
            work=work / f"case-{n}",
            runner=runner,
            timeout_s=timeout_s,
        )
        out.append(verdict(run, case["expected"]))
    return out
