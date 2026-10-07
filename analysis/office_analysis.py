"""Preregistered analysis for cleanslate-v1 office cells (PREREG-office-v1).

    python3 analysis/office_analysis.py --cell DIR [--cell DIR ...] [--baseline LABEL=CELLDIR ...]
                                        [--fit] [--out report.json]

Inputs are cell folders (records.jsonl, run.json). cleanslate cells come from kit/run_cell.sh. Unify
baseline cells are the everyday office runner's (records with passed / usd / usd_priced /
unpriced_calls / calls / visit). Task ids are evaluation labels here and nowhere else.

Per arm (cells grouped by arm label), it reports:
  * solved per run, mean, spread, and a run-then-task bootstrap interval (2,000 draws, seed 0);
  * USD per instance (provider-reported charges; a run with unpriced calls reports its priced subtotal
    and the unpriced count, and its total stays unknown), first visits vs returns, and the ratio;
  * calls per instance;
  * hold rate (instances with at least one smell hold), false-hold rate (holds whose held state the
    checker passes), offers made, offers accepted, offers accepted and correct, references shown;
  * with --fit (needs the office data and bubblewrap): the fit rate. For each return visit whose job
    has a procedure stored by an earlier visit in the same run, the procedure is bound to the
    return's request, run on the return's own files in a fresh workspace, and scored by the
    checker. Reported over returns with a procedure, with bindable / same-request-shape counts.
Then it evaluates the preregistered thresholds (THRESHOLDS below, mirrored in the prereg).
"""
from __future__ import annotations

import argparse
import json
import random
import shutil
import statistics
import sys
import tempfile
from decimal import Decimal
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROTO = HERE.parent
sys.path[:0] = [str(PROTO), str(PROTO / "adapters")]

THRESHOLDS = {
    "S1_return_cost_ratio_max": "0.50",        # cleanslate-full: USD of returns / USD of the same jobs' first visits
    "S1_return_cost_vs_lean_all_max": "0.70",  # cleanslate-full return USD / A1 return USD
    "S2_solved_margin_min": -1.0,              # mean solved (of 24) minus A1's mean, at least
    "S3_offer_precision_min": 0.95,            # accepted ready offers that pass
    "S3_lookalike_wrong_accepts_max": 0,
    "S4_false_hold_rate_max": 0.10,            # false holds / correct deliveries
    "H2_wrong_offer_accepts_per_run_max": 1,
    "H3_false_hold_share_of_holds_max": 0.20,
    "H4_usd_per_run_vs_lean_all_max": "2.0",
    "H5_solved_below_upstream_max": 3.0,
}


def load_cell(cell: Path) -> dict:
    records = [json.loads(x) for x in (cell / "records.jsonl").read_text().splitlines() if x.strip()]
    run = json.loads((cell / "run.json").read_text()) if (cell / "run.json").exists() else {}
    return {"dir": str(cell), "records": records, "run": run}


READOUT_ARMS = {"unify@main": "A0", "overhauled": "A1"}


def load_readout(path: Path) -> dict[str, list[dict]]:
    """EVAL's office readout rows (artifacts/everyday-v1/readout-confirmation-v1/office-rows.jsonl):
    one row per instance, arms 'unify@main' (A0, Unify @ main 592685713) and 'overhauled' (A1, lean-all
    af8958e5d). Converted to cells of records in this script's schema, one cell per run_id."""
    cells: dict[tuple, list] = {}
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if r.get("part") != "office":
            continue
        cells.setdefault((READOUT_ARMS.get(r["arm"], r["arm"]), r["run_id"]), []).append({
            "task_id": r["instance_id"], "job": r.get("job"), "visit": "return" if r["variant"] == "return" else "first",
            "passed": r["solved"], "usd": r["usd"], "usd_priced": r["usd"] or "0",
            "unpriced_calls": 0 if r["usd"] is not None else None, "calls": r.get("calls")})
    out: dict[str, list[dict]] = {}
    for (arm, run_id), recs in sorted(cells.items()):
        out.setdefault(arm, []).append({"dir": f"{path}#{run_id}", "records": recs, "run": {"run_id": run_id}})
    return out


def usd_of(rec: dict) -> tuple[Decimal | None, Decimal, int, int]:
    """(known USD or None, priced subtotal, unpriced calls, calls) for either record schema."""
    if "cost" in rec and isinstance(rec["cost"], dict):
        c = rec["cost"]
        return (Decimal(c["usd"]) if c.get("usd") is not None else None, Decimal(c.get("priced_usd") or "0"),
                c.get("unpriced_calls") or 0, c.get("calls") or 0)
    priced = Decimal(str(rec.get("usd_priced") or "0"))
    unpriced = rec.get("unpriced_calls") or 0
    return (Decimal(str(rec["usd"])) if rec.get("usd") is not None else None, priced, unpriced, rec.get("calls") or 0)


def cell_metrics(cell: dict) -> dict:
    recs = cell["records"]
    n = len(recs)
    solved = sum(r.get("passed") is True for r in recs)
    priced = sum((usd_of(r)[1] for r in recs), Decimal(0))
    unpriced = sum(usd_of(r)[2] for r in recs)
    calls = sum(usd_of(r)[3] for r in recs)

    def mean_usd(visit):
        sub = [usd_of(r)[1] for r in recs if r.get("visit") == visit]
        return (sum(sub, Decimal(0)) / len(sub)) if sub else None
    first, ret = mean_usd("first"), mean_usd("return")
    jobs_returned = {r.get("job") for r in recs if r.get("visit") == "return"}
    firsts_of_returned = [usd_of(r)[1] for r in recs if r.get("visit") == "first" and r.get("job") in jobs_returned]
    returns = [usd_of(r)[1] for r in recs if r.get("visit") == "return"]
    matched = (sum(returns, Decimal(0)) / sum(firsts_of_returned, Decimal(0))) if firsts_of_returned and \
        sum(firsts_of_returned, Decimal(0)) else None
    holds = [h for r in recs for h in r.get("held_would_have_passed", [])]
    out = {"n": n, "solved": solved, "unknown": sum(r.get("passed") is None for r in recs),
           "usd_priced": str(priced), "unpriced_calls": unpriced, "usd": str(priced) if unpriced == 0 else None,
           "usd_per_instance_priced": str(priced / n) if n else None, "calls_per_instance": calls / n if n else None,
           "usd_first_mean": None if first is None else str(first), "usd_return_mean": None if ret is None else str(ret),
           "return_over_first": None if not first or ret is None else str(ret / first),
           # primary: USD of the returns over USD of the same jobs' first visits (7 recurring jobs)
           "return_over_own_first": None if matched is None else str(matched),
           "solved_first": sum(r.get("passed") is True for r in recs if r.get("visit") == "first"),
           "solved_return": sum(r.get("passed") is True for r in recs if r.get("visit") == "return")}
    if any("holds" in r for r in recs):
        correct = sum(r.get("passed") is True for r in recs)
        false_holds = sum(h is True for h in holds)
        accepted = [r for r in recs if r.get("used_offer")]
        out.update({
            "instances_held": sum(bool(r.get("holds")) for r in recs), "holds": len(holds),
            "hold_rate": sum(bool(r.get("holds")) for r in recs) / n if n else None,
            "false_holds": false_holds, "holds_unknown": sum(h is None for h in holds),
            "false_hold_rate_of_correct": false_holds / correct if correct else None,
            "false_hold_share_of_holds": false_holds / len(holds) if holds else None,
            "offers_made": sum(bool(r.get("offers")) for r in recs), "offers_accepted": len(accepted),
            "offers_accepted_correct": sum(r.get("passed") is True for r in accepted),
            "offers_accepted_wrong": sum(r.get("passed") is False for r in accepted),
            "references_shown": sum(bool(r.get("references")) for r in recs),
            "noop_cells": sum(r.get("noop_cells") or 0 for r in recs), "caps_hit": sum(bool(r.get("cap")) for r in recs)})
    return out


def bootstrap_mean_solved(runs: list[list[dict]], draws: int = 2000, seed: int = 0) -> list[float] | None:
    """Resample runs, then tasks within each drawn run; the statistic is mean solved per run."""
    if not runs:
        return None
    rng = random.Random(seed)
    stats = []
    for _ in range(draws):
        picked = [runs[rng.randrange(len(runs))] for _ in runs]
        totals = [sum(r[rng.randrange(len(r))].get("passed") is True for _ in r) for r in picked if r]
        stats.append(statistics.mean(totals))
    stats.sort()
    return [stats[int(0.025 * draws)], stats[int(0.975 * draws) - 1]]


def fit_for_cell(cell: dict) -> list[dict]:
    import office
    from cleanslate.memory import bind, similarity
    from cleanslate.sandbox import Sandbox
    recs = cell["records"]
    prompts = {e["task_id"]: e["prompt"] for e in office.load_entries([r["task_id"] for r in recs])}
    out = []
    for r in recs:
        if r.get("visit") != "return":
            continue
        earlier = [x for x in recs if x["index"] < r["index"] and x["job"] == r["job"]]
        pid = next((x.get("stored") or x.get("used_offer") for x in reversed(earlier)
                    if x.get("stored") or x.get("used_offer")), None)
        store_file = Path(cell["dir"]) / f"procedures-before-{r['index']:02d}.json"
        item = {"task_id": r["task_id"], "job": r["job"], "procedure": pid}
        if not pid or not store_file.exists():
            out.append({**item, "has_procedure": False})
            continue
        proc = next((p for p in json.loads(store_file.read_text()) if p["id"] == pid), None)
        if proc is None:
            out.append({**item, "has_procedure": False})
            continue
        req = prompts[r["task_id"]]
        case = max(proc["cases"], key=lambda c: similarity(req, c["request"]))
        params, extra = bind(case["request"], case["params"], req)
        item.update({"has_procedure": True, "bindable": params is not None, "same_request_shape": params is not None
                     and not extra, "differences": extra})
        if params is None:
            out.append({**item, "passed": False, "why": "parameters could not be bound"})
            continue
        with tempfile.TemporaryDirectory(prefix="fit-") as tmp:
            ws = Path(tmp, "ws")
            shutil.copytree(office.OFFICE_DIR / r["task_id"] / "workspace", ws)
            with Sandbox(str(ws), hidden=office.hidden_paths(), timeout=60) as sb:
                sb.set(**params)
                ran = sb.run(proc["code"])
            scored = Path(tmp, "out")
            office.copy_out(ws, scored)
            verdict = office.score(r["task_id"], scored, Path(tmp, "score"))
        out.append({**item, "ran_ok": ran.get("ok"), "passed": verdict.get("passed")})
    return out


def _mean_dec(values):
    values = [Decimal(v) for v in values if v is not None]
    return sum(values, Decimal(0)) / len(values) if values else None


def evaluate(report: dict) -> dict:
    """The preregistered criteria. Baselines must be labelled A0 (Unify @ main) and A1 (lean-all).
    A criterion whose inputs are missing is None (not evaluated), never a pass."""
    arms, T = report["arms"], THRESHOLDS
    full, a0, a1 = arms.get("cleanslate-full") or arms.get("cleanslate-full-no-offers"), arms.get("A0"), arms.get("A1")
    out = {}

    def put(name, value, ok):
        out[name] = {"value": None if value is None else str(value), "pass": None if value is None else bool(ok)}
    if full:
        runs = full["runs"]
        ret = _mean_dec(m["usd_return_mean"] for m in runs)
        ratio = _mean_dec(m["return_over_own_first"] for m in runs)
        put("S1_return_cost_ratio", ratio, ratio is not None and ratio <= Decimal(T["S1_return_cost_ratio_max"]))
        if a1:
            a1_ret = _mean_dec(m["usd_return_mean"] for m in a1["runs"])
            r = ret / a1_ret if ret is not None and a1_ret else None
            put("S1_return_cost_vs_lean_all", r, r is not None and r <= Decimal(T["S1_return_cost_vs_lean_all_max"]))
            margin = full["solved_mean"] - a1["solved_mean"]
            put("S2_solved_margin_vs_lean_all", margin, margin >= T["S2_solved_margin_min"])
            usd_full = _mean_dec(m["usd_priced"] for m in runs)
            usd_a1 = _mean_dec(m["usd_priced"] for m in a1["runs"])
            r = usd_full / usd_a1 if usd_full is not None and usd_a1 else None
            put("H4_usd_per_run_vs_lean_all", r, r is not None and r <= Decimal(T["H4_usd_per_run_vs_lean_all_max"]))
        acc = sum(m.get("offers_accepted", 0) for m in runs)
        good = sum(m.get("offers_accepted_correct", 0) for m in runs)
        put("S3_offer_precision", good / acc if acc else None, acc and good / acc >= T["S3_offer_precision_min"])
        out["S3_lookalike_wrong_accepts"] = {"value": None, "pass": None,
                                             "note": "office core has no look-alike instances; not applicable"}
        correct = sum(m["solved"] for m in runs)
        false_holds = sum(m.get("false_holds", 0) for m in runs)
        holds = sum(m.get("holds", 0) for m in runs)
        put("S4_false_hold_rate", false_holds / correct if correct else None,
            correct and false_holds / correct <= T["S4_false_hold_rate_max"])
        worst = max((m.get("offers_accepted_wrong", 0) for m in runs), default=None)
        put("H2_wrong_offer_accepts_worst_run", worst, worst is not None and worst <= T["H2_wrong_offer_accepts_per_run_max"])
        put("H3_false_hold_share", false_holds / holds if holds else None,
            holds and false_holds / holds <= T["H3_false_hold_share_of_holds_max"])
        if a0:
            gap = a0["solved_mean"] - full["solved_mean"]
            put("H5_solved_below_upstream", gap, gap < T["H5_solved_below_upstream_max"])
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--cell", action="append", default=[], help="a cleanslate cell folder")
    ap.add_argument("--baseline", action="append", default=[], help="LABEL=CELLDIR for a Unify office cell")
    ap.add_argument("--baseline-readout", help="EVAL's office-rows.jsonl (A0 and A1 runs)")
    ap.add_argument("--fit", action="store_true")
    ap.add_argument("--out")
    a = ap.parse_args(argv)
    arms: dict[str, list[dict]] = {}
    for d in a.cell:
        c = load_cell(Path(d))
        label = ("cleanslate-full" if c["run"].get("offers", True) else "cleanslate-full-no-offers") \
            if c["run"].get("memory", True) else "cleanslate-no-memory"
        arms.setdefault(label, []).append(c)
    for spec in a.baseline:
        label, d = spec.split("=", 1)
        arms.setdefault(label, []).append(load_cell(Path(d)))
    if a.baseline_readout:
        for label, cells in load_readout(Path(a.baseline_readout)).items():
            arms.setdefault(label, []).extend(cells)
    report = {"thresholds": THRESHOLDS, "arms": {}}
    for label, cells in arms.items():
        per_run = [cell_metrics(c) for c in cells]
        solved = [m["solved"] for m in per_run]
        entry = {"runs": per_run, "cells": [c["dir"] for c in cells], "solved_per_run": solved,
                 "solved_mean": statistics.mean(solved) if solved else None,
                 "solved_spread": [min(solved), max(solved)] if solved else None,
                 "solved_interval_95": bootstrap_mean_solved([c["records"] for c in cells])}
        if a.fit and label == "cleanslate-full":
            fits = [f for c in cells for f in fit_for_cell(c)]
            with_proc = [f for f in fits if f.get("has_procedure")]
            entry["fit"] = {"returns": len(fits), "with_procedure": len(with_proc),
                            "bindable": sum(bool(f.get("bindable")) for f in with_proc),
                            "same_request_shape": sum(bool(f.get("same_request_shape")) for f in with_proc),
                            "passed": sum(f.get("passed") is True for f in with_proc),
                            "fit_rate": (sum(f.get("passed") is True for f in with_proc) / len(with_proc))
                            if with_proc else None, "items": fits}
        report["arms"][label] = entry
    report["criteria"] = evaluate(report)
    text = json.dumps(report, indent=1, default=str)
    if a.out:
        Path(a.out).write_text(text + "\n")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
