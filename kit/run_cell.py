"""Run one cleanslate-v1 office cell on a bench worker (OPS runs this; see kit/README.md).

    kit/run_cell.sh --arm full|no-memory --run-index N --tasks all|ID,... --order frozen
                    (--fake | --confirm-paid --prereg PATH --prereg-sha256 HEX)
                    [--workbench DIR] [--max-fs-mb N] [--dedicated-user NAME] [--limits-profile worker|local]

Layout (what cost_ledger.py and `bench_workers sync` already read for everyday office cells):
    <workbench>/appworld-r0/<cell>/                 attempt.json, cell.json, run.json, records.jsonl,
                                                    summary.json, cleanup.json, cost-record.jsonl,
                                                    procedures*.json, workspaces/, held/
    <workbench>/runs/<cell>/attempts/<attempt>/costs.jsonl   the lab request journal
<cell> = [fake-]everyday-office-cleanslate-<arm>-r<N>-<UTC stamp>; fake- cells are filed as dry runs.

A loopback --base-url turns --confirm-paid into a rehearsal of the paid path against a local stand-in
(cell prefix rehearse-, filed as a dry run; no prereg needed, the local limits profile allowed).

Exit codes: 0 done and clean; 1 cleanup not verified; 2 refused before starting; 4 runaway guard stop.

Model, effort and caps default to the matched Unify office cells (benchmarks/everyday/ops/office_campaign.py
and artifacts/everyday-v1/cells-conf-office-core-p*.json): openai/gpt-6-luna, effort low, per task
USD 0.50, 200 calls, 900 s. The key is read from OPENROUTER_API_KEY in this process's environment
(OPS's usual secret route); it is checked for presence only, never printed, logged or passed on.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import sys
import tempfile
import time
import urllib.parse
import uuid
from decimal import Decimal
from pathlib import Path

KIT = Path(__file__).resolve().parent
PROTO = KIT.parent
sys.path[:0] = [str(PROTO), str(PROTO / "adapters")]

import office  # noqa: E402
from cleanslate import Caps, ChatClient, CostLedger, Limits, RunawayGuard  # noqa: E402
from cleanslate import limits as L  # noqa: E402
from cleanslate.llm import OPENROUTER  # noqa: E402
from cleanslate.sandbox import ConfinementError, Sandbox  # noqa: E402

MODEL, EFFORT = "openai/gpt-6-luna", "low"
# USD per 1M tokens from the pinned OpenRouter catalogue (runtime-appworld/unify-agent/.catalog/openrouter_models.json,
# sha256 cbb212a90a75ba8b7326ba14dd15eb3548dde6923846c005c2f71eb22a50beae). Used for cap/guard estimates only;
# charges come from the provider's per-request report.
PRICES = {MODEL: {"input": "0.1", "output": "0.5"}}
CATALOGUE = Path.home() / ".local/share/continual-harness-research/runtime-appworld/unify-agent/.catalog/openrouter_models.json"
DEFAULT_WORKBENCH = Path.home() / ".local/share/continual-harness-research/runtime-appworld/continual-arc-baselines/.workbench"
REFUSED, CLEANUP_FAILED, RUNAWAY = 2, 1, 4


def refuse(msg: str) -> int:
    print(json.dumps({"refused": msg}))
    return REFUSED


def catalogue_prices() -> dict | None:
    if not CATALOGUE.is_file():
        return None
    raw = CATALOGUE.read_bytes()
    entry = json.loads(raw)["models"].get(MODEL, {})
    per_m = {k: format(Decimal(str(entry[f])) * 1_000_000, "f").rstrip("0").rstrip(".")
             for k, f in (("input", "input_cost_per_token"), ("output", "output_cost_per_token")) if f in entry}
    return {"sha256": hashlib.sha256(raw).hexdigest(), "prices": per_m}


def preflight(workbench: Path) -> dict:
    """Start one confined workspace with the chosen limits and stop it again."""
    workbench.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="preflight-", dir=workbench) as d:
        with Sandbox(d, hidden=office.hidden_paths()) as sb:
            ok = sb.run("1 + 1").get("last") == "2"
            report = sb.limits_report
    if not ok:
        raise ConfinementError("preflight workspace did not run code")
    return report


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--arm", choices=("full", "no-memory"), required=True)
    ap.add_argument("--run-index", type=int, required=True)
    ap.add_argument("--tasks", default="all")
    ap.add_argument("--order", choices=("frozen",), default="frozen")
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--fake", action="store_true", help="scripted fake model; no provider request")
    mode.add_argument("--confirm-paid", action="store_true")
    ap.add_argument("--prereg")
    ap.add_argument("--prereg-sha256")
    ap.add_argument("--workbench", default=str(DEFAULT_WORKBENCH))
    ap.add_argument("--limits-profile", choices=("worker", "local"), default="worker")
    ap.add_argument("--dedicated-user")
    ap.add_argument("--max-fs-mb", type=int, help="refuse unless the run folder's filesystem is at most this size")
    ap.add_argument("--max-task-usd", default="0.50")
    ap.add_argument("--max-task-calls", type=int, default=200)
    ap.add_argument("--max-wall-s", type=float, default=900.0)
    ap.add_argument("--max-usd-per-hour", default="40")
    ap.add_argument("--cell-timeout", type=float, default=60.0)
    ap.add_argument("--no-offers", action="store_true",
                    help="stored procedures are shown only as reference code: no binding, no pre-computed results")
    ap.add_argument("--no-capture-content", action="store_true",
                    help="do not keep transcripts (the prompt, the model's replies and the observations)")
    ap.add_argument("--base-url", default=OPENROUTER,
                    help="a loopback URL makes this a rehearsal of the paid path against a local stand-in (no spend)")
    a = ap.parse_args(argv)

    host = urllib.parse.urlparse(a.base_url).hostname
    rehearsal = a.confirm_paid and host in ("127.0.0.1", "localhost", "::1")
    if a.confirm_paid and not rehearsal and a.base_url != OPENROUTER:
        return refuse("paid cells use the OpenRouter route of the matched Unify cells")
    prereg = None
    if rehearsal:
        if not os.environ.get("OPENROUTER_API_KEY"):
            return refuse("OPENROUTER_API_KEY is not in this process's environment")
    elif a.confirm_paid:
        if a.limits_profile != "worker":
            return refuse("paid cells run only with the worker limits profile")
        if not os.environ.get("OPENROUTER_API_KEY"):
            return refuse("OPENROUTER_API_KEY is not in this process's environment")
        if not (a.prereg and a.prereg_sha256):
            return refuse("paid cells need --prereg and --prereg-sha256 (a frozen prereg and MAIN's GO)")
        raw = Path(a.prereg).read_bytes()
        if hashlib.sha256(raw).hexdigest() != a.prereg_sha256:
            return refuse("the prereg file does not match --prereg-sha256")
        text = raw.decode("utf-8", "replace")
        title = text.splitlines()[0] if text else ""
        if not re.search(r"^Status: FROZEN \S+ by MAIN", text, re.M) or re.search(r"^Status: DRAFT", text, re.M) \
                or "DRAFT" in title:
            return refuse("the prereg is not frozen")
        prereg = {"path": a.prereg, "sha256": a.prereg_sha256}
    workbench = Path(a.workbench).expanduser().resolve()
    L.set_default(Limits.worker(a.dedicated_user) if a.limits_profile == "worker" else Limits.local())
    try:
        limits_report = preflight(workbench)
    except (ConfinementError, L.LimitError) as exc:
        return refuse(f"confinement preflight failed: {exc}")
    if a.max_fs_mb is not None:
        st = os.statvfs(workbench)
        size_mb = st.f_blocks * st.f_frsize / (1 << 20)
        if size_mb > a.max_fs_mb:
            return refuse(f"{workbench} is on a {size_mb:.0f} MB filesystem, larger than --max-fs-mb {a.max_fs_mb}")
    entries = office.load_entries(None if a.tasks == "all" else a.tasks.split(","))

    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    prefix = "fake-" if a.fake else "rehearse-" if rehearsal else ""
    arm_label = f"{a.arm}-no-offers" if a.no_offers else a.arm
    cell = f"{prefix}everyday-office-cleanslate-{arm_label}-r{a.run_index}-{stamp}"
    attempt = f"att-{stamp}-{uuid.uuid4().hex[:6]}"
    cell_dir = workbench / "appworld-r0" / cell
    attempt_dir = workbench / "runs" / cell / "attempts" / attempt
    cell_dir.mkdir(parents=True)
    attempt_dir.mkdir(parents=True)
    (cell_dir / "attempt.json").write_text(json.dumps({"attempt_dir": str(attempt_dir), "run_id": cell,
                                                       "attempt_id": attempt}, indent=1))
    ledger = CostLedger(run_id=cell, attempt_id=attempt, path=str(cell_dir / "cost-record.jsonl"),
                        journal_path=str(attempt_dir / "costs.jsonl"), prices=PRICES)
    guard = RunawayGuard(ledger, max_usd_per_hour=a.max_usd_per_hour, window_minutes=10)
    caps = Caps(max_usd=Decimal(a.max_task_usd), max_calls=a.max_task_calls, max_wall_s=a.max_wall_s)
    if a.fake:
        from office_fake import OfficeFake as factory
    else:
        def factory():
            return ChatClient(model=MODEL, effort=EFFORT, base_url=a.base_url)
    meta = {"cell": cell, "offers": not a.no_offers, "mode": "fake" if a.fake else "rehearsal" if rehearsal else "paid", "base_url": a.base_url, "model": "office-fake" if a.fake else MODEL,
            "effort": None if a.fake else EFFORT, "order": a.order, "run_index": a.run_index,
            "limits_profile": a.limits_profile, "limits": limits_report, "prereg": prereg,
            "price_estimates_usd_per_m": PRICES, "catalogue": catalogue_prices(),
            "max_usd_per_hour": a.max_usd_per_hour, "argv": sys.argv[1:] if argv is None else argv,
            "python": sys.version.split()[0], "host": os.uname().nodename}
    (cell_dir / "cell.json").write_text(json.dumps(meta, indent=1))
    summary = office.run(entries, cell_dir, factory, arm="persistent", memory=a.arm == "full", ledger=ledger,
                         caps=caps, guard=guard, cell_timeout=a.cell_timeout, meta=meta,
                         capture_content=not a.no_capture_content, offers=not a.no_offers)
    leftovers = [p.name for p in workbench.glob("preflight-*")]
    summary["cleanup"]["preflight_dirs_left"] = leftovers
    (cell_dir / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps({"cell": cell, "cell_dir": str(cell_dir), "journal": str(attempt_dir / "costs.jsonl"),
                      **{k: summary[k] for k in ("records", "planned", "passed", "unknown", "runaway_stop", "cost")},
                      "clean": summary["cleanup"]["clean"] and not leftovers}, indent=1))
    if summary["runaway_stop"]:
        return RUNAWAY
    return 0 if summary["cleanup"]["clean"] and not leftovers else CLEANUP_FAILED


if __name__ == "__main__":
    sys.exit(main())
