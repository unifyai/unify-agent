"""Office adapter: serves everyday office-v1 instances to cleanslate-v1 the way the office runner
(benchmarks/everyday/src/everyday/office/runner.py) serves them to Unify.

    python3 adapters/office.py dryrun --tasks ofc-d01,ofc-r01 --out DIR [--arm persistent|wiped]

What matches the Unify runner:
  * the model receives the stream's prompt text, byte for byte; its sha256 is checked against
    stream.json, task.json and MANIFEST.json before the first task starts;
  * the task's workspace/ is copied into an empty working folder and its tree hash checked
    against task.json;
  * after the turn, the working folder is copied out (regular files only) and scored by
    `python -m everyday.score <task> <ws>` in a separate process that never imports harness code
    and gets a built, credential-free environment;
  * the persistent arm keeps the procedure store across tasks; the wiped arm starts each task
    with an empty store.
What the harness never sees: task ids, job names, the office data folder, expected answers,
checkers, generators or the repository. Task ids are written only to the offline records. The
working folder is the only host path inside the workspace (bubblewrap), and the adapter names
the data, the benchmark source and the repository as hidden paths, so the sandbox refuses any
working folder that would expose them.

Only the scripted fake model is wired here. Paid runs need a frozen preregistration and MAIN's
GO, and are deliberately not implemented in this file.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import uuid
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROTO = HERE.parent
REPO = PROTO.parents[1]
sys.path.insert(0, str(PROTO))

from cleanslate import Agent, Caps, ProcedureStore  # noqa: E402
from cleanslate.cost import CostLedger, RunawayGuard, RunawayStop  # noqa: E402
from cleanslate import sandbox  # noqa: E402

EVERYDAY_SRC = REPO / "benchmarks/everyday/src"
MANIFEST = REPO / "benchmarks/everyday/MANIFEST.json"
DATA = Path(os.environ.get("EVERYDAY_DATA") or Path.home() / ".local/share/continual-harness-research/everyday")
OFFICE_DIR = DATA / "office-v1"
SCORER_PYTHON = "/usr/bin/python3"
SCORE_TIMEOUT_S = 600
SCHEMA = "cleanslate-office-record-v1"


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def tree_sha256(ws: Path) -> str:
    """The benchmark's own tree hash (trusted adapter side; never imported by the harness)."""
    sys.path.insert(0, str(EVERYDAY_SRC))
    try:
        from everyday.office.common import tree_sha256 as benchmark_tree_sha256
    finally:
        sys.path.remove(str(EVERYDAY_SRC))
    return benchmark_tree_sha256(ws)


def hidden_paths() -> list[str]:
    return [str(p) for p in (DATA, OFFICE_DIR, EVERYDAY_SRC, MANIFEST.parent, REPO)]


def load_entries(task_ids: list[str] | None) -> list[dict]:
    """Core stream entries in the frozen order, each prompt verified three ways."""
    stream = json.loads((OFFICE_DIR / "stream.json").read_text())["core"]
    manifest = json.loads(MANIFEST.read_text())
    man_prompt = {e["task_id"]: e["prompt"] for e in manifest["office"]["stream"]}
    if task_ids is not None:
        unknown = sorted(set(task_ids) - {e["task_id"] for e in stream})
        if unknown:
            raise SystemExit(f"not core office-v1 tasks: {unknown}")
    chosen = [e for e in stream if task_ids is None or e["task_id"] in task_ids]
    out = []
    for e in chosen:
        task = json.loads((OFFICE_DIR / e["task_id"] / "task.json").read_text())
        digest = sha256_text(e["prompt"])
        problems = [what for what, ok in (("stream", e.get("prompt_sha256") == digest),
                                          ("task.json", sha256_text(task["prompt"]) == digest),
                                          ("manifest", sha256_text(man_prompt.get(e["task_id"], "")) == digest)) if not ok]
        if problems:
            raise SystemExit(f"{e['task_id']}: prompt differs from {problems}")
        if task.get("clarification"):
            raise SystemExit(f"{e['task_id']}: clarification tasks need a second turn; not supported yet")
        out.append({**e, "deliverables": task["deliverables"], "tree": task["workspace_tree_sha256"]})
    return out


def copy_out(ws: Path, dest: Path) -> dict:
    """Regular files only, no links (as the Unify runner copies out)."""
    files = 0
    for root, dirs, names in os.walk(ws):
        rel = Path(root).relative_to(ws)
        (dest / rel).mkdir(parents=True, exist_ok=True)
        for n in names:
            src = Path(root, n)
            if src.is_file() and not src.is_symlink():
                shutil.copyfile(src, dest / rel / n)
                files += 1
    return {"files": files, "tree_sha256": tree_sha256(dest)}


def score(task_id: str, ws: Path, tmp: Path) -> dict:
    tmp.mkdir(parents=True, exist_ok=True)
    env = {"PATH": "/usr/bin:/bin", "LANG": "C.UTF-8", "PYTHONPATH": str(EVERYDAY_SRC), "EVERYDAY_DATA": str(DATA),
           "TMPDIR": str(tmp), "HOME": str(tmp), "PYTHONDONTWRITEBYTECODE": "1"}
    try:
        done = subprocess.run([SCORER_PYTHON, "-B", "-m", "everyday.score", task_id, str(ws)], env=env,
                              cwd=str(tmp), capture_output=True, text=True, timeout=SCORE_TIMEOUT_S)
        verdict = json.loads([x for x in done.stdout.splitlines() if x.strip()][-1])
        if done.returncode != 0 or verdict.get("task_id") != task_id or verdict.get("passed") not in (True, False, None):
            raise ValueError(f"scorer exit {done.returncode}")
        return verdict
    except (OSError, subprocess.TimeoutExpired, ValueError, IndexError) as exc:
        return {"task_id": task_id, "passed": None, "reason": f"infrastructure error: {exc}"[:300]}


def source_hashes() -> dict:
    files = sorted((PROTO / "cleanslate").glob("*.py")) + sorted(HERE.glob("*.py")) + sorted((PROTO / "kit").glob("*")) \
        + sorted((PROTO / "analysis").glob("*.py"))
    return {str(p.relative_to(PROTO)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files if p.is_file()}


def run(entries: list[dict], out: Path, model_factory, arm: str = "persistent", memory: bool = True,
        ledger: CostLedger | None = None, caps: Caps | None = None, guard: RunawayGuard | None = None,
        cell_timeout: float = 10.0, meta: dict | None = None, capture_content: bool = True,
        offers: bool = True) -> dict:
    """Serve `entries` in order. `arm` persistent keeps the procedure store across tasks, wiped empties
    it before each; memory=False runs with no store at all (the no-memory arm). A cap ends one task
    (still scored); a RunawayStop ends the cell (exit 4 in the kit)."""
    out = out.resolve()
    hidden = hidden_paths()
    for h in hidden:
        if out == Path(h).resolve() or Path(h).resolve() in out.parents:
            raise SystemExit(f"--out must lie outside {h}: evidence stays outside Git and away from the data")
    out.mkdir(parents=True, exist_ok=True)
    unexpected = sorted(p.name for p in out.iterdir() if p.name not in ("attempt.json", "cell.json"))
    if unexpected:
        raise SystemExit(f"{out} is not a fresh run folder (holds {unexpected[:5]})")
    if ledger is None:
        run_id = f"run-{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}-{uuid.uuid4().hex[:6]}"
        ledger = CostLedger(run_id=run_id, path=str(out / "cost-record.jsonl"))
    run_id = ledger.run_id
    (out / "run.json").write_text(json.dumps({
        "run_id": run_id, "attempt_id": ledger.attempt_id, "arm": arm, "memory": memory, "office_set": "office-v1",
        "tasks": [e["task_id"] for e in entries], "source_sha256": source_hashes(),
        "caps": {k: str(v) for k, v in vars(caps or Caps()).items()}, "cell_timeout_s": cell_timeout,
        "capture_content": capture_content, "offers": offers,
        **(meta or {}), "started": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}, indent=1))
    store_path = out / "procedures.json"
    records, stopped = [], None
    launched_before = len(sandbox.LAUNCHED)
    for index, e in enumerate(entries):
        if store_path.exists():  # the store each task starts from, kept for the offline fit-rate analysis
            shutil.copyfile(store_path, out / f"procedures-before-{index:02d}.json")
            if arm == "wiped":
                store_path.unlink()
        store = ProcedureStore(str(store_path)) if memory else None
        task_dir = Path(tempfile.mkdtemp(prefix="w-", dir=out))  # opaque name; seen inside only as /work
        shutil.copytree(OFFICE_DIR / e["task_id"] / "workspace", task_dir, dirs_exist_ok=True)
        tree_ok = tree_sha256(task_dir) == e["tree"]
        rec = {"schema": SCHEMA, "run_id": run_id, "attempt_id": ledger.attempt_id, "arm": arm, "memory": memory,
               "index": index, "task_id": e["task_id"], "job": e["job"], "kind": e["kind"],
               "visit": "return" if e["kind"] == "return" else "first", "prompt_sha256": sha256_text(e["prompt"]),
               "workspace_tree_ok": tree_ok, "started": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
        held_dirs = []

        def keep_held(workdir, n, index=index, held_dirs=held_dirs):
            dest = out / "held" / f"{index:02d}-{e['task_id']}-hold{n}"
            copy_out(Path(workdir), dest)
            held_dirs.append(dest)
        try:
            if not tree_ok:
                raise RuntimeError("workspace tree hash differs from task.json")
            agent = Agent(model_factory(), store, ledger=ledger, caps=caps, guard=guard, cell_timeout=cell_timeout,
                          hold_hook=keep_held, offers=offers)
            res = agent.solve(e["prompt"], str(task_dir), hidden=hidden, label=f"office-v1 #{index}")
            if capture_content and res.transcript is not None:  # office prompts and the model's own text only
                tdir = out / "transcripts"
                tdir.mkdir(exist_ok=True)
                (tdir / f"{index:02d}-{e['task_id']}.json").write_text(json.dumps(res.transcript, indent=1))
            ws_out = out / "workspaces" / f"{index:02d}-{e['task_id']}"
            copied = copy_out(task_dir, ws_out)
            verdict = score(e["task_id"], ws_out, out / "score-tmp")
            # what each held delivery would have scored: a hold on a passing state is a false hold
            held = [{"path": str(d.relative_to(out)), "passed": score(e["task_id"], d, out / "score-tmp").get("passed")}
                    for d in held_dirs]
            rec.update({"solve_id": res.solve_id, "passed": verdict.get("passed"), "reason": verdict.get("reason"),
                        "checker_sha256": verdict.get("checker_sha256"), "steps": res.steps, "cap": res.cap,
                        "delivered": res.delivered, "delivered_files": sorted(res.files or {}),
                        "noop_cells": res.noop_cells, "offers": res.offers, "references": res.references,
                        "used_offer": res.used_offer, "stored": res.stored, "holds": res.holds,
                        "held_would_have_passed": [h["passed"] for h in held], "held_snapshots": held,
                        "events": res.events, "cost": res.cost, "workspace_out": copied, "error": None})
        except RunawayStop as exc:
            stopped = str(exc)
            rec.update({"passed": None, "error": f"RunawayStop: {exc}"})
        except Exception as exc:  # one task's failure is recorded; the grade is unknown, never a fail
            rec.update({"passed": None, "error": f"{type(exc).__name__}: {exc}"[:500]})
        finally:
            shutil.rmtree(task_dir, ignore_errors=True)
            rec["task_dir_removed"] = not task_dir.exists()
            rec["finished"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        records.append(rec)
        with open(out / "records.jsonl", "a") as f:
            f.write(json.dumps(rec, default=str) + "\n")
        if stopped:
            break
    shutil.rmtree(out / "score-tmp", ignore_errors=True)
    cleanup = {"task_dirs_left": sorted(p.name for p in out.glob("w-*")),
               "workspaces_started": len(sandbox.LAUNCHED) - launched_before,
               "workspace_processes_left": sandbox.survivors(launched_before),
               "checked": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    cleanup["clean"] = not cleanup["task_dirs_left"] and not cleanup["workspace_processes_left"]
    (out / "cleanup.json").write_text(json.dumps(cleanup, indent=1))
    summary = {"run_id": run_id, "attempt_id": ledger.attempt_id, "records": len(records),
               "planned": len(entries), "passed": sum(r.get("passed") is True for r in records),
               "unknown": sum(r.get("passed") is None for r in records), "runaway_stop": stopped,
               "cost": ledger.summary(), "cleanup": cleanup}
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    return summary


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("dryrun", help="scripted fake model; no provider requests")
    d.add_argument("--tasks", required=True)
    d.add_argument("--out", required=True)
    d.add_argument("--arm", choices=("persistent", "wiped"), default="persistent")
    args = ap.parse_args(argv)
    from office_fake import OfficeFake
    summary = run(load_entries(args.tasks.split(",")), Path(args.out), OfficeFake, args.arm)
    print(json.dumps(summary, indent=1))
    return 0 if summary["cleanup"]["clean"] else 1


if __name__ == "__main__":
    sys.exit(main())
