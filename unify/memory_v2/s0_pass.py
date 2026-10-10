"""Memory v2.1 r5 (r4 §4): S0 at the start of each WRITE pass, as files for the writer and facts for the analysts.

Tier (a) (:mod:`.mining`) runs over the batch and every earlier episode; tier (b) (:mod:`.mining_behaviour`)
replays each episode once in its lifetime (the replay results are kept in a cache folder beside the state) and
cross-runs only pairs with a unit from the batch. Office code is replayed in the episode's recorded workspace,
exported once from the run's worktree repo. The writer gets ``/inputs/s0/clusters.json`` (tier a, the clusters that
touch the batch) and ``/inputs/s0/behaviour.json`` (tier b's clusters and their units). The analysts get the batch
episodes in a cluster and the code shapes seen before the batch (:func:`.analysts.flagged`).

S0 never stops a pass: any failure (no bubblewrap, a box error, an unreadable episode) is returned as ``error`` and
the pass goes on without clusters. Nothing is keyed on a task id or the stream: episodes are distinguished by
their request bytes.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from pathlib import Path
from typing import Callable

from . import mining as _mining
from . import mining_behaviour as _behaviour


def _as_mining(ep) -> tuple[str, str, list]:
    req = hashlib.sha256(
        json.dumps(ep.request[:1], sort_keys=True, default=str).encode(),
    ).hexdigest()[:16]
    cells = [
        _mining.Cell(
            int(c.index),
            c.code or "",
            c.output or "",
            c.error,
            c.language or "python",
        )
        for c in ep.cells
    ]
    return ep.episode_id, req, cells


def _export_tree(worktree: str, sha: str, dest: Path) -> Path | None:
    if dest.exists():
        return dest
    data = subprocess.run(
        ["git", "--git-dir", worktree, "archive", sha],
        capture_output=True,
        timeout=300,
    )
    if data.returncode:
        return None
    tmp = dest.with_name(dest.name + ".part")
    tmp.mkdir(parents=True, exist_ok=True)
    if subprocess.run(
        ["tar", "x", "-C", str(tmp)],
        input=data.stdout,
        capture_output=True,
    ).returncode:
        return None
    tmp.rename(dest)
    return dest


def run(
    load: Callable,
    batch: list[str],
    history: list[str],
    dest: Path,
    cache: Path,
    *,
    worktree: str | None = None,
    jobs: int | None = None,
) -> dict:
    """S0 for one pass; the summary, with ``clustered`` (batch episodes in any cluster) and ``seen`` (code shapes
    of the episodes before the batch); ``error`` when it could not run."""
    try:
        dest.mkdir(parents=True, exist_ok=True)
        cache.mkdir(parents=True, exist_ok=True)
        order = list(dict.fromkeys([*history, *batch]))
        eps = {e: load(e) for e in order}
        mined = [_as_mining(eps[e]) for e in order]
        clusters, _ = _mining.mine(mined)
        in_batch = set(batch)
        touching = [
            c for c in clusters if any(i.episode in in_batch for i in c.instances)
        ]
        (dest / "clusters.json").write_text(
            json.dumps(_mining.as_json(touching), indent=1, default=str) + "\n",
        )
        clustered = {
            i.episode for c in touching for i in c.instances if i.episode in in_batch
        }
        seen: set[str] = set()
        for e in order:
            if e not in in_batch:
                from .analysts import _cell_keys

                seen |= _cell_keys(eps[e])
        summary = {
            "tier_a_clusters": len(touching),
            "clustered": sorted(clustered),
            "seen": seen,
        }
    except Exception as exc:  # noqa: BLE001 - S0 never stops a pass
        return {"error": f"{type(exc).__name__}: {str(exc)[:200]}"}
    try:
        # tier (b): replay results cached per episode; its work folder stays out of /inputs
        cached = {}
        replays = cache / "replays"
        replays.mkdir(exist_ok=True)
        for e in order:
            f = replays / f"{e}.json"
            if f.is_file():
                try:
                    cached[e] = json.loads(f.read_text())
                except ValueError:
                    pass
        trees = {}
        if worktree:
            for e in order:
                sha = getattr(eps[e], "worktree_before", None)
                if sha:
                    t = _export_tree(worktree, sha, cache / "trees" / e)
                    if t is not None:
                        trees[e] = t
        doc, fresh = _behaviour.run_behaviour(
            mined,
            cache / "work",
            jobs=jobs or max(1, (os.cpu_count() or 2) - 2),
            trees=trees,
            batch=in_batch,
            cached=cached,
        )
        for e, r in fresh.items():
            (replays / f"{e}.json").write_text(json.dumps(r, default=str))
        units = doc["units"]
        (dest / "behaviour.json").write_text(
            json.dumps(
                {
                    "summary": doc["summary"],
                    "clusters": doc["clusters"],
                    "units": {m: units[m] for c in doc["clusters"] for m in c["units"]},
                },
                indent=1,
                default=str,
            )
            + "\n",
        )
        summary["tier_b"] = {
            k: doc["summary"][k] for k in ("behavioural_clusters", "new_vs_a", "pairs")
        }
        summary["clustered"] = sorted(
            clustered
            | {units[m]["episode"] for c in doc["clusters"] for m in c["units"]}
            & in_batch,
        )
    except (
        Exception
    ) as exc:  # noqa: BLE001 - tier (b) failing keeps tier (a)'s clusters
        summary["tier_b_error"] = f"{type(exc).__name__}: {str(exc)[:200]}"
    return summary
