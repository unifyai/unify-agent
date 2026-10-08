"""Test-only stand-ins for the modules RequestRun drives (online Tracks A and B), at their fixed interfaces.

``install(monkeypatch)`` puts recording fakes of ``integration.worktree_capture``, ``trajectory``,
``consolidate`` and ``cost`` in place (``sys.modules`` and the package attribute), whether or not the real
modules exist, so the request lifecycle is tested on its own. Each fake records the arguments it was
given; ``Fakes.calls`` holds them in call order. Nothing here is shipped.
"""

from __future__ import annotations

import json
import sys
import types
from dataclasses import dataclass, field
from decimal import Decimal
from pathlib import Path

from unify.memory_v2.blobs import BlobStore
from unify.memory_v2.episodes import Action, Episode
from unify.memory_v2.evidence import EvidenceStore
from unify.memory_v2.gitio import Repo
from unify.memory_v2.redact import Redactor

PKG = "unify.memory_v2.integration"
WT_BEFORE = "b" * 40
WT_AFTER = "a" * 40
WT_DIFF = "diff --git a/out.json b/out.json\n+{}\n"
WT_ACTION = Action(
    0,
    "worktree:workspace",
    "write",
    ["out.json"],
    {},
    {"blob_before": None, "blob_after": "c" * 64, "size": 2, "shape": {}},
    "ok",
    "unknown",
    None,
    "worktree",
)


@dataclass
class Fakes:
    calls: list[tuple[str, tuple, dict]] = field(default_factory=list)
    cost_active: list[bool] = field(default_factory=list)
    passes_raise: BaseException | None = None
    assemble_raise: BaseException | None = None
    pass_events: bool = True
    redactors: list = field(default_factory=list)
    tool_actions: list = field(default_factory=list)

    def names(self) -> list[str]:
        return [c[0] for c in self.calls]

    def of(self, name: str) -> tuple[tuple, dict]:
        hits = [(a, k) for n, a, k in self.calls if n == name]
        assert len(hits) == 1, f"{name}: {len(hits)} calls"
        return hits[0]


def _bare(path: Path) -> Repo:
    return Repo(path) if (path / "HEAD").exists() else Repo.init_bare(path)


def install(monkeypatch) -> Fakes:
    import importlib

    pkg = importlib.import_module(PKG)
    f = Fakes()

    # -- Track A: worktree_capture --------------------------------------------------------------
    wc = types.ModuleType(f"{PKG}.worktree_capture")

    @dataclass
    class WorktreeResult:
        actions: list
        before: str | None
        after: str | None
        diff: str

    class WorktreeCapture:
        def __init__(self, paths, workspace, redactor_factory) -> None:
            f.calls.append(("WorktreeCapture", (paths, workspace), {}))
            self.redactor_factory = redactor_factory

        def begin(self) -> None:
            f.calls.append(("worktree.begin", (), {}))

        def finish(self, cells):
            f.calls.append(("worktree.finish", (cells,), {}))
            f.redactors.append(self.redactor_factory())
            return WorktreeResult([WT_ACTION], WT_BEFORE, WT_AFTER, WT_DIFF)

    wc.WorktreeCapture, wc.WorktreeResult = WorktreeCapture, WorktreeResult

    # -- Track B: cost ------------------------------------------------------------------------
    cost = types.ModuleType(f"{PKG}.cost")

    class CostListener:
        def __init__(self) -> None:
            self.rows: list = []

        def activate(self) -> None:
            f.cost_active.append(True)

        def deactivate(self) -> None:
            f.cost_active.append(False)

    listener = CostListener()
    cost.CostListener = CostListener
    cost.install = lambda: listener

    # -- Track B: trajectory ------------------------------------------------------------------
    traj = types.ModuleType(f"{PKG}.trajectory")

    def read_jsonl(path):
        f.calls.append(("read_jsonl", (Path(path),), {}))
        return [json.loads(x) for x in Path(path).read_text().splitlines() if x.strip()]

    def fold(lines):
        return list(lines)

    def timed_cells(folded):
        return [
            ("cell", i) for i, line in enumerate(folded) if line.get("type") == "cell"
        ]

    def assemble(
        run,
        lines,
        memory_diff,
        ended_at,
        *,
        extra_actions=(),
        worktree_before=None,
        worktree_after=None,
        worktree_diff="",
    ):
        kw = dict(
            extra_actions=list(extra_actions),
            worktree_before=worktree_before,
            worktree_after=worktree_after,
            worktree_diff=worktree_diff,
        )
        f.calls.append(("assemble", (run, lines, memory_diff, ended_at), kw))
        if f.assemble_raise is not None:
            raise f.assemble_raise
        drained = run.observer.drain() if run.observer is not None else None
        f.tool_actions = list(getattr(drained, "actions", None) or [])
        ep = Episode(
            run.episode_id,
            run.started_at,
            ended_at,
            run.build,
            run.model,
            run.effort,
            "dense",
            run.pin,
            worktree_before,
            worktree_after,
            [run.request],
            list(lines),
            [],
            list(extra_actions),
            memory_diff=memory_diff,
            worktree_diff=worktree_diff,
        )
        return ep, Redactor()

    def learned_secrets(values):
        out = {}
        for prefix, value in values:
            if isinstance(value, dict):
                for k, v in value.items():
                    if "TOKEN" in str(k).upper() and isinstance(v, str) and len(v) >= 8:
                        out[f"{prefix}.{k}"] = v
        return out

    from unify.memory_v2.integration.adapters.tool import RecordingObserver

    traj.read_jsonl, traj.fold, traj.timed_cells, traj.assemble = (
        read_jsonl,
        fold,
        timed_cells,
        assemble,
    )
    traj.learned_secrets, traj.TimedObserver = learned_secrets, RecordingObserver

    # -- Track B: consolidate -----------------------------------------------------------------
    cons = types.ModuleType(f"{PKG}.consolidate")

    def open_stores(paths):
        f.calls.append(("open_stores", (paths,), {}))
        return types.SimpleNamespace(
            paths=paths,
            memory=_bare(paths.memory),
            episodes=_bare(paths.episodes),
            blobs=BlobStore(paths.blobs),
            evidence=EvidenceStore(paths.evidence),
        )

    def post_checker(stores, eid, sha, solved, ts):
        f.calls.append(("post_checker", (stores, eid, sha, solved, ts), {}))
        return solved is not None

    async def run_due_passes(
        stores,
        eid,
        sha,
        state,
        *,
        effort,
        settings,
        emit,
        clock=None,
    ):
        f.calls.append(
            (
                "run_due_passes",
                (stores, eid, sha, state),
                {"effort": effort, "settings": settings, "emit": emit},
            ),
        )

        def emit_and_file(
            event,
        ):  # as the real driver: the events file always, the emitter if any
            path = stores.paths.events
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a") as fh:
                fh.write(json.dumps(event, default=str) + "\n")
            if emit is not None:
                emit(event)

        if f.pass_events:
            emit_and_file(
                {
                    "type": "consolidation",
                    "phase": "start",
                    "pass_id": f"{eid}.p0",
                    "trigger_tokens": 1,
                    "episodes": [eid],
                    "sol_model": settings.UNIFY_MEMORY_V2_SOL_MODEL,
                    "sol_effort": effort,
                    "cap_usd": Decimal("0.00000073") * 1,  # a Decimal on purpose
                },
            )
            emit_and_file(
                {
                    "type": "consolidation",
                    "phase": "end",
                    "pass_id": f"{eid}.p0",
                    "usd": Decimal("1E-7"),
                    "unknown_cost_calls": 0,
                    "calls": 1,
                    "seconds": 0.5,
                    "gate_passed": False,
                    "items": [],
                    "index_tokens": 0,
                    "reason_codes": ["no_manifest"],
                },
            )
        if f.passes_raise is not None:
            raise f.passes_raise
        return []

    cons.open_stores, cons.post_checker, cons.run_due_passes = (
        open_stores,
        post_checker,
        run_due_passes,
    )

    for name, mod in (
        ("worktree_capture", wc),
        ("cost", cost),
        ("trajectory", traj),
        ("consolidate", cons),
    ):
        monkeypatch.setitem(sys.modules, f"{PKG}.{name}", mod)
        monkeypatch.setattr(pkg, name, mod, raising=False)
    return f


def dump_home(home: Path) -> bytes:
    """Every byte stored under *home*: regular files outside git dirs, every object of each git dir (notes
    and unreachable objects included, decompressed) and each SQLite database's rows."""
    import sqlite3
    import subprocess

    out: list[bytes] = []
    gits = [p.parent for p in home.rglob("HEAD") if (p.parent / "objects").is_dir()]
    for path in sorted(home.rglob("*")):
        if any(path == g or g in path.parents for g in gits) or not path.is_file():
            continue
        if path.is_symlink():
            continue
        data = path.read_bytes()
        out.append(data)
        if data.startswith(b"SQLite format 3"):
            con = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
            try:
                out.append("\n".join(con.iterdump()).encode())
            finally:
                con.close()
    env = {
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_CONFIG_GLOBAL": "/dev/null",
        "PATH": "/usr/bin:/bin",
    }
    for g in gits:
        # every object, reachable or not (notes refs included)
        out.append(
            subprocess.run(
                [
                    "git",
                    "--git-dir",
                    str(g),
                    "cat-file",
                    "--batch-all-objects",
                    "--batch",
                ],
                capture_output=True,
                env=env,
                check=True,
            ).stdout,
        )
    return b"\n".join(out)
