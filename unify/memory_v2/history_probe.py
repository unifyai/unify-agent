"""The history probe of memory v2.1 (spec §13.4; consumed by the bisect of §10.2, P5).

The test-quality study found two things:
- a function's own tests were green on every version the gate refused;
- tests written later were red on 11 of 15 such versions.

So the probe runs the *later* tests of an item (from a later library commit) on a refused or suspect *version*.
That is the version's tree, with the later test files and their packages' ``tests/data/`` laid over it. Earlier
``main`` versions are not probed: later tests are red on them by construction (§13.4).

Every run is the gate's confined pytest run (:func:`.sandbox_run.run_pytest`): the tree read-only at ``/memory``,
no network, ``-c /dev/null --import-mode=importlib``, one run per test file, each bounded. A refused version is a
pass row's candidate commit, while git still holds it (:func:`refused_versions`). Suspect versions come from P5.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from .evidence import EvidenceStore
from .gate import _PYTEST_ENV
from .gitio import GitError, Repo
from .sandbox_run import PYTHON, PytestOutcome, run_pytest
from .snapshot import listing, materialise

PROBE_S = 300.0


@dataclass
class ProbeRow:
    label: str
    version: str
    test: str
    passed: list[str] = field(default_factory=list)
    failed: list[str] = field(default_factory=list)
    valid: bool = True
    timed_out: bool = False

    @property
    def red(self) -> bool:
        return self.valid and not self.timed_out and bool(self.failed)


def _resolve(mem: Repo, rev: str) -> str | None:
    if not isinstance(rev, str) or not rev or rev.startswith("-") or "\n" in rev:
        return None
    try:
        return (
            mem.run("rev-parse", "--verify", "--quiet", f"{rev}^{{commit}}").strip()
            or None
        )
    except GitError:
        return None


def overlay(mem: Repo, version: str, later: str, tests: list[str], dest: Path) -> Path:
    """*version*'s tree with *later*'s *tests* and their packages' ``tests/data/`` laid over it."""
    dest.mkdir(parents=True, exist_ok=True)
    v_files, _ = listing(mem, version)
    tree = materialise(mem, v_files, dest / "tree")
    l_files, _ = listing(mem, later)
    missing = [t for t in tests if t not in l_files]
    if missing:
        raise ValueError(f"the later commit holds no {missing[:3]}")
    data_dirs = {t.split("/tests/", 1)[0] + "/tests/data/" for t in tests}
    keep = {
        p: l_files[p]
        for p in l_files
        if p in tests or any(p.startswith(d) for d in data_dirs)
    }
    later_tree = materialise(mem, keep, dest / "later")
    for d in data_dirs:
        shutil.rmtree(
            tree / d,
            ignore_errors=True,
        )  # the later tests read the later fixtures only
    for p in keep:
        (tree / p).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(later_tree / p, tree / p)
    return tree


def probe(
    mem: Repo,
    versions: list[tuple[str, str]],
    later: str,
    tests: list[str],
    *,
    work: Path,
    python: Path = PYTHON,
    pytest_runner: Callable[..., PytestOutcome] = run_pytest,
    timeout_s: float = PROBE_S,
) -> list[ProbeRow]:
    later_sha = _resolve(mem, later)
    if later_sha is None:
        raise ValueError("the later revision does not resolve")
    rows: list[ProbeRow] = []
    for n, (label, rev) in enumerate(versions):
        sha = _resolve(mem, rev)
        if sha is None:
            rows += [ProbeRow(label, str(rev), t, valid=False) for t in tests]
            continue
        tree = overlay(mem, sha, later_sha, tests, work / f"v{n}")
        for t in tests:
            o = pytest_runner(
                t,
                python=python,
                ro={tree: "/memory"},
                rw={},
                cwd="/memory",
                timeout_s=timeout_s,
                env=dict(_PYTEST_ENV),
            )
            rows.append(
                ProbeRow(
                    label,
                    sha,
                    t,
                    sorted(o.passed),
                    sorted(o.failed),
                    o.valid,
                    o.timed_out,
                ),
            )
    return rows


def refused_versions(ev: EvidenceStore, mem: Repo, item: str) -> list[tuple[str, str]]:
    """``(refused:<pass id>, candidate)`` for every recorded pass that refused *item* and whose candidate git holds."""
    out = []
    for pass_id, candidate, refused in ev.db.execute(
        "SELECT pass_id, candidate, items_refused FROM passes ORDER BY rowid",
    ).fetchall():
        try:
            items = json.loads(refused or "{}")
        except ValueError:
            continue
        if (
            isinstance(items, dict)
            and item in items
            and _resolve(mem, candidate) is not None
        ):
            out.append((f"refused:{pass_id}", candidate))
    return out
