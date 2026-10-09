"""Bisect and the rollback proposal of memory v2.1 (spec §10.2, §13.4). Deterministic; no model calls.

When a function turns ``suspect`` by its own use (errors or negative signals, not taint), the harness runs its
current tests on each version of it in ``main``'s history, and on each version the gate refused
(:func:`.history_probe.refused_versions`). The current tests read its recorded inputs, the drawn inputs appended
to them and its exact expected values (P4), so they are the item's recorded inputs, re-run. Every run is
:func:`.history_probe.probe`'s confined pytest run: the tree read-only, no network, bounded. Nothing here imports
library code.

The result names the version that introduced the current behaviour (the oldest version of the newest run of
versions that behave like the current one) and every version at which behaviour changed. At most
:data:`BISECT_MAX_VERSIONS` newest versions are run; how many were not is recorded (``versions_cut``).

:func:`rollback_target` proposes the newest older version whose use record is clean. CURATE decides (P6).
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from .gitio import GitError, Repo
from .layout import item_path, module_bodies

BISECT_MAX_VERSIONS = 20
VERSION_SCAN = 200
#: The newest refused versions run (the rest are counted in ``versions_cut``).
BISECT_MAX_REFUSED = 10
#: Seconds all of one consolidation's bisects may take together (AGENTS.md: bounded runtime; MAIN, 9 Oct). An item
#: the budget does not reach is recorded ``{"skipped": "budget"}``, and the records are still written.
BISECT_BUDGET_S = 600.0


def versions_of(
    repo: Repo,
    item: str,
    head: str,
    *,
    scan: int = VERSION_SCAN,
) -> list[str]:
    """The commits up to *head* that changed *item*, oldest first: for a function, those whose unparsed definition
    differs from their parent's (as :func:`.library_export.item_history` decides); for a note, those that changed
    its file. At most the newest *scan* commits that touched its file are read."""
    path = item if ":" not in item else item_path(item)
    name = item.split(":", 1)[1] if ":" in item else None

    def body(rev: str | None) -> str | None:
        if rev is None:
            return None
        try:
            return module_bodies(repo.show(rev, path)).get(name)
        except GitError:
            return None

    out: list[str] = []
    for line in repo.run(
        "log",
        f"--max-count={scan}",
        "--format=%H %P",
        head,
        "--",
        path,
    ).splitlines():
        parts = line.split()
        if not parts:
            continue
        commit, parent = parts[0], (parts[1] if len(parts) > 1 else None)
        if name is None:
            out.append(commit)
            continue
        after = body(commit)
        if after is not None and after != body(parent):
            out.append(commit)
    return list(reversed(out))


def _takes(fn: Callable[..., Any], kwargs: Mapping[str, Any]) -> bool:
    import inspect

    try:
        params = inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return False
    return all(k in params for k in kwargs) or any(
        p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()
    )


def _signature(rows: list[Any]) -> tuple:
    """How one version behaved under the tests: invalid, timed out, or (passed ids, failed ids)."""
    if not rows or any(not r.valid for r in rows):
        return ("invalid",)
    if any(r.timed_out for r in rows):
        return ("timed_out",)
    return (
        tuple(sorted(t for r in rows for t in r.passed)),
        tuple(sorted(t for r in rows for t in r.failed)),
    )


def bisect_item(
    mem: Repo,
    ev: Any,
    item: str,
    head: str,
    tests: list[str],
    *,
    work: Path,
    probe: Callable[..., list] | None = None,
    refused: Callable[..., list[tuple[str, str]]] | None = None,
    max_versions: int = BISECT_MAX_VERSIONS,
    deadline: float | None = None,
    stop: Callable[[], bool] | None = None,
) -> dict:
    """Run *item*'s current *tests* (from *head*) on its versions and on its newest :data:`BISECT_MAX_REFUSED`
    refused versions (module docstring). *deadline* and *stop* bound the probe
    (:func:`.history_probe.probe`); a probe cut short gives ``{"skipped": "budget" | "stopped"}``.
    """
    if ":" not in item:
        return {"skipped": "a note has no tests to run"}
    if not tests:
        return {"skipped": "its verification record names no tests"}
    from . import history_probe

    probe = probe or history_probe.probe
    refused_fn = refused or history_probe.refused_versions
    every = versions_of(mem, item, head)
    versions = every[-max_versions:] if max_versions > 0 else []
    labels = [(f"v{i}", sha) for i, sha in enumerate(versions)]
    every_refused = list(refused_fn(ev, mem, item))
    refused_rows = every_refused[
        -BISECT_MAX_REFUSED:
    ]  # recorded oldest first: the newest are kept
    extra = {k: v for k, v in (("deadline", deadline), ("stop", stop)) if v is not None}
    if extra and not _takes(probe, extra):
        extra = {}  # a probe without the bounds (a test's stand-in)
    try:
        rows = probe(
            mem,
            labels + refused_rows,
            head,
            list(tests),
            work=Path(work),
            **extra,
        )
    except history_probe.ProbeCut as cut:
        return {"skipped": cut.reason}
    by_label: dict[str, list[Any]] = {}
    for r in rows:
        by_label.setdefault(r.label, []).append(r)
    sigs = [_signature(by_label.get(label, [])) for label, _ in labels]
    introduced = None
    for i in range(len(versions) - 1, -1, -1):
        if sigs[i] != sigs[-1]:
            break
        introduced = versions[i]
    return {
        "versions": versions,
        "versions_cut": len(every)
        - len(versions)
        + len(every_refused)
        - len(refused_rows),
        "introduced": introduced,
        "changes": [
            versions[i] for i in range(1, len(versions)) if sigs[i] != sigs[i - 1]
        ],
        "red": [sha for (_, sha), s in zip(labels, sigs) if len(s) == 2 and s[1]],
        "refused": [
            {
                "label": label,
                "version": sha,
                "red": any(r.red for r in by_label.get(label, [])),
            }
            for label, sha in refused_rows
        ],
        "tests": list(tests),
    }


def rollback_target(
    use: Mapping[str, Mapping],
    versions: list[str],
    introduced: str | None,
    order: list[str],
) -> dict | None:
    """The newest version older than *introduced* whose use record is clean (spec §10.2): episodes ran it (on
    the commits from that version up to the next), and none raised, refused or was followed by a negative
    signal. *use* is the item's record ``use`` (library commit -> counts); *order* is ``main`` oldest first.
    """
    if introduced is None or introduced not in versions:
        return None
    pos = {c: i for i, c in enumerate(order)}
    idx = versions.index(introduced)
    for k in range(idx - 1, -1, -1):
        start, end = pos.get(versions[k]), pos.get(versions[k + 1])
        if start is None or end is None:
            continue
        rows = [use[c] for c in order[start:end] if c in use]
        episodes = sum(int(r.get("episodes", 0)) for r in rows)
        if episodes and not any(
            r.get("errors", 0) or r.get("negative_signals", 0) for r in rows
        ):
            return {"version": versions[k], "episodes": episodes}
    return None
