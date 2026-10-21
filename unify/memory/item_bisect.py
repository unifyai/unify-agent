"""Bisect and the rollback proposal of the library mode. Deterministic; no model calls.

When a function is ``retired`` by its own use (errors or negative signals, not taint), the harness runs its
current tests on each version of it in ``main``'s history, and on each version the gate refused
(:func:`.history_probe.refused_versions`). The current tests read its recorded inputs, the drawn inputs appended
to them and its exact expected values, so they are the item's recorded inputs, re-run. Every run is
:func:`.history_probe.probe`'s confined pytest run: the tree read-only, no network, bounded. Nothing here imports
library code.

The result names the version that introduced the current behaviour (the oldest version of the newest run of
versions that behave like the current one) and every version at which behaviour changed. At most
:data:`BISECT_MAX_VERSIONS` newest versions are run; how many were not is recorded (``versions_cut``).

:func:`rollback_target` proposes the newest older version whose use record is clean. CURATE decides.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .gitio import GitError, Repo
from .layout import item_path, module_bodies

BISECT_MAX_VERSIONS = 20
VERSION_SCAN = 200
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


def rollback_target(
    use: Mapping[str, Mapping],
    versions: list[str],
    introduced: str | None,
    order: list[str],
) -> dict | None:
    """The newest version older than *introduced* whose use record is clean: episodes ran it (on
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
