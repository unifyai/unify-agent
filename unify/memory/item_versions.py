"""The versions of a library item in ``main``'s history (:func:`versions_of`). Deterministic; reads git only."""

from __future__ import annotations

from .gitio import GitError, Repo
from .layout import item_path, module_bodies

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
