"""The recorded input shapes each memory commit's catalogue shows, frozen per commit (v2.1 surfacing).

A function's input shapes are the shapes of the recorded inputs the gate admitted it on
(:func:`descriptor`, with :mod:`.memory_helper`'s ``file_shape`` and ``value_shape``). They are kept as a
**snapshot per memory commit** in the evidence store (``commit_shapes``): item -> ``{"body": digest,
"shapes": [...], "backfilled": bool}``, written once and never changed. So exporting any commit again, a
pinned replay of an old one included, renders byte-identical catalogue files.

* **A landed merge** (:meth:`.gate.Gate.merge`) writes the candidate's snapshot (:func:`snapshot_rows`):
  for every environment function in the candidate, the parent's shapes for the same body, plus the shapes
  of the covers this merge validated for that body.
* **An export** (:func:`shapes_at`) reads the snapshot of its commit, else of its nearest first-parent
  ancestor that has one (a hide commit only removes functions), keeping only rows whose body digest still
  matches. A function with no row there, such as one merged before snapshots existed, is **backfilled**:
  its shapes are derived from the evidence store's validated covers of that item (the ``covers`` table and
  the recorded observations they point at), deterministically, and marked ``backfilled``. With
  ``freeze=True`` (the request's export), a commit that needed a backfill or had no snapshot of its own gets
  one now, so its later exports read the same rows.

Descriptors whose canonical JSON holds a key-shaped string are dropped (:data:`.redact.KEY_SHAPED`).
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

from .admission import is_rejection
from .catalogue import body_digest
from .evidence import MAX_INPUT_SHAPES
from .held_out import _blob_sha
from .memory_helper import file_shape, value_shape
from .memory_repo import items
from .redact import KEY_SHAPED
from .snapshot import item_bodies

Rows = dict[str, dict]
# Covers read per function for a backfill.
MAX_BACKFILL_COVERS = 32
MAX_ANCESTORS = 10_000


def descriptor(a: Any, input_kind: str | None, blobs: Any) -> dict | None:
    """The input-shape descriptor of one recorded covered action, or None.

    A covered file is shaped from its recorded blob (by its path's name), a shell output from its tail, and
    an observation (a dialogue's, or a tool response for an item taking ``observation``) as a value. An
    item taking the environment object (``env``) has no data input to shape; rejections have none either.
    """
    kind = getattr(a, "kind", "tool")
    if is_rejection(a) or (kind != "shell" and a.status != "ok"):
        return None
    desc: dict | None = None
    if kind == "worktree":
        sha = _blob_sha(a)
        if sha is not None and blobs.has(sha):
            path = a.args[0] if a.args and isinstance(a.args[0], str) else ""
            desc = file_shape(path, blobs.get(sha))
    elif kind == "shell":
        tail = a.response.get("tail") if isinstance(a.response, dict) else None
        desc = file_shape("", tail.encode("utf-8")) if isinstance(tail, str) else None
    elif kind == "dialogue" or (input_kind or "env") == "observation":
        desc = value_shape(a.response)
    if desc is not None and KEY_SHAPED.search(json.dumps(desc, sort_keys=True)):
        return None
    return desc


def merge_shapes(*groups: list[dict]) -> list[dict]:
    """The union of descriptor lists, sorted by canonical JSON, at most :data:`.evidence.MAX_INPUT_SHAPES`."""
    canon = {
        json.dumps(d, sort_keys=True, ensure_ascii=False) for g in groups for d in g
    }
    return [json.loads(c) for c in sorted(canon)[:MAX_INPUT_SHAPES]]


def descriptors(actions: list[Any], input_kind: str | None, blobs: Any) -> list[dict]:
    out: list[dict] = []
    for a in actions:
        try:
            desc = descriptor(a, input_kind, blobs)
        except Exception:  # noqa: BLE001 - shapes are a catalogue aid, never a failure
            desc = None
        if desc is not None:
            out.append(desc)
    return merge_shapes(out)


def _functions(tree: Path) -> dict[str, tuple[str, str | None]]:
    """item -> (body digest, declared input form or None) of the environment functions in *tree*."""
    bodies = item_bodies(Path(tree))
    forms = {
        it.item_id: (it.input or None)
        for it in items(Path(tree)).items
        if it.kind == "env_function"
    }
    return {
        item: (body_digest(body), forms.get(item))
        for item, (kind, body, _) in bodies.items()
        if kind == "env_function"
    }


def snapshot_rows(
    tree: Path,
    prev: Rows,
    new: dict[str, tuple[str, list[dict]]],
) -> Rows:
    """A candidate's snapshot: per function, the parent's shapes for the same body plus this merge's."""
    out: Rows = {}
    for item, (digest, _) in sorted(_functions(tree).items()):
        old = prev.get(item)
        carried = old["shapes"] if old and old["body"] == digest else []
        added = new[item][1] if item in new and new[item][0] == digest else []
        shapes = merge_shapes(carried, added)
        if shapes:
            out[item] = {
                "body": digest,
                "shapes": shapes,
                "backfilled": bool(carried and old and old.get("backfilled")),
            }
    return out


def _nearest(repo: Any, evidence: Any, sha: str) -> tuple[str, Rows] | None:
    try:
        shas = repo.run(
            "rev-list",
            "--first-parent",
            f"--max-count={MAX_ANCESTORS}",
            sha,
        ).split()
    except Exception:  # noqa: BLE001 - an unreadable history has no snapshot
        return None
    for c in shas:
        rows = evidence.commit_shapes(c)
        if rows is not None:
            return c, rows
    return None


def backfill(
    tree: Path,
    evidence: Any,
    lookup: Callable[[str, int], Any],
    blobs: Any,
    only: set[str] | None = None,
) -> Rows:
    """Shapes derived from the evidence store's validated covers of each function (of *only*, if given)."""
    out: Rows = {}
    for item, (digest, form) in sorted(_functions(tree).items()):
        if only is not None and item not in only:
            continue
        actions = []
        for eid, idx in evidence.covers_of(item)[:MAX_BACKFILL_COVERS]:
            try:
                a = lookup(eid, idx)
            except Exception:  # noqa: BLE001 - an unreadable episode shows nothing
                a = None
            if a is not None:
                actions.append(a)
        shapes = descriptors(actions, form, blobs)
        if shapes:
            out[item] = {"body": digest, "shapes": shapes, "backfilled": True}
    return out


def shapes_at(
    repo: Any,
    evidence: Any,
    sha: str,
    tree: Path,
    *,
    lookup: Callable[[str, int], Any] | None = None,
    blobs: Any = None,
    freeze: bool = False,
) -> Rows:
    """The shape rows of commit *sha* (whose files are at *tree*); see the module docstring."""
    functions = _functions(tree)
    found = _nearest(repo, evidence, sha)
    if found is not None and found[0] == sha:
        return found[1]  # frozen
    rows = {
        item: row
        for item, row in (found[1] if found is not None else {}).items()
        if item in functions and row["body"] == functions[item][0]
    }
    missing = {item for item in functions if item not in rows}
    if missing and lookup is not None and blobs is not None:
        rows.update(backfill(tree, evidence, lookup, blobs, only=missing))
    if freeze:
        evidence.write_commit_shapes(sha, rows)
        return evidence.commit_shapes(sha) or {}
    return rows


def lookup_from(rows: Rows) -> Callable[[str, str], tuple[list[dict], bool] | None]:
    """The catalogue's shape lookup over *rows*: (shapes, backfilled) for a matching body, else None."""

    def lookup(item: str, digest: str) -> tuple[list[dict], bool] | None:
        row = rows.get(item)
        if row is None or row["body"] != digest or not row["shapes"]:
            return None
        return row["shapes"], bool(row.get("backfilled"))

    return lookup
