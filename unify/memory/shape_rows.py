"""The recorded input shapes each memory commit's catalogue shows, frozen per commit (library-mode surfacing).

A function's input shapes are the shapes of the recorded inputs the gate admitted it on (with
:mod:`.memory_helper`'s ``file_shape`` and ``value_shape``). In library mode they are the shapes of the arguments
its example calls returned on (:func:`arg_descriptors`): each descriptor names the parameter it was passed as
(``"param"``), so ``memory.find`` can say which parameter a value fits. Rows recorded before that (shapes of a
function's covers, :func:`descriptor`) have no ``"param"``; they are kept and still found. Shapes are kept as a
**snapshot per memory commit** in the evidence store (``commit_shapes``): item -> ``{"body": digest,
"shapes": [...], "backfilled": bool}``, written once and never changed. So exporting any commit again, a
pinned replay of an old one included, renders byte-identical catalogue files.

* **A landed merge** (:meth:`.gate.Gate.merge`) writes the candidate's snapshot (:func:`snapshot_rows`):
  for every function in the candidate, the parent's shapes for the same body, plus the shapes this merge
  recorded for that body (in library mode, of its example calls' arguments).
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
import os
import stat
from collections.abc import Callable
from pathlib import Path
from typing import Any

from .admission import is_rejection
from .analysis.shapes import PARSE_LIMIT
from .catalogue import body_digest
from .evidence import MAX_INPUT_SHAPES
from .example_calls import ExampleCall
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


def _tree_file(tree: Path, rel: str) -> bytes | None:
    """The first ``PARSE_LIMIT + 1`` bytes of the regular file *rel* names inside *tree*, else None.

    *rel* comes from a call record written beside model-written tests, so it is untrusted: an absolute path, a
    ``..`` part, a link out of the tree or anything but a regular file gives None.
    """
    parts = rel.split("/")
    if not rel or rel.startswith("/") or ".." in parts or "" in parts:
        return None
    root = Path(tree).resolve()
    path = root / rel
    try:
        if not path.resolve().is_relative_to(root):
            return None
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    except (OSError, ValueError, RuntimeError):
        return None
    try:
        if not stat.S_ISREG(os.fstat(fd).st_mode):
            return None
        chunks: list[bytes] = []
        left = PARSE_LIMIT + 1
        while left > 0:
            chunk = os.read(fd, left)
            if not chunk:
                break
            chunks.append(chunk)
            left -= len(chunk)
        return b"".join(chunks)
    except OSError:
        return None
    finally:
        os.close(fd)


def arg_descriptors(
    calls: list[ExampleCall],
    tree: Path,
    *,
    note: Callable[[str], None] | None = None,
) -> list[dict]:
    """The argument shapes of *calls* (a function's example calls), each naming its parameter as ``"param"``.

    Per call that returned and per argument: a path argument (one naming a file of *tree*, the candidate's
    files) is shaped from that file's bytes (``file_shape``), any other from its recorded JSON form
    (``value_shape``). A scalar or plain text has no shape and adds nothing; a call that refused or raised adds
    nothing. Descriptors holding a key-shaped string are dropped, as are arguments too large to be recorded and
    descriptors past :data:`.evidence.MAX_INPUT_SHAPES`; *note* is told how many, and which path arguments named
    no readable file of the tree. Never raises: shapes are a catalogue aid, never a reason.
    """
    out: list[dict] = []
    dropped = 0
    unreadable: set[str] = set()
    for call in calls:
        if call.outcome != "returned":
            continue
        for arg in call.args:
            desc: dict | None = None
            try:
                if arg.path is not None:
                    data = _tree_file(tree, arg.path)
                    if data is None:
                        unreadable.add(arg.path)
                        continue
                    desc = file_shape(arg.path, data)
                elif arg.cut:
                    dropped += 1
                    continue
                else:
                    desc = value_shape(arg.json)
            except Exception:  # noqa: BLE001 - shapes are a catalogue aid, never a failure
                dropped += 1  # a value the shape reader could not walk is counted with the dropped ones
                continue
            if desc is None:
                continue
            desc = {**desc, "param": arg.name}
            if KEY_SHAPED.search(json.dumps(desc, sort_keys=True)):
                dropped += 1
                continue
            out.append(desc)
    distinct = {json.dumps(d, sort_keys=True, ensure_ascii=False) for d in out}
    dropped += max(0, len(distinct) - MAX_INPUT_SHAPES)
    if note is not None:
        item = ", ".join(sorted({c.item for c in calls}))
        if dropped:
            note(
                f"G2 shapes: {dropped} argument shapes of {item} dropped (key-shaped or over the size bound)",
            )
        if unreadable:
            note(
                f"G2 shapes: {len(unreadable)} path arguments of {item} name no regular file of the library "
                "(" + ", ".join(sorted(unreadable)) + "); no shape is taken from them",
            )
    return merge_shapes(out)


def _functions(
    tree: Path,
    library_mode: bool = False,
) -> dict[str, tuple[str, str | None]]:
    """item -> (body digest, declared input form or None) of the environment functions in *tree*; with *library_mode*
    the library function items, whose input form comes from the item records (None until then).
    """
    if library_mode:
        from .layout import function_bodies

        return {
            item: (body_digest(body), None)
            for item, body in sorted(function_bodies(Path(tree)).items())
        }
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
    *,
    library_mode: bool = False,
) -> Rows:
    """A candidate's snapshot: per function, the parent's shapes for the same body plus this merge's.

    *new* maps an item to (body digest, descriptors); in library mode the descriptors are
    :func:`arg_descriptors`' (with ``"param"``), and a parent's rows without ``"param"`` are carried as they are.
    """
    out: Rows = {}
    for item, (digest, _) in sorted(_functions(tree, library_mode).items()):
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
    *,
    library_mode: bool = False,
) -> Rows:
    """Shapes derived from the evidence store's validated covers of each function (of *only*, if given)."""
    out: Rows = {}
    for item, (digest, form) in sorted(_functions(tree, library_mode).items()):
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
    library_mode: bool = False,
) -> Rows:
    """The shape rows of commit *sha* (whose files are at *tree*); see the module docstring."""
    functions = _functions(tree, library_mode)
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
        rows.update(
            backfill(
                tree,
                evidence,
                lookup,
                blobs,
                only=missing,
                library_mode=library_mode,
            ),
        )
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
