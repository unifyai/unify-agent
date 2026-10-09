"""Drafts (spec v2.1 §8.4, D41): work the gate refused after the repair rounds, given to the next WRITE pass.

A draft is not a branch (D13 stands). It is the patch the gate already saves with a pass's row
(``passes.patch_blob``; :meth:`.gate.Gate._record`: a refused pass keeps its whole patch, a reduced one what it
refused) plus the pass's final full gate result (``pass_rounds.gate_blob``). Only v2.1 passes leave drafts.

States, decided only from the evidence store, in pass order (:meth:`.evidence.EvidenceStore.pass_rounds`):

* ``finished``: every item the draft names (its pass's ``items_refused``) has since landed (some later WRITE
  pass's ``items_merged``);
* ``archived``: :data:`DRAFT_IDLE_PASSES` later WRITE passes in a row, since the draft or its last touch, did
  not touch it. A pass touches a draft when its manifest named one of the draft's items (in its
  ``items_merged`` or ``items_refused``), whether it landed or not. A draft that names no item is never touched;
* ``open``: neither. Open drafts are staged for the next WRITE pass; drafts never reach the actor.

A state, once reached, is final. Finished and archived drafts stay in the evidence and blob stores.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

from .blobs import BlobStore
from .evidence import EvidenceStore

DRAFT_IDLE_PASSES = 3
_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}\Z")


def draft_states(rows: list[dict], idle: int = DRAFT_IDLE_PASSES) -> dict[str, str]:
    """Each pass among *rows* (oldest first) that left a draft -> ``open``, ``finished`` or ``archived``."""
    out: dict[str, str] = {}
    for i, row in enumerate(rows):
        if not row.get("patch_blob"):
            continue
        names = set(row["items_refused"])
        landed: set[str] = set()
        quiet = 0
        state = "open"
        for later in rows[i + 1 :]:
            landed |= set(later["items_merged"])
            if names and names <= landed:
                state = "finished"
                break
            attempted = set(later["items_merged"]) | set(later["items_refused"])
            quiet = 0 if names & attempted else quiet + 1
            if quiet >= idle:
                state = "archived"
                break
        out[row["pass_id"]] = state
    return out


def _blob(blobs: BlobStore, sha: object, what: str) -> bytes:
    """A blob's bytes, or a one-line marker naming what is missing or corrupt (never a silent gap, and never a
    damaged patch passed on as if whole): the bytes must hash to their id, as the store wrote them.
    """
    if not (isinstance(sha, str) and blobs.has(sha)):
        return f"({what} {sha} is not in the blob store)\n".encode()
    data = blobs.get(sha)
    if hashlib.sha256(data).hexdigest() != sha:
        return f"({what} {sha} is corrupt in the blob store)\n".encode()
    return data


def stage_drafts(
    ev: EvidenceStore,
    blobs: BlobStore,
    inputs: Path,
    idle: int = DRAFT_IDLE_PASSES,
) -> list[dict]:
    """Write ``<inputs>/drafts/<pass_id>/patch.diff`` and ``gate.md`` per open draft, ``drafts/index.json``,
    and ``<inputs>/gate/previous.md`` (the last v2.1 WRITE pass's final full result, landed or not: spec
    §9.3). Returns the staged drafts as ``[{"pass_id", "items"}]``, in pass order.
    """
    rows = ev.pass_rounds("write")
    states = draft_states(rows, idle)
    gate_dir = Path(inputs) / "gate"
    gate_dir.mkdir(exist_ok=True)
    if rows:
        last = rows[-1]
        (gate_dir / "previous.md").write_bytes(
            _blob(
                blobs,
                last["gate_blob"],
                f"the full gate result of pass {last['pass_id']}",
            ),
        )
    staged: list[dict] = []
    skipped: list[str] = []
    for row in rows:
        pid = row["pass_id"]
        if states.get(pid) != "open":
            continue
        if not isinstance(pid, str) or not _SAFE_ID.match(pid):
            skipped.append(str(pid)[:200])
            continue
        d = Path(inputs) / "drafts" / pid
        d.mkdir(parents=True)
        (d / "patch.diff").write_bytes(_blob(blobs, row["patch_blob"], "patch"))
        (d / "gate.md").write_bytes(_blob(blobs, row["gate_blob"], "gate result"))
        staged.append({"pass_id": pid, "items": sorted(row["items_refused"])})
    if staged or skipped:
        (Path(inputs) / "drafts").mkdir(exist_ok=True)
        index = {"open": staged, **({"skipped": skipped} if skipped else {})}
        (Path(inputs) / "drafts" / "index.json").write_text(
            json.dumps(index, indent=1) + "\n",
        )
    return staged


def drafts_message(drafts: list[dict], previous: bool) -> str:
    """The lines of Sol's first message that name the open drafts and the last full result."""
    lines = [
        "Open drafts (work an earlier pass's gate refused; each is /inputs/drafts/<pass>/patch.diff and "
        "gate.md, and you may finish it):",
    ]
    lines += [
        f"- {d['pass_id']}: {', '.join(d['items']) or '(no item named)'}"
        for d in drafts
    ] or ["- none"]
    if previous:
        lines.append("The last pass's full gate result: /inputs/gate/previous.md")
    return "\n".join(lines)
