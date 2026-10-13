"""Item records written by the code that writes them in production, for the memory test worlds.

A world names its recorded episodes, the library committed before the pass, the captured procedures it lands and the
writer's own files, fixtures and manifest entries. :func:`build` then runs what a writer pass and the consolidation
after it run, in order:

1. the capture step (``captures.capture``) on each landed episode's staged ``procedure.py`` and ``summary.md``: the
   capture store's procedure, applicability signature and record;
2. ``land_capture`` in the writer's box, which holds the library: the reader, the procedure, their fixtures and test,
   the manifest entries, and the landing in the capture store (``landed.json``);
3. the writer's own files, its fixtures through the fixture tool (a parsed fixture runs its reader confined) and its
   manifest entries;
4. the library gate's check of that candidate (``Gate.check``), whose verification records hold each function's
   level, example and readers;
5. ``lifecycle.next_records`` over the candidate's tree, given the gate's verification and the capture store's
   landings (``captures.landed_items``).

Nothing here sets a field of a record. The capture step, the landing, the parsed fixtures and the gate run code
confined, so a world needs bubblewrap and prlimit (:data:`needs_box`).
"""

from __future__ import annotations

import json
import os
import shutil
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

from tests.memory.library_example_world import candidate
from tests.memory.test_library_helper import _commit
from unify.memory import captures
from unify.memory import fixtures as fx
from unify.memory import land_capture
from unify.memory import lifecycle
from unify.memory import sandbox_run as sr
from unify.memory import writer_tools
from unify.memory.blobs import BlobStore
from unify.memory.evidence import EvidenceStore
from unify.memory.gate import Gate
from unify.memory.gitio import Repo
from unify.memory.layout import discover, import_graph
from unify.memory.library_checks import LibraryConfig

needs_box = pytest.mark.skipif(
    shutil.which("bwrap") is None or not os.access(sr.PRLIMIT, os.X_OK),
    reason="bubblewrap and prlimit required",
)


@dataclass(frozen=True)
class Landing:
    """A captured procedure the writer lands: the episode it was captured from, the ``procedure.py`` and
    ``summary.md`` the episode-end fork staged, and the package and name ``land_capture`` is given."""

    episode: str
    procedure: str
    summary: str
    package: str
    name: str


@dataclass
class Built:
    """A built world: the library the gate checked (path -> text, tests and fixtures included), its item records,
    the gate's result and manifest, the capture store and each landed episode's capture id."""

    files: dict[str, str]
    records: dict[str, dict]
    result: Any
    manifest: dict
    store: Path
    captures: dict[str, str] = field(default_factory=dict)
    commit: str = ""


def _merge(manifest: dict, entries: Sequence[Mapping]) -> dict:
    """*manifest* with the writer's *entries*: a new item is added; an item already listed (a landing's) gains the
    entry's tests and source episodes."""
    items = [dict(i) for i in manifest.get("items") or [] if isinstance(i, dict)]
    for entry in entries:
        old = next((i for i in items if i.get("item") == entry["item"]), None)
        if old is None:
            items.append(dict(entry))
            continue
        for key in ("tests", "source_episodes"):
            have = list(old.get(key) or [])
            old[key] = have + [v for v in entry.get(key) or [] if v not in have]
    return {**manifest, "items": items}


def _tree_files(box: Path) -> dict[str, str]:
    """Every file of the box as the pass commits it: all but ``.pass``."""
    out: dict[str, str] = {}
    for p in sorted(box.rglob("*")):
        rel = p.relative_to(box).as_posix()
        if p.is_file() and not rel.startswith(".pass/"):
            out[rel] = p.read_text(encoding="utf-8")
    return out


def build(
    root: Path,
    *,
    episodes: Mapping[str, Any],
    base: Mapping[str, str],
    landings: Sequence[Landing] = (),
    files: Mapping[str, str] | None = None,
    fixtures: Sequence[Mapping] = (),
    items: Sequence[Mapping] = (),
) -> Built:
    """The world under *root* (module docstring): *base* is committed first; the pass lands *landings*, then writes
    *files* (path -> text), makes *fixtures* (fixture tool arguments) and lists *items* beside the landings' manifest
    entries."""
    root = Path(root)
    episodes = dict(episodes)
    batch = sorted(episodes)
    mem = Repo.init_bare(root / "mem.git")
    if base:
        _commit(mem, dict(base), "the library before the pass")
    ev = EvidenceStore(root / "evidence.sqlite")
    for n, ep in enumerate(episodes.values()):
        ev.index_episode(ep, str(n + 1) * 40)
    blobs = BlobStore(root / "blobs")
    store = root / "captures"
    box = root / "box"
    for rel, text in base.items():
        (box / rel).parent.mkdir(parents=True, exist_ok=True)
        (box / rel).write_text(text, encoding="utf-8")
    box.mkdir(parents=True, exist_ok=True)

    captured: dict[str, str] = {}
    for landing in landings:
        staging = root / "staging" / landing.episode
        staging.mkdir(parents=True)
        (staging / captures.PROCEDURE_FILE).write_text(landing.procedure, encoding="utf-8")
        (staging / captures.SUMMARY_FILE).write_text(landing.summary, encoding="utf-8")
        row = captures.capture(staging, episodes[landing.episode], store)
        assert row["safety"] == "ok" and row["capture_id"], row
        captured[landing.episode] = row["capture_id"]
        reply = land_capture.land(
            {"capture_id": row["capture_id"], "package": landing.package, "name": landing.name},
            store=store,
            box=box,
            episode=episodes.get,
            blob=blobs.get,
            batch=batch,
        )
        assert reply.status == "ok", reply.text

    for rel, text in (files or {}).items():
        (box / rel).parent.mkdir(parents=True, exist_ok=True)
        (box / rel).write_text(text, encoding="utf-8")
    for args in fixtures:
        reply = writer_tools.fixture(dict(args), episodes.get, box, blobs.get, batch)
        assert reply.status == "ok", reply.text
    drafted = box / writer_tools.MANIFEST
    manifest = json.loads(drafted.read_text(encoding="utf-8")) if drafted.is_file() else {}
    manifest = _merge(manifest, items)
    writer_tools.write_manifest(box, manifest)

    # what the pass hands the gate: the draft manifest, with the fixture provenance the harness derives from the box
    checked = {
        "deleted": [],
        "unlisted": [],
        "support": [],
        **manifest,
        "fixtures": fx.derive_ledger(box),
    }
    tree = _tree_files(box)
    commit = candidate(mem, tree)

    def lookup(eid: str, i: int):
        ep = episodes.get(eid)
        return ep.actions[i] if ep is not None and 0 <= i < len(ep.actions) else None

    gate = Gate(
        mem,
        ev,
        blobs,
        action_lookup=lookup,
        library_mode=LibraryConfig(checks=True, episodes=episodes.get),
    )
    result = gate.check(mem.head(), commit, checked)
    assert result.passed, result.reasons

    with mem.temp_checkout(commit) as wt:
        lib = discover(wt)
        graph = import_graph(wt)
    kinds = {f.item_id: "function" for f in lib.functions} | {n.item_id: "note" for n in lib.notes}
    provenance = {
        i["item"]: {"episodes": list(i.get("source_episodes") or []), "pass": None}
        for i in manifest.get("items") or []
    }
    records, _ = lifecycle.next_records(
        items=kinds,
        prev={},
        facts=[],
        since={i: {commit} for i in kinds},
        changed_at={i: commit for i in kinds},
        functions={f.item_id: f.module for f in lib.functions},
        graph=graph,
        note_uses={n.item_id: list(n.uses) for n in lib.notes},
        verification=result.verification,
        provenance=provenance,
        landed=captures.landed_items(store),
    )
    return Built(tree, records, result, checked, store, captured, commit)
