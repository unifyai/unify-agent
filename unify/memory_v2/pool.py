"""Memory v2.1 candidate pool (the lead, 11 Oct; design ``docs/design/memory-v2.1-candidate-pool.md``).

Proposals are never thrown away after one pass. Every fork or analyst candidate and lesson, every S0 cluster, every
draft a pass left, and every episode the writer dismissed with ``defer=true`` becomes an entry of a git-tracked pool:
the memory repo's ref ``refs/heads/pool``, never served to the actor. Each WRITE (and CURATE) pass sees every open
entry (``/inputs/pool/INDEX.md`` and ``/inputs/pool/<id>/``, read-only, untrusted). Only the writer changes an
entry's status, through its manifest's ``pool`` block: ``merged`` and ``superseded`` take effect when the gate lands
the pass, ``retired`` (with a reason) when finish is accepted. There is no timer, cap or expiry: an entry stays open
until the writer decides. Closed entries leave the index (they are never re-checked) and stay in git history.

Entries are keyed by their content (or, for a cluster, its S0 key; for a draft, its pass; for a deferral, its
episode), never by a task id, a stream position or a count. Layout per entry: ``pool/<kind>/<id>/entry.json`` and
its payload files.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

POOL_REF = "pool"
KINDS = ("candidate", "lesson", "cluster", "draft", "deferred")
STATUSES = ("open", "merged", "superseded", "retired")


def entry_id(kind: str, key: str) -> str:
    return hashlib.sha256(f"{kind}\0{key}".encode()).hexdigest()[:16]


@dataclass
class Entry:
    id: str
    kind: str
    title: str
    status: str = "open"
    status_detail: dict = field(
        default_factory=dict,
    )  # merged: {item, pass}; superseded: {by, pass}; retired
    provenance: dict = field(
        default_factory=lambda: {"episodes": [], "passes": [], "sources": []},
    )
    opened_in: str = ""
    last_changed_in: str = ""
    data: dict = field(
        default_factory=dict,
    )  # cluster instances, a deferral's reason, ...
    files: dict[str, str] = field(
        default_factory=dict,
    )  # payload files by relative path (text)

    def meta(self) -> dict:
        return {
            "id": self.id,
            "kind": self.kind,
            "title": self.title,
            "status": self.status,
            "status_detail": self.status_detail,
            "provenance": self.provenance,
            "opened_in": self.opened_in,
            "last_changed_in": self.last_changed_in,
            "data": self.data,
            "files": sorted(self.files),
        }

    def add_provenance(self, episodes=(), passes=(), sources=()) -> None:
        for key, vals in (
            ("episodes", episodes),
            ("passes", passes),
            ("sources", sources),
        ):
            have = self.provenance.setdefault(key, [])
            for v in vals:
                if v not in have:
                    have.append(v)


def _canon(obj) -> str:
    return json.dumps(obj, sort_keys=True, default=str)


class Pool:
    """The pool on *repo*'s ``refs/heads/pool``: :meth:`load` reads it, :meth:`save` commits one change set."""

    def __init__(self, repo) -> None:
        self.repo = repo
        self.entries: dict[str, Entry] = {}
        self.head: str | None = None

    # -- read ------------------------------------------------------------------------------------------------
    def load(self) -> "Pool":
        try:
            self.head = (
                self.repo.run(
                    "rev-parse",
                    "--verify",
                    "-q",
                    f"refs/heads/{POOL_REF}",
                ).strip()
                or None
            )
        except Exception:  # noqa: BLE001 - no pool yet
            self.head = None
        self.entries = {}
        if not self.head:
            return self
        names = [
            n
            for n in self.repo.run(
                "ls-tree",
                "-r",
                "--name-only",
                self.head,
            ).splitlines()
            if n
        ]
        by_dir: dict[str, list[str]] = {}
        for n in names:
            parts = n.split("/")
            if len(parts) >= 4 and parts[0] == "pool":
                by_dir.setdefault("/".join(parts[:3]), []).append("/".join(parts[3:]))
        for d, rels in sorted(by_dir.items()):
            if "entry.json" not in rels:
                continue
            meta = json.loads(self.repo.show(self.head, f"{d}/entry.json").decode())
            files = {
                r: self.repo.show(self.head, f"{d}/{r}").decode("utf-8", "replace")
                for r in rels
                if r != "entry.json"
            }
            e = Entry(
                id=meta["id"],
                kind=meta["kind"],
                title=meta.get("title", ""),
                status=meta.get("status", "open"),
                status_detail=meta.get("status_detail") or {},
                provenance=meta.get("provenance")
                or {"episodes": [], "passes": [], "sources": []},
                opened_in=meta.get("opened_in", ""),
                last_changed_in=meta.get("last_changed_in", ""),
                data=meta.get("data") or {},
                files=files,
            )
            self.entries[e.id] = e
        return self

    def open_entries(self) -> list[Entry]:
        return sorted(
            (e for e in self.entries.values() if e.status == "open"),
            key=lambda e: (
                KINDS.index(e.kind) if e.kind in KINDS else 99,
                e.opened_in,
                e.id,
            ),
        )

    # -- change ----------------------------------------------------------------------------------------------
    def upsert(self, entry: Entry, pass_id: str) -> Entry:
        """Add *entry*, or merge its provenance and data into the existing entry with the same id (one id, more
        evidence). A closed entry stays closed: new evidence for it is recorded, its status is the writer's.
        """
        have = self.entries.get(entry.id)
        if have is None:
            entry.opened_in = entry.opened_in or pass_id
            entry.last_changed_in = pass_id
            self.entries[entry.id] = entry
            return entry
        before = _canon(have.meta()) + _canon(have.files)
        have.add_provenance(
            entry.provenance.get("episodes", []),
            entry.provenance.get("passes", []),
            entry.provenance.get("sources", []),
        )
        inst = entry.data.get("instances")
        if inst:
            seen = {_canon(i) for i in have.data.get("instances", [])}
            have.data.setdefault("instances", []).extend(
                i for i in inst if _canon(i) not in seen
            )
        for rel, text in entry.files.items():
            have.files.setdefault(rel, text)
        if _canon(have.meta()) + _canon(have.files) != before:
            have.last_changed_in = pass_id
        return have

    def set_status(self, eid: str, status: str, detail: dict, pass_id: str) -> None:
        e = self.entries[eid]
        e.status, e.status_detail, e.last_changed_in = (
            status,
            {**detail, "pass": pass_id},
            pass_id,
        )

    # -- write -----------------------------------------------------------------------------------------------
    def _blob(self, text: str) -> str:
        return self.repo.run("hash-object", "-w", "--stdin", input=text).strip()

    def _tree(self, node: dict) -> str:
        """*node*: name -> blob sha (str) or a sub-node (dict)."""
        lines = []
        for name in sorted(node):
            v = node[name]
            if isinstance(v, dict):
                lines.append(f"040000 tree {self._tree(v)}\t{name}")
            else:
                lines.append(f"100644 blob {v}\t{name}")
        if not lines:
            return self.repo.run("mktree", input="").strip()  # the empty tree
        return self.repo.run("mktree", input="\n".join(lines) + "\n").strip()

    def save(self, pass_id: str) -> str | None:
        """Commit the pool as it now is on ``refs/heads/pool`` (one commit, trailer ``Pass:``); None when nothing
        changed. The ref moves only from the head :meth:`load` read (a concurrent writer fails, never overwrites).
        """
        root: dict = {}
        for e in self.entries.values():
            d = root.setdefault("pool", {}).setdefault(e.kind, {}).setdefault(e.id, {})
            d["entry.json"] = self._blob(
                json.dumps(e.meta(), indent=1, sort_keys=True, default=str) + "\n",
            )
            for rel, text in e.files.items():
                node = d
                parts = rel.split("/")
                for p in parts[:-1]:
                    node = node.setdefault(p, {})
                node[parts[-1]] = self._blob(text)
        tree = self._tree(root)
        if (
            self.head
            and self.repo.run("rev-parse", f"{self.head}^{{tree}}").strip() == tree
        ):
            return None
        args = ["commit-tree", tree, "-m", f"pool: pass {pass_id}\n\nPass: {pass_id}"]
        if self.head:
            args[2:2] = ["-p", self.head]
        commit = self.repo.run(*args).strip()
        self.repo.run(
            "update-ref",
            f"refs/heads/{POOL_REF}",
            commit,
            self.head or "0" * 40,
        )
        self.head = commit
        return commit

    # -- the writer's view -----------------------------------------------------------------------------------
    def stage(self, dest: Path) -> int:
        """``/inputs/pool``: INDEX.md (every open entry, one line each, deterministic) and each entry's files."""
        dest.mkdir(parents=True, exist_ok=True)
        rows = self.open_entries()
        lines = [
            "# Candidate pool: open entries",
            "",
            "Proposals earlier passes did not settle. Each stays open until a pass merges, supersedes or retires it "
            "in its manifest's `pool` block. Files: /inputs/pool/<id>/.",
            "",
        ]
        for e in rows:
            n = len(e.provenance.get("episodes", []))
            lines.append(
                f"- `{e.id}` {e.kind}: {e.title} ({n} episode(s); opened in {e.opened_in})",
            )
            d = dest / e.id
            d.mkdir(exist_ok=True)
            (d / "entry.json").write_text(
                json.dumps(e.meta(), indent=1, sort_keys=True, default=str) + "\n",
            )
            for rel, text in e.files.items():
                out = d / rel
                out.parent.mkdir(parents=True, exist_ok=True)
                out.write_text(text)
        if not rows:
            lines.append("(no open entries)")
        (dest / "INDEX.md").write_text("\n".join(lines) + "\n")
        return len(rows)


# -- the writer's status changes ----------------------------------------------------------------------------


def block_problems(manifest: object, pool: Pool, kind: str = "write") -> list[str]:
    """Problems with *manifest*'s ``pool`` block (G1 at finish): it must name open entries; merged needs an item the
    manifest lists; retired needs a reason; CURATE may supersede or retire, not merge.
    """
    if not isinstance(manifest, dict) or "pool" not in manifest:
        return []
    block = manifest.get("pool")
    if not isinstance(block, dict):
        return ["pool must be an object with merged, superseded and retired"]
    out = []
    items = {i.get("item") for i in manifest.get("items") or [] if isinstance(i, dict)}
    for key in block:
        if key not in ("merged", "superseded", "retired"):
            out.append(f"pool: unknown key {key!r}")
    for key in ("merged", "superseded", "retired"):
        val = block.get(key) or {}
        if not isinstance(val, dict):
            out.append(f"pool.{key} must map entry ids to values")
            continue
        for eid, v in val.items():
            e = pool.entries.get(str(eid))
            if e is None or e.status != "open":
                out.append(f"pool.{key}: {str(eid)[:20]} is not an open entry")
            elif key == "merged" and (kind == "curate" or v not in items):
                out.append(f"pool.merged: {eid} must name an item this manifest lists")
            elif not isinstance(v, str) or not v.strip():
                out.append(
                    f"pool.{key}: {eid} needs {'a reason' if key == 'retired' else 'a value'}",
                )
    return out


def apply_block(
    pool: Pool,
    manifest: object,
    pass_id: str,
    *,
    landed: set[str],
    finished: bool,
) -> list[str]:
    """Apply the writer's status changes: merged only for an item the gate landed (*landed*); superseded by an item
    the gate landed, or by another pool entry once finish was accepted (no library change for the gate to land);
    retired only when finish was accepted. Returns the ids changed."""
    if (
        not isinstance(manifest, dict)
        or not isinstance(manifest.get("pool"), dict)
        or block_problems(manifest, pool)
    ):
        return []
    block, changed = manifest["pool"], []
    for eid, item in (block.get("merged") or {}).items():
        if item in landed:
            pool.set_status(eid, "merged", {"item": item}, pass_id)
            changed.append(eid)
    for eid, by in (block.get("superseded") or {}).items():
        if by in landed or (finished and by in pool.entries and by != eid):
            pool.set_status(eid, "superseded", {"by": by}, pass_id)
            changed.append(eid)
    if finished:
        for eid, reason in (block.get("retired") or {}).items():
            pool.set_status(eid, "retired", {"reason": reason}, pass_id)
            changed.append(eid)
    return changed


# -- what feeds the pool ------------------------------------------------------------------------------------


def from_staging(eid: str, view) -> list[Entry]:
    """A fork's or analyst's staging (:class:`.staging.StagingView`) as entries: each candidate with its case, and
    the notes as a lesson."""
    out = []
    files = {k: v.decode("utf-8", "replace") for k, v in view.files.items()}
    for rel, text in sorted(files.items()):
        if not rel.startswith("candidates/") or not rel.endswith(".py"):
            continue
        name = rel[len("candidates/") : -3]
        payload = {"candidate.py": text}
        case = files.get(f"cases/{name}.json")
        if case is not None:
            payload["cases.json"] = case
        e = Entry(
            id=entry_id("candidate", _canon(payload)),
            kind="candidate",
            title=f"candidate {name} from episode {eid}",
            files=payload,
        )
        e.add_provenance([eid], [], [view.source])
        out.append(e)
    if "notes.md" in files and files["notes.md"].strip():
        e = Entry(
            id=entry_id("lesson", files["notes.md"]),
            kind="lesson",
            title=f"lesson from episode {eid}: {files['notes.md'].strip().splitlines()[0][:80]}",
            files={"notes.md": files["notes.md"]},
        )
        e.add_provenance([eid], [], [view.source])
        out.append(e)
    return out


def from_clusters(tier_a: list, tier_b: dict) -> list[Entry]:
    """S0's clusters (``clusters.json``, ``behaviour.json``) as entries: one id per cluster key, instances added."""
    out = []
    for c in tier_a or []:
        inst = [
            {"episode": a["episode"], "cell": a["cell"], "lines": a["lines"]}
            for a in c.get("at", [])
        ]
        e = Entry(
            id=entry_id("cluster", f"a:{c['key']}"),
            kind="cluster",
            title=f"code shape {c['key'][:8]} ({c.get('kind')}, {c.get('size')} nodes)",
            data={
                "key": c["key"],
                "template": c.get("template", ""),
                "instances": inst,
            },
        )
        e.add_provenance(sorted({i["episode"] for i in inst}), [], ["s0"])
        out.append(e)
    units = (tier_b or {}).get("units") or {}
    for c in (tier_b or {}).get("clusters") or []:
        members = sorted(c.get("units") or [])
        us = [units[m] for m in members if m in units]
        inst = [
            {"episode": u["episode"], "cell": u["cell"], "lines": u["lines"]}
            for u in us
        ]
        e = Entry(
            id=entry_id("cluster", "b:" + ",".join(sorted(c.get("akeys") or []))),
            kind="cluster",
            title=f"behaviour group of {len(c.get('akeys') or [])} code shapes",
            data={"akeys": sorted(c.get("akeys") or []), "instances": inst},
            files={f"unit-{i}.py": u.get("src", "") for i, u in enumerate(us)},
        )
        e.add_provenance(sorted({i["episode"] for i in inst}), [], ["s0"])
        out.append(e)
    return out


def deferred(eid: str, reason: str) -> Entry:
    e = Entry(
        id=entry_id("deferred", eid),
        kind="deferred",
        title=f"episode {eid} deferred: {reason[:100]}",
        data={"reason": reason},
    )
    e.add_provenance([eid], [], ["writer"])
    return e


def draft(pass_id: str, patch: str, result: str, items: list[str]) -> Entry:
    e = Entry(
        id=entry_id("draft", pass_id),
        kind="draft",
        title=f"draft of pass {pass_id}: {', '.join(items[:4])}",
        data={"items": items},
        files={"patch.diff": patch, "gate.md": result},
    )
    e.add_provenance([], [pass_id], ["gate"])
    return e
