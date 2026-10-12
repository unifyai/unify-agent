"""The candidate pool: proposals persist across passes until the writer settles them.

Every staged candidate and lesson, every draft a pass left and every episode the writer deferred becomes an entry of
a git-tracked pool: the memory repo's ref ``refs/heads/pool``, never served to the actor. A store pass's own staging
enters the pool when the pass *starts* (:meth:`Pool.enter_staging`), so the writer can merge this pass's candidates
in the same pass. Each pass sees every open entry (``/inputs/pool/INDEX.md`` and ``/inputs/pool/<id>/``, read-only,
untrusted), and the closed entries that received new evidence after they were closed. Only the writer changes an
entry's status: through its manifest's ``pool`` block (``merged`` and ``superseded`` take effect when the gate lands
the pass, ``retired`` and ``reopened`` when finish is accepted), or by adopting the entry in an item (the item's
``adopts``), which closes it as merged when that item lands. There is no timer, cap or expiry: an entry stays open
until the writer decides. Closed entries leave the open list and stay in git history.

Entries are keyed by their content (a draft by its pass, a deferral by its episode), never by a task id, a stream
position or a count. Layout per entry: ``pool/<kind>/<id>/entry.json`` and its payload files.
"""

from __future__ import annotations

import ast
import hashlib
import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path

POOL_REF = "pool"
# ``cluster`` entries came from a recurrence miner that no longer feeds the pool; old ones still load, and
# :meth:`Pool.close_cluster_entries` closes them
KINDS = ("candidate", "lesson", "cluster", "draft", "deferred", "unread")
# the order the writer sees: proposals first (candidates, lessons, deferrals), then everything else
ORDER = ("candidate", "lesson", "deferred", "draft", "unread", "cluster")
STATUSES = ("open", "merged", "superseded", "retired")
BLOCK_KEYS = ("merged", "superseded", "retired", "reopened")
TITLE_CHARS = 160
MINER_RETIRED = "miner retired"
_ID = re.compile(r"[0-9a-f]{16}")


def _clip(text: str, n: int = TITLE_CHARS) -> str:
    """*text* on one line (whitespace collapsed), at most *n* characters."""
    one = " ".join(str(text).split())
    return one if len(one) <= n else one[: n - 1].rstrip() + "…"


def candidate_title(name: str, source: str) -> str:
    """A candidate's title from its content: the signature and the docstring's first line of the function named
    *name* (else the file's first top-level function); the file's first line when it does not parse.
    """
    try:
        tree = ast.parse(source)
    except (SyntaxError, ValueError):
        tree = None
    fns = [
        n
        for n in (tree.body if tree is not None else [])
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
    ]
    fn = next((f for f in fns if f.name == name), fns[0] if fns else None)
    if fn is None:
        first = next((ln.strip() for ln in source.splitlines() if ln.strip()), "")
        return _clip(f"{name}: {first}" if first and name else first or name)
    sig = f"{fn.name}({ast.unparse(fn.args)})"
    if fn.returns is not None:
        sig += f" -> {ast.unparse(fn.returns)}"
    doc = (ast.get_docstring(fn) or "").strip().splitlines()
    return _clip(f"{sig}: {doc[0].strip()}" if doc and doc[0].strip() else sig)


def lesson_title(notes: str) -> str:
    """A lesson's title: the first non-empty line of its notes, without a heading marker."""
    first = next((ln.strip() for ln in notes.splitlines() if ln.strip()), "")
    return _clip(first.lstrip("#").strip() or "notes")


def title_of(entry: "Entry") -> str:
    """The content title of *entry*: a candidate's signature and docstring line, a lesson's first line, a deferral's
    reason; any other kind keeps its title (clipped)."""
    if entry.kind == "candidate" and "candidate.py" in entry.files:
        return candidate_title(
            str(entry.data.get("name") or ""),
            entry.files["candidate.py"],
        )
    if entry.kind == "lesson" and "notes.md" in entry.files:
        return lesson_title(entry.files["notes.md"])
    if entry.kind == "deferred" and entry.data.get("reason"):
        return _clip(str(entry.data["reason"]))
    return _clip(entry.title)


def safe_rel(rel: str) -> bool:
    """A payload path the pool may hold: relative, its parts plain names (no ``..``, ``.`` or empty part), no
    backslash or NUL (the host writes these under /inputs/pool)."""
    if not rel or rel.startswith("/") or "\\" in rel or "\0" in rel:
        return False
    return all(part not in ("", ".", "..") for part in rel.split("/"))


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
    )  # merged: {item, pass}; superseded: {by, pass} or {reason, pass}; retired: {reason, pass}
    provenance: dict = field(
        default_factory=lambda: {"episodes": [], "passes": [], "sources": []},
    )
    opened_in: str = ""
    last_changed_in: str = ""
    data: dict = field(
        default_factory=dict,
    )  # a candidate's name, a deferral's reason, evidence that arrived after the entry closed, ...
    files: dict[str, str] = field(
        default_factory=dict,
    )  # payload files by relative path (text)
    # the staging refs (``staging:<episode>/<file>``) this entry was built from; held in memory only, never saved
    refs: list[str] = field(default_factory=list, compare=False, repr=False)

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

    def add_provenance(self, episodes=(), passes=(), sources=()) -> list[str]:
        """Add provenance; returns the episodes that were new to this entry."""
        new_eps: list[str] = []
        for key, vals in (
            ("episodes", episodes),
            ("passes", passes),
            ("sources", sources),
        ):
            have = self.provenance.setdefault(key, [])
            for v in vals:
                if v not in have:
                    have.append(v)
                    if key == "episodes":
                        new_eps.append(v)
        return new_eps

    def note_new_evidence(self, episodes: list[str], pass_id: str) -> None:
        """A closed entry received evidence: the episodes are listed under ``data.new_evidence`` until it reopens."""
        if self.status == "open" or not episodes:
            return
        rows = self.data.setdefault("new_evidence", [])
        for row in rows:
            if row.get("pass") == pass_id:
                row["episodes"] = row.get("episodes", []) + [
                    e for e in episodes if e not in row.get("episodes", [])
                ]
                return
        rows.append({"pass": pass_id, "episodes": list(episodes)})

    def new_evidence_episodes(self) -> list[str]:
        out: list[str] = []
        for row in self.data.get("new_evidence") or []:
            for e in row.get("episodes") or []:
                if e not in out:
                    out.append(e)
        return out


def _canon(obj) -> str:
    return json.dumps(obj, sort_keys=True, default=str)


def _order(e: Entry) -> tuple:
    return (ORDER.index(e.kind) if e.kind in ORDER else 99, e.opened_in, e.id)


class Pool:
    """The pool on *repo*'s ``refs/heads/pool``: :meth:`load` reads it, :meth:`save` commits one change set."""

    def __init__(self, repo) -> None:
        self.repo = repo
        self.entries: dict[str, Entry] = {}
        self.head: str | None = None
        self.skipped: list[str] = []  # entries load() refused
        # this pass's staging refs (``staging:<episode>/<file>``) -> the entry holding them (:meth:`enter_staging`)
        self.staged: dict[str, str] = {}

    # -- read ------------------------------------------------------------------------------------------------
    def load(self) -> "Pool":
        """Read ``refs/heads/pool``. Only well-formed entries are loaded (``pool/<kind>/<id>/``, a known kind, a
        sha16 id equal to its ``entry.json``'s, plain relative file names, regular blobs only); anything else is
        skipped and listed in :attr:`skipped`, never staged. Titles are recomputed from content.
        """
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
        self.entries, self.skipped = {}, []
        if not self.head:
            return self
        by_dir: dict[tuple[str, str], list[str]] = {}
        bad: set[tuple[str, str]] = set()
        for rec in self.repo.run("ls-tree", "-r", "-z", self.head).split("\0"):
            if not rec:
                continue
            head, _, path = rec.partition("\t")
            mode, typ = (head.split() + ["", ""])[:2]
            parts = path.split("/")
            if len(parts) < 4 or parts[0] != "pool":
                self.skipped.append(f"outside pool/<kind>/<id>/: {path[:120]!r}")
                continue
            key, rel = (parts[1], parts[2]), "/".join(parts[3:])
            if parts[1] not in KINDS or not _ID.fullmatch(parts[2]):
                self.skipped.append(f"bad kind or id: {path[:120]!r}")
                continue
            if mode != "100644" or typ != "blob" or not safe_rel(rel):
                self.skipped.append(f"not a plain file: {path[:120]!r} ({mode} {typ})")
                bad.add(key)
                continue
            by_dir.setdefault(key, []).append(rel)
        for (kind, eid), rels in sorted(by_dir.items()):
            d = f"pool/{kind}/{eid}"
            if (kind, eid) in bad or "entry.json" not in rels:
                self.skipped.append(f"entry {d} skipped (a bad file or no entry.json)")
                continue
            try:
                meta = json.loads(self.repo.show(self.head, f"{d}/entry.json").decode())
            except ValueError:
                meta = None
            if (
                not isinstance(meta, dict)
                or meta.get("id") != eid
                or meta.get("kind") != kind
            ):
                self.skipped.append(f"entry {d} skipped (entry.json does not name it)")
                continue
            files = {
                r: self.repo.show(self.head, f"{d}/{r}").decode("utf-8", "replace")
                for r in rels
                if r != "entry.json"
            }
            e = Entry(
                id=eid,
                kind=kind,
                title=str(meta.get("title", "")),
                status=(
                    meta.get("status", "open")
                    if meta.get("status") in STATUSES
                    else "open"
                ),
                status_detail=meta.get("status_detail") or {},
                provenance=meta.get("provenance")
                or {"episodes": [], "passes": [], "sources": []},
                opened_in=meta.get("opened_in", ""),
                last_changed_in=meta.get("last_changed_in", ""),
                data=meta.get("data") or {},
                files=files,
            )
            e.title = title_of(e)
            self.entries[e.id] = e
        return self

    def open_entries(self) -> list[Entry]:
        return sorted(
            (e for e in self.entries.values() if e.status == "open"),
            key=_order,
        )

    def closed_with_new_evidence(self) -> list[Entry]:
        """Closed entries that received evidence after they closed (the writer may reopen them)."""
        return sorted(
            (
                e
                for e in self.entries.values()
                if e.status != "open" and e.new_evidence_episodes()
            ),
            key=_order,
        )

    # -- change ----------------------------------------------------------------------------------------------
    def upsert(self, entry: Entry, pass_id: str) -> Entry:
        """Add *entry*, or merge its provenance and data into the existing entry with the same id (one id, more
        evidence). A closed entry stays closed: new episodes for it are recorded as new evidence, and its status
        stays the writer's."""
        have = self.entries.get(entry.id)
        if have is None:
            entry.opened_in = entry.opened_in or pass_id
            entry.last_changed_in = pass_id
            entry.title = title_of(entry)
            self.entries[entry.id] = entry
            return entry
        before = _canon(have.meta()) + _canon(have.files)
        new_eps = have.add_provenance(
            entry.provenance.get("episodes", []),
            entry.provenance.get("passes", []),
            entry.provenance.get("sources", []),
        )
        have.note_new_evidence(new_eps, pass_id)
        inst = entry.data.get("instances")
        if inst:
            seen = {_canon(i) for i in have.data.get("instances", [])}
            have.data.setdefault("instances", []).extend(
                i for i in inst if _canon(i) not in seen
            )
        for rel, text in entry.files.items():
            have.files.setdefault(rel, text)
        for ref in entry.refs:
            if ref not in have.refs:
                have.refs.append(ref)
        if _canon(have.meta()) + _canon(have.files) != before:
            have.last_changed_in = pass_id
        return have

    def add_evidence(self, eid: str, episode: str, reason: str, pass_id: str) -> bool:
        """The writer's deferral that names entry *eid*: *episode* and its reason join that entry (its status is
        unchanged). False when there is no such entry."""
        e = self.entries.get(eid)
        if e is None:
            return False
        e.note_new_evidence(e.add_provenance([episode], [pass_id], ["writer"]), pass_id)
        ev = e.data.setdefault("evidence", [])
        row = {"episode": episode, "reason": reason, "pass": pass_id}
        if row not in ev:
            ev.append(row)
        e.last_changed_in = pass_id
        return True

    def set_status(self, eid: str, status: str, detail: dict, pass_id: str) -> None:
        e = self.entries[eid]
        e.status, e.status_detail, e.last_changed_in = (
            status,
            {**detail, "pass": pass_id},
            pass_id,
        )
        if status == "open":
            e.data.pop("new_evidence", None)

    def enter_staging(self, views: dict, pass_id: str) -> dict:
        """A store pass's staging enters the pool at the pass's start: each batch episode's staging
        (:class:`.staging.StagingView` by episode) as entries (:func:`from_staging`), and :attr:`staged` maps each
        staged file's ref to the entry holding it. Returns ``{"entries": n, "added": n}``.
        """
        added, held = 0, set()
        for eid, view in sorted(views.items()):
            for e in from_staging(eid, view):
                e.add_provenance([], [pass_id], [])
                added += e.id not in self.entries
                kept = self.upsert(e, pass_id)
                held.add(kept.id)
                for ref in e.refs:
                    self.staged[ref] = kept.id
        return {"entries": len(held), "added": added}

    def close_cluster_entries(self, pass_id: str) -> list[str]:
        """Close every open ``cluster`` entry as superseded (the miner that fed them is retired); returns their ids.
        A cleanup that runs when a pass loads the pool, so a pool with no cluster entries is unchanged.
        """
        out = []
        for e in self.open_entries():
            if e.kind == "cluster":
                self.set_status(e.id, "superseded", {"reason": MINER_RETIRED}, pass_id)
                out.append(e.id)
        return out

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
        """``/inputs/pool``: INDEX.md (every open entry, one line each, proposals first, deterministic; then the
        closed entries with new evidence) and each listed entry's files. Returns the number of open entries.

        *dest* must not exist yet (a fresh directory, never one reached through a link); every file is created new,
        never through a link, and must resolve under *dest*."""
        dest = Path(dest)
        dest.mkdir(parents=True, exist_ok=False)
        root = dest.resolve()

        def write(rel: str, text: str) -> None:
            if not safe_rel(rel):
                raise ValueError(f"pool: unsafe path {rel[:120]!r}")
            out = dest / rel
            out.parent.mkdir(parents=True, exist_ok=True)
            if not out.parent.resolve().is_relative_to(root):
                raise ValueError(f"pool: {rel[:120]!r} leaves /inputs/pool")
            fd = os.open(
                out,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC,
                0o644,
            )
            with os.fdopen(fd, "w") as fh:
                fh.write(text)

        def files_of(e: Entry) -> None:
            write(
                f"{e.id}/entry.json",
                json.dumps(e.meta(), indent=1, sort_keys=True, default=str) + "\n",
            )
            for rel, text in sorted(e.files.items()):
                if safe_rel(rel) and rel != "entry.json":
                    write(f"{e.id}/{rel}", text)

        def count(n: int) -> str:
            return f"{n} episode{'' if n == 1 else 's'}"

        rows = [e for e in self.open_entries() if _ID.fullmatch(e.id)]
        lines = [
            "# Candidate pool: open entries",
            "",
            "Proposals not yet settled, candidates first, then lessons and deferrals. Each stays open until a pass "
            "merges, supersedes or retires it in its manifest's `pool` block, or an item that adopts it lands. "
            "Files: /inputs/pool/<id>/.",
            "",
        ]
        for e in rows:
            n = len(e.provenance.get("episodes", []))
            lines.append(
                f"- `{e.id}` {e.kind}: {e.title} ({count(n)}; opened in {e.opened_in})",
            )
            files_of(e)
        if not rows:
            lines.append("(no open entries)")
        again = [e for e in self.closed_with_new_evidence() if _ID.fullmatch(e.id)]
        if again:
            lines += [
                "",
                "## Closed entries with new evidence",
                "",
                "Closed entries that received evidence after they were closed. A manifest's "
                '`"pool": {"reopened": {"<id>": "<reason>"}}` reopens one.',
                "",
            ]
            for e in again:
                lines.append(
                    f"- `{e.id}` {e.kind} ({e.status}): {e.title} "
                    f"({count(len(e.new_evidence_episodes()))} since it closed)",
                )
                files_of(e)
        write("INDEX.md", "\n".join(lines) + "\n")
        return len(rows)


# -- the writer's status changes ----------------------------------------------------------------------------


def adopted_entries(pool: Pool, refs) -> list[str]:
    """The pool entries the adopt refs *refs* name: ``pool:<id>`` directly, a staged file through
    :attr:`Pool.staged`. Unknown refs name nothing."""
    out: list[str] = []
    for ref in refs or []:
        ref = str(ref)
        eid = ref[len("pool:") :] if ref.startswith("pool:") else pool.staged.get(ref)
        if eid and eid in pool.entries and eid not in out:
            out.append(eid)
    return out


def _adopts_of(manifest: dict) -> dict[str, list[str]]:
    """Each manifest item's ``adopts`` refs (as written; the parser canonicalises and validates their form)."""
    out: dict[str, list[str]] = {}
    for it in manifest.get("items") or []:
        if isinstance(it, dict) and isinstance(it.get("adopts"), list):
            # entries are {"ref", "how"}; the ref names the draft
            out[str(it.get("item"))] = [
                str(r.get("ref")) if isinstance(r, dict) else str(r)
                for r in it["adopts"]
            ]
    return out


def block_problems(manifest: object, pool: Pool, kind: str = "write") -> list[str]:
    """Problems with *manifest*'s ``pool`` block and its items' ``adopts`` (G1 at finish): decisions must name open
    entries (``reopened``: closed ones); merged needs an item the manifest lists; retired and reopened need a
    reason; CURATE may supersede or retire, not merge; an adopted ref must name a pool entry or a file this pass
    staged, and an entry an item adopts cannot also be superseded or retired.
    """
    if not isinstance(manifest, dict):
        return []
    from .manifest import ManifestError, adopt_ref

    out = []
    adopted: dict[str, str] = {}
    for item, refs in _adopts_of(manifest).items():
        for raw in refs:
            try:
                ref = adopt_ref(raw)
            except ManifestError:
                continue  # the manifest parser reports a malformed ref
            if ref.startswith("helper:"):
                continue  # helper output is checked where it is exported
            eids = adopted_entries(pool, [ref])
            if not eids:
                where = (
                    "a pool entry"
                    if ref.startswith("pool:")
                    else "a file this pass staged"
                )
                out.append(f"{item}: adopts {ref[:120]!r}, which is not {where}")
            for eid in eids:
                adopted.setdefault(eid, item)
    if "pool" not in manifest:
        return out
    block = manifest.get("pool")
    if not isinstance(block, dict):
        return out + [
            "pool must be an object with merged, superseded, retired or reopened",
        ]
    items = {i.get("item") for i in manifest.get("items") or [] if isinstance(i, dict)}
    for key in block:
        if key not in BLOCK_KEYS:
            out.append(f"pool: unknown key {key!r}")
    for key in BLOCK_KEYS:
        val = block.get(key) or {}
        if not isinstance(val, dict):
            out.append(f"pool.{key} must map entry ids to values")
            continue
        for eid, v in val.items():
            e = pool.entries.get(str(eid))
            if key == "reopened":
                if e is None or e.status == "open":
                    out.append(f"pool.reopened: {str(eid)[:20]} is not a closed entry")
                elif not isinstance(v, str) or not v.strip():
                    out.append(f"pool.reopened: {eid} needs a reason")
                continue
            if e is None or e.status != "open":
                out.append(f"pool.{key}: {str(eid)[:20]} is not an open entry")
            elif key == "merged" and (kind == "curate" or v not in items):
                out.append(f"pool.merged: {eid} must name an item this manifest lists")
            elif not isinstance(v, str) or not v.strip():
                out.append(
                    f"pool.{key}: {eid} needs {'a value' if key == 'superseded' else 'a reason'}",
                )
            elif key in ("superseded", "retired") and str(eid) in adopted:
                out.append(
                    f"pool.{key}: {eid} is adopted by {adopted[str(eid)]}, which merges it when it lands",
                )
    # one decision per entry, and a supersession keeps its replacement open
    named: dict[str, str] = {}
    for key in BLOCK_KEYS:
        val = block.get(key)
        for eid in val if isinstance(val, dict) else {}:
            if str(eid) in named:
                out.append(
                    f"pool: {str(eid)[:20]} is named in both {named[str(eid)]} and {key}",
                )
            named.setdefault(str(eid), key)
    sup = block.get("superseded")
    for eid, by in (sup if isinstance(sup, dict) else {}).items():
        if by == eid:
            out.append(f"pool.superseded: {eid} cannot supersede itself")
        elif isinstance(by, str) and by in named:
            out.append(
                f"pool.superseded: {eid}'s replacement {by} is closed by this block too",
            )
    return out


def apply_block(
    pool: Pool,
    manifest: object,
    pass_id: str,
    *,
    landed: set[str],
    finished: bool,
    provenance: list[dict] | None = None,
) -> list[str]:
    """Apply the writer's status changes. From the ``pool`` block of an accepted manifest: merged only for an item
    the gate landed (*landed*); superseded by an item the gate landed, or by another pool entry once finish was
    accepted (no library change for the gate to land); retired and reopened only when finish was accepted. From the
    gate's *provenance* (``{"item", "adopts"}`` per landed item): every open entry a landed item adopts closes as
    merged. Returns the ids changed."""
    changed: list[str] = []
    if (
        isinstance(manifest, dict)
        and isinstance(manifest.get("pool"), dict)
        and not block_problems(manifest, pool)
    ):
        block = manifest["pool"]
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
            for eid, reason in (block.get("reopened") or {}).items():
                pool.set_status(eid, "open", {"reopened": reason}, pass_id)
                changed.append(eid)
    for row in provenance or []:
        item = row.get("item")
        if item not in landed:
            continue
        for eid in adopted_entries(pool, row.get("adopts") or []):
            if pool.entries[eid].status == "open":
                pool.set_status(eid, "merged", {"item": item, "via": "adopts"}, pass_id)
                changed.append(eid)
    return changed


# -- what feeds the pool ------------------------------------------------------------------------------------


def from_staging(eid: str, view) -> list[Entry]:
    """A reader's staging (:class:`.staging.StagingView`) as entries: each candidate with its case, and the notes
    as a lesson. Each entry's :attr:`Entry.refs` names the staged files it holds."""
    out = []
    files = {k: v.decode("utf-8", "replace") for k, v in view.files.items()}
    for rel, text in sorted(files.items()):
        if not rel.startswith("candidates/") or not rel.endswith(".py"):
            continue
        name = rel[len("candidates/") : -3]
        payload = {"candidate.py": text}
        refs = [f"staging:{eid}/{rel}"]
        case = files.get(f"cases/{name}.json")
        if case is not None:
            payload["cases.json"] = case
            refs.append(f"staging:{eid}/cases/{name}.json")
        e = Entry(
            id=entry_id("candidate", _canon(payload)),
            kind="candidate",
            title=candidate_title(name, text),
            data={"name": name},
            files=payload,
            refs=refs,
        )
        e.add_provenance([eid], [], [view.source])
        out.append(e)
    if "notes.md" in files and files["notes.md"].strip():
        e = Entry(
            id=entry_id("lesson", files["notes.md"]),
            kind="lesson",
            title=lesson_title(files["notes.md"]),
            files={"notes.md": files["notes.md"]},
            refs=[f"staging:{eid}/notes.md"],
        )
        e.add_provenance([eid], [], [view.source])
        out.append(e)
    return out


def deferred(eid: str, reason: str) -> Entry:
    e = Entry(
        id=entry_id("deferred", eid),
        kind="deferred",
        title=_clip(reason),
        data={"reason": reason},
    )
    e.add_provenance([eid], [], ["writer"])
    return e


def unread(eid: str, parts_read: int, parts_required: int) -> Entry:
    """An episode a store pass left neither read nor dismissed; kept, never dropped, until a writer closes it."""
    e = Entry(
        id=entry_id("unread", eid),
        kind="unread",
        title=f"episode unread ({parts_read} of {parts_required} parts read)",
        data={"parts_read": parts_read, "parts_required": parts_required},
    )
    e.add_provenance([eid], [], ["writer"])
    return e


def draft(pass_id: str, patch: str, result: str, items: list[str]) -> Entry:
    e = Entry(
        id=entry_id("draft", pass_id),
        kind="draft",
        title=_clip(f"draft of pass {pass_id}: {', '.join(items[:4]) or 'no items'}"),
        data={"items": items},
        files={"patch.diff": patch, "gate.md": result},
    )
    e.add_provenance([], [pass_id], ["gate"])
    return e
