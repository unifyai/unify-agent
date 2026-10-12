"""The writer's tools: the handlers that need no pass state. Each tool's definition is in the shared registry
(:mod:`unify.common.tool_registry`).

Every tool JSON the writer, the curator or a helper receives is rendered from the registry (through
:mod:`.agent_profiles`), and every refusal text comes from a spec's ``refusals`` (or :data:`GENERAL_REFUSALS` for the
answers the pass loop itself gives), each with a stable form code that the structured tool events record.

Tool state is derived from files the model can see: fixture provenance from the fixture files themselves
(:func:`.fixtures.derive_ledger`), pool decisions from the manifest's ``pool`` block, and the pool from
``/inputs/pool``. Reading records nothing that changes what a later call may do; the read log is a measure only.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from . import batch_map as _bm
from . import fixtures as _fx
from . import pool as _pool
from . import views as _views
from unify.common import tool_registry as _registry

from .tool_spec import GENERAL, refusal
from .tool_spec import GENERAL_REFUSALS as _GENERAL_REFUSALS

#: parts per read_episode call, reader calls per model turn and cells per model turn (stated in the descriptions)
READ_PARTS_PER_CALL = _registry.READ_PARTS_PER_CALL
READERS_PER_TURN = _registry.READERS_PER_TURN
CELLS_PER_TURN = _registry.CELLS_PER_TURN
#: where the manifest lives in the writer's box
MANIFEST = ".pass/manifest.json"

_POOL_STATUS = ("open", "closed", "any")

# The tools' definitions live in the shared registry (unify/common/tool_registry.py); these names stay for callers.
READ = _registry.READ
GREP = _registry.GREP
OVERVIEW = _registry.OVERVIEW
READ_EPISODE = _registry.READ_EPISODE
POOL = _registry.POOL
FIXTURE = _registry.FIXTURE
REPLAY = _registry.REPLAY
#: the writer's and curator's execute_code as the model-API tool sends it while UNIFY_TOOLS_IN_CODE is off
EXECUTE_CODE = _registry.earlier_tool("writer", "execute_code")
CHECK = _registry.CHECK
FINISH = _registry.FINISH

SPECS = (READ, GREP, OVERVIEW, READ_EPISODE, POOL, FIXTURE, REPLAY, EXECUTE_CODE, CHECK, FINISH)

#: Answers the pass loop gives on any tool, by stable form code (defined with the specs in :mod:`.tool_spec`).
GENERAL_REFUSALS = _GENERAL_REFUSALS
#: The numbers the per-turn answers state, filled in when a caller does not give them.
_GENERAL_FIELDS = {
    "readers_per_turn": {"limit": READERS_PER_TURN},
    "cells_per_turn": {"limit": CELLS_PER_TURN},
}


@dataclass(frozen=True)
class Reply:
    """A tool's answer: its text, its status (``ok``, ``refused`` or ``error``), the refusal's form code and the
    resolved paths it touched (``{"path", "mode"}``; mode ``read``, ``write``, ``list`` or ``search``).
    """

    text: str
    status: str = "ok"
    form: str | None = None
    paths: tuple = ()


def refused(tool: str, form: str, paths: tuple = (), **fields: object) -> Reply:
    return Reply(refusal(tool, form, **fields)[:600], "refused", form, tuple(paths))


def general(form: str, **fields: object) -> Reply:
    fields = {**_GENERAL_FIELDS.get(form, {}), **fields}
    return Reply(refusal(GENERAL, form, **fields)[:600], "refused", form)


def path_row(path: str, mode: str) -> dict:
    return {"path": path, "mode": mode}


# --- reading episodes ----------------------------------------------------------------------------------------


@dataclass
class ReadLog:
    """What one agent was shown through read_episode and overview: a measure only (no tool reads it back)."""

    parts: dict[str, dict[str, int]] = field(default_factory=dict)

    def record(self, eid: str, part: str, nbytes: int) -> None:
        got = self.parts.setdefault(eid, {})
        got[part] = got.get(part, 0) + max(0, int(nbytes))

    def opened(self) -> list[str]:
        return sorted(self.parts)

    def summary(self, batch: list[str]) -> dict:
        return {
            "episodes": len(batch),
            "opened": self.opened(),
            "unread": [e for e in batch if e not in self.parts],
            "parts_read": {e: sorted(ps) for e, ps in sorted(self.parts.items())},
            "bytes_read": sum(n for ps in self.parts.values() for n in ps.values()),
        }


def part_names(ep: Any) -> list[str]:
    """Every part name of *ep*, in the order the table of contents lists them."""
    return (
        ["request"]
        + [f"observation:{i}" for i in range(max(0, len(ep.request) - 1))]
        + [f"cell:{c.index}" for c in ep.cells]
        + [f"action:{i}" for i in range(len(ep.actions))]
        + (["diff"] if ep.worktree_diff else [])
    )


def valid_names(ep: Any) -> str:
    """*ep*'s part names in short: ranges for the numbered kinds."""
    n_obs, n_act = max(0, len(ep.request) - 1), len(ep.actions)
    cells = [c.index for c in ep.cells]
    bits = ["request"]
    if n_obs:
        bits.append(f"observation:0..{n_obs - 1}")
    if cells:
        bits.append(
            (
                f"cell:{cells[0]}..{cells[-1]}"
                if cells == list(range(cells[0], cells[-1] + 1))
                else "cell:" + "/".join(map(str, cells))
            ),
        )
    if n_act:
        bits.append(f"action:0..{n_act - 1}")
    if ep.worktree_diff:
        bits.append("diff")
    return ", ".join(bits)


def _part_bytes(ep: Any, name: str, shared: dict[str, str] | None) -> bytes | None:
    if name.startswith("shared:") and not shared:
        return None
    try:
        return _bm.part_text(ep, name, shared).encode()
    except (KeyError, ValueError, IndexError, StopIteration, TypeError):
        return None


def _sources_of_part(ep: Any, name: str) -> list[str]:
    if name == "request":
        return ["request:0"]
    if name.startswith("observation:"):
        return [f"request:{int(name[12:]) + 1}"]
    if name.startswith("action:"):
        i = name[7:]
        have = set(_fx.sources_of(ep))
        return [s for s in (i, f"args:{i}") if s in have]
    if name.startswith("cell:"):
        return [name] if name in set(_fx.sources_of(ep)) else []
    return []


def _raw_len(ep: Any, src: str, blob: Callable[[str], bytes]) -> str:
    try:
        return str(len(_fx.source_bytes(ep, src, blob)))
    except (_fx.FixtureError, OSError, KeyError, ValueError, IndexError):
        return "unknown"


def contents(
    ep: Any,
    eid: str,
    blob: Callable[[str], bytes],
    shared: dict[str, str] | None = None,
) -> str:
    """read_episode's table of contents: each part, its size as shown, and its fixture sources with raw lengths."""
    names = part_names(ep)
    copies = _bm.observation_copies(ep)
    lines = [
        f"{eid}: {len(names)} part(s). Read them with parts=[...] ({READ_PARTS_PER_CALL} per call, one 8000-byte "
        "page); fixture slices count the raw bytes of the source named here.",
    ]
    for name in names:
        data = _part_bytes(ep, name, shared)
        size = "unknown" if data is None else str(len(data))
        srcs = _sources_of_part(ep, name)
        src_text = (
            ", ".join(f"{s} ({_raw_len(ep, s, blob)} bytes)" for s in srcs) or "-"
        )
        line = f"- {name}: {size} bytes; fixture source: {src_text}"
        if name.startswith("observation:") and name[12:] in copies:
            line += f"; same bytes as observation:{copies[name[12:]]}"
        lines.append(line)
    return "\n".join(lines)


def _wanted_parts(args: dict) -> tuple[list | None, str | None]:
    """The parts a read_episode call names (None: the table of contents), or (None, form) for a malformed list.
    ``parts`` given as text (a JSON list, ``parts=[...]`` or names separated by commas) is read as that list.
    """
    raw = args.get("parts", None)
    if raw is None:
        one = args.get("part")
        if one is None:
            return None, None
        raw = one if isinstance(one, list) else str(one)
        if isinstance(raw, str) and raw.strip() in ("", "parts", "contents", "toc"):
            return None, None
        if isinstance(raw, str) and not (
            raw.strip().startswith(("[", "parts=", "parts:")) or "," in raw
        ):
            return [{"part": raw.strip(), "offset": args.get("offset", 0) or 0}], None
    if isinstance(raw, str):
        text = raw.strip()
        for prefix in ("parts=", "parts:"):
            if text.startswith(prefix):
                text = text[len(prefix) :].strip()
        try:
            val = json.loads(text)
        except ValueError:
            val = None
        if not isinstance(val, list):
            val = [x.strip().strip("'\"") for x in text.strip("[]").split(",")]
        raw = val
    if not isinstance(raw, list):
        return None, "bad_parts"
    raw = [v for v in raw if not (isinstance(v, str) and not v.strip())]
    if not raw or raw == ["parts"]:
        return None, None
    return raw, None


def read_episode(
    ep: Any,
    eid: str,
    args: dict,
    *,
    blob: Callable[[str], bytes],
    log: ReadLog,
    shown: dict[str, tuple[str, str]],
    shared: dict[str, str] | None = None,
) -> Reply:
    """The read_episode tool on a resolved episode *ep*. *shown* is this agent's own record of parts it was shown in
    full (sha256 -> (episode, part)), so a byte-identical part is answered with a pointer; *log* is the read log.
    """
    where = (path_row(f"/episodes/{eid}", "read"),)
    wanted, problem = _wanted_parts(args)
    if problem is not None:
        return refused("read_episode", problem, where)
    if wanted is None:
        return Reply(contents(ep, eid, blob, shared), paths=where)
    if wanted == [{"part": "overview", "offset": args.get("offset", 0) or 0}]:
        return overview_episode(ep, eid, args.get("offset", 0), log=log, shared=shared)
    rest: list = []
    if len(wanted) > READ_PARTS_PER_CALL:
        wanted, rest = wanted[:READ_PARTS_PER_CALL], wanted[READ_PARTS_PER_CALL:]
    items: list[tuple[str, int, bytes]] = []
    unknown: list[str] = []
    for item in wanted:
        name, off = (
            (item.get("part"), item.get("offset", 0))
            if isinstance(item, dict)
            else (item, 0)
        )
        name = str(name).strip()
        off = 0 if off is None else off
        if isinstance(off, bool) or not isinstance(off, int) or off < 0:
            return refused("read_episode", "bad_offset", where, part=name[:80])
        data = _part_bytes(ep, name, shared)
        if data is None:
            unknown.append(name[:80])
            continue
        items.append((name, off, data))
    if unknown and not items:
        return refused(
            "read_episode",
            "unknown_part",
            where,
            names=", ".join(repr(u) for u in unknown),
            episode=eid,
            valid=valid_names(ep),
        )
    left, out = _views.VIEW_BYTES, []
    if unknown:
        out.append(
            f"unknown part(s) in {eid}: {', '.join(repr(u) for u in unknown)}; valid: {valid_names(ep)}",
        )
    paths = [path_row(f"/episodes/{eid}/{name}", "read") for name, _, _ in items]
    for name, off, data in items:
        key = hashlib.sha256(data).hexdigest()
        src = shown.get(key)
        if off == 0 and data and src is not None and src != (eid, name):
            out.append(
                f"== {name} ==\nidentical to {src[0]} {src[1]} (sha256 {key[:12]}), shown to you in full earlier "
                f"in this pass; read {src[0]} parts=[{json.dumps(src[1])}] to see it again",
            )
            continue
        if left <= 0:
            out.append(
                f"== {name} ==\nnot shown (page full): read again with parts={json.dumps([name])}",
            )
            continue
        text, a, b = _views.view_range(data, off, left)
        log.record(eid, name, b - a)
        if a == 0 and b >= len(data) and data:
            shown.setdefault(key, (eid, name))
        left -= b - a
        out.append(f"== {name} ==\n{text}")
    text = "\n\n".join(out)
    if rest:
        text += (
            f"\n\n[{len(rest)} more part(s) not read in this call ({READ_PARTS_PER_CALL} per call); "
            f"next: parts={json.dumps(rest)}]"
        )
    return Reply(text, paths=tuple(paths) or where)


def overview_episode(
    ep: Any,
    eid: str,
    offset: object = 0,
    *,
    log: ReadLog,
    shared: dict[str, str] | None = None,
) -> Reply:
    text, end_parts = _bm.solution_first(ep, shared)
    data = text.encode()
    off = (
        offset
        if isinstance(offset, int) and not isinstance(offset, bool) and offset >= 0
        else 0
    )
    page, a, b = _views.view_range(data, off)
    if a == 0 and b >= len(data):
        for part in end_parts:
            got = _part_bytes(ep, part, shared)
            log.record(eid, part, len(got or b""))
    else:
        log.record(eid, "overview", b - a)
    return Reply(page, paths=(path_row(f"/episodes/{eid}", "read"),))


def overview_batch(rows: list[dict], offset: object = 0) -> Reply:
    """One line per batch episode from the batch map's rows."""
    lines = []
    for r in rows:
        staging = r.get("staging") if isinstance(r.get("staging"), dict) else {}
        lines.append(
            f"{r.get('episode_id')}: {r.get('observations', 0)} observation(s), {r.get('cells', 0)} cell(s), "
            f"{r.get('actions', 0)} action(s), {len(r.get('errors') or [])} error(s), "
            f"{len(staging.get('candidates') or [])} staged candidate(s), "
            f"{len(r.get('pool_entries') or [])} open pool entr{'y' if len(r.get('pool_entries') or []) == 1 else 'ies'}",
        )
    text = "\n".join(lines) or "(no batch episodes)"
    off = (
        offset
        if isinstance(offset, int) and not isinstance(offset, bool) and offset >= 0
        else 0
    )
    return Reply(
        _views.view(text.encode(), off),
        paths=(path_row("/inputs/batch_map.json", "read"),),
    )


# --- fixtures ------------------------------------------------------------------------------------------------


def fixture(
    args: dict,
    episode: Callable[[str], Any],
    box: Path,
    blob: Callable[[str], bytes],
    batch: list[str] | tuple[str, ...] = (),
) -> Reply:
    """The fixture tool: recorded bytes into ``tests/data/``. *episode* resolves an id to a recorded episode (None:
    unknown). A copy is listed in ``.pass/fixtures.json``; an appended line names its own source.
    """
    eid = str(args.get("episode", ""))
    dest = str(args.get("dest", ""))
    where = (path_row(f"/memory/{dest}", "write"),)
    ep = episode(eid)
    if ep is None:
        return refused(
            "fixture",
            "unknown_episode",
            where,
            episode=eid[:80],
            batch=", ".join(batch) or "-",
        )
    source = args.get("source", args.get("action"))
    if source is None:
        return refused("fixture", "no_source", where)
    append = args.get("append") is True
    replace = args.get("replace") is True
    box = Path(box)
    cur = box
    for part in Path(dest).parts[:-1]:
        cur = cur / part
        if cur.is_symlink():
            return refused(
                "fixture",
                "link",
                where,
                what="a directory on the way to dest",
            )
    target = box / dest
    if target.is_symlink():
        return refused("fixture", "link", where, what="dest")
    existing = None
    if append and not replace and target.exists():
        data = _fx._read_plain(target)
        existing = _fx.assembled_entry(data) if data else None
        if data and existing is None:
            return refused("fixture", "not_input_lines", where, dest=dest[:200])
    notes: list[str] = []
    try:
        data, entry = _fx.make(
            ep,
            source,
            dest,
            blob=blob,
            cut=args.get("slice"),
            append=append,
            existing=existing,
            notes=notes,
        )
    except _fx.FixtureError as exc:
        return refused("fixture", "bad_fixture", where, reason=str(exc)[:300])
    target.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_NOFOLLOW | os.O_CLOEXEC
    flags |= os.O_APPEND if (append and not replace) else os.O_TRUNC
    try:
        with os.fdopen(os.open(target, flags, 0o644), "wb") as f:
            f.write(data)
        _fx.write_listing(box, dest, None if append else entry)
    except (OSError, _fx.FixtureError) as exc:
        return refused(
            "fixture",
            "bad_fixture",
            where,
            reason=f"cannot write {dest[:200]}: {exc.__class__.__name__}",
        )
    verb = (
        "started"
        if (append and replace)
        else ("appended a line to" if append else "wrote")
    )
    return Reply(
        f"ok: {verb} {dest} ({len(data)} bytes, sha256 {hashlib.sha256(data).hexdigest()[:12]}; source "
        f"{entry.get('source', (entry.get('lines') or [{}])[-1].get('source'))} of {eid})"
        + "".join(f"; {n}" for n in notes),
        paths=(path_row(f"/episodes/{eid}", "read"), *where),
    )


# --- the pool ------------------------------------------------------------------------------------------------

_DECISIONS = ("merged", "superseded", "retired", "reopened")


def _entries(pool_dir: Path) -> list[dict]:
    out = []
    if not pool_dir.is_dir() or pool_dir.is_symlink():
        return out
    for d in sorted(pool_dir.iterdir()):
        if d.is_symlink() or not d.is_dir():
            continue
        raw = _fx._read_plain(d / "entry.json", 4 * 1024 * 1024)
        try:
            meta = json.loads(raw.decode()) if raw else None
        except (UnicodeDecodeError, ValueError):
            meta = None
        if isinstance(meta, dict) and meta.get("id") == d.name:
            out.append(meta)
    return out


_RELATED_UNREADABLE = (
    f"/inputs/pool/{_pool.RELATED_FILE} could not be read, so the links are unknown"
)


def _related(pool_dir: Path) -> dict | None:
    """/inputs/pool/related_closed.json: ``{"entries": {id: [link]}, "episodes": {episode: [link]}}``; {} when the
    pool has no such file, None when it cannot be read."""
    path = Path(pool_dir) / _pool.RELATED_FILE
    if not path.exists() and not path.is_symlink():
        return {}
    raw = _fx._read_plain(path)
    try:
        val = json.loads(raw.decode()) if raw is not None else None
    except (UnicodeDecodeError, ValueError, RecursionError):
        val = None
    return val if isinstance(val, dict) else None


def _closing(e: dict) -> str:
    """How a closed entry closed, from its entry.json: status, reason and the pass."""
    d = e.get("status_detail") or {}
    status = e.get("status")
    if status == "merged":
        why = f"into {d.get('item')}"
    elif status == "superseded":
        why = f"by {d['by']}" if d.get("by") else str(d.get("reason") or "")
    else:
        why = str(d.get("reason") or "")
    return f"{status}{': ' + why if why else ''}; closed in {d.get('pass') or e.get('last_changed_in')}"


def _line(e: dict) -> str:
    eps = (e.get("provenance") or {}).get("episodes") or []
    closed = "" if e.get("status", "open") == "open" else f" ({_closing(e)})"
    return f"- `{e.get('id')}` {e.get('kind')}{closed}: {e.get('title')} ({len(eps)} episode(s))"


def _link_line(link: dict) -> str:
    """One related closed entry: what it is, how it closed, its evidence and what it shares."""
    shared = [
        f"{n} code unit(s) with {label}"
        for label, n in sorted((link.get("shared_code") or {}).items())
    ]
    if link.get("shared_episodes"):
        shared.append("episode(s) " + ", ".join(link["shared_episodes"]))
    return (
        f"- `{link.get('id')}` {link.get('kind')} ({link.get('status')}: {link.get('reason') or 'no reason given'}; "
        f"closed in {link.get('closed_in')}): {link.get('title')}; evidence: "
        f"{', '.join(link.get('evidence_episodes') or []) or 'none'}; shares {'; '.join(shared)}"
    )


def _read_manifest(box: Path) -> tuple[dict | None, str | None]:
    if (box / ".pass").is_symlink():
        return None, "/memory/.pass is a link"
    path = box / MANIFEST
    if not path.exists() and not path.is_symlink():
        return {}, None
    raw = _fx._read_plain(path, 1024 * 1024)
    if raw is None:
        return (
            None,
            "/memory/.pass/manifest.json is not a regular file of at most 1 MiB",
        )
    try:
        val = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, ValueError, RecursionError):
        return None, "/memory/.pass/manifest.json is not valid JSON"
    if not isinstance(val, dict):
        return None, "/memory/.pass/manifest.json is not a JSON object"
    return val, None


def write_manifest(box: Path, manifest: object) -> None:
    """Write *manifest* (an object, or JSON text) to ``/memory/.pass/manifest.json``, never through a link."""
    pass_dir = Path(box) / ".pass"
    if pass_dir.is_symlink():
        raise OSError("/memory/.pass is a link")
    pass_dir.mkdir(exist_ok=True)
    text = (
        manifest
        if isinstance(manifest, str)
        else json.dumps(manifest, indent=1, sort_keys=True) + "\n"
    )
    fd = os.open(
        Path(box) / MANIFEST,
        os.O_WRONLY | os.O_CREAT | os.O_TRUNC | os.O_NOFOLLOW | os.O_CLOEXEC,
        0o644,
    )
    with os.fdopen(fd, "w") as fh:
        fh.write(text)


def defer_rows(manifest: object) -> list[dict]:
    """The well-formed ``defer`` rows of *manifest*'s pool block: ``{"id": <entry id or "new">, "episodes", "reason"}``."""
    block = manifest.get("pool") if isinstance(manifest, dict) else None
    rows = block.get("defer") if isinstance(block, dict) else None
    out = []
    for r in rows if isinstance(rows, list) else []:
        if (
            isinstance(r, dict)
            and isinstance(r.get("id"), str)
            and isinstance(r.get("reason"), str)
            and r["reason"].strip()
            and isinstance(r.get("episodes", []), list)
            and all(isinstance(e, str) for e in r.get("episodes", []))
        ):
            out.append(
                {
                    "id": r["id"],
                    "episodes": list(r.get("episodes", [])),
                    "reason": r["reason"].strip(),
                },
            )
    return out


def without_defer(manifest: object) -> object:
    """*manifest* with its pool block's ``defer`` rows set aside (they are applied by the pass, not the block)."""
    if not isinstance(manifest, dict) or not isinstance(manifest.get("pool"), dict):
        return manifest
    block = {k: v for k, v in manifest["pool"].items() if k != "defer"}
    return (
        {**manifest, "pool": block}
        if block
        else {k: v for k, v in manifest.items() if k != "pool"}
    )


def pool(
    args: dict,
    pool_dir: Path,
    box: Path,
    write: Callable[[Path, object], None] | None = None,
) -> Reply:
    """The pool tool over the entry files in *pool_dir* (``/inputs/pool``); decide edits the manifest in *box*,
    through *write* when given (in-code calls: a descriptor-relative replace), else :func:`write_manifest`."""
    op = str(args.get("op", ""))
    entries = _entries(Path(pool_dir))
    listed = (path_row("/inputs/pool", "list"),)
    off = args.get("offset", 0)
    off = off if isinstance(off, int) and not isinstance(off, bool) and off >= 0 else 0
    if op in ("list", "filter"):
        kind = args.get("kind")
        related = str(args.get("related") or "") if op == "filter" else ""
        status = args.get("status") or ("any" if related else "open")
        if status not in _POOL_STATUS:
            return refused("pool", "bad_status", listed)
        rows = [
            e
            for e in entries
            if (not kind or e.get("kind") == kind)
            and (
                status == "any"
                or (status == "open") == (e.get("status", "open") == "open")
            )
        ]
        if op == "filter":
            text = str(args.get("text") or "").lower()
            ep = args.get("episode")
            if ep:
                rows = [
                    e
                    for e in rows
                    if ep in ((e.get("provenance") or {}).get("episodes") or [])
                ]
            if related:
                links = _related(Path(pool_dir))
                if links is None:
                    return Reply(_RELATED_UNREADABLE, paths=listed)
                got = (
                    (links.get("entries") or {}).get(related)
                    or (links.get("episodes") or {}).get(related)
                    or []
                )
                linked = [str(x.get("id")) for x in got if isinstance(x, dict)]
                rows = sorted(
                    (e for e in rows if str(e.get("id")) in linked),
                    key=lambda e: linked.index(str(e.get("id"))),
                )
            if text:
                keep = []
                for e in rows:
                    blob = str(e.get("title", "")).lower()
                    for rel in e.get("files") or []:
                        raw = _fx._read_plain(
                            Path(pool_dir) / str(e.get("id")) / str(rel),
                            4 * 1024 * 1024,
                        )
                        blob += "\n" + (raw or b"").decode("utf-8", "replace").lower()
                    if text in blob:
                        keep.append(e)
                rows = keep
        page = rows[off : off + 100]
        word = "" if status == "any" else f"{status} "
        body = "\n".join(_line(e) for e in page) or f"(no matching {word}entries)"
        if off + 100 < len(rows):
            body += f"\n[… entries {off}–{off + 100} of {len(rows)}; next: offset={off + 100}]"
        return Reply(
            f"{len(rows)} {word}entr{'y' if len(rows) == 1 else 'ies'}\n" + body,
            paths=listed,
        )
    eid = str(args.get("id", ""))
    known = {str(e.get("id")): e for e in entries}
    if op == "show":
        e = known.get(eid)
        if e is None:
            return refused("pool", "unknown_entry", listed, id=eid[:40])
        files = [f"/inputs/pool/{eid}/{rel}" for rel in e.get("files") or []]
        found = _related(Path(pool_dir))
        links = [
            x
            for x in ((found or {}).get("entries") or {}).get(eid) or []
            if isinstance(x, dict)
        ]
        size = _pool.RELATED_PAGE
        related = f"\nRelated closed entries: {len(links)}" + (
            "".join("\n" + _link_line(x) for x in links[off : off + size])
        )
        if found is None:
            related = f"\nRelated closed entries: {_RELATED_UNREADABLE}"
        elif off + size < len(links):
            related += f"\n[… related {off}–{off + size} of {len(links)}; next: offset={off + size}]"
        text = (
            json.dumps(e, indent=1, sort_keys=True)
            + "\nFiles:\n"
            + ("\n".join(files) or "(none)")
            + related
        )
        return Reply(
            _views.view(text.encode()),
            paths=(path_row(f"/inputs/pool/{eid}/entry.json", "read"),),
        )
    if op != "decide":
        return refused("pool", "bad_op", listed)
    man_path = (path_row("/memory/.pass/manifest.json", "write"),)
    decision = str(args.get("decision", ""))
    reason = str(args.get("reason") or "").strip()
    target = str(args.get("target") or "").strip()
    episodes = args.get("episodes") or []
    if decision not in (*_DECISIONS, "defer", "open"):
        return refused(
            "pool",
            "bad_decision",
            man_path,
            reason="decision must be merged, superseded, retired, defer, reopened or open",
        )
    if not (decision == "defer" and eid == "new"):
        e = known.get(eid)
        if e is None:
            return refused("pool", "unknown_entry", man_path, id=eid[:40])
        is_open = e.get("status", "open") == "open"
        if decision == "reopened" and is_open:
            return refused("pool", "open_entry", man_path, id=eid[:40])
        if decision not in ("reopened", "open") and not is_open:
            return refused(
                "pool",
                "closed_entry",
                man_path,
                id=eid[:40],
                status=str(e.get("status"))[:20],
            )
    if decision in ("merged", "superseded") and not target:
        return refused(
            "pool",
            "bad_decision",
            man_path,
            reason=f"{decision} needs target",
        )
    if decision in ("retired", "defer", "reopened") and not reason:
        return refused(
            "pool",
            "bad_decision",
            man_path,
            reason=f"{decision} needs a reason",
        )
    if not isinstance(episodes, list) or not all(isinstance(x, str) for x in episodes):
        return refused(
            "pool",
            "bad_decision",
            man_path,
            reason="episodes must be a list of episode ids",
        )
    if decision == "defer" and eid == "new" and not episodes:
        return refused(
            "pool",
            "bad_decision",
            man_path,
            reason="id='new' needs the episodes it defers",
        )
    manifest, problem = _read_manifest(Path(box))
    if manifest is None:
        return refused("pool", "manifest_unreadable", man_path, reason=problem)
    block = manifest.get("pool") if isinstance(manifest.get("pool"), dict) else {}
    for (
        key
    ) in _DECISIONS:  # one decision per entry: an earlier one on this id is replaced
        if isinstance(block.get(key), dict):
            block[key].pop(eid, None)
            if not block[key]:
                block.pop(key)
    if isinstance(block.get("defer"), list) and eid != "new":
        block["defer"] = [
            r
            for r in block["defer"]
            if not (isinstance(r, dict) and r.get("id") == eid)
        ]
        if not block["defer"]:
            block.pop("defer")
    note = ""
    if decision in _DECISIONS:
        block.setdefault(decision, {})[eid] = (
            target if decision in ("merged", "superseded") else reason
        )
        items = {
            i.get("item") for i in manifest.get("items") or [] if isinstance(i, dict)
        }
        if decision == "merged" and target not in items:
            note = f"\nnote: the manifest lists no item {target!r} yet; finish needs it listed"
    elif decision == "defer":
        block.setdefault("defer", []).append(
            {"id": eid, "episodes": list(episodes), "reason": reason},
        )
    if block:
        manifest["pool"] = block
    else:
        manifest.pop("pool", None)
    try:
        (write or write_manifest)(Path(box), manifest)
    except OSError as exc:
        return refused(
            "pool",
            "manifest_unreadable",
            man_path,
            reason=f"cannot write the manifest ({exc.__class__.__name__})",
        )
    return Reply(
        "ok: the manifest's pool block is now\n"
        + json.dumps(manifest.get("pool", {}), indent=1, sort_keys=True)
        + note,
        paths=man_path,
    )
