"""Memory during an episode (``UNIFY_MEMORY=in_episode``): checkpoints, segment forks, and a re-pin at the next cell.

A long episode can learn and use memory before it ends.

**Checkpoints are natural units**, never a count or a timer (:data:`SOURCES`). A pause whose cells since the last one
computed nothing (:func:`..capture_shape.computes`: only calls with literal arguments) starts no fork; its cells stay
for the next segment or the episode-end fork:

=================  ===========================================================================================
``actor_call``     the actor calls ``tools.checkpoint(note="")``
``compaction``     the harness compacts the actor's context (a pause it already has)
``new_message``    a new user or environment message arrives after at least one cell
``signal``         an in-episode signal arrives (a checker line the actor was shown)
=================  ===========================================================================================

At a checkpoint (:meth:`InEpisode.segment`) the harness records the episode as it stands
(:func:`..segments.write_snapshot`, ``<episode>.s<n>``), starts a **segment fork** for the cells since the last
checkpoint (the episode-end fork's code, summary and capture contract in a short form: :func:`.fork.segment_instruction`,
with its own staging dir, ``memory-staging/<episode>.s<n>/``), and calls the writer service (``ensure_service``). A
settled segment fork becomes a ``new_experience`` entry with its segment (:mod:`..attention`). The fork continues the
actor's conversation, which can be forked only between turns, never with a tool call unanswered and never after a
compaction: so a checkpoint the actor or a signal marks runs at the next turn boundary (:meth:`InEpisode.at_boundary`),
a new message's runs before the message joins the conversation, and a compaction's runs before the compaction.

**Re-pin.** When the writer service lands a commit while the episode is open, the next cell boundary (never mid-cell,
:meth:`InEpisode.before_cell`) exports it into a fresh pin directory (``<checkout>/.pins/<k>-<sha12>``), swaps the
actor's read-only library path atomically (the symlink ``<checkout>/current``, renamed over), and the worker drops the
library's modules and invalidates its import caches before the cell runs. That cell's result ends with one line
naming exactly what changed between the two pins (:func:`notice`); the first such line also says that names bound
before keep their old code. At most one re-pin per boundary, always to the latest commit. Each pin is kept in the
episode's ``memory_pins`` (``memory_main`` stays the first), and every use the cells make is counted under the pin
served when it was called.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Literal, get_args

logger = logging.getLogger(__name__)

Source = Literal["actor_call", "compaction", "new_message", "signal"]
SOURCES: tuple[str, ...] = get_args(Source)
PINS_DIR = ".pins"
CURRENT = "current"
#: A pause's note, as the actor gives it, at most.
NOTE_CHARS = 500
HELD_LINE = "Names you bound from the library before this keep their old code; get or import them again to use the new."


@dataclass
class Pin:
    """One pin of the episode: the cell from which it was served, its commit and its directory."""

    from_cell: int
    commit: str
    path: Path
    landed: list = field(default_factory=list)
    retired: list = field(default_factory=list)

    def row(self) -> dict:
        out: dict = {"from_cell": self.from_cell, "commit": self.commit}
        if self.from_cell:
            out["landed"] = list(self.landed)
            out["retired"] = list(self.retired)
        return out


def _swap_link(link: Path, target_name: str) -> None:
    """Point *link* at *target_name* (relative) atomically: a new link renamed over the old one."""
    tmp = link.with_name(f".{link.name}.{os.getpid()}.{time.monotonic_ns()}")
    os.symlink(target_name, tmp)
    os.replace(tmp, link)


def _items_of(tree: Path) -> dict[str, str]:
    """Every item a pin offers, by its notice name, to a digest of what it is now: functions (``fn:<id>``), notes and
    the captures it serves (``capture:<id>``); retired items left out."""
    import hashlib
    import json

    from ..layout import discover, function_bodies

    out: dict[str, str] = {}
    try:
        rows = json.loads((tree / ".memory" / "items.json").read_text(encoding="utf-8")).get("items") or {}
    except (OSError, ValueError):
        rows = {}
    hidden = {i for i, r in rows.items() if isinstance(r, dict) and r.get("status") in ("retired", "superseded")}
    try:
        lib = discover(tree)
        bodies = function_bodies(tree)
    except Exception:  # noqa: BLE001 - an unreadable pin offers nothing
        return out
    for f in lib.functions:
        if f.item_id not in hidden:
            out[f"fn:{f.item_id}"] = hashlib.sha256(bodies.get(f.item_id, "").encode()).hexdigest()
    for n in lib.notes:
        if n.item_id in hidden:
            continue
        try:
            data = (tree / n.item_id).read_bytes()
        except OSError:
            data = b""
        out[n.item_id] = hashlib.sha256(data).hexdigest()
    caps = tree / ".memory" / "captures"
    try:
        names = sorted(p.name for p in caps.iterdir() if p.is_dir())
    except OSError:
        names = []
    for cid in names:
        try:
            data = (caps / cid / "procedure.py").read_bytes()
        except OSError:
            data = b""
        out[f"capture:{cid}"] = hashlib.sha256(data).hexdigest()
    return out


def _why_gone(tree: Path, name: str) -> str:
    """Why an item left between pins: ``retired`` or ``superseded`` (its record or the withdrawn captures say so),
    else ``removed``."""
    import json

    try:
        rows = json.loads((tree / ".memory" / "items.json").read_text(encoding="utf-8")).get("items") or {}
    except (OSError, ValueError):
        rows = {}
    item = name[3:] if name.startswith("fn:") else name
    status = (rows.get(item) or {}).get("status") if isinstance(rows.get(item), dict) else None
    if status in ("retired", "superseded"):
        return status
    if name.startswith("capture:"):
        try:
            gone = json.loads((tree / ".memory" / "captures" / "withdrawn.json").read_text(encoding="utf-8"))
            row = (gone.get("captures") or {}).get(name[len("capture:") :])
            if isinstance(row, dict) and row.get("status") in ("retired", "superseded"):
                return row["status"]
        except (OSError, ValueError, AttributeError):
            pass
        return "retired"
    return "removed"


def diff(old: Path, new: Path) -> tuple[list[str], list[str], list[tuple[str, str]]]:
    """``(added, changed, gone)`` between two pins' trees; *gone* holds each name with why it left."""
    a, b = _items_of(old), _items_of(new)
    added = sorted(k for k in b if k not in a)
    changed = sorted(k for k in b if k in a and a[k] != b[k])
    gone = [(k, _why_gone(new, k)) for k in sorted(a) if k not in b]
    return added, changed, gone


def notice(commit: str, added: list[str], changed: list[str], gone: list[tuple[str, str]], first: bool) -> str:
    """The one line a re-pinned cell's result ends with: exactly what changed between the two pins."""
    parts = [f"+{k}" for k in added] + [f"~{k}" for k in changed] + [f"−{k} ({why})" for k, why in gone]
    line = f"memory changed (commit {commit[:12]}): {' '.join(parts)}; call memory.items() to see them."
    return f"{line} {HELD_LINE}" if first else line


class InEpisode:
    """One request's memory during its episode (module docstring). *run* is the request's
    :class:`.request.RequestRun`; *export* writes the actor's copy of a commit into a directory
    (``RequestRun._export_copy``)."""

    def __init__(
        self,
        run: Any,
        export: Callable[[str, Path], None],
        *,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self.run = run
        self.export = export
        self.clock = clock
        self.checkout = Path(run.paths.checkout)
        self.pins: list[Pin] = []
        self.cells = 0  # the episode's code cells whose result has come back (the next cell's index)
        self.next_cell = 0  # the first cell of the next segment
        self.segments = 0
        self.pending: list[tuple[str, str]] = []
        self.forks: list[Any] = []
        self.warned = False
        self.events: list[dict] = []
        self.running = 0  # cells running now, in any of the request's workers
        self._lock = threading.Lock()

    # -- the copy and its pins -------------------------------------------------------------------------------------

    @property
    def copy_root(self) -> Path:
        """The actor's library copy as cells and host readers reach it: ``<checkout>/current``."""
        return self.checkout / CURRENT

    @property
    def current(self) -> str | None:
        return self.pins[-1].commit if self.pins else None

    def open(self, sha: str) -> None:
        """At the request's start: the first pin, exported into its own directory, and the link to it."""
        from .checkout import remove_checkout

        remove_checkout(self.checkout)
        self.checkout.mkdir(parents=True)
        (self.checkout / PINS_DIR).mkdir()
        dest = self.checkout / PINS_DIR / f"0-{sha[:12]}"
        self.export(sha, dest)
        _swap_link(self.copy_root, f"{PINS_DIR}/{dest.name}")
        self.pins.append(Pin(0, sha, dest))

    def cell_done(self) -> None:
        """One of the episode's code cells came back (``hooks.tool_result``, ``execute_code``): the episode's cells are
        numbered in that order, as its record numbers them (``cell:<n>``)."""
        self.cells += 1

    def commit_of(self, name: str) -> str | None:
        """The commit of the pin whose directory is *name* (``<k>-<12 hex>``), or None."""
        for p in self.pins:
            if p.path.name == name:
                return p.commit
        return None

    @property
    def generation(self) -> int:
        """How many times the copy was re-pinned: a worker that last saw another number drops the library's modules
        before its next cell."""
        return len(self.pins) - 1

    def before_cell(self, head: str | None) -> dict:
        """At a cell boundary, in any of the request's workers: when *head* (the commit requests now pin) differs from
        the pin and no other cell is running (so no running cell reads the copy as it changes), re-pin to it (module
        docstring). Returns ``{"generation": n, "notice": <line or None>}``: the notice only for the cell whose boundary
        re-pinned. The new pin is served from the episode's next cell (``from_cell``). Each call is paired with one
        :meth:`after_cell`."""
        with self._lock:
            notice = None
            if self.running == 0 and head and head != self.current:
                notice = self._repin(head)
            self.running += 1
            return {"generation": self.generation, "notice": notice}

    def after_cell(self) -> None:
        """The cell that :meth:`before_cell` began has ended (returned, raised, timed out or was stopped)."""
        with self._lock:
            self.running = max(0, self.running - 1)

    def _repin(self, head: str) -> str | None:
        index = self.cells
        old = self.pins[-1].path
        dest = self.checkout / PINS_DIR / f"{len(self.pins)}-{head[:12]}"
        try:
            self.export(head, dest)
        except Exception as exc:  # noqa: BLE001 - the cell runs on the pin it has; the next boundary tries again
            logger.warning("memory: re-pin to %s not exported (%s)", head[:12], type(exc).__name__)
            self._log({"type": "repin", "cell": index, "commit": head, "error": type(exc).__name__})
            return None
        _swap_link(self.copy_root, f"{PINS_DIR}/{dest.name}")
        repinned = getattr(self.run, "repinned", None)
        if callable(repinned):
            repinned(dest)
        added, changed, gone = diff(old, dest)
        self.pins.append(Pin(index, head, dest, sorted(added + changed), [k for k, _ in gone]))
        line = None
        if added or changed or gone:
            line = notice(head, added, changed, gone, first=not self.warned)
            self.warned = True
        self._log({"type": "repin", "cell": index, "commit": head, "landed": len(added) + len(changed),
                   "retired": len(gone)})  # fmt: skip
        return line

    def pin_rows(self) -> list[dict]:
        """The episode's ``memory_pins``."""
        return [p.row() for p in self.pins]

    # -- checkpoints -------------------------------------------------------------------------------------------------

    def mark(self, source: str, note: str = "") -> None:
        """Mark a pause (:data:`SOURCES`); its segment fork starts at the next turn boundary."""
        if source not in SOURCES:
            raise ValueError(f"a checkpoint's source is one of {', '.join(SOURCES)}, not {source!r}")
        self.pending.append((source, str(note or "")[:NOTE_CHARS]))

    def at_boundary(self, handle: Any = None, *, compacting: bool = False, new_message: bool = False) -> dict | None:
        """Between turns: run the marked pauses, a new message's (before the message joins the conversation) and a
        compaction's (before it), as one segment fork for the cells since the last checkpoint. None when nothing is
        due or no cell ran since the last checkpoint."""
        sources = [s for s, _ in self.pending]
        notes = [n for _, n in self.pending if n]
        if compacting:
            sources.append("compaction")
        if new_message:
            sources.append("new_message")
        if not sources:
            return None
        self.pending.clear()
        return self.segment(sorted(set(sources), key=SOURCES.index), notes, handle)

    def segment(self, sources: list[str], notes: list[str], handle: Any = None) -> dict:
        """One segment fork (module docstring) for the cells from :attr:`next_cell` to the last one the episode
        recorded so far; never raises (the request goes on, and its event says why nothing started)."""
        event: dict = {"type": "checkpoint", "sources": sources}
        try:
            ep, redactor = self.run.partial_episode()
        except Exception as exc:  # noqa: BLE001 - the episode as it stands cannot be read; the next pause tries again
            event.update(cells=None, error=type(exc).__name__)
            logger.warning("memory: the episode so far not read at a checkpoint (%s)", type(exc).__name__)
            self._log(event)
            return event
        first, last = self.next_cell, len(ep.cells) - 1
        event["cells"] = [first, last] if last >= first else None
        if last < first:
            event["skipped"] = "no cell ran since the last pause"
            self._log(event)
            return event
        from ..capture_shape import computes

        if not computes(ep.cells[first : last + 1]):
            # rev 2: cells that computed nothing (calls with literal arguments only, read from their syntax alone)
            # start no fork; they stay for the next segment or the episode-end fork
            event["skipped"] = "no computation"
            self._log(event)
            return event
        self.segments += 1
        name = f"{self.run.episode_id}.s{self.segments}"
        event["segment"] = self.segments
        try:
            event.update(self._start(name, ep, redactor, first, last, notes, handle))
        except Exception as exc:  # noqa: BLE001 - a segment that cannot start leaves the episode as it was
            event["error"] = type(exc).__name__
            logger.warning("memory: segment %s not started (%s)", name, type(exc).__name__)
        self.next_cell = last + 1
        self._log(event)
        return event

    def finish(self, progress: Callable[[str], None] = lambda _m: None) -> None:
        """After the reply: a pause marked after the last turn boundary is logged as left to the episode-end fork, and
        each segment fork is handed over, and waited for, as the episode's own fork is (:func:`.fork.finish`); never
        raises."""
        from . import fork as _fork

        if self.pending:
            # marked after the last turn boundary: the episode-end fork covers these cells
            self._log({"type": "checkpoint", "sources": sorted({src for src, _ in self.pending}, key=SOURCES.index),
                       "cells": None, "skipped": "the episode ended first; its own fork covers these cells"})  # fmt: skip
            self.pending.clear()
        for fork in self.forks:
            try:
                _fork.finish(fork, progress)
            except Exception as exc:  # noqa: BLE001 - the fork settles on its own; the next sweep resolves it
                logger.warning("memory: segment fork not handed over (%s)", type(exc).__name__)

    def _log(self, event: dict) -> None:
        """Keep *event* (a checkpoint or a re-pin) in the episode's list and in
        ``<state>/memory-segments/checkpoints.jsonl`` (one row each, appended)."""
        import json

        from ..segments import SEGMENTS_DIR

        self.events.append(event)
        row = {"v": 1, "episode": self.run.episode_id, "at": _now(), **event}
        try:
            root = Path(self.run.paths.state_dir) / SEGMENTS_DIR
            root.mkdir(parents=True, exist_ok=True)
            fd = os.open(root / "checkpoints.jsonl", os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o600)
            try:
                os.write(fd, (json.dumps(row, sort_keys=True) + "\n").encode("utf-8"))
                os.fsync(fd)
            finally:
                os.close(fd)
        except (OSError, TypeError, ValueError) as exc:
            logger.warning("memory: checkpoint row not kept (%s)", type(exc).__name__)

    def _start(self, name: str, ep: Any, redactor: Any, first: int, last: int, notes: list[str], handle: Any) -> dict:
        from .. import segments
        from . import fork as _fork

        run = self.run
        ep.episode_id = name
        segments.write_snapshot(run.paths.state_dir, name, ep, redactor)
        reservation = _fork.reserve(run.paths, name)
        if reservation is None:
            return {"fork": "not started: staging dir"}
        parts = _fork.part_list(ep)
        if notes:
            parts += "\n\nThe actor's note at this pause: " + " / ".join(notes)
        fork = _fork.start(
            run.paths,
            handle if handle is not None else getattr(run, "handle", None),
            name,
            reservation=reservation,
            parts=parts,
            instruction=_fork.segment_instruction(first, last),
        )
        out = {"fork": "started" if fork is not None else "refused"}
        if fork is not None:
            fork.writer.join(_fork.WRITE_GUARD_S)
            self.forks.append(fork)
        _ensure_service(run.paths.home)
        return out


# --- segments from a recorded episode (the offline replay's recorded pauses) -------------------------------------------


@dataclass(frozen=True)
class RecordedSegment:
    """One segment of a recorded episode, ready for its fork: its id and staging dir (``<episode>.s<n>``), the cells
    it covers, the fork's instruction for them (:func:`.fork.segment_instruction`), the part list of the episode as it
    stood (:func:`.fork.part_list`) and that record's path (:func:`..segments.snapshot_path`)."""

    name: str
    n: int
    staging: Path
    cells: tuple[int, int]
    instruction: str
    parts: str
    snapshot: Path


def episode_until(ep: Any, last_cell: int) -> Any:
    """The recorded episode *ep* as it stood when its cell *last_cell* came back: its cells to that one, its actions
    from those cells, and, when it holds its transcript, the transcript to that cell's result with the messages and
    replies in it (a later observation or reply is left out)."""
    from dataclasses import replace as _replace

    from . import trajectory

    cells = [c for c in ep.cells if c.index <= last_cell]
    actions = [a for a in ep.actions if getattr(a, "cell", None) is None or a.cell <= last_cell]
    out = _replace(ep, cells=cells, actions=actions)
    lines = [ln for ln in (ep.transcript or []) if isinstance(ln, dict)]
    timed = trajectory.timed_cells(trajectory.fold(lines)) if lines else []
    if 0 <= last_cell < len(timed):
        end = timed[last_cell].end
        cut = [ln for ln in lines if trajectory._seconds(ln.get("ts")) <= end]
        requests, replies = trajectory.dialogue_text(trajectory.fold(cut))
        out = _replace(out, transcript=cut, request=requests or list(ep.request[:1]), replies=replies)
    return out


def recorded_segments(paths: Any, ep: Any, points: list[int], redactor: Any = None) -> list[RecordedSegment]:
    """The segments a recorded episode *ep* would have had with a pause after each cell in *points* (cell indices, the
    last cell before each pause: the replay's recorded compaction points), as :meth:`InEpisode.segment` makes them
    live: for each, the episode as it stood (:func:`episode_until`) recorded once as the segment's record
    (:func:`..segments.write_snapshot`; an existing record is kept), and what its fork needs (:class:`RecordedSegment`).
    A pause with no new cell since the last makes no segment, as live. Nothing is started: the caller runs each fork
    (recorded replies, a stub or a live lane) into its staging dir, which it reserves (:func:`.fork.reserve`)."""
    from dataclasses import replace as _replace

    from .. import segments
    from . import fork as _fork

    out: list[RecordedSegment] = []
    first = 0
    n_cells = len(ep.cells)
    for point in sorted({int(p) for p in points}):
        last = min(point, n_cells - 1)
        if last < first:
            continue
        name = segments.segment_name(ep.episode_id, len(out) + 1)
        cut = episode_until(ep, last)
        path = segments.snapshot_path(paths.state_dir, name)
        if not path.is_file():
            segments.write_snapshot(paths.state_dir, name, _replace(cut, episode_id=name), redactor)
        out.append(
            RecordedSegment(
                name=name,
                n=len(out) + 1,
                staging=_fork.staging_dir(paths, name),
                cells=(first, last),
                instruction=_fork.segment_instruction(first, last),
                parts=_fork.part_list(cut),
                snapshot=path,
            ),
        )
        first = last + 1
    return out


def _now() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _ensure_service(home: Path) -> None:
    """The writer service, so the segment is read while the episode goes on (the service's own module; nothing when
    it is not there or off). Never raises."""
    try:
        from ..writer_service import ensure_service
    except ImportError:
        return
    try:
        ensure_service(Path(home))
    except Exception as exc:  # noqa: BLE001 - the service is started again at the request's end
        logger.warning("memory: writer service not ensured at a checkpoint (%s)", type(exc).__name__)


# --- the actor's tool ------------------------------------------------------------------------------------------------


def checkpoint(ctx: Any, args: dict) -> Any:
    """``tools.checkpoint(note="")``: the actor marks a pause; its segment fork starts at the end of this turn."""
    from unify.actor.tool_handlers import _refused, _reply
    from unify.common.tool_registry import refusal

    from .request import current

    run = current()
    ie = getattr(run, "in_episode", None) if run is not None else None
    if ie is None:
        return _refused(refusal("checkpoint", "refused", reason="memory during an episode is off in this run"))
    note = args.get("note", "")
    if note is not None and not isinstance(note, str):
        return _refused(refusal("checkpoint", "refused", reason="note is a few words of text, or left out"))
    ie.mark("actor_call", note or "")
    return _reply("the pause is marked: what your cells did since the last pause is written down at the end of this turn")
