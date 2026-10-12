"""The ``similar`` tool: where else the same code occurs among the recorded episodes.

Each recorded episode's Python cells are cut into normalised code units (:func:`.mining.units`: the whole cell and
every statement subtree of at least :data:`.mining.MIN_NODES` AST nodes, names renamed in order of use and
literals lifted, the kept names decided by the unit's own text). The units are written to a per-episode index file
once, when the episode is recorded (:func:`record_episode`), under a time limit, and never during a writer pass.
A file is keyed by the content hash of the cells it was built from: building it again from the same cells writes
nothing, and the same cells always give the same bytes.

``similar(episode, cell)`` or ``similar(code)`` normalises the cell (or the code) the same way and returns, for each
unit, the other indexed episodes holding the same normalised unit: counts, the places (episode, cell, lines) and
short excerpts as written there. Nothing is ranked or filtered beyond dropping a unit whose every match lies inside
a larger unit's match already shown; episodes that are not indexed are named in the reply.
"""

from __future__ import annotations

import ast
import hashlib
import json
import os
import time
from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

from . import mining
from .tool_spec import ToolSpec

FORMAT = 1  # the index file's layout; a file of another format is rebuilt
INDEX_TIME_LIMIT_S = 10.0  # one episode's indexing at recording time (operational: the request must end)
EXCERPT_LINES = 3
EXCERPT_WIDTH = 120
INDEX_FOLDER = "memory-similar"

SPEC = ToolSpec(
    name="similar",
    description=(
        "Find the other recorded episodes that contain the same code as one cell of an episode, or as code you "
        "give. Code is compared after normalising it: variable names are renamed in order of use and literal "
        "values are lifted out, while builtins, names the code imports, attribute names and keyword names are "
        "kept, so the same code written with other variable names or constants matches. The reply lists, for the "
        "whole cell and for each statement of at least "
        f"{mining.MIN_NODES} AST nodes, the episodes and cells where it occurs, with counts and a short excerpt "
        "as written there. Episodes are indexed when they are recorded; the reply names any recorded episode "
        "that is not indexed. Give `episode` and `cell`, or `code`."
    ),
    parameters={
        "type": "object",
        "properties": {
            "episode": {"type": "string", "description": "A recorded episode id."},
            "cell": {
                "type": "integer",
                "description": "The index of a cell of that episode.",
            },
            "code": {
                "type": "string",
                "description": "Python source to look for, instead of an episode's cell.",
            },
        },
        "additionalProperties": False,
    },
    roles=frozenset({"write", "helper"}),
    refusals={
        "args": "give `episode` and `cell` together, or `code` alone.",
        "unknown_episode": "no recorded episode {episode}.",
        "unknown_cell": "episode {episode} has no cell {cell}; its cells are {cells}.",
        "no_parse": "the code does not parse as Python ({error}).",
    },
    summary=(
        "other recorded episodes containing the same normalised code as a cell (`episode`, `cell`) or as "
        "given `code`, with counts and excerpts."
    ),
)


def _refuse(form: str, **fields: object) -> str:
    """The same text as :func:`.tool_spec.refusal` for this tool, without loading the whole registry."""
    return "refused: " + SPEC.refusals[form].format(**fields)


# --- the index -------------------------------------------------------------------------------------------------


def index_dir(state_dir: str | os.PathLike) -> Path:
    """Where the index lives: a folder beside the run's other state."""
    return Path(state_dir) / INDEX_FOLDER


def _safe_name(eid: str) -> bool:
    return (
        bool(eid) and "/" not in eid and "\\" not in eid and not eid.startswith(".")
    )


def _entry_path(root: Path, eid: str) -> Path:
    if not _safe_name(eid):
        raise ValueError(f"not an episode id: {eid!r}")
    return Path(root) / "episodes" / f"{eid}.json"


def _as_cell(c: Any) -> mining.Cell:
    """A recorded cell (an episode's Cell, a dict as stored in cells.jsonl, or a :class:`.mining.Cell`)."""
    row = asdict(c) if is_dataclass(c) else dict(c)
    return mining.Cell(
        int(row.get("index", 0)),
        str(row.get("code") or ""),
        str(row.get("output") or ""),
        row.get("error"),
        str(row.get("language") or "python"),
    )


def _python(cells: Iterable[Any]) -> list[mining.Cell]:
    out = [_as_cell(c) for c in cells]
    return sorted(
        (c for c in out if (c.language or "python") == "python"),
        key=lambda c: c.index,
    )


def content_key(cells: Iterable[Any]) -> str:
    """The hash of what the index of these cells depends on: each Python cell's index, code and whether it ran."""
    rows = [
        [c.index, c.code, mining.successful(c)] for c in _python(cells)
    ]
    return hashlib.sha256(
        json.dumps(rows, separators=(",", ":")).encode(),
    ).hexdigest()


def excerpt(code: str, lines: tuple[int, int]) -> str:
    """The first non-empty lines of *lines* (1-based, inclusive) of *code*, each cut to a width."""
    rows = code.splitlines()[lines[0] - 1 : lines[1]]
    got = [r.rstrip()[:EXCERPT_WIDTH] for r in rows if r.strip()][:EXCERPT_LINES]
    return "\n".join(got)


def _unit_rows(
    cell: mining.Cell,
    stop: Callable[[], bool] | None = None,
) -> list[dict]:
    ok = mining.successful(cell)
    return [
        {
            "key": u.key,
            "kind": u.kind,
            "size": u.size,
            "cell": u.cell,
            "lines": list(u.lines),
            "ran": ok,
            "excerpt": excerpt(cell.code, u.lines),
        }
        for u in mining.units(cell, stop=stop)
    ]


def build_entry(
    episode: str,
    cells: Iterable[Any],
    *,
    time_limit_s: float = INDEX_TIME_LIMIT_S,
    clock: Callable[[], float] = time.monotonic,
) -> dict:
    """The index entry of one episode: every Python cell's units, in cell order. When the time limit passes, the
    cells indexed so far are kept and ``complete`` is false."""
    py = _python(cells)
    end = clock() + time_limit_s

    def late() -> bool:
        return clock() > end

    units: list[dict] = []
    done = 0
    complete = True
    for c in py:
        if late():
            complete = False
            break
        rows = _unit_rows(c, stop=late)
        if late():  # the cell may have been cut short: its units are not all there
            complete = False
            break
        units += rows
        done += 1
    units.sort(key=lambda u: (u["cell"], u["lines"], u["kind"], u["key"]))
    return {
        "format": FORMAT,
        "episode": episode,
        "content": content_key(py),
        "complete": complete,
        "cells": [c.index for c in py],
        "cells_indexed": done,
        "units": units,
    }


def read_entry(root: Path | None, eid: str) -> dict | None:
    if root is None or not _safe_name(eid):
        return None
    try:
        data = json.loads(_entry_path(root, eid).read_text())
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) and data.get("format") == FORMAT else None


def index_episode(
    root: Path,
    episode: str,
    cells: Iterable[Any],
    *,
    time_limit_s: float = INDEX_TIME_LIMIT_S,
    clock: Callable[[], float] = time.monotonic,
) -> dict:
    """Index one episode under *root* unless a complete entry for the same content is there already. The result:
    ``{"episode", "status": "kept" | "written" | "partial", "units"}``."""
    cells = list(cells)
    path = _entry_path(root, episode)
    have = read_entry(root, episode)
    if have is not None and have.get("complete") and have.get("content") == content_key(cells):
        return {"episode": episode, "status": "kept", "units": len(have.get("units") or [])}
    entry = build_entry(episode, cells, time_limit_s=time_limit_s, clock=clock)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".part")
    tmp.write_text(json.dumps(entry, sort_keys=True, separators=(",", ":")) + "\n")
    os.replace(tmp, path)
    return {
        "episode": episode,
        "status": "written" if entry["complete"] else "partial",
        "units": len(entry["units"]),
    }


def record_episode(
    state_dir: str | os.PathLike,
    ep: Any,
    redactor: Any = None,
    *,
    time_limit_s: float = INDEX_TIME_LIMIT_S,
) -> dict:
    """The recording hook: index a just-recorded episode. *redactor* (the episode writer's) is applied to the cells
    first, so the index holds what the stored episode holds."""
    rows = [asdict(c) if is_dataclass(c) else dict(c) for c in ep.cells]
    if redactor is not None:
        rows = [redactor.obj(r) for r in rows]
    return index_episode(
        index_dir(state_dir),
        str(ep.episode_id),
        rows,
        time_limit_s=time_limit_s,
    )


def index_missing(
    root: Path,
    load: Callable[[str], Any],
    episodes: Iterable[str],
    *,
    time_limit_s: float = INDEX_TIME_LIMIT_S,
    clock: Callable[[], float] = time.monotonic,
) -> dict:
    """Index the *episodes* that have no complete entry (episodes recorded before the index existed), oldest first,
    until *time_limit_s* passes in all; counts of what was done and what is left."""
    end = clock() + time_limit_s
    done, left, failed = 0, 0, 0
    for eid in episodes:
        eid = str(eid)
        have = read_entry(root, eid)
        if have is not None and have.get("complete"):
            continue
        remaining = end - clock()
        if remaining <= 0:
            left += 1
            continue
        try:
            index_episode(root, eid, load(eid).cells, time_limit_s=remaining, clock=clock)
            done += 1
        except Exception:  # noqa: BLE001 - an unreadable episode stays unindexed and is named by the tool
            failed += 1
    return {"indexed": done, "left": left, "failed": failed}


def load_index(root: Path | None) -> dict[str, dict]:
    """Every readable entry under *root*, by episode id."""
    out: dict[str, dict] = {}
    if root is None:
        return out
    folder = Path(root) / "episodes"
    if not folder.is_dir():
        return out
    for f in sorted(folder.glob("*.json")):
        try:
            data = json.loads(f.read_text())
        except (OSError, ValueError):
            continue
        if isinstance(data, dict) and data.get("format") == FORMAT and data.get("episode"):
            out[str(data["episode"])] = data
    return out


# --- the tool --------------------------------------------------------------------------------------------------


@dataclass
class Context:
    """What the handler needs. ``index_dir``: the index folder (:func:`index_dir` of the run's state folder; None:
    nothing is indexed);
    ``load``: episode id -> the recorded episode (with ``cells``), raising for an unknown id; ``recorded``: every
    recorded episode id (to name the ones not indexed). Without ``load``, a cell is looked up in the index only."""

    index_dir: Path | None
    load: Callable[[str], Any] | None = None
    recorded: Callable[[], Iterable[str]] | None = None


def _spans(cells: list[int]) -> str:
    """``0-4, 7, 9-10`` for sorted cell indices."""
    out: list[str] = []
    for i in cells:
        if out and i == int(out[-1].split("-")[-1]) + 1:
            out[-1] = f"{out[-1].split('-')[0]}-{i}"
        else:
            out.append(str(i))
    return ", ".join(out) or "none"


def _inside(small: tuple[int, int], big: tuple[int, int]) -> bool:
    return big[0] <= small[0] and small[1] <= big[1]


class _Refusal(Exception):
    pass


def _query(args: dict, ctx: Context) -> tuple[str | None, list[dict], str]:
    """(the query's episode or None, its unit rows, a heading), or raises _Refusal."""
    has_ep, has_cell = "episode" in args, "cell" in args
    code = args.get("code")
    if code is not None and not (has_ep or has_cell):
        if not isinstance(code, str):
            raise _Refusal(_refuse("args"))
        try:
            ast.parse(code)
        except (SyntaxError, ValueError, RecursionError, MemoryError) as exc:
            error = getattr(exc, "msg", None) or type(exc).__name__
            raise _Refusal(_refuse("no_parse", error=error)) from None
        cell = mining.Cell(0, code)
        return None, _unit_rows(cell), "the given code"
    if not (has_ep and has_cell) or code is not None:
        raise _Refusal(_refuse("args"))
    eid, raw = args.get("episode"), args.get("cell")
    if not isinstance(eid, str) or not eid:
        raise _Refusal(_refuse("args"))
    if isinstance(raw, bool) or not (
        isinstance(raw, int) or (isinstance(raw, str) and raw.strip().isdigit())
    ):
        raise _Refusal(_refuse("args"))
    idx = int(raw)
    cells: list[mining.Cell] | None = None
    if ctx.load is not None:
        try:
            cells = [_as_cell(c) for c in ctx.load(eid).cells]
        except Exception:  # noqa: BLE001 - any lookup failure is an unknown episode to the caller
            cells = None
    if cells is None:
        entry = read_entry(ctx.index_dir, eid)
        if entry is None:
            raise _Refusal(_refuse("unknown_episode", episode=eid))
        if idx not in (entry.get("cells") or []):
            raise _Refusal(
                _refuse("unknown_cell", episode=eid, cell=idx, cells=_spans(sorted(entry.get("cells") or []))),
            )
        rows = [u for u in entry.get("units") or [] if u.get("cell") == idx]
        return eid, rows, f"episode {eid} cell {idx}"
    found = next((c for c in cells if c.index == idx), None)
    if found is None:
        raise _Refusal(
            _refuse("unknown_cell", episode=eid, cell=idx, cells=_spans(sorted(c.index for c in cells))),
        )
    if (found.language or "python") != "python":
        return eid, [], f"episode {eid} cell {idx} (a {found.language} cell: only Python is indexed)"
    return eid, _unit_rows(found), f"episode {eid} cell {idx}"


def _describe(u: dict) -> str:
    what = "whole cell" if u.get("kind") == "cell" else "statement"
    a, b = u["lines"]
    return f"lines {a}-{b} ({what}, {u['size']} nodes)"


def _place(p: dict) -> str:
    a, b = p["lines"]
    return f"{p['episode']} cell {p['cell']} lines {a}-{b}" + ("" if p.get("ran", True) else " (cell errored)")


def handle(args: dict, ctx: Context) -> str:
    """The tool's reply to *args* (``episode`` and ``cell``, or ``code``)."""
    if not isinstance(args, dict):
        return _refuse("args")
    try:
        own, rows, heading = _query(args, ctx)
    except _Refusal as r:
        return str(r)
    index = load_index(ctx.index_dir)
    others = {e: d for e, d in index.items() if e != own}
    postings: dict[str, list[dict]] = {}
    for eid in sorted(others):
        for u in others[eid].get("units") or []:
            postings.setdefault(u.get("key"), []).append({**u, "episode": eid})
    recorded: list[str] = []
    if ctx.recorded is not None:
        try:
            recorded = [str(e) for e in ctx.recorded()]
        except Exception:  # noqa: BLE001 - without the list, the reply cannot name unindexed episodes
            recorded = []
    unindexed = sorted(set(recorded) - set(index) - ({own} if own else set()))
    partial = sorted(e for e, d in others.items() if not d.get("complete"))

    out = [
        f"Code units of {heading}: {len(rows)} (the whole cell and each statement of at least "
        f"{mining.MIN_NODES} AST nodes). Searched {len(others)} other indexed episodes.",
    ]
    if unindexed:
        out.append(f"Not indexed ({len(unindexed)} recorded episodes, not searched): {', '.join(unindexed)}.")
    if partial:
        out.append(
            f"Indexed in part (indexing stopped at its time limit; later cells not searched): {', '.join(partial)}.",
        )
    if not rows:
        out.append("No code unit to look for.")
        return "\n".join(out)
    shown: list[tuple[tuple[int, int], list[dict]]] = []
    implied = 0
    unmatched: list[str] = []
    for u in sorted(rows, key=lambda r: (-r["size"], r["lines"], r["key"])):
        places = postings.get(u["key"], [])
        span = (u["lines"][0], u["lines"][1])
        if not places:
            unmatched.append(f"lines {span[0]}-{span[1]}")
            continue
        if any(
            _inside(span, big_span)
            and all(
                any(
                    p["episode"] == q["episode"]
                    and p["cell"] == q["cell"]
                    and _inside(tuple(p["lines"]), tuple(q["lines"]))
                    for q in big_places
                )
                for p in places
            )
            for big_span, big_places in shown
        ):
            implied += 1
            continue
        shown.append((span, places))
        eps = sorted({p["episode"] for p in places})
        out.append("")
        out.append(
            f"{_describe(u)}: the same normalised code in {len(eps)} other episode(s), {len(places)} place(s):",
        )
        out.append("  " + "; ".join(_place(p) for p in places))
        seen_text: set[str] = set()
        for p in places:
            text = p.get("excerpt") or ""
            if not text or text in seen_text:
                continue
            seen_text.add(text)
            out.append(f"  as written in {p['episode']} cell {p['cell']}:")
            out += ["    | " + ln for ln in text.splitlines()]
    if implied:
        out.append("")
        out.append(f"{implied} smaller unit(s) match only inside the places above (not listed again).")
    if unmatched:
        out.append("")
        out.append(f"No other episode holds: {', '.join(unmatched)}.")
    return "\n".join(out)
