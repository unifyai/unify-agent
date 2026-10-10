"""Memory v2.1 r5 arm C (r3 S1, r5 §6): Sol analysts, one per flagged episode, writing staging.

An episode is flagged by code, never by a task id (:func:`flagged`): it has a recorded signal, code the S0 miner
clustered, code no earlier episode had (novel), or a cell that failed followed by one that ran. Each flagged
episode gets one Sol sub-call with a fresh context. Every sibling starts with the same bytes (the brief, then the
library index view), so that prefix is cached once; the episode's own message follows. The analyst reads its one
episode with the writer's readers (``read_episode``, ``read``, ``grep``) and stages files with ``stage``: the same
paths and rules as the fork (:mod:`.integration.fork_worker`), written once at its end, then ``analyst.json``.
Nothing staged is run. Turns are bounded only by the operational step guard.
"""

from __future__ import annotations

import asyncio
import json
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Awaitable, Callable

from . import batch_map as _bm
from .mining import keep_names, normalise, successful, _parse
from .integration.fork_worker import write_files

ANALYST_BRIEF = (
    "You read one recorded episode of an agent's work and write down what would help with similar requests "
    "later. The episode's parts are read with `read_episode` (start with part 'overview': its end first, then "
    "every step with the part name that reads it); `read` and `grep` read the pass's other inputs. Stage files "
    "with `stage(path, text)`: for code the agent wrote that worked and could be reused, a general function with "
    "a docstring in `candidates/<name>.py`, and the inputs and outputs it had in the episode in "
    "`cases/<name>.json`; for each thing that failed and why, and for any other lesson, a short note in "
    "`notes.md`, stated in general terms so it applies beyond this request, naming the part it comes from (for "
    "example `cell:3`). If nothing here is worth keeping, stage `notes.md` saying so. Call `done` when finished."
)
STAGE_TOOL = {
    "type": "function",
    "function": {
        "name": "stage",
        "description": "Stage one file for the writer: candidates/<name>.py, cases/<name>.json or notes.md.",
        "parameters": {
            "type": "object",
            "properties": {"path": {"type": "string"}, "text": {"type": "string"}},
            "required": ["path", "text"],
        },
    },
}
DONE_TOOL = {
    "type": "function",
    "function": {
        "name": "done",
        "description": "End this analysis.",
        "parameters": {"type": "object", "properties": {}},
    },
}
READERS = ("read_episode", "read", "grep")


def _cell_keys(ep) -> set[str]:
    trees = [t for t in (_parse(c.code) for c in ep.cells) if t is not None]
    keep = keep_names(trees)
    out = set()
    for c in ep.cells:
        t = _parse(c.code) if successful(c) else None
        if t is not None:
            out.add(normalise(t, keep)[0])
    return out


def flagged(ep, clustered: set[str], seen_keys: set[str]) -> list[str]:
    """Why *ep* gets an analyst ([] when it does not): ``signal``, ``clustered_code`` (in an S0 cluster),
    ``novel_code`` (a successful cell whose normalised shape is not in *seen_keys*), ``fail_then_succeed``.
    """
    why = []
    if _bm.structural_signals(ep):
        why.append("signal")
    if ep.episode_id in clustered:
        why.append("clustered_code")
    if _cell_keys(ep) - seen_keys:
        why.append("novel_code")
    failed = False
    for c in ep.cells:
        if not successful(c):
            failed = True
        elif failed:
            why.append("fail_then_succeed")
            break
    return why


def _usd(value: Any) -> Decimal | None:
    try:
        return Decimal(str(value))
    except (InvalidOperation, ValueError):
        return None


async def analyse_one(
    eid: str,
    why: list[str],
    *,
    model_turn: Callable[[list, list], Awaitable[tuple[dict, Any]]],
    reader: Callable[[str, dict], Awaitable[str]],
    tools: list[dict],
    prefix: list[dict],
    root: Path,
    step_guard: int,
) -> dict:
    """One analyst: its turns, its staging and ``analyst.json``; never raises."""
    dest = root / eid
    if dest.exists():
        return {"episode": eid, "status": "refused: staging exists"}
    messages = [
        *prefix,
        {
            "role": "user",
            "content": f"Episode `{eid}` (flagged: {', '.join(why)}). Read it end to end, then stage your findings.",
        },
    ]
    found: list[tuple[str, str, str]] = []
    usd, unknown, turns, status = Decimal(0), 0, 0, "ok"
    try:
        while turns < step_guard:
            msg, cost = await model_turn(messages, tools)
            turns += 1
            c = _usd(cost)
            if c is None:
                unknown += 1
            else:
                usd += c
            messages.append(msg)
            calls = msg.get("tool_calls") or []
            if not calls:
                break
            finished = False
            for tc in calls:
                fn = tc.get("function") or {}
                name = fn.get("name", "")
                try:
                    args = json.loads(fn.get("arguments") or "{}")
                except ValueError:
                    args = {}
                if name == "stage":
                    found.append(
                        (
                            str(args.get("path", "")),
                            str(args.get("text", "")),
                            "tool:stage",
                        ),
                    )
                    content = "staged"
                elif name == "done":
                    finished, content = True, "ok"
                elif name in READERS:
                    content = await reader(name, args)
                else:
                    content = f"unknown tool {name!r}"
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc.get("id", ""),
                        "content": content,
                    },
                )
            if finished:
                break
        else:
            status = "operational_step_guard"
    except (
        Exception
    ) as exc:  # noqa: BLE001 - one analyst's failure never stops the pass
        status = f"error: {type(exc).__name__}"
    dest.mkdir(parents=True)
    files, refused, over = write_files(dest, found)
    doc = {
        "episode": eid,
        "status": status,
        "flags": why,
        "turns": turns,
        "usd": str(usd),
        "unknown_cost_calls": unknown,
        "files": files,
        "refused_files": refused,
        "quota_reached": over,
    }
    (dest / "analyst.json").write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n")
    return doc


async def run(
    flags: dict[str, list[str]],
    *,
    model_turn: Callable[[list, list], Awaitable[tuple[dict, Any]]],
    reader: Callable[[str, dict], Awaitable[str]],
    reader_tools: list[dict],
    index_view: str,
    root: Path,
    step_guard: int,
) -> list[dict]:
    """Every flagged episode's analyst, in parallel; siblings share the brief and the index view byte for byte."""
    prefix = [
        {"role": "system", "content": ANALYST_BRIEF},
        {"role": "user", "content": index_view or "Current library index: (empty)"},
    ]
    tools = [t for t in reader_tools if t["function"]["name"] in READERS] + [
        STAGE_TOOL,
        DONE_TOOL,
    ]
    root.mkdir(parents=True, exist_ok=True)
    return list(
        await asyncio.gather(
            *(
                analyse_one(
                    eid,
                    why,
                    model_turn=model_turn,
                    reader=reader,
                    tools=tools,
                    prefix=prefix,
                    root=root,
                    step_guard=step_guard,
                )
                for eid, why in sorted(flags.items())
                if why
            ),
        ),
    )
