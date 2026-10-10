"""The writer's batch map (spec v2.1 §7.1): deterministic facts and pointers per episode, no summaries.

Signals are structural only (spec §5): a cell error, an action's error status, and a retry of an identical call
after an error. The episode's declared ``regime`` passes through. Environment text is never pattern-matched.
"""

from __future__ import annotations

import ast
import json
import re
import textwrap
from collections import Counter
from typing import Callable, Iterable

from .episodes import Episode

_ADDED_PY = re.compile(r"^\+\+\+ b/(?P<path>.+\.py)$")
_HUNK = re.compile(r"^@@ -\d+(?:,\d+)? \+(?P<start>\d+)(?:,\d+)? @@")


def _call_key(a) -> str:
    return json.dumps(
        [a.kind, a.channel, a.method, a.args, a.kwargs],
        sort_keys=True,
        default=str,
    )


def structural_signals(ep: Episode) -> list[dict]:
    out: list[dict] = []
    for c in ep.cells:
        if c.error:
            out.append({"kind": "cell_error", "cell": c.index, "action": None})
    failed: set[str] = set()
    for i, a in enumerate(ep.actions):
        key = _call_key(a)
        if key in failed:
            out.append({"kind": "retry_after_error", "cell": a.cell, "action": i})
        if a.status == "error" or a.error:
            out.append({"kind": "action_error", "cell": a.cell, "action": i})
            failed.add(key)
    return out


def _defs(tree: ast.AST) -> list[ast.FunctionDef | ast.AsyncFunctionDef]:
    return [
        n
        for n in getattr(tree, "body", [])
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
    ]


def _called_names(code: str) -> Counter:
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return Counter()
    return Counter(
        n.func.id
        for n in ast.walk(tree)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
    )


def _diff_python_runs(diff: str) -> dict[str, list[tuple[int, list[str]]]]:
    """Added lines of each ``.py`` file in a unified diff, as runs of consecutive new-file lines.

    Returns ``{path: [(first new-file line, [line, ...]), ...]}``. Header lines count only outside a hunk, so an
    added line that itself starts with ``++`` is still an added line.
    """
    runs: dict[str, list[tuple[int, list[str]]]] = {}
    path, in_hunk, new_line = None, False, 0
    for line in diff.splitlines():
        if line.startswith("diff --git "):
            path, in_hunk = None, False
            continue
        if not in_hunk and line.startswith("+++ "):
            m = _ADDED_PY.match(line)
            path = m.group("path") if m else None
            continue
        h = _HUNK.match(line)
        if h:
            in_hunk, new_line = True, int(h.group("start"))
            continue
        if not in_hunk or path is None:
            continue
        if line.startswith("+"):
            file_runs = runs.setdefault(path, [])
            if file_runs and file_runs[-1][0] + len(file_runs[-1][1]) == new_line:
                file_runs[-1][1].append(line[1:])
            else:
                file_runs.append((new_line, [line[1:]]))
            new_line += 1
        elif line.startswith(" ") or line == "":
            new_line += 1
    return runs


def _diff_functions(ep: Episode) -> tuple[list[dict], list[str], bool]:
    """The actor's functions in the work-tree diff, the ``.py`` files with an unparsable run, and whether any
    ``.py`` file has added lines. Each run is dedented and parsed on its own, so one bad file or fragment
    hides nothing else."""
    out: list[dict] = []
    unparsed: list[str] = []
    runs = _diff_python_runs(ep.worktree_diff or "")
    for path, file_runs in runs.items():
        for start, lines in file_runs:
            try:
                tree = ast.parse(textwrap.dedent("\n".join(lines)))
            except SyntaxError:
                if path not in unparsed:
                    unparsed.append(path)
                continue
            for d in _defs(tree):
                out.append(
                    {
                        "name": d.name,
                        "signature": "(" + ast.unparse(d.args) + ")",
                        "source": "diff",
                        "cell": None,
                        "path": path,
                        "lineno": start + d.lineno - 1,
                        "cell_error": False,
                        "called_later": 0,
                    },
                )
    return out, unparsed, bool(runs)


def actor_functions(
    ep: Episode,
    diff: tuple[list[dict], list[str], bool] | None = None,
) -> list[dict]:
    """The actor's functions: top-level defs of its Python cells, then those its work-tree diff adds. *diff* is
    ``_diff_functions(ep)`` when the caller already has it."""
    out: list[dict] = []
    py = [c for c in ep.cells if (c.language or "python") == "python"]
    for c in py:
        try:
            tree = ast.parse(c.code)
        except SyntaxError:
            continue
        for d in _defs(tree):
            later = sum(_called_names(x.code)[d.name] for x in py if x.index > c.index)
            out.append(
                {
                    "name": d.name,
                    "signature": "(" + ast.unparse(d.args) + ")",
                    "source": "cell",
                    "cell": c.index,
                    "lineno": d.lineno,
                    "cell_error": bool(c.error),
                    "called_later": later,
                },
            )
    out.extend((diff or _diff_functions(ep))[0])
    return out


def _observations(ep: Episode) -> list[str]:
    return list(ep.request[1:])


def canonical_part(ep: Episode, part: str) -> str:
    """The required part that *part* stands for: a byte-identical observation is required once, as its first
    copy, and reading any copy credits that one. Every other part is its own."""
    if not part.startswith("observation:"):
        return part
    obs = _observations(ep)
    text = obs[int(part[12:])]
    return f"observation:{obs.index(text)}"


def observation_copies(ep: Episode) -> dict[str, int]:
    """Each observation that repeats an earlier one byte for byte, mapped to the first copy (keys are the
    indices as strings, as in the JSON map)."""
    obs = _observations(ep)
    return {str(i): obs.index(t) for i, t in enumerate(obs) if obs.index(t) != i}


def _cells_before_replies(ep: Episode) -> dict[int, int]:
    """Episode action index of each dialogue action -> the index of the last cell completed before its reply,
    from the transcript's order. The dialogue recorder's own reply rule pairs actions with replies: its
    answered replies when the counts match (the online recorder), else all of them; with neither, no pairing.
    """
    from .integration.adapters.dialogue import _is_observation, _is_reply, _messages

    msgs = _messages(ep.transcript)
    calls: set[str] = set()
    done: set[str] = set()
    replies: list[tuple[int | None, bool]] = []  # (last cell index before it, answered)
    for i, m in enumerate(msgs):
        for tc in m.get("tool_calls") or []:
            fn = (
                tc.get("function")
                if isinstance(tc, dict) and isinstance(tc.get("function"), dict)
                else {}
            )
            if fn.get("name") == "execute_code" and isinstance(tc.get("id"), str):
                calls.add(tc["id"])
        if m.get("role") == "tool" and m.get("tool_call_id") in calls:
            done.add(m["tool_call_id"])
        if _is_reply(m):
            answered = False
            for later in msgs[i + 1 :]:
                if later.get("role") == "assistant":
                    break
                if _is_observation(later):
                    answered = True
                    break
            replies.append((len(done) - 1 if done else None, answered))
    dialogue = [i for i, a in enumerate(ep.actions) if a.kind == "dialogue"]
    answered = [c for c, ok in replies if ok]
    pairs = answered if len(answered) == len(dialogue) else [c for c, _ in replies]
    if len(pairs) != len(dialogue):
        return {}
    return {i: c for i, c in zip(dialogue, pairs) if c is not None}


def required_parts(
    ep: Episode,
    signals: list[dict],
    functions: list[dict],
    diff: tuple[list[dict], list[str], bool] | None = None,
) -> list[str]:
    """Every part a writer must read to cover *ep*: the request, every distinct observation (where any verdict
    lives; the harness never decides which by its text), every cell that defines a function or carries a
    signal, every action a signal names, every dialogue action and every action that is not a read (with its
    source cell), and the work-tree diff whenever a ``.py`` file in it has added lines.
    """
    parts = ["request"]
    seen: set[str] = set()
    for i, text in enumerate(_observations(ep)):
        if text not in seen:
            seen.add(text)
            parts.append(f"observation:{i}")
    cells = {f["cell"] for f in functions if f["source"] == "cell"}
    cells |= {
        s["cell"]
        for s in signals
        if s["kind"] == "cell_error" and s["cell"] is not None
    }
    actions = {s["action"] for s in signals if s["action"] is not None}
    # T8 (what the actor did): every dialogue action and every action that is not a read, with its source cell
    # (a dialogue action's: the last cell completed before its reply). Structural only: kind and effect.
    acted = [
        i
        for i, a in enumerate(ep.actions)
        if a.kind == "dialogue" or a.effect != "read"
    ]
    actions |= set(acted)
    known = {c.index for c in ep.cells}
    before = _cells_before_replies(ep)
    for i in acted:
        a = ep.actions[i]
        source = before.get(i) if a.kind == "dialogue" else a.cell
        if source in known:
            cells.add(source)
    for c in sorted(cells):
        parts.append(f"cell:{c}")
    for a in sorted(actions):
        parts.append(f"action:{a}")
    if (diff or _diff_functions(ep))[2]:
        parts.append("diff")
    return parts


def part_text(ep: Episode, part: str, shared: dict[str, str] | None = None) -> str:
    """*part* of *ep* as the writer reads it; *shared* (r5, :mod:`.shared_text`): the request's shared blocks shown
    as their markers (the raw episode files keep every byte)."""
    if part.startswith("shared:"):
        # r5 (RUNTIME B2): a shared request block is its own part, so the writer sees it once; the identical-part
        # credit then covers it in every other episode whose request holds it
        if not shared or part[7:] not in shared:
            raise KeyError(part)
        obj = shared[part[7:]]
    elif part == "request":
        obj = ep.request[0] if ep.request else ""
        if shared:
            from .shared_text import mark

            obj = mark(obj, shared)
    elif part.startswith("observation:"):
        obj = _observations(ep)[int(part[12:])]
    elif part == "diff":
        obj = ep.worktree_diff or ""
    elif part.startswith("cell:"):
        c = next(c for c in ep.cells if c.index == int(part[5:]))
        obj = {
            "index": c.index,
            "language": c.language,
            "code": c.code,
            "output": c.output,
            "error": c.error,
        }
    elif part.startswith("action:"):
        a = ep.actions[int(part[7:])]
        obj = {
            "index": int(part[7:]),
            "cell": a.cell,
            "kind": a.kind,
            "channel": a.channel,
            "method": a.method,
            "args": a.args,
            "kwargs": a.kwargs,
            "status": a.status,
            "error": a.error,
            "response": a.response,
        }
    else:
        raise KeyError(part)
    return json.dumps(obj, sort_keys=True, default=str)


_GIST = 100
_TRACEBACK = "Traceback (most recent call last)"


def _gist_cell(code: str) -> str:
    try:
        tree = ast.parse(code)
    except (SyntaxError, ValueError, RecursionError):
        tree = None
    defs = [d.name for d in _defs(tree)] if tree is not None else []
    calls = (
        [n for n, _ in _called_names(code).most_common(4)] if tree is not None else []
    )
    first = next((ln.strip() for ln in code.splitlines() if ln.strip()), "")
    bits = ([f"defines {', '.join(defs)}"] if defs else []) + (
        [f"calls {', '.join(calls)}"] if calls else []
    )
    return ("; ".join(bits) + " | " if bits else "") + first[:_GIST]


def solution_first(
    ep: Episode,
    shared: dict[str, str] | None = None,
) -> tuple[str, list[str]]:
    """r5 (r4 §5): *ep* end first, then an index of every step, newest first; and the parts shown in full.

    The end: the last cell that ran without an error, the last action and the last observation, each verbatim as
    its own part (``read_episode`` returns the same bytes). The index: one line per cell, action and observation,
    newest first, with its part name (the handle to read it), its size, whether it errored, and a deterministic
    gist (a cell's definitions, calls and first line; an action's channel, method and status; an observation's
    first line). Nothing is summarised by a model and nothing is left out: every step is listed and readable.
    """
    ok = [c for c in ep.cells if c.error is None and _TRACEBACK not in (c.output or "")]
    obs = _observations(ep)
    end_parts = []
    if ok:
        end_parts.append(f"cell:{ok[-1].index}")
    if ep.actions:
        end_parts.append(f"action:{len(ep.actions) - 1}")
    if obs:
        end_parts.append(f"observation:{len(obs) - 1}")
    out = ["== end =="]
    for part in end_parts:
        out.append(f"-- {part} --\n{part_text(ep, part, shared)}")
    if not end_parts:
        out.append("(no cell, action or observation was recorded)")
    out.append("\n== steps, newest first ==")
    for c in reversed(ep.cells):
        size = len(part_text(ep, f"cell:{c.index}").encode())
        err = (
            "error" if (c.error is not None or _TRACEBACK in (c.output or "")) else "ok"
        )
        out.append(f"cell:{c.index}  {size}B  {err}  {_gist_cell(c.code or '')}")
    for i in range(len(ep.actions) - 1, -1, -1):
        a = ep.actions[i]
        size = len(part_text(ep, f"action:{i}").encode())
        err = "error" if (a.status == "error" or a.error) else a.status
        out.append(
            f"action:{i}  {size}B  {err}  {a.kind} {a.channel}.{a.method} (cell {a.cell})",
        )
    for i in range(len(obs) - 1, -1, -1):
        text = str(obs[i])
        first = next((ln.strip() for ln in text.splitlines() if ln.strip()), "")
        out.append(
            f"observation:{i}  {len(part_text(ep, f'observation:{i}').encode())}B  {first[:_GIST]}",
        )
    return "\n".join(out) + "\n", end_parts


def _errors(ep: Episode) -> list[dict]:
    out = []
    for i, a in enumerate(ep.actions):
        if a.status == "error" or a.error:
            nxt = ep.actions[i + 1] if i + 1 < len(ep.actions) else None
            out.append(
                {
                    "action": i,
                    "error": a.error,
                    "next_call": (
                        f"{nxt.channel}.{nxt.method}" if nxt is not None else None
                    ),
                },
            )
    return out


def _shared_parts(ep: Episode, shared: dict[str, str]) -> list[str]:
    """``shared:<id>`` for each shared block *ep*'s request holds (RUNTIME B2: shown once, never only a marker)."""
    from .shared_text import blocks

    held = set(blocks(ep.request[0] if ep.request else ""))
    return [f"shared:{b}" for b, text in sorted(shared.items()) if text in held]


def build_batch_map(
    load: Callable[[str], Episode],
    eids: Iterable[str],
    shared: dict[str, str] | None = None,
) -> dict:
    """The batch map; *shared* (r5): requests carry shared blocks as markers, and ``shared_blocks`` lists each
    block's file once."""
    if shared:
        from .shared_text import mark
    rows = []
    for eid in eids:
        ep = load(eid)
        sig = structural_signals(ep)
        diff = _diff_functions(ep)
        fns = actor_functions(ep, diff)
        rows.append(
            {
                "episode_id": ep.episode_id,
                "memory_main": ep.memory_main,
                "regime": ep.regime,
                "request": (
                    mark(ep.request[0] if ep.request else "", shared)
                    if shared
                    else (ep.request[0] if ep.request else "")
                ),
                "observations": len(_observations(ep)),
                "observation_copies": observation_copies(ep),
                "cells": len(ep.cells),
                "actions": len(ep.actions),
                "items_used": ep.memory_use,
                "calls": dict(Counter(f"{a.channel}.{a.method}" for a in ep.actions)),
                "errors": _errors(ep),
                "signals": sig,
                "functions": fns,
                "diff_unparsed": diff[1],
                "required_parts": required_parts(ep, sig, fns, diff)
                + (_shared_parts(ep, shared) if shared else []),
            },
        )
    out = {"version": 1, "episodes": rows}
    if shared:
        out["shared_blocks"] = [
            {"id": b, "chars": len(t), "file": f"/inputs/shared/{b}.txt"}
            for b, t in sorted(shared.items())
        ]
    return out
