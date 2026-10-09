"""The writer's batch map (spec v2.1 §7.1): deterministic facts and pointers per episode, no summaries.

Signals are structural only (spec §5): a cell error, an action's error status, and a retry of an identical call
after an error. The episode's declared ``regime`` passes through. Environment text is never pattern-matched.
"""

from __future__ import annotations

import ast
import json
import re
from collections import Counter
from typing import Callable, Iterable

from .episodes import Episode

_ADDED_PY = re.compile(r"^\+\+\+ b/(?P<path>.+\.py)$")


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


def _diff_added_python(diff: str) -> str:
    out, in_py = [], False
    for line in diff.splitlines():
        if line.startswith("+++ "):
            in_py = bool(_ADDED_PY.match(line))
            continue
        if in_py and line.startswith("+") and not line.startswith("+++"):
            out.append(line[1:])
    return "\n".join(out)


def actor_functions(ep: Episode) -> list[dict]:
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
    added = _diff_added_python(ep.worktree_diff or "")
    if added:
        try:
            tree = ast.parse(added)
        except SyntaxError:
            tree = None
        for d in _defs(tree) if tree is not None else []:
            out.append(
                {
                    "name": d.name,
                    "signature": "(" + ast.unparse(d.args) + ")",
                    "source": "diff",
                    "cell": None,
                    "lineno": d.lineno,
                    "cell_error": False,
                    "called_later": 0,
                },
            )
    return out


def required_parts(
    ep: Episode,
    signals: list[dict],
    functions: list[dict],
) -> list[str]:
    parts = ["request"]
    for c in sorted({f["cell"] for f in functions if f["source"] == "cell"}):
        parts.append(f"cell:{c}")
    for a in sorted({s["action"] for s in signals if s["action"] is not None}):
        parts.append(f"action:{a}")
    if any(f["source"] == "diff" for f in functions):
        parts.append("diff")
    return parts


def part_text(ep: Episode, part: str) -> str:
    if part == "request":
        obj = ep.request[0] if ep.request else ""
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


def build_batch_map(load: Callable[[str], Episode], eids: Iterable[str]) -> dict:
    rows = []
    for eid in eids:
        ep = load(eid)
        sig = structural_signals(ep)
        fns = actor_functions(ep)
        rows.append(
            {
                "episode_id": ep.episode_id,
                "memory_main": ep.memory_main,
                "regime": ep.regime,
                "request": ep.request[0] if ep.request else "",
                "cells": len(ep.cells),
                "actions": len(ep.actions),
                "items_used": ep.memory_use,
                "calls": dict(Counter(f"{a.channel}.{a.method}" for a in ep.actions)),
                "errors": _errors(ep),
                "signals": sig,
                "functions": fns,
                "required_parts": required_parts(ep, sig, fns),
            },
        )
    return {"version": 1, "episodes": rows}
