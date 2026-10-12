"""Procedure verification: the ``episode`` cover type and its runners.

A procedure is a function that does a whole recorded job. Its ``episode`` cover names the episode and the runner
of the episode's action kind. The runner re-does the job confined and compares the effect with what the
environment recorded:

* **worktree**:
  - the scratch tree is seeded from the episode's ``worktree_before`` snapshot;
  - the procedure runs as ``fn("/work", **params)`` with ``/work`` the working directory;
  - every path the episode changed (or the cover's ``paths``, a subset of them) must then equal
    ``worktree_after``;
  - other paths it changed are noted.
* **tool**:
  - ``fn(env, **params)`` runs in its own box, a fresh interpreter that sees neither the case nor the recording:
    ``env`` is a proxy whose every call goes over a socket to a replay server in a second box, which holds the
    recorded rows (``memlab.replay``), answers only the calls the function makes, and writes the result;
  - no call may miss the recording;
  - the write calls it issues must equal the recorded write calls, in order, with their arguments;
  - when the recording's effects are ``unknown``, the whole served sequence must equal the recorded one;
  - an episode with no write call has no effect to confirm, and is refused (a reader takes action covers).
* **dialogue**:
  - ``fn(observation, **params)``, given the observation the recorded action answered (the previous dialogue
    observation on its channel, else the request);
  - it must return the recorded action's value: its keyword arguments, else its one argument, else its arguments;
  - the episode must carry a positive signal (checker ``pass`` or provenance ``support``) and no contrary one. The
    harness never reads environment text for this. A ``checker`` signal counts only when the bed
    declares its verdict visible to the actor (*checker_visible*, from ``switch.checker_visible``); a grader the
    actor never sees is never read.

A procedure's parameters must be values its episode recorded (:func:`param_problems`). A procedure of any other
kind has no runner; :func:`parse_cover` refuses it, so it goes to drafts.

Model-written code runs only through *runner* (:func:`.sandbox_run.run_confined`):
- the library tree read-only at ``/memory``, the kit read-only at ``/kit``;
- for the work-tree and dialogue runners, the case read-only at ``/case`` (it holds only the runner and the
  inputs; the expected result stays on the host) and writable only ``/out`` (and ``/work`` for a work-tree run);
- for the tool runner, no ``/case`` and no ``/out``: only the item's name and parameters at ``/item`` and the
  replay server's socket at ``/chan``;
- no network and an empty environment.
The host only compares files and JSON.
"""

from __future__ import annotations

import contextlib
import json
import os
import re
import shutil
import socket
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable

from . import testkit
from .episodes import Action, Episode
from .fixtures import canonical
from .gitio import Repo
from .replay import UnknownEffect
from .replay import calls as recorded_calls
from .sandbox_run import SandboxResult, run_confined
from .signals import _CONTRARY, _SUPPORT
from .snapshot import listing, materialise
from .quality import recorded_literals

RUNNERS = ("worktree", "tool", "dialogue")
COVER_TYPES = ("cell", "diff", "episode")
PROCEDURE_S = 60.0
_EPISODE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}\Z")
_KEYS = frozenset({"episode", "type", "cell", "runner", "action", "params", "paths"})

_RUN = r"""
import importlib, json, os, sys
sys.path[:0] = ["/memory", "/kit"]
sys.dont_write_bytecode = True
spec = json.load(open("/case/case.json"))
out = {}
try:
    module, _, name = spec["item"].partition(":")
    fn = getattr(importlib.import_module(module), name)
    params = spec.get("params") or {}
    if spec["runner"] == "worktree":
        os.chdir("/work")
        fn("/work", **params)
    else:
        out["value"] = json.dumps(fn(spec["observation"], **params), sort_keys=True, default=repr)
    out["outcome"] = "returned"
except BaseException as exc:
    out["outcome"] = "raised"
    out["error_class"] = type(exc).__name__
with open("/out/result.json", "w") as f:
    json.dump(out, f, default=repr)
"""


# The tool runner's two boxes. The item's box runs _ITEM_RUN: a fresh interpreter with the item's name and parameters
# only, whose env proxy sends each call over the socket. The server's box runs _SERVE: it holds the recorded rows,
# answers each call through the exact-call replay, and writes the result once the item ends.
SOCKET = "env.sock"
SERVER_GRACE_S = 10.0

_ITEM_RUN = r"""
import importlib, json, socket, sys, time
sys.path[:0] = ["/memory", "/kit"]
sys.dont_write_bytecode = True


def _main():
    with open("/item/item.json") as f:
        target = json.load(f)
    conn = None
    for _ in range(1200):
        conn = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            conn.connect("/chan/env.sock")
            break
        except OSError:
            conn.close()
            conn = None
            time.sleep(0.05)
    if conn is None:
        return
    rx, tx = conn.makefile("r", encoding="utf-8"), conn.makefile("w", encoding="utf-8")
    try:
        from memlab.replay import RecordedError, ReplayMiss
    except Exception:
        class ReplayMiss(LookupError):
            pass

        class RecordedError(RuntimeError):
            pass

    def ask(message):
        tx.write(json.dumps(message, default=str) + "\n")
        tx.flush()
        line = rx.readline()
        if not line:
            raise RuntimeError("the recorded environment ended")
        return json.loads(line)

    class Channel:
        def __init__(self, name):
            self._name = name

        def __getattr__(self, method):
            if method.startswith("_"):
                raise AttributeError(method)
            channel = self._name

            def call(*args, **kwargs):
                got = ask({"op": "call", "channel": channel, "method": method, "args": list(args), "kwargs": kwargs})
                if got.get("error") == "miss":
                    raise ReplayMiss(str(got.get("message") or ""))
                if got.get("error") == "recorded":
                    raise RecordedError(str(got.get("message") or "recorded error"))
                return got.get("value")

            return call

    class Env:
        def __getattr__(self, channel):
            if channel.startswith("_"):
                raise AttributeError(channel)
            return Channel(channel)

    end = {"op": "end"}
    try:
        module, _, name = target["item"].partition(":")
        fn = getattr(importlib.import_module(module), name)
        fn(Env(), **(target.get("params") or {}))
        end["outcome"] = "returned"
    except BaseException as exc:
        end["outcome"] = "raised"
        end["error_class"] = type(exc).__name__
    tx.write(json.dumps(end) + "\n")
    tx.flush()


_main()
"""

_SERVE = r"""
import json, socket, sys
sys.path[:0] = ["/kit"]
sys.dont_write_bytecode = True
from memlab.replay import RecordedEnv, RecordedError, ReplayMiss


def _main():
    with open("/case/case.json") as f:
        case = json.load(f)
    env = RecordedEnv(case["rows"])
    out = {"outcome": "raised"}
    srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    srv.bind("/chan/env.sock")
    srv.listen(1)
    srv.settimeout(float(sys.argv[1]))
    try:
        conn, _ = srv.accept()
    except OSError:
        conn = None
    if conn is not None:
        conn.settimeout(None)
        rx, tx = conn.makefile("r", encoding="utf-8"), conn.makefile("w", encoding="utf-8")
        try:
            for line in rx:
                msg = json.loads(line)
                op = msg.get("op") if isinstance(msg, dict) else None
                if op == "end":
                    if msg.get("outcome") == "returned":
                        out["outcome"] = "returned"
                    elif isinstance(msg.get("error_class"), str):
                        out["error_class"] = msg["error_class"][:64]
                    break
                channel, method = msg.get("channel"), msg.get("method")
                args, kwargs = msg.get("args"), msg.get("kwargs")
                if op != "call" or not (
                    isinstance(channel, str) and isinstance(method, str)
                    and isinstance(args, list) and isinstance(kwargs, dict)
                ):
                    break
                try:
                    reply = {"value": env._serve(channel, method, args, kwargs)}
                except ReplayMiss as exc:
                    reply = {"error": "miss", "message": str(exc)}
                except RecordedError as exc:
                    reply = {"error": "recorded", "message": str(exc)}
                tx.write(json.dumps(reply) + "\n")
                tx.flush()
        except (OSError, ValueError):
            pass
    out["issued"] = env.issued()
    out["misses"] = len(env.misses)
    if case.get("effect") == "write":
        try:
            out["issued_writes"] = env.issued(effect="write")
        except Exception:
            out["issued_writes"] = None
    with open("/out/result.json", "w") as f:
        json.dump(out, f, default=repr)


_main()
"""


@dataclass(frozen=True)
class Cover:
    episode: str
    type: str
    index: int | None = None
    runner: str | None = None
    action: int | None = None
    params: dict = field(default_factory=dict, hash=False)
    paths: tuple[str, ...] = ()


@dataclass
class Outcome:
    ok: bool
    reason: str | None = None
    notes: list[str] = field(default_factory=list)


def _index(v: object) -> bool:
    return isinstance(v, int) and not isinstance(v, bool) and v >= 0


def parse_cover(raw: object) -> tuple[str, int] | Cover:
    """An action cover ``[episode, index]`` (v2's form), or a typed cover (an object)."""
    if (
        isinstance(raw, (list, tuple))
        and len(raw) == 2
        and isinstance(raw[0], str)
        and _index(raw[1])
    ):
        return (raw[0], raw[1])
    if not isinstance(raw, dict):
        raise ValueError(
            "a cover is [episode, action] or an object with episode and type",
        )
    eid, kind = raw.get("episode"), raw.get("type")
    if not isinstance(eid, str) or not _EPISODE_ID.match(eid):
        raise ValueError("a cover's episode must be an episode id")
    if kind not in COVER_TYPES:
        raise ValueError(f"a cover's type is one of {', '.join(COVER_TYPES)}")
    unknown = sorted(set(raw) - _KEYS)
    if unknown:
        raise ValueError(f"unknown cover keys {unknown[:5]}")
    if kind == "cell":
        if not _index(raw.get("cell")):
            raise ValueError("a cell cover names its cell index")
        return Cover(eid, "cell", index=raw["cell"])
    if kind == "diff":
        return Cover(eid, "diff")
    runner = raw.get("runner")
    if runner not in RUNNERS:
        raise ValueError(
            f"an episode cover names its runner, one of {', '.join(RUNNERS)}; a procedure of another kind goes "
            "to drafts",
        )
    params = raw.get("params", {})
    if not isinstance(params, dict) or not all(
        isinstance(k, str) and k.isidentifier() for k in params
    ):
        raise ValueError("params maps parameter names to recorded values")
    action = raw.get("action")
    if runner == "dialogue" and not _index(action):
        raise ValueError(
            "a dialogue episode cover names the action the procedure must reproduce",
        )
    paths = raw.get("paths", [])
    if not isinstance(paths, list) or not all(isinstance(p, str) for p in paths):
        raise ValueError("paths lists work-tree paths")
    return Cover(
        eid,
        "episode",
        runner=runner,
        action=action if runner == "dialogue" else None,
        params=dict(params),
        paths=tuple(paths),
    )


def cover_raw(c: Cover) -> dict:
    """*c* in the manifest's form, which :func:`parse_cover` reads back to an equal cover."""
    if c.type == "cell":
        return {"episode": c.episode, "type": "cell", "cell": c.index}
    if c.type == "diff":
        return {"episode": c.episode, "type": "diff"}
    raw: dict = {"episode": c.episode, "type": "episode", "runner": c.runner}
    if c.action is not None:
        raw["action"] = c.action
    if c.params:
        raw["params"] = dict(c.params)
    if c.paths:
        raw["paths"] = list(c.paths)
    return raw


def _scalars(v: Any) -> Iterable[Any]:
    if isinstance(v, dict):
        for x in v.values():
            yield from _scalars(x)
    elif isinstance(v, (list, tuple)):
        for x in v:
            yield from _scalars(x)
    else:
        yield v


def param_problems(
    params: dict,
    ep: Episode,
    blob: Callable[[str], bytes],
) -> list[str]:
    """Each parameter holding a string or number the episode did not record (booleans and None aside)."""
    if not params:
        return []
    recorded = recorded_literals([ep], blob)
    out = []
    for name, value in params.items():
        if any(
            not (isinstance(x, bool) or x is None) and canonical(x) not in recorded
            for x in _scalars(value)
        ):
            out.append(
                f"parameter {name} holds a value the episode did not record; a procedure's parameters come from "
                "its recording",
            )
    return out


def action_value(a: Action) -> Any:
    if a.kwargs:
        return dict(a.kwargs)
    return a.args[0] if len(a.args) == 1 else list(a.args)


def snapshot_files(repo: Repo) -> Callable[[str, Path], Path | None]:
    """A reader of the work-tree snapshots (``<UNIFY_HOME>/worktree.git``): *sha*'s files materialised at *dest*."""

    def read(sha: str, dest: Path) -> Path | None:
        try:
            files, _ = listing(repo, sha)
            return materialise(repo, files, dest)
        except (
            Exception
        ):  # noqa: BLE001 - an unreadable snapshot gives nothing to compare
            return None

    return read


def _files(root: Path) -> dict[str, bytes]:
    return {
        str(p.relative_to(root)): p.read_bytes()
        for p in sorted(root.rglob("*"))
        if p.is_file() and not p.is_symlink()
    }


def _cls(result: dict) -> str:
    name = result.get("error_class")
    return (
        name
        if isinstance(name, str) and name.isidentifier() and len(name) <= 64
        else "an exception"
    )


def _box(
    item: str,
    spec: dict,
    *,
    tree: Path,
    python: Path,
    work: Path,
    runner: Callable[..., SandboxResult],
    timeout_s: float,
    scratch: Path | None = None,
) -> tuple[dict | None, SandboxResult]:
    case, out, kit = work / "case", work / "out", work / "kit"
    case.mkdir(parents=True)
    out.mkdir()
    testkit.stage_package(kit / testkit.PACKAGE)
    (case / "run.py").write_text(_RUN)
    (case / "case.json").write_text(json.dumps({**spec, "item": item}, default=str))
    rw = {out: "/out", **({scratch: "/work"} if scratch is not None else {})}
    res = runner(
        [str(python), "-I", "/case/run.py"],
        ro={tree: "/memory", case: "/case", kit: "/kit"},
        rw=rw,
        cwd="/work" if scratch is not None else "/memory",
        timeout_s=timeout_s,
        env={},
    )
    try:
        result = json.loads((out / "result.json").read_text())
    except (OSError, ValueError):
        result = None
    return (result if isinstance(result, dict) else None), res


def _ran(result: dict | None, res: SandboxResult, timeout_s: float) -> str | None:
    if res.timed_out:
        return f"it did not finish within {timeout_s:g} s"
    if result is None:
        return "it could not be run (no result)"
    return None


def run_procedure(
    item: str,
    cover: Cover,
    *,
    ep: Episode,
    tree: Path,
    python: Path,
    work: Path,
    blob: Callable[[str], bytes],
    worktree_files: Callable[[str, Path], Path | None],
    signals: Callable[[str], list],
    runner: Callable[..., SandboxResult] = run_confined,
    timeout_s: float = PROCEDURE_S,
    checker_visible: bool = False,
) -> Outcome:
    problems = param_problems(cover.params, ep, blob)
    if problems:
        return Outcome(False, problems[0])
    work.mkdir(parents=True, exist_ok=True)
    kw = {
        "tree": tree,
        "python": python,
        "work": work,
        "runner": runner,
        "timeout_s": timeout_s,
    }
    if cover.runner == "worktree":
        return _worktree(item, cover, ep, worktree_files, kw)
    if cover.runner == "tool":
        return _tool(item, cover, ep, kw)
    return _dialogue(item, cover, ep, signals, kw, checker_visible)


def _worktree(
    item: str,
    cover: Cover,
    ep: Episode,
    worktree_files,
    kw: dict,
) -> Outcome:
    if not ep.worktree_before or not ep.worktree_after:
        return Outcome(False, "the episode recorded no work-tree snapshots")
    work: Path = kw["work"]
    before = worktree_files(ep.worktree_before, work / "before")
    after = worktree_files(ep.worktree_after, work / "after")
    if before is None or after is None:
        return Outcome(False, "the episode's work-tree snapshots cannot be read")
    b, a = _files(before), _files(after)
    written = {p for p in set(b) | set(a) if b.get(p) != a.get(p)}
    paths = list(cover.paths) or sorted(written)
    if not paths:
        return Outcome(
            False,
            "the episode changed no file: there is no effect to confirm",
        )
    stray = [p for p in paths if p not in written]
    if stray:
        return Outcome(False, f"paths {stray[:5]} were not changed by the episode")
    scratch = work / "scratch"
    shutil.copytree(before, scratch)
    result, res = _box(
        item,
        {"runner": "worktree", "params": cover.params},
        scratch=scratch,
        **kw,
    )
    why = _ran(result, res, kw["timeout_s"])
    if why is not None:
        return Outcome(False, why)
    if result.get("outcome") != "returned":
        return Outcome(False, f"it raised {_cls(result)}")
    got = _files(scratch)
    differ = [p for p in paths if got.get(p) != a.get(p)]
    if differ:
        return Outcome(
            False,
            f"its result differs from the recorded work tree at {len(differ)} of {len(paths)} paths: {differ[:5]}",
        )
    extra = sorted(
        p for p in set(got) | set(b) if p not in written and got.get(p) != b.get(p)
    )
    notes = (
        [f"it also changed {len(extra)} path(s) the episode did not change"]
        if extra
        else []
    )
    return Outcome(True, None, notes)


def _rows(ep: Episode) -> list[dict]:
    return [
        {
            "channel": a.channel,
            "method": a.method,
            "args": list(a.args),
            "kwargs": dict(a.kwargs or {}),
            "response": a.response,
            "status": a.status,
            "effect": a.effect,
            "error": a.error,
            "kind": "tool",
        }
        for a in ep.actions
        if getattr(a, "kind", "tool") == "tool"
    ]


def _tool_boxes(
    item: str,
    params: dict,
    case_data: dict,
    *,
    tree: Path,
    python: Path,
    work: Path,
    runner: Callable[..., SandboxResult],
    timeout_s: float,
) -> tuple[dict | None, SandboxResult]:
    """Run *item* against the recording in two boxes: (the server's result, the item box's result).

    The item's box sees the library, the kit, its name and parameters, and the server's socket, never the case:
    so it cannot reproduce the recording without computing. The server's box sees the case and the socket's
    folder and writes ``/out/result.json``.
    """
    case, out, kit, chan, item_dir = (
        work / "case",
        work / "out",
        work / "kit",
        work / "chan",
        work / "item",
    )
    for d in (case, out, chan, item_dir):
        d.mkdir(parents=True)
    testkit.stage_package(kit / testkit.PACKAGE)
    (case / "serve.py").write_text(_SERVE)
    (case / "case.json").write_text(json.dumps(case_data, default=str))
    (item_dir / "run.py").write_text(_ITEM_RUN)
    (item_dir / "item.json").write_text(
        json.dumps({"item": item, "params": params}, default=str),
    )
    served: dict = {}

    def serve() -> None:
        try:
            served["res"] = runner(
                [str(python), "-I", "/case/serve.py", f"{timeout_s:g}"],
                ro={case: "/case", kit: "/kit"},
                rw={out: "/out", chan: "/chan"},
                cwd="/tmp",
                timeout_s=timeout_s + SERVER_GRACE_S,
                env={},
            )
        except BaseException as exc:  # noqa: BLE001 - reported as no result
            served["error"] = exc

    server = threading.Thread(target=serve, name="replay-env-server", daemon=True)
    server.start()
    try:
        res = runner(
            [str(python), "-I", "/item/run.py"],
            ro={tree: "/memory", kit: "/kit", item_dir: "/item", chan: "/chan"},
            cwd="/memory",
            timeout_s=timeout_s,
            env={},
        )
    finally:
        # a server still waiting for an item that never connected ends now instead of at its own limit
        with contextlib.suppress(OSError), socket.socket(socket.AF_UNIX) as poke:
            poke.settimeout(1.0)
            poke.connect(str(chan / SOCKET))
        server.join(timeout_s + 2 * SERVER_GRACE_S)
    try:
        result = json.loads((out / "result.json").read_text())
    except (OSError, ValueError):
        result = None
    return (result if isinstance(result, dict) else None), res


def _tool(item: str, cover: Cover, ep: Episode, kw: dict) -> Outcome:
    rows = _rows(ep)
    if not rows:
        return Outcome(False, "the episode recorded no tool call")
    try:
        writes: list[dict] | None = recorded_calls(rows, effect="write")
    except UnknownEffect:
        writes = None
    if writes is not None and not writes:
        return Outcome(
            False,
            "the episode recorded no write call: there is no effect to confirm (a reader takes action covers)",
        )
    result, res = _tool_boxes(
        item,
        cover.params,
        {"rows": rows, "effect": "write" if writes is not None else None},
        **kw,
    )
    why = _ran(result, res, kw["timeout_s"])
    if why is not None:
        return Outcome(False, why)
    misses = result.get("misses")
    if isinstance(misses, int) and misses:
        return Outcome(False, f"it made {misses} call(s) the episode did not record")
    if result.get("outcome") != "returned":
        return Outcome(False, f"it raised {_cls(result)}")
    issued = result.get("issued_writes") if writes is not None else result.get("issued")
    want = writes if writes is not None else recorded_calls(rows)
    if not isinstance(issued, list) or issued != want:
        n = len(issued) if isinstance(issued, list) else 0
        what = "write " if writes is not None else ""
        return Outcome(
            False,
            f"it issued {n} {what}call(s); the episode recorded {len(want)} (compared in order, with arguments)",
        )
    return Outcome(True)


def _dialogue(
    item: str,
    cover: Cover,
    ep: Episode,
    signals,
    kw: dict,
    checker_visible: bool = False,
) -> Outcome:
    i = cover.action
    a = ep.actions[i] if i is not None and i < len(ep.actions) else None
    if a is None or getattr(a, "kind", "tool") != "dialogue" or a.status != "ok":
        return Outcome(False, f"action {i} is not an answered dialogue action")
    # a checker verdict counts only when the bed declares its verdicts visible AND this signal is
    # one the actor saw (visible_to_actor), as lifecycle.counted_signals; a hidden grader is never evidence
    sigs = [
        s
        for s in (signals(ep.episode_id) or [])
        if getattr(s, "source", None) != "checker"
        or (checker_visible and getattr(s, "visible_to_actor", False) is True)
    ]
    if any((s.source, s.label) in _CONTRARY for s in sigs):
        return Outcome(False, "the episode carries a contrary signal")
    if not any((s.source, s.label) in _SUPPORT for s in sigs):
        return Outcome(
            False,
            "no positive signal is recorded for the episode; a dialogue procedure must reproduce an action a "
            "positive signal followed",
        )
    prev = [
        b
        for b in ep.actions[:i]
        if getattr(b, "kind", "tool") == "dialogue"
        and b.channel == a.channel
        and b.status == "ok"
    ]
    observation = prev[-1].response if prev else (ep.request[0] if ep.request else "")
    result, res = _box(
        item,
        {"runner": "dialogue", "params": cover.params, "observation": observation},
        **kw,
    )
    why = _ran(result, res, kw["timeout_s"])
    if why is not None:
        return Outcome(False, why)
    if result.get("outcome") != "returned":
        return Outcome(False, f"it raised {_cls(result)}")
    if result.get("value") != json.dumps(action_value(a), sort_keys=True, default=repr):
        return Outcome(False, "its result differs from the recorded action")
    return Outcome(True)
