"""Typed covers of library items, and stored readers.

**Covers** (:func:`parse_cover`, :func:`cover_raw`): a function's typed cover names the episode it comes from and
its type: an actor ``cell`` that defined it, the work-tree ``diff`` that added it, or an ``episode`` cover naming the
runner of the episode's action kind (``worktree``, ``tool`` or ``dialogue``). The gate reads covers against the
recording and re-runs none of them (rev 2.1, :mod:`.gate_policy`); an ``episode`` cover is recorded on the item's
label as not run. A procedure of another kind has no runner; :func:`parse_cover` refuses it, so it goes to drafts.

**Stored readers** (:func:`reader_parse`): a library function with one text parameter that a fixture names
(``fixture(..., parser="<id>")``). The writer's fixture tool runs it on the recorded bytes to make the fixture,
confined (:func:`.sandbox_run.run_confined`): the library read-only at ``/memory``, the kit at ``/kit``, the texts
read-only at ``/case``, writable only ``/out``, no network and an empty environment. The host only reads JSON.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from . import testkit
from .episodes import Action
from .request_value import REQUEST_CLASS, request_data
from .sandbox_run import SandboxResult, run_confined

RUNNERS = ("worktree", "tool", "dialogue")
COVER_TYPES = ("cell", "diff", "episode")
PROCEDURE_S = 60.0
_EPISODE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}\Z")
_KEYS = frozenset({"episode", "type", "cell", "runner", "action", "params", "paths"})

@dataclass(frozen=True)
class Cover:
    episode: str
    type: str
    index: int | None = None
    runner: str | None = None
    action: int | None = None
    params: dict = field(default_factory=dict, hash=False)
    paths: tuple[str, ...] = ()


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


def action_value(a: Action) -> Any:
    if a.kwargs:
        return dict(a.kwargs)
    return a.args[0] if len(a.args) == 1 else list(a.args)


# --- stored readers --------------------------------------------------------------------------------------------
#
# A reader is a library function with one text parameter that returns a value JSON can encode (and raises
# MemoryInputError on text it was not built for). Nothing marks a function as a reader in advance: it is one when a
# fixture names it (``fixture(..., parser="<id>")``). The writer's fixture tool runs it on the recorded bytes; the
# gate never runs it. It runs only confined: the library read-only at /memory, the texts read-only at /case, writable
# only /out, no network, an empty environment.

_PARSE = r"""
import importlib, inspect, json, sys
sys.path[:0] = ["/memory", "/kit"]
sys.dont_write_bytecode = True
spec = json.load(open("/case/case.json"))
out = {"results": []}
""" + REQUEST_CLASS + r"""


def _text(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


try:
    module, _, name = spec["item"].partition(":")
    fn = getattr(importlib.import_module(module), name)
    try:
        params = list(inspect.signature(fn).parameters.values())
    except (TypeError, ValueError):
        params = None
    positional = (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
    out["one_text_parameter"] = bool(
        params is not None
        and len(params) == 1
        and params[0].kind in positional
        and params[0].annotation in (inspect.Parameter.empty, str, "str")
    )
    if out["one_text_parameter"]:
        for text, data in zip(spec["texts"], spec["data"]):
            try:
                first = fn(Request(text, data))
            except BaseException as exc:
                out["results"].append({"outcome": "raised", "error_class": type(exc).__name__})
                continue
            try:
                value = _text(first)
            except (TypeError, ValueError, RecursionError):
                out["results"].append({"outcome": "not_json"})
                continue
            try:
                again = _text(fn(Request(text, data)))
            except BaseException:
                again = None
            out["results"].append({"outcome": "returned", "value": value, "same": value == again})
except BaseException as exc:
    out["error_class"] = type(exc).__name__
with open("/out/result.json", "w") as f:
    json.dump(out, f)
"""


def reader_digest(tree: Path, item: str) -> str | None:
    """The sha256 of reader *item*'s ``def`` source in *tree* (None: its module or ``def`` cannot be read). A parsed
    fixture records it, so a later change of the reader is seen and its fixtures are made again."""
    import ast
    import hashlib

    from .fixtures import module_path

    try:
        source = (Path(tree) / module_path(item)).read_text(encoding="utf-8")
        mod = ast.parse(source)
    except (OSError, SyntaxError, ValueError, UnicodeDecodeError):
        return None
    name = item.split(":", 1)[1]
    for node in mod.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
            segment = ast.get_source_segment(source, node) or ""
            return hashlib.sha256(segment.encode("utf-8")).hexdigest()
    return None


def run_reader(
    item: str,
    texts: list[str],
    *,
    tree: Path,
    python: Path,
    work: Path,
    runner: Callable[..., SandboxResult] = run_confined,
    timeout_s: float = PROCEDURE_S,
) -> tuple[list[str | None], str | None]:
    """Run reader *item* confined on each of *texts*, twice each: (per text its result as canonical JSON text, or
    None; why the reader cannot be used, or None). The reader is handed each text with its data, as every reader is
    (:mod:`.request_value`).

    The reader conditions, each a reason: it takes one positional text parameter; it returns on the text (a
    ``MemoryInputError`` or any other exception is named by class); its result encodes as JSON; two runs give the
    same result; the whole batch ends within ``timeout_s`` per text.
    """
    if not texts:
        return [], None
    case, out, kit = work / "case", work / "out", work / "kit"
    case.mkdir(parents=True)
    out.mkdir()
    testkit.stage_package(kit / testkit.PACKAGE)
    (case / "run.py").write_text(_PARSE)
    texts = [str(t) for t in texts]
    data = [request_data(t) for t in texts]
    (case / "case.json").write_text(json.dumps({"item": item, "texts": texts, "data": data}, ensure_ascii=False))
    limit = float(timeout_s) * len(texts)
    res = runner(
        [str(python), "-I", "/case/run.py"],
        ro={tree: "/memory", case: "/case", kit: "/kit"},
        rw={out: "/out"},
        cwd="/memory",
        timeout_s=limit,
        env={},
    )
    if res.timed_out:
        return [None] * len(texts), f"reader {item} did not finish within {limit:g} s"
    try:
        got = json.loads((out / "result.json").read_text())
    except (OSError, ValueError):
        got = None
    if not isinstance(got, dict):
        return [None] * len(texts), f"reader {item} could not be run (no result)"
    if got.get("error_class"):
        return [None] * len(texts), f"reader {item} could not be loaded ({got['error_class']})"
    if not got.get("one_text_parameter"):
        return [None] * len(texts), f"reader {item} does not take one positional text parameter"
    rows = got.get("results") if isinstance(got.get("results"), list) else []
    if len(rows) != len(texts):
        return [None] * len(texts), f"reader {item} gave {len(rows)} results for {len(texts)} texts"
    values: list[str | None] = []
    for row in rows:
        outcome = row.get("outcome") if isinstance(row, dict) else None
        if outcome == "raised":
            return [None] * len(texts), f"reader {item} raises {row.get('error_class')} on the recorded text"
        if outcome == "not_json":
            return [None] * len(texts), f"reader {item} returns a value JSON cannot encode"
        if outcome != "returned" or not isinstance(row.get("value"), str):
            return [None] * len(texts), f"reader {item} could not be run (no result)"
        if not row.get("same"):
            return [None] * len(texts), f"reader {item} gives different results on two runs over the same text"
        values.append(row["value"])
    return values, None


def reader_parse(
    tree: Path,
    *,
    python: Path,
    work: Path,
    runner: Callable[..., SandboxResult] = run_confined,
    allowed: Callable[[str], str | None] | None = None,
) -> Callable[[str, str], tuple[str, str]]:
    """A ``parse(reader, text) -> (canonical JSON text, reader digest)`` over the library at *tree*, for
    :func:`.fixtures.make` and :func:`.fixtures.verify`; each (reader, text) runs once. *allowed* (reader -> why it
    may not be used, or None) adds the caller's own conditions. Raises :class:`.fixtures.FixtureError`.
    """
    import hashlib

    from .fixtures import FixtureError

    memo: dict[tuple[str, str], tuple[str, str]] = {}
    n = [0]

    def parse(reader: object, text: str) -> tuple[str, str]:
        from .layout import FUNCTION_ID

        if not isinstance(reader, str) or not FUNCTION_ID.match(reader):
            raise FixtureError(f"parser {str(reader)[:120]!r} is not a library function id")
        why = allowed(reader) if allowed is not None else None
        if why:
            raise FixtureError(why)
        key = (reader, hashlib.sha256(text.encode("utf-8")).hexdigest())
        if key in memo:
            return memo[key]
        digest = reader_digest(tree, reader)
        if digest is None:
            raise FixtureError(f"reader {reader} is not a function of the library")
        n[0] += 1
        values, problem = run_reader(
            reader,
            [text],
            tree=tree,
            python=python,
            work=work / f"reader-{n[0]}",
            runner=runner,
        )
        if problem is not None or values[0] is None:
            raise FixtureError(problem or f"reader {reader} could not be run")
        memo[key] = (values[0], digest)
        return memo[key]

    return parse
