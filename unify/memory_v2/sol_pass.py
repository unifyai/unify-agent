"""One consolidation pass: Sol as a sandboxed coding agent, then the gate (spec §7, D9).

The pass copies the memory library's ``main`` into a plain directory, creates an empty ``env/<channel>/``
folder there for every memory channel of the pass's recorded actions, and gives Sol three tools:
``execute_code`` (the code as a read-only file run in :func:`.sandbox_run.run_confined`, with the copy
read-write at ``/memory`` and the pass's inputs read-only at ``/inputs``), ``check`` (the gate's cheap,
read-only checks of a manifest against the current copy, host-side; :meth:`.gate.Gate.preview`; at most
:data:`MAX_CHECKS` per pass, each counted as a call) and ``finish``. It ends on
``finish``, ``max_calls``, the USD cap, the pass deadline or a breach of the write quota (which refuses the
pass without a commit). The harness then reads ``/memory/.pass/manifest.json`` without following
links, mirrors the copy into a git checkout (dropping ``.pass``, nested ``.git`` entries and caches),
commits with ``Pass``/``Episode``/``Evidence`` trailers and hands the commit to :meth:`.gate.Gate.merge`,
which records the pass. A missing manifest is recorded here as failed with the reason ``no manifest``.

The box never sees the git checkout itself: a ``.git`` file the model could rewrite would point the host's
``git add`` at a repository (and configuration) of its choosing.

Sol sees only what :func:`export_for_sol` and :func:`export_blobs` write (ruling R10): request, cells,
actions, and the file blobs worktree actions recorded. Nothing about outcomes, signals or checkers reaches it.
"""

from __future__ import annotations

import asyncio
import contextvars
import copy
import json
import os
import re
import shutil
import stat
import tempfile
import threading
import time
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Callable, Iterable, Protocol

from pydantic import SecretStr

from . import manifest as _manifest
from .blobs import BLOB_ID, BlobStore
from .episodes import Episode, env_channel
from .evidence import EvidenceStore
from .gate import Gate, ParentSnapshot
from .gitio import Repo
from .index import build_index
from .redact import redact_error
from .sandbox_run import PRLIMIT, PYTHON, run_confined
from .trigger import PassRequest

# Write bounds on /memory, checked after every cell and before the commit; a breach refuses the pass.
QUOTA_TOTAL_BYTES = 20 * 1024**2
QUOTA_FILE_BYTES = 1024**2
QUOTA_ENTRIES = 2000
# RLIMIT_FSIZE inside a Sol cell, lowered from run_confined's by prlimit in the box (lowering needs no
# privilege); the quota walk is the precise check, this only bounds a single file between walks.
CELL_FSIZE_BYTES = 2 * QUOTA_FILE_BYTES
# Recorded file blobs exported to /inputs/blobs: one blob at most a library file's quota (so any exported
# blob can become a test fixture), all of them at most this much.
EXPORT_BLOB_BYTES = QUOTA_FILE_BYTES
EXPORT_TOTAL_BYTES = 32 * 1024**2

# check calls per pass (each also counts against PassConfig.max_calls); a reply lists at most this many reasons
MAX_CHECKS = 5
_CHECK_REASONS = 20
_CHECK_REASON_CHARS = 200
# A check starts only with at least this much of the pass deadline left (it runs to completion).
CHECK_FLOOR_S = 60.0

_RULES_HEADING = "Manifest rules for consolidators"


def _manifest_rules() -> str:
    """The consolidator rules of :mod:`.manifest`'s docstring, verbatim, so prompt and gate cannot drift."""
    doc = _manifest.__doc__ or ""
    start = doc.find(_RULES_HEADING)
    if start < 0:
        raise ImportError(
            f"unify.memory_v2.manifest's docstring lacks the section {_RULES_HEADING!r} "
            "(docstrings stripped by -OO?)",
        )
    return doc[start:].strip()


_PROMPT = """\
You consolidate an agent's recorded work into a shared library that a cheaper working model will import.
You work in a sandbox with no network. /memory is the library (a copy of its git checkout that you edit; the
harness commits it when you finish); /inputs is read-only and holds request.json (this pass's request: kind,
channel, episodes), episodes/<episode_id>.json (the recorded episodes for this pass), blobs/ (the file contents
the episodes' worktree actions recorded, one file per blob id, with index.json naming any skipped) and `memlab`, a
toolkit: memlab.analysis.cells (call_sites), memlab.analysis.provenance (def_use_edges, value_edges),
memlab.analysis.slicing (backward_slice, generic_methods, prelude), memlab.analysis.antiunify (antiunify),
memlab.analysis.recorded (actions_of_kind, worktree_files, load_blob), memlab.analysis.shapes (shape, signature,
conforms: file format, delimiter, columns and types, key trees, sheets), memlab.analysis.shellout (records,
output_shape, exit_codes, by_shape), memlab.analysis.transitions (transitions, table, conflicts, check, fixture),
memlab.replay.RecordedEnv, memlab.episodes (Cell, Action, env_channel).
Each episode file holds "episode_id", "request", "cells" (code and printed output; memlab.episodes.Cell(**cell))
and "actions" ("index" is the action's action_index, and memlab.episodes.Action(**{k: v for k, v in a.items()
if k != "index"}) rebuilds it), and "memory_channels" (memory_channels[i] is the channel items covering
actions[i] live under).
Every action has a "kind":
- tool: a call through an environment namespace, with its response or error;
- shell: a command (args) with response {"exit_code", "tail"} (the output's last part);
- worktree: a file read or write (args[0] is the path) with response {"blob_before", "blob_after", "size", "shape"};
- dialogue: the action an agent's reply carried (args[0]) with the next observation as its response.
An item covering actions[i] lives in env/<memory_channels[i]>/ exactly (never derive another name); an action
whose memory channel is null cannot be covered. The recordings say
nothing about whether a request was solved; do not guess.
Each execute_code call runs your code as a Python script in a fresh process, cwd /memory,
PYTHONPATH=/inputs:/memory; only files persist between calls. /memory may hold at most {entries} files and
directories, {file_mib} MiB per file and {total_mib} MiB in all, or the pass is refused. Run tests as the gate does:
subprocess.run([sys.executable, "-m", "pytest", "env/<channel>/tests", "-q", "-p", "no:cacheprovider",
"--rootdir", "/memory", "-c", "/dev/null", "--import-mode=importlib"],
env={"PYTHONPATH": "/memory", "PYTHONDONTWRITEBYTECODE": "1"}).

What to build, in priority order:
1. env/<channel>/__init__.py: general, parametric Python functions for the environment work that recurs:
   - tool channels: auth once, typed call wrappers, pagination, known pitfalls, taking the environment object as
     the first argument `apis`;
   - shell channels: functions that parse a command's output text into values, with an output-shape check (the
     gate never runs commands; tests feed recorded outputs);
   - worktree channels: readers and writers for recurring files (taking a path, text or bytes), with schema checks
     (delimiter, columns, types, sign and date conventions) that raise on files of another shape;
   - dialogue channels: observation parsers, an action grammar, transition facts and small predictors.
   Each public function's docstring has a one-line summary and a line `Effect: read`, `Effect: write` or
   `Effect: unknown`, and a line `Input: <form>` saying what its first parameter takes, the same form as
   "input" in its manifest entry (the gate passes each covered input in that form), one of: {input_kinds}.
   Each function checks the shape of its inputs and raises MemoryInputError(diagnosis) when it
   differs. Define MemoryInputError in the module (that is module skeleton: declare "skeleton": ["env/<channel>"]
   when you add it, together with every public function of the module in items).
   Scope is shape, not observed values. A check or refusal names only types, columns or fields, required keys, the
   call (channel, method, parameter names) and formats; never the range or the list of values the recordings
   happened to show. Bad: `if colour not in ("red", "blue"): raise MemoryInputError(...)` because only red and blue
   were recorded. Good: `if not isinstance(colour, str) or not colour: raise MemoryInputError("colour must be a
   non-empty string")`. The one exception: a recording in which the environment itself rejected a value (an error
   response or a nonzero exit) may justify refusing such values, and that rejection must then be one of the
   function's covers. The gate reruns each function on its covered inputs with unseen values of the same types
   (new strings, numbers beyond the recorded range, dates outside the recorded span) and refuses it if it raises
   MemoryInputError on them. Restrict a field's values only through this fixed list of types, declared in the item's
   manifest entry as "field_types": {"<field>": "<type>"}: {semantic_types}.
2. env/<channel>/NOTES.md: `## ` sections for pitfalls observed in the recordings (a failed call followed by one that
   worked) that code cannot express. Each section is an item env/<channel>/NOTES.md#<section-slug> (the slug rule is
   below).
Never store: results of calls with effects, credentials, session tokens (always re-acquire them), rules drawn from a
single judgement, anything about whether a request was solved. Do not write workflow notes in this pass. Do not write
job functions (end-to-end code for a whole request): v0 refuses them.

Test first. For every function you add or change, first write env/<channel>/tests/test_<name>.py over the recorded
observations it covers: replayed calls (use the recorded responses; /memory/unify_memory_testkit.py, the only helper
module allowed at the root, may build the fake environment), recorded shell outputs, recorded file blobs, or every
recorded (action, next observation) pair in its scope. The gate mounts only /memory, so copy the fixtures a test
needs into data files under env/<channel>/tests/ (never memlab) and list them in "support". Test agreement on every covered observation, and that
shape checks reject recorded observations of another shape. Run the test and see it fail,
then write the code and see it pass. Revise existing
modules in place; do not add near-duplicates. Prefer nothing over a weak function: the library must earn its place.

Finish by writing /memory/.pass/manifest.json (never committed):
{"items":[{"item":"env/<channel>:<function>","kind":"env_function","input":"<form>","source_episodes":[...],
  "tests":["env/<channel>/tests/test_<function>.py"],"covers":[[episode_id, action_index], ...]}],
 "support":["unify_memory_testkit.py"],"skeleton":[],"unlisted":[],"deleted":[],"deleted_tests":[],"summary":"..."}
`covers` lists the recorded actions on the function's own channel whose observations its tests check: tool calls
with status "ok" and a response, shell commands with an output tail, worktree reads or writes with a recorded blob,
dialogue actions with status "ok" and an observation, and any recorded rejection (status "error" with its error, or a
nonzero exit) that justifies a value check (never covers made only of rejections).
Before finish, call check(manifest) with the manifest JSON and fix every reason it returns: it runs the gate's
manifest, provenance, scope, cover-channel, index and safety checks (not the tests) on your current files, changes
nothing, and counts as a call ({checks} per pass at most). The folders env/<channel>/ for this pass's memory channels
already exist; put each item in the one its covers' memory_channels name. Then call finish(summary).
A deterministic gate will check provenance, that each new test fails before your change and passes after, the full
test suite, an index budget, that the library only grows when it covers new recorded calls, and safety. Its rules
follow; a pass that breaks one is refused whole.
"""

SOL_SYSTEM = (
    _PROMPT.replace("{entries}", str(QUOTA_ENTRIES))
    .replace("{file_mib}", str(QUOTA_FILE_BYTES // 1024**2))
    .replace("{total_mib}", str(QUOTA_TOTAL_BYTES // 1024**2))
    .replace("{checks}", str(MAX_CHECKS))
    .replace("{semantic_types}", _manifest.describe_semantic_types())
    .replace("{input_kinds}", _manifest.describe_input_kinds())
    + "\n"
    + _manifest_rules()
    + "\n"
)

# The toolkit copied into /inputs/memlab. Not gitio (git), the gate or the sandbox runner.
_MEMLAB_FILES = (
    "analysis",
    "replay.py",
    "episodes.py",
    "fingerprint.py",
    "blobs.py",
    "redact.py",
)
_GITIO_STUB = '''\
"""memlab has no git inside the consolidation sandbox; this stands in for the names episodes.py imports."""


class GitError(RuntimeError):
    pass


class Repo:
    def __init__(self, *args, **kwargs):
        raise GitError("git is not available inside the consolidation sandbox")
'''
_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "execute_code",
            "description": "Run Python code as a script in the sandbox, cwd /memory; returns stdout and stderr.",
            "parameters": {
                "type": "object",
                "properties": {"code": {"type": "string"}},
                "required": ["code"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "check",
            "description": (
                "Check a manifest against your current /memory files with the gate's cheap checks (manifest, "
                "provenance and scope, covers and their channels, index, safety; not the tests). Returns 'ok' "
                f"or the gate's reasons. Changes nothing; at most {MAX_CHECKS} per pass, each counted as a call."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "manifest": {
                        "type": "string",
                        "description": "The manifest JSON, as you will write it to /memory/.pass/manifest.json.",
                    },
                },
                "required": ["manifest"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "finish",
            "description": "End the pass after writing /memory/.pass/manifest.json.",
            "parameters": {
                "type": "object",
                "properties": {"summary": {"type": "string"}},
                "required": ["summary"],
            },
        },
    },
]
_OUTPUT_CAP = 8000
_MAX_CELLS_PER_TURN = 8
_MANIFEST_MAX_BYTES = 1024**2
_MANIFEST_MAX_DEPTH = 16
# Never mirrored out of the box: git metadata (anywhere), the pass directory (top level), caches.
_NEVER_COPIED = frozenset({".git", "__pycache__", ".pytest_cache"})
_EPISODE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}\Z")
_UNKNOWN = "unknown"

# Structured causes of a pass's end (PassOutcome.codes), set where each cause arises, never read back from
# reason text. A passed pass is exactly ["ok"]; a refused pass lists the gate checks that refused (G1..G6).
CODE_OK = "ok"
CODE_NO_MANIFEST = "no_manifest"
CODE_MANIFEST_INVALID = "manifest_invalid"
CODE_OVER_QUOTA = "over_quota"
CODE_DEADLINE = "deadline"
CODE_PASS_CAP = "pass_cap"
CODE_SOL_ERROR = "sol_error"
# a Sol call found Sol's declared route not in effect
CODE_ROUTE_NOT_IN_EFFECT = "route_not_in_effect"


class ModelTurn(Protocol):
    async def __call__(
        self,
        messages: list[dict],
        tools: list[dict],
    ) -> tuple[dict, str]: ...


@dataclass
class PassConfig:
    """Per-pass bounds. ``model`` and ``effort`` build the real turn: ``unillm_turn(cfg.model, cfg.effort)``."""

    model: str = "openai/gpt-6-sol"
    effort: str = "low"
    max_calls: int = 40
    max_usd: Decimal = Decimal("1.00")
    cell_timeout_s: float = 60.0
    deadline_s: float = 900.0


@dataclass
class PassOutcome:
    """``usd`` is the known spend as a pure decimal string (ruling R22); unpriced calls are counted apart."""

    pass_id: str
    passed: bool
    commit: str | None
    usd: str
    calls: int  # model calls plus check calls: both count against PassConfig.max_calls
    reasons: list[str] = field(default_factory=list)
    summary: str = ""
    unknown_cost_calls: int = 0
    codes: list[str] = field(
        default_factory=list,
    )  # structured causes (the CODE_* constants)
    checks: int = 0  # check calls run (included in calls)


# --- inputs ------------------------------------------------------------------------------------------------


def export_for_sol(
    load: Callable[[str], Episode],
    eids: Iterable[str],
    dest: Path,
) -> None:
    """Write ``<dest>/<episode_id>.json`` per episode: the request, cells and actions, and nothing else.

    Ruling R10: no outcome, signal or checker data reaches Sol. The fields are an allowlist: cells carry
    their index, code and printed output; actions their index (the ``action_index`` covers cite), cell,
    channel, method, args, kwargs, response, status, effect, error and kind; ``memory_channels`` lists, per
    action, the gate's :func:`.episodes.env_channel` of its kind and channel (kept beside the actions so the
    documented ``Action(**...)`` rebuild never meets an unknown key). Anything else an episode carries
    (transcript, regime, costs, diffs, fingerprints, cell errors, or attributes a later schema adds) is
    dropped.
    """
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    for eid in eids:
        if not isinstance(eid, str) or not _EPISODE_ID.match(eid):
            raise ValueError(f"unsafe episode id {eid!r}"[:200])
        ep = load(eid)
        row = {
            "episode_id": eid,
            "request": list(ep.request),
            "cells": [
                {"index": c.index, "code": c.code, "output": c.output} for c in ep.cells
            ],
            "actions": [
                {
                    "index": i,
                    "cell": a.cell,
                    "channel": a.channel,
                    "method": a.method,
                    "args": a.args,
                    "kwargs": a.kwargs,
                    "response": a.response,
                    "status": a.status,
                    "effect": a.effect,
                    "error": a.error,
                    "kind": getattr(a, "kind", "tool"),
                }
                for i, a in enumerate(ep.actions)
            ],
            # the gate's own mapping (G2), so Sol never derives a channel name
            "memory_channels": [
                env_channel(getattr(a, "kind", "tool"), a.channel) for a in ep.actions
            ],
        }
        (dest / f"{eid}.json").write_text(
            json.dumps(row, sort_keys=True, default=str, indent=1) + "\n",
        )


def export_blobs(
    load: Callable[[str], Episode],
    eids: Iterable[str],
    blobs: BlobStore | None,
    dest: Path,
    *,
    per_blob_bytes: int = EXPORT_BLOB_BYTES,
    total_bytes: int = EXPORT_TOTAL_BYTES,
) -> dict:
    """Copy the file blobs the episodes' worktree actions recorded to ``<dest>/<blob id>``, size-capped.

    Only well-formed ids (:data:`.blobs.BLOB_ID`) of ``blob_before``/``blob_after`` on ``kind ==
    "worktree"`` actions are looked up, in episode and action order. A blob larger than *per_blob_bytes*,
    one past *total_bytes* in all, or one missing from the store is skipped. ``<dest>/index.json`` lists
    ``{"exported": [...], "skipped": {id: reason}}`` (memlab's ``analysis.recorded.load_blob`` reads it).
    """
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    exported: list[str] = []
    skipped: dict[str, str] = {}
    used = 0
    for eid in eids:
        if not isinstance(eid, str) or not _EPISODE_ID.match(eid):
            raise ValueError(f"unsafe episode id {eid!r}"[:200])
        for a in load(eid).actions:
            if getattr(a, "kind", "tool") != "worktree" or not isinstance(
                a.response,
                dict,
            ):
                continue
            for key in ("blob_before", "blob_after"):
                sha = a.response.get(key)
                if not isinstance(sha, str) or not BLOB_ID.match(sha):
                    continue
                if sha in skipped or sha in exported:
                    continue
                if blobs is None or not blobs.has(sha):
                    skipped[sha] = "not in the blob store"
                    continue
                size = blobs.size(sha)
                if size > per_blob_bytes:
                    skipped[sha] = (
                        f"{size} bytes, over the per-blob cap of {per_blob_bytes}"
                    )
                elif used + size > total_bytes:
                    skipped[sha] = f"over the export's total cap of {total_bytes} bytes"
                else:
                    (dest / sha).write_bytes(blobs.get(sha))
                    used += size
                    exported.append(sha)
    (dest / "index.json").write_text(
        json.dumps({"exported": exported, "skipped": skipped}, sort_keys=True) + "\n",
    )
    return {"exported": exported, "skipped": skipped}


def _stage_memlab(lab: Path) -> None:
    src = Path(__file__).parent
    lab.mkdir()
    for name in _MEMLAB_FILES:
        s = src / name
        if s.is_dir():
            shutil.copytree(
                s,
                lab / name,
                ignore=shutil.ignore_patterns("__pycache__"),
            )
        else:
            shutil.copy2(s, lab / name)
    (lab / "__init__.py").write_text(
        '"""memlab: analysis and replay tools for consolidation passes."""\n',
    )
    (lab / "gitio.py").write_text(_GITIO_STUB)


# --- the box's tree ----------------------------------------------------------------------------------------


def _unlock(path: Path, st: os.stat_result) -> os.stat_result:
    """Give the owner read (and, on directories, write and search) permission; never through a link.

    The box runs as this user, so it can ``chmod`` what it wrote; the host restores its own access before
    measuring or copying. Called only on an entry that ``lstat`` showed to be a directory or regular file,
    while no cell runs (the box is dead between cells), so it cannot have been swapped for a link.
    """
    want = 0o700 if stat.S_ISDIR(st.st_mode) else 0o600
    if stat.S_IMODE(st.st_mode) & want != want:
        os.chmod(path, stat.S_IMODE(st.st_mode) | want)
        return os.lstat(path)
    return st


def _measure(root: Path) -> str | None:
    """The quota the tree breaks, or None. Entries are made owner-accessible on the way.

    Walks with ``lstat`` (no link is followed, no file is read) and stops at the first breach: more than
    :data:`QUOTA_ENTRIES` files, links and directories, a regular file over :data:`QUOTA_FILE_BYTES`,
    regular files over :data:`QUOTA_TOTAL_BYTES` in all, or an entry that stays unreadable after
    :func:`_unlock` (a ``chmod 0`` directory could otherwise hide any amount).
    """
    entries = total = 0
    unreadable: list[str] = []
    root = Path(root)
    try:
        _unlock(root, os.lstat(root))
    except OSError:
        return "over quota: unreadable entries ['.']"
    stack = [(root, "")]
    while stack:
        d, rel = stack.pop()
        try:
            it = os.scandir(d)
        except OSError:
            unreadable.append(rel.rstrip("/") or ".")
            continue
        with it:
            for e in it:
                r = rel + e.name
                try:
                    st = os.lstat(e.path)
                    if stat.S_ISDIR(st.st_mode) or stat.S_ISREG(st.st_mode):
                        st = _unlock(Path(e.path), st)
                except OSError:
                    unreadable.append(r)
                    continue
                entries += 1
                if entries > QUOTA_ENTRIES:
                    return f"over quota: more than {QUOTA_ENTRIES} files and directories under /memory"
                if stat.S_ISDIR(st.st_mode):
                    stack.append((Path(e.path), r + "/"))
                elif stat.S_ISREG(st.st_mode):
                    if st.st_size > QUOTA_FILE_BYTES:
                        return (
                            f"over quota: /memory/{r[:200]} has {st.st_size} bytes "
                            f"(at most {QUOTA_FILE_BYTES} per file)"
                        )
                    total += st.st_size
                    if total > QUOTA_TOTAL_BYTES:
                        return f"over quota: more than {QUOTA_TOTAL_BYTES} bytes under /memory"
    if unreadable:
        return f"over quota: unreadable entries {sorted(unreadable)[:5]}"
    return None


def _mirror(
    src: Path,
    dst: Path,
    skip_top: frozenset[str] = frozenset(),
) -> list[str]:
    """Copy *src* into the existing *dst* without following links; return the entries left out.

    Directories and regular files (with their permission bits) are copied, links are recreated as links
    (the gate refuses them); FIFOs, sockets, devices and anything that cannot be read are left out (an
    unreadable directory is left empty). ``.git``, ``__pycache__`` and ``.pytest_cache`` are never copied,
    at any depth, nor *skip_top* names at the top.
    """
    left_out: list[str] = []

    def walk(s: Path, d: Path, rel: str, top: bool) -> None:
        try:
            with os.scandir(s) as it:
                entries = sorted(it, key=lambda e: e.name)
        except OSError:
            left_out.append(rel.rstrip("/") or ".")
            return
        for e in entries:
            if e.name in _NEVER_COPIED or (top and e.name in skip_top):
                continue
            sp, dp, r = Path(e.path), d / e.name, rel + e.name
            try:
                st = os.lstat(sp)
                if stat.S_ISLNK(st.st_mode):
                    os.symlink(os.readlink(sp), dp)
                elif stat.S_ISDIR(st.st_mode):
                    dp.mkdir()
                    walk(sp, dp, r + "/", False)
                elif stat.S_ISREG(st.st_mode):
                    shutil.copyfile(sp, dp, follow_symlinks=False)
                    os.chmod(dp, stat.S_IMODE(st.st_mode))
                else:
                    left_out.append(r)
            except OSError:
                left_out.append(r)

    walk(Path(src), Path(dst), "", True)
    return left_out


def _exported_channels(episodes: Path) -> list[str]:
    """The distinct memory channels (non-null ``memory_channels``) of the actions :func:`export_for_sol` wrote.

    Read back from the host's own export before Sol's first cell, so it is exactly what Sol is shown.
    """
    out: set[str] = set()
    for f in sorted(episodes.glob("*.json")) if episodes.is_dir() else []:
        row = json.loads(f.read_text())
        out.update(c for c in row.get("memory_channels", []) if isinstance(c, str))
    return sorted(out)


def _channel_dirs(box: Path, channels: Iterable[str]) -> None:
    """Create an empty ``env/<channel>/`` in *box* for each well-formed channel; never a file, never via a link.

    Git tracks no empty directory, so a folder left empty is neither committed nor seen by the gate.
    """
    for ch in channels:
        if not _manifest.SKELETON_ID.match(f"env/{ch}"):
            continue
        try:
            for d in (box / "env", box / "env" / ch):
                try:
                    st = os.lstat(d)
                except FileNotFoundError:
                    os.mkdir(d, 0o755)
                    continue
                if not stat.S_ISDIR(st.st_mode):
                    break  # a file or link already holds the name: leave it to the gate
        except OSError:
            continue


def _check_manifest(raw: object) -> tuple[object, str | None]:
    """A check call's manifest argument (JSON text, or an object) parsed and bounded, or (None, why)."""
    if isinstance(raw, str):
        if len(raw.encode("utf-8", errors="replace")) > _MANIFEST_MAX_BYTES:
            return None, f"manifest: larger than {_MANIFEST_MAX_BYTES} bytes"
        try:
            raw = json.loads(raw)
        except (ValueError, RecursionError) as exc:
            return (
                None,
                f"manifest: not valid JSON ({type(exc).__name__}: {exc})"[
                    :_CHECK_REASON_CHARS
                ],
            )
    elif not isinstance(raw, dict):
        return None, "manifest: pass the manifest JSON as a string"
    if _depth(raw) > _MANIFEST_MAX_DEPTH:
        return None, f"manifest: nested deeper than {_MANIFEST_MAX_DEPTH} levels"
    return raw, None


def _check_reply(reasons: list[str]) -> str:
    """``ok``, or the gate's reasons, one per line, each redacted and at most 200 characters."""
    if not reasons:
        return "ok"
    lines = [_redact(r)[:_CHECK_REASON_CHARS] for r in reasons[:_CHECK_REASONS]]
    if len(reasons) > _CHECK_REASONS:
        lines.append(f"(and {len(reasons) - _CHECK_REASONS} more)")
    return "\n".join(lines)


def _clear_checkout(wt: Path) -> None:
    for e in os.scandir(wt):
        if e.name == ".git":
            continue
        if e.is_dir(follow_symlinks=False):
            shutil.rmtree(e.path)
        else:
            os.unlink(e.path)


def _read_manifest(box: Path) -> tuple[bool, object, str | None]:
    """(found, parsed manifest or None, problem). The file is never followed through a link.

    ``.pass`` must be a real directory and ``manifest.json`` a regular file of at most 1 MiB holding JSON
    that parses (a nesting too deep to parse is a problem, not a crash).
    """
    pass_dir = box / ".pass"
    try:
        st = os.lstat(pass_dir)
    except FileNotFoundError:
        return False, None, None
    except OSError as exc:  # /memory itself made unreadable
        return True, None, f"manifest: /memory/.pass cannot be read ({exc.strerror})"
    if not stat.S_ISDIR(st.st_mode):
        return (
            True,
            None,
            "manifest: /memory/.pass is not a directory, so no regular file was read",
        )
    try:
        fd = os.open(
            pass_dir / "manifest.json",
            os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK | os.O_CLOEXEC,
        )
    except FileNotFoundError:
        return False, None, None
    except OSError as exc:
        return (
            True,
            None,
            f"manifest: /memory/.pass/manifest.json is not a regular file that can be read ({exc.strerror})",
        )
    try:
        fst = os.fstat(fd)
        if not stat.S_ISREG(fst.st_mode):
            return (
                True,
                None,
                "manifest: /memory/.pass/manifest.json is not a regular file",
            )
        if fst.st_size > _MANIFEST_MAX_BYTES:
            return True, None, f"manifest: larger than {_MANIFEST_MAX_BYTES} bytes"
        data = os.read(fd, _MANIFEST_MAX_BYTES + 1)
    finally:
        os.close(fd)
    try:
        parsed = json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, ValueError, RecursionError) as exc:
        return (
            True,
            None,
            f"manifest: not valid JSON ({type(exc).__name__}: {exc})"[:300],
        )
    if _depth(parsed) > _MANIFEST_MAX_DEPTH:  # the gate walks it recursively
        return True, None, f"manifest: nested deeper than {_MANIFEST_MAX_DEPTH} levels"
    return True, parsed, None


def _depth(value: object) -> int:
    """Nesting depth of parsed JSON, without recursion."""
    deepest, stack = 0, [(value, 1)]
    while stack:
        v, d = stack.pop()
        deepest = max(deepest, d)
        if isinstance(v, dict):
            stack.extend((x, d + 1) for x in v.values())
        elif isinstance(v, list):
            stack.extend((x, d + 1) for x in v)
    return deepest


def _parse_args(raw: object) -> tuple[dict | None, str | None]:
    """Tool-call arguments as an object, or (None, why): malformed or too deeply nested JSON is refused."""
    try:
        args = json.loads(raw or "{}") if isinstance(raw, (str, type(None))) else raw
    except (ValueError, RecursionError) as exc:
        return None, f"unreadable arguments: {type(exc).__name__}: {exc}"[:300]
    if not isinstance(args, dict):
        return None, "unreadable arguments: not a JSON object"
    return args, None


def _remove(path: Path) -> None:
    """Remove *path* (a link itself, never its target); what cannot be removed is never mirrored anyway."""
    try:
        st = os.lstat(path)
        if stat.S_ISDIR(st.st_mode):
            shutil.rmtree(path)
        else:
            os.unlink(path)
    except OSError:
        pass


def _sources(manifest: object) -> list[str]:
    """Source episodes named by the manifest's items; ids that are not plain episode ids are left out."""
    out: set[str] = set()
    items = manifest.get("items") if isinstance(manifest, dict) else None
    for it in items if isinstance(items, list) else []:
        eids = it.get("source_episodes") if isinstance(it, dict) else None
        for e in eids if isinstance(eids, list) else []:
            if isinstance(e, str) and _EPISODE_ID.match(e):
                out.add(e)
    return sorted(out)


def _money(usd: object) -> Decimal | None:
    """A finite, non-negative decimal string as a Decimal; anything else (``unknown``) is None."""
    if not isinstance(usd, str):
        return None
    try:
        value = Decimal(usd)
    except InvalidOperation:
        return None
    return value if value.is_finite() and value >= 0 else None


def _redact(text: str) -> str:
    return redact_error(text)


def _usd(value: Decimal) -> str:
    """Money as a plain decimal string: never exponent notation (``str(Decimal("1E-7"))`` is ``1E-7``)."""
    return format(value, "f")


def _unpriced(n: int) -> list[str]:
    return [f"note: {n} unpriced calls"] if n else []


# --- the pass ----------------------------------------------------------------------------------------------


@dataclass
class _Spend:
    usd: Decimal = Decimal("0")
    unknown: int = 0


class SolPass:
    """One consolidation pass, bounded by ``max_calls``, ``max_usd`` (a Decimal) and ``deadline_s``.

    Each model call's reported cost is added to the spend; an unknown (or unreadable) cost is counted and
    reserves ``max_usd / max_calls`` against the cap, so unpriced calls cannot run away. Money is a pure
    decimal string (ruling R22, never exponent notation): the recorded and returned USD is the known spend
    only, and the count of unpriced calls is recorded as the reason ``note: N unpriced calls`` and in
    :attr:`PassOutcome.unknown_cost_calls`. The pass's own notes follow the gate's reasons in the row.

    Sol sees the episodes through :func:`export_for_sol` over *load* (ruling R10 is enforced here, not by
    the caller). A pass that ends with an exception, cancellation included, is recorded as failed.
    """

    def __init__(
        self,
        memory: Repo,
        gate: Gate,
        evidence: EvidenceStore,
        load: Callable[[str], Episode],
        model_turn: ModelTurn,
        config: PassConfig,
    ) -> None:
        self.mem, self.gate, self.ev = memory, gate, evidence
        self.load, self.turn, self.cfg = load, model_turn, config

    def _stage_inputs(self, req: PassRequest, inputs: Path) -> None:
        inputs.mkdir()
        export_for_sol(self.load, list(req.episodes), inputs / "episodes")
        export_blobs(
            self.load,
            list(req.episodes),
            getattr(self.gate, "blobs", None),
            inputs / "blobs",
        )
        (inputs / "request.json").write_text(
            json.dumps(
                {
                    "kind": req.kind,
                    "channel": req.channel,
                    "episodes": list(req.episodes),
                    "lift": req.lift,
                },
            ),
        )
        _stage_memlab(inputs / "memlab")

    def _cell(
        self,
        code: str,
        box: Path,
        inputs: Path,
        cells: Path,
        timeout_s: float,
    ) -> str:
        """Run *code* as ``/cell/code.py`` (a fresh read-only bind; never argv) under a lowered file-size limit."""
        cdir = Path(tempfile.mkdtemp(prefix="cell-", dir=cells))
        try:
            (cdir / "code.py").write_bytes(code.encode("utf-8", errors="replace"))
            r = run_confined(
                [
                    PRLIMIT,
                    f"--fsize={CELL_FSIZE_BYTES}:{CELL_FSIZE_BYTES}",
                    "--",
                    str(PYTHON),
                    "/cell/code.py",
                ],
                ro={inputs: "/inputs", cdir: "/cell"},
                rw={box: "/memory"},
                cwd="/memory",
                timeout_s=timeout_s,
                env={"PYTHONPATH": "/inputs:/memory"},
            )
        finally:
            shutil.rmtree(cdir, ignore_errors=True)
        content = (r.stdout + ("\n" + r.stderr if r.stderr else "")) or "(no output)"
        content = content[-_OUTPUT_CAP:]
        if r.timed_out:
            content += f"\n(cell timed out after {timeout_s:.0f} s)"
        elif r.returncode != 0:
            content += f"\n(exit status {r.returncode})"
        return content

    def _check(
        self,
        raw: object,
        box: Path,
        base: ParentSnapshot | str,
    ) -> tuple[str, str | None]:
        """The ``check`` tool: (reply, host-side detail of an error). Runs on the loop's thread.

        :meth:`.gate.Gate.preview` of a manifest against a copy of *box*: the copy is what the commit would
        hold (:func:`_mirror` without ``.pass``), so the box is only read. Called between cells, when no box
        runs, after :func:`_measure` made every entry readable. It stays on the event loop's thread, as
        :meth:`.gate.Gate.merge` does, because the evidence store's connection belongs to that thread. An
        error is answered with its type only (its text can name host paths) and returned as the detail.
        """
        manifest, problem = _check_manifest(raw)
        if problem is not None:
            return problem, None
        try:
            with tempfile.TemporaryDirectory(prefix="memv2-check-") as tmp:
                tree = Path(tmp) / "tree"
                tree.mkdir()
                _mirror(box, tree, skip_top=frozenset({".pass"}))
                reasons = self.gate.preview(base, tree, manifest)
        except Exception as exc:  # a broken check never ends the pass
            detail = _redact(f"check error: {type(exc).__name__}: {exc}")[:300]
            return f"check error: {type(exc).__name__}", detail
        return _check_reply(reasons), None

    async def _run_cell(self, *args) -> str:
        """The confined cell on a worker thread; on cancellation, wait for the box before the pass cleans up."""
        fut = asyncio.ensure_future(asyncio.to_thread(self._cell, *args))
        try:
            return await asyncio.shield(fut)
        except asyncio.CancelledError:
            await asyncio.wait([fut])  # bounded by the cell's own timeout
            raise

    def _fail(
        self,
        req: PassRequest,
        pass_id: str,
        parent: str,
        spend: _Spend,
        reasons: list[str],
    ) -> list[str]:
        reasons = reasons + _unpriced(spend.unknown)
        self.ev.record_pass(
            {
                "pass_id": pass_id,
                "kind": req.kind,
                "channel": req.channel,
                "parent": parent,
                "candidate": None,
                "passed": 0,
                "reasons": json.dumps(reasons),
                "usd": _usd(spend.usd),
                "patch_blob": None,
            },
        )
        return reasons

    async def run(self, req: PassRequest, pass_id: str) -> PassOutcome:
        if self.ev.pass_exists(pass_id):
            # never overwrite an earlier attempt's record, and spend nothing on a pass the gate would refuse
            return PassOutcome(
                pass_id,
                False,
                None,
                "0",
                0,
                [f"pass {pass_id} is already recorded"],
            )
        parent = self.mem.head()
        spend = _Spend()
        try:
            return await self._run(req, pass_id, parent, spend)
        except BaseException as exc:  # cancellation and interrupts included
            if not self.ev.pass_exists(pass_id):  # the gate records its own errors
                reason = _redact(f"pass error: {type(exc).__name__}: {exc}")[:500]
                self._fail(req, pass_id, parent, spend, [reason])
            raise

    async def _run(
        self,
        req: PassRequest,
        pass_id: str,
        parent: str,
        spend: _Spend,
    ) -> PassOutcome:
        cap, max_calls = Decimal(self.cfg.max_usd), int(self.cfg.max_calls)
        reserve = cap / max_calls if max_calls > 0 else cap
        deadline = time.monotonic() + float(self.cfg.deadline_s)
        calls, checks, summary = 0, 0, ""
        base: ParentSnapshot | str | None = (
            None  # the parent, taken once for every check
        )
        notes: list[str] = []
        stop: str | None = None  # why no further cell runs: deadline or quota
        over_quota: str | None = None
        causes: list[str] = []  # PassOutcome.codes, set where each cause arises
        errored = False

        def cause(code: str) -> None:
            if code not in causes:
                causes.append(code)

        def remaining() -> float:
            return deadline - time.monotonic()

        def outcome(
            passed: bool,
            commit: str | None,
            reasons: list[str],
        ) -> PassOutcome:
            return PassOutcome(
                pass_id,
                passed,
                commit,
                _usd(spend.usd),
                calls,
                reasons,
                summary,
                spend.unknown,
                [CODE_OK] if passed else list(causes),
                checks,
            )

        with (
            self.mem.temp_checkout() as wt,
            tempfile.TemporaryDirectory(
                prefix="memv2-sol-",
            ) as tmp,
        ):
            box, inputs, cells = (Path(tmp) / n for n in ("memory", "inputs", "cells"))
            box.mkdir()
            cells.mkdir()
            _mirror(wt, box)
            self._stage_inputs(req, inputs)
            _channel_dirs(box, _exported_channels(inputs / "episodes"))
            try:
                index = build_index(wt)
            except ValueError as exc:  # over budget, or an unreadable notes file
                index = f"(index not built: {exc})"
            messages: list[dict] = [
                {"role": "system", "content": SOL_SYSTEM},
                {
                    "role": "user",
                    "content": f"Pass {pass_id}: {json.dumps(req.__dict__)}\n\nCurrent index:\n{index}",
                },
            ]
            finished = False
            while (
                not finished
                and stop is None
                and calls < max_calls
                and spend.usd + spend.unknown * reserve < cap
            ):
                if remaining() <= 0:
                    stop = f"pass deadline of {self.cfg.deadline_s} s reached"
                    cause(CODE_DEADLINE)
                    break
                calls += 1
                try:
                    msg, usd = await asyncio.wait_for(
                        self.turn(messages, _TOOLS),
                        timeout=remaining(),
                    )
                except TimeoutError:
                    spend.unknown += 1
                    stop = f"pass deadline of {self.cfg.deadline_s} s reached during a model call"
                    cause(CODE_DEADLINE)
                    break
                except Exception as exc:  # the call may still have cost money
                    spend.unknown += 1
                    errored = True
                    failed_as = (
                        CODE_ROUTE_NOT_IN_EFFECT
                        if isinstance(exc, SolRouteError)
                        else CODE_SOL_ERROR
                    )
                    cause(failed_as)
                    notes.append(
                        _redact(f"model call failed: {type(exc).__name__}: {exc}")[
                            :300
                        ],
                    )
                    break
                cost = _money(usd)
                if cost is None:
                    spend.unknown += 1
                else:
                    spend.usd += cost
                if not isinstance(msg, dict):
                    notes.append(f"model returned {type(msg).__name__}, not a message")
                    errored = True
                    cause(CODE_SOL_ERROR)
                    break
                messages.append(msg)
                tool_calls = msg.get("tool_calls") or []
                tool_calls = tool_calls if isinstance(tool_calls, list) else []
                ran = 0
                for n, tc in enumerate(tool_calls):
                    if not isinstance(tc, dict):
                        messages.append(
                            {
                                "role": "tool",
                                "tool_call_id": f"call_{calls}_{n}",
                                "content": "unreadable tool call",
                            },
                        )
                        continue
                    if not isinstance(tc.get("id"), str) or not tc["id"]:
                        tc["id"] = (
                            f"call_{calls}_{n}"  # the reply must name the call it answers
                        )
                    fn = tc.get("function")
                    fn = fn if isinstance(fn, dict) else {}
                    name = fn.get("name")
                    args, content = _parse_args(fn.get("arguments"))
                    if args is None:
                        pass  # content says why
                    elif finished:
                        content = "not run: pass finished"
                    elif name == "finish":
                        summary, finished = str(args.get("summary", "")), True
                        content = "ok"
                    elif name == "check":
                        if stop is not None:
                            content = f"not run: {stop}"
                        elif checks >= MAX_CHECKS:
                            content = (
                                f"not run: at most {MAX_CHECKS} check calls per pass"
                            )
                        elif calls >= max_calls:
                            content = "not run: the pass's call cap is reached"
                        elif remaining() < CHECK_FLOOR_S:
                            content = (
                                "not run: too little time left before the pass deadline"
                            )
                        else:
                            checks += 1
                            calls += 1  # a check is a call against max_calls
                            if base is None:
                                try:
                                    base = self.gate.parent_snapshot(
                                        parent,
                                        Path(tmp) / "check-parent",
                                    )
                                except (
                                    Exception
                                ):  # GitError, OSError: the preview retries
                                    base = None
                                base = base or parent
                            content, detail = self._check(
                                args.get("manifest"),
                                box,
                                base,
                            )
                            if detail is not None:
                                notes.append(detail)
                    elif name != "execute_code":
                        content = (
                            f"unknown tool {name!r}; use execute_code, check or finish"[
                                :300
                            ]
                        )
                    elif stop is not None:
                        content = f"not run: {stop}"
                    elif ran >= _MAX_CELLS_PER_TURN:
                        content = (
                            f"not run: at most {_MAX_CELLS_PER_TURN} cells per turn"
                        )
                    elif remaining() <= 0:
                        stop = f"pass deadline of {self.cfg.deadline_s} s reached"
                        cause(CODE_DEADLINE)
                        content = f"not run: {stop}"
                    else:
                        ran += 1
                        timeout = max(
                            1.0,
                            min(float(self.cfg.cell_timeout_s), remaining()),
                        )
                        content = await self._run_cell(
                            str(args.get("code", "")),
                            box,
                            inputs,
                            cells,
                            timeout,
                        )
                        over_quota = _measure(box)
                        if over_quota is not None:
                            stop = over_quota
                            content += f"\n({over_quota}; the pass is refused)"
                    messages.append(
                        {"role": "tool", "tool_call_id": tc["id"], "content": content},
                    )
                if not tool_calls:
                    messages.append(
                        {
                            "role": "user",
                            "content": "Use execute_code, check, or finish.",
                        },
                    )
            if (
                not finished
                and stop is None
                and not errored
                and (calls >= max_calls or spend.usd + spend.unknown * reserve >= cap)
            ):
                cause(CODE_PASS_CAP)  # the call cap or the USD cap ended the pass
            if stop is not None and over_quota is None:
                notes.append(stop)
            summary = _redact(summary)
            if over_quota is None:
                over_quota = _measure(
                    box,
                )  # also restores the host's access to every entry
            if over_quota is not None:
                # no mirror, no commit: the tree is not looked at further
                cause(CODE_OVER_QUOTA)
                return outcome(
                    False,
                    None,
                    self._fail(req, pass_id, parent, spend, notes + [over_quota]),
                )
            found, manifest, problem = _read_manifest(box)
            if not found:
                cause(CODE_NO_MANIFEST)
                return outcome(
                    False,
                    None,
                    self._fail(req, pass_id, parent, spend, notes + ["no manifest"]),
                )
            if problem is not None:
                notes.append(problem)
                cause(CODE_MANIFEST_INVALID)
            _remove(box / ".pass")
            _clear_checkout(wt)
            left_out = _mirror(box, wt, skip_top=frozenset({".pass"}))
            if left_out:
                notes.append(f"special or unreadable entries left out: {left_out[:5]}")
            sources = _sources(manifest)
            candidate = self.mem.commit_all(
                wt,
                f"consolidation pass {pass_id}: {summary[:200]}".replace("\0", ""),
                {"Pass": pass_id, "Episode": sources, "Evidence": sources},
            )
        # a manifest that could not be read (not a regular file, not JSON) reaches the gate as None: G1 refuses it
        res = self.gate.merge(
            parent,
            candidate,
            manifest,
            pass_id,
            req.kind,
            req.channel,
            _usd(spend.usd),
        )
        if getattr(res, "manifest_invalid", False):
            cause(CODE_MANIFEST_INVALID)
        for check in getattr(res, "refused", []):
            cause(check)  # the gate checks that refused: G1..G6
        # the pass's own notes (left-out entries, deadline, model failures) follow the gate's reasons
        tail = notes + _unpriced(spend.unknown)
        self.ev.add_pass_notes(pass_id, tail)
        return outcome(
            res.passed,
            candidate if res.passed else None,
            res.reasons + tail,
        )


# --- the real model ----------------------------------------------------------------------------------------


@dataclass(frozen=True)
class SolRoute:
    """Sol's own route (``UNIFY_MEMORY_V2_SOL_BASE_URL`` / ``UNIFY_MEMORY_V2_SOL_TOKEN``): an OpenAI-compatible
    base URL and its token. The token is a ``SecretStr``: its repr and str are masked.
    """

    base_url: str
    token: SecretStr


#: The route of the Sol call running in this context, read by unillm's per-call gateway lookup
#: (:func:`_install_sol_gateway`). Context-local: a call in any other task or thread never sees it.
_SOL_GATEWAY: contextvars.ContextVar[SolRoute | None] = contextvars.ContextVar(
    "memory_v2_sol_gateway",
    default=None,
)
_GATEWAY_LOCK = threading.Lock()


class SolRouteError(RuntimeError):
    """Sol's declared route was not in effect for a Sol call (before it was sent, or seen in its event)."""


class SolCallError(RuntimeError):
    """A Sol call on the declared route failed. Its text is the exception's class and a fixed category only:
    the original text (an echoed request header, say) never reaches pass notes, error rows or logs.
    """


_CLASS_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]{0,63}")


def _call_failure(exc: BaseException) -> tuple[str, str]:
    """``(class name, category)`` of a failed call, from its type and status code only, never its text."""
    name = type(exc).__name__
    name = name if _CLASS_NAME.fullmatch(name) else "Exception"
    status = getattr(exc, "status_code", None)
    lowered = name.lower()
    if (
        isinstance(status, int)
        and not isinstance(status, bool)
        and 100 <= status <= 599
    ):
        category = f"http_{status}"
    elif "timeout" in lowered:
        category = "timeout"
    elif "connect" in lowered:
        category = "connection"
    elif "ratelimit" in lowered:
        category = "rate_limit"
    elif "auth" in lowered or "permission" in lowered:
        category = "auth"
    else:
        category = "other"
    return name, category


def _check_route_in_effect(route: SolRoute) -> None:
    """Refuse before any call unless unillm's request preparation, in this context, sends to *route*.

    A pure call to the same function every attempt (retries included) prepares its transport with; it shows
    the lookup it reads is the wrapped one. The prepared credential is compared, never kept or shown.
    """
    from unillm.clients import uni_llm

    kw: dict = {"model": "openrouter/memory-v2-sol-route-check", "messages": []}
    uni_llm._prepare_provider_request_kw(kw=kw, provider="openrouter", stream=False)
    ok = (
        str(kw.get("api_base") or "").rstrip("/") == route.base_url
        and kw.get("api_key") == route.token.get_secret_value()
    )
    kw.clear()
    if not ok:
        raise SolRouteError(
            "Sol's declared route is not in effect; no Sol call was made",
        )


def _install_sol_gateway() -> None:
    """Make unillm's per-call gateway lookup return Sol's route inside a Sol call; idempotent.

    unillm sends an OpenRouter call to ``_llm_gateway()`` (read on every call, before the provider credential
    is filled in, which never overwrites a chosen ``api_key``), but its clients take no per-call
    ``api_base``/``api_key``: ``generate`` drops extra keyword arguments. Wrapping that lookup with a
    context variable gives the per-call route without touching the process environment, which subprocesses
    inherit. Outside a Sol call the wrapper returns what unillm's own lookup returns. Refuses (no call made)
    when the installed unillm has no such lookup.
    """
    from unillm.clients import uni_llm

    with _GATEWAY_LOCK:
        lookup = getattr(uni_llm, "_llm_gateway", None)
        if getattr(lookup, "_memory_v2_sol", False):
            return
        if not callable(lookup) or not callable(
            getattr(uni_llm, "_prepare_provider_request_kw", None),
        ):
            raise RuntimeError(
                "Sol's route is set, but the installed unillm has no per-call gateway lookup to apply it",
            )

        def gateway() -> tuple[str, str] | None:
            route = _SOL_GATEWAY.get()
            if route is not None:
                return route.base_url, route.token.get_secret_value()
            return lookup()

        gateway._memory_v2_sol = True  # type: ignore[attr-defined]
        gateway.__wrapped__ = lookup  # type: ignore[attr-defined]
        uni_llm._llm_gateway = gateway


def unillm_turn(
    model: str,
    effort: str,
    *,
    origin: str = "memory_v2.sol",
    route: SolRoute | None = None,
) -> ModelTurn:
    """The real model call: one ``generate`` per turn on a unillm client, priced from unillm's LLM events.

    * The model is passed explicitly, so ``UNIFY_REASONING_EFFORT`` and the assistant default do not
      override it; ``effort`` is fixed for the pass. unillm needs a ``model@provider`` endpoint: a bare id
      (``openai/gpt-6-sol``) gets ``@openrouter``, as the repo's tests write ``openai/gpt-6-luna@openrouter``.
    * ``generate(messages=..., stateful=True)`` replaces the client's history with the full message list
      (the system message included) and appends the assistant reply, which is read back from
      ``client.messages[-1]`` (an OpenAI-shaped dict with ``content`` and ``tool_calls``), as
      :func:`unify.common.single_shot.single_shot_tool_decision` does.
    * Cost: every ``LLMEvent`` the call emits (postprocessing retries emit their own) is caught by a
      context-local hook (``unillm.allm_event_hook_scope``), not a process-global
      ``add_llm_event_listener``, so concurrent calls elsewhere in the process are never billed to the pass.
      The turn's USD is the sum of ``provider_cost``; it is ``unknown`` when no event arrived or any event's
      cost is ``None`` (cache hits, errors), which :class:`SolPass` then reserves against its cap.
    * Credentials: unillm reads the provider key from the controller's environment. It never enters argv,
      logs, the box (whose environment is cleared) or anything this module writes.

    Verifying a real call (Task 13, the first paid pass, under ``PassConfig.max_usd``): with the key in the
    environment, run one pass on a recorded episode with a small ``max_calls``; check that the ``passes`` row
    holds a decimal USD string and no ``note: N unpriced calls`` reason, that it matches the provider's billed amount for origin
    ``memory_v2.sol``, and that the unillm log shows ``reasoning_effort`` as configured. No test makes this
    call.

    Sol's route (*route*, from ``UNIFY_MEMORY_V2_SOL_BASE_URL`` and ``UNIFY_MEMORY_V2_SOL_TOKEN``): ``None``
    sends the call exactly as above. Set, each call goes to ``route.base_url`` with ``route.token`` (through
    :func:`_install_sol_gateway`, for this call's context only) and carries ``X-Unify-Call-Kind: <origin>``;
    the endpoint must be an ``@openrouter`` one (the route replaces OpenRouter's transport). The scoped
    hook, which unillm calls before any process-wide listener, drops ``api_key`` from each event's request,
    so the token never reaches the event bus, and a call whose event shows another ``api_base`` fails the
    turn (its cost then counts as unknown). Before each call, a pure check of unillm's request preparation
    in this context refuses (no call made) unless it sends to the route; the same preparation serves every
    retry. A failed call on the route raises :class:`SolCallError` naming only the exception's class and a
    fixed category (``http_<status>``, ``timeout``, ``connection``, ``rate_limit``, ``auth``, ``other``), so
    no error text (an echoed header, say) reaches pass notes, error rows or logs. The pricing is unchanged:
    the request still asks OpenRouter's API for its charged cost, and a call the route does not price is
    ``unknown``.
    """
    import unillm

    from unify.common.llm_client import CALL_KIND_HEADER, new_llm_client

    endpoint = model if "@" in model else f"{model}@openrouter"
    if route is not None:
        if not endpoint.endswith("@openrouter"):
            raise ValueError(
                "Sol's route replaces the OpenRouter transport; the Sol model must be an @openrouter endpoint",
            )
        _install_sol_gateway()
    client = new_llm_client(
        endpoint,
        origin=origin,
        reasoning_effort=effort,
        stateful=True,
    )
    # only on the declared route, so the shipped call is unchanged
    routed: dict[str, Any] = (
        {} if route is None else {"extra_headers": {CALL_KIND_HEADER: origin}}
    )

    async def turn(messages: list[dict], tools: list[dict]) -> tuple[dict, str]:
        costs: list[float | None] = []
        misrouted: list[bool] = []

        def hook(event) -> None:
            if route is not None and isinstance(event.request, dict):
                event.request.pop("api_key", None)
                base = str(event.request.get("api_base") or "").rstrip("/")
                misrouted.append(base != route.base_url)
            if event.origin == origin:
                costs.append(event.provider_cost)

        failure: tuple[str, str] | None = None
        timed_out = False
        async with unillm.allm_event_hook_scope(hook):
            scope = _SOL_GATEWAY.set(route) if route is not None else None
            try:
                if route is not None:
                    _check_route_in_effect(route)
                await client.generate(
                    messages=copy.deepcopy(messages),
                    tools=tools,
                    tool_choice="auto",
                    stateful=True,
                    **routed,
                )
            except Exception as exc:
                if route is None or isinstance(exc, SolRouteError):
                    raise
                failure = _call_failure(exc)  # class and category only, never the text
                timed_out = isinstance(exc, TimeoutError)
            finally:
                if scope is not None:
                    _SOL_GATEWAY.reset(scope)
        if failure is not None:
            # raised outside the handler, so the original exception is not chained to it
            name, category = failure
            if (
                timed_out
            ):  # still a timeout to the pass, which treats it as its deadline
                raise TimeoutError("a Sol call timed out")
            raise SolCallError(f"Sol call failed: {name} ({category})")
        if any(misrouted):
            raise SolRouteError("a Sol call did not take Sol's declared route")
        msg = client.messages[-1]
        msg = msg if isinstance(msg, dict) else dict(msg)
        if not costs or any(c is None for c in costs):
            return msg, _UNKNOWN
        return msg, _usd(sum((Decimal(str(c)) for c in costs), Decimal("0")))

    return turn
