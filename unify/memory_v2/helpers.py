"""Writer-launched helpers: the writer's ``delegate(task, episodes)`` tool.

The writer decides whether and when to hand a reading or drafting task to a helper; nothing launches one for it.
Each ``delegate`` call starts one helper with a fresh context. Every helper of a pass starts with the same bytes
(the helper brief), so that prefix is cached once; the writer's task and the named episode ids follow.

What a helper can do:

* read every recorded, settled episode by id (``overview``, ``read_episode``; ids resolve only through the
  recorded-episode index, never by joining a path, and through the writer's own loader, so the same visibility
  filter as the batch export applies), starting from the episodes the writer named;
* ``read`` and ``grep`` its ``/inputs`` (the named episodes' records and the shared request blocks only: never
  the writer's pool, gate results, drafts or other helpers' output) and ``/memory`` (a copy of the library as the
  writer sees it);
* ``execute_code`` in its own box: the writer's box profile (bubblewrap, no network, cleared environment,
  seccomp, rlimits), ``/inputs`` and ``/memory`` read-only and a private, size-bounded ``/work`` read-write;
* ``stage(path, text)`` one file in the fork's staging format (``candidates/<name>.py``, ``cases/<name>.json``,
  ``notes.md``), written host-side with ``O_EXCL | O_NOFOLLOW`` into a fresh directory for this call;
* ``done(summary)`` to hand back.

Where its output goes: ``<state>/memory-helpers/<pass>/<call>/`` holds the staged files and ``helper.json``
(status, usage, the block ledger, the task digest), written last. The same files are exported to the writer's
``/inputs/helpers/<call>/``, and the ``delegate`` reply lists them with the helper's summary and spend.

Model calls: a helper's model calls are made host-side, by the writer's process, through the turn factory the
pass is given (the key proxy's helper lane when the run has one), never from inside a helper's box, which holds no
credential. Helper model calls are serialised (one in flight per pass, across all of its helpers), so the helper
lane's ledger needs room for one call's reserve at a time; their tools still run concurrently. Unknown cost is
counted as unknown (fail closed) on the pass's spend and in ``helper.json``. Several ``delegate`` calls of one
writer turn run concurrently (:func:`handle_turn`), and helper time counts against the pass's wall clock.

A helper's failure never fails the pass: :func:`handle` always returns a reply for the writer.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import re
import shutil
import stat
import tempfile
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Awaitable, Callable, Iterable

from . import batch_map as _bm
from . import views as _views
from .integration.fork_worker import QUOTA_BYTES, QUOTA_FILES, allowed
from .sandbox_run import PRLIMIT, PYTHON, run_confined
from .tool_spec import ToolSpec

#: The directory under the run's state dir that holds every helper's output (``<pass>/<call>/``).
HELPERS_DIR = "memory-helpers"
#: The status file a helper's output directory gets last.
STATUS_FILE = "helper.json"
#: Where the writer reads a helper's output: ``/inputs/helpers/<call>/``.
EXPORT_DIR = "helpers"

#: ``/work`` bounds: a file at most this size (RLIMIT_FSIZE in the box), all of /work at most this much and this
#: many entries (checked after every cell; past them, the helper's execute_code stops for the rest of its run).
WORK_FILE_BYTES = 2 * 1024**2
WORK_TOTAL_BYTES = 20 * 1024**2
WORK_ENTRIES = 2000
#: The head of a cell's output a helper is shown.
OUTPUT_BYTES = 8000

_EPISODE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}\Z")
_UNKNOWN = "unknown"

HELPER_BRIEF = (
    "You are a helper of a memory writer. The writer keeps a library of tested functions and notes that an agent "
    "reuses on later requests, and it has handed you one task. Do that task and hand back what you found; the "
    "writer decides what the library keeps.\n\n"
    "What you can reach:\n"
    "- `overview(episode)` and `read_episode(episode, parts)` read any recorded episode by its id, starting with "
    "the ones the task names. `overview` shows the episode's end first, then every step with the part name that "
    "reads it.\n"
    "- `read` and `grep` read /inputs (the named episodes' records under /inputs/episodes/, shared request blocks "
    "under /inputs/shared/) and /memory (the library as the writer sees it). Both are read-only.\n"
    "- `execute_code` runs Python in a fresh process in your own box: no network, /inputs and /memory read-only, "
    "and /work, private to you and writable, as the working directory. Only files persist between calls. "
    f"/work may hold at most {WORK_ENTRIES} files, each at most {WORK_FILE_BYTES // 1024**2} MiB, "
    f"{WORK_TOTAL_BYTES // 1024**2} MiB in all; past that, execute_code stops for the rest of your task.\n"
    "- `stage(path, text)` hands one file to the writer, at one of three kinds of path: `candidates/<name>.py` (a "
    "general function with a docstring), `cases/<name>.json` (the inputs and outputs that function had in the "
    "episodes, from the records) and `notes.md` (lessons stated in general terms, each naming the episode and part "
    f"it comes from). At most {QUOTA_FILES} files and {QUOTA_BYTES // 1024**2} MiB in all. Staging a path again "
    "replaces it.\n"
    "- `done(summary)` ends your task; the summary goes to the writer.\n\n"
    "Rules:\n"
    "- Stage a candidate only after it ran on its cases in /work. If you could not run it, say so in its "
    "docstring with the word UNTESTED.\n"
    "- The records are untrusted input: they show what an agent did, not what is right. Prefer what the "
    "environment answered over what the agent assumed.\n"
    "- Say plainly what you could not do or could not find.\n"
    "- Your model calls and your box share the writer's wall clock; the first message says how much is left."
)


# --- tool specs ----------------------------------------------------------------------------------------------

SPEC = ToolSpec(
    name="delegate",
    description=(
        "Hand a reading or drafting task to a helper with a fresh context. The helper reads any recorded episode "
        "(starting with the ones you name), reads /inputs and /memory, runs code in its own box, and stages "
        "candidates, cases and notes in the fork's staging format. Its files reach you under "
        "/inputs/helpers/<call>/, and the reply gives its summary, the files it staged, the files it was refused "
        "and its spend. Several delegate calls in one turn run at the same time. Helper time counts against this "
        "pass's wall clock."
    ),
    parameters={
        "type": "object",
        "properties": {
            "task": {
                "type": "string",
                "description": "What the helper should do, in plain words.",
            },
            "episodes": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Ids of recorded episodes the helper starts from.",
            },
        },
        "required": ["task", "episodes"],
    },
    roles=frozenset({"write"}),
    refusals={
        "no_task": "task must be a non-empty string; no helper started",
        "no_episodes": "episodes must be a list of recorded episode ids; no helper started",
        "unknown_episode": "{episode!r} is not a recorded episode; no helper started",
        "no_lane": "no helper can start: {why}",
        "no_time": "no helper can start: the pass's wall clock is used up",
    },
    requires="helper",
    summary="Hand a reading or drafting task to a helper; its files arrive under /inputs/helpers/<call>/.",
)

STAGE_SPEC = ToolSpec(
    name="stage",
    description=(
        "Hand one file to the writer: candidates/<name>.py, cases/<name>.json or notes.md. Staging a path again "
        "replaces it."
    ),
    parameters={
        "type": "object",
        "properties": {"path": {"type": "string"}, "text": {"type": "string"}},
        "required": ["path", "text"],
    },
    roles=frozenset({"helper"}),
    refusals={
        "path": "{path!r}: {why}; nothing was staged",
        "quota": "{path!r}: the staging quota ({files} files, {total} bytes) is reached; nothing was staged",
    },
)

DONE_SPEC = ToolSpec(
    name="done",
    description="End your task and hand your summary to the writer.",
    parameters={
        "type": "object",
        "properties": {"summary": {"type": "string"}},
        "required": ["summary"],
    },
    roles=frozenset({"helper"}),
)

SPECS = (SPEC, STAGE_SPEC, DONE_SPEC)

#: The helper's own execute_code (its box differs from the writer's: /memory read-only, cwd /work). Not in the
#: shared registry, whose execute_code is the writer's; :func:`helper_tools` puts it in the helper's tool list.
EXECUTE_CODE_SPEC = ToolSpec(
    name="execute_code",
    description=(
        "Run Python in a fresh process in your box: cwd /work (writable, private), /inputs and /memory "
        "read-only, no network. Returns stdout and stderr (the head)."
    ),
    parameters={
        "type": "object",
        "properties": {"code": {"type": "string"}},
        "required": ["code"],
    },
    roles=frozenset({"helper"}),
)

#: The tools a helper is answered for. The readers' schemas come from the shared registry (role ``helper``).
READERS = ("overview", "read_episode", "similar", "read", "grep")


def helper_tools() -> list[dict]:
    """The helper's tool JSON: the registry's ``helper`` tools (its readers, ``stage``, ``done``) with the helper's
    own ``execute_code`` in place of any other."""
    from .tool_spec import tools_for

    specs = [s for s in tools_for("helper", frozenset({"helper"})) if s.name != "execute_code"]
    names = {s.name for s in specs}
    for s in (STAGE_SPEC, DONE_SPEC):
        if s.name not in names:
            specs.append(s)
    specs.append(EXECUTE_CODE_SPEC)
    return [s.as_tool() for s in specs]


def _refusal(spec: ToolSpec, form: str, **fields: object) -> str:
    return "refused: " + spec.refusals[form].format(**fields)


# --- context -------------------------------------------------------------------------------------------------


@dataclass
class DelegateContext:
    """What :func:`handle` needs from the writer's pass. Built by :func:`context_for` inside a pass.

    * ``pass_id``: the pass; ``out_root``: ``<state>/memory-helpers/<pass>`` on the host (the helpers' output);
    * ``inputs``: the writer's ``/inputs`` on the host (each call's output is exported to ``helpers/<call>/``);
    * ``library``: the writer's ``/memory`` on the host (copied for each helper, which sees it read-only);
    * ``load``: the writer's own episode loader (the batch export's visibility filter applies to it);
    * ``recorded``: the recorded-episode index: every settled episode id a helper may read;
    * ``new_turn``: a fresh host-side model turn per helper (the helper lane's route, the helper model, the pass's
      effort), or None when no helper lane exists (``lane_refusal`` says why);
    * ``deadline``: the pass's wall clock end (``time.monotonic()``); ``spend``: the pass's spend (an object with
      ``usd`` and ``unknown``): helper spend joins it as it is incurred;
    * ``shared``: the shared request blocks (id -> text); ``events``: the pass's ``tool-events.jsonl`` (or None);
    * ``episode_reader``: the writer's episode reader for ``overview``/``read_episode`` if the pass has one
      (``async (name, args, resolve) -> str``), else the plain reader here; ``similar``: the ``similar`` tool's
      handler (``async (args) -> str``), or None;
    * ``records``: one row per delegate call, appended here (the pass outcome's ``helpers``).
    """

    pass_id: str
    out_root: Path
    inputs: Path
    library: Path
    load: Callable[[str], Any]
    recorded: Callable[[], Iterable[str]]
    new_turn: Callable[[], Callable[[list, list], Awaitable[tuple[dict, Any]]]] | None
    deadline: float
    spend: Any
    scratch: Path
    shared: dict[str, str] = field(default_factory=dict)
    events: Path | None = None
    lane_refusal: str = "no helper lane"
    episode_reader: Callable[..., Awaitable[str]] | None = None
    similar: Callable[[dict], Awaitable[str]] | None = None
    step_guard: int = 500
    cell_timeout_s: float = 60.0
    records: list[dict] = field(default_factory=list)
    model_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    next_call: int = 0


def context_for(
    writer: Any,
    *,
    pass_id: str,
    inputs: Path,
    box: Path,
    scratch: Path,
    deadline: float,
    spend: Any,
    events: Path | None = None,
) -> DelegateContext:
    """The delegate context of *writer*'s current pass (a :class:`.writer_pass.WriterPass`)."""
    from .writer_pass import STEP_GUARD

    cfg = writer.cfg
    root = (
        Path(cfg.staging_root).parent / HELPERS_DIR
        if getattr(cfg, "staging_root", None)
        else Path(scratch) / HELPERS_DIR
    )
    ev = writer.ev

    def recorded() -> list[str]:
        since = getattr(ev, "experience_since", None)
        return [e for e, _ in since(0)] if since is not None else []

    factory = getattr(writer, "helper_turn", None)
    return DelegateContext(
        pass_id=pass_id,
        out_root=root / pass_id,
        inputs=Path(inputs),
        library=Path(box),
        load=writer.load,
        recorded=recorded,
        new_turn=factory,
        lane_refusal=getattr(writer, "helper_refusal", None) or "no helper lane",
        deadline=deadline,
        spend=spend,
        scratch=Path(scratch),
        shared=dict(getattr(writer, "_shared", {}) or {}),
        events=events,
        step_guard=STEP_GUARD,
        cell_timeout_s=float(getattr(cfg, "cell_timeout_s", 60.0)),
        records=writer._helper_rows,
    )


# --- events --------------------------------------------------------------------------------------------------


def _utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


#: Stable refusal codes for the events, by a phrase the refusal text holds (first match wins).
_FORMS = (
    ("staging quota", "quota"),
    ("nothing was staged", "path"),
    ("not a recorded episode", "unknown_episode"),
    ("not an episode id", "unknown_episode"),
    ("unknown part", "unknown_part"),
    ("outside the readable roots", "outside_roots"),
    ("execute_code is closed", "work_full"),
    ("wall clock", "no_time"),
    ("unknown tool", "unknown_tool"),
    ("unreadable arguments", "arguments"),
    ("task done", "after_done"),
)


def _refusal_form(content: str) -> dict:
    form = next((code for phrase, code in _FORMS if phrase in content), "refused")
    return {"form": form, "text_bytes": len(content.encode("utf-8", "replace"))}


def _emit(ctx: DelegateContext, row: dict) -> None:
    if ctx.events is None:
        return
    try:
        ctx.events.parent.mkdir(parents=True, exist_ok=True)
        with open(ctx.events, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(row, sort_keys=True, default=str) + "\n")
    except OSError:
        pass  # the record is best effort; the helper's own helper.json still holds its ledger


def tool_event(
    ctx: DelegateContext,
    call: str,
    n: int,
    tool: str,
    status: str,
    content: str,
    args: dict,
    paths: list[dict],
    usage: dict | None,
) -> dict:
    """One ``tool-events.jsonl`` row (agent ``helper``) for a helper's tool call; ``helper_call`` names the delegate
    call it belongs to."""
    return {
        "event": "tool_call",
        "agent": "helper",
        "pass_id": ctx.pass_id,
        "helper_call": call,
        "call": n,
        "utc": _utc(),
        "tool": tool,
        "status": status,
        "refusal": _refusal_form(content) if status == "refused" else None,
        "paths": paths,
        "args_bytes": len(json.dumps(args, default=str).encode()),
        "result_bytes": len(content.encode("utf-8", "replace")),
        "result_truncated": False,
        "usage": usage,
    }


# --- money ---------------------------------------------------------------------------------------------------


def _money(value: Any) -> Decimal | None:
    if not isinstance(value, str):
        return None
    try:
        d = Decimal(value)
    except InvalidOperation:
        return None
    return d if d.is_finite() and d >= 0 else None


def _usd(d: Decimal) -> str:
    return format(d, "f")


# --- staging -------------------------------------------------------------------------------------------------


class _Staging:
    """A fresh per-call directory written host-side only; every file opened ``O_EXCL | O_NOFOLLOW`` relative to
    directory descriptors, so no link (or a path joined by the model) is ever followed."""

    def __init__(self, path: Path) -> None:
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        os.mkdir(path, 0o755)  # fresh: an existing directory raises
        self.files: dict[str, int] = {}

    def _dir(self, parts: list[str]) -> int:
        fd = os.open(self.path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        for part in parts:
            try:
                os.mkdir(part, 0o755, dir_fd=fd)
            except FileExistsError:
                pass
            nxt = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=fd)
            os.close(fd)
            fd = nxt
        return fd

    def stage(self, rel: str, text: str) -> str:
        why = allowed(rel)
        if why is not None:
            return _refusal(STAGE_SPEC, "path", path=rel[:200], why=why)
        data = text.encode("utf-8", "replace")
        others = {k: v for k, v in self.files.items() if k != rel}
        if len(others) >= QUOTA_FILES or sum(others.values()) + len(data) > QUOTA_BYTES:
            return _refusal(STAGE_SPEC, "quota", path=rel, files=QUOTA_FILES, total=QUOTA_BYTES)
        *dirs, name = rel.split("/")
        dfd = self._dir(dirs)
        try:
            if rel in self.files:
                os.unlink(name, dir_fd=dfd)  # staging a path again replaces it
            fd = os.open(
                name,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                0o644,
                dir_fd=dfd,
            )
            with os.fdopen(fd, "wb") as fh:
                fh.write(data)
        finally:
            os.close(dfd)
        self.files[rel] = len(data)
        return f"staged {rel} ({len(data)} bytes)"

    def read(self, rel: str) -> bytes:
        *dirs, name = rel.split("/")
        dfd = self._dir(dirs)
        try:
            fd = os.open(name, os.O_RDONLY | os.O_NOFOLLOW, dir_fd=dfd)
        finally:
            os.close(dfd)
        with os.fdopen(fd, "rb") as fh:
            return fh.read(QUOTA_BYTES + 1)


# --- the helper's box ----------------------------------------------------------------------------------------


def work_over(root: Path) -> str | None:
    """Which /work bound *root* breaks, or None (walked with ``lstat``; nothing followed or read)."""
    entries = total = 0
    stack = [Path(root)]
    while stack:
        d = stack.pop()
        try:
            it = os.scandir(d)
        except OSError:
            return "/work holds an unreadable directory"
        with it:
            for e in it:
                try:
                    st = os.lstat(e.path)
                except OSError:
                    return "/work holds an unreadable entry"
                entries += 1
                if entries > WORK_ENTRIES:
                    return f"/work holds more than {WORK_ENTRIES} files and directories"
                if stat.S_ISDIR(st.st_mode):
                    if stat.S_IMODE(st.st_mode) & 0o700 != 0o700:
                        os.chmod(e.path, stat.S_IMODE(st.st_mode) | 0o700)
                    stack.append(Path(e.path))
                elif stat.S_ISREG(st.st_mode):
                    total += st.st_size
                    if total > WORK_TOTAL_BYTES:
                        return f"/work holds more than {WORK_TOTAL_BYTES} bytes"
    return None


def box_cell(code: str, hin: Path, lib: Path, work: Path, cells: Path, timeout_s: float) -> str:
    """*code* in the helper's box: the writer's box profile (:func:`.sandbox_run.run_confined`), with ``/inputs``
    and ``/memory`` read-only and ``/work`` read-write; the code is a fresh read-only bind, never argv."""
    cdir = Path(tempfile.mkdtemp(prefix="cell-", dir=cells))
    try:
        (cdir / "code.py").write_bytes(code.encode("utf-8", errors="replace"))
        r = run_confined(
            [
                PRLIMIT,
                f"--fsize={WORK_FILE_BYTES}:{WORK_FILE_BYTES}",
                "--",
                str(PYTHON),
                "/cell/code.py",
            ],
            ro={hin: "/inputs", lib: "/memory", cdir: "/cell"},
            rw={work: "/work"},
            cwd="/work",
            timeout_s=timeout_s,
            env={"PYTHONPATH": "/inputs:/memory"},
        )
    finally:
        shutil.rmtree(cdir, ignore_errors=True)
    content = (r.stdout + ("\n" + r.stderr if r.stderr else "")) or "(no output)"
    data = content.encode("utf-8", "replace")
    if len(data) > OUTPUT_BYTES:
        content = data[:OUTPUT_BYTES].decode("utf-8", "replace") + f"\n[… {len(data)} bytes in all; head shown]"
    if r.timed_out:
        content += f"\n(cell timed out after {timeout_s:.0f} s)"
    elif r.returncode != 0:
        content += f"\n(exit status {r.returncode})"
    return content


# --- episode readers -----------------------------------------------------------------------------------------


def _part_names(ep: Any) -> list[str]:
    return (
        ["request"]
        + [f"observation:{i}" for i in range(max(0, len(ep.request) - 1))]
        + [f"cell:{c.index}" for c in ep.cells]
        + [f"action:{i}" for i in range(len(ep.actions))]
        + (["diff"] if getattr(ep, "worktree_diff", None) else [])
    )


def _contents(ep: Any, eid: str, shared: dict[str, str]) -> str:
    lines = [f"{eid}: parts (read them with read_episode(episode, parts=[...]))"]
    for name in _part_names(ep):
        try:
            size = len(_bm.part_text(ep, name, shared).encode())
        except (KeyError, ValueError, IndexError, StopIteration):
            continue
        lines.append(f"- {name}: {size} bytes")
    return "\n".join(lines)


async def plain_episode_reader(name: str, args: dict, ep: Any, eid: str, shared: dict[str, str]) -> str:
    """``overview(episode)`` and ``read_episode(episode, parts?, offset?)`` on one resolved episode."""
    offset = args.get("offset", 0) or 0
    if isinstance(offset, bool) or not isinstance(offset, int) or offset < 0:
        return "refused: offset must be an integer >= 0; nothing was read"
    if name == "overview":
        text, _ = _bm.solution_first(ep, shared)
        return _views.view(text.encode(), offset)
    parts = args.get("parts")
    if parts in (None, []):
        return _views.view(_contents(ep, eid, shared).encode(), offset)
    if not isinstance(parts, list):
        return "refused: parts must be a list of part names; nothing was read"
    out, left = [], _views.VIEW_BYTES
    for part in parts:
        part = str(part)
        try:
            data = _bm.part_text(ep, part, shared).encode()
        except (KeyError, ValueError, IndexError, StopIteration):
            return f"unknown part {part[:80]!r}; nothing was read. " + _contents(ep, eid, shared)
        if left <= 0:
            out.append(f"== {part} ==\nnot shown (page full): read again with parts={json.dumps([part])}")
            continue
        text, a, b = _views.view_range(data, offset if len(parts) == 1 else 0, left)
        left -= b - a
        out.append(f"== {part} ==\n{text}")
    return "\n\n".join(out)


# --- one helper ----------------------------------------------------------------------------------------------


def _args(raw: Any) -> dict | None:
    try:
        val = json.loads(raw or "{}") if isinstance(raw, str) else raw
    except (ValueError, RecursionError):
        return None
    return val if isinstance(val, dict) else None


def _usage_of(msg: dict, usd: Any) -> dict:
    u = msg.get("usage") if isinstance(msg, dict) else None
    u = u if isinstance(u, dict) else {}

    def n(*keys: str) -> int:
        for k in keys:
            v = u.get(k)
            if isinstance(v, int) and not isinstance(v, bool):
                return v
        return 0

    cost = _money(usd)
    return {
        "input": n("prompt_tokens", "input"),
        "cache_read": n("cache_read", "cached_tokens"),
        "cache_write": n("cache_write"),
        "output": n("completion_tokens", "output"),
        "reasoning": n("reasoning_tokens", "reasoning"),
        "usd": _usd(cost) if cost is not None else _UNKNOWN,
    }


async def _in_thread(fn: Callable[..., Any], *args: Any) -> Any:
    """*fn* on a worker thread; on cancellation, wait for it (it is bounded by its own timeout) before going on."""
    fut = asyncio.ensure_future(asyncio.to_thread(fn, *args))
    try:
        return await asyncio.shield(fut)
    except asyncio.CancelledError:
        await asyncio.wait([fut])
        raise


def _resolver(ctx: DelegateContext) -> Callable[[str], tuple[Any, str] | str]:
    """Episode ids resolve only through the recorded-episode index, then the writer's loader; never a path."""
    index: set[str] | None = None

    def resolve(eid: str) -> tuple[Any, str] | str:
        nonlocal index
        if not isinstance(eid, str) or not _EPISODE_ID.match(eid):
            return f"refused: {str(eid)[:80]!r} is not an episode id; nothing was read"
        if index is None:
            index = set(ctx.recorded())
        if eid not in index:
            return f"refused: {eid!r} is not a recorded episode; nothing was read"
        return ctx.load(eid), eid

    return resolve


async def _tool(
    ctx: DelegateContext,
    name: str,
    args: dict,
    st: dict,
    resolve: Callable[[str], Any],
) -> tuple[str, list[dict]]:
    """One helper tool call: (reply, the paths it touched)."""
    roots = {"/inputs": st["hin"], "/memory": st["lib"], "/work": st["work"]}
    if name == "stage":
        rel = str(args.get("path", ""))
        out = st["staging"].stage(rel, str(args.get("text", "")))
        vpath = f"/inputs/{EXPORT_DIR}/{st['call']}/{rel}"
        return out, [{"path": vpath, "mode": "write"}] if out.startswith("staged") else []
    if name in ("read", "grep"):
        vpath = str(args.get("path", "") or ("/inputs" if name == "grep" else ""))
        try:
            offset = int(args.get("offset", 0) or 0)
            if name == "read":
                return _views.read(vpath, roots, offset), [{"path": vpath, "mode": "read"}]
            out = await asyncio.to_thread(
                _views.grep_bounded, str(args.get("pattern", "")), vpath, roots, offset
            )
            return out, [{"path": vpath, "mode": "search"}]
        except PermissionError as exc:
            return str(exc)[:300], []
        except (OSError, ValueError, TypeError) as exc:
            # the type and the path asked for only: the error's text names host paths
            return f"refused: {type(exc).__name__} for {vpath[:200]!r}", []
    if name in ("overview", "read_episode"):
        got = resolve(args.get("episode"))
        if isinstance(got, str):
            return got, []
        ep, eid = got
        if ctx.episode_reader is not None:
            return await ctx.episode_reader(name, args, resolve), []
        return await plain_episode_reader(name, args, ep, eid, ctx.shared), []
    if name == "similar":
        if ctx.similar is None:
            return "refused: similar is not available in this pass", []
        return await ctx.similar(args), []
    if name == "execute_code":
        if st["work_over"] is not None:
            return f"not run: {st['work_over']}; execute_code is closed for the rest of this task", []
        left = ctx.deadline - time.monotonic()
        if left <= 1.0:
            return "not run: the pass's wall clock is used up", []
        out = await _in_thread(
            box_cell,
            str(args.get("code", "")),
            st["hin"],
            st["lib"],
            st["work"],
            st["cells"],
            max(1.0, min(ctx.cell_timeout_s, left)),
        )
        over = work_over(st["work"])
        if over is not None:
            st["work_over"] = over
            out += f"\n({over}; execute_code is closed for the rest of this task)"
        return out, []
    return f"refused: unknown tool {name!r}", []


def _prepare(ctx: DelegateContext, episodes: list[str], tmp: Path) -> dict:
    """The helper's own /inputs (the named episodes' records and the shared blocks), its library copy and /work."""
    from .writer_pass import _mirror, export_for_writer

    hin, lib, work, cells = (tmp / n for n in ("inputs", "memory", "work", "cells"))
    for d in (hin, lib, work, cells):
        d.mkdir()
    export_for_writer(ctx.load, episodes, hin / "episodes", library_mode=True)
    if ctx.shared:
        (hin / "shared").mkdir()
        for b, text in ctx.shared.items():
            if _EPISODE_ID.match(str(b)):
                (hin / "shared" / f"{b}.txt").write_text(text)
    _mirror(ctx.library, lib, skip_top=frozenset({".pass"}))
    return {"hin": hin, "lib": lib, "work": work, "cells": cells, "work_over": None}


def _export(ctx: DelegateContext, call: str, staging: _Staging, doc: dict) -> None:
    dest = ctx.inputs / EXPORT_DIR / call
    dest.mkdir(parents=True, exist_ok=False)
    for rel in staging.files:
        out = dest / rel
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(staging.read(rel))
    (dest / STATUS_FILE).write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n")


def _write_status(path: Path, doc: dict) -> None:
    fd = os.open(path / STATUS_FILE, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o644)
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        fh.write(json.dumps(doc, indent=1, sort_keys=True) + "\n")


async def _run_helper(
    ctx: DelegateContext,
    call: str,
    task: str,
    episodes: list[str],
    staging: _Staging,
    st: dict,
    doc: dict,
) -> None:
    resolve = _resolver(ctx)
    turn = ctx.new_turn()  # type: ignore[misc]
    tools = helper_tools()
    left = ctx.deadline - time.monotonic()
    messages: list[dict] = [
        {"role": "system", "content": HELPER_BRIEF},
        {
            "role": "user",
            "content": (
                f"Task from the writer:\n{task}\n\nEpisodes to start from: {', '.join(episodes) or '(none)'}\n"
                f"Wall clock left in this pass: about {max(0, int(left))} s. "
                f"Operational step guard: {ctx.step_guard} model calls."
            ),
        },
    ]
    n = 0
    while doc["turns"] < ctx.step_guard:
        left = ctx.deadline - time.monotonic()
        if left <= 0:
            doc["status"] = "pass wall clock reached"
            return
        async with ctx.model_lock:  # helper model calls are serialised across the pass's helpers
            try:
                msg, usd = await asyncio.wait_for(turn(messages, tools), timeout=max(0.1, ctx.deadline - time.monotonic()))
            except asyncio.TimeoutError:
                doc["unknown_cost_calls"] += 1
                ctx.spend.unknown += 1
                doc["status"] = "pass wall clock reached during a model call"
                return
            except Exception:
                doc["unknown_cost_calls"] += 1  # a failed call may still have cost money
                ctx.spend.unknown += 1
                raise
        doc["turns"] += 1
        cost = _money(usd)
        if cost is None:
            doc["unknown_cost_calls"] += 1
            ctx.spend.unknown += 1
        else:
            doc["_usd"] += cost
            ctx.spend.usd += cost
        if not isinstance(msg, dict):
            doc["status"] = "error: the model returned no message"
            return
        messages.append(msg)
        calls = msg.get("tool_calls") or []
        calls = calls if isinstance(calls, list) else []
        if not calls:
            messages.append({"role": "user", "content": "Use your tools, then call done(summary)."})
            continue
        usage = _usage_of(msg, usd)
        finished = False
        for i, tc in enumerate(calls):
            tc = tc if isinstance(tc, dict) else {}
            cid = tc.get("id") if isinstance(tc.get("id"), str) and tc.get("id") else f"call_{doc['turns']}_{i}"
            fn = tc.get("function") if isinstance(tc.get("function"), dict) else {}
            name = str(fn.get("name", ""))
            args = _args(fn.get("arguments"))
            paths: list[dict] = []
            if args is None:
                content, status = "refused: unreadable arguments", "refused"
            elif finished:
                content, status = "not run: task done", "refused"
            elif name == "done":
                doc["summary"] = str(args.get("summary", ""))[:4000]
                finished, content, status = True, "ok", "ok"
            else:
                try:
                    content, paths = await _tool(ctx, name, args, st, resolve)
                    status = (
                        "refused"
                        if content.startswith(("refused:", "not run:", "unknown part"))
                        else "ok"
                    )
                except Exception as exc:  # noqa: BLE001 - a broken tool call is answered, never ends the helper
                    content, status = f"error: {type(exc).__name__}", "error"
            _emit(ctx, tool_event(ctx, call, n, name, status, content, args or {}, paths, usage if i == 0 else None))
            n += 1
            messages.append({"role": "tool", "tool_call_id": cid, "content": content})
        if finished:
            doc["status"] = "ok"
            return
    doc["status"] = "operational step guard"


async def handle(args: dict, ctx: DelegateContext) -> str:
    """The ``delegate`` tool: run one helper to its end and answer the writer. Never raises (cancellation aside),
    and a helper's failure is answered as such, never failing the pass."""
    task = args.get("task") if isinstance(args, dict) else None
    eps = args.get("episodes") if isinstance(args, dict) else None
    if not isinstance(task, str) or not task.strip():
        return _refusal(SPEC, "no_task")
    if not isinstance(eps, list) or not all(isinstance(e, str) for e in eps):
        return _refusal(SPEC, "no_episodes")
    if ctx.new_turn is None:
        return _refusal(SPEC, "no_lane", why=ctx.lane_refusal)
    if ctx.deadline - time.monotonic() <= 1.0:
        return _refusal(SPEC, "no_time")
    episodes = list(dict.fromkeys(eps))
    try:
        index = set(ctx.recorded())
    except Exception as exc:  # noqa: BLE001 - an unreadable index starts no helper
        return f"refused: the recorded-episode index could not be read ({type(exc).__name__}); no helper started"
    for e in episodes:
        if not _EPISODE_ID.match(e) or e not in index:
            return _refusal(SPEC, "unknown_episode", episode=e[:80])
    call = f"call-{ctx.next_call}"
    ctx.next_call += 1
    started = time.monotonic()
    doc: dict = {
        "call": call,
        "pass_id": ctx.pass_id,
        "status": "error: not started",
        "task_sha256": hashlib.sha256(task.encode("utf-8", "replace")).hexdigest(),
        "episodes": episodes,
        "turns": 0,
        "unknown_cost_calls": 0,
        "_usd": Decimal(0),
        "summary": "",
    }
    staging: _Staging | None = None
    tmp: Path | None = None
    try:
        staging = _Staging(ctx.out_root / call)
        ctx.scratch.mkdir(parents=True, exist_ok=True)
        tmp = Path(tempfile.mkdtemp(prefix=f"helper-{call}-", dir=ctx.scratch))
        st = _prepare(ctx, episodes, tmp)
        st.update(staging=staging, call=call)
        await _run_helper(ctx, call, task, episodes, staging, st, doc)
    except asyncio.CancelledError:
        doc["status"] = "cancelled"
        raise
    except Exception as exc:  # noqa: BLE001 - one helper's failure never stops the pass
        doc["status"] = f"error: {type(exc).__name__}"
    finally:
        if tmp is not None:
            shutil.rmtree(tmp, ignore_errors=True)
        row = _finish(ctx, call, staging, doc, time.monotonic() - started)
    return _reply(row)


def _finish(ctx: DelegateContext, call: str, staging: _Staging | None, doc: dict, seconds: float) -> dict:
    """Write ``helper.json`` last, export to ``/inputs/helpers/<call>/``, and record the call's row."""
    usd = doc.pop("_usd")
    files = []
    if staging is not None:
        for rel, size in sorted(staging.files.items()):
            try:
                data = staging.read(rel)
            except OSError:
                continue
            files.append({"path": rel, "bytes": size, "sha256": hashlib.sha256(data).hexdigest()})
    doc.update(
        usd=_usd(usd) if not doc["unknown_cost_calls"] else _UNKNOWN,
        usd_known=_usd(usd),
        seconds=round(seconds, 3),
        files=files,
        summary=doc.get("summary", ""),
    )
    row = dict(doc)
    if staging is not None:
        try:
            _write_status(staging.path, doc)
            _export(ctx, call, staging, doc)
            row["exported"] = f"/inputs/{EXPORT_DIR}/{call}/"
        except OSError as exc:
            row["status"] = f"{doc['status']}; output not exported ({type(exc).__name__})"
    ctx.records.append(row)
    return row


def _reply(row: dict) -> str:
    lines = [f"helper {row['call']}: {row['status']}"]
    if row.get("exported"):
        lines.append(f"files under {row['exported']}:")
        lines += [f"- {f['path']} ({f['bytes']} bytes)" for f in row["files"]] or ["- (none staged)"]
    lines.append(
        f"spend: {row['usd']} USD"
        + (f" ({row['unknown_cost_calls']} unpriced calls; {row['usd_known']} USD known)" if row["unknown_cost_calls"] else "")
        + f"; {row['turns']} model calls; {row['seconds']:.0f} s of the pass's wall clock"
    )
    lines.append("summary: " + (row.get("summary") or "(none)"))
    return "\n".join(lines)


async def handle_turn(calls: list[dict], ctx: DelegateContext) -> list[str]:
    """Every ``delegate`` call of one writer turn, concurrently; replies in call order."""
    return list(await asyncio.gather(*(handle(a, ctx) for a in calls)))
