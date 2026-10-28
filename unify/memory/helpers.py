"""Writer-launched helpers: the writer's ``delegate(task, episodes)`` tool.

The writer decides whether and when to hand a reading or drafting task to a helper; nothing launches one for it.
Each ``delegate`` call starts one helper with a fresh context. Every helper of a pass starts with the same bytes
(the helper brief), so that prefix is cached once; the writer's task and the named episode ids follow.

What a helper can do:

* read every recorded, settled episode by id (``overview``, ``read_episode``; ids resolve only through the
  recorded-episode index, never by joining a path, and through the writer's own loader, so the same visibility
  filter as the batch export applies), starting from the episodes the writer named;
* ``read`` and ``grep`` ``/inputs`` (the writer's own inputs, read-only and live: staging, pool, batch map,
  drafts, gate results, shared request blocks and other helpers' output), ``/task`` (the named episodes'
  records) and ``/memory`` (a copy of the library as the writer sees it);
* ``execute_code`` in its own box: the writer's box profile (bubblewrap, no network, cleared environment,
  seccomp, rlimits), ``/inputs``, ``/task`` and ``/memory`` read-only and a private, size-bounded ``/work``
  read-write;
* ``stage(path, text)`` one file in the fork's staging format (``candidates/<name>.py``, ``cases/<name>.json``,
  ``notes.md``), written host-side with ``O_EXCL | O_NOFOLLOW`` into a fresh directory for this call;
* ``done(summary, claims)`` to hand back: every claim names the episode parts it rests on
  (``<episode>/<part>``, the part names ``overview`` lists), and a ``done`` with a claim whose sources are missing or
  do not resolve to a part of a recorded episode is refused, so the helper can fix it.

A helper cannot launch helpers: ``delegate`` is never in its tool list.

Where its output goes: ``<state>/memory-helpers/<pass>/<name>/`` holds the staged files and ``helper.json``
(status, usage, the block ledger, the task digest, the claims with their sources), written last. The same files
are exported to the writer's ``/inputs/helpers/<name>/``, and the ``delegate`` reply lists them with the helper's
summary, its claims with their sources, and its spend. The writer is told that a claim is untrusted until it has
read the parts it points to.

Model calls: a helper's model calls are made host-side, by the writer's process, through the turn factory the
pass is given (the key proxy's helper lane when the run has one), never from inside a helper's box, which holds no
credential. Helpers run at the same time and so do their model calls. Unknown cost is counted as unknown (fail
closed) on the pass's spend and in ``helper.json``.

Messaging (:mod:`.messaging`): ``delegate`` starts the helper through the pass record's pool and returns at once with
its name, ``call-<n>``: one string for its record entries, events and output folder. It talks on the
record (``send``, ``ask``, ``messages``, ``wait``, ``who``); its ``done`` text is its last entry. A helper ends
when it calls done, when the writer stops it, when the writer finishes (a model call in flight then finishes
first and no new one starts) or at the pass's wall clock, the outer backstop. Whichever way it ends,
``helper.json`` and its staged files are written and kept.

A helper's failure never fails the pass: its last entry on the record says what happened.
"""

from __future__ import annotations

import asyncio
import hashlib
import inspect
import json
import os
import re
import shutil
import stat
import tempfile
import textwrap
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Awaitable, Callable, Iterable

from unify.agents.pool import Pool
from unify.agents.record import PostRefused
from unify.common import tool_registry as _registry

from . import batch_map as _bm
from . import messaging as _messaging
from . import safe_paths as _safe
from . import views as _views
from . import writer_tools as _wt
from .integration.fork_worker import QUOTA_BYTES, QUOTA_FILES, allowed
from .library_prompts import CANDIDATE_CONTRACT_IN_CODE
from . import sandbox_run as _sandbox
from .tool_spec import GENERAL, Refusal, ToolSpec, form_of, refusal

#: The directory under the run's state dir that holds every helper's output (``<pass>/<name>/``).
HELPERS_DIR = "memory-helpers"
#: The status file a helper's output directory gets last.
STATUS_FILE = "helper.json"
#: Where the writer reads a helper's output: ``/inputs/helpers/<name>/``.
EXPORT_DIR = "helpers"

#: ``/work`` bounds: a file at most this size (RLIMIT_FSIZE in the box), all of /work at most this much and this
#: many entries (checked after every cell; past them, the helper's execute_code stops for the rest of its run).
WORK_FILE_BYTES = 2 * 1024**2
WORK_TOTAL_BYTES = 20 * 1024**2
WORK_ENTRIES = 2000
#: A helper cell's time, at most (less when the pass's wall clock has less left).
CELL_TIMEOUT_S = 60.0
#: The head of a cell's output a helper is shown.
OUTPUT_BYTES = 8000

_EPISODE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}\Z")
_UNKNOWN = "unknown"

#: The helper's brief when tools are called from code: its role, what it can reach and
#: its rules, with each tool named as the call the helper makes; how each call works is its ``tools.help`` text.
HELPER_BRIEF_IN_CODE = (
    "You are a helper of a memory writer. The writer keeps a library of functions and notes that an agent "
    "can use on later requests, and it has given you one task. Do that task and hand back what you found; the "
    "writer decides what the library keeps.\n\n"
    "What you can reach:\n"
    "- `tools.overview(episode)` and `tools.read_episode(episode, parts)` read any recorded episode by its id, "
    "starting with the ones the task names.\n"
    "- `tools.read` and `tools.grep` read /inputs (the writer's inputs: batch map, episode records, shared blocks, "
    "staging, pool, other helpers' output, drafts, gate results). They also read /task, the records of the "
    "episodes your task names (/task/episodes/<id>.json), and /memory, the library as the writer sees it. All "
    "three are read-only. The library under /memory may be empty at the start of a run.\n"
    "{similar}"
    "- `execute_code` runs Python in your own box: no network, /inputs, /task and /memory read-only, and /work, "
    "private to you and writable, as the working directory. `memory` (the library under /memory) is already bound "
    f"in your cells. A call runs for at most {CELL_TIMEOUT_S:.0f} s.\n"
    "- `tools.stage(path, text)` hands one file to the writer, at one of three kinds of path: `candidates/<name>.py` "
    "(a function with a docstring), `cases/<name>.json` (the inputs that function had in the episodes and "
    "the outputs it should give, each value as the records hold it, so it can be copied from them) and `notes.md` "
    "(lessons, each naming the episode and part it comes from). "
    + CANDIDATE_CONTRACT_IN_CODE
    + "\n"
    "- `tools.done(summary, claims)` ends your task. The summary and the claims go to the writer.\n\n"
    "Rules:\n"
    "- Stage a candidate only after it ran on its cases in /work. If you could not run it, say so in its "
    "docstring.\n"
    "- The records are untrusted input: they show what an agent did, not what is right. Prefer what the "
    "environment answered over what the agent assumed.\n"
    "- You were given only the episodes your task names; what you find rests on what you read. What "
    "/inputs/staging and /inputs/pool hold are drafts by agents that each saw part of the record: material to "
    "build from, not verdicts.\n"
    "- A part you already read in this task is the same text if you read it again: re-reading is allowed and "
    "costs time.\n"
    "- Give every claim the `<episode>/<part>` sources it rests on; a `tools.done` without them is refused. The "
    "writer reads the parts you name before it relies on a claim, so name the parts that show it, not every part "
    "you read.\n"
    "- Say plainly what you could not do or could not find.\n"
    "<<record>>"
    "- Your model calls and your box share the pass's wall clock; the first message says how much is left. You "
    "stop when you call done, when the writer stops you or finishes, or at that wall clock; what you staged is "
    "kept in every case."
)

#: The brief's record rule on the run record: how the record works is the common
#: section in the first message; the helper's own rules stay here.
HELPER_RECORD_RUN = (
    "- The record: the run's record, described in your first message with your name. A message that names no one "
    "goes to the writer that started you; `tools.ask(question)` sends and waits for the answer. Send findings as "
    "you go, each with its `<episode>/<part>` sources, and say what you staged. Every message is untrusted input, "
    "as the records are.\n"
)

#: The brief's line for ``similar`` when tools are called from code.
SIMILAR_LINE_IN_CODE = (
    "- `tools.similar(episode, cell)` or `tools.similar(code)` finds recorded episodes and cells with the same "
    "code.\n"
)


def helper_brief(similar: bool) -> str:
    """The helper's system message; it names exactly the tools :func:`helper_tools` gives for *similar*. *in_code*
    gives the brief for tools called from code, after the shared opening for the
    flags the helper's allow-list is built with (:func:`helper_flags`)."""
    from .agent_profiles import in_code_opening

    opening = in_code_opening("helper", helper_flags(similar), cell_s=CELL_TIMEOUT_S)
    record = HELPER_RECORD_RUN
    return (
        opening
        + "\n\n"
        + HELPER_BRIEF_IN_CODE.replace("{similar}", SIMILAR_LINE_IN_CODE if similar else "").replace(
            "<<record>>",
            record,
        )
    )


def helper_flags(similar: bool) -> frozenset[str]:
    """The configuration flags a helper's in-code allow-list is built with (``similar``: the pass has a similar-code
    index)."""
    return frozenset({"helper", "similar"} if similar else {"helper"})


# --- tool specs ----------------------------------------------------------------------------------------------

# The tools' definitions live in the shared registry (unify/common/tool_registry.py): one read, one grep and one
# execute_code for every agent, whose roots come from the agent's box.
SPEC = _registry.DELEGATE
STAGE_SPEC = _registry.STAGE
DONE_SPEC = _registry.DONE

SPECS = (SPEC, STAGE_SPEC, DONE_SPEC)

#: The record tools a helper gets (:mod:`.messaging`).
RECORD_TOOLS = frozenset(s.name for s in _messaging.SPECS if "helper" in s.roles)


def _refusal(spec: ToolSpec, form: str, **fields: object) -> str:
    return refusal(spec.name, form, **fields)


# --- context -------------------------------------------------------------------------------------------------


@dataclass
class DelegateContext:
    """What :func:`handle` needs from the writer's pass. Built by :func:`context_for` inside a pass.

    * ``pass_id``: the pass; ``out_root``: ``<state>/memory-helpers/<pass>`` on the host (the helpers' output);
    * ``inputs``: the writer's ``/inputs`` on the host (each call's output is exported to ``helpers/<name>/``);
    * ``library``: the writer's ``/memory`` on the host (copied for each helper, which sees it read-only);
    * ``load``: the writer's own episode loader (the batch export's visibility filter applies to it);
    * ``recorded``: the recorded-episode index: every settled episode id a helper may read;
    * ``new_turn``: a fresh host-side model turn per helper (the helper lane's route, the helper model, the pass's
      effort), or None when no helper lane exists (``lane_refusal`` says why);
    * ``deadline``: the pass's wall clock end (``time.monotonic()``), the outer backstop of every helper;
      ``spend``: the pass's spend (an object with
      ``usd`` and ``unknown``): helper spend joins it as it is incurred;
    * ``shared``: the shared request blocks (id -> text); ``events``: the pass's ``tool-events.jsonl`` (or None);
    * ``episode_reader``: the writer's episode reader for ``overview``/``read_episode`` if the pass has one
      (``async (name, args, resolve) -> str``), else the writer's own handlers with this helper's read log
      (:func:`_episode_reply`); ``blob``: the gate's recorded-blob reader for the raw lengths those handlers list
      (None: the lengths are "unknown"); ``similar``: the ``similar`` tool's
      handler (``async (args) -> str``), or None;
    * ``records``: one row per delegate call, appended here (the pass outcome's ``helpers``);
    * ``pass_record``: the pass on the run record (:class:`.messaging.PassRecord`), whose pool starts the helpers;
    * ``pool``: the pass's pool of drafts (:class:`.pool.Pool`), or None: the closed entries a helper's staged code
      or episodes resemble are listed in its ``helper.json`` and its reply, as the batch map lists them for an
      episode.
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
    pass_record: _messaging.PassRecord
    shared: dict[str, str] = field(default_factory=dict)
    events: Path | None = None
    lane_refusal: str = "no helper lane"
    episode_reader: Callable[..., Awaitable[str]] | None = None
    #: the gate's recorded-blob reader (``sha -> bytes``), for the raw lengths in read_episode's table of contents
    blob: Callable[[str], bytes] | None = None
    similar: Callable[[dict], Awaitable[str]] | None = None
    step_guard: int = 500
    #: whether one more model call fits under the pass's USD stop (:func:`.writer_pass.usd_room`); None: no stop
    usd_room: Callable[[], bool] | None = None
    cell_timeout_s: float = CELL_TIMEOUT_S
    records: list[dict] = field(default_factory=list)
    pool: Any = None
    #: host-stamped open rows from the helpers' boxes (also appended to ``box-opens.jsonl`` beside ``events``)
    box_opens: list[dict] = field(default_factory=list)
    next_call: int = 0
    #: each started helper's task and episodes, until its body takes them
    pending: dict[str, tuple[str, list[str]]] = field(default_factory=dict)
    #: each ended helper's last text (its ``done`` reply, or how it ended)
    replies: dict[str, str] = field(default_factory=dict)
    #: each helper's message list and tool JSON, as it grows (recorded with the writer's transcript)
    transcripts: dict[str, tuple[list, list]] = field(default_factory=dict)
    #: helpers compact their conversations as the agent loop does (:func:`.writer_pass.compact_by_summary`); what
    #: each compaction took out, by helper, for its transcript
    archived: dict[str, list] = field(default_factory=dict)
    #: a helper's requests carry only execute_code and its other tools are calls from its cells; its interpreter
    #: starts with ``interpreter_start`` (None: the default) and its handler threads join the pass's tracker
    #: ``threads``
    interpreter_start: Any = None
    threads: Any = None

    def __post_init__(self) -> None:
        pr = self.pass_record
        if pr.pool is None and self.new_turn is not None:

            async def start(name: str, spawner: str, request: str, seq: int) -> str:
                return await _body(self, name)

            pr.pool = Pool(pr.record, start_helper=start)


def context_for(
    writer: Any,
    *,
    pass_id: str,
    inputs: Path,
    box: Path,
    scratch: Path,
    deadline: float,
    spend: Any,
    pass_record: _messaging.PassRecord,
    events: Path | None = None,
) -> DelegateContext:
    """The delegate context of *writer*'s current pass (a :class:`.writer_pass.WriterPass`)."""
    from .writer_pass import STEP_GUARD, usd_room

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
        usd_room=(lambda: usd_room(cfg, spend)) if getattr(cfg, "usd_limit", None) is not None else None,
        cell_timeout_s=float(getattr(cfg, "cell_timeout_s", 60.0)),
        records=writer._helper_rows,
        blob=getattr(getattr(writer, "gate", None), "_blob", None),
        pass_record=pass_record,
        pool=getattr(writer, "_pool", None),
        interpreter_start=getattr(writer, "interpreter_start", None),
        threads=getattr(writer, "_in_code_threads", None),
    )


# --- events --------------------------------------------------------------------------------------------------


def _utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def _refusal_form(content: str) -> dict:
    """The refusal's own form code (the reply was made by :func:`.tool_spec.refusal`) and its size."""
    return {
        "form": form_of(content),
        "text_bytes": len(content.encode("utf-8", "replace")),
    }


def _emit(ctx: DelegateContext, row: dict) -> None:
    if ctx.events is None:
        return
    try:
        ctx.events.parent.mkdir(parents=True, exist_ok=True)
        with open(ctx.events, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(row, sort_keys=True, default=str) + "\n")
    except OSError:
        pass  # the record is best effort; the helper's own helper.json still holds its ledger


def record_box_opens(ctx: DelegateContext, row: dict, audit: dict | None) -> None:
    """Add an ``execute_code`` call's open-log fields to its tool event *row*, and keep its open rows, stamped by
    the host (agent, pass, delegate call, call number; the box supplies only event, path and mode), on
    ``ctx.box_opens`` and in ``box-opens.jsonl`` beside the pass's ``tool-events.jsonl``. *audit* None: no box ran.
    """
    row.update(_sandbox.box_open_fields(audit))
    rows = [
        {
            "agent": "helper",
            "pass_id": ctx.pass_id,
            "helper_call": row.get("helper_call"),
            "call": row.get("call"),
            **o,
        }
        for o in (audit or {}).get("rows") or []
    ]
    if not rows:
        return
    ctx.box_opens.extend(rows)
    if ctx.events is not None:
        _sandbox.append_open_rows(ctx.events.parent / "box-opens.jsonl", rows)


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
        # one helper identity: its record name, delegate's answer and its output folder are this same string
        "name": call,
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
    directory descriptors, so no link (or a path joined by the model) is ever followed.
    """

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
            return _refusal(
                STAGE_SPEC,
                "quota",
                path=rel,
                files=QUOTA_FILES,
                total=QUOTA_BYTES,
            )
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
    """Which /work bound *root* breaks, or None. Walked from descriptors (:func:`.safe_paths.walk_unlocking`):
    nothing followed or read, and the owner's access to each directory restored on the opened directory itself,
    so a directory the box swaps for a link meanwhile never changes a mode outside /work."""
    entries = total = 0
    walk = _safe.walk_unlocking(root, 0o700)
    try:
        for event, _, st in walk:
            if event == "unlisted":
                return "/work holds an unreadable directory"
            if event != "entry" or st is None:
                return "/work holds an unreadable entry"
            entries += 1
            if entries > WORK_ENTRIES:
                return f"/work holds more than {WORK_ENTRIES} files and directories"
            if stat.S_ISREG(st.st_mode):
                total += st.st_size
                if total > WORK_TOTAL_BYTES:
                    return f"/work holds more than {WORK_TOTAL_BYTES} bytes"
    except _safe.NoProc:
        return "/work cannot be measured safely (/proc/self/fd is not available)"
    finally:
        walk.close()
    return None


# --- episode readers -----------------------------------------------------------------------------------------


def _part_names(ep: Any) -> list[str]:
    """The part names ``read_episode`` reads for *ep* (the writer's own list, so a helper and the writer agree)."""
    return _wt.part_names(ep)


def episode_paths(name: str, args: dict, eid: str) -> list[dict]:
    """The episode paths an answered ``overview`` or ``read_episode`` call read, as the writer's events name them:
    ``/episodes/<id>`` for the overview or the table of contents, ``/episodes/<id>/<part>`` for each named part.
    """
    wanted, _ = _wt._wanted_parts(args) if name == "read_episode" else (None, None)
    if not wanted:
        return [{"path": f"/episodes/{eid}", "mode": "read"}]
    names = [str(p.get("part")) if isinstance(p, dict) else str(p) for p in wanted]
    return [{"path": f"/episodes/{eid}/{p.strip()}", "mode": "read"} for p in names]


def _no_blob(sha: str) -> bytes:
    raise KeyError(sha)


def _episode_reply(
    ctx: DelegateContext,
    name: str,
    args: dict,
    ep: Any,
    eid: str,
    st: dict,
) -> tuple[str, list[dict]]:
    """``overview(episode)`` and ``read_episode`` on one resolved episode through the writer's own handlers
    (:func:`.writer_tools.overview_episode`, :func:`.writer_tools.read_episode`): the same part limit per call, the
    same table of contents (fixture sources and raw lengths) and the same pointer for a part identical to one this
    helper was shown in full earlier. Each helper keeps its own read log and its own record of parts shown."""
    log = st.setdefault("read_log", _wt.ReadLog())
    shown = st.setdefault("shown_full", {})
    if name == "overview":
        reply = _wt.overview_episode(
            ep,
            eid,
            args.get("offset", 0),
            log=log,
            shared=ctx.shared,
        )
    else:
        reply = _wt.read_episode(
            ep,
            eid,
            args,
            blob=ctx.blob or _no_blob,
            log=log,
            shown=shown,
            shared=ctx.shared,
        )
    if reply.status == "refused":
        return Refusal(reply.text, name, reply.form or "refused"), []
    return reply.text, [dict(p) for p in reply.paths]


# --- claims and their sources ------------------------------------------------------------------------------


def _source_problem(
    source: Any,
    resolve: Callable[[str], tuple[Any, str] | str],
    shared: dict[str, str],
) -> str | None:
    """Why *source* (``<episode>/<part>``) does not name a part of a recorded episode, or None when it does.

    The episode resolves the way ``overview`` and ``read_episode`` resolve it (the recorded-episode index, then the
    writer's loader); the part must be one of the names ``overview`` lists for that episode (or a shared request
    block, ``shared:<id>``) and must read the way ``read_episode`` reads it.
    """
    if not isinstance(source, str) or "/" not in source:
        return "not written `<episode>/<part>`"
    eid, part = source.split("/", 1)
    try:
        got = resolve(eid)
    except (
        Exception
    ) as exc:  # noqa: BLE001 - an episode that cannot be loaded is not a source
        return f"episode {eid[:80]!r} could not be loaded ({type(exc).__name__})"
    if isinstance(got, str):
        if "not an episode id" in got:
            return f"{eid[:80]!r} is not an episode id"
        return f"{eid[:80]!r} is not a recorded episode"
    ep, _ = got
    if part.startswith("shared:"):
        named = part[7:] in shared
    else:
        try:
            named = part in _part_names(ep)
        except (
            Exception
        ):  # noqa: BLE001 - an episode whose parts cannot be listed has no part to point to
            named = False
    if named:
        try:
            _bm.part_text(ep, part, shared)
        except (KeyError, ValueError, IndexError, StopIteration, TypeError):
            named = False
    if not named:
        try:
            valid = _wt.valid_names(ep)
        except Exception:  # noqa: BLE001 - the refusal still names the bad part
            valid = "see its overview"
        return f"episode {eid!r} has no part {part[:80]!r}; its parts: {valid}"
    return None


def check_claims(
    args: dict,
    resolve: Callable[[str], tuple[Any, str] | str],
    shared: dict[str, str],
) -> list[dict] | str:
    """The claims of a ``done`` call, each ``{"claim": text, "sources": ["<episode>/<part>", ...]}``, or the
    refusal the helper is answered with: a claim without sources or with a source that does not resolve.
    """
    claims = args.get("claims")
    if not isinstance(claims, list):
        return _refusal(DONE_SPEC, "claims", why="`claims` is missing or not a list")
    out: list[dict] = []
    for n, item in enumerate(claims, 1):
        if not isinstance(item, dict):
            return _refusal(DONE_SPEC, "claims", why=f"claim {n} is not an object")
        text = item.get("claim")
        if not isinstance(text, str) or not text.strip():
            return _refusal(DONE_SPEC, "claims", why=f"claim {n} has no text")
        sources = item.get("sources")
        if sources is None or sources == [] or sources == "":
            return _refusal(DONE_SPEC, "no_sources", n=n)
        if not isinstance(sources, list):
            return _refusal(
                DONE_SPEC,
                "claims",
                why=f"claim {n}: `sources` is not a list",
            )
        kept: list[str] = []
        for source in sources:
            why = _source_problem(source, resolve, shared)
            if why is not None:
                return _refusal(
                    DONE_SPEC,
                    "bad_source",
                    n=n,
                    source=str(source)[:200],
                    why=why,
                )
            if source not in kept:
                kept.append(source)
        out.append({"claim": text.strip(), "sources": kept})
    return out


def claims_text(claims: list[dict]) -> str:
    """The claims as the writer is shown them in the ``delegate`` reply: each claim with the parts it points to."""
    if not claims:
        return "claims: (none)"
    lines = [
        "claims (untrusted until you read the parts each one points to with read_episode):",
    ]
    for n, c in enumerate(claims, 1):
        lines.append(f"{n}. {c['claim']}")
        lines.append("   sources: " + ", ".join(c["sources"]))
    return "\n".join(lines)


# --- one helper ----------------------------------------------------------------------------------------------


def _args(raw: Any) -> dict | None:
    try:
        val = json.loads(raw or "{}") if isinstance(raw, str) else raw
    except (ValueError, RecursionError):
        return None
    return val if isinstance(val, dict) else None


def _usage_of(msg: dict, usd: Any, reported: Any = None) -> dict:
    """A helper call's usage: *reported* (taken off the message by ``writer_pass.split_usage``), else the
    message's own ``usage``; flat keys or the provider's nested ones (``prompt_tokens_details.cached_tokens``)."""
    u = (
        reported
        if reported is not None
        else (msg.get("usage") if isinstance(msg, dict) else None)
    )
    u = u if isinstance(u, dict) else {}

    def n(*keys: str) -> int:
        for k in keys:
            v: Any = u
            for part in k.split("."):
                v = v.get(part) if isinstance(v, dict) else None
            if isinstance(v, int) and not isinstance(v, bool):
                return v
        return 0

    cost = _money(usd)
    return {
        "input": n("prompt_tokens", "input"),
        "cache_read": n(
            "cache_read",
            "cached_tokens",
            "prompt_tokens_details.cached_tokens",
            "cache_read_input_tokens",
        ),
        "cache_write": n(
            "cache_write",
            "cache_creation_input_tokens",
            "prompt_tokens_details.cache_write_tokens",
        ),
        "output": n("completion_tokens", "output"),
        "reasoning": n(
            "reasoning_tokens",
            "reasoning",
            "completion_tokens_details.reasoning_tokens",
        ),
        "usd": _usd(cost) if cost is not None else _UNKNOWN,
    }


def _resolver(ctx: DelegateContext) -> Callable[[str], tuple[Any, str] | str]:
    """Episode ids resolve only through the recorded-episode index, then the writer's loader; never a path."""
    index: set[str] | None = None

    def resolve(eid: str, tool: str = "read_episode") -> tuple[Any, str] | str:
        nonlocal index
        if not isinstance(eid, str) or not _EPISODE_ID.match(eid):
            return refusal(tool, "not_episode_id", episode=str(eid)[:80])
        if index is None:
            index = set(ctx.recorded())
        if eid not in index:
            return refusal(tool, "not_recorded", episode=eid)
        return ctx.load(eid), eid

    return resolve


def _tool_sync(
    ctx: DelegateContext,
    name: str,
    args: dict,
    st: dict,
    resolve: Callable[[str], Any],
) -> tuple[str, list[dict]]:
    """A helper's ``stage``, ``read`` or ``grep`` call, synchronously (:func:`_tool` runs grep on a worker thread; an
    in-code call runs any of them from the worker pool): (reply, the paths it touched)."""
    roots = {
        "/inputs": st["hin"],
        "/task": st["task"],
        "/memory": st["lib"],
        "/work": st["work"],
    }
    if name == "stage":
        rel = str(args.get("path", ""))
        out = st["staging"].stage(rel, str(args.get("text", "")))
        vpath = f"/inputs/{EXPORT_DIR}/{st['call']}/{rel}"
        return out, ([] if form_of(out) else [{"path": vpath, "mode": "write"}])
    if name in ("read", "grep"):
        vpath = str(args.get("path", "") or ("/inputs" if name == "grep" else ""))
        try:
            offset = int(args.get("offset", 0) or 0)
            # the box path asked for, normalised, as the writer's events record it
            prefix, parts = _safe.split(vpath, list(roots))
            virtual = _safe.virtual(prefix, parts)
            if name == "read":
                text, mode = _views.read_entry(vpath, roots, offset)
                return text, [{"path": virtual, "mode": mode}]
            out = _views.grep_bounded(
                str(args.get("pattern", "")),
                vpath,
                roots,
                offset,
            )
            return out, [{"path": virtual, "mode": "search"}]
        except _safe.Outside:
            return _own_refusal(name, "outside_box", path=vpath[:200]), []
        except _views.PatternProblem as exc:
            return _own_refusal("grep", "bad_pattern", reason=str(exc)[:300]), []
        except _views.SearchFailed:
            return _own_refusal("grep", "search_failed"), []
        except FileNotFoundError:
            missing = _own_refusal(name, "not_found", path=vpath[:200])
            listing = _parent_listing(vpath, roots)
            return (
                Refusal(
                    str(missing) + listing,
                    name,
                    "not_found",
                ),
                [],
            )
        except (_safe.LinkFound, _safe.NotRegular) as exc:
            return (
                _own_refusal(name, "unreadable", error=exc.reason, path=vpath[:200]),
                [],
            )
        except (OSError, ValueError, TypeError) as exc:
            # the type and the path asked for only: the error's text names host paths
            return (
                _own_refusal(
                    name,
                    "unreadable",
                    error=type(exc).__name__,
                    path=vpath[:200],
                ),
                [],
            )
    raise ValueError(f"not a synchronous helper tool: {name}")


async def _tool(
    ctx: DelegateContext,
    name: str,
    args: dict,
    st: dict,
    resolve: Callable[[str], Any],
) -> tuple[str, list[dict]]:
    """One helper tool call: (reply, the paths it touched)."""
    if name in RECORD_TOOLS:
        info = st["messaging"] = {}
        return await _messaging.call(ctx.pass_record, st["call"], name, args, info), []
    if name == "stage":
        return _tool_sync(ctx, name, args, st, resolve)
    if name in ("read", "grep"):
        # both open box paths and may wait on the file system: never on the event loop
        return await asyncio.to_thread(_tool_sync, ctx, name, args, st, resolve)
    if name == "overview" and not args.get("episode"):
        return _batch_overview(ctx, args), [
            {"path": "/inputs/batch_map.json", "mode": "read"},
        ]
    if name in ("overview", "read_episode"):
        got = resolve(args.get("episode"), name)
        if isinstance(got, str):
            return got, []
        ep, eid = got
        if ctx.episode_reader is not None:
            out = await ctx.episode_reader(name, args, resolve)
            return out, ([] if form_of(out) else episode_paths(name, args, eid))
        return await asyncio.to_thread(_episode_reply, ctx, name, args, ep, eid, st)
    if name == "similar":
        if ctx.similar is None:
            return refusal("similar", "not_available"), []
        out = await ctx.similar(args)
        eid = args.get("episode")
        named = isinstance(eid, str) and not form_of(out)
        return out, ([{"path": f"/episodes/{eid}", "mode": "search"}] if named else [])
    if name == "execute_code":
        from unify.common.tool_registry import THOUGHT_ONLY_REPLY, thought_only

        if thought_only(args):
            # a thought alone: nothing runs and nothing is refused; the thought stays in the conversation
            return THOUGHT_ONLY_REPLY, []
        if st["work_over"] is not None:
            return refusal("execute_code", "work_full", why=st["work_over"]), []
        stopping = ctx.pass_record.stopping.get(st["call"])
        if stopping is not None:
            return refusal("execute_code", "stopping", reason=stopping), []
        left = ctx.deadline - time.monotonic()
        if left <= 1.0:
            return refusal("execute_code", "no_time"), []
        out = await _in_code_cell(ctx, st, str(args.get("code", "")), left)
        over = work_over(st["work"])
        if over is not None:
            st["work_over"] = over
            out += f"\n({over}; execute_code is closed for the rest of this task)"
        return out, []
    raise ValueError(f"not a helper tool: {name}")


def _own_refusal(tool: str, form: str, **fields: object) -> str:
    """A refusal in the helper's own words for read or grep (its roots: the registry's helper texts, the same words
    its ``tools.help`` shows), keeping its form code."""
    return _registry.role_refusal("helper", tool, form, memory=True, **fields)


def _parent_listing(vpath: str, roots: dict) -> str:
    """What the nearest existing directory above *vpath* holds, for a path that does not exist (same form). Each
    directory is opened from the root descriptor without following a link (:mod:`.safe_paths`)."""
    parent = vpath.rstrip("/").rsplit("/", 1)[0] or "/"
    with _safe.using(roots) as r:
        while parent not in ("", "/"):
            try:
                fd, _, _ = _safe.open_path(r, parent)
            except _safe.Missing:
                parent = parent.rsplit("/", 1)[0]
                continue
            except OSError:
                break
            try:
                names = _safe.listing(fd) if _safe.is_dir_fd(fd) else None
            except OSError:
                break
            finally:
                os.close(fd)
            if names is None:
                parent = parent.rsplit("/", 1)[0]
                continue
            shown = names[:50] + (
                [f"... {len(names) - 50} more"] if len(names) > 50 else []
            )
            return f"; {parent} holds: " + (", ".join(shown) or "(nothing)")
    return ""


def _batch_overview(ctx: DelegateContext, args: dict) -> str:
    """``overview`` without an episode: one line per batch episode, from the writer's /inputs/batch_map.json."""
    try:
        rows = json.loads((ctx.inputs / "batch_map.json").read_text())["episodes"]
    except (OSError, ValueError, KeyError, TypeError):
        return "this pass has no batch map; give `episode` to read one recorded episode"
    return _wt.overview_batch(
        rows if isinstance(rows, list) else [],
        args.get("offset", 0),
    ).text


def _prepare(ctx: DelegateContext, episodes: list[str], tmp: Path) -> dict:
    """The helper's view: the writer's own /inputs (read-only and live, so other helpers' output appears as it is
    exported), its /task (the named episodes' records), its library copy and /work."""
    from .writer_pass import _mirror, export_for_writer

    task, lib, work, cells = (tmp / n for n in ("task", "memory", "work", "cells"))
    for d in (task, lib, work, cells):
        d.mkdir()
    # the writer's /inputs/helpers/ exists from the first helper on, so the helpers' bind of /inputs shows each
    # helper's output as it is exported
    (ctx.inputs / EXPORT_DIR).mkdir(parents=True, exist_ok=True)
    if not (ctx.inputs / "sitecustomize.py").exists():
        try:  # the writer's pass stages it; the box's observe-only open log needs it at /inputs/sitecustomize.py
            _sandbox.stage_open_hook(ctx.inputs)
        except FileExistsError:
            pass
    export_for_writer(ctx.load, episodes, task / "episodes", library_mode=True)
    _mirror(ctx.library, lib, skip_top=frozenset({".pass"}))
    return {
        "hin": ctx.inputs,
        "task": task,
        "lib": lib,
        "work": work,
        "cells": cells,
        "work_over": None,
    }


def _export(ctx: DelegateContext, call: str, staging: _Staging, doc: dict) -> None:
    dest = ctx.inputs / EXPORT_DIR / call
    dest.mkdir(parents=True, exist_ok=False)
    for rel in staging.files:
        out = dest / rel
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(staging.read(rel))
    (dest / STATUS_FILE).write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n")


def _write_status(path: Path, doc: dict) -> None:
    fd = os.open(
        path / STATUS_FILE,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
        0o644,
    )
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        fh.write(json.dumps(doc, indent=1, sort_keys=True) + "\n")


async def _compact_helper(
    ctx: DelegateContext,
    call: str,
    turn: Callable[[list, list], Awaitable[tuple[dict, Any]]],
    messages: list[dict],
    tools: list[dict],
    doc: dict,
) -> None:
    """A helper's compaction (:func:`.writer_pass.compact_by_summary`): the summary call goes through the helper's
    own turn (the helper lane's route and model), counts as one of its turns, and its charge joins the helper's
    and the pass's spend like any call; a ``compaction`` event row carries its usage. On a summary, what it took
    out is kept for the helper's transcript and the conversation restarts in place."""
    from .writer_pass import _prompt_chars, call_in_flight, compact_by_summary, compaction_archive

    pr = ctx.pass_record
    pr.calls_in_flight.add(call)
    try:
        with call_in_flight(ctx.spend):
            comp = await asyncio.wait_for(
                compact_by_summary(messages, turn, tools),
                timeout=max(0.1, ctx.deadline - time.monotonic()),
            )
    except asyncio.CancelledError:
        doc["unknown_cost_calls"] += 1
        ctx.spend.unknown += 1
        raise
    except (
        Exception
    ):  # noqa: BLE001 - a failed summary call may still have cost money; the conversation stays
        doc["unknown_cost_calls"] += 1
        ctx.spend.unknown += 1
        return
    finally:
        pr.calls_in_flight.discard(call)
    doc["turns"] += 1
    cost = _money(comp.usd)
    if cost is None:
        doc["unknown_cost_calls"] += 1
        ctx.spend.unknown += 1
    else:
        doc["_usd"] += cost
        ctx.spend.usd += cost
    kept = {id(m) for m in comp.restart or []}
    out = [m for m in messages if id(m) not in kept] if comp.restart is not None else []
    if comp.restart is not None:
        ctx.archived.setdefault(call, []).extend(
            compaction_archive(comp, list(messages)),
        )
        messages[:] = comp.restart
        doc["compactions"] = doc.get("compactions", 0) + 1
    _emit(
        ctx,
        {
            "event": "compaction",
            "agent": "helper",
            "helper": call,
            "pass_id": ctx.pass_id,
            "utc": _utc(),
            "restarted": comp.restart is not None,
            "messages_moved": len(out),
            "chars_moved": sum(_prompt_chars(m) for m in out),
            "usage": _usage_of(
                comp.reply if isinstance(comp.reply, dict) else {},
                comp.usd,
            ),
        },
    )


async def _run_helper(
    ctx: DelegateContext,
    call: str,
    task: str,
    episodes: list[str],
    staging: _Staging,
    st: dict,
    doc: dict,
) -> None:
    """One helper's turns inside its interpreter session, which ends with it (verified kill): with tools in code, a
    session with the tool channel; off, one with no channel (a fresh box per cell,
    :func:`box_cell`, and no session)."""
    session = _in_code_session(ctx, call, st, doc)
    try:
        return await _run_helper_turns(ctx, call, task, episodes, staging, st, doc)
    finally:
        await session.close()


async def _run_helper_turns(
    ctx: DelegateContext,
    call: str,
    task: str,
    episodes: list[str],
    staging: _Staging,
    st: dict,
    doc: dict,
) -> None:
    resolve = _resolver(ctx)
    st["in_code"].resolve = resolve
    session = f"helper.{ctx.pass_id}.{call.rsplit('-', 1)[-1]}"
    # a factory takes the per-call session label (the live lane) or nothing (scripted turns)
    takes_session = bool(inspect.signature(ctx.new_turn).parameters)  # type: ignore[arg-type]
    turn = ctx.new_turn(session) if takes_session else ctx.new_turn()  # type: ignore[misc]
    from .tool_handlers import in_code_json_tools

    tools = in_code_json_tools("helper")
    pr = ctx.pass_record
    left = ctx.deadline - time.monotonic()
    messages: list[dict] = [
        {"role": "system", "content": helper_brief(ctx.similar is not None)},
        {
            "role": "user",
            "content": (
                f"Task from the writer:\n{task}\n\nEpisodes to start from: {', '.join(episodes) or '(none)'}\n"
                f"Your name on the record: {call}; the agent that runs this pass: {pr.root}. "
                f"Wall clock left in this pass: about {max(0, int(left))} s. "
                f"Operational step guard: {ctx.step_guard} model calls."
                + (_run_lines(call, pr.root))
            ),
        },
    ]
    ctx.transcripts[call] = (messages, tools)
    from unify.common._async_tool import loop_stop as _loop_stop

    # the shared no-progress rule (loop_stop.AgentNoProgress), always on: a notice at k no-progress model calls in
    # a row, and at 2 * k the helper ends with what it staged
    no_progress = _loop_stop.AgentNoProgress(_loop_stop.terminal_tool("helper"), in_code=True)
    n = 0
    last_prompt = (
        None  # the last model call's prompt tokens (writer_pass.compaction_due)
    )
    while doc["turns"] < ctx.step_guard:
        # asked to stop (the writer finished): no new model call starts; one in flight was allowed to finish
        reason = pr.stopping.get(call)
        if reason is not None:
            doc["status"] = f"stopped: {reason}"
            return
        if ctx.deadline - time.monotonic() <= 0:
            doc["status"] = "pass wall clock reached"
            return
        if ctx.usd_room is not None and not ctx.usd_room():
            doc["status"] = "stopped: the writer service's room for this pass reached"
            return
        from .writer_pass import compaction_due

        if compaction_due(last_prompt, getattr(turn, "max_input_tokens", None)):
            last_prompt = None
            await _compact_helper(ctx, call, turn, messages, tools, doc)
            if doc["turns"] >= ctx.step_guard:
                break
        if doc["turns"]:
            # the run record's turn boundary: what is new for this helper, after its turn's tool results
            try:
                block = pr.record.take_block(call)
            except Exception:  # noqa: BLE001 - delivery never stops a helper
                block = None
            if block:
                messages.append({"role": "user", "content": block})
        if ctx.usd_room is not None and not ctx.usd_room():  # a compaction call may have used the room
            doc["status"] = "stopped: the writer service's room for this pass reached"
            return
        # helpers' model calls run at the same time: each helper waits only for its own call
        pr.calls_in_flight.add(call)
        from .writer_pass import call_in_flight, call_retrying, unbooked_cost

        async def one_call() -> tuple[dict, str]:
            with call_in_flight(ctx.spend):
                return await asyncio.wait_for(
                    turn(messages, tools),
                    timeout=max(0.1, ctx.deadline - time.monotonic()),
                )

        def book_unpriced() -> None:
            doc["unknown_cost_calls"] += 1
            ctx.spend.unknown += 1

        try:
            # a transient refusal (the route busy, a rate limit, a server error) is tried again, bounded
            msg, usd = await call_retrying(
                one_call,
                book_unpriced=book_unpriced,
                remaining=lambda: ctx.deadline - time.monotonic(),
                room=ctx.usd_room,
            )
        except asyncio.CancelledError:
            # cut mid-flight (stop, or the wall clock at the end of the pass): it may still have cost money
            doc["calls_cut"] += 1
            doc["unknown_cost_calls"] += 1
            ctx.spend.unknown += 1
            raise
        except asyncio.TimeoutError:
            doc["calls_cut"] += 1
            doc["unknown_cost_calls"] += 1
            ctx.spend.unknown += 1
            doc["status"] = "pass wall clock reached during a model call"
            return
        except Exception as exc:
            if unbooked_cost(exc):  # a failed call may still have cost money (not one the route refused unsent)
                doc["unknown_cost_calls"] += 1
                ctx.spend.unknown += 1
            raise
        finally:
            pr.calls_in_flight.discard(call)
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
        from .writer_pass import split_usage

        # the call's usage report stays with this turn; the next request never carries it
        msg, reported = split_usage(msg)
        last_prompt = (
            reported.get("prompt_tokens") if isinstance(reported, dict) else None
        )
        if isinstance(last_prompt, int) and not isinstance(last_prompt, bool):
            doc["max_prompt_tokens"] = max(doc["max_prompt_tokens"] or 0, last_prompt)
        messages.append(msg)
        calls = msg.get("tool_calls") or []
        calls = calls if isinstance(calls, list) else []
        if not calls:
            messages.append(
                {
                    "role": "user",
                    "content": "Use your tools, then call done(summary, claims).",
                },
            )
            no_progress.after_call(messages)  # the nudge starts the count again
            continue
        usage = _usage_of(msg, usd, reported)
        finished = False
        for i, tc in enumerate(calls):
            tc = tc if isinstance(tc, dict) else {}
            cid = (
                tc.get("id")
                if isinstance(tc.get("id"), str) and tc.get("id")
                else f"call_{doc['turns']}_{i}"
            )
            fn = tc.get("function") if isinstance(tc.get("function"), dict) else {}
            name = str(fn.get("name", ""))
            args = _args(fn.get("arguments"))
            paths: list[dict] = []
            if args is None:
                content = refusal(
                    GENERAL,
                    "bad_arguments",
                    reason="unreadable arguments",
                )
                status = "refused"
            elif finished:
                content, status = refusal(GENERAL, "task_done"), "refused"
            elif name != "execute_code":
                content = refusal(
                    GENERAL,
                    "unknown_tool",
                    name=str(name)[:80],
                    tools="execute_code",
                )
                status = "refused"
            else:
                st["n"] = n + 1  # the cell's row keeps n; its in-code calls are numbered after it
                try:
                    content, paths = await _tool(ctx, name, args, st, resolve)
                    status = "refused" if form_of(content) else "ok"
                except (
                    Exception
                ) as exc:  # noqa: BLE001 - a broken tool call is answered, never ends the helper
                    content, status = f"error: {type(exc).__name__}", "error"
            row = tool_event(
                ctx,
                call,
                n,
                name,
                status,
                content,
                args or {},
                paths,
                usage if i == 0 else None,
            )
            if name == "execute_code":
                record_box_opens(ctx, row, st.pop("box_audit", None) or None)
            if name in RECORD_TOOLS:
                row["messaging"] = st.pop("messaging", None) or {}
            row.update(
                via="model",
                cell=None,
                args_sha256=_args_digest(args or {}),
                box_gone=False,
            )
            _emit(ctx, row)
            n += 1
            n = max(n, st.get("n", 0))
            if st["in_code"].finished and not finished and name == "execute_code":
                finished = True  # tools.done was accepted in this cell
            # plain text: a reply may be a str subclass carrying its refusal form
            messages.append(
                {"role": "tool", "tool_call_id": cid, "content": str(content)},
            )
        if finished:
            doc["status"] = "ok"
            no_progress.after_call(messages, act=False)
            doc["no_progress"] = no_progress.report()
            return
        acted = no_progress.after_call(messages)
        doc["no_progress"] = no_progress.report()
        if acted == "end":
            doc["status"] = no_progress.note(acted)  # "no progress: ..."
            return
    doc["status"] = "operational step guard"


def _args_digest(args: dict) -> str:
    text = json.dumps(args, sort_keys=True, ensure_ascii=False, separators=(",", ":"), default=str)
    return hashlib.sha256(text.encode("utf-8", "replace")).hexdigest()


def _helper_box(ctx: DelegateContext, st: dict) -> tuple[dict, Any]:
    """The binds and profile of the helper's interpreter box, with tools in code on or off: the writer's /inputs,
    /task and its /memory copy read-only, /work read-write and the cwd; the file-size limit; the names bound in every
    cell (``memory`` where the copy holds the package)."""
    from . import tool_handlers as _th

    binds = {
        "ro": {st["hin"]: "/inputs", st["task"]: "/task", st["lib"]: "/memory"},
        "rw": {st["work"]: "/work"},
    }
    profile = _th.memory_profile(
        pass_id=ctx.pass_id,
        cwd="/work",
        sys_path=("/inputs", "/memory"),
        fsize_bytes=WORK_FILE_BYTES,
        # the helper loop stamps and records the open rows with each cell's event, as for a fresh cell
        opens_path=None,
        # /work's limits are checked after each cell by the helper loop (work_over), as today
        measure=None,
        globals=_th.memory_globals("helper", st["lib"]),
    )
    return binds, profile


def _in_code_session(ctx: DelegateContext, call: str, st: dict, doc: dict) -> Any:
    """Tools in code: the helper's handlers' state and its interpreter session (its box: the writer's /inputs,
    /task and its /memory copy read-only, /work read-write), kept on *st*."""
    from unify.common.tool_channel import HandlerThreads

    from . import tool_handlers as _th

    hc = _th.HelperCalls(ctx=ctx, call=call, st=st, resolve=None, doc=doc)
    allow = _registry.allow_list("helper", helper_flags(ctx.similar is not None))

    def on_event(row: dict) -> None:
        # the helper's rows: the pass's events file, numbered in the helper's own sequence
        if row.get("event") == "tool_call":
            row["call"] = st.get("n", 0)
            st["n"] = row["call"] + 1
            row["helper_call"] = call
        _emit(ctx, row)

    binds, profile = _helper_box(ctx, st)
    pr = ctx.pass_record
    session = _th.InterpreterSession(
        record=pr.view(call) if pr is not None else None,
        agent="helper",
        name=call,
        role="helper",
        pass_id=ctx.pass_id,
        state=hc,
        allow=allow,
        binds=binds,
        env={},
        profile=profile,
        roots={"/inputs": st["hin"], "/task": st["task"], "/memory": st["lib"], "/work": st["work"]},
        deadline=ctx.deadline,
        cell_limit_s=ctx.cell_timeout_s,
        on_event=on_event,
        threads=ctx.threads if ctx.threads is not None else HandlerThreads(),
        start=ctx.interpreter_start,
    )
    hc.kill_box = session.kill
    hc.threads = session.threads
    st["in_code"] = hc
    st["session"] = session
    return session


async def _in_code_cell(ctx: DelegateContext, st: dict, code: str, left: float) -> str:
    """One helper cell in its interpreter, its in-code calls served by the channel: the head of the output, then the
    host's notes."""
    from . import tool_handlers as _th

    hc = st["in_code"]
    session = st["session"]
    hc.finish_text = None
    cell_no = st["n"] - 1
    note = session.take_note()
    res = await session.cell(code, cell_no)
    audit = getattr(res, "opens", None)
    st["box_audit"] = audit if isinstance(audit, dict) else {}
    body, tail = _th.cell_parts(res, hc.finish_text, session.gone_during(), note)
    data = body.encode("utf-8", "replace")
    if len(data) > OUTPUT_BYTES:
        body = (
            data[:OUTPUT_BYTES].decode("utf-8", "replace")
            + f"\n[… {len(data)} bytes in all; head shown]"
        )
    return body + tail


async def handle(args: dict, ctx: DelegateContext, info: dict | None = None) -> str:
    """The ``delegate`` tool: start one helper and return at once with its name, or the refusal that started none.
    The helper runs as a task of the pass record's pool; its messages and its last text arrive on the record.
    """
    task = args.get("task") if isinstance(args, dict) else None
    eps = args.get("episodes") if isinstance(args, dict) else None
    if not isinstance(task, str) or not task.strip():
        return _refusal(SPEC, "no_task")
    if not isinstance(eps, list) or not all(isinstance(e, str) for e in eps):
        return _refusal(SPEC, "no_episodes")
    pr = ctx.pass_record
    if ctx.new_turn is None or pr.pool is None:
        return _refusal(SPEC, "no_lane", why=ctx.lane_refusal)
    if ctx.deadline - time.monotonic() <= 1.0:
        return _refusal(SPEC, "no_time")
    episodes = list(dict.fromkeys(eps))
    try:
        index = set(ctx.recorded())
    except Exception as exc:  # noqa: BLE001 - an unreadable index starts no helper
        return _refusal(SPEC, "index_unreadable", error=type(exc).__name__)
    for e in episodes:
        if not _EPISODE_ID.match(e) or e not in index:
            return _refusal(SPEC, "unknown_episode", episode=e[:80])
    request = f"Task from the writer: {task}\nEpisodes to start from: {', '.join(episodes) or '(none)'}"
    call, failed = await _spawn_named(ctx, pr, task, episodes, request)
    if failed is not None:
        ctx.pending.pop(call, None)
        if call not in pr.record.participants:
            pr.roles.pop(call, None)
            pr.help.pop(call, None)
            pr.ended.pop(call, None)
        else:
            ctx.next_call += 1  # the name was taken by the record; it is never reused
        form = (
            "not_allowed"
            if isinstance(failed, PermissionError)
            else "helper_limit" if isinstance(failed, RuntimeError) else "bad_request"
        )
        return _refusal(SPEC, form, why=str(failed)[:500])
    ctx.next_call += 1
    if info is not None:
        info.update(helper=call)
    return (
        f"helper {call} started; its messages arrive in the record: use wait or messages. "
        f"Running now: {', '.join(pr.running())}."
    )


def _run_lines(call: str, starter: str) -> str:
    """The common section and this helper's roster lines, after what it was asked to do (the run record)."""
    from unify.agents.run_record import prompt_lines

    return "\n\n" + prompt_lines(call, "helper", starter)


#: How many names a helper start on the run record tries when other processes keep taking the one it picked.
_RUN_NAME_TRIES = 20


async def _spawn_named(
    ctx: DelegateContext,
    pr: _messaging.PassRecord,
    task: str,
    episodes: list[str],
    request: str,
) -> tuple[str, Exception | None]:
    """Start the helper under its name: ``call-<n>`` of this pass, or on the run record
    the first ``call-<k>`` no agent of the whole run has had, trying the next one when another process takes it
    first. Returns ``(name, None)``, or the name tried last and why none started."""
    tries = _RUN_NAME_TRIES
    call = ""
    for attempt in range(tries):
        call = pr.record.next_name("call-")
        pr.add_helper(call, task)
        ctx.pending[call] = (task, episodes)
        try:
            await pr.pool.spawn(pr.root, request, name=call)
        except PostRefused as exc:
            if "is taken" in str(exc) and attempt + 1 < tries and call not in pr.record.participants:
                ctx.pending.pop(call, None)
                pr.roles.pop(call, None)
                pr.help.pop(call, None)
                pr.ended.pop(call, None)
                continue
            return call, exc
        except (PermissionError, RuntimeError, ValueError) as exc:
            return call, exc
        # let the helper's task take its first step, so even a stop right away finds its body running
        await asyncio.sleep(0)
        return call, None
    return call, RuntimeError("no free helper name could be taken")


async def _body(ctx: DelegateContext, call: str) -> str:
    """One helper, from its box to its ``helper.json``: the pool runs this as the helper's task and posts what it
    returns as the helper's last entry. Never raises (cancellation aside); ``helper.json`` and the staged files are
    written on every path, a stop included."""
    task, episodes = ctx.pending.pop(call)
    pr = ctx.pass_record
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
        "claims": [],
        # done, stop, finish or wall_clock (error, step_guard or no_progress when it ended so); model calls cut
        # mid-flight
        "stopped_by": None,
        "calls_cut": 0,
        # the largest prompt (tokens) one of its model calls reported (None: none reported one)
        "max_prompt_tokens": None,
        # the shared no-progress rule (loop_stop.AgentNoProgress.report): {"k", "notices", "ended", "at_call"}, or
        # None when no notice was given; "ended": 2 * k no-progress model calls in a row ended the helper
        "no_progress": None,
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
        doc["stopped_by"] = _stopped_by(pr, call, doc["status"])
    except asyncio.CancelledError:
        doc["status"] = f"stopped: {pr.stopping.get(call, 'cancelled')}"
        doc["stopped_by"] = pr.stopped_by.get(call, "stop")
        raise
    except Exception as exc:  # noqa: BLE001 - one helper's failure never stops the pass
        doc["status"] = f"error: {type(exc).__name__}"
        doc["stopped_by"] = "error"
    finally:
        if tmp is not None:
            shutil.rmtree(tmp, ignore_errors=True)
        row = _finish(ctx, call, staging, doc, time.monotonic() - started)
        ctx.replies[call] = _reply(row)
        event = pr.ended.get(call)
        if event is not None:
            event.set()
    return ctx.replies[call]


def _stopped_by(pr: _messaging.PassRecord, call: str, status: str) -> str:
    """What ended a helper whose body returned: done, stop, finish, wall_clock, no_progress, error or step_guard."""
    if status == "ok":
        return "done"
    if status.startswith("no progress: "):
        return "no_progress"
    if status.startswith("stopped: "):
        return pr.stopped_by.get(call, "finish")
    if status.startswith("pass wall clock"):
        return "wall_clock"
    if status == "operational step guard":
        return "step_guard"
    return "error"


def _finish(
    ctx: DelegateContext,
    call: str,
    staging: _Staging | None,
    doc: dict,
    seconds: float,
) -> dict:
    """Write ``helper.json`` last, export to ``/inputs/helpers/<name>/``, and record the call's row."""
    usd = doc.pop("_usd")
    files = []
    if staging is not None:
        for rel, size in sorted(staging.files.items()):
            try:
                data = staging.read(rel)
            except OSError:
                continue
            files.append(
                {
                    "path": rel,
                    "bytes": size,
                    "sha256": hashlib.sha256(data).hexdigest(),
                },
            )
    if ctx.pool is not None:
        doc["related_closed"] = _related_closed(
            ctx.pool,
            call,
            staging,
            doc.get("episodes") or [],
        )
    doc.update(
        usd=_usd(usd) if not doc["unknown_cost_calls"] else _UNKNOWN,
        usd_known=_usd(usd),
        seconds=round(seconds, 3),
        files=files,
        summary=doc.get("summary", ""),
        claims=doc.get("claims", []),
    )
    row = dict(doc)
    if staging is not None:
        try:
            _write_status(staging.path, doc)
            _export(ctx, call, staging, doc)
            row["exported"] = f"/inputs/{EXPORT_DIR}/{call}/"
        except OSError as exc:
            row["status"] = (
                f"{doc['status']}; output not exported ({type(exc).__name__})"
            )
    ctx.records.append(row)
    return row


def _related_closed(
    pool: Any,
    call: str,
    staging: _Staging | None,
    episodes: list[str],
) -> dict:
    """The closed pool entries resembling what helper *call* staged (its ``.py`` files and the added Python lines of
    its ``.diff`` files, labelled ``helper:<name>/<file>``) or citing one of its episodes: ``{"count", "entries"}``
    with the first :data:`.pool.RELATED_PAGE` links, or ``{"error"}`` when they cannot be computed.
    """
    from . import pool as _pool

    try:
        code: dict[str, frozenset[str]] = {}
        for rel in sorted(staging.files) if staging is not None else []:
            if not rel.endswith((".py", ".diff")):
                continue
            text = staging.read(rel).decode("utf-8", "replace")
            label = f"helper:{call}/{rel}"
            if rel.endswith(".py"):
                code[label] = _pool.code_units(text)
            else:
                for _path, runs in sorted(_bm._diff_python_runs(text).items()):
                    for _start, lines in runs:
                        code[label] = code.get(label, frozenset()) | _pool.code_units(
                            textwrap.dedent("\n".join(lines)),
                        )
        links = pool.related_closed(code, episodes)
    except (
        Exception
    ) as exc:  # noqa: BLE001 - the links are reported missing, never shown as none
        return {"error": f"{type(exc).__name__}: {str(exc)[:200]}"}
    return {"count": len(links), "entries": links[: _pool.RELATED_PAGE]}


def _reply(row: dict) -> str:
    lines = [f"helper {row['call']}: {row['status']}"]
    if row.get("exported"):
        lines.append(f"files under {row['exported']}:")
        lines += [f"- {f['path']} ({f['bytes']} bytes)" for f in row["files"]] or [
            "- (none staged)",
        ]
    lines.append(
        f"spend: {row['usd']} USD"
        + (
            f" ({row['unknown_cost_calls']} unpriced calls; {row['usd_known']} USD known)"
            if row["unknown_cost_calls"]
            else ""
        )
        + f"; {row['turns']} model calls; {row['seconds']:.0f} s of the pass's wall clock",
    )
    rel = row.get("related_closed")
    if rel is not None:
        if "error" in rel:
            lines.append(
                f"closed pool entries resembling this output: could not be read ({rel['error']})",
            )
        elif rel["count"]:
            lines.append(
                f"closed pool entries resembling this output ({rel['count']}; all in helper.json, "
                f"`pool show <id>` for each):",
            )
            lines += [
                f"- {e['id']} ({e['status']}, pass {e.get('closed_in')}): {e.get('reason') or ''}"
                for e in rel["entries"]
            ]
    lines.append("summary: " + (row.get("summary") or "(none)"))
    lines.append(claims_text(row.get("claims") or []))
    return "\n".join(lines)
