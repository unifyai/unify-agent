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

Where its output goes: ``<state>/memory-helpers/<pass>/<call>/`` holds the staged files and ``helper.json``
(status, usage, the block ledger, the task digest, the claims with their sources), written last. The same files
are exported to the writer's ``/inputs/helpers/<call>/``, and the ``delegate`` reply lists them with the helper's
summary, its claims with their sources, and its spend. The writer is told that a claim is untrusted until it has
read the parts it points to.

Model calls: a helper's model calls are made host-side, by the writer's process, through the turn factory the
pass is given (the key proxy's helper lane when the run has one), never from inside a helper's box, which holds no
credential. Helpers run at the same time and so do their model calls. Unknown cost is counted as unknown (fail
closed) on the pass's spend and in ``helper.json``.

Messaging (:mod:`.messaging`): ``delegate`` starts the helper through the pass record's pool and returns at once with
its name (its call id, ``call-<n>``). The helper and the writer talk on the pass's record (``send``, ``ask``,
``messages``, ``wait``, ``who``); the helper's ``done`` text is its last entry. A helper ends when it calls done,
when the writer stops it, when the writer finishes (a model call in flight then finishes first and no new one
starts) or at the pass's wall clock, the outer backstop. Whichever way it ends, ``helper.json`` and its staged
files are written and kept.

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
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Awaitable, Callable, Iterable

from unify.agents.pool import Pool

from . import batch_map as _bm
from . import messaging as _messaging
from . import views as _views
from . import writer_tools as _wt
from .integration.fork_worker import QUOTA_BYTES, QUOTA_FILES, allowed
from .library_prompts import CANDIDATE_CONTRACT
from . import sandbox_run as _sandbox
from .sandbox_run import PRLIMIT, PYTHON, run_confined
from .tool_spec import GENERAL, REFUSED, Refusal, ToolSpec, form_of, refusal

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
#: A helper cell's time, at most (less when the pass's wall clock has less left).
CELL_TIMEOUT_S = 60.0
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
    "reads it; without an episode it lists the batch. `read_episode` without parts lists the parts.\n"
    "- `read` and `grep` read /inputs, the writer's own inputs as the writer sees them: the batch map "
    "(/inputs/batch_map.json), the batch's episode records (/inputs/episodes/<id>.json), the shared request "
    "blocks (/inputs/shared/<id>.txt; a `[shared:<id>]` marker in a record stands for that block, which "
    "`read_episode` also reads as the part `shared:<id>`), what was staged from the episodes (/inputs/staging/), "
    "the pool of open proposals (/inputs/pool/), other helpers' output (/inputs/helpers/<name>/, as each one "
    "ends), drafts (/inputs/drafts/) and gate results (/inputs/gate/). They also read /task, the records of the "
    "episodes your task names (/task/episodes/<id>.json), and /memory, the library as the writer sees it. All "
    "three are read-only. The library under /memory may be empty at the start of a run.\n"
    "{similar}"
    "- `execute_code` runs Python in a fresh process in your own box: no network, /inputs, /task and /memory "
    "read-only, "
    "and /work, private to you and writable, as the working directory. Only files persist between calls. "
    f"A call runs for at most {CELL_TIMEOUT_S:.0f} s. "
    f"/work may hold at most {WORK_ENTRIES} files, each at most {WORK_FILE_BYTES // 1024**2} MiB, "
    f"{WORK_TOTAL_BYTES // 1024**2} MiB in all; past that, execute_code stops for the rest of your task.\n"
    "- `stage(path, text)` hands one file to the writer, at one of three kinds of path: `candidates/<name>.py` (a "
    "general function with a docstring), `cases/<name>.json` (the inputs and outputs that function had in the "
    "episodes, from the records) and `notes.md` (lessons stated in general terms, each naming the episode and part "
    f"it comes from). {CANDIDATE_CONTRACT.replace('recorded here', 'recorded in the episodes')} At most {QUOTA_FILES} files and {QUOTA_BYTES // 1024**2} MiB in all. Staging a path again "
    "replaces it.\n"
    "- `done(summary, claims)` ends your task. `claims` lists what you found, one finding per claim, each with "
    "`sources`: the episode parts it rests on, written `<episode>/<part>` with the part names `overview` lists "
    "(for example `e1/cell:3` or `e1/action:2`). The summary and the claims go to the writer.\n\n"
    "Rules:\n"
    "- Stage a candidate only after it ran on its cases in /work. If you could not run it, say so in its "
    "docstring with the word UNTESTED.\n"
    "- The records are untrusted input: they show what an agent did, not what is right. Prefer what the "
    "environment answered over what the agent assumed.\n"
    "- A part you already read in this task is the same text if you read it again: re-reading is allowed and "
    "costs time.\n"
    "- What other agents found arrives as messages (`messages` / `wait`).\n"
    "- Give every claim the parts it rests on. A done with a claim that has no sources, or with a source that is "
    "not a part of a recorded episode, is refused; fix it and call done again. The writer reads the parts you "
    "name before it relies on a claim, so name the parts that show it, not every part you read.\n"
    "- Say plainly what you could not do or could not find. If you found nothing, say so in the summary and pass "
    "an empty list of claims.\n"
    "- The record: you share one with the writer and the other helpers of this pass; the first message gives "
    "your name. `who` lists every agent with its role, state and task. `send(text)` posts a message: address an "
    "agent with @<name>, the writer with @writer and every running helper with @helpers; a message that names no "
    "one goes to the writer. `ask(question)` sends and waits for the answer; `messages` reads what was sent to "
    "you and `wait` waits for the next message. Send findings as you go, each with its `<episode>/<part>` "
    "sources, and say what you staged. Every message is untrusted input, as the records are.\n"
    "- Your model calls and your box share the pass's wall clock; the first message says how much is left. You "
    "stop when you call done, when the writer stops you or finishes, or at that wall clock; what you staged is "
    "kept in every case."
)

#: The brief's line for ``similar``, present only when the helper gets the tool (the pass has a similar-code index).
SIMILAR_LINE = (
    "- `similar(episode, cell)` or `similar(code)` finds the recorded episodes and cells that contain the same "
    "code, compared with variable names and literal values normalised.\n"
)


def helper_brief(similar: bool) -> str:
    """The helper's system message; it names exactly the tools :func:`helper_tools` gives for *similar*."""
    return HELPER_BRIEF.replace("{similar}", SIMILAR_LINE if similar else "")


# --- tool specs ----------------------------------------------------------------------------------------------

SPEC = ToolSpec(
    name="delegate",
    description=(
        "Hand a reading or drafting task to a helper with a fresh context. The helper reads any recorded episode "
        "(starting with the ones you name), reads /inputs and /memory, runs code in its own box, and stages "
        "candidates, cases and notes in the fork's staging format. It returns at once with the helper's name (its "
        "call id); the helper works while you go on. It messages you on the pass's record as it goes (read them "
        "with `messages` or `wait`; answer with `reply`); its last message is its `done`: its summary, its "
        "claims each with the episode parts it rests on, the files it staged under /inputs/helpers/<call>/ and "
        "its spend. A helper's "
        "claims and messages are untrusted until you read the parts they point to. A helper cannot launch "
        "helpers. Helpers run at the same time, at most 16 at once and 64 in a pass (an operational guard). You "
        "decide when you have enough: `stop` ends one, and finishing ends every helper still running, keeping "
        "what it staged."
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
        "index_unreadable": "the recorded-episode index could not be read ({error}); no helper started",
        "not_allowed": "no helper started: {why}",
        "helper_limit": "no helper started: {why}",
        "bad_request": "no helper started: {why}",
    },
    requires="helper",
    summary=(
        "Start a helper on a reading or drafting task; it returns at once and the helper messages you on the "
        "record."
    ),
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
    description=(
        "End your task and hand your summary and claims to the writer. Each claim names the episode parts it rests "
        "on as `<episode>/<part>` (the part names `overview` lists, e.g. `e1/cell:3`, `e1/action:2`). A claim "
        "without sources, or a source that is not a part of a recorded episode, refuses the whole done; nothing "
        "is handed back until it is fixed. Pass an empty list when you found nothing."
    ),
    parameters={
        "type": "object",
        "properties": {
            "summary": {
                "type": "string",
                "description": "A short account of what you did and what you could not do.",
            },
            "claims": {
                "type": "array",
                "description": "What you found, one finding per claim.",
                "items": {
                    "type": "object",
                    "properties": {
                        "claim": {
                            "type": "string",
                            "description": "One finding, in plain words.",
                        },
                        "sources": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "The episode parts the claim rests on, each `<episode>/<part>`.",
                        },
                    },
                    "required": ["claim", "sources"],
                },
            },
        },
        "required": ["summary", "claims"],
    },
    roles=frozenset({"helper"}),
    refusals={
        "claims": "claims must be a list of objects with a non-empty `claim` and a list of `sources` ({why}); "
        "the task is not done, call done again",
        "no_sources": "claim {n} has no sources: give it the episode parts it rests on, each written "
        "`<episode>/<part>` (for example `e1/cell:3`); the task is not done, call done again",
        "bad_source": "claim {n}: source {source!r} does not resolve ({why}); name a recorded episode and one of "
        "the parts its overview lists, as `<episode>/<part>`; the task is not done, call done again",
    },
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
    refusals={
        "work_full": "not run: {why}; execute_code is closed for the rest of this task",
        "no_time": "not run: the pass's wall clock is used up",
        "stopping": "not run: you are asked to stop ({reason})",
    },
)

#: The helper's own read and grep: its roots (/inputs, /task, /memory and /work; no /outputs), no per-turn count.
READ_SPEC = ToolSpec(
    name="read",
    description=(
        "Read a file or list a directory under /inputs, /task, /memory or /work. Returns at most 8000 bytes from "
        "offset; a marker gives the next offset. Nothing is ever cut silently. A path that does not exist is "
        "answered with what its directory holds."
    ),
    parameters={
        "type": "object",
        "properties": {
            "path": {"type": "string"},
            "offset": {"type": "integer", "description": "byte offset to start the page at"},
        },
        "required": ["path"],
    },
    roles=frozenset({"helper"}),
    refusals={
        "outside_box": "{path} is outside /inputs, /task, /memory and /work",
        "not_found": "{path} does not exist",
        "unreadable": "{error} for {path!r}",
    },
)
GREP_SPEC = ToolSpec(
    name="grep",
    description=(
        "Search files under a path (default /inputs) of /inputs, /task, /memory or /work with a Python regular "
        "expression; returns path:line: text hits, one bounded page at a time (a marker gives the next offset)."
    ),
    parameters={
        "type": "object",
        "properties": {
            "pattern": {"type": "string"},
            "path": {"type": "string", "description": "file or directory to search; default /inputs"},
            "offset": {"type": "integer", "description": "hit number to start the page at"},
        },
        "required": ["pattern"],
    },
    roles=frozenset({"helper"}),
    refusals={
        "outside_box": "{path} is outside /inputs, /task, /memory and /work",
        "not_found": "{path} does not exist",
        "bad_pattern": "{reason}",
        "unreadable": "{error} for {path!r}",
    },
)

#: Specs answering as a registered tool name for the helper only (their refusal forms are listed by
#: :func:`.tool_spec.refusal_codes` under that name).
VARIANT_SPECS = (EXECUTE_CODE_SPEC, READ_SPEC, GREP_SPEC)

#: The tools a helper is answered for. The readers' schemas come from the shared registry (role ``helper``).
READERS = ("overview", "read_episode", "similar", "read", "grep")
#: The record tools a helper gets (:mod:`.messaging`).
RECORD_TOOLS = frozenset(s.name for s in _messaging.SPECS if "helper" in s.roles)


def helper_tools(similar: bool = True) -> list[dict]:
    """The helper's tool JSON: the registry's ``helper`` tools (its readers, ``stage``, ``done``) with the helper's
    own ``execute_code`` in place of any other. *similar*: the pass has a similar-code index, so the helper gets
    ``similar`` too. A helper never gets ``delegate`` (a writer-only tool)."""
    from .tool_spec import tools_for

    flags = frozenset({"helper", "similar"} if similar else {"helper"})
    own = {s.name: s for s in VARIANT_SPECS if s.name != "execute_code"}
    specs = [own.get(s.name, s) for s in tools_for("helper", flags) if s.name != "execute_code"]
    names = {s.name for s in specs}
    for s in (STAGE_SPEC, DONE_SPEC):
        if s.name not in names:
            specs.append(s)
    specs.append(EXECUTE_CODE_SPEC)
    return [s.as_tool() for s in specs]


def _refusal(spec: ToolSpec, form: str, **fields: object) -> str:
    return refusal(spec.name, form, **fields)


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
    * ``deadline``: the pass's wall clock end (``time.monotonic()``), the outer backstop of every helper;
      ``spend``: the pass's spend (an object with
      ``usd`` and ``unknown``): helper spend joins it as it is incurred;
    * ``shared``: the shared request blocks (id -> text); ``events``: the pass's ``tool-events.jsonl`` (or None);
    * ``episode_reader``: the writer's episode reader for ``overview``/``read_episode`` if the pass has one
      (``async (name, args, resolve) -> str``), else the plain reader here; ``similar``: the ``similar`` tool's
      handler (``async (args) -> str``), or None;
    * ``records``: one row per delegate call, appended here (the pass outcome's ``helpers``);
    * ``pass_record``: the pass's record (:class:`.messaging.PassRecord`; made here when not given), whose pool
      starts the helpers.
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
    cell_timeout_s: float = CELL_TIMEOUT_S
    records: list[dict] = field(default_factory=list)
    #: host-stamped open rows from the helpers' boxes (also appended to ``box-opens.jsonl`` beside ``events``)
    box_opens: list[dict] = field(default_factory=list)
    next_call: int = 0
    pass_record: _messaging.PassRecord | None = None
    #: each started helper's task and episodes, until its body takes them
    pending: dict[str, tuple[str, list[str]]] = field(default_factory=dict)
    #: each ended helper's last text (its ``done`` reply, or how it ended)
    replies: dict[str, str] = field(default_factory=dict)
    #: each helper's message list and tool JSON, as it grows (recorded with the writer's transcript)
    transcripts: dict[str, tuple[list, list]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.pass_record is None:
            self.pass_record = _messaging.PassRecord(root="writer", deadline=self.deadline)
        pr = self.pass_record
        if pr.pool is None and self.new_turn is not None:

            async def start(name: str, spawner: str, request: str, seq: int) -> str:
                return await _body(self, name)

            pr.pool = Pool(pr.record, start_helper=start, spawn_allowed=True)


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
    pass_record: _messaging.PassRecord | None = None,
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
        pass_record=pass_record,
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


def box_cell(
    code: str,
    hin: Path,
    lib: Path,
    work: Path,
    cells: Path,
    timeout_s: float,
    audit: dict | None = None,
    task: Path | None = None,
) -> str:
    """*code* in the helper's box: the writer's box profile (:func:`.sandbox_run.run_confined`), with ``/inputs``
    (the writer's own, live), ``/task`` and ``/memory`` read-only and ``/work`` read-write; the code is a fresh
    read-only bind, never argv.
    *audit*, when given, receives the call's open log (:meth:`.sandbox_run.BoxOpenLog.finish`).
    """
    cdir = Path(tempfile.mkdtemp(prefix="cell-", dir=cells))
    open_log = _sandbox.BoxOpenLog()
    try:
        (cdir / "code.py").write_bytes(code.encode("utf-8", errors="replace"))
        r = run_confined(
            [
                PRLIMIT,
                f"--fsize={WORK_FILE_BYTES}:{WORK_FILE_BYTES}",
                "--",
                str(PYTHON),
                "-s",  # no user site-packages
                "/cell/code.py",
            ],
            ro={
                hin: "/inputs",
                **({task: "/task"} if task is not None else {}),
                lib: "/memory",
                cdir: "/cell",
            },
            rw={work: "/work"},
            cwd="/work",
            timeout_s=timeout_s,
            env={"PYTHONPATH": "/inputs:/memory", **open_log.env},
            pass_fds=open_log.pass_fds,
        )
    finally:
        result = open_log.finish()
        if audit is not None:
            audit.update(result)
        shutil.rmtree(cdir, ignore_errors=True)
    content = (r.stdout + ("\n" + r.stderr if r.stderr else "")) or "(no output)"
    data = content.encode("utf-8", "replace")
    if len(data) > OUTPUT_BYTES:
        content = (
            data[:OUTPUT_BYTES].decode("utf-8", "replace")
            + f"\n[… {len(data)} bytes in all; head shown]"
        )
    if r.timed_out:
        content += f"\n(cell timed out after {timeout_s:.0f} s)"
    elif r.returncode != 0:
        content += f"\n(exit status {r.returncode})"
    return content


# --- episode readers -----------------------------------------------------------------------------------------


def _part_names(ep: Any) -> list[str]:
    """The part names ``read_episode`` reads for *ep* (the writer's own list, so a helper and the writer agree)."""
    return _wt.part_names(ep)


def _contents(ep: Any, eid: str, shared: dict[str, str]) -> str:
    lines = [f"{eid}: parts (read them with read_episode(episode, parts=[...]))"]
    for name in _part_names(ep):
        try:
            size = len(_bm.part_text(ep, name, shared).encode())
        except (KeyError, ValueError, IndexError, StopIteration):
            continue
        lines.append(f"- {name}: {size} bytes")
    return "\n".join(lines)


def episode_paths(name: str, args: dict, eid: str) -> list[dict]:
    """The episode paths an answered ``overview`` or ``read_episode`` call read, as the writer's events name them:
    ``/episodes/<id>`` for the overview or the table of contents, ``/episodes/<id>/<part>`` for each named part.
    """
    wanted, _ = _wt._wanted_parts(args) if name == "read_episode" else (None, None)
    if not wanted:
        return [{"path": f"/episodes/{eid}", "mode": "read"}]
    names = [str(p.get("part")) if isinstance(p, dict) else str(p) for p in wanted]
    return [{"path": f"/episodes/{eid}/{p.strip()}", "mode": "read"} for p in names]


async def plain_episode_reader(
    name: str,
    args: dict,
    ep: Any,
    eid: str,
    shared: dict[str, str],
) -> str:
    """``overview(episode)`` and ``read_episode(episode, parts?, offset?)`` on one resolved episode."""
    offset = args.get("offset", 0) or 0
    if isinstance(offset, bool) or not isinstance(offset, int) or offset < 0:
        return refusal(
            name,
            "bad_arguments",
            reason="offset must be an integer >= 0; nothing was read",
        )
    if name == "overview":
        text, _ = _bm.solution_first(ep, shared)
        return _views.view(text.encode(), offset)
    # the writer's own parsing: "" and [] mean the table of contents, {"part", "offset"} items continue a part
    wanted, problem = _wt._wanted_parts(args)
    if problem is not None:
        return refusal("read_episode", problem)
    if wanted is None:
        return _views.view(_contents(ep, eid, shared).encode(), offset)
    items: list[tuple[str, int, bytes]] = []
    unknown: list[str] = []
    for item in wanted:
        part, off = (
            (item.get("part"), item.get("offset", 0)) if isinstance(item, dict) else (item, None)
        )
        part = str(part).strip()
        if off is None:
            off = offset if len(wanted) == 1 else 0
        if isinstance(off, bool) or not isinstance(off, int) or off < 0:
            return refusal("read_episode", "bad_offset", part=part[:80])
        try:
            items.append((part, off, _bm.part_text(ep, part, shared).encode()))
        except (KeyError, ValueError, IndexError, StopIteration, TypeError):
            unknown.append(part[:80])
    if unknown and not items:
        return refusal(
            "read_episode",
            "unknown_part",
            names=", ".join(repr(u) for u in unknown),
            episode=eid,
            valid=_wt.valid_names(ep),
        )
    out, left = [], _views.VIEW_BYTES
    if unknown:
        out.append(
            f"unknown part(s) in {eid}: {', '.join(repr(u) for u in unknown)}; valid: {_wt.valid_names(ep)}",
        )
    for part, off, data in items:
        if left <= 0:
            out.append(
                f"== {part} ==\nnot shown (page full): read again with parts={json.dumps([part])}",
            )
            continue
        text, a, b = _views.view_range(data, off, left)
        left -= b - a
        out.append(f"== {part} ==\n{text}")
    return "\n\n".join(out)


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
            # the resolved virtual path, as the writer's events record it
            real = _views.resolve(vpath, roots)
            virtual = _views._virtual(real, roots)
            if virtual.endswith("/."):
                virtual = virtual[:-2]
            if name == "read":
                mode = "list" if real.is_dir() else "read"
                return _views.read(vpath, roots, offset), [
                    {"path": virtual, "mode": mode},
                ]
            out = await asyncio.to_thread(
                _views.grep_bounded,
                str(args.get("pattern", "")),
                vpath,
                roots,
                offset,
            )
            return out, [{"path": virtual, "mode": "search"}]
        except PermissionError:
            return _own_refusal(name, "outside_box", path=vpath[:200]), []
        except _views.PatternProblem as exc:
            return _own_refusal("grep", "bad_pattern", reason=str(exc)[:300]), []
        except FileNotFoundError:
            missing = _own_refusal(name, "not_found", path=vpath[:200])
            return Refusal(str(missing) + _parent_listing(vpath, roots), name, "not_found"), []
        except (OSError, ValueError, TypeError) as exc:
            # the type and the path asked for only: the error's text names host paths
            return (
                _own_refusal(name, "unreadable", error=type(exc).__name__, path=vpath[:200]),
                [],
            )
    if name == "overview" and not args.get("episode"):
        return _batch_overview(ctx, args), [{"path": "/inputs/batch_map.json", "mode": "read"}]
    if name in ("overview", "read_episode"):
        got = resolve(args.get("episode"), name)
        if isinstance(got, str):
            return got, []
        ep, eid = got
        if ctx.episode_reader is not None:
            out = await ctx.episode_reader(name, args, resolve)
        else:
            out = await plain_episode_reader(name, args, ep, eid, ctx.shared)
        return out, ([] if form_of(out) else episode_paths(name, args, eid))
    if name == "similar":
        if ctx.similar is None:
            return refusal("similar", "not_available"), []
        out = await ctx.similar(args)
        eid = args.get("episode")
        named = isinstance(eid, str) and not form_of(out)
        return out, ([{"path": f"/episodes/{eid}", "mode": "search"}] if named else [])
    if name == "execute_code":
        if st["work_over"] is not None:
            return refusal("execute_code", "work_full", why=st["work_over"]), []
        stopping = ctx.pass_record.stopping.get(st["call"])
        if stopping is not None:
            return refusal("execute_code", "stopping", reason=stopping), []
        left = ctx.deadline - time.monotonic()
        if left <= 1.0:
            return refusal("execute_code", "no_time"), []
        audit: dict = {}
        st["box_audit"] = audit
        out = await _in_thread(
            box_cell,
            str(args.get("code", "")),
            st["hin"],
            st["lib"],
            st["work"],
            st["cells"],
            max(1.0, min(ctx.cell_timeout_s, left)),
            audit,
            st["task"],
        )
        over = work_over(st["work"])
        if over is not None:
            st["work_over"] = over
            out += f"\n({over}; execute_code is closed for the rest of this task)"
        return out, []
    tools = ", ".join(
        t["function"]["name"] for t in helper_tools(similar=ctx.similar is not None)
    )
    return refusal(GENERAL, "unknown_tool", name=str(name)[:80], tools=tools), []


def _own_refusal(tool: str, form: str, **fields: object) -> str:
    """A refusal from the helper's own read or grep spec (its roots), keeping its form code."""
    spec = READ_SPEC if tool == "read" else GREP_SPEC
    template = spec.refusals.get(form)
    if template is None:
        return refusal(tool, form, **fields)
    return Refusal(REFUSED + template.format(**fields), tool, form)


def _parent_listing(vpath: str, roots: dict) -> str:
    """What the nearest existing directory above *vpath* holds, for a path that does not exist (same form)."""
    parent = vpath.rstrip("/").rsplit("/", 1)[0] or "/"
    while parent not in ("", "/"):
        try:
            real = _views.resolve(parent, roots)
        except PermissionError:
            break
        if real.is_dir():
            names = sorted(x.name + ("/" if x.is_dir() else "") for x in real.iterdir())
            shown = names[:50] + ([f"... {len(names) - 50} more"] if len(names) > 50 else [])
            return f"; {parent} holds: " + (", ".join(shown) or "(nothing)")
        parent = parent.rsplit("/", 1)[0]
    return ""


def _batch_overview(ctx: DelegateContext, args: dict) -> str:
    """``overview`` without an episode: one line per batch episode, from the writer's /inputs/batch_map.json."""
    try:
        rows = json.loads((ctx.inputs / "batch_map.json").read_text())["episodes"]
    except (OSError, ValueError, KeyError, TypeError):
        return "this pass has no batch map; give `episode` to read one recorded episode"
    return _wt.overview_batch(rows if isinstance(rows, list) else [], args.get("offset", 0)).text


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
    session = f"helper.{ctx.pass_id}.{call.rsplit('-', 1)[-1]}"
    # a factory takes the per-call session label (the live lane) or nothing (scripted turns)
    takes_session = bool(inspect.signature(ctx.new_turn).parameters)  # type: ignore[arg-type]
    turn = ctx.new_turn(session) if takes_session else ctx.new_turn()  # type: ignore[misc]
    tools = helper_tools(similar=ctx.similar is not None)
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
            ),
        },
    ]
    ctx.transcripts[call] = (messages, tools)
    n = 0
    while doc["turns"] < ctx.step_guard:
        # asked to stop (the writer finished): no new model call starts; one in flight was allowed to finish
        reason = pr.stopping.get(call)
        if reason is not None:
            doc["status"] = f"stopped: {reason}"
            return
        if ctx.deadline - time.monotonic() <= 0:
            doc["status"] = "pass wall clock reached"
            return
        # helpers' model calls run at the same time: each helper waits only for its own call
        pr.calls_in_flight.add(call)
        try:
            msg, usd = await asyncio.wait_for(
                turn(messages, tools),
                timeout=max(0.1, ctx.deadline - time.monotonic()),
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
        except Exception:
            doc["unknown_cost_calls"] += 1  # a failed call may still have cost money
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
            continue
        usage = _usage_of(msg, usd)
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
            elif name == "done":
                try:
                    got = check_claims(args, resolve, ctx.shared)
                except (
                    Exception
                ) as exc:  # noqa: BLE001 - a broken check is answered, never ends the helper
                    got = f"error: {type(exc).__name__}"
                if isinstance(got, str):
                    content = got
                    status = "error" if got.startswith("error:") else "refused"
                else:
                    doc["summary"] = str(args.get("summary", ""))[:4000]
                    doc["claims"] = got
                    finished, content, status = True, "ok", "ok"
            else:
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
            _emit(ctx, row)
            n += 1
            # plain text: a reply may be a str subclass carrying its refusal form
            messages.append(
                {"role": "tool", "tool_call_id": cid, "content": str(content)},
            )
        if finished:
            doc["status"] = "ok"
            return
    doc["status"] = "operational step guard"


async def handle(args: dict, ctx: DelegateContext, info: dict | None = None) -> str:
    """The ``delegate`` tool: start one helper and return at once with its name, or the refusal that started none.
    The helper runs as a task of the pass record's pool; its messages and its last text arrive on the record."""
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
    call = f"call-{ctx.next_call}"
    request = f"Task from the writer: {task}\nEpisodes to start from: {', '.join(episodes) or '(none)'}"
    pr.add_helper(call, task)
    ctx.pending[call] = (task, episodes)
    try:
        await pr.pool.spawn(pr.root, request, name=call)
        # let the helper's task take its first step, so even a stop right away finds its body running
        await asyncio.sleep(0)
    except (PermissionError, RuntimeError, ValueError) as exc:
        ctx.pending.pop(call, None)
        if call not in pr.record.participants:
            pr.roles.pop(call, None)
            pr.help.pop(call, None)
            pr.ended.pop(call, None)
        else:
            ctx.next_call += 1  # the name was taken by the record; it is never reused
        form = (
            "not_allowed"
            if isinstance(exc, PermissionError)
            else "helper_limit" if isinstance(exc, RuntimeError) else "bad_request"
        )
        return _refusal(SPEC, form, why=str(exc)[:500])
    ctx.next_call += 1
    if info is not None:
        info.update(helper=call)
    return (
        f"helper {call} started; its messages arrive in the record: use wait or messages. "
        f"Running now: {', '.join(pr.running())}."
    )


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
        # done, stop, finish or wall_clock (error or step_guard when it ended so); model calls cut mid-flight
        "stopped_by": None,
        "calls_cut": 0,
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
    """What ended a helper whose body returned: done, stop, finish, wall_clock, error or step_guard."""
    if status == "ok":
        return "done"
    if status.startswith("stopped: "):
        return pr.stopped_by.get(call, "finish")
    if status.startswith("pass wall clock"):
        return "wall_clock"
    if status == "operational step guard":
        return "step_guard"
    return "error"


async def result(ctx: DelegateContext, call: str) -> str:
    """Wait for helper *call* to end and return its last text (for callers that need the end, such as tests)."""
    await ctx.pass_record.ended[call].wait()
    return ctx.replies[call]


def _finish(
    ctx: DelegateContext,
    call: str,
    staging: _Staging | None,
    doc: dict,
    seconds: float,
) -> dict:
    """Write ``helper.json`` last, export to ``/inputs/helpers/<call>/``, and record the call's row."""
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
    lines.append("summary: " + (row.get("summary") or "(none)"))
    lines.append(claims_text(row.get("claims") or []))
    return "\n".join(lines)
