"""The memory run of the request in progress (one request per CLI process; integration Task 25).

``RequestRun.begin`` opens a run while memory is on (``UNIFY_MEMORY=on``): it takes the request
lock, applies the writer's landed passes, writes the read-only copy of memory ``main`` the worker mounts, notes its
items (for the use record), renders the memory section the system prompt ends with (the librarian's constant
pointer, :func:`.prompt.render_memory_section`), takes the work tree's before snapshot, and opens the scope the
actor runs in (its transcript continues the episode id and model costs are recorded). Before that, each episode
journal a process that died mid-request left is recorded as an incomplete episode, and this request's journal is
opened: the transcript's lines, kept turn by turn until the episode is recorded (:mod:`.journal`). ``finish`` records
the request as one episode (with its transcript's dialogue actions: :mod:`.adapters.dialogue`), with how it used the
library (``memory_use.json``, indexed in the evidence store's ``item_use`` table), removes the journal, starts the
episode-end fork and makes sure the writer service is up; it never raises (an episode that could not be recorded keeps
its journal, recorded as incomplete at the next start). The service's events go to the CLI's ``--jsonl`` output when
it has one, and to the state directory's ``events.jsonl`` (``Paths.events``) either way. ``abort`` cleans up
and records nothing now: its journal stays, and the next start records it as an incomplete episode. The harness hooks read the current run (``current()``) for ``index``
and ``paths``.

Only pass/fail of a posted outcome is kept: ``take_outcome`` keeps ``solved`` and drops
everything else at once, so no checker text reaches an episode, the evidence, the writer or a cell.
"""

from __future__ import annotations

import asyncio
import contextlib
import datetime as dt
import json
import logging
import os
import re
import secrets
import subprocess
from collections.abc import Callable, Mapping
from decimal import Decimal
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_CURRENT: Any = None
#: The memory homes (resolved paths) whose run this process has begun (:func:`..writer_service.begin_run`): one run per
#: process and home.
_RUNS_BEGUN: set[str] = set()

#: The reasoning effort the actor's client gets when nothing sets one (unify/common/llm_client.py
#: ``_build_llm_client``'s ``reasoning_effort`` default).
CLIENT_DEFAULT_EFFORT = "high"
LOCK_TIMEOUT_S = 60.0
#: The evidence store's busy wait while the writer service shares it.
_BUSY_TIMEOUT_S = 60.0
#: The most code-cell statuses a run keeps (use telemetry); every cell of a request fits many times over.
MAX_STATUSES = 10_000
#: The most distinct items a request's runtime call counts keep; calls of more are counted as dropped.
MAX_RUNTIME_ITEMS = 2000
RUNTIME_OUTCOMES = ("ok", "refused", "error")
#: ``Item.fits`` results and tracked item calls a request's evidence lists; past these they are counted only.
MAX_MATCH_RESULTS = 200
MAX_RUN_ROWS = 1000
#: Programs that search files: a cell's process start naming one of them and a path in the library copy is a search.
SEARCH_PROGRAMS = frozenset({"grep", "egrep", "fgrep", "rg", "ag", "ack"})
_SPAWN_EVENTS = frozenset({"subprocess.Popen", "os.posix_spawn", "os.exec", "os.system"})
#: The last line of a cell error that names the library's package as a missing name, or as a failed import.
_NAME_ERROR = re.compile(r"NameError: name 'memory' is not defined")
_IMPORT_ERROR = re.compile(
    r"(?:ImportError|ModuleNotFoundError): .*'memory(?:\.[A-Za-z0-9_.]*)?'",
)
_ERROR_CHARS = 2000
#: The most library failures a request's use record lists (``library_failures``); later ones are
#: counted in ``library_failures_dropped``.
MAX_LIBRARY_FAILURES = 100
#: The package's calls whose raised errors are listed: the lookups, a question to the librarian (``ask``),
#: ``Item.fits`` (``match``), and a function item's call refused before it ran (``run``: a note called); a tracked
#: call reports its item's own failure through the item's ``memory.call`` event instead.
_FAILURE_HELPERS = ("get", "items", "ask", "match", "run")
#: The most characters kept of a failure's message line and of a helper call's arguments' shape.
LIBRARY_FAILURE_TEXT = 200
_EXCEPTION_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]{0,99}")


class MemoryUnavailable(RuntimeError):
    """Memory cannot run here; the message names what to change."""


def current() -> Any:
    return _CURRENT


def _now() -> str:
    return dt.datetime.now(dt.UTC).isoformat()


def actor_model() -> str:
    """The model the actor runs on for this process (the CLI's actor takes the session default)."""
    from unify.common.llm_client import resolve_default_model

    return resolve_default_model()[0]


def actor_effort() -> str:
    """The actor's reasoning effort for this run, read from the setting the actor's client reads. The writer's passes
    run at this effort too."""
    from unify.common.llm_client import resolve_default_model

    return resolve_default_model()[1] or CLIENT_DEFAULT_EFFORT


def build_id() -> str:
    """The harness checkout's commit, or ``unknown`` when it is not a git checkout."""
    from ..gitio import git_base_env

    root = Path(__file__).resolve().parents[3]
    env = git_base_env()
    env.update(
        GIT_CONFIG_NOSYSTEM="1",
        GIT_CONFIG_GLOBAL="/dev/null",
        GIT_TERMINAL_PROMPT="0",
    )
    try:
        out = subprocess.run(
            [
                "git",
                "-c",
                "core.hooksPath=/dev/null",
                "-c",
                "core.fsmonitor=false",
                "-C",
                str(root),
                "rev-parse",
                "--verify",
                "HEAD",
            ],
            capture_output=True,
            env=env,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    sha = out.stdout.decode("ascii", "replace").strip()
    return sha if out.returncode == 0 and len(sha) == 40 else "unknown"


def new_episode_id(transcripts_dir: Path) -> str:
    """``%Y%m%dT%H%M%S-<8 hex>`` (the transcript session id rule), naming no existing transcript."""
    while True:
        stamp = dt.datetime.now(dt.UTC).strftime("%Y%m%dT%H%M%S")
        eid = f"{stamp}-{secrets.token_hex(4)}"
        if not (transcripts_dir / f"{eid}.jsonl").exists():
            return eid


#: The dialogue channel every episode records its transcript's dialogue actions on (each answered reply's action,
#: paired with the counterpart's next message: :mod:`.adapters.dialogue`).
DIALOGUE_CHANNEL = "env"


# --- the role dispatcher ------------------------------------------------------------------

#: How often the dispatcher reads the run record for messages to a role (other processes' posts reach it within this).
ROLE_POLL_S = 0.5
#: The claim states after which a message to a role is done (unify.agents.run_record.CLAIM_STATES).
ROLE_DONE = ("answered", "failed", "withdrawn")


def start_role_dispatcher(run: Any) -> Any:
    """Seed the librarian role once per run (``run_record.seed_role``: from then on ``@librarian`` resolves to the
    role) and start this request's dispatcher (:func:`role_dispatcher`) on the running loop. None when no loop runs.
    Never raises.
    """
    try:
        from .. import librarian as _lib

        loop = asyncio.get_running_loop()  # first: with no loop to answer on, the role is not seeded either
        from unify.agents import run_record as rr

        rr.seed_role(run.paths.home, _lib.AGENT, by=f"request {run.episode_id}")
    except RuntimeError:  # no running loop: nothing to dispatch on
        return None
    except Exception as exc:  # noqa: BLE001 - the roles never stop a request
        logger.warning("memory: the role dispatcher was not started (%s)", type(exc).__name__)
        return None
    return loop.create_task(role_dispatcher(run), name="memory-role-dispatcher")


async def role_dispatcher(run: Any, *, poll_s: float = ROLE_POLL_S) -> None:
    """The request's dispatcher for the run's roles (``run_record.INSTANCE_ROLES``: the
    librarian). Every :data:`ROLE_POLL_S` it reads the run record (every process's posts); each message to the role
    that is not done (``run_record.role_claims``), in order, gets one message instance (``librarian.answer_message``):
    one at a time across the run's processes (``run_record.role_lock``; a lock another process holds waits for the next
    round), claimed before it starts and marked with how it ended. A failure of the instance still gives the message's
    author one reply, from the harness, naming the failure's class. Cancelled with the request (``_cleanup``): the
    message in flight is marked abandoned and the next request's dispatcher takes it again. Never raises otherwise."""
    from unify.agents import run_record as rr

    from .. import librarian as _lib

    home = run.paths.home
    record = rr.join_run_record(home)
    open_session = _lib.opener(on_event=lambda row: rr.role_event(home, _lib.AGENT, row),
                               request_id=f"{run.episode_id}.roles")  # fmt: skip
    while True:
        try:
            await _role_round(record, home, open_session)
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 - a round never stops the request
            logger.warning("memory: a role dispatcher round failed (%s)", type(exc).__name__)
        await asyncio.sleep(poll_s)


async def _role_round(record: Any, home: Any, open_session: Callable[..., Any]) -> int:
    """One round of :func:`role_dispatcher`: the librarian's messages that are not done, oldest first, while this
    process holds the role's instance lock; returns how many it finished."""
    from unify.agents import run_record as rr
    from unify.agents.record import HARNESS

    from .. import librarian as _lib

    role, done = _lib.AGENT, 0
    while True:
        claims = rr.role_claims(home, role)
        pending = [e for e in record.role_messages(role) if claims.get(e.seq, {}).get("state") not in ROLE_DONE]
        if not pending:
            return done
        entry = pending[0]
        try:
            with rr.role_lock(home, role, wait=False):
                if rr.role_claims(home, role).get(entry.seq, {}).get("state") in ROLE_DONE:
                    continue  # another process's instance finished it meanwhile
                rr.claim(home, role, entry.seq, "claimed", pid=os.getpid())
                try:
                    out = await _lib.answer_message(entry, record=record, open_session=open_session)
                except asyncio.CancelledError:
                    rr.claim(home, role, entry.seq, "abandoned", pid=os.getpid())
                    raise
                except Exception as exc:  # noqa: BLE001 - the author still gets one reply
                    reply, _ = record.append(HARNESS, f"no answer to #{entry.seq}: the librarian failed ({type(exc).__name__})",
                                             kind="reply", mentions=[entry.author], reply_to=entry.seq)  # fmt: skip
                    out = {"state": "failed", "reply": reply.seq, "by": HARNESS, "error_class": type(exc).__name__}
                rr.claim(home, role, entry.seq, out["state"], reply=out.get("reply"), by=out.get("by"),
                         **({"error_class": out["error_class"]} if "error_class" in out else {}))  # fmt: skip
                rr.role_event(home, role, {"event": "message", "seq": entry.seq, "state": out["state"],
                                           "reply": out.get("reply"), "by": out.get("by")})  # fmt: skip
                done += 1
        except rr.RoleBusy:
            return done  # another process's instance runs: the next round


def recorded_dialogue(lines: list[dict], redactor: Any) -> list:
    """The dialogue actions a request's episode records: answered replies only, so a reply nothing followed (a
    single-turn request's final reply) adds no action. The observation and the payload are kept whole."""
    from .adapters import dialogue

    return dialogue.dialogue_actions(
        lines,
        DIALOGUE_CHANNEL,
        redactor=redactor,
        answered_only=True,
        max_observation_chars=None,
        max_payload_chars=None,
        payload_scan_chars=None,
    )


def check_hidden(paths: Any, policy: Any) -> None:
    """Refuse to run while a cell could read any harness-side memory path."""
    readable = [
        str(p) for p in paths.harness_only() if policy.readable_violation(p) is None
    ]
    if readable:
        raise MemoryUnavailable(
            "memory needs its state hidden from cells, but the sandbox would let a cell "
            f"read {readable}; keep UNIFY_HOME outside the workspace and every mount",
        )


def _export_roots(checkout: Path) -> list[str]:
    """The paths cells may see the export at: its host paths (an in-process cell's) and the fixed path the
    worker's sandbox binds it at (``unify.sandbox.LIBRARY_MOUNT``)."""
    from unify.sandbox import LIBRARY_MOUNT

    from ..analysis import use

    return use.roots_of(checkout) + [LIBRARY_MOUNT]


def pinned_items(checkout: Path) -> list[str]:
    """The ids of the library's function items in the copy at *checkout* (listed or not), in id order."""
    from ..layout import discover

    return sorted(f.item_id for f in discover(Path(checkout)).functions)


def _count(value: Any) -> int:
    return value if type(value) is int and value > 0 else 0


#: The longest constant a ``memory.get`` call may name and still be kept as an id (``ask_library``'s bound).
MAX_GET_ID = 300


def cells_get_ids(lines: Any, start: int = 0) -> list[str]:
    """The constants the transcript's cells passed to ``memory.get`` (``memory.get('<id>')``, its first argument as a
    string literal), in order of first appearance, from the cells that have a result (``analysis.use.transcript_cells``),
    from cell *start* on. Parsed as code, never matched as text: a cell that does not parse names nothing."""
    import ast

    from ..analysis import use

    out: list[str] = []
    for cell in use.transcript_cells(lines)[start:]:
        try:
            tree = ast.parse(cell.get("code") or "")
        except (SyntaxError, ValueError):
            continue
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "get"
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == "memory"
                and node.args
                and isinstance(node.args[0], ast.Constant)
                and isinstance(node.args[0].value, str)
                and 0 < len(node.args[0].value) <= MAX_GET_ID
                and node.args[0].value not in out
            ):
                out.append(node.args[0].value)
    return out


def runtime_calls_empty() -> dict:
    """A request's runtime call evidence before any cell: ``cells`` whose audit records arrived, per-item
    ``counts`` ({ok, refused, error}) and what was lost (:func:`add_runtime_calls`)."""
    return {
        "cells": 0,
        "counts": {},
        "dropped": 0,
        "records_dropped": 0,
        "malformed": 0,
        "helpers_failed": 0,
        "helpers": {},
        "matches": [],
        "runs": [],
        "match_run_dropped": 0,
        "file_lookups": {"reads": 0, "searches": 0},
    }


#: The library package's own calls the use record counts (``memory.helper_call`` audit events): the lookups
#: (``get``, ``items``), the applicability checks (``match``: ``Item.fits``) and the tracked calls (``run``: a
#: function item's call).
LIBRARY_HELPERS = ("get", "items", "match", "run")
#: ``memory.ask``: counted with the helpers.
ASK_HELPER = "ask"
#: ``memory.ask`` results a request's evidence lists; past this they are counted only.
MAX_ASK_ROWS = 200


def library_helper_calls(runtime: Any) -> dict:
    """The use record's ``library_helper_calls``: for each of the package's calls (:data:`LIBRARY_HELPERS`:
    ``memory.get``, ``memory.items``, ``Item.fits`` as ``match`` and a function item's call as ``run``; and
    ``memory.ask``), the calls the audit hook
    saw, by outcome ({ok, refused, error}). Counts only; zeros when none was seen."""
    seen = runtime.get("helpers") if isinstance(runtime, dict) else None
    seen = seen if isinstance(seen, dict) else {}
    out = {}
    for name in (*LIBRARY_HELPERS, ASK_HELPER):
        row = seen.get(name)
        row = row if isinstance(row, dict) else {}
        out[name] = {k: _count(row.get(k)) for k in RUNTIME_OUTCOMES}
    return out


def library_file_lookups_total(runtime: Any) -> dict:
    """The library-mode use record's ``library_file_lookups``: ``{"reads": n, "searches": n}``, the request's file
    reads and searches of the library copy that went around the package's helpers (:func:`library_file_lookups`).
    Counts only; zeros when none was seen."""
    seen = runtime.get("file_lookups") if isinstance(runtime, dict) else None
    seen = seen if isinstance(seen, dict) else {}
    return {k: _count(seen.get(k)) for k in ("reads", "searches")}


def library_cell_error(error: Any) -> str | None:
    """``name`` when a cell's error (its traceback text) ended in a ``NameError`` on ``memory``, ``import``
    when it ended in an ``ImportError`` or ``ModuleNotFoundError`` naming ``memory`` or one of its modules,
    else None. Only the error's last line is read."""
    if not isinstance(error, str):
        return None
    lines = [line for line in error.splitlines() if line.strip()]
    last = lines[-1] if lines else ""
    if _NAME_ERROR.match(last):
        return "name"
    if _IMPORT_ERROR.match(last):
        return "import"
    return None


def library_failures(runtime: Any, cell_errors: Any) -> dict:
    """The library-mode use record's failure counts: ``library_calls_failed``, the library calls the audit hook
    saw raise (outcome ``refused`` or ``error``, items and the package's helpers alike);
    ``library_name_errors``, cells whose error was a ``NameError`` on ``memory``; ``library_import_errors``,
    cells whose error was a failed import of ``memory`` or of one of its modules. Counts only."""
    failed = 0
    if isinstance(runtime, dict):
        counts = runtime.get("counts")
        for row in (counts if isinstance(counts, dict) else {}).values():
            if isinstance(row, dict):
                failed += _count(row.get("refused")) + _count(row.get("error"))
        failed += _count(runtime.get("helpers_failed"))
    errors = cell_errors if isinstance(cell_errors, dict) else {}
    return {
        "library_calls_failed": failed,
        "library_name_errors": _count(errors.get("name")),
        "library_import_errors": _count(errors.get("import")),
    }


def library_failures_empty() -> dict:
    """A request's library failures before any cell: ``entries`` stamped with their cell's tool call id,
    ``pending`` ones whose cell has not reported its result yet, and how many were ``dropped`` over
    :data:`MAX_LIBRARY_FAILURES` (or by the audit hook's own cap)."""
    return {"entries": [], "pending": [], "dropped": 0}


def _failure_room(acc: dict) -> bool:
    if len(acc["entries"]) + len(acc["pending"]) < MAX_LIBRARY_FAILURES:
        return True
    acc["dropped"] += 1
    return False


def _clip(value: Any) -> str:
    return value[:LIBRARY_FAILURE_TEXT] if isinstance(value, str) else ""


def add_library_failures(acc: dict, drained: Any) -> None:
    """Add one cell's failed library calls (the drained audit records' ``calls.failures``, ``adapters.audit``) to
    *acc* as pending entries: ``kind`` ``item`` with the item id as ``target``, or the helper's name with the
    arguments' shape (types and sizes) as ``target``; the exception's type and the message's first line. The
    hook's own dropped count is added. Malformed parts are skipped."""
    calls = drained.get("calls") if isinstance(drained, dict) else None
    if not isinstance(calls, dict):
        return
    acc["dropped"] += _count(calls.get("failures_dropped"))
    failures = calls.get("failures")
    for row in failures if isinstance(failures, list) else []:
        if not isinstance(row, dict):
            continue
        item, exception = row.get("item"), row.get("exception")
        if not isinstance(item, str) or not isinstance(exception, str):
            continue
        if not _EXCEPTION_NAME.fullmatch(exception):
            continue
        helper = item.split(":", 1)[1] if item.startswith("memory:") else None
        if helper is not None and helper not in _FAILURE_HELPERS:
            continue
        if not _failure_room(acc):
            continue
        acc["pending"].append(
            {
                "kind": helper or "item",
                "target": _clip(row.get("shape")) if helper else item,
                "exception": exception,
                "message": _clip(row.get("message")),
            },
        )


def _error_line(error: str) -> tuple[str, str]:
    """A cell error's exception type (its last dotted part) and message, from its last non-blank line."""
    lines = [line for line in error.splitlines() if line.strip()]
    head, sep, message = (lines[-1] if lines else "").partition(": ")
    name = head.strip().rsplit(".", 1)[-1]
    if not sep or not _EXCEPTION_NAME.fullmatch(name):
        return "", ""
    return name, message.strip()


def settle_library_failures(acc: dict, call_id: Any, error: Any = None) -> None:
    """A cell's result is in: its error, when it named the package (:func:`library_cell_error`), becomes a
    ``name`` or ``import`` entry, and every pending entry is stamped with the cell's tool call id."""
    kind = library_cell_error(error)
    if kind is not None and _failure_room(acc):
        exception, message = _error_line(error)
        acc["pending"].append(
            {
                "kind": kind,
                "target": None,
                "exception": exception,
                "message": _clip(message),
            },
        )
    call = None if call_id is None else str(call_id)
    for entry in acc["pending"]:
        acc["entries"].append({**entry, "call": call})
    acc["pending"] = []


def library_failure_list(acc: Any, cells: list[dict], redactor: Any = None) -> tuple[list[dict], int]:
    """The use record's ``library_failures`` and its dropped count: each entry's ``kind``, ``target``,
    ``exception``, ``message`` (the first line, through the run's *redactor* and the harness's error
    redaction) and ``cell``, the cell's index in the episode's transcript (:func:`..analysis.use.transcript_cells`;
    None when its cell is not known), where its full traceback is."""
    from ..redact import redact_error

    if not isinstance(acc, dict):
        return [], 0
    index = {c.get("call"): c.get("index") for c in cells if isinstance(c, dict)}

    def redact(text: Any) -> Any:
        if not isinstance(text, str):
            return text
        if redactor is not None:
            text = redactor.text(text)
        return redact_error(text)[:LIBRARY_FAILURE_TEXT]

    out = []
    for entry in [*acc.get("entries", []), *({**e, "call": None} for e in acc.get("pending", []))]:
        out.append(
            {
                "kind": entry.get("kind"),
                "target": redact(entry.get("target")),
                "exception": entry.get("exception"),
                "message": redact(entry.get("message")),
                "cell": index.get(entry.get("call")),
            },
        )
    return out, _count(acc.get("dropped"))


def _under(path: Any, roots: tuple[str, ...]) -> bool:
    return isinstance(path, str) and any(path == r or path.startswith(r + "/") for r in roots)


def _read_mode(mode: Any) -> bool:
    return isinstance(mode, str) and not any(c in mode for c in "wax+")


def _argv_tokens(argv: Any) -> list[str]:
    """A process start's words: each argument split as a shell would (a ``sh -c`` command, ``os.system``'s string)."""
    import shlex

    out: list[str] = []
    for arg in argv if isinstance(argv, list) else []:
        if not isinstance(arg, str):
            continue
        try:
            out += shlex.split(arg)
        except ValueError:
            out += arg.split()
    return out


def _searches_library(argv: Any, roots: tuple[str, ...]) -> bool:
    """Whether a process start runs a search program (:data:`SEARCH_PROGRAMS`, or ``git grep``) over a path in
    the library copy."""
    words = _argv_tokens(argv)
    names = [os.path.basename(w) for w in words]
    program = any(n in SEARCH_PROGRAMS for n in names) or any(
        n == "git" and "grep" in names[i + 1 :] for i, n in enumerate(names)
    )
    return program and any(_under(w.rstrip("/") or w, roots) for w in words)


def library_file_lookups(records: Any, roots: Any) -> dict:
    """One cell's lookups of the library copy at *roots* (its paths as the cell sees them) that went around the
    package's helpers: ``reads``, the files under the copy the cell's own code opened for reading (an open made by
    the import system, by traceback formatting or by the library package is marked ``via`` by the audit hook and
    not counted), and ``searches``, the process starts that ran a search program over a path in the copy. Counts
    only: no path or content is kept. ``via`` is the cell's own report (model code can make an open look like the
    import system's), so it filters this metric only and must never feed a security audit or a stop rule."""
    roots_t = tuple(r.rstrip("/") for r in (roots or ()) if isinstance(r, str) and r.strip("/"))
    out = {"reads": 0, "searches": 0}
    if not roots_t:
        return out
    for rec in records if isinstance(records, list) else []:
        if not isinstance(rec, dict):
            continue
        event = rec.get("event")
        if event == "open":
            if "via" not in rec and _read_mode(rec.get("mode")) and _under(rec.get("path"), roots_t):
                out["reads"] += 1
        elif event in _SPAWN_EVENTS and _searches_library(rec.get("argv"), roots_t):
            out["searches"] += 1
    return out


def _labelled_rows(rows: Any) -> list[dict]:
    keep = ("id", "level", "status", "matched")
    rows = rows if isinstance(rows, list) else []
    return [
        {k: r.get(k) for k in keep}
        for r in rows
        if isinstance(r, dict) and isinstance(r.get("id"), str)
    ]


def _add_matches_and_runs(acc: dict, calls: dict, cell: int) -> None:
    """Add one cell's ``Item.fits`` results and tracked item calls to *acc*, and mark which listed ids were
    called afterwards: by a library call or a run in a later cell, or later in the same cell (the hook's
    ``calls_before``/``runs_before`` positions)."""
    records = calls.get("records") if isinstance(calls.get("records"), list) else []
    called = [r.get("item") for r in records if isinstance(r, dict) and isinstance(r.get("item"), str)]
    counts = calls.get("counts") if isinstance(calls.get("counts"), dict) else {}
    runs = [r for r in calls.get("runs") or [] if isinstance(r, dict) and isinstance(r.get("target"), str)]
    matches = [m for m in calls.get("matches") or [] if isinstance(m, dict)]
    here = {k for k in counts if isinstance(k, str)} | {r["target"] for r in runs}
    for m in acc["matches"]:  # earlier cells' results: anything this cell called came after them
        listed = {row["id"] for row in m["listed"]}
        m["called_after"] = sorted(set(m["called_after"]) | (listed & here))
    earlier = {row["id"] for m in acc["matches"] for row in m["listed"]}

    def position(value: Any) -> int:
        return value if type(value) is int and value >= 0 else 0

    new: list[dict] = []
    for m in matches:
        _log_no_outcome("Item.fits", m.get("no_outcome"))
        listed = _labelled_rows(m.get("listed"))
        after = set(called[position(m.get("calls_before")) :]) | {
            r["target"] for r in runs[position(m.get("runs_before")) :]
        }
        new.append(
            {
                "cell": cell,
                "listed": listed,
                "failed": _labelled_rows(m.get("failed")),
                "called_after": sorted({row["id"] for row in listed} & after),
                "runs_before": position(m.get("runs_before")),
            },
        )
    # the ids memory.ask listed: in earlier cells (acc["asks"], added before this), or earlier in this cell
    asked_before = {i for a in acc.get("asks") or [] if a.get("cell") != cell for i in a.get("listed") or []}
    asks_here = [a for a in calls.get("asks") or [] if isinstance(a, dict)]
    for i, r in enumerate(runs):
        listed_here = any(
            m["runs_before"] <= i and r["target"] in {row["id"] for row in m["listed"]} for m in new
        )
        by_ask = r["target"] in asked_before or any(
            position(a.get("runs_before")) <= i and r["target"] in (a.get("listed") or []) for a in asks_here
        )
        if len(acc["runs"]) >= MAX_RUN_ROWS:
            acc["match_run_dropped"] += 1
            continue
        acc["runs"].append(
            {
                "cell": cell,
                "target": r["target"],
                "level": r.get("level") if isinstance(r.get("level"), str) else None,
                "outcome": r.get("outcome") if isinstance(r.get("outcome"), str) else None,
                "conditions": r.get("conditions") if isinstance(r.get("conditions"), str) else None,
                "listed_by_match": r["target"] in earlier or listed_here,
                "listed_by_ask": by_ask,
            },
        )
    for m in new:
        m.pop("runs_before")
        if len(acc["matches"]) >= MAX_MATCH_RESULTS:
            acc["match_run_dropped"] += 1
            continue
        acc["matches"].append(m)


def _log_no_outcome(call: str, facts: Any) -> None:
    """Log each stored reader that gave no outcome in *call* (``not-started`` or ``failed``) with its fixed short
    reason: ids, labels and the reason only (the audit hook let nothing else through)."""
    for f in facts if isinstance(facts, list) else []:
        if isinstance(f, dict) and f.get("read") in ("not-started", "failed"):
            logger.warning("%s: the stored reader of %s %s (%s)", call, f.get("id"), f.get("read"), f.get("reason"))


def _add_asks(acc: dict, calls: dict, cell: int) -> None:
    """Add one cell's ``memory.ask`` results to *acc*: each with the ids it listed as
    fitting, whether it said nothing fits and how many rows the copy's structural catalogue held (no reader
    runs, so there are no reader facts); and mark which listed ids were called afterwards (a library call or a run in a
    later cell, or later in the same cell). Ids and labels only. Added under ``asks`` and ``listed_by_ask`` only
    once an ask was seen, so a request without one is unchanged."""
    asks = [a for a in calls.get("asks") or [] if isinstance(a, dict)]
    records = calls.get("records") if isinstance(calls.get("records"), list) else []
    called = [r.get("item") for r in records if isinstance(r, dict) and isinstance(r.get("item"), str)]
    counts = calls.get("counts") if isinstance(calls.get("counts"), dict) else {}
    runs = [r for r in calls.get("runs") or [] if isinstance(r, dict) and isinstance(r.get("target"), str)]
    here = {k for k in counts if isinstance(k, str)} | {r["target"] for r in runs}
    if "asks" in acc:
        for a in acc["asks"]:  # earlier cells' answers: anything this cell called came after them
            a["called_after"] = sorted(set(a["called_after"]) | (set(a["listed"]) & here))
    if not asks:
        return
    acc.setdefault("asks", [])
    acc.setdefault("asks_dropped", 0)

    def position(value: Any) -> int:
        return value if type(value) is int and value >= 0 else 0

    for a in asks:
        listed = [i for i in a.get("listed") or [] if isinstance(i, str)]
        after = set(called[position(a.get("calls_before")) :]) | {
            r["target"] for r in runs[position(a.get("runs_before")) :]
        }
        if len(acc["asks"]) >= MAX_ASK_ROWS:
            acc["asks_dropped"] += 1
            continue
        acc["asks"].append(
            {
                "cell": cell,
                "listed": listed,
                "nothing_fits": bool(a.get("nothing_fits")),
                "catalogue_rows": a.get("catalogue_rows") if type(a.get("catalogue_rows")) is int else None,
                "called_after": sorted(set(listed) & after),
            },
        )


def add_runtime_calls(acc: dict, drained: Any, library_roots: Any = ()) -> None:
    """Add one cell's drained audit records (the worker's ``done`` message ``audit``) to *acc*: the cell is
    counted as observed, and its ``calls`` block (``adapters.audit``), when present, adds its per-item counts, its
    ``Item.fits`` results and its tracked item calls (:func:`_add_matches_and_runs`). Its path and process records
    add to ``file_lookups`` (:func:`library_file_lookups`) for the sandbox's library mount and *library_roots* (the
    copy's other paths as the cell sees them). Malformed parts are skipped; calls of items beyond :data:`MAX_RUNTIME_ITEMS`
    count as dropped.
    """
    if not isinstance(drained, dict):
        return
    acc["cells"] += 1
    if isinstance(drained.get("records"), list) and drained["records"]:
        from unify.sandbox import LIBRARY_MOUNT

        # the sandbox's fixed mount is always the library copy; *library_roots* add its other paths
        roots = tuple(library_roots or ()) + (LIBRARY_MOUNT,)
        found = library_file_lookups(drained.get("records"), roots)
        have = acc.setdefault("file_lookups", {"reads": 0, "searches": 0})
        for k in ("reads", "searches"):
            have[k] = _count(have.get(k)) + found[k]
    calls = drained.get("calls")
    if not isinstance(calls, dict):
        return
    acc.setdefault("matches", [])
    acc.setdefault("runs", [])
    acc.setdefault("match_run_dropped", 0)
    _add_asks(acc, calls, acc["cells"] - 1)
    _add_matches_and_runs(acc, calls, acc["cells"] - 1)
    for key in ("dropped", "records_dropped", "malformed"):
        acc[key] += _count(calls.get(key))
    helpers = calls.get("helpers")
    for helper_id, row in (helpers if isinstance(helpers, dict) else {}).items():
        if isinstance(row, dict):
            acc["helpers_failed"] = (
                _count(acc.get("helpers_failed"))
                + _count(row.get("refused"))
                + _count(row.get("error"))
            )
            # per helper, by outcome: "memory:get" counts as "get"; any other id is not a helper
            name = (
                helper_id.split(":", 1)[1]
                if isinstance(helper_id, str) and helper_id.startswith("memory:")
                else None
            )
            if name in LIBRARY_HELPERS or name == ASK_HELPER:
                have = acc.setdefault("helpers", {}).setdefault(
                    name,
                    {k: 0 for k in RUNTIME_OUTCOMES},
                )
                for k in RUNTIME_OUTCOMES:
                    have[k] += _count(row.get(k))
            else:
                acc["malformed"] += 1
    counts = calls.get("counts")
    if not isinstance(counts, dict):
        return
    for item, row in counts.items():
        if not isinstance(item, str) or not isinstance(row, dict):
            acc["malformed"] += 1
            continue
        got = {k: _count(row.get(k)) for k in RUNTIME_OUTCOMES}
        have = acc["counts"].get(item)
        if have is None:
            if len(acc["counts"]) >= MAX_RUNTIME_ITEMS:
                acc["dropped"] += sum(got.values())
                continue
            have = acc["counts"][item] = dict.fromkeys(RUNTIME_OUTCOMES, 0)
        for k in RUNTIME_OUTCOMES:
            have[k] += got[k]


def with_runtime_calls(record: dict, runtime: Any) -> dict:
    """*record* (a use record) with the request's runtime call evidence: each called item's row gains
    ``runtime`` = {ok, refused, error} (a row is added for an item the static reading did not see), and the
    record gains ``runtime_calls`` = {cells, items, dropped, records_dropped, malformed}. With no observed
    cell (the audit hook was not installed, or no cell ran) the record is returned unchanged, so readers fall
    back to the static call sites. Every item of a record that has ``runtime_calls`` and no ``runtime`` row
    was observed making no call, unless ``dropped`` is non-zero (some items' calls were not counted).
    """
    if not isinstance(runtime, dict) or _count(runtime.get("cells")) == 0:
        return record
    if not isinstance(record, dict) or "error" in record:
        return record
    rows = record.get("items")
    if not isinstance(rows, dict):
        rows = record["items"] = {}
    counts = runtime.get("counts") if isinstance(runtime.get("counts"), dict) else {}
    for item in sorted(counts):
        row = rows.get(item)
        if not isinstance(row, dict):
            row = rows[item] = {}
        row["runtime"] = dict(counts[item])
    record["runtime_calls"] = {
        "cells": _count(runtime.get("cells")),
        "items": len(counts),
        "dropped": _count(runtime.get("dropped")),
        "records_dropped": _count(runtime.get("records_dropped")),
        "malformed": _count(runtime.get("malformed")),
    }
    return record


def memory_use(
    ep: Any,
    item_ids: list[str],
    *,
    redactor: Any = None,
    export_roots: list[str] | tuple[str, ...] = (),
    surface: dict | None = None,
    shown: dict | None = None,
    shown_text: str = "",
    cell_status: dict | None = None,
    runtime_calls: dict | None = None,
    cell_errors: dict | None = None,
    memory_bound: bool | None = None,
    failures: dict | None = None,
    librarian: list | None = None,
) -> dict:
    """The request's use record (:func:`..analysis.use.request_use`), with *runtime_calls*, the calls the
    actor's cells actually made as the worker's audit hook observed them (:func:`with_runtime_calls`).

    With *librarian* (the request's librarian sessions) the record also has ``memory.ask``'s fields (:func:`..analysis.use.ask_fields`: ``ask_calls``, ``ask_rows``,
    ``listed_by_ask``), ``ask`` among ``library_helper_calls``, and ``librarian``: its model calls' tokens, cache
    reads and USD (:func:`..librarian.usage_fields`).

    It reads the transcript, actions and ``memory.diff`` as the episode writes them (through *redactor*,
    so the offline analyser recomputes the same record from the written episode), against the items at
    the pin, the export's roots and the library's import surface taken before the actor ran, *shown*,
    what the memory-section renderer recorded (its digest is taken again over *shown_text*, the section
    as rendered, through *redactor*, as the transcript's copy of the prompt is), and *cell_status*, each
    cell's status from the runtime's structured result by tool call id (``RequestRun.note_result``). The record
    also counts failures (:func:`library_failures`, from *runtime_calls* and
    *cell_errors*, the cells whose error named the library's package) and lists them
    (``library_failures``, at most :data:`MAX_LIBRARY_FAILURES`, the rest counted in
    ``library_failures_dropped``; :func:`library_failure_list`, from *failures*). A failure of this function
    is recorded as its exception type, with the items at the pin; it never stops the episode. *memory_bound*
    says whether the worker bound ``memory`` in the cells' globals (None: bound; see ``use.request_use``).
    """
    from ..analysis import use

    def clean(value: Any) -> Any:
        return redactor.obj(value) if redactor is not None else value

    try:
        if isinstance(shown, dict) and redactor is not None and shown_text:
            shown = {**shown, **use.section_digest(redactor.text(shown_text))}
        lines = clean(list(ep.transcript))
        record = use.request_use(
            lines,
            item_ids,
            [
                clean(
                    {
                        "cell": a.cell,
                        "kind": getattr(a, "kind", "tool"),
                        "channel": a.channel,
                        "status": a.status,
                    },
                )
                for a in ep.actions
            ],
            memory_diff=(
                redactor.text(ep.memory_diff or "")
                if redactor is not None
                else ep.memory_diff or ""
            ),
            export_roots=export_roots,
            surface=surface,
            shown=shown,
            cell_status=cell_status,
            memory_bound=memory_bound,
        )
        record = with_runtime_calls(record, runtime_calls)
        if isinstance(record, dict) and "error" not in record:
            record.update(library_failures(runtime_calls, cell_errors))
            record["library_helper_calls"] = library_helper_calls(runtime_calls)
            listed, dropped = library_failure_list(
                failures,
                use.transcript_cells(lines) if failures else [],
                redactor,
            )
            record["library_failures"] = listed
            record["library_failures_dropped"] = dropped
            record["library_file_lookups"] = library_file_lookups_total(runtime_calls)
            record.update(use.match_run_fields(runtime_calls, clean(list(ep.transcript))))
            record.update(use.ask_fields(runtime_calls))
        if librarian is not None and isinstance(record, dict) and "error" not in record:
            from ..librarian import usage_fields

            record["librarian"] = usage_fields(librarian)
        return record
    except Exception as exc:  # noqa: BLE001 - telemetry never stops recording
        return {
            "version": use.VERSION,
            "error": type(exc).__name__,
            "items_at_pin": list(item_ids)[: use.MAX_ITEMS_AT_PIN],
        }


def _plain(value: Any) -> Any:
    """*value* with every Decimal as a plain decimal string (never exponent notation)."""
    if isinstance(value, Decimal):
        return format(value, "f")
    if isinstance(value, dict):
        return {k: _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    return value


class RequestRun:
    """One request's memory run; create it with :meth:`begin`."""

    def __init__(self, request: str, paths: Any) -> None:
        self.request = request
        self.paths = paths
        self.index = ""  # the memory section of the system prompt (prompt.render_memory_section)
        # the export's generated files (relative path -> bytes), left out of memory.diff while unchanged
        self.generated: dict[str, bytes] = {}
        # Whether the library copy holds the package the worker binds as ``memory`` in every cell (set at open).
        self.memory_bound = False
        # What the memory section shows (``analysis.use.record_shown``), set where it is rendered.
        self.shown: dict | None = None
        # The memory functions of the export at the pin, taken before the actor runs (use telemetry).
        self.item_ids: list[str] = []
        self.surface: dict | None = None
        self.export_roots: list[str] = []
        # Each code cell's status from the runtime's structured result, by tool call id (note_result).
        self.cell_status: dict[str, dict] = {}
        # The library calls the cells made, from the worker's audit records (note_calls).
        self.runtime_calls: dict = runtime_calls_empty()
        # Cells whose error named the library's package: ``name`` and ``import`` (library_cell_error).
        self.cell_errors: dict[str, int] = {"name": 0, "import": 0}
        # The failed library calls and library cell errors, listed (library_failure_list).
        self.failures: dict = library_failures_empty()
        # The request's librarian sessions (librarian.opener registers each), whose totals the use record reads
        self.librarian_sessions: list = []
        # The episode's journal (journal.EpisodeJournal), fed by the transcript's tap; None while there is none.
        self.journal: Any = None
        # The episodes recorded at open from journals a process that died mid-request left (journal.recover).
        self.recovered: list[str] = []
        self.episode_id = ""
        self.started_at = ""
        self.pin = ""
        self.model = ""
        self.effort = ""
        self.build = ""
        #: what serving changed in this request's copy (withdrawn captures)
        self.capture_serving: dict | None = None
        self.stores: Any = None
        self.state: Any = None
        # The tool adapter is not wired in this build (deferred): ``assemble`` records no tool calls.
        self.observer: Any = None
        self.costs: Any = None
        #: the embedding requests the library export sent (``embeddings.embed``'s ``on_call``), booked in _open
        self.export_embedding_calls: list[dict] = []
        self.worktree: Any = None
        self._worktree_finished = False
        # Only the checker's verdict of a posted outcome: True, False or None (none posted).
        self.solved: bool | None = None
        self.outcome_at: str | None = None
        # The verdicts the bed showed the actor, one per ``{"checker": ...}`` line, in order:
        # {"label": "pass"|"fail", "obs": observations in the transcript when it came (or None), "at": time}; with
        # "text" when the line carried the verdict's text, and for a "fail" with "used_before", the items this
        # request had used when it came (:meth:`used_so_far`).
        self.checker_seen: list[dict] = []
        # The regime is inferred, never defaulted: the verdicts on earlier episodes this session took, posted when this
        # request is recorded (outcomes.post_late); and the callables told of each verdict taken within this episode,
        # with its arrival row's fields.
        self.late_checkers: list[dict] = []
        # The requests a persistent session submitted after its first (:meth:`mark_request`), and the library use when
        # the current one began (None: the episode's start), so a verdict bears on the request it followed.
        self.requests = 0
        self._mark: dict | None = None
        self.signal_listeners: list[Callable[[dict], None]] = []
        #: this request's role dispatcher (:func:`start_role_dispatcher`), running on the loop; None when none runs
        self.role_task: Any = None
        self._lock: int | None = None
        self._scope = contextlib.ExitStack()
        self._closed = False
        self._consolidate_mod: Any = None
        self._landed: list[dict] = []  # the writer's results applied at open, reported at finish
        #: what ensure_service found at the request's start
        self.service: Any = None

    @property
    def memory_main(self) -> str:
        return self.pin

    # -- begin ------------------------------------------------------------------------------------

    @classmethod
    def begin(cls, request: str) -> "RequestRun":
        global _CURRENT
        from unify import sandbox, transcripts
        from unify.db import store_home

        from .paths import Paths
        from .state import acquire_lock

        # The sandboxed worker, the core surface and transcripts are the only mode at this base (the
        # code freeze removed their switches), so nothing else needs checking before the lock.
        run = cls(request, Paths.under(store_home()))
        run._lock = acquire_lock(run.paths.lock, timeout_s=LOCK_TIMEOUT_S)
        try:
            run._open(sandbox, transcripts)
        except BaseException:
            run._drop_journal()  # a request that never began records nothing
            run._cleanup()
            raise
        _CURRENT = run
        return run

    def _open(self, sandbox: Any, transcripts: Any) -> None:
        from unify.settings import SETTINGS

        from . import consolidate, cost, worktree_capture
        from .async_pass import apply_results
        from ..memory_writer import served_head
        from ..writer_service import begin_run, ensure_service
        from .state import State

        paths = self.paths
        policy = sandbox.build_policy(fresh=True)
        check_hidden(paths, policy)
        self.stores = consolidate.open_stores(paths, busy_timeout_s=_BUSY_TIMEOUT_S)
        self.state = State.load(paths.state)
        # under the request lock, before _open_library copies the pin
        self._landed = apply_results(self.state, paths)
        # the served head, whose item records are already written
        self.pin = served_head(self.stores.memory)
        self._open_library(consolidate)
        # this process's run, once per memory home however it is spelled (memory's spend cap counts from it, whatever
        # service restarts follow). A run row that cannot be written never fails the request, and fails safe on spend:
        # the count runs on from the previous run's row, which only counts more.
        home_key = str(Path(paths.home).resolve())
        if home_key not in _RUNS_BEGUN:
            _RUNS_BEGUN.add(home_key)
            try:
                begin_run(paths.home)
            except OSError as exc:
                logger.warning("memory: the run's start was not recorded (%s); the spend count runs on", type(exc).__name__)
        # the writer service: up at every request's start (never waited for; it never fails the request)
        self.service = ensure_service(paths.home, settings=SETTINGS, wait_s=0.0)
        # under the request lock, before this request's own journal exists
        self._recover_journals()
        self.episode_id = new_episode_id(transcripts.transcripts_dir())
        self.started_at = _now()
        if self.capture_serving is not None:
            # what the capture switches changed in this request's copy: the evidence of each switch's effect
            consolidate._deliver(
                self.stores,
                None,
                {"type": "capture_serving", "episode_id": self.episode_id, **self.capture_serving},
            )
        self.model, self.effort, self.build = actor_model(), actor_effort(), build_id()
        # Before the actor starts, so the before snapshot precedes the first cell and the worker's
        # audit hook is installed while the capture is active (hooks.worker_audit).
        self.worktree = worktree_capture.WorktreeCapture(
            paths,
            Path(policy.workspace),
            self.redactor,
        )
        self.worktree.begin()
        self._open_journal(transcripts)
        self._scope.enter_context(transcripts.resume_session(self.episode_id))
        self.costs = cost.install()
        self.costs.activate()
        self._scope.callback(self.costs.deactivate)
        # the embedding requests the library export sent before the cost record opened
        for call in self.export_embedding_calls:
            self.costs.add_embedding(call)
        # the run's roles (the librarian), seeded once per run; this request answers their messages
        self.role_task = start_role_dispatcher(self)

    def _recover_journals(self) -> None:
        """Record each episode journal a process that died mid-request left, as an incomplete episode
        (:func:`.journal.recover`, with the similar index); never raises."""
        from . import journal

        def similar_index(ep: Any, redactor: Any) -> None:
            from .. import similar

            similar.record_episode(self.paths.state_dir, ep, redactor)

        try:
            self.recovered = journal.recover(
                self.paths,
                self.stores,
                library_index=similar_index,
            )
        except Exception as exc:  # noqa: BLE001 - a recovery never stops a request
            logger.warning("memory: episode journals not recovered (%s)", type(exc).__name__)

    def _open_journal(self, transcripts: Any) -> None:
        """This request's journal (:mod:`.journal`): its header, then every line of the episode's transcript as it
        is written (the transcript's tap). A request whose journal cannot be opened runs and is recorded as before,
        without one."""
        from . import journal

        recorder = getattr(self.worktree, "_recorder", None)
        header = {
            "episode_id": self.episode_id,
            "started_at": self.started_at,
            "build": self.build,
            "model": self.model,
            "effort": self.effort,
            "memory_main": self.pin,
            "worktree_before": getattr(recorder, "before", None),
            "request": self.request,
        }
        try:
            self.journal = journal.EpisodeJournal.open(self.paths.journal, header, self.redactor)
        except Exception as exc:  # noqa: BLE001 - the request runs without a journal
            logger.warning("memory: no episode journal for this request (%s)", type(exc).__name__)
            self.journal = None
            return
        transcripts.tap(self.episode_id, self.journal.append)

    def _close_journal(self) -> None:
        """Stop journaling; the journal stays (for the next start to record)."""
        found, self.journal = self.journal, None
        if found is None:
            return
        try:
            from unify import transcripts

            transcripts.tap(self.episode_id, None)
        except Exception:  # noqa: BLE001
            pass
        found.close()

    def _drop_journal(self) -> None:
        """Stop journaling and remove the journal: its episode is recorded, or the request records nothing."""
        found = self.journal
        self._close_journal()
        if found is not None:
            try:
                found.remove()
            except OSError as exc:
                logger.warning("memory: the episode journal was not removed (%s)", type(exc).__name__)

    def _open_library(self, consolidate: Any) -> None:
        """Writes the read-only copy of the pin: no tests, plus the generated helper, INDEX.md and links.json
        (:func:`.checkout.export_actor_library_mode`). The memory section is the librarian's constant pointer. No
        memory.diff is recorded, because the copy cannot be written.
        """
        from ..analysis import use
        from .prompt import render_memory_section

        paths = self.paths
        self._consolidate_mod = consolidate
        self._export_copy(self.pin, paths.checkout)
        copy = self.copy_root
        self.item_ids = pinned_items(copy)
        # the worker binds ``memory`` in every cell when the copy holds the package (hooks.worker_globals)
        try:
            self.memory_bound = (copy / "memory" / "__init__.py").is_file()
        except (OSError, ValueError):
            self.memory_bound = False
        self.export_roots = _export_roots(copy)
        self.surface = use.library_surface(copy)  # re-exports for the use record
        self.index, self.shown = render_memory_section(copy)

    @property
    def copy_root(self) -> Path:
        """The actor's library copy as cells and host readers reach it: the checkout itself."""
        return Path(self.paths.checkout)

    def _export_copy(self, sha: str, dest: Path) -> None:
        """Write the actor's read-only copy of memory commit *sha* into *dest*: the commit's library, its generated
        files with the item records at that commit, the capture store's snapshot as it is now (withdrawn captures left
        out; every other capture served the same way, whatever its code looks like) and the librarian's files. It also
        records what serving changed in the copy, and the generated files."""
        from ..shape_rows import lookup_from
        from .checkout import export_actor_library_mode

        paths = self.paths
        consolidate = getattr(self, "_consolidate_mod", None)
        try:
            shapes = lookup_from(self._shape_rows(consolidate, sha=sha))
        except (
            Exception
        ) as exc:  # noqa: BLE001 - find then matches nothing; the library stays usable
            logger.warning(
                "memory library mode: input shapes not read (%s)",
                type(exc).__name__,
            )
            shapes = None
        from ..gitio import Repo
        from ..item_records import records_at, status_of

        # the item records at the pin: statuses for the index and find, and what memory.get prints
        records, _ = records_at(Repo(paths.memory), sha)
        # the capture store as this request starts (captures.py; empty unless the episode-end fork captured
        # attempts): read only through memory helpers, never in the index or the prompt
        from ..captures import export_files, store_dir
        from ..pool import withdrawn_captures

        # the captures the writer superseded or retired, as the pool says now, and what replaces each
        try:
            withdrawn = withdrawn_captures(Repo(paths.memory))
        except Exception as exc:  # noqa: BLE001 - an unreadable pool withdraws nothing; the request goes on
            logger.warning("memory: the pool's closed captures not read (%s)", type(exc).__name__)
            withdrawn = {}
        # and the captures a writer's decision retired (the attention ledger's close rows), with the writer's reason
        from ..attention import retired_captures

        try:
            withdrawn = {**retired_captures(paths.state_dir), **withdrawn}
        except Exception as exc:  # noqa: BLE001 - an unreadable ledger retires nothing; the request goes on
            logger.warning("memory: the writer's retired captures not read (%s)", type(exc).__name__)
        try:
            captured = export_files(store_dir(paths.state_dir), withdrawn=withdrawn)
        except OSError as exc:
            logger.warning(
                "memory library mode: captures not exported (%s)",
                type(exc).__name__,
            )
            captured = {}
        gone = withdrawn or {}
        # counts, and the items offered in a withdrawn capture's place (the pool holds which capture each replaced)
        self.capture_serving = {
            "withdrawn": len(gone),
            "withdrawn_replaced": sum(1 for row in gone.values() if row.get("by")),
            "offered_in_place": sorted({row["by"] for row in gone.values() if row.get("by")}),
        }
        from ..library_export import staging_summaries
        from ..library_search import backend

        summary_of = staging_summaries(Path(paths.state_dir) / "memory-staging")
        # the search index's vectors: each request the embedder sends is booked once the request's cost record opens
        # (_open)
        book = self.export_embedding_calls.append
        generated = export_actor_library_mode(
            paths.memory,
            sha,
            dest,
            shapes=shapes,
            status_of=status_of(records),
            records=records,
            captures=captured,
            summary_of=summary_of,
            search=backend(on_call=book),
            serving=self._serving(records, status_of(records)),
        )
        self.generated = generated

    def _serving(self, records: Mapping[str, dict], status_of: Callable[[str], str]) -> Callable[[Path], dict]:
        """What the export computes from the copy it materialised: each item's evidence (its failure record,
        contradictions, status basis and stated conditions, :func:`..serving.item_evidence`) and the library items the
        writer retired or superseded (:func:`..serving.withdrawn_items`). An evidence store that cannot be read gives no
        evidence, and the request goes on."""
        from .. import serving as sv
        from ..captures import load, store_dir

        def compute(tree: Path) -> dict:
            caps = load(store_dir(self.paths.state_dir))
            try:
                found = sv.item_evidence(
                    self.stores,
                    captures=caps,
                    home=self.paths.home,
                )
            except Exception as exc:  # noqa: BLE001 - served without evidence; logged
                logger.warning("memory: item evidence not read (%s)", type(exc).__name__)
                found = {}
            # a captured procedure's status is its uses' (serving.item_evidence), never its record's
            statuses = {i: status_of(i) for i in records}
            return {
                "evidence": sv.evidence_doc(found, sv.conditions(tree, records, caps), statuses),
                "withdrawn_items": sv.withdrawn_items(records),
            }

        return compute

    def _shape_rows(self, consolidate: Any, sha: str | None = None) -> dict:
        """The pinned commit's input-shape rows (or *sha*'s), frozen on first export
        (:func:`..shape_rows.shapes_at`); functions without recorded shapes are backfilled from the evidence store's
        covers.
        """
        from ..shape_rows import shapes_at

        episodes = getattr(consolidate, "EpisodeLookup", None)
        return shapes_at(
            self.stores.memory,
            self.stores.evidence,
            sha or self.pin,
            self.paths.checkout,
            lookup=episodes(self.stores).action if episodes is not None else None,
            blobs=self.stores.blobs,
            freeze=True,
        )

    def redactor(self) -> Any:
        """The run's redactor as far as it is known before the episode is assembled: the environment's
        secrets. The work-tree capture calls it at finish. (No tool adapter records calls in this build,
        so no credential is learned from a call's response before ``assemble``.)"""
        from ..redact import Redactor

        return Redactor.from_environ(os.environ)

    # -- cell results -----------------------------------------------------------------------------

    def note_result(self, call_id: Any, result: Any) -> None:
        """Keep a code cell's status from its ``ExecutionResult`` (``hooks.tool_result``), reduced at once
        to names (``analysis.use.runtime_status``): the use record's only source of refusals, errors and
        sessions. A cell whose error named the library's package is counted (:func:`library_cell_error`); it and the
        cell's failed library calls are listed (:func:`settle_library_failures`).
        A plain mapping (the code tool's own result for an empty cell, or when its session executor
        raised) is read by ``analysis.use.dict_status``. At most ``MAX_STATUSES`` calls are
        kept; later ones are then unknown.
        """
        from ..analysis import use

        error = (
            result.get("error")
            if isinstance(result, Mapping)
            else getattr(result, "error", None)
        )
        kind = library_cell_error(error)
        if kind is not None:
            self.cell_errors[kind] = self.cell_errors.get(kind, 0) + 1
        settle_library_failures(self.failures, call_id, error)
        if call_id is None or len(self.cell_status) >= MAX_STATUSES:
            return
        if isinstance(result, Mapping):
            status = use.dict_status(
                result,
                items=self.item_ids,
                roots=self.export_roots,
            )
        else:
            status = use.runtime_status(
                getattr(result, "error", None),
                getattr(result, "session_id", None),
                getattr(result, "session_created", None),
                items=self.item_ids,
                roots=self.export_roots,
            )
        self.cell_status[str(call_id)] = status

    def note_calls(self, drained: Any) -> None:
        """Add one cell's drained audit records to the request's runtime call evidence
        (:func:`add_runtime_calls`; ``hooks.worker_cell_done``), and its failed library calls to the pending failures
        the cell's result then settles (:func:`add_library_failures`)."""
        add_runtime_calls(self.runtime_calls, drained, self.export_roots)
        add_library_failures(self.failures, drained)

    def note_failure(self, call_id: Any, exc: BaseException) -> None:
        """Keep that a code cell's tool call raised instead of returning (``hooks.tool_result``): its
        outcome is unknown (``tool_raised``), with the exception's class name when it is a builtin one.
        """
        from ..analysis import use

        settle_library_failures(self.failures, call_id)
        if call_id is None or len(self.cell_status) >= MAX_STATUSES:
            return
        self.cell_status[str(call_id)] = use.failed_status(type(exc).__name__)

    # -- the outcome ------------------------------------------------------------------------------

    def take_outcome(self, raw: Any) -> dict:
        """Keep the checker's verdict only; the answer the CLI writes for an ``{"outcome": ...}`` line."""
        from unify import outcome as outcome_mod

        try:
            normalized = outcome_mod.normalize(raw)
        except (
            outcome_mod.OutcomeError
        ) as exc:  # our own text; it never quotes the outcome
            return {"type": "outcome", "accepted": False, "reason": str(exc)}
        solved, checks = normalized["solved"], int(normalized["checks_total"])
        del normalized, raw
        self.solved = solved if isinstance(solved, bool) else None
        self.outcome_at = _now()
        return {
            "type": "outcome",
            "accepted": True,
            "solved": self.solved,
            "checks": checks,
        }

    def take_checker(self, raw: Any, episode: Any = None, about: Any = None) -> dict:
        """Keep one verdict the bed showed the actor; the answer the CLI writes for a
        ``{"checker": {"label": "pass"|"fail"}}`` line. The bed sends it just before the message that shows the
        actor the same verdict, so the transcript's observation count now is that message's ordinal. Refused, with
        nothing kept, when the line is anything but ``{"label": "pass"}`` or ``{"label": "fail"}``, optionally with
        ``"text"``: the verdict's text exactly as the environment showed it to the actor (:func:`.outcomes.check_text`;
        kept for the writer through the run's redactor, never read for a verdict: the label is the verdict). The
        answer never quotes the line.

        A ``fail`` taken within this episode also keeps the items this request had used when it came
        (:meth:`used_so_far`): a visible verdict that follows an item's use counts as a failed use of that item in
        this request (:meth:`failed_after_verdicts`), so its librarian does not offer it again here.

        The line may also say what the verdict is about (*about*: ``action:<n>``, ``item:<id>`` or ``capture:<id>``)
        and name an earlier episode by its id (*episode*): a verdict that arrived after that episode ended, kept to be
        posted on it when this request is recorded (:func:`.outcomes.post_late`)."""
        from . import outcomes

        if (
            not isinstance(raw, dict)
            or not ({"label"} <= set(raw) <= {"label", "text"})
            or raw["label"] not in ("pass", "fail")
        ):
            return {
                "type": "checker",
                "accepted": False,
                "reason": 'a checker line is {"label": "pass"} or {"label": "fail"}, optionally with "text"',
            }
        text, bad = outcomes.check_text(raw.get("text"))
        if bad:
            return {"type": "checker", "accepted": False, "reason": bad}
        if text is not None:
            text = self.redactor().text(text)
        if episode is not None and episode != self.episode_id:
            entry, why = outcomes.late_entry(
                raw["label"],
                episode,
                about,
                current=self.episode_id,
                exists=self.stores.evidence.episode_exists,
                text=text,
            )
            if entry is None:
                return {"type": "checker", "accepted": False, "reason": why}
            self.late_checkers.append(entry)
            return {
                "type": "checker",
                "accepted": True,
                "label": raw["label"],
                "episode": episode,
                "count": len(self.late_checkers),
            }
        from ..signals import valid_about

        about = "" if about is None else about
        if not valid_about(about):
            return {
                "type": "checker",
                "accepted": False,
                "reason": "about must be action:<n>, item:<id> or capture:<16 hex digits>",
            }
        entry = {"label": raw["label"], "obs": self._observations_now(), "at": _now(), "about": about}
        if text is not None:
            entry["text"] = text
        entry["request"] = self.requests
        if raw["label"] == "fail":
            entry["used_before"] = self.used_so_far()
        self.checker_seen.append(entry)
        self._tell_listeners(entry)
        return {
            "type": "checker",
            "accepted": True,
            "label": raw["label"],
            "count": len(self.checker_seen),
        }

    def _transcript_lines(self) -> list:
        from unify import transcripts

        from . import trajectory

        return trajectory.read_jsonl(transcripts.transcripts_dir() / f"{self.episode_id}.jsonl")

    def _usage_now(self) -> dict:
        """The library use so far, as :meth:`used_so_far` counts it: calls per item, tracked runs and executed cells."""
        runtime = self.runtime_calls if isinstance(self.runtime_calls, dict) else {}
        counts = runtime.get("counts") if isinstance(runtime.get("counts"), dict) else {}
        try:
            from ..analysis import use

            cells = len(use.transcript_cells(self._transcript_lines()))
        except Exception:  # noqa: BLE001 - unreadable: every cell counts as later
            cells = 0
        return {
            "counts": {
                item: sum(_count(v) for v in row.values())
                for item, row in counts.items()
                if isinstance(item, str) and isinstance(row, dict)
            },
            "runs": len(runtime.get("runs") or []),
            "cells": cells,
        }

    def mark_request(self) -> None:
        """A persistent session begins its next request (the CLI calls it for a message the world marks so:
        ``{"message": ..., "request": "new"}``; a plain follow-up message is more of the same request): from here a
        verdict bears on this request's uses (:meth:`used_so_far`), and an in-request exclusion lasts from the verdict
        to the end of the request after the one it followed (:meth:`failed_after_verdicts`). An episode of one request
        is never marked."""
        self._mark = self._usage_now()
        self.requests += 1

    def used_so_far(self) -> list[str]:
        """The library items the current request's cells have used so far (since :meth:`mark_request`, else since the
        episode began), in id order: each item a cell called (the runtime's call counts and tracked runs) and each id
        a cell passed to ``memory.get`` as a constant, from the cells whose results the transcript holds
        (:func:`cells_get_ids`). Ids only; [] when nothing can be read."""
        mark = self._mark or {"counts": {}, "runs": 0, "cells": 0}
        runtime = self.runtime_calls if isinstance(self.runtime_calls, dict) else {}
        counts = runtime.get("counts") if isinstance(runtime.get("counts"), dict) else {}
        before = mark["counts"]
        used = {
            item
            for item, row in counts.items()
            if isinstance(item, str)
            and isinstance(row, dict)
            and sum(_count(v) for v in row.values()) > before.get(item, 0)
        }
        used |= {
            r["target"]
            for r in (runtime.get("runs") or [])[mark["runs"] :]
            if isinstance(r, dict) and isinstance(r.get("target"), str)
        }
        try:
            used |= set(cells_get_ids(self._transcript_lines(), mark["cells"]))
        except Exception as exc:  # noqa: BLE001 - the runtime's calls still count
            logger.warning("memory: the cells' lookups were not read (%s)", type(exc).__name__)
        return sorted(used)

    def failed_after_verdicts(self) -> list[str]:
        """The items a visible ``fail`` verdict taken within this episode bears on (each ``used_before``: the uses of
        the request it followed), in id order, while it holds: through the request it was taken in and the next one
        (the request that shows it), never later. Failed uses in this request, never stored as such."""
        out: set[str] = set()
        for entry in self.checker_seen:
            if entry.get("label") == "fail" and entry.get("request", 0) >= self.requests - 1:
                out.update(i for i in entry.get("used_before") or () if isinstance(i, str))
        return sorted(out)

    def _tell_listeners(self, entry: dict) -> None:
        """Tell each of :attr:`signal_listeners` of a verdict taken within this episode (``{"episode_id", "label", "about", "arrived_at", "timing": "in_episode"}``); a listener that raises
        is logged and never stops the line."""
        row = {
            "episode_id": self.episode_id,
            "label": entry["label"],
            "about": entry.get("about") or "",
            "arrived_at": entry["at"],
            "timing": "in_episode",
        }
        for listen in list(self.signal_listeners):
            try:
                listen(dict(row))
            except Exception as exc:  # noqa: BLE001 - a listener never stops the session
                logger.warning("memory: a signal listener failed (%s)", type(exc).__name__)

    def last_message(self) -> str | None:
        """The latest user or environment message this request received after its first (an observation: a user
        message the loop did not author, :func:`.adapters.dialogue.observation_texts`), from its transcript, or None
        when none came after the request or the transcript cannot be read. The situation the librarian is given."""
        from unify import transcripts

        from . import trajectory
        from .adapters import dialogue

        try:
            lines = trajectory.read_jsonl(transcripts.transcripts_dir() / f"{self.episode_id}.jsonl")
            texts = [t for t in dialogue.observation_texts(trajectory.fold(lines)) if t.strip()]
        except Exception:  # noqa: BLE001 - nothing to give: the question goes without it
            return None
        return texts[-1] if len(texts) > 1 else None

    def _observations_now(self) -> int | None:
        """How many observations this request's transcript holds now, or None when it cannot be read."""
        from unify import transcripts

        from . import trajectory
        from .adapters import dialogue

        try:
            lines = trajectory.read_jsonl(
                transcripts.transcripts_dir() / f"{self.episode_id}.jsonl",
            )
        except (OSError, ValueError):
            return None
        return dialogue.observation_count(lines)

    # -- finish -----------------------------------------------------------------------------------

    @staticmethod
    def _emitter(emit: Callable[[dict], None] | None) -> Callable[[dict], None] | None:
        """*emit* (the CLI's ``--jsonl`` output) with money as plain decimal strings, or None. The
        consolidation driver also appends every event to the events file (``Paths.events``).
        """
        if emit is None:
            return None
        return lambda event: emit(_plain(dict(event)))

    def _held(
        self,
        eid: str,
        progress: Callable[[str], None],
        emit: Callable[[dict], None] | None,
    ) -> None:
        """The episode is recorded; the driver asked that no pass start (it is ending its stream)."""
        from . import consolidate

        event = {
            "type": "consolidation",
            "phase": "held",
            "episode_id": eid,
            "reason_codes": ["quit_without_consolidation"],
        }
        consolidate._deliver(
            self.stores,
            self._emitter(emit),
            event,
        )  # the events file and --jsonl
        progress(
            "memory: episode recorded; no consolidation pass started (quit without consolidation)",
        )

    def _error(
        self,
        stage: str,
        exc: BaseException,
        progress: Callable[[str], None],
    ) -> None:
        from ..redact import redact_error

        text = redact_error(f"{type(exc).__name__}: {exc}")[:_ERROR_CHARS]
        row = {
            "ts": _now(),
            "episode_id": self.episode_id,
            "stage": stage,
            "error": text,
        }
        try:
            self.paths.errors.parent.mkdir(parents=True, exist_ok=True)
            with self.paths.errors.open("a") as fh:
                fh.write(json.dumps(row) + "\n")
        except OSError:
            logger.warning("memory: %s failed: %s", stage, text)
        try:
            progress(f"memory: {stage} failed: {type(exc).__name__}")
        except Exception:  # noqa: BLE001
            pass

    async def finish(
        self,
        handle: Any,
        *,
        progress: Callable[[str], None],
        emit: Callable[[dict], None] | None = None,
        consolidate: bool = True,
    ) -> None:
        """Record the request, start the episode-end fork and make sure the writer service is up; never raises (the
        request never waits for a pass). The transcript is the record; *handle* is read only by the episode-end fork,
        which starts after the episode is recorded and changes nothing in it. With *consolidate* False (a driver's
        ``{"quit": true, "consolidate": false}``) the episode is recorded and the writer is not woken; what needs its
        attention stays owed, and one ``held`` event says so."""
        if self._closed:
            return
        fork = reservation = None
        try:
            self._close_scope()
            # the fork's staging dir and lock exist before the episode is recorded, so admission never sees a
            # recorded episode whose fork has not been reserved
            reservation = self._reserve_fork(progress)
            try:
                eid, sha = self._record_episode()
            except Exception as exc:  # noqa: BLE001
                self._error("episode", exc, progress)
                if reservation is not None:
                    reservation.abandon("refused: the episode was not recorded")
                return
            # the recorded episode's part list, read before the similar index takes the recorded episode
            parts = self._fork_parts(reservation, progress)
            self._index_similar(progress)
            fork = self._start_fork(handle, eid, progress, reservation, parts)
            if consolidate:
                await self._consolidate(eid, sha, progress, emit)
            else:
                self._report_landed(progress, emit)
                self._held(eid, progress, emit)
        except BaseException as exc:
            self._error("finish", exc, progress)
            if not isinstance(exc, Exception):
                raise
        finally:
            if reservation is not None:
                reservation.close()  # a no-op once the fork has started or settled
            self._cleanup()
        if fork is not None:
            from . import fork as _fork

            # after the reply and the lock: hand over the request, and wait for the fork
            await asyncio.to_thread(_fork.finish, fork, progress)

    def _index_similar(self, progress: Callable[[str], None]) -> None:
        """The ``similar`` tool's index entry for the episode just recorded, built here once, under its time limit, so no
        writer pass spends its time on it; never raises."""
        recorded, self._recorded = getattr(self, "_recorded", None), None
        if recorded is None:
            return
        try:
            from .. import similar

            similar.record_episode(self.paths.state_dir, *recorded)
        except Exception as exc:  # noqa: BLE001 - the index never stops a request
            self._error("similar index", exc, progress)

    def _reserve_fork(self, progress: Callable[[str], None]) -> Any:
        """The episode-end fork's staging dir and lock; never raises."""

        try:
            from . import fork as _fork

            return _fork.reserve(self.paths, self.episode_id, progress)
        except Exception as exc:  # noqa: BLE001
            self._error("fork", exc, progress)
            return None

    def _fork_parts(self, reservation: Any, progress: Callable[[str], None]) -> str | None:
        """The episode just recorded as the fork's part list (:func:`.fork.part_list`), only when its fork is
        reserved; None otherwise or when it cannot be read (the fork then names parts by its numbering rule alone).
        Never raises."""
        recorded = getattr(self, "_recorded", None)
        if reservation is None or recorded is None:
            return None
        try:
            from . import fork as _fork

            return _fork.part_list(recorded[0])
        except Exception as exc:  # noqa: BLE001 - the part list never stops a request
            self._error("fork part list", exc, progress)
            return None

    def _start_fork(
        self,
        handle: Any,
        eid: str,
        progress: Callable[[str], None],
        reservation: Any = None,
        parts: str | None = None,
    ) -> Any:
        """The episode-end fork; never raises."""

        try:
            from . import fork as _fork

            return _fork.start(
                self.paths,
                handle,
                eid,
                progress=progress,
                reservation=reservation,
                parts=parts,
            )
        except Exception as exc:  # noqa: BLE001
            self._error("fork", exc, progress)
            if reservation is not None:
                reservation.abandon(f"error: {type(exc).__name__}")
            return None

    def _record_episode(self) -> tuple[str, str]:
        from unify import transcripts

        from ..episodes import EpisodeWriter
        from . import trajectory

        stores = self.stores
        lines = trajectory.read_jsonl(
            transcripts.transcripts_dir() / f"{self.episode_id}.jsonl",
        )
        cells = trajectory.timed_cells(trajectory.fold(lines))
        self._worktree_finished = True  # finish ends the capture, whatever it returns
        wt = self.worktree.finish(cells)
        ended_at = _now()
        # the transcript's dialogue actions follow the work-tree rows
        extra_actions = [*wt.actions, *recorded_dialogue(lines, self.redactor())]
        ep, redactor = trajectory.assemble(
            self,
            lines,
            ended_at,
            extra_actions=extra_actions,
            worktree_before=wt.before,
            worktree_after=wt.after,
            worktree_diff=wt.diff,
        )
        ep.memory_use = memory_use(
            ep,
            self.item_ids,
            redactor=redactor,
            export_roots=self.export_roots,
            surface=self.surface,
            shown=self.shown,
            shown_text=self.index,
            cell_status=dict(self.cell_status),
            runtime_calls=self.runtime_calls,
            cell_errors=dict(self.cell_errors),
            memory_bound=self.memory_bound,
            failures=self.failures,
            librarian=self.librarian_sessions,
        )
        # the regime from this episode's own signals and use record, before it is written, so the meta, the index and
        # every signal posted below carry it (never a default)
        basis, visible = self._infer_regime(ep, lines)
        sha = EpisodeWriter(stores.episodes, stores.blobs, redactor).write(ep)
        stores.evidence.index_episode(ep, sha)
        # recorded: the journal has served its turn
        self._drop_journal()
        # the episode and the redactor it was stored with, for the similar index (:meth:`_index_similar`)
        self._recorded = (ep, redactor)
        self._post_inferred(ep, sha, ended_at, basis, visible)
        self._record_capture_uses(ep)
        bumped = self.state.generations.observe(ep.fingerprints or {})
        self.state.drift |= bumped
        self.state.suspect |= bumped
        self.state.save()
        return ep.episode_id, sha

    def _record_capture_uses(self, ep: Any) -> None:
        """Each captured procedure this episode called gets its uses written into its capture record from the evidence
        store's rows just indexed (:func:`..serving.record_capture_uses`), so the record and the evidence store agree
        from this request on; a failure is logged and never stops the request."""
        from .. import captures, serving

        use = ep.memory_use if isinstance(ep.memory_use, dict) else {}
        items = use.get("items") if isinstance(use.get("items"), dict) else {}
        try:
            serving.record_capture_uses(self.stores, captures.store_dir(self.paths.state_dir), list(items))
        except Exception as exc:  # noqa: BLE001 - logged, value-free; the request goes on
            logger.warning("memory: capture uses not recorded (%s)", type(exc).__name__)

    def _infer_regime(self, ep: Any, lines: list[dict]) -> tuple[str, list]:
        """Set *ep*'s regime from the verdicts the actor saw in this episode (each
        referring to the action it judged where the pairing is certain) and its use record, with the run's other
        episodes as the prior (:func:`.outcomes.infer`); a hidden grader's verdict decides nothing. Returns the
        basis and the visible verdicts' signals, to be posted once the episode is indexed."""
        from . import consolidate, outcomes

        visible = (
            consolidate.visible_checker_signals(
                self.stores,
                ep,
                list(self.checker_seen),
                lines,
                DIALOGUE_CHANNEL,
                "none",
            )
            if self.checker_seen
            else []
        )
        for sig in visible:
            entry = self.checker_seen[int(sig.signal_id.rsplit(".", 1)[1]) - 1]
            outcomes.timed(sig, entry["at"], ep.ended_at, entry.get("about") or "")
        regime, basis = outcomes.infer(self.stores.evidence, ep.episode_id, visible, ep.memory_use)
        ep.regime = regime
        for sig in visible:
            sig.regime = regime
        return basis, visible

    def _post_inferred(self, ep: Any, sha: str, ended_at: str, basis: str, visible: list) -> None:
        """Record the episode's inferred regime with its basis, post its signals with
        their timing (a regime that cannot observe one masks it, as always) and write their arrival rows, then post
        the verdicts this session took on earlier episodes (:func:`.outcomes.post_late`), each in turn; a late
        verdict that cannot be posted is logged and never stops the request."""
        from ..signals import SignalMasked, post_signal
        from . import consolidate, outcomes

        stores, eid = self.stores, ep.episode_id
        stores.evidence.record_regime(eid, ep.regime, basis, ended_at)
        hidden = consolidate.checker_signal(stores, eid, self.solved, self.outcome_at or ended_at)
        if hidden is not None:
            outcomes.timed(hidden, self.outcome_at or ended_at, ended_at)
        for sig in ([hidden] if hidden is not None else []) + list(visible):
            try:
                post_signal(sig, stores.episodes, sha, stores.evidence)
            except SignalMasked:
                continue
            outcomes.log_arrival(self.paths.state_dir, sig, regime_before=None, regime_after=ep.regime)
            text = self._checker_text(sig) if sig is not hidden else None
            if text is not None:
                outcomes.log_text(self.paths.state_dir, sig.signal_id, text)
            entry = self._checker_entry(sig) if sig is not hidden else None
            if self.requests and entry is not None and entry.get("label") == "fail":
                # a session of several requests: the verdict bears on the request it followed
                try:
                    stores.evidence.add_signal_items(sig.signal_id, entry.get("used_before") or ())
                except Exception as exc:  # noqa: BLE001 - logged, value-free; it then bears on the episode
                    consolidate._error(stores, f"verdict scope {sig.signal_id}: {type(exc).__name__}")
        if self.requests:
            # and a verdict on this episode that arrives later bears on its last request (outcomes.post_late)
            try:
                stores.evidence.set_last_request(eid, self.requests + 1, self.used_so_far())
            except Exception as exc:  # noqa: BLE001 - logged, value-free
                consolidate._error(stores, f"last request of {eid}: {type(exc).__name__}")
        if not self.late_checkers:
            return
        lookup = consolidate.EpisodeLookup(stores)
        for n, entry in enumerate(self.late_checkers, 1):
            try:
                outcomes.post_late(
                    stores,
                    self.paths.state_dir,
                    entry,
                    n,
                    eid,
                    lookup=lookup,
                )
            except Exception as exc:  # noqa: BLE001 - logged, value-free; the request goes on
                consolidate._error(stores, f"late verdict {n} on {entry.get('episode')}: {type(exc).__name__}")

    def _checker_entry(self, sig: Any) -> dict | None:
        """The checker line a visible signal of this episode was posted from (``<episode>.checker.<n>``,
        :func:`.consolidate.visible_checker_signals`), or None."""
        try:
            return self.checker_seen[int(str(sig.signal_id).rsplit(".", 1)[1]) - 1]
        except (IndexError, ValueError):
            return None

    def _checker_text(self, sig: Any) -> str | None:
        """The verdict text of the checker line a visible signal of this episode was posted from, or None."""
        entry = self._checker_entry(sig)
        if entry is None:
            return None
        text = entry.get("text")
        return text if isinstance(text, str) else None

    def _report_landed(
        self,
        progress: Callable[[str], None],
        emit: Callable[[dict], None] | None,
    ) -> None:
        """One ``landed`` event per writer result applied at open, naming this request (the first that ran after it);
        once."""
        rows, self._landed = self._landed, []
        if not rows:
            return
        try:
            from . import consolidate
            from .async_pass import landed_event

            for row in rows:
                consolidate._deliver(
                    self.stores,
                    self._emitter(emit),
                    landed_event(
                        row,
                        self.episode_id,
                        self.pin,
                        consolidate.library_items_at(self.stores, row.get("commit")),
                    ),
                )
        except Exception as exc:  # noqa: BLE001
            self._error("passes", exc, progress)

    async def _consolidate(
        self,
        eid: str,
        sha: str,
        progress: Callable[[str], None],
        emit: Callable[[dict], None] | None,
    ) -> None:
        """Report the passes that landed since the last request, and make sure the writer service is up (the request
        never waits for a pass)."""
        from unify.settings import SETTINGS

        from .async_pass import request_end

        self._report_landed(progress, emit)
        try:
            request_end(self.stores, eid, settings=SETTINGS, emit=self._emitter(emit))
        except Exception as exc:  # noqa: BLE001
            self._error("passes", exc, progress)
        try:
            self.state.save()
        except Exception as exc:  # noqa: BLE001
            self._error("state", exc, progress)

    # -- abort and cleanup ------------------------------------------------------------------------

    def _close_scope(self) -> None:
        """Leave the actor's scope. Every exit runs; one that cannot (a context variable reset from
        another context, which only a caller outside the CLI's task causes) is logged, never raised.
        """
        try:
            self._scope.close()
        except Exception as exc:  # noqa: BLE001
            logger.warning("memory: request scope not closed cleanly: %s", exc)

    def abort(self, stopped: dict | None = None) -> None:
        """End the run with nothing recorded now (the process is closing without a finished request). Its journal
        stays, with why it stopped (*stopped*: ``{"reason": "timeout"}`` or ``{"reason": "error", "error": <class
        name>}``), so the next start records the turns that ran as an incomplete episode. After ``finish`` it changes
        nothing."""
        if not self._closed and self.journal is not None and stopped:
            self.journal.stopped(stopped)
        self._cleanup()

    def _cleanup(self) -> None:
        global _CURRENT
        if self._closed:
            return
        self._closed = True
        from .checkout import remove_checkout
        from .state import release_lock

        if self.role_task is not None:
            # the dispatcher ends with the request; a message instance still answering posts nothing, and its
            # message is taken again by the next request's dispatcher (the claim is marked abandoned)
            self.role_task.cancel()
        self._close_scope()
        # a journal still open (the episode was not recorded) stays for the next start
        self._close_journal()
        if self.worktree is not None and not self._worktree_finished:
            # A run that never reached the capture's finish (abort, a failed begin or a failed
            # finish before it) ends the capture so the worker installs no audit hook for it.
            self._worktree_finished = True
            try:
                self.worktree.abort()
            except (
                Exception
            ) as exc:  # noqa: BLE001 - abort never raises; belt and braces
                logger.warning("memory: work-tree capture not aborted: %s", exc)
        try:
            remove_checkout(self.paths.checkout)
        except OSError as exc:
            logger.warning("memory: export not removed: %s", exc)
        if self._lock is not None:
            try:
                release_lock(self._lock)
            except OSError:
                pass
            self._lock = None
        if _CURRENT is self:
            _CURRENT = None
