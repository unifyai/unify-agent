"""One consolidation pass: the writer as a sandboxed coding agent, then the gate.

The pass copies the memory library's ``main`` into a plain directory, creates an empty ``env/<channel>/``
folder there for every memory channel of the pass's recorded actions, and gives the writer one model-API tool,
``execute_code`` (the code as a read-only file run in :func:`.sandbox_run.run_confined`, with the copy
read-write at ``/memory`` and the pass's inputs read-only at ``/inputs``); its other tools are calls ``tools.<name>``
from its cells (its profile in :mod:`.agent_profiles` lists them), among them ``check`` (the gate's cheap,
read-only checks of a manifest against the current copy, host-side; :meth:`.gate.Gate.preview`; each counted as a
call) and ``finish``. It ends on ``finish``, the operational step guard (``max_calls``), the run's spend room for the
pass (``usd_limit``), the pass deadline or a breach of the write quota (which refuses the pass without a commit). The harness then reads ``/memory/.pass/manifest.json`` without following
links, mirrors the copy into a git checkout (dropping ``.pass``, nested ``.git`` entries and caches),
commits with ``Pass``/``Episode``/``Evidence`` trailers and hands the commit to :meth:`.gate.Gate.merge`,
which records the pass. A missing manifest is recorded here as failed with the reason ``no manifest``.

The box never sees the git checkout itself: a ``.git`` file the model could rewrite would point the host's
``git add`` at a repository (and configuration) of its choosing.

The writer sees only what :func:`export_for_writer` and :func:`export_blobs` write: request, cells,
actions, and the file blobs worktree actions recorded; and, from the harness, the library's generated files, its
attention entries and, with ``PassConfig.show_usage``, a table of how requests used each function
(:func:`.usage.usage_table`, harness counts). An outcome reaches it only as a recorded fact: each capture's outcome
(``positive``, ``none`` or ``contrary``, from the signals the actor could see) in the capture's record and, for a
group, in its group document; no checker or grader runs for it. The
pass's messages are kept (:meth:`WriterPass.transcript`, redacted and bounded) for the driver to store on the
episode repo.
"""

from __future__ import annotations

import ast
import asyncio
import contextlib
import contextvars
import copy
import hashlib
import json
import logging
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
from typing import Any, Callable, Iterable, Iterator, Mapping, Protocol

from pydantic import SecretStr

from . import manifest as _manifest
from .blobs import BLOB_ID, BlobStore
from .episodes import Episode, env_channel
from .evidence import EvidenceStore
from .gate import Gate, GateResult, ParentSnapshot
from .gitio import Repo
from . import messaging as _messaging
from . import repair as _repair
from . import safe_paths as _safe
from . import views as _views
from . import writer_tools as _wt
from .tool_spec import form_of as _form_of
from .redact import Redactor, redact_error
from . import sandbox_run as _sandbox
from .sandbox_run import PRLIMIT, PYTHON, run_confined
from .trigger import PassRequest

# Write bounds on /memory, checked after every cell and before the commit; a breach refuses the pass.
QUOTA_TOTAL_BYTES = 20 * 1024**2
QUOTA_FILE_BYTES = 1024**2
QUOTA_ENTRIES = 2000
# RLIMIT_FSIZE inside a writer cell, lowered from run_confined's by prlimit in the box (lowering needs no
# privilege); the quota walk is the precise check, this only bounds a single file between walks.
CELL_FSIZE_BYTES = 2 * QUOTA_FILE_BYTES
# Recorded file blobs exported to /inputs/blobs: one blob at most a library file's quota (so any exported
# blob can become a test fixture), all of them at most this much.
EXPORT_BLOB_BYTES = QUOTA_FILE_BYTES
EXPORT_TOTAL_BYTES = 32 * 1024**2

# a pass has no USD cap of its own and no count limits: this many model and check calls end a runaway pass, labelled
# operational (a safety stop), never a design budget
STEP_GUARD = 500
_CHECK_REASONS = 20
_CHECK_REASON_CHARS = 200
# A check starts only with at least this much of the pass deadline left (it runs to completion).
CHECK_FLOOR_S = 60.0

def _thought_only(args: object) -> bool:
    """Whether execute_code's *args* hold no code to run (:func:`unify.common.tool_registry.thought_only`)."""
    from unify.common.tool_registry import thought_only

    return thought_only(args)


def _thought_only_reply() -> str:
    from unify.common.tool_registry import THOUGHT_ONLY_REPLY

    return THOUGHT_ONLY_REPLY


#: the memory agents (the writer, each helper) compact their conversation
#: as the agent loop does (unify/common/_async_tool): once a model call's prompt reaches
#: ``context_compression.COMPACTION_THRESHOLD`` of the model's context window (the window unillm gives for the
#: agent's endpoint, ``unillm.get_max_input_tokens``, the agent loop's own source), the agent writes a summary
#: (``FORK_SUMMARY``: its last request, unchanged, plus ``cache_discipline.COMPRESSION_FORK_INSTRUCTION``), sent by
#: the agent's own model turn over its own route, and the conversation restarts from the system message and that
#: summary (``restart_text``); the request messages are kept before it
#: (``kept_prefix``). A model with no known window is never compacted, as in the agent loop.
#: The files an agent works with stay where they were; the summary message says so.
RESTART_POINTER = (
    "Nothing under /memory, /inputs or /outputs changed: read them again (gate results, fixtures, drafts, staging) "
    "when you need them."
)


def compaction_due(prompt_tokens: object, window: object) -> bool:
    """The agent loop's trigger (``context_over_threshold``: :data:`COMPACTION_THRESHOLD` of *window*, or the fixed
    compaction point, whichever comes first) for a call that used *prompt_tokens*; with no known *window* the fixed
    point applies alone, and an unknown prompt size never triggers.
    """
    from unify.common._async_tool.context_compression import (
        COMPACTION_THRESHOLD,
        context_over_threshold,
    )

    if (
        not isinstance(prompt_tokens, int)
        or isinstance(prompt_tokens, bool)
        or prompt_tokens <= 0
    ):
        return False
    # an unknown window leaves the fixed compaction point, as in the agent loop
    return context_over_threshold(prompt_tokens, COMPACTION_THRESHOLD, window)


def _natural_path(vpath: str, roots: Mapping[str, Any]) -> str:
    """A path as an agent naturally writes it, resolved to one of its *roots*: an absolute path as it is; a relative
    one under the first root (``/memory``, then ``/inputs``, ``/outputs``) where it exists, with a leading
    ``memory/`` also read as the library's root (``memory/notes/x.md`` -> ``/memory/notes/x.md``); a relative path
    found under none is left as written (the read then refuses it with its next step). Nothing outside a root is
    reached: the result is checked against the roots as any path is."""
    if not vpath or vpath.startswith("/"):
        return vpath
    rel = vpath.strip("/")
    tries = [f"{r.rstrip('/')}/{rel}" for r in ("/memory", "/inputs", "/outputs") if r in roots]
    if rel.startswith("memory/") and "/memory" in roots:
        tries.append("/memory/" + rel[len("memory/") :])
    for t in tries:
        prefix = next((r for r in roots if t == r or t.startswith(r.rstrip("/") + "/")), None)
        if prefix is None or ".." in t.split("/"):
            continue
        if os.path.exists(os.path.join(os.fspath(roots[prefix]), t[len(prefix) :].lstrip("/"))):
            return t
    return vpath


@dataclass
class Compaction:
    """One compaction's summary call and its outcome: the call's reply and charge (booked like any other call
    of the agent), the request it sent, and the conversation to restart from (None: the reply held no summary,
    so the conversation stays as it is)."""

    request: list[dict]
    reply: object
    usd: object
    restart: list[dict] | None


async def compact_by_summary(
    messages: list[dict],
    turn: Callable[[list[dict], list[dict]], Any],
    tools: list[dict],
    pointer: str = RESTART_POINTER,
) -> Compaction:
    """The agent loop's summary compaction (``AsyncToolLoopHandle._summarise_as_fork``) for a memory agent: the
    summary request is *messages*, unchanged, plus the agent loop's instruction, sent through the agent's own
    model *turn* (its route, its model, its accounting) with its *tools*, so the cached prefix serves all but the
    instruction. Only message content goes in: what the agent's own calls already sent. The restart is the leading
    system message and the summary as the agent loop builds it (``restart_text``, here with *pointer*), after the
    kept request messages when a prefix is kept (``kept_prefix``, the summary then marked as the
    loop's own message)."""
    from unify.common._async_tool import cache_discipline as _cd
    from unify.common._async_tool import context_compression as _cc

    request = [*messages, {"role": "user", "content": _cd.compression_fork_instruction()}]
    reply, usd = await turn(request, tools)
    summary = (
        _cd.completion_text(reply.get("content")) if isinstance(reply, dict) else ""
    )
    if not summary or (isinstance(reply, dict) and reply.get("tool_calls")):
        return Compaction(request, reply, usd, None)
    system = [
        m for m in messages[:1] if isinstance(m, dict) and m.get("role") == "system"
    ]
    restart = [
        *system,
        {"role": "user", "content": _cc.restart_text(summary, pointer)},
    ]
    return Compaction(request, reply, usd, restart)


def compaction_archive(compaction: Compaction, history: list[dict]) -> list[dict]:
    """What a compaction takes out of an agent's conversation, for its transcript: the *history* it replaced, then
    the summary call's instruction and reply, each marked ``"compaction_call": true``.
    """
    reply = (
        compaction.reply
        if isinstance(compaction.reply, dict)
        else {"content": repr(compaction.reply)}
    )
    return [
        *history,
        {**compaction.request[-1], "compaction_call": True},
        {**reply, "compaction_call": True},
    ]


def _prompt_chars(m: object) -> int:
    """A message's prompt text: its content and its tool calls' names and arguments; never provider fields."""
    if not isinstance(m, dict):
        return 0
    c = m.get("content")
    n = (
        len(c)
        if isinstance(c, str)
        else (len(json.dumps(c, default=str)) if c is not None else 0)
    )
    for tc in m.get("tool_calls") or []:
        f = tc.get("function") if isinstance(tc, dict) else None
        if isinstance(f, dict):
            n += len(str(f.get("name") or "")) + len(str(f.get("arguments") or ""))
    return n


def split_usage(msg: dict) -> tuple[dict, object]:
    """A turn's message as the conversation sends it next, and the provider's ``usage`` report it came with (None:
    none). The report is a reply-only field, never sent back to the provider, so the message is a copy without it;
    the turn's own dict is left as it is."""
    if "usage" not in msg:
        return msg, None
    return {k: v for k, v in msg.items() if k != "usage"}, msg["usage"]


def _usage_of(msg: object, usd: object, reported: object = None) -> dict:
    """The usage of the model call that produced a turn's tool calls: *reported* (taken off the message by
    :func:`split_usage`), else the message's own ``usage``; an unreported count stays None (unknown)."""
    u = (
        reported
        if reported is not None
        else (msg.get("usage") if isinstance(msg, dict) else None)
    )
    u = u if isinstance(u, dict) else {}

    def num(*path: str) -> int | None:
        for key in path:
            cur: object = u
            for part in key.split("."):
                cur = cur.get(part) if isinstance(cur, dict) else None
            if isinstance(cur, int) and not isinstance(cur, bool):
                return cur
        return None

    cost = _money(usd)
    return {
        "input": num("prompt_tokens", "input_tokens"),
        "cache_read": num(
            "prompt_tokens_details.cached_tokens",
            "cache_read_input_tokens",
        ),
        "cache_write": num(
            "cache_creation_input_tokens",
            "prompt_tokens_details.cache_write_tokens",
        ),
        "output": num("completion_tokens", "output_tokens"),
        "reasoning": num(
            "completion_tokens_details.reasoning_tokens",
            "reasoning_tokens",
        ),
        "usd": _usd(cost) if cost is not None else "unknown",
    }


class _PassTemp:
    """The pass's temporary folder (``tempfile.TemporaryDirectory``), removed on the way out unless :attr:`keep`
    names why it stays (a tool handler thread may still use it)."""

    def __init__(self, prefix: str) -> None:
        self._dir = tempfile.TemporaryDirectory(prefix=prefix)
        self.keep: str | None = None

    def __enter__(self) -> str:
        return self._dir.name

    def __exit__(self, *exc: object) -> None:
        if self.keep is None:
            self._dir.cleanup()
        else:
            self._dir._finalizer.detach()  # noqa: SLF001 - kept on purpose; never removed at exit either


def _args_sha256(args: object) -> str:
    """The digest of a call's canonical arguments (sorted keys, compact JSON)."""
    import hashlib

    text = json.dumps(
        args,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
        default=str,
    )
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()


def _lookup_store(gate: object) -> object:
    """The evidence store behind the gate's episode lookup (an ``EpisodeLookup``'s stores), if it has one."""
    owner = getattr(getattr(gate, "lookup", None), "__self__", None)
    return getattr(getattr(owner, "stores", None), "evidence", None)


def _messaging_extra(info: dict) -> dict:
    """The event fields of a delegate or record tool call: its ``messaging`` object (``helper`` names the helper a
    delegate or stop is about)."""
    return {"messaging": dict(info)}


def _delegate_reply(text: str) -> "_wt.Reply":
    """A ``delegate`` or record tool call's reply: what it did, or the refusal (its own form code) that did nothing."""
    form = _form_of(text)
    return _wt.Reply(text, "refused" if form else "ok", form)


def _unread_entry(pool_mod: Any, eid: str) -> Any:
    """The pool's ``unread`` entry for a batch episode the writer never opened (kept until a writer closes it)."""
    e = pool_mod.Entry(
        id=pool_mod.entry_id("unread", eid),
        kind="unread",
        title=f"episode {eid} was not opened by the writer",
        data={},
    )
    e.add_provenance([eid], [], ["writer"])
    return e


def _staged_candidates(entry: dict) -> list[dict]:
    """A batch-map staging entry's candidate functions, each with its cases file (or None)."""
    files = set(entry.get("files") or [])
    base = entry.get("path", "")
    out = []
    for f in sorted(files):
        if f.startswith("candidates/") and f.endswith(".py"):
            name = f[len("candidates/") : -3]
            case = f"cases/{name}.json"
            out.append(
                {
                    "name": name,
                    "file": f"{base}/{f}",
                    "cases": f"{base}/{case}" if case in files else None,
                },
            )
    return out


def _capture_lands(store: Path, cid: str) -> bool:
    """Whether capture *cid* of the store can be landed: its record (a regular file, never through a link) says its
    attempt did not fail, the test :func:`.land_capture.read_capture` applies."""
    from .captures import MAX_RESULT_BYTES, RECORD_FILE, negative

    p = Path(store) / cid / RECORD_FILE
    try:
        st = os.lstat(p)
        if not stat.S_ISREG(st.st_mode) or st.st_size > MAX_RESULT_BYTES:
            return False
        rec = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    if not isinstance(rec, dict) or negative(rec):
        return False
    # a trace capture lands by this route only in the value form (a procedure of the request's value); the others are
    # landed by writing them (land_capture refuses them, naming that route)
    form = rec.get("entry_point", {}).get("form") if isinstance(rec.get("entry_point"), dict) else None
    return form in (None, "value")


def _with_fixtures(manifest: object, made: dict) -> object:
    """The harness's fixture provenance replaces whatever the manifest says under ``fixtures``."""
    if not isinstance(manifest, dict):
        return manifest
    return {**manifest, "fixtures": {k: made[k] for k in sorted(made)}}


def _tests_changed(manifest: object) -> list[str]:
    """``Tests-Changed:`` trailers from the manifest's stated reasons (no unexplained test loss)."""
    stated = manifest.get("tests_changed") if isinstance(manifest, dict) else None
    if not isinstance(stated, dict):
        return []
    return [
        f"{k}: {v.strip()}"[:300].replace("\n", " ")
        for k, v in sorted(stated.items())
        if isinstance(k, str) and isinstance(v, str) and v.strip()
    ]


def _trailers(
    pass_id: str,
    sources: object,
    manifest: object,
) -> dict:
    """A candidate commit's trailers: ``Pass``, ``Episode`` and ``Evidence``, then ``Why:``, ``Episodes:`` and
    ``Items:`` (:func:`_library_trailers`), and ``Tests-Changed:`` when the manifest states reasons."""
    out: dict = {"Pass": pass_id, "Episode": sources, "Evidence": sources}
    out.update(
        _library_trailers(
            manifest,
            list(sources) if isinstance(sources, list) else [],
        ),
    )
    if _tests_changed(manifest):
        out["Tests-Changed"] = _tests_changed(manifest)
    return out


_OUTPUT_CAP = 8000
READERS_PER_TURN = _wt.READERS_PER_TURN


def _head_marked(text: str) -> tuple[str, str]:
    """A cell's output in library mode: its head, marked when cut (never the tail alone), and the full text."""
    return _views.view(text.encode("utf-8", errors="replace"), 0, _OUTPUT_CAP), text


_MAX_CELLS_PER_TURN = _wt.CELLS_PER_TURN
_MANIFEST_MAX_BYTES = 1024**2
_MANIFEST_MAX_DEPTH = 16
# Never mirrored out of the box: git metadata (anywhere), the pass directory (top level), caches. Test-run
# caches (pytest's, hypothesis's) are the writer's run byproducts, never library content; the gate refuses them.
_NEVER_COPIED = frozenset({".git", "__pycache__", ".pytest_cache", ".hypothesis"})
_EPISODE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}\Z")
_UNKNOWN = "unknown"

# The writer's transcript: kept whole, never cut (:mod:`.transcripts`). On this notes ref of the request's episode
# commit go its JSON lines while they fit transcripts.NOTE_BYTES, else one row pointing to its chunk files
TRANSCRIPT_REF = "writer-transcripts"

# Structured causes of a pass's end (PassOutcome.codes), set where each cause arises, never read back from
# reason text. A passed pass is exactly ["ok"]; a refused pass lists the gate checks that refused (G1..G6).
CODE_OK = "ok"
CODE_NO_MANIFEST = "no_manifest"
CODE_MANIFEST_INVALID = "manifest_invalid"
CODE_OVER_QUOTA = "over_quota"
CODE_DEADLINE = "deadline"
CODE_PASS_CAP = "pass_cap"
CODE_STEP_GUARD = (
    "operational_step_guard"  # a pass reached its step guard (max_calls, STEP_GUARD in production)
)
CODE_WRITER_ERROR = "writer_error"
# a writer call found the writer's declared route not in effect
CODE_ROUTE_NOT_IN_EFFECT = "route_not_in_effect"
# library mode: the shared no-progress rule ended the agent's round (loop_stop.AgentNoProgress); the pass then went
# to its final merge, so a refused pass lists the gate checks beside it
CODE_NO_PROGRESS = "no_progress"


class ModelTurn(Protocol):
    async def __call__(
        self,
        messages: list[dict],
        tools: list[dict],
    ) -> tuple[dict, str]: ...


@dataclass
class PassConfig:
    """Per-pass bounds. ``model`` and ``effort`` build the real turn: ``unillm_turn(cfg.model, cfg.effort)``."""

    # the memory home whose run record the pass's agent and its helpers join (unify/agents/run_record.py)
    run_home: str
    model: str = "openai/gpt-6-sol"
    effort: str = "low"
    # the operational step guard: model and check calls that end a runaway pass (no USD cap of its own; the run's
    # spend room is usd_limit)
    max_calls: int = STEP_GUARD
    cell_timeout_s: float = 60.0
    deadline_s: float = 900.0
    # where the fork leaves staging (<state>/memory-staging; None: a scratch dir in the pass); writer-launched
    # helpers write beside it (<state>/memory-helpers, helpers.py)
    staging_root: str | None = None
    # the ``similar`` tool's index folder (similar.index_dir; None: the tool has no index to search)
    similar_index: str | None = None
    # where the pass's structured records go: <pass_record_root>/<pass id>/tool-events.jsonl (None: kept in memory
    # only, on WriterPass.tool_events)
    pass_record_root: str | None = None
    # the memory home whose attention ledger the pass reads and decides (attention.py, attention_tools.py)
    attention_home: str | None = None
    # the writer service's room for this pass (its cap less what its other passes committed, less what a resumed pass
    # already lost), a decimal string: a hard USD stop that leaves the pass's other limits as they are. No model call, the writer's or a helper's, starts unless the pass's spend so far,
    # with its unpriced calls and every call in flight and the new one each at usd_call_worst, stays within it
    # (usd_room). None: no such stop, as built
    usd_limit: str | None = None
    # one call's worst case for the pass's lanes (the pinned catalogue's bound for the writer's model and, with a
    # helper lane, the helpers'; call_bounds.py), a decimal string; required with usd_limit
    usd_call_worst: str | None = None


def usd_room(cfg: Any, spend: Any) -> bool:
    """Whether one more model call fits under ``cfg.usd_limit`` (always, without one): the pass's known spend (the
    writer's and its helpers'), and each of its unpriced calls, its calls in flight (``spend.in_flight``) and the new
    call at ``cfg.usd_call_worst``, so the pass never spends past the limit while no single call costs more than
    that worst case. A limit with no worst case to count with lets no call start (fail closed)."""
    limit = getattr(cfg, "usd_limit", None)
    if limit is None:
        return True
    worst = getattr(cfg, "usd_call_worst", None)
    if worst is None:
        return False
    calls = int(spend.unknown) + int(getattr(spend, "in_flight", 0) or 0) + 1
    return Decimal(spend.usd) + Decimal(calls) * Decimal(worst) <= Decimal(limit)


@contextlib.contextmanager
def call_in_flight(spend: Any) -> Iterator[None]:
    """Count a model call as in flight on the pass's *spend* (shared by the writer and its helpers) while it runs."""
    spend.in_flight = int(getattr(spend, "in_flight", 0) or 0) + 1
    try:
        yield
    finally:
        spend.in_flight -= 1


@dataclass
class PassOutcome:
    """``usd`` is the known spend as a pure decimal string; unpriced calls are counted apart."""

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
    # per-item admission (the gate's): the manifest items that landed, and each refused item's codes
    items_merged: list[str] = field(default_factory=list)
    items_refused: dict[str, list[str]] = field(default_factory=dict)
    # the writer's read log (writer_tools.ReadLog.summary: episodes opened, unread batch episodes, parts and bytes
    # read); a measure only (None with library_mode off)
    coverage: dict | None = None
    reads: int = 0  # reader calls made (library_mode)
    exported_bytes: int | None = (
        None  # blob bytes staged in /inputs (library_mode: uncapped)
    )
    # repair rounds: the writer's rounds (1 + the repair rounds started; 0 with library_mode off) and the full
    # gate results written for the writer during the pass, in order (/inputs/gate/result-<n>.md)
    rounds: int = 0
    round_results: list[str] = field(default_factory=list)
    # library_mode: the writer's last finish was accepted; and its manifest, parsed by the gate's parser at that
    # finish, listed no item and no deletion: the writer's explicit decision that the batch holds nothing to store
    finished: bool = False
    nothing_to_store: bool = False
    # one row per delegate call (helpers.py: call, status, episodes, turns, usd, unknown_cost_calls, files); their
    # spend is in usd too, and the rows split it by lane
    helpers: list[dict] = field(default_factory=list)
    # the batch's staging by source (fork, replay_fork, none); None with library_mode off
    staging: dict | None = None
    # with staging on, one row per batch episode (:func:`.summaries.episode_row`): its staging's source, status,
    # files, candidates and the files withheld from an unfinished reader; its summary.md's size and how many of the
    # summary's pointers resolve in the recorded episode; and whether the writer read the summary. None otherwise
    staging_episodes: list[dict] | None = None
    # the candidate pool after this pass: open entries, added, status changes, its commit (or its error); None off
    pool: dict | None = None
    # no longer computed (the read log in ``coverage`` replaces it); kept so older readers find the field
    coverage_measures: dict | None = None
    # library layout: counts of staged candidates at each hand-off, fork reply to adoption (:func:`.chain.counts`)
    chain: dict | None = None
    # library layout: the gate's {"item", "adopts"} row per landed item
    provenance: list[dict] = field(default_factory=list)
    # the writer's compactions, the messages they took out of its
    # conversation and those messages' prompt characters (summary calls are counted in calls and usd)
    compactions: int = 0
    chars_moved: int = 0
    messages_moved: int = 0
    # the writer's model calls' prompt tokens and the part of them read from the provider's cache, each summed over
    # the calls that reported it (None: no call did); input_tokens_calls and cache_read_calls say how many calls
    # reported each, out of writer_calls, so a partial sum is never read as a whole one. cache_read_tokens /
    # input_tokens is the cache-hit rate when both counts come from the same calls
    input_tokens: int | None = None
    cache_read_tokens: int | None = None
    input_tokens_calls: int = 0
    cache_read_calls: int = 0
    # the output tokens (reasoning included, as providers count them) summed over the calls that reported them (None:
    # no call did), and how many calls did
    output_tokens: int | None = None
    output_tokens_calls: int = 0
    writer_calls: int = 0
    # the first writer call's prompt tokens (the brief and the first message, before any reading); None: unreported
    writer_first_prompt_tokens: int | None = None
    # the largest prompt (tokens) one of the writer's model calls reported (None: none reported one), and the
    # model's context window as the compaction trigger reads it (None: unknown, so it never compacts)
    max_prompt_tokens: int | None = None
    context_window: int | None = None
    # library mode: the refusals the gate repeated for the same reason in repair.STUCK_REPEATS consecutive
    # rounds, so no further round was offered for them (repair.StuckRefusals.report: "repeats", "stopped" (every
    # refusal of a round was stuck, so repair ended there) and one {"check", "item", "reason", "rounds"} row each);
    # None when none became stuck
    repair_stuck: dict | None = None
    # library mode: the no-progress rule (loop_stop.AgentNoProgress.report): {"k", "notices", "ended", "at_call"}
    # when the writer made loop_stop.MEMORY_K no-progress model calls in a row at least once; "ended":
    # 2 * k in a row ended its round and the pass went to its final merge. None when no notice was given
    no_progress: dict | None = None
    # library mode: the landed functions' evidence levels, as counts per level (:mod:`.item_states`), and the
    # evidence findings that set them, as counts by reason code (``[examples:grounded]``, or the check for a finding
    # without a code); None when the final merge recorded none
    levels: dict | None = None
    capped_reasons: dict | None = None


def level_counts(res: Any) -> tuple[dict | None, dict | None]:
    """(counts per evidence level, counts of the findings that set them by reason code) over the functions
    *res* (a gate result) landed; (None, None) when it recorded no level."""
    levels = getattr(res, "levels", None) or {}
    if not levels:
        return None, None
    landed = set(getattr(res, "items_merged", None) or [])
    per: dict[str, int] = {}
    codes: dict[str, int] = {}
    for item, level in sorted(levels.items()):
        if landed and item not in landed:
            continue
        per[level] = per.get(level, 0) + 1
        for line in (getattr(res, "level_reasons", None) or {}).get(item, []):
            m = re.search(r"\[[a-z_]+:[a-z_-]+\]", line)
            code = m.group(0) if m else line.split(":", 1)[0]
            codes[code] = codes.get(code, 0) + 1
    return dict(sorted(per.items())), dict(sorted(codes.items()))


# --- inputs ------------------------------------------------------------------------------------------------


def export_for_writer(
    load: Callable[[str], Episode],
    eids: Iterable[str],
    dest: Path,
    *,
    library_mode: bool = False,
    usage_dest: Path | None = None,
) -> None:
    """Write ``<dest>/<episode_id>.json`` per episode: the request, cells and actions, and nothing else.

    With *library_mode* the row also carries the declared regime, the memory pin, the work-tree diff, the
    structural events, and each cell's error and language. The episode's library-use record is harness telemetry,
    not part of the episode: with *usage_dest* it goes to ``<usage_dest>/<episode_id>.json``
    (:func:`.usage.writer_usage_record`, host paths removed), and it is never in the episode file. The
    allowlist below applies only with *library_mode* off.

    No outcome, signal or checker data reaches the writer. The fields are an allowlist: cells carry
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
            # the gate's own mapping (G2), so the writer never derives a channel name
            "memory_channels": [
                env_channel(getattr(a, "kind", "tool"), a.channel) for a in ep.actions
            ],
        }
        if library_mode:
            from .batch_map import structural_events

            row["regime"] = ep.regime
            row["memory_main"] = ep.memory_main
            row["worktree_diff"] = ep.worktree_diff or ""
            row["structural_events"] = structural_events(ep)
            if usage_dest is not None:
                from .usage import writer_usage_record

                Path(usage_dest).mkdir(parents=True, exist_ok=True)
                (Path(usage_dest) / f"{eid}.json").write_text(
                    json.dumps(
                        writer_usage_record(ep.memory_use),
                        sort_keys=True,
                        default=str,
                        indent=1,
                    )
                    + "\n",
                )
            for c_row, c in zip(row["cells"], ep.cells):
                c_row["error"] = c.error
                c_row["language"] = c.language
        (dest / f"{eid}.json").write_text(
            json.dumps(row, sort_keys=True, default=str, indent=1) + "\n",
        )


def export_blobs(
    load: Callable[[str], Episode],
    eids: Iterable[str],
    blobs: BlobStore | None,
    dest: Path,
    *,
    per_blob_bytes: int | None = EXPORT_BLOB_BYTES,
    total_bytes: int | None = EXPORT_TOTAL_BYTES,
    record_bytes: bool = False,
) -> dict:
    """Copy the file blobs the episodes' worktree actions recorded to ``<dest>/<blob id>``, size-capped.

    Only well-formed ids (:data:`.blobs.BLOB_ID`) of ``blob_before``/``blob_after`` on ``kind ==
    "worktree"`` actions are looked up, in episode and action order. A blob larger than *per_blob_bytes*,
    one past *total_bytes* in all, or one missing from the store is skipped; a cap of ``None`` is no cap
    ``<dest>/index.json`` lists
    ``{"exported": [...], "skipped": {id: reason}}``.
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
                if per_blob_bytes is not None and size > per_blob_bytes:
                    skipped[sha] = (
                        f"{size} bytes, over the per-blob cap of {per_blob_bytes}"
                    )
                elif total_bytes is not None and used + size > total_bytes:
                    skipped[sha] = f"over the export's total cap of {total_bytes} bytes"
                else:
                    (dest / sha).write_bytes(blobs.get(sha))
                    used += size
                    exported.append(sha)
    out: dict = {"exported": exported, "skipped": skipped}
    if (
        record_bytes
    ):  # library mode: the bytes actually staged, so an uncapped export's size is on record
        out["exported_bytes"] = used
    (dest / "index.json").write_text(json.dumps(out, sort_keys=True) + "\n")
    return out


def _plain(o: object, chars: int) -> object:
    """JSON data with every string cut to *chars* characters (keys included)."""
    if isinstance(o, str):
        return o[:chars]
    if isinstance(o, dict):
        return {str(k)[:chars]: _plain(v, chars) for k, v in o.items()}
    if isinstance(o, list):
        return [_plain(v, chars) for v in o]
    return o


def _scrub(o: object) -> object:
    """JSON data with :func:`.redact.redact_error` applied to every string and key: every secret registered
    with :mod:`unify.process_secrets` (the writer's route token among them) as it stands now, credential structures
    and key shapes."""
    if isinstance(o, str):
        return redact_error(o)
    if isinstance(o, dict):
        return {redact_error(str(k)): _scrub(v) for k, v in o.items()}
    if isinstance(o, list):
        return [_scrub(v) for v in o]
    return o


#: sha256 of each brief or tool JSON whose text a transcript of this process already carries (when no blob store
#: holds it): the text is stored once per run, and later transcripts name it by its hash.
_TRANSCRIPT_TEXTS: set[str] = set()


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _reasoning_text(m: Mapping) -> str | None:
    """The reasoning summary a provider returned on an assistant message, if any: ``reasoning_details`` summaries
    and texts, ``reasoning_content``/``reasoning``, and the same under ``provider_specific_fields`` (with its
    ``summary`` and ``thinking_blocks``). None when there is none."""
    found: list[str] = []

    def take(v: object) -> None:
        if isinstance(v, str) and v.strip():
            found.append(v)
        elif isinstance(v, list):
            for item in v:
                if isinstance(item, dict):
                    for key in ("summary", "text", "thinking"):
                        take(item.get(key))
                else:
                    take(item)

    psf = m.get("provider_specific_fields")
    for src in (m, psf if isinstance(psf, Mapping) else {}):
        for key in (
            "reasoning_details",
            "reasoning_content",
            "reasoning",
            "summary",
            "thinking_blocks",
        ):
            take(src.get(key))
    return "\n".join(found) if found else None


def transcript_lines(
    messages: list,
    pass_id: str,
    redactor: Redactor | None = None,
    *,
    max_bytes: int | None = None,
    string_chars: int | None = None,
    tools: list | None = None,
    originals: Mapping[int, str] | None = None,
    store: Callable[[bytes], str] | None = None,
    label: Mapping[str, str] | None = None,
) -> list[str]:
    """The writer's message list as JSON lines ``{"pass_id", "i", "message"}``, redacted, every message whole.
    *label* (for a helper: ``{"agent": "helper", "helper": <name>}``) is added to every line.

    The first line is a header ``{"pass_id", "header": {...}}`` naming the brief (the system message) and the tool
    JSON by sha256 with their sizes. With *store* (a content-addressed store such as the gate's blob store) their
    bytes are put there; without one, the header carries their text the first time this process writes them, so
    each is stored once per run. The system message itself is then named, ``(brief sha256 <hash>)``, not copied.

    Each message becomes plain JSON, then *redactor* (the run's environment credentials and registered secrets by
    value, then key shapes; key shapes alone without one) and :func:`.redact.redact_error` (every secret registered
    with :mod:`unify.process_secrets` when the lines are made, credential structures, key shapes) rewrite every
    string and key. A tool result that compaction moved out of the conversation is recorded as it was before
    (*originals*, by message index) and marked ``"compacted": true``. A message that is not JSON data is named by
    its type.

    With neither *max_bytes* nor *string_chars* (the stored transcript, :mod:`.transcripts`) nothing is cut. A
    bounded view may ask for one: redaction comes before any cut, so a cut never leaves part of a secret
    unredacted; every string is cut to *string_chars* characters, except in the first user message, which is kept
    whole; and the lines stop before *max_bytes* in all, ending with ``{"pass_id", "truncated": n}`` for the *n*
    messages left out.
    """
    red = redactor if redactor is not None else Redactor()
    originals = originals or {}
    label = dict(label or {})
    marker_room = 200
    lines: list[str] = []
    used = 0

    carried: list[str] = []  # hashes whose text this header carries

    def text_ref(kind: str, text: str, inline: bool) -> dict:
        data = text.encode("utf-8")
        sha = _sha(data)
        ref: dict = {f"{kind}_sha256": sha, f"{kind}_chars": len(text)}
        if store is not None:
            try:
                store(data)
                ref[f"{kind}_stored"] = "blobs"
                return ref
            except Exception:  # noqa: BLE001 - the transcript keeps the text instead
                pass
        if sha in _TRANSCRIPT_TEXTS:
            ref[f"{kind}_stored"] = "earlier transcript"
        elif inline:
            ref[kind] = _scrub(red.obj(text))
            ref[f"{kind}_stored"] = "transcript"
            carried.append(sha)
        else:
            ref[f"{kind}_stored"] = "not stored: larger than the transcript's size"
        return ref

    first = messages[0] if messages else None
    brief = (
        first.get("content")
        if isinstance(first, dict) and first.get("role") == "system"
        else None
    )
    tools_text = (
        json.dumps(tools, sort_keys=True, default=str, ensure_ascii=False)
        if tools
        else None
    )

    def header_line(inline: bool) -> str | None:
        header: dict = {}
        if isinstance(brief, str):
            header |= text_ref("brief", brief, inline)
        if tools_text is not None:
            header |= text_ref("tools", tools_text, inline)
        if not header:
            return None
        return json.dumps(
            {"pass_id": pass_id, "header": header, **label},
            sort_keys=True,
        )

    line = header_line(True)
    if (
        line is not None
        and max_bytes is not None
        and len(line.encode("utf-8")) + 1 > (max_bytes - marker_room) // 2
    ):
        # the texts would crowd out the messages: name them by hash only
        carried.clear()
        line = header_line(False)
    if line is not None:
        _TRANSCRIPT_TEXTS.update(carried)
        lines.append(line)
        used += len(line.encode("utf-8")) + 1
    first_user = next(
        (
            i
            for i, m in enumerate(messages)
            if isinstance(m, dict) and m.get("role") == "user"
        ),
        None,
    )
    for i, m in enumerate(messages):
        extra: dict = {}
        # the brief, at the start and again after each compaction's restart, is named by its hash
        is_brief = (
            isinstance(brief, str)
            and isinstance(m, dict)
            and m.get("role") == "system"
            and m.get("content") == brief
        )
        if is_brief:
            m = {
                "role": "system",
                "content": f"(brief sha256 {_sha(brief.encode('utf-8'))})",
            }
        elif i in originals and isinstance(m, dict):
            m = {**m, "content": originals[i]}
            extra["compacted"] = True
        if label and isinstance(m, dict) and m.get("role") == "assistant":
            # a helper's turn: the keys its message kept in the history, and the provider's reasoning summary
            extra["message_keys"] = sorted(str(k) for k in m)
            summary = _reasoning_text(m)
            if summary is not None:
                extra["reasoning_summary"] = _plain(
                    _scrub(red.obj(summary)),
                    string_chars,
                )
        try:
            plain = _scrub(red.obj(json.loads(json.dumps(m, default=str))))
            # the first user message and the brief's name are kept whole; every other string is cut
            whole = i == first_user or is_brief
            msg = (
                plain if whole or string_chars is None else _plain(plain, string_chars)
            )
            line = json.dumps(
                {"pass_id": pass_id, "i": i, "message": msg, **extra, **label},
                sort_keys=True,
            )
        except (RecursionError, ValueError, TypeError):
            line = json.dumps(
                {
                    "pass_id": pass_id,
                    "i": i,
                    "message": f"(unserialisable {type(m).__name__})",
                    **label,
                },
            )
        size = len(line.encode("utf-8")) + 1
        if max_bytes is not None and used + size > max_bytes - marker_room:
            lines.append(
                json.dumps(
                    {"pass_id": pass_id, "truncated": len(messages) - i, **label},
                ),
            )
            break
        lines.append(line)
        used += size
    return lines


def record_lines(
    entries: list,
    pass_id: str,
    redactor: Redactor | None = None,
    *,
    max_bytes: int | None = None,
    string_chars: int | None = None,
) -> list[str]:
    """The pass record's entries (:mod:`.messaging`) as JSON lines ``{"pass_id", "record": {seq, ts, author, kind,
    text, mentions}}``, redacted, every entry whole; a bounded view (*max_bytes*, *string_chars*) cuts as
    :func:`transcript_lines` does."""
    red = redactor if redactor is not None else Redactor()
    lines: list[str] = []
    used = 0
    for n, e in enumerate(entries):
        entry = _scrub(red.obj(e.to_dict()))
        entry = entry if string_chars is None else _plain(entry, string_chars)
        line = json.dumps({"pass_id": pass_id, "record": entry}, sort_keys=True)
        size = len(line.encode("utf-8")) + 1
        if max_bytes is not None and used + size > max_bytes - 200:
            lines.append(
                json.dumps({"pass_id": pass_id, "record_truncated": len(entries) - n}),
            )
            break
        lines.append(line)
        used += size
    return lines


# --- the box's tree ----------------------------------------------------------------------------------------


#: the quota measure's answer when the host cannot restore its access without a path (no /proc): fail closed
NO_PROC_BREACH = "over quota: the tree cannot be measured safely (/proc/self/fd is not available)"


def _measure(root: Path) -> str | None:
    """The quota the tree breaks, or None. Entries are made owner-accessible on the way.

    Walks from descriptors (:func:`.safe_paths.walk_unlocking`: no link is followed, no file is read, a mode is
    changed only on the opened directory or regular file itself, through /proc) and stops at the first breach:
    more than :data:`QUOTA_ENTRIES` files, links and directories, a regular file over :data:`QUOTA_FILE_BYTES`,
    regular files over :data:`QUOTA_TOTAL_BYTES` in all, or an entry that stays unreadable after its owner's
    access is restored (a ``chmod 0`` directory could otherwise hide any amount). Box code may change the tree
    while it is measured: a directory swapped for a link is never followed, so no mode outside the tree changes.
    """
    entries = total = 0
    unreadable: list[str] = []
    walk = _safe.walk_unlocking(root, 0o700, 0o600)
    try:
        for event, r, st in walk:
            if event != "entry" or st is None:
                unreadable.append(r)
                continue
            entries += 1
            if entries > QUOTA_ENTRIES:
                return f"over quota: more than {QUOTA_ENTRIES} files and directories under /memory"
            if stat.S_ISREG(st.st_mode):
                if st.st_size > QUOTA_FILE_BYTES:
                    return (
                        f"over quota: /memory/{r[:200]} has {st.st_size} bytes "
                        f"(at most {QUOTA_FILE_BYTES} per file)"
                    )
                total += st.st_size
                if total > QUOTA_TOTAL_BYTES:
                    return f"over quota: more than {QUOTA_TOTAL_BYTES} bytes under /memory"
    except _safe.NoProc:
        return NO_PROC_BREACH
    finally:
        walk.close()
    if unreadable:
        return f"over quota: unreadable entries {sorted(unreadable)[:5]}"
    return None


def _layout_problems(
    box: Path,
    skip_top: frozenset[str] = frozenset({".pass"}),
) -> list[str]:
    """What the gate refuses to import from the writer's ``/memory``: the candidate is taken as regular files and
    directories only. Refused: any path component starting with ``.git`` (git metadata, attributes or modules would
    change how the harness's own checkout behaves), symbolic links, hard links (a regular file with more than one
    link) and special files. The walk never follows a link; *skip_top* names at the top are not part of the
    candidate."""
    found: list[str] = []
    stack = [(Path(box), "", True)]
    while stack:
        d, rel, top = stack.pop()
        try:
            with os.scandir(d) as it:
                entries = sorted(it, key=lambda e: e.name)
        except OSError:
            continue  # unreadable: the quota measure refuses it
        for e in entries:
            if top and e.name in skip_top:
                continue
            r = rel + e.name
            if e.name.startswith(".git"):
                found.append(f"{r}: a path starting with .git")
                continue
            if e.name in _NEVER_COPIED:
                continue
            try:
                st = os.lstat(e.path)
            except OSError:
                continue
            if stat.S_ISLNK(st.st_mode):
                found.append(f"{r}: a symbolic link")
            elif stat.S_ISDIR(st.st_mode):
                stack.append((Path(e.path), r + "/", False))
            elif stat.S_ISREG(st.st_mode):
                if st.st_nlink > 1:
                    found.append(f"{r}: a hard link ({st.st_nlink} links)")
            else:
                found.append(f"{r}: not a regular file or directory")
    return sorted(found)


def _accepted_manifest(box: Path) -> dict | None:
    """The raw manifest dict in *box* at an accepted finish, or None when there is none that parses to an object."""
    try:
        found, manifest, problem = _read_manifest(box)
    except Exception:  # noqa: BLE001 - recording the manifest never fails the finish
        return None
    return (
        manifest if found and problem is None and isinstance(manifest, dict) else None
    )


def _layout_refusal(found: list[str]) -> str:
    shown = "; ".join(f"/memory/{x}" for x in found[:10])
    more = f" (and {len(found) - 10} more)" if len(found) > 10 else ""
    return (
        "layout: the library is taken as regular files and directories only; remove or replace "
        f"{shown}{more}"
    )[:900]


#: How :func:`_mirror` opens what it copies: never through a link, so an entry swapped while it walks is not followed
_MIRROR_DIR = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC
_MIRROR_FILE = os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK | os.O_CLOEXEC


def _mirror(
    src: Path,
    dst: Path,
    skip_top: frozenset[str] = frozenset(),
) -> list[str]:
    """Copy *src* into the existing *dst* without following links; return the entries left out.

    Directories and regular files (with their permission bits) are copied, links are recreated as links
    (the gate refuses them); FIFOs, sockets, devices and anything that cannot be read are left out (an
    unreadable directory is left empty). ``.git``, ``__pycache__``, ``.pytest_cache`` and ``.hypothesis`` are
    never copied, at any depth, nor *skip_top* names at the top.

    *src* may be written while it is copied (a box with it bound read-write runs beside a helper's copy), so the
    walk holds a descriptor for each directory and opens every entry relative to it with ``O_NOFOLLOW``: an
    entry swapped for a link after it was listed is recreated as the link or left out, never followed.
    """
    left_out: list[str] = []

    def copy_file(dfd: int, name: str, dp: Path) -> None:
        fd = os.open(name, _MIRROR_FILE, dir_fd=dfd)
        try:
            st = os.fstat(fd)
            if not stat.S_ISREG(st.st_mode):
                raise OSError(f"{name} is not a regular file")
            with open(dp, "xb") as out:
                while True:
                    chunk = os.read(fd, 1 << 20)
                    if not chunk:
                        break
                    out.write(chunk)
            os.chmod(dp, stat.S_IMODE(st.st_mode))
        finally:
            os.close(fd)

    def walk(dfd: int, d: Path, rel: str, top: bool) -> None:
        try:
            with os.scandir(dfd) as it:
                names = sorted(e.name for e in it)
        except OSError:
            left_out.append(rel.rstrip("/") or ".")
            return
        for name in names:
            if (
                name in _NEVER_COPIED
                or name.startswith(".git")
                or (top and name in skip_top)
            ):
                continue  # git metadata never enters a checkout the harness runs git in
            dp, r = d / name, rel + name
            try:
                st = os.stat(name, dir_fd=dfd, follow_symlinks=False)
                if stat.S_ISLNK(st.st_mode):
                    os.symlink(os.readlink(name, dir_fd=dfd), dp)
                elif stat.S_ISDIR(st.st_mode):
                    dp.mkdir()
                    try:
                        cfd = os.open(name, _MIRROR_DIR, dir_fd=dfd)
                    except OSError:
                        left_out.append(
                            r,
                        )  # unreadable, or swapped for a link since it was listed
                        continue
                    try:
                        walk(cfd, dp, r + "/", False)
                    finally:
                        os.close(cfd)
                elif stat.S_ISREG(st.st_mode):
                    copy_file(dfd, name, dp)
                else:
                    left_out.append(r)
            except OSError:
                left_out.append(r)

    root = os.open(src, _MIRROR_DIR)
    try:
        walk(root, Path(dst), "", True)
    finally:
        os.close(root)
    return left_out


def _drop_unchanged(tree: Path, generated: Mapping[str, bytes]) -> None:
    """Remove each generated file of a library-mode box (never committed) that still holds the bytes the
    harness wrote, then ``.memory/`` if it is left empty. An edited one stays, so the gate sees and refuses it.

    *tree* can hold links the box made (a copy keeps them as links): each file is read and removed from its own
    directory's descriptor, never through a link (:mod:`.safe_paths`).
    """
    from .layout import DATA_DIR

    try:
        root = os.open(tree, _safe.DIR_FLAGS)
    except OSError:
        return
    try:
        for rel, data in generated.items():
            try:
                parts = _safe.rel_parts(rel)
                dfd = _safe.walk_dirs(root, parts[:-1])
            except (OSError, ValueError):
                continue
            try:
                if _safe.read_at(dfd, parts[-1], len(data)) == data:
                    os.unlink(parts[-1], dir_fd=dfd)
            except OSError:
                pass
            finally:
                os.close(dfd)
        try:
            dfd = _safe.walk_dirs(root, [DATA_DIR])
        except OSError:
            return
        try:
            empty = not _safe.listing(dfd)
        except OSError:
            empty = False
        finally:
            os.close(dfd)
        if empty:
            try:
                os.rmdir(DATA_DIR, dir_fd=root)
            except OSError:
                pass
    finally:
        os.close(root)


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
    that parses (a nesting too deep to parse is a problem, not a crash). Both are opened from the box's descriptor
    one component at a time (:mod:`.safe_paths`), so what is parsed, and then judged, is this read's own copy of
    the bytes, whatever box code does to the file meanwhile.
    """
    try:
        root = os.open(box, _safe.DIR_FLAGS)
    except OSError as exc:  # /memory itself made unreadable
        return True, None, f"manifest: /memory cannot be read ({exc.strerror})"
    try:
        try:
            pfd = _safe.walk_dirs(root, [".pass"])
        except _safe.Missing:
            if _safe.lstat_at(root, ".pass") is None:
                return False, None, None
            return (
                True,
                None,
                "manifest: /memory/.pass is not a directory, so no regular file was read",
            )
        except _safe.LinkFound:
            return (
                True,
                None,
                "manifest: /memory/.pass is not a directory, so no regular file was read",
            )
        except OSError as exc:
            return True, None, f"manifest: /memory/.pass cannot be read ({exc.strerror})"
        try:
            try:
                fd = _safe.openat("manifest.json", _safe.READ_FLAGS, pfd, _safe.NO_CREATE_MODE)
            except FileNotFoundError:
                return False, None, None
            except OSError as exc:
                return (
                    True,
                    None,
                    f"manifest: /memory/.pass/manifest.json is not a regular file that can be read ({exc.strerror})",
                )
        finally:
            os.close(pfd)
    finally:
        os.close(root)
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
    if len(data) > _MANIFEST_MAX_BYTES:
        return True, None, f"manifest: larger than {_MANIFEST_MAX_BYTES} bytes"
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


def _library_trailers(manifest: object, sources: list[str]) -> dict[str, list[str]]:
    """The library-mode commit trailers: ``Why`` (the manifest's one line, at most 200 characters),
    ``Episodes`` (the source episodes) and ``Items`` (the item ids the manifest names).
    """
    m = manifest if isinstance(manifest, dict) else {}
    why = m.get("why")
    line = (
        " ".join(why.split())[:200]
        if isinstance(why, str) and why.strip()
        else "(not given)"
    )
    raw = m.get("items") if isinstance(m.get("items"), list) else []
    items = sorted(
        {
            it["item"]
            for it in raw
            if isinstance(it, dict) and isinstance(it.get("item"), str)
        },
    )
    return {"Why": [line], "Episodes": list(sources), "Items": items or ["(none)"]}


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
    #: model calls running now (the writer's and its helpers'), for the USD stop (usd_room)
    in_flight: int = 0


#: The full result a library-mode pass cancelled before its gate leaves in its draft.
CANCELLED_RESULT = (
    "# Gate result: pass {pass_id}, cancelled before the gate ran\n"
    "\n"
    "- passed: no\n"
    "- candidate: none\n"
    "- refused checks: none (nothing was checked)\n"
    "- items refused: {items}\n"
    "\n"
    "The pass was cancelled by its wall-clock bound or at shutdown before its work reached the gate. patch.diff "
    "holds the work as it stood; none of it has been checked.\n"
)


def _library_item_ids(box: Path) -> set[str]:
    """The ids of the functions and notes the library copy at *box* holds (its files alone; nothing runs)."""
    from .layout import discover

    try:
        lib = discover(Path(box))
    except (OSError, ValueError):
        return set()
    return {f.item_id for f in lib.functions} | {n.item_id for n in lib.notes}


def _named_items(box: Path) -> list[str]:
    """The item ids the box's manifest names, if it can be read (a cancelled pass's draft names them)."""
    try:
        found, raw, problem = _read_manifest(Path(box))
    except (OSError, ValueError):
        return []
    if not found or problem is not None:
        return []
    items = raw.get("items") if isinstance(raw, dict) else None
    return sorted(
        {
            it["item"]
            for it in items or []
            if isinstance(it, dict) and isinstance(it.get("item"), str)
        },
    )


class _KeepOnCancel:
    """Library mode: a pass cancelled at its bound or at shutdown keeps the writer's work as a draft. It is the
    innermost context of :meth:`WriterPass._run`'s ``with``, so it exits first, while the box still exists.
    """

    def __init__(self, writer: "WriterPass", parent: str, pass_id: str) -> None:
        self.writer, self.parent, self.pass_id = writer, parent, pass_id

    def __enter__(self) -> "_KeepOnCancel":
        return self

    def __exit__(self, et, ev, tb) -> bool:
        live = getattr(self.writer, "_live", None)
        if (
            et is not None and issubclass(et, asyncio.CancelledError) and live is not None
        ):
            self.writer._cancel_patch = self.writer._keep_cancelled_work(
                live[0],
                live[1],
                self.parent,
                self.pass_id,
            )
        return False


class WriterPass:
    """One consolidation pass, bounded by its step guard ``max_calls``, the run's room ``usd_limit`` and ``deadline_s``.

    Each model call's reported cost is added to the spend; an unknown (or unreadable) cost is counted, and the room
    books it at ``usd_call_worst``, so unpriced calls cannot run away. Money is a pure
    decimal string (never exponent notation): the recorded and returned USD is the known spend
    only, and the count of unpriced calls is recorded as the reason ``note: N unpriced calls`` and in
    :attr:`PassOutcome.unknown_cost_calls`. The pass's own notes follow the gate's reasons in the row.

    The writer sees the episodes through :func:`export_for_writer` over *load* (the allowlist is enforced here, not by
    the caller). A pass that ends with an exception, cancellation included, is recorded as failed.

    In library mode a request of kind ``curate`` runs the writer's second role (:mod:`.curate`) on *curate*, the library
    state its trigger read.
    """

    #: tools in code: how an agent's persistent interpreter starts (``BoxInterpreter.start``'s signature); None
    #: uses :class:`unify.common.box_interpreter.BoxInterpreter`
    #: tools in code: how long the pass's end waits for its tool handler threads before it refuses its gate
    in_code_idle_s: float = 60.0
    interpreter_start: Any = None

    def __init__(
        self,
        memory: Repo,
        gate: Gate,
        evidence: EvidenceStore,
        load: Callable[[str], Episode],
        model_turn: ModelTurn,
        config: PassConfig,
        *,
        redactor: Redactor | None = None,
        helper_turn: Callable[[], ModelTurn] | None = None,
        helper_refusal: str | None = None,
    ) -> None:
        self.mem, self.gate, self.ev = memory, gate, evidence
        # writer-launched helpers (helpers.py): a fresh host-side model turn per helper on the helper lane, or None
        # (no helper lane: delegate is not offered; helper_refusal says why)
        self.helper_turn, self.helper_refusal = helper_turn, helper_refusal
        self._helper_rows: list[dict] = []
        self._delegate_ctx: Any = (
            None  # the current pass run's helpers.DelegateContext, made at its first delegate
        )
        # the current pass run's record (messaging.PassRecord): the writer and every helper it starts
        self._pass_record: Any = None
        self.load, self.turn, self.cfg = load, model_turn, config
        self.redactor = redactor  # for the transcript: the run's registered secrets
        # the last run's message list, kept for :meth:`transcript`
        self.messages: list[dict] = []
        # what the writer saw, for the transcript: the tool JSON it was given and each compacted tool result's
        # text before compaction (message index -> text)
        self._tools_json: list[dict] = []
        # what compactions took out of the writer's conversation, in order (compaction_archive), for the transcript
        self._archived: list[dict] = []
        self._compaction = {"compactions": 0, "chars_moved": 0, "messages_moved": 0}
        self._max_prompt: int | None = None
        # input, cache read, calls that reported input, calls that reported cache read, calls counted
        self._cache_tokens: list[int] = [0, 0, 0, 0, 0]
        # output tokens, calls that reported them
        self._output_tokens: list[int] = [0, 0]
        # the first counted call's prompt tokens: (counted, value)
        self._first_prompt: tuple[bool, int | None] = (False, None)
        # with staging on, one end-event row per batch episode (summaries.episode_row), filled when staging is listed
        self._staging_episodes: list[dict] | None = None
        # the fixture provenance last derived from the box's files (fixtures.derive_ledger); the manifest's own is ignored
        self.fixtures_made: dict[str, dict] = {}
        # per agent: sha256 of a part shown to that agent in full -> (episode, part), so a byte-identical part is
        # answered with a pointer; and that agent's read log (a measure only)
        self._shown_full: dict[str, dict[str, tuple[str, str]]] = {}
        self._read_logs: dict[str, _wt.ReadLog] = {}
        # the recorded-episode index (loaded once per pass) and the episodes loaded through it
        self._recorded: set[str] | None = None
        self._episodes: dict[str, Episode] = {}
        self._batch: list[str] = []
        self._bmap_rows: list[dict] = []
        # one structured event per tool call (tool-events.jsonl rows), in call order; and the pass's events file
        # (None: not opened yet; False: not written this pass)
        self.tool_events: list[dict] = []
        self._events_file: Path | None | bool = None
        # the raw manifest dict of the pass's last accepted finish (either role), else None; consolidation reads it
        # for what the pass kept
        self.final_manifest: dict | None = None
        # request blocks shared by most episodes (id -> text), shown as markers; {} until staged
        self._shared: dict[str, str] = {}
        self._inputs_dir: Path | None = (
            None  # the pass's /inputs on the host, once staged
        )
        self._staging_counts: dict | None = None
        # per batch episode, its capture's batch-map row (captures.batch_row); empty without the capture store
        self._capture_rows: dict = {}
        # the candidate pool (pool.py): loaded and fed with this pass's staging at the pass's start, saved again at
        # its end with the writer's decisions
        self._pool = None
        self._pool_open: int | None = None
        self._pool_error: str | None = None
        self._pool_at_start: dict | None = None
        # this pass's staging by episode (empty with staging off), and the staging root it was read from
        self._staged_views: dict = {}
        self._staging_read_root: Path | None = None
        # the gate's {"item", "adopts"} rows for the items that landed (empty until the merge)
        self._provenance: list[dict] = []
        self._levels: tuple[dict | None, dict | None] = (None, None)
        self._chain_rows: list[dict] = (
            []
        )  # the batch map's rows once staging is listed in it
        self._pool_manifest: object = None
        # the open drafts the last _stage_inputs staged; always [] with library_mode off
        self._drafts: list[dict] = []
        self._library_generated: dict[str, bytes] = (
            {}
        )  # library mode: the generated files written into the box
        # library mode: the live box and checkout, and a cancelled pass's (patch blob, named items)
        self._live: tuple[Path, Path] | None = None
        self._cancel_patch: tuple[str | None, list[str]] = (None, [])

    def transcript(self, pass_id: str) -> list[str]:
        """The last run's messages as redacted JSON lines, every message whole (:func:`transcript_lines`): the brief
        and the tool JSON by sha256 (their bytes in the gate's blob store when it has one), and each compacted tool
        result as it was before the compaction."""
        blobs = getattr(self.gate, "blobs", None)
        store = blobs.put if isinstance(blobs, BlobStore) else None
        lines = transcript_lines(
            [*self._archived, *self.messages],
            pass_id,
            self.redactor,
            tools=self._tools_json,
            store=store,
        )
        # each helper's own transcript, bounded the same way, every line naming the helper; then the pass's record
        ctx = self._delegate_ctx
        if ctx is not None and ctx.pass_id == pass_id:
            for name, (messages, tools) in ctx.transcripts.items():
                lines += transcript_lines(
                    [*ctx.archived.get(name, []), *messages],
                    pass_id,
                    self.redactor,
                    tools=tools,
                    store=store,
                    label={"agent": "helper", "helper": name},
                )
        pr = self._pass_record
        if pr is not None and pr.record.entries:
            lines += record_lines(list(pr.record.entries), pass_id, self.redactor)
        return lines

    def _count_cache(self, usage: dict) -> None:
        """Count a model call: add its prompt, cache-read and output tokens to the pass's sums, each when it
        reported it; keep the largest prompt any call reported and the first call's prompt."""
        inp, cached, out = usage.get("input"), usage.get("cache_read"), usage.get("output")
        inp = inp if isinstance(inp, int) and not isinstance(inp, bool) else None
        cached = cached if isinstance(cached, int) and not isinstance(cached, bool) else None
        if isinstance(out, int) and not isinstance(out, bool):
            self._output_tokens[0] += out
            self._output_tokens[1] += 1
        self._cache_tokens[4] += 1
        if not self._first_prompt[0]:
            self._first_prompt = (True, inp)
        if inp is not None:
            self._max_prompt = max(self._max_prompt or 0, inp)
            self._cache_tokens[0] += inp
            self._cache_tokens[2] += 1
        if cached is not None:
            self._cache_tokens[1] += cached
            self._cache_tokens[3] += 1

    def _compaction_event(
        self,
        pass_id: str,
        messages: list[dict],
        comp: Compaction,
    ) -> None:
        """Count a compaction (when it restarted the conversation) and write its ``compaction`` row to the pass's
        events: the summary call's usage and charge, as a tool call's row carries its model call's, and what it
        took out of the conversation."""
        usage = _usage_of(comp.reply, comp.usd)
        if isinstance(comp.reply, dict):
            self._count_cache(usage)
        kept = {id(m) for m in comp.restart or []}
        out = (
            [m for m in messages if id(m) not in kept]
            if comp.restart is not None
            else []
        )
        chars = sum(_prompt_chars(m) for m in out)
        if comp.restart is not None:
            self._compaction["compactions"] += 1
            self._compaction["messages_moved"] += len(out)
            self._compaction["chars_moved"] += chars
        self._write_event(
            pass_id,
            {
                "event": "compaction",
                "agent": self._role_agent(),
                "pass_id": pass_id,
                "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "restarted": comp.restart is not None,
                "messages_before": len(messages),
                "messages_after": (
                    len(comp.restart) if comp.restart is not None else len(messages)
                ),
                "messages_moved": len(out),
                "chars_moved": chars,
                "usage": usage,
            },
        )

    def _role_agent(self) -> str:
        pr = self._pass_record
        return (
            pr.root
            if pr is not None
            else ("writer")
        )

    def transcript_note(self, pass_id: str) -> list[str]:
        """Store the last run's whole transcript (:meth:`transcript`) as chunk files under the pass's record
        folder (:func:`.transcripts.write_chunks`, with ``pass_record_root``), and return the git note's lines: the
        transcript itself while it fits, else a row pointing to the chunks with their sha256
        (:func:`.transcripts.note_lines`). With no record folder, or one that cannot be written, the note carries
        the whole transcript: no message is dropped."""
        from . import transcripts as _transcripts

        lines = self.transcript(pass_id)
        manifest = None
        root = getattr(self.cfg, "pass_record_root", None)
        if root:
            try:
                manifest = _transcripts.write_chunks(
                    root,
                    pass_id,
                    _transcripts.WRITER,
                    lines,
                )
            except (OSError, ValueError):
                manifest = None  # the note keeps it all instead
        return _transcripts.note_lines(pass_id, lines, manifest)

    def _usage(self, req: PassRequest, tree: Path) -> str:
        """The library-use table over the pass's requests (:func:`.usage.usage_table`) for the functions
        of memory ``main`` at *tree*: harness counts only, nothing of a checker.
        """
        from .usage import usage_table

        try:
            # library mode: the whole table is a file the brief names, /inputs/usage/table.md; nothing of it
            # is inlined in the first message
            from .layout import discover

            ids = [f.item_id for f in discover(Path(tree)).functions]
            full = usage_table(self.ev, list(req.episodes), ids, max_rows=None)
            if self._inputs_dir is not None:
                (self._inputs_dir / "usage").mkdir(exist_ok=True)
                (self._inputs_dir / "usage" / "table.md").write_text(full)
            return full
        except Exception as exc:  # noqa: BLE001 - a measurement never stops a pass
            return f"(library use not available: {type(exc).__name__})\n"

    def _stage_inputs(
        self,
        req: PassRequest,
        inputs: Path,
        tree: Path | None = None,
    ) -> None:
        """The pass's read-only ``/inputs``; *tree* is the parent library (read by nothing here now; kept for the
        callers)."""
        inputs.mkdir()
        _sandbox.stage_open_hook(
            inputs,
        )  # /inputs/sitecustomize.py: the cells' observe-only open log
        store = getattr(self.gate, "blobs", None)
        export_for_writer(
            self.load,
            list(req.episodes),
            inputs / "episodes",
            library_mode=True,
            # the store pass reads each episode's library use as its own file (none for a curate pass, whose use
            # records are /inputs/curate/use.json)
            usage_dest=(
                inputs / "usage"
            ),
        )
        exported = export_blobs(
            self.load,
            list(req.episodes),
            getattr(self.gate, "blobs", None),
            inputs / "blobs",
            per_blob_bytes=None,
            total_bytes=None,
            record_bytes=True,
        )
        self._exported_bytes = exported.get("exported_bytes")
        from . import shared_text as _shared_text
        from .batch_map import build_batch_map

        # count this batch's request blocks once each, then show the shared ones as markers, each
        # block once under /inputs/shared/
        if hasattr(
            self.ev,
            "db",
        ):  # an evidence store (test doubles without one show requests as they are)
            for e in req.episodes:
                ep_ = self.load(e)
                _shared_text.update_counts(
                    self.ev,
                    e,
                    ep_.request[0] if ep_.request else "",
                    ep_.request[1:],
                )
            self._shared = _shared_text.shared(self.ev)
        if self._shared:
            (inputs / "shared").mkdir(exist_ok=True)
            for b, text in self._shared.items():
                (inputs / "shared" / f"{b}.txt").write_text(text)
        (inputs / "batch_map.json").write_text(
            json.dumps(
                build_batch_map(self.load, list(req.episodes), self._shared),
                sort_keys=True,
                default=str,
                indent=1,
            )
            + "\n",
        )

        # the reference tables the brief points to instead of carrying them
        from .library_prompts import input_pointer_files

        for name, text in input_pointer_files().items():
            (inputs / name).write_text(text)
        if self.helper_turn is not None:
            # the brief names /inputs/helpers/ whenever helpers can be launched: it opens from the start, empty
            # until a helper's output is exported there
            from .helpers import EXPORT_DIR as _HELPERS_DIR

            (inputs / _HELPERS_DIR).mkdir(exist_ok=True)
        self._drafts = []
        from .drafts import stage_drafts

        if isinstance(store, BlobStore):
            self._drafts = stage_drafts(self.ev, store, inputs)
        else:
            (inputs / "gate").mkdir(exist_ok=True)

    def _stage_library_library_mode(self, box: Path, parent: str) -> dict[str, bytes]:
        """Library mode: write the generated helper, INDEX.md, links.json and ``.memory/`` into the writer's box
        (its root goes on the cells' path, so ``import memory`` gives ``get`` and ``items``).
        Returns them; :func:`_drop_unchanged` keeps unchanged copies out of the candidate.
        """
        from .catalogue import write_files
        from .library_export import generated_library_mode, item_history

        from .item_records import records_at, status_of

        records, _ = records_at(
            self.mem,
            parent,
        )  # the writer reads the records through memory.get
        files = generated_library_mode(
            box,
            history=item_history(self.mem, parent),
            status_of=status_of(records),
            records=records,
        )
        write_files(box, files)
        return files

    def similar_context(self):
        """The ``similar`` tool's context for this pass (:class:`.similar.Context`): the index folder, the episode
        loader and every recorded episode id (so the reply can name episodes that are not indexed).
        """
        from . import similar as _similar

        since = getattr(self.ev, "experience_since", None)
        return _similar.Context(
            index_dir=Path(self.cfg.similar_index) if self.cfg.similar_index else None,
            load=self.load,
            recorded=(lambda: [e for e, _ in since(0)]) if since is not None else None,
        )

    def _similar_call(self, args: dict) -> "_wt.Reply":
        """The ``similar`` tool on the pass thread; a refusal keeps its form code."""
        from . import similar as _similar

        text = _similar.handle(args, self.similar_context())
        form = _form_of(text)
        if form is None:
            eid = args.get("episode")
            # the episode a cell lookup named; a lookup by code names none
            paths = (
                ({"path": f"/episodes/{eid}", "mode": "search"},)
                if isinstance(eid, str)
                else ()
            )
            return _wt.Reply(text, paths=paths)
        return _wt.Reply(text, "refused", form)

    async def _similar_async(self, args: dict) -> str:
        """The ``similar`` tool for a helper (its handler is synchronous and bounded)."""
        from . import similar as _similar

        return _similar.handle(args, self.similar_context())

    def _events_path(self, pass_id: str) -> Path | None:
        """The pass's ``tool-events.jsonl``, created now if no row has been written yet (so helpers append to the
        same file the writer created); None when the pass writes no events file."""
        root = getattr(self.cfg, "pass_record_root", None)
        if not root or self._events_file is False:
            return None
        if isinstance(self._events_file, Path):
            return self._events_file
        self._events_file = False
        try:
            folder = Path(root) / pass_id
            for d in (Path(root), folder):
                d.mkdir(parents=True, exist_ok=True)
                if d.is_symlink() or not d.is_dir():
                    return None
            path = folder / "tool-events.jsonl"
            fd = os.open(
                path,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC,
                0o644,
            )
            os.close(fd)
        except OSError:
            return None  # a measurement never stops a pass
        self._events_file = path
        return path

    def _helper_context(
        self,
        pass_id: str,
        inputs: Path,
        box: Path,
        scratch: Path,
        deadline: float,
        spend: Any,
    ) -> Any:
        """This pass run's delegate context (made at its first delegate), on the pass's own record."""
        from . import helpers as _helpers

        # one context per pass run (its spend object is the run's own), so call ids never repeat within a pass
        ctx = self._delegate_ctx
        if ctx is None or ctx.pass_id != pass_id or ctx.spend is not spend:
            ctx = _helpers.context_for(
                self,
                pass_id=pass_id,
                inputs=inputs,
                box=box,
                scratch=scratch,
                deadline=deadline,
                spend=spend,
                events=self._events_path(pass_id),
                pass_record=self._pass_record,
            )
            if self.cfg.similar_index:
                ctx.similar = self._similar_async
            self._delegate_ctx = ctx
        return ctx

    async def _close_record(self) -> None:
        """End every helper of this pass run still running (:func:`.messaging.close`); finishing and the end of the
        writer's loop call it."""
        pr = self._pass_record
        if pr is not None:
            await _messaging.close(pr, f"the {pr.root} finished")

    async def _end_record(self) -> None:
        """Every way out of the pass: end its helpers, then write the ``pass_end`` event with ``helpers_closed``
        (each helper: ended or still running) and where the record is. On the run record the pass's agent then
        leaves it (the writer service's writer goes idle)."""
        pr = self._pass_record
        if pr is None:
            return
        try:
            await self._close_record()
        finally:
            pr.end_root()
        # each helper's transcript beside the pass's events (record_transcript), named in the end event
        transcripts: list[dict] = []
        ctx = self._delegate_ctx
        root = getattr(self.cfg, "pass_record_root", None)
        if root and ctx is not None and ctx.pass_id == pr.pass_id:
            from . import transcripts as _transcripts

            for name, (messages, tools) in ctx.transcripts.items():
                transcripts.append(
                    _transcripts.record_transcript(
                        root,
                        pr.pass_id,
                        name,
                        [*ctx.archived.get(name, []), *messages],
                        redactor=self.redactor,
                        tools=tools,
                        label={"agent": "helper", "helper": name},
                    ),
                )
        self._write_event(
            pr.pass_id,
            {
                "event": "pass_end",
                "agent": pr.root,
                "name": pr.root,
                "pass_id": pr.pass_id,
                "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "helpers_closed": dict(pr.closed),
                "helpers": pr.agents()[1:],
                "record": str(pr.record.path),
                "record_entries": len(pr.record.entries),
                "helper_transcripts": transcripts,
                # tools in code: the pass ended with a handler thread still running, so its folder was kept
                **(
                    {"temp_kept": self._temp_kept}
                    if getattr(self, "_temp_kept", None)
                    else {}
                ),
                # tools in code: the handler threads alive when the gate would start; 0 whenever the gate ran,
                # more than 0 beside temp_kept when a handler kept it from starting (None: the pass ended before)
                **(
                    {"handlers_alive_at_gate": getattr(self, "_handlers_at_gate", None)}
                    if getattr(self, "_in_code_pass", False)
                    else {}
                ),
            },
        )

    def _pool_update(self, req: PassRequest, pass_id: str, out: PassOutcome) -> dict:
        """The candidate pool after this pass: the writer's status changes (merged / superseded when the gate landed
        them, retired and reopened when finish was accepted, every open entry a landed item adopts merged, and each
        capture's entry closed by the route the writer took, :func:`.pool.close_captures`), then,
        on a store pass, its deferrals and its draft when nothing landed; one commit on refs/heads/pool. The pass's
        staging entered the pool at its start (:meth:`_staging_library_mode`). Never raises: a failure is recorded
        and the pass's result stands."""
        from . import pool as _pool_mod

        try:
            p = self._pool
            unapplied: list[str] = []
            changed = _pool_mod.apply_block(
                p,
                _wt.without_defer(self._pool_manifest),
                pass_id,
                landed=set(out.items_merged or []),
                finished=bool(out.finished),
                provenance=self._provenance,
                library_items=getattr(self, "_pool_library_items", None),
                notes=unapplied,
            )
            from .captures import landings

            # each capture's entry closes by the one route the writer took: landed by land_capture, or a draft
            # staged from its episode adopted instead
            store = self._captures_root()
            changed += _pool_mod.close_captures(
                p,
                pass_id,
                landed=set(out.items_merged or []),
                landings=landings(store) if store is not None else {},
                provenance=self._provenance,
            )
            new: list = []
            extended: list[str] = []
            # the manifest's defer rows (pool decide defer): episodes join a named entry, or open deferred entries
            deferred: set[str] = set()
            for row in _wt.defer_rows(self._pool_manifest):
                deferred.update(row["episodes"])
                for e in row["episodes"]:
                    if row["id"] == "new":
                        new.append(_pool_mod.deferred(e, row["reason"]))
                    elif p.add_evidence(row["id"], e, row["reason"], pass_id):
                        extended.append(row["id"])
            # a batch episode the writer never opened is kept as unread (a measure: nothing gates on it)
            log = self._read_logs.get("writer") or _wt.ReadLog()
            for e in self._batch:
                if e not in log.parts and e not in deferred:
                    new.append(_unread_entry(_pool_mod, e))
            if not out.passed:
                new += self._pool_draft(pass_id)
            added = 0
            for e in new:
                e.add_provenance([], [pass_id], [])
                added += e.id not in p.entries
                p.upsert(e, pass_id)
            res = {
                "open": len(p.open_entries()),
                "added": added,
                "extended": sorted(set(extended)),
                "changed": changed,
                # decisions that could not apply, each with why (an item that did not land)
                "unapplied": unapplied,
                "commit": p.save(pass_id),
                "at_start": self._pool_at_start,
            }
            # per batch episode, the attempt captured from it and the item it landed as (its pool entry merged), or
            # the item that superseded it (a draft staged from its episode, adopted instead)
            caps = {}
            for eid, cap in sorted((self._capture_rows or {}).items()):
                ent = p.entries.get(_pool_mod.entry_id("capture", cap["id"]))
                status = ent.status if ent is not None else None
                detail = ent.status_detail if ent is not None else {}
                caps[eid] = {
                    "capture_id": cap["id"],
                    "landed": detail.get("item") if status == "merged" else None,
                    "superseded_by": (detail.get("item") or detail.get("by")) if status == "superseded" else None,
                }
            if caps:
                res["captures"] = caps
            return res
        except Exception as exc:  # noqa: BLE001 - the pool never fails a pass
            return {"error": f"{type(exc).__name__}: {str(exc)[:200]}"}

    def _pool_draft(self, pass_id: str) -> list:
        """This pass's draft for the pool (its patch and gate result), [] when it recorded no (or an empty) patch."""
        from . import pool as _pool_mod

        row = next(
            (r for r in self.ev.pass_rounds() if r.get("pass_id") == pass_id),
            None,
        )
        blobs = getattr(self.gate, "blobs", None)
        if not row or not row.get("patch_blob") or blobs is None:
            return []
        patch = blobs.get(row["patch_blob"]).decode("utf-8", "replace")
        if not patch.strip():  # nothing was drafted
            return []
        res = (
            blobs.get(row["gate_blob"]).decode("utf-8", "replace")
            if row.get("gate_blob")
            else ""
        )
        return [
            _pool_mod.draft(
                pass_id,
                patch,
                res,
                sorted(row.get("items_refused") or {}),
            ),
        ]

    async def _staging_library_mode(
        self,
        req,
        inputs: Path,
        box: Path,
        roots,
        spend,
        *,
        pass_id: str = "",
    ) -> None:
        """With staging on (:meth:`_staging_on`), each batch episode's staging: entered into the pool at once (so
        this pass can merge it), copied to ``/inputs/staging/<episode>/`` and listed in batch_map.json. An episode
        whose capture can be landed reaches the writer by that one route: its staging is copied without the fork's
        ``procedure.py``, and its row's ``staging.capture`` names the capture and the call that lands it. Staging is
        untrusted: :mod:`.staging` takes only allowed regular files and nothing staged is run. Helpers are launched
        by the writer itself (the ``delegate`` tool, :mod:`.helpers`), never here.
        """
        import dataclasses

        from . import captures as _captures
        from . import staging as _staging
        from .pool import entry_id as _pool_entry_id

        capture_ids: list[str] = []
        captures_error: str | None = None
        root = (
            Path(self.cfg.staging_root)
            if self.cfg.staging_root
            else Path(inputs).parent / "staging-local"
        )
        path = Path(inputs) / "batch_map.json"
        bmap = json.loads(path.read_text())
        counts: dict[str, int] = {}
        # staging off (no reader leaves any): no staging directory, row field or pool entry reaches the writer
        staging_on = self._staging_on()
        views: dict = {}
        if staging_on:
            for row in bmap.get("episodes", []):
                view = _staging.read(root, row["episode_id"])
                if view is not None:
                    views[row["episode_id"]] = view
        self._staged_views, self._staging_read_root = views, root
        # the capture store beside staging (captures.py): written by the harness when each fork settles, read here
        captures_root = self._captures_root()
        if self._pool is not None and (views or captures_root is not None):
            # this pass's staging and the capture store enter the pool now, so the writer can merge or promote them
            # in this same pass
            try:
                self._pool_at_start = self._pool.enter_staging(views, pass_id)
                if captures_root is not None:
                    self._pool_at_start["captures"] = self._pool.enter_captures(
                        captures_root,
                        pass_id,
                    )
                self._pool_at_start["commit"] = self._pool.save(pass_id)
            except Exception as exc:  # noqa: BLE001 - the pool never stops a pass
                self._pool_at_start = {
                    "error": f"{type(exc).__name__}: {str(exc)[:200]}",
                }
        # each row points at what else this pass holds about its episode, where the writer reads: staged
        # candidates and open pool entries citing it
        open_pool = self._pool.open_entries() if self._pool is not None else []
        if staging_on:
            # the brief names /inputs/staging/ whenever staging is on: it exists even when nothing was staged
            (Path(inputs) / "staging").mkdir(parents=True, exist_ok=True)
        # the call that lands a capture, as the writer makes it (in code: through ``tools``)
        land_call = ("tools." if getattr(self, "_in_code_pass", False) else "") + "land_capture('{}', package, name)"
        for row in bmap.get("episodes", []):
            eid = row["episode_id"]
            src = "none"
            landable: str | None = None
            if captures_root is not None:
                # the attempt captured from this episode, if any: what reproducing it gave, its outcome, level and
                # status, and the stored captures that gave the same result on this request
                row["capture"] = _captures.batch_row(captures_root, eid)
                if row["capture"] is not None:
                    capture_ids.append(row["capture"]["id"])
                    if _capture_lands(captures_root, row["capture"]["id"]):
                        landable = row["capture"]["id"]
            elif staging_on:
                # the brief lists the capture key whenever staging is on; with no capture store it is null
                row["capture"] = None
            if staging_on:
                view = views.get(eid)
                if landable is not None and view is not None and _captures.PROCEDURE_FILE in view.files:
                    # one route per episode: a capture that can be landed reaches the writer as the capture alone,
                    # so the fork's copy of its procedure is left out of the export (the host's staging, and the
                    # pool's view of it, keep the file)
                    view = dataclasses.replace(
                        view,
                        files={k: v for k, v in view.files.items() if k != _captures.PROCEDURE_FILE},
                    )
                row["staging"] = _staging.export(view, Path(inputs) / "staging" / eid)
                row["staging"]["candidates"] = _staged_candidates(row["staging"])
                # the episode's capture and the one call that lands it, when it can be landed
                row["staging"]["capture"] = (
                    None if landable is None else {"id": landable, "land": land_call.format(landable)}
                )
                # the reader is counted host-side (an unfinished one as withheld); the writer's row never names it
                view = views.get(eid)
                src = "none" if view is None else (view.source if view.finished else "withheld")
            if landable is not None and self._pool is not None:
                # its pool entry closes when the writer lands it, or instead adopts a draft staged from the episode
                self._pool.episode_captures[eid] = _pool_entry_id("capture", landable)
            row["pool_entries"] = [
                {"id": e.id, "kind": e.kind, "title": e.title}
                for e in open_pool
                if eid in (e.provenance.get("episodes") or [])
            ]
            counts[src] = counts.get(src, 0) + 1
        # the brief names /inputs/captures whenever the fork is on: it exists even when nothing was captured
        (Path(inputs) / "captures").mkdir(parents=True, exist_ok=True)
        if captures_root is not None:
            try:
                _captures.export_for_writer(captures_root, capture_ids, Path(inputs) / "captures")
            except OSError as exc:  # the capture store never stops a pass; the rows still name what it held
                captures_error = type(exc).__name__
        if self._pool is not None:
            # each row names the closed pool entries resembling its episode or a draft staged from it
            from .batch_map import add_related_closed

            try:
                add_related_closed(
                    bmap.get("episodes", []),
                    self._pool,
                    self.load,
                    staged={
                        e: {
                            k: v.decode("utf-8", "replace")
                            for k, v in view.files.items()
                            if k.startswith("candidates/") and k.endswith(".py")
                        }
                        for e, view in views.items()
                    },
                    index_root=self.cfg.similar_index,
                )
            except (
                Exception
            ) as exc:  # noqa: BLE001 - the pool never stops a pass; the rows say why
                for row in bmap.get("episodes", []):
                    row["related_closed"] = {
                        "count": None,
                        "entries": [],
                        "error": f"{type(exc).__name__}: {str(exc)[:200]}",
                    }
        path.write_text(json.dumps(bmap, sort_keys=True, default=str, indent=1) + "\n")
        self._staging_counts = {"episodes": len(bmap.get("episodes", [])), **counts}
        if captures_error is not None:
            self._staging_counts["captures_error"] = captures_error
        self._capture_rows = {
            row["episode_id"]: row["capture"]
            for row in bmap.get("episodes", [])
            if row.get("capture")
        }
        self._chain_rows = list(bmap.get("episodes", []))
        if staging_on:
            # evidence only: each episode's staging as the writer got it, and its summary's pointers checked
            from .summaries import episode_row

            self._staging_episodes = [
                episode_row(row["episode_id"], row.get("staging"), views.get(row["episode_id"]), self.load)
                for row in bmap.get("episodes", [])
            ]

    def _staging_episodes_out(self) -> list[dict] | None:
        """The end event's ``staging_episodes``: the rows made when staging was listed, each with whether the writer
        read the episode's summary (a ``read`` call, a tool event or a box open of its /inputs path). Never raises.
        """
        rows = self._staging_episodes
        if rows is None:
            return None
        try:
            from . import chain as _chain

            read = _chain.read_paths(self.messages, list(getattr(self, "tool_events", None) or []))
        except Exception:  # noqa: BLE001 - a measure never stops a pass
            read = None
        out = []
        for r in rows:
            path = f"/inputs/staging/{r['episode']}/summary.md"
            seen = None if read is None else (path in read if r.get("summary_bytes") is not None else False)
            out.append({**r, "summary_read": seen})
        return out

    def _captures_root(self) -> Path | None:
        """The capture store (:mod:`.captures`, ``<state>/memory-captures``), beside the fork's staging; None when
        the fork is off or there is no staging root."""
        from .captures import STORE_DIR

        if not (self.cfg.staging_root):
            return None
        return Path(self.cfg.staging_root).parent / STORE_DIR

    def _staging_on(self) -> bool:
        """Whether any reader leaves staging for this pass: a staging root (the episode-end fork or its offline
        replay). Off, the writer sees no staging directory, batch-map field or staged pool entry.
        """
        return bool(self.cfg.staging_root)

    def _chain_counts(self) -> dict:
        """This store pass's hop counts (:func:`.chain.counts`); reads come from the read tool calls' arguments and
        from any structured tool or open events the tool layer kept (``self.tool_events``).
        """
        from . import chain as _chain

        pool = self._pool
        return _chain.counts(
            views=self._staged_views,
            root=self._staging_read_root,
            staged_entry=dict(pool.staged) if pool is not None else {},
            pool_ids=set(pool.entries) if pool is not None else set(),
            batch_rows=self._chain_rows,
            read=_chain.read_paths(
                self.messages,
                list(getattr(self, "tool_events", None) or []),
            ),
            provenance=self._provenance,
        )

    def _first_message_library_mode(self, req: PassRequest, pass_id: str) -> str:
        """The first user message of a library-mode pass: the pass id, its episode ids and its wall clock, and
        nothing that grows (no library index, pool index, use table, drafts or curate reasons: those are files the
        brief points to). Byte-stable apart from the ids and the wall-clock number."""
        wall = format(Decimal(str(float(self.cfg.deadline_s))).normalize(), "f")
        eids = [e for e in req.episodes if isinstance(e, str)]
        from .library_prompts import MAX_INLINE_LIST

        # the ids are inlined only while they are a short list; a larger batch is named by its count
        named = f": {', '.join(eids)}" if len(eids) <= MAX_INLINE_LIST else ""
        return (
            f"Pass {pass_id}: a store pass over {len(eids)} recorded episode"
            f"{'' if len(eids) == 1 else 's'}{named}.\n"
            "Their ids and rows are in /inputs/batch_map.json.\n"
            f"Wall clock for this pass: {wall} s.\n"
        )

    @staticmethod
    def _generated_binds(box: Path) -> dict[Path, str]:
        """Library mode: the generated paths of the writer's box (``memory/__init__.py``, ``INDEX.md``,
        ``links.json``, ``.memory/``), each bound read-only over the writable box in every cell. A write to one
        fails with the ordinary OS error. A generated path that became a link refuses the cell (fail closed).
        """
        from .layout import DATA_DIR, GENERATED_ROOT_FILES

        box = Path(box)
        top = Path(os.path.realpath(box))
        out: dict[Path, str] = {}
        for rel in (*GENERATED_ROOT_FILES, DATA_DIR):
            p = box / rel
            if p.is_symlink() or (
                p.exists() and not Path(os.path.realpath(p)).is_relative_to(top)
            ):
                raise RuntimeError(f"a generated path of the box is a link: {rel}")
            if p.exists():
                out[p] = f"/memory/{rel}"
        return out

    def _cell(
        self,
        code: str,
        box: Path,
        inputs: Path,
        cells: Path,
        timeout_s: float,
    ) -> str:
        """Run *code* as ``/cell/code.py`` (a fresh read-only bind; never argv) under a lowered file-size limit."""
        self._box_audit = None
        cdir = Path(tempfile.mkdtemp(prefix="cell-", dir=cells))
        # the opens the code makes under /inputs and /memory, from the audit hook (/inputs/sitecustomize.py) through
        # a pipe the host drains; stamped and recorded by the call's tool event
        open_log = _sandbox.BoxOpenLog()
        try:
            (cdir / "code.py").write_bytes(code.encode("utf-8", errors="replace"))
            r = run_confined(
                [
                    PRLIMIT,
                    f"--fsize={CELL_FSIZE_BYTES}:{CELL_FSIZE_BYTES}",
                    "--",
                    str(PYTHON),
                    "-s",  # no user site-packages
                    "/cell/code.py",
                ],
                ro={inputs: "/inputs", cdir: "/cell"},
                rw={box: "/memory"},
                cwd="/memory",
                timeout_s=timeout_s,
                env={"PYTHONPATH": "/inputs:/memory", **open_log.env},
                pass_fds=open_log.pass_fds,
                on_spawn=open_log.close_write,
                # library mode: the generated files are read-only binds, so a write fails at once
                **(
                    {"late_ro": self._generated_binds(box)}
                    if getattr(self, "_library_generated", None)
                    else {}
                ),
            )
        finally:
            self._box_audit = open_log.finish()
            shutil.rmtree(cdir, ignore_errors=True)
        content = (r.stdout + ("\n" + r.stderr if r.stderr else "")) or "(no output)"
        shown, full = _head_marked(content)
        from unify.common.box_interpreter import keep_traceback_end

        shown = keep_traceback_end(shown, r.stderr)
        out_dir = cells.parent / "outputs"
        out_dir.mkdir(exist_ok=True)
        n = len(list(out_dir.iterdir()))
        (out_dir / f"cell-{n}.txt").write_text(full)
        if len(full.encode("utf-8", errors="replace")) > _OUTPUT_CAP:
            shown += f"\n(full output: /outputs/cell-{n}.txt)"
        content = shown
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
        reasons_out: list[str] | None = None,
    ) -> tuple[str, str | None]:
        """The ``check`` tool: (reply, host-side detail of an error). Runs on the loop's thread. *reasons_out*, when
        given, receives every reason the gate gave (the reply shows the first :data:`_CHECK_REASONS`).

        :meth:`.gate.Gate.preview` of a manifest against a copy of *box*: the copy is what the commit would
        hold (:func:`_mirror` without ``.pass``), so the box is only read. Called between cells, when no box
        runs, after :func:`_measure` made every entry readable. It stays on the event loop's thread, as
        :meth:`.gate.Gate.merge` does, because the evidence store's connection belongs to that thread. An
        error is answered with its type only (its text can name host paths) and returned as the detail.
        """
        manifest, problem = _check_manifest(raw)
        if problem is not None:
            return problem, None
        found = _layout_problems(box)
        if found:
            return _layout_refusal(found), None
        manifest = _with_fixtures(manifest, self._ledger(box))
        try:
            with tempfile.TemporaryDirectory(prefix="memory-check-") as tmp:
                tree = Path(tmp) / "tree"
                tree.mkdir()
                _mirror(box, tree, skip_top=frozenset({".pass"}))
                _drop_unchanged(tree, self._library_generated)
                reasons = self.gate.preview(base, tree, manifest)
        except Exception as exc:  # a broken check never ends the pass
            detail = _redact(f"check error: {type(exc).__name__}: {exc}")[:300]
            return f"check error: {type(exc).__name__}", detail
        if reasons_out is not None:
            reasons_out.extend(_redact(r) for r in reasons)
        return _check_reply(reasons), None

    def _finish_manifest_problem(self, box: Path) -> tuple[str | None, bool]:
        """(The G1 refusal ``finish`` would meet, from the gate's own manifest parser; whether the parsed manifest
        lists no item and no deletion).

        No refusal when the manifest parses. A manifest the host cannot read yet (the final check restores access
        first) is left to the final check rather than refused here.
        """
        self._pool_manifest = (
            None  # only an accepted finish's manifest changes the pool
        )
        layout = _layout_problems(box)
        if layout:
            return _layout_refusal(layout), False
        found, manifest, problem = _read_manifest(box)
        if not found:
            return (
                "G1: no manifest: write /memory/.pass/manifest.json, then finish",
                False,
            )
        if problem is not None:
            return (None if "cannot be read" in problem else f"G1: {problem}"), False
        try:
            man = self.gate._parse_manifest(
                _with_fixtures(manifest, self._ledger(box)),
            )
        except _manifest.ManifestError as exc:
            return _redact(f"G1: malformed manifest: {exc}")[:600], False
        if self._pool is not None:
            from .pool import block_problems

            library_items = _library_item_ids(box)
            probs = block_problems(
                _wt.without_defer(manifest),
                self._pool,
                "write",
                library_items=library_items,
            )
            block = manifest.get("pool") if isinstance(manifest, dict) else None
            raw_defer = block.get("defer") if isinstance(block, dict) else None
            rows = _wt.defer_rows(manifest)
            if raw_defer is not None and (
                not isinstance(raw_defer, list) or len(rows) != len(raw_defer)
            ):
                probs.append(
                    'pool.defer must be a list of {"id", "episodes", "reason"} rows (pool decide writes them)',
                )
            open_ids = {e.id for e in self._pool.open_entries()}
            for row in rows:
                if row["id"] != "new" and row["id"] not in open_ids:
                    probs.append(f"pool.defer: {row['id'][:20]} is not an open entry")
                elif row["id"] == "new" and not row["episodes"]:
                    probs.append("pool.defer: a new deferral needs its episodes")
            if probs:
                return _redact(f"G1: {probs[0]}")[:600], False
            self._pool_manifest = manifest
            self._pool_library_items = library_items
        return None, not man.items and not man.deleted

    def _event(
        self,
        pass_id: str,
        tool: object,
        raw_args: object,
        reply: "_wt.Reply | None",
        usage: dict | None,
        extra: dict | None = None,
        call: int | None = None,
    ) -> None:
        """One ``tool_call`` row of tool-events.jsonl, from the tool layer's own reply (never from transcript text),
        appended when the call ends; *extra* adds fields (delegate and the record's tools: ``messaging``). *call*: a
        number reserved when the cell started (tools in code: the cell's own calls are numbered after it).
        """
        reply = reply or _wt.Reply("")
        args_text = (
            raw_args if isinstance(raw_args, str) else json.dumps(raw_args, default=str)
        )
        text_bytes = len(reply.text.encode("utf-8", errors="replace"))
        row = {
            "event": "tool_call",
            "agent": "writer",
            # the agent's name on the pass's record
            "name": "writer",
            "pass_id": pass_id,
            "call": len(self.tool_events) if call is None else call,
            "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "tool": tool if isinstance(tool, str) else None,
            "status": reply.status,
            "refusal": (
                {"form": reply.form, "text_bytes": text_bytes}
                if reply.status == "refused"
                else None
            ),
            "paths": [dict(p) for p in reply.paths],
            "args_bytes": len(str(args_text or "").encode("utf-8", errors="replace")),
            "result_bytes": text_bytes,
            "result_truncated": "; next: offset=" in reply.text,
            "usage": usage,
            **(extra or {}),
        }
        opens: list[dict] = []
        if tool == "execute_code":
            # the host stamps who, which pass and which call; the box supplies only event, path and mode
            audit = self.__dict__.pop("_box_audit", None)
            row.update(_sandbox.box_open_fields(audit))
            opens = [
                {"agent": row["agent"], "pass_id": pass_id, "call": row["call"], **o}
                for o in (audit or {}).get("rows") or []
            ]
        if call is None:
            self.tool_events.append(row)
        else:
            self.tool_events[call] = row
        self._write_event(pass_id, row)
        if opens:
            self._record_box_opens(pass_id, opens)

    def _crashed_cell_event(
        self,
        pass_id: str,
        raw_args: object,
        args: object,
        usage: dict | None,
        call: int,
        exc: BaseException,
    ) -> None:
        """The ``execute_code`` row of a cell that raised out of :meth:`_in_code_cell`, at the *call* number
        reserved for it: status ``error`` and the exception's type only. Never raises: the cell's own exception
        is the one the pass sees."""
        try:
            self.__dict__.pop("_box_audit", None)  # a crashed cell returned no audit of its own
            self._event(
                pass_id,
                "execute_code",
                raw_args,
                _wt.Reply(f"error: the cell did not end normally ({type(exc).__name__})", "error"),
                usage,
                {
                    "via": "model",
                    "cell": None,
                    "args_sha256": _args_sha256(args if args is not None else raw_args),
                    "box_gone": False,
                    "exception": type(exc).__name__,
                },
                call,
            )
        except Exception:  # noqa: BLE001 - the row is evidence; it never masks the cell's exception
            logging.getLogger(__name__).warning(
                "the row of a cell that raised was not written (%s)",
                type(exc).__name__,
            )
            if 0 <= call < len(self.tool_events) and self.tool_events[call].get("event") == "reserved":
                self.tool_events[call] = {
                    "event": "tool_call",
                    "call": call,
                    "tool": "execute_code",
                    "status": "error",
                    "exception": type(exc).__name__,
                }

    def _reserve_event(self) -> int:
        """Reserve the next ``call`` number for a row written later (a cell's, after its in-code calls)."""
        self.tool_events.append({"event": "reserved"})
        return len(self.tool_events) - 1

    def _channel_event(self, pass_id: str, row: dict) -> None:
        """The tool channel's rows, on the event loop in the order calls end: a ``tool_call`` row gets the pass's
        next ``call`` number and joins :attr:`tool_events`; a ``channel_closed`` row is written as it is."""
        if row.get("event") == "tool_call":
            row["call"] = len(self.tool_events)
            self.tool_events.append(row)
        self._write_event(pass_id, row)

    def _record_box_opens(self, pass_id: str, rows: list[dict]) -> None:
        """Keep host-stamped open rows on the pass (``self.box_opens``) and append them to
        ``<pass_record_root>/<pass id>/box-opens.jsonl`` (:func:`.sandbox_run.append_open_rows`).
        """
        kept = self.__dict__.setdefault("box_opens", [])
        kept.extend(rows)
        root = getattr(self.cfg, "pass_record_root", None)
        if root:
            _sandbox.append_open_rows(Path(root) / pass_id / "box-opens.jsonl", rows)

    def _write_refusal_codes(self, pass_id: str) -> None:
        """Write ``<pass_record_root>/<pass id>/refusal-codes.json``: every refusal form code by tool
        (:func:`.tool_spec.refusal_codes`), the codes this pass's ``refusal.form`` fields are drawn from. A regular
        file created new (never through a link, never over an existing file); a failure skips it, never the pass.
        """
        root = getattr(self.cfg, "pass_record_root", None)
        if not root:
            return
        from .tool_spec import refusal_codes

        try:
            folder = Path(root) / pass_id
            for d in (Path(root), folder):
                d.mkdir(parents=True, exist_ok=True)
                if d.is_symlink() or not d.is_dir():
                    return
            fd = os.open(
                folder / "refusal-codes.json",
                os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC,
                0o644,
            )
            with os.fdopen(fd, "w") as fh:
                json.dump(refusal_codes(), fh, sort_keys=True, indent=1)
                fh.write("\n")
        except OSError:
            pass  # a measurement never stops a pass

    def _write_event(self, pass_id: str, row: dict) -> None:
        """Append *row* to ``<pass_record_root>/<pass id>/tool-events.jsonl``. The file is created new (never through
        a link, never over an existing file) at the pass's first event; a failure stops the file, never the pass.
        """
        root = getattr(self.cfg, "pass_record_root", None)
        if not root or self._events_file is False:
            return
        try:
            if self._events_file is None:
                self._events_file = False
                folder = Path(root) / pass_id
                for d in (Path(root), folder):
                    d.mkdir(parents=True, exist_ok=True)
                    if d.is_symlink() or not d.is_dir():
                        return
                path = folder / "tool-events.jsonl"
                flags = (
                    os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC
                )
            else:
                path = self._events_file
                flags = os.O_WRONLY | os.O_APPEND | os.O_NOFOLLOW | os.O_CLOEXEC
            fd = os.open(path, flags, 0o644)
            with os.fdopen(fd, "a") as fh:
                if not stat.S_ISREG(os.fstat(fh.fileno()).st_mode):
                    self._events_file = False
                    return
                fh.write(json.dumps(row, sort_keys=True) + "\n")
            self._events_file = path
        except OSError:
            self._events_file = False  # a measurement never stops a pass

    def _recorded_ids(self) -> set[str]:
        """The recorded-episode index: every episode id the evidence store has recorded (an episode is listed once it
        is recorded in full). Ids resolve only through this index and the batch, never through a path.
        """
        if self._recorded is None:
            since = getattr(self.ev, "experience_since", None)
            try:
                self._recorded = (
                    {e for e, _ in since(0)} if since is not None else set()
                )
            except Exception:  # noqa: BLE001 - an unreadable index: the batch alone
                self._recorded = set()
        return self._recorded

    def _record_dir(self, pass_id: str, inputs: Path) -> Path:
        """Where the pass keeps its records: ``<pass_record_root>/<pass id>`` (beside the inputs when it has none)."""
        root = getattr(self.cfg, "pass_record_root", None)
        return Path(root) / pass_id if root else Path(inputs).parent / "pass-record"

    def _attention_start(self, req: PassRequest, pass_id: str, inputs: Path) -> Any:
        """An attention pass's state (:class:`.attention_tools.AttentionPass`): the home's ledger, the entries handed
        to the pass, its record dir, the box's inputs and the home's records the material is read from."""
        from . import attention as _at
        from . import attention_tools as _at_tools

        home = Path(self.cfg.attention_home)
        ledger = _at.Ledger(home)
        return _at_tools.AttentionPass(
            ledger=ledger,
            pass_id=pass_id,
            record_dir=self._record_dir(pass_id, inputs),
            inputs=Path(inputs),
            home=home,
            handed=list(getattr(req, "attention", None) or []),
            sources=_at.HomeSources(home, state_dir=ledger.state_dir),
        )

    def _tests_start(self, pass_id: str, box: Path, tmp: Path, inputs: Path) -> Any:
        """The stated tests' context (:class:`.writer_tests.TestContext`): ``tests.jsonl`` beside the pass's records,
        the writer's /memory and the episodes it may read."""
        from . import writer_tests as _tests

        return _tests.TestContext(
            pass_id=pass_id,
            tests_path=self._record_dir(pass_id, inputs) / _tests.TESTS_FILE,
            library=Path(box),
            episode=self._episode,
            work=Path(tmp) / "tests",
        )

    def _episode(self, eid: object) -> Episode | None:
        """A recorded episode by id (the batch's, or any in the recorded-episode index), loaded once per pass; None
        for an id the index does not hold. Read tools show only the parts the batch export shows (request,
        observations, cells, actions, the work-tree diff): no checker or outcome field is a part.
        """
        if not isinstance(eid, str) or not _EPISODE_ID.match(eid):
            return None
        if eid in self._episodes:
            return self._episodes[eid]
        if eid not in self._recorded_ids():
            return None
        try:
            ep = self.load(eid)
        except (
            Exception
        ):  # noqa: BLE001 - an episode that cannot be loaded is not readable
            return None
        self._episodes[eid] = ep
        return ep

    def _ledger(self, box: Path) -> dict[str, dict]:
        """Fixture provenance derived from the files now in *box* (:func:`.fixtures.derive_ledger`)."""
        from .fixtures import derive_ledger

        self.fixtures_made = derive_ledger(box)
        return self.fixtures_made

    def _reader_call(
        self,
        name: str,
        args: dict,
        roots: dict[str, Path],
        agent: str = "writer",
    ) -> _wt.Reply:
        """The reader tools (read, grep, overview, read_episode, pool) for *agent*. Each agent has its own read log
        and its own record of parts it was shown in full. A bad argument is answered, never raised; the answer names
        the domain problem, never an exception type. An in-code call runs it from the worker pool."""
        readers = ("read", "grep", "overview", "read_episode", "pool")
        if name not in readers:
            return _wt.general(
                "unknown_tool",
                name=str(name)[:80],
                tools=", ".join(readers),
            )
        log = self._read_logs.setdefault(agent, _wt.ReadLog())
        shown = self._shown_full.setdefault(agent, {})
        batch = list(self._batch)
        offset = args.get("offset", 0)
        offset = (
            offset
            if isinstance(offset, int) and not isinstance(offset, bool) and offset >= 0
            else 0
        )
        if name in ("read", "grep"):
            vpath = _natural_path(str(args.get("path") or ("/inputs" if name == "grep" else "")), roots)
            try:
                prefix, parts = _safe.split(vpath, list(roots))
            except _safe.Outside:
                return _wt.refused(name, "outside_roots", path=vpath[:200])
            virtual = _safe.virtual(prefix, parts)
            if name == "read":
                where = ({"path": virtual, "mode": "read"},)
                try:
                    text, mode = _views.read_entry(vpath, roots, offset)
                except _safe.Outside:
                    return _wt.refused("read", "outside_roots", path=vpath[:200])
                except (_safe.LinkFound, _safe.NotRegular) as exc:
                    return _wt.refused(
                        "read",
                        "unreadable",
                        where,
                        error=exc.reason,
                        path=vpath[:200],
                    )
                except OSError:
                    return _wt.refused("read", "not_found", where, path=vpath[:200])
                return _wt.Reply(text, paths=({"path": virtual, "mode": mode},))
            where = ({"path": virtual, "mode": "search"},)
            try:
                text = _views.grep_bounded(
                    str(args.get("pattern", "")),
                    vpath,
                    roots,
                    offset,
                )
            except _views.PatternProblem as exc:
                return _wt.refused("grep", "bad_pattern", where, reason=str(exc))
            except _views.SearchFailed:
                return _wt.refused("grep", "search_failed", where)
            except _safe.Outside:
                return _wt.refused("grep", "outside_roots", where, path=vpath[:200])
            except _safe.LinkFound as exc:
                return _wt.refused(
                    "grep",
                    "unreadable",
                    where,
                    error=exc.reason,
                    path=vpath[:200],
                )
            except OSError as exc:
                return _wt.refused(
                    "grep",
                    "unreadable",
                    where,
                    error=_safe.errno_name(exc),
                    path=vpath[:200],
                )
            return _wt.Reply(text, paths=where)
        if name == "pool":
            return _wt.pool(args, roots["/inputs"] / "pool", roots["/memory"])
        eid = args.get("episode")
        if name == "overview" and eid in (None, ""):
            return _wt.overview_batch(self._bmap_rows, offset)
        ep = self._episode(eid)
        if ep is None:
            return _wt.refused(
                name,
                "unknown_episode",
                episode=str(eid)[:80],
                batch=", ".join(batch) or "-",
            )
        if name == "overview":
            return _wt.overview_episode(
                ep,
                str(eid),
                offset,
                log=log,
                shared=self._shared,
            )
        return _wt.read_episode(
            ep,
            str(eid),
            args,
            blob=self.gate._blob,
            log=log,
            shown=shown,
            shared=self._shared,
        )

    def _in_code_start(
        self,
        *,
        pass_id: str,
        box: Path,
        inputs: Path,
        tmp: Path,
        roots: dict[str, Path],
        parent: str,
        max_calls: int,
        remaining: Any,
        cause: Any,
        check_file: Any,
        phase: dict,
        notes: list,
        deadline: float,
        spend: Any,
        allow: tuple[str, ...],
        scope: contextlib.AsyncExitStack,
    ) -> tuple[Any, Any]:
        """Tools in code: the handlers' pass state (:class:`.tool_handlers.WriterCalls`) and the agent's interpreter
        session with its channel. The evidence store is readable from the worker pool for the pass; on the way out
        the session ends first (the box is killed, the channel closed)."""
        from unify.common.tool_channel import HandlerThreads

        from . import tool_handlers as _th

        agent = "writer"
        ic = _th.WriterCalls(
            writer=self,
            agent=agent,
            pass_id=pass_id,
            box=box,
            inputs=inputs,
            tmp=tmp,
            roots=roots,
            parent=parent,
            max_calls=max_calls,
            remaining=remaining,
            cause=cause,
            check_file=check_file,
            phase=phase,
            notes=notes,
            deadline_s=self.cfg.deadline_s,
            deadline=deadline,
            spend=spend,
            threads=HandlerThreads(),
        )

        binds, profile = self._interpreter_box(agent, pass_id, box, inputs)
        pr = self._pass_record
        session = _th.InterpreterSession(
            agent=agent,
            name=agent,
            role="write",
            pass_id=pass_id,
            state=ic,
            allow=allow,
            binds=binds,
            env={},
            roots=roots,
            deadline=deadline,
            cell_limit_s=float(self.cfg.cell_timeout_s),
            on_event=lambda row: self._channel_event(pass_id, row),
            threads=ic.threads,
            profile=profile,
            start=self.interpreter_start,
            # the run record: the comms tools' record is the pass agent's view of it
            record=pr.view(pr.root) if pr is not None else None,
        )
        ic.kill_box = session.kill
        ic.measure_quota = session.measure_quota
        self._in_code_threads = ic.threads  # helpers' handler threads join the pass's tracker (the gate waits for them)
        scope.callback(
            _th.thread_reads(
                self.ev,
                getattr(self.gate, "ev", None),
                _lookup_store(self.gate),
            ),
        )
        scope.push_async_callback(session.close)
        return ic, session

    def _interpreter_box(self, agent: str, pass_id: str, box: Path, inputs: Path) -> tuple[dict, Any]:
        """The binds and profile of the agent's interpreter box, with tools in code on or off: the binds _cell gives
        (the generated files read-only over the writable box), cwd /memory, the import path, the file-size limit,
        the /memory quota measure and the names bound in every cell (``memory`` where the box holds the package)."""
        from . import tool_handlers as _th

        binds = {
            "ro": {inputs: "/inputs"},
            "rw": {box: "/memory"},
            "late_ro": (
                self._generated_binds(box)
                if getattr(self, "_library_generated", None)
                else {}
            ),
        }
        profile = _th.memory_profile(
            pass_id=pass_id,
            cwd="/memory",
            sys_path=("/inputs", "/memory"),
            fsize_bytes=CELL_FSIZE_BYTES,
            # the pass stamps and records the open rows with each cell's event (as for a fresh cell)
            opens_path=None,
            measure=lambda: _measure(box),
            globals=_th.memory_globals(agent, box),
        )
        return binds, profile

    async def _in_code_cell(
        self,
        session: Any,
        ic: Any,
        code: str,
        cells: Path,
        cell_no: int,
    ) -> str:
        """One cell in the agent's interpreter, its in-code calls served by the channel (*ic* None: tools in code
        off, no channel); the output as _cell gives it in library mode (the first 8000 bytes, the full output in
        /outputs/cell-<n>.txt), then the host's notes."""
        from . import tool_handlers as _th

        note = session.take_note()
        res = await session.cell(code, cell_no)
        audit = getattr(res, "opens", None)
        self._box_audit = audit if isinstance(audit, dict) else None
        finish_text = ic.finish_text if ic is not None else None
        body, tail = _th.cell_parts(res, finish_text, session.gone_during(), note)
        shown, full = _head_marked(body)
        from unify.common.box_interpreter import keep_traceback_end

        shown = keep_traceback_end(shown, getattr(res, "error", None))
        out_dir = cells.parent / "outputs"
        out_dir.mkdir(exist_ok=True)
        n = len(list(out_dir.iterdir()))
        (out_dir / f"cell-{n}.txt").write_text(full)
        if len(full.encode("utf-8", errors="replace")) > _OUTPUT_CAP:
            shown += f"\n(full output: /outputs/cell-{n}.txt)"
        return shown + tail

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
        *,
        patch_blob: str | None = None,
        refused: dict | None = None,
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
                "patch_blob": patch_blob,
                "items_merged": "[]",
                "items_refused": json.dumps(refused or {}, sort_keys=True),
            },
        )
        return reasons

    def _candidate_key(self, candidate: str | None, manifest: object) -> tuple | None:
        """What a repair round's check judged: the candidate commit's tree and the manifest, so a round that
        changed neither compares equal to the last one (a commit's own id differs every round). None when
        unknown."""
        if candidate is None:
            return None
        try:
            tree = self.mem.run("rev-parse", f"{candidate}^{{tree}}").strip()
            judged = json.dumps(manifest, sort_keys=True, default=str)
        except Exception:  # noqa: BLE001 - unknown is never a repeat
            return None
        return (tree, judged)

    def _commit_round(
        self,
        box: Path,
        parent: str,
        pass_id: str,
        summary: str,
        manifest: object,
    ) -> str:
        """A library-mode round's candidate: the box as the commit would hold it (no ``.pass``), committed on a
        fresh checkout of *parent* with the message and trailers v2's final commit has, and no round number
        (Amendment A: the merge lands the last checked commit itself; the round lives in ``pass_rounds``). It is
        one commit on the parent that no ref names, so a refused round never enters ``main``'s history.
        Called after :func:`_measure` restored the host's access to the box."""
        sources = _sources(manifest)
        with self.mem.temp_checkout(parent) as rwt:
            _clear_checkout(rwt)
            _mirror(box, rwt, skip_top=frozenset({".pass"}))
            if getattr(self, "_library_generated", None):
                _drop_unchanged(
                    rwt,
                    self._library_generated,
                )  # the generated files are never committed
            return self.mem.commit_all(
                rwt,
                f"consolidation pass {pass_id}: {_redact(summary)[:200]}".replace(
                    "\0",
                    "",
                ),
                _trailers(pass_id, sources, manifest),
            )

    def _close_rounds(
        self,
        pass_id: str,
        round_no: int,
        round_blobs: list[str],
        res: GateResult,
        candidate: str | None,
        pass_notes: Iterable[str],
    ) -> None:
        """Library mode: the pass's final full gate result into the blob store and its ``pass_rounds`` row;
        nothing with library_mode off."""
        text = _repair.render_result(
            pass_id,
            round_no,
            res,
            final=True,
            candidate=candidate,
            pass_notes=pass_notes,
        )
        store = getattr(self.gate, "blobs", None)
        self.ev.record_pass_rounds(
            {
                "pass_id": pass_id,
                "role": "write",
                "rounds": round_no + 1,
                "round_blobs": json.dumps(round_blobs),
                "gate_blob": (
                    store.put(text.encode()) if isinstance(store, BlobStore) else None
                ),
            },
        )

    def _brief_library_mode(self) -> str:
        """The writer's system message: the WRITE brief for this pass's inputs."""
        from .library_prompts import write_brief_now

        return write_brief_now(**self._write_inputs())

    def _write_inputs(self) -> dict:
        """The WRITE brief's inputs, constant per run: a helper lane and a similar-code index."""
        return {
            "cell_s": float(self.cfg.cell_timeout_s),
            "helper": self.helper_turn is not None,
            "similar": bool(self.cfg.similar_index),
        }

    def _keep_cancelled_work(
        self,
        box: Path,
        wt: Path,
        parent: str,
        pass_id: str,
    ) -> tuple[str | None, list[str]]:
        """The writer's box as a patch on *parent* (a commit no ref names, as refused work is saved) and the item ids its
        manifest named. Never raises: a cancellation must not be masked."""
        named = _named_items(box)
        store = getattr(self.gate, "blobs", None)
        try:
            _remove(box / ".pass")
            _clear_checkout(wt)
            _mirror(box, wt, skip_top=frozenset({".pass"}))
            if getattr(self, "_library_generated", None):
                _drop_unchanged(
                    wt,
                    self._library_generated,
                )  # the generated files the writer left unchanged
            commit = self.mem.commit_all(
                wt,
                f"cancelled pass {pass_id}",
                {"Pass": pass_id},
            )
            if commit == parent or not isinstance(store, BlobStore):
                return None, named
            return store.put(self.mem.diff(parent, commit).encode()), named
        except Exception:  # noqa: BLE001
            return None, named

    def _cancelled_result(self, pass_id: str, named: list[str]) -> str | None:
        """The blob id of :data:`CANCELLED_RESULT` for a cancelled library-mode pass (its draft's gate result)."""
        store = getattr(self.gate, "blobs", None)
        if not isinstance(store, BlobStore):
            return None
        text = CANCELLED_RESULT.format(
            pass_id=pass_id,
            items=", ".join(named) or "none named",
        )
        return store.put(text.encode())

    async def run(self, req: PassRequest, pass_id: str) -> PassOutcome:
        self.messages = []
        self._tools_json, self._archived = [], []
        self._compaction = {"compactions": 0, "chars_moved": 0, "messages_moved": 0}
        self._cache_tokens = [0, 0, 0, 0, 0]
        self._output_tokens = [0, 0]
        self._first_prompt = (False, None)
        self._staging_episodes = None
        self._max_prompt = None
        self.fixtures_made = {}
        self._shown_full, self._read_logs = {}, {}
        self._recorded, self._episodes = None, {}
        self._batch = list(req.episodes)
        self.tool_events = []
        self._events_file = None
        self.final_manifest = None
        self._shared = {}
        self._helper_rows = []
        self._pass_record = None
        self._staging_counts = None
        self._capture_rows = {}
        self._inputs_dir = None
        self._pool, self._pool_open, self._pool_error = None, None, None
        self._pool_at_start = None
        self._staged_views, self._staging_read_root, self._provenance = {}, None, []
        self._levels = (None, None)
        self._chain_rows = []
        self._pool_manifest = None
        self._bmap_rows = []
        self._live, self._cancel_patch = None, (None, [])
        # an attention pass's state (attention_tools.AttentionPass) and the stated tests' context (writer_tests)
        self._attention, self._tests = None, None
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
        self._write_refusal_codes(pass_id)
        try:
            out = await self._run(req, pass_id, parent, spend)
            out.pool = (
                self._pool_update(req, pass_id, out)
                if self._pool is not None
                else {"error": self._pool_error or "not loaded"}
            )
            return out
        except BaseException as exc:  # cancellation and interrupts included
            if not self.ev.pass_exists(pass_id):  # the gate records its own errors
                reason = _redact(f"pass error: {type(exc).__name__}: {exc}")[:500]
                cancelled = isinstance(
                    exc,
                    asyncio.CancelledError,
                )
                patch, named = self._cancel_patch if cancelled else (None, [])
                self._fail(
                    req,
                    pass_id,
                    parent,
                    spend,
                    [reason],
                    patch_blob=patch,
                    refused={i: ["cancelled"] for i in named} if cancelled else None,
                )
                try:
                    self.ev.record_pass_rounds(
                        {
                            "pass_id": pass_id,
                            "role": "write",
                            "rounds": None,
                            "round_blobs": "[]",
                            # a cancelled pass's draft points at its fixed gate result
                            "gate_blob": (
                                self._cancelled_result(pass_id, named)
                                if cancelled
                                else None
                            ),
                        },
                    )
                except (
                    Exception
                ):  # noqa: BLE001 - never masks the pass's own exception
                    pass
            raise

    async def _run(
        self,
        req: PassRequest,
        pass_id: str,
        parent: str,
        spend: _Spend,
    ) -> PassOutcome:
        # no USD stop of its own and no count limits: the run's room (usd_limit), the step guard and the deadline end a
        # pass; unpriced calls are booked at their worst case by the run-level guard
        max_calls = int(self.cfg.max_calls)
        deadline = time.monotonic() + float(self.cfg.deadline_s)
        # the pass on the run record, for its agent and every helper it starts (messaging.py)
        self._pass_record = _messaging.PassRecord(
            root="writer",
            deadline=deadline,
            run_home=self.cfg.run_home,
            pass_id=pass_id,
            held_by_service=True,
        )
        calls, checks, summary = 0, 0, ""
        # repair rounds (repair.py states when one may open)
        round_no = 0  # the round the writer is in: 0 until its first accepted finish
        # library mode: the last accepted finish
        phase: dict[str, Any] = {
            "finish_accepted": False,
            "nothing_to_store": False,
        }
        round_s: list[float] = (
            []
        )  # each completed round's seconds, its gate check included
        round_paths: list[str] = []  # /inputs/gate/result-<n>.md written for the writer
        round_blobs: list[str] = []  # the same texts in the blob store
        # the refusals repeated round to round (repair.py states the stuck rule)
        stuck = _repair.StuckRefusals()
        mark = [time.monotonic()]  # the current round's start
        # the last checked candidate, which the merge lands itself when the box is unchanged since
        check_files = [
            0,
        ]  # Amendment B: /inputs/gate/check-<k>.md written for the writer's check calls
        checked: dict[str, Any] = {
            "candidate": None,
            "manifest": None,
            "current": False,
            "seconds": 0.0,
        }
        base: ParentSnapshot | str | None = (
            None  # the parent, taken once for every check
        )
        notes: list[str] = []
        stop: str | None = None  # why no further cell runs: deadline or quota
        over_quota: str | None = None
        causes: list[str] = []  # PassOutcome.codes, set where each cause arises
        errored = False
        reads = 0  # library_mode reader calls, counted
        self._exported_bytes = None
        self._temp_kept = None
        # tools in code: one JSON tool, the rest called from cells
        self._in_code_pass = True
        # tools in code: the handler threads still running when the gate would start (None: that point was not
        # reached); written on the pass end event
        self._handlers_at_gate = None
        # library mode: the shared no-progress rule (loop_stop.AgentNoProgress), always on; a capped pass without the
        # library is bounded by its call and USD caps as before
        no_progress = None
        from unify.common._async_tool import loop_stop as _loop_stop

        no_progress = _loop_stop.AgentNoProgress(
            _loop_stop.terminal_tool("writer"),
            in_code=True,
        )

        def cause(code: str) -> None:
            if code not in causes:
                causes.append(code)

        def remaining() -> float:
            return deadline - time.monotonic()

        def outcome(
            passed: bool,
            commit: str | None,
            reasons: list[str],
            merged: list[str] | None = None,
            refused: dict[str, list[str]] | None = None,
        ) -> PassOutcome:
            o = PassOutcome(
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
                list(merged or []),
                dict(refused or {}),
                (
                    (self._read_logs.get("writer") or _wt.ReadLog()).summary(
                        self._batch,
                    )
                ),
                reads,
                self._exported_bytes,
            )
            o.rounds, o.round_results = round_no + 1, list(round_paths)
            o.repair_stuck = stuck.report()
            o.no_progress = no_progress.report() if no_progress is not None else None
            o.compactions = self._compaction["compactions"]
            o.chars_moved = self._compaction["chars_moved"]
            o.messages_moved = self._compaction["messages_moved"]
            o.max_prompt_tokens = self._max_prompt
            window = getattr(self.turn, "max_input_tokens", None)
            o.context_window = (
                window
                if isinstance(window, int) and not isinstance(window, bool)
                else None
            )
            if self._cache_tokens[2]:
                o.input_tokens = self._cache_tokens[0]
            if self._cache_tokens[3]:
                o.cache_read_tokens = self._cache_tokens[1]
            o.input_tokens_calls, o.cache_read_calls = self._cache_tokens[2], self._cache_tokens[3]
            if self._output_tokens[1]:
                o.output_tokens = self._output_tokens[0]
            o.output_tokens_calls = self._output_tokens[1]
            o.writer_calls = self._cache_tokens[4]
            o.writer_first_prompt_tokens = self._first_prompt[1]
            o.finished = bool(phase["finish_accepted"])
            o.provenance = list(self._provenance)
            o.levels, o.capped_reasons = self._levels
            try:  # measures only: they never fail a pass
                o.chain = self._chain_counts()
            except Exception as exc:  # noqa: BLE001
                o.chain = {"error": f"{type(exc).__name__}: {str(exc)[:200]}"}
            o.staging = self._staging_counts
            o.staging_episodes = self._staging_episodes_out()
            o.helpers = [
                {
                    k: r.get(k)
                    for k in (
                        "call",
                        "status",
                        "episodes",
                        "turns",
                        "usd",
                        "usd_known",
                        "unknown_cost_calls",
                        "seconds",
                        "no_progress",
                    )
                }
                | {"files": len(r.get("files") or [])}
                for r in sorted(self._helper_rows, key=_started_order)
            ]
            o.nothing_to_store = bool(
                phase["finish_accepted"] and phase["nothing_to_store"],
            )
            return o

        async with contextlib.AsyncExitStack() as scope:
            wt = scope.enter_context(self.mem.temp_checkout())
            pass_temp = _PassTemp(prefix="memory-writer-")
            tmp = scope.enter_context(pass_temp)
            scope.enter_context(_KeepOnCancel(self, parent, pass_id))
            # on every way out (finish, deadline, error, cancellation) no helper outlives the pass: they end before
            # the pass's directories go
            scope.push_async_callback(self._end_record)
            box, inputs, cells = (Path(tmp) / n for n in ("memory", "inputs", "cells"))
            box.mkdir()
            cells.mkdir()
            self._live = (box, wt)
            _mirror(wt, box)
            self._stage_inputs(req, inputs, wt)
            if req.kind == "attention":
                # the entries handed to this pass, its ledger and its record
                self._attention = self._attention_start(req, pass_id, Path(inputs))
                from . import attention_tools as _at_tools

                # on every way out: the pass's read evidence, which the service verifies the entries' reads against
                scope.callback(_at_tools.write_read_log, self)
            # the tests the writer states, recorded beside the pass's records
            self._tests = self._tests_start(pass_id, box, Path(tmp), Path(inputs))
            bmap_path = inputs / "batch_map.json"
            bmap = (
                json.loads(bmap_path.read_text())
                if bmap_path.is_file()
                else {"episodes": []}
            )  # a curate pass has no batch map
            self._episodes = {e: self.load(e) for e in req.episodes}
            self._bmap_rows = list(bmap["episodes"])
            outputs = Path(tmp) / "outputs"
            outputs.mkdir()
            roots = {"/inputs": inputs, "/memory": box, "/outputs": outputs}
            from . import tool_handlers as _th

            from unify.common import tool_registry as _registry

            from .library_prompts import tool_flags

            # the registry's allow-list for the agent, with the pass's configuration flags
            in_code_allow = _registry.allow_list(
                "writer",
                tool_flags(helper=self.helper_turn is not None, similar=bool(self.cfg.similar_index)),
            )
            tools = _th.in_code_json_tools(
                "writer",
            )
            tool_names = [t["function"]["name"] for t in tools]
            # the writer-chosen tree with its generated helper and index; no channel
            # folders and no library.json (the gate inputs replace previous_gate.json)
            self._library_generated = self._stage_library_library_mode(box, parent)
            # every open proposal of earlier passes; this pass's staging joins it below, before it is staged
            from . import pool as _pool_mod

            try:
                self._pool = _pool_mod.Pool(self.mem).load()
            except (
                Exception
            ) as exc:  # noqa: BLE001 - the pool never stops a pass
                self._pool = None
                self._pool_error = (
                    f"load: {type(exc).__name__}: {str(exc)[:200]}"
                )
            # every batch episode's staging, read-only for the writer
            await self._staging_library_mode(
                req,
                inputs,
                box,
                roots,
                spend,
                pass_id=pass_id,
            )
            # the staging rows were added to the batch map: overview reads them
            staged_map = inputs / "batch_map.json"
            if staged_map.is_file():
                self._bmap_rows = list(
                    json.loads(staged_map.read_text()).get("episodes", []),
                )
            if (
                self._pool is not None
            ):  # a pass with no pool skips this
                try:
                    self._pool_open = self._pool.stage(Path(inputs) / "pool")
                except (
                    Exception
                ) as exc:  # noqa: BLE001 - the pool never stops a pass
                    self._pool = None
                    self._pool_error = (
                        f"stage: {type(exc).__name__}: {str(exc)[:200]}"
                    )
            first = self._first_message_library_mode(req, pass_id)
            if self._attention is not None:
                from . import attention_tools as _at_tools

                # what each handed entry owes a read, under /inputs (its staging, captures and attention files)
                _at_tools.export_entries(self, self._attention, list(req.attention))
                # a capture group's related closed entries and the library items it touches
                _at_tools.export_group(self._attention, list(req.attention))
                wall = format(Decimal(str(float(self.cfg.deadline_s))).normalize(), "f")
                first = _at_tools.first_message(self._attention, req, wall)
            pr = self._pass_record
            if pr is not None:
                # the run record: the common section and this agent's roster lines, after the constant brief
                from unify.agents.run_record import prompt_lines

                first += "\n\n" + prompt_lines(pr.root, "writer")
            self._inputs_dir = Path(inputs)
            # point, don't paste: the use table is a file (/inputs/usage/table.md), named by the brief
            self._usage(req, wt)
            messages: list[dict] = [
                # the library-mode surfacing switches' brief, then the stage-5 rewrites and paragraph (each
                # unchanged while its switches are off)
                {
                    "role": "system",
                    "content": (
                        self._brief_library_mode()
                    ),
                },
                {"role": "user", "content": first},
            ]
            self.messages = messages
            self._tools_json = list(tools)
            finished = False


            def commit_and_check(
                manifest: object,
                box: Path = box,
            ) -> tuple[str, GateResult]:
                """Library mode: the box as one commit on the parent, and the whole gate on it without merging."""
                candidate = self._commit_round(box, parent, pass_id, summary, manifest)
                return candidate, self.gate.check(parent, candidate, manifest)

            def check_tests(raw: object, box: Path = box) -> str:
                """Amendment B: in library mode ``check`` is the whole gate (``gate.check``, no merge, nothing written
                to the evidence store; it runs no code). A refusal is answered with the gate's full result as a
                head-first view of /inputs/gate/check-<k>.md, which read pages.
                """
                manifest, problem = _check_manifest(raw)
                if problem is not None:
                    return problem
                manifest = _with_fixtures(manifest, self._ledger(box))
                try:
                    candidate, res = commit_and_check(manifest, box)
                except (
                    Exception
                ) as exc:  # noqa: BLE001 - a broken check never ends the pass
                    notes.append(
                        _redact(f"check error: {type(exc).__name__}: {exc}")[:300],
                    )
                    return f"check error: {type(exc).__name__}"
                if res.passed:
                    return "ok: the gate's checks pass (the gate runs no code; your tests are the ones you run)"
                text = _repair.render_result(
                    pass_id,
                    round_no,
                    res,
                    final=False,
                    candidate=candidate,
                )
                # a file under /inputs, so a long result can be paged with read at the marker's offset
                vpath = f"/inputs/gate/check-{check_files[0]}.md"
                (inputs / "gate").mkdir(exist_ok=True)
                (inputs / "gate" / f"check-{check_files[0]}.md").write_text(text)
                check_files[0] += 1
                return _views.view(text.encode()) + f"\n(full result: {vpath})"

            def check_file(
                args: dict,
                base: ParentSnapshot | str,
                box: Path = box,
            ) -> _wt.Reply:
                """``check`` with the library on: it judges /memory/.pass/manifest.json, the file finish and the gate
                judge; a ``manifest`` argument is written there first. Every reason the static checks give is saved
                in /inputs/gate/check-<k>.md and the reply names that file."""
                where = [{"path": "/memory/.pass/manifest.json", "mode": "read"}]
                given = args.get("manifest")
                if given not in (
                    None,
                    "",
                ):  # an empty argument means: judge the file as it is
                    parsed, problem = _check_manifest(given)
                    if problem is not None:
                        return _wt.refused(
                            "check",
                            "manifest_problem",
                            tuple(where),
                            reason=problem,
                        )
                    try:
                        _wt.write_manifest(
                            box,
                            given if isinstance(given, str) else parsed,
                        )
                    except OSError as exc:
                        return _wt.refused(
                            "check",
                            "manifest_problem",
                            tuple(where),
                            reason=f"cannot write /memory/.pass/manifest.json ({type(exc).__name__})",
                        )
                    where.insert(
                        0,
                        {"path": "/memory/.pass/manifest.json", "mode": "write"},
                    )
                found, manifest, problem = _read_manifest(box)
                if not found or problem is not None:
                    return _wt.refused(
                        "check",
                        "manifest_problem",
                        tuple(where),
                        reason=problem
                        or "no manifest: write /memory/.pass/manifest.json (or pass it as manifest), then check",
                    )
                reasons: list[str] = []
                content, detail = self._check(manifest, box, base, reasons)
                if detail is not None:
                    notes.append(detail)
                    return _wt.Reply(content, "error", None, tuple(where))
                if content == "ok":
                    return _wt.Reply(check_tests(manifest, box), paths=tuple(where))
                vpath = f"/inputs/gate/check-{check_files[0]}.md"
                (inputs / "gate").mkdir(exist_ok=True)
                (inputs / "gate" / f"check-{check_files[0]}.md").write_text(
                    "\n".join(reasons) + "\n",
                )
                check_files[0] += 1
                return _wt.Reply(
                    f"{content}\n(full result: {vpath})",
                    paths=tuple(where),
                )

            def repair() -> bool:
                """Library mode, after an accepted finish: True opens repair round ``round_no + 1``.

                False ends the pass: its final candidate then goes to the one merge (per-item admission).
                """
                nonlocal round_no, stop, over_quota
                round_s.append(time.monotonic() - mark[0])

                def refused_round(check_s: float) -> str | None:
                    return _repair.may_repair(
                        round_no,
                        remaining(),
                        max_calls - calls,
                        round_s,
                        last_check_s=check_s,
                    )

                why = refused_round(
                    checked["seconds"],
                )  # before the check: no jail time on a pass that cannot repair
                if why is not None:
                    notes.append(why)
                    return False
                over_quota = _measure(
                    box,
                )  # also restores the host's access before the mirror
                if over_quota is not None:
                    stop = over_quota
                    return False
                started = time.monotonic()
                found, manifest, problem = _read_manifest(box)
                manifest = _with_fixtures(manifest, self._ledger(box))
                candidate = None
                if not found or problem is not None:
                    res = GateResult(False, {}, [problem or "no manifest"], [])
                else:
                    try:
                        candidate, res = commit_and_check(manifest)
                    except (
                        Exception
                    ) as exc:  # noqa: BLE001 - the final merge records the gate's own error
                        notes.append(
                            _redact(
                                f"repair: the gate's check failed: {type(exc).__name__}: {exc}",
                            )[:300],
                        )
                        return False
                    checked.update(candidate=candidate, manifest=manifest, current=True)
                check_s = time.monotonic() - started
                checked["seconds"] = check_s
                round_s[
                    -1
                ] += check_s  # the gate's jail time counts against the deadline
                if res.passed:
                    return False  # the one merge lands this very commit
                vpath = f"/inputs/gate/result-{round_no}.md"
                text = _repair.render_result(
                    pass_id,
                    round_no,
                    res,
                    final=False,
                    candidate=candidate,
                )
                (inputs / "gate").mkdir(exist_ok=True)
                (inputs / "gate" / f"result-{round_no}.md").write_text(text)
                round_paths.append(vpath)
                store = getattr(self.gate, "blobs", None)
                if isinstance(store, BlobStore):
                    round_blobs.append(store.put(text.encode()))
                # no round count: a refusal repeated round to round is stuck instead (repair.py)
                was = set(stuck.rows)
                held = stuck.update(res, self._candidate_key(candidate, manifest))
                fresh = [h for h in held if (h[0], h[1]) not in was]
                if fresh:
                    notes.append(_repair.stuck_note(fresh, stuck.repeats))
                why = refused_round(
                    check_s,
                )  # after the check, with its time: this one decides
                if why is not None:
                    notes.append(why)
                    return False
                if stuck.all_stuck():
                    stuck.stopped = True
                    notes.append(_repair.stopped_note(round_no + 1, held, stuck.repeats))
                    return False
                round_no += 1
                checked["current"] = False  # the writer may change the box again
                messages.append(
                    {
                        "role": "user",
                        "content": _repair.repair_message(
                            round_no,
                            vpath,
                            res,
                            stuck=held,
                            repeats=stuck.repeats,
                        ),
                    },
                )
                mark[0] = time.monotonic()
                return True

            ic: Any = None  # tools in code: the handlers' pass state and the interpreter session
            session: Any = None
            ic, session = self._in_code_start(
                pass_id=pass_id,
                box=box,
                inputs=inputs,
                tmp=Path(tmp),
                roots=roots,
                parent=parent,
                max_calls=max_calls,
                remaining=remaining,
                cause=cause,
                check_file=check_file,
                phase=phase,
                notes=notes,
                deadline=deadline,
                spend=spend,
                allow=in_code_allow,
                scope=scope,
            )
            # tools in code off, library mode: the cells' persistent interpreter, no channel (None: a fresh box per
            # cell, or a fresh box per cell)
            cell_session: Any = None
            last_prompt: list = [
                None,
            ]  # the last model call's prompt tokens (compaction_due)
            while (
                not finished
                and stop is None
                and calls < max_calls
            ):
                if remaining() <= 0:
                    stop = f"pass deadline of {self.cfg.deadline_s} s reached"
                    cause(CODE_DEADLINE)
                    break
                if not usd_room(self.cfg, spend):
                    stop = f"the writer service's room of {self.cfg.usd_limit} USD for this pass reached"
                    cause(CODE_PASS_CAP)
                    break
                calls += 1
                if (
                    compaction_due(
                        last_prompt[0],
                        getattr(self.turn, "max_input_tokens", None),
                    )
                ):
                    # the agent loop's compaction: a summary call on the writer's own turn, booked as a call
                    last_prompt[0] = None
                    try:
                        with call_in_flight(spend):
                            comp = await asyncio.wait_for(
                                compact_by_summary(messages, self.turn, tools),
                                timeout=remaining(),
                            )
                    except TimeoutError:
                        spend.unknown += 1
                        stop = f"pass deadline of {self.cfg.deadline_s} s reached during a model call"
                        cause(CODE_DEADLINE)
                        break
                    except (
                        Exception
                    ) as exc:  # it may still have cost money (unless the route refused it unsent); the conversation stays
                        if unbooked_cost(exc):
                            spend.unknown += 1
                        notes.append(
                            _redact(f"compaction call failed: {type(exc).__name__}")[
                                :300
                            ],
                        )
                        comp = None
                    if comp is not None:
                        cost = _money(comp.usd)
                        if cost is None:
                            spend.unknown += 1
                        else:
                            spend.usd += cost
                        self._compaction_event(pass_id, messages, comp)
                        if comp.restart is not None:
                            self._archived += compaction_archive(comp, list(messages))
                            messages[:] = comp.restart
                    if calls >= max_calls:
                        continue  # the summary call took the last call the pass had
                    calls += 1
                pr_run = self._pass_record
                if pr_run is not None:
                    # the run record's turn boundary: what is new for this agent, after the turn's tool results
                    try:
                        block = pr_run.record.take_block(pr_run.root)
                    except Exception as exc:  # noqa: BLE001 - delivery never stops a pass
                        block = None
                        notes.append(_redact(f"record block not delivered: {type(exc).__name__}")[:300])
                    if block:
                        messages.append({"role": "user", "content": block})
                if not usd_room(self.cfg, spend):  # a compaction call may have used the room just checked
                    stop = f"the writer service's room of {self.cfg.usd_limit} USD for this pass reached"
                    cause(CODE_PASS_CAP)
                    break
                async def one_call() -> tuple[dict, str]:
                    with call_in_flight(spend):
                        return await asyncio.wait_for(
                            self.turn(messages, tools),
                            timeout=remaining(),
                        )

                def book_unpriced() -> None:
                    spend.unknown += 1

                try:
                    # a transient refusal (the route busy, a rate limit, a server error) is tried again, bounded
                    msg, usd = await call_retrying(
                        one_call,
                        book_unpriced=book_unpriced,
                        remaining=remaining,
                        # the pass's room, as before the first attempt
                        room=lambda: usd_room(self.cfg, spend),
                        note=lambda text: notes.append(_redact(text)[:300]),
                    )
                except TimeoutError:
                    spend.unknown += 1
                    stop = f"pass deadline of {self.cfg.deadline_s} s reached during a model call"
                    cause(CODE_DEADLINE)
                    break
                except Exception as exc:  # the call may still have cost money (not one the route refused unsent)
                    if unbooked_cost(exc):
                        spend.unknown += 1
                    errored = True
                    failed_as = (
                        CODE_ROUTE_NOT_IN_EFFECT
                        if isinstance(exc, WriterRouteError)
                        else CODE_WRITER_ERROR
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
                    cause(CODE_WRITER_ERROR)
                    break
                # the call's usage report stays with this turn; the next request never carries it
                msg, reported = split_usage(msg)
                messages.append(msg)
                tool_calls = msg.get("tool_calls") or []
                tool_calls = tool_calls if isinstance(tool_calls, list) else []
                ran = 0
                readers = 0  # library_mode reader calls in this turn
                turn_usage: dict | None = _usage_of(msg, usd, reported)
                last_prompt[0] = turn_usage["input"]
                self._count_cache(turn_usage)
                for n, tc in enumerate(tool_calls):
                    if not isinstance(tc, dict):
                        messages.append(
                            {
                                "role": "tool",
                                "tool_call_id": f"call_{calls}_{n}",
                                "content": "unreadable tool call",
                            },
                        )
                        self._event(
                            pass_id,
                            None,
                            None,
                            _wt.general("bad_arguments", reason="unreadable tool call"),
                            turn_usage,
                        )
                        turn_usage = None
                        continue
                    if not isinstance(tc.get("id"), str) or not tc["id"]:
                        tc["id"] = (
                            f"call_{calls}_{n}"  # the reply must name the call it answers
                        )
                    fn = tc.get("function")
                    fn = fn if isinstance(fn, dict) else {}
                    name = fn.get("name")
                    args, content = _parse_args(fn.get("arguments"))
                    reply: _wt.Reply | None = None
                    # delegate and the record's tools: the event row's `messaging` object (and the helper's name)
                    event_extra: dict | None = None
                    event_call: int | None = None  # tools in code: the cell row's reserved number
                    if args is None:
                        reply = _wt.general("bad_arguments", reason=str(content))
                    elif finished:
                        reply = _wt.general("pass_finished")
                    elif name != "execute_code":
                        reply = _wt.general(
                            "unknown_tool",
                            name=str(name)[:80],
                            tools=", ".join(tool_names),
                        )
                    elif _thought_only(args):
                        # a thought alone: nothing runs and nothing is refused; the thought stays in the conversation
                        reply = _wt.Reply(_thought_only_reply())
                    elif stop is not None:
                        reply = _wt.general("stopped", reason=stop)
                    elif ran >= _MAX_CELLS_PER_TURN:
                        reply = _wt.general("cells_per_turn")
                    elif remaining() <= 0:
                        stop = f"pass deadline of {self.cfg.deadline_s} s reached"
                        cause(CODE_DEADLINE)
                        reply = _wt.general("stopped", reason=stop)
                    else:
                        ran += 1
                        timeout = max(
                            1.0,
                            min(float(self.cfg.cell_timeout_s), remaining()),
                        )
                        if cell_session is not None:
                            # tools in code off: the pass's persistent interpreter, no channel
                            content = await self._in_code_cell(
                                cell_session,
                                None,
                                str(args.get("code", "")),
                                cells,
                                len(self.tool_events),  # the cell's event row number
                            )
                        elif ic is None:
                            content = await self._run_cell(
                                str(args.get("code", "")),
                                box,
                                inputs,
                                cells,
                                timeout,
                            )
                        else:
                            # the cell's in-code calls run through the channel against these counters
                            event_call = self._reserve_event()
                            ic.calls, ic.checks, ic.reads, ic.readers = (
                                calls,
                                checks,
                                reads,
                                readers,
                            )
                            ic.stop, ic.finished, ic.summary, ic.base = (
                                stop,
                                finished,
                                summary,
                                base,
                            )
                            ic.finish_text = None
                            try:
                                content = await self._in_code_cell(
                                    session,
                                    ic,
                                    str(args.get("code", "")),
                                    cells,
                                    event_call,
                                )
                            except BaseException as exc:
                                # the cell's row still lands at its reserved number, with an error status, so no
                                # placeholder outlives the pass; the exception goes on as before
                                self._crashed_cell_event(
                                    pass_id,
                                    fn.get("arguments"),
                                    args,
                                    turn_usage,
                                    event_call,
                                    exc,
                                )
                                raise
                            calls, checks, reads, readers = (
                                ic.calls,
                                ic.checks,
                                ic.reads,
                                ic.readers,
                            )
                            stop, finished, summary, base = (
                                ic.stop,
                                ic.finished,
                                ic.summary,
                                ic.base,
                            )
                            if session.stuck and stop is None:
                                stop = "a tool handler did not end within its deadline"
                        over_quota = _measure(box)
                        if over_quota is not None:
                            stop = over_quota
                            content += f"\n({over_quota}; the pass is refused)"
                        reply = _wt.Reply(content)
                    if reply is not None:
                        content = reply.text
                    messages.append(
                        # plain text: a reply may be a str subclass carrying its refusal form
                        {
                            "role": "tool",
                            "tool_call_id": tc["id"],
                            "content": str(content),
                        },
                    )
                    # the additive fields of every row with tools in code (a model-made call: no cell)
                    event_extra = {
                        "via": "model",
                        "cell": None,
                        "args_sha256": _args_sha256(
                            args if args is not None else fn.get("arguments"),
                        ),
                        "box_gone": False,
                        **(event_extra or {}),
                    }
                    self._event(
                        pass_id,
                        name,
                        fn.get("arguments"),
                        reply,
                        turn_usage,
                        event_extra,
                        event_call,
                    )
                    turn_usage = None
                if not tool_calls:
                    messages.append(
                        {
                            "role": "user",
                            "content": (
                                ("Use your tools: "
                                + ", ".join(tool_names)
                                + "; finish ends the pass.")
                            ),
                        },
                    )
                if no_progress is not None:
                    # a call made while a helper of the pass runs is not counted: waiting on it is progress the loop
                    # cannot see. At the end the round ends as if finish were accepted, without a repair round: the
                    # final merge lands the items that pass and keeps the rest as a draft
                    acted = no_progress.after_call(
                        messages,
                        in_flight=bool(self._pass_record is not None and self._pass_record.helpers_running()),
                        act=not finished and stop is None,
                    )
                    if acted is not None:
                        notes.append(no_progress.note(acted))
                    if acted == "end":
                        cause(CODE_NO_PROGRESS)
                        break
                if finished and stop is None and repair():
                    finished = (
                        False  # a repair round: the loop goes on under the same caps
                    )
            # the loop has ended: every helper still running ends now, so its row is in the outcome
            await self._close_record()
            if ic is not None or cell_session is not None:
                # no box can write /memory and no tool handler of the pass may run while the gate judges it (tools
                # in code on or off)
                live = session if ic is not None else cell_session
                try:
                    await live.close()
                except Exception:  # noqa: BLE001 - a stop that failed is a box that may still run: fail closed
                    logging.getLogger(__name__).warning("the pass's interpreter did not stop cleanly", exc_info=True)
                    alive = True
                else:
                    alive = live.box_alive()
                idle = not alive and (ic is None or await ic.threads.wait_idle(self.in_code_idle_s))
                self._handlers_at_gate = ic.threads.running() if ic is not None else 0
                if not idle:
                    why = (
                        "a box of this pass is still running"
                        if alive
                        else "a tool handler of this pass is still running"
                    )
                    # fail closed: no gate, and the pass's folders stay for the thread that may still use them
                    pass_temp.keep = why
                    self._temp_kept = {"reason": why, "path": str(tmp)}
                    cause(CODE_WRITER_ERROR)
                    reasons = self._fail(
                        req,
                        pass_id,
                        parent,
                        spend,
                        notes + [f"{why}: the gate did not start; the pass's temporary folder was kept"],
                    )
                    return outcome(False, None, reasons)
            if (
                not finished
                and stop is None
                and not errored
                and calls >= max_calls
            ):
                # the operational step guard ended the pass
                cause(CODE_STEP_GUARD)
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
                reasons = self._fail(req, pass_id, parent, spend, notes + [over_quota])
                self._close_rounds(
                    pass_id,
                    round_no,
                    round_blobs,
                    GateResult(False, {}, list(reasons), []),
                    None,
                    (),
                )
                return outcome(False, None, reasons)
            found, manifest, problem = _read_manifest(box)
            if not found:
                cause(CODE_NO_MANIFEST)
                reasons = self._fail(
                    req,
                    pass_id,
                    parent,
                    spend,
                    notes + ["no manifest"],
                )
                self._close_rounds(
                    pass_id,
                    round_no,
                    round_blobs,
                    GateResult(False, {}, list(reasons), []),
                    None,
                    (),
                )
                return outcome(False, None, reasons)
            if problem is not None:
                notes.append(problem)
                cause(CODE_MANIFEST_INVALID)
            manifest = _with_fixtures(manifest, self._ledger(box))
            _remove(box / ".pass")
            _clear_checkout(wt)
            left_out = _mirror(box, wt, skip_top=frozenset({".pass"}))
            _drop_unchanged(wt, self._library_generated)
            if left_out:
                notes.append(f"special or unreadable entries left out: {left_out[:5]}")
            sources = _sources(manifest)
            if checked["current"]:
                # Amendment A: the box is unchanged since the last check, so the merge lands that very commit
                candidate = checked["candidate"]
            else:
                candidate = self.mem.commit_all(
                    wt,
                    f"consolidation pass {pass_id}: {summary[:200]}".replace("\0", ""),
                    _trailers(pass_id, sources, manifest),
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
        self._provenance = list(getattr(res, "provenance", None) or [])
        self._levels = level_counts(res) if res.passed else (None, None)
        for check in getattr(res, "refused", []):
            cause(check)  # the gate checks that refused: G1..G6
        # the pass's own notes (left-out entries, deadline, model failures) follow the gate's reasons
        tail = notes + _unpriced(spend.unknown)
        self.ev.add_pass_notes(pass_id, tail)
        self._close_rounds(pass_id, round_no, round_blobs, res, candidate, tail)
        # the commit that landed: the candidate, or its reduction to the admitted items
        landed = getattr(res, "merged", None) or candidate
        return outcome(
            res.passed,
            landed if res.passed else None,
            res.reasons + tail,
            getattr(res, "items_merged", None),
            getattr(res, "items_refused", None),
        )


# --- the real model ----------------------------------------------------------------------------------------


@dataclass(frozen=True)
class WriterRoute:
    """The writer's own route (``UNIFY_MEMORY_WRITER_BASE_URL`` / ``UNIFY_MEMORY_WRITER_TOKEN``): an OpenAI-compatible
    base URL and its token. The token is a ``SecretStr``: its repr and str are masked.
    """

    base_url: str
    token: SecretStr


#: The route of the writer call running in this context, read by unillm's per-call gateway lookup
#: (:func:`_install_writer_gateway`). Context-local: a call in any other task or thread never sees it.
_WRITER_GATEWAY: contextvars.ContextVar[WriterRoute | None] = contextvars.ContextVar(
    "memory_writer_gateway",
    default=None,
)
_GATEWAY_LOCK = threading.Lock()


class WriterRouteError(RuntimeError):
    """The writer's declared route was not in effect for a writer call (before it was sent, or seen in its event)."""


class WriterCallError(RuntimeError):
    """A writer call on the declared route failed. Its text is the exception's class and a fixed category only:
    the original text (an echoed request header, say) never reaches pass notes, error rows or logs.

    ``category`` is that category (:func:`_call_failure`). ``transient``: a failure a later attempt may not get
    (:data:`TRANSIENT_CATEGORIES`: the route's one-call-at-a-time refusal, a rate limit, a server error), tried again
    by :func:`call_retrying`. ``no_cost``: the route refused the call before sending it on (its refusal body says so,
    :func:`_route_refusal`), so the call cost nothing. ``counted``: the pass already booked this failed attempt.
    """

    def __init__(self, text: str, category: str = "other") -> None:
        super().__init__(text)
        self.category = category
        self.transient = transient_category(category)
        self.no_cost = category in (REFUSED_BUSY, REFUSED_CAP)
        self.counted = False


#: A 403 whose body is the route's refusal for its cap (``reason`` ``*_cap_reached``, or a ``*_budget_exhausted``
#: refusal that is not about a call in flight): the route's spend cap, as HTTP 402 is.
REFUSED_CAP = "http_403_cap"
#: A 403 whose body is the route's refusal because another call is in flight (``reason`` ``*_in_flight`` or
#: ``*_race``, or a ``*_budget_exhausted`` refusal for a call in flight or a reserve not yet released): for now only.
REFUSED_BUSY = "http_403_busy"
#: The categories a later attempt may not get: the route busy, a rate limit, a server error.
TRANSIENT_CATEGORIES = frozenset({REFUSED_BUSY, "http_429"} | {f"http_{s}" for s in range(500, 600)})
#: The categories that are the route's spend cap reached: HTTP 402, and a 403 whose body says it is the cap.
CAP_CATEGORIES = frozenset({"http_402", REFUSED_CAP})
#: A transient failure is tried again at most this many times in a row, waiting TRANSIENT_BACKOFF_S, then twice
#: that, and so on (2 + 4 + 8 + 16 s), never past the pass's deadline.
TRANSIENT_RETRIES = 4
TRANSIENT_BACKOFF_S = 2.0


def transient_category(category: str) -> bool:
    return category in TRANSIENT_CATEGORIES


def unbooked_cost(exc: BaseException) -> bool:
    """Whether a failed call still has to be booked as one unpriced call: not when the route refused it before
    sending it on (``no_cost``), nor when :func:`call_retrying` booked it already (``counted``)."""
    return not (getattr(exc, "no_cost", False) or getattr(exc, "counted", False))


def _error_object(value: Any) -> dict | None:
    """The refusal's error object in a parsed body: ``{"error": {...}}`` or the inner object itself."""
    if not isinstance(value, dict):
        return None
    inner = value.get("error")
    obj = inner if isinstance(inner, dict) else value
    if isinstance(obj.get("type"), str) or isinstance(obj.get("reason"), str):
        return obj
    return None


#: How much of an exception's text is searched for the refusal body, from how many opening braces, and how long
#: one object may be (a refusal body is a few hundred characters).
_REFUSAL_SCAN_CHARS = 65536
_REFUSAL_STARTS = 64
_REFUSAL_OBJECT_CHARS = 8192


def _objects_in_text(text: str) -> Iterator[dict]:
    """The brace-balanced objects in *text* that parse as JSON or as a Python literal (a client may show the body
    either way), first to last, within the bounds above; literals only, nothing is run."""
    text = text[:_REFUSAL_SCAN_CHARS]
    start = text.find("{")
    starts = 0
    while start != -1 and starts < _REFUSAL_STARTS:
        starts += 1
        depth = 0
        for end in range(start, min(len(text), start + _REFUSAL_OBJECT_CHARS)):
            if text[end] == "{":
                depth += 1
            elif text[end] == "}":
                depth -= 1
                if depth == 0:
                    piece = text[start : end + 1]
                    for parse in (json.loads, ast.literal_eval):
                        try:
                            value = parse(piece)
                        except (ValueError, SyntaxError, TypeError, MemoryError, RecursionError):
                            continue
                        if isinstance(value, dict):
                            yield value
                        break
                    break
        start = text.find("{", start + 1)


def _route_refusal(exc: BaseException) -> str | None:
    """``"busy"`` or ``"cap"`` when *exc*, or an exception it was raised from or during, carries the route's refusal
    body (``{"error": {"type", "message", "reason"?}}``, from the client's parsed body, its response, or its text);
    else None. Read from the body, never from the status code: its ``reason`` first (``*_in_flight`` or ``*_race``
    busy, ``*_cap_reached`` the cap), then, for a body without one of those, a ``*_budget_exhausted`` type, busy when
    its message is about a call in flight or a reserve not yet released, else the cap. Only the kind is returned; no
    text of the body is kept."""
    seen: set[int] = set()
    e: BaseException | None = exc
    while e is not None and id(e) not in seen:
        seen.add(id(e))
        found: list[dict] = []
        body = _error_object(getattr(e, "body", None))
        if body is not None:
            found.append(body)
        response = getattr(e, "response", None)
        if response is not None and callable(getattr(response, "json", None)):
            try:
                parsed = _error_object(response.json())
            except Exception:  # noqa: BLE001 - a response not read, or not JSON: its text below
                parsed = None
            if parsed is not None:
                found.append(parsed)
        texts = [getattr(e, "message", None), *getattr(e, "args", ())]
        for text in texts:
            if isinstance(text, str):
                found.extend(o for o in map(_error_object, _objects_in_text(text)) if o is not None)
        for obj in found:
            kind = _refusal_kind(obj)
            if kind is not None:
                return kind
        e = e.__cause__ or e.__context__
    return None


def _refusal_kind(obj: dict) -> str | None:
    reason = obj.get("reason")
    if isinstance(reason, str):
        if reason.endswith(("_in_flight", "_race")):
            return "busy"
        if reason.endswith("_cap_reached"):
            return "cap"
    kind = obj.get("type")
    if isinstance(kind, str) and kind.endswith("_budget_exhausted"):
        message = obj.get("message")
        text = message.lower() if isinstance(message, str) else ""
        return "busy" if ("in flight" in text or "release race" in text) else "cap"
    return None


async def call_retrying(
    call: Callable[[], Any],
    *,
    book_unpriced: Callable[[], None],
    remaining: Callable[[], float],
    room: Callable[[], bool] | None = None,
    note: Callable[[str], None] | None = None,
    sleep: Callable[[float], Any] = asyncio.sleep,
) -> Any:
    """``await call()``, tried again after a transient :class:`WriterCallError` (:data:`TRANSIENT_CATEGORIES`) at
    most :data:`TRANSIENT_RETRIES` times in a row, waiting :data:`TRANSIENT_BACKOFF_S`, doubled each time. Each failed
    attempt that is tried again is booked first: nothing when the route refused it before sending it on
    (``no_cost``), else one unpriced call (*book_unpriced*). No new attempt starts past the pass's deadline
    (*remaining*) or its room (*room*, checked with the failed attempt booked); then, and on any other failure, the
    error is raised (``counted`` when it was booked here)."""
    attempt = 0
    while True:
        try:
            return await call()
        except WriterCallError as exc:
            wait = TRANSIENT_BACKOFF_S * (2**attempt)
            if not exc.transient or attempt >= TRANSIENT_RETRIES or remaining() <= wait:
                raise
            if not exc.no_cost:
                book_unpriced()
            exc.counted = True
            if room is not None and not room():
                raise
            attempt += 1
            if note is not None:
                note(
                    f"model call refused for now ({exc.category}); tried again after {wait:g} s "
                    f"({attempt} of {TRANSIENT_RETRIES})",
                )
            await sleep(wait)


_CLASS_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]{0,63}")


#: Packages whose errors carry a provider's or a transport's text, which may echo the request.
_CLIENT_LIBRARIES = ("unillm", "litellm", "openai", "httpx", "httpcore", "aiohttp")


def _touches_client(exc: BaseException) -> bool:
    """Whether any frame of *exc*'s traceback, or of any exception in its cause and context chain, ran in a package
    of :data:`_CLIENT_LIBRARIES`."""
    seen: set[int] = set()
    todo: list[BaseException] = [exc]
    while todo:
        e = todo.pop()
        if id(e) in seen:
            continue
        seen.add(id(e))
        tb = e.__traceback__
        while tb is not None:
            path = tb.tb_frame.f_code.co_filename
            if any(f"/{lib}/" in path for lib in _CLIENT_LIBRARIES):
                return True
            tb = tb.tb_next
        todo.extend(x for x in (e.__cause__, e.__context__) if x is not None)
    return False


def _status_in_chain(exc: BaseException) -> object:
    """The HTTP status code of *exc*, or of the first exception it was raised from or during that has one. A client
    may re-raise a provider's error as a bare exception (unillm does, dropping the code); the original stays in the
    chain, so the status is still known."""
    seen: set[int] = set()
    e: BaseException | None = exc
    while e is not None and id(e) not in seen:
        seen.add(id(e))
        status = getattr(e, "status_code", None)
        if isinstance(status, int) and not isinstance(status, bool):
            return status
        e = e.__cause__ or e.__context__
    return None


def _call_failure(exc: BaseException) -> tuple[str, str, str]:
    """``(class name, category, detail)`` of a failed call. The category comes from its type and status code only.
    *detail* is empty for an answer from the provider or the route (an HTTP status, a timeout, a connection or
    authentication failure), whose text may echo the request; for any other failure, an error raised by code in
    this process, it names where the error was raised and its message, redacted and clipped.
    """
    name = type(exc).__name__
    name = name if _CLASS_NAME.fullmatch(name) else "Exception"
    status = _status_in_chain(exc)
    lowered = name.lower()
    if (
        isinstance(status, int)
        and not isinstance(status, bool)
        and 100 <= status <= 599
    ):
        category = f"http_{status}"
        if status == 403:
            # the route's own refusal, told apart by its body only (never by the status code): busy for now, or the cap
            kind = _route_refusal(exc)
            category = {"busy": REFUSED_BUSY, "cap": REFUSED_CAP}.get(kind or "", category)
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
    detail = ""
    if category == "other":
        tb, where = exc.__traceback__, ""
        while tb is not None:
            code = tb.tb_frame.f_code
            where = f"{Path(code.co_filename).name}:{tb.tb_lineno} in {code.co_name}"
            tb = tb.tb_next
        detail = f" at {where}" if where else ""
        # the message only when no frame of the error or of any error it was raised from or during ran in the model
        # client: a client's error text may echo the request, and a local error may carry it on
        if not _touches_client(exc):
            message = redact_error(str(exc)).replace("\n", " ")[:200]
            detail += f": {message}" if message else ""
    return name, category, detail


def _check_route_in_effect(route: WriterRoute) -> None:
    """Refuse before any call unless unillm's request preparation, in this context, sends to *route*.

    A pure call to the same function every attempt (retries included) prepares its transport with; it shows
    the lookup it reads is the wrapped one. The prepared credential is compared, never kept or shown.
    """
    from unillm.clients import uni_llm

    kw: dict = {"model": "openrouter/memory-writer-route-check", "messages": []}
    uni_llm._prepare_provider_request_kw(kw=kw, provider="openrouter", stream=False)
    ok = (
        str(kw.get("api_base") or "").rstrip("/") == route.base_url
        and kw.get("api_key") == route.token.get_secret_value()
    )
    kw.clear()
    if not ok:
        raise WriterRouteError(
            "the writer's declared route is not in effect; no writer call was made",
        )


#: Why the writer's route does not start while unillm records OTel spans: a span keeps a failed call's text as is.
OTEL_REFUSAL = (
    "UNILLM_OTEL is on (or its state cannot be read): unillm's spans record a failed call's error text, "
    "which the writer's route keeps out of every record, so the writer's route does not start and no consolidation pass "
    "runs; turn UNILLM_OTEL off for a run with the writer's route"
)


def otel_on() -> bool:
    """Whether unillm records OTel spans (``UNILLM_OTEL``); ``True`` when that cannot be read (fail closed)."""
    try:
        import importlib

        return bool(importlib.import_module("unillm.logger").is_otel_enabled())
    except Exception:  # noqa: BLE001
        return True


class _RedactRecords(logging.Filter):
    """Redacts a log record before any handler sees it: its message with :func:`redact_error` applied, no
    arguments left to format, and any traceback text redacted the same way.

    Installed on ``unillm.retry`` itself (a filter on a parent logger does not run for a child's records),
    whose warnings carry the first 200 characters of a failed call's text and reach the controller's stderr.
    It only removes registered credentials and credential structures, so it applies to every record there.
    """

    _memory_writer = True

    def filter(self, record: logging.LogRecord) -> bool:
        try:
            text = record.getMessage()
        except Exception:  # noqa: BLE001
            # a record that cannot format keeps its template only
            text = str(record.msg)
        record.msg = redact_error(text)
        record.args = ()
        if record.exc_info:
            record.exc_text = redact_error(
                logging.Formatter().formatException(record.exc_info),
            )
            record.exc_info = None
        elif record.exc_text:
            record.exc_text = redact_error(record.exc_text)
        if record.stack_info:
            record.stack_info = redact_error(record.stack_info)
        return True


#: The unillm logger whose records carry a failed call's text (``unillm/helpers.py``).
RETRY_LOGGER = "unillm.retry"


def _install_log_redaction() -> None:
    """Put :class:`_RedactRecords` on ``unillm.retry``; idempotent (the caller holds the lock)."""
    log = logging.getLogger(RETRY_LOGGER)
    if not any(getattr(f, "_memory_writer", False) for f in log.filters):
        log.addFilter(_RedactRecords())


def _redact_log_file(path: Any) -> None:
    """unillm's per-call log file of a writer call (written only when ``UNILLM_LOG_DIR`` is set), which holds a
    failed call's error text as is, rewritten with :func:`redact_error` applied (a temporary file beside it
    replaces it). If it cannot be rewritten it is removed. Never raises.
    """
    target = Path(path)
    try:
        text = target.read_bytes().decode("utf-8", "surrogateescape")
        clean = redact_error(text)
        if clean == text:
            return
        fd, tmp = tempfile.mkstemp(dir=target.parent, prefix=f".{target.name}.")
        try:
            with os.fdopen(fd, "wb") as fh:
                fh.write(clean.encode("utf-8", "surrogateescape"))
            os.replace(tmp, target)
        except BaseException:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise
    except Exception:  # noqa: BLE001 - unredacted, it may hold the token: it goes
        try:
            target.unlink()
        except OSError:
            pass


def _install_writer_gateway() -> None:
    """Make unillm's per-call gateway lookup return the writer's route inside a writer call; idempotent.

    unillm sends an OpenRouter call to ``_llm_gateway()`` (read on every call, before the provider credential
    is filled in, which never overwrites a chosen ``api_key``), but its clients take no per-call
    ``api_base``/``api_key``: ``generate`` drops extra keyword arguments. Wrapping that lookup with a
    context variable gives the per-call route without touching the process environment, which subprocesses
    inherit. Outside a writer call the wrapper returns what unillm's own lookup returns. Refuses (no call made)
    when the installed unillm has no such lookup, or records OTel spans (:data:`OTEL_REFUSAL`). Also puts a
    redacting filter on the ``unillm.retry`` logger (:class:`_RedactRecords`).
    """
    from unillm.clients import uni_llm

    if otel_on():
        raise WriterRouteError(OTEL_REFUSAL)
    with _GATEWAY_LOCK:
        _install_log_redaction()
        lookup = getattr(uni_llm, "_llm_gateway", None)
        if getattr(lookup, "_memory_writer", False):
            return
        if not callable(lookup) or not callable(
            getattr(uni_llm, "_prepare_provider_request_kw", None),
        ):
            raise RuntimeError(
                "the writer's route is set, but the installed unillm has no per-call gateway lookup to apply it",
            )

        def gateway() -> tuple[str, str] | None:
            route = _WRITER_GATEWAY.get()
            if route is not None:
                return route.base_url, route.token.get_secret_value()
            return lookup()

        gateway._memory_writer = True  # type: ignore[attr-defined]
        gateway.__wrapped__ = lookup  # type: ignore[attr-defined]
        uni_llm._llm_gateway = gateway


def _started_order(row: dict) -> tuple[int, str]:
    """A helper row's place in the pass outcome: the order the writer started its helpers (``call-<n>``, numbered as
    they start), whatever order they ended in."""
    call = str(row.get("call") or "")
    n = call.rsplit("-", 1)[-1]
    return (int(n) if n.isdigit() else 1 << 30, call)


def unillm_turn(
    model: str,
    effort: str,
    *,
    origin: str = "memory_writer",
    session: str | None = None,
    route: WriterRoute | None = None,
) -> ModelTurn:
    """The real model call: one ``generate`` per turn on a unillm client, priced from unillm's LLM events.

    * The model is passed explicitly, so ``UNIFY_REASONING_EFFORT`` and the assistant default do not
      override it; ``effort`` is fixed for the pass. unillm needs a ``model@provider`` endpoint: a bare id
      (``<vendor>/<model>``) gets ``@openrouter``, as in ``<vendor>/<model>@openrouter``.
    * ``generate(messages=..., stateful=True)`` replaces the client's history with the full message list
      (the system message included) and appends the assistant reply, which is read back from
      ``client.messages[-1]`` (an OpenAI-shaped dict with ``content`` and ``tool_calls``), as
      :func:`unify.common.single_shot.single_shot_tool_decision` does.
    * Cost: every ``LLMEvent`` the call emits (postprocessing retries emit their own) is caught by a
      context-local hook (``unillm.allm_event_hook_scope``), not a process-global
      ``add_llm_event_listener``, so concurrent calls elsewhere in the process are never billed to the pass.
      The turn's USD is the sum of ``provider_cost``; it is ``unknown`` when no event arrived or any event's
      cost is ``None`` (cache hits, errors), which :class:`WriterPass` then reserves against its cap.
    * Usage: the kept reply carries no ``usage``; unillm reports it only on each event's ``response``. The
      returned message is a copy of the reply with the last such report as its ``usage`` (the call that produced
      the reply), so the compaction trigger and the pass's token counts see the prompt size. The loops take it
      off (:func:`split_usage`) before the message joins the conversation, so no request sends it back.
    * Credentials: unillm reads the provider key from the controller's environment. It never enters argv,
      logs, the box (whose environment is cleared) or anything this module writes.

    Verifying a real call (the first paid pass, under ``PassConfig.usd_limit``): with the key in the
    environment, run one pass on a recorded episode with a small ``max_calls``; check that the ``passes`` row
    holds a decimal USD string and no ``note: N unpriced calls`` reason, that it matches the provider's billed amount for origin
    ``memory.writer``, and that the unillm log shows ``reasoning_effort`` as configured. No test makes this
    call.

    The writer's route (*route*, from ``UNIFY_MEMORY_WRITER_BASE_URL`` and ``UNIFY_MEMORY_WRITER_TOKEN``): ``None``
    sends the call exactly as above. Set, each call goes to ``route.base_url`` with ``route.token`` (through
    :func:`_install_writer_gateway`, for this call's context only) and carries ``X-Unify-Call-Kind: <origin>``;
    the endpoint must be an ``@openrouter`` one (the route replaces OpenRouter's transport). The scoped
    hook, which unillm calls before any process-wide listener, drops ``api_key`` from each event's request,
    so the token never reaches the event bus, and a call whose event shows another ``api_base`` fails the
    turn (its cost then counts as unknown). Before each call, a pure check of unillm's request preparation
    in this context refuses (no call made) unless it sends to the route; the same preparation serves every
    retry. A failed call on the route raises :class:`WriterCallError` naming only the exception's class and a
    fixed category (``http_<status>``, ``timeout``, ``connection``, ``rate_limit``, ``auth``, ``other``; a 403
    whose body is the route's own refusal is :data:`REFUSED_BUSY` or :data:`REFUSED_CAP`), so
    no error text (an echoed header, say) reaches pass notes, error rows or logs. unillm's own copies of that
    text are redacted too: its ``unillm.retry`` warnings by a filter on that logger, and the writer's per-call log
    file (``UNILLM_LOG_DIR``) by a rewrite once unillm finalises it; the route refuses to start while unillm
    records OTel spans (``UNILLM_OTEL``), which would keep the text as is. The pricing is unchanged:
    the request still asks OpenRouter's API for its charged cost, and a call the route does not price is
    ``unknown``.
    """
    import unillm

    from unify.common.llm_client import CALL_KIND_HEADER, SESSION_HEADER, new_llm_client

    endpoint = model if "@" in model else f"{model}@openrouter"
    if route is not None:
        if not endpoint.endswith("@openrouter"):
            raise ValueError(
                "the writer's route replaces the OpenRouter transport; the writer model must be an @openrouter endpoint",
            )
        _install_writer_gateway()
    client = new_llm_client(
        endpoint,
        origin=origin,
        reasoning_effort=effort,
        stateful=True,
    )
    meta = getattr(client, "_unify_request_metadata", None)
    if session and meta is not None:
        meta.session = session  # the request-metadata headers would otherwise send a random session id
    if route is not None and callable(getattr(client, "set_on_log_file", None)):
        # the writer's and helpers' clients only; unify sets no callback of its own
        client.set_on_log_file(_redact_log_file)
    # only on the declared route, so the shipped call is unchanged
    routed: dict[str, Any] = (
        {}
        if route is None
        else {
            "extra_headers": {
                CALL_KIND_HEADER: origin,
                **({SESSION_HEADER: session} if session else {}),
            },
        }
    )

    async def turn(messages: list[dict], tools: list[dict] | None) -> tuple[dict, str]:
        costs: list[float | None] = []
        misrouted: list[bool] = []

        # each call's usage report, from its event's response (the client's kept message carries none)
        usages: list[dict] = []

        def hook(event) -> None:
            if route is not None and isinstance(event.request, dict):
                event.request.pop("api_key", None)
                base = str(event.request.get("api_base") or "").rstrip("/")
                misrouted.append(base != route.base_url)
            if event.origin == origin:
                costs.append(event.provider_cost)
                response = event.response
                usage = (
                    response.get("usage") if isinstance(response, dict) else None
                )
                if isinstance(usage, dict):
                    usages.append(usage)

        failure: tuple[str, str, str] | None = None
        timed_out = False
        async with unillm.allm_event_hook_scope(hook):
            scope = _WRITER_GATEWAY.set(route) if route is not None else None
            try:
                if route is not None:
                    _check_route_in_effect(route)
                await client.generate(
                    messages=copy.deepcopy(messages),
                    # a call given no tool list (None: the judge of behaviour's) sends neither tools nor a tool choice
                    **({"tools": tools, "tool_choice": "auto"} if tools is not None else {}),
                    stateful=True,
                    **routed,
                )
            except Exception as exc:
                if route is None or isinstance(exc, WriterRouteError):
                    raise
                failure = _call_failure(
                    exc,
                )  # class and category; a local failure also says where
                timed_out = isinstance(exc, TimeoutError)
            finally:
                if scope is not None:
                    _WRITER_GATEWAY.reset(scope)
        if failure is not None:
            # raised outside the handler, so the original exception is not chained to it
            name, category, detail = failure
            if (
                timed_out
            ):  # still a timeout to the pass, which treats it as its deadline
                raise TimeoutError("a writer call timed out")
            raise WriterCallError(f"writer call failed: {name} ({category}){detail}", category)
        if any(misrouted):
            raise WriterRouteError(
                "a writer call did not take the writer's declared route",
            )
        # a copy, so the client's own history is never changed; with retries, the last call that reported usage
        # is the one whose reply this is
        msg = dict(client.messages[-1])
        if usages:
            msg["usage"] = copy.deepcopy(usages[-1])
        if not costs or any(c is None for c in costs):
            return msg, _UNKNOWN
        return msg, _usd(sum((Decimal(str(c)) for c in costs), Decimal("0")))

    # the context window the agent loop reads for its compaction trigger, for the same endpoint (None: unknown)
    try:
        turn.max_input_tokens = unillm.get_max_input_tokens(endpoint)  # type: ignore[attr-defined]
    except (
        Exception
    ):  # noqa: BLE001 - an unknown window: no compaction, as in the agent loop
        turn.max_input_tokens = None  # type: ignore[attr-defined]
    return turn
