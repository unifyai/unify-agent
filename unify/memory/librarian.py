"""The librarian: an agent the actor asks which stored items fit (``UNIFY_MEMORY_LIBRARIAN=on``; design J(b)).

``memory.ask()`` in an actor cell computes the recognisers' outcomes on the current request in the actor's own box
(:mod:`.library_helper`'s ``_facts``, used by the ``memory.ask`` section of the generated package, :mod:`.library_ask`)
and sends the question (the request's text when none is given) and those outcomes to the host through
the in-code tool ``ask_library`` (:func:`unify.actor.tool_handlers.ask_library`). The handler opens the request's
:class:`LibrarianSession` on the first question and keeps it for the request's follow-ups; the session ends with the
request (``InCodeTools.close``).

The session keeps the first answer the librarian gives for each distinct ask (:func:`answer_key`: the digest of exactly
what the ask sends it, :func:`sent_to_librarian`: the question with its checked facts, and the library copy's commit and
dispatch file as the ask read them). Asked the same again, it returns
that answer, marked as already given (:data:`ALREADY_GIVEN`), with no model call; a question with other facts is
asked anew. Every ask is answered and none is refused for repeating; the use record counts the asks, the answers
given again and the model calls apart (:func:`usage_fields`).

The session runs on the memory agents' runtime, as a helper does: the shared opening
(:func:`unify.common.tool_registry.preamble`) for the agent ``librarian``, one model-API tool (``execute_code``), its
cells in a persistent confined interpreter (:class:`.tool_handlers.InterpreterSession`) whose ``tools`` module holds
the allow-list (``search``, ``read``, ``grep`` and ``finish``), the memory agents' compaction
(:func:`.writer_pass.compact_by_summary`) and the shared no-progress rule
(:class:`unify.common._async_tool.loop_stop.AgentNoProgress`). Its box mounts the actor's library copy read-only at
``/memory``; it has no write root (a private ``/tmp`` only), no network and an empty environment, as every memory box.
Its system message is the shared opening and the brief (:data:`BRIEF`): a pointer to where the library is and how it
is laid out, the same bytes whatever the library holds, so the prompt prefix stays cached as the library grows. It
finds items with its tools: ``search`` (:func:`search`) ranks every item by its "When to use" text against a query
(:mod:`.library_search`, over the index the export wrote), ``read`` and ``grep`` (:func:`read`, :func:`grep`) read
and search the library's files, inside the library only.

Its model calls go through the actor's route on the actor's model and effort (:func:`.writer_pass.unillm_turn` with
the origin :data:`ORIGIN`): the request's cost listener counts them as the actor's own (``integration.cost``), and
under ``UNIFY_REQUEST_METADATA_HEADERS=on`` each carries ``X-Unify-Call-Kind: librarian`` (:data:`CALL_KIND`). The
session also keeps its own totals (calls, tokens, cache reads, USD) for the use record; an unpriced call keeps the
USD unknown.

It answers with ``tools.finish(answer)`` (:func:`finish`): ``{"fits": [{"id", "call", "why"}, ...]}`` or
``{"fits": [], "nearest": [{"id", "differs"}, ...]}``. The handler checks the form and every id against the items of
the library copy (:func:`library_items`: what ``memory.items`` lists, and the captured procedures); a malformed
answer or an unknown id goes back as a refusal for the librarian to correct, within the loop's existing limits (no
cap of its own). The answer reaches the actor as the librarian gave it: the harness ranks, scores and reorders
nothing.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import time
from dataclasses import dataclass, field
from decimal import Decimal
from pathlib import Path
from collections.abc import Mapping
from typing import Any, Awaitable, Callable

#: The origin of the librarian's model calls (the key proxy refuses a "." in a call kind) and their call kind.
ORIGIN = "memory_librarian"
CALL_KIND = "librarian"
AGENT = "librarian"
#: The library copy as the librarian's box sees it.
LIBRARY_ROOT = "/memory"
#: A librarian cell's time and the head of its output it is shown (the helper's own limits).
CELL_TIMEOUT_S = 60.0
OUTPUT_BYTES = 8000
#: The fact labels ``memory.ask`` sends (:mod:`.library_helper`'s ``READ_OUTCOMES`` and ``APPLIES_OUTCOMES``).
READ_OUTCOMES = ("parsed", "did-not-parse", "not-started", "failed")
APPLIES_OUTCOMES = ("held", "did-not-hold", "raised", "not-run", "none")
CAPTURE_PREFIX = "capture:"
#: The field a reply carries when it is an answer the librarian already gave in this request (``true``); a first
#: answer never has it.
ALREADY_GIVEN = "already_given"
#: What the summary after a compaction says of the files (the memory agents' restart pointer, for this box).
RESTART_POINTER = "Nothing under /memory changed: open an item again with memory.get when you need it."
_HEADING = re.compile(r"^### (\S+)\s*$")


def enabled() -> bool:
    """Whether ``UNIFY_MEMORY_LIBRARIAN=on``."""
    from unify.settings import SETTINGS

    return getattr(SETTINGS, "UNIFY_MEMORY_LIBRARIAN", "off") == "on"


# --- the prompt -----------------------------------------------------------------------------------------------------

BRIEF = (
    "## Your role\n"
    "You are the librarian of a library of stored procedures and notes that an agent built from its earlier work. "
    "The agent is working on a request, and it asks you which stored items apply to a question: its request's "
    "text, or a question of its own. Your job is to find every item that applies and say why. You do not do the "
    "agent's work, and you change nothing.\n"
    "\n"
    "## The library\n"
    "- It is under /memory, read-only. The stored functions are the Python package `memory`: one folder per package "
    "under /memory/memory, each module a file of functions. The notes are Markdown files under /memory/notes, one "
    "folder per topic. Captured procedures, recent work not yet reviewed, are under /memory/.memory/captures, one "
    "folder each with its procedure.py.\n"
    "- Every package and topic folder has a README.md: what the folder holds and how its items relate.\n"
    "- Every stored function's docstring and every note has a When to use section: what the item was saved for, how "
    "to recognise that situation, and what it is not for.\n"
    "- A procedure may have a recipe: one function that does the procedure's whole job on the agent's current "
    "request. The procedure's card in a search result shows the call that applies the recipe.\n"
    "\n"
    "## Your tools\n"
    "- `tools.search` ranks every item of the library by how close its When to use text is to a situation, best "
    "first, with each item's path, id and one-line card. Search first, with the situation the question describes; "
    "search again in other words when the first ranking does not settle it.\n"
    "- `tools.read` reads a file or lists a folder of the library. Before you name an item, read the README of its "
    "folder and the item's When to use section.\n"
    "- `tools.grep` finds where a name or a call occurs in the library's files: which functions call a function, "
    "which notes mention it.\n"
    "- `memory.get` in a cell gives an item's record: its status, its evidence level and how it has been used.\n"
    "- `tools.finish` gives your answer.\n"
    "\n"
    "## What each question brings\n"
    "Each question comes with facts: for each stored procedure that has a reader, whether that reader parsed the "
    "text of the agent's current request, and, where the procedure states conditions, whether its `applies` held on "
    "what the reader gave. A reader that parsed the text, or an `applies` that held, is evidence that the item's "
    "input fits, and evidence only: it does not show that the item does what the question needs. A reader that did "
    "not parse the text is evidence that the input differs; one that could not be started or failed to run is "
    "evidence of nothing. Many requests carry no structured data, and then there are no such facts: the When to use "
    "text decides.\n"
    "\n"
    "## How you answer\n"
    "- Name every item that applies, not only one, each with why it applies: the When to use line, the fact or the "
    "record you relied on.\n"
    "- For each item that applies, give a call the agent can run as written: for a procedure whose card shows the "
    "call that applies its recipe to the current request, that call; for another function, the item called on the "
    "question's input.\n"
    "- When nothing applies, answer that nothing fits, and name the nearest items with what differs between each of "
    "them and the question.\n"
    '- Answer with `tools.finish(answer)`, where answer is `{"fits": [{"id": ..., "call": ..., "why": ...}, ...]}`, '
    'or `{"fits": [], "nearest": [{"id": ..., "differs": ...}, ...]}` when nothing fits. Every id is the id of an '
    "item of the library, as search shows it. An answer in another form, or with an id the library does not hold, "
    "is refused with the reason; correct it and call `tools.finish` again.\n"
    "- The agent may ask follow-up questions in the same request. Each arrives as a new message with its own "
    "facts, and you answer each one with `tools.finish`.\n"
)


def environment_text(cell_s: float = CELL_TIMEOUT_S) -> str:
    """The librarian's box and limits, box paths only."""
    return (
        f"Your box: {LIBRARY_ROOT} (read-only), the agent's copy of the library; a private /tmp is the only "
        "writable path, and there is no network.\n"
        f"Limits: cell time: {cell_s:g} s; cell output shown: the first {OUTPUT_BYTES} bytes."
    )


def opening(cell_s: float = CELL_TIMEOUT_S) -> str:
    """The shared opening of the librarian (:func:`unify.common.tool_registry.preamble`)."""
    from unify.common import tool_registry
    from unify.common.box_interpreter import persistent

    from .agent_profiles import _LIBRARY_LINE

    return tool_registry.preamble(
        AGENT,
        {
            "environment": environment_text(cell_s),
            "memory": _LIBRARY_LINE[AGENT],
            "flags": frozenset(),
            "persistent": persistent(),
        },
    )


def system_prompt(cell_s: float = CELL_TIMEOUT_S) -> str:
    """The librarian's system message: the shared opening and the brief, in that order. It holds nothing of the
    library itself, so it is the same bytes for every library."""
    return opening(cell_s) + "\n\n" + BRIEF


def catalogue_ids(catalogue: str) -> list[str]:
    """Every item id the catalogue holds (its ``### <id>`` headings), in the catalogue's order."""
    out: list[str] = []
    for line in catalogue.splitlines():
        m = _HEADING.match(line)
        if m and m.group(1) not in out:
            out.append(m.group(1))
    return out


#: The captured procedures of a library copy, as its export holds them.
_CAPTURES_DIR = ".memory/captures"
_CAPTURE_DIR = re.compile(r"[0-9a-f]{16}")
#: The statuses ``memory.items`` leaves out (:data:`.library_index.HIDDEN`).
_HIDDEN = ("retired",)


def library_items(copy: Path) -> tuple[list[str], set[str], dict[str, str]] | None:
    """The items of the library copy at *copy*, as the actor's ``memory.items`` lists them (``.memory/items.json``:
    the functions and notes, the retired left out), then its captured procedures in id order; the captured
    procedures that were tried and failed (never offered to run); and, for each procedure with a recipe and for the
    recipe itself, the call that applies the recipe to the current request (:func:`.item_records.recipe_call`). None
    when the copy's items cannot be read."""
    from .item_records import recipe_call

    from .library_helper import ITEMS_FILE

    try:
        data = json.loads((Path(copy) / ITEMS_FILE).read_text(encoding="utf-8"))
    except (OSError, ValueError, UnicodeDecodeError):
        return None
    rows = data.get("items") if isinstance(data, dict) else None
    if not isinstance(rows, dict):
        return None
    ids = [i for i, r in rows.items() if isinstance(i, str) and not (isinstance(r, dict) and r.get("status") in _HIDDEN)]
    recipes = {
        i: recipe_call(r["recipe"])
        for i, r in rows.items()
        if i in ids and isinstance(r, dict) and isinstance(r.get("recipe"), str) and r["recipe"]
    }
    failed: set[str] = set()
    try:
        names = sorted(os.listdir(Path(copy) / _CAPTURES_DIR))
    except OSError:
        names = []
    for name in names:
        if _CAPTURE_DIR.fullmatch(name) is None:
            continue
        ids.append(CAPTURE_PREFIX + name)
        try:
            rec = json.loads((Path(copy) / _CAPTURES_DIR / name / "record.json").read_text(encoding="utf-8"))
        except (OSError, ValueError, UnicodeDecodeError):
            rec = None
        if isinstance(rec, dict) and rec.get("outcome") == "failed":
            failed.add(CAPTURE_PREFIX + name)
    return ids, failed, recipes


# --- the facts ------------------------------------------------------------------------------------------------------


def read_dispatch(copy: Path) -> dict[str, dict]:
    """The procedures of the copy's ``.memory/dispatch.json`` by id (empty when it cannot be read)."""
    from .library_export import DISPATCH_FILE

    try:
        data = json.loads((Path(copy) / DISPATCH_FILE).read_text(encoding="utf-8"))
    except (OSError, ValueError, UnicodeDecodeError):
        return {}
    rows = data.get("procedures") if isinstance(data, dict) else None
    return {
        r["id"]: r
        for r in (rows if isinstance(rows, list) else [])
        if isinstance(r, dict) and isinstance(r.get("id"), str)
    }


def checked_facts(facts: Any, dispatch: dict[str, dict]) -> tuple[list[dict], int]:
    """The facts a cell sent that name a procedure of *dispatch* with its own reader and known outcome labels, in
    the dispatch file's order, and how many were left out. Only ids from the dispatch file and fixed labels pass,
    so nothing else a cell sends reaches the librarian."""
    kept: dict[str, dict] = {}
    dropped = 0
    for f in facts if isinstance(facts, list) else []:
        row = dispatch.get(f.get("id")) if isinstance(f, dict) and isinstance(f.get("id"), str) else None
        if (
            row is None
            or f.get("reader") != row.get("reader")
            or f.get("read") not in READ_OUTCOMES
            or f.get("applies") not in APPLIES_OUTCOMES
            or f["id"] in kept
        ):
            dropped += 1
            continue
        kept[f["id"]] = {"id": f["id"], "reader": row.get("reader"), "read": f["read"], "applies": f["applies"]}
    return [kept[i] for i in dispatch if i in kept], dropped


_APPLIES_TEXT = {
    "held": "its `applies` held on what the reader gave",
    "did-not-hold": "its `applies` did not hold on what the reader gave",
    "raised": "its `applies` raised",
    "not-run": "its `applies` was not run",
    "none": "it states no `applies`",
}


_READ_TEXT = {
    "parsed": "parsed the text",
    "did-not-parse": "did not parse the text",
    "not-started": "could not be started on the text",
    "failed": "failed to run on the text (no outcome)",
}


def facts_text(facts: list[dict], dropped: int = 0) -> str:
    """The facts as the librarian reads them, one line per procedure."""
    if not facts:
        lines = ["- none: no stored reader ran (no stored procedure has one, or the request was not available)"]
    else:
        lines = []
        for f in facts:
            reader = f"reader `{f['reader']}`" if f.get("reader") else "no reader"
            read = _READ_TEXT[f["read"]]
            lines.append(f"- `{f['id']}`: {reader} {read}; {_APPLIES_TEXT[f['applies']]}.")
    if dropped:
        lines.append(f"- ({dropped} fact(s) could not be read and are left out.)")
    return "\n".join(lines)


def sent_to_librarian(question: str, facts: list[dict], dropped: int, library: Mapping[str, Any]) -> dict:
    """Exactly what one ask sends the librarian beyond its fixed opening (the shared opening and the brief, the same
    for every ask and every library): the user message (:func:`question_message`: the question and its checked
    facts) and the library it reads in its box with its tools (*library*: the copy's commit and its dispatch file as
    the ask read it). :func:`answer_key` is this structure's digest, and the message the session sends is its
    ``message``, so nothing the librarian reads can be missing from the key."""
    return {"message": question_message(question, facts, dropped), "library": dict(library)}


def answer_key(sent: Mapping[str, Any]) -> str:
    """The digest of what an ask sent (:func:`sent_to_librarian`): two asks with the same key sent the librarian the
    same, so its first answer answers both."""
    import hashlib

    text = json.dumps(sent, ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha256(text.encode("utf-8", "surrogatepass")).hexdigest()


def _digest_of(value: Any) -> str:
    import hashlib

    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode("utf-8", "replace")).hexdigest()[:16]


def question_message(question: str, facts: list[dict], dropped: int = 0) -> str:
    """The user message that carries one question and its facts."""
    return (
        "Question from the agent:\n"
        f"{question}\n\n"
        "Facts: the stored readers, run on the text of the agent's current request in its own sandbox (outcomes "
        "only):\n"
        f"{facts_text(facts, dropped)}\n\n"
        "Answer with tools.finish(answer)."
    )


# --- the answer -----------------------------------------------------------------------------------------------------


@dataclass
class LibrarianCalls:
    """The librarian's in-code handler state (``CallContext.state``): the library copy its tools read, its items'
    ids (:func:`library_items`), and the answer of the question in progress."""

    ids: list[str]
    failed: set[str] = field(default_factory=set)
    copy: Path | None = None
    #: item id -> the call that applies its recipe to the current request (:func:`library_items`)
    recipes: dict[str, str] = field(default_factory=dict)
    answer: dict | None = None
    finish_text: str | None = None
    finished: bool = False


def _text(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def check_answer(
    answer: Any,
    ids: list[str],
    failed: set[str] = frozenset(),
    recipes: Mapping[str, str] | None = None,
) -> tuple[dict | None, str]:
    """The answer as the librarian gave it, or None with the reason it is refused: the form of design J(b).4, every
    id an item of the library (*ids*), no captured attempt that failed among the fits, and each fit that names a
    procedure with a recipe (or the recipe) carrying the call that applies it to the current request (*recipes*:
    id -> that call), so the agent has the one call. The harness adds, drops and reorders nothing."""
    recipes = recipes or {}
    known = set(ids)
    if not isinstance(answer, dict):
        return None, 'the answer must be an object: {"fits": [...]} or {"fits": [], "nearest": [...]}'
    extra = set(answer) - {"fits", "nearest"}
    if extra:
        return None, f"unknown keys {sorted(extra)}; the answer has `fits` and, when it is empty, `nearest`"
    fits = answer.get("fits")
    if not isinstance(fits, list):
        return None, "`fits` must be a list"
    for i, f in enumerate(fits):
        if not isinstance(f, dict) or set(f) != {"id", "call", "why"} or not all(_text(f[k]) for k in f):
            return None, f"fit {i} must be an object with the non-empty texts `id`, `call` and `why`, and no other key"
        if f["id"] not in known:
            return None, f"fit {i}: {f['id'][:200]!r} is not an item of the library"
        if f["id"] in failed:
            return None, f"fit {i}: {f['id']!r} was tried and failed; it is never offered to run (name it as nearest)"
        entry = recipes.get(f["id"])
        if entry and entry not in f["call"]:
            return None, (
                f"fit {i}: {f['id']!r} has a recipe that does its whole job on the current request; give its call "
                f"as written: {entry}"
            )
    if len({f["id"] for f in fits}) != len(fits):
        return None, "an id is listed twice among the fits"
    if fits:
        if "nearest" in answer:
            return None, "`nearest` is given only when nothing fits"
        return {"fits": [dict(f) for f in fits]}, ""
    nearest = answer.get("nearest")
    if not isinstance(nearest, list):
        return None, "when nothing fits, `nearest` must be a list of {id, differs} (empty when the library is empty)"
    for i, n in enumerate(nearest):
        if not isinstance(n, dict) or set(n) != {"id", "differs"} or not all(_text(n[k]) for k in n):
            return None, f"nearest {i} must be an object with the non-empty texts `id` and `differs`, and no other key"
        if n["id"] not in known:
            return None, f"nearest {i}: {n['id'][:200]!r} is not an item of the library"
    if len({n["id"] for n in nearest}) != len(nearest):
        return None, "an id is listed twice among the nearest"
    return {"fits": [], "nearest": [dict(n) for n in nearest]}, ""


async def finish(ctx: Any, args: dict) -> Any:
    """The librarian's ``tools.finish(answer)``: the answer, checked (:func:`check_answer`), is kept for the
    question in progress; a malformed one is refused with the reason and the question goes on."""
    from unify.common.tool_registry import refusal
    from unify.memory.writer_tools import Reply

    state: LibrarianCalls = ctx.state
    if state.finished:
        text = refusal("finish", "answered")
        return Reply(str(text), "refused", text.form)
    got, why = check_answer(args.get("answer"), state.ids, state.failed, state.recipes)
    if got is None:
        text = refusal("finish", "bad_answer", reason=why[:2000])
        state.finish_text = str(text)
        return Reply(str(text), "refused", text.form)
    state.answer, state.finished = got, True
    state.finish_text = "finished: your answer goes to the agent that asked"
    return Reply(state.finish_text)


# --- finding items: search, read, grep -------------------------------------------------------------------------------


def _offset(args: dict) -> int:
    value = args.get("offset", 0)
    return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else 0


def search(ctx: Any, args: dict) -> Any:
    """The librarian's ``tools.search(query, offset=None)``: every item of the library ranked by its "When to use"
    text against *query* (:func:`.library_search.search_text`, over the copy's index), by meaning with the run's
    embedding backend (each request it sends booked: ``integration.cost.book_embedding``), else by a lexical score,
    and the result says which. Runs in the worker pool."""
    from unify.memory import writer_tools as _wt

    from .integration.cost import book_embedding
    from .library_search import INDEX_FILE, backend, read_index, search_text

    state: LibrarianCalls = ctx.state
    query = args.get("query")
    if not _text(query):
        return _wt.refused("search", "no_query")
    where = ({"path": f"{LIBRARY_ROOT}/{INDEX_FILE}", "mode": "search"},)
    data = read_index(state.copy) if state.copy is not None else None
    return _wt.Reply(search_text(data, str(query), backend(on_call=book_embedding), _offset(args)), paths=where)


def _library_path(tool: str, args: dict, default: str = "") -> tuple[str, str | None, Any]:
    """(the path asked for, its box path, None) inside the library; or a refusal as the third item."""
    from unify.memory import safe_paths as _safe
    from unify.memory import writer_tools as _wt

    vpath = str(args.get("path") or default)
    try:
        prefix, parts = _safe.split(vpath, [LIBRARY_ROOT])
    except _safe.Outside:
        return vpath, None, _wt.refused(tool, "outside_library", path=vpath[:200])
    return vpath, _safe.virtual(prefix, parts), None


def read(ctx: Any, args: dict) -> Any:
    """The librarian's ``tools.read(path, offset=None)``: a page of a file of the library, or a folder's listing;
    a path outside the library is refused and nothing is read. Runs in the worker pool."""
    from unify.memory import safe_paths as _safe
    from unify.memory import views as _views
    from unify.memory import writer_tools as _wt

    state: LibrarianCalls = ctx.state
    vpath, virtual, refusal_reply = _library_path("read", args)
    if refusal_reply is not None:
        return refusal_reply
    where = ({"path": virtual, "mode": "read"},)
    try:
        text, mode = _views.read_entry(vpath, {LIBRARY_ROOT: state.copy}, _offset(args))
    except _safe.Outside:
        return _wt.refused("read", "outside_library", path=vpath[:200])
    except (_safe.LinkFound, _safe.NotRegular) as exc:
        return _wt.refused("read", "unreadable", where, error=exc.reason, path=vpath[:200])
    except OSError:
        return _wt.refused("read", "not_found", where, path=vpath[:200])
    return _wt.Reply(text, paths=({"path": virtual, "mode": mode},))


def grep(ctx: Any, args: dict) -> Any:
    """The librarian's ``tools.grep(pattern, path=None)``: the lines of the library's files under *path* (default
    the whole library) that match *pattern*, one page at a time; a path outside the library is refused and nothing
    is searched. The pattern runs in a child process (:func:`.views.grep_bounded`). Runs in the worker pool."""
    from unify.memory import safe_paths as _safe
    from unify.memory import views as _views
    from unify.memory import writer_tools as _wt

    state: LibrarianCalls = ctx.state
    vpath, virtual, refusal_reply = _library_path("grep", args, LIBRARY_ROOT)
    if refusal_reply is not None:
        return refusal_reply
    where = ({"path": virtual, "mode": "search"},)
    try:
        text = _views.grep_bounded(str(args.get("pattern", "")), vpath, {LIBRARY_ROOT: state.copy}, _offset(args))
    except _views.PatternProblem as exc:
        return _wt.refused("grep", "bad_pattern", where, reason=str(exc))
    except _views.SearchFailed:
        return _wt.refused("grep", "search_failed", where)
    except _safe.Outside:
        return _wt.refused("grep", "outside_library", where, path=vpath[:200])
    except _safe.LinkFound as exc:
        return _wt.refused("grep", "unreadable", where, error=exc.reason, path=vpath[:200])
    except OSError as exc:
        return _wt.refused("grep", "unreadable", where, error=_safe.errno_name(exc), path=vpath[:200])
    return _wt.Reply(text, paths=where)


# --- the session ----------------------------------------------------------------------------------------------------


class NoAnswer(RuntimeError):
    """The librarian ended a question without an accepted answer; the message says why."""


@dataclass
class Usage:
    """The librarian's model calls in this request: counts, tokens (unknown ones counted apart) and USD."""

    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    unreported_calls: int = 0
    usd: Decimal = Decimal(0)
    unpriced_calls: int = 0

    def add(self, usage: dict, usd: Any) -> None:
        from .integration.cost import UNKNOWN, money

        self.calls += 1
        if not isinstance(usage.get("input"), int):
            self.unreported_calls += 1
        for key, attr in (("input", "input_tokens"), ("output", "output_tokens"), ("cache_read", "cache_read_tokens")):
            if isinstance(usage.get(key), int):
                setattr(self, attr, getattr(self, attr) + usage[key])
        amount = money(usd)
        if amount == UNKNOWN:
            self.unpriced_calls += 1
        else:
            self.usd += Decimal(amount)

    def fields(self) -> dict:
        """The use record's ``librarian`` block: decimal strings for money; ``usd`` is ``unknown`` while any call
        is unpriced (``usd_priced`` is the sum of the priced ones)."""
        from unify.common.llm_meter import decimal_string

        return {
            "calls": self.calls,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cache_read_tokens": self.cache_read_tokens,
            "unreported_calls": self.unreported_calls,
            "usd": "unknown" if self.unpriced_calls else decimal_string(self.usd),
            "usd_priced": decimal_string(self.usd),
            "unpriced_calls": self.unpriced_calls,
        }


Turn = Callable[[list, list], Awaitable[tuple[dict, Any]]]


class LibrarianSession:
    """One actor request's librarian (module docstring): opened at the first question, kept for the follow-ups,
    closed with the request. *library* is the library copy's commit (part of what an answer answered)."""

    def __init__(
        self,
        *,
        copy: Path,
        turn: Turn,
        start: Callable[..., Awaitable[Any]] | None = None,
        on_event: Callable[[dict], None] | None = None,
        request_id: str | None = None,
        cell_limit_s: float = CELL_TIMEOUT_S,
        step_guard: int | None = None,
        compact: bool | None = None,
        library: str | None = None,
        run_record: Any = None,
        name: str = AGENT,
    ) -> None:
        from unify.common.tool_channel import HandlerThreads

        from . import tool_handlers as _th
        from .writer_pass import STEP_GUARD

        #: UNIFY_AGENTS_RUN_RECORD=on: the run record this librarian joined (as ``librarian-<hex>``), else None
        self.run_record = run_record
        self.name = name
        self.view: Any = None
        if run_record is not None:
            from unify.agents.run_record import RunRecordView

            self.view = RunRecordView(run_record, name, AGENT)
        #: whether the first question (which carries the common section and the roster lines) was sent
        self._introduced = False

        self.copy = Path(copy)
        self.library = library if isinstance(library, str) and library else None
        self.dispatch = read_dispatch(self.copy)
        ids, failed, recipes = library_items(self.copy) or ([], set(), {})
        self.state = LibrarianCalls(ids=ids, failed=failed, copy=self.copy, recipes=recipes)
        self.turn = turn
        self.cell_limit_s = float(cell_limit_s)
        self.step_guard = STEP_GUARD if step_guard is None else int(step_guard)
        self.compact = _compact_on() if compact is None else bool(compact)
        self.usage = Usage()
        #: every ask this session answered or tried to (the use record's ``questions``), and those answered with an
        #: answer it already gave (``answers_from_cache``)
        self.questions = 0
        self.from_cache = 0
        #: the first accepted answer for each distinct ask (:func:`answer_key`)
        self.answers: dict[str, dict] = {}
        self.closed = False
        self._lock = asyncio.Lock()
        self._cell = 0
        self._on_event = on_event
        self.messages: list[dict] = [{"role": "system", "content": system_prompt(self.cell_limit_s)}]
        from unify.common import tool_registry

        self.tools = tool_registry.json_tools(AGENT)
        self.allow = tool_registry.allow_list(AGENT)
        self.interp = _th.InterpreterSession(
            agent=AGENT,
            name=AGENT,
            role=AGENT,
            pass_id=request_id or "",
            state=self.state,
            allow=self.allow,
            # the actor's library copy, read-only; no writable bind
            binds={"ro": {self.copy: LIBRARY_ROOT}, "rw": {}},
            env={},
            profile=_th.memory_profile(
                pass_id=None,
                cwd=LIBRARY_ROOT,
                sys_path=(LIBRARY_ROOT,),
                fsize_bytes=None,
                audit_roots=(LIBRARY_ROOT,),
                open_areas=(LIBRARY_ROOT,),
                opens_path=None,
                measure=None,
                # the library package, bound in every cell and imported by the first cell that uses it
                globals={"memory": ["lazy_module", "memory"]},
            ),
            roots={LIBRARY_ROOT: self.copy},
            deadline=time.monotonic() + 30 * 86400.0,
            cell_limit_s=self.cell_limit_s,
            on_event=self._event,
            threads=HandlerThreads(),
            start=start,
            record=self.view,
        )

    def _event(self, row: dict) -> None:
        if self._on_event is None:
            return
        try:
            self._on_event({**row, "agent": AGENT})
        except Exception:  # noqa: BLE001 - a measurement never stops the librarian
            pass

    async def ask(self, question: str, facts: Any, request: str | None = None) -> str:
        """One question about the current request (*request* is not sent to the librarian, so it is not in the key):
        its facts checked against the dispatch file. When the librarian was already sent the same
        (:func:`answer_key` of :func:`sent_to_librarian`), that answer as it gave
        it, with :data:`ALREADY_GIVEN`, and no model call; otherwise the question goes as the next user message and
        the librarian's accepted answer is kept and returned. JSON text either way. Raises :class:`NoAnswer` when it
        ends without one (nothing is kept, so the next ask asks again)."""
        if self.closed:
            raise NoAnswer("the request has ended")
        async with self._lock:
            # the library as the ask finds it: a copy that changed since the session opened is asked about afresh
            self.dispatch = read_dispatch(self.copy)
            ids, failed, recipes = library_items(self.copy) or ([], set(), {})
            self.state.ids, self.state.failed, self.state.recipes = ids, failed, recipes
            kept, dropped = checked_facts(facts, self.dispatch)
            self.questions += 1
            library = {"commit": self.library, "dispatch": _digest_of(self.dispatch)}
            sent = sent_to_librarian(question, kept, dropped, library)
            key = answer_key(sent)
            given = self.answers.get(key)
            if given is not None:
                self.from_cache += 1
                return json.dumps({**given, ALREADY_GIVEN: True}, ensure_ascii=False, sort_keys=True)
            self.state.answer, self.state.finished, self.state.finish_text = None, False, None
            content = sent["message"]
            if self.view is not None and not self._introduced:
                # the run record: the common section and this agent's roster lines, once, after the constant brief
                from unify.agents.run_record import prompt_lines

                content = f"{content}\n\n{prompt_lines(self.name, AGENT)}"
                self._introduced = True
            self.messages.append({"role": "user", "content": content})
            answer = await self._turns()
            self.answers[key] = answer
            return json.dumps(answer, ensure_ascii=False, sort_keys=True)

    async def _model_call(self) -> tuple[dict, Any]:
        from unify.common.llm_client import call_kind

        if self.view is not None:
            # the run record's turn boundary: what is new for the librarian, after its turn's tool results
            try:
                block = self.view.take_block()
            except Exception:  # noqa: BLE001 - delivery never stops an answer
                block = None
            if block:
                self.messages.append({"role": "user", "content": block})
        with call_kind(CALL_KIND):
            return await self.turn(self.messages, self.tools)

    async def _compact(self, last_prompt: Any) -> None:
        from unify.common.llm_client import call_kind

        from .writer_pass import compact_by_summary, compaction_due

        if not self.compact or not compaction_due(last_prompt, getattr(self.turn, "max_input_tokens", None)):
            return
        with call_kind(CALL_KIND):
            comp = await compact_by_summary(self.messages, self.turn, self.tools, pointer=RESTART_POINTER)
        from .writer_pass import _usage_of

        reply = comp.reply if isinstance(comp.reply, dict) else {}
        self.usage.add(_usage_of(reply, comp.usd), comp.usd)
        if comp.restart is not None:
            self.messages[:] = comp.restart

    async def _turns(self) -> dict:
        from unify.common._async_tool import loop_stop as _loop_stop

        from .writer_pass import _usage_of, split_usage

        no_progress = _loop_stop.AgentNoProgress(_loop_stop.terminal_tool(AGENT), in_code=True)
        turns = 0
        last_prompt = None
        while turns < self.step_guard:
            await self._compact(last_prompt)
            try:
                msg, usd = await self._model_call()
            except BaseException:
                self.usage.unpriced_calls += 1  # a failed or cut call may still have cost money
                self.usage.calls += 1
                raise
            turns += 1
            if not isinstance(msg, dict):
                self.usage.add({}, usd)
                raise NoAnswer("the model returned no message")
            msg, reported = split_usage(msg)
            usage = _usage_of(msg, usd, reported)
            self.usage.add(usage, usd)
            last_prompt = usage.get("input")
            self.messages.append(msg)
            calls = msg.get("tool_calls") if isinstance(msg.get("tool_calls"), list) else []
            if not calls:
                self.messages.append({"role": "user", "content": "Answer with tools.finish(answer) in a cell."})
                no_progress.after_call(self.messages)
                continue
            for i, tc in enumerate(calls):
                tc = tc if isinstance(tc, dict) else {}
                cid = tc.get("id") if isinstance(tc.get("id"), str) and tc.get("id") else f"call_{turns}_{i}"
                content = await self._tool_call(tc)
                self.messages.append({"role": "tool", "tool_call_id": cid, "content": content})
            if self.state.finished and self.state.answer is not None:
                no_progress.after_call(self.messages, act=False)
                return self.state.answer
            if no_progress.after_call(self.messages) == "end":
                raise NoAnswer(no_progress.note("end"))
        raise NoAnswer("the operational step guard was reached")

    async def _tool_call(self, tc: dict) -> str:
        from unify.common.tool_registry import GENERAL, refusal

        from . import tool_handlers as _th

        fn = tc.get("function") if isinstance(tc.get("function"), dict) else {}
        name = str(fn.get("name", ""))
        if self.state.finished:
            return str(refusal(GENERAL, "task_done"))
        if name != "execute_code":
            return str(refusal(GENERAL, "unknown_tool", name=name[:80], tools="execute_code"))
        try:
            args = json.loads(fn.get("arguments") or "{}")
        except (TypeError, ValueError):
            args = None
        if not isinstance(args, dict):
            return str(refusal(GENERAL, "bad_arguments", reason="execute_code takes strings: thought and code"))
        from unify.common.tool_registry import THOUGHT_ONLY_REPLY, thought_only

        if thought_only(args):
            # a thought alone: nothing runs and nothing is refused; the thought stays in the conversation
            return THOUGHT_ONLY_REPLY
        code = args["code"]
        self.state.finish_text = None
        note = self.interp.take_note()
        res = await self.interp.cell(code, self._cell)
        self._cell += 1
        body, tail = _th.cell_parts(res, self.state.finish_text, self.interp.gone_during(), note)
        data = body.encode("utf-8", "replace")
        if len(data) > OUTPUT_BYTES:
            body = data[:OUTPUT_BYTES].decode("utf-8", "replace") + f"\n[… {len(data)} bytes in all; head shown]"
            from unify.common.box_interpreter import keep_traceback_end

            body = keep_traceback_end(body, getattr(res, "error", None))
        return body + tail

    def box_alive(self) -> bool:
        return self.interp.box_alive()

    async def close(self) -> None:
        """End the session with the request: the interpreter is stopped (verified kill) and its channel closed.
        Safe to repeat."""
        if self.closed:
            return
        self.closed = True
        try:
            async with self._lock:
                await self.interp.close()
        finally:
            if self.run_record is not None:
                # it leaves the run record with the request (its name stays taken)
                try:
                    self.run_record.leave(self.name)
                except Exception:  # noqa: BLE001 - leaving never fails the request's end
                    pass


def _compact_on() -> bool:
    try:
        from unify.settings import SETTINGS

        return str(getattr(SETTINGS, "UNIFY_MEMORY_LIBRARY_COMPACT", "on") or "on") != "off"
    except Exception:  # noqa: BLE001 - no settings: the default
        return True


# --- opening the request's session ------------------------------------------------------------------------------------


class Unavailable(RuntimeError):
    """The librarian cannot be opened for this request; the message says why."""


def actor_turn() -> Turn:
    """A model turn on the actor's route, model and effort (``integration.request.actor_model`` and
    ``actor_effort``), with the librarian's origin."""
    from .integration.request import actor_effort, actor_model
    from .writer_pass import unillm_turn

    return unillm_turn(actor_model(), actor_effort(), origin=ORIGIN)


def opener(
    *,
    on_event: Callable[[dict], None] | None = None,
    request_id: str | None = None,
    turn_factory: Callable[[], Turn] | None = None,
    start: Callable[..., Awaitable[Any]] | None = None,
) -> Callable[[], Awaitable[LibrarianSession]]:
    """What opens this request's session at its first question: the run's library copy
    (``RequestRun.paths.checkout``; refused while no library copy is mounted or its items cannot be read), a turn on
    the actor's model
    (:func:`actor_turn`, or *turn_factory*'s), and the commit the copy was exported at (``RequestRun.pin``). The
    session is registered with the run, whose use record reads its totals."""

    async def open_session() -> LibrarianSession:
        from .integration import request as _request

        run = _request.current()
        copy = getattr(getattr(run, "paths", None), "checkout", None)
        if run is None or copy is None or not getattr(run, "library_mode", False):
            raise Unavailable("no library is mounted in this request")
        if library_items(Path(copy)) is None:
            raise Unavailable("the library's items cannot be read")
        record, name = _join_run_record(run)
        session = LibrarianSession(
            copy=Path(copy),
            turn=(turn_factory or actor_turn)(),
            start=start,
            on_event=on_event,
            request_id=request_id,
            library=getattr(run, "pin", None),
            run_record=record,
            name=name,
        )
        sessions = getattr(run, "librarian_sessions", None)
        if isinstance(sessions, list):
            sessions.append(session)
        return session

    return open_session


def _join_run_record(run: Any) -> tuple[Any, str]:
    """``UNIFY_AGENTS_RUN_RECORD=on``: the run record this request's librarian joins at its first question, as
    ``librarian-<hex of the episode>`` with the request's episode, and its name; ``(None, "librarian")`` while off."""
    from unify.common.tool_registry import run_record_on

    if not run_record_on():
        return None, AGENT
    from unify.agents.run_record import join_run_record

    home = getattr(getattr(run, "paths", None), "home", None)
    if home is None:
        raise Unavailable("no memory home for the run record")
    episode = getattr(run, "episode_id", None) or None
    record = join_run_record(home)
    import uuid

    naming = episode or uuid.uuid4().hex[:16]
    for _ in range(20):
        name = record.unique_name(AGENT, naming)
        try:
            record.join(name, AGENT, episode=episode)
            return record, name
        except Exception:  # noqa: BLE001 - a name another process took: the next one
            continue
    raise Unavailable("no free name for the librarian on the run record")


def usage_fields(sessions: Any) -> dict:
    """The use record's ``librarian`` block over a request's sessions (:meth:`Usage.fields`): ``calls`` the model
    calls, ``questions`` every ask, and ``answers_from_cache`` the asks answered with an answer already given."""
    total = Usage()
    for s in sessions if isinstance(sessions, list) else []:
        u = getattr(s, "usage", None)
        if not isinstance(u, Usage):
            continue
        total.calls += u.calls
        total.input_tokens += u.input_tokens
        total.output_tokens += u.output_tokens
        total.cache_read_tokens += u.cache_read_tokens
        total.unreported_calls += u.unreported_calls
        total.usd += u.usd
        total.unpriced_calls += u.unpriced_calls
    out = total.fields()
    rows = sessions if isinstance(sessions, list) else []
    out["questions"] = sum(int(getattr(s, "questions", 0) or 0) for s in rows)
    out["answers_from_cache"] = sum(int(getattr(s, "from_cache", 0) or 0) for s in rows)
    return out


__all__ = [
    "ALREADY_GIVEN",
    "BRIEF",
    "CALL_KIND",
    "LibrarianCalls",
    "LibrarianSession",
    "NoAnswer",
    "ORIGIN",
    "Unavailable",
    "Usage",
    "answer_key",
    "sent_to_librarian",
    "check_answer",
    "checked_facts",
    "enabled",
    "finish",
    "grep",
    "library_items",
    "opener",
    "read",
    "search",
    "system_prompt",
    "usage_fields",
]
