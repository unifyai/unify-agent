"""The librarian: an agent the actor asks which stored items fit (design J(b)).

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
each carries ``X-Unify-Call-Kind: librarian`` (:data:`CALL_KIND`). The
session also keeps its own totals (calls, tokens, cache reads, USD) for the use record; an unpriced call keeps the
USD unknown.

Rev 2 (PLAN-REV2 item 1): the librarian is a role of the run record (``run_record.INSTANCE_ROLES``). Its state lives as
long as the run, under ``memory-state/agents/roles/librarian/``: the answers its instances gave (an exact-match cache:
:func:`answer_key` names the library's commit, its dispatch digest and the schema's digest, so a changed library never
serves an old answer) and the notes they kept; every instance reads it at its start. An instance starts in one of two
ways (:data:`MODES`): ``memory.ask``'s (:class:`AskInstance`: one per request, kept for its follow-ups as before, whose
asks also go on the record) or one per message to ``@librarian`` (:func:`answer_message`, started by the request's role
dispatcher, ``integration.request.role_dispatcher``), which posts one reply and ends.

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
CAPTURE_PREFIX = "capture:"
#: The field a reply carries when it is an answer the librarian already gave in this request (``true``); a first
#: answer never has it.
ALREADY_GIVEN = "already_given"
#: A structured answer (``memory.ask(..., schema=...)``): the reply's value, and its form when the value did not match
#: the schema after :data:`SCHEMA_REPAIRS` repair turn(s).
SCHEMA_VALUE = "value"
SCHEMA_ERROR = "schema_error"
SCHEMA_REPAIRS = 1
#: What the summary after a compaction says of the files (the memory agents' restart pointer, for this box).
RESTART_POINTER = "Nothing under /memory changed: open an item again with memory.get when you need it."
def enabled() -> bool:
    """Whether the run has the librarian: whenever it has memory (``UNIFY_MEMORY`` is ``on`` or ``in_episode``)."""
    from unify.memory.integration.switch import memory_on
    from unify.settings import SETTINGS

    return memory_on(SETTINGS)


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
    "folder per topic. Recorded procedures (`capture:<id>`), each re-running work an earlier request accepted, are "
    "under /memory/.memory/captures, one "
    "folder each with its procedure.py.\n"
    "- Every package and topic folder has a README.md: what the folder holds and how its items relate.\n"
    "- Every stored function's docstring and every note has a When to use section: what the item was saved for, how "
    "to recognise that situation, and what it is not for.\n"
    "- A procedure may have a recipe: one function that applies it to the agent's current request where one call "
    "can do the job. Work of several steps, or a conversation, uses the procedure's functions and notes step by "
    "step. The procedure's card in a search result shows the call that applies the recipe.\n"
    "\n"
    "## Your tools\n"
    "- `tools.search` lists every item of the library with its path, id and one-line card, ranked by how close its "
    "When to use text is to a situation when the library has an embedding index, else in the library's order (the "
    "result says which). Search first, with the situation the question describes; read the cards, and search again "
    "in other words when a ranking does not settle it.\n"
    "- `tools.read` reads a file or lists a folder of the library. Before you name an item, read the README of its "
    "folder and the item's When to use section.\n"
    "- `tools.grep` finds where a name or a call occurs in the library's files: which functions call a function, "
    "which notes mention it.\n"
    "- `memory.get` in a cell gives an item's record: its status, its evidence level and how it has been used.\n"
    "- `tools.keep_note` keeps one short note for the later librarians of this run: something about the library "
    "that would save them time, never a request's details.\n"
    "- `tools.finish` gives your answer.\n"
    "\n"
    "## What each question brings\n"
    "Each question names the catalogue: /memory/.memory/dispatch.json, one row per stored procedure, made by the "
    "harness from each procedure's code and records, never by running it: its form, its input line, its `applies` "
    "as text, the tool calls its code names (`effects`), whether it reads its input, whether it gives a fixed "
    "answer (`constant`: an earlier answer kept as code, to read, never to call), and for a captured procedure how "
    "often it occurred and with what outcomes. Judge relevance from these rows and the items' text: open any item "
    "with your read tools. Nothing has run on the agent's request: a row is evidence only of what an item's code "
    "does, never that the item fits the request.\n"
    "\n"
    "## How you answer\n"
    "- Rank by fit to the question: an item's kind (a curated item, a note or a recorded procedure) is never a "
    "reason to prefer it or to pass it over.\n"
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


#: The brief's paragraph on what each question brings (rev 2: the structural catalogue; no reader runs).
_BRINGS = (
    "## What each question brings\n"
    "Each question names the catalogue: /memory/.memory/dispatch.json, one row per stored procedure, made by the "
    "harness from each procedure's code and records, never by running it: its form, its input line, its `applies` "
    "as text, the tool calls its code names (`effects`), whether it reads its input, whether it gives a fixed "
    "answer (`constant`: an earlier answer kept as code, to read, never to call), and for a captured procedure how "
    "often it occurred and with what outcomes. Judge relevance from these rows and the items' text: open any item "
    "with your read tools. Nothing has run on the agent's request: a row is evidence only of what an item's code "
    "does, never that the item fits the request.\n"
)
#: The same paragraph with the harness's evidence serving on: the situation
#: and each item's failure record.
_BRINGS_EVIDENCE = (
    "## What each question brings\n"
    "Each question comes with the situation: the agent's question (its own words, or its request's text) and, when "
    "a message has come after its request, the latest message it received. It names the catalogue: "
    "/memory/.memory/dispatch.json, one row per stored procedure, made by the harness from each procedure's code and "
    "records, never by running it: its form, its input line, its `applies` as text, the tool calls its code names "
    "(`effects`), whether it reads its input, whether it gives a fixed answer (`constant`: an earlier answer kept as "
    "code, to read, never to call), and for a captured procedure how often it occurred and with what outcomes. Judge "
    "relevance from these rows and the items' text: open any item with your read tools. Nothing has run on the "
    "agent's request: a row is evidence only of what an item's code does, never that the item fits the request. "
    "Every card in a search result and in the "
    "catalogue states what its status rests on and its recorded failures (a use that raised, whose stated result "
    "condition was false, that a contrary signal followed, or that an agent reported): weigh them before you name an "
    "item, and say how in why.\n"
)
if BRIEF.count(_BRINGS) != 1:
    raise RuntimeError("the librarian brief no longer holds the paragraph its evidence form replaces")
#: The brief with the harness's evidence serving on.
BRIEF_EVIDENCE = BRIEF.replace(_BRINGS, _BRINGS_EVIDENCE)


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

    from .agent_profiles import _LIBRARY_LINE

    return tool_registry.preamble(
        AGENT,
        {
            "environment": environment_text(cell_s),
            "memory": _LIBRARY_LINE[AGENT],
            "flags": frozenset(),
        },
    )


def system_prompt(cell_s: float = CELL_TIMEOUT_S, evidence: bool = False) -> str:
    """The librarian's system message: the shared opening and the brief, in that order (with *evidence*, the
    harness's evidence serving on, :data:`BRIEF_EVIDENCE`). It holds nothing of the library itself, so it is the
    same bytes for every library."""
    return opening(cell_s) + "\n\n" + (BRIEF_EVIDENCE if evidence else BRIEF)


def read_evidence(copy: Path) -> dict | None:
    """The copy's item evidence (:data:`..serving.EVIDENCE_FILE`) by item id, or None when the harness serves none."""
    from .serving import EVIDENCE_FILE

    try:
        data = json.loads((Path(copy) / EVIDENCE_FILE).read_text(encoding="utf-8"))
    except (OSError, ValueError, UnicodeDecodeError):
        return None
    rows = data.get("items") if isinstance(data, dict) else None
    return rows if isinstance(rows, dict) else None


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
        if isinstance(rec, dict) and rec.get("outcome") in ("failed", "contrary"):
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


def catalogue_line(rows: int) -> str:
    """The line that names the structural catalogue (:data:`.library_export.DISPATCH_FILE`) for one question."""
    return (
        f"The catalogue: {LIBRARY_ROOT}/.memory/dispatch.json, {rows} row(s), one per stored procedure, from its code "
        "and records (nothing was run on the request)."
    )


def evidence_question_message(question: str, last_message: str | None, rows: int = 0) -> str:
    """The user message that carries one question with the harness's evidence serving on: the question, the
    situation (the latest message the agent received after its request, when there is one) and the catalogue."""
    last = last_message.strip() if isinstance(last_message, str) else ""
    situation = (
        f"- The latest message the agent received after its request:\n{last}"
        if last
        else "- No message has come after the agent's request."
    )
    return (
        "Question from the agent:\n"
        f"{question}\n\n"
        "The situation:\n"
        f"{situation}\n\n"
        f"{catalogue_line(rows)}\n\n"
        "Answer with tools.finish(answer)."
    )


def sent_to_librarian(
    question: str,
    library: Mapping[str, Any],
    *,
    evidence: Mapping[str, Any] | None = None,
    last_message: str | None = None,
    rows: int = 0,
    schema: dict | None = None,
) -> dict:
    """Exactly what one ask sends the librarian beyond its fixed opening (the shared opening and the brief, the same
    for every ask and every library): the user message (:func:`question_message`: the question and the catalogue
    line) and the library it reads in its box with its tools (*library*: the copy's commit and its catalogue's
    digest as the ask read it). The question and the situation's message are sent as the asker wrote them,
    identifiers included (rev 2.1: the librarian sees the same identifiers the library's items carry, so it can tell
    an item keyed on another request's identifier from a fit). :func:`answer_key` is this structure's digest, and
    the message the session sends is its ``message``, so nothing the librarian reads can be missing from the key. With *evidence* (the copy's item
    evidence: the harness's evidence serving on) the message is :func:`evidence_question_message`, with
    *last_message* as the situation's latest message."""
    q = question if isinstance(question, str) else ""
    if evidence is not None:
        situation = last_message if isinstance(last_message, str) and last_message else None
        message = evidence_question_message(q, situation, rows)
    else:
        message = question_message(q, rows)
    if schema is None:
        return {"message": message, "library": dict(library)}
    # a structured answer: the schema replaces the answer's form for this question (its digest is in the key)
    message = message.rsplit("Answer with tools.finish(answer).", 1)[0] + schema_instruction(schema)
    return {"message": message, "library": dict(library), "schema": _digest_of(schema)}


def schema_instruction(schema: dict) -> str:
    """The closing lines of a question with a schema: the answer is one JSON value matching it, not the fits form."""
    return (
        "For this question the answer is not the fits form: answer with tools.finish(answer), where answer is one "
        "JSON value that matches this JSON Schema. Base it on the library's items, as for any question. A value "
        "that does not match is refused with where it does not.\n"
        + json.dumps(schema, ensure_ascii=False, sort_keys=True)
    )


def answer_key(sent: Mapping[str, Any]) -> str:
    """The digest of what an ask sent (:func:`sent_to_librarian`): two asks with the same key sent the librarian the
    same, so its first answer answers both."""
    import hashlib

    text = json.dumps(sent, ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha256(text.encode("utf-8", "surrogatepass")).hexdigest()


def _digest_of(value: Any) -> str:
    import hashlib

    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode("utf-8", "replace")).hexdigest()[:16]


def question_message(question: str, rows: int = 0) -> str:
    """The user message that carries one question and names the catalogue."""
    return (
        "Question from the agent:\n"
        f"{question}\n\n"
        f"{catalogue_line(rows)}\n\n"
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
    #: the JSON Schema of the question in progress (``memory.ask(..., schema=...)``), and its answers that missed it
    schema: dict | None = None
    schema_failures: int = 0
    #: rev 2: the run's memory home when the run has the librarian role (its notes are kept there), and this
    #: instance's name on the record (:func:`keep_note`); None and "" without the role
    role_home: Path | None = None
    instance: str = ""


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
                f"fit {i}: {f['id']!r} has a recipe, the one call that applies it to the current request; give "
                f"that call as written: {entry}"
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


def check_schema(schema: Any) -> None:
    """Raise ValueError when *schema* is not a JSON Schema the checker can use (Draft 2020-12)."""
    from jsonschema import Draft202012Validator
    from jsonschema.exceptions import SchemaError

    if not isinstance(schema, dict):
        raise ValueError("a schema is a JSON object")
    try:
        Draft202012Validator.check_schema(schema)
    except SchemaError as exc:
        raise ValueError(f"not a JSON Schema: {exc.message[:300]}") from None


def schema_errors(value: Any, schema: dict) -> list[list[str]]:
    """Where *value* misses *schema*: ``[path, message]`` pairs, the path a JSON pointer (``""`` for the whole value),
    in path order; [] when it matches."""
    from jsonschema import Draft202012Validator

    out = []
    for err in Draft202012Validator(schema).iter_errors(value):
        path = "".join(f"/{p}" for p in err.absolute_path)
        out.append([path, str(err.message)[:300]])
    return sorted(out)


def _finish_schema(state: "LibrarianCalls", answer: Any) -> Any:
    """``finish`` for a question with a schema: a value that matches it is the answer; one that misses it goes back
    with the checker's errors (never the request) once, and a second miss ends the question with them."""
    from unify.common.tool_registry import refusal
    from unify.memory.writer_tools import Reply

    errors = schema_errors(answer, state.schema or {})
    if not errors:
        state.answer, state.finished = {SCHEMA_VALUE: answer, "repairs": state.schema_failures}, True
        state.finish_text = "finished: your answer goes to the agent that asked"
        return Reply(state.finish_text)
    state.schema_failures += 1
    if state.schema_failures > SCHEMA_REPAIRS:
        raw = json.dumps(answer, ensure_ascii=False, default=str)[:20000]
        state.answer = {SCHEMA_ERROR: {"raw": raw, "errors": errors}}
        state.finished = True
        state.finish_text = "finished: the answer does not match the schema; the agent is told where"
        return Reply(state.finish_text)
    shown = "; ".join(f"{p or '/'}: {m}" for p, m in errors[:20])
    text = refusal("finish", "bad_answer", reason=f"the answer does not match the schema: {shown}"[:2000])
    state.finish_text = str(text)
    return Reply(str(text), "refused", text.form)


async def finish(ctx: Any, args: dict) -> Any:
    """The librarian's ``tools.finish(answer)``: the answer, checked (:func:`check_answer`, or against the question's
    schema, :func:`_finish_schema`), is kept for the question in progress; a malformed one is refused with the reason
    and the question goes on."""
    from unify.common.tool_registry import refusal
    from unify.memory.writer_tools import Reply

    state: LibrarianCalls = ctx.state
    if state.finished:
        text = refusal("finish", "answered")
        return Reply(str(text), "refused", text.form)
    if state.schema is not None:
        return _finish_schema(state, args.get("answer"))
    got, why = check_answer(args.get("answer"), state.ids, state.failed, state.recipes)
    if got is None:
        text = refusal("finish", "bad_answer", reason=why[:2000])
        state.finish_text = str(text)
        return Reply(str(text), "refused", text.form)
    state.answer, state.finished = got, True
    state.finish_text = "finished: your answer goes to the agent that asked"
    return Reply(state.finish_text)


#: Rev 2: one kept note's characters at most, and the role's notes file's bytes at most (a full file refuses more).
NOTE_CHARS = 1000
NOTES_MAX_BYTES = 256 * 1024


def keep_note(ctx: Any, args: dict) -> Any:
    """The librarian's ``tools.keep_note(text)``: append one note for the later instances of the run's librarian role
    to ``memory-state/agents/roles/librarian/notes.md`` in the run's memory home (``run_record.role_dir``), and nothing
    else: append-only, one line, at most :data:`NOTE_CHARS` characters, as written, refused when it holds a
    key-shaped string, when the run has no librarian
    role, or when the notes are full (:data:`NOTES_MAX_BYTES`)."""
    import fcntl
    import os

    from unify.agents import run_record as _rr
    from unify.memory import writer_tools as _wt

    from .redact import KEY_SHAPED

    state = ctx.state
    text = args.get("text")
    if not _text(text):
        return _wt.refused("keep_note", "no_text")
    home = getattr(state, "role_home", None)
    if home is None or not _rr.role_seeded(home, AGENT):
        return _wt.refused("keep_note", "no_role")
    if KEY_SHAPED.search(str(text)):
        return _wt.refused("keep_note", "key_shaped")
    line = " ".join(str(text).split())[:NOTE_CHARS]
    row = f"- {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} {getattr(state, 'instance', '') or AGENT}: {line}\n"
    folder = _rr.role_dir(home, AGENT)
    path = folder / _rr.ROLE_NOTES
    lock = os.open(folder / (_rr.ROLE_NOTES + ".lock"), os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW | os.O_CLOEXEC, 0o600)
    try:
        fcntl.flock(lock, fcntl.LOCK_EX)
        fd = os.open(path, os.O_WRONLY | os.O_APPEND | os.O_CREAT | os.O_NOFOLLOW | os.O_CLOEXEC, 0o600)
        try:
            data = row.encode("utf-8")
            if os.fstat(fd).st_size + len(data) > NOTES_MAX_BYTES:
                return _wt.refused("keep_note", "full", limit=NOTES_MAX_BYTES)
            os.write(fd, data)
            os.fsync(fd)
        finally:
            os.close(fd)
    finally:
        os.close(lock)
    return _wt.Reply(f"kept: a note of {len(line)} characters for the later librarians of this run")


# --- finding items: search, read, grep -------------------------------------------------------------------------------


def _offset(args: dict) -> int:
    value = args.get("offset", 0)
    return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else 0


def search(ctx: Any, args: dict) -> Any:
    """The librarian's ``tools.search(query, offset=None)``: every item of the library ranked by its "When to use"
    text against *query* (:func:`.library_search.search_text`, over the copy's index), by meaning with the run's
    embedding backend (each request it sends booked: ``integration.cost.book_embedding``), else not ranked (every
    item listed in the library's order), and the result says which. Runs in the worker pool."""
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
        library: str | None = None,
        run_record: Any = None,
        name: str = AGENT,
    ) -> None:
        from unify.common.tool_channel import HandlerThreads

        from . import tool_handlers as _th
        from .writer_pass import STEP_GUARD

        #: the run record this librarian joined (as ``librarian-<hex>``), else None
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
        #: the copy's item evidence (the harness's evidence serving on), read again at each ask; None: as built
        self.evidence = read_evidence(self.copy)
        ids, failed, recipes = library_items(self.copy) or ([], set(), {})
        self.state = LibrarianCalls(ids=ids, failed=failed, copy=self.copy, recipes=recipes)
        self.turn = turn
        self.cell_limit_s = float(cell_limit_s)
        self.step_guard = STEP_GUARD if step_guard is None else int(step_guard)
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
        self.messages: list[dict] = [
            {"role": "system", "content": system_prompt(self.cell_limit_s, evidence=self.evidence is not None)},
        ]
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

    async def ask(
        self,
        question: str,
        facts: Any = None,
        request: str | None = None,
        last_message: str | None = None,
        schema: dict | None = None,
    ) -> str:
        """One question about the current request, as JSON text (:meth:`answer`; *request* is not sent to the
        librarian, so it is not in the key; *facts*, which an older cell may still send, are not read: rev 2 runs no
        reader)."""
        got = await self.answer(question, schema=schema, last_message=last_message)
        return json.dumps(got, ensure_ascii=False, sort_keys=True)

    async def answer(self, question: str, *, schema: dict | None = None, last_message: str | None = None) -> dict:
        """One question about the current request. When the librarian was already sent the same
        (:func:`answer_key` of :func:`sent_to_librarian`), that answer as it gave it, with :data:`ALREADY_GIVEN`, and
        no model call; otherwise the question goes as the next user message and the librarian's accepted answer is
        kept and returned. With *schema* (a JSON Schema, :func:`check_schema`; ValueError otherwise) the answer is
        ``{"value": ..., "repairs": n}``, or ``{"schema_error": {"raw", "errors"}}`` when the value missed it after
        :data:`SCHEMA_REPAIRS` repair (never kept). Raises :class:`NoAnswer` when it ends without one (nothing is
        kept, so the next ask asks again)."""
        if self.closed:
            raise NoAnswer("the request has ended")
        if schema is not None:
            check_schema(schema)
        async with self._lock:
            # the library as the ask finds it: a copy that changed since the session opened is asked about afresh
            self.dispatch = read_dispatch(self.copy)
            ids, failed, recipes = library_items(self.copy) or ([], set(), {})
            self.state.ids, self.state.failed, self.state.recipes = ids, failed, recipes
            self.questions += 1
            library = {"commit": self.library, "dispatch": _digest_of(self.dispatch)}
            if self.evidence is not None:
                # the evidence as the ask finds it, and the situation's latest message: what the answer answered
                self.evidence = read_evidence(self.copy) or {}
                sent = sent_to_librarian(question, library, evidence=self.evidence, last_message=last_message,
                                         rows=len(self.dispatch), schema=schema)  # fmt: skip
            else:
                sent = sent_to_librarian(question, library, rows=len(self.dispatch), schema=schema)
            key = answer_key(sent)
            given = self.answers.get(key)
            if given is not None:
                self.from_cache += 1
                return {**given, ALREADY_GIVEN: True}
            self.state.answer, self.state.finished, self.state.finish_text = None, False, None
            self.state.schema, self.state.schema_failures = schema, 0
            content = sent["message"]
            if self.view is not None and not self._introduced:
                # the run record: the common section and this agent's roster lines, once, after the constant brief
                from unify.agents.run_record import prompt_lines

                content = f"{content}\n\n{prompt_lines(self.name, AGENT)}"
                self._introduced = True
            self.messages.append({"role": "user", "content": content})
            try:
                answer = await self._turns()
            finally:
                failures, self.state.schema = self.state.schema_failures, None
            if schema is not None:
                # what the run keeps of a schema: its digest only
                self._event({"event": "ask_schema", "schema": sent.get("schema"), "repairs": failures,
                             "ok": SCHEMA_ERROR not in answer})  # fmt: skip
            if SCHEMA_ERROR not in answer:
                self.answers[key] = answer
            return answer

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

        if not compaction_due(last_prompt, getattr(self.turn, "max_input_tokens", None)):
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


# --- opening the request's session ------------------------------------------------------------------------------------


class Unavailable(RuntimeError):
    """The librarian cannot be opened for this request; the message says why."""


def actor_turn() -> Turn:
    """A model turn on the actor's route, model and effort (``integration.request.actor_model`` and
    ``actor_effort``), with the librarian's origin."""
    from .integration.request import actor_effort, actor_model
    from .writer_pass import unillm_turn

    return unillm_turn(actor_model(), actor_effort(), origin=ORIGIN)


#: Rev 2 (PLAN-REV2 item 1): the two ways an instance of the librarian role starts. ``ask``: ``memory.ask``'s instance, the
#: synchronous client of the role (one per request, kept for the request's follow-ups, as before); ``message``: one
#: instance per message to ``@librarian`` on the run record (:func:`answer_message`), which answers it and ends.
MODES = ("ask", "message")
#: What an instance is shown first of the notes earlier instances of the role kept (only when there are some).
NOTES_HEAD = "Notes earlier librarians of this run kept (read them; they may be out of date):"


class AskInstance(LibrarianSession):
    """Rev 2: ``memory.ask``'s instance of the librarian role. Answers as :class:`LibrarianSession` does (its answer is
    the tool's result, as before); each ask is also put on the run record, so the record's readers see asks and
    answers the same way as messages: one harness entry naming the asker and the question, and this instance's
    ``reply`` to it. Neither mentions anyone (the asker already has the answer as its tool result). The answer goes on
    the record without the schema (only the value); the role's answer cache keeps it (:func:`keep_answers`)."""

    async def ask(self, question: str, facts: Any = None, request: str | None = None, last_message: str | None = None,
                  schema: dict | None = None) -> str:  # fmt: skip
        text = await super().ask(question, facts, request, last_message, schema)
        keep_answers(self)
        try:
            self._post_ask(question, text)
        except Exception:  # noqa: BLE001 - the record never costs the ask its answer
            pass
        return text

    def _post_ask(self, question: str, text: str) -> None:
        if self.run_record is None:
            return
        from unify.agents.record import HARNESS

        rec = self.run_record
        asker = next((r["name"] for r in rec.roster_rows(include_ended=True)
                      if r.get("role") == "actor" and getattr(self, "episode", None) and r.get("episode") == self.episode), None)  # fmt: skip
        ask, _ = rec.append(HARNESS, f"{asker or 'the actor'} asked the librarian (memory.ask): {question}", mentions=[])
        try:
            got = json.loads(text)
        except ValueError:
            got = {"text": text}
        rec.append(self.name, reply_text(ask.seq, got), kind="reply", mentions=[], reply_to=ask.seq)


def reply_text(seq: int, got: dict) -> str:
    """The text of an instance's reply to entry *seq*: the answer as JSON, cut to the record's entry limit."""
    from unify.agents.entry import MAX_TEXT_BYTES

    body = json.dumps(got, ensure_ascii=False, sort_keys=True, default=str)
    text = f"answer to #{seq}: {body}"
    room = MAX_TEXT_BYTES - 256
    data = text.encode("utf-8")
    if len(data) > room:
        text = data[:room].decode("utf-8", "ignore") + f" [... cut at {room} of {len(data)} bytes]"
    return text


def keep_answers(session: LibrarianSession) -> int:
    """Rev 2: keep in the role's state the answers *session* gave that the role did not have yet (its exact-match cache:
    the key is :func:`answer_key` of exactly what was asked, which names the library's commit and dispatch digest and
    the schema's digest, so a changed library or schema never matches). Returns how many were kept."""
    home = getattr(session, "role_home", None)
    if home is None:
        return 0
    from unify.agents import run_record as _rr

    known = session.__dict__.setdefault("role_keys", set())
    kept = 0
    for key, answer in list(session.answers.items()):
        if key in known:
            continue
        _rr.keep_role_answer(home, AGENT, key, answer, by=session.name, library=session.library)
        known.add(key)
        kept += 1
    return kept


def _role_state(session: LibrarianSession, home: Any) -> None:
    """Rev 2: every instance reads the role's state at its start: the answers earlier instances kept (exact-match cache:
    :meth:`LibrarianSession.answer` returns one for exactly the same ask) and the notes they kept."""
    from unify.agents import run_record as _rr

    if home is None or not _rr.role_seeded(home, AGENT):
        return
    session.role_home = home
    # the notes tool appends to the role's notes in this home, as this instance (keep_note); a session with no tool
    # state (one that takes no tool calls) keeps no notes
    state = getattr(session, "state", None)
    if state is not None:
        state.role_home, state.instance = Path(home), str(getattr(session, "name", "") or AGENT)
    keys = set()
    for row in _rr.role_answers(home, AGENT):
        session.answers.setdefault(row["key"], row["answer"])
        keys.add(row["key"])
    session.role_keys = keys
    notes = _rr.role_notes(home, AGENT).strip()
    if notes:
        session.messages.append({"role": "user", "content": f"{NOTES_HEAD}\n{notes}"})


def opener(
    *,
    on_event: Callable[[dict], None] | None = None,
    request_id: str | None = None,
    turn_factory: Callable[[], Turn] | None = None,
    start: Callable[..., Awaitable[Any]] | None = None,
) -> Callable[..., Awaitable[LibrarianSession]]:
    """What opens an instance of the librarian: the run's library copy (``RequestRun.copy_root``; refused while no
    library copy is mounted or its items cannot be read), a turn on the actor's model (:func:`actor_turn`, or
    *turn_factory*'s), and the commit the copy was exported at (``RequestRun.pin``). The session is registered with the
    run, whose use record reads its totals. Rev 2: ``mode`` (:data:`MODES`): ``ask`` (the default: ``memory.ask``'s
    instance, :class:`AskInstance`) or ``message`` (one per message to ``@librarian``); either reads the role's state at
    its start (:func:`_role_state`)."""

    async def open_session(*, mode: str = "ask") -> LibrarianSession:
        from .integration import request as _request

        if mode not in MODES:
            raise ValueError(f"mode is one of {', '.join(MODES)}")
        run = _request.current()
        # the copy as cells read it (UNIFY_MEMORY=in_episode: its link to the current pin), else the export itself
        copy = getattr(run, "copy_root", None) or getattr(getattr(run, "paths", None), "checkout", None)
        if run is None or copy is None:
            raise Unavailable("no library is mounted in this request")
        if library_items(Path(copy)) is None:
            raise Unavailable("the library's items cannot be read")
        record, name = _join_run_record(run, mode=mode)
        session = (AskInstance if mode == "ask" else LibrarianSession)(
            copy=Path(copy),
            turn=(turn_factory or actor_turn)(),
            start=start,
            on_event=on_event,
            request_id=request_id,
            library=getattr(getattr(run, "in_episode", None), "current", None) or getattr(run, "pin", None),
            run_record=record,
            name=name,
        )
        session.mode = mode
        session.episode = (getattr(run, "episode_id", None) or None) if mode == "ask" else None
        if record is not None:
            _role_state(session, getattr(getattr(run, "paths", None), "home", None))
        sessions = getattr(run, "librarian_sessions", None)
        if isinstance(sessions, list):
            sessions.append(session)
        return session

    return open_session


def _join_run_record(run: Any, *, mode: str = "ask") -> tuple[Any, str]:
    """The run record an instance of the librarian joins, and its name. ``ask``: ``librarian-<hex of the episode>``
    with the request's episode (it lives as long as the request); ``message``: ``librarian-<random hex>`` with no
    episode (it lives until its reply)."""
    from unify.agents.run_record import join_run_record

    home = getattr(getattr(run, "paths", None), "home", None)
    if home is None:
        raise Unavailable("no memory home for the run record")
    episode = (getattr(run, "episode_id", None) or None) if mode == "ask" else None
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


def question_of(text: str) -> str:
    """A message to the role as the question its instance is asked: the text without the role's own address."""
    q = re.sub(r"(?<![\w@.])@librarian\b", "", str(text or ""), flags=re.IGNORECASE).strip()
    return q or str(text or "").strip()


async def answer_message(entry: Any, *, record: Any, open_session: Callable[..., Awaitable[LibrarianSession]]) -> dict:
    """Rev 2 (PLAN-REV2 item 1): one message to ``@librarian`` (*entry*, from *record*): a message instance is started
    (``open_session(mode="message")``), given the message as its question (the delivery log names it), and posts ONE
    reply entry (``reply_to`` = the message, mentioning its author): the answer, or why there is none. A message its
    author withdrew (a ``cancel`` of theirs that answers it) gets no reply. The instance keeps its new answers in the
    role's state and ends (its name is refused from then on). Returns the claim's outcome: ``{"state", "reply", "by"}``.
    The instance is stopped with the request: a cancelled answer posts nothing (the caller marks the claim abandoned)."""
    from unify.agents.record import HARNESS

    if record.withdrawn(entry.seq):
        return {"state": "withdrawn", "reply": None, "by": None}
    try:
        session = await open_session(mode="message")
    except Unavailable as exc:
        # no instance can start: the author still gets one reply, from the harness, saying why
        reply, _ = record.append(HARNESS, f"no librarian could answer #{entry.seq}: {str(exc)[:300]}", kind="reply",
                                 mentions=[entry.author], reply_to=entry.seq)  # fmt: skip
        return {"state": "failed", "reply": reply.seq, "by": HARNESS}
    rec = session.run_record if session.run_record is not None else record
    try:
        rec.log_role_delivery(session.name, [entry.seq])
        state = "answered"
        try:
            got = await session.answer(question_of(entry.text))
            text = reply_text(entry.seq, got)
        except NoAnswer as exc:
            state, text = "failed", f"no answer to #{entry.seq}: {str(exc)[:300]}"
        if rec.withdrawn(entry.seq):
            return {"state": "withdrawn", "reply": None, "by": session.name}
        reply, _ = rec.append(session.name, text, kind="reply", mentions=[entry.author], reply_to=entry.seq)
        keep_answers(session)
        return {"state": state, "reply": reply.seq, "by": session.name}
    finally:
        await session.close()


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
    "SCHEMA_ERROR",
    "SCHEMA_VALUE",
    "CALL_KIND",
    "AskInstance",
    "MODES",
    "answer_message",
    "keep_answers",
    "question_of",
    "reply_text",
    "LibrarianCalls",
    "LibrarianSession",
    "NoAnswer",
    "ORIGIN",
    "Unavailable",
    "Usage",
    "answer_key",
    "sent_to_librarian",
    "catalogue_line",
    "check_answer",
    "check_schema",
    "schema_errors",
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
