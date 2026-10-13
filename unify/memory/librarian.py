"""The librarian: an agent the actor asks which stored items fit (``UNIFY_MEMORY_LIBRARIAN=on``; design J(b)).

``memory.ask(text)`` in an actor cell computes the recognisers' outcomes in the actor's own box (the ``memory.ask``
section of the generated package, :mod:`.library_ask`) and sends the question and those outcomes to the host through
the in-code tool ``ask_library`` (:func:`unify.actor.tool_handlers.ask_library`). The handler opens the request's
:class:`LibrarianSession` on the first question and keeps it for the request's follow-ups; the session ends with the
request (``InCodeTools.close``).

The session runs on the memory agents' runtime, as a helper does: the shared opening
(:func:`unify.common.tool_registry.preamble`) for the agent ``librarian``, one model-API tool (``execute_code``), its
cells in a persistent confined interpreter (:class:`.tool_handlers.InterpreterSession`) whose ``tools`` module holds
the allow-list (``finish`` only), the memory agents' compaction (:func:`.writer_pass.compact_by_summary`) and the
shared no-progress rule (:class:`unify.common._async_tool.loop_stop.AgentNoProgress`). Its box mounts the actor's
library copy read-only at ``/memory``; it has no write root (a private ``/tmp`` only), no network and an empty
environment, as every memory box. Its system message is the shared opening, the brief (:data:`BRIEF`) and the
catalogue (``.memory/catalogue.md``), in that order, so the catalogue is a cached prefix every question shares.

Its model calls go through the actor's route on the actor's model and effort (:func:`.writer_pass.unillm_turn` with
the origin :data:`ORIGIN`): the request's cost listener counts them as the actor's own (``integration.cost``), and
under ``UNIFY_REQUEST_METADATA_HEADERS=on`` each carries ``X-Unify-Call-Kind: librarian`` (:data:`CALL_KIND`). The
session also keeps its own totals (calls, tokens, cache reads, USD) for the use record; an unpriced call keeps the
USD unknown.

It answers with ``tools.finish(answer)`` (:func:`finish`): ``{"fits": [{"id", "call", "why"}, ...]}`` or
``{"fits": [], "nearest": [{"id", "differs"}, ...]}``. The handler checks the form and every id against the
catalogue; a malformed answer or an unknown id goes back as a refusal for the librarian to correct, within the
loop's existing limits (no cap of its own). The answer reaches the actor as the librarian gave it: the harness ranks,
scores and reorders nothing.
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
#: The fact labels ``memory.ask`` sends (:mod:`.library_ask`).
READ_OUTCOMES = ("parsed", "did-not-parse")
APPLIES_OUTCOMES = ("held", "did-not-hold", "raised", "not-run", "none")
CAPTURE_PREFIX = "capture:"
#: What the summary after a compaction says of the files (the memory agents' restart pointer, for this box).
RESTART_POINTER = "Nothing under /memory changed: open an item again with memory.show when you need it."
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
    "text, or a question of its own. Your job is to say which items apply. You do not do the agent's work, and you "
    "change nothing.\n"
    "\n"
    "## What you read\n"
    "- The catalogue below holds one card per stored function: its id, its signature and the first paragraph of "
    "its docstring; what it was saved against (what the work it was built from asked, and how that work went about "
    "it); and its record: its evidence level, its status, its reader, whether it states the conditions it applies "
    "under (`applies`), the structure of the input it was saved against, and how it has been used. Notes follow "
    "the functions. Captured procedures, recent work not yet reviewed, come last; one marked as tried and failed "
    "shows what broke and is never offered to run.\n"
    "- Each question comes with facts: for each stored procedure that has a reader, whether that reader parsed "
    "the question's text, and, where the procedure states conditions, whether its `applies` held on what the "
    "reader gave. A reader that parsed the text, or an `applies` that held, is evidence that the item's input "
    "fits, and evidence only: it does not show that the item does what the question needs. A reader that did not "
    "parse the text is evidence that the input differs.\n"
    '- `memory.show("<id>")` in a cell prints one item in full: its source, docstring, links and records. Open an '
    "item when its card leaves you unsure whether it applies.\n"
    "\n"
    "## How you answer\n"
    "- Read every card before you answer. Name every item that applies, not only one, in the catalogue's order.\n"
    "- For each item that applies, give a ready call the agent can run as written, such as "
    '`memory.run("<id>", text)` for a stored procedure, and a one-line reason that names the saved-against line, '
    "the fact or the record you relied on.\n"
    "- When the evidence is thin, answer that nothing fits, and name the nearest items with what differs between "
    "each of them and the question.\n"
    '- Answer with `tools.finish(answer)`, where answer is `{"fits": [{"id": ..., "call": ..., "why": ...}, ...]}`, '
    'or `{"fits": [], "nearest": [{"id": ..., "differs": ...}, ...]}` when nothing fits. Every id is one the '
    "catalogue holds. An answer in another form, or with an id the catalogue does not hold, is refused with the "
    "reason; correct it and call `tools.finish` again.\n"
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


def system_prompt(catalogue: str, cell_s: float = CELL_TIMEOUT_S) -> str:
    """The librarian's system message: the shared opening, the brief and the catalogue, in that order."""
    return opening(cell_s) + "\n\n" + BRIEF + "\n" + catalogue


def catalogue_ids(catalogue: str) -> list[str]:
    """Every item id the catalogue holds (its ``### <id>`` headings), in the catalogue's order."""
    out: list[str] = []
    for line in catalogue.splitlines():
        m = _HEADING.match(line)
        if m and m.group(1) not in out:
            out.append(m.group(1))
    return out


def failed_captures(catalogue: str) -> set[str]:
    """The captured attempts the catalogue lists as tried and failed (never offered to run)."""
    out: set[str] = set()
    current = None
    for line in catalogue.splitlines():
        m = _HEADING.match(line)
        if m:
            current = m.group(1)
        elif current and current.startswith(CAPTURE_PREFIX) and "tried and failed" in line:
            out.add(current)
    return out


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


def facts_text(facts: list[dict], dropped: int = 0) -> str:
    """The facts as the librarian reads them, one line per procedure."""
    if not facts:
        lines = ["- none: no stored procedure has a reader"]
    else:
        lines = []
        for f in facts:
            reader = f"reader `{f['reader']}`" if f.get("reader") else "no reader"
            read = "parsed the text" if f["read"] == "parsed" else "did not parse the text"
            lines.append(f"- `{f['id']}`: {reader} {read}; {_APPLIES_TEXT[f['applies']]}.")
    if dropped:
        lines.append(f"- ({dropped} fact(s) could not be read and are left out.)")
    return "\n".join(lines)


def question_message(question: str, facts: list[dict], dropped: int = 0) -> str:
    """The user message that carries one question and its facts."""
    return (
        "Question from the agent:\n"
        f"{question}\n\n"
        "Facts: the stored readers, run on the question's text in the agent's own sandbox (outcomes only):\n"
        f"{facts_text(facts, dropped)}\n\n"
        "Answer with tools.finish(answer)."
    )


# --- the answer -----------------------------------------------------------------------------------------------------


@dataclass
class LibrarianCalls:
    """The librarian's in-code handler state (``CallContext.state``): the catalogue's ids, and the answer of the
    question in progress."""

    ids: list[str]
    failed: set[str] = field(default_factory=set)
    answer: dict | None = None
    finish_text: str | None = None
    finished: bool = False


def _text(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def check_answer(answer: Any, ids: list[str], failed: set[str] = frozenset()) -> tuple[dict | None, str]:
    """The answer as the librarian gave it, or None with the reason it is refused: the form of design J(b).4, every
    id one the catalogue holds, and no captured attempt that failed among the fits. The harness adds, drops and
    reorders nothing."""
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
            return None, f"fit {i}: {f['id'][:200]!r} is not an id the catalogue holds"
        if f["id"] in failed:
            return None, f"fit {i}: {f['id']!r} was tried and failed; it is never offered to run (name it as nearest)"
    if len({f["id"] for f in fits}) != len(fits):
        return None, "an id is listed twice among the fits"
    if fits:
        if "nearest" in answer:
            return None, "`nearest` is given only when nothing fits"
        return {"fits": [dict(f) for f in fits]}, ""
    nearest = answer.get("nearest")
    if not isinstance(nearest, list):
        return None, "when nothing fits, `nearest` must be a list of {id, differs} (empty when the catalogue is empty)"
    for i, n in enumerate(nearest):
        if not isinstance(n, dict) or set(n) != {"id", "differs"} or not all(_text(n[k]) for k in n):
            return None, f"nearest {i} must be an object with the non-empty texts `id` and `differs`, and no other key"
        if n["id"] not in known:
            return None, f"nearest {i}: {n['id'][:200]!r} is not an id the catalogue holds"
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
    got, why = check_answer(args.get("answer"), state.ids, state.failed)
    if got is None:
        text = refusal("finish", "bad_answer", reason=why[:2000])
        state.finish_text = str(text)
        return Reply(str(text), "refused", text.form)
    state.answer, state.finished = got, True
    state.finish_text = "finished: your answer goes to the agent that asked"
    return Reply(state.finish_text)


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
    closed with the request."""

    def __init__(
        self,
        *,
        copy: Path,
        catalogue: str,
        turn: Turn,
        start: Callable[..., Awaitable[Any]] | None = None,
        on_event: Callable[[dict], None] | None = None,
        request_id: str | None = None,
        cell_limit_s: float = CELL_TIMEOUT_S,
        step_guard: int | None = None,
        compact: bool | None = None,
    ) -> None:
        from unify.common.tool_channel import HandlerThreads

        from . import tool_handlers as _th
        from .writer_pass import STEP_GUARD

        self.copy = Path(copy)
        self.dispatch = read_dispatch(self.copy)
        self.state = LibrarianCalls(ids=catalogue_ids(catalogue), failed=failed_captures(catalogue))
        self.turn = turn
        self.cell_limit_s = float(cell_limit_s)
        self.step_guard = STEP_GUARD if step_guard is None else int(step_guard)
        self.compact = _compact_on() if compact is None else bool(compact)
        self.usage = Usage()
        self.questions = 0
        self.closed = False
        self._lock = asyncio.Lock()
        self._cell = 0
        self._on_event = on_event
        self.messages: list[dict] = [{"role": "system", "content": system_prompt(catalogue, self.cell_limit_s)}]
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
                prebound={"memory": ["lazy_module", "memory"]},
            ),
            roots={LIBRARY_ROOT: self.copy},
            deadline=time.monotonic() + 30 * 86400.0,
            cell_limit_s=self.cell_limit_s,
            on_event=self._event,
            threads=HandlerThreads(),
            start=start,
        )

    def _event(self, row: dict) -> None:
        if self._on_event is None:
            return
        try:
            self._on_event({**row, "agent": AGENT})
        except Exception:  # noqa: BLE001 - a measurement never stops the librarian
            pass

    async def ask(self, question: str, facts: Any) -> str:
        """One question: its facts checked against the dispatch file, sent as the next user message; the
        librarian's accepted answer as JSON text. Raises :class:`NoAnswer` when it ends without one."""
        if self.closed:
            raise NoAnswer("the request has ended")
        async with self._lock:
            kept, dropped = checked_facts(facts, self.dispatch)
            self.questions += 1
            self.state.answer, self.state.finished, self.state.finish_text = None, False, None
            self.messages.append({"role": "user", "content": question_message(question, kept, dropped)})
            answer = await self._turns()
            return json.dumps(answer, ensure_ascii=False, sort_keys=True)

    async def _model_call(self) -> tuple[dict, Any]:
        from unify.common.llm_client import call_kind

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
        code = args.get("code") if isinstance(args, dict) else None
        if not isinstance(code, str):
            return str(refusal(GENERAL, "bad_arguments", reason="execute_code takes one string argument, code"))
        self.state.finish_text = None
        note = self.interp.take_note()
        res = await self.interp.cell(code, self._cell)
        self._cell += 1
        body, tail = _th.cell_parts(res, self.state.finish_text, self.interp.gone_during(), note)
        data = body.encode("utf-8", "replace")
        if len(data) > OUTPUT_BYTES:
            body = data[:OUTPUT_BYTES].decode("utf-8", "replace") + f"\n[… {len(data)} bytes in all; head shown]"
        return body + tail

    def box_alive(self) -> bool:
        return self.interp.box_alive()

    async def close(self) -> None:
        """End the session with the request: the interpreter is stopped (verified kill) and its channel closed.
        Safe to repeat."""
        if self.closed:
            return
        self.closed = True
        async with self._lock:
            await self.interp.close()


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
    """What opens this request's session at its first question: the run's library copy and its catalogue
    (``RequestRun.paths.checkout``; refused while no library copy is mounted), a turn on the actor's model
    (:func:`actor_turn`, or *turn_factory*'s). The session is registered with the run, whose use record reads its
    totals."""

    async def open_session() -> LibrarianSession:
        from .integration import request as _request
        from .library_export import CATALOGUE_FILE

        run = _request.current()
        copy = getattr(getattr(run, "paths", None), "checkout", None)
        if run is None or copy is None or not getattr(run, "library_mode", False):
            raise Unavailable("no library is mounted in this request")
        try:
            catalogue = (Path(copy) / CATALOGUE_FILE).read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as exc:
            raise Unavailable(f"the catalogue cannot be read ({type(exc).__name__})") from None
        session = LibrarianSession(
            copy=Path(copy),
            catalogue=catalogue,
            turn=(turn_factory or actor_turn)(),
            start=start,
            on_event=on_event,
            request_id=request_id,
        )
        sessions = getattr(run, "librarian_sessions", None)
        if isinstance(sessions, list):
            sessions.append(session)
        return session

    return open_session


def usage_fields(sessions: Any) -> dict:
    """The use record's ``librarian`` block over a request's sessions (:meth:`Usage.fields`)."""
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
    out["questions"] = sum(
        int(getattr(s, "questions", 0) or 0) for s in (sessions if isinstance(sessions, list) else [])
    )
    return out


__all__ = [
    "BRIEF",
    "CALL_KIND",
    "LibrarianCalls",
    "LibrarianSession",
    "NoAnswer",
    "ORIGIN",
    "Unavailable",
    "Usage",
    "check_answer",
    "checked_facts",
    "enabled",
    "finish",
    "opener",
    "system_prompt",
    "usage_fields",
]
