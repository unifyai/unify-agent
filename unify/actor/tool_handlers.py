"""The host handlers of the actor's and its teammates' in-code tools (``UNIFY_TOOLS_IN_CODE=on``).

Every handler has the registry's one signature, ``handler(ctx, args) -> Reply``: ``ctx`` is the channel's
``CallContext`` for the calling agent (built by :func:`actor_binding`, never from the box's frame), ``args`` the
call's validated arguments. A handler that does file or subprocess work is a plain function (the channel runs it in
its worker pool); one that waits on the record or the request is a coroutine function (the channel awaits it).

* ``read`` and ``grep`` read the request's workspace, named as the box shows it (``/workspace/...``; a relative path
  is under it). Both keep the workspace sandbox's rules: a path the sandbox hides is refused with the rule.
  ``read`` opens the path from the workspace's root descriptor one component at a time, never through a link;
  ``grep`` runs the actor's sandboxed search. Replies name paths as the box shows them, never a host path.
* ``install`` runs the workspace installer on the host, as the actor's ``install`` does today, and its reply names
  paths as the box shows them.
* The record tools go through ``ctx.record`` (:class:`unify.common.tool_registry.Record`: the agent record).
* ``answer(answer)`` records the request's answer, validated against the request's response format as the loop
  validates a response tool's payload; the loop ends the request when the cell returns.
* ``store_skills(request)`` starts the storage review as host work that outlives the call.
* ``current_request()`` gives the memory package the request the agent works on (``memory.ask()`` and
  ``memory.match()`` run the stored readers on it); it is on the channel but never in a cell's ``tools``.
"""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Optional

from unify.common import tool_registry as _registry
from unify.common.tool_registry import REFUSED, Refusal, RecordRefused, form_of, refusal

#: The box path of the request's workspace (the root the actor's read and grep are given).
WORKSPACE = "/workspace"
#: The longest a record wait may block (the agent record's own limit).
MAX_WAIT_S = 600.0


def _reply(text: str, status: str = "ok", form: Optional[str] = None, paths: tuple = ()) -> Any:
    from unify.memory.writer_tools import Reply

    return Reply(str(text), status, form, paths)


def _refused(text: Refusal) -> Any:
    return _reply(text, "refused", form_of(text))


def _from_record(tool: str, exc: RecordRefused) -> Any:
    """A record's refusal as *tool*'s reply: a finished refusal as it is, else the reason in the tool's template
    for the form (or plainly, keeping the form, when that template takes other fields)."""
    if form_of(exc.text) is not None:
        return _refused(exc.text)
    try:
        return _refused(refusal(tool, exc.form, why=exc.text))
    except (KeyError, IndexError):
        return _refused(Refusal(REFUSED + str(exc.text), tool, exc.form))


@dataclass
class ActorRequestState:
    """What the actor's handlers use of the request they serve (``CallContext.state``)."""

    #: the loop's slot for the turn's reply (``unify.common._async_tool.cell_reply.ReplySlot``); answer fills it
    reply_slot: Any = None
    #: the response format's check: the answer -> its JSON text (raises with the reason); None: any answer
    validate: Optional[Callable[[Any], str]] = None
    #: today's storage-review starter (the actor's ``store_skills`` tool function); None: not available
    store_skills: Optional[Callable[[str], Awaitable[Any]]] = None
    #: why skill storage is not available, when it is not
    store_refusal: str = "this session does not store skills"
    #: the host directory ``/workspace`` is bound to (a wheel path under /workspace is installed from there)
    workspace_root: Optional[str] = None
    #: the workspace sandbox's policy the request's cells run under (``unify.sandbox.build_policy()``); None: built
    #: at the call
    policy: Any = None
    #: who started this agent (a teammate's question that names no one goes to it); None for the actor
    spawner: Optional[str] = None
    #: host work started by a call that outlives it (a storage review)
    host_tasks: set = field(default_factory=set)
    #: UNIFY_MEMORY_LIBRARIAN=on: what opens the request's librarian at its first question
    #: (``unify.memory.librarian.opener``); None: the librarian is not available
    open_librarian: Optional[Callable[[], Awaitable[Any]]] = None
    #: the request's librarian session, once a question opened it; it ends with the request (``InCodeTools.close``)
    librarian: Any = None
    #: the request this agent works on, as ``current_request`` gives it to the memory package: its text, and the
    #: structured request as JSON data (None for a text or a message); None: not known here
    request_text: Optional[str] = None
    request_data: Any = None


def response_validator(rf_norm: Any) -> Callable[[Any], str]:
    """The answer check of a request with a response format (``NormalizedResponseFormat``): validate the payload
    and return its JSON text, as the loop does for a response tool's ``answer``."""

    def validate(payload: Any) -> str:
        if payload is None:
            raise ValueError("Missing 'answer' in tool arguments.")
        validated = rf_norm.validate(payload)
        dump = getattr(validated, "model_dump", None)
        return json.dumps(dump(mode="json") if callable(dump) else validated)

    return validate


def actor_flags(agent: str, can_store: bool, librarian: Optional[bool] = None) -> frozenset:
    """The configuration flags the actor's or a teammate's allow-list is built with: teammates are offered
    (``helper``), storing only where the actor may store (*can_store*), and asking the librarian only for the actor
    with ``UNIFY_MEMORY_LIBRARIAN=on`` (*librarian*; None reads the setting)."""
    from unify.settings import SETTINGS

    if librarian is None:
        librarian = getattr(SETTINGS, "UNIFY_MEMORY_LIBRARIAN", "off") == "on"
    return frozenset(
        {"helper"}
        | ({_registry.STORE_FLAG} if can_store and agent == "actor" else set())
        | ({_registry.LIBRARIAN_FLAG} if librarian and agent == "actor" else set())
        # UNIFY_MEMORY_IN_EPISODE=on: the actor marks the pauses of its work (tools.checkpoint)
        | ({_registry.CHECKPOINT_FLAG} if agent == "actor" and _in_episode(SETTINGS) else set()),
    )


def _in_episode(settings: Any) -> bool:
    """Whether ``UNIFY_MEMORY_IN_EPISODE`` is on for this run (with memory on); False when it cannot be read."""
    try:
        from unify.memory.integration.switch import in_episode, memory_on

        return memory_on(settings) and in_episode(settings)
    except Exception:  # noqa: BLE001 - a setting that cannot be read offers nothing new
        return False


def actor_binding(
    *,
    agent: str,
    name: str,
    request_id: Optional[str],
    roots: dict,
    record: Any,
    deadline: float,
    state: ActorRequestState,
    can_store: bool,
    limits: Any = None,
    counters: Any = None,
    context_cls: Any = None,
    librarian: bool = False,
) -> tuple[Any, tuple[str, ...]]:
    """The ``CallContext`` and the allow-list of the actor (``agent="actor"``) or a teammate (``"teammate"``) for
    one request: *roots* maps box roots (``/workspace``) to open directory descriptors, *record* is the agent
    record (:class:`unify.common.record_adapters.AgentRecordAdapter`), *name* the agent's name on it. Starting
    teammates is offered through ``delegate`` (the pool refuses it where the run does not allow it);
    ``store_skills`` only where the actor may store (*can_store*)."""
    if agent not in ("actor", "teammate"):
        raise ValueError(f"not the actor or a teammate: {agent!r}")
    allow = _registry.allow_list(agent, actor_flags(agent, can_store, librarian))
    if context_cls is None:
        from unify.common.tool_channel import CallContext as context_cls
    ctx = context_cls(
        agent=agent,
        name=name,
        role=agent,
        pass_id=None,
        request_id=request_id,
        roots=dict(roots),
        limits=limits,
        counters=counters,
        record=record,
        episodes=None,
        deadline=deadline,
        state=state,
    )
    return ctx, allow


# --- the workspace ---

#: Hits shown per ``grep`` page (and the most the search collects past the page's offset).
GREP_PAGE = 100
GREP_MAX_HITS = 1000


def _policy(ctx: Any) -> Any:
    policy = getattr(ctx.state, "policy", None)
    if policy is None:
        from unify import sandbox

        policy = sandbox.build_policy()
    return policy


def _box_path(path: Any) -> str:
    """*path* as a box path: a relative path is under ``/workspace`` (the box's working directory)."""
    text = "" if path is None else str(path)
    if not text.startswith("/"):
        text = WORKSPACE + ("/" + text if text else "")
    return text


def _outside(tool: str, path: str) -> Any:
    return _refused(Refusal(f"{REFUSED}{path[:200]} is outside {WORKSPACE}", tool, "outside_roots"))


def _hidden(virtual: str, policy: Any) -> Optional[str]:
    """The workspace sandbox's rule that hides *virtual* from a cell, as text naming box paths only; None when a
    cell may read it."""
    from pathlib import Path

    from unify import sandbox

    found = policy.readable_violation(Path(sandbox.host_path(virtual, policy)))
    if found is None:
        return None
    rule, detail = found
    return sandbox.neutral_text(f"hidden by the workspace sandbox rule `{rule}` ({detail})", policy)


def read(ctx: Any, args: dict) -> Any:
    """A page of a file under ``/workspace`` (8000 bytes from ``offset``), or a directory's listing."""
    from unify.memory import safe_paths as _safe
    from unify.memory import views as _views

    vpath = _box_path(args.get("path"))
    offset = args.get("offset", 0)
    offset = offset if isinstance(offset, int) and not isinstance(offset, bool) and offset >= 0 else 0
    root = (ctx.roots or {}).get(WORKSPACE)
    try:
        prefix, parts = _safe.split(vpath, [WORKSPACE])
    except _safe.Outside:
        return _outside("read", vpath)
    virtual = _safe.virtual(prefix, parts)
    where = ({"path": virtual, "mode": "read"},)
    if root is None:
        return _refused(refusal("read", "not_found", path=virtual))
    policy = _policy(ctx)
    hidden = _hidden(virtual, policy)
    if hidden is not None:
        return _reply(refusal("read", "unreadable", error=hidden, path=virtual), "refused", "unreadable", where)
    try:
        text, mode = _views.read_entry(virtual, {WORKSPACE: root}, offset)
    except _safe.Outside:
        return _outside("read", vpath)
    except (_safe.LinkFound, _safe.NotRegular) as exc:
        return _reply(refusal("read", "unreadable", error=exc.reason, path=virtual), "refused", "unreadable", where)
    except OSError:
        return _reply(refusal("read", "not_found", path=virtual), "refused", "not_found", where)
    return _reply(text, paths=({"path": virtual, "mode": mode},))


async def grep(ctx: Any, args: dict) -> Any:
    """Hits of a regular expression under a ``/workspace`` path (default the whole workspace), one page from hit
    ``offset``; the search runs in the workspace sandbox, as the actor's ``grep`` does."""
    from unify import sandbox
    from unify.actor.execution import file_tools
    from unify.common.tool_errors import ToolInputError
    from unify.memory import safe_paths as _safe

    pattern = args.get("pattern")
    if not isinstance(pattern, str) or not pattern:
        return _refused(refusal("grep", "bad_pattern", reason="pattern must be a non-empty string"))
    vpath = _box_path(args.get("path") or WORKSPACE)
    offset = args.get("offset", 0)
    offset = offset if isinstance(offset, int) and not isinstance(offset, bool) and offset >= 0 else 0
    try:
        prefix, parts = _safe.split(vpath, [WORKSPACE])
    except _safe.Outside:
        return _outside("grep", vpath)
    virtual = _safe.virtual(prefix, parts)
    where = ({"path": virtual, "mode": "search"},)
    policy = _policy(ctx)
    try:
        found = await file_tools.grep(pattern, virtual, min(GREP_MAX_HITS, offset + GREP_PAGE), policy=policy)
    except sandbox.SandboxRefusal as exc:
        error = sandbox.neutral_text(str(exc), policy)
        return _reply(refusal("grep", "unreadable", error=error, path=virtual), "refused", "unreadable", where)
    except ToolInputError as exc:
        text = str(exc)
        if text.startswith("No such file or directory"):
            return _reply(refusal("grep", "not_found", path=virtual), "refused", "not_found", where)
        return _reply(refusal("grep", "bad_pattern", reason=text[:300]), "refused", "bad_pattern", where)
    hits = list(found.get("hits") or [])
    page = hits[offset : offset + GREP_PAGE]
    text = "\n".join(page) or "(no hits)"
    end = offset + len(page)
    if len(hits) > end or (found.get("truncated") and end < GREP_MAX_HITS):
        text += f"\n[… hits {offset}–{end}; next: offset={end}]"
    elif found.get("truncated"):
        text += f"\n[… the search stopped at {GREP_MAX_HITS} hits; narrow the pattern or the path]"
    return _reply(text, paths=where)


def _host_spec(spec: str, workspace_root: Optional[str]) -> str:
    """A wheel named by its box path is installed from the host directory /workspace is bound to."""
    if workspace_root and (spec == WORKSPACE or spec.startswith(WORKSPACE + "/")):
        return workspace_root.rstrip("/") + spec[len(WORKSPACE) :]
    return spec


def install(ctx: Any, args: dict) -> Any:
    packages = args.get("packages")
    specs = [packages] if isinstance(packages, str) else packages
    if not isinstance(specs, list) or not specs or not all(isinstance(p, str) and p.strip() for p in specs):
        return _refused(refusal("install", "bad_packages"))
    from unify import environment, sandbox

    root = getattr(ctx.state, "workspace_root", None)
    result = environment.install([_host_spec(p.strip(), root) for p in specs])
    # the installer's report names the environment's host path; the box sees it elsewhere
    return _reply(sandbox.neutral_text(json.dumps(result, default=str), _policy(ctx)))


# --- the record ---


def _rec(ctx: Any):
    return ctx.record


def _running(ctx: Any) -> list:
    # an agent waiting for a message on the run record is running too: it reads what it is sent
    return [r["name"] for r in _rec(ctx).roster() if r.get("state") in ("running", "waiting")]


_TAGS = {"reply": " (done)", "cancel": " (stop)", "system": " (notice)"}


def _render(ctx: Any, entries: list) -> str:
    lines = [f"#{e['seq']} {e['author']}{_TAGS.get(e.get('kind'), '')}: {e['text']}" for e in entries]
    lines.append(f"(running now: {', '.join(_running(ctx))})")
    return "\n".join(lines)


def _left(ctx: Any) -> float:
    return min(MAX_WAIT_S, float(ctx.deadline) - time.monotonic())


def _post(ctx: Any, tool: str, text: Any, to: Optional[list] = None) -> tuple[Any, dict]:
    if not isinstance(text, str) or not text.strip():
        return _refused(refusal(tool, "no_text")), {}
    rec = _rec(ctx)
    try:
        if hasattr(rec, "last_from"):
            # the run record: a reply names the entry it answers, and a question goes where it says (or to the
            # agent that started this one)
            reply_to = rec.last_from(to[0]) if tool == "reply" and to else None
            default_to = getattr(ctx.state, "spawner", None) if tool == "ask" else None
            out = rec.post(ctx.name, text, to=to, reply_to=reply_to, default_to=default_to)
        else:
            out = rec.post(ctx.name, text, to=to)
    except RecordRefused as exc:
        return _from_record(tool, exc), {}
    sent = f"sent to {', '.join(out.get('to') or []) or 'no one by name'} (#{out.get('seq')})"
    if out.get("unresolved"):
        sent += f"; these names match no agent: {', '.join('@' + n for n in out['unresolved'])}"
    return _reply(sent), out


def send(ctx: Any, args: dict) -> Any:
    return _post(ctx, "send", args.get("text"))[0]


def messages(ctx: Any, args: dict) -> Any:
    new = _rec(ctx).unseen(ctx.name)
    if new:
        return _reply(_render(ctx, new))
    return _reply(f"no new messages (running now: {', '.join(_running(ctx))})")


async def _wait(ctx: Any, timeout: Optional[float] = None) -> str:
    left = _left(ctx) if timeout is None else min(_left(ctx), float(timeout))
    if left <= 0:
        return "no new messages; the request's time is used up"
    if not [n for n in _running(ctx) if n != ctx.name]:
        new = _rec(ctx).unseen(ctx.name)
        return _render(ctx, new) if new else "no new messages; no other agent is running"
    new = await _rec(ctx).wait(ctx.name, left)
    return _render(ctx, new) if new else "no new messages before the wait ended"


async def wait(ctx: Any, args: dict) -> Any:
    """Wait for the next entries for this agent, at most ``timeout`` seconds (required: the agent record's wait has
    no default)."""
    timeout = args.get("timeout")
    if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or not 0 < timeout <= MAX_WAIT_S:
        return _refused(refusal("wait", "bad_timeout"))
    return _reply(await _wait(ctx, timeout))


async def who(ctx: Any, args: dict) -> Any:
    lines = ["agents on the record:"]
    lines += [f"- {r['name']} ({r['role']}, {r['state']}): {r.get('help', '')}" for r in _rec(ctx).roster()]
    lines.append("Address an agent with @<name>.")
    return _reply("\n".join(lines))


def reply(ctx: Any, args: dict) -> Any:
    target, text = args.get("agent"), args.get("text")
    if not isinstance(text, str) or not text.strip():
        return _refused(refusal("reply", "no_text"))
    running = _running(ctx)
    if not isinstance(target, str) or target.lower() not in running or target.lower() == ctx.name:
        valid = ", ".join(f"@{n}" for n in running if n != ctx.name) or "(none)"
        return _refused(refusal("reply", "unknown_name", names=repr(str(target)[:80]), valid=valid))
    return _post(ctx, "reply", text, to=[target.lower()])[0]


async def stop(ctx: Any, args: dict) -> Any:
    name = args.get("helper")
    try:
        out = await _rec(ctx).stop(ctx.name, str(name))
    except RecordRefused as exc:
        return _from_record("stop", exc)
    return _reply(f"{out['name']} stopped (running now: {', '.join(_running(ctx))})")


async def ask(ctx: Any, args: dict) -> Any:
    question = args.get("question")
    spawner = getattr(ctx.state, "spawner", None)
    names_someone = isinstance(question, str) and "@" in question
    sent, out = _post(ctx, "ask", question, to=None if names_someone or not spawner else [spawner])
    if sent.status != "ok":
        return sent
    got = await _wait(ctx)
    return _reply(got if got.startswith("#") else f"no reply yet ({got})")


async def delegate(ctx: Any, args: dict) -> Any:
    task = args.get("task")
    if not isinstance(task, str) or not task.strip():
        return _refused(refusal("delegate", "no_task"))
    try:
        out = await _rec(ctx).start(ctx.name, task)
    except RecordRefused as exc:
        return _from_record("delegate", exc)
    return _reply(
        f"started {out['name']}; it works while you go on and answers you on the record "
        "(tools.messages() reads it, tools.wait() waits for it)",
    )


# --- ending the request, and storing skills ---


async def answer(ctx: Any, args: dict) -> Any:
    """Record the request's answer. With a response format, the answer must validate against it (the reason is
    returned when it does not, and the request goes on); the loop then ends the request with it when the cell
    returns, without another model call."""
    state = ctx.state
    slot = getattr(state, "reply_slot", None)
    # the slot holds this request's answer until the loop takes it, and is empty again when the session's next
    # request starts, so each request of a persistent session can answer once
    if slot is not None and getattr(slot, "replied", False):
        return _refused(refusal("answer", "answered"))
    if "answer" not in args:
        return _refused(refusal("answer", "bad_answer", reason="Missing 'answer' in tool arguments."))
    given = args.get("answer")
    validate = getattr(state, "validate", None)
    if validate is not None:
        try:
            text = validate(given)
        except Exception as exc:  # the validation's reason goes back to the cell
            return _refused(refusal("answer", "bad_answer", reason=str(exc)[:2000]))
    else:
        text = given if isinstance(given, str) else json.dumps(given, default=str)
    if slot is None:
        return _refused(refusal("answer", "bad_answer", reason="this request takes no answer from a cell"))
    slot.text, slot.from_value = text, True
    return _reply("answered: the answer is recorded; the request ends when this cell returns")


async def store_skills(ctx: Any, args: dict) -> Any:
    """Start today's storage review and keep it running after the call (its handle is not returned to the loop,
    which stops still-running handles when a call returns)."""
    request = args.get("request")
    if not isinstance(request, str) or not request.strip():
        return _refused(refusal("store_skills", "no_request"))
    state = ctx.state
    start = getattr(state, "store_skills", None)
    if start is None:
        return _refused(refusal("store_skills", "unavailable", why=getattr(state, "store_refusal", "")))
    handle = await start(request)
    if isinstance(handle, str):
        form = "no_context" if handle.startswith("No active execution context") else "unavailable"
        return _refused(refusal("store_skills", form, why=handle))
    task = asyncio.ensure_future(handle.result())
    state.host_tasks.add(task)
    task.add_done_callback(state.host_tasks.discard)
    return _reply("the skill-storage review started; it runs on after this call")


# --- asking the librarian (UNIFY_MEMORY_LIBRARIAN=on) ---


async def ask_library(ctx: Any, args: dict) -> Any:
    """One question to the request's librarian (``memory.ask`` sends it): the session is opened at the request's
    first question and kept for its follow-ups; the reply is the librarian's checked answer as JSON text. The
    librarian judges from the copy's structural catalogue and the items' text; nothing is run (rev 2). A question it
    already answered about the current request gets that answer again, marked as already given, with no model call
    (``LibrarianSession.ask``)."""
    question = args.get("question")
    if not isinstance(question, str) or not question.strip():
        return _refused(refusal("ask_library", "no_question"))
    state = ctx.state
    session = getattr(state, "librarian", None)
    if session is None:
        start = getattr(state, "open_librarian", None)
        if start is None:
            return _refused(refusal("ask_library", "unavailable", why="this request has no librarian"))
        from unify.memory import librarian as _librarian

        try:
            session = await start()
        except _librarian.Unavailable as exc:
            return _refused(refusal("ask_library", "unavailable", why=str(exc)[:300]))
        state.librarian = session
    from unify.memory import librarian as _librarian

    # the current request, as current_request gives it to the cells: part of what an answer answered
    request = getattr(state, "request_text", None)
    if not isinstance(request, str) or not request:
        request = _memory_run_request() if getattr(ctx, "agent", None) == "actor" else None
    try:
        if getattr(session, "evidence", None) is not None:
            # the harness's evidence serving on: the situation's latest message goes with the question
            text = await session.ask(
                question,
                request=request,
                last_message=_memory_run_last_message() if getattr(ctx, "agent", None) == "actor" else None,
            )
        else:
            text = await session.ask(question, request=request)
    except _librarian.NoAnswer as exc:
        return _refused(refusal("ask_library", "no_answer", why=str(exc)[:300]))
    return _reply(text)


# --- the current request, for the memory package (never in a cell's tools) ---

#: The most bytes of JSON text ``current_request`` replies with (a channel reply carries at most 1 MiB).
REQUEST_REPLY_BYTES = 1_000_000


def request_of(request: Any) -> tuple[Optional[str], Any]:
    """An ``act()`` request as ``current_request`` gives it: ``(text, data)``, exactly what the actor's first user
    message shows and nothing more. The text is the requester's text as the loop gives it (``bound_request.text_of``:
    a string, a user message's content, the latest user message of a list); a field of the request the message does
    not show (an id, a label) never reaches the box. *data* is the list of the JSON objects and arrays that text
    contains, in order of appearance, exactly as the bound request's ``data`` (:func:`request_data`). ``(None, None)``
    when the request shows no text."""
    from unify.common._async_tool.bound_request import text_of

    text = text_of(request)
    if not isinstance(text, str) or not text.strip():
        return None, None
    return text, request_data(text)


def request_data(text: str) -> list:
    """The JSON objects and arrays *text* contains, in order of appearance: ``worker_child.json_values``, the one
    definition the bound ``request.data`` uses."""
    from unify.actor.execution.worker_child import json_values

    return json_values(text)


def _memory_run_request() -> Optional[str]:
    """The text of the memory run's request (``integration.request.current()``), when one is open."""
    try:
        from unify.memory.integration import request as _request

        text = getattr(_request.current(), "request", None)
    except Exception:  # noqa: BLE001 - no memory run here
        return None
    return text if isinstance(text, str) and text else None


def _memory_run_last_message() -> Optional[str]:
    """The latest message the memory run's request received after its first, from its transcript
    (``RequestRun.last_message``); None when there is none or no run is open."""
    try:
        from unify.memory.integration import request as _request

        run = _request.current()
        text = run.last_message() if run is not None else None
    except Exception:  # noqa: BLE001 - no memory run here, or its transcript cannot be read
        return None
    return text if isinstance(text, str) and text else None


def current_request(ctx: Any, args: dict) -> Any:
    """The request the calling agent works on, as JSON text ``{"text": ..., "data": ...}``: the ``act()`` request the
    request's state holds (:func:`request_of`), else, for the actor only, the memory run's (``integration.request``);
    *data* is the JSON objects and arrays the text contains, in order (:func:`request_data`); a teammate gets only the
    task it was given. The memory package calls it through the channel
    client; it is never in a cell's ``tools``. Refused when no request is known, or when its JSON is larger than a
    reply carries."""
    state = ctx.state
    text, data = getattr(state, "request_text", None), getattr(state, "request_data", None)
    if (not isinstance(text, str) or not text) and getattr(ctx, "agent", None) == "actor":
        text = _memory_run_request()
        data = request_data(text) if text else None
    if not isinstance(text, str) or not text:
        text = None
    if text is None:
        return _refused(refusal("current_request", "unavailable", why="no current request is known"))
    reply = json.dumps({"text": text, "data": data})
    if len(reply.encode("utf-8")) > REQUEST_REPLY_BYTES:
        return _refused(refusal("current_request", "unavailable", why="the request is larger than a reply carries"))
    return _reply(reply)


__all__ = [
    "ActorRequestState",
    "actor_binding",
    "actor_flags",
    "answer",
    "ask",
    "ask_library",
    "current_request",
    "delegate",
    "grep",
    "install",
    "messages",
    "read",
    "reply",
    "request_of",
    "response_validator",
    "send",
    "stop",
    "store_skills",
    "wait",
    "who",
]
