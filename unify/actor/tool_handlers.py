"""The host handlers of the actor's and its teammates' in-code tools (``UNIFY_TOOLS_IN_CODE=on``).

Every handler has the registry's one signature, ``handler(ctx, args) -> Reply``: ``ctx`` is the channel's
``CallContext`` for the calling agent (built by :func:`actor_binding`, never from the box's frame), ``args`` the
call's validated arguments. A handler that does file or subprocess work is a plain function (the channel runs it in
its worker pool); one that waits on the record or the request is a coroutine function (the channel awaits it).

* ``read`` and ``grep`` are the memory agents' handlers over the request's roots (``/workspace``).
* ``install`` runs the workspace installer on the host, as the actor's ``install`` does today.
* The record tools go through ``ctx.record`` (:class:`unify.common.tool_registry.Record`: the agent record).
* ``finish(answer)`` records the request's answer, validated against the request's response format as the loop
  validates a response tool's payload; the loop ends the request when the cell returns.
* ``store_skills(request)`` starts the storage review as host work that outlives the call.
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

    #: the loop's slot for the turn's reply (``unify.common._async_tool.cell_reply.ReplySlot``); finish fills it
    reply_slot: Any = None
    #: the response format's check: the answer -> its JSON text (raises with the reason); None: any answer
    validate: Optional[Callable[[Any], str]] = None
    #: today's storage-review starter (the actor's ``store_skills`` tool function); None: not available
    store_skills: Optional[Callable[[str], Awaitable[Any]]] = None
    #: why skill storage is not available, when it is not
    store_refusal: str = "this session does not store skills"
    #: the host directory ``/workspace`` is bound to (a wheel path under /workspace is installed from there)
    workspace_root: Optional[str] = None
    #: who started this agent (a teammate's question that names no one goes to it); None for the actor
    spawner: Optional[str] = None
    #: host work started by a call that outlives it (a storage review)
    host_tasks: set = field(default_factory=set)
    finished: bool = False


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
) -> tuple[Any, tuple[str, ...]]:
    """The ``CallContext`` and the allow-list of the actor (``agent="actor"``) or a teammate (``"teammate"``) for
    one request: *roots* maps box roots (``/workspace``) to open directory descriptors, *record* is the agent
    record (:class:`unify.common.record_adapters.AgentRecordAdapter`), *name* the agent's name on it. Starting
    teammates is offered through ``delegate`` (the pool refuses it where the run does not allow it);
    ``store_skills`` only where the actor may store (*can_store*)."""
    if agent not in ("actor", "teammate"):
        raise ValueError(f"not the actor or a teammate: {agent!r}")
    flags = frozenset({"helper"} | ({_registry.STORE_FLAG} if can_store and agent == "actor" else set()))
    allow = _registry.allow_list(agent, flags)
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


def read(ctx: Any, args: dict) -> Any:
    from unify.memory import tool_handlers as _memory

    return _memory.read(ctx, args)


def grep(ctx: Any, args: dict) -> Any:
    from unify.memory import tool_handlers as _memory

    return _memory.grep(ctx, args)


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
    from unify import environment

    root = getattr(ctx.state, "workspace_root", None)
    result = environment.install([_host_spec(p.strip(), root) for p in specs])
    return _reply(json.dumps(result, default=str))


# --- the record ---


def _rec(ctx: Any):
    return ctx.record


def _running(ctx: Any) -> list:
    return [r["name"] for r in _rec(ctx).roster() if r.get("state") == "running"]


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
    try:
        out = _rec(ctx).post(ctx.name, text, to=to)
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


async def finish(ctx: Any, args: dict) -> Any:
    """Record the request's answer. With a response format, the answer must validate against it (the reason is
    returned when it does not, and the request goes on); the loop then ends the request with it when the cell
    returns, without another model call."""
    state = ctx.state
    slot = getattr(state, "reply_slot", None)
    if getattr(state, "finished", False) or (slot is not None and getattr(slot, "replied", False)):
        return _refused(refusal("finish", "answered"))
    if "answer" not in args:
        return _refused(refusal("finish", "bad_answer", reason="Missing 'answer' in tool arguments."))
    answer = args.get("answer")
    validate = getattr(state, "validate", None)
    if validate is not None:
        try:
            text = validate(answer)
        except Exception as exc:  # the validation's reason goes back to the cell
            return _refused(refusal("finish", "bad_answer", reason=str(exc)[:2000]))
    else:
        text = answer if isinstance(answer, str) else json.dumps(answer, default=str)
    if slot is None:
        return _refused(refusal("finish", "bad_answer", reason="this request takes no answer from a cell"))
    slot.text, slot.from_value = text, True
    state.finished = True
    return _reply("finished: the answer is recorded; the request ends when this cell returns")


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


__all__ = [
    "ActorRequestState",
    "actor_binding",
    "ask",
    "delegate",
    "finish",
    "grep",
    "install",
    "messages",
    "read",
    "reply",
    "response_validator",
    "send",
    "stop",
    "store_skills",
    "wait",
    "who",
]
