"""The host handlers of the actor's and its teammates' in-code tools (``tools.<name>(...)`` in a cell).

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
* ``current_request()`` gives the memory package the request the agent works on (``memory.ask()`` asks about it and
  ``memory.request()`` gives it); it is on the channel but never in a cell's ``tools``.
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Awaitable, Callable, Optional

from unify.common import tool_registry as _registry
from unify.common.tool_registry import REFUSED, Refusal, RecordRefused, form_of

#: The box path of the request's workspace (the root the actor's read and grep are given).
WORKSPACE = "/workspace"
#: The longest a record wait may block (the agent record's own limit).
MAX_WAIT_S = 600.0


def _reply(text: str, status: str = "ok", form: Optional[str] = None, paths: tuple = ()) -> Any:
    from unify.memory.writer_tools import Reply

    return Reply(str(text), status, form, paths)


def _refused(text: Refusal) -> Any:
    return _reply(text, "refused", form_of(text))


def _refusal(ctx: Any, tool: str, form: str, **fields: object) -> Refusal:
    """The refusal *form* of *tool* as the calling agent reads it in this run (``tool_registry.role_refusal``): the
    actor's and a teammate's own texts, which name only the roots and calls they have (none of memory's in a run
    without it)."""
    return _registry.role_refusal(getattr(ctx, "agent", None), tool, form, **fields)


def _from_record(ctx: Any, tool: str, exc: RecordRefused) -> Any:
    """A record's refusal as *tool*'s reply: a finished refusal as it is, else the reason in the tool's template
    for the form (or plainly, keeping the form, when that template takes other fields)."""
    if form_of(exc.text) is not None:
        return _refused(exc.text)
    try:
        return _refused(_refusal(ctx, tool, exc.form, why=exc.text))
    except (KeyError, IndexError):
        return _refused(Refusal(REFUSED + str(exc.text), tool, exc.form))


@dataclass
class ActorRequestState:
    """What the actor's handlers use of the request they serve (``CallContext.state``)."""

    #: the loop's slot for the turn's reply (``unify.common._async_tool.cell_reply.ReplySlot``); answer fills it
    reply_slot: Any = None
    #: the response format's check: the answer -> its JSON text (raises with the reason); None: any answer
    validate: Optional[Callable[[Any], str]] = None
    #: the host directory ``/workspace`` is bound to (a wheel path under /workspace is installed from there)
    workspace_root: Optional[str] = None
    #: the workspace sandbox's policy the request's cells run under (``unify.sandbox.build_policy()``); None: built
    #: at the call
    policy: Any = None
    #: who started this agent (a teammate's question that names no one goes to it); None for the actor
    spawner: Optional[str] = None
    #: what opens the request's librarian at its first question
    #: (``unify.memory.librarian.opener``); None: the librarian is not available
    open_librarian: Optional[Callable[[], Awaitable[Any]]] = None
    #: the request's librarian session, once a question opened it; it ends with the request (``InCodeTools.close``)
    librarian: Any = None
    #: the request this agent works on, as ``current_request`` gives it to the memory package: its text, and the
    #: structured request as JSON data (None for a text or a message); for a teammate the run's request, which its
    #: task serves (the task is its first message); None: not known here
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


def actor_flags(agent: str, *, librarian: Optional[bool] = None) -> frozenset:
    """The configuration flags the actor's or a teammate's allow-list is built with: teammates are offered
    (``helper``), and asking the librarian for the actor and its teammates alike (*librarian*; None: whenever the run
    has memory)."""
    from unify.settings import SETTINGS

    if librarian is None:
        from unify.memory.integration.switch import memory_on

        librarian = memory_on(SETTINGS)
    return frozenset(
        {"helper"}
        | ({_registry.LIBRARIAN_FLAG} if librarian and agent in ("actor", "teammate") else set())
        # UNIFY_MEMORY=in_episode: the actor marks the pauses of its work (tools.checkpoint)
        | ({_registry.CHECKPOINT_FLAG} if agent == "actor" and _in_episode(SETTINGS) else set())
        # a world whose requests give examples offers the example check (UNIFY_ACTOR_EXAMPLES_CHECK)
        | ({_registry.EXAMPLES_FLAG} if agent in ("actor", "teammate") and SETTINGS.UNIFY_ACTOR_EXAMPLES_CHECK
           else set()),
    )


def _in_episode(settings: Any) -> bool:
    """Whether ``UNIFY_MEMORY=in_episode`` is on for this run (with memory on); False when it cannot be read."""
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
    limits: Any = None,
    counters: Any = None,
    context_cls: Any = None,
    librarian: bool = False,
) -> tuple[Any, tuple[str, ...]]:
    """The ``CallContext`` and the allow-list of the actor (``agent="actor"``) or a teammate (``"teammate"``) for
    one request: *roots* maps box roots (``/workspace``) to open directory descriptors, *record* is the agent
    record (:class:`unify.common.record_adapters.AgentRecordAdapter`), *name* the agent's name on it. Starting
    teammates is offered through ``delegate`` (the pool refuses it where the run does not allow it)."""
    if agent not in ("actor", "teammate"):
        raise ValueError(f"not the actor or a teammate: {agent!r}")
    allow = _registry.allow_list(agent, actor_flags(agent, librarian=librarian))
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


#: The box path of the run's library copy (read-only), as the actor's memory guide names it (``unify.sandbox``'s
#: library mount); ``/memory`` is the same library as the librarian and the writer name it. Either names the request's
#: copy, in every memory mode.
LIBRARY = "/memory-library"
MEMORY_ALIAS = "/memory"


@dataclass(frozen=True)
class LibraryView:
    """The library copy as the workspace sandbox mounts it for this request's cells.

    *mount* is the host directory the sandbox binds read-only at :data:`LIBRARY` (``unify.sandbox.late_readonly_mounts``
    of the worker's own late mounts, so a read sees exactly what a cell sees), *mounts* those mounts, and *copy* the host
    directory of the request's copy: the mount itself, or under ``UNIFY_MEMORY=in_episode`` the pin its link *link*
    (inside the mount) points to now. A box path under :data:`LIBRARY` or ``/memory`` names the copy; under
    in_episode a path through the link names the copy too, and one under another entry of the mount (the pins) names
    that entry as a cell sees it."""

    mount: Path
    copy: Path
    link: str
    mounts: tuple


def library_view(policy: Any) -> Optional[LibraryView]:
    """The request's :class:`LibraryView`; None when the workspace sandbox mounts no library."""
    from unify import sandbox
    from unify.memory.integration import hooks

    try:
        late = hooks.worker_readonly_mounts()
        mounts = tuple(sandbox.late_readonly_mounts(policy, late)) if late else ()
    except Exception:  # noqa: BLE001 - no library mounted for these cells
        return None
    mount = next((Path(host) for host, at in mounts if at == LIBRARY), None)
    if mount is None or not mount.is_dir():
        return None
    copy, link = mount, ""
    root = getattr(hooks._run(), "copy_root", None)
    if root is not None:
        # the copy's link inside the mount, named as the box names it (its own name, never followed here)
        named = Path(os.path.realpath(os.path.dirname(os.fspath(root)))) / os.path.basename(os.fspath(root))
        real = Path(os.path.realpath(root))
        rel = os.path.relpath(named, mount)
        if (
            named != mount
            and "/" not in rel
            and rel not in (".", "..")
            and real.is_dir()
            and real != mount
            and real.is_relative_to(mount)
        ):
            copy, link = real, rel
    return LibraryView(mount, copy, link, mounts)


def _library_place(view: LibraryView, vpath: str) -> Optional[tuple[str, Path, str]]:
    """Where a box path names the library: ``(box path as shown, host root, box path under that root)``, the box path
    shown being under :data:`LIBRARY`; None when *vpath* names no library path. Raises ``safe_paths.Outside`` for a
    path that leaves its root."""
    from unify.memory import safe_paths as _safe

    if not (vpath in (LIBRARY, MEMORY_ALIAS) or vpath.startswith((LIBRARY + "/", MEMORY_ALIAS + "/"))):
        return None
    prefix, parts = _safe.split(vpath, [LIBRARY, MEMORY_ALIAS])
    if prefix == LIBRARY and view.link and parts:
        if parts[0] == view.link:
            parts = parts[1:]
        elif parts[0] in _mount_entries(view):
            # an entry of the mount besides the copy's link (in_episode: the pins), as a cell sees it
            shown = _safe.virtual(LIBRARY, parts)
            return shown, view.mount, shown
    shown = _safe.virtual(LIBRARY, parts)
    return shown, view.copy, shown


def _mount_entries(view: LibraryView) -> frozenset:
    """The names directly in the mount other than the copy's link (empty when the copy is the mount)."""
    if not view.link:
        return frozenset()
    try:
        return frozenset(n for n in os.listdir(view.mount) if n != view.link)
    except OSError:
        return frozenset()


def _host_of(root: Path, shown: str) -> Path:
    """The host path of *shown* (a box path under :data:`LIBRARY`) under the host *root*, lexically."""
    rel = shown[len(LIBRARY) :].lstrip("/")
    return root / rel if rel else root


def _outside(tool: str, path: str, ctx: Any) -> Any:
    """A path outside the agent's roots, refused in its own words: with memory, the workspace and the library (and the
    call that opens an entry); without, the workspace alone."""
    return _refused(_refusal(ctx, tool, "outside_roots", path=path[:200]))


def _library_hidden(view: LibraryView, host: Path, policy: Any) -> Optional[str]:
    """The sandbox's rule that keeps *host* (under the library mount) from a cell or from the harness's file tools,
    as text naming no host path; None when it may be read."""
    from unify import sandbox

    found = sandbox.late_readable_violation(host, policy, view.mounts)
    return None if found is None else f"hidden by the workspace sandbox rule `{found[0]}`"


def _read_library(ctx: Any, view: LibraryView, place: tuple[str, Path, str], asked: str, offset: int) -> Any:
    """``read`` of a path in the library copy, read-only, as a cell would read it: from the sandbox's own mount of the
    copy (:class:`LibraryView`), under its rules (:func:`_library_hidden`), opened from the root one component at a
    time, never through a link. A refusal names the path as the agent wrote it (*asked*)."""
    from unify.memory import safe_paths as _safe
    from unify.memory import views as _views

    shown, root, _ = place
    where = ({"path": shown, "mode": "read"},)
    hidden = _library_hidden(view, _host_of(root, shown), _policy(ctx))
    if hidden is not None:
        return _reply(_refusal(ctx, "read", "unreadable", error=hidden, path=asked), "refused", "unreadable", where)
    try:
        text, mode = _views.read_entry(shown, {LIBRARY: str(root)}, offset)
    except _safe.Outside:
        return _outside("read", asked, ctx)
    except (_safe.LinkFound, _safe.NotRegular) as exc:
        return _reply(_refusal(ctx, "read", "unreadable", error=exc.reason, path=asked), "refused", "unreadable", where)
    except OSError:
        return _reply(_refusal(ctx, "read", "not_in_library", path=asked), "refused", "not_in_library", where)
    return _reply(text, paths=({"path": shown, "mode": mode},))


def _hidden(virtual: str, policy: Any) -> Optional[str]:
    """The workspace sandbox's rule that hides *virtual* from a cell, as text naming box paths only; None when a
    cell may read it."""
    from unify import sandbox

    found = policy.readable_violation(Path(sandbox.host_path(virtual, policy)))
    if found is None:
        return None
    rule, detail = found
    return sandbox.neutral_text(f"hidden by the workspace sandbox rule `{rule}` ({detail})", policy)


def _library_place_of(ctx: Any, vpath: str) -> tuple[Optional[LibraryView], Optional[tuple[str, Path, str]]]:
    """``(view, place)`` when *vpath* names the library of a request that mounts one (:func:`_library_place`), else
    ``(view or None, None)``."""
    if not (vpath in (LIBRARY, MEMORY_ALIAS) or vpath.startswith((LIBRARY + "/", MEMORY_ALIAS + "/"))):
        return None, None
    view = library_view(_policy(ctx))
    if view is None:
        return None, None
    return view, _library_place(view, vpath)


def read(ctx: Any, args: dict) -> Any:
    """A page of a file under ``/workspace`` or of the library copy (8000 bytes from ``offset``), or a directory's
    listing."""
    from unify.memory import safe_paths as _safe
    from unify.memory import views as _views

    vpath = _box_path(args.get("path"))
    offset = args.get("offset", 0)
    offset = offset if isinstance(offset, int) and not isinstance(offset, bool) and offset >= 0 else 0
    try:
        view, place = _library_place_of(ctx, vpath)
    except _safe.Outside:
        return _outside("read", vpath, ctx)
    if place is not None:
        return _read_library(ctx, view, place, vpath, offset)
    root = (ctx.roots or {}).get(WORKSPACE)
    try:
        prefix, parts = _safe.split(vpath, [WORKSPACE])
    except _safe.Outside:
        return _outside("read", vpath, ctx)
    virtual = _safe.virtual(prefix, parts)
    where = ({"path": virtual, "mode": "read"},)
    if root is None:
        return _refused(_refusal(ctx, "read", "not_found", path=virtual))
    policy = _policy(ctx)
    hidden = _hidden(virtual, policy)
    if hidden is not None:
        return _reply(_refusal(ctx, "read", "unreadable", error=hidden, path=virtual), "refused", "unreadable", where)
    try:
        text, mode = _views.read_entry(virtual, {WORKSPACE: root}, offset)
    except _safe.Outside:
        return _outside("read", vpath, ctx)
    except (_safe.LinkFound, _safe.NotRegular) as exc:
        return _reply(_refusal(ctx, "read", "unreadable", error=exc.reason, path=virtual), "refused", "unreadable", where)
    except OSError:
        return _reply(_refusal(ctx, "read", "not_found", path=virtual), "refused", "not_found", where)
    return _reply(text, paths=({"path": virtual, "mode": mode},))


def _grep_library(
    ctx: Any, view: LibraryView, place: tuple[str, Path, str], asked: str, pattern: str, offset: int
) -> Any:
    """``grep`` in the library copy: the files are opened from the sandbox's own mount of the copy, one component at
    a time and never through a link, those its rules keep from a cell left out (:func:`_library_hidden`); the pattern
    runs in a bounded child process that is handed their bytes and box paths only (``views.grep_bounded``). A refusal
    names the path as the agent wrote it (*asked*)."""
    from unify.memory import safe_paths as _safe
    from unify.memory import views as _views

    shown, root, _ = place
    where = ({"path": shown, "mode": "search"},)
    policy = _policy(ctx)
    host = _host_of(root, shown)
    hidden = _library_hidden(view, host, policy)
    if hidden is not None:
        return _reply(_refusal(ctx, "grep", "unreadable", error=hidden, path=asked), "refused", "unreadable", where)
    if not os.path.lexists(host):
        return _reply(_refusal(ctx, "grep", "not_in_library", path=asked), "refused", "not_in_library", where)
    try:
        text = _views.grep_bounded(
            pattern,
            shown,
            {LIBRARY: str(root)},
            offset,
            want=lambda box: _library_hidden(view, _host_of(root, box), policy) is None,
        )
    except _safe.Outside:
        return _outside("grep", asked, ctx)
    except _safe.LinkFound as exc:
        return _reply(_refusal(ctx, "grep", "unreadable", error=exc.reason, path=asked), "refused", "unreadable", where)
    except _views.PatternProblem as exc:
        return _reply(_refusal(ctx, "grep", "bad_pattern", reason=str(exc)[:300]), "refused", "bad_pattern", where)
    except _views.SearchFailed:
        return _reply(_refusal(ctx, "grep", "search_failed"), "refused", "search_failed", where)
    return _reply(text, paths=where)


async def grep(ctx: Any, args: dict) -> Any:
    """Hits of a regular expression under a ``/workspace`` path (default the whole workspace) or a library path, one
    page from hit ``offset``; a workspace search runs in the workspace sandbox, as the actor's ``grep`` does."""
    from unify import sandbox
    from unify.actor.execution import file_tools
    from unify.common.tool_errors import ToolInputError
    from unify.memory import safe_paths as _safe

    pattern = args.get("pattern")
    if not isinstance(pattern, str) or not pattern:
        return _refused(_refusal(ctx, "grep", "bad_pattern", reason="pattern must be a non-empty string"))
    vpath = _box_path(args.get("path") or WORKSPACE)
    offset = args.get("offset", 0)
    offset = offset if isinstance(offset, int) and not isinstance(offset, bool) and offset >= 0 else 0
    try:
        view, place = _library_place_of(ctx, vpath)
    except _safe.Outside:
        return _outside("grep", vpath, ctx)
    if place is not None:
        return await asyncio.to_thread(_grep_library, ctx, view, place, vpath, pattern, offset)
    try:
        prefix, parts = _safe.split(vpath, [WORKSPACE])
    except _safe.Outside:
        return _outside("grep", vpath, ctx)
    virtual = _safe.virtual(prefix, parts)
    where = ({"path": virtual, "mode": "search"},)
    policy = _policy(ctx)
    try:
        found = await file_tools.grep(pattern, virtual, min(GREP_MAX_HITS, offset + GREP_PAGE), policy=policy)
    except sandbox.SandboxRefusal as exc:
        error = sandbox.neutral_text(str(exc), policy)
        return _reply(_refusal(ctx, "grep", "unreadable", error=error, path=virtual), "refused", "unreadable", where)
    except ToolInputError as exc:
        text = str(exc)
        if text.startswith("No such file or directory"):
            return _reply(_refusal(ctx, "grep", "not_found", path=virtual), "refused", "not_found", where)
        return _reply(_refusal(ctx, "grep", "bad_pattern", reason=text[:300]), "refused", "bad_pattern", where)
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
        return _refused(_refusal(ctx, "install", "bad_packages"))
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
        return _refused(_refusal(ctx, tool, "no_text")), {}
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
        return _from_record(ctx, tool, exc), {}
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
        return _refused(_refusal(ctx, "wait", "bad_timeout"))
    return _reply(await _wait(ctx, timeout))


async def who(ctx: Any, args: dict) -> Any:
    lines = ["agents on the record:"]
    lines += [f"- {r['name']} ({r['role']}, {r['state']}): {r.get('help', '')}" for r in _rec(ctx).roster()]
    lines.append("Address an agent with @<name>.")
    return _reply("\n".join(lines))


def reply(ctx: Any, args: dict) -> Any:
    target, text = args.get("agent"), args.get("text")
    if not isinstance(text, str) or not text.strip():
        return _refused(_refusal(ctx, "reply", "no_text"))
    running = _running(ctx)
    if not isinstance(target, str) or target.lower() not in running or target.lower() == ctx.name:
        valid = ", ".join(f"@{n}" for n in running if n != ctx.name) or "(none)"
        return _refused(_refusal(ctx, "reply", "unknown_name", names=repr(str(target)[:80]), valid=valid))
    return _post(ctx, "reply", text, to=[target.lower()])[0]


async def stop(ctx: Any, args: dict) -> Any:
    name = args.get("helper")
    try:
        out = await _rec(ctx).stop(ctx.name, str(name))
    except RecordRefused as exc:
        return _from_record(ctx, "stop", exc)
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
        return _refused(_refusal(ctx, "delegate", "no_task"))
    try:
        out = await _rec(ctx).start(ctx.name, task)
    except RecordRefused as exc:
        return _from_record(ctx, "delegate", exc)
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
        return _refused(_refusal(ctx, "answer", "answered"))
    if "answer" not in args:
        return _refused(_refusal(ctx, "answer", "bad_answer", reason="Missing 'answer' in tool arguments."))
    given = args.get("answer")
    validate = getattr(state, "validate", None)
    if validate is not None:
        try:
            text = validate(given)
        except Exception as exc:  # the validation's reason goes back to the cell
            return _refused(_refusal(ctx, "answer", "bad_answer", reason=str(exc)[:2000]))
    else:
        text = given if isinstance(given, str) else json.dumps(given, default=str)
    if slot is None:
        return _refused(_refusal(ctx, "answer", "bad_answer", reason="this request takes no answer from a cell"))
    slot.text = text
    return _reply("answered: the answer is recorded; the request ends when this cell returns")


# --- asking the librarian ---


async def ask_library(ctx: Any, args: dict) -> Any:
    """One question to the request's librarian (``memory.ask`` sends it): the session is opened at the request's
    first question and kept for its follow-ups; the reply is the librarian's checked answer as JSON text. The
    librarian judges from the copy's structural catalogue and the items' text; nothing is run (rev 2). A question it
    already answered about the current request gets that answer again, marked as already given, with no model call
    (``LibrarianSession.ask``)."""
    question = args.get("question")
    if not isinstance(question, str) or not question.strip():
        return _refused(_refusal(ctx, "ask_library", "no_question"))
    schema = args.get("schema")
    if schema is not None:
        from unify.memory.librarian import check_schema

        try:
            check_schema(schema)
        except ValueError as exc:
            return _refused(_refusal(ctx, "ask_library", "bad_schema", why=str(exc)[:300]))
    failed = args.get("failed")
    if failed is None:
        failed = []
    if not isinstance(failed, list) or len(failed) > MAX_FAILED_ITEMS or not all(
        isinstance(i, str) and 0 < len(i) <= 300 for i in failed
    ):
        return _refused(_refusal(ctx, "ask_library", "bad_failed"))
    # the items whose call failed in this request: those the asking cell's package saw, and those the request's earlier
    # cells recorded; none is offered again in this request (a use outcome, never stored as one)
    failed = sorted(set(failed) | set(_memory_run_failed()))
    state = ctx.state
    session = getattr(state, "librarian", None)
    if session is None:
        start = getattr(state, "open_librarian", None)
        if start is None:
            return _refused(_refusal(ctx, "ask_library", "unavailable", why="this request has no librarian"))
        from unify.memory import librarian as _librarian

        try:
            session = await start()
        except _librarian.Unavailable as exc:
            return _refused(_refusal(ctx, "ask_library", "unavailable", why=str(exc)[:300]))
        state.librarian = session
    from unify.memory import librarian as _librarian

    # the current request, as current_request gives it to the cells: part of what an answer answered
    request = getattr(state, "request_text", None)
    if not isinstance(request, str) or not request:
        request = _memory_run_request()
    try:
        extra = {"schema": schema} if schema is not None else {}
        if failed:
            extra["failed"] = failed
        if getattr(session, "evidence", None) is not None:
            # the harness's evidence serving on: the situation's latest message goes with the question
            text = await session.ask(question, request=request, last_message=_memory_run_last_message(), **extra)
        else:
            text = await session.ask(question, request=request, **extra)
    except _librarian.NoAnswer as exc:
        return _refused(_refusal(ctx, "ask_library", "no_answer", why=str(exc)[:300]))
    return _reply(text)


# --- the current request, for the memory package (never in a cell's tools) ---

#: The most bytes of JSON text ``current_request`` replies with (a channel reply carries at most 1 MiB).
REQUEST_REPLY_BYTES = 1_000_000


def request_of(request: Any) -> tuple[Optional[str], Any]:
    """An ``act()`` request as ``current_request`` gives it: ``(text, data)``, exactly what the actor's first user
    message shows and nothing more. *data* is the list of the JSON objects and arrays that text contains, in order
    of appearance (:func:`request_data`). ``(None, None)`` when the request shows no text."""
    if not isinstance(request, str) or not request.strip():
        return None, None
    return request, request_data(request)


def request_data(text: str) -> list:
    """The JSON objects and arrays *text* contains, in order of appearance (``worker_child.json_values``)."""
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


#: The most failed items one question may carry.
MAX_FAILED_ITEMS = 200


def _memory_run_failed() -> list[str]:
    """The items whose calls raised or refused their input in the memory run's cells so far (its runtime call
    counts), in id order; [] when no run is open."""
    try:
        from unify.memory.integration import request as _request

        run = _request.current()
        counts = (getattr(run, "runtime_calls", None) or {}).get("counts") or {}
    except Exception:  # noqa: BLE001 - no memory run here
        return []
    return sorted(
        item
        for item, row in counts.items()
        if isinstance(item, str) and isinstance(row, dict) and any(
            isinstance(row.get(k), int) and row[k] > 0 for k in ("refused", "error")
        )
    )


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
    """The request the calling agent works on, as JSON text ``{"text": ..., "data": ...}``: the request the request's
    state holds (:func:`request_of`: the actor's ``act()`` request, or for a teammate the run's request its task
    serves), else the memory run's (``integration.request``); *data* is the JSON objects and arrays the text contains,
    in order (:func:`request_data`). A teammate's own task is never given here. The memory package calls it through
    the channel client; it is never in a cell's ``tools``. Refused when no request is known, or when its JSON is
    larger than a reply carries."""
    state = ctx.state
    text, data = getattr(state, "request_text", None), getattr(state, "request_data", None)
    if not isinstance(text, str) or not text:
        text = _memory_run_request()
        data = request_data(text) if text else None
    if not isinstance(text, str) or not text:
        text = None
    if text is None:
        return _refused(_refusal(ctx, "current_request", "unavailable", why="no current request is known"))
    reply = json.dumps({"text": text, "data": data})
    if len(reply.encode("utf-8")) > REQUEST_REPLY_BYTES:
        return _refused(_refusal(ctx, "current_request", "unavailable", why="the request is larger than a reply carries"))
    return _reply(reply)


def request(ctx: Any, args: dict) -> Any:
    """``tools.request()``: the exact text of the request the calling agent works on (:func:`current_request`'s
    text: for a teammate the run's request, which its task serves), in every run, with or without memory. Refused when
    no request is known."""
    state = ctx.state
    text = getattr(state, "request_text", None)
    if not isinstance(text, str) or not text:
        text = _memory_run_request()
    if not isinstance(text, str) or not text:
        return _refused(_refusal(ctx, "request", "unavailable", why="no current request is known"))
    if len(text.encode("utf-8")) > REQUEST_REPLY_BYTES:
        return _refused(_refusal(ctx, "request", "unavailable", why="the request is larger than a reply carries"))
    return _reply(text)


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
    "wait",
    "who",
]
