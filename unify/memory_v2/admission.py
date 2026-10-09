"""What a cover must point at, per action kind (spec §F3; the gate's G2).

An environment function's ``covers`` name recorded actions ``[episode_id, action_index]``. G2 checks only
that each is a real recorded observation of the right kind on the item's own channel; the tests the item
ships (red→green, regression) are what check that the item agrees with those observations.

=========  ==============================================================================================
Kind       A cover must be
=========  ==============================================================================================
tool       a call with status ``ok`` and a recorded response (unchanged from v0)
shell      a command with a recorded output tail: ``response = {"tail": str, ...}`` (any exit code: a
           failing run is an observation too)
worktree   a read or write with status ``ok`` and a recorded file blob: ``response["blob_before"]`` or
           ``response["blob_after"]`` is a blob id held by the blob store
dialogue   an action with status ``ok`` and a recorded observation (``response`` is not None or empty)
=========  ==============================================================================================

A tool, worktree or dialogue action the environment rejected (status ``error`` with its recorded error
text or error response; :func:`is_rejection`) is also a cover: it is how an item shows that the
environment itself restricts a value (spec §F3a). The gate requires at least one cover that is not such a
rejection.

In v2 the action's memory channel (:func:`.episodes.env_channel` of its kind and channel key) must be the
item's channel. v2.1 has no channel rule (D42).
"""

from __future__ import annotations

from typing import Callable

from .episodes import ACTION_KINDS, Action, env_channel

_BLOB_FIELDS = ("blob_before", "blob_after")


def is_rejection(a: Action) -> bool:
    """Whether the environment rejected *a*: a nonzero exit (shell), else status ``error`` with a record of it."""
    if getattr(a, "kind", "tool") == "shell":
        r = a.response
        code = r.get("exit_code") if isinstance(r, dict) else None
        return isinstance(code, int) and not isinstance(code, bool) and code != 0
    recorded = (
        isinstance(a.error, str) and a.error.strip() != ""
    ) or a.response not in (
        None,
        "",
    )
    return a.status == "error" and recorded


def _tool(a: Action, has_blob: Callable[[str], bool]) -> str | None:
    if is_rejection(a):
        return None
    if a.status != "ok" or a.response is None:
        return "not a recorded successful call or rejection"
    return None


def _shell(a: Action, has_blob: Callable[[str], bool]) -> str | None:
    r = a.response
    if not isinstance(r, dict) or not isinstance(r.get("tail"), str):
        return "not a recorded shell command with an output tail"
    return None


def _worktree(a: Action, has_blob: Callable[[str], bool]) -> str | None:
    if is_rejection(a):
        return None
    r = a.response
    if a.status != "ok" or not isinstance(r, dict):
        return "not a recorded file read or write"
    if not any(has_blob(r.get(k)) for k in _BLOB_FIELDS):
        return "a file action without a recorded blob"
    return None


def _dialogue(a: Action, has_blob: Callable[[str], bool]) -> str | None:
    if is_rejection(a):
        return None
    if a.status != "ok" or a.response is None or a.response == "":
        return "not a recorded dialogue action with an observation"
    return None


_RULES = {"tool": _tool, "shell": _shell, "worktree": _worktree, "dialogue": _dialogue}
assert set(_RULES) == ACTION_KINDS


def cover_problem(
    a: Action | None,
    channel: str | None,
    has_blob: Callable[[str], bool],
    *,
    channel_rule: bool = True,
) -> str | None:
    """Why recorded action *a* cannot be a cover of an item on memory channel *channel*, or None.

    With *channel_rule* False (memory v2.1, spec §4.1 and D42) the action's channel is not compared: a cover is
    any real recorded observation of its kind, and the item's source channel lives in its record (P5).
    """
    if a is None:
        return "not a recorded action"
    kind = getattr(a, "kind", "tool")
    rule = _RULES.get(kind)
    if rule is None:
        return f"an action of unknown kind {kind!r}"[:120]
    problem = rule(a, has_blob)
    if problem is not None:
        return f"{problem} ({kind})"
    if not channel_rule:
        return None
    mapped = env_channel(kind, a.channel)
    if mapped != channel:
        return f"a {kind} action on {a.channel}"[:200]
    return None
