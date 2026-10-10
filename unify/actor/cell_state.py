"""``UNIFY_STATEFUL_CELLS``: every ``execute_code`` cell runs in the task's session.

As shipped ``execute_code`` takes ``state_mode`` (an untyped optional string),
``session_id`` and ``session_name``, and four tools manage sessions. Omitted, a
cell runs in session 0, the task's persistent session; but a model that fills
in every argument writes a mode into every cell, and with the tools around it
(``execute_function`` defaults to ``"stateless"``, ``inspect_state`` offers to
"run stateless") it writes ``"stateless"``: nothing one cell computed is there
for the next, and values are typed again by hand. An enum led by the default,
described as such, still drew "stateless" in 15 of 18 replayed first cells.

With the switch on the cell tool has no mode to choose: every cell runs in
session 0, like a notebook. What a cell keeps is the model's to decide in
Python (its names, ``del``, functions). The session tools are not offered,
since there is one session; ``execute_function`` keeps its own ``state_mode``
(a stored function runs ``"stateless"`` by default, or in that session) and
loses the session selectors.
"""

from __future__ import annotations

import inspect
import re
from typing import Any, Dict

__all__ = [
    "SESSION_TOOLS",
    "correct_tools",
    "describe_execute_code",
    "describe_execute_function",
    "enabled",
]

#: The tools that list, inspect and close sessions.
SESSION_TOOLS = (
    "list_sessions",
    "inspect_state",
    "close_session",
    "close_all_sessions",
)
_SELECTORS = ("session_id", "session_name")

# execute_code's two session bullets as shipped, up to the next bullet.
_SESSION_BULLETS = re.compile(
    r"- \*\*state_mode\*\*: omit it and the cell runs.*?"
    r"- \*\*session_id/session_name\*\*:.*?\n(?P<i>[ \t]*)(?=- \*\*)",
    re.DOTALL,
)
# The workspace variant's bash bullet (unify/actor/workspace_tools.py).
_BASH_PER_SESSION = re.compile(r"a persistent bash session per session_id:")
_BASH_MODES = re.compile(
    r"\s+Bash takes\s+state_mode \"stateful\" or \"stateless\", not \"read_only\"\.",
)
_IN_A_STATE_MODE = re.compile(r" in a specified state mode\.")
_FUNCTION_MODES = re.compile(
    r"``state_mode`` / ``session_id`` /(?P<ws>\s+)``session_name`` keep ``execute_code``"
    r" semantics, except\s+``state_mode`` here defaults to ``\"stateless\"``\.",
)


def enabled() -> bool:
    from unify.settings import SETTINGS

    return bool(SETTINGS.UNIFY_STATEFUL_CELLS)


def describe_execute_code(doc: str) -> str:
    """*doc* (``execute_code``'s description) for cells that all run in session 0."""

    def bullet(m: "re.Match[str]") -> str:
        i = m.group("i")
        return (
            "- **Session**: every cell runs in this task's persistent session,\n"
            f"{i}  like a notebook: variables, imports and definitions carry\n"
            f"{i}  over from one cell to the next.\n"
            f"{i}"
        )

    doc = _SESSION_BULLETS.sub(bullet, doc, count=1)
    doc = _BASH_PER_SESSION.sub("a persistent bash session:", doc, count=1)
    doc = _BASH_MODES.sub("", doc, count=1)
    return _IN_A_STATE_MODE.sub(".", doc, count=1)


def describe_execute_function(doc: str) -> str:
    """*doc* (``execute_function``'s description) without the session selectors."""

    def modes(m: "re.Match[str]") -> str:
        ws = m.group("ws")
        return (
            f'``state_mode``:{ws}``"stateless"`` (the default) runs the function in a'
            f'{ws}fresh namespace; ``"stateful"`` runs it in the session'
            f"{ws}``execute_code`` cells run in, so it sees and keeps their"
            f'{ws}variables; ``"read_only"`` runs it there and discards its'
            f"{ws}changes."
        )

    return _FUNCTION_MODES.sub(modes, doc, count=1)


def _without(fn: Any, names: tuple) -> None:
    sig = inspect.signature(fn)
    if not any(n in sig.parameters for n in names):
        return
    fn.__signature__ = sig.replace(
        parameters=[p for p in sig.parameters.values() if p.name not in names],
    )


def correct_tools(tools: Dict[str, Any]) -> None:
    """Apply the switch to the actor's tools, in place.

    ``execute_code`` shows no ``state_mode``, ``session_id`` or
    ``session_name`` (the signature the loop reads leaves them out; the
    function is unchanged and runs an omitted mode in session 0), and
    ``execute_function`` no session selector; both descriptions say so; the
    session tools are removed. The tools are built per actor, so nothing
    reaches another actor.
    """
    from unify.common.tool_spec import ToolSpec

    def fn_of(name: str) -> Any:
        tool = tools.get(name)
        return tool.fn if isinstance(tool, ToolSpec) else tool

    code = fn_of("execute_code")
    if code is not None:
        _without(code, ("state_mode", *_SELECTORS))
        if code.__doc__:
            code.__doc__ = describe_execute_code(code.__doc__)
    function = fn_of("execute_function")
    if function is not None:
        _without(function, _SELECTORS)
        if function.__doc__:
            function.__doc__ = describe_execute_function(function.__doc__)
    for name in SESSION_TOOLS:
        tools.pop(name, None)
