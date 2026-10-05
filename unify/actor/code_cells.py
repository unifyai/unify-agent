"""What the code-cell tool asks for: ``UNIFY_CODE_ONLY_CELLS``.

As shipped ``execute_code`` requires a ``thought``, "a brief, first-person,
one-sentence explanation ... shown to the user as the rationale for this
step", and takes ``code`` as an optional, nullable argument. At low effort
that schema invites announcing instead of computing: in the Python-tool-mode
ARC LOW runs 54-66% of the cells were narration (a printed sentence, a
comment, ``None``), the hypothesis went into ``thought``, and 60-70% of the
requests used no reasoning tokens. prime-agent's code tool takes only
``code`` and narrates in 0-7% of its cells.

``UNIFY_CODE_ONLY_CELLS``: ``code`` is the cell tool's only required argument
and the tool takes no ``thought``; without a primitives environment it does
not offer ``include_parent_chat_context`` either (only primitives read the
conversation it passes). The function behind the tool is unchanged: it is
called with an empty thought. A call that still passes ``thought`` is refused
by the tool loop's unknown-argument rule, naming the tool's parameters.

The switch changes only the actor's own copy of the tool (the tools are built
per actor), on the JSON surface and, since the core surface copies the same
function, on ``UNIFY_TOOL_SURFACE=core``.
"""

from __future__ import annotations

import dataclasses
import functools
import inspect
from typing import Any, Callable, Dict, Mapping

__all__ = [
    "code_only",
    "correct_tools",
]


def _setting(name: str) -> bool:
    from unify.settings import SETTINGS

    return bool(getattr(SETTINGS, name, False))


def code_only() -> bool:
    return _setting("UNIFY_CODE_ONLY_CELLS")


# ---------------------------------------------------------------------------
# UNIFY_CODE_ONLY_CELLS
# ---------------------------------------------------------------------------


def _code_only_tool(fn: Callable[..., Any], *, keep_parent_context: bool) -> Any:
    """*fn* (``execute_code``) as a tool that takes ``code`` first and no ``thought``.

    The signature the loop reads (and the schema is built from) has ``code``
    as a required string, no ``thought`` and, unless *keep_parent_context*,
    no ``_parent_chat_context``; everything else, hidden plumbing included, is
    passed through to *fn* with an empty thought.
    """
    sig = inspect.signature(fn)
    params = []
    for p in sig.parameters.values():
        if p.name == "thought":
            continue
        if p.name == "_parent_chat_context" and not keep_parent_context:
            continue
        if p.name == "code":
            p = p.replace(default=inspect.Parameter.empty, annotation=str)
        params.append(p)

    @functools.wraps(
        fn,
        assigned=("__module__", "__name__", "__qualname__", "__doc__"),
        updated=(),
    )
    async def execute_code(code: str, *args: Any, **kwargs: Any) -> Any:
        return await fn("", code, *args, **kwargs)

    execute_code.__signature__ = sig.replace(parameters=params)
    annotations = {
        k: v for k, v in getattr(fn, "__annotations__", {}).items() if k != "thought"
    }
    annotations["code"] = str
    execute_code.__annotations__ = annotations
    # get_type_hints resolves the annotations in *fn*'s module.
    execute_code.__wrapped__ = fn
    return execute_code


# ---------------------------------------------------------------------------
# The actor's tools
# ---------------------------------------------------------------------------


def correct_tools(
    tools: Dict[str, Any],
    environments: Mapping[str, Any],
) -> None:
    """Apply the switch to the actor's ``execute_code``, in place."""
    from unify.common.tool_spec import ToolSpec

    tool = tools.get("execute_code")
    if tool is None or not code_only():
        return
    fn = _code_only_tool(
        tool.fn if isinstance(tool, ToolSpec) else tool,
        keep_parent_context="primitives" in (environments or {}),
    )
    if isinstance(tool, ToolSpec):
        tools["execute_code"] = dataclasses.replace(tool, fn=fn)
    else:
        tools["execute_code"] = fn
