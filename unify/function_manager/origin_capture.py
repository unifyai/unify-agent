"""The Python and bash code cells a session ran, read from its trajectory.

The storage review's helpers read them: ``session_source`` stores a function
the session defined by its name, and ``escape_drift`` checks a stored literal
against the cells' own.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Mapping, Sequence

CODE_TOOLS = frozenset({"execute_code"})


def _content(message: Mapping[str, Any]) -> str:
    content = message.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(
            str(part.get("text") or "") if isinstance(part, dict) else str(part)
            for part in content
        )
    return "" if content is None else json.dumps(content, default=str)


def _code_cells(
    trajectory: Sequence[Mapping[str, Any]],
) -> List[tuple[int, str, str, str]]:
    """``(index of the result, code, language, output)`` for each code cell, in order."""
    from unify.actor import notebook_cells

    notebook = notebook_cells.enabled()
    calls: Dict[str, tuple[str, str]] = {}
    cells: List[tuple[int, str, str, str]] = []
    for index, message in enumerate(trajectory):
        if not isinstance(message, Mapping):
            continue
        if message.get("role") == "assistant":
            for call in message.get("tool_calls") or []:
                fn = (call or {}).get("function") or {}
                if fn.get("name") not in CODE_TOOLS:
                    continue
                try:
                    args = json.loads(fn.get("arguments") or "{}")
                except (TypeError, ValueError):
                    continue
                if isinstance(args, dict) and isinstance(args.get("code"), str):
                    code, language = args["code"], str(args.get("language") or "python")
                    if "language" not in args and notebook:
                        # UNIFY_CODE_PROJECTION=notebook: a %%bash first line.
                        language, code = notebook_cells.language_and_code(code)
                    calls[str(call.get("id"))] = (code, language)
        elif message.get("role") == "tool":
            found = calls.pop(str(message.get("tool_call_id")), None)
            if found is not None:
                cells.append((index, found[0], found[1], _content(message)))
    return cells


__all__ = ["CODE_TOOLS"]
