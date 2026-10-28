"""A scripted model's call of one of a memory agent's tools, made as those agents make every call: from a code cell.

Each memory agent's one model-API tool is ``execute_code``; its other tools are calls from its cells (``tools.<name>``).
:func:`as_cell` turns a call of tool *name* with *args* into the ``execute_code`` arguments of a cell that makes it and
prints its reply, or the refusal or error it raised, exactly as the tool gave it.
"""

from __future__ import annotations

from typing import Any


def as_cell(name: str, args: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    """``("execute_code", {"code": ...})`` for a call of *name* with *args*; ``execute_code`` itself unchanged."""
    if name == "execute_code":
        return name, args
    code = (
        "try:\n"
        f"    print(tools.{name}(**{args!r}))\n"
        "except Exception as _exc:\n"
        "    print(_exc)\n"
    )
    return "execute_code", {"code": code}
