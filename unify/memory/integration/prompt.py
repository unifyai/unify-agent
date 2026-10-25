"""The memory section at the end of the cached system prompt.

The actor asks the librarian which stored items fit (``memory.ask`` in a cell), so the section is the librarian's
constant pointer (:data:`..library_prompts.GUIDE_LIBRARIAN`) and the line saying where the library copy is: the same
bytes for every library, whatever it holds, so the prompt's prefix never changes as the library grows. The hooks
append it last in the system prompt, after the clock line (:func:`.hooks.system_prompt`).
"""

from __future__ import annotations

from pathlib import Path

from .. import library_prompts


def location_line(checkout: Path) -> str:
    """Names the copy where the cells' sandbox binds it (``unify.sandbox.LIBRARY_MOUNT``), never *checkout*, the host
    path, which can name the run. ``tools.read``, ``tools.grep`` and ``memory.get`` resolve a path under it to the
    request's copy (``unify.actor.tool_handlers.library_view``)."""
    from unify.sandbox import LIBRARY_MOUNT

    return f"Library files: `{LIBRARY_MOUNT}`, holding INDEX.md, links.json, memory/ and notes/.\n"


def render_memory_section(checkout: Path) -> tuple[str, dict]:
    """The actor's memory section, and what it shows (:func:`..analysis.use.record_shown`: the pointer alone)."""
    from ..analysis.use import record_shown

    text = library_prompts.GUIDE_LIBRARIAN + location_line(Path(checkout))
    return text, record_shown(text, channels=(), items=(), renderer="librarian_pointer")
