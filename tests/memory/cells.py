"""Cells run in the actor's real sandboxed worker, for the memory tests that read what a cell's result or its audit
records tell the request (``hooks.tool_result``, ``hooks.worker_cell_done``).

:class:`Cells` keeps one actor interpreter per session id (:func:`tests.actor.code_act.sandbox_world.actor_interpreter`)
and answers each cell with the runtime's result as a mapping (``stdout`` as text parts, the interpreter's notes around
what the cell printed, as the actor's result has them; ``error``, ``result``,
``session_id``, ``session_created``), the form ``RequestRun.note_result`` reads. Close it with ``await cells.close()``.
"""

from __future__ import annotations

from typing import Any


class Cells:
    def __init__(self, *, limit: float = 60.0, shadow: dict | None = None) -> None:
        self.limit, self.shadow = limit, shadow
        self._interps: dict[int, Any] = {}

    async def execute(self, *, code: str, session_id: int = 0, **_ignored: Any) -> dict:
        from tests.actor.code_act.sandbox_world import actor_interpreter
        from unify.actor.execution.types import TextPart

        created = session_id not in self._interps
        if created:
            shadow = dict(self.shadow) if self.shadow is not None else None
            self._interps[session_id] = await actor_interpreter(shadow=shadow, limit=self.limit)
        res = await self._interps[session_id].run_cell(code)
        # as the actor's result (``in_code_tools``): the interpreter's notes (a restart, a timeout) are text around
        # what the cell printed; a timed-out cell has no error of its own
        out = [TextPart(text=n + "\n") for n in res.notes_before]
        out += [TextPart(text=res.output)] if res.output else []
        out += [TextPart(text=n + "\n") for n in res.notes_after]
        return {
            "stdout": out,
            "stderr": [],
            "error": res.error,
            "result": res.result,
            "session_id": session_id,
            "session_created": created,
        }

    async def close(self) -> None:
        interps, self._interps = list(self._interps.values()), {}
        for interp in interps:
            await interp.stop()
