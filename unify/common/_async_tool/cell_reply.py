"""The actor's answer from a cell: ``tools.answer(answer)`` gives the turn's answer.

The actor's loop (``tools_in_code``) sets a :class:`ReplySlot` in its context before its first model call. The
``answer`` handler (unify/actor/tool_handlers.py) fills the slot of the loop that ran the cell; where the loop would
ask the model for its next step, it finds the answer in its slot, appends it as the assistant's message instead, and
goes on exactly as for a text reply. Each loop sets its own slot, so a nested loop never sees its parent's.
"""

from __future__ import annotations

import contextvars
import dataclasses
from typing import Optional


@dataclasses.dataclass
class ReplySlot:
    """One loop's answer from a cell: filled by ``tools.answer``, taken by the loop."""

    text: Optional[str] = None

    @property
    def replied(self) -> bool:
        return self.text is not None

    def take(self) -> Optional[dict]:
        """The assistant message a given answer becomes; clears the slot."""
        if self.text is None:
            return None
        msg = {"role": "assistant", "content": self.text, "tool_calls": None}
        self.text = None
        return msg

    def clear(self) -> None:
        self.text = None


_SLOT: contextvars.ContextVar[Optional[ReplySlot]] = contextvars.ContextVar(
    "unify_cell_reply_slot",
    default=None,
)


def bind() -> contextvars.Token:
    """Set a new slot for the running loop."""
    return _SLOT.set(ReplySlot())


def unbind(token: Optional[contextvars.Token]) -> None:
    if token is not None:
        _SLOT.reset(token)


def current() -> Optional[ReplySlot]:
    return _SLOT.get()


__all__ = ["ReplySlot", "bind", "current", "unbind"]
