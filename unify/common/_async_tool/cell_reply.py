"""``UNIFY_REPLY_CHANNEL=code+text``: a cell sends the turn's reply.

As shipped the model answers only by replying with text. With the switch on,
model code in a cell can also call ``reply(text)``: the cell ends at the
call, and the loop ends the turn with exactly that text as the reply,
without another model call, as if the model had replied with it.

Pieces, in the order a reply travels:

* ``reply`` (``worker_child.Reply``) is a global of the actor's sandbox, in
  process and in the sandboxed worker. It refuses a non-``str`` argument and
  a call from inside a stored function, and raises ``worker_child.CellReply``
  (a ``BaseException``, so a model's ``except Exception`` does not swallow
  it), which ends the cell.
* The session executor catches it (in process, or as the worker reports it)
  and hands it to :func:`deliver`, which records it on the :class:`ReplySlot`
  of the loop that ran the cell, or refuses it with the reason.
* The loop (``loop.py``) finds the reply in its slot where it would ask the
  model for its next step, appends the reply as the assistant's message
  instead, and goes on exactly as for a text reply.

Each loop sets its own slot in its context (``None`` when it takes no cell
replies: the storage review, a request answered by ``final_response``), so a
nested loop never sees its parent's. Which kind of reply ended a turn is
recorded on the turn's final assistant message (``_reply_source``: "cell" or
"text"; ``_reply_from_value``: whether reply()'s argument was computed rather
than a string literal); unillm strips underscore keys before a request, so
the model never sees them.
"""

from __future__ import annotations

import ast
import contextlib
import contextvars
import dataclasses
from typing import Any, Iterator, Optional

#: The metadata keys on a turn's final assistant message.
SOURCE_KEY = "_reply_source"
FROM_VALUE_KEY = "_reply_from_value"

#: The method a literal ``reply("...")`` call is rewritten to.
LITERAL_METHOD = "_literal_reply"


def enabled() -> bool:
    """Whether ``UNIFY_REPLY_CHANNEL=code+text``."""
    from unify.settings import SETTINGS

    return getattr(SETTINGS, "UNIFY_REPLY_CHANNEL", "") == "code+text"


@dataclasses.dataclass
class ReplySlot:
    """One loop's cell reply: claimed by a cell, taken by the loop."""

    text: Optional[str] = None
    from_value: bool = False

    @property
    def replied(self) -> bool:
        return self.text is not None

    def take(self) -> Optional[dict]:
        """The assistant message a claimed reply becomes; clears the slot."""
        if self.text is None:
            return None
        msg = {"role": "assistant", "content": self.text, "tool_calls": None}
        stamp(msg, source="cell", from_value=self.from_value)
        self.text, self.from_value = None, False
        return msg

    def clear(self) -> None:
        self.text, self.from_value = None, False


_SLOT: contextvars.ContextVar[Optional[ReplySlot]] = contextvars.ContextVar(
    "unify_cell_reply_slot",
    default=None,
)
# Why a reply cannot be taken in the running call (execute_function), if so.
_REFUSED: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar(
    "unify_cell_reply_refused",
    default=None,
)

NOT_AVAILABLE = (
    "reply() is not available here: this session does not answer with a text "
    "reply (a storage review, or a request answered by `final_response`)"
)
ALREADY_REPLIED = "reply() was already called in this turn; a turn has one reply"


def bind(reply_channel: bool) -> Optional[contextvars.Token]:
    """Set the running loop's slot; ``None`` (nothing set) while the switch is off."""
    if not enabled():
        return None
    return _SLOT.set(ReplySlot() if reply_channel else None)


def unbind(token: Optional[contextvars.Token]) -> None:
    if token is not None:
        _SLOT.reset(token)


def current() -> Optional[ReplySlot]:
    return _SLOT.get()


@contextlib.contextmanager
def refused(reason: str) -> Iterator[None]:
    """Refuse reply() in the calls made inside the block, saying *reason*."""
    token = _REFUSED.set(reason)
    try:
        yield
    finally:
        _REFUSED.reset(token)


def refusal() -> Optional[str]:
    """Why reply() cannot be taken now, or ``None``."""
    reason = _REFUSED.get()
    if reason is not None:
        return reason
    slot = _SLOT.get()
    if slot is None:
        return NOT_AVAILABLE
    if slot.replied:
        return ALREADY_REPLIED
    return None


def precheck() -> None:
    """In process, raised at the call, so the cell sees why."""
    reason = refusal()
    if reason is not None:
        raise RuntimeError(reason)


def deliver(text: str, from_value: bool) -> Optional[str]:
    """Claim *text* as the turn's reply; the refusal, if it cannot be taken."""
    reason = refusal()
    if reason is not None:
        return reason
    slot = _SLOT.get()
    assert slot is not None
    slot.text, slot.from_value = text, bool(from_value)
    return None


def stamp(msg: dict, *, source: str, from_value: bool) -> None:
    """Record on a turn's final assistant message which reply ended it."""
    msg[SOURCE_KEY] = source
    msg[FROM_VALUE_KEY] = bool(from_value)


# ---------------------------------------------------------------------------
# Provenance: a reply of a literal string
# ---------------------------------------------------------------------------


def _is_literal(node: ast.AST) -> bool:
    """A string literal, or an f-string (or concatenation) of literals only."""
    if isinstance(node, ast.Constant):
        return isinstance(node.value, str)
    if isinstance(node, ast.JoinedStr):
        return all(isinstance(v, ast.Constant) for v in node.values)
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        return _is_literal(node.left) and _is_literal(node.right)
    return False


class _MarkLiteralReplies(ast.NodeTransformer):
    """``reply("...")`` -> ``getattr(reply, "_literal_reply", reply)("...")``.

    The call is unchanged when ``reply`` is something the model bound
    itself; ``Reply._literal_reply`` records that the text was written out
    in the call rather than computed."""

    def __init__(self) -> None:
        self.changed = False

    def visit_Call(self, node: ast.Call) -> ast.AST:  # noqa: N802
        self.generic_visit(node)
        if (
            isinstance(node.func, ast.Name)
            and node.func.id == "reply"
            and len(node.args) == 1
            and not node.keywords
            and _is_literal(node.args[0])
        ):
            self.changed = True
            node.func = ast.Call(
                func=ast.Name(id="getattr", ctx=ast.Load()),
                args=[
                    ast.Name(id="reply", ctx=ast.Load()),
                    ast.Constant(LITERAL_METHOD),
                    ast.Name(id="reply", ctx=ast.Load()),
                ],
                keywords=[],
            )
        return node


def mark_literal_replies(code: str) -> str:
    """*code* with each ``reply(<literal>)`` call marked; unchanged otherwise."""
    if "reply" not in code:
        return code
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return code
    marker = _MarkLiteralReplies()
    tree = marker.visit(tree)
    if not marker.changed:
        return code
    ast.fix_missing_locations(tree)
    return ast.unparse(tree)


# ---------------------------------------------------------------------------
# Run stats
# ---------------------------------------------------------------------------


def run_stats(runtime_state: Any) -> dict:
    """The counters the CLI reports beside the run's tokens."""
    return {
        "replies_from_cell": int(getattr(runtime_state, "replies_from_cell", 0)),
        "replies_from_value": int(getattr(runtime_state, "replies_from_value", 0)),
    }


__all__ = [
    "ALREADY_REPLIED",
    "FROM_VALUE_KEY",
    "NOT_AVAILABLE",
    "ReplySlot",
    "SOURCE_KEY",
    "bind",
    "current",
    "deliver",
    "enabled",
    "mark_literal_replies",
    "precheck",
    "refusal",
    "refused",
    "run_stats",
    "stamp",
    "unbind",
]
