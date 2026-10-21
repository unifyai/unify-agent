"""The no-progress rule: an agent whose tool calls stop making progress is told so, then stopped.

The loop census of 7 October 2026 (87,147 recorded requests) found that
most loops inside one request are no-op narration. The model runs a cell
such as ``print('Request another demo.')``, ``print('')`` or ``pass`` instead
of replying. A rule that looks only for exact repeats misses it.

A model call makes *no progress* when every tool call it makes either

* runs a Python cell that does nothing: only ``pass``, comments, prints of
  constant text (constant f-strings included), bare constants, or nothing,
  with magics ignored (:func:`is_noop_cell`); or
* repeats one of the ``WINDOW`` model calls before it: the same tool with the
  same arguments (``thought`` dropped; a cell's code without comments,
  whitespace or magics), string and number literals ignored, and it got the
  same result (times, ids, durations and the step budget footer ignored).

A result that is not known yet (missing, empty, a pending placeholder) never
matches. A call made while other calls are still running is not counted. A
text reply, any other call and every user message start the count again.

The memory agents (the writer, the curator, their helpers and the librarian)
run their own loops, and :class:`AgentNoProgress` applies the rule to their
conversations: ``MEMORY_K`` no-progress calls in a row get one notice, twice
as many end the agent's round.
"""

from __future__ import annotations

import ast
import hashlib
import io
import json
import re
import tokenize
from dataclasses import dataclass, field
from typing import Any, Optional, Sequence

from .messages import is_loop_authored_message, is_non_final_tool_reply

# How many earlier model calls a repeat is compared with.
WINDOW = 2


# ── cells that do nothing ────────────────────────────────────────────────


def _strip_magics(code: str) -> str:
    return "\n".join(
        line for line in code.splitlines() if not line.lstrip().startswith("%")
    )


def _constant(node: ast.AST) -> bool:
    if isinstance(node, ast.Constant):
        return True
    return isinstance(node, ast.JoinedStr) and all(
        isinstance(value, ast.Constant) for value in node.values
    )


def is_noop_cell(code: Any) -> bool:
    """Whether a Python cell does nothing but print constant text."""
    text = _strip_magics("" if code is None else str(code))
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return False
    for statement in tree.body:
        if isinstance(statement, ast.Pass):
            continue
        if isinstance(statement, ast.Expr) and _constant(statement.value):
            continue
        value = getattr(statement, "value", None)
        if (
            isinstance(statement, ast.Expr)
            and isinstance(value, ast.Call)
            and isinstance(value.func, ast.Name)
            and value.func.id == "print"
            and all(_constant(arg) for arg in value.args)
            and all(_constant(keyword.value) for keyword in value.keywords)
        ):
            continue
        return False
    return True


# ── actions ──────────────────────────────────────────────────────────────

_STRING = re.compile(
    r"(?s)(\"\"\".*?\"\"\"|'''.*?'''|\"(?:\\.|[^\"\\])*\"|'(?:\\.|[^'\\])*')",
)
_NUMBER = re.compile(r"(?<![A-Za-z_])\d+(?:\.\d+)?")


def _blank_literals(text: str) -> str:
    return _NUMBER.sub("N", _STRING.sub("S", text))


def _code_tokens(code: str) -> str:
    """The code without comments, blank lines or layout."""
    try:
        tokens = tokenize.generate_tokens(io.StringIO(code).readline)
        kept = [
            token.string
            for token in tokens
            if token.type
            not in (
                tokenize.COMMENT,
                tokenize.NL,
                tokenize.NEWLINE,
                tokenize.INDENT,
                tokenize.DEDENT,
            )
            and token.string
        ]
        return " ".join(kept)
    except (tokenize.TokenError, IndentationError, SyntaxError):
        lines = [line for line in code.splitlines() if not line.strip().startswith("#")]
        return " ".join(" ".join(lines).split())


def _digest(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8", "replace")).hexdigest()


@dataclass(frozen=True)
class CallRecord:
    """One tool call as compared: exact and near action keys, no-op flag."""

    exact: str
    near: str
    noop: bool


def call_record(name: str, arguments: Any) -> CallRecord:
    """*name* called with *arguments* (a dict or its JSON text)."""
    if isinstance(arguments, str):
        try:
            args = json.loads(arguments) if arguments.strip() else {}
        except (ValueError, RecursionError):  # a model's arguments: malformed, or nested past the decoder's depth
            args = {"_raw": arguments}
    else:
        args = arguments or {}
    if not isinstance(args, dict):
        args = {"_value": args}
    args = {k: v for k, v in args.items() if k != "thought"}
    noop = False
    if name == "execute_code":
        code = args.pop("code", "")
        code = "" if code is None else str(code)
        language = str(args.get("language") or args.get("_language") or "python")
        noop = language.lower() == "python" and is_noop_cell(code)
        code_key = _code_tokens(_strip_magics(code))
        rest = json.dumps(args, sort_keys=True, default=str)
        exact = f"{name}|{code_key}|{rest}"
        near = f"{name}|{_blank_literals(code_key)}|{_blank_literals(rest)}"
    else:
        rest = " ".join(json.dumps(args, sort_keys=True, default=str).split())
        exact = f"{name}|{rest}"
        near = f"{name}|{_blank_literals(rest)}"
    return CallRecord(exact=_digest(exact), near=_digest(near), noop=noop)


# ── results ──────────────────────────────────────────────────────────────

_RESULT_SUBS = [
    (re.compile(r"\n*\[step budget\][^\n]*$"), ""),
    (
        re.compile(
            r"\d{4}-\d\d-\d\d[T ]\d\d:\d\d:\d\d(?:\.\d+)?(?:[+-]\d\d:?\d\d|Z)?",
        ),
        "TS",
    ),
    (
        re.compile(
            r"\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b",
        ),
        "UUID",
    ),
    (re.compile(r"0x[0-9a-fA-F]+"), "ADDR"),
    (re.compile(r"\b[0-9a-f]{8,}\b"), "HEX"),
    (re.compile(r"\b\d+(?:\.\d+)?\s*(?:ms|s|sec|secs|seconds)\b"), "DUR"),
    (re.compile(r'"(?:duration|elapsed|wall|took)\w*"\s*:\s*[\d.]+'), '"DUR"'),
    (re.compile(r"\s+"), " "),
]

_STEERABLE = re.compile(r"\[[^\]\n]*: steerable\]")


def _text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(
            (
                (
                    str(part.get("text") or "")
                    if part.get("type") == "text"
                    else json.dumps(part, sort_keys=True, default=str)
                )
                if isinstance(part, dict)
                else str(part)
            )
            for part in content
        )
    return "" if content is None else str(content)


def normalise_result(text: str) -> str:
    """A tool result as compared."""
    for pattern, replacement in _RESULT_SUBS:
        text = pattern.sub(replacement, text)
    return text.strip()


def _result(message: Optional[dict]) -> Optional[str]:
    """The normalised result a tool message carries; ``None`` if not known."""
    if message is None or is_non_final_tool_reply(message):
        return None
    text = _text(message.get("content"))
    if not text.strip() or _STEERABLE.search(text):
        return None
    return _digest(normalise_result(text))


# ── the tracker ──────────────────────────────────────────────────────────


@dataclass
class _Turn:
    calls: list[tuple[CallRecord, Optional[str]]] = field(default_factory=list)


def _is_completion_stub(message: dict) -> bool:
    """The loop's own assistant stub that carries a late async result."""
    calls = message.get("tool_calls") or []
    return bool(calls) and all(
        str(call.get("id") or "").endswith("_completed") for call in calls
    )


class Tracker:
    """Counts the no-progress model calls in a row of the current request."""

    def __init__(self, k: int, window: int = WINDOW) -> None:
        self.k = k
        self.window = window
        self.count = 0
        self._anchor: Optional[dict] = None
        self._seen = 0
        self._turns: list[_Turn] = []

    def _reset(self, anchor: Optional[dict]) -> None:
        self._anchor = anchor
        self._seen = 0
        self._turns = []
        self.count = 0

    def observe(self, messages: Sequence[dict], *, in_flight: bool = False) -> bool:
        """Count the model calls of the request not yet seen; ``True`` at K.

        *in_flight*: other calls are still running, so the calls seen now
        are not counted.
        """
        start = 0
        anchor: Optional[dict] = None
        for i in range(len(messages) - 1, -1, -1):
            message = messages[i]
            if message.get("role") == "user" and not is_loop_authored_message(
                message,
            ):
                start, anchor = i + 1, message
                break
        turns = [
            message
            for message in messages[start:]
            if message.get("role") == "assistant" and not _is_completion_stub(message)
        ]
        if anchor is not self._anchor or len(turns) < self._seen:
            self._reset(anchor)
        if len(turns) == self._seen:
            return False
        results: dict[str, Optional[dict]] = {}
        for message in messages[start:]:
            if message.get("role") == "tool" and message.get("tool_call_id"):
                results[str(message["tool_call_id"])] = message
        for message in turns[self._seen :]:
            turn = _Turn()
            for call in message.get("tool_calls") or []:
                function = call.get("function") or {}
                call_id = str(call.get("id") or "")
                reply = results.get(f"{call_id}_completed") or results.get(call_id)
                turn.calls.append(
                    (
                        call_record(
                            str(function.get("name") or ""),
                            function.get("arguments"),
                        ),
                        _result(reply),
                    ),
                )
            if turn.calls and not in_flight and self._no_progress(turn):
                self.count += 1
            else:
                self.count = 0
            self._turns.append(turn)
        self._seen = len(turns)
        return self.count >= self.k

    def _no_progress(self, turn: _Turn) -> bool:
        earlier = [
            (record, result)
            for previous in self._turns[-self.window :]
            for record, result in previous.calls
            if result is not None
        ]
        for record, result in turn.calls:
            if record.noop:
                continue
            if result is None or not any(
                record.near == other.near and result == other_result
                for other, other_result in earlier
            ):
                return False
        return True


# ── what the stop says ───────────────────────────────────────────────────


# ── agents that run their own loop ───────────────────────────────────────

#: K for the memory agents (the writer, the curator and their helpers), which run their own loops rather than the
#: actor's: K no-progress model calls in a row get one notice, 2K end the agent's round. Five is past the one or two
#: retries a refused call reasonably gets, and short enough that a loop is noticed within a few minutes and a few
#: calls' cost; a model that changes approach after the notice starts the count again. A recorded writer pass ran
#: one cell that rewrote a file unchanged 98 times in a row, until its wall clock.
MEMORY_K = 5


def terminal_tool(agent: str) -> str:
    """The registry's terminal tool for *agent*: ``finish`` for the writer and the curator, ``done`` for a helper."""
    from unify.common import tool_registry

    for name, spec in tool_registry.registry().items():
        if spec.kind == "terminal" and agent in spec.roles:
            return name
    raise KeyError(f"no terminal tool for {agent}")


class AgentNoProgress:
    """The no-progress rule for an agent that runs its own loop over an OpenAI-format conversation.

    The agent calls :meth:`after_call` once after each of its model calls, when that call's tool results are in
    its conversation. The calls are counted by :class:`Tracker` from the agent's own messages. With tools called from code, a tool call is an ``execute_code`` cell whose in-code calls appear in
    its output, so the same rule applies to it. At ``k`` no-progress calls in a row one notice is appended to the
    conversation; at ``2 * k`` the agent's round ends. The notices are left out of what the tracker reads, so a
    notice is not taken for a new request; any other user message (a repair round's, a nudge after a text reply)
    starts the count again.
    """

    def __init__(self, terminal: str, *, in_code: bool = False, k: Optional[int] = None) -> None:
        self.k = MEMORY_K if k is None else int(k)
        self.terminal = f"tools.{terminal}" if in_code else terminal
        self.tracker = Tracker(2 * self.k)
        self.calls = 0  # the agent's model calls read so far
        self.notices = 0
        self.ended = False
        self.at_call: Optional[int] = None  # the model call of the last notice, or of the end
        self._notes: list[dict] = []

    @property
    def notice(self) -> str:
        return (
            f"You have repeated this call {self.k} times with the same result (or ran a cell that does nothing). "
            f"Change approach, or call {self.terminal} and keep the item as a draft."
        )

    def after_call(self, messages: list, *, in_flight: bool = False, act: bool = True) -> Optional[str]:
        """Read the model call just answered in *messages*. ``"notice"``: the notice was appended to *messages*;
        ``"end"``: the agent's round ends now; None otherwise.

        *in_flight*: other work of the agent (a helper it started) is still running, so the call is not counted.
        *act* False: the agent's round is ending anyway (it finished or was stopped): the call is read, nothing is
        appended and nothing ends.
        """
        self.calls += 1
        view = [m for m in messages if not any(m is note for note in self._notes)]
        self.tracker.observe(view, in_flight=in_flight)
        if not act:
            return None
        if self.tracker.count >= 2 * self.k:
            self.ended, self.at_call = True, self.calls
            return "end"
        if self.tracker.count == self.k:
            note = {"role": "user", "content": self.notice}
            messages.append(note)
            self._notes.append(note)
            self.notices += 1
            self.at_call = self.calls
            return "notice"
        return None

    def note(self, what: str) -> str:
        """The note line for *what* (``"notice"`` or ``"end"``)."""
        if what == "end":
            return (
                f"no progress: {2 * self.k} model calls in a row made no progress (each ran a cell that does nothing, "
                "or repeated one of the two calls before it and got the same result), so the round ended at model "
                f"call {self.at_call}"
            )
        return (
            f"no progress: {self.k} model calls in a row made no progress; a notice was given at model call "
            f"{self.at_call}"
        )

    def report(self) -> Optional[dict]:
        """The evidence row: None when no notice was given."""
        if not self.notices and not self.ended:
            return None
        return {"k": self.k, "notices": self.notices, "ended": self.ended, "at_call": self.at_call}


__all__ = [
    "AgentNoProgress",
    "CallRecord",
    "MEMORY_K",
    "Tracker",
    "WINDOW",
    "call_record",
    "is_noop_cell",
    "normalise_result",
    "terminal_tool",
]
