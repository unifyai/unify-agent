"""``UNIFY_CAPTURE_ACCEPTED``: keep the code behind a session's answer, and the call that gave it.

On the 5 Oct ARC reuse-chain diagnosis (research artifact
``overhaul-v1-screen/arc-reuse-chain-diagnosis.md``) 80 of 192 repeat visits had
no stored function although an earlier visit had been solved: the storage
review kept a note, or its gate called the work "a one-off grid", and of the
functions that were stored most left the rule's deciding value as an argument
the next caller guessed wrong. The review never saw which code produced the
answer, and the library never recorded how that code was called.

This module does the plumbing; the review decides:

- :func:`find_answer_cell` finds the code cell whose output the session's
  answer repeats: the answer is the reply the checked outcome arrived on
  (``UNIFY_OUTCOME``) or else the session's last reply with at least
  :data:`MIN_REPLY_TOKENS` letter and digit runs, and a cell's output repeats
  it when one contiguous run of the cell's output tokens covers all but
  :func:`_slack` of the reply's tokens. Nothing about any reply or request
  format is parsed;
- :func:`review_note` and :func:`gate_note` say so to the storage review and
  its gate: which cell, the checker's verdict when one was posted, and that
  the code may be kept whole as an entry-point function;
- while that review runs (:func:`reviewing`), a function it adds or updates
  that can be called with the cell's literal values (its top-level
  ``name = <literal>`` assignments, by parameter name, or the one required
  parameter) is run on them as a case replay runs it
  (:mod:`~unify.function_manager.store_cases`: no environment, network or
  model, :data:`~unify.function_manager.store_cases.REPLAY_TIMEOUT_S` per
  run, at most :data:`MAX_TRIES` runs), and a call whose return repeats the
  answer is recorded as the function's case (``UNIFY_FUNCTION_CASES``), so
  search results show how it was called. Nothing is compared with examples
  the task gave: only with the session's own answer.

With the switch off nothing here runs.
"""

from __future__ import annotations

import ast
import contextlib
import contextvars
import inspect
import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Mapping, Optional, Sequence

logger = logging.getLogger(__name__)

MIN_REPLY_TOKENS = 8
"""A reply with fewer letter and digit runs (a bare action word) is not an answer to link."""

MAX_SLACK = 12
"""The most reply tokens a linked output may leave uncovered (the reply's own wording)."""

MAX_TRIES = 3
"""Runs of one stored function on the cell's values, at most."""

MAX_CODE_CHARS = 6000
"""Characters of the cell's code the review is shown (head and tail around an elision)."""

MAX_BINDING_CHARS = 64_000
"""A literal whose repr is longer is not used as an argument."""

CODE_TOOLS = frozenset({"execute_code"})
"""The tools whose calls are code cells."""

_TOKEN = re.compile(r"[^\W\d_]+|\d+")
_SEP = "\x1f"


def enabled() -> bool:
    from unify.settings import SETTINGS

    return bool(getattr(SETTINGS, "UNIFY_CAPTURE_ACCEPTED", False))


def tokens(text: Any) -> List[str]:
    """The lower-cased runs of letters and of digits of *text* (JSON for non-strings)."""
    if not isinstance(text, str):
        try:
            text = json.dumps(text, default=str)
        except (TypeError, ValueError):
            text = str(text)
    return _TOKEN.findall(text.lower())


def _slack(n: int) -> int:
    return min(MAX_SLACK, max(3, n // 10))


def repeats(answer: Sequence[str], output: Sequence[str]) -> bool:
    """Whether one contiguous run of *output* covers all but :func:`_slack` of *answer*'s tokens."""
    n = len(answer)
    if n < MIN_REPLY_TOKENS or not output:
        return False
    need = n - _slack(n)
    hay = _SEP + _SEP.join(output) + _SEP
    for length in range(n, need - 1, -1):
        for start in range(0, n - length + 1):
            if (_SEP + _SEP.join(answer[start : start + length]) + _SEP) in hay:
                return True
    return False


@dataclass(frozen=True)
class AnswerCell:
    """The code cell a session's answer repeats."""

    answer: str
    code: str
    output: str
    language: str
    bindings: Mapping[str, Any] = field(default_factory=dict)

    @property
    def answer_tokens(self) -> List[str]:
        return tokens(self.answer)


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


def literal_bindings(code: str) -> Dict[str, Any]:
    """The cell's top-level ``name = <literal>`` values (the last of each name)."""
    try:
        tree = ast.parse(code)
    except (SyntaxError, ValueError):
        return {}
    out: Dict[str, Any] = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target, value = node.targets[0], node.value
        elif isinstance(node, ast.AnnAssign) and node.value is not None:
            target, value = node.target, node.value
        else:
            continue
        if not isinstance(target, ast.Name):
            continue
        try:
            literal = ast.literal_eval(value)
        except (ValueError, TypeError, SyntaxError, MemoryError, RecursionError):
            continue
        try:
            if len(repr(literal)) > MAX_BINDING_CHARS:
                continue
        except Exception:  # noqa: BLE001 - a repr that fails is not used
            continue
        out[target.id] = literal
    return out


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


def find_answer_cell(
    trajectory: Sequence[Mapping[str, Any]],
    *,
    answer: Optional[str] = None,
) -> Optional[AnswerCell]:
    """The latest code cell whose output the session's answer repeats, or ``None``.

    *answer* is the reply the checked outcome arrived on, when one did;
    otherwise the session's last reply (an assistant message without tool
    calls) with at least :data:`MIN_REPLY_TOKENS` tokens. Only cells whose
    result came before that reply count.
    """
    if not isinstance(trajectory, Sequence):
        return None
    end = len(trajectory)
    reply = answer if isinstance(answer, str) and answer.strip() else None
    if reply is None:
        for index in range(len(trajectory) - 1, -1, -1):
            message = trajectory[index]
            if (
                isinstance(message, Mapping)
                and message.get("role") == "assistant"
                and not message.get("tool_calls")
                and len(tokens(_content(message))) >= MIN_REPLY_TOKENS
            ):
                reply, end = _content(message), index
                break
    else:
        for index in range(len(trajectory) - 1, -1, -1):
            message = trajectory[index]
            if (
                isinstance(message, Mapping)
                and message.get("role") == "assistant"
                and _content(message).strip() == reply.strip()
            ):
                end = index
                break
    if reply is None:
        return None
    wanted = tokens(reply)
    if len(wanted) < MIN_REPLY_TOKENS:
        return None
    for index, code, language, output in reversed(_code_cells(trajectory)):
        if index >= end:
            continue
        if repeats(wanted, tokens(output)):
            return AnswerCell(
                answer=reply,
                code=code,
                output=output,
                language=language,
                bindings=(
                    literal_bindings(code) if language.lower() == "python" else {}
                ),
            )
    return None


def _clipped(code: str) -> str:
    if len(code) <= MAX_CODE_CHARS:
        return code
    half = MAX_CODE_CHARS // 2
    return code[:half] + "\n# … [code omitted] …\n" + code[-half:]


def _verdict_line(outcome: Optional[Mapping[str, Any]]) -> str:
    solved = (outcome or {}).get("solved")
    if solved is True:
        return "The environment's checker accepted that answer."
    return (
        "No checked outcome was posted: whether the answer was accepted is "
        "for you to read from the conversation."
    )


def review_note(cell: AnswerCell, outcome: Optional[Mapping[str, Any]] = None) -> str:
    """The storage review's section on the code behind the session's answer."""
    names = ", ".join(f"`{name}`" for name in cell.bindings)
    inputs = f" (the values the cell wrote in: {names})" if names else ""
    return (
        "## Code That Produced The Answer\n\n"
        "The session's last answer repeats the output of this code cell:\n\n"
        f"```{cell.language.lower()}\n{_clipped(cell.code)}\n```\n\n"
        f"{_verdict_line(outcome)} If it was accepted, this code is a whole "
        "procedure for this kind of request, and you may store it as one "
        "entry-point function that takes the request's input"
        f"{inputs} and returns the answer, with the choices the code made "
        "kept in its body rather than left as arguments for a later caller "
        "to choose. A function you add or update now that can be called with "
        "those values is run once on them, with no environment; if it "
        "returns this answer, that call is kept as its first recorded case, "
        "so later searches show how it was called.\n\n"
    )


def gate_note(cell: AnswerCell, outcome: Optional[Mapping[str, Any]] = None) -> str:
    """The review gate's line on the code behind the session's answer."""
    return (
        "## Answer from code\n\n"
        "The session's last answer repeats the output of a code cell it ran "
        f"(shown in the transcript). {_verdict_line(outcome)}"
    )


# ---------------------------------------------------------------------------
# Recording the answering call during the review
# ---------------------------------------------------------------------------

_REVIEWING: contextvars.ContextVar[Optional[AnswerCell]] = contextvars.ContextVar(
    "unify_capture_accepted",
    default=None,
)


@contextlib.contextmanager
def reviewing(cell: Optional[AnswerCell]) -> Iterator[None]:
    """While the block runs (and in the tasks it starts), functions stored are tried on *cell*."""
    token = _REVIEWING.set(cell)
    try:
        yield
    finally:
        _REVIEWING.reset(token)


def current() -> Optional[AnswerCell]:
    return _REVIEWING.get() if enabled() else None


def _parameters(source: str, name: str) -> Optional[List[inspect.Parameter]]:
    from .store_trust import source_signature

    signature = source_signature(source, name)
    if signature is None:
        return None
    return list(signature.parameters.values())


def argument_sets(
    source: str,
    name: str,
    bindings: Mapping[str, Any],
) -> List[Dict[str, Any]]:
    """The calls of ``name`` to try on *bindings*: ``[{"args": [...], "kwargs": {...}}, ...]``.

    Every required parameter matched by name; else, for a function with one
    required parameter, each value in turn, the longest first. At most
    :data:`MAX_TRIES`.
    """
    params = _parameters(source, name)
    if params is None or not bindings:
        return []
    kinds = (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
    named = [p for p in params if p.kind in kinds]
    required = [p for p in named if p.default is inspect.Parameter.empty]
    if any(
        p.kind is inspect.Parameter.POSITIONAL_ONLY
        and p.default is inspect.Parameter.empty
        for p in params
    ):
        return []
    out: List[Dict[str, Any]] = []
    if required and all(p.name in bindings for p in required):
        out.append(
            {
                "args": [],
                "kwargs": {
                    p.name: bindings[p.name] for p in named if p.name in bindings
                },
            },
        )
    elif len(required) == 1:
        ordered = sorted(
            bindings.items(),
            key=lambda item: -len(repr(item[1])),
        )
        for _, value in ordered:
            out.append({"args": [], "kwargs": {required[0].name: value}})
    return out[:MAX_TRIES]


def _run(
    fm: Any,
    *,
    name: str,
    source: str,
    depends_on: List[str],
    call: Dict[str, Any],
):
    """Run ``name`` on *call* as a case replay runs it: ``(how, value)``."""
    from . import store_cases as sc

    world = sc._World(
        sc.Case(
            case_id=0,
            function_id=0,
            kind=sc.PASS,
            status=sc.ACTIVE,
            source_hash="",
            call=call,
            args_shown="",
            result=None,
            error=None,
            trace=(),
            trace_complete=True,
            session=None,
            outcome=None,
            retired_why=None,
            recorded_at="",
        ),
    )
    token = sc._REPLAYING.set(True)
    try:
        candidate = fm._verify_candidate(
            name=name,
            source=source,
            depends_on=depends_on,
        )
        fn = candidate.load(sc._ReplayPrimitives(world), sc._replay_globals(world))
    except BaseException as exc:  # noqa: BLE001 - a source that does not load here
        return "unloadable", exc
    finally:
        sc._REPLAYING.reset(token)
    how, value = sc._run_bounded(fn, call, world)
    if world.verdict is not None:
        return "refused", world.verdict[1]
    return how, value


def _ending(how: str, value: Any) -> str:
    from . import store_cases as sc

    if how == "timeout":
        return f"it ran over {sc.REPLAY_TIMEOUT_S:g}s"
    if how == "refused":
        return str(value)
    if how == "unloadable":
        return f"it does not load: {sc._error_text(value)}"
    return f"it raised {sc._error_text(value)}"


def record_answering_call(fm: Any, name: str) -> Optional[str]:
    """During a review offered an answer cell: record the stored ``name``'s call that gives the answer.

    Returns a note for the review (what was recorded, or why nothing was),
    or ``None`` when there was nothing to try. Never raises.
    """
    cell = current()
    if cell is None:
        return None
    from . import store_cases as sc

    if not sc.enabled():
        return None
    try:
        row = fm._get_function_data_by_name(name=name)
        if not row or row.get("is_primitive") or not row.get("implementation"):
            return None
        source = str(row["implementation"])
        depends_on = [d for d in row.get("depends_on") or [] if isinstance(d, str)]
        calls = argument_sets(source, name, cell.bindings)
        if not calls:
            return None
        why = sc._unreplayable(
            fm,
            source=source,
            depends_on=depends_on,
            dependencies=row.get("dependencies") or (),
        )
        if why is not None:
            return f"not run on the answer cell's values: {why}"
        wanted = cell.answer_tokens
        returned, why_not = False, ""
        for call in calls:
            how, value = _run(
                fm,
                name=name,
                source=source,
                depends_on=depends_on,
                call=call,
            )
            if how != "returned":
                why_not = why_not or _ending(how, value)
                continue
            returned = True
            if not repeats(wanted, tokens(sc.plain(value)[0])):
                continue
            recorder = sc.CaseRecorder.for_function(row)
            if recorder is None:
                return None
            pending = recorder.begin(list(call["args"]), dict(call["kwargs"]))
            recorder.end(pending, result=value)
            shown = ", ".join(
                f"{key}=<{type(val).__name__}>" for key, val in call["kwargs"].items()
            )
            return (
                f"called as {name}({shown}) on the answer cell's values, it "
                "returned the session's answer; that call is recorded as its case"
            )
        if not returned:
            return (
                f"not run to the end on the answer cell's values ({why_not}); "
                "no case recorded"
            )
        return (
            "run on the answer cell's values, it did not return the session's "
            "answer; no case recorded"
        )
    except Exception as exc:  # noqa: BLE001 - never breaks a store
        logger.warning(f"answering call of {name!r} not recorded: {exc}")
        return None


__all__ = [
    "AnswerCell",
    "argument_sets",
    "enabled",
    "find_answer_cell",
    "gate_note",
    "literal_bindings",
    "record_answering_call",
    "repeats",
    "review_note",
    "reviewing",
]
