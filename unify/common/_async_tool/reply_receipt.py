"""``UNIFY_REPLY_RECEIPT=on``: facts about a drafted reply, shown once.

As shipped a text reply ends the turn as soon as the model writes it. With
the switch on, a reply that would end a turn of a loop answering a requester
is first checked, and when a check holds the model reads the facts once and
replies again; whatever it replies then ends the turn.

The checks are the two that passed the offline gate of the first-principles
memo (§E2; 2,126 logged final replies from office, AppWorld and
Continual-ARC; pooled trigger 4.0%, precision 0.80), ported from its
``checks.py`` and generalised: no benchmark is named, and the request is
read only for its tables (blocks of integers in text, and lists of lists in
its JSON values, parsed as ``request.data`` is under UNIFY_BIND_REQUEST):

* C2_nc, a degenerate answer: the reply's primary answer (the bare reply;
  a bold, backticked or fenced value; the value after "is", "total", "="
  or ":"; for a reply holding a JSON object, its value field with the most
  scalars) is 0, NaN, None, null, empty, or a list of two or more identical
  scalars; or that answer, a list of lists, is identical to a table in the
  request: a block of two or more lines of integers of one length in its
  text, or a list of lists in its JSON values. Other fields of a JSON reply
  are not checked (the gate never measured them), and a list of identical
  rows is not "every item the same": the gate dropped the single-colour
  rule.
* C4_last, an error the reply does not mention: the last computing cell
  (``execute_code`` or ``execute_function`` that does something) since the
  requester's latest message raised, or a cell since then caught an
  exception and printed an error marker, and the reply says nothing of an
  error (fail, error, could not, unable, ...).

:func:`render` words the facts as the model reads them: at most 3 lines,
facts only, no instruction, and quoted code or error text with any token
naming a demo, an example or a pair elided.

The loop (``loop.py``) calls :func:`receipt` where a text reply would end
the turn, keeps the per-request state on its runtime state, and counts
``receipts_shown`` and ``receipts_revised`` (:func:`run_stats`).
"""

from __future__ import annotations

import ast
import json
import math
import re
from typing import Any, Iterable, Optional

from .messages import is_loop_authored_message

#: The marker on the loop-authored message that holds a receipt.
MARKER = "_reply_receipt"

#: Steps (messages) a receipt takes: itself and the reply after it. A reply
#: with no more than this many steps left before ``max_steps`` gets none.
STEPS = 2

#: The tools whose calls are computing cells.
CELL_TOOLS = frozenset({"execute_code", "execute_function"})

#: Requests longer than this are not parsed for JSON values.
MAX_REQUEST_CHARS = 1_000_000


def enabled() -> bool:
    """Whether ``UNIFY_REPLY_RECEIPT=on``."""
    from unify.settings import SETTINGS

    return getattr(SETTINGS, "UNIFY_REPLY_RECEIPT", "") == "on"


def has_room(remaining_steps: Optional[int]) -> bool:
    """Whether the receipt and the reply after it fit before ``max_steps``."""
    return remaining_steps is None or remaining_steps > STEPS


# ---------------------------------------------------------------------------
# Values in the reply (offline checks.py, unchanged where not noted)
# ---------------------------------------------------------------------------

NUM = re.compile(r"(?<![\w./:@#-])-?\d[\d,]*(?:\.\d+)?%?(?![\w/@-]|\.\d|:\d)")
PRIMARY = re.compile(r"\*\*([^*\n]{1,80})\*\*|`([^`\n]{1,80})`")
AFTER_IS = re.compile(
    r"\b(?:is|was|are|were|total(?:s|led)?|answer|equals?|comes to|=|:)\s*\**`?"
    r"(-?\d[\d,]*(?:\.\d+)?%?|NaN|nan|None|null)\b",
    re.I,
)
FILEISH = re.compile(
    r"^[\w./-]+\.(?:txt|csv|json|jsonl|py|md|log|xlsx|html|yaml|yml|toml|ini|cfg)$"
    r"|^[\w-]+/$|/",
    re.I,
)
FENCE = re.compile(r"```[\w-]*\s*\n?(.*?)```", re.S)
UNCERT = re.compile(r"(?is)(\*\*|#+\s*)?Uncertaint(?:y|ies)\b.*$")
EMPTY_TOKENS = ("", "[]", "{}", '""', "''")


def _num(tok: str) -> Optional[float]:
    try:
        return float(tok.replace(",", "").rstrip("%"))
    except ValueError:
        return None


def body_of(reply: str) -> str:
    """The reply without its trailing 'Uncertainties' section."""
    return UNCERT.sub("", reply or "")


def bare_value(reply: str) -> Optional[str]:
    """The whole reply when it is a bare value (a number, a short token or a list)."""
    r = (reply or "").strip().strip("*`").strip()
    if not r or len(r) > 60 or "\n" in r:
        return None
    if NUM.fullmatch(r) or re.fullmatch(r"\[.*\]|\{.*\}", r) or len(r.split()) <= 3:
        return r
    return None


def primary_values(reply: str) -> list[str]:
    """Values the reply presents as its answer: the bare reply, bold or
    backticked values, values after 'is/total/=/:', short fenced blocks."""
    vals = []
    b = bare_value(reply)
    if b is not None:
        vals.append(b)
    body = body_of(reply)
    for m in PRIMARY.finditer(body):
        v = (m.group(1) or m.group(2) or "").strip()
        if v and not FILEISH.search(v):
            vals.append(v)
    for m in AFTER_IS.finditer(body):
        vals.append(m.group(1))
    for m in FENCE.finditer(body):
        v = m.group(1).strip()
        if len(v) <= 60:
            vals.append(v)
    return vals


def _is_scalar(value: Any) -> bool:
    return value is None or isinstance(value, (str, int, float, bool))


def _all_same(items: Any) -> bool:
    """Two or more scalars, all equal (a list of rows is never one)."""
    return (
        isinstance(items, list)
        and len(items) >= 2
        and all(_is_scalar(e) for e in items)
        and all(type(e) is type(items[0]) and e == items[0] for e in items)
    )


def _degenerate_text_value(v: str) -> Optional[str]:
    vv = v.strip().strip("*`").strip()
    x = _num(vv) if NUM.fullmatch(vv) else None
    if vv in EMPTY_TOKENS:
        return f"the answer is an empty value ({vv or repr(vv)})"
    if vv.lower() in ("nan", "none", "null"):
        return f"the answer is {vv}"
    if x is not None and x == 0:
        return f"the answer is zero ({vv})"
    if vv.startswith("[") and vv.endswith("]"):
        try:
            items = ast.literal_eval(vv)
        except Exception:
            return None
        if _all_same(items):
            return f"every item of the answer is the same ({vv[:40]})"
    return None


# ---------------------------------------------------------------------------
# The primary JSON value of the reply, and the values of the request
# ---------------------------------------------------------------------------


def json_values(text: str) -> list:
    """The top-level JSON objects and arrays in *text*, as UNIFY_BIND_REQUEST
    parses a request (``worker_child.json_values``)."""
    from unify.actor.execution.worker_child import json_values as parse

    return parse(text or "")


def _items(value: Any) -> int:
    """The scalars a JSON value holds."""
    if isinstance(value, list):
        return sum(_items(v) for v in value)
    if isinstance(value, dict):
        return sum(_items(v) for v in value.values())
    return 1


#: No primary JSON value.
_NONE = object()


def primary_json(reply: str) -> Any:
    """The JSON value the reply presents as its answer, or ``_NONE``.

    For a reply holding a JSON object (an action object, say) it is the
    object's value field holding the most scalars (the first on a tie); the
    offline gate read the first object of a reply the same way. Otherwise it
    is a JSON array that is the whole reply, a fenced block or a backticked
    value. A JSON array written in prose is not an answer.
    """
    body = body_of(reply)
    for value in json_values(body):
        if isinstance(value, dict):
            return max(value.values(), key=_items) if value else value
    texts = [body.strip().strip("*`").strip()]
    texts += [m.group(1).strip() for m in FENCE.finditer(body)]
    texts += [m.group(2).strip() for m in PRIMARY.finditer(body) if m.group(2)]
    for text in texts:
        if text.startswith("["):
            try:
                return json.loads(text)
            except ValueError:
                continue
    return _NONE


def _table(value: Any) -> bool:
    """A non-empty list of lists: the only kind of answer compared with the request."""
    return (
        isinstance(value, list)
        and bool(value)
        and all(isinstance(row, list) for row in value)
    )


_DIGITS = re.compile(r"\d+")
_INT_ROW = re.compile(r"-?\d+(?:(?:\s*,\s*|\s+)-?\d+)+,?")


def _row(line: str) -> Optional[list[int]]:
    """A line of integers (separated by spaces or commas, or a run of single
    digits with no separator) as a list, else ``None``."""
    text = line.strip()
    if _DIGITS.fullmatch(text):
        return [int(c) for c in text]
    if _INT_ROW.fullmatch(text):
        return [int(t) for t in re.split(r"[\s,]+", text.rstrip(",")) if t]
    return None


def number_tables(text: str) -> list[list[list[int]]]:
    """Every rectangular block of integers in *text*: two or more consecutive
    lines of integers, all of one length, each read as a list of ints."""
    tables: list[list[list[int]]] = []
    run: list[list[int]] = []
    for line in (text or "").splitlines() + [""]:
        row = _row(line)
        if row is not None and run and len(row) == len(run[0]):
            run.append(row)
            continue
        if len(run) >= 2:
            tables.append(run)
        run = [row] if row is not None else []
    return tables


def _json_tables(values: Iterable[Any], into: list) -> list:
    """Every list of lists at any depth of *values*."""
    for value in values:
        if isinstance(value, dict):
            _json_tables(value.values(), into)
        elif isinstance(value, list):
            if _table(value):
                into.append(value)
            _json_tables(value, into)
    return into


def request_tables(request: Optional[str]) -> list:
    """The request's tables: its blocks of integers in text, and every list of
    lists in its JSON values."""
    if not request or len(request) > MAX_REQUEST_CHARS:
        return []
    return number_tables(request) + _json_tables(json_values(request), [])


# ---------------------------------------------------------------------------
# C2_nc
# ---------------------------------------------------------------------------


def degenerate(reply: str, request: Optional[str]) -> Optional[str]:
    """Why the reply's primary answer is degenerate (lower-case fact), or ``None``."""
    if not (reply or "").strip():
        return "the reply is empty"
    for v in primary_values(reply):
        why = _degenerate_text_value(v)
        if why:
            return why
    value = primary_json(reply)
    if value is _NONE or isinstance(value, bool):
        return None
    shown = json.dumps(value, ensure_ascii=False)
    if value is None:
        return "the answer is null"
    if isinstance(value, float) and math.isnan(value):
        return "the answer is NaN"
    if isinstance(value, (int, float)) and value == 0:
        return f"the answer is zero ({shown})"
    if value in ("", [], {}):
        return f"the answer is an empty value ({shown})"
    if not isinstance(value, (list, dict)):
        return None
    if _all_same(value):
        return f"every item of the answer is the same ({shown[:40]})"
    if _table(value) and any(value == t for t in request_tables(request)):
        return "the answer is identical to a value in the request"
    return None


# ---------------------------------------------------------------------------
# C4_last (offline checks.py)
# ---------------------------------------------------------------------------

TB = "Traceback (most recent call last)"
ERRLINE = re.compile(r"^\s*(?:[A-Z]\w*(?:Error|Exception)|ERROR|Error)\b\s*[:\-]", re.M)
ERRJSON = re.compile(r'"error"\s*:\s*"(?!")')
CAUGHT = re.compile(
    r"^\W*(?:error|exception|failed|failure)\b"
    r"|\b(?:error|exception|failed|failure)\b\s*[:!=-]"
    r"|\b[A-Z]\w*(?:Error|Exception)\b",
    re.I | re.M,
)
REPLY_MENTIONS_ERR = re.compile(
    r"error|fail|exception|could ?n[o'’]t|cannot|can['’]t|did ?n[o'’]t|unsuccessful"
    r"|unable|traceback|not found|missing|warning|problem|issue|retr(?:y|ied)|fixed|crash",
    re.I,
)
META = {
    "state_mode",
    "session_id",
    "session_created",
    "duration_ms",
    "session_name",
    "language",
    "venv_id",
    "shell",
    "session_reused",
}


def noop(code: Optional[str]) -> bool:
    """Code that does nothing (empty, pass, print())."""
    lines = [
        line
        for line in (code or "").splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]
    return "\n".join(lines).strip() in (
        "",
        "pass",
        "print()",
        "print('')",
        'print("")',
        "...",
        "None",
    )


def clean_output(parts: list) -> str:
    """Tool result parts -> output text: drops the JSON status header of a
    code tool's result (keeping its error) and the stream markers."""
    out = []
    for i, p in enumerate(parts):
        t = p if isinstance(p, str) else ""
        if i == 0 and t.lstrip().startswith("{"):
            try:
                head = json.loads(t)
            except Exception:
                head = None
            if isinstance(head, dict) and (
                "duration_ms" in head or "state_mode" in head or "error" in head
            ):
                if head.get("error"):
                    out.append(str(head["error"]))
                rest = {
                    k: v
                    for k, v in head.items()
                    if k not in META and k != "error" and v not in (None, "", [], {})
                }
                if rest:
                    out.append(json.dumps(rest))
                continue
        t = re.sub(
            r"^\s*--- (?:stdout|stderr|result|output) ---\s*$",
            "",
            t,
            flags=re.M,
        )
        out.append(t)
    return "\n".join(x for x in out if x.strip())


def error_of(cell: dict) -> Optional[tuple[str, str]]:
    """('raised'|'caught', first error line) when the cell errored or printed
    an error marker from an except."""
    out, code = cell.get("output") or "", cell.get("code") or ""
    if not out:
        return None
    if TB in out or ERRJSON.search(out[:2000]):
        lines = [
            line
            for line in out.replace("\\n", "\n").splitlines()
            if re.match(r"^\s*\w*(?:Error|Exception)\b", line)
        ]
        line = (
            lines[-1].strip().rstrip('"}').rstrip("\\").strip()
            if lines
            else "Traceback"
        )
        return "raised", line[:160]
    m = ERRLINE.search(out)
    if m:
        line = out[m.start() :].strip().splitlines()[0]
        return ("raised" if "except" not in code else "caught"), line[:160]
    if "except" in code and CAUGHT.search(out):
        line = next(x for x in out.splitlines() if CAUGHT.search(x))
        return "caught", line.strip()[:160]
    return None


def last_error(reply: str, cells: list[dict]) -> Optional[dict]:
    """C4_last: the last computing cell errored, or a cell caught and printed
    an error, and the reply mentions no error."""
    if REPLY_MENTIONS_ERR.search(reply or ""):
        return None
    comp = [c for c in cells if not c.get("noop")]
    hits = [(i, error_of(c)) for i, c in enumerate(comp)]
    hits = [(i, e) for i, e in hits if e and (i == len(comp) - 1 or e[0] == "caught")]
    if not hits:
        return None
    i, (kind, line) = hits[-1]
    return {"cell": i + 1, "kind": kind, "line": line}


# ---------------------------------------------------------------------------
# The transcript
# ---------------------------------------------------------------------------


def _since_request(messages: list) -> tuple[Optional[dict], list]:
    """The requester's latest message and the messages after it."""
    for i in range(len(messages) - 1, -1, -1):
        m = messages[i]
        if m.get("role") == "user" and not is_loop_authored_message(m):
            return m, messages[i + 1 :]
    return None, list(messages)


def _parts(content: Any) -> list:
    if isinstance(content, list):
        return [
            (p.get("text") or "") if isinstance(p, dict) else str(p) for p in content
        ]
    if isinstance(content, str):
        try:
            loaded = json.loads(content)
        except Exception:
            return [content]
        if isinstance(loaded, list):
            return [
                (p.get("text") or "") if isinstance(p, dict) else str(p) for p in loaded
            ]
        return [content]
    return [] if content is None else [str(content)]


def request_text(messages: list) -> Optional[str]:
    """The text of the requester's latest message in *messages*."""
    from .bound_request import text_of

    message, _ = _since_request(messages)
    return None if message is None else text_of(message)


def cells_since_request(messages: list) -> list[dict]:
    """The computing cells since the requester's latest message, in order:
    ``tool``, ``code``, ``noop`` and ``output`` (``None`` when no result)."""
    _, after = _since_request(messages)
    results = {
        m.get("tool_call_id"): m.get("content")
        for m in after
        if m.get("role") == "tool" and m.get("tool_call_id")
    }
    cells = []
    for m in after:
        if m.get("role") != "assistant":
            continue
        for call in m.get("tool_calls") or []:
            fn = call.get("function") or {}
            name = fn.get("name")
            if name not in CELL_TOOLS:
                continue
            args = fn.get("arguments")
            if isinstance(args, str):
                try:
                    args = json.loads(args or "{}")
                except Exception:
                    args = {}
            args = args if isinstance(args, dict) else {}
            if name == "execute_code":
                code = args.get("code")
                code = (
                    code
                    if isinstance(code, str)
                    else ("" if code is None else str(code))
                )
                is_noop = noop(code)
            else:
                kwargs = json.dumps(args.get("call_kwargs"), default=str)
                code = f"{args.get('function_name')}(**{kwargs})"
                is_noop = False
            content = results.get(call.get("id"))
            cells.append(
                {
                    "tool": name,
                    "code": code,
                    "noop": is_noop,
                    "output": (
                        None if content is None else clean_output(_parts(content))
                    ),
                },
            )
    return cells


# ---------------------------------------------------------------------------
# The receipt
# ---------------------------------------------------------------------------

NO_EXAMPLES = re.compile(r"\w*(?:demo|example|pair)\w*", re.I)


def _safe(text: str) -> str:
    """Quoted code or error text never names a task's examples: such tokens are elided."""
    return NO_EXAMPLES.sub("…", text or "")


def checks(reply: str, request: Optional[str], cells: list[dict]) -> dict:
    """C2 (C2_nc) and C4_last for a drafted *reply*; ``None`` where silent."""
    why = degenerate(reply, request)
    return {"C2": {"why": why} if why else None, "C4_last": last_error(reply, cells)}


def render(res: dict) -> list[str]:
    """The facts as the model reads them: at most 3 lines, no instruction."""
    lines = []
    if res.get("C2"):
        why = _safe(res["C2"]["why"])
        lines.append(why[0].upper() + why[1:] + ".")
    if res.get("C4_last"):
        d = res["C4_last"]
        verb = "raised" if d["kind"] == "raised" else "caught and printed"
        lines.append(
            f"Code cell {d['cell']} since the request {verb} "
            f"`{_safe(d['line'][:80])}`; the reply does not mention an error.",
        )
    return lines[:3]


def receipt(reply: str, request: Optional[str], messages: list) -> Optional[str]:
    """The receipt for *reply* drafted at the end of *messages*, or ``None``."""
    lines = render(checks(reply, request, cells_since_request(messages)))
    return "\n".join(lines) if lines else None


def revised(draft: str, final: str) -> bool:
    """Whether the reply after a receipt differs from the draft (whitespace aside)."""
    return " ".join(str(draft or "").split()) != " ".join(str(final or "").split())


def run_stats(runtime_state: Any) -> dict:
    """The counters the CLI reports beside the run's tokens."""
    return {
        "receipts_shown": int(getattr(runtime_state, "receipts_shown", 0)),
        "receipts_revised": int(getattr(runtime_state, "receipts_revised", 0)),
    }


__all__ = [
    "MARKER",
    "cells_since_request",
    "checks",
    "degenerate",
    "enabled",
    "has_room",
    "last_error",
    "receipt",
    "render",
    "request_text",
    "revised",
    "run_stats",
]
