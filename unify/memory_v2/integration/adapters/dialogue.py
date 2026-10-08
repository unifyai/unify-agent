"""Adapter 4, dialogue (spec §C2): the action carried in each reply, paired with the next observation.

A dialogue action is a turn-ending assistant reply (text, no tool calls) followed by the next genuine
user-role message, which is what the counterpart sent back: a typed environment's observation
(Crafter, ScienceWorld), ARC's feedback after a JSON ``submit``/``request_demos``, or a person's
follow-up. The transcript is the only instrument; nothing else is observed.

Everything here is structural:

- the counterpart (the channel key) is supplied by the harness, the source it serves; it is never read
  from content and never keyed on a task;
- the payload is the reply's trailing JSON object when one parses, else its final non-empty line; no
  word in the reply is ever matched;
- the observation fingerprint (``response["shape"]``) keeps line-count buckets, whether the text ends in a ``(… k/N)`` counter,
  and its JSON kind and key shape, never a value.

Assistant messages that carry tool calls (``execute_code`` cells) are not dialogue; the tool, shell and
worktree adapters record those. Loop-authored user messages (``_loop_authored``: progress notices,
context headers) are harness text, not the counterpart's, and are skipped when looking for the
observation. Only ``type == "message"`` lines are read, so ``message_update``, ``system_prompt``,
``session_start``, ``compaction`` and outcome lines never become actions or observations.

Not wired into the actor yet.
"""

from __future__ import annotations

import json
import re
from typing import Any, Iterable

from ...episodes import Action
from ...fingerprint import shape
from ...redact import Redactor

METHOD = "act"
DEFAULT_OBSERVATION_CAP = 4000
DEFAULT_PAYLOAD_CAP = 16000
# The memory repo's channel naming: a lowercase identifier with no "." (``Generations`` splits the
# channel off at the first ".") and no ":" (kind-qualified channels).
COUNTERPART_RE = re.compile(r"^[a-z][a-z0-9_-]*$")
# A trailing object is looked for only when the segment from the reply's first "{" to its end is at
# most this long, at most this many opening braces are tried, and a parsed object deeper than
# _MAX_DEPTH is refused, so a pathological reply costs bounded time and never recurses deeply.
_SCAN_CHARS = 65536
_SCAN_TRIES = 256
_MAX_DEPTH = 64
_SHAPE_CHARS = 200
_FENCE = "```"
# "(step 12/2000)", "(12/2000)": a parenthesised k/N counter at the very end; the label is any letters.
_TRAILING_COUNTER = re.compile(r"\(\s*[^\W\d_]*\s*\d+\s*/\s*\d+\s*\)\s*$")
_DECODER = json.JSONDecoder()


def _text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            p.get("text", "")
            for p in content
            if isinstance(p, dict) and isinstance(p.get("text", ""), str)
        )
    return ""


def _messages(lines: Iterable[dict]) -> list[dict]:
    out = []
    for ln in lines:
        if not isinstance(ln, dict) or ln.get("type") != "message":
            continue
        msg = ln.get("message")
        if isinstance(msg, dict):
            out.append(msg)
    return out


def _is_reply(msg: dict) -> bool:
    return (
        msg.get("role") == "assistant"
        and not msg.get("tool_calls")
        and bool(_text(msg.get("content")).strip())
    )


def _is_observation(msg: dict) -> bool:
    return msg.get("role") == "user" and not msg.get("_loop_authored")


def _strip_closing_fence(body: str) -> str:
    body = body.rstrip()
    if body.endswith(_FENCE):
        body = body[: -len(_FENCE)].rstrip()
    return body


def _shallow(obj: Any, limit: int = _MAX_DEPTH) -> bool:
    """Whether a parsed JSON value nests at most ``limit`` containers deep (iterative, no recursion)."""
    stack = [(obj, 1)]
    while stack:
        value, depth = stack.pop()
        if isinstance(value, (dict, list)):
            if depth > limit:
                return False
            children = value.values() if isinstance(value, dict) else value
            stack.extend((c, depth + 1) for c in children)
    return True


def trailing_object(text: str) -> dict | None:
    """The JSON object that ends the text (after an optional closing code fence), else None.

    Never raises: an over-long trailing segment, a parse failure, a too-deep nesting
    (``RecursionError`` inside the decoder) or a result deeper than ``_MAX_DEPTH`` all give None.
    """
    body = _strip_closing_fence(text)
    if not body.endswith("}"):
        return None
    offset = max(0, len(body) - _SCAN_CHARS)
    window = body[offset:]
    tries = 0
    pos = window.find("{")
    while pos != -1 and tries < _SCAN_TRIES:
        tries += 1
        try:
            obj, end = _DECODER.raw_decode(window, pos)
        except (json.JSONDecodeError, RecursionError, ValueError):
            obj, end = None, -1
        if end == len(window) and isinstance(obj, dict):
            # leftmost start that runs to the end: the outermost trailing object
            return obj if _shallow(obj) else None
        pos = window.find("{", pos + 1)
    return None


def action_payload(text: str) -> Any:
    """The action a reply carries: its trailing JSON object, else its final non-empty line."""
    obj = trailing_object(text)
    if obj is not None:
        return obj
    lines = [ln.strip() for ln in _strip_closing_fence(text).splitlines() if ln.strip()]
    return lines[-1] if lines else ""


def cap_text(text: str, cap: int) -> str:
    """At most ``cap`` characters, keeping the head and the tail (where counters and JSON sit)."""
    if cap <= 0:
        raise ValueError("cap must be positive")
    if len(text) <= cap:
        return text
    marker = f"\n[... {len(text)} chars, middle elided ...]\n"
    room = cap - len(marker)
    if room <= 1:
        return text[:cap]
    tail = room // 4
    head = room - tail
    return text[:head] + marker + (text[-tail:] if tail else "")


def dialogue_actions(
    transcript_lines: Iterable[dict],
    counterpart: str,
    *,
    redactor: Redactor | None = None,
    max_observation_chars: int = DEFAULT_OBSERVATION_CAP,
    max_payload_chars: int = DEFAULT_PAYLOAD_CAP,
) -> list[Action]:
    """One ``kind="dialogue"`` action per turn-ending reply, in transcript order.

    ``counterpart`` is the source the harness serves (it becomes the channel key verbatim) and must
    match ``COUNTERPART_RE``. The observation is the next non-loop-authored user message before the
    next assistant message; a reply with none is ``unrecorded`` with no response. Payload and
    observation are redacted, then capped: the observation at ``max_observation_chars``, the payload
    at ``max_payload_chars`` (an object payload whose JSON is longer becomes its capped JSON text).
    ``response["shape"]`` is the observation's shape taken before the cap.
    """
    if not isinstance(counterpart, str) or not COUNTERPART_RE.match(counterpart):
        raise ValueError(
            "counterpart must be a harness-supplied channel name matching "
            f"{COUNTERPART_RE.pattern}, got {counterpart!r}",
        )
    red = redactor if redactor is not None else Redactor()
    msgs = _messages(transcript_lines)
    actions: list[Action] = []
    for i, msg in enumerate(msgs):
        if not _is_reply(msg):
            continue
        reply = _text(msg.get("content"))
        # redact before capping, so no secret is cut in two
        payload = red.obj(action_payload(reply))
        if isinstance(payload, str):
            payload = cap_text(payload, max_payload_chars)
        else:
            serialised = json.dumps(payload, sort_keys=True, ensure_ascii=False)
            if len(serialised) > max_payload_chars:
                payload = cap_text(serialised, max_payload_chars)
        observation = None
        for nxt in msgs[i + 1 :]:
            if nxt.get("role") == "assistant":
                break
            if _is_observation(nxt):
                observation = _text(nxt.get("content"))
                break
        if observation is None:
            response, status = None, "unrecorded"
        else:
            redacted = red.text(observation)
            response = {
                "observation": cap_text(redacted, max_observation_chars),
                "shape": observation_shape(redacted),
            }
            status = "ok"
        actions.append(
            Action(
                cell=-1,
                channel=counterpart,
                method=METHOD,
                args=[payload],
                kwargs={},
                response=response,
                status=status,
                effect="unknown",
                kind="dialogue",
            ),
        )
    return actions


def _line_bucket(n: int) -> str:
    if n == 0:
        return "0"
    if n >= 64:
        return "64+"
    k = n.bit_length() - 1
    lo, hi = 1 << k, (1 << (k + 1)) - 1
    return str(lo) if lo == hi else f"{lo}-{hi}"


def _json_kind(text: str) -> str:
    stripped = text.strip()
    try:
        whole = json.loads(stripped) if stripped else None
        parsed = bool(stripped)
    except (json.JSONDecodeError, RecursionError, ValueError):
        whole, parsed = None, False
    if parsed and not _shallow(whole):
        return "deep"
    if parsed:
        if isinstance(whole, dict):
            return ("object:" + shape(whole))[:_SHAPE_CHARS]
        if isinstance(whole, list):
            return ("array:" + shape(whole))[:_SHAPE_CHARS]
        return "scalar"
    obj = trailing_object(text)
    if obj is not None:
        return ("trailing:" + shape(obj))[:_SHAPE_CHARS]
    return "none"


def observation_shape(text: str) -> str:
    """A value-free structural summary of one observation."""
    lines = sum(1 for ln in text.splitlines() if ln.strip())
    counter = "y" if _TRAILING_COUNTER.search(text) else "n"
    return f"lines={_line_bucket(lines)};counter={counter};json={_json_kind(text)}"


def observation_fingerprint(
    actions: Iterable[Action],
) -> dict[str, dict[str, list[str]]]:
    """Observation shapes per ``<channel>.<method>``, in the format ``fingerprint.Generations`` reads.

    Only recorded dialogue actions count; ``errors`` is always empty (an observation is never an
    error). The shape stored at recording time (taken before the cap) is used when present.
    """
    out: dict[str, set[str]] = {}
    for a in actions:
        if a.kind != "dialogue" or a.status != "ok" or not isinstance(a.response, dict):
            continue
        stored = a.response.get("shape")
        obs = a.response.get("observation")
        if isinstance(stored, str):
            got = stored
        elif isinstance(obs, str):
            got = observation_shape(obs)
        else:
            continue
        out.setdefault(f"{a.channel}.{a.method}", set()).add(got)
    return {k: {"shapes": sorted(v), "errors": []} for k, v in out.items()}
