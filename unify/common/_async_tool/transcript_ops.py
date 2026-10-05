"""Pure transforms over a tool loop's message list.

A read-only inspection (``AsyncToolLoopHandle.ask``) sees the inspected
transcript in one of two shapes: a compact digest of a completed run, paired
with a drill-down tool that returns any one message verbatim, or a live
snapshot of a still-running loop with roles renamed so the inspecting model
can tell the inspected conversation from its own. Nothing here calls an LLM,
and the digest is deterministic for a given message list so repeat asks
reuse identical bytes.
"""

import json
import re
from contextlib import suppress
from typing import Any, Callable, Dict, List, Optional

from .formatting import TOOL_RESULT_TEXT_CHAR_LIMIT, _truncate_tool_text

# ── Live snapshot of a running loop ──────────────────────────────────────────

_PARENT_CTX_POINTER = (
    "## Parent Chat Context\n"
    "[The parent chat context that was available to this loop has been omitted "
    "from this transcript to avoid duplication. Refer to the Parent Chat Context "
    "section in your system context for the full, up-to-date version.]"
)


def _transform_inner_roles(messages: list[dict]) -> list[dict]:
    """Rename 'user'/'assistant' to 'inner_user'/'inner_assistant'.

    Disambiguates the inspected loop's transcript from the inspection loop's
    own conversation and from the outer parent context (which uses
    'outer_user'/'outer_assistant').
    """
    transformed = []
    for msg in messages:
        new_msg = dict(msg)
        role = new_msg.get("role", "")
        if role == "user":
            new_msg["role"] = "inner_user"
        elif role == "assistant":
            new_msg["role"] = "inner_assistant"
        transformed.append(new_msg)
    return transformed


def _replace_runtime_parent_context(messages: list[dict]) -> list[dict]:
    """Replace the embedded parent-context section with a short pointer.

    When the inspection loop receives fresh parent context of its own, the
    stale copy inside the inspected transcript is redundant. Only the
    "## Parent Chat Context" portion of a ``_parent_chat_context`` message
    is replaced; sections before it (e.g. Caller Context) are preserved.
    """
    result = []
    for msg in messages:
        if msg.get("_parent_chat_context"):
            new_msg = dict(msg)
            content = new_msg.get("content") or ""
            pcc_idx = content.find("## Parent Chat Context")
            if pcc_idx >= 0:
                new_msg["content"] = content[:pcc_idx] + _PARENT_CTX_POINTER
            result.append(new_msg)
        else:
            result.append(msg)
    return result


# ── Digest of a completed loop ───────────────────────────────────────────────

_DIGEST_URL_RE = re.compile(r"https?://[^\s\"'<>\]\)]+")
_DIGEST_SOURCES_CAP = 20
_DIGEST_RESULT_HEAD_CHARS = 240

# A digest grows ~45 tokens per turn, so an uncapped one blows past its ~2k
# token budget well before 100 turns. Turns are capped at head+tail with an
# explicit elision marker in between; elided turns keep their transcript
# ``idx`` so they stay reachable through read_child_message.
_DIGEST_TURNS_HEAD = 17
_DIGEST_TURNS_TAIL = 17
_DIGEST_MAX_TURNS = _DIGEST_TURNS_HEAD + _DIGEST_TURNS_TAIL


def _digest_result_head(text: str) -> str:
    """Pick a representative preview line from a tool result's text.

    The first line is often purely structural (a JSON opening brace, a rule
    of dashes) with the content on line 2+, so prefer the first line with
    real content and fall back to a flat character slice.
    """
    stripped = text.strip()
    if not stripped:
        return ""
    for line in stripped.splitlines():
        candidate = line.strip()
        if len(candidate) >= 3 and any(ch.isalnum() for ch in candidate):
            return candidate[:_DIGEST_RESULT_HEAD_CHARS]
    return stripped[:_DIGEST_RESULT_HEAD_CHARS]


def _message_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if content:
        try:
            return json.dumps(content, separators=(",", ":"))
        except Exception:
            return str(content)
    return ""


def _tool_call_meta(messages: List[dict]) -> Dict[str, dict]:
    """Index every tool call by id so its result message can recover the
    tool's name and ``thought`` argument.

    Under ``UNIFY_CODE_PROJECTION=notebook`` a cell has no ``thought``
    argument; its first comment, or else its first line of code, stands in.
    """
    from unify.actor import notebook_cells

    notebook = notebook_cells.enabled()
    call_meta: Dict[str, dict] = {}
    for m in messages:
        if m.get("role") != "assistant":
            continue
        for tc in m.get("tool_calls") or []:
            fn = tc.get("function") or {}
            thought = None
            with suppress(Exception):
                parsed_args = json.loads(fn.get("arguments") or "{}")
                if isinstance(parsed_args, dict) and isinstance(
                    parsed_args.get("thought"),
                    str,
                ):
                    thought = parsed_args["thought"]
                elif (
                    notebook
                    and fn.get("name") == "execute_code"
                    and isinstance(parsed_args, dict)
                    and isinstance(parsed_args.get("code"), str)
                ):
                    thought = notebook_cells.caption(parsed_args["code"]) or None
            call_meta[tc.get("id")] = {"name": fn.get("name"), "thought": thought}
    return call_meta


def _elide_middle_turns(turns: List[dict]) -> List[dict]:
    head = turns[:_DIGEST_TURNS_HEAD]
    tail = turns[-_DIGEST_TURNS_TAIL:] if _DIGEST_TURNS_TAIL else []
    elided = turns[_DIGEST_TURNS_HEAD : len(turns) - _DIGEST_TURNS_TAIL]
    marker = {
        "elided": True,
        "count": len(elided),
        "idx_range": [elided[0]["idx"], elided[-1]["idx"]],
        "note": (
            f"{len(elided)} turns elided to keep the digest compact "
            "(count is turns; idx_range is transcript positions, "
            "which interleave with non-tool messages so it is not "
            "contiguous per turn) — each elided turn still has a "
            "stable idx; retrieve any one verbatim via "
            "read_child_message(idx)."
        ),
    }
    return [*head, marker, *tail]


def build_digest(
    messages: List[dict],
    *,
    request: Any,
    final_result: Optional[str],
) -> str:
    """Compact JSON digest of a completed run, built mechanically.

    Lists the original ``request``, each tool call in execution order (name,
    ``thought`` argument when supplied, a preview of its result, the result's
    size in bytes and the message ``idx`` it sits at — capped at
    ``_DIGEST_MAX_TURNS`` with an elision marker), source URLs seen across
    the run, and the ``final_result``.

    ``final_result`` is the authoritative answer the loop's task returned.
    Pass ``None`` only when that is unavailable; the digest then falls back
    to the last bare assistant message, which is wrong whenever the loop
    submitted its answer through a tool (the last such message is then the
    narration that preceded the real answer).
    """
    call_meta = _tool_call_meta(messages)

    turns: list[dict] = []
    seen_urls: list[str] = []
    seen_urls_set: set[str] = set()
    heuristic_final_result = None

    for idx, m in enumerate(messages):
        role = m.get("role")
        content = m.get("content")
        text = _message_text(content)

        if text:
            for url in _DIGEST_URL_RE.findall(text):
                if url not in seen_urls_set:
                    seen_urls_set.add(url)
                    seen_urls.append(url)

        if role == "tool":
            meta = call_meta.get(m.get("tool_call_id"), {})
            turns.append(
                {
                    "idx": idx,
                    "tool": meta.get("name") or m.get("name"),
                    "thought": meta.get("thought"),
                    "result_head": _digest_result_head(text),
                    "result_bytes": len(text.encode("utf-8")),
                },
            )
        elif (
            role == "assistant"
            and not m.get("tool_calls")
            and isinstance(content, str)
            and content.strip()
        ):
            heuristic_final_result = content

    if final_result is None:
        final_result = heuristic_final_result

    if len(turns) > _DIGEST_MAX_TURNS:
        turns = _elide_middle_turns(turns)

    digest_obj = {
        "request": request,
        "turns": turns,
        "sources": seen_urls[:_DIGEST_SOURCES_CAP],
        "final_result": final_result,
    }
    return json.dumps(digest_obj, separators=(",", ":"), default=str)


def make_read_child_message_tool(messages: List[dict]) -> Callable:
    """Drill-down counterpart to the digest's ``idx`` fields: returns one
    message of *messages* verbatim, compact-serialized and capped at 32KB."""

    async def read_child_message(idx: int) -> str:
        if not (0 <= idx < len(messages)):
            return (
                f"⚠️ No message at idx={idx}. Valid range: "
                f"0-{max(len(messages) - 1, 0)}."
            )
        serialized = json.dumps(messages[idx], separators=(",", ":"), default=str)
        return _truncate_tool_text(serialized, limit=TOOL_RESULT_TEXT_CHAR_LIMIT)

    read_child_message.__doc__ = (
        "Fetch one message from the completed tool's transcript, verbatim.\n\n"
        "Parameters\n"
        "----------\n"
        "idx : int\n"
        "    The message index, as listed in the digest's `turns` entries "
        "(`idx` field).\n\n"
        "Returns\n"
        "-------\n"
        "str\n"
        "    The message, compact-serialized JSON, capped at 32KB (beyond "
        "that the middle is omitted with a marker)."
    )
    return read_child_message


# ── Clarification tail messages ──────────────────────────────────────────────

_CLARIFICATION_MSG_RE = re.compile(r"^\[clarification ([^\]]+)\] ")


def extract_clarifications(
    msgs: List[dict],
    *,
    callid_to_tool_name: Optional[Dict[str, str]] = None,
) -> List[dict]:
    """Summarise the transcript's "[clarification <call_id>]" user-role tail
    messages (see ``ToolsData.record_clarification``) as
    ``{call_id, tool, question}`` entries.

    ``question`` is the tail message's text verbatim with the prefix and the
    "Tool incomplete..." framing stripped. One entry per call_id: a tool that
    asks more than once produces a fresh tail message per question, and the
    LAST matching message wins, reflecting its current question.
    """
    callid_to_tool_name = callid_to_tool_name or {}
    by_call_id: Dict[str, dict] = {}
    for m in msgs or []:
        try:
            if not isinstance(m, dict) or m.get("role") != "user":
                continue
            content = m.get("content")
            if not isinstance(content, str):
                continue
            match = _CLARIFICATION_MSG_RE.match(content)
            if not match:
                continue
            call_id = match.group(1)
            before, sep, after = content[match.end() :].partition("\n")
            question = after if sep else before
        except Exception:
            continue
        by_call_id[call_id] = {
            "call_id": call_id,
            "tool": callid_to_tool_name.get(call_id, ""),
            "question": question,
        }
    return list(by_call_id.values())
