"""``UNIFY_REVIEW_GATE``: one cheap yes/no call before the session's storage review.

The review that follows a session reads the whole trajectory and stores
functions and guidance; it runs after every session, solved or not, and on
the 4 Oct ARC LOW runs it was 4.1-4.5 calls and 10-16% of USD per episode.
Most sessions leave nothing to store. The gate asks one tool-free question
first -- did this session leave reusable working code, a lesson found by
trial and error, or a stored entry that needs repair? -- and the review runs
only on a yes. It is modelled on Prime's automatic refine gate, which asks
whether a checkpoint should run /refine and rejects "one-off noise,
unsupported hypotheses, and transient tool outputs".

The gate is a separate request after the session, so the session's own
requests and prefix are untouched. It reads the end of the trajectory as the
review would (``_prepare_trajectory_for_storage_review``), the checked
outcome when the environment posted one (``UNIFY_OUTCOME``) and the final
reply. A reply it cannot read, or a failed call, runs the review as
shipped: the gate only ever saves a review, never loses one by accident.

While the library holds nothing (no stored function and no guidance entry)
the gate is not asked and the review runs: the first sessions of a run are
the ones that seed the library, and on the 80 captured reviews the gate's
one costly miss was such a session (AppWorld HIGH's first task, whose three
stored functions later tasks called).

``UNIFY_REVIEW_GATE_FORK`` asks the same question as a fork of the session's
conversation (:func:`decide_in_fork`). The standalone gate is a new prompt:
every one of its 24 calls per ARC LOW lean-all run (about 17k tokens each)
read 0 tokens from the provider's cache, 30-35% of that arm's cache writes.
Forked, its request is the session's last request as sent, the session's
reply and one appended user message, so all but that message is the prefix
the session already cached.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)

# The most of the trajectory the gate reads, from its end (Prime's gate reads
# the last 40,000 characters).
GATE_TAIL_CHARS = 40_000
# The most of one message it shows, head and tail around an elision.
_MESSAGE_HEAD = 2_500
_MESSAGE_TAIL = 1_000
# The effort the gate runs at unless UNIFY_REVIEW_REASONING_EFFORT sets the
# review's.
GATE_EFFORT = "low"
ORIGIN = "StorageCheckGate"

GATE_SYSTEM_PROMPT = (
    "You decide whether a finished agent session is worth a library review. "
    "The review turns what the session did into stored functions (code that "
    "ran and worked, which a later task of the same kind would run again, "
    "with what varies between tasks as parameters) and short guidance (a "
    "lesson that changes how a later task is done). A review costs several "
    "model calls, so it should run only when the session left something "
    "worth keeping.\n\n"
    "Answer yes when the session holds at least one of:\n"
    "- code that ran successfully, would be run again by a later task of "
    "the same kind, and is not already a stored function the session "
    "called;\n"
    "- a correction found by trial and error: an approach that failed in a "
    "non-obvious way, and what worked instead;\n"
    "- a stored function or guidance entry that failed or misled the "
    "session and needs repair.\n\n"
    "Answer no for one-off work: results worked out in text without "
    "reusable code, steps a stored function already did successfully, "
    "transient tool outputs, and hypotheses the session never confirmed.\n\n"
    'Reply with one JSON object on one line: {"review": true or false, '
    '"reason": "<one sentence>"}'
)


@dataclass(frozen=True)
class GateDecision:
    review: bool
    reason: str
    # Whether the model's reply decided it (False: the gate failed open).
    decided: bool
    # UNIFY_REVIEW_OUTCOME: the reply's "answer_outcome", when it states one.
    answer_outcome: Optional[str] = None


def enabled() -> bool:
    from unify.settings import SETTINGS

    return bool(SETTINGS.UNIFY_REVIEW_GATE)


def fork_enabled() -> bool:
    """Whether ``UNIFY_REVIEW_GATE_FORK`` asks the gate in a fork of the session."""
    from unify.settings import SETTINGS

    return bool(getattr(SETTINGS, "UNIFY_REVIEW_GATE_FORK", False))


def library_is_empty(counts: tuple[Optional[int], Optional[int]]) -> bool:
    """Whether *counts* (stored functions, guidance entries) say the library
    holds nothing. An unknown count (``None``) is not taken as empty."""
    return tuple(counts) == (0, 0)


def _text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, dict):
                text = part.get("text")
                parts.append(text if isinstance(text, str) else f"[{part.get('type')}]")
            else:
                parts.append(str(part))
        return "\n".join(parts)
    return "" if content is None else json.dumps(content, default=str)


def _clip(text: str) -> str:
    if len(text) <= _MESSAGE_HEAD + _MESSAGE_TAIL:
        return text
    omitted = len(text) - _MESSAGE_HEAD - _MESSAGE_TAIL
    return (
        text[:_MESSAGE_HEAD]
        + f"\n… [{omitted:,} chars omitted] …\n"
        + text[-_MESSAGE_TAIL:]
    )


def render_trajectory(messages: list[dict]) -> str:
    """The trajectory as plain text, one block per message, its end kept.

    System messages are left out (the gate judges what the session did, not
    what it was told), tool calls are shown by name and arguments, and the
    text is cut from the front to :data:`GATE_TAIL_CHARS`.
    """
    blocks: list[str] = []
    for msg in messages:
        if not isinstance(msg, dict) or msg.get("role") == "system":
            continue
        role = str(msg.get("role") or "")
        label = role
        if role == "tool":
            label = f"tool result ({msg.get('name') or msg.get('tool_call_id') or ''})"
        body = _clip(_text(msg.get("content")))
        calls = []
        for call in msg.get("tool_calls") or []:
            fn = (call or {}).get("function") or {}
            calls.append(
                f"-> {fn.get('name')}({_clip(str(fn.get('arguments') or ''))})",
            )
        blocks.append("\n".join([f"[{label}]", body, *calls]).strip())
    text = "\n\n".join(blocks)
    if len(text) > GATE_TAIL_CHARS:
        text = "… [earlier messages omitted] …\n" + text[-GATE_TAIL_CHARS:]
    return text


def build_user_message(
    *,
    trajectory: list[dict],
    final_result: str,
    outcome_note: str = "",
) -> str:
    parts = [
        "## Session (its transcript, ending with the final reply)\n\n"
        + render_trajectory(trajectory),
    ]
    if outcome_note.strip():
        parts.append(outcome_note.strip())
    parts.append("## Final reply\n\n" + _clip(str(final_result or "")))
    parts.append("Should this session's work be reviewed for the library?")
    return "\n\n".join(parts)


# UNIFY_REVIEW_GATE_FORK: the gate's criteria, asked at the end of the
# session's own conversation. The criteria are the standalone gate's; only
# the opening differs, since the conversation above is the session.
_FORK_OPENING = "You decide whether a finished agent session is worth"
GATE_FORK_PROMPT = (
    "## Library Review Gate\n\n"
    "The task above is over. Do not continue it and do not call any tool: "
    "answer in text. "
    + GATE_SYSTEM_PROMPT.replace(
        _FORK_OPENING,
        "Decide whether this finished session (the conversation above) is worth",
        1,
    )
)


def build_fork_message(*, final_result: str, outcome_note: str = "") -> str:
    """The one user message a forked gate appends to the session's conversation."""
    parts = [GATE_FORK_PROMPT]
    if outcome_note.strip():
        parts.append(outcome_note.strip())
    parts.append("## Final reply\n\n" + _clip(str(final_result or "")))
    parts.append("Should this session's work be reviewed for the library?")
    return "\n\n".join(parts)


_JSON_OBJECT = re.compile(r"\{.*\}", re.DOTALL)


def parse_decision(raw: Any) -> Optional[GateDecision]:
    """The decision in *raw*, or ``None`` when it states none."""
    text = str(raw or "").strip()
    match = _JSON_OBJECT.search(text)
    if match is None:
        return None
    try:
        data = json.loads(match.group(0))
    except ValueError:
        return None
    if not isinstance(data, dict) or not isinstance(data.get("review"), bool):
        return None
    reason = data.get("reason")
    from unify.actor import review_outcome

    return GateDecision(
        review=data["review"],
        reason=" ".join(str(reason or "").split())[:300],
        decided=True,
        answer_outcome=review_outcome.parse(match.group(0)),
    )


async def decide(
    *,
    client_factory: Callable[[], Any],
    trajectory: list[dict],
    final_result: str,
    outcome_note: str = "",
) -> GateDecision:
    """Ask the gate; a failed call or an unreadable reply runs the review."""
    try:
        client = client_factory()
        raw = await client.generate(
            user_message=build_user_message(
                trajectory=trajectory,
                final_result=final_result,
                outcome_note=outcome_note,
            ),
            system_message=GATE_SYSTEM_PROMPT,
        )
    except Exception as exc:  # the review runs as shipped
        logger.warning(
            f"StorageCheck gate failed ({type(exc).__name__}: {exc}); reviewing",
        )
        return GateDecision(True, f"gate call failed: {type(exc).__name__}", False)
    decision = parse_decision(raw)
    if decision is None:
        logger.warning("StorageCheck gate reply stated no decision; reviewing")
        return GateDecision(True, "gate reply stated no decision", False)
    return decision


async def decide_in_fork(
    *,
    client_factory: Callable[[], Any],
    fork_source: dict,
    final_result: str,
    outcome_note: str = "",
    prompt_caching: Any = None,
) -> GateDecision:
    """Ask the gate as a fork of the session (``UNIFY_REVIEW_GATE_FORK``).

    *fork_source* is the session's (``_review_fork_source`` in the actor):
    ``sent_messages`` is its history as its last request sent it (system
    prompt first) followed by its reply, ``tools`` and ``tool_choice`` what
    that request carried. The request is that history plus one user message,
    the same tools, and the same tool choice -- ``auto`` for a forced one,
    since the answer is text. The gate runs no tool: a reply that calls one
    states no decision. A failed call or an unreadable reply runs the
    review, as with the standalone gate.
    """
    from unify.common._async_tool import cache_discipline

    request: dict[str, Any] = {
        "messages": [
            *fork_source["sent_messages"],
            {
                "role": "user",
                "content": build_fork_message(
                    final_result=final_result,
                    outcome_note=outcome_note,
                ),
            },
        ],
        "stateful": False,
        "return_full_completion": True,
    }
    if fork_source.get("tools"):
        tool_choice = fork_source.get("tool_choice")
        if cache_discipline.is_forced_tool_choice(tool_choice):
            tool_choice = "auto"
        request["tools"] = fork_source["tools"]
        request["tool_choice"] = tool_choice
    if prompt_caching is not None:
        request["prompt_caching"] = prompt_caching
    try:
        client = client_factory()
        completion = await client.generate(**request)
        cache_discipline.log_cache_use(client, completion, label=ORIGIN)
        raw = cache_discipline.completion_text(completion.choices[0].message.content)
    except Exception as exc:  # the review runs as shipped
        logger.warning(
            f"StorageCheck gate failed ({type(exc).__name__}: {exc}); reviewing",
        )
        return GateDecision(True, f"gate call failed: {type(exc).__name__}", False)
    decision = parse_decision(raw)
    if decision is None:
        logger.warning("StorageCheck gate reply stated no decision; reviewing")
        return GateDecision(True, "gate reply stated no decision", False)
    return decision
