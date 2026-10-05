"""``UNIFY_REVIEW_OUTCOME``: the storage review states whether the session's answer was confirmed.

``UNIFY_ORIGIN_PROVENANCE`` can tell a later session that a stored function
came from a session whose answer was accepted, but only when the environment
posts its checker's verdict to the harness (``UNIFY_OUTCOME``), which most
environments do not. The conversation itself usually shows it: the
requester thanks the agent, says the answer was wrong, or the environment
replies that a submission was correct. The storage review reads that
conversation anyway, so with this switch it also states, as the last line
of its reply, one JSON object: ``{"answer_outcome": "confirmed"}``,
``"rejected"`` or ``"unknown"``. The review gate, when asked, adds the same
key to its reply, so a session whose review the gate skips is still judged.
The harness reads only that line of the model's own reply; it parses
nothing of the conversation and knows nothing of any environment's wording.
The judgement (the review's, else the gate's; ``unknown`` keeps nothing) is
kept under the session's request in the request log
(:func:`unify.function_manager.task_origin.record_outcome`, source
``review``), where ``UNIFY_ORIGIN_PROVENANCE`` reads it.
"""

from __future__ import annotations

import json
import re
from typing import Any, Optional

KEY = "answer_outcome"
CONFIRMED = "confirmed"
REJECTED = "rejected"
UNKNOWN = "unknown"
VALUES = (CONFIRMED, REJECTED, UNKNOWN)

REVIEW_SECTION = (
    "## Answer Outcome\n\n"
    "As the last line of your final reply, add one JSON object saying whether "
    "this conversation shows that the session's final answer was confirmed or "
    "rejected by the requester or the environment: "
    '{"answer_outcome": "confirmed"}, {"answer_outcome": "rejected"}, or '
    '{"answer_outcome": "unknown"} when it shows neither. Judge from the '
    "conversation alone.\n\n"
)

GATE_SECTION = (
    "## Answer outcome\n\n"
    'Also put "answer_outcome" in your JSON object: "confirmed" or "rejected" '
    "if this conversation shows that the session's final answer was confirmed "
    'or rejected by the requester or the environment, "unknown" if it shows '
    "neither. Judge from the conversation alone."
)

_OBJECT = re.compile(r"\{[^{}]*\"" + KEY + r"\"[^{}]*\}")


def enabled() -> bool:
    from unify.function_manager import task_origin

    return task_origin.review_outcome_enabled()


def parse(text: Any) -> Optional[str]:
    """The last ``answer_outcome`` *text* states (``confirmed``, ``rejected`` or ``unknown``), or ``None``."""
    for match in reversed(_OBJECT.findall(str(text or ""))):
        try:
            data = json.loads(match)
        except ValueError:
            continue
        value = data.get(KEY) if isinstance(data, dict) else None
        if isinstance(value, str) and value.strip().lower() in VALUES:
            return value.strip().lower()
    return None


def solved(judgement: Optional[str]) -> Optional[bool]:
    """``True`` for confirmed, ``False`` for rejected, ``None`` otherwise."""
    return {CONFIRMED: True, REJECTED: False}.get(judgement or "")


def settle(*judgements: Optional[str]) -> Optional[str]:
    """The first judgement that is confirmed or rejected, else ``unknown`` if any was stated."""
    stated = [j for j in judgements if j in VALUES]
    for judgement in stated:
        if judgement != UNKNOWN:
            return judgement
    return UNKNOWN if stated else None


def record(*judgements: Optional[str]) -> Optional[str]:
    """Keep the settled judgement under the current request; returns it (``None`` while off)."""
    if not enabled():
        return None
    from unify.function_manager import task_origin

    judgement = settle(*judgements)
    task_origin.record_outcome(solved(judgement), source=task_origin.REVIEW)
    return judgement


__all__ = [
    "GATE_SECTION",
    "REVIEW_SECTION",
    "enabled",
    "parse",
    "record",
    "settle",
    "solved",
]
