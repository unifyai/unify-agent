"""``UNIFY_MEMORY_KIND``: keep what a session learned in one form only.

Unify's libraries hold two kinds of entry -- stored functions (code) and
guidance (written workflows and lessons) -- and the storage review, a model,
decides after each session what to distil into which. Whether the *form* of
what is kept matters is an open question; this switch holds the rest of the
harness fixed and changes only the form:

* ``functions``: the review runs as shipped but may write no guidance -- its
  guidance write tools are not offered.
* ``notes``: the review runs as shipped but may write no function -- its
  function write tools are not offered (as a lessons-only review, without
  the failure framing).
* ``examples``: no review runs. After the session the harness keeps one entry
  verbatim: the request, the code cell whose output the answer repeats
  (:func:`unify.function_manager.origin_capture.find_answer_cell`, whole and
  unedited) and the answer, with its outcome when a checker posted one. It is
  a guidance entry titled ``Worked example: ...``, so the same listings and
  searches find it; nothing is summarised, parameterised or edited.

Empty: both kinds, reviewed, as shipped. For "keep nothing" use
``UNIFY_STORE_ADMISSION=never``. The actor's own mid-task writes
(``UNIFY_INLINE_CURATION``) would bypass the review's restriction, so a kind
other than empty refuses to start with them on.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

KINDS = ("", "functions", "notes", "examples")

#: The review's guidance write tools (withheld under ``functions``).
GUIDANCE_WRITE_TOOLS: Tuple[str, ...] = (
    "GuidanceManager_add_guidance",
    "GuidanceManager_update_guidance",
    "GuidanceManager_patch_guidance",
    "GuidanceManager_delete_guidance",
    "GuidanceManager_reconcile_dependencies",
)

RULES = {
    "functions": "this harness keeps what it learns as stored functions only, so no guidance is written",
    "notes": "this harness keeps what it learns as written guidance only, so no function is written",
}

TITLE = "Worked example: "
MAX_REQUEST = 2000
MAX_CODE = 8000
MAX_ANSWER = 2000


def kind() -> str:
    """``UNIFY_MEMORY_KIND``, empty when off."""
    from unify.settings import SETTINGS

    return str(getattr(SETTINGS, "UNIFY_MEMORY_KIND", "") or "")


def require_prerequisites() -> None:
    """Refuse a kind the actor's own mid-task writes would bypass."""
    from unify.settings import SETTINGS

    if kind() and str(getattr(SETTINGS, "UNIFY_INLINE_CURATION", "") or ""):
        raise RuntimeError(
            "UNIFY_MEMORY_KIND keeps one form of memory through the storage "
            "review, and UNIFY_INLINE_CURATION lets the actor write either "
            "kind mid-task: turn one of them off.",
        )


def refused_tools(review_kind: str) -> Tuple[str, ...]:
    """The review tools a kind withholds (none for empty or ``examples``, which runs no review)."""
    if review_kind == "functions":
        return GUIDANCE_WRITE_TOOLS
    if review_kind == "notes":
        from unify import outcome as outcome_mod

        return tuple(outcome_mod.LESSON_REFUSED_TOOLS) + (
            "FunctionManager_retire_case",
        )
    return ()


def _clip(text: Any, limit: int) -> str:
    text = str(text or "").strip()
    return text if len(text) <= limit else text[:limit] + "\n… [cut]"


def _outcome_line(outcome: Optional[Mapping[str, Any]]) -> str:
    solved = (outcome or {}).get("solved") if isinstance(outcome, Mapping) else None
    if solved is True:
        return "Outcome: the environment's checker accepted this answer."
    if solved is False:
        return "Outcome: the environment's checker rejected this answer."
    return "Outcome: unknown (no checker verdict was posted)."


def example_text(
    *,
    request: str,
    code: Optional[str],
    answer: str,
    outcome: Optional[Mapping[str, Any]],
) -> Tuple[str, str]:
    """``(title, content)`` of a worked example, kept verbatim."""
    first_line = " ".join(str(request or "").split())
    title = TITLE + (first_line[:80] + ("…" if len(first_line) > 80 else ""))
    parts = [
        "A request handled earlier, kept as it was: nothing here was summarised or edited.",
        "Request:\n" + _clip(request, MAX_REQUEST),
    ]
    if code:
        parts.append(
            "Code whose output gave the answer:\n```python\n"
            + _clip(code, MAX_CODE)
            + "\n```",
        )
    else:
        parts.append("No code cell's output repeats the answer.")
    parts.append("Answer given:\n" + _clip(answer, MAX_ANSWER))
    parts.append(_outcome_line(outcome))
    return title, "\n\n".join(parts)


def _last_reply(trajectory: Sequence[Mapping[str, Any]]) -> str:
    for message in reversed(list(trajectory)):
        if message.get("role") == "assistant" and not message.get("tool_calls"):
            content = message.get("content")
            if isinstance(content, list):
                return " ".join(
                    str(p.get("text") or "") for p in content if isinstance(p, Mapping)
                ).strip()
            return str(content or "").strip()
    return ""


def store_example(
    actor: Any,
    *,
    request: Optional[str],
    trajectory: Sequence[Mapping[str, Any]],
    answer: Optional[str] = None,
    outcome: Optional[Mapping[str, Any]] = None,
) -> Optional[Dict[str, Any]]:
    """Keep the session as one worked example (``examples``); never raises. Returns what was stored.

    *answer* is the reply a checked outcome arrived on, when one did; else
    the session's last reply is the answer (never a stop notice).
    """
    if kind() != "examples":
        return None
    try:
        from unify.function_manager import origin_capture, task_origin

        gm = getattr(actor, "guidance_manager", None)
        if gm is None:
            return None
        text = request or task_origin.current_request() or ""
        cell = origin_capture.find_answer_cell(list(trajectory), answer=answer)
        code = getattr(cell, "code", None) if cell is not None else None
        given = (
            (cell.answer if cell is not None else None)
            or answer
            or _last_reply(trajectory)
        )
        title, content = example_text(
            request=text,
            code=code,
            answer=given,
            outcome=outcome,
        )
        gm.add_guidance(title=title, content=content)
        return {"title": title, "has_code": bool(code)}
    except (
        Exception
    ) as exc:  # noqa: BLE001 - keeping an example must never break a session
        logger.warning("worked example not kept: %s: %s", type(exc).__name__, exc)
        return None


__all__ = [
    "GUIDANCE_WRITE_TOOLS",
    "KINDS",
    "RULES",
    "example_text",
    "kind",
    "refused_tools",
    "require_prerequisites",
    "store_example",
]
