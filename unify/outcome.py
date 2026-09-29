"""The checked outcome of a session, posted by the environment that ran it (``UNIFY_OUTCOME``).

The storage review that follows a session decides what the libraries keep,
and until now it knew only what the agent said about its own work. An
environment that checks the work (a benchmark's grader, a user's verdict)
can post the result here, into the harness process: the review of that
session then reads it in a section of its own, marked as coming from the
checker and not from the agent.

The outcome is held in memory, on the session's handle. It is never
written to a file and never put in the environment, so nothing the agent
runs through the workspace can read it from disk. ``unify act --jsonl``
receives it as a control line on the stdin channel the driving program
already writes (``{"outcome": {...}}``); a program that runs the actor in
process calls :func:`post` itself.

The schema::

    {"solved": true | false | null,
     "score": <number> | null,
     "checks": [{"name": str, "passed": true | false | null, "reason": str}],
     "source": "grader" | "user" | ...,
     "summary": str}

Every key is optional. At most :data:`MAX_CHECKS` checks are kept (failed
checks first, each group in the order given) and every text is cut to a
fixed length, so a posted outcome cannot flood the review's prompt.
Anything else is refused with :class:`OutcomeError`.
"""

from __future__ import annotations

import math
import re
import weakref
from typing import Any, Optional, Protocol

MAX_CHECKS = 20
MAX_NAME = 120
MAX_REASON = 400
MAX_SOURCE = 40
MAX_SUMMARY = 1200

OUTCOME_HEADER = "## Verified outcome (from the environment's checker, not the agent)"
LESSONS_HEADER = "## Failure lessons"

# The library writes a review of a failed run may not make.
LESSON_REFUSED_TOOLS = (
    "FunctionManager_add_functions",
    "FunctionManager_delete_function",
    "FunctionManager_reconcile_dependencies",
    "FunctionManager_check_function",
    "FunctionManager_patch_function",
)
LESSON_MASK_RULE = (
    "this review covers a run that failed its check, so it records lessons "
    "as guidance and makes no function writes"
)


class OutcomeError(ValueError):
    """A posted outcome that is malformed or has no session to go to."""


class OutcomeReceiver(Protocol):
    def receive_outcome(self, outcome: dict) -> None: ...


_RECEIVERS: "weakref.WeakValueDictionary[str, Any]" = weakref.WeakValueDictionary()


def enabled() -> bool:
    """Whether ``UNIFY_OUTCOME`` is on."""
    from unify.settings import SETTINGS

    return bool(getattr(SETTINGS, "UNIFY_OUTCOME", False))


def review_failed_mode() -> str:
    """``UNIFY_REVIEW_FAILED``: ``"lessons"`` or ``""``."""
    from unify.settings import SETTINGS

    return str(getattr(SETTINGS, "UNIFY_REVIEW_FAILED", "") or "")


def _text(value: Any, limit: int, what: str) -> str:
    if value is None:
        return ""
    if not isinstance(value, str):
        raise OutcomeError(f"{what} must be a string")
    flat = re.sub(r"\s+", " ", value).strip()
    if len(flat) > limit:
        flat = flat[: limit - 1].rstrip() + "…"
    return flat


def normalize(raw: Any) -> dict:
    """The outcome as the review will read it; :class:`OutcomeError` if malformed."""
    if not isinstance(raw, dict):
        raise OutcomeError("an outcome is a JSON object")
    solved = raw.get("solved")
    if solved is not None and not isinstance(solved, bool):
        raise OutcomeError("solved must be true, false or null")
    score = raw.get("score")
    if score is not None:
        if isinstance(score, bool) or not isinstance(score, (int, float)):
            raise OutcomeError("score must be a number or null")
        if not math.isfinite(score):
            raise OutcomeError("score must be finite")
    checks_raw = raw.get("checks")
    if checks_raw is None:
        checks_raw = []
    if not isinstance(checks_raw, list):
        raise OutcomeError("checks must be a list")
    checks: list[dict] = []
    for i, item in enumerate(checks_raw):
        if not isinstance(item, dict):
            raise OutcomeError(f"check {i} is not an object")
        passed = item.get("passed")
        if passed is not None and not isinstance(passed, bool):
            raise OutcomeError(f"check {i}: passed must be true, false or null")
        checks.append(
            {
                "name": _text(item.get("name"), MAX_NAME, f"check {i}: name")
                or f"check {i + 1}",
                "passed": passed,
                "reason": _text(item.get("reason"), MAX_REASON, f"check {i}: reason"),
            },
        )
    kept = checks
    if len(checks) > MAX_CHECKS:
        failed = [c for c in checks if c["passed"] is not True]
        others = [c for c in checks if c["passed"] is True]
        kept = (failed + others)[:MAX_CHECKS]
    out: dict = {
        "solved": solved,
        "score": score,
        "checks": kept,
        "checks_total": len(checks),
        "checks_passed": sum(1 for c in checks if c["passed"] is True),
        "source": _text(raw.get("source"), MAX_SOURCE, "source") or "unspecified",
    }
    summary = _text(raw.get("summary"), MAX_SUMMARY, "summary")
    if summary:
        out["summary"] = summary
    return out


def register(session_id: str, receiver: OutcomeReceiver) -> None:
    """Name *receiver* as the session an outcome posted to *session_id* goes to.

    Held weakly: a session that has ended and been dropped takes no outcome.
    """
    _RECEIVERS[session_id] = receiver


def post(session_id: Optional[str], outcome: Any) -> dict:
    """Give the session *session_id* its checked outcome; returns it normalized.

    Raises :class:`OutcomeError` when ``UNIFY_OUTCOME`` is off, the outcome is
    malformed, no live session has that id, or the session no longer takes
    one (its review has started).
    """
    if not enabled():
        raise OutcomeError("UNIFY_OUTCOME is off")
    normalized = normalize(outcome)
    receiver = _RECEIVERS.get(session_id) if session_id else None
    if receiver is None:
        raise OutcomeError(f"no session takes an outcome under id {session_id!r}")
    receiver.receive_outcome(normalized)
    return normalized


def _verdict_word(value: Optional[bool]) -> str:
    return {True: "yes", False: "no", None: "not stated"}[value]


def render(outcome: Optional[dict], *, lessons: bool = False) -> str:
    """The review's section on the checked outcome, and on failure lessons.

    Empty when there is neither an outcome nor a lessons-only review.
    """
    parts: list[str] = []
    if outcome is not None:
        lines = [
            f"{OUTCOME_HEADER}\n",
            "After the session the environment checked the task itself. This "
            f"verdict comes from its checker (source: `{outcome['source']}`), "
            "not from the agent: where the conversation claims a different "
            "result, the verdict is right and the claim is wrong.\n",
            f"- Solved: {_verdict_word(outcome['solved'])}",
        ]
        if outcome.get("score") is not None:
            lines.append(f"- Score: {outcome['score']:g}")
        total = outcome.get("checks_total", len(outcome["checks"]))
        if total:
            shown = len(outcome["checks"])
            more = f"; {total - shown} not shown" if total > shown else ""
            lines.append(
                f"- Checks: {outcome.get('checks_passed', 0)} of {total} passed{more}",
            )
            for check in outcome["checks"]:
                mark = {True: "passed", False: "FAILED", None: "unknown"}[
                    check["passed"]
                ]
                reason = f": {check['reason']}" if check["reason"] else ""
                lines.append(f"  - {mark} `{check['name']}`{reason}")
        if outcome.get("summary"):
            lines.append(f"\nChecker's summary: {outcome['summary']}")
        if outcome["solved"] is False:
            lines.append(
                "\nThe task was not solved, so the procedure the conversation "
                "followed did not work as a whole, whatever the agent said. Do "
                "not store it as a working function; keep only a part the "
                "checks show to be sound, if any.",
            )
        elif outcome["solved"] is True:
            lines.append("\nThe task was solved.")
        else:
            lines.append(
                "\nThe checker gave no overall verdict; weigh the checks above.",
            )
        parts.append("\n".join(lines) + "\n\n")
    if lessons:
        refused = ", ".join(f"`{name}`" for name in LESSON_REFUSED_TOOLS)
        parts.append(
            f"{LESSONS_HEADER}\n\n"
            "This run failed its check, so this review records lessons, not "
            f"functions: function writes ({refused}) are refused. If the "
            "failure teaches something a future task of this kind should do "
            "differently -- the mistake, the check that caught it, and what "
            "would have passed -- record it as guidance "
            "(`GuidanceManager_add_guidance`, or update the entry that covers "
            "it). Name the mistake concretely, and never record the failed "
            "procedure as a recipe. If the failure teaches nothing general, "
            "store nothing.\n\n",
        )
    return "".join(parts)
