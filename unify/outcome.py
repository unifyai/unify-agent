"""The checked outcome of a request, posted by the environment that ran it.

An environment that checks the work (a benchmark's grader, a user's verdict) gives it to the harness as a control
line on the stdin channel ``unify act --jsonl`` already reads (``{"outcome": {...}}``). With memory on, the
request's memory run keeps the verdict as the episode's checked signal (unify/memory/integration/request.py);
with memory off, the line is refused.

The schema::

    {"solved": true | false | null,
     "score": <number> | null,
     "checks": [{"name": str, "passed": true | false | null, "reason": str}],
     "source": "grader" | "user" | ...,
     "summary": str}

Every key is optional. At most :data:`MAX_CHECKS` checks are kept (failed checks first, each group in the order
given) and every text is cut to a fixed length. Anything else is refused with :class:`OutcomeError`.
"""

from __future__ import annotations

import math
import re
from typing import Any

MAX_CHECKS = 20
MAX_NAME = 120
MAX_REASON = 400
MAX_SOURCE = 40
MAX_SUMMARY = 1200


class OutcomeError(ValueError):
    """A posted outcome that is malformed."""


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
    """The outcome as memory keeps it; :class:`OutcomeError` if malformed."""
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
