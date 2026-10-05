"""``UNIFY_PROTECT_VERIFIED``: a session not known to be accepted never replaces a verified entry.

On the 5 Oct Continual-ARC paper-protocol run (research artifact
long-horizon-v1) a failed visit twice replaced a specific rule that a solved
visit had written with generic advice; one of those rules was the task's
correct one. Voyager, LEGO-Prover and the Continual-ARC program library admit
only what a check accepted. This switch keeps that one rule for a library a
model curates, reading only which request a write runs under and the
outcomes the harness already keeps (the checker's, else the storage
review's judgement):

* **Who wrote the content.** A function or guidance entry written under a
  keyed request records that request's hash as the writer of its current
  content (``content_by``, in a function's ``metadata`` or a guidance entry's
  hidden ``origin``).
* **Verified** content was written in a session whose answer was accepted,
  or was called or relied on in a later accepted session
  (:func:`unify.function_manager.entry_record.status`).
* **Never replaced from an unaccepted session.** While the current session's
  answer is not known to be accepted (not accepted, or not known yet), an
  overwrite, patch, update or deletion of a verified entry written by
  another session is not applied; the writer is told why and that an entry
  of its own (marked unverified) is the way to keep its lesson.
  Entries it writes are shown unverified until accepted.

It changes no prompt. Versions kept beside the content are a separate
question (not built here).
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from . import task_origin

FIELD = "content_by"


def enabled() -> bool:
    """``UNIFY_PROTECT_VERIFIED`` (with request records on)."""
    from unify.settings import SETTINGS

    return task_origin.enabled() and bool(
        getattr(SETTINGS, "UNIFY_PROTECT_VERIFIED", False),
    )


def current_key() -> Optional[str]:
    text = task_origin.current_request()
    return task_origin.text_key(text) if text else None


def stamped(metadata: Any) -> Optional[Dict[str, Any]]:
    """*metadata* with the current session as the writer of the content; ``None`` when off or unkeyed."""
    key = current_key()
    if not enabled() or key is None:
        return None
    out = dict(metadata) if isinstance(metadata, dict) else {}
    out[FIELD] = key
    return out


def content_by(row: Dict[str, Any]) -> Optional[str]:
    metadata = row.get("metadata")
    value = metadata.get(FIELD) if isinstance(metadata, dict) else None
    return value if isinstance(value, str) and value else None


def refusal(
    kind: str,
    ident: Any,
    row: Dict[str, Any],
    *,
    action: str,
) -> Optional[str]:
    """Why *action* on this entry is not applied; ``None`` to apply it.

    *row* carries the entry's origin as ``metadata``.
    """
    from . import entry_record

    if not enabled():
        return None
    key = current_key()
    if key is None:
        return None
    writer = content_by(row)
    if writer is not None and writer == key:
        return None
    if entry_record.outcome_of_key(key) is True:
        return None
    uses = None
    if entry_record.enabled():
        uses = entry_record.uses_of([(kind, str(ident))])[(kind, str(ident))]
    status = entry_record.status(kind, row, uses)
    if not status.startswith("verified"):
        return None
    what = f"guidance {ident}" if kind == "guidance" else f"function `{ident}`"
    verb = {"delete": "deleted", "update": "changed"}.get(action, "changed")
    return (
        f"{what} is kept as it is: it is {status}, and this session's answer "
        f"is not known to be accepted, so it is not {verb} now. A lesson it "
        "does not cover can go in an entry of its own, which is listed as "
        "unverified until a session using it is accepted."
    )


__all__ = ["FIELD", "content_by", "current_key", "enabled", "refusal", "stamped"]
