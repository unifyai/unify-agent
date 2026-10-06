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
* **Two modes.** ``refuse``: such a change is not applied and the writer is
  told why. ``versioned``: it is kept beside the entry as an unverified
  version (``versions``, at most :data:`VERSIONS_KEPT`, in the same
  metadata; the canonical content is what every read and listing shows),
  and replaces the content as soon as its session's answer is accepted --
  whenever that outcome is kept (:func:`promote_for`, called by
  :func:`~unify.function_manager.task_origin.record_outcome`). A version is
  not listed or used, so no later session can confirm it by use; a later
  accepted use verifies the canonical content instead. A deletion is
  refused in both modes.

It changes no prompt.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from . import task_origin

logger = logging.getLogger(__name__)

FIELD = "content_by"
VERSIONS = "versions"
VERSIONS_KEPT = 3
REFUSE = "refuse"
VERSIONED = "versioned"


def mode() -> Optional[str]:
    """``refuse`` or ``versioned`` (``UNIFY_PROTECT_VERIFIED``, with request records on); ``None`` when off."""
    from unify.settings import SETTINGS

    value = getattr(SETTINGS, "UNIFY_PROTECT_VERIFIED", "")
    if not task_origin.enabled() or not value:
        return None
    return VERSIONED if str(value).strip().lower() == VERSIONED else REFUSE


def enabled() -> bool:
    """``UNIFY_PROTECT_VERIFIED`` in either mode (with request records on)."""
    return mode() is not None


def versioned() -> bool:
    return mode() == VERSIONED


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


def protected(kind: str, ident: Any, row: Dict[str, Any]) -> Optional[str]:
    """The entry's verified status when this session may not change it; ``None`` when it may.

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
    return status if status.startswith("verified") else None


def refusal(
    kind: str,
    ident: Any,
    row: Dict[str, Any],
    *,
    action: str,
) -> Optional[str]:
    """Why *action* on this entry is not applied; ``None`` to apply it."""
    status = protected(kind, ident, row)
    if status is None:
        return None
    what = f"guidance {ident}" if kind == "guidance" else f"function `{ident}`"
    verb = {"delete": "deleted", "update": "changed"}.get(action, "changed")
    return (
        f"{what} is kept as it is: it is {status}, and this session's answer "
        f"is not known to be accepted, so it is not {verb} now. A lesson it "
        "does not cover can go in an entry of its own, which is listed as "
        "unverified until a session using it is accepted."
    )


# ── versioned: changes kept beside the content until their session is accepted ──


def versions(metadata: Any) -> List[Dict[str, Any]]:
    """The unverified versions kept beside an entry, oldest first."""
    value = metadata.get(VERSIONS) if isinstance(metadata, dict) else None
    return [v for v in value if isinstance(v, dict)] if isinstance(value, list) else []


def with_version(metadata: Any, fields: Dict[str, Any]) -> Dict[str, Any]:
    """*metadata* with *fields* kept as the current session's unverified version (replacing its earlier one)."""
    from unify import db

    key = current_key()
    task = task_origin._CURRENT.get()
    out = dict(metadata) if isinstance(metadata, dict) else {}
    kept = [v for v in versions(out) if v.get("by") != key]
    kept.append(
        {
            "by": key,
            "task": task.key if task is not None else None,
            "text": task.text if task is not None else None,
            "at": db.now_iso(),
            "fields": dict(fields),
        },
    )
    out[VERSIONS] = kept[-VERSIONS_KEPT:]
    return out


def version_note(kind: str, ident: Any, status: str) -> str:
    """What the writer is told when its change is kept as an unverified version."""
    what = f"guidance {ident}" if kind == "guidance" else f"function `{ident}`"
    return (
        f"{what} keeps its content: it is {status}, and this session's answer "
        "is not known to be accepted. This change is kept beside it as an "
        "unverified version, and replaces the content if this session's "
        "answer is accepted."
    )


def promoted(
    metadata: Any,
    key: str,
) -> Optional[tuple[Dict[str, Any], Dict[str, Any]]]:
    """``(fields, metadata)`` to apply for session *key*'s version, or ``None`` if it has none.

    The metadata loses that version, records *key* as the content's writer
    and the version's request among the origins.
    """
    found = next((v for v in versions(metadata) if v.get("by") == key), None)
    if found is None:
        return None
    out = dict(metadata) if isinstance(metadata, dict) else {}
    out[VERSIONS] = [v for v in versions(out) if v.get("by") != key]
    if not out[VERSIONS]:
        del out[VERSIONS]
    out[FIELD] = key
    if found.get("task") and found.get("text"):
        keys = [t for t in (out.get(task_origin.FIELD) or []) if isinstance(t, str)]
        if found["task"] not in keys:
            keys.append(found["task"])
        texts = [
            t for t in (out.get(task_origin.REQUESTS_FIELD) or []) if isinstance(t, str)
        ]
        texts = [t for t in texts if t != found["text"]] + [found["text"]]
        out[task_origin.FIELD] = keys
        out[task_origin.REQUESTS_FIELD] = texts[-task_origin.MAX_ORIGIN_REQUESTS :]
    return dict(found.get("fields") or {}), out


PROMOTED_REASON = "unverified version promoted: its session's answer was accepted"


def promote_for(key: Optional[str]) -> List[str]:
    """Apply every version session *key* wrote, now that its answer is accepted; the entries changed.

    Nothing unless the mode is ``versioned``. Reads and writes the store
    directly (both kinds), recording the replaced content in history as an
    update does.
    """
    import sqlite3

    from unify import db

    if not versioned() or not key:
        return []
    changed: List[str] = []
    try:
        from unify.guidance_manager.guidance_manager import GuidanceManager

        for row in db.query(
            "SELECT guidance_id, origin FROM guidance WHERE origin IS NOT NULL",
        ):
            origin = db.loads(row["origin"])
            found = promoted(origin, key)
            if found is None:
                continue
            fields, origin = found
            updates = {
                k: v
                for k, v in fields.items()
                if k in ("title", "content", "function_ids")
            }
            updates["origin"] = db.dumps(origin)
            GuidanceManager._update_row(
                int(row["guidance_id"]),
                updates,
                reason=PROMOTED_REASON,
            )
            if "function_ids" in updates:
                from . import entry_links

                entry_links.set_guidance_links(
                    int(row["guidance_id"]),
                    updates["function_ids"],
                )
            changed.append(f"guidance {row['guidance_id']}")
        from .function_manager import FunctionManager, VERSION_FIELDS

        for row in db.query(
            "SELECT function_id, name, metadata FROM functions WHERE metadata LIKE ?",
            (f'%"{VERSIONS}"%',),
        ):
            metadata = db.loads(row["metadata"])
            found = promoted(metadata, key)
            if found is None:
                continue
            fields, metadata = found
            changes = {k: v for k, v in fields.items() if k in VERSION_FIELDS}
            changes["metadata"] = metadata
            with db.transaction():
                FunctionManager._update_function(
                    int(row["function_id"]),
                    changes,
                    reason=PROMOTED_REASON,
                )
            changed.append(f"function {row['name']}")
    except (sqlite3.Error, ValueError, TypeError) as exc:
        logger.warning(f"versions not promoted: {type(exc).__name__}: {exc}")
    if changed:
        logger.info(f"unverified versions promoted: {changed}")
    return changed


__all__ = [
    "FIELD",
    "PROMOTED_REASON",
    "REFUSE",
    "VERSIONED",
    "VERSIONS",
    "VERSIONS_KEPT",
    "content_by",
    "current_key",
    "enabled",
    "mode",
    "promote_for",
    "promoted",
    "protected",
    "refusal",
    "stamped",
    "version_note",
    "versioned",
    "versions",
    "with_version",
]
