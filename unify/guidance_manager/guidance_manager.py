from __future__ import annotations

import functools
import logging
import re
import sqlite3
from contextvars import ContextVar
from typing import Any, Dict, FrozenSet, List, Optional, Union

from unify import db
from ..common.exact_patch import PatchEdit
from ..common.sql_filters import and_clauses, invalid_filter_error, not_in
from ..common.stale_reason import StaleReason, merge_stale_reasons
from ..common.semantic_search import rank_by_similarity
from ..common.tool_outcome import ToolOutcome
from .base import BaseGuidanceManager
from .builtins import builtin_guidance_enabled, ensure_seeded
from .types.guidance import (
    Guidance,
    GuidanceWithLinks,
    GuidanceWithLinksAndRecord,
    GuidanceWithRecord,
)


def _origin_for(guidance_id: Optional[int]) -> Optional[str]:
    """``UNIFY_GUIDANCE_ORIGIN``: the entry's ``origin`` with the current request added.

    The JSON to store, or ``None`` while the switch is off or no request is
    keyed (the column is then left as it is). ``UNIFY_LISTING_PROVENANCE``
    and ``UNIFY_LESSON_STATUS`` record it too, for the shortlist's lines.
    """
    from ..function_manager import task_origin

    if not task_origin.guidance_recorded():
        return None
    prior = _stored_origin(guidance_id) if guidance_id is not None else None
    stamped = task_origin.stamped(prior)
    return None if stamped is None else db.dumps(stamped)


def _stored_origin(guidance_id: Any) -> Optional[Dict[str, Any]]:
    """The hidden ``origin`` of stored entry *guidance_id* (a dict), or ``None``."""
    row = db.query_one(
        "SELECT origin FROM guidance WHERE guidance_id = ?",
        (int(guidance_id),),
    )
    prior = db.loads(row["origin"]) if row and row["origin"] else None
    return prior if isinstance(prior, dict) else None


def _record_enabled() -> bool:
    from ..function_manager import entry_record

    return entry_record.enabled()


def _identifier_search_enabled() -> bool:
    from unify.settings import SETTINGS
    from ..function_manager import task_origin

    return task_origin.enabled() and bool(
        getattr(SETTINGS, "UNIFY_SEARCH_IDENTIFIERS", False),
    )


logger = logging.getLogger(__name__)

# Content cap for search/filter result payloads. Entries (notably imported
# builtin skills) can run to 100KB+; returning them wholesale from list-style
# reads floods the caller's context window. Reads above this cap return a
# preview and the full text is fetched per entry via ``get_guidance``.
GUIDANCE_PREVIEW_CHARS = 2000

_SELECT = f"SELECT {', '.join(db.GUIDANCE_COLUMNS)} FROM all_guidance"

# UNIFY_FUNCTION_PATCH: the reason ``guidance_history`` records for an update.
# ``patch_guidance`` sets it around its ``update_guidance`` call; a plain
# ``update_guidance`` records the default.
_UPDATE_REASON: ContextVar[Optional[str]] = ContextVar(
    "guidance_update_reason",
    default=None,
)
DEFAULT_UPDATE_REASON = "updated with update_guidance"


def _patch_enabled() -> bool:
    from unify.settings import SETTINGS

    return bool(SETTINGS.UNIFY_FUNCTION_PATCH)


def _linked_names_enabled() -> bool:
    from unify.settings import SETTINGS

    return bool(getattr(SETTINGS, "UNIFY_GUIDANCE_LINKED_NAMES", False))


def _call_signature(row: Dict[str, Any]) -> str:
    """``name(signature)``, with `` (async)`` for an ``async def``."""
    name = str(row.get("name") or "")
    argspec = str(row.get("argspec") or "").strip()
    text = name + (argspec if argspec.startswith("(") else f"({argspec})")
    source = str(row.get("implementation") or "")
    if re.search(rf"^\s*async\s+def\s+{re.escape(name)}\b", source, re.M):
        text += " (async)"
    return text


def _with_linked_functions(entries: List[Guidance]) -> List[Guidance]:
    """``UNIFY_GUIDANCE_LINKED_NAMES``: *entries* naming the functions they link.

    Each becomes a ``Guidance`` read with ``linked_functions``: the name and
    signature of each id in ``function_ids`` that a stored function has, in
    that order. Off: *entries* as they are.
    """
    if not _linked_names_enabled():
        return entries
    ids = sorted(
        {int(i) for entry in entries for i in (entry.function_ids or [])},
    )
    found: Dict[int, str] = {}
    if ids:
        marks = ", ".join("?" for _ in ids)
        for row in db.query(
            "SELECT function_id, name, argspec, implementation FROM functions "
            f"WHERE function_id IN ({marks})",
            tuple(ids),
        ):
            found[int(row["function_id"])] = _call_signature(row)
    return [
        GuidanceWithLinks(
            **entry.model_dump(),
            linked_functions=[
                found[int(i)] for i in entry.function_ids if int(i) in found
            ],
        )
        for entry in entries
    ]


def _instance_warning(title: Optional[str], content: Optional[str]) -> Optional[str]:
    """``UNIFY_STORE_INSTANCE_LINT``: a warning when the entry names this task instance."""
    from unify.function_manager import instance_lint

    if not instance_lint.enabled():
        return None
    return instance_lint.join_warnings(
        [
            instance_lint.text_warning("its title", title or ""),
            instance_lint.text_warning("its content", content or ""),
        ],
    )


def _with_warning(outcome: Dict[str, Any], warning: Optional[str]) -> Dict[str, Any]:
    if warning:
        outcome["warning"] = warning
    return outcome


def _stored_only_without_reference(references: Optional[Dict[str, str]]) -> bool:
    """True when ``UNIFY_GUIDANCE_EMPTY_QUERY=stored`` and no reference text is given.

    Without reference text every row is unscored and ordered newest first by
    id. Built-in ids are hashes (up to 2**31) while stored ids count up from 1,
    so the built-in catalogue would fill every slot ahead of the stored
    entries; under the switch such a search reads only the stored entries.
    """
    from unify.settings import SETTINGS

    if SETTINGS.UNIFY_GUIDANCE_EMPTY_QUERY != "stored":
        return False
    if not references:
        return True
    if not isinstance(references, dict):
        return False
    return not any(str(text or "").strip() for text in references.values())


class GuidanceManager(BaseGuidanceManager):
    """Guidance stored in the ``guidance`` table, read alongside the builtins."""

    def __init__(
        self,
        *,
        rolling_summary_in_prompts: bool = True,
        filter_scope: Optional[str] = None,
        exclude_ids: Optional[FrozenSet[int]] = None,
    ) -> None:
        super().__init__()
        self._filter_scope = filter_scope
        self._exclude_ids = frozenset(exclude_ids) if exclude_ids else None
        self._rolling_summary_in_prompts = rolling_summary_in_prompts
        ensure_seeded()

    # -- Scope / exclusion properties ----------------------------------------

    @property
    def filter_scope(self) -> Optional[str]:
        """A SQL WHERE clause permanently applied to all read queries."""
        return self._filter_scope

    @filter_scope.setter
    def filter_scope(self, value: Optional[str]) -> None:
        self._filter_scope = value

    @property
    def exclude_ids(self) -> Optional[FrozenSet[int]]:
        """Guidance IDs excluded from all read queries."""
        return self._exclude_ids

    @exclude_ids.setter
    def exclude_ids(self, value: Optional[FrozenSet[int]]) -> None:
        self._exclude_ids = frozenset(value) if value else None

    def _scope(self, caller_filter: Optional[str] = None) -> Optional[str]:
        """Compose *caller_filter* with ``filter_scope``, the id exclusions and,
        under ``UNIFY_BUILTIN_GUIDANCE=0``, the stored entries only."""
        return and_clauses(
            caller_filter,
            self._filter_scope,
            not_in("guidance_id", self._exclude_ids),
            None if builtin_guidance_enabled() else "is_builtin = 0",
        )

    # -- Reads ------------------------------------------------------------------

    def _rows(
        self,
        where: Optional[str],
        *,
        limit: Optional[int] = None,
        offset: int = 0,
        readonly: bool = False,
    ) -> List[Dict[str, Any]]:
        sql = _SELECT
        if where:
            sql += f" WHERE {where}"
        sql += " ORDER BY is_builtin, guidance_id"
        if limit is not None:
            sql += f" LIMIT {int(limit)} OFFSET {int(offset)}"
        rows = db.query_readonly(sql) if readonly else db.query(sql)
        return [db.decode(row, db.GUIDANCE_JSON_COLUMNS) for row in rows]

    def _own_row(self, guidance_id: int) -> Optional[Dict[str, Any]]:
        row = db.query_one(
            "SELECT guidance_id, title, content, function_ids, stale_reasons "
            "FROM guidance WHERE guidance_id = ?",
            (int(guidance_id),),
        )
        return db.decode(row, db.GUIDANCE_JSON_COLUMNS) if row else None

    @staticmethod
    def _is_builtin_guidance(guidance_id: int) -> bool:
        if not builtin_guidance_enabled():
            return False
        return (
            db.query_one(
                "SELECT 1 FROM builtin_guidance WHERE guidance_id = ?",
                (int(guidance_id),),
            )
            is not None
        )

    def _raise_if_builtin(self, guidance_id: int, action: str) -> None:
        """Refuse mutations of builtins entries with an actionable error."""
        if self._is_builtin_guidance(guidance_id):
            raise ValueError(
                f"guidance_id {guidance_id} is a built-in platform guidance "
                f"entry and cannot be {action}. Built-in guidance is "
                "read-only for everyone. To tailor it, create your own "
                "entry with add_guidance (optionally adapting the built-in "
                "content); that copy can then be updated or deleted freely.",
            )

    @staticmethod
    def _available_functions_by_id() -> dict[int, str]:
        """Return the stored functions keyed by id."""
        return {
            int(row["function_id"]): str(row["name"])
            for row in db.query("SELECT function_id, name FROM functions")
        }

    @staticmethod
    def _missing_function_reasons(
        guidance: Guidance,
        *,
        available: dict[int, str],
        preserve_historical: bool,
    ) -> list[StaleReason]:
        preserved = [
            reason for reason in guidance.stale_reasons if reason.dep_kind != "function"
        ]
        candidates: dict[int, str | None] = {
            int(function_id): None for function_id in guidance.function_ids
        }
        if preserve_historical:
            for reason in guidance.stale_reasons:
                if reason.dep_kind == "function" and reason.id is not None:
                    candidates.setdefault(int(reason.id), reason.name)
        missing = [
            StaleReason(
                dep_kind="function",
                id=function_id,
                name=name,
                message=(
                    f"missing function_id={function_id}"
                    + (f" name={name}" if name else "")
                ),
            )
            for function_id, name in candidates.items()
            if function_id not in available
        ]
        return merge_stale_reasons(preserved, *missing)

    @staticmethod
    def _with_content_preview(row: Guidance) -> Guidance:
        """Return *row* with content truncated to the list-read preview cap."""
        if len(row.content) <= GUIDANCE_PREVIEW_CHARS:
            return row
        preview = (
            row.content[:GUIDANCE_PREVIEW_CHARS]
            + f"\n\n… [content preview truncated at {GUIDANCE_PREVIEW_CHARS:,} "
            f"of {len(row.content):,} chars — fetch the full entry with "
            f"get_guidance(guidance_id={row.guidance_id})]"
        )
        return row.model_copy(update={"content": preview})

    def _num_items(self) -> int:
        sql = "SELECT COUNT(*) AS n FROM all_guidance"
        where = self._scope()
        if where:
            sql += f" WHERE {where}"
        return int(db.query_one(sql)["n"])

    @functools.wraps(BaseGuidanceManager.clear, updated=())
    def clear(self) -> None:
        with db.transaction() as conn:
            conn.execute("DELETE FROM guidance")
            conn.execute("DELETE FROM sqlite_sequence WHERE name = 'guidance'")

    # -- Writes -----------------------------------------------------------------

    @functools.wraps(BaseGuidanceManager.add_guidance, updated=())
    def add_guidance(
        self,
        *,
        title: Optional[str] = None,
        content: Optional[str] = None,
        function_ids: Optional[List[int]] = None,
    ) -> ToolOutcome:
        if not title and not content:
            raise ValueError(
                "At least one field (title/content) must be provided.",
            )
        g = Guidance(
            title=title or "",
            content=content or "",
            function_ids=function_ids or [],
        )
        values = (
            g.title,
            g.content,
            db.dumps(g.function_ids),
            db.dumps(
                [reason.model_dump(mode="json") for reason in g.stale_reasons],
            ),
            db.now_iso(),
        )
        # UNIFY_GUIDANCE_ORIGIN: the request this entry was written for.
        origin = _origin_for(None)
        if origin is None:
            cursor = db.execute(
                "INSERT INTO guidance (title, content, function_ids, stale_reasons, created_at)"
                " VALUES (?, ?, ?, ?, ?)",
                values,
            )
        else:
            cursor = db.execute(
                "INSERT INTO guidance (title, content, function_ids, stale_reasons,"
                " created_at, origin) VALUES (?, ?, ?, ?, ?, ?)",
                (*values, origin),
            )
        # UNIFY_ENTRY_RECORD: the links, in the many-to-many link table too.
        if g.function_ids and _record_enabled():
            from ..function_manager import entry_links

            entry_links.set_guidance_links(int(cursor.lastrowid), g.function_ids)
        return _with_warning(
            {
                "outcome": "guidance created successfully",
                "details": {"guidance_id": int(cursor.lastrowid)},
            },
            _instance_warning(g.title, g.content),
        )

    @functools.wraps(BaseGuidanceManager.update_guidance, updated=())
    def update_guidance(
        self,
        *,
        guidance_id: int,
        title: Optional[str] = None,
        content: Optional[str] = None,
        function_ids: Optional[List[int]] = None,
    ) -> ToolOutcome:
        updates: Dict[str, Any] = {}
        if title is not None:
            updates["title"] = title
        if content is not None:
            updates["content"] = content
        if function_ids is not None:
            validated = Guidance(
                title=title or "tmp",
                content=content or "tmp",
                function_ids=function_ids,
            )
            updates["function_ids"] = validated.function_ids
        if not updates:
            raise ValueError("At least one field must be provided for an update.")

        self._raise_if_builtin(guidance_id, "updated")
        row = self._own_row(guidance_id)
        if row is None:
            raise ValueError(
                f"No guidance found with guidance_id {guidance_id} to update.",
            )
        if function_ids is not None:
            candidate = Guidance(**{**row, **updates})
            updates["stale_reasons"] = [
                reason.model_dump(mode="json")
                for reason in self._missing_function_reasons(
                    candidate,
                    available=self._available_functions_by_id(),
                    preserve_historical=False,
                )
            ]
        # UNIFY_GUIDANCE_ORIGIN: the request this revision was written for.
        origin = _origin_for(guidance_id)
        if origin is not None:
            updates["origin"] = origin
        self._update_row(
            guidance_id,
            updates,
            reason=_UPDATE_REASON.get() or DEFAULT_UPDATE_REASON,
        )
        # UNIFY_ENTRY_RECORD: the links, in the many-to-many link table too.
        if "function_ids" in updates and _record_enabled():
            from ..function_manager import entry_links

            entry_links.set_guidance_links(guidance_id, updates["function_ids"])
        return _with_warning(
            {"outcome": "guidance updated", "details": {"guidance_id": guidance_id}},
            _instance_warning(title, content),
        )

    @staticmethod
    def _update_row(
        guidance_id: int,
        updates: Dict[str, Any],
        *,
        reason: Optional[str] = None,
    ) -> None:
        """Apply ``updates`` to one stored guidance entry.

        ``reason`` marks an edit of the entry (not a refresh of its stale
        reasons); with ``UNIFY_FUNCTION_PATCH`` on, the row as it was is first
        appended to ``guidance_history`` with that reason, in one transaction.
        """
        assignments = ", ".join(f"{column} = ?" for column in updates)
        values = [
            db.dumps(value) if column in db.GUIDANCE_JSON_COLUMNS else value
            for column, value in updates.items()
        ]
        sql = f"UPDATE guidance SET {assignments} WHERE guidance_id = ?"
        if reason is not None and _patch_enabled():
            with db.transaction():
                GuidanceManager._record_guidance_history(guidance_id, reason)
                db.execute(sql, [*values, int(guidance_id)])
            return
        db.execute(sql, [*values, int(guidance_id)])

    @staticmethod
    def _record_guidance_history(guidance_id: int, reason: str) -> None:
        """Append the stored row of ``guidance_id``, as it is now, to ``guidance_history``."""
        row = db.query_one(
            "SELECT * FROM guidance WHERE guidance_id = ?",
            (int(guidance_id),),
        )
        if row is None:
            return
        previous = db.decode(dict(row), db.GUIDANCE_JSON_COLUMNS)
        db.execute(
            "INSERT INTO guidance_history"
            " (guidance_id, title, previous, reason, replaced_at)"
            " VALUES (?, ?, ?, ?, ?)",
            (
                int(guidance_id),
                previous["title"],
                db.dumps(previous),
                str(reason),
                db.now_iso(),
            ),
        )

    def patch_guidance(
        self,
        *,
        id_or_title: Union[int, str],
        old: Optional[str] = None,
        new: Optional[str] = None,
        why: str,
        edits: Optional[List[PatchEdit]] = None,
        replace_all: bool = False,
        old_string: Optional[str] = None,
        new_string: Optional[str] = None,
    ) -> ToolOutcome:
        """Fix a stored guidance entry in place by replacing excerpts of its content.

        Prefer this to adding a second entry or rewriting the whole entry.
        Read the current content first (``get_guidance``) and copy the text to
        change as it appears. Give one change as ``old``/``new``, or several
        as ``edits``: they apply in order, each to the content as the edits
        before it left it, and all or none are stored. Each ``old`` must occur
        exactly once (unless ``replace_all``); if it is not found exactly,
        differences in line endings, trailing spaces, indentation and runs of
        spaces are tolerated while the match stays unique. Otherwise nothing
        changes and the error shows the closest text or every occurrence. The
        entry keeps its id, title and ``function_ids``, and the version it
        replaces is kept in history with ``why``. Built-in entries cannot be
        patched.

        Args:
            id_or_title: The entry's ``guidance_id``, or its exact title.
            old: The text to replace, with enough surrounding text to occur
                only once.
            new: The replacement text (an empty string deletes ``old``).
            why: One sentence on what was wrong or missing; kept with the
                replaced version.
            edits: Several changes in one call, instead of ``old``/``new``:
                ``[{"old", "new", "replace_all"?}, ...]``.
            replace_all: With ``old``/``new``, replace every occurrence of
                ``old`` instead of exactly one.
            old_string: Another name for ``old``.
            new_string: Another name for ``new``.

        Returns:
            ``{"outcome": "guidance patched", "details": {"guidance_id",
            "edits"}}``, where ``edits`` gives, per edit, how it matched
            (``exact``, ``trailing_whitespace``, ``indentation`` or
            ``collapsed_whitespace``) and how many occurrences it replaced.

        Raises:
            ValueError: Nothing was changed; the message says why.
        """
        from unify.common.exact_patch import PatchRefused, apply_edits, collect_edits

        if not _patch_enabled():
            raise ValueError(
                "patching is not enabled here (UNIFY_FUNCTION_PATCH is off)",
            )
        if not str(why or "").strip():
            raise ValueError("say `why` the entry needs this patch")
        try:
            wanted = collect_edits(
                old=old,
                new=new,
                edits=edits,
                replace_all=replace_all,
                old_string=old_string,
                new_string=new_string,
            )
        except PatchRefused as exc:
            raise ValueError(str(exc)) from None
        row = self._stored_entry(id_or_title)
        guidance_id = int(row["guidance_id"])
        try:
            patched, report = apply_edits(
                row["content"],
                wanted,
                what=f"the content of guidance {guidance_id}",
            )
        except PatchRefused as exc:
            raise ValueError(str(exc)) from None
        token = _UPDATE_REASON.set(str(why).strip())
        try:
            updated = self.update_guidance(guidance_id=guidance_id, content=patched)
        finally:
            _UPDATE_REASON.reset(token)
        return _with_warning(
            {
                "outcome": "guidance patched",
                "details": {"guidance_id": guidance_id, "edits": report},
            },
            updated.get("warning"),
        )

    def _stored_entry(self, id_or_title: Union[int, str]) -> Dict[str, Any]:
        """The stored entry named by id (or a string of digits) or by exact title."""
        if isinstance(id_or_title, bool) or not isinstance(id_or_title, (int, str)):
            raise ValueError("id_or_title must be a guidance_id or a title")
        text = str(id_or_title).strip()
        if isinstance(id_or_title, int) or text.isdigit():
            guidance_id = int(text)
            self._raise_if_builtin(guidance_id, "patched")
            row = self._own_row(guidance_id)
            if row is not None:
                return row
            if isinstance(id_or_title, int):
                raise ValueError(f"No guidance found with guidance_id {guidance_id}.")
        matches = db.query(
            "SELECT guidance_id FROM guidance WHERE title = ? ORDER BY guidance_id",
            (text,),
        )
        if len(matches) > 1:
            ids = ", ".join(str(m["guidance_id"]) for m in matches)
            raise ValueError(
                f"{len(matches)} stored entries are titled {text!r} "
                f"(guidance_ids {ids}); patch one by its guidance_id.",
            )
        if matches:
            return self._own_row(int(matches[0]["guidance_id"]))
        builtin = db.query_one(
            "SELECT guidance_id FROM builtin_guidance WHERE title = ?",
            (text,),
        )
        if builtin is not None:
            self._raise_if_builtin(int(builtin["guidance_id"]), "patched")
        raise ValueError(f"No stored guidance is titled {text!r}.")

    @functools.wraps(BaseGuidanceManager.delete_guidance, updated=())
    def delete_guidance(
        self,
        *,
        guidance_id: int,
    ) -> ToolOutcome:
        self._raise_if_builtin(guidance_id, "deleted")
        deleted = db.execute(
            "DELETE FROM guidance WHERE guidance_id = ?",
            (int(guidance_id),),
        ).rowcount
        if not deleted:
            raise ValueError(
                f"No guidance found with guidance_id {guidance_id} to delete.",
            )
        # UNIFY_ENTRY_RECORD: its links go with it.
        if _record_enabled():
            from ..function_manager import entry_links

            entry_links.drop(guidance_id=guidance_id)
        return {"outcome": "guidance deleted", "details": {"guidance_id": guidance_id}}

    @functools.wraps(BaseGuidanceManager.reconcile_dependencies, updated=())
    def reconcile_dependencies(
        self,
        *,
        guidance_ids: Optional[List[int]] = None,
    ) -> ToolOutcome:
        sql = "SELECT guidance_id, title, content, function_ids, stale_reasons FROM guidance"
        if guidance_ids:
            ids = ", ".join(str(int(gid)) for gid in guidance_ids)
            sql += f" WHERE guidance_id IN ({ids})"
        rows = [db.decode(row, db.GUIDANCE_JSON_COLUMNS) for row in db.query(sql)]
        available = self._available_functions_by_id()
        stale_guidance_ids: list[int] = []
        for row in rows:
            guidance = Guidance(**row)
            refreshed = self._missing_function_reasons(
                guidance,
                available=available,
                preserve_historical=True,
            )
            if refreshed:
                stale_guidance_ids.append(int(guidance.guidance_id))
            serialized = [reason.model_dump(mode="json") for reason in refreshed]
            if serialized == [
                reason.model_dump(mode="json") for reason in guidance.stale_reasons
            ]:
                continue
            self._update_row(guidance.guidance_id, {"stale_reasons": serialized})
        return {
            "outcome": "dependencies reconciled",
            "details": {
                "checked": len(rows),
                "stale_guidance_ids": stale_guidance_ids,
                "stale_count": len(stale_guidance_ids),
            },
        }

    # -- Public reads -----------------------------------------------------------

    @functools.wraps(BaseGuidanceManager.search, updated=())
    def search(
        self,
        *,
        references: Optional[Dict[str, str]] = None,
        k: int = 10,
    ) -> List[Guidance]:
        caller_filter = None
        if _stored_only_without_reference(references):
            caller_filter = "is_builtin = 0"
        rows = rank_by_similarity(
            self._rows(self._scope(caller_filter)),
            references,
            limit=k,
            id_field="guidance_id",
        )
        # UNIFY_SEARCH_IDENTIFIERS: entries recorded under a request naming
        # an identifier of the query come first.
        shared: Dict[int, List[str]] = {}
        if _identifier_search_enabled():
            rows, shared = self._identifier_first(rows, references, k)
        return self._with_records(
            _with_linked_functions(
                [self._with_content_preview(Guidance(**row)) for row in rows],
            ),
            shared=shared,
        )

    def _identifier_first(
        self,
        rows: List[Dict[str, Any]],
        references: Optional[Dict[str, str]],
        k: int,
    ) -> tuple[List[Dict[str, Any]], Dict[int, List[str]]]:
        """``UNIFY_SEARCH_IDENTIFIERS``: *rows* with the entries whose recorded requests name a query identifier first."""
        from ..function_manager import task_origin

        query = " ".join(str(v or "") for v in (references or {}).values())
        recorded = self._origin_rows()
        matched = task_origin.identifier_matches(query, recorded)
        if not matched:
            return rows, {}
        by_id = {int(r["guidance_id"]): r for r in self._rows(self._scope(None))}
        first = [by_id[gid] for gid in matched if gid in by_id]
        rest = [r for r in rows if int(r["guidance_id"]) not in matched]
        return (first + rest)[: max(k, len(first))], matched

    def _with_records(
        self,
        entries: List[Guidance],
        *,
        shared: Optional[Dict[int, List[str]]] = None,
    ) -> List[Guidance]:
        """``UNIFY_ENTRY_RECORD``: *entries* each with its record (and kept revisions). Off: as they are."""
        from ..function_manager import entry_record, task_origin

        if not entries or not (_record_enabled() or shared):
            return entries
        recorded = {int(r["guidance_id"]): r for r in self._origin_rows()}
        library = [*recorded.values(), *self._function_origin_rows()]
        marker = task_origin.Marker(library)
        uses = (
            entry_record.uses_of(
                [(entry_record.GUIDANCE, str(e.guidance_id)) for e in entries],
            )
            if _record_enabled()
            else {}
        )
        out = []
        for entry in entries:
            gid = int(entry.guidance_id)
            row = recorded.get(gid) or {
                "guidance_id": gid,
                "metadata": {},
                "is_builtin": entry.is_builtin,
            }
            if entry.is_builtin:
                row = {**row, "is_builtin": True}
            parts = []
            if shared and shared.get(gid):
                parts.append(
                    "written while handling a request that also named "
                    + " and ".join(f"`{w}`" for w in shared[gid]),
                )
            if _record_enabled():
                parts.append(
                    entry_record.record_text(
                        marker,
                        entry_record.GUIDANCE,
                        row,
                        uses.get((entry_record.GUIDANCE, str(gid))),
                    ),
                )
            model = (
                GuidanceWithLinksAndRecord
                if isinstance(entry, GuidanceWithLinks)
                else GuidanceWithRecord
            )
            out.append(
                model(
                    **entry.model_dump(),
                    record="; ".join(parts),
                ),
            )
        return out

    @staticmethod
    def _function_origin_rows() -> List[Dict[str, Any]]:
        """The stored functions' origins (``metadata``), to weigh requests as the shortlist does."""
        rows = []
        for row in db.query("SELECT function_id, name, metadata FROM functions"):
            metadata = db.loads(row["metadata"]) if row["metadata"] else None
            if isinstance(metadata, dict):
                rows.append({"name": row["name"], "metadata": metadata})
        return rows

    @staticmethod
    def _evidence_rows_static() -> List[Dict[str, Any]]:
        """Every stored entry with a recorded origin, as ``{guidance_id, metadata}`` (to weigh requests)."""
        out = []
        for row in db.query(
            "SELECT guidance_id, origin FROM guidance WHERE origin IS NOT NULL",
        ):
            origin = db.loads(row["origin"])
            if isinstance(origin, dict):
                out.append({"guidance_id": row["guidance_id"], "metadata": origin})
        return out

    def _evidence_rows(self) -> List[Dict[str, Any]]:
        """The stored entries in scope (built-ins excluded), each with its origin as ``metadata`` ({} without one).

        Rows carry ``guidance_id``, ``title``, ``content``, ``function_ids``
        and ``metadata``. For the evidence list; no
        read the model makes returns the origin.
        """
        visible = self._rows(self._scope("is_builtin = 0"))
        if not visible:
            return []
        origins = {
            int(row["guidance_id"]): db.loads(row["origin"])
            for row in db.query(
                "SELECT guidance_id, origin FROM guidance WHERE origin IS NOT NULL",
            )
        }
        out = []
        for row in visible:
            origin = origins.get(int(row["guidance_id"]))
            out.append(
                {
                    "guidance_id": row["guidance_id"],
                    "title": row["title"],
                    "content": row["content"],
                    "function_ids": list(row.get("function_ids") or []),
                    "metadata": origin if isinstance(origin, dict) else {},
                },
            )
        return out

    def _shortlist_rows(self, text: str, k: int) -> List[Dict[str, Any]]:
        """``UNIFY_LIBRARY_SHORTLIST``: the *k* guidance entries in scope closest to *text*.

        Ranked as ``search`` ranks them, by the similarity of the title and
        the content to *text*. Rows carry ``guidance_id``, ``title``,
        ``content`` and ``_similarity``.
        """
        if not str(text or "").strip() or k <= 0:
            return []
        rows = rank_by_similarity(
            self._rows(self._scope(None)),
            {"title": text, "content": text},
            limit=k,
            id_field="guidance_id",
        )
        from ..function_manager import task_origin

        notes = task_origin.listing_notes_enabled()
        return [
            {
                "guidance_id": row.get("guidance_id"),
                "title": row.get("title"),
                "content": row.get("content"),
                "_similarity": float(row.get("_similarity") or 0.0),
                # UNIFY_LESSON_STATUS: a built-in entry is not a lesson.
                **({"is_builtin": bool(row.get("is_builtin"))} if notes else {}),
            }
            for row in rows
        ]

    def _origin_rows(self) -> List[Dict[str, Any]]:
        """``UNIFY_GUIDANCE_ORIGIN``: the stored entries in scope with a recorded origin.

        Rows carry ``guidance_id``, ``title``, ``content`` and the origin as
        ``metadata`` (the shape :class:`~unify.function_manager.task_origin.Marker`
        reads). Built-in entries never have one. For the gated shortlist
        only: no read the model makes returns the origin.
        """
        visible = {
            int(row["guidance_id"]): row
            for row in self._rows(self._scope("is_builtin = 0"))
        }
        if not visible:
            return []
        out = []
        for row in db.query(
            "SELECT guidance_id, origin FROM guidance"
            " WHERE origin IS NOT NULL ORDER BY guidance_id",
        ):
            entry = visible.get(int(row["guidance_id"]))
            origin = db.loads(row["origin"])
            if entry is None or not isinstance(origin, dict):
                continue
            out.append(
                {
                    "guidance_id": entry["guidance_id"],
                    "title": entry["title"],
                    "content": entry["content"],
                    "metadata": origin,
                },
            )
        return out

    def _set_use_when(self, guidance_id: Any, statement: str) -> bool:
        """``UNIFY_SHORTLIST_RELATED``: keep *statement* in the entry's recorded origin.

        Only for an entry in scope (not built-in) whose origin records the
        current request (``UNIFY_GUIDANCE_ORIGIN``: added or updated while
        handling it); ``False`` otherwise. No guidance read returns it.
        """
        from ..actor.related_shortlist import STATEMENT
        from ..function_manager import task_origin

        key = task_origin.current()
        try:
            gid = int(guidance_id)
        except (TypeError, ValueError):
            return False
        if key is None or not self._rows(
            self._scope(f"is_builtin = 0 AND guidance_id = {gid}"),
        ):
            return False
        row = db.query_one(
            "SELECT origin FROM guidance WHERE guidance_id = ?",
            (gid,),
        )
        origin = db.loads(row["origin"]) if row and row["origin"] else None
        if not isinstance(origin, dict) or key not in (
            origin.get(task_origin.FIELD) or []
        ):
            return False
        origin[STATEMENT] = statement
        db.execute(
            "UPDATE guidance SET origin = ? WHERE guidance_id = ?",
            (db.dumps(origin), gid),
        )
        return True

    @functools.wraps(BaseGuidanceManager.filter, updated=())
    def filter(
        self,
        *,
        filter: Optional[str] = None,
        offset: int = 0,
        limit: int = 100,
    ) -> List[Guidance]:
        try:
            rows = self._rows(
                self._scope(filter),
                limit=limit,
                offset=offset,
                readonly=True,
            )
        except sqlite3.Error as exc:
            return invalid_filter_error(exc, filter, db.GUIDANCE_COLUMNS).payload
        return self._with_records(
            _with_linked_functions(
                [self._with_content_preview(Guidance(**row)) for row in rows],
            ),
        )

    @functools.wraps(BaseGuidanceManager.get_guidance, updated=())
    def get_guidance(
        self,
        *,
        guidance_id: int,
    ) -> Guidance:
        rows = self._rows(
            self._scope(f"guidance_id = {int(guidance_id)}"),
            limit=1,
        )
        if not rows:
            raise ValueError(f"No guidance found with guidance_id {guidance_id}.")
        # UNIFY_ENTRY_RECORD: this session read the entry (a review's reads
        # are not counted).
        if _record_enabled() and not rows[0].get("is_builtin"):
            from ..function_manager import entry_record

            entry_record.record_use(
                entry_record.GUIDANCE,
                rows[0]["guidance_id"],
                entry_record.READ,
            )
        return self._with_records(_with_linked_functions([Guidance(**rows[0])]))[0]
