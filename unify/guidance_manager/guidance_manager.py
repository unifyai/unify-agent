from __future__ import annotations

import functools
import logging
import sqlite3
from typing import Any, Dict, FrozenSet, List, Optional

from unify import db
from ..common.sql_filters import and_clauses, invalid_filter_error, not_in
from ..common.stale_reason import StaleReason, merge_stale_reasons
from ..common.text_search import rank_by_text
from ..common.tool_outcome import ToolErrorException, ToolOutcome
from .base import BaseGuidanceManager
from .builtins import ensure_seeded
from .types.guidance import Guidance

logger = logging.getLogger(__name__)

# Content cap for search/filter result payloads. Entries (notably imported
# builtin skills) can run to 100KB+; returning them wholesale from list-style
# reads floods the caller's context window. Reads above this cap return a
# preview and the full text is fetched per entry via ``get_guidance``.
GUIDANCE_PREVIEW_CHARS = 2000

_SELECT = f"SELECT {', '.join(db.GUIDANCE_COLUMNS)} FROM all_guidance"


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
        """Compose *caller_filter* with ``filter_scope`` and the id exclusions."""
        return and_clauses(
            caller_filter,
            self._filter_scope,
            not_in("guidance_id", self._exclude_ids),
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
        cursor = db.execute(
            "INSERT INTO guidance (title, content, function_ids, stale_reasons, created_at)"
            " VALUES (?, ?, ?, ?, ?)",
            (
                g.title,
                g.content,
                db.dumps(g.function_ids),
                db.dumps(
                    [reason.model_dump(mode="json") for reason in g.stale_reasons],
                ),
                db.now_iso(),
            ),
        )
        return {
            "outcome": "guidance created successfully",
            "details": {"guidance_id": int(cursor.lastrowid)},
        }

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
        self._update_row(guidance_id, updates)
        return {"outcome": "guidance updated", "details": {"guidance_id": guidance_id}}

    @staticmethod
    def _update_row(guidance_id: int, updates: Dict[str, Any]) -> None:
        assignments = ", ".join(f"{column} = ?" for column in updates)
        values = [
            db.dumps(value) if column in db.GUIDANCE_JSON_COLUMNS else value
            for column, value in updates.items()
        ]
        db.execute(
            f"UPDATE guidance SET {assignments} WHERE guidance_id = ?",
            [*values, int(guidance_id)],
        )

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
        if not any(str(v or "").strip() for v in (references or {}).values()):
            from unify.settings import SETTINGS

            if SETTINGS.UNIFY_REQUIRE_SEARCH_QUERY:
                # An empty search would return the newest entries; with the flag the model
                # must say what it is looking for first.
                return ToolErrorException(
                    {
                        "error_kind": "missing_query",
                        "message": (
                            "references is required: no guidance was searched. Give the words "
                            "a matching entry's title or content would contain, such as the "
                            "task name or id this work is for, and call search again."
                        ),
                        "details": {"references": references, "k": k},
                    },
                ).payload
        rows = rank_by_text(
            self._rows(self._scope()),
            references,
            limit=k,
            id_field="guidance_id",
            backfill=True,
        )
        return [self._with_content_preview(Guidance(**row)) for row in rows]

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
        return [self._with_content_preview(Guidance(**row)) for row in rows]

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
        return Guidance(**rows[0])
