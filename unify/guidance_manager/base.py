from __future__ import annotations

from abc import abstractmethod
from typing import Dict, List, Optional

from ..manager_registry import SingletonABCMeta
from ..common.global_docstrings import CLEAR_METHOD_DOCSTRING
from ..common.state_managers import BaseStateManager
from ..common.tool_outcome import ToolOutcome
from .types.guidance import Guidance


class BaseGuidanceManager(BaseStateManager, metaclass=SingletonABCMeta):
    """
    Public contract that every concrete guidance-manager must satisfy.

    Stores procedural how-to information: step-by-step instructions,
    standard operating procedures, software usage walkthroughs, and
    strategies for composing functions together.

    Exposes CRUD operations (search, filter, add_guidance,
    update_guidance, delete_guidance) as first-class JSON tool calls
    on the CodeActActor — both in the main doing loop and in the
    post-completion storage review loop.
    """

    _as_caller_description: str = (
        "the GuidanceManager, managing procedural instructions and operating procedures"
    )

    # ------------------------------------------------------------------ #
    # Public interface                                                   #
    # ------------------------------------------------------------------ #

    @abstractmethod
    def search(
        self,
        *,
        references: Optional[Dict[str, str]] = None,
        k: int = 10,
    ) -> List["Guidance"]:
        """Search for guidance entries by semantic similarity to reference content.

        Guidance entries contain procedural how-to information: step-by-step
        instructions, operating procedures, software walkthroughs, and
        strategies for composing functions together.

        Results blend your own stored guidance with a built-in, read-only
        library of platform-provided guidance (entries with
        ``is_builtin=True``). Built-in entries can be searched and read like
        any other guidance but never modified.

        Long entries are returned with a truncated content preview; fetch
        the complete text of a specific entry with ``get_guidance`` before
        following its procedure in detail.

        Parameters
        ----------
        references : Dict[str, str] | None, default None
            Mapping of field name (``title`` or ``content``) to reference
            text compared with that field by meaning; with several fields an
            entry scores the mean of their similarities. Describe the task
            in natural language: a paraphrase finds a procedure that words
            it differently.
        k : int, default 10
            Maximum number of results to return. Must be <= 1000.

        Returns
        -------
        List[Guidance]
            Up to *k* rows ranked by similarity, backfilled with the newest
            remaining entries when fewer than *k* can be ranked.
        """
        raise NotImplementedError

    @abstractmethod
    def filter(
        self,
        *,
        filter: Optional[str] = None,
        offset: int = 0,
        limit: int = 100,
    ) -> List["Guidance"]:
        """Filter guidance entries with a SQL ``WHERE`` clause.

        Guidance entries contain procedural how-to information: step-by-step
        instructions, operating procedures, software walkthroughs, and
        strategies for composing functions together.

        Results blend your own stored guidance with a built-in, read-only
        library of platform-provided guidance (entries with
        ``is_builtin=True``). Filter on ``is_builtin`` to target either
        population explicitly.

        Long entries are returned with a truncated content preview; fetch
        the complete text of a specific entry with ``get_guidance`` before
        following its procedure in detail.

        Parameters
        ----------
        filter : str | None, default None
            A SQLite ``WHERE`` clause (without the ``WHERE`` keyword) over the
            columns ``guidance_id``, ``title``, ``content``, ``function_ids``
            (JSON list; test membership with
            ``EXISTS (SELECT 1 FROM json_each(function_ids) WHERE value = 42)``),
            ``stale_reasons`` and ``is_builtin`` (0 or 1), e.g.
            ``"title LIKE '%deploy%' AND is_builtin = 0"``. When ``None``,
            returns all guidance records subject to pagination. The clause is
            executed read-only; anything other than a read is rejected.
        offset : int, default 0
            Zero-based index of the first result to include.
        limit : int, default 100
            Maximum number of records to return. Must be <= 1000.

        Returns
        -------
        List[Guidance]
            Matching guidance records as Guidance models.
        """
        raise NotImplementedError

    @abstractmethod
    def get_guidance(
        self,
        *,
        guidance_id: int,
    ) -> "Guidance":
        """Fetch one guidance entry by id with its complete, untruncated content.

        ``search`` and ``filter`` return truncated content previews for long
        entries; call this to read the full procedure before executing it
        step by step. Works for your own entries and for built-in
        platform guidance alike.

        Parameters
        ----------
        guidance_id : int
            Identifier of the entry to fetch.

        Returns
        -------
        Guidance
            The complete guidance entry.
        """
        raise NotImplementedError

    @abstractmethod
    def add_guidance(
        self,
        *,
        title: Optional[str] = None,
        content: Optional[str] = None,
        function_ids: Optional[List[int]] = None,
    ) -> "ToolOutcome":
        """Create a new guidance entry for procedural or operational how-to
        information: step-by-step instructions, standard operating procedures,
        software usage walkthroughs, composition strategies for combining
        functions, or any other actionable "how to do X" content.

        At least one of ``title`` or ``content`` must be provided.

        Parameters
        ----------
        title : str | None
            Short human-readable title for the guidance entry.
        content : str | None
            Longer freeform guidance text describing the procedure.
        function_ids : list[int] | None
            Optional ids of related functions to cross-reference.

        Returns
        -------
        ToolOutcome
            Outcome string and details containing the newly assigned
            ``guidance_id``.
        """
        raise NotImplementedError

    @abstractmethod
    def update_guidance(
        self,
        *,
        guidance_id: int,
        title: Optional[str] = None,
        content: Optional[str] = None,
        function_ids: Optional[List[int]] = None,
    ) -> "ToolOutcome":
        """Update fields of an existing guidance entry by id.

        Use this to revise procedural instructions, operating procedures,
        or compositional strategies that are already stored. Built-in
        platform guidance (``is_builtin=True``) is read-only and cannot be
        updated; create your own entry instead when a tailored variant is
        needed.

        Parameters
        ----------
        guidance_id : int
            Identifier of the row to update.
        title : str | None
            New title (omit to keep existing value).
        content : str | None
            New content (omit to keep existing value).
        function_ids : list[int] | None
            Replacement list of related function ids.

        Returns
        -------
        ToolOutcome
            Outcome string and details with the ``guidance_id``.
        """
        raise NotImplementedError

    @abstractmethod
    def delete_guidance(
        self,
        *,
        guidance_id: int,
    ) -> "ToolOutcome":
        """Delete a guidance entry by id.

        Built-in platform guidance (``is_builtin=True``) is read-only and
        cannot be deleted.

        Parameters
        ----------
        guidance_id : int
            Identifier of the row to delete.

        Returns
        -------
        ToolOutcome
            Outcome string and details with the removed ``guidance_id``.
        """
        raise NotImplementedError

    @abstractmethod
    def reconcile_dependencies(
        self,
        *,
        guidance_ids: Optional[List[int]] = None,
    ) -> "ToolOutcome":
        """Refresh structured link debt for related functions.

        Checks each selected entry's declared ``function_ids`` and any
        previously recorded missing-function identities. Missing links remain
        represented in ``stale_reasons`` without reconstructing
        ``function_ids`` from names.

        Parameters
        ----------
        guidance_ids : list[int] | None
            Optional subset to audit; when omitted, checks all stored guidance.

        Returns
        -------
        ToolOutcome
            Outcome with checked and stale guidance identifiers.
        """
        raise NotImplementedError

    @abstractmethod
    def clear(self) -> None:
        raise NotImplementedError


# Attach centralised docstring
BaseGuidanceManager.clear.__doc__ = CLEAR_METHOD_DOCSTRING
