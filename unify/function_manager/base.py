from __future__ import annotations

from abc import abstractmethod
from typing import Any, Dict, List, Literal, Optional, Union

from ..common.global_docstrings import CLEAR_METHOD_DOCSTRING
from ..common.state_managers import BaseStateManager

# State modes for function execution
StateMode = Literal["stateful", "read_only", "stateless"]


class BaseFunctionManager(BaseStateManager):
    """
    Public contract for a function catalogue that stores and retrieves
    user‑supplied functions and their metadata.

    Overview
    --------
    Implementations may talk to a real database (e.g. Unify logs), an
    in‑memory mock, or a purely simulated LLM – but they all expose the
    same public methods documented below.

    Data Model
    ----------
    All function records conform to the Pydantic model
    ``unify.function_manager.types.function.Function`` (referred to as
    "Function" in the method docs). Implementations may return either
    instances of this model or JSON‑serialisable dictionaries whose keys
    and value types match the model schema. Fields that are not
    applicable to a particular operation (e.g. ``implementation`` when
    not requested) may be omitted or set to suitable defaults by the
    implementation, but the schema serves as the single source of truth
    for field names and types.
    """

    _as_caller_description: str = "the FunctionManager, managing user-defined functions"

    # ------------------------------------------------------------------ #
    # Public interface                                                   #
    # ------------------------------------------------------------------ #
    @abstractmethod
    def add_functions(
        self,
        *,
        implementations: Union[str, List[str]],
        preconditions: Optional[Dict[str, Dict]] = None,
        overwrite: bool = False,
        raise_on_error: bool = True,
        dependencies: Optional[List[str]] = None,
    ) -> Dict[str, str]:
        """
        Validate, compile and persist one or more function implementations.

        Parameters
        ----------
        implementations : str | list[str]
            Python function source strings: exactly one top-level ``def`` /
            ``async def`` per string, starting at column 0; signatures and
            docstrings are extracted automatically.
        preconditions : dict[str, dict] | None
            Mapping of function name → precondition payload, stored as the
            record's ``precondition`` field.
        overwrite : bool, default ``False``
            When ``True``, an existing name is updated in place with a stable
            ``function_id`` so existing references (task entrypoints,
            guidance links) keep resolving — always prefer this over
            delete-and-re-add. When ``False``, existing names are skipped and
            reported as duplicates.
        raise_on_error : bool, default ``True``
            Raise ``ValueError`` naming the failed functions and errors,
            instead of reporting them in the result. Failures for one
            function never block the rest of the batch.
        dependencies : list[str] | None
            PEP 508 requirement strings for the third-party packages the
            functions import, e.g. ``["pandas>=2.0", "tabulate"]`` — the
            same specifiers that installed them. **Required** when a
            function imports anything beyond the standard library and the
            execution environment; otherwise ``ValueError`` is raised.
            Detection covers every import form — ``import``,
            ``from … import`` and dynamic imports with a literal name
            (``importlib.import_module("pkg")``, ``__import__("pkg")``) — so
            the recorded dependency, not the import syntax, is what makes
            the package available: it is installed wherever the function
            runs, before it runs. Recorded on every function in the batch.

        Returns
        -------
        dict[str, str]
            Function name → status, e.g. ``{"my_func": "added"}`` or
            ``{"my_func": "error: <message>"}``.

        Anti-patterns
        -------------
        - Silent early returns or empty/"ok" results with no ``PHASE`` /
          ``SKIP`` / ``SOFT_FAIL`` trail — soft failures need stdlib
          ``logging`` calls with those markers (not user-facing
          notifications), and the trail must survive distillation from a
          live trajectory.
        - Nesting ``asyncio.run(...)`` inside sync helpers / entrypoints —
          offline Jobs and actor sandboxes already own a loop. Prefer
          ``async def`` + ``await``, or the injected ``run_coro_sync`` helper.
        """

    @abstractmethod
    def list_functions(
        self,
        *,
        include_implementations: bool = False,
        _return_callable: bool = False,
        _namespace: Optional[Dict[str, Any]] = None,
        _also_return_metadata: bool = False,
    ) -> Dict[str, Dict[str, Any]]:
        """
        Return a mapping of function name to function metadata.

        Each record conforms to the ``Function`` schema and includes a
        ``guidance_ids`` field — a list of identifiers for related guidance
        entries that describe compositional procedures using these functions.

        Parameters
        ----------
        include_implementations : bool, default ``False``
            When ``True``, values include the full source code in the
            ``implementation`` field. When ``False``, implementations may be
            omitted to reduce payload size.
        _return_callable : bool, default ``False``
            When ``True``, return Python callables instead of metadata dicts.
            Implementations SHOULD inject the resulting callables (and any of their
            transitive dependencies) into the provided ``_namespace``.
        _namespace : dict[str, Any] | None, default ``None``
            Target namespace dict for dependency injection when
            ``_return_callable=True``. Required when ``_return_callable=True``.
        _also_return_metadata : bool, default ``False``
            When ``True`` (and only valid with ``_return_callable=True``), return a
            dict containing both callables and metadata:
            ``{"callables": <...>, "metadata": <...>}``.

        Returns
        -------
        dict[str, Function] | dict[str, Callable[..., Any]] | dict[str, Any]
            - When ``_return_callable=False``: mapping of function name → record
              conforming to the ``Function`` schema (as dicts or Function objects).
              When ``include_implementations=False``, the ``implementation`` field
              may be omitted.
            - When ``_return_callable=True``: mapping of function name → callable.
            - When ``_also_return_metadata=True``: a dict with keys ``callables`` and
              ``metadata`` containing the two corresponding mappings.

        Raises
        ------
        ValueError
            If ``_return_callable=True`` but ``_namespace`` is ``None``.
        ValueError
            If ``_also_return_metadata=True`` but ``_return_callable`` is ``False``.
        """

    @abstractmethod
    def get_precondition(self, *, function_name: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve the stored precondition for a given function.

        Signature
        ---------
        get_precondition(*, function_name: str) -> dict[str, Any] | None

        Parameters
        ----------
        function_name : str
            The canonical function ``name`` (as stored in the corresponding
            ``Function`` record).

        Returns
        -------
        dict[str, Any] | None
            The ``Function.precondition`` payload if present, otherwise ``None``.
        """

    @abstractmethod
    def delete_function(
        self,
        *,
        function_id: int,
        delete_dependents: bool = True,
    ) -> Dict[str, str]:
        """
        Delete a function by its unique identifier.

        Signature
        ---------
        delete_function(
            *,
            function_id: int,
            delete_dependents: bool = True,
        ) -> dict[str, str]

        Parameters
        ----------
        function_id : int
            Identifier of the function to delete (``Function.function_id``).
        delete_dependents : bool, default ``True``
            When ``True``, also remove every function that directly or
            transitively calls the target (recursive cascade).

        Returns
        -------
        dict[str, str]
            Status mapping, typically ``{<function_name>: "deleted"}``.
        """

    @abstractmethod
    def reconcile_dependencies(
        self,
        *,
        function_ids: Optional[List[int]] = None,
    ) -> Dict[str, Any]:
        """Refresh structured link debt for compositional dependencies.

        Audits each selected compositional function's declared ``depends_on``
        names against the current compositional and primitive catalogues.
        Existing ``stale_reasons`` for dependencies that resolve again are
        removed; missing dependencies are recorded without changing
        ``depends_on``.

        Parameters
        ----------
        function_ids : list[int] | None
            Optional subset to audit; when omitted, checks all compositional
            functions.

        Returns
        -------
        dict[str, Any]
            Outcome with checked and stale function identifiers.
        """

    @abstractmethod
    def filter_functions(
        self,
        *,
        filter: Optional[str] = None,
        offset: int = 0,
        limit: int = 100,
        include_implementations: bool = True,
        _return_callable: bool = False,
        _namespace: Optional[Dict[str, Any]] = None,
        _also_return_metadata: bool = False,
    ) -> List[Dict[str, Any]]:
        """
        Filter stored function metadata with a SQL ``WHERE`` clause.

        Each result conforms to the ``Function`` schema and includes a
        ``guidance_ids`` field — a list of identifiers for related guidance
        entries that describe compositional procedures using these functions.

        Parameters
        ----------
        filter : str | None, default ``None``
            A SQLite ``WHERE`` clause (without the ``WHERE`` keyword) over the
            function table's columns: ``function_id``, ``name``, ``argspec``,
            ``docstring``, ``implementation``, ``depends_on``,
            ``stale_reasons``, ``precondition``, ``metadata``,
            ``dependencies``, ``created_at``, ``usage_calls``,
            ``usage_last_called_at``, ``usage_search_hits``, ``is_primitive``
            (0 or 1), ``primitive_class``, ``primitive_method``. List and dict
            columns hold JSON text: test membership with
            ``EXISTS (SELECT 1 FROM json_each(depends_on) WHERE value = 'helper')``.
            When ``None``, returns all rows subject to pagination. The clause
            is executed read-only; anything other than a read is rejected.
        offset : int, default ``0``
            Zero‑based index of the first result to return.
        limit : int, default ``100``
            Maximum number of results to return. Must be <= 1000.
        include_implementations : bool, default ``True``
            When ``True``, results include the full source code in the
            ``implementation`` field. When ``False``, implementations are
            omitted to reduce payload size.
        _return_callable : bool, default ``False``
            When ``True``, return Python callables instead of metadata dicts.
            Implementations SHOULD inject the resulting callables (and any of their
            transitive dependencies) into the provided ``_namespace``.
        _namespace : dict[str, Any] | None, default ``None``
            Target namespace dict for dependency injection when
            ``_return_callable=True``. Required when ``_return_callable=True``.
        _also_return_metadata : bool, default ``False``
            When ``True`` (and only valid with ``_return_callable=True``), return a
            dict containing both callables and metadata:
            ``{"callables": [...], "metadata": [...]}``.

        Returns
        -------
        list[Function] | list[Callable[..., Any]] | dict[str, Any]
            - When ``_return_callable=False``: list of records conforming to the
              ``Function`` schema (as dicts or Function objects). When
              ``include_implementations=False``, the ``implementation`` field
              is omitted.
            - When ``_return_callable=True``: list of callables corresponding to the
              returned records.
            - When ``_also_return_metadata=True``: a dict with keys ``callables`` and
              ``metadata`` containing the two corresponding lists.

        Raises
        ------
        ValueError
            If ``_return_callable=True`` but ``_namespace`` is ``None``.
        ValueError
            If ``_also_return_metadata=True`` but ``_return_callable`` is ``False``.

        Examples
        --------
        >>> mgr.filter_functions(filter="docstring LIKE '%price%' AND is_primitive = 0")
        >>> mgr.filter_functions(filter="name LIKE 'get_%'")
        """

    @abstractmethod
    def search_functions(
        self,
        *,
        query: str = "",
        n: int = 5,
        include_implementations: bool = True,
        include_dormant: bool = False,
        _return_callable: bool = False,
        _namespace: Optional[Dict[str, Any]] = None,
        _also_return_metadata: bool = False,
    ) -> List[Dict[str, Any]]:
        """
        Search for functions by semantic similarity to a natural‑language query.

        A function is compared by the meaning of its name, signature and
        docstring, so describe what you need done in your own words: a
        paraphrase or synonym finds a function whose docstring words it
        differently. Results are ranked like memory, not just like an
        index: semantic similarity dominates, and a function's *standing* —
        how often and how recently it has actually been used, judged against
        its own usage rhythm — acts as the tiebreaker. Functions whose
        standing has fully lapsed drop out of results entirely (they still
        exist and still run; ``filter_functions``/``list_functions`` always
        see the whole store, and ``include_dormant=True`` brings them back
        here). Freshly stored functions surface normally: creation counts as
        a first use.

        Every result carries the ranking components in the open:
        ``_similarity`` (semantic match, 0–1), ``_standing`` (usage-based
        memory strength, 0–1 — recency against the function's own rhythm ×
        log-saturating call count), and ``_retrieval_score`` (the combined
        rank, ``similarity × (floor + (1−floor) × standing)``), beside the
        raw trace (``usage_calls``, ``usage_last_called_at``). Read them:
        a highly similar result with near-zero standing is a plausible but
        long-unexercised skill, while a moderate match with high standing
        is a battle-tested recent workhorse.

        Each result conforms to the ``Function`` schema and includes a
        ``guidance_ids`` field — a list of identifiers for related guidance
        entries that describe compositional procedures using these functions.

        Parameters
        ----------
        query : str, default ``""``
            Natural‑language text describing the desired function(s). An empty
            query is allowed (soft models sometimes omit it during discovery)
            and returns a broad sample of the catalogue rather than failing.
        n : int, default ``5``
            Number of similar results to return.
        include_implementations : bool, default ``True``
            When ``True``, results include the full source code in the
            ``implementation`` field. When ``False``, implementations are
            omitted to reduce payload size.
        include_dormant : bool, default ``False``
            When ``True``, functions whose usage standing has fully lapsed
            are ranked and returned like any other instead of being dropped
            from the results. Use when hunting for a skill you believe was
            stored long ago but has not been exercised recently.
        _return_callable : bool, default ``False``
            When ``True``, return Python callables instead of metadata dicts.
            Implementations SHOULD inject the resulting callables (and any of their
            transitive dependencies) into the provided ``_namespace``.
        _namespace : dict[str, Any] | None, default ``None``
            Target namespace dict for dependency injection when
            ``_return_callable=True``. Required when ``_return_callable=True``.
        _also_return_metadata : bool, default ``False``
            When ``True`` (and only valid with ``_return_callable=True``), return a
            dict containing both callables and metadata:
            ``{"callables": [...], "metadata": [...]}``.

        Returns
        -------
        list[dict[str, Any]] | list[Callable[..., Any]] | dict[str, Any]
            - When ``_return_callable=False``: up to ``n`` results ordered by similarity.
              Each element SHOULD include the fields of the ``Function`` model.
              When ``include_implementations=False``, the ``implementation`` field
              is omitted.
            - When ``_return_callable=True``: list of callables corresponding to the
              returned records.
            - When ``_also_return_metadata=True``: a dict with keys ``callables`` and
              ``metadata`` containing the two corresponding lists.

        Raises
        ------
        ValueError
            If ``_return_callable=True`` but ``_namespace`` is ``None``.
        ValueError
            If ``_also_return_metadata=True`` but ``_return_callable`` is ``False``.
        """

    @abstractmethod
    async def execute_function(
        self,
        *,
        function_name: str,
        call_kwargs: Optional[Dict[str, Any]] = None,
        state_mode: Literal["stateful", "read_only", "stateless"] = "stateless",
        session_id: int = 0,
        extra_namespaces: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Execute a stored function by name with optional state mode overrides.

        The function's recorded ``dependencies`` are installed if any are
        missing, then the implementation runs in-process.

        Signature
        ---------
        execute_function(
            *,
            function_name: str,
            call_kwargs: dict[str, Any] | None = None,
            state_mode: Literal["stateful", "read_only", "stateless"] = "stateless",
            session_id: int = 0,
            extra_namespaces: dict[str, Any] | None = None,
        ) -> dict[str, Any]

        Parameters
        ----------
        function_name : str
            Name of the function to execute (must exist in the function table).
        call_kwargs : dict[str, Any] | None, default ``None``
            Keyword arguments to pass to the function.
        state_mode : Literal["stateful", "read_only", "stateless"], default ``"stateless"``
            Controls how global state is handled during execution:
            - ``"stateless"``: Executes with fresh globals/no inherited state.
              Every execution starts with a clean environment; the default, and
              the right choice for pure functions that should not depend on or
              affect session state.
            - ``"stateful"``: Uses a persistent globals dict per session.
              Variables and state from previous executions persist, enabling
              Jupyter-notebook-style incremental development.
            - ``"read_only"``: Reads the current state from the persistent session
              but executes in a fresh environment. Changes are not persisted.
              Useful for "what-if" exploration.
        session_id : int, default ``0``
            The session whose state ``"stateful"`` and ``"read_only"`` use.
            Multiple sessions allow independent stateful execution contexts,
            like concurrent "notebook panes" with isolated state.
        extra_namespaces : dict[str, Any] | None, default ``None``
            Named objects to inject into the function's execution globals.

        Returns
        -------
        dict[str, Any]
            Execution result with keys:
            - ``result``: The function's return value.
            - ``error``: Error message if execution failed, ``None`` otherwise.
            - ``stdout``: Captured stdout from the function.
            - ``stderr``: Captured stderr from the function.

        Raises
        ------
        ValueError
            If the function does not exist or has no implementation.
        RuntimeError
            If one of the function's dependencies cannot be installed.

        Examples
        --------
        >>> # Execute a function statefully
        >>> result = await fm.execute_function(
        ...     function_name="my_func",
        ...     call_kwargs={"x": 1},
        ...     state_mode="stateful",
        ... )

        >>> # Execute with extra namespaces (e.g. sub-agent environment)
        >>> result = await fm.execute_function(
        ...     function_name="my_func",
        ...     extra_namespaces={"primitives": prims, "sub_agents": agent_env},
        ... )
        """

    @abstractmethod
    def clear(self) -> None:
        raise NotImplementedError


# Attach centralised docstring
BaseFunctionManager.clear.__doc__ = CLEAR_METHOD_DOCSTRING
