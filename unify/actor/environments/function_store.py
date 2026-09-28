"""FunctionStore environment for CodeActActor.

Exposes FunctionManager-stored functions for prompt injection and sandbox
execution, identified by name or ID rather than a live Python instance.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from unify.actor.environments.base import (
    BaseEnvironment,
    ToolMetadata,
    _ClarificationQueueInjector,
)

if TYPE_CHECKING:
    from unify.function_manager.function_manager import FunctionManager


class FunctionStoreEnvironment(BaseEnvironment):
    """Environment backed by FunctionManager-stored compositional functions.

    Promotes specific stored functions from "discoverable via search" to
    "prompt-injected and directly callable in the sandbox".  The tagged
    ``function_id`` values are automatically excluded from FunctionManager
    search/list/filter results by the CodeActActor's exclusion wiring, so
    the environment reads its own functions by exact name or ID, which
    discovery scoping does not touch.

    Parameters
    ----------
    function_manager : FunctionManager
        The FunctionManager instance to fetch function metadata and callables from.
    function_names : list[str] | None
        Names of stored functions to include (e.g., ``["alpha", "beta"]``).
        At least one of ``function_names`` or ``function_ids`` must be provided.
    function_ids : list[int] | None
        IDs of stored functions to include.
        At least one of ``function_names`` or ``function_ids`` must be provided.
    namespace : str
        Sandbox namespace under which functions are accessible
        (default ``"functions"``). The LLM calls ``await functions.alpha(...)``.
    clarification_up_q : asyncio.Queue | None
        Queue for sending clarification requests to the user.
    clarification_down_q : asyncio.Queue | None
        Queue for receiving clarification responses from the user.
    """

    def __init__(
        self,
        function_manager: "FunctionManager",
        *,
        function_names: Optional[List[str]] = None,
        function_ids: Optional[List[int]] = None,
        namespace: str = "functions",
        clarification_up_q: Optional[asyncio.Queue[str]] = None,
        clarification_down_q: Optional[asyncio.Queue[str]] = None,
    ):
        if not function_names and not function_ids:
            raise ValueError(
                "At least one of function_names or function_ids must be provided.",
            )

        self._function_manager = function_manager
        self._requested_names = list(function_names) if function_names else []
        self._requested_ids = list(function_ids) if function_ids else []

        # Fetch the records once at construction time, so the prompt documents
        # exactly what every sandbox runs.
        self._func_metadata: List[Dict[str, Any]] = self._resolve_metadata()

        # Placeholder instance — callables are resolved lazily in get_sandbox_instance.
        super().__init__(
            instance=None,
            namespace=namespace,
            clarification_up_q=clarification_up_q,
            clarification_down_q=clarification_down_q,
        )

    def _resolve_metadata(self) -> List[Dict[str, Any]]:
        """Fetch the requested stored functions by exact name or ID.

        Discovery reads (search, filter, list) cannot serve here: the actor
        hides every function this environment documents from them.
        """
        fm = self._function_manager
        found = [fm._get_function_data_by_name(name=n) for n in self._requested_names]
        found += [
            fm._get_log_by_function_id(function_id=i, raise_if_missing=False)
            for i in self._requested_ids
        ]
        by_id = {row["function_id"]: row for row in found if row is not None}
        return [by_id[function_id] for function_id in sorted(by_id)]

    @property
    def namespace(self) -> str:
        return self._namespace

    def get_instance(self) -> Any:
        """Return None — callables are resolved lazily in get_sandbox_instance."""
        return self._instance

    def get_sandbox_instance(self) -> Any:
        """Resolve stored functions to callables and return a namespace object.

        Each function becomes an attribute on the returned object, callable
        as ``await namespace.function_name(...)``.
        """
        callables = self._function_manager._inject_callables_for_functions(
            self._func_metadata,
            namespace={},
        )
        sandbox_ns = SimpleNamespace(**{fn.__name__: fn for fn in callables})

        # Optionally wrap for clarification queue injection.
        if self._clarification_up_q is not None:
            return _ClarificationQueueInjector(
                target=sandbox_ns,
                clarification_up_q=self._clarification_up_q,
                clarification_down_q=self._clarification_down_q,
            )

        return sandbox_ns

    def get_tools(self) -> Dict[str, ToolMetadata]:
        """Return tool metadata for each stored function.

        Each tool is tagged with its ``function_id`` and
        ``function_context="compositional"`` so the CodeActActor's exclusion
        wiring automatically masks these from FunctionManager search results.
        """
        tools: Dict[str, ToolMetadata] = {}
        for row in self._func_metadata:
            name = row.get("name")
            if not name:
                continue
            fq_name = f"{self.namespace}.{name}"
            tools[fq_name] = ToolMetadata(
                name=fq_name,
                is_impure=True,
                is_steerable=False,
                docstring=row.get("docstring"),
                signature=row.get("argspec"),
                function_id=row.get("function_id"),
                function_context="compositional",
            )
        return tools

    def get_prompt_context(self) -> str:
        """Generate prompt context from stored function metadata.

        Formats each function's signature, docstring, and LLM-meaningful
        metadata as markdown, using the FunctionManager's stored metadata
        as the source of truth.
        """
        if not self._func_metadata:
            return ""

        lines = [f"### `{self.namespace}` — Injected Functions\n"]
        lines.append(
            "These functions are prompt-injected, directly callable in the sandbox "
            f"via `await {self.namespace}.<name>(...)`, and excluded from "
            "FunctionManager search.\n",
        )

        for row in self._func_metadata:
            name = row.get("name", "unknown")
            argspec = row.get("argspec", "(...)")
            docstring = row.get("docstring", "")

            lines.append(
                f"\n**`{self.namespace}.{name}{argspec}`**"
                f" [function_id: {row['function_id']}]",
            )
            if docstring:
                for doc_line in docstring.splitlines():
                    lines.append(f"  {doc_line}")

            gids = row.get("guidance_ids")
            if gids:
                lines.append(f"  Related guidance: {gids}")
            deps = row.get("depends_on")
            if deps:
                lines.append(f"  Depends on: {', '.join(deps)}")
            precond = row.get("precondition")
            if precond:
                lines.append(f"  Precondition: {precond}")
            dependencies = row.get("dependencies")
            if dependencies:
                lines.append(f"  Dependencies: {', '.join(dependencies)}")

        return "\n".join(lines)
