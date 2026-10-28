"""An environment: one namespace bound in the actor's cells, and what the actor's prompt says about it."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List


class BaseEnvironment(ABC):
    """One namespace in the actor's cells (``primitives``, say): the object bound under it and its prompt context.

    The object lives in the harness; a cell reaches it through the sandboxed worker's proxy
    (unify/actor/execution/worker.py).
    """

    def __init__(self, *, instance: Any, namespace: str) -> None:
        self._instance = instance
        self._namespace = namespace

    @property
    def namespace(self) -> str:
        """The global name bound in the actor's cells (e.g. ``primitives``)."""
        return self._namespace

    def get_instance(self) -> Any:
        """The object bound under :attr:`namespace`."""
        return self._instance

    @abstractmethod
    def get_prompt_context(self) -> str:
        """Markdown rules and examples for using this environment, for the actor's system prompt."""


def by_namespace(environments: List[BaseEnvironment]) -> Dict[str, BaseEnvironment]:
    """The environments by namespace; two with one namespace are refused."""
    out: Dict[str, BaseEnvironment] = {}
    for env in environments:
        if env.namespace in out:
            raise ValueError(f"two environments bind the namespace {env.namespace!r}")
        out[env.namespace] = env
    return out
