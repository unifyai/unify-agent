from __future__ import annotations

import functools
import re

from typing import Any

import pytest

from unify.actor.execution.capture import StreamLike
from unify.actor.execution.session import SessionExecutor
from unify.actor.execution.types import ExecutionResult
from unify.manager_registry import ManagerRegistry

_ADDR_RE = re.compile(r" at 0x[0-9a-fA-F]+")


def _normalize_execute_function_duration(result: Any) -> Any:
    if result is None:
        return result
    if isinstance(result, dict):
        result["duration_ms"] = 0
    elif hasattr(result, "duration_ms"):
        result.duration_ms = 0
    return result


@pytest.fixture(autouse=True)
def _reset_manager_registry() -> None:
    """Give every actor test fresh manager singletons."""
    ManagerRegistry.clear()


@pytest.fixture(autouse=True)
def stabilize_execute_function_duration(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep execution result duration_ms deterministic so LLM cache keys are stable."""
    original_execute = SessionExecutor.execute

    @functools.wraps(original_execute)
    async def _patched_execute(self, *args, **kwargs):
        result = await original_execute(self, *args, **kwargs)
        return _normalize_execute_function_duration(result)

    monkeypatch.setattr(SessionExecutor, "execute", _patched_execute, raising=True)


@pytest.fixture(autouse=True)
def _sanitize_sandbox_addresses(monkeypatch: pytest.MonkeyPatch) -> None:
    """Strip non-deterministic ``at 0x…`` addresses from sandbox output so LLM cache keys stay stable.

    Covers what the sandbox prints and the repr of a block's last expression,
    such as the coroutine a missing ``await`` leaves behind.
    """
    original_write = StreamLike.write

    @functools.wraps(original_write)
    def _sanitized_write(self, obj: str) -> int:
        return original_write(self, _ADDR_RE.sub(" at 0x...", obj))

    monkeypatch.setattr(StreamLike, "write", _sanitized_write)

    original_to_llm_content = ExecutionResult.to_llm_content

    @functools.wraps(original_to_llm_content)
    def _sanitized_to_llm_content(self) -> list[dict]:
        return [
            (
                {**block, "text": _ADDR_RE.sub(" at 0x...", block["text"])}
                if block.get("type") == "text"
                else block
            )
            for block in original_to_llm_content(self)
        ]

    monkeypatch.setattr(ExecutionResult, "to_llm_content", _sanitized_to_llm_content)
