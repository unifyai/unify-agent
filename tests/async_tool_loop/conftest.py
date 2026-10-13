"""
tests/async_tool_loop/conftest.py
======================================

Shared pytest fixtures for async tool loop tests.

A test of what the loop does with a model's reply runs against a scripted
model (``tests/scripted_model.py``): the provider transport is replaced
below unillm's request building, the replies are written in the test, and
nothing leaves the process. Such a test still takes ``llm_config``, so each
request is built for the production default model (``UNIFY_MODEL``); it
builds its client after installing the script. No test reaches a real provider.
"""

from __future__ import annotations

import asyncio
import contextlib
from typing import Any, Callable

import pytest
from openai.types.chat import ChatCompletion

from tests.scripted_model import (  # noqa: F401  (scripted_model is a fixture)
    Always,
    ScriptedModel,
    scripted,
    scripted_model,
)
from unify.settings import SETTINGS

_DEFAULT_MODEL = SETTINGS.UNIFY_MODEL

# Single production-default config — ``new_llm_client`` supplies
# reasoning_effort unless a test overrides it explicitly.
DEFAULT_LLM_CONFIG: dict[str, str] = {"model": _DEFAULT_MODEL}

LLM_CONFIGS = [
    pytest.param(
        DEFAULT_LLM_CONFIG,
        id=_DEFAULT_MODEL.replace("@", "-"),
    ),
]


@pytest.fixture(params=LLM_CONFIGS)
def llm_config(request) -> dict[str, str]:
    """Fixture providing the production-default LLM client configuration."""

    return request.param


#: The kinds a loop's own turns are classified as: ``actor`` when the request
#: offers tools (a loop with no tools of its own still offers
#: ``compress_context``), ``query_llm`` when it offers none.
LOOP_KINDS = ("actor", "query_llm")


def loop_script(*entries: Any, **kinds: Any) -> ScriptedModel:
    """A :class:`ScriptedModel` answering ``actor`` and ``query_llm`` requests
    from one ordered list, *entries*, and every other kind from *kinds*.

    An entry is what a :class:`ScriptedModel` script holds: a reply, an
    exception, or a callable taking the request's ``Call`` (sync or async)
    and returning either. A test ends with ``assert model.kinds() == [...]``,
    so a request classified as another kind fails the test instead of taking
    a reply meant for a loop turn.
    """
    clash = set(kinds) & set(LOOP_KINDS)
    if clash:
        raise ValueError(f"{sorted(clash)} are answered from the shared entries")
    shared = list(entries)

    async def take(call):
        if not shared:
            raise AssertionError(
                f"no scripted reply left for a {call.kind!r} request",
            )
        entry = shared.pop(0)
        if callable(entry) and not isinstance(entry, ChatCompletion):
            entry = entry(call)
            if asyncio.iscoroutine(entry):
                entry = await entry
        return entry

    model = ScriptedModel(**{k: Always(take) for k in LOOP_KINDS}, **kinds)
    # The shared entries not yet asked for, for a test to check.
    model.shared = shared
    return model


@pytest.fixture
def loop_model() -> Callable[..., ScriptedModel]:
    """``loop_model(*entries, **kinds)`` builds :func:`loop_script`'s model
    and installs it for the rest of the test; build clients after it."""
    stack = contextlib.ExitStack()

    def build(*entries: Any, **kinds: Any) -> ScriptedModel:
        return stack.enter_context(scripted(loop_script(*entries, **kinds)))

    with stack:
        yield build
