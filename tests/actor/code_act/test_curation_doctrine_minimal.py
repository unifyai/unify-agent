"""Symbolic: ``UNIFY_CURATION_DOCTRINE=minimal``: a storage rulebook without the office assistant.

Every storage review request carries about 3.9k tokens of rulebook written
for a colleague product: preserving user notifications in stored functions,
weekly recurring deliverables, specialist sub-agents, trials of candidate
models inside the review, PHASE/SKIP/SOFT_FAIL logging markers and a
distillation essay. ``minimal`` keeps the compose rules and what storage
mechanically needs (what can be stored and how it runs, dependencies, what
guidance is for) and drops the rest.
"""

from __future__ import annotations

import asyncio
import re

import pytest

from tests import cache_discipline_helpers as h
from unify.actor import code_act_actor as caa

_DROPPED = (
    "### Preserving user-facing communication points",
    "## Recurring Deliverables",
    "## Sub-Agent Delegation Patterns",
    "### Model choice is part of distillation",
    "### The distillation dial",
    "### Expressive logging in stored functions",
    "PHASE",
    "send_notification",
)


# The minimal rulebook's own text, before the notes that follow it.
_MINIMAL = caa._STORAGE_MINIMAL_WHAT + caa._STORAGE_MINIMAL_GUIDANCE


def _sections() -> str:
    return caa._storage_doctrine_sections()


def test_minimal_keeps_what_storage_needs():
    text = _sections()
    assert text.startswith(_MINIMAL)
    assert caa._STORAGE_COMPOSE_DOCTRINE in text
    flat = " ".join(text.split())
    for kept in (
        "`FunctionManager_add_functions`",
        "`dependencies`",
        "`query_llm(...)`",
        "`run_coro_sync(factory)`",
        "`function_ids`",
    ):
        assert kept in flat, kept


@pytest.mark.parametrize("dropped", _DROPPED)
def test_minimal_drops_the_office_assistant(dropped):
    assert dropped not in _sections()


def test_minimal_names_no_benchmark():
    words = set(re.findall(r"[a-z]+", _MINIMAL.lower()))
    for word in (
        "arc",
        "appworld",
        "scienceworld",
        "crafter",
        "grid",
        "demo",
        "example",
    ):
        assert word not in words


@pytest.mark.asyncio
@pytest.mark.timeout(180)
async def test_the_sent_review_carries_the_minimal_rulebook():
    from unify.actor.code_act_actor import CodeActActor

    actor = CodeActActor()
    try:
        with h.scripted(h.ACTOR_REPLIES) as provider:
            handle = await actor.act("List the files in the workspace.", persist=False)
            await asyncio.wait_for(handle.result(), 60)
            await asyncio.wait_for(handle._completion_event.wait(), 60)
    finally:
        await actor.close()
    # The review forks the session and names the core surface's calls, so it
    # is found by the rulebook's heading.
    reviews = [
        r for r in provider.requests if "## What Can Be Stored" in str(r["messages"])
    ]
    assert reviews
    assert "## Recurring Deliverables" not in str(reviews[0]["messages"])


# The minimal rulebook once said "Functions are `async def` and `await` their
# calls". AppWorld's environment methods are synchronous, so the storage
# review rewrote working calls as `await primitives.<app>.<api>(...)` and
# stored 11 of 11 environment-calling functions that raise TypeError when run
# from code (the compose rulebook stored 0 of 13). The rulebook now says to
# await only what is asynchronous, and the environment note says which of the
# registered methods are, from the callables themselves.


def _sync_call(**kwargs):
    return [kwargs]


async def _async_call(**kwargs):
    return [kwargs]


class _AwaitableObject:
    async def __call__(self, **kwargs):
        return [kwargs]


def _namespace(name: str, **calls):
    from unify.function_manager.primitives import (
        EnvironmentMethod,
        EnvironmentNamespace,
    )

    return EnvironmentNamespace(
        name=name,
        methods=tuple(
            EnvironmentMethod(name=method, call=call, effect="read")
            for method, call in calls.items()
        ),
    )


@pytest.fixture
def registered():
    from unify.function_manager.primitives import (
        EnvironmentSurface,
        register_environment,
    )
    from unify.function_manager.primitives.environment import (
        clear_environment_namespaces,
    )

    def register(*namespaces):
        clear_environment_namespaces()
        register_environment(
            EnvironmentSurface(namespaces=tuple(namespaces)),
            source="tests:kinds",
        )

    clear_environment_namespaces()
    yield register
    clear_environment_namespaces()


def test_minimal_awaits_only_what_is_asynchronous():
    flat = " ".join(_sections().split())
    assert "Functions are `async def` and `await` their calls" not in flat
    assert (
        "Await only what is asynchronous: `query_llm(...)`, the "
        "`primitives.actor` methods and stored functions defined with "
        "`async def`"
    ) in flat
    assert "awaiting that value raises `TypeError`, so call it without `await`" in flat


def test_minimal_says_a_synchronous_environment_is_synchronous(registered):
    registered(_namespace("music", show_library=_sync_call, play=_sync_call))
    text = " ".join(_sections().split())
    assert "`primitives.music`" in text
    assert (
        "Every method of these namespaces is synchronous: it returns its value "
        "directly, so call it without `await`."
    ) in text


def test_minimal_says_an_asynchronous_environment_is_asynchronous(registered):
    registered(_namespace("web", fetch=_async_call, post=_AwaitableObject()))
    text = " ".join(_sections().split())
    assert "Every method of these namespaces is asynchronous: `await` it." in text
    assert "synchronous: call" not in text


def test_minimal_names_the_kinds_of_a_mixed_environment(registered):
    registered(
        _namespace("music", show_library=_sync_call),
        _namespace("web", fetch=_async_call),
        _namespace("files", read=_sync_call, write=_sync_call, watch=_async_call),
    )
    text = " ".join(_sections().split())
    assert (
        "The methods of `primitives.music` are synchronous: call them without `await`."
        in text
    )
    assert "The methods of `primitives.web` are asynchronous: `await` them." in text
    assert (
        "In `primitives.files`, `watch` is asynchronous (`await` it) and the "
        "other methods are synchronous."
    ) in text


def test_is_async_method_reads_the_callable():
    import functools

    from unify.function_manager.primitives import EnvironmentMethod
    from unify.function_manager.primitives.environment import is_async_method

    def kind(call):
        return is_async_method(EnvironmentMethod(name="m", call=call, effect="read"))

    @functools.wraps(_async_call)
    def forwarding(**kwargs):
        return _async_call(**kwargs)

    assert kind(_async_call)
    assert kind(_AwaitableObject())
    assert kind(forwarding)
    assert kind(functools.partial(_async_call, x=1))
    assert not kind(_sync_call)
    assert not kind(lambda **kw: kw)
    assert not kind(functools.partial(_sync_call, x=1))
