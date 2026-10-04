"""Symbolic: a sub-actor gets at most its caller's grants.

``primitives.actor.act`` is called by model-written code, so its arguments
are a request: a caller without library writes or sub-actors cannot start a
child that has them, a caller without code cannot start a child with code,
and a child's library scopes are joined with its caller's. Outside an actor
run (harness code, tests) the arguments are used as given. The scripted
runs go through ``CodeActActor.act`` with unillm's transport replaced, so
nothing leaves the process.
"""

from __future__ import annotations

import asyncio
import contextlib
import json

import pytest

from tests import cache_discipline_helpers as h
from tests.helpers import _handle_project
from unify.actor import code_act_actor as caa
from unify.actor.environments.actor import (
    ActorEnvironment,
    _ActorRunner,
    _build_inner_actor,
)
from unify.actor.grants import (
    CALLER_GRANTS,
    ActorGrants,
    GrantEscalationError,
    bound_child_grants,
    caller_grants,
)
from unify.common.asyncio_compat import run_coro_sync
from unify.common.sql_filters import UnsafeClauseError
from unify.settings import SETTINGS

FULL = ActorGrants(can_compose=True, can_store=True, can_spawn_sub_agents=True)
NO_STORE = ActorGrants(can_compose=True, can_store=False, can_spawn_sub_agents=True)
LEAF = ActorGrants(can_compose=True, can_store=False, can_spawn_sub_agents=False)
NO_CODE = ActorGrants(can_compose=False, can_store=False, can_spawn_sub_agents=True)


def _request(**overrides):
    """The grant arguments of a ``primitives.actor.act`` call, its defaults first."""
    request = dict(
        can_compose=True,
        can_store=False,
        can_spawn_sub_agents=False,
        discovery_scope=None,
        guidance_scope=None,
    )
    request.update(overrides)
    return request


@contextlib.contextmanager
def _caller(grants: ActorGrants | None):
    token = CALLER_GRANTS.set(grants)
    try:
        yield
    finally:
        CALLER_GRANTS.reset(token)


def _build(**overrides):
    kwargs = dict(
        guidelines=None,
        prompt_guidance=None,
        guidance_scope=None,
        prompt_functions=None,
        discovery_scope=None,
        timeout=30,
        can_compose=True,
        can_store=False,
        can_spawn_sub_agents=False,
    )
    kwargs.update(overrides)
    child, _guidelines = _build_inner_actor(**kwargs)
    return child


def _grants_of(actor) -> ActorGrants:
    return ActorGrants.of_actor(
        actor,
        can_compose=actor.can_compose,
        can_store=actor.can_store,
    )


# ── the bound itself ─────────────────────────────────────────────────────


@pytest.mark.timeout(30)
def test_outside_an_actor_run_the_request_is_used_as_given():
    assert caller_grants() is None
    bounded = bound_child_grants(
        None,
        **_request(can_store=True, can_spawn_sub_agents=True, discovery_scope="a"),
    )
    assert bounded == ActorGrants(
        can_compose=True,
        can_store=True,
        can_spawn_sub_agents=True,
        discovery_scope="a",
    )


@pytest.mark.timeout(30)
@pytest.mark.parametrize(
    "parent",
    [FULL, NO_STORE, LEAF],
    ids=["full", "no-store", "leaf"],
)
def test_a_default_request_is_unchanged(parent):
    """Normal delegation: the defaults ask for nothing a caller lacks."""
    assert bound_child_grants(parent, **_request()) == ActorGrants(
        can_compose=True,
        can_store=False,
        can_spawn_sub_agents=False,
    )


@pytest.mark.timeout(30)
def test_a_caller_holding_every_grant_passes_them_on():
    bounded = bound_child_grants(
        FULL,
        **_request(can_store=True, can_spawn_sub_agents=True),
    )
    assert bounded == FULL


@pytest.mark.timeout(30)
@pytest.mark.parametrize(
    ("parent", "asked", "refused"),
    [
        (NO_STORE, {"can_store": True}, ["can_store"]),
        (LEAF, {"can_spawn_sub_agents": True}, ["can_spawn_sub_agents"]),
        (
            ActorGrants(can_compose=True, can_store=False, can_spawn_sub_agents=False),
            {"can_store": True, "can_spawn_sub_agents": True},
            ["can_store", "can_spawn_sub_agents"],
        ),
    ],
    ids=["store", "spawn", "both"],
)
def test_a_grant_the_caller_lacks_is_refused(parent, asked, refused):
    with pytest.raises(GrantEscalationError) as excinfo:
        bound_child_grants(parent, **_request(**asked))
    message = str(excinfo.value)
    for name in refused:
        assert f"{name}=True" in message
    assert "at most its caller's grants" in message


@pytest.mark.timeout(30)
def test_code_is_dropped_when_the_caller_has_none():
    """can_compose defaults to True, so it narrows rather than refuses."""
    assert bound_child_grants(NO_CODE, **_request()).can_compose is False
    assert bound_child_grants(NO_CODE, **_request(can_compose=True)).can_compose is (
        False
    )
    assert bound_child_grants(FULL, **_request(can_compose=False)).can_compose is (
        False
    )


@pytest.mark.timeout(30)
def test_scopes_are_joined_with_the_callers():
    parent = ActorGrants(
        can_compose=True,
        can_store=False,
        can_spawn_sub_agents=True,
        discovery_scope="name LIKE 'report_%'",
        guidance_scope="guidance_id < 100",
    )
    joined = bound_child_grants(
        parent,
        **_request(discovery_scope="docstring LIKE '%csv%'"),
    )
    assert joined.discovery_scope == (
        "(name LIKE 'report_%') AND (docstring LIKE '%csv%')"
    )
    assert joined.guidance_scope == "guidance_id < 100"
    # A child asking for no scope still gets its caller's.
    assert bound_child_grants(parent, **_request()).discovery_scope == (
        "name LIKE 'report_%'"
    )


@pytest.mark.timeout(30)
@pytest.mark.parametrize(
    "scope",
    ["1=1) OR (1=1", "1=1 --", "1=1; DELETE FROM functions", "name = 'x", "/* */ 1"],
)
def test_a_scope_that_escapes_its_parentheses_is_refused(scope):
    parent = ActorGrants(
        can_compose=True,
        can_store=False,
        can_spawn_sub_agents=True,
        discovery_scope="name LIKE 'report_%'",
    )
    with pytest.raises(UnsafeClauseError, match="self-contained"):
        bound_child_grants(parent, **_request(discovery_scope=scope))
    with pytest.raises(UnsafeClauseError, match="self-contained"):
        bound_child_grants(None, **_request(guidance_scope=scope))


@pytest.mark.timeout(30)
@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("can_store", "False"),
        ("can_spawn_sub_agents", 1),
        ("can_compose", None),
        ("discovery_scope", 3),
    ],
)
def test_a_grant_of_the_wrong_type_is_refused(name, value):
    """``bool("False")`` is True; a grant must be a real bool."""
    with pytest.raises(TypeError, match=name):
        bound_child_grants(FULL, **_request(**{name: value}))


@pytest.mark.asyncio
@pytest.mark.timeout(30)
async def test_an_unknown_grant_is_refused():
    with pytest.raises(TypeError, match="can_network"):
        await _ActorRunner().act(request="x", can_network=True)


# ── the child the runner builds ──────────────────────────────────────────


@pytest.mark.timeout(60)
@_handle_project
def test_escalation_refused_before_the_child_is_built(monkeypatch):
    built = []
    monkeypatch.setattr(
        "unify.actor.code_act_actor.CodeActActor",
        lambda *a, **kw: built.append(kw),
    )
    with _caller(NO_STORE):
        with pytest.raises(GrantEscalationError, match="can_store=True"):
            _build(can_store=True)
    with _caller(LEAF):
        with pytest.raises(GrantEscalationError, match="can_spawn_sub_agents=True"):
            _build(can_spawn_sub_agents=True)
    assert built == []


@pytest.mark.timeout(60)
@_handle_project
def test_normal_delegation_builds_the_same_child():
    """Under a caller holding every grant the child is what the request asks."""
    outside = _build(can_store=True, can_spawn_sub_agents=True, discovery_scope="a=1")
    with _caller(FULL):
        inside = _build(
            can_store=True,
            can_spawn_sub_agents=True,
            discovery_scope="a=1",
        )
    for child in (outside, inside):
        assert child.can_store is True
        assert child.can_compose is True
        assert child.function_manager.filter_scope == "a=1"
        assert isinstance(child.environments.get("primitives"), ActorEnvironment)
    assert _grants_of(inside) == _grants_of(outside)


@pytest.mark.timeout(60)
@_handle_project
def test_the_child_reads_its_libraries_within_the_callers_scope():
    parent = ActorGrants(
        can_compose=True,
        can_store=False,
        can_spawn_sub_agents=True,
        discovery_scope="name LIKE 'report_%'",
        guidance_scope="guidance_id < 100",
    )
    with _caller(parent):
        child = _build(discovery_scope="docstring LIKE '%csv%'")
    assert child.function_manager.filter_scope == (
        "(name LIKE 'report_%') AND (docstring LIKE '%csv%')"
    )
    assert child.guidance_manager.filter_scope == "guidance_id < 100"
    assert child.can_compose is True
    assert "primitives" not in child.environments


_FUNCTIONS = [
    "def report_total(xs):\n" '    """Sum a report column."""\n' "    return sum(xs)\n",
    "def wipe_store():\n" '    """Delete every stored row."""\n' "    return None\n",
]


@pytest.mark.timeout(60)
@_handle_project
def test_a_prompt_function_outside_the_callers_scope_is_not_handed_on():
    from unify.function_manager.function_manager import FunctionManager

    FunctionManager(include_primitives=False).add_functions(implementations=_FUNCTIONS)
    parent = ActorGrants(
        can_compose=True,
        can_store=False,
        can_spawn_sub_agents=False,
        discovery_scope="name LIKE 'report_%'",
    )
    with _caller(parent):
        allowed = _build(prompt_functions=["report_total"])
        with pytest.raises(ValueError, match="'wipe_store' did not match"):
            _build(prompt_functions=["wipe_store"])
    assert any(
        name.endswith("report_total")
        for env in allowed.environments.values()
        for name in env.get_tools()
    )
    # Outside an actor run the same name resolves.
    _build(prompt_functions=["wipe_store"])


@pytest.mark.timeout(60)
@_handle_project
def test_prompt_guidance_outside_the_callers_scope_is_refused():
    from unify.guidance_manager.guidance_manager import GuidanceManager

    gm = GuidanceManager()
    gm.add_guidance(title="Reports", content="Sum the report columns.")
    gm.add_guidance(title="Owner's runbook", content="Never shown to children.")
    inside = gm.filter(filter="title = 'Reports'")[0].guidance_id
    parent = ActorGrants(
        can_compose=True,
        can_store=False,
        can_spawn_sub_agents=False,
        guidance_scope=f"guidance_id = {inside}",
    )
    with _caller(parent):
        _build_inner_actor(
            guidelines=None,
            prompt_guidance=["Reports"],
            guidance_scope=None,
            prompt_functions=None,
            discovery_scope=None,
            timeout=30,
            can_compose=True,
            can_store=False,
            can_spawn_sub_agents=False,
        )
        with pytest.raises(GrantEscalationError, match="Owner's runbook"):
            _build(prompt_guidance=["Owner's runbook"])
        # A quote in a title cannot end the clause and widen the lookup.
        _child, injected = _build_inner_actor(
            guidelines=None,
            prompt_guidance=["x' OR title = 'Owner''s runbook"],
            guidance_scope=None,
            prompt_functions=None,
            discovery_scope=None,
            timeout=30,
            can_compose=True,
            can_store=False,
            can_spawn_sub_agents=False,
        )
        assert injected is None
    _child, text = _build_inner_actor(
        guidelines=None,
        prompt_guidance=["Owner's runbook"],
        guidance_scope=None,
        prompt_functions=None,
        discovery_scope=None,
        timeout=30,
        can_compose=True,
        can_store=False,
        can_spawn_sub_agents=False,
    )
    assert "Never shown to children." in text


@pytest.mark.timeout(60)
@_handle_project
def test_nested_spawns_narrow_at_every_level():
    """A grandchild is bounded by its parent, which is bounded by the root."""
    with _caller(NO_STORE):
        child = _build(can_spawn_sub_agents=True)
    assert _grants_of(child) == ActorGrants(
        can_compose=True,
        can_store=False,
        can_spawn_sub_agents=True,
    )
    with _caller(_grants_of(child)):
        with pytest.raises(GrantEscalationError, match="can_store=True"):
            _build(can_store=True, can_spawn_sub_agents=True)
        grandchild = _build(discovery_scope="name LIKE 'g%'")
    assert _grants_of(grandchild) == ActorGrants(
        can_compose=True,
        can_store=False,
        can_spawn_sub_agents=False,
        discovery_scope="name LIKE 'g%'",
    )
    with _caller(_grants_of(grandchild)):
        with pytest.raises(GrantEscalationError, match="can_spawn_sub_agents=True"):
            _build(can_spawn_sub_agents=True)
        great = _build(discovery_scope="name LIKE '%x'")
    assert great.function_manager.filter_scope == (
        "(name LIKE 'g%') AND (name LIKE '%x')"
    )


@pytest.mark.asyncio
@pytest.mark.timeout(30)
async def test_the_grants_follow_a_call_through_run_coro_sync():
    async def read():
        return caller_grants()

    with _caller(NO_STORE):
        assert run_coro_sync(read) == NO_STORE
    assert run_coro_sync(read) is None


# ── through CodeActActor.act ─────────────────────────────────────────────


def _spawn_then_answer(code: str):
    return [
        lambda: h.completion(calls=[("execute_code", {"code": code})]),
        *[lambda: h.completion(content="done")] * 8,
    ]


def _tool_results(requests: list[dict]) -> list[str]:
    return [
        json.dumps(m["content"], default=str)
        for m in requests[-1]["messages"]
        if m.get("role") == "tool"
    ]


async def _act(actor, replies) -> list[dict]:
    try:
        with h.scripted(replies) as provider:
            handle = await actor.act("Delegate the sum.", persist=False)
            result = await asyncio.wait_for(handle.result(), 60)
    finally:
        await actor.close()
    assert result == "done"
    return h.session_requests(provider.requests)


@pytest.fixture
def no_gate(monkeypatch):
    monkeypatch.setattr(SETTINGS, "UNIFY_DISCOVERY_GATE", False)
    monkeypatch.setattr(SETTINGS, "UNIFY_BUILTIN_GUIDANCE", False)


@pytest.fixture
def inner_actors(monkeypatch):
    """Record each sub-actor built; it answers at once without a model."""
    built: list[dict] = []

    class _Inner:
        def __init__(self, *args, **kwargs):
            built.append(kwargs)

        async def act(self, request, **kwargs):
            from unify.actor.simulated import _StaticAnswerHandle

            return _StaticAnswerHandle(f"sub-actor answered {request!r}")

        async def close(self):
            return None

    real = caa.CodeActActor

    def _factory(*args, **kwargs):
        if "prompt_caching" in kwargs and kwargs.get("timeout") is not None:
            return _Inner(*args, **kwargs)
        return real(*args, **kwargs)

    monkeypatch.setattr(caa, "CodeActActor", _factory)
    return built, real


@pytest.mark.asyncio
@pytest.mark.timeout(120)
async def test_an_actor_without_store_writes_cannot_start_a_child_with_them(
    no_gate,
    inner_actors,
):
    built, real = inner_actors
    parent = real(environments=[ActorEnvironment()], can_store=False)
    requests = await _act(
        parent,
        _spawn_then_answer(
            "await primitives.actor.act(request='sum', can_store=True, "
            "can_spawn_sub_agents=True)",
        ),
    )
    (result,) = _tool_results(requests)
    assert "GrantEscalationError" in result
    assert "can_store=True" in result
    assert built == []


@pytest.mark.asyncio
@pytest.mark.timeout(120)
async def test_an_actor_with_store_writes_starts_a_child_as_before(
    no_gate,
    inner_actors,
):
    built, real = inner_actors
    parent = real(environments=[ActorEnvironment()], can_store=True)
    requests = await _act(
        parent,
        _spawn_then_answer(
            "await primitives.actor.act(request='sum', can_store=True, "
            "can_spawn_sub_agents=True)",
        ),
    )
    (result,) = _tool_results(requests)
    assert "GrantEscalationError" not in result
    assert len(built) == 1
    assert built[0]["can_store"] is True
    assert built[0]["can_compose"] is True
    assert any(isinstance(env, ActorEnvironment) for env in built[0]["environments"])


@pytest.mark.timeout(60)
@_handle_project
def test_a_childs_guidance_scope_leaves_everyone_elses_alone():
    """The guidance manager is a singleton; a child's scoped one is its own."""
    from unify.guidance_manager.guidance_manager import GuidanceManager

    gm = GuidanceManager()
    gm.add_guidance(title="Reports", content="Sum the report columns.")
    before = (gm.filter_scope, gm.exclude_ids)
    child = _build(guidance_scope="guidance_id < 0", prompt_guidance=["Reports"])
    assert child.guidance_manager is not gm
    assert child.guidance_manager.filter_scope == "guidance_id < 0"
    assert (GuidanceManager().filter_scope, GuidanceManager().exclude_ids) == before
    assert [g.title for g in gm.filter(filter="title = 'Reports'")] == ["Reports"]
