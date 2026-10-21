"""What an actor may do, and the most a sub-actor it starts may do.

A ``primitives.actor.act`` call names the grants its sub-actor gets. The
caller is model-written code, so the call's arguments are a request, not an
authority: a sub-actor gets at most what its caller holds. Every
``CodeActActor.act`` run sets its own :class:`ActorGrants` in
:data:`CALLER_GRANTS`, and the runner bounds each request by them with
:func:`bound_child_grants` before the sub-actor is built. Nested spawns narrow
the same way, since a sub-actor's grants are themselves bounded. The grants
cross into ``run_coro_sync``'s worker thread with the call
(unify/common/asyncio_compat.py), so a synchronous façade cannot shed them.

A call made outside any actor run (harness code, a test, a stored function
executed directly) has no caller grants: its arguments come from
configuration, not from a model, and are used as given.
"""

from __future__ import annotations

import contextvars
from dataclasses import dataclass
from typing import Any, Optional

from unify.common.asyncio_compat import carry_across_threads
from unify.common.sql_filters import and_clauses, require_self_contained


class GrantEscalationError(PermissionError):
    """A sub-actor was asked for a grant its caller does not hold."""


@dataclass(frozen=True)
class ActorGrants:
    """The grants one actor run holds.

    ``discovery_scope`` and ``guidance_scope`` are the SQL clauses its
    libraries read with; ``None`` means unrestricted.
    """

    can_compose: bool
    can_store: bool
    can_spawn_sub_agents: bool
    discovery_scope: Optional[str] = None
    guidance_scope: Optional[str] = None

    @classmethod
    def of_actor(
        cls,
        actor: Any,
        *,
        can_compose: bool,
        can_store: bool,
    ) -> "ActorGrants":
        """The grants of *actor* for one run with the given effective flags."""
        return cls(
            can_compose=bool(can_compose),
            can_store=bool(can_store),
            can_spawn_sub_agents=_can_spawn(actor),
            discovery_scope=_scope_of(getattr(actor, "function_manager", None)),
            guidance_scope=_scope_of(getattr(actor, "guidance_manager", None)),
        )


def _scope_of(manager: Any) -> Optional[str]:
    scope = getattr(manager, "filter_scope", None)
    return scope if isinstance(scope, str) and scope else None


def _can_spawn(actor: Any) -> bool:
    """Whether any path in *actor* reaches ``primitives.actor.act``.

    Its sandbox holds the primitive when an environment exposes it, and a
    stored function gets it injected when the FunctionManager's primitive
    scope includes ``actor``.
    """
    from unify.actor.environments.actor import ActorEnvironment

    act_name = f"{ActorEnvironment.NAMESPACE}.{ActorEnvironment.MANAGER_ALIAS}.act"
    environments = getattr(actor, "environments", None) or {}
    env = environments.get(ActorEnvironment.NAMESPACE)
    if env is not None:
        try:
            if act_name in env.get_tools():
                return True
        except Exception:
            pass
    fm = getattr(actor, "function_manager", None)
    scope = getattr(fm, "primitive_scope", None)
    includes = getattr(scope, "includes", None)
    return bool(callable(includes) and includes(ActorEnvironment.MANAGER_ALIAS))


CALLER_GRANTS: contextvars.ContextVar[Optional[ActorGrants]] = carry_across_threads(
    contextvars.ContextVar("unify_actor_caller_grants", default=None),
)


def caller_grants() -> Optional[ActorGrants]:
    """The grants of the actor run this call is made from; ``None`` outside one."""
    grants = CALLER_GRANTS.get()
    return grants if isinstance(grants, ActorGrants) else None


def _require_bool(name: str, value: Any) -> bool:
    if not isinstance(value, bool):
        raise TypeError(
            f"primitives.actor.act: {name} must be True or False, not {value!r}.",
        )
    return value


def _require_scope(name: str, value: Any) -> Optional[str]:
    if value is not None and not isinstance(value, str):
        raise TypeError(
            f"primitives.actor.act: {name} must be a SQL WHERE clause string "
            f"or None, not {value!r}.",
        )
    return require_self_contained(value)


def bound_child_grants(
    parent: Optional[ActorGrants],
    *,
    can_compose: Any,
    can_store: Any,
    can_spawn_sub_agents: Any,
    discovery_scope: Any,
    guidance_scope: Any,
) -> ActorGrants:
    """The grants a sub-actor gets: the request, bounded by *parent*.

    A request for ``can_store`` or ``can_spawn_sub_agents`` the caller does
    not hold is refused with :class:`GrantEscalationError`; both default to
    False, so asking for one is always explicit. ``can_compose`` defaults to
    True, so a request cannot say whether it asked for it: a caller without
    it gets a sub-actor without it. The scopes are joined with the caller's
    by ``AND``, so the sub-actor sees at most what its caller sees. A grant
    that is not a bool, or a scope that is not a string, is refused.
    """
    can_compose = _require_bool("can_compose", can_compose)
    can_store = _require_bool("can_store", can_store)
    can_spawn_sub_agents = _require_bool("can_spawn_sub_agents", can_spawn_sub_agents)
    discovery_scope = _require_scope("discovery_scope", discovery_scope)
    guidance_scope = _require_scope("guidance_scope", guidance_scope)

    if parent is None:
        return ActorGrants(
            can_compose=can_compose,
            can_store=can_store,
            can_spawn_sub_agents=can_spawn_sub_agents,
            discovery_scope=discovery_scope,
            guidance_scope=guidance_scope,
        )

    refused = [
        name
        for name, asked, held in (
            ("can_store", can_store, parent.can_store),
            ("can_spawn_sub_agents", can_spawn_sub_agents, parent.can_spawn_sub_agents),
        )
        if asked and not held
    ]
    if refused:
        raise GrantEscalationError(
            "primitives.actor.act refused "
            + ", ".join(f"{name}=True" for name in refused)
            + ": this actor does not hold "
            + (" or ".join(refused))
            + ", so it cannot grant it to a sub-actor. A sub-actor gets at most "
            "its caller's grants; call it again without "
            + (" or ".join(refused))
            + ".",
        )

    return ActorGrants(
        can_compose=can_compose and parent.can_compose,
        can_store=can_store,
        can_spawn_sub_agents=can_spawn_sub_agents,
        discovery_scope=and_clauses(parent.discovery_scope, discovery_scope),
        guidance_scope=and_clauses(parent.guidance_scope, guidance_scope),
    )


__all__ = [
    "CALLER_GRANTS",
    "ActorGrants",
    "GrantEscalationError",
    "bound_child_grants",
    "caller_grants",
]
