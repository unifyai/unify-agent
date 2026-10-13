import pytest

from tests.helpers import _handle_project
from unify.manager_registry import ManagerRegistry, SingletonABCMeta


class _DummySingleton(metaclass=SingletonABCMeta):
    """Tiny throw-away class to verify the **generic** singleton contract."""

    def __init__(self):
        # Store *id(self)* so we can later assert a new instance is created
        self.identity = id(self)


@pytest.mark.asyncio
async def test_same_instance():
    """Multiple constructor calls must return the *identical* object."""

    first = _DummySingleton()
    second = _DummySingleton()

    # Both variables must point to *exactly* the same object
    assert first is second
    assert first.identity == second.identity

    # Registry should return that very instance as well
    assert ManagerRegistry.get_instance(_DummySingleton) is first


@pytest.mark.asyncio
async def test_clear_creates_fresh_instance():
    """`ManagerRegistry.clear` must drop the cached instance so that the
    next instantiation yields a *new* object.
    """

    original = _DummySingleton()

    # Purge the registry manually (the session-wide fixture only runs between
    # tests; we also check the behaviour *within* a single test).
    ManagerRegistry.clear()

    replacement = _DummySingleton()

    assert original is not replacement
    assert original.identity != replacement.identity


@_handle_project
def test_a_forced_new_guidance_manager_leaves_the_shared_one_alone():
    """``_force_new`` must get past the singleton metaclass.

    A sub-actor scopes the guidance manager it is given; when the "new"
    instance was the registered one, its scope applied to every reader.
    """
    from unify.guidance_manager.guidance_manager import GuidanceManager

    shared = ManagerRegistry.get_guidance_manager()
    fresh = ManagerRegistry.get_guidance_manager(_force_new=True)
    assert fresh is not shared
    fresh.filter_scope = "guidance_id < 0"
    fresh.exclude_ids = frozenset({1})
    assert shared.filter_scope is None
    assert not shared.exclude_ids
    assert GuidanceManager() is shared
    assert ManagerRegistry.get_guidance_manager() is shared
