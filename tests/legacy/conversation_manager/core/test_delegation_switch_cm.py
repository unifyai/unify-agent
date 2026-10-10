"""Symbolic: the legacy conversation manager builds its top-level actor from
the delegation switch (moved from tests/actor/code_act/test_delegation_switch.py).
"""

from __future__ import annotations


def test_the_conversation_manager_builds_from_the_switch():
    import inspect

    from unify.legacy.conversation_manager.domains import managers_utils

    source = inspect.getsource(managers_utils)
    assert "top_level_environments()" in source
    assert "ActorEnvironment(), *registered_environments()" not in source
