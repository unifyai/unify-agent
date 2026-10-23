"""Fixtures of the system suite: a fresh memory home per test, the real harness end to end, every model scripted.

``system`` gives a :class:`tests.system.drive.System` for the test's ``mode`` (``off``, ``on`` or ``in_episode``,
from the test's own parametrisation): the lanes' loopback server (fork, writer, helper and the embedding route), the
run's settings as a launcher gives them, the lanes settled as a controller settles them, and at the end the kit's
drain (verified termination of the writer service). Nothing of the unit under test is replaced.
"""

from __future__ import annotations

import pytest

from tests.actor.code_act.core_world import core_world  # noqa: F401 (fixture)
from tests.actor.code_act.sandbox_world import world  # noqa: F401 (fixture)
# the writer's lane refuses while unillm records OTel spans (the test runner turns them on); autouse
from tests.memory.test_writer_route import _otel_off_unless_a_test_turns_it_on  # noqa: F401


@pytest.fixture
def system(core_world, monkeypatch, mode):  # noqa: F811
    from tests.system.drive import System

    s = System.open(core_world, monkeypatch, mode)
    try:
        yield s
    finally:
        s.close()
