"""The example check is offered to the actor only where the world gives examples (``UNIFY_ACTOR_EXAMPLES_CHECK``).

Off (the default), neither ``tools.check_examples`` nor its opening line reaches the actor or a teammate; the memory
agents keep the tool. On, both come back exactly as before.
"""

from __future__ import annotations

import pytest

from unify.actor import tool_handlers as H
from unify.common import tool_registry as R
from unify.settings import SETTINGS


def _opening(agent: str) -> str:
    return R.preamble(agent, {"flags": H.actor_flags(agent, librarian=False)})


@pytest.mark.parametrize("agent", ["actor", "teammate"])
def test_off_by_default_the_actor_has_neither_the_tool_nor_its_line(agent, monkeypatch):
    monkeypatch.setattr(SETTINGS, "UNIFY_ACTOR_EXAMPLES_CHECK", False)
    flags = H.actor_flags(agent, librarian=False)
    assert R.EXAMPLES_FLAG not in flags
    assert "check_examples" not in R.allow_list(agent, flags)
    assert R.EXAMPLES_LINE not in _opening(agent)
    assert "check_examples" not in R.render_stub(R.allow_list(agent, flags), agent)


@pytest.mark.parametrize("agent", ["actor", "teammate"])
def test_on_the_actor_has_the_tool_and_its_one_line(agent, monkeypatch):
    monkeypatch.setattr(SETTINGS, "UNIFY_ACTOR_EXAMPLES_CHECK", True)
    flags = H.actor_flags(agent, librarian=False)
    assert R.EXAMPLES_FLAG in flags and "check_examples" in R.allow_list(agent, flags)
    assert _opening(agent).count(R.EXAMPLES_LINE) == 1


@pytest.mark.parametrize("agent", ["writer", "helper"])
def test_the_memory_agents_keep_the_tool_whatever_the_setting(agent, monkeypatch):
    monkeypatch.setattr(SETTINGS, "UNIFY_ACTOR_EXAMPLES_CHECK", False)
    assert "check_examples" in R.allow_list(agent, frozenset({"helper", "similar"}))


def test_off_the_opening_is_the_opening_without_the_line(monkeypatch):
    monkeypatch.setattr(SETTINGS, "UNIFY_ACTOR_EXAMPLES_CHECK", True)
    on = _opening("actor")
    monkeypatch.setattr(SETTINGS, "UNIFY_ACTOR_EXAMPLES_CHECK", False)
    off = _opening("actor")
    assert off == on.replace("\n\n" + R.EXAMPLES_LINE, "")
