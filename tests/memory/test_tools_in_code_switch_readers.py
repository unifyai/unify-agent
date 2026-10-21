"""Symbolic: every reader of ``UNIFY_TOOLS_IN_CODE`` and ``UNIFY_CELL_STATE`` follows the same settings, and the
writer's allow-list with tools in code is its tools one by one whatever the setting says."""

from __future__ import annotations

import pytest

from unify.settings import SETTINGS


@pytest.mark.parametrize("value", ["persistent", "fresh"])
def test_the_cell_state_reader_follows_the_settings(monkeypatch, value):
    from unify.common import box_interpreter

    monkeypatch.setattr(SETTINGS, "UNIFY_CELL_STATE", value)
    assert box_interpreter.cell_state() == value
    assert box_interpreter.persistent() is (value == "persistent")


