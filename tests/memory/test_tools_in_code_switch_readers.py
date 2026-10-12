"""Symbolic: every reader of ``UNIFY_TOOLS_IN_CODE`` and ``UNIFY_CELL_STATE`` follows the same settings, and the
writer's allow-list with tools in code is its tools one by one whatever the setting says."""

from __future__ import annotations

import pytest

from unify.settings import SETTINGS


@pytest.mark.parametrize("value", ["on", "off"])
def test_every_reader_of_the_tools_switch_agrees(monkeypatch, value):
    from unify.common import in_code_switch
    from unify.common._async_tool import loop
    from unify.memory import agent_profiles

    monkeypatch.setattr(SETTINGS, "UNIFY_TOOLS_IN_CODE", value)
    on = value == "on"
    assert in_code_switch.tools_in_code() is on
    assert agent_profiles.tools_in_code() is on
    assert loop._tools_in_code_on() is on
    from unify.actor import in_code_tools

    assert in_code_tools.enabled() is on


@pytest.mark.parametrize("value", ["persistent", "fresh"])
def test_the_cell_state_reader_follows_the_settings(monkeypatch, value):
    from unify.common import box_interpreter

    monkeypatch.setattr(SETTINGS, "UNIFY_CELL_STATE", value)
    assert box_interpreter.cell_state() == value
    assert box_interpreter.persistent() is (value == "persistent")


def test_the_writers_allow_list_is_its_tools_one_by_one_with_the_setting_on(monkeypatch):
    from unify.memory.writer_pass import writer_tools

    monkeypatch.setattr(SETTINGS, "UNIFY_TOOLS_IN_CODE", "on")
    names = lambda tools: [t["function"]["name"] for t in tools]  # noqa: E731
    assert names(writer_tools(library_mode=True)) == ["execute_code"]
    listed = names(writer_tools(library_mode=True, in_code=False))
    assert {"read", "grep", "check", "finish"} <= set(listed)
    monkeypatch.setattr(SETTINGS, "UNIFY_TOOLS_IN_CODE", "off")
    assert names(writer_tools(library_mode=True)) == listed
