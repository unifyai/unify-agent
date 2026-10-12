"""The two switches of the in-code tools, validated when the settings load."""

from __future__ import annotations

import pytest

from unify.settings import ProductionSettings


def test_tools_in_code_is_on_or_off_and_off_by_default():
    assert ProductionSettings.model_fields["UNIFY_TOOLS_IN_CODE"].default == "off"
    assert ProductionSettings().UNIFY_TOOLS_IN_CODE == "off"
    assert ProductionSettings(UNIFY_TOOLS_IN_CODE=" ON ").UNIFY_TOOLS_IN_CODE == "on"
    assert ProductionSettings(UNIFY_TOOLS_IN_CODE="").UNIFY_TOOLS_IN_CODE == "off"
    for bad in ("yes", "true", "1", "code"):
        with pytest.raises(ValueError, match="UNIFY_TOOLS_IN_CODE"):
            ProductionSettings(UNIFY_TOOLS_IN_CODE=bad)


def test_cell_state_is_persistent_or_fresh_and_fresh_by_default():
    assert ProductionSettings.model_fields["UNIFY_CELL_STATE"].default == "fresh"
    assert ProductionSettings().UNIFY_CELL_STATE == "fresh"
    assert ProductionSettings(UNIFY_CELL_STATE="Persistent").UNIFY_CELL_STATE == "persistent"
    for bad in ("on", "stateful", "keep"):
        with pytest.raises(ValueError, match="UNIFY_CELL_STATE"):
            ProductionSettings(UNIFY_CELL_STATE=bad)


def test_both_are_read_from_the_environment(monkeypatch):
    monkeypatch.setenv("UNIFY_TOOLS_IN_CODE", "on")
    monkeypatch.setenv("UNIFY_CELL_STATE", "persistent")
    s = ProductionSettings()
    assert (s.UNIFY_TOOLS_IN_CODE, s.UNIFY_CELL_STATE) == ("on", "persistent")
    monkeypatch.setenv("UNIFY_TOOLS_IN_CODE", "maybe")
    with pytest.raises(ValueError):
        ProductionSettings()
