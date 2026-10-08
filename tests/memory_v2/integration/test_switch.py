"""The UNIFY_MEMORY_V2 switches: off by default, strict values (integration Task 17)."""

import pytest

from unify.settings import ProductionSettings

_NAMES = (
    "UNIFY_MEMORY_V2",
    "UNIFY_MEMORY_V2_SOL_MODEL",
    "UNIFY_MEMORY_V2_SOL_BUDGET_USD",
    "UNIFY_MEMORY_V2_TRIGGER",
)


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for name in _NAMES:
        monkeypatch.delenv(name, raising=False)


@pytest.mark.parametrize(
    "raw,want",
    [("", ""), ("off", ""), ("OFF", ""), ("on", "on"), (" On ", "on")],
)
def test_values(raw, want, monkeypatch):
    monkeypatch.setenv("UNIFY_MEMORY_V2", raw)
    assert ProductionSettings().UNIFY_MEMORY_V2 == want


@pytest.mark.parametrize(
    "name,raw",
    [
        ("UNIFY_MEMORY_V2", "yes"),
        ("UNIFY_MEMORY_V2", "true"),
        ("UNIFY_MEMORY_V2_SOL_MODEL", "openai/gpt 6"),
        ("UNIFY_MEMORY_V2_SOL_BUDGET_USD", "-1"),
        ("UNIFY_MEMORY_V2_SOL_BUDGET_USD", "nan"),
        ("UNIFY_MEMORY_V2_SOL_BUDGET_USD", "inf"),
        ("UNIFY_MEMORY_V2_SOL_BUDGET_USD", "two"),
        ("UNIFY_MEMORY_V2_TRIGGER", "every"),
    ],
)
def test_refuses_other_values(name, raw, monkeypatch):
    monkeypatch.setenv(name, raw)
    with pytest.raises(Exception):
        ProductionSettings()


def test_defaults():
    s = ProductionSettings()
    assert s.UNIFY_MEMORY_V2 == ""
    assert s.UNIFY_MEMORY_V2_SOL_MODEL == "openai/gpt-6-sol"
    assert s.UNIFY_MEMORY_V2_SOL_BUDGET_USD == "2.50"
    assert s.UNIFY_MEMORY_V2_TRIGGER == "d6"


def test_companion_values(monkeypatch):
    monkeypatch.setenv("UNIFY_MEMORY_V2_SOL_MODEL", " openai/gpt-6-luna ")
    monkeypatch.setenv("UNIFY_MEMORY_V2_SOL_BUDGET_USD", "0.75")
    monkeypatch.setenv("UNIFY_MEMORY_V2_TRIGGER", "Batched")
    s = ProductionSettings()
    assert s.UNIFY_MEMORY_V2_SOL_MODEL == "openai/gpt-6-luna"
    assert s.UNIFY_MEMORY_V2_SOL_BUDGET_USD == "0.75"
    assert isinstance(s.UNIFY_MEMORY_V2_SOL_BUDGET_USD, str)
    assert s.UNIFY_MEMORY_V2_TRIGGER == "batched"


@pytest.mark.parametrize("raw", ["", "D6"])
def test_trigger_empty_is_d6(raw, monkeypatch):
    monkeypatch.setenv("UNIFY_MEMORY_V2_TRIGGER", raw)
    assert ProductionSettings().UNIFY_MEMORY_V2_TRIGGER == "d6"
