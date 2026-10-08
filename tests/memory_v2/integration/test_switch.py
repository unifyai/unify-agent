"""The UNIFY_MEMORY_V2 switches: off by default, strict values (integration Task 17, online contract).

The contract: ``UNIFY_MEMORY_V2`` (``on`` / ``off`` / empty), ``UNIFY_MEMORY_V2_E`` (a positive int,
150000), ``UNIFY_MEMORY_V2_SOL_MODEL`` (``openai/gpt-6-sol``),
``UNIFY_MEMORY_V2_SOL_ALLOWANCE_USD_PER_TOKENS`` (a decimal string, ``0.00000073``, no exponent) and
``UNIFY_MEMORY_V2_SOL_RUN_GUARD_USD`` (a decimal string, or empty for no guard) and
``UNIFY_MEMORY_V2_DIALOGUE`` (``off`` / empty, or ``env``). Each value is loaded
through ``ProductionSettings`` from the environment, as a run sets it.
"""

import pytest

from unify.settings import ProductionSettings

_NAMES = (
    "UNIFY_MEMORY_V2",
    "UNIFY_MEMORY_V2_E",
    "UNIFY_MEMORY_V2_SOL_MODEL",
    "UNIFY_MEMORY_V2_SOL_ALLOWANCE_USD_PER_TOKENS",
    "UNIFY_MEMORY_V2_SOL_RUN_GUARD_USD",
    "UNIFY_MEMORY_V2_SOL_EFFORT_SCALE",
    "UNIFY_MEMORY_V2_SOL_MAX_CALLS",
    "UNIFY_MEMORY_V2_SURFACING",
    "UNIFY_MEMORY_V2_DOCSTRINGS",
    "UNIFY_MEMORY_V2_SOFT_BUDGET",
    "UNIFY_MEMORY_V2_SOL_USAGE",
    "UNIFY_MEMORY_V2_QA_FIXTURES",
    "UNIFY_MEMORY_V2_QA_MUTATION",
    "UNIFY_MEMORY_V2_QA_MUTATION_MIN_KILL",
    "UNIFY_MEMORY_V2_QA_DETERMINISM",
    "UNIFY_MEMORY_V2_QA_REPLAY",
    "UNIFY_MEMORY_V2_QA_FIXTURE_SIZE",
    "UNIFY_MEMORY_V2_DIALOGUE",
    # retired by the online contract; a stale one in the environment must change nothing
    "UNIFY_MEMORY_V2_TRIGGER",
    "UNIFY_MEMORY_V2_SOL_BUDGET_USD",
)


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for name in _NAMES:
        monkeypatch.delenv(name, raising=False)


def _load(monkeypatch, name, raw):
    monkeypatch.setenv(name, raw)
    return getattr(ProductionSettings(), name)


def test_defaults():
    s = ProductionSettings()
    assert s.UNIFY_MEMORY_V2 == ""
    assert s.UNIFY_MEMORY_V2_E == 150000 and type(s.UNIFY_MEMORY_V2_E) is int
    assert s.UNIFY_MEMORY_V2_SOL_MODEL == "openai/gpt-6-sol"
    assert s.UNIFY_MEMORY_V2_SOL_ALLOWANCE_USD_PER_TOKENS == "0.00000073"
    assert s.UNIFY_MEMORY_V2_SOL_RUN_GUARD_USD == ""
    # per-effort pass limits: one rule for every bed
    assert s.UNIFY_MEMORY_V2_SOL_EFFORT_SCALE == "low:1,medium:2,high:5"
    assert s.UNIFY_MEMORY_V2_SOL_MAX_CALLS == "low:40,medium:80,high:80"
    # v2.1 surfacing: each default is the v2 screen build's behaviour (test_v21_switch_defaults.py)
    assert s.UNIFY_MEMORY_V2_SURFACING == "index"
    assert s.UNIFY_MEMORY_V2_DOCSTRINGS == "off"
    assert s.UNIFY_MEMORY_V2_SOFT_BUDGET == "off"
    assert s.UNIFY_MEMORY_V2_SOL_USAGE == ""
    assert s.UNIFY_MEMORY_V2_DIALOGUE == ""


def test_the_retired_switches_are_gone(monkeypatch):
    fields = ProductionSettings.model_fields
    assert "UNIFY_MEMORY_V2_TRIGGER" not in fields
    assert "UNIFY_MEMORY_V2_SOL_BUDGET_USD" not in fields
    monkeypatch.setenv("UNIFY_MEMORY_V2_TRIGGER", "every")
    monkeypatch.setenv("UNIFY_MEMORY_V2_SOL_BUDGET_USD", "two")
    s = ProductionSettings()  # ignored, not refused: settings ignore unknown names
    assert not hasattr(s, "UNIFY_MEMORY_V2_TRIGGER")
    assert not hasattr(s, "UNIFY_MEMORY_V2_SOL_BUDGET_USD")


@pytest.mark.parametrize(
    "raw,want",
    [("", ""), ("off", ""), ("OFF", ""), (" Off ", ""), ("on", "on"), (" On ", "on")],
)
def test_switch_values(raw, want, monkeypatch):
    assert _load(monkeypatch, "UNIFY_MEMORY_V2", raw) == want


@pytest.mark.parametrize(
    "raw,want",
    [("", ""), ("off", ""), ("OFF", ""), (" Off ", ""), ("on", "on"), (" On ", "on")],
)
def test_sol_usage_values(raw, want, monkeypatch):
    assert _load(monkeypatch, "UNIFY_MEMORY_V2_SOL_USAGE", raw) == want


@pytest.mark.parametrize(
    "raw,want",
    [("", ""), ("off", ""), (" OFF ", ""), ("env", "env"), (" Env ", "env")],
)
def test_dialogue_values(raw, want, monkeypatch):
    assert _load(monkeypatch, "UNIFY_MEMORY_V2_DIALOGUE", raw) == want


@pytest.mark.parametrize(
    "raw,want",
    [("", 150000), ("150000", 150000), (" 1 ", 1), ("2000000", 2000000)],
)
def test_experience_budget_values(raw, want, monkeypatch):
    got = _load(monkeypatch, "UNIFY_MEMORY_V2_E", raw)
    assert got == want and type(got) is int


@pytest.mark.parametrize(
    "raw,want",
    [
        ("", "openai/gpt-6-sol"),
        (" openai/gpt-6-luna ", "openai/gpt-6-luna"),
        ("openai/gpt-6-sol@openrouter", "openai/gpt-6-sol@openrouter"),
    ],
)
def test_sol_model_values(raw, want, monkeypatch):
    assert _load(monkeypatch, "UNIFY_MEMORY_V2_SOL_MODEL", raw) == want


@pytest.mark.parametrize(
    "raw,want",
    [
        ("", "0.00000073"),
        ("0.00000073", "0.00000073"),
        (" 0.000002 ", "0.000002"),
        ("1", "1"),
        ("0.5", "0.5"),
    ],
)
def test_allowance_values(raw, want, monkeypatch):
    got = _load(monkeypatch, "UNIFY_MEMORY_V2_SOL_ALLOWANCE_USD_PER_TOKENS", raw)
    assert got == want and isinstance(got, str)


@pytest.mark.parametrize(
    "raw,want",
    [
        ("", ""),
        ("  ", ""),
        ("2.50", "2.50"),
        (" 10 ", "10"),
        ("0", "0"),
        ("0.75", "0.75"),
    ],
)
def test_run_guard_values(raw, want, monkeypatch):
    got = _load(monkeypatch, "UNIFY_MEMORY_V2_SOL_RUN_GUARD_USD", raw)
    assert got == want and isinstance(got, str)


@pytest.mark.parametrize(
    "name,raw,want",
    [
        ("UNIFY_MEMORY_V2_SOL_EFFORT_SCALE", "", "low:1,medium:2,high:5"),
        (
            "UNIFY_MEMORY_V2_SOL_EFFORT_SCALE",
            " High:5, low:1,MEDIUM:2 ",
            "low:1,medium:2,high:5",
        ),
        (
            "UNIFY_MEMORY_V2_SOL_EFFORT_SCALE",
            "low:0.5,medium:1.25,high:10",
            "low:0.5,medium:1.25,high:10",
        ),
        ("UNIFY_MEMORY_V2_SOL_MAX_CALLS", "", "low:40,medium:80,high:80"),
        (
            "UNIFY_MEMORY_V2_SOL_MAX_CALLS",
            "high:100,medium:80,low:040",
            "low:40,medium:80,high:100",
        ),
    ],
)
def test_per_effort_limit_values(name, raw, want, monkeypatch):
    got = _load(monkeypatch, name, raw)
    assert got == want and isinstance(got, str)


@pytest.mark.parametrize(
    "name,raw",
    [
        ("UNIFY_MEMORY_V2_SOL_EFFORT_SCALE", "low:1,medium:2"),
        ("UNIFY_MEMORY_V2_SOL_EFFORT_SCALE", "low:1,medium:2,high:5,xhigh:8"),
        ("UNIFY_MEMORY_V2_SOL_EFFORT_SCALE", "low:1,medium:2,high:5,high:5"),
        ("UNIFY_MEMORY_V2_SOL_EFFORT_SCALE", "low:0,medium:2,high:5"),
        ("UNIFY_MEMORY_V2_SOL_EFFORT_SCALE", "low:1,medium:2,high:5e0"),
        ("UNIFY_MEMORY_V2_SOL_EFFORT_SCALE", "low:1,medium:-2,high:5"),
        ("UNIFY_MEMORY_V2_SOL_EFFORT_SCALE", "actor:1,low:1,medium:2,high:5"),
        ("UNIFY_MEMORY_V2_SOL_EFFORT_SCALE", "5"),
        ("UNIFY_MEMORY_V2_SOL_MAX_CALLS", "low:40,medium:80"),
        ("UNIFY_MEMORY_V2_SOL_MAX_CALLS", "low:40,medium:80,high:0"),
        ("UNIFY_MEMORY_V2_SOL_MAX_CALLS", "low:40,medium:80,high:80.0"),
        ("UNIFY_MEMORY_V2_SOL_MAX_CALLS", "low:40,medium:80,high:+80"),
        ("UNIFY_MEMORY_V2_SOL_MAX_CALLS", "low=40,medium=80,high=80"),
        ("UNIFY_MEMORY_V2", "yes"),
        ("UNIFY_MEMORY_V2", "true"),
        ("UNIFY_MEMORY_V2", "1"),
        ("UNIFY_MEMORY_V2_E", "0"),
        ("UNIFY_MEMORY_V2_E", "-5"),
        ("UNIFY_MEMORY_V2_E", "1.5"),
        ("UNIFY_MEMORY_V2_E", "1e5"),
        ("UNIFY_MEMORY_V2_E", "150_000"),
        ("UNIFY_MEMORY_V2_E", "+7"),
        ("UNIFY_MEMORY_V2_E", "lots"),
        ("UNIFY_MEMORY_V2_SOL_MODEL", "openai/gpt 6"),
        ("UNIFY_MEMORY_V2_SOL_ALLOWANCE_USD_PER_TOKENS", "7.3e-7"),
        ("UNIFY_MEMORY_V2_SOL_ALLOWANCE_USD_PER_TOKENS", "7.3E-7"),
        ("UNIFY_MEMORY_V2_SOL_ALLOWANCE_USD_PER_TOKENS", "0"),
        ("UNIFY_MEMORY_V2_SOL_ALLOWANCE_USD_PER_TOKENS", "0.000"),
        ("UNIFY_MEMORY_V2_SOL_ALLOWANCE_USD_PER_TOKENS", "-0.1"),
        ("UNIFY_MEMORY_V2_SOL_ALLOWANCE_USD_PER_TOKENS", "+0.1"),
        ("UNIFY_MEMORY_V2_SOL_ALLOWANCE_USD_PER_TOKENS", ".5"),
        ("UNIFY_MEMORY_V2_SOL_ALLOWANCE_USD_PER_TOKENS", "nan"),
        ("UNIFY_MEMORY_V2_SOL_ALLOWANCE_USD_PER_TOKENS", "inf"),
        ("UNIFY_MEMORY_V2_SOL_ALLOWANCE_USD_PER_TOKENS", "1,5"),
        ("UNIFY_MEMORY_V2_SOL_RUN_GUARD_USD", "1e1"),
        ("UNIFY_MEMORY_V2_SOL_RUN_GUARD_USD", "-1"),
        ("UNIFY_MEMORY_V2_SOL_RUN_GUARD_USD", "nan"),
        ("UNIFY_MEMORY_V2_SOL_RUN_GUARD_USD", "Infinity"),
        ("UNIFY_MEMORY_V2_SOL_RUN_GUARD_USD", "two"),
        ("UNIFY_MEMORY_V2_SOL_RUN_GUARD_USD", "$5"),
        ("UNIFY_MEMORY_V2_SOL_USAGE", "yes"),
        ("UNIFY_MEMORY_V2_SOL_USAGE", "1"),
        ("UNIFY_MEMORY_V2_SOL_USAGE", "true"),
        ("UNIFY_MEMORY_V2_DIALOGUE", "on"),
        ("UNIFY_MEMORY_V2_DIALOGUE", "user"),
        ("UNIFY_MEMORY_V2_DIALOGUE", "env/env"),
        ("UNIFY_MEMORY_V2_DIALOGUE", "dialogue:env"),
    ],
)
def test_refuses_other_values(name, raw, monkeypatch):
    monkeypatch.setenv(name, raw)
    with pytest.raises(Exception, match=name):
        ProductionSettings()


def test_the_allowance_assignment_passes_the_credential_guards():
    """The job controller refuses argv matching ``token\\s*=`` and the launcher treats a name part
    containing TOKEN as a credential unless it is exactly ``TOKENS``: the allowance's name passes both.
    """
    import re

    line = "UNIFY_MEMORY_V2_SOL_ALLOWANCE_USD_PER_TOKENS=0.00000073"
    assert re.search(r"(?i)token\s*=", line) is None
    parts = line.split("=", 1)[0].split("_")
    assert [p for p in parts if "TOKEN" in p] == ["TOKENS"]


# --- stage-5 test checks in the gate (memory v2.1) ---------------------------------------------------------

_QA_ON_OFF = (
    "UNIFY_MEMORY_V2_QA_MUTATION",
    "UNIFY_MEMORY_V2_QA_DETERMINISM",
    "UNIFY_MEMORY_V2_QA_REPLAY",
    "UNIFY_MEMORY_V2_QA_FIXTURE_SIZE",
)


def test_qa_switches_default_off():
    s = ProductionSettings()
    for name in _QA_ON_OFF + ("UNIFY_MEMORY_V2_QA_FIXTURES",):
        assert getattr(s, name) == ""
    assert s.UNIFY_MEMORY_V2_QA_MUTATION_MIN_KILL == "0.5"


@pytest.mark.parametrize("name", _QA_ON_OFF)
@pytest.mark.parametrize(
    "raw,want",
    [("", ""), ("off", ""), (" OFF ", ""), ("on", "on"), (" On ", "on")],
)
def test_qa_on_off_values(name, raw, want, monkeypatch):
    assert _load(monkeypatch, name, raw) == want


@pytest.mark.parametrize(
    "raw,want",
    [("", ""), ("off", ""), ("on", "on"), (" Strict ", "strict")],
)
def test_qa_fixtures_values(raw, want, monkeypatch):
    assert _load(monkeypatch, "UNIFY_MEMORY_V2_QA_FIXTURES", raw) == want


@pytest.mark.parametrize(
    "raw,want",
    [("", "0.5"), ("0", "0"), ("1", "1"), (" 0.75 ", "0.75")],
)
def test_qa_min_kill_values(raw, want, monkeypatch):
    got = _load(monkeypatch, "UNIFY_MEMORY_V2_QA_MUTATION_MIN_KILL", raw)
    assert got == want and isinstance(got, str)


@pytest.mark.parametrize(
    "name,raw",
    [
        ("UNIFY_MEMORY_V2_QA_FIXTURES", "yes"),
        ("UNIFY_MEMORY_V2_QA_FIXTURES", "1"),
        ("UNIFY_MEMORY_V2_QA_MUTATION", "true"),
        ("UNIFY_MEMORY_V2_QA_DETERMINISM", "strict"),
        ("UNIFY_MEMORY_V2_QA_REPLAY", "1"),
        ("UNIFY_MEMORY_V2_QA_FIXTURE_SIZE", "yes"),
        ("UNIFY_MEMORY_V2_QA_MUTATION_MIN_KILL", "1.5"),
        ("UNIFY_MEMORY_V2_QA_MUTATION_MIN_KILL", "5e-1"),
        ("UNIFY_MEMORY_V2_QA_MUTATION_MIN_KILL", "-0.1"),
        ("UNIFY_MEMORY_V2_QA_MUTATION_MIN_KILL", "half"),
    ],
)
def test_qa_refuses_other_values(name, raw, monkeypatch):
    monkeypatch.setenv(name, raw)
    with pytest.raises(Exception, match=name):
        ProductionSettings()
