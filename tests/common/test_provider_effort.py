from __future__ import annotations

from decimal import Decimal

import pytest

from unify.common import llm_client
from unify.memory.integration.switch import writer_effort_scale_map, writer_max_calls_map
from unify.settings import SETTINGS, ProductionSettings


class _Captured:
    def __init__(self, model, **config):
        self.model = model
        self.config = config
        self.reasoning_effort = config.get("reasoning_effort")

    def set_stream_transport(self, on):  # the streamed-calls switch touches the client
        self.stream = on


@pytest.fixture
def captured(monkeypatch):
    monkeypatch.setattr(llm_client.unillm, "Unify", _Captured)
    monkeypatch.setattr(llm_client.unillm, "AsyncUnify", _Captured)
    return _Captured


def test_setting_accepts_provider():
    assert ProductionSettings.parse_reasoning_effort("provider") == "provider"
    assert ProductionSettings.parse_reasoning_effort(" Provider ") == "provider"


def test_provider_run_sends_no_effort_even_when_asked(captured, monkeypatch):
    monkeypatch.setattr(SETTINGS, "UNIFY_REASONING_EFFORT", "provider")
    for kwargs in ({}, {"reasoning_effort": "high"}, {"reasoning_effort": "low"}):
        client = llm_client.new_llm_client("google/gemini-3-flash-preview@openrouter", **kwargs)
        assert client.config["reasoning_effort"] is None


def test_explicit_provider_effort_sends_none(captured, monkeypatch):
    monkeypatch.setattr(SETTINGS, "UNIFY_REASONING_EFFORT", "high")
    client = llm_client.new_llm_client("google/gemini-3-flash-preview@openrouter", reasoning_effort="provider")
    assert client.config["reasoning_effort"] is None


def test_other_runs_keep_their_effort(captured, monkeypatch):
    monkeypatch.setattr(SETTINGS, "UNIFY_REASONING_EFFORT", "high")
    assert llm_client.new_llm_client("openai/gpt-6-luna@openrouter").config["reasoning_effort"] == "high"
    assert llm_client.new_llm_client("openai/gpt-6-luna@openrouter", reasoning_effort="low").config["reasoning_effort"] == "low"


def test_effort_maps_take_an_optional_provider_key():
    assert writer_effort_scale_map("low:1,medium:2,high:4,provider:4")["provider"] == Decimal("4")
    assert "provider" not in writer_effort_scale_map("low:1,medium:2,high:4")
    assert writer_max_calls_map("low:40,medium:80,high:80,provider:80")["provider"] == 80
    with pytest.raises(ValueError):
        writer_effort_scale_map("low:1,medium:2,high:4,provider:4,provider:4")
    with pytest.raises(ValueError):
        writer_effort_scale_map("low:1,medium:2,provider:4")
