"""Symbolic: the run's reasoning effort (``UNIFY_REASONING_EFFORT``) applies to a client unless its call site names
its own."""

from __future__ import annotations

import pytest

from unify.common import llm_client
from unify.settings import SETTINGS


class _Captured:
    def __init__(self, model, **config):
        self.model = model
        self.config = config
        self.reasoning_effort = config.get("reasoning_effort")

    def set_stream_transport(self, on):  # every client's calls are streamed
        self.stream = on

    def _generate(self, *args, **kwargs):  # the request headers wrap it
        raise AssertionError("no call is made")


@pytest.fixture
def captured(monkeypatch):
    monkeypatch.setattr(llm_client.unillm, "Unify", _Captured)
    monkeypatch.setattr(llm_client.unillm, "AsyncUnify", _Captured)
    return _Captured


def test_the_run_effort_applies_unless_the_call_site_names_its_own(captured, monkeypatch):
    monkeypatch.setattr(SETTINGS, "UNIFY_REASONING_EFFORT", "high")
    assert llm_client.new_llm_client("openai/gpt-6-luna@openrouter").config["reasoning_effort"] == "high"
    assert llm_client.new_llm_client("openai/gpt-6-luna@openrouter", reasoning_effort="low").config["reasoning_effort"] == "low"
