"""Symbolic: ``UNIFY_THOUGHT_FIELD=optional`` makes the code tools' ``thought`` optional.

On the long lean-all Continual-ARC LOW run (``arc-pm-lean8958-h-low-ws0``)
394 of 754 ``execute_code`` cells (52%) did nothing (``pass``, a comment,
``print('')``) and 335 of them carried a required ``thought`` announcing the
next protocol action, which the model then took in text on the next call.
As shipped, ``thought`` is required on ``execute_code`` and
``execute_function``. With the switch the schemas no longer require it and
say it may be left out; a call without it still runs. Screened only together
with ``UNIFY_REPLY_WORDING`` or ``UNIFY_REPLY_CHANNEL``: in 57% of the no-op
calls at LOW effort the ``thought`` was the only reasoning the model gave.
"""

from __future__ import annotations

import pytest

from unify.settings import SETTINGS


def _schemas() -> dict:
    from unify.actor.code_act_actor import CodeActActor
    from unify.common.llm_helpers import method_to_schema

    actor = CodeActActor()
    tools = dict(actor.get_tools("act"))
    return {
        name: method_to_schema(getattr(tools[name], "fn", tools[name]), name)
        for name in ("execute_code", "execute_function")
    }


def _thought(schema: dict) -> tuple[dict, list]:
    params = schema["function"]["parameters"]
    return params["properties"]["thought"], params.get("required", [])


def test_the_switch_is_validated():
    from unify.settings import ProductionSettings

    assert ProductionSettings(UNIFY_THOUGHT_FIELD="optional").UNIFY_THOUGHT_FIELD == (
        "optional"
    )
    assert ProductionSettings(UNIFY_THOUGHT_FIELD="required").UNIFY_THOUGHT_FIELD == ""
    with pytest.raises(ValueError):
        ProductionSettings(UNIFY_THOUGHT_FIELD="never")


def test_off_thought_is_required_as_shipped(monkeypatch):
    monkeypatch.setattr(SETTINGS, "UNIFY_THOUGHT_FIELD", "")
    for name, schema in _schemas().items():
        prop, required = _thought(schema)
        assert "thought" in required, name
        assert "always provide it" in prop["description"], name


def test_optional_thought_is_not_required_and_says_so(monkeypatch):
    monkeypatch.setattr(SETTINGS, "UNIFY_THOUGHT_FIELD", "")
    shipped = _schemas()
    monkeypatch.setattr(SETTINGS, "UNIFY_THOUGHT_FIELD", "optional")
    optional = _schemas()
    for name in ("execute_code", "execute_function"):
        prop, required = _thought(optional[name])
        assert "thought" not in required, name
        assert "you may leave it out" in prop["description"], name
        assert "always provide it" not in prop["description"], name
        # Nothing else in the schema changes.
        before = shipped[name]["function"]["parameters"]
        after = optional[name]["function"]["parameters"]
        assert [r for r in before["required"] if r != "thought"] == after.get(
            "required",
            [],
        )
        assert list(before["properties"]) == list(after["properties"])
        for key in before["properties"]:
            if key != "thought":
                assert before["properties"][key] == after["properties"][key]
    assert "Always provide it." not in (
        optional["execute_function"]["function"]["description"]
    )


def test_a_call_without_thought_is_backfilled(monkeypatch):
    """The loop fills a left-out ``thought`` with "" (``llm_soft_required``)."""
    from unify.common.tool_spec import LLM_SOFT_REQUIRED_DEFAULTS_ATTR

    monkeypatch.setattr(SETTINGS, "UNIFY_THOUGHT_FIELD", "optional")
    from unify.actor.code_act_actor import CodeActActor

    tools = dict(CodeActActor().get_tools("act"))
    for name in ("execute_code", "execute_function"):
        fn = getattr(tools[name], "fn", tools[name])
        assert getattr(fn, LLM_SOFT_REQUIRED_DEFAULTS_ATTR) == {"thought": ""}
