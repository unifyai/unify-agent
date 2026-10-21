"""Symbolic: cell code may call only the LLM endpoints UNIFY_CELL_LLM_MODELS allows.

Nothing here reaches a model or the network: every client constructor a test
could reach is a fake, or fails the test when it is called. The base
(45322b89d) bound the harness's own ``query_llm``, ``list_llms`` and the
``unillm`` module as cell globals, so any model a cell named was called with
the harness's keys; each refusal below fails there.
"""

from __future__ import annotations

from typing import Any

import pytest

from unify.common import cell_models, reasoning
from unify.common.act_llm_profiles import (
    ACT_LLM_PROFILES,
    CURRENT_ACT_LLM_PROFILE,
)
from unify.session_details import SESSION_DETAILS
from unify.settings import SETTINGS, ProductionSettings

LUNA = "openai/gpt-6-luna@openrouter"
OPUS = "anthropic/claude-opus-4.6@openrouter"
MINI = "openai/gpt-5.4-mini@openrouter"


@pytest.fixture
def luna_session(monkeypatch: pytest.MonkeyPatch) -> None:
    """A session configured with gpt-6-luna and the default policy."""
    monkeypatch.setattr(SETTINGS, "UNIFY_MODEL", LUNA)
    monkeypatch.setattr(SETTINGS, "UNIFY_CELL_LLM_MODELS", "")
    monkeypatch.setattr(SESSION_DETAILS.assistant, "default_model", "")


def _fail(*_args: Any, **_kwargs: Any) -> Any:
    raise AssertionError("a client was built for a refused model")


class _FakeClient:
    def __init__(self, endpoint: str = LUNA, result: Any = "ok") -> None:
        self.endpoint = endpoint
        self.result = result
        self.generated: list[dict] = []

    async def generate(self, **kwargs: Any) -> Any:
        self.generated.append(kwargs)
        return self.result

    def set_endpoint(self, value: str) -> "_FakeClient":
        self.endpoint = value
        return self


# ── the setting ──────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("raw", "parsed"),
    [
        ("", ""),
        ("  ", ""),
        ("any", "any"),
        (" ANY ", "any"),
        (f" {MINI} , {OPUS} ", f"{MINI},{OPUS}"),
    ],
)
def test_the_setting_parses(raw: str, parsed: str) -> None:
    assert ProductionSettings(UNIFY_CELL_LLM_MODELS=raw).UNIFY_CELL_LLM_MODELS == (
        parsed
    )


@pytest.mark.parametrize("raw", ["gpt-5.5", "any,openai/gpt-5.5@openrouter", "@x"])
def test_the_setting_refuses_what_is_not_an_endpoint_list(raw: str) -> None:
    with pytest.raises(ValueError, match="UNIFY_CELL_LLM_MODELS"):
        ProductionSettings(UNIFY_CELL_LLM_MODELS=raw)


def test_the_default_is_the_session_model_only() -> None:
    assert ProductionSettings().UNIFY_CELL_LLM_MODELS == ""


# ── check_model ──────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "alias",
    [LUNA, f"openrouter/{LUNA}", f"  {LUNA} "],
)
def test_check_model_allows_the_session_model_and_its_aliases(
    luna_session: None,
    alias: str,
) -> None:
    cell_models.check_model(alias)
    cell_models.check_model(None)


@pytest.mark.parametrize(
    "other",
    [
        OPUS,
        "openai/gpt-5.5@openrouter",
        "openai/gpt-6-luna",  # no provider: not unillm's endpoint
        "gpt-6-luna@openrouter",  # another OpenRouter id
        "openai/gpt-6-luna@vertex-ai",  # another provider
    ],
)
def test_check_model_refuses_any_other_endpoint(
    luna_session: None,
    other: str,
) -> None:
    with pytest.raises(cell_models.CellModelRefused) as raised:
        cell_models.check_model(other)
    message = str(raised.value)
    assert f"session's model {LUNA}" in message
    assert "UNIFY_CELL_LLM_MODELS" in message
    assert "omit model= to use it" in message
    assert isinstance(raised.value, PermissionError)


def test_check_model_refuses_a_model_that_is_not_a_string(luna_session) -> None:
    with pytest.raises(cell_models.CellModelRefused, match="model@provider"):
        cell_models.check_model(["openai/gpt-5.5@openrouter"])


def test_any_allows_every_endpoint(luna_session, monkeypatch) -> None:
    monkeypatch.setattr(SETTINGS, "UNIFY_CELL_LLM_MODELS", "any")
    for model in (OPUS, MINI, "anything@else"):
        cell_models.check_model(model)
    assert cell_models.allowed_models() is None


def test_an_explicit_list_adds_to_the_session_model(luna_session, monkeypatch):
    monkeypatch.setattr(SETTINGS, "UNIFY_CELL_LLM_MODELS", MINI)
    cell_models.check_model(LUNA)
    cell_models.check_model(MINI)
    cell_models.check_model(f"openrouter/{MINI}")
    with pytest.raises(cell_models.CellModelRefused, match=f"or {MINI}"):
        cell_models.check_model(OPUS)


def test_the_act_profile_model_is_the_sessions(luna_session) -> None:
    profile = ACT_LLM_PROFILES["gpt_5_5_high"]
    token = CURRENT_ACT_LLM_PROFILE.set(profile)
    try:
        assert cell_models.configured_models() == [profile.model, LUNA]
        cell_models.check_model(profile.model)
        with pytest.raises(cell_models.CellModelRefused):
            cell_models.check_model(OPUS)
    finally:
        CURRENT_ACT_LLM_PROFILE.reset(token)


def test_the_assistants_default_model_is_the_sessions(luna_session, monkeypatch):
    monkeypatch.setattr(SESSION_DETAILS.assistant, "default_model", MINI)
    assert cell_models.configured_models() == [MINI]
    cell_models.check_model(MINI)
    with pytest.raises(cell_models.CellModelRefused):
        cell_models.check_model(LUNA)


# ── the cell's query_llm ─────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_cell_query_llm_refuses_another_model_before_any_client(
    luna_session,
    monkeypatch,
) -> None:
    monkeypatch.setattr(reasoning, "new_llm_client", _fail)
    with pytest.raises(cell_models.CellModelRefused, match="omit model= to use it"):
        await cell_models.cell_query_llm("x", model=OPUS)
    # Nor through the client's or the call's options.
    with pytest.raises(cell_models.CellModelRefused):
        await cell_models.cell_query_llm("x", client_kwargs={"endpoint": OPUS})
    with pytest.raises(cell_models.CellModelRefused):
        await cell_models.cell_query_llm("x", model=None, endpoint=OPUS)


@pytest.mark.asyncio
async def test_cell_query_llm_without_a_model_uses_the_configured_one(
    luna_session,
    monkeypatch,
) -> None:
    client = _FakeClient()
    built: list = []

    def fake_new_llm_client(model: Any = None, **kwargs: Any) -> _FakeClient:
        built.append(model)
        return client

    monkeypatch.setattr(reasoning, "new_llm_client", fake_new_llm_client)
    assert await cell_models.cell_query_llm("x") == "ok"
    # Naming the session's model (in any alias form) is the same model.
    assert await cell_models.cell_query_llm("x", model=f"openrouter/{LUNA}") == "ok"
    # None resolves in new_llm_client to the configured model, as before.
    assert built == [None, f"openrouter/{LUNA}"]


@pytest.mark.asyncio
async def test_harness_query_llm_and_unillm_are_unrestricted(
    luna_session,
    monkeypatch,
) -> None:
    """The policy is cell-scoped: the controller's own query_llm and unillm
    (memory passes, reviews) still name any model under the default."""
    import unillm

    built: list = []

    def fake_new_llm_client(model: Any = None, **kwargs: Any) -> _FakeClient:
        built.append(model)
        return _FakeClient(endpoint=model)

    monkeypatch.setattr(reasoning, "new_llm_client", fake_new_llm_client)
    assert await reasoning.query_llm("x", model=OPUS) == "ok"
    assert built == [OPUS]
    # Building a client sends nothing.
    assert unillm.AsyncUnify(OPUS).endpoint == OPUS
    # The same through the cell's globals is refused.
    with pytest.raises(cell_models.CellModelRefused):
        await cell_models.cell_query_llm("x", model=OPUS)
    with pytest.raises(cell_models.CellModelRefused):
        cell_models.cell_unillm.AsyncUnify(OPUS)
    assert built == [OPUS]


# ── the cell's list_llms ─────────────────────────────────────────────────────


def test_cell_list_llms_lists_only_allowed_endpoints(luna_session, monkeypatch):
    registry = [MINI, OPUS, "openai/gpt-5.5@openrouter", "claude-4.6-opus@anthropic"]
    monkeypatch.setattr(
        reasoning,
        "list_llms",
        lambda provider=None: [
            e for e in registry if provider is None or e.endswith(f"@{provider}")
        ],
    )
    # The session's model is routable though the registry does not list it.
    assert cell_models.cell_list_llms() == [LUNA]
    assert cell_models.cell_list_llms("anthropic") == []
    monkeypatch.setattr(SETTINGS, "UNIFY_CELL_LLM_MODELS", MINI)
    assert cell_models.cell_list_llms() == [MINI, LUNA]
    assert cell_models.cell_list_llms("openrouter") == [MINI, LUNA]
    monkeypatch.setattr(SETTINGS, "UNIFY_CELL_LLM_MODELS", "any")
    assert cell_models.cell_list_llms() == registry


# ── the cell's unillm ────────────────────────────────────────────────────────


def test_cell_unillm_refuses_another_model_at_construction(luna_session, monkeypatch):
    import unillm

    monkeypatch.setattr(unillm, "AsyncUnify", _fail)
    monkeypatch.setattr(unillm, "Unify", _fail)
    facade = cell_models.cell_unillm
    for build in (facade.AsyncUnify, facade.Unify):
        with pytest.raises(cell_models.CellModelRefused, match=LUNA):
            build(OPUS)
        with pytest.raises(cell_models.CellModelRefused):
            build(endpoint=OPUS)
        with pytest.raises(cell_models.CellModelRefused, match="needs an endpoint"):
            build()


def test_cell_unillm_builds_the_session_model_and_holds_its_endpoint(
    luna_session,
    monkeypatch,
) -> None:
    import unillm

    monkeypatch.setattr(unillm, "AsyncUnify", _FakeClient)
    client = cell_models.cell_unillm.AsyncUnify(LUNA)
    assert isinstance(client, cell_models.CellLLMClient)
    assert client.endpoint == LUNA
    with pytest.raises(cell_models.CellModelRefused, match="keep the client's"):
        client.set_endpoint(OPUS)
    assert client.endpoint == LUNA
    # An allowed alias is the same model; the setter's result stays held.
    assert isinstance(
        client.set_endpoint(f"openrouter/{LUNA}"),
        cell_models.CellLLMClient,
    )
    with pytest.raises(cell_models.CellModelRefused):
        client.generate(user_message="x", model=OPUS)
    with pytest.raises(AttributeError):
        client.endpoint = OPUS


def test_cell_unillm_refuses_every_other_attribute(luna_session) -> None:
    facade = cell_models.cell_unillm
    for name in ("SETTINGS", "set_cache_backend", "add_llm_event_listener", "os"):
        with pytest.raises(cell_models.CellAttributeRefused, match="not available"):
            getattr(facade, name)
    # A probe with a default reads a refused attribute as missing.
    assert getattr(facade, "SETTINGS", None) is None
    assert getattr(facade, "__wrapped__", None) is None
    assert dir(facade) == ["AsyncUnify", "Unify"]
    with pytest.raises(AttributeError):
        facade.AsyncUnify = _fail
