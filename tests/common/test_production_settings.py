"""
Tests for ProductionSettings LLM provider validation.

Verifies that unify.init() hard-fails when unillm resolved no key that serves
an LLM call and UNIFY_VALIDATE_LLM_PROVIDERS is enabled (the default).
"""

import pytest
from pydantic import SecretStr
from unillm.settings import PROVIDER_KEYS
from unillm.settings import SETTINGS as UNILLM_SETTINGS

from unify.settings import ProductionSettings


@pytest.fixture
def resolved_keys(monkeypatch, tmp_path):
    """Stand in for a machine whose keys reach unillm from Secret Manager.

    No provider key is in the environment or in a ``.env``, and unillm holds
    exactly the keys passed, with every other key it resolves left empty.
    """
    monkeypatch.chdir(tmp_path)
    for name in PROVIDER_KEYS:
        monkeypatch.delenv(name, raising=False)

    def resolve(**keys: str) -> None:
        for name in PROVIDER_KEYS:
            monkeypatch.setattr(UNILLM_SETTINGS, name, SecretStr(keys.get(name, "")))

    return resolve


class TestLLMProviderValidation:
    """Tests for validate_llm_providers method."""

    def test_default_model_is_gpt_5_6_sol(self):
        """UNIFY_MODEL defaults to the primary production reasoning model."""
        field_info = ProductionSettings.model_fields["UNIFY_MODEL"]
        assert field_info.default == "openai/gpt-5.6-sol@openrouter"
        effort = ProductionSettings.model_fields["UNIFY_REASONING_EFFORT"]
        assert effort.default == "high"

    def test_validation_fails_when_no_credential_resolved(self, resolved_keys):
        """Validation raises RuntimeError when unillm resolved no key."""
        resolved_keys()
        settings = ProductionSettings(UNIFY_VALIDATE_LLM_PROVIDERS=True)
        with pytest.raises(RuntimeError) as exc_info:
            settings.validate_llm_providers()

        error_msg = str(exc_info.value)
        assert "At least one LLM provider credential is required" in error_msg

    def test_validation_passes_when_openrouter_credential_resolved(
        self,
        resolved_keys,
    ):
        """An OpenRouter key serves the default model and every non-Anthropic one."""
        resolved_keys(OPENROUTER_API_KEY="sk-or-test")
        settings = ProductionSettings(UNIFY_VALIDATE_LLM_PROVIDERS=True)
        settings.validate_llm_providers()

    def test_validation_passes_when_anthropic_credential_resolved(
        self,
        resolved_keys,
    ):
        """An Anthropic key serves the Anthropic models, which unillm calls directly."""
        resolved_keys(ANTHROPIC_API_KEY="sk-ant-test")
        settings = ProductionSettings(UNIFY_VALIDATE_LLM_PROVIDERS=True)
        settings.validate_llm_providers()

    def test_validation_rejects_keys_no_call_uses(self, resolved_keys):
        """unillm resolves the management and Together keys but sends neither."""
        resolved_keys(
            OPENROUTER_MANAGEMENT_API_KEY="sk-test",
            TOGETHER_API_KEY="sk-test",
        )
        settings = ProductionSettings(UNIFY_VALIDATE_LLM_PROVIDERS=True)
        with pytest.raises(RuntimeError):
            settings.validate_llm_providers()

    def test_validation_skipped_when_disabled(self, resolved_keys):
        """Validation is skipped when UNIFY_VALIDATE_LLM_PROVIDERS=False."""
        resolved_keys()
        settings = ProductionSettings(UNIFY_VALIDATE_LLM_PROVIDERS=False)
        # Should not raise even with no key resolved
        settings.validate_llm_providers()

    def test_validation_enabled_by_default(self):
        """UNIFY_VALIDATE_LLM_PROVIDERS defaults to True in code."""
        # Verify the class-level default is True (env vars may override at runtime)
        field_info = ProductionSettings.model_fields["UNIFY_VALIDATE_LLM_PROVIDERS"]
        assert field_info.default is True
