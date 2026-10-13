"""
ConversationManager-specific settings.

These settings are composed into the global ProductionSettings.
Environment variables use the prefix UNIFY_CONVERSATION_.
"""

from pydantic_settings import BaseSettings, SettingsConfigDict


class ConversationSettings(BaseSettings):
    """ConversationManager settings.

    Attributes:
        IMPL: Implementation type - "real" or "simulated".
        SLOW_BRAIN_MODEL: Shared ConversationManager slow-brain model. Empty
            falls back to the global shared model (UNIFY_MODEL / assistant
            default resolution). Override via
            UNIFY_CONVERSATION_SLOW_BRAIN_MODEL.
        SLOW_BRAIN_REASONING_EFFORT: Reasoning effort paired with
            SLOW_BRAIN_MODEL when that setting is non-empty. Empty leaves
            call-site effort intact. Override via
            UNIFY_CONVERSATION_SLOW_BRAIN_REASONING_EFFORT.
    """

    SLOW_BRAIN_MODEL: str = "openai/gpt-5.6-terra@openrouter"
    SLOW_BRAIN_REASONING_EFFORT: str = "high"
    IMPL: str = "real"

    model_config = SettingsConfigDict(
        env_prefix="UNIFY_CONVERSATION_",
        case_sensitive=True,
        extra="ignore",
    )
