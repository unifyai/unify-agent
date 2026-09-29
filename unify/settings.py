"""
unify/settings.py
==================

Centralized runtime settings using pydantic-settings.

All settings can be overridden via environment variables or the ``.env`` file
in the working directory.
"""

from typing import Any

import unillm
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from unify.actor.settings import ActorSettings
from unify.conversation_manager.settings import ConversationSettings
from unify.function_manager.settings import FunctionSettings
from unify.guidance_manager.settings import GuidanceSettings


def _parse_bool(v: Any) -> bool:
    """Parse a value as boolean."""
    if isinstance(v, bool):
        return v
    if isinstance(v, str):
        return v.lower() in ("true", "yes", "1", "on")
    return bool(v)


class ProductionSettings(BaseSettings):
    """Runtime settings; test settings (TestingSettings) inherit from this class."""

    # ─────────────────────────────────────────────────────────────────────────
    # Local Workspace
    # ─────────────────────────────────────────────────────────────────────────
    # Root directory for local file operations, CodeActActor working directory
    # and virtual environments. Defaults to ``<UNIFY_HOME>/workspace`` (with
    # ``UNIFY_HOME`` defaulting to ``~/.unify``) when empty.
    UNIFY_LOCAL_ROOT: str = ""

    # ─────────────────────────────────────────────────────────────────────────
    # Core LLM Settings
    # ─────────────────────────────────────────────────────────────────────────
    UNIFY_MODEL: str = "openai/gpt-5.6-sol@openrouter"
    # Reasoning effort paired with UNIFY_MODEL when no per-assistant default is
    # set. Empty leaves per-call-site effort levels untouched.
    UNIFY_REASONING_EFFORT: str = "high"
    # Ceiling on output tokens for one actor turn. Unset, the provider ceiling
    # applies (128k on current OpenAI models), so a turn that degenerates into
    # repetition bills and blocks for the full window — observed at eight
    # minutes. Sized to bound that, not to shape normal turns: reasoning tokens
    # count toward it, so keep it well above what a long reasoning turn plus a
    # large code block needs. Set to 0 to restore the provider ceiling.
    UNIFY_MAX_OUTPUT_TOKENS: int = 32768

    # Ceiling on tool-calling iterations for one agent run. Unset, a run ends
    # only when the model decides to stop, so a loop that never converges
    # bills indefinitely. Sized to bound that rather than to shape normal
    # runs: a long agentic task uses far fewer steps than this, and a caller
    # that genuinely needs more can pass ``max_steps`` explicitly. Set to 0 to
    # restore unbounded iteration.
    UNIFY_MAX_TOOL_LOOP_STEPS: int = 300

    # Fail init when unillm holds no provider key. The keys live only in
    # unillm's settings, which read them from the environment, ``.env`` and, on
    # a machine with the team's service-account key, Google Secret Manager.
    UNIFY_VALIDATE_LLM_PROVIDERS: bool = True
    # Storage review of a persistent session: by default it runs once, when the session
    # ends, over the whole trajectory. Set true to also review at every completed turn
    # that ran tools (for sessions that live long enough that waiting for the end would
    # defer distillation indefinitely).
    UNIFY_TURN_STORAGE_REVIEWS: bool = False

    # ─────────────────────────────────────────────────────────────────────────
    # Skill Search
    # ─────────────────────────────────────────────────────────────────────────
    # Embed skills for semantic search with BAAI/bge-small-en-v1.5 in process
    # instead of openai/text-embedding-3-small through OpenRouter. It needs no
    # network once its weights (134 MB) are downloaded, but reads only English
    # and the first 512 tokens of each text.
    UNIFY_LOCAL_EMBEDDINGS: bool = False
    # The endpoint OpenRouter-style embedding requests are posted to, for
    # example a proxy that tracks their cost. Empty posts to openrouter.ai.
    UNIFY_EMBED_URL: str = ""
    # Leave a stored function that cannot be loaded out of a search, list or
    # filter that loads its results, naming it in a warning, instead of
    # failing the whole result.
    UNIFY_SEARCH_SKIP_UNLOADABLE: bool = False

    # ─────────────────────────────────────────────────────────────────────────
    # Environment Namespaces and the Storage Check
    # ─────────────────────────────────────────────────────────────────────────
    # ``package.module:factory`` entries, comma-separated: factories that
    # register the environment's own callable surface as ``primitives.<name>``
    # namespaces at start-up (unify/function_manager/primitives/environment.py).
    # Empty registers nothing.
    UNIFY_ENV_NAMESPACES: str = ""
    # ``resolve``: before a function is stored, every name and every
    # ``primitives.*`` reference in it must resolve against the sandbox's
    # globals and the registered namespaces, and it must load as a search
    # would load it; otherwise ``add_functions`` refuses it and says why.
    # Empty stores without the check.
    UNIFY_STORE_CHECK: str = ""
    # Path of a JSON file in which an external check of the session's outcome
    # admits (``{"admit": true}``) the review that runs when a session ends.
    # A missing, unreadable or malformed file, or any other ``admit``, skips
    # that review. While set, the session's own library write tools are
    # withheld and turn-level reviews are off, so the libraries change only
    # through an admitted review. Empty reviews every session as shipped.
    UNIFY_STORE_ADMISSION: str = ""

    # ─────────────────────────────────────────────────────────────────────────
    # Builtins Catalogue
    # ─────────────────────────────────────────────────────────────────────────
    # Name of the project holding the builtins catalogues (function primitives
    # and guidance), seeded from the committed snapshots at start-up.
    UNIFY_BUILTINS_PROJECT: str = "Builtins"

    # ─────────────────────────────────────────────────────────────────────────
    # Logging / Observability
    # ─────────────────────────────────────────────────────────────────────────
    PYTEST_LOG_TO_FILE: bool = True
    # Directory for Unify LOGGER file output (async tool loop, managers, etc.)
    # When set, logs are written to {UNIFY_LOG_DIR}/unify.log
    # Default: None (console only)
    UNIFY_LOG_DIR: str = ""

    # ─────────────────────────────────────────────────────────────────────────
    # Terminal Logging
    # ─────────────────────────────────────────────────────────────────────────
    UNIFY_TERMINAL_LOG: bool = True
    UNIFY_TERMINAL_LOG_LEVEL: str = "INFO"

    # ─────────────────────────────────────────────────────────────────────────
    # Composed Manager Settings
    # ─────────────────────────────────────────────────────────────────────────
    # Each manager owns its settings in its own settings.py file.
    # Access via SETTINGS.function.IMPL, SETTINGS.guidance.IMPL, etc.
    actor: ActorSettings = Field(default_factory=ActorSettings)
    conversation: ConversationSettings = Field(default_factory=ConversationSettings)
    function: FunctionSettings = Field(default_factory=FunctionSettings)
    guidance: GuidanceSettings = Field(default_factory=GuidanceSettings)

    # ─────────────────────────────────────────────────────────────────────────
    # Validators
    # ─────────────────────────────────────────────────────────────────────────
    @field_validator(
        "UNIFY_TERMINAL_LOG",
        "PYTEST_LOG_TO_FILE",
        "UNIFY_VALIDATE_LLM_PROVIDERS",
        "UNIFY_TURN_STORAGE_REVIEWS",
        "UNIFY_LOCAL_EMBEDDINGS",
        "UNIFY_SEARCH_SKIP_UNLOADABLE",
        mode="before",
    )
    @classmethod
    def parse_bool_fields(cls, v: Any) -> bool:
        return _parse_bool(v)

    @field_validator("UNIFY_STORE_CHECK", mode="before")
    @classmethod
    def parse_store_check(cls, v: Any) -> str:
        value = str(v or "").strip().lower()
        if value not in ("", "resolve"):
            raise ValueError(f"UNIFY_STORE_CHECK must be empty or 'resolve', not {v!r}")
        return value

    model_config = SettingsConfigDict(
        env_file=".env",
        case_sensitive=True,
        extra="ignore",
    )

    def validate_llm_providers(self) -> None:
        """Validate that unillm holds a key it can reach an LLM provider with.

        unillm calls Anthropic models directly and every other model through
        OpenRouter, so those two keys are the only ones that serve a call.

        Raises:
            RuntimeError: If unillm resolved neither key.
        """
        if not self.UNIFY_VALIDATE_LLM_PROVIDERS:
            return
        keys = (unillm.SETTINGS.OPENROUTER_API_KEY, unillm.SETTINGS.ANTHROPIC_API_KEY)
        if not any(key.get_secret_value() for key in keys):
            raise RuntimeError(
                "At least one LLM provider credential is required. "
                "Set OPENROUTER_API_KEY and/or ANTHROPIC_API_KEY.",
            )


# Singleton instance for production code
SETTINGS = ProductionSettings()
