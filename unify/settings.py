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
    # ``package.module:factory``: a verifier for the storage review. A
    # function is stored only after its exact source has passed a run on a
    # held-out task of the same kind (FunctionManager_check_function, offered
    # to the review only while this is set) and static checks against request
    # details and credentials (unify/function_manager/store_verify.py). Needs
    # UNIFY_STORE_ADMISSION. Empty stores without the check.
    UNIFY_STORE_VERIFY: str = ""
    # When a provider refuses a forced tool choice ("required", "any" or one
    # named tool) with HTTP 400 because the model does not support it, retry
    # that call once with tool_choice "auto" and an instruction to make the
    # required call first (re-prompting once if the reply makes none), and
    # send later forced calls to that model this way for the rest of the
    # process (unify/common/tool_choice_fallback.py). Off: the error
    # propagates as shipped.
    UNIFY_TOOL_CHOICE_FALLBACK: bool = False
    # ``stored``: a guidance search without reference text returns the stored
    # guidance entries, newest first, instead of the built-in catalogue (whose
    # hash-derived ids otherwise sort ahead of every stored entry). Empty
    # searches as shipped.
    UNIFY_GUIDANCE_EMPTY_QUERY: str = ""
    # Offer FunctionManager_patch_function and GuidanceManager_patch_guidance,
    # which replace one exact excerpt of a stored entry in place (the patched
    # function is stored through add_functions, so its checks still apply),
    # and keep the previous version of every overwritten function or guidance
    # entry in function_history / guidance_history. Off: no patch tools and
    # nothing is written to history.
    UNIFY_FUNCTION_PATCH: bool = False
    # ``warn``: adding a new function whose normalised code nearly matches a
    # stored one (token Jaccard >= 0.9) stores it and returns a warning naming
    # the stored function. Empty adds as shipped.
    UNIFY_STORE_DEDUPE: str = ""
    # ``ramp``: keep a trust record per stored function (probation, trusted,
    # quarantined) in function_trust. Every reuse is evidence: a call that
    # returns is a pass, one that raises quarantines the function, which is
    # then left out of the searches that load functions and listed to the
    # next storage review as needing repair. A function is trusted after 3
    # passes over 2 distinct inputs (5 over 3 if it can change anything); a
    # changed source or callee puts it back on probation. With
    # UNIFY_STORE_VERIFY set, a reuse is also re-checked in a fresh world
    # with probability 1/2^k after k clean uses
    # (unify/function_manager/store_trust.py). Empty keeps no record.
    UNIFY_STORE_TRUST: str = ""
    # Keep every tool loop's requests a growing, byte-stable prefix, so the
    # provider's prompt cache is reused call after call: the tool list is
    # computed once per session and a tool the phase does not allow is
    # refused by rule instead of removed; messages already sent are never
    # edited; compression asks for its summary as a fork of the conversation;
    # a per-session cache affinity key is passed when the LLM client takes
    # one; and each call logs how much of its input came from the cache
    # (unify/common/_async_tool/cache_discipline.py). Off: as shipped.
    UNIFY_CACHE_DISCIPLINE: bool = False
    # Run the storage review that follows a session as a fork of the session's
    # own conversation: its request is the actor's system prompt, messages,
    # last tools and tool choice, plus one user message with the review
    # rulebook, so it is served from the cache the actor built. Tools outside
    # the library are refused. Needs UNIFY_CACHE_DISCIPLINE (for the fixed
    # tool list); without it, or when the session was compressed or its
    # history changed since its last request, the review runs as shipped and
    # the log says why. Off: as shipped.
    UNIFY_REVIEW_FORK: bool = False

    # ─────────────────────────────────────────────────────────────────────────
    # Session Transcripts
    # ─────────────────────────────────────────────────────────────────────────
    # Append every agent conversation (actor, sub-agents, storage review,
    # compressor) as JSON lines to ``<UNIFY_HOME>/transcripts/<session>.jsonl``
    # and one line per ended session to ``transcripts/index.jsonl``; after a
    # context compression the compressed context points at the file
    # (unify/transcripts.py). Off: nothing is written.
    UNIFY_TRANSCRIPTS: bool = False

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
        "UNIFY_TOOL_CHOICE_FALLBACK",
        "UNIFY_FUNCTION_PATCH",
        "UNIFY_CACHE_DISCIPLINE",
        "UNIFY_REVIEW_FORK",
        "UNIFY_TRANSCRIPTS",
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

    @field_validator("UNIFY_GUIDANCE_EMPTY_QUERY", mode="before")
    @classmethod
    def parse_guidance_empty_query(cls, v: Any) -> str:
        value = str(v or "").strip().lower()
        if value not in ("", "stored"):
            raise ValueError(
                f"UNIFY_GUIDANCE_EMPTY_QUERY must be empty or 'stored', not {v!r}",
            )
        return value

    @field_validator("UNIFY_STORE_DEDUPE", mode="before")
    @classmethod
    def parse_store_dedupe(cls, v: Any) -> str:
        value = str(v or "").strip().lower()
        if value not in ("", "warn"):
            raise ValueError(f"UNIFY_STORE_DEDUPE must be empty or 'warn', not {v!r}")
        return value

    @field_validator("UNIFY_STORE_TRUST", mode="before")
    @classmethod
    def parse_store_trust(cls, v: Any) -> str:
        value = str(v or "").strip().lower()
        if value not in ("", "ramp"):
            raise ValueError(f"UNIFY_STORE_TRUST must be empty or 'ramp', not {v!r}")
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
