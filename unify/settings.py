"""
unify/settings.py
==================

Centralized runtime settings using pydantic-settings.

Each setting is read from the environment variable of the same name (the CLI
also loads a ``.env`` file from the working directory into the environment
first). CONFIG.md lists every variable the package reads.
"""

import os
import re
from typing import Any

import unillm
from pydantic import field_validator
from pydantic_settings import BaseSettings

# The reasoning efforts unillm forwards to providers (it maps them per
# provider, e.g. DeepSeek's high/max), with "" for the model's own default.
_REASONING_EFFORTS = ("", "none", "minimal", "low", "medium", "high", "xhigh", "max")


class ProductionSettings(BaseSettings):
    """The package's runtime configuration: models, paths, limits and the memory mode."""

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
    # Reasoning effort paired with UNIFY_MODEL. Empty leaves the model's own
    # default.
    UNIFY_REASONING_EFFORT: str = "high"
    # The LLM endpoints model-written code may name (the ``query_llm``,
    # ``list_llms`` and ``unillm`` globals of a cell, unify/common/cell_models.py).
    # Empty: only the session's configured model (what ``model=None``
    # resolves to). ``any``: every endpoint.
    # Otherwise a comma list of ``model@provider`` endpoints allowed besides
    # the session's model. Harness calls (the actor's loop, reviews,
    # compression) never read it.
    UNIFY_CELL_LLM_MODELS: str = ""
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

    # ─────────────────────────────────────────────────────────────────────────
    # Embeddings (memory's semantic search, unify/common/embeddings.py):
    # deployment configuration, not a behaviour switch (unify/memory/integration/switch.py)
    # ─────────────────────────────────────────────────────────────────────────
    # The loopback lane OpenRouter-style embedding requests are posted to,
    # ``http://127.0.0.1:<port>/<path>``: the run's proxy, which holds the real
    # credential and books the calls. Empty: a text not yet embedded is refused;
    # nothing is posted to a provider directly.
    UNIFY_EMBED_URL: str = ""
    # Embed with BAAI/bge-small-en-v1.5 in process instead (the
    # ``local-embeddings`` extra). It needs no network once its weights (134 MB)
    # are downloaded, but reads only English and the first 512 tokens of a text.
    UNIFY_LOCAL_EMBEDDINGS: bool = False

    # ─────────────────────────────────────────────────────────────────────────
    # Environment Namespaces
    # ─────────────────────────────────────────────────────────────────────────
    # ``package.module:factory`` entries, comma-separated: factories that
    # register the environment's own callable surface as ``primitives.<name>``
    # namespaces at start-up (unify/function_manager/primitives/environment.py).
    # Empty registers nothing.
    UNIFY_ENV_NAMESPACES: str = ""
    # Memory (unify/memory/integration/switch.py; CONFIG.md and docs/memory.md). ``UNIFY_MEMORY`` is the one memory
    # switch: ``on`` (the default; also unset or empty) keeps memory between episodes, ``off`` is the no-memory
    # baseline; anything else is refused.
    # ``UNIFY_MEMORY_CAP_USD`` is memory's spend cap for the memory home (a plain decimal string; empty: 20.00).
    # Every other memory setting is deployment configuration, not a behaviour switch, each described in switch.py's
    # "Deployment configuration" section: ``UNIFY_MEMORY_WRITER_MODEL`` is the writer's and its helpers' model
    # (empty: UNIFY_MODEL).
    UNIFY_MEMORY: str = "on"
    UNIFY_MEMORY_WRITER_MODEL: str = ""
    UNIFY_MEMORY_CAP_USD: str = "20.00"
    # The model-call lanes a launcher hands the controller (deployment configuration, switch.py; never set by hand):
    # each lane's loopback base URL and the inherited descriptor holding its token, read once at start-up and then
    # removed from the environment, plus the writer lane's journal (an absolute path) and the writer model's pinned
    # prices (JSON, USD per million tokens). Empty: no lane.
    UNIFY_MEMORY_WRITER_BASE_URL: str = ""
    UNIFY_MEMORY_WRITER_TOKEN_FD: str = ""
    UNIFY_MEMORY_WRITER_JOURNAL: str = ""
    UNIFY_MEMORY_WRITER_PRICES_USD_PER_MILLION: str = ""
    UNIFY_MEMORY_LIBRARY_FORK_BASE_URL: str = ""
    UNIFY_MEMORY_LIBRARY_FORK_TOKEN_FD: str = ""
    UNIFY_MEMORY_LIBRARY_HELPER_BASE_URL: str = ""
    UNIFY_MEMORY_LIBRARY_HELPER_TOKEN_FD: str = ""

    # ─────────────────────────────────────────────────────────────────────────
    # Workspace Sandbox
    # ─────────────────────────────────────────────────────────────────────────
    # The Python worker that runs every cell, and every subprocess a cell
    # starts, are confined by bubblewrap (unify/sandbox.py,
    # unify/actor/execution/worker.py); without bubblewrap they are refused,
    # never run unconfined.
    # ``proxy``: the sandbox's only network is one loopback port forwarded to
    # the proxy listening on 127.0.0.1:UNIFY_WORKSPACE_PROXY_PORT on the host.
    # Empty: no network at all.
    UNIFY_WORKSPACE_NETWORK: str = ""
    UNIFY_WORKSPACE_PROXY_PORT: int = 0
    # Variables of the harness's environment a sandboxed command (the Python
    # worker that runs every cell, and what it starts) gets besides the built-in
    # allow-list (unify/sandbox.py, CELL_ENV_NAMES and CELL_ENV_PATTERNS): a
    # comma list of names and ``PREFIX_*`` patterns, for a runner's own cell
    # variables. A credential's name or a key-shaped value is never passed.
    UNIFY_CELL_ENV_ALLOW: str = ""
    # Whether the actor and its teammates are offered the example check (``tools.check_examples``) and the one
    # opening line about it. A world whose requests give input/output examples may turn it on; off (the default)
    # the actor's tools and opening carry neither.
    UNIFY_ACTOR_EXAMPLES_CHECK: bool = False

    # ─────────────────────────────────────────────────────────────────────────
    # Logging / Observability
    # ─────────────────────────────────────────────────────────────────────────
    # Directory of the package's log files (``unify.log`` and
    # ``unify_info_only.log``, unify/logger.py). Empty: the ``unify`` command
    # writes them under ``<UNIFY_HOME>/logs``; a program importing the package
    # gets none.
    UNIFY_LOG_DIR: str = ""

    # ─────────────────────────────────────────────────────────────────────────
    # Validators
    # ─────────────────────────────────────────────────────────────────────────
    @field_validator("UNIFY_LOCAL_EMBEDDINGS", mode="before")
    @classmethod
    def parse_local_embeddings(cls, v: Any) -> bool:
        if isinstance(v, str):
            return v.strip().lower() in ("true", "yes", "1", "on")
        return bool(v)

    @field_validator("UNIFY_REASONING_EFFORT", mode="before")
    @classmethod
    def parse_reasoning_effort(cls, v: Any) -> str:
        value = str(v or "").strip().lower()
        if value not in _REASONING_EFFORTS:
            raise ValueError(
                "UNIFY_REASONING_EFFORT must be empty or one of "
                f"{', '.join(_REASONING_EFFORTS[1:])}, not {v!r}",
            )
        return value

    @field_validator(
        "UNIFY_MEMORY",
        "UNIFY_MEMORY_WRITER_MODEL",
        "UNIFY_MEMORY_CAP_USD",
        "UNIFY_MEMORY_WRITER_BASE_URL",
        "UNIFY_MEMORY_WRITER_TOKEN_FD",
        "UNIFY_MEMORY_WRITER_JOURNAL",
        "UNIFY_MEMORY_WRITER_PRICES_USD_PER_MILLION",
        "UNIFY_MEMORY_LIBRARY_FORK_BASE_URL",
        "UNIFY_MEMORY_LIBRARY_FORK_TOKEN_FD",
        "UNIFY_MEMORY_LIBRARY_HELPER_BASE_URL",
        "UNIFY_MEMORY_LIBRARY_HELPER_TOKEN_FD",
        mode="before",
    )
    @classmethod
    def parse_memory(cls, v: Any, info: Any) -> Any:
        from unify.memory.integration import switch

        return switch.PARSERS[info.field_name](v)

    @field_validator("UNIFY_CELL_LLM_MODELS", mode="before")
    @classmethod
    def parse_cell_llm_models(cls, v: Any) -> str:
        text = str(v or "").strip()
        if text.lower() == "any":
            return "any"
        endpoints = [part.strip() for part in text.split(",") if part.strip()]
        for endpoint in endpoints:
            model, _, provider = endpoint.rpartition("@")
            if endpoint.lower() == "any" or not model or not provider:
                raise ValueError(
                    "UNIFY_CELL_LLM_MODELS must be empty, 'any' or a comma list "
                    f"of 'model@provider' endpoints, not {v!r}",
                )
        return ",".join(endpoints)

    @field_validator("UNIFY_CELL_ENV_ALLOW", mode="before")
    @classmethod
    def parse_cell_env_allow(cls, v: Any) -> str:
        entries = [part.strip() for part in str(v or "").split(",") if part.strip()]
        for entry in entries:
            if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*\*?", entry):
                raise ValueError(
                    "UNIFY_CELL_ENV_ALLOW must be a comma list of variable names "
                    f"and PREFIX_* patterns, not {v!r}",
                )
        return ",".join(entries)

    @field_validator("UNIFY_WORKSPACE_NETWORK", mode="before")
    @classmethod
    def parse_workspace_network(cls, v: Any) -> str:
        value = str(v or "").strip().lower()
        if value not in ("", "proxy"):
            raise ValueError(
                f"UNIFY_WORKSPACE_NETWORK must be empty or 'proxy', not {v!r}",
            )
        return value

    def validate_llm_providers(self) -> None:
        """Validate that unillm holds a key it can reach an LLM provider with.

        unillm calls Anthropic models directly and every other model through
        OpenRouter, so those two keys are the only ones that serve a call.

        Raises:
            RuntimeError: If unillm resolved neither key.
        """
        keys = (unillm.SETTINGS.OPENROUTER_API_KEY, unillm.SETTINGS.ANTHROPIC_API_KEY)
        if not any(key.get_secret_value() for key in keys):
            raise RuntimeError(
                "At least one LLM provider credential is required. "
                "Set OPENROUTER_API_KEY and/or ANTHROPIC_API_KEY.",
            )


# Singleton instance for production code
SETTINGS = ProductionSettings()
# The writer lane's token: with UNIFY_MEMORY_WRITER_TOKEN_FD set, this first
# settle reads the token from that inherited descriptor and closes it, before
# the CLI or any controller code can spawn a child; the name leaves the
# environment and the token stays registered with the value-based redactors.
# The CLI checks again after loading .env. Unset, nothing changes.
from unify.memory.integration.switch import settle_writer_route_env  # noqa: E402

settle_writer_route_env(os.environ, SETTINGS)
# The fork lane's token likewise: read once from its descriptor, both route names out of the environment.
from unify.memory.integration.switch import settle_fork_route_env  # noqa: E402

settle_fork_route_env(os.environ, SETTINGS)
# The helper lane's token likewise: read once from its descriptor, both route names out of the environment.
from unify.memory.integration.switch import settle_helper_route_env  # noqa: E402

settle_helper_route_env(os.environ, SETTINGS)
# The embedding lane's token likewise: read once from its descriptor, its name out of the environment.
from unify.lanes.embed_token import settle as settle_embed_token  # noqa: E402

settle_embed_token(os.environ)
