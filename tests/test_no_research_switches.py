"""Symbolic: no research switch remains in the settings but the ones still in progress.

The code freeze (7 Oct 2026) made the chosen configuration the only code
path: every other research switch was baked or deleted, with its losing
paths. What may remain is a switch still in progress (each one has a
section in ``WIP_SWITCHES.md``) or a configuration setting, which selects
a model, a limit, a path or an endpoint rather than a behaviour under test.
A new ``UNIFY_`` setting therefore needs either a ``WIP_SWITCHES.md``
section or a place in ``CONFIGURATION`` here.
"""

from __future__ import annotations

import re
from pathlib import Path

from unify.settings import ProductionSettings

WIP_SWITCHES_MD = Path(__file__).resolve().parents[1] / "WIP_SWITCHES.md"

# The configuration settings (the freeze's step-3 classification).
CONFIGURATION = frozenset(
    {
        "UNIFY_BUILTINS_PROJECT",
        # Deployment policy: a runner's variables a cell gets beside the allow-list.
        "UNIFY_CELL_ENV_ALLOW",
        # Deployment policy: which LLM endpoints cell code may name.
        "UNIFY_CELL_LLM_MODELS",
        "UNIFY_EMBED_URL",
        "UNIFY_ENV_NAMESPACES",
        "UNIFY_GUIDANCE_EMPTY_QUERY",
        "UNIFY_LOCAL_EMBEDDINGS",
        "UNIFY_LOCAL_ROOT",
        "UNIFY_LOG_DIR",
        "UNIFY_MAX_OUTPUT_TOKENS",
        "UNIFY_MAX_TOOL_LOOP_STEPS",
        "UNIFY_MODEL",
        "UNIFY_REASONING_EFFORT",
        # Observability for a proxy: HTTP headers only, the request body as shipped.
        "UNIFY_REQUEST_METADATA_HEADERS",
        "UNIFY_STORE_ADMISSION",
        "UNIFY_STORE_VERIFY",
        "UNIFY_TERMINAL_LOG",
        "UNIFY_TERMINAL_LOG_LEVEL",
        "UNIFY_TOOL_CHOICE_FALLBACK",
        "UNIFY_TURN_STORAGE_REVIEWS",
        "UNIFY_VALIDATE_LLM_PROVIDERS",
        "UNIFY_WORKSPACE_NETWORK",
        "UNIFY_WORKSPACE_PROXY_PORT",
    },
)


def wip_switches() -> set[str]:
    """Every ``UNIFY_`` name a ``WIP_SWITCHES.md`` section heading names."""
    names: set[str] = set()
    for line in WIP_SWITCHES_MD.read_text().splitlines():
        if line.startswith("## "):
            names.update(re.findall(r"\bUNIFY_[A-Z0-9_]+\b", line))
    return names


def test_the_wip_list_names_the_switches_in_progress():
    assert wip_switches() >= {
        "UNIFY_CODE_PROJECTION",
        "UNIFY_STATEFUL_CELLS",
        "UNIFY_LOOP_STOP",
        "UNIFY_LOOP_STOP_K",
    }
    assert not wip_switches() & CONFIGURATION


def test_no_research_switch_remains():
    fields = {
        name for name in ProductionSettings.model_fields if name.startswith("UNIFY_")
    }
    assert sorted(fields - wip_switches() - CONFIGURATION) == []
