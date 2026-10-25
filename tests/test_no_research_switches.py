"""Symbolic: the settings hold configuration only.

Every experiment switch was collapsed to the behaviour the rev 2 runs used,
with its other paths deleted. What remains selects a model, a limit, a path or
a deployment policy, never a behaviour under test. The memory settings (and
the run record's) are the memory side's, checked against CONFIG.md by
tests/test_env_allowlist.py; this test covers every other ``UNIFY_`` setting.
"""

from __future__ import annotations

from unify.settings import ProductionSettings

#: The memory side's settings, checked elsewhere.
MEMORY_PREFIXES = ("UNIFY_MEMORY", "UNIFY_AGENTS_")
MEMORY_NAMES = frozenset({"UNIFY_HOST_LEASE_END", "UNIFY_PROFILE"})

#: Each configuration setting, with why it is configuration.
CONFIGURATION = {
    # The model and its reasoning effort.
    "UNIFY_MODEL": "the model every request runs on",
    "UNIFY_REASONING_EFFORT": "the effort paired with the model",
    # Limits that bound a runaway request's cost.
    "UNIFY_MAX_OUTPUT_TOKENS": "bounds one turn's output",
    "UNIFY_MAX_TOOL_LOOP_STEPS": "bounds one request's steps",
    # Paths.
    "UNIFY_LOCAL_ROOT": "the workspace directory",
    "UNIFY_LOG_DIR": "where the log files go",
    # Deployment policy.
    "UNIFY_ENV_NAMESPACES": "the environment's own namespaces (an embedding application's factory)",
    "UNIFY_CELL_LLM_MODELS": "which model endpoints cell code may name",
    "UNIFY_ACTOR_EXAMPLES_CHECK": "whether the world's requests give input/output examples the actor may check against",
    "UNIFY_CELL_ENV_ALLOW": "a runner's own variables a cell gets beside the allow-list",
    "UNIFY_WORKSPACE_NETWORK": "whether cells reach the network, and only through the host's proxy",
    "UNIFY_WORKSPACE_PROXY_PORT": "that proxy's port",
    "UNIFY_EMBED_URL": "the loopback lane provider embeddings go through (an endpoint, like the model's)",
    "UNIFY_LOCAL_EMBEDDINGS": "where embeddings are computed: in process instead of through that lane",
}


def _memory(name: str) -> bool:
    return name.startswith(MEMORY_PREFIXES) or name in MEMORY_NAMES


def test_every_setting_is_configuration():
    fields = {
        name
        for name in ProductionSettings.model_fields
        if name.startswith("UNIFY_") and not _memory(name)
    }
    assert sorted(fields - set(CONFIGURATION)) == []
    assert sorted(set(CONFIGURATION) - fields) == []
