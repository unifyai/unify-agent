"""The memory agents' profiles: one row per agent, binding it to its tools, its box mounts (with the mode the box
really gives them), what it is denied, and what it consumes and produces.

Rows use role names only; every model is a setting. The writer pass takes its tool set from here
(:func:`tool_json`), and the tools themselves come from the shared spec (:mod:`.tool_spec`), so a profile can only
name tools that exist.
"""

from __future__ import annotations

from . import tool_spec as _tool_spec

#: Which spec role each tool-using agent draws from.
SPEC_ROLE = {"writer": "write", "curator": "curate", "helper": "helper"}

AGENT_PROFILES: tuple[dict, ...] = (
    {
        "agent": "actor",
        "mode": "consumer",
        "instructions": "unify.memory_v2.library_prompts:guide_library_mode",
        # memory.index / memory.show / memory.find are library functions called in cells, not tools
        "tools": [],
        "mounts": [{"path": "<library scratch copy>", "mode": "rw"}],
        "denied": ["/inputs", "staging", "pool", "other episodes", "library repository"],
        "consumes": [{"artifact": "library export", "path": "<library scratch copy>"}],
        "produces": [
            {"artifact": "episode record", "path": "<state>/episodes/<episode>", "consumer": "writer"},
            {"artifact": "conversation", "path": "<request conversation>", "consumer": "fork"},
        ],
    },
    {
        "agent": "fork",
        "mode": "review",
        "instructions": "unify.memory_v2.integration.fork:request_doc",
        # the actor's tool schemas stay in the request only to keep the cached prefix; no call is executed
        "tools": [],
        "mounts": [
            {"path": "/usr", "mode": "ro"},
            {"path": "/s", "mode": "ro"},
            {"path": "/staging", "mode": "rw"},
        ],
        "denied": ["tool execution", "home", "state dir", "library", "network except the fork lane"],
        "consumes": [{"artifact": "conversation", "path": "<request conversation>"}],
        "produces": [
            {
                "artifact": "staging",
                "path": "<state>/memory-staging/<episode>/",
                "consumer": "writer",
            },
        ],
    },
    {
        "agent": "writer",
        "mode": "write",
        "instructions": "unify.memory_v2.library_prompts:write_brief_now",
        "tools": [
            "read",
            "grep",
            "overview",
            "read_episode",
            "pool",
            "fixture",
            "replay",
            "execute_code",
            "check",
            "finish",
        ],
        "mounts": [
            {"path": "/inputs", "mode": "ro"},
            {"path": "/memory", "mode": "rw"},
            {"path": "/memory/INDEX.md", "mode": "ro"},
            {"path": "/memory/links.json", "mode": "ro"},
            {"path": "/memory/memory/__init__.py", "mode": "ro"},
        ],
        "denied": ["network", "credentials", "/outputs inside cells", ".git"],
        "consumes": [
            {"artifact": "batch map", "path": "/inputs/batch_map.json"},
            {"artifact": "staging", "path": "/inputs/staging"},
            {"artifact": "pool", "path": "/inputs/pool"},
            {"artifact": "episode record", "path": "read_episode"},
            {"artifact": "gate result", "path": "/inputs/gate"},
            {"artifact": "drafts", "path": "/inputs/drafts"},
        ],
        "produces": [
            {"artifact": "library item", "path": "/memory/memory/", "consumer": "gate"},
            {"artifact": "manifest", "path": "/memory/.pass/manifest.json", "consumer": "gate"},
            {"artifact": "fixture listing", "path": "/memory/.pass/fixtures.json", "consumer": "gate"},
        ],
    },
    {
        "agent": "curator",
        "mode": "curate",
        "instructions": "unify.memory_v2.library_prompts:curate_brief_now",
        "tools": ["read", "grep", "read_episode", "fixture", "execute_code", "check", "finish"],
        "mounts": [
            {"path": "/inputs", "mode": "ro"},
            {"path": "/memory", "mode": "rw"},
            {"path": "/memory/INDEX.md", "mode": "ro"},
            {"path": "/memory/links.json", "mode": "ro"},
            {"path": "/memory/memory/__init__.py", "mode": "ro"},
        ],
        "denied": [
            "pool",
            "similar",
            "replay",
            "delegate",
            "overview",
            "staging",
            "batch map",
            "network",
            "credentials",
        ],
        "consumes": [
            {"artifact": "curate inputs", "path": "/inputs/curate"},
            {"artifact": "gate result", "path": "/inputs/gate"},
            {"artifact": "episode record", "path": "read_episode"},
        ],
        "produces": [
            {"artifact": "library item", "path": "/memory/memory/", "consumer": "gate"},
            {"artifact": "manifest", "path": "/memory/.pass/manifest.json", "consumer": "gate"},
        ],
    },
    {
        "agent": "helper",
        "mode": "helper",
        "instructions": "unify.memory_v2.analysts:run",
        "tools": ["read", "grep", "overview", "read_episode", "execute_code"],
        "mounts": [
            {"path": "/inputs", "mode": "ro"},
            {"path": "/memory", "mode": "ro"},
            {"path": "/work", "mode": "rw"},
        ],
        "denied": ["library writes", "pool", "writer staging", "network", "credentials"],
        "consumes": [
            {"artifact": "task", "path": "delegate arguments"},
            {"artifact": "episode record", "path": "read_episode"},
        ],
        "produces": [
            {
                "artifact": "staging",
                "path": "<state>/memory-helpers/<pass>/<call>/",
                "consumer": "writer",
            },
        ],
    },
    {
        "agent": "gate",
        "mode": "repair",
        "instructions": "unify.memory_v2.repair:repair_message",
        "tools": [],
        "mounts": [],
        "denied": ["network"],
        "consumes": [
            {"artifact": "manifest", "path": "/memory/.pass/manifest.json"},
            {"artifact": "fixture listing", "path": "/memory/.pass/fixtures.json"},
        ],
        "produces": [
            {"artifact": "gate result", "path": "/inputs/gate/result-<round>.md", "consumer": "writer"},
            {"artifact": "library item", "path": "library main", "consumer": "actor"},
        ],
    },
)


def profile(agent: str) -> dict:
    for row in AGENT_PROFILES:
        if row["agent"] == agent:
            return row
    raise KeyError(f"no agent profile {agent!r}")


def tool_names(agent: str, flags: frozenset[str] = frozenset()) -> list[str]:
    """The tools *agent* gets with the configuration *flags* on, in the spec's order: the spec's tools for its role
    that its profile names (a tool the spec gates on a flag is left out while the flag is off)."""
    row = profile(agent)
    role = SPEC_ROLE.get(agent)
    if role is None:
        return []
    named = set(row["tools"])
    return [s.name for s in _tool_spec.tools_for(role, flags) if s.name in named]


def tool_json(agent: str, flags: frozenset[str] = frozenset()) -> list[dict]:
    names = set(tool_names(agent, flags))
    return [s.as_tool() for s in _tool_spec.tools_for(SPEC_ROLE[agent], flags) if s.name in names]
