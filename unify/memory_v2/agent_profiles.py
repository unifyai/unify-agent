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

#: The real limits of a writer, curator or helper pass, each with its number (the briefs state them as information).
_PASS_LIMITS = [
    {"limit": "reader page", "value": 8000, "unit": "bytes"},
    {"limit": "reader calls per model turn", "value": 16, "unit": "calls"},
    {"limit": "read_episode parts per call", "value": 8, "unit": "parts"},
    {"limit": "cells per model turn", "value": 8, "unit": "cells"},
    {"limit": "cell time", "value": 60, "unit": "s"},
    {"limit": "file under /memory", "value": 1048576, "unit": "bytes"},
    {"limit": "all files under /memory", "value": 20971520, "unit": "bytes"},
    {"limit": "entries under /memory", "value": 2000, "unit": "entries"},
    {"limit": "step guard", "value": 500, "unit": "model and check calls"},
    {"limit": "time left for a check", "value": 60, "unit": "s"},
    {
        "limit": "pass wall clock",
        "value": "UNIFY_MEMORY_LIBRARY_PASS_WALL_S",
        "unit": "s",
    },
]
#: What any memory prompt may inline: constant text, ids and numbers; never a list past these bounds.
_INLINE_BOUND = {"items": 10, "bytes": 1000, "tokens": 250}
_EVENTS = [{"event": "tool_call", "path": "<pass output dir>/tool-events.jsonl"}]
_BOX_MOUNTS = [
    {"path": "/inputs", "mode": "ro"},
    {"path": "/memory", "mode": "rw"},
    {"path": "/memory/INDEX.md", "mode": "ro"},
    {"path": "/memory/links.json", "mode": "ro"},
    {"path": "/memory/memory/__init__.py", "mode": "ro"},
    # host-side only: `read` opens the saved cell outputs; cells cannot open /outputs
    {"path": "/outputs", "mode": "ro", "via": "read"},
]


def _fork_tool_reply() -> str:
    """The one reply every tool call of the fork gets (it writes blocks; no tool runs)."""
    from .integration.fork_worker import TOOL_REPLY

    return TOOL_REPLY


AGENT_PROFILES: tuple[dict, ...] = (
    {
        "agent": "actor",
        "mode": "consumer",
        "instructions": "unify.memory_v2.library_prompts:guide_library_mode",
        # memory.index / memory.show / memory.find are library functions called in cells, not tools
        "tools": [],
        "mounts": [{"path": "<library scratch copy>", "mode": "rw"}],
        "denied": [
            "/inputs",
            "staging",
            "pool",
            "other episodes",
            "library repository",
        ],
        "consumes": [{"artifact": "library export", "path": "<library scratch copy>"}],
        "produces": [
            {
                "artifact": "episode record",
                "path": "<state>/episodes/<episode>",
                "consumer": "writer",
            },
            {
                "artifact": "conversation",
                "path": "<request conversation>",
                "consumer": "fork",
            },
        ],
        "limits": [],
        "inlined": [{"what": "memory guide paragraph", "bound": _INLINE_BOUND}],
        "capabilities": ["call memory.index, memory.show and memory.find in cells"],
        "absent_paths": [
            "/inputs",
            "/inputs/staging",
            "/inputs/pool",
            "<state>/memory-staging",
        ],
        "expectations": [
            {
                "expectation": "look for an item before writing code for a step",
                "tool": "memory.find",
                "permission": "library copy rw",
            },
            {
                "expectation": "read an item in full",
                "tool": "memory.show",
                "permission": "library copy rw",
            },
        ],
        "fields": ["memory_use"],
        "emits": [],
        "pointers": [{"pointer": "library index", "tool": "memory.index"}],
    },
    {
        "agent": "fork",
        "mode": "fork",
        "instructions": "unify.memory_v2.integration.fork:request_doc",
        # the one allowed told-about-but-not-usable exception: the actor's tool schemas stay in the request only to
        # keep the cached prefix; no call is executed, and the instruction says to write blocks instead
        "tools": ["<the actor's tool schemas, not executed>"],
        "tools_executed": False,
        "exceptions": ["told_about_not_usable: actor tools"],
        "fixed_reply": _fork_tool_reply(),
        "mounts": [
            {"path": "/usr", "mode": "ro"},
            {"path": "/s", "mode": "ro"},
            {"path": "/staging", "mode": "rw"},
        ],
        "denied": [
            "tool execution",
            "home",
            "state dir",
            "library",
            "network except the fork lane",
        ],
        "consumes": [{"artifact": "conversation", "path": "<request conversation>"}],
        "produces": [
            {
                "artifact": "staging",
                "path": "<state>/memory-staging/<episode>/",
                "consumer": "writer",
            },
        ],
        "limits": [
            {"limit": "turns", "value": 16, "unit": "turns"},
            {"limit": "call time", "value": 900, "unit": "s"},
            {"limit": "staged bytes per episode", "value": 4194304, "unit": "bytes"},
            {"limit": "staged files per episode", "value": 200, "unit": "files"},
        ],
        "inlined": [{"what": "fork instruction (constant)", "bound": _INLINE_BOUND}],
        "capabilities": ["write fenced file blocks in its reply"],
        "absent_paths": ["/inputs", "/memory", "<state>"],
        "expectations": [
            {
                "expectation": "write candidates, cases and notes as file blocks",
                "tool": "reply text",
                "permission": "/staging rw (worker)",
            },
        ],
        "fields": [
            "candidates/<name>.py",
            "cases/<name>.json",
            "notes.md",
            "fork.json",
        ],
        "emits": [
            {
                "event": "fork blocks",
                "path": "memory-staging/<episode>/fork-blocks.json",
            },
        ],
        "pointers": [],
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
            "similar",
            "pool",
            "fixture",
            "replay",
            "delegate",
            "execute_code",
            "check",
            "finish",
        ],
        "mounts": list(_BOX_MOUNTS),
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
            {
                "artifact": "manifest",
                "path": "/memory/.pass/manifest.json",
                "consumer": "gate",
            },
            {
                "artifact": "fixture listing",
                "path": "/memory/.pass/fixtures.json",
                "consumer": "gate",
            },
        ],
        "limits": list(_PASS_LIMITS),
        "inlined": [
            {"what": "brief (constant)", "bound": _INLINE_BOUND},
            {
                "what": "first message: pass id, batch episode ids, wall clock, pointers",
                "bound": _INLINE_BOUND,
            },
        ],
        "capabilities": [
            "read any recorded episode read-only",
            "run Python in a confined box",
            "write the library copy",
            "record pool decisions in the manifest",
        ],
        "absent_paths": ["/memory/.git"],
        "expectations": [
            {
                "expectation": "read the batch",
                "tool": "overview",
                "permission": "/inputs ro",
            },
            {
                "expectation": "read the episodes behind a proposal",
                "tool": "read_episode",
                "permission": "recorded episodes ro",
            },
            {
                "expectation": "read staged proposals",
                "tool": "read",
                "permission": "/inputs ro",
            },
            {
                "expectation": "decide on pool entries",
                "tool": "pool",
                "permission": "/memory rw",
            },
            {
                "expectation": "make fixtures from recorded values",
                "tool": "fixture",
                "permission": "/memory rw",
            },
            {
                "expectation": "write and run tests",
                "tool": "execute_code",
                "permission": "/memory rw",
            },
            {
                "expectation": "check before finishing",
                "tool": "check",
                "permission": "/memory rw",
            },
            {
                "expectation": "end the pass",
                "tool": "finish",
                "permission": "/memory rw",
            },
        ],
        "fields": [
            "manifest.items",
            "manifest.support",
            "manifest.tests_changed",
            "manifest.why",
            "manifest.summary",
            "manifest.adopts",
            "manifest.pool",
        ],
        "emits": list(_EVENTS),
        "pointers": [
            {"pointer": "/inputs/batch_map.json", "tool": "overview"},
            {"pointer": "recorded episodes", "tool": "read_episode"},
            {"pointer": "/inputs/pool/INDEX.md", "tool": "pool"},
            {"pointer": "/inputs/staging/<episode>/", "tool": "read"},
            {"pointer": "/memory/INDEX.md", "tool": "read"},
            {"pointer": "/inputs/gate/previous.md", "tool": "read"},
            {"pointer": "/outputs/cell-<n>.txt", "tool": "read"},
        ],
    },
    {
        "agent": "curator",
        "mode": "curate",
        "instructions": "unify.memory_v2.library_prompts:curate_brief_now",
        "tools": [
            "read",
            "grep",
            "overview",
            "read_episode",
            "fixture",
            "execute_code",
            "check",
            "finish",
        ],
        "mounts": list(_BOX_MOUNTS),
        "denied": [
            "pool",
            "similar",
            "replay",
            "delegate",
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
            {
                "artifact": "manifest",
                "path": "/memory/.pass/manifest.json",
                "consumer": "gate",
            },
        ],
        "limits": list(_PASS_LIMITS),
        "inlined": [{"what": "brief (constant)", "bound": _INLINE_BOUND}],
        "capabilities": [
            "read any recorded episode read-only",
            "run Python in a confined box",
            "write the library copy",
        ],
        "absent_paths": [
            "/inputs/staging",
            "/inputs/pool",
            "/inputs/batch_map.json",
            "/memory/.git",
        ],
        "expectations": [
            {
                "expectation": "read why the pass runs",
                "tool": "read",
                "permission": "/inputs ro",
            },
            {
                "expectation": "read episodes cited by suspects",
                "tool": "read_episode",
                "permission": "recorded episodes ro",
            },
            {
                "expectation": "make a recorded value a fixture",
                "tool": "fixture",
                "permission": "/memory rw",
            },
            {
                "expectation": "merge, repair or retire items",
                "tool": "execute_code",
                "permission": "/memory rw",
            },
            {
                "expectation": "check before finishing",
                "tool": "check",
                "permission": "/memory rw",
            },
            {
                "expectation": "end the pass",
                "tool": "finish",
                "permission": "/memory rw",
            },
        ],
        "fields": [
            "manifest.items",
            "manifest.deleted",
            "manifest.retired",
            "manifest.deleted_tests",
            "manifest.aliases",
            "manifest.tests_changed",
            "manifest.why",
            "manifest.summary",
        ],
        "emits": list(_EVENTS),
        "pointers": [
            {"pointer": "/inputs/curate/trigger.json", "tool": "read"},
            {"pointer": "recorded episodes", "tool": "read_episode"},
            {"pointer": "an episode's end and steps", "tool": "overview"},
            {"pointer": "/memory/INDEX.md", "tool": "read"},
        ],
    },
    {
        "agent": "helper",
        "mode": "helper",
        "instructions": "unify.memory_v2.analysts:run",
        # its own execute_code (in a box with /memory read-only and a private /work) and stage/done come with the
        # helper's own specs
        "tools": [
            "read",
            "grep",
            "overview",
            "read_episode",
            "similar",
            "execute_code",
            "stage",
            "done",
        ],
        "mounts": [
            {"path": "/inputs", "mode": "ro"},
            {"path": "/memory", "mode": "ro"},
            {"path": "/work", "mode": "rw"},
        ],
        # a helper never launches helpers: delegate is the writer's alone
        "denied": [
            "library writes",
            "pool",
            "writer staging",
            "delegate",
            "network",
            "credentials",
        ],
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
        "limits": list(_PASS_LIMITS),
        "inlined": [
            {
                "what": "helper brief and task (constant per call)",
                "bound": _INLINE_BOUND,
            },
        ],
        "capabilities": [
            "read any recorded episode read-only",
            "run Python in its own box",
        ],
        "absent_paths": ["/inputs/pool", "/memory/.pass"],
        "expectations": [
            {
                "expectation": "read the named episodes",
                "tool": "read_episode",
                "permission": "recorded episodes ro",
            },
            {
                "expectation": "try a draft on its cases",
                "tool": "execute_code",
                "permission": "/work rw",
            },
            {
                "expectation": "name the episode parts each claim rests on",
                "tool": "done",
                "permission": "recorded episodes ro",
            },
        ],
        "fields": [
            "candidates/<name>.py",
            "cases/<name>.json",
            "notes.md",
            "helper.json",
        ],
        "emits": list(_EVENTS),
        "pointers": [{"pointer": "recorded episodes", "tool": "read_episode"}],
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
            {
                "artifact": "gate result",
                "path": "/inputs/gate/result-<round>.md",
                "consumer": "writer",
            },
            {"artifact": "library item", "path": "library main", "consumer": "actor"},
        ],
        "limits": [{"limit": "time left for a check", "value": 60, "unit": "s"}],
        "inlined": [
            {"what": "repair message (names the result file)", "bound": _INLINE_BOUND},
        ],
        "capabilities": ["judge a manifest and candidate", "merge admitted items"],
        "absent_paths": [],
        "expectations": [],
        "fields": ["manifest.fixtures", "manifest.adopts", "manifest.pool"],
        "emits": [],
        # it names its result file to the writer, who reads it with read
        "pointers": [],
    },
)


def profile(agent: str) -> dict:
    for row in AGENT_PROFILES:
        if row["agent"] == agent:
            return row
    raise KeyError(f"no agent profile {agent!r}")


def tool_names(agent: str, flags: frozenset[str] = frozenset()) -> list[str]:
    """The tools *agent* gets with the configuration *flags* on, in the spec's order: the spec's tools for its role
    that its profile names (a tool the spec gates on a flag is left out while the flag is off).
    """
    row = profile(agent)
    role = SPEC_ROLE.get(agent)
    if role is None:
        return []
    named = set(row["tools"])
    return [s.name for s in _tool_spec.tools_for(role, flags) if s.name in named]


def tool_json(agent: str, flags: frozenset[str] = frozenset()) -> list[dict]:
    names = set(tool_names(agent, flags))
    return [
        s.as_tool()
        for s in _tool_spec.tools_for(SPEC_ROLE[agent], flags)
        if s.name in names
    ]
