"""The agents' profiles: one row per agent, binding it to its tools, its box mounts (with the mode the box really
gives them), what it is denied, and what it consumes and produces.

Rows use role names only; every model is a setting. Each tool-using row names its tools three ways: ``tools``, the
model-API tools it was given before every agent had one (the memory agents' earlier surface); ``json_tools``, its one
model-API tool with ``UNIFY_TOOLS_IN_CODE=on`` (``execute_code``); and ``in_code``, the tools it may call from its
cells then (the allow-list the host enforces, :func:`unify.common.tool_registry.allow_list`). The tools themselves
come from the shared registry (:mod:`unify.common.tool_registry`), so a profile can only name tools that exist.
"""

from __future__ import annotations

from . import tool_spec as _tool_spec
from .example_calls import CALLS_DIR, CALLS_OUT

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
#: A helper's limits: its own /work and cell time, its staging quota, the writer pass's step guard and wall clock.
_HELPER_LIMITS = [
    {"limit": "cell time", "value": 60, "unit": "s"},
    {"limit": "files under /work", "value": 2000, "unit": "files"},
    {"limit": "file under /work", "value": 2097152, "unit": "bytes"},
    {"limit": "all files under /work", "value": 20971520, "unit": "bytes"},
    {"limit": "staged files", "value": 200, "unit": "files"},
    {"limit": "staged bytes", "value": 4194304, "unit": "bytes"},
    {"limit": "step guard", "value": 500, "unit": "model calls"},
    {
        "limit": "pass wall clock",
        "value": "UNIFY_MEMORY_LIBRARY_PASS_WALL_S",
        "unit": "s",
    },
]
#: What any memory prompt may inline: constant text, ids and numbers; never a list past these bounds.
_INLINE_BOUND = {"items": 10, "bytes": 1000, "tokens": 250}
_EVENTS = [{"event": "tool_call", "path": "<pass output dir>/tool-events.jsonl"}]
#: The pass's own agent also writes the pass's end (each helper: ended or still running) and keeps the record.
_PASS_EVENTS = [
    *_EVENTS,
    {"event": "pass_end", "path": "<pass output dir>/tool-events.jsonl"},
    {"event": "record entry", "path": "<pass output dir>/record.jsonl"},
]
#: Every agent on a pass's record reads it and posts to it (messaging.py: one in-memory record per pass).
_RECORD = "the pass's record (send, messages, wait, who)"
_RECORD_TOOLS = ["send", "messages", "wait", "who"]
#: The actor and its teammates share the run's agent record (unify/agents/record.py).
_AGENT_RECORD = "the agent record (send, messages, wait, who)"
#: Every tool-using agent's one model-API tool with UNIFY_TOOLS_IN_CODE=on.
_JSON_TOOLS = ["execute_code"]
#: The actor's and a teammate's in-code tools (store_skills only where the actor may store).
_ACTOR_IN_CODE = [
    "read",
    "grep",
    "install",
    *_RECORD_TOOLS,
    "reply",
    "stop",
    "delegate",
    "finish",
    "store_skills",
    # only with UNIFY_MEMORY_LIBRARIAN=on (registry flag "librarian")
    "ask_library",
    "sh",
]
_TEAMMATE_IN_CODE = [
    "read",
    "grep",
    "install",
    *_RECORD_TOOLS,
    "reply",
    "stop",
    "ask",
    "delegate",
    "finish",
    "sh",
]
_BOX_MOUNTS = [
    {"path": "/inputs", "mode": "ro"},
    {"path": "/memory", "mode": "rw"},
    {"path": "/memory/INDEX.md", "mode": "ro"},
    {"path": "/memory/links.json", "mode": "ro"},
    {"path": "/memory/memory/__init__.py", "mode": "ro"},
    # host-side only: `read` opens the saved cell outputs; cells cannot open /outputs
    {"path": "/outputs", "mode": "ro", "via": "read"},
]


#: The record of the calls a function's tests make, written by the observe-only plugin in the gate's test box while
#: the gate runs the tests the writer or curator wrote. Gate-internal: no writer-visible artifact.
_CALL_RECORD = {"artifact": "call record", "path": CALLS_OUT}
#: The gate's test box mounts the plugin read-only and a fresh directory for the record read-write.
_CALL_RECORD_MOUNTS = [
    {"path": CALLS_DIR, "mode": "ro"},
    {"path": CALLS_OUT.rsplit("/", 1)[0], "mode": "rw"},
]


def _fork_tool_reply() -> str:
    """The one reply every tool call of the fork gets (it writes blocks; no tool runs)."""
    from .integration.fork_worker import TOOL_REPLY

    return TOOL_REPLY


AGENT_PROFILES: tuple[dict, ...] = (
    {
        "agent": "actor",
        "mode": "consumer",
        "instructions": "unify.memory.library_prompts:guide_library_mode",
        # memory.index / memory.show / memory.find are library functions called in cells, not tools
        "tools": [],
        "json_tools": list(_JSON_TOOLS),
        "in_code": list(_ACTOR_IN_CODE),
        # what its guide names that it does in its own cells, through no tool
        "in_cells": ["memory.index", "memory.show", "memory.find"],
        "mounts": [{"path": "<library scratch copy>", "mode": "rw"}],
        "denied": [
            "/inputs",
            "staging",
            "pool",
            "other episodes",
            "library repository",
        ],
        "consumes": [
            {"artifact": "library export", "path": "<library scratch copy>"},
            {"artifact": "agent record", "path": _AGENT_RECORD},
            # UNIFY_MEMORY_LIBRARIAN=on: what memory.ask returns
            {"artifact": "librarian answer", "path": "memory.ask"},
        ],
        "produces": [
            {"artifact": "agent record", "path": _AGENT_RECORD, "consumer": "teammate"},
            {"artifact": "question", "path": "memory.ask (tools.ask_library)", "consumer": "librarian"},
            {
                "artifact": "episode record",
                "path": "<state>/episodes/<episode>",
                "consumer": "writer",
            },
            {
                "artifact": "episode record",
                "path": "<state>/episodes/<episode>",
                "consumer": "curator",
            },
            {
                "artifact": "episode record",
                "path": "<state>/episodes/<episode>",
                "consumer": "helper",
            },
            {
                "artifact": "episode record",
                "path": "<state>/episodes/<episode>",
                "consumer": "harness",
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
        "agent": "teammate",
        "mode": "teammate",
        "instructions": "unify.agents.binding:helper_request",
        # the actor's own loop and box (a teammate is an actor started on the agent record)
        "tools": [],
        "json_tools": list(_JSON_TOOLS),
        "in_code": list(_TEAMMATE_IN_CODE),
        "in_cells": ["memory.index", "memory.show", "memory.find"],
        "mounts": [{"path": "<library scratch copy>", "mode": "rw"}],
        "denied": ["store_skills", "/inputs", "library repository"],
        "consumes": [{"artifact": "agent record", "path": _AGENT_RECORD}],
        "produces": [{"artifact": "agent record", "path": _AGENT_RECORD, "consumer": "actor"}],
        "limits": [],
        "inlined": [{"what": "its task, as the record entry that started it", "bound": _INLINE_BOUND}],
        "capabilities": ["work on one part of the actor's request in its own cells"],
        "absent_paths": ["/inputs"],
        "expectations": [],
        "fields": [],
        "emits": [],
        "pointers": [],
    },
    {
        "agent": "fork",
        "mode": "fork",
        "instructions": "unify.memory.integration.fork:request_doc",
        # the one allowed told-about-but-not-usable exception: the actor's tool schemas stay in the request only to
        # keep the cached prefix; no call is executed, and the instruction says to write blocks instead
        "tools": ["<the actor's tool schemas, not executed>"],
        "tools_executed": False,
        "exceptions": ["told_about_not_usable: actor tools", "inlined_list: the episode's part list"],
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
                "consumer": "harness",
            },
        ],
        "limits": [
            {"limit": "turns", "value": 16, "unit": "turns"},
            {"limit": "call time", "value": 900, "unit": "s"},
            {"limit": "staged bytes per episode", "value": 4194304, "unit": "bytes"},
            {"limit": "staged files per episode", "value": 200, "unit": "files"},
        ],
        "inlined": [
            {"what": "fork instruction (constant)", "bound": _INLINE_BOUND},
            # the one list that grows with its input: the fork reads no files, so its episode's part names (each
            # with that part's first line, at most 80 characters) follow the instruction, after the cached prefix
            {"what": "the recorded episode's part list", "bound": {"items": "one per recorded part"}},
        ],
        "capabilities": ["write fenced file blocks in its reply"],
        "absent_paths": ["/inputs", "/memory", "<state>"],
        "expectations": [
            {
                "expectation": "write a summary, candidates, cases and notes as file blocks",
                "tool": "reply text",
                "permission": "/staging rw (worker)",
            },
        ],
        "fields": [
            "summary.md",
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
        "instructions": "unify.memory.library_prompts:write_brief_now",
        "tools": [
            "read",
            "grep",
            "overview",
            "read_episode",
            "similar",
            "pool",
            "fixture",
            "replay",
            "land_capture",
            "delegate",
            *_RECORD_TOOLS,
            "reply",
            "stop",
            # write and edit exist only with UNIFY_MEMORY_LIBRARY_EDIT_TOOLS=on (spec flag "edit")
            "write",
            "edit",
            "execute_code",
            "check",
            "finish",
        ],
        "json_tools": list(_JSON_TOOLS),
        "in_code": [
            "read",
            "grep",
            "overview",
            "read_episode",
            "similar",
            "pool",
            "fixture",
            "replay",
            "land_capture",
            "delegate",
            *_RECORD_TOOLS,
            "reply",
            "stop",
            "write",
            "edit",
            "check",
            "finish",
            "sh",
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
            {"artifact": "helper output", "path": "/inputs/helpers"},
            {"artifact": "record", "path": _RECORD},
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
            # its tests, run by the gate, give the call record
            {**_CALL_RECORD, "consumer": "gate"},
            {"artifact": "task", "path": "delegate arguments", "consumer": "helper"},
            {"artifact": "record", "path": _RECORD, "consumer": "helper"},
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
        "emits": list(_PASS_EVENTS),
        "pointers": [
            {"pointer": "/inputs/batch_map.json", "tool": "overview"},
            {"pointer": "recorded episodes", "tool": "read_episode"},
            {"pointer": "/inputs/pool/INDEX.md", "tool": "pool"},
            {"pointer": "/inputs/staging/<episode>/", "tool": "read"},
            {"pointer": "/memory/INDEX.md", "tool": "read"},
            {"pointer": "/inputs/gate/previous.md", "tool": "read"},
            {"pointer": "/outputs/cell-<n>.txt", "tool": "read"},
            {"pointer": "agents on the record", "tool": "who"},
        ],
    },
    {
        "agent": "curator",
        "mode": "curate",
        "instructions": "unify.memory.library_prompts:curate_brief_now",
        "tools": [
            "read",
            "grep",
            "overview",
            "read_episode",
            "fixture",
            *_RECORD_TOOLS,
            "reply",
            "stop",
            # write and edit exist only with UNIFY_MEMORY_LIBRARY_EDIT_TOOLS=on (spec flag "edit")
            "write",
            "edit",
            "execute_code",
            "check",
            "finish",
        ],
        "json_tools": list(_JSON_TOOLS),
        "in_code": [
            "read",
            "grep",
            "overview",
            "read_episode",
            "fixture",
            *_RECORD_TOOLS,
            "reply",
            "stop",
            "write",
            "edit",
            "check",
            "finish",
            "sh",
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
            {"artifact": "record", "path": _RECORD},
        ],
        "produces": [
            {"artifact": "library item", "path": "/memory/memory/", "consumer": "gate"},
            {
                "artifact": "manifest",
                "path": "/memory/.pass/manifest.json",
                "consumer": "gate",
            },
            {"artifact": "record", "path": _RECORD, "consumer": "helper"},
            # its tests, run by the gate, give the call record
            {**_CALL_RECORD, "consumer": "gate"},
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
                "expectation": "read episodes cited by retired items",
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
        "emits": list(_PASS_EVENTS),
        "pointers": [
            {"pointer": "/inputs/curate/trigger.json", "tool": "read"},
            {"pointer": "recorded episodes", "tool": "read_episode"},
            {"pointer": "an episode's end and steps", "tool": "overview"},
            {"pointer": "/memory/INDEX.md", "tool": "read"},
            {"pointer": "agents on the record", "tool": "who"},
        ],
    },
    {
        "agent": "helper",
        "mode": "helper",
        "instructions": "unify.memory.helpers:helper_brief",
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
            *_RECORD_TOOLS,
            "ask",
            "done",
        ],
        "json_tools": list(_JSON_TOOLS),
        "in_code": [
            "read",
            "grep",
            "overview",
            "read_episode",
            "similar",
            "stage",
            *_RECORD_TOOLS,
            "ask",
            "done",
            "sh",
        ],
        "mounts": [
            # the writer's own /inputs (staging, pool, batch map, drafts, gate results, other helpers' output)
            {"path": "/inputs", "mode": "ro"},
            {"path": "/task", "mode": "ro"},
            {"path": "/memory", "mode": "ro"},
            {"path": "/work", "mode": "rw"},
        ],
        # a helper never launches helpers: delegate is the writer's alone
        "denied": [
            "library writes",
            "writes under /inputs",
            "delegate",
            "network",
            "credentials",
        ],
        "consumes": [
            {"artifact": "task", "path": "delegate arguments"},
            {"artifact": "episode record", "path": "read_episode"},
            {"artifact": "record", "path": _RECORD},
        ],
        "produces": [
            {
                "artifact": "helper output",
                "path": "<state>/memory-helpers/<pass>/<name>/ (the writer's /inputs/helpers/<name>/)",
                "consumer": "writer",
            },
            # helper to helper included: any agent on the record can address any other
            {"artifact": "record", "path": _RECORD, "consumer": "writer"},
            {"artifact": "record", "path": _RECORD, "consumer": "curator"},
            {"artifact": "record", "path": _RECORD, "consumer": "helper"},
        ],
        # the helper's own limits, as its brief and first message state them (helpers.py constants)
        "limits": list(_HELPER_LIMITS),
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
        "absent_paths": ["/memory/.pass"],
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
        "pointers": [
            {"pointer": "recorded episodes", "tool": "read_episode"},
            {"pointer": "agents on the record", "tool": "who"},
        ],
    },
    {
        # UNIFY_MEMORY_LIBRARIAN=on: one session per actor request, on the actor's model and effort, answering which
        # stored items fit the actor's questions (memory.ask); its system prompt holds the whole catalogue
        "agent": "librarian",
        "mode": "librarian",
        "instructions": "unify.memory.librarian:system_prompt",
        "tools": [],
        "json_tools": list(_JSON_TOOLS),
        "in_code": ["finish"],
        "in_cells": ["memory.show"],
        # the actor's library copy, read-only; no write root, no network
        "mounts": [{"path": "/memory", "mode": "ro"}],
        "denied": ["network", "credentials", "writes", "library repository", "episodes", "state dir"],
        "consumes": [
            {"artifact": "library export", "path": "/memory"},
            {"artifact": "question", "path": "the request's next user message"},
        ],
        "produces": [{"artifact": "librarian answer", "path": "tools.finish(answer)", "consumer": "actor"}],
        "limits": [{"limit": "cell time", "value": 60, "unit": "s"}],
        "inlined": [{"what": "the brief and the library's catalogue (constant per library commit)", "bound": {}}],
        "capabilities": ["read the library copy read-only", "run Python in its own box"],
        "absent_paths": ["/inputs", "/workspace", "<state>"],
        "expectations": [
            {"expectation": "open an item's source before answering", "tool": "memory.show", "permission": "/memory ro"},
        ],
        "fields": ["ask_calls", "ask_rows", "listed_by_ask"],
        "emits": [],
        "pointers": [{"pointer": "an item's source", "tool": "memory.show"}],
    },
    {
        "agent": "gate",
        "mode": "repair",
        "instructions": "unify.memory.repair:repair_message",
        "tools": [],
        # it reads the candidate library copy, its manifest and fixture listing under /memory/.pass
        "mounts": [{"path": "/memory", "mode": "ro"}, *_CALL_RECORD_MOUNTS],
        "denied": ["network"],
        "consumes": [
            {"artifact": "library item", "path": "/memory/memory/"},
            {"artifact": "manifest", "path": "/memory/.pass/manifest.json"},
            {"artifact": "fixture listing", "path": "/memory/.pass/fixtures.json"},
            dict(_CALL_RECORD),
        ],
        "produces": [
            {
                "artifact": "gate result",
                "path": "/inputs/gate/result-<round>.md",
                "consumer": "writer",
            },
            {
                "artifact": "gate result",
                "path": "/inputs/gate/result-<round>.md",
                "consumer": "curator",
            },
            {
                "artifact": "library export",
                "path": "library main, copied to <library scratch copy>",
                "consumer": "actor",
            },
            {
                "artifact": "library export",
                "path": "library main, copied to <library scratch copy>",
                "consumer": "librarian",
            },
        ],
        "limits": [
            {"limit": "time left for a check", "value": 60, "unit": "s"},
            # only in a pass with a spend cap; an uncapped pass has no set number (repair.may_repair)
            {"limit": "repair rounds (capped pass)", "value": 2, "unit": "rounds"},
        ],
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
    {
        "agent": "harness",
        "mode": "inputs",
        "instructions": "unify.memory.writer_pass:WriterPass._stage_inputs",
        "tools": [],
        "mounts": [{"path": "/inputs", "mode": "rw"}],
        "denied": ["network"],
        "consumes": [
            {"artifact": "episode record", "path": "<state>/episodes/<episode>"},
            {"artifact": "staging", "path": "<state>/memory-staging/<episode>/"},
        ],
        "produces": [
            {
                "artifact": "batch map",
                "path": "/inputs/batch_map.json",
                "consumer": "writer",
            },
            {"artifact": "staging", "path": "/inputs/staging", "consumer": "writer"},
            {"artifact": "pool", "path": "/inputs/pool", "consumer": "writer"},
            {"artifact": "drafts", "path": "/inputs/drafts", "consumer": "writer"},
            {
                "artifact": "curate inputs",
                "path": "/inputs/curate",
                "consumer": "curator",
            },
        ],
        "limits": [],
        "inlined": [],
        "capabilities": ["stage each pass's inputs read-only for its agent"],
        "absent_paths": [],
        "expectations": [],
        "fields": [],
        "emits": [],
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


def tools_in_code() -> bool:
    """Whether ``UNIFY_TOOLS_IN_CODE=on``: every agent keeps one model-API tool and calls the rest in code."""
    from unify.settings import SETTINGS

    return getattr(SETTINGS, "UNIFY_TOOLS_IN_CODE", "off") == "on"


def tool_json(
    agent: str,
    flags: frozenset[str] = frozenset(),
    in_code: bool | None = None,
) -> list[dict]:
    """*agent*'s model-API tools: with the tools in code (*in_code*, by default the setting), ``execute_code``
    alone, the same bytes for every agent; otherwise the tools it was given before, byte for byte."""
    if tools_in_code() if in_code is None else in_code:
        from unify.common import tool_registry

        return tool_registry.json_tools(agent)
    names = set(tool_names(agent, flags))
    return [
        s.as_tool()
        for s in _tool_spec.tools_for(SPEC_ROLE[agent], flags)
        if s.name in names
    ]


def in_code_names(agent: str, flags: frozenset[str] = frozenset()) -> tuple[str, ...]:
    """The tools *agent* may call from its cells with the configuration *flags* on (the host's allow-list)."""
    from unify.common import tool_registry

    return tool_registry.allow_list(agent, flags)


# --- the opening a memory agent reads first when tools are called from code ---

_MODE_TEXT = {"ro": "read-only", "rw": "writable"}

#: Where each memory agent reads the library: one line, box paths only.
_LIBRARY_LINE = {
    "writer": (
        "The library is the Python package `memory` under /memory, bound as `memory` in your cells: "
        '`memory.index()`, `memory.show("<id>")` and `memory.find(value)`.'
    ),
    "curator": (
        "The library is the Python package `memory` under /memory, bound as `memory` in your cells: "
        '`memory.index()`, `memory.show("<id>")` and `memory.find(value)`.'
    ),
    "helper": "/memory holds the library as the writer sees it, read-only; it is the package `memory` in your cells.",
    "librarian": (
        "The library is the Python package `memory` under /memory, read-only, already bound as `memory` in every "
        'cell: `memory.show("<id>")` prints an item\'s source, docstring and records.'
    ),
}


def _number(value: float) -> str:
    return f"{value:g}" if isinstance(value, float) else str(value)


def _root_text(mount: dict) -> str:
    mode = _MODE_TEXT.get(mount["mode"], mount["mode"])
    if mount.get("via"):
        return f"{mount['path']} ({mode}, through tools.{mount['via']} only; cells cannot open it)"
    return f"{mount['path']} ({mode})"


def _limit_text(row: dict, cell_s: float | None) -> str:
    value = cell_s if row["limit"] == "cell time" and cell_s is not None else row["value"]
    if isinstance(value, str):
        # a setting's name: its value for this pass is in the first message
        return f"{row['limit']}: stated in the first message"
    return f"{row['limit']}: {_number(value)} {row['unit']}"


def _output_bytes(agent: str) -> int:
    """The head of a cell's output the agent is shown (the constant that cuts it)."""
    if agent == "helper":
        from .helpers import OUTPUT_BYTES

        return OUTPUT_BYTES
    from .writer_pass import _OUTPUT_CAP

    return _OUTPUT_CAP


def environment_text(agent: str, cell_s: float | None = None) -> str:
    """The memory agent's box roots with their modes and its limits, from its profile row (box paths only);
    *cell_s* is the pass's own cell time where it is set."""
    row = profile(agent)
    roots = "; ".join(_root_text(m) for m in row["mounts"])
    limits = [_limit_text(r, cell_s) for r in row["limits"]]
    limits.append(f"cell output shown: the first {_output_bytes(agent)} bytes")
    return f"Your box: {roots}.\nLimits: " + "; ".join(limits) + "."


def in_code_opening(agent: str, flags: frozenset[str] = frozenset(), cell_s: float | None = None) -> str:
    """The shared opening (:func:`unify.common.tool_registry.preamble`) of the writer, curator or a helper when tools
    are called from code: its box and limits, its library line, the tools its allow-list has with *flags* on, whether
    variables are kept between cells (``UNIFY_CELL_STATE``) and the names bound in its cells (``tools`` and
    ``memory``: the box holds the library package in library mode, the only mode with tools in code). Constant for a
    run, so the system message stays a stable prefix."""
    from unify.common import tool_registry
    from unify.common.box_interpreter import persistent

    return tool_registry.preamble(
        agent,
        {
            "environment": environment_text(agent, cell_s),
            "memory": _LIBRARY_LINE[agent],
            "flags": frozenset(flags),
            "persistent": persistent(),
            "names": tool_registry.namespace_names(agent, {"tools": True, "memory": True}),
        },
    )
