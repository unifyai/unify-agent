"""One definition per memory tool: its name, description, parameter schema, the agents that get it, and its refusal
texts. The tool JSON each agent receives, the Tools section of its instructions and the refusal texts are all
rendered from here, so they cannot disagree.

Modules that implement a tool define its spec next to its handler (``SPEC = ToolSpec(...)``) and are listed in
:data:`SPEC_MODULES`; :func:`registry` collects them in a fixed order.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass, field

#: The agents a tool can be given to (see agent_profiles.py).
ROLES = ("write", "curate", "helper")


@dataclass(frozen=True)
class ToolSpec:
    name: str
    #: what the tool does, as the agent reads it (stable text: no dynamic lists)
    description: str
    #: JSON schema of the arguments (an object schema)
    parameters: dict
    #: the agents that get the tool
    roles: frozenset[str]
    #: refusal texts by stable form code; ``{name}`` fields are filled by :func:`refusal`
    refusals: dict[str, str] = field(default_factory=dict)
    #: a configuration flag that must be on for the tool to exist (e.g. "pool", "fork"); None: always
    requires: str | None = None
    #: one line for the Tools section of the instructions (defaults to the description's first sentence)
    summary: str = ""

    def as_tool(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


#: Modules holding ``SPEC`` (or ``SPECS``, a tuple) for their tools, in the order the tools are listed.
SPEC_MODULES: tuple[str, ...] = (".writer_tools",)


def registry() -> tuple[ToolSpec, ...]:
    specs: list[ToolSpec] = []
    for mod in SPEC_MODULES:
        m = importlib.import_module(mod, __package__)
        found = getattr(m, "SPECS", None) or (getattr(m, "SPEC"),)
        specs.extend(found)
    names = [s.name for s in specs]
    if len(set(names)) != len(names):
        raise ValueError(
            f"duplicate tool names: {sorted(n for n in names if names.count(n) > 1)}",
        )
    return tuple(specs)


def tools_for(role: str, flags: frozenset[str] = frozenset()) -> list[ToolSpec]:
    """The tools *role* gets with the configuration *flags* on."""
    if role not in ROLES:
        raise ValueError(f"unknown role {role!r}")
    return [
        s
        for s in registry()
        if role in s.roles and (s.requires is None or s.requires in flags)
    ]


def tool_json(role: str, flags: frozenset[str] = frozenset()) -> list[dict]:
    return [s.as_tool() for s in tools_for(role, flags)]


def tools_section(role: str, flags: frozenset[str] = frozenset()) -> str:
    """The Tools section of *role*'s instructions: one bullet per tool it actually gets."""
    lines = []
    for s in tools_for(role, flags):
        text = s.summary or s.description.split(". ")[0].rstrip(".") + "."
        lines.append(f"- `{s.name}`: {text}")
    return "\n".join(lines)


def refusal(tool: str, form: str, **fields: object) -> str:
    """The refusal text *form* of *tool*, filled; starts with "refused: " so every refusal reads the same."""
    spec = next((s for s in registry() if s.name == tool), None)
    if spec is None or form not in spec.refusals:
        raise KeyError(f"{tool}: no refusal form {form!r}")
    return "refused: " + spec.refusals[form].format(**fields)
