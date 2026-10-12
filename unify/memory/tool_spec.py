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
SPEC_MODULES: tuple[str, ...] = (".writer_tools", ".similar", ".helpers")


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


#: Answers the pass loop gives on any tool (limits, unknown tools, malformed calls), by stable form code. A text
#: starting with "not run: " is a call that was not run (it can be sent again); any other is a refusal.
#: How every refusal text that is not a "not run" answer starts.
REFUSED = "refused: "

GENERAL_REFUSALS: dict[str, str] = {
    "unknown_tool": "unknown tool {name!r}; your tools: {tools}",
    "bad_arguments": "{reason}",
    "readers_per_turn": "not run: at most {limit} reader calls run per model turn; send it again",
    "cells_per_turn": "not run: at most {limit} cells run per model turn; send it again",
    "reads_limit": "not run: the pass's reader-call limit ({limit}) is reached",
    "stopped": "not run: {reason}",
    "pass_finished": "not run: pass finished",
    "task_done": "not run: task done",
    "check_limit": "not run: at most {limit} check calls per pass",
    "call_limit": "not run: the pass's call cap is reached",
}

#: Starts a template keeps as they are: a call that was not run (send it again) and a finish that did not end the
#: pass. Every other refusal text starts with "refused: ".
KEPT_STARTS = ("not run: ", "not finished: ")

#: The key :func:`refusal_codes` lists the general forms under.
GENERAL = "*"


class Refusal(str):
    """A refusal text that keeps the tool and the stable form code it was made from, so a tool event records the
    code itself (never a code guessed back from the text). Slicing or joining gives a plain ``str``.
    """

    tool: str
    form: str

    def __new__(cls, text: str, tool: str, form: str) -> "Refusal":
        obj = super().__new__(cls, text)
        obj.tool, obj.form = tool, form
        return obj

    def __getnewargs__(self) -> tuple[str, str, str]:
        # copy.deepcopy and pickle rebuild the object through __new__, which needs the tool and the form too
        return (str(self), self.tool, self.form)


def _variants() -> list[ToolSpec]:
    """Specs that are not in the shared registry but answer as a registered tool name for one agent (a module's
    ``VARIANT_SPECS``, e.g. the helper's own execute_code)."""
    out: list[ToolSpec] = []
    for mod in SPEC_MODULES:
        m = importlib.import_module(mod, __package__)
        out.extend(getattr(m, "VARIANT_SPECS", ()))
    return out


def refusal_codes() -> dict[str, list[str]]:
    """Every refusal form code by tool (registered specs and their variants), and the general forms under
    :data:`GENERAL`. A tool event's ``refusal.form`` is always one of these."""
    out: dict[str, list[str]] = {}
    for s in (*registry(), *_variants()):
        out.setdefault(s.name, [])
        out[s.name].extend(f for f in s.refusals if f not in out[s.name])
    out[GENERAL] = list(GENERAL_REFUSALS)
    return {k: sorted(v) for k, v in out.items()}


def refusal(tool: str, form: str, **fields: object) -> Refusal:
    """The refusal text *form* of *tool* (its spec's ``refusals``, then the general forms), filled. It starts with
    "refused: " so every refusal reads the same, except a template starting with one of :data:`KEPT_STARTS`.
    """
    template = None
    for s in (*registry(), *_variants()):
        if s.name == tool and form in s.refusals:
            template = s.refusals[form]
            break
    if template is None:
        template = GENERAL_REFUSALS.get(form)
    if template is None:
        raise KeyError(f"{tool}: no refusal form {form!r}")
    text = template.format(**fields)
    if not text.startswith(KEPT_STARTS):
        text = REFUSED + text
    return Refusal(text, tool, form)


def form_of(text: object) -> str | None:
    """The form code a reply was made with (:class:`Refusal`), else None."""
    return getattr(text, "form", None)
