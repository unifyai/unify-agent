"""Symbolic: with tools called from code, a role's prompt orchestrates and the library's calls describe themselves.

A prompt says when to use which call and how to combine them, and names each call without its arguments
(``memory.items``, ``memory.get``, and ``memory.ask`` where the librarian answers it). Signatures, argument forms,
item id formats, result shapes and error mechanics live in each call's help (``help(memory)``), never in a prompt.
This test renders
every role's in-code prompt from the role's own builders (the actor, a teammate, the writer, the curator, a helper,
the librarian, and the episode-end fork) and fails on any of:

* a call of ``tools``, ``memory`` or ``env`` shown with arguments (``memory.get("<id>")``, ``tools.read(path)``);
* any call form of a library call (``memory.items()``): the library's calls are named, never shown called;
* an item id or an id template (``memory.<package>.<module>:<name>``, ``memory.text.dates:parse_date``);
* a note id template (``notes/<topic>/<slug>.md``);
* the tool error classes (``tools.Refused``, ``tools.NotRun``, ``tools.ToolError``, ``tools.ChannelClosed``).

:data:`ALLOWED` lists what may still match, each with its reason: the writer's authoring contract (where the files
it writes go, the ids they get, and a note's manifest entry) states a layout, not a call's mechanics; and the
``tools.*`` texts are left for the change that moves the tools' own mechanics into their help. Every entry must
still match something, so the list is reviewed whenever a prompt changes. Nothing here starts a box or a model.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


from unify.common import tool_registry as R

#: Each rule: its name and the pattern a prompt line must not hold.
RULES: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("call with arguments", re.compile(r"\b(?:tools|memory|env)\.[A-Za-z_]\w*\((?!\))")),
    ("library call form", re.compile(r"\b(?:memory|env)\.[A-Za-z_]\w*\(")),
    (
        "item id",
        re.compile(r"\bmemory\.(?:[A-Za-z_]\w*|<[\w ]+>)\.(?:[A-Za-z_]\w*|<[\w ]+>):(?:[A-Za-z_]\w*|<[\w ]+>)"),
    ),
    ("note id template", re.compile(r"\bnotes/<[\w ]+>/<[\w ]+>\.md")),
    ("tool error class", re.compile(r"\btools\.(?:Refused|NotRun|ToolError|ChannelClosed)\b")),
)
RULE_NAMES = frozenset(name for name, _ in RULES)
ROLES = ("actor", "teammate", "writer", "helper", "librarian", "fork")  # rev 2 has no curator
#: The reason every hit on a ``tools.*`` text is allowed for now.
TOOLS_LATER = "tools.* mechanics move in the next build"


@dataclass(frozen=True)
class Allowed:
    """What may still match a rule: the rule, the roles, the text the matched span (*match*) or its line (*line*)
    holds, and why."""

    rule: str
    roles: frozenset[str]
    reason: str
    match: str | None = None
    line: str | None = None

    def covers(self, rule: str, role: str, span: str, line: str) -> bool:
        if rule != self.rule or role not in self.roles:
            return False
        if self.match is not None and not re.match(self.match, span):
            return False
        return self.line is None or self.line in line


ALL = frozenset(ROLES)
WRITER = frozenset({"writer"})
#: The reviewed exceptions, each with its reason.
ALLOWED: tuple[Allowed, ...] = (
    Allowed("call with arguments", ALL, TOOLS_LATER, match=r"tools\."),
    Allowed("tool error class", ALL, TOOLS_LATER, match=r"tools\."),
    Allowed(
        "item id",
        WRITER,
        "the writer's authoring contract: the id each function it writes gets (a layout, not a call's mechanics)",
        line="A function's id is `memory.<package>.<module>:<name>`",
    ),
    Allowed(
        "note id template",
        WRITER,
        "the writer's authoring contract: where each note it writes goes (a layout, not a call's mechanics)",
        line="Notes go in `notes/<topic>/<slug>.md`",
    ),
    Allowed(
        "note id template",
        WRITER,
        "the writer's authoring contract: a note's entry in the manifest it writes (the manifest's JSON form)",
        line='`{"item": "notes/<topic>/<slug>.md", "kind": "note"',
    ),
)

#: An index past its view's budget, so the actor's section shows the head and the overflow paragraph (no item lines:
#: an item's line is the library's data, not prompt text).
_SECTION = "\n## memory.text\n\n- `memory.text.dates:parse_date(s)` — Parse a date. [on-trial]\n"


def _actor_memory_sections() -> list[str]:
    """The actor's memory section: the librarian's guide (the actor reads no index of its own)."""
    from unify.memory import library_prompts as lp

    return [lp.GUIDE_LIBRARIAN]


def _actor_core() -> str:
    from unify.actor import prompt_builders as pb

    return pb.build_code_act_prompt(environments={})


def _opening(agent: str) -> str:
    return R.preamble(
        agent,
        {"environment": "Your box: /workspace and a private /tmp are writable.", "names": ("tools", "memory")},
    )


def role_texts() -> dict[str, list[str]]:
    """Every role's in-code prompt texts, rendered by the role's own builders (tools called from code)."""
    from unify.memory import helpers
    from unify.memory import librarian as L
    from unify.memory import library_prompts as lp
    from unify.memory.integration import fork

    full = {"helper": True, "similar": True}
    tool_text = R.EXECUTE_CODE.description
    actor = [_opening("actor"), _actor_core(), *_actor_memory_sections(), tool_text]
    return {
        "actor": actor,
        "teammate": [_opening("teammate"), _actor_core(), *_actor_memory_sections(), tool_text],
        "writer": [
            lp.write_brief_now(),
            lp.write_brief_now(**full),
            tool_text,
        ],
        "helper": [helpers.helper_brief(True), helpers.helper_brief(False), tool_text],
        "librarian": [L.system_prompt(), tool_text],
        "fork": [fork.fork_instruction()],
    }


@dataclass(frozen=True)
class Hit:
    role: str
    rule: str
    span: str
    line: str


def hits(role: str, text: str) -> list[Hit]:
    out = []
    for line in text.splitlines():
        for rule, pattern in RULES:
            for m in pattern.finditer(line):
                out.append(Hit(role, rule, m.group(0), line))
    return out


def test_the_rules_find_what_they_are_for():
    found = {
        h.rule
        for text in (
            'print(memory.show("<id>")) and memory.items() and tools.read(path)',
            "a function as `memory.<package>.<module>:<name>(<parameters>)`, a note as `notes/<topic>/<slug>.md`",
            "Import `memory.text.dates:parse_date`; a refused call raises tools.Refused or tools.NotRun.",
            "env.get('env.files.read_rows')",
        )
        for h in hits("x", text)
    }
    assert found == RULE_NAMES
    # a call named without its arguments, and the help pointer, are what a prompt may say
    assert hits("x", "Ask with `memory.ask`, open with `memory.get`; `help(memory)` shows the library's calls.") == []
    assert hits("x", "`tools.list()` prints one line per tool; `tools.overview()` shows the rows.") == []


def test_no_role_prompt_shows_a_calls_mechanics():
    """Every rule hit in every role's in-code prompt is on the reviewed list, and every list entry is still used."""
    texts = role_texts()
    assert set(texts) == set(ROLES)
    bad, used = [], set()
    for role, parts in texts.items():
        assert all(parts), role
        for text in parts:
            for h in hits(role, text):
                covering = [i for i, a in enumerate(ALLOWED) if a.covers(h.rule, h.role, h.span, h.line)]
                if covering:
                    used.update(covering)
                else:
                    bad.append(f"{h.role}: {h.rule}: {h.span!r} in {h.line[:200]!r}")
    assert not bad, "prompt text that shows a call's mechanics:\n" + "\n".join(sorted(set(bad)))
    stale = [ALLOWED[i] for i in range(len(ALLOWED)) if i not in used]
    assert not stale, f"allow-list entries nothing matches any more: {stale}"


def test_each_allowed_entry_has_a_known_rule_and_a_reason():
    for a in ALLOWED:
        assert a.rule in RULE_NAMES, a
        assert a.roles and a.roles <= ALL, a
        assert len(a.reason) >= 20 and (a.match or a.line), a


