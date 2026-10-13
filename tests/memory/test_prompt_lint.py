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

import pytest

from unify.common import in_code_switch
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
ROLES = ("actor", "teammate", "writer", "curator", "helper", "librarian", "fork")
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
    """Each form of the actor's memory section with tools in code: the inline guide with the index head and the
    overflow paragraph, the pointer guide, and the librarian's guide."""
    from unify.memory import library_index as li
    from unify.memory import library_prompts as lp

    view = li.prompt_view_head(li.index_view(li.INDEX_HEAD + _SECTION, budget_tokens=1))
    assert view.startswith(li.INDEX_HEAD_IN_CODE) and li.SECTION_CALL_IN_CODE in view, view
    return [lp.guide_library() + "\n" + view, lp.guide_pointer(), lp.GUIDE_LIBRARIAN]


def _actor_core() -> str:
    from unify.actor import prompt_builders as pb
    from unify.actor.core_surface import PromptSurface

    return pb.build_code_act_prompt(
        environments={},
        core=PromptSurface(functions=False, guidance=False),
        session_sections=False,
    )


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

    full = {"staging": True, "pool": True, "helper": True, "similar": True, "edit_tools": True}
    tool_text = R.EXECUTE_CODE.description
    actor = [_opening("actor"), _actor_core(), *_actor_memory_sections(), tool_text]
    return {
        "actor": actor,
        "teammate": [_opening("teammate"), _actor_core(), *_actor_memory_sections(), tool_text],
        "writer": [
            lp.write_brief_now(),
            lp.write_brief_now(**full, compact=True, finish_capped=True),
            tool_text,
        ],
        "curator": [lp.curate_brief_now(compact=True, finish_capped=True, edit_tools=True), tool_text],
        "helper": [helpers.helper_brief(True), helpers.helper_brief(False), tool_text],
        "librarian": [L.system_prompt(""), tool_text],
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


@pytest.fixture
def in_code(monkeypatch):
    monkeypatch.setattr(in_code_switch, "tools_in_code", lambda: True)


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


def test_no_role_prompt_shows_a_calls_mechanics(in_code):
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


def test_the_library_memory_texts_name_only_the_librarys_calls(in_code):
    """The memory texts with tools in code name the library's three read calls, each with no arguments, and point
    to ``help(memory)``; none names an older call."""
    from unify.memory import agent_profiles
    from unify.memory import librarian as L
    from unify.memory import library_prompts as lp

    texts = {
        "guide": lp.guide_library(),
        "pointer": lp.guide_pointer(),
        "librarian guide": lp.GUIDE_LIBRARIAN,
        "writer bullet": lp._LIBRARY_BULLET_IN_CODE,
        "curator brief": lp.curate_brief_now(edit_tools=True),
        "librarian brief": L.BRIEF,
        **{f"{agent} opening line": line for agent, line in agent_profiles._LIBRARY_LINE.items()},
    }
    old = re.compile(r"\bmemory\.(?:index|show|find|match|run|catalog|describe)\b")
    for name, text in texts.items():
        assert not old.search(text), (name, old.search(text).group(0))
        assert set(re.findall(r"\bmemory\.([a-z_]+)\b", text)) <= {"ask", "items", "get"}, name
    for name in ("guide", "pointer", "librarian guide"):
        assert lp.MEMORY_HELP_LINE in texts[name], name
    for agent in ("writer", "curator", "librarian"):
        assert lp.MEMORY_HELP_LINE in agent_profiles._LIBRARY_LINE[agent], agent


@pytest.mark.parametrize("librarian", [False, True], ids=["librarian off", "librarian on"])
def test_memory_ask_is_named_only_where_the_librarian_answers_it(in_code, tmp_path, librarian):
    """``memory.ask`` is the librarian: the actor's section names it only with the librarian on, in either index
    form; both switch states name ``memory.items`` and ``memory.get`` and show no call's mechanics."""
    from unify.memory import library_index as li
    from unify.memory.integration.prompt import render_memory_library_mode

    copy = tmp_path / "copy"
    copy.mkdir()
    (copy / "INDEX.md").write_text(li.INDEX_HEAD + _SECTION)
    for form in ("inline", "pointer"):
        text = render_memory_library_mode(copy, budget_tokens=1, actor_index=form, librarian=librarian)[0]
        assert text, form
        assert ("`memory.ask`" in text) is librarian, form
        assert "`memory.items`" in text and "`memory.get`" in text, form
        assert ("memory.text" in text) is (form == "inline" and not librarian), form  # the index view's heading
        assert hits("actor", text) == [], (form, hits("actor", text))


def test_the_tools_off_texts_are_unchanged(monkeypatch):
    """With tools in code off the memory texts are as they were: the guides still show the earlier calls."""
    from unify.memory import library_index as li
    from unify.memory import library_prompts as lp

    monkeypatch.setattr(in_code_switch, "tools_in_code", lambda: False)
    assert lp.guide_library() == lp.GUIDE_LIBRARY and lp.guide_pointer() == lp.GUIDE_POINTER
    for text in (lp.GUIDE_LIBRARY, lp.GUIDE_POINTER):
        assert lp.MATCH_LINE in text and lp.RUN_LINE in text and '`print(memory.show("<id>"))`' in text
    view = li.index_view(li.INDEX_HEAD + _SECTION, budget_tokens=1)
    assert li.prompt_view_head(view) == view and li.in_code_section_call(view) == view
    assert '`memory.index("<section>")` prints one section in full (for example `memory.index("memory.text")`)' in view
