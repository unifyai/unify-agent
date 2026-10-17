"""The tool reference at the end of the writer's and the curator's brief (``UNIFY_MEMORY_LIBRARY_TOOL_REFERENCE``):
the switch, the reference generated from the tool registry, and the briefs with the switch off and on.
"""

import pytest

from unify.common import tool_registry
from unify.memory import library_prompts as lp
from unify.memory.integration import switch
from unify.memory.writer_pass import PassConfig
from unify.settings import SETTINGS
from tests.memory.test_writer_pass import Turns, _writer


def test_the_setting_is_off_by_default_and_parses_on_and_off():
    assert SETTINGS.UNIFY_MEMORY_LIBRARY_TOOL_REFERENCE == "off"
    assert PassConfig().tool_reference is False
    assert (
        switch.PARSERS[switch.LIBRARY_TOOL_REFERENCE]
        is switch.parse_library_tool_reference
    )
    assert switch.parse_library_tool_reference("") == "off"
    assert switch.parse_library_tool_reference(" ON ") == "on"
    with pytest.raises(ValueError, match="UNIFY_MEMORY_LIBRARY_TOOL_REFERENCE"):
        switch.parse_library_tool_reference("yes")

    class S:
        UNIFY_MEMORY_LIBRARY_TOOL_REFERENCE = "on"

    assert switch.library_tool_reference(S()) is True
    S.UNIFY_MEMORY_LIBRARY_TOOL_REFERENCE = ""
    assert switch.library_tool_reference(S()) is False


@pytest.mark.parametrize("agent", ["writer", "curator"])
def test_the_reference_is_the_registry_list_of_the_agents_tools(agent):
    names = tool_registry.allow_list(agent, lp.tool_flags(pool=True, edit=True))
    text = lp.tool_reference(agent, names)
    assert text.startswith(lp.TOOL_REFERENCE_HEAD)
    lines = text[len(lp.TOOL_REFERENCE_HEAD) :].splitlines()
    listed = [n for n in names if n not in tool_registry.CHANNEL_ONLY]
    assert lines == [
        f"- tools.{line}"
        for line in tool_registry.list_text(listed, agent).splitlines()
    ]
    assert len(lines) == len(listed)
    for name, line in zip(listed, lines):
        assert line.startswith(
            f"- tools.{tool_registry.signature_text(name, agent)} - "
        )
    assert not any(f"tools.{n}(" in text for n in tool_registry.CHANNEL_ONLY)
    assert "tools.help(name)" in text
    assert lp.tool_reference(agent, ()) == ""


def test_the_briefs_end_with_the_reference_only_in_code(monkeypatch):
    names = tool_registry.allow_list("writer", lp.tool_flags())
    monkeypatch.setattr(SETTINGS, "UNIFY_TOOLS_IN_CODE", "on")
    plain = lp.write_brief_now()
    with_ref = lp.write_brief_now(reference=names)
    assert with_ref.startswith(plain.rstrip("\n"))
    assert with_ref.endswith(lp.tool_reference("writer", names))
    assert lp.write_brief_now(reference=()) == plain
    cur = tool_registry.allow_list("curator", lp.tool_flags())
    assert lp.curate_brief_now(reference=cur).endswith(
        lp.tool_reference("curator", cur)
    )
    monkeypatch.setattr(SETTINGS, "UNIFY_TOOLS_IN_CODE", "off")
    assert lp.write_brief_now(reference=names) == lp.write_brief_now()
    assert lp.curate_brief_now(reference=cur) == lp.curate_brief_now()


@pytest.mark.parametrize("role", ["write", "curate"])
def test_the_pass_brief_carries_the_reference_of_its_own_allow_list(
    tmp_path, monkeypatch, role
):
    monkeypatch.setattr(SETTINGS, "UNIFY_TOOLS_IN_CODE", "on")
    briefs = {}
    for on in (False, True):
        sub = tmp_path / f"{role}-{on}"
        sub.mkdir()
        _, _, writer = _writer(
            sub, Turns([]), library_mode=True, pool=True, tool_reference=on
        )
        writer._role = role
        briefs[on] = (writer._brief_library_mode(), writer._in_code_allow())
    (off, _), (on, allow) = briefs[False], briefs[True]
    agent = "curator" if role == "curate" else "writer"
    assert lp.TOOL_REFERENCE_HEAD not in off
    assert on == off.rstrip("\n") + "\n\n" + lp.tool_reference(agent, allow)
    assert allow == tool_registry.allow_list(agent, lp.tool_flags(pool=True))
