"""Symbolic: with ``UNIFY_TOOLS_IN_CODE=on`` the writer's, the curator's and a helper's system message starts with the
shared opening (``tool_registry.preamble``) built from the agent's profile, its allow-list flags and kept variables,
then the role's own brief. The opening holds box paths only. With the setting off, no brief has it."""

from __future__ import annotations

import os
import re
import warnings

import pytest

from unify.common import in_code_switch
from unify.common import tool_registry as R

HEADING = "## How you work here"
LIST_SENTENCE = "`tools.list()` prints one line per tool you have"
KEPT_SENTENCE = "Your Python sandbox is persistent and stateful"
FRESH_SENTENCE = "Each cell runs in a fresh Python sandbox"
NAMES_SENTENCE = "`tools` and `memory` are already available as variables in every cell; no import needed."
PLANTED_VALUE = "planted-value-for-the-opening-test"


@pytest.fixture
def planted(tmp_path, monkeypatch):
    """A run root in ``UNIFY_HOME`` and an environment value, neither of which an opening may name."""
    root = tmp_path / "planted-run-root"
    root.mkdir()
    monkeypatch.setenv("UNIFY_HOME", str(root))
    monkeypatch.setenv("PLANTED_VALUE_FOR_THE_OPENING_TEST", PLANTED_VALUE)
    return {"root": str(root), "tmp": str(tmp_path)}


def _role_start(agent: str) -> str:
    """Where the role's own brief starts, after the opening and its blank line."""
    return "\n\nYou are a helper" if agent == "helper" else "\n\n# Role\n"


def _briefs() -> dict[str, tuple[str, str, frozenset]]:
    """Each memory agent's system message as its pass sends it: (agent, text, the flags its allow-list uses)."""
    from unify.memory import helpers
    from unify.memory import library_prompts as lp

    full = {"staging": True, "pool": True, "helper": True, "similar": True, "edit_tools": True}
    return {
        "writer base": ("writer", lp.write_brief_now(), lp.tool_flags()),
        "writer full": (
            "writer",
            lp.write_brief_now(**full, finish_capped=True),
            lp.tool_flags(helper=True, similar=True, edit=True),
        ),
        "curator": ("curator", lp.curate_brief_now(edit_tools=True), lp.tool_flags(edit=True)),
        "helper": ("helper", helpers.helper_brief(True), helpers.helper_flags(True)),
        "helper without similar": ("helper", helpers.helper_brief(False), helpers.helper_flags(False)),
    }


def test_every_memory_agent_starts_with_the_shared_opening(monkeypatch, planted):
    from unify.memory import agent_profiles

    monkeypatch.setattr(in_code_switch, "tools_in_code", lambda: True)
    report = []
    for name, (agent, text, flags) in _briefs().items():
        opening, sep, _ = text.partition(_role_start(agent))
        assert sep, f"{name}: the role's own text does not follow the opening"
        assert text.startswith(HEADING + "\n"), name
        assert LIST_SENTENCE in opening and KEPT_SENTENCE in opening, name
        # the shared sandbox paragraph is said once, in the opening, never again in the role's text
        assert NAMES_SENTENCE in opening and text.count(KEPT_SENTENCE) == 1, name
        assert "import memory" not in text, name
        assert text.count(HEADING) == 1, name
        # the opening names only tools the agent's allow-list has, with the flags its pass builds the list with
        allow = set(R.allow_list(agent, flags))
        named = R.tool_names_in(opening) - {"list", "help", "Refused", "NotRun"}
        assert named <= allow, f"{name}: {sorted(named - allow)}"
        # box paths and limits only: no host path, no run root, no environment value
        for value in (planted["root"], planted["tmp"], PLANTED_VALUE, os.getcwd(), os.path.expanduser("~")):
            assert value not in opening, f"{name}: {value}"
        assert "UNIFY_" not in opening, name
        for mount in agent_profiles.profile(agent)["mounts"]:
            assert mount["path"] in opening, f"{name}: {mount['path']}"
        assert "cell time: 60 s" in opening and "stated in the first message" in opening, name
        report.append(f"{name}: {text[:300]!r}")
    warnings.warn("FIRST 300 ON\n" + "\n".join(report), stacklevel=1)


def test_the_openings_library_line_names_no_call_and_points_to_the_help(monkeypatch):
    """The opening says where the library is and that it is bound; the library's calls, with their arguments and
    results, are in ``help(memory)``, which the line points to (a helper's line names no call at all)."""
    from unify.memory import agent_profiles
    from unify.memory import library_prompts as lp

    monkeypatch.setattr(in_code_switch, "tools_in_code", lambda: True)
    for name, (agent, text, _) in _briefs().items():
        opening = text.partition(_role_start(agent))[0]
        assert agent_profiles._LIBRARY_LINE[agent] in opening, name
        assert not re.search(r"\bmemory\.[a-z_]+", opening), name
        assert (lp.MEMORY_HELP_LINE in opening) is (agent != "helper"), name


def test_with_the_setting_off_no_brief_has_the_opening(monkeypatch):
    """With tools in code off there is no opening: each brief starts with the shared sandbox paragraph instead, which
    names `memory` alone (its tools stay JSON tools)."""
    monkeypatch.setattr(in_code_switch, "tools_in_code", lambda: False)
    for name, (_, text, _) in _briefs().items():
        assert HEADING not in text, name
        assert LIST_SENTENCE not in text, name
        assert text.startswith(KEPT_SENTENCE), name
        assert "`memory` is already available as a variable in every cell; no import needed." in text, name
        assert text.count(KEPT_SENTENCE) == 1 and "`tools`" not in text.split("\n\n", 1)[0], name
        assert "import memory" not in text, name


def test_under_the_fresh_opt_out_the_opening_says_cells_start_fresh(monkeypatch):
    from unify.settings import SETTINGS

    monkeypatch.setattr(SETTINGS, "UNIFY_CELL_STATE", "fresh")
    monkeypatch.setattr(in_code_switch, "tools_in_code", lambda: True)
    for name, (agent, text, _) in _briefs().items():
        opening = text.partition(_role_start(agent))[0]
        assert FRESH_SENTENCE in opening and KEPT_SENTENCE not in text, name
    # with tools in code off, the opt-out's briefs are the earlier ones: each cell is a fresh script that imports
    # the library itself, and the briefs say so
    monkeypatch.setattr(in_code_switch, "tools_in_code", lambda: False)
    for name, (_, text, _) in _briefs().items():
        assert KEPT_SENTENCE not in text and FRESH_SENTENCE not in text, name
        assert "import memory" in text or name.startswith("helper"), name


def test_brief_sizes_without_and_with_the_opening(monkeypatch):
    """Each memory agent's system message in characters, with the setting off, and on with the opening (printed
    and warned for the report)."""
    monkeypatch.setattr(in_code_switch, "tools_in_code", lambda: False)
    off = {name: len(text) for name, (_, text, _) in _briefs().items()}
    monkeypatch.setattr(in_code_switch, "tools_in_code", lambda: True)
    on = {}
    role_only = {}
    for name, (agent, text, _) in _briefs().items():
        on[name] = len(text)
        role_only[name] = len(text) - len(text.split(_role_start(agent), 1)[0]) - 2
    lines = [f"{k}: off {off[k]}, on role text {role_only[k]}, on with opening {on[k]}" for k in off]
    warnings.warn("MEMORY BRIEF SIZES (characters)\n" + "\n".join(lines), stacklevel=1)
