"""Symbolic: the writer's and a helper's system message starts with the shared opening (``tool_registry.preamble``)
built from the agent's profile, its allow-list flags and kept variables, then the role's own brief. The opening holds
box paths only."""

from __future__ import annotations

import os
import re
import warnings

import pytest

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

    full = {"helper": True, "similar": True}
    return {
        "writer base": ("writer", lp.write_brief_now(), lp.tool_flags()),
        "writer full": (
            "writer",
            lp.write_brief_now(**full, finish_capped=True),
            lp.tool_flags(helper=True, similar=True),
        ),
        "helper": ("helper", helpers.helper_brief(True), helpers.helper_flags(True)),
        "helper without similar": ("helper", helpers.helper_brief(False), helpers.helper_flags(False)),
    }


def test_every_memory_agent_starts_with_the_shared_opening(monkeypatch, planted):
    from unify.memory import agent_profiles

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
        # rev 2's opening states the pass's wall clock as a number beside the cell time
        assert "cell time: 60 s" in opening and re.search(r"pass wall clock: [0-9]+ s", opening), name
        report.append(f"{name}: {text[:300]!r}")
    warnings.warn("FIRST 300 ON\n" + "\n".join(report), stacklevel=1)


def test_the_openings_library_line_names_no_call_and_points_to_the_help(monkeypatch):
    """The opening says where the library is and that it is bound; the library's calls, with their arguments and
    results, are in ``help(memory)``, which the line points to (a helper's line names no call at all)."""
    from unify.memory import agent_profiles
    from unify.memory import library_prompts as lp

    for name, (agent, text, _) in _briefs().items():
        opening = text.partition(_role_start(agent))[0]
        assert agent_profiles._LIBRARY_LINE[agent] in opening, name
        assert not re.search(r"\bmemory\.[a-z_]+", opening), name
        assert (lp.MEMORY_HELP_LINE in opening) is (agent != "helper"), name
