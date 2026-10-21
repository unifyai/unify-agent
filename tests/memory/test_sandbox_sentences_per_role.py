"""What every agent is told about its sandbox, without a box: each role's system message (actor, teammate, writer,
curator, helper) opens with the shared opening, and inside it says once that the sandbox is persistent and stateful
(or, under ``UNIFY_CELL_STATE=fresh``, that each cell starts fresh), names the restarts that lose variables, and says
that its pre-bound names (``memory`` and ``tools`` among them) are already available as variables with no import
needed. No role says ``import memory`` or calls the library "importable" any more. The pre-bound sets come from one
function per role and configuration, and never widen a role's tools.

Keyless and box-free: the prompts are rendered by the roles' own builders.
"""

from __future__ import annotations

import re
from pathlib import Path
from types import SimpleNamespace

import pytest

from unify.common import box_interpreter as bi
from unify.common import tool_registry as R
from unify.memory import helpers
from unify.memory import library_prompts as lp
from unify.memory.integration import request as request_mod
from unify.memory.integration.paths import Paths
from unify.memory.library_export import generated_library_mode
from unify.settings import SETTINGS

# ==== names and sentences this file assumes (the spec gives them; adapt them here, in one place) ====================
#: tool_registry.sandbox_paragraph(agent, names, persistent) -> str (design D.1)
SANDBOX_PARAGRAPH = "sandbox_paragraph"
#: tool_registry.prebound_names(agent, facts) -> iterable of names (design B.1)
PREBOUND_NAMES = "prebound_names"
#: the facts keys prebound_names reads: tools called from code, a library copy with its package, clarification on
FACT_IN_CODE = "tools"
FACT_LIBRARY = "memory"
FACT_CLARIFICATION = "clarification"
#: design D.1, persistent form (the default), in two sentences
PERSISTENT_SENTENCE = (
    "Your Python sandbox is persistent and stateful: variables, imports and functions you define in one cell are "
    "still there in the next, so build on what you already have instead of defining or loading it again."
)
RESTART_SENTENCE = (
    "If the sandbox restarts (a cell ran out of time, broke a limit, left something running, or sat idle too long), "
    "the cell's output says so; then variables are gone and files remain."
)
#: design D.1, fresh form (only under UNIFY_CELL_STATE=fresh)
FRESH_SENTENCE = (
    "Each cell runs in a fresh Python sandbox: files you write stay, but variables, imports and functions do not "
    "carry over to the next cell."
)
#: design D.1, the names sentence: "{names} are already available as variables in every cell; no import needed."
NAMES_SENTENCE = re.compile(
    r"((?:`[A-Za-z_][A-Za-z0-9_]*`(?:, | and |, and )?)+) (?:is|are) already available as (?:a )?variables? in every "
    r"cell; no import needed\.",
)
#: design D.3 (B.4): the writer's and curator's sentence about a library file edited after it was imported
REIMPORT_SENTENCE = (
    "Names you bound earlier keep the code they were bound to; after you edit a library file, import it again."
)
#: design D.2: the actor's library guide and role
GUIDE_BOUND = "is already available in your cells as `memory`"
# =====================================================================================================================

ROLES = ("actor", "teammate", "writer", "curator", "helper")
MEMORY_ROLES = ("writer", "curator", "helper")
#: what no role says any more (design A.3, D.2-D.4)
IMPORT_MEMORY = re.compile(r"\bimport memory\b")
IMPORTABLE = "importable in your cells"
#: the bare names `tools` replaced; never pre-bound (design B.2, B.3)
REPLACED = ("install", "read_file", "grep", "record", "agents")
#: model and network helpers a memory box must never get (design B.2, B.3)
NO_MODEL_HELPERS = ("query_llm", "list_llms", "unillm", "run_coro_sync", "display", "ModelRequestError")


def flat(text: str) -> str:
    """*text* with every run of whitespace as one space (a sentence may be wrapped)."""
    return " ".join(str(text).split())


def names_in_sentence(text: str) -> set[str]:
    """The names the (single) names sentence of *text* lists."""
    found = NAMES_SENTENCE.findall(flat(text))
    assert len(found) == 1, f"expected one names sentence, found {len(found)}: {found}"
    return set(re.findall(r"`([A-Za-z_][A-Za-z0-9_]*)`", found[0]))


def opening_of(text: str, role: str) -> str:
    """The shared opening at the start of *text*: up to the line that says how the agent ends its work. The actor's
    and a teammate's text here is their opening alone (their role section says how they answer)."""
    assert text.startswith("## How you work here\n"), text[:200]
    if role in ("actor", "teammate"):
        assert "You end your work with" not in text, text[:2000]
        return text
    end = text.find("You end your work with tools.")
    assert end > 0, text[:2000]
    stop = text.find("\n", end)
    return text if stop < 0 else text[:stop]


# --- the roles' system messages ---------------------------------------------------------------------------------


@pytest.fixture
def in_code(monkeypatch):
    return monkeypatch


def _library_run(monkeypatch, home: Path) -> Paths:
    """A library-mode request whose read-only copy holds the generated package of an empty library."""
    monkeypatch.setattr(SETTINGS, "UNIFY_MEMORY", "on")
    paths = Paths.under(home)
    empty = home / "empty-library"
    empty.mkdir(parents=True)
    for rel, data in generated_library_mode(empty).items():
        (paths.checkout / rel).parent.mkdir(parents=True, exist_ok=True)
        (paths.checkout / rel).write_bytes(data)
    monkeypatch.setattr(
        request_mod,
        "_CURRENT",
        SimpleNamespace(index="", paths=paths, library_mode=True),
    )
    return paths


def _actor_session(tmp_path: Path, agent: str):
    from unify.actor import in_code_tools

    work = tmp_path / f"{agent}-workspace"
    work.mkdir(exist_ok=True)
    return in_code_tools.InCodeTools(
        agent=agent,
        name="root" if agent == "actor" else "mate",
        record=SimpleNamespace(),
        workspace_root=str(work),
        shadow={},
        cell_limit_s=30.0,
        events_folder=tmp_path / f"{agent}-events",
    )


async def _openings(tmp_path: Path) -> dict[str, str]:
    """Each role's system message (the actor's and the teammate's opening; the memory agents' whole brief)."""
    out = {
        "writer": lp.write_brief_now(cell_s=60.0, helper=True),
        "curator": lp.curate_brief_now(cell_s=60.0),
        "helper": helpers.helper_brief(False),
    }
    for agent in ("actor", "teammate"):
        session = _actor_session(tmp_path, agent)
        try:
            out[agent] = session.opening()
        finally:
            await session.close(completed=True)
    return out


@pytest.mark.asyncio
async def test_every_role_says_once_that_its_sandbox_is_persistent_and_names_the_restarts(in_code, tmp_path):
    _library_run(in_code, tmp_path / "home")
    texts = await _openings(tmp_path)
    assert set(texts) == set(ROLES)
    for role, text in texts.items():
        body = flat(text)
        assert text.startswith("## How you work here\n"), (role, text[:200])
        assert body.count(PERSISTENT_SENTENCE) == 1, (role, body[:1500])
        assert body.count(RESTART_SENTENCE) == 1, (role, body[:1500])
        assert FRESH_SENTENCE not in body, role
        # the shared opening carries it, before the role's own instructions
        assert PERSISTENT_SENTENCE in flat(opening_of(text, role)), role


@pytest.mark.asyncio
async def test_every_role_says_memory_and_tools_are_variables_with_no_import(in_code, tmp_path):
    _library_run(in_code, tmp_path / "home")
    texts = await _openings(tmp_path)
    for role, text in texts.items():
        names = names_in_sentence(text)
        assert {"tools", "memory"} <= names, (role, names)
        # the sentence never offers a bare name that `tools` replaced
        assert not names & set(REPLACED), (role, names)
        if role in MEMORY_ROLES:
            assert not names & set(NO_MODEL_HELPERS), (role, names)


@pytest.mark.asyncio
async def test_no_role_says_import_memory_or_importable(in_code, tmp_path):
    _library_run(in_code, tmp_path / "home")
    texts = await _openings(tmp_path)
    for role, text in texts.items():
        assert not IMPORT_MEMORY.search(text), (role, IMPORT_MEMORY.search(text))
        assert IMPORTABLE not in flat(text), role


@pytest.mark.asyncio
async def test_under_fresh_cell_state_every_role_is_told_each_cell_starts_fresh(in_code, tmp_path):
    _library_run(in_code, tmp_path / "home")
    in_code.setattr(SETTINGS, "UNIFY_CELL_STATE", "fresh")
    texts = await _openings(tmp_path)
    for role, text in texts.items():
        body = flat(text)
        assert body.count(FRESH_SENTENCE) == 1, (role, body[:1500])
        assert PERSISTENT_SENTENCE not in body and "persistent and stateful" not in body, role
        assert "keeps its variables between cells" not in body, role


@pytest.mark.asyncio
async def test_the_actor_without_a_library_is_told_tools_only(in_code, tmp_path):
    in_code.setattr(SETTINGS, "UNIFY_MEMORY", "off")
    session = _actor_session(tmp_path, "actor")
    try:
        text = session.opening()
    finally:
        await session.close(completed=True)
    names = names_in_sentence(text)
    assert "tools" in names and "memory" not in names, names
    assert flat(text).count(PERSISTENT_SENTENCE) == 1


@pytest.mark.asyncio
async def test_the_new_sentences_hold_no_host_path_secret_or_forcing(in_code, tmp_path, monkeypatch):
    planted = "planted-value-for-the-sandbox-sentences"
    monkeypatch.setenv("PLANTED_VALUE_FOR_THE_SANDBOX_SENTENCES", planted)
    _library_run(in_code, tmp_path / "home")
    texts = await _openings(tmp_path)
    for role, text in texts.items():
        opening = opening_of(text, role)
        for value in (str(tmp_path), planted):
            assert value not in opening, role
        assert lp.example_checks(opening) == [], role
        assert lp.benchmark_words(opening) == [], role
    # the shared sentences are the same bytes for every role (a stable, cached prefix)
    paragraph = {role: PERSISTENT_SENTENCE in flat(t) for role, t in texts.items()}
    assert all(paragraph.values()), paragraph


# --- the actor's guide, role and sandbox section ------------------------------------------------------------------


def test_the_actor_in_code_prompt_leaves_persistence_to_the_opening():
    from unify.actor import prompt_builders as pb

    rules = flat(pb._LEAN_EXECUTION_RULES_IN_CODE)
    assert "cells share one persistent sandbox" not in rules
    assert "usually means the sandbox restarted" not in rules
    assert "**Sessions**: bind results to variables and build on them" in rules
    sandbox = flat(pb._SANDBOX_IN_CODE)
    assert "already bound" in sandbox and "no import needed" in sandbox
    assert "`tools`" in sandbox and "`display(obj)`" in sandbox
    assert "importable" not in flat(pb._LEAN_ROLE_LIBRARY_MEMORY)


# --- the functions behind the sentences -----------------------------------------------------------------------------


def _paragraph(agent: str, names, persistent: bool) -> str:
    return getattr(R, SANDBOX_PARAGRAPH)(agent, names, persistent)


@pytest.mark.parametrize("agent", ROLES)
def test_sandbox_paragraph_persistent_and_fresh_forms(agent):
    on = flat(_paragraph(agent, ["tools", "memory"], True))
    assert PERSISTENT_SENTENCE in on and RESTART_SENTENCE in on and FRESH_SENTENCE not in on
    assert names_in_sentence(on) == {"tools", "memory"}
    off = flat(_paragraph(agent, ["tools", "memory"], False))
    assert FRESH_SENTENCE in off and PERSISTENT_SENTENCE not in off
    assert names_in_sentence(off) == {"tools", "memory"}


def test_sandbox_paragraph_names_sentence_forms():
    one = flat(_paragraph("writer", ["memory"], True))
    assert "`memory` is already available" in one, one
    assert names_in_sentence(one) == {"memory"}
    many = flat(_paragraph("helper", ["tools", "memory", "help"], True))
    assert names_in_sentence(many) == {"tools", "memory", "help"}
    # no names: no names sentence at all
    none = flat(_paragraph("writer", [], True))
    assert not NAMES_SENTENCE.search(none) and "no import needed" not in none
    assert PERSISTENT_SENTENCE in none
    # the same bytes for every role, apart from the names
    texts = {_paragraph(a, ["tools", "memory"], True) for a in ROLES}
    assert len(texts) == 1


def test_sandbox_paragraph_is_constant_and_names_no_host_value(tmp_path):
    text = _paragraph("actor", ["tools"], True)
    assert text == _paragraph("actor", ["tools"], True)
    assert str(tmp_path) not in text and str(Path.home()) not in text


def test_the_preamble_carries_the_paragraph_for_the_cell_state_it_is_given():
    for agent in ROLES:
        on = flat(R.preamble(agent, {"persistent": True}))
        off = flat(R.preamble(agent, {"persistent": False}))
        assert on.count(PERSISTENT_SENTENCE) == 1 and FRESH_SENTENCE not in on, agent
        assert off.count(FRESH_SENTENCE) == 1 and PERSISTENT_SENTENCE not in off, agent


def _prebound(agent: str, **facts) -> set[str]:
    return set(getattr(R, PREBOUND_NAMES)(agent, facts))


def test_prebound_names_of_the_memory_agents():
    for agent in MEMORY_ROLES:
        lib = {FACT_IN_CODE: True, FACT_LIBRARY: True}
        assert _prebound(agent, **lib) == {"tools", "memory", "help"}, agent
        assert _prebound(agent, **{FACT_IN_CODE: True, FACT_LIBRARY: False}) == {"tools", "help"}, agent
        # tools stay JSON tools with the switch off: no `tools`, the rest the same
        assert _prebound(agent, **{FACT_IN_CODE: False, FACT_LIBRARY: True}) == {"memory", "help"}, agent


def test_prebound_names_of_the_actor_and_teammate():
    facts = {FACT_IN_CODE: True, FACT_LIBRARY: True, FACT_CLARIFICATION: True}
    actor = _prebound("actor", **facts)
    assert {
        "tools",
        "memory",
        "help",
        "display",
        "query_llm",
        "list_llms",
        "unillm",
        "run_coro_sync",
        "ModelRequestError",
        "request_clarification",
    } <= actor, actor
    mate = _prebound("teammate", **facts)
    assert "request_clarification" not in mate
    assert actor - {"request_clarification"} == mate
    assert "memory" not in _prebound("actor", **{**facts, FACT_LIBRARY: False})
    assert "request_clarification" not in _prebound("actor", **{**facts, FACT_CLARIFICATION: False})
    for names in (actor, mate):
        assert not names & set(REPLACED), names
        assert not names & {"functions", "guidance"}, names


def test_the_default_cell_state_is_persistent(monkeypatch):
    from unify.settings import ProductionSettings

    monkeypatch.delenv(bi.CELL_STATE_SETTING, raising=False)
    assert ProductionSettings.model_fields["UNIFY_CELL_STATE"].default == "persistent"
    assert bi.cell_state() == "persistent"
