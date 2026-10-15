"""``UNIFY_LIBRARY=off`` removes memory and library surface only, and ``UNIFY_SYSTEM_SUFFIX`` ends the system prompt
with its text, pinned by ``UNIFY_SYSTEM_SUFFIX_SHA256`` (unify/actor/library_switch.py, unify/actor/system_suffix.py).

The model is scripted (no key, no network). The end-to-end tests run with memory off and the tools called from code,
cells in the real sandboxed worker: with the library off the sandbox has no ``functions``, ``guidance`` or
``memory``, ``tools`` has no ``store_skills`` and keeps the rest, the system prompt names no stored function, and no
storage review runs. Off by default: an unset switch and an unset suffix leave the first request as it is.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
from types import SimpleNamespace

import pytest

from tests import cache_discipline_helpers as h
from tests.actor.code_act.core_world import core_world, new_actor  # noqa: F401 (fixture)
from tests.actor.code_act.sandbox_world import needs_bwrap
from tests.actor.code_act.test_in_code_tools_wiring import _act, _system_texts, _tool_results, _why, code
from tests.helpers import _handle_project
from tests.scripted_model import ScriptedModel, reply
from unify import cli
from unify.actor import library_switch, prompt_builders, system_suffix
from unify.settings import SETTINGS, ProductionSettings

#: Text that names the library in the actor's system prompt (the function store, its search, storing skills, the
#: library's size line, memory).
LIBRARY_TEXT = (
    "functions.search",
    "stored functions",
    "store_skills",
    "guidance.search",
    "Library at task start",
    "`memory`",
)
#: A suffix for the tests (not a run's paragraph); one non-ASCII character, so the sha256 is of UTF-8 bytes.
SUFFIX = "A closing paragraph for this test only: cells are cheap, turns are not – plan."
SUFFIX_SHA = hashlib.sha256(SUFFIX.encode("utf-8")).hexdigest()

#: The tools the probe looks for in the ``tools`` module.
LOOKED_FOR = ("store_skills", "ask_library", "read", "grep", "install", "sh", "check_examples", "answer")
#: A first cell that reads only names (the cell's builtins are a reduced set): which of LOOKED_FOR the ``tools``
#: module has, and which library names are bound (a bare name read in a ``try``, as the namespace tests do).
PROBE = (
    f"_has = [n for n in {LOOKED_FOR!r} if n in tools.__dict__]\n"
    "print('TOOLS ' + ','.join(_has))\n"
    "_bound = []\n"
    + "".join(
        f"try:\n    {n}\n    _bound.append({n!r})\nexcept NameError:\n    pass\n" for n in ("functions", "guidance", "memory")
    )
    + "print('BOUND ' + ','.join(_bound))\n"
)


def _line(output: str, mark: str) -> list[str]:
    line = next(ln for ln in output.splitlines() if ln.startswith(mark + " ") or ln == mark)
    return [x for x in line[len(mark) :].strip().split(",") if x]


# --- the settings ----------------------------------------------------------------------------------------------------


def test_the_library_setting_is_empty_on_or_off_and_the_sha256_is_lowercase_hex():
    assert [library_switch.parse(v) for v in (None, "", "on", " ON ", "off", "Off ")] == ["", "", "", "", "off", "off"]
    with pytest.raises(ValueError, match="UNIFY_LIBRARY must be"):
        library_switch.parse("maybe")
    assert ProductionSettings(UNIFY_LIBRARY=" OFF ").UNIFY_LIBRARY == "off"
    assert system_suffix.parse_sha256(f" {SUFFIX_SHA} ") == SUFFIX_SHA and system_suffix.parse_sha256("") == ""
    for bad in (SUFFIX_SHA.upper(), SUFFIX_SHA[:63], "x" * 64):
        with pytest.raises(ValueError, match="64 lowercase hex"):
            system_suffix.parse_sha256(bad)
    with pytest.raises(ValueError):
        ProductionSettings(UNIFY_SYSTEM_SUFFIX_SHA256=SUFFIX_SHA.upper())


def test_off_with_memory_on_is_refused_by_name():
    refusal = library_switch.refusal(SimpleNamespace(UNIFY_LIBRARY="off", UNIFY_MEMORY="on"))
    assert "UNIFY_LIBRARY=off" in refusal and "UNIFY_MEMORY" in refusal
    assert library_switch.refusal(SimpleNamespace(UNIFY_LIBRARY="off", UNIFY_MEMORY="")) is None
    assert library_switch.refusal(SimpleNamespace(UNIFY_LIBRARY="", UNIFY_MEMORY="on")) is None


@pytest.mark.parametrize(
    "text, sha, named",
    [
        (SUFFIX, "", "set together"),
        ("", SUFFIX_SHA, "set together"),
        (SUFFIX + " ", SUFFIX_SHA, "does not match"),
        (SUFFIX, hashlib.sha256(SUFFIX.encode("latin-1", "replace")).hexdigest(), "does not match"),
    ],
)
def test_a_suffix_without_its_sha256_or_not_matching_it_is_refused_without_the_text(text, sha, named):
    refusal = system_suffix.refusal(SimpleNamespace(UNIFY_SYSTEM_SUFFIX=text, UNIFY_SYSTEM_SUFFIX_SHA256=sha))
    assert named in refusal and "UNIFY_SYSTEM_SUFFIX" in refusal and "closing paragraph" not in refusal
    with pytest.raises(system_suffix.SystemSuffixRefused):
        system_suffix.append("prompt", SimpleNamespace(UNIFY_SYSTEM_SUFFIX=text, UNIFY_SYSTEM_SUFFIX_SHA256=sha))


def test_the_suffix_ends_the_text_after_one_blank_line_and_is_recorded_by_its_sha256():
    pinned = SimpleNamespace(UNIFY_SYSTEM_SUFFIX=SUFFIX, UNIFY_SYSTEM_SUFFIX_SHA256=SUFFIX_SHA)
    unset = SimpleNamespace(UNIFY_SYSTEM_SUFFIX="", UNIFY_SYSTEM_SUFFIX_SHA256="")
    assert system_suffix.refusal(pinned) is None and system_suffix.digest(SUFFIX) == SUFFIX_SHA
    assert system_suffix.append("prompt", pinned) == "prompt\n\n" + SUFFIX
    assert system_suffix.append("prompt", unset) == "prompt"
    assert system_suffix.record(pinned) == {"system_suffix_sha256": SUFFIX_SHA} and system_suffix.record(unset) == {}


@pytest.mark.parametrize(
    "settings",
    [
        {"UNIFY_LIBRARY": "off", "UNIFY_MEMORY": "on"},
        {"UNIFY_SYSTEM_SUFFIX": SUFFIX},
        {"UNIFY_SYSTEM_SUFFIX": SUFFIX + "!", "UNIFY_SYSTEM_SUFFIX_SHA256": SUFFIX_SHA},
    ],
)
def test_the_cli_refuses_before_anything_runs(monkeypatch, capsys, settings):
    for name, value in settings.items():
        monkeypatch.setattr(SETTINGS, name, value)
    monkeypatch.setattr(cli, "_configure_environment", lambda args: None)

    async def _never(args):
        raise AssertionError("the run started")

    monkeypatch.setattr(cli, "_run_act", _never)
    assert cli.main(["act", "Do the task."]) == 2
    err = capsys.readouterr().err
    assert err.startswith("unify: ") and "UNIFY_" in err and "closing paragraph" not in err


def test_off_removes_only_the_library_objects_and_the_lines_naming_stored_functions(monkeypatch):
    objects = {"functions": object(), "guidance": object(), "tools": object(), "install": object()}
    text = "\n\n".join(
        (
            prompt_builders._LEAN_ROLE_LIBRARY,
            prompt_builders._SANDBOX_GLOBALS_INTRO,
            prompt_builders._CORE_SANDBOX_SEARCH_PYTHON,
            "Everything else stays.",
        ),
    )
    monkeypatch.setattr(SETTINGS, "UNIFY_LIBRARY", "")
    assert library_switch.sandbox_objects(objects) is objects and library_switch.system_prompt(text) is text
    assert library_switch.can_store(True) is True
    monkeypatch.setattr(SETTINGS, "UNIFY_LIBRARY", "off")
    assert set(library_switch.sandbox_objects(objects)) == {"tools", "install"}
    assert library_switch.can_store(True) is False
    stripped = library_switch.system_prompt(text)
    assert not any(mark in stripped for mark in LIBRARY_TEXT), stripped
    assert stripped == "\n\n".join(
        (
            prompt_builders._LEAN_ROLE_NO_LIBRARY,
            prompt_builders._SANDBOX_GLOBALS_INTRO_MEMORY,
            prompt_builders._CORE_SANDBOX_HELP_ONLY,
            "Everything else stays.",
        ),
    )


# --- the first request ------------------------------------------------------------------------------------------------


@pytest.fixture
def minimal(core_world, monkeypatch):  # noqa: F811
    """The minimal arms' setting: memory off, the tools called from code, a persistent interpreter."""
    monkeypatch.setattr(SETTINGS, "UNIFY_MEMORY", "")
    monkeypatch.setattr(SETTINGS, "UNIFY_TOOLS_IN_CODE", "on")
    monkeypatch.setattr(SETTINGS, "UNIFY_CELL_STATE", "persistent")
    yield core_world


def _system(request: dict) -> str:
    return "\n".join(str(m.get("content")) for m in request["messages"] if m.get("role") == "system")


async def _first_request() -> dict:
    actor = new_actor(can_store=False)
    try:
        with h.scripted((lambda: h.completion(content="done"),)) as provider:
            handle = await actor.act("Do the task.", persist=False)
            assert await asyncio.wait_for(handle.result(), 120) == "done"
    finally:
        await actor.close()
    return provider.requests[0]


@pytest.mark.asyncio
@pytest.mark.timeout(240)
@_handle_project
async def test_unset_the_prompt_names_the_library_and_the_suffix_only_adds_its_text(minimal, monkeypatch):
    monkeypatch.setattr(SETTINGS, "UNIFY_LIBRARY", "")
    unset = _system(await _first_request())
    assert any(mark in unset for mark in LIBRARY_TEXT), unset[:1500]
    monkeypatch.setattr(SETTINGS, "UNIFY_LIBRARY", "on")
    assert _system(await _first_request()) == unset
    monkeypatch.setattr(SETTINGS, "UNIFY_SYSTEM_SUFFIX", SUFFIX)
    monkeypatch.setattr(SETTINGS, "UNIFY_SYSTEM_SUFFIX_SHA256", SUFFIX_SHA)
    assert _system(await _first_request()) == unset + "\n\n" + SUFFIX


@pytest.mark.asyncio
@pytest.mark.timeout(240)
@_handle_project
async def test_off_the_prompt_names_no_library_and_ends_with_the_suffix(minimal, monkeypatch):
    monkeypatch.setattr(SETTINGS, "UNIFY_LIBRARY", "off")
    monkeypatch.setattr(SETTINGS, "UNIFY_SYSTEM_SUFFIX", SUFFIX)
    monkeypatch.setattr(SETTINGS, "UNIFY_SYSTEM_SUFFIX_SHA256", SUFFIX_SHA)
    first = await _first_request()
    system = _system(first)
    assert not [mark for mark in LIBRARY_TEXT if mark in system], system[:2500]
    assert system.endswith("\n\n" + SUFFIX)
    first_user = json.dumps([m for m in first["messages"] if m.get("role") == "user"][:1])
    assert "Library at task start" not in first_user


# --- end to end ------------------------------------------------------------------------------------------------------


@needs_bwrap
@pytest.mark.asyncio
@pytest.mark.timeout(400)
async def test_off_the_sandbox_has_no_library_tools_keeps_the_rest_and_no_review_runs(minimal, monkeypatch):
    monkeypatch.setattr(SETTINGS, "UNIFY_LIBRARY", "off")
    model = ScriptedModel(actor=[reply(calls=[code(PROBE)]), reply("done")])
    # The actor may store by default: with the library off it does not, so no review call is ever made (an
    # unscripted kind fails the test).
    assert await _act(model, "Probe the sandbox.") == "done", _why(model)
    assert model.kinds() == ["actor", "actor"]
    output = _tool_results(model.calls[1])
    names, bound = set(_line(output, "TOOLS")), _line(output, "BOUND")
    assert "store_skills" not in names and "ask_library" not in names, sorted(names)
    assert {"read", "grep", "install", "sh", "check_examples", "answer"} <= names, sorted(names)
    assert bound == [], output
    system = "\n".join(_system_texts(model.calls[0]))
    assert not [mark for mark in LIBRARY_TEXT if mark in system], system[:2500]
