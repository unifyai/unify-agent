"""Symbolic: ``UNIFY_PLACEHOLDER_NOTE`` names a credential argument that received a stand-in.

In an AppWorld HIGH cell the model held a fresh token in the session variable
``access_token`` and called a stored function through ``execute_function``
with ``{"access_token": "{{access_token}}"}``; ``call_kwargs`` are passed as
written, so the function received those 16 characters and both calls failed
with 401. With the switch on, the tool's result carries a note naming the
argument and the stand-in it received, the call still runs, and the tool's
description says that values are literals. A real-looking value gets no note
and is never echoed. Off: no note, and the description as shipped.
Functions run in-process through the actor's own session executor; no model
is called.
"""

from __future__ import annotations

from typing import Any

import pytest

from tests.helpers import _handle_project
from unify.actor import placeholder_note
from unify.actor.code_act_actor import CodeActActor
from unify.common.llm_helpers import method_to_schema
from unify.function_manager.function_manager import FunctionManager
from unify.settings import SETTINGS

LOGIN = (
    "def fetch_profile(access_token: str) -> dict:\n"
    "    if not access_token.isalnum():\n"
    "        raise PermissionError(\n"
    "            f'401 Unauthorized: {access_token!r} is not a token'\n"
    "        )\n"
    "    return {'user': 'ada'}\n"
)
COUNT = (
    "def count_playlists(password: str, limit: int = 5) -> int:\n" "    return limit\n"
)
REAL_TOKEN = "eyJhbGciOiJIUzI1NiJ9realtoken42"
PLACEHOLDER_NOTE = (
    "`access_token` received the literal text `{{access_token}}`. "
    "`call_kwargs` values are passed as written and session variables are "
    "not substituted; to pass a variable, call the function from "
    "`execute_code`."
)


def _note(out: Any) -> Any:
    return out.get("note") if isinstance(out, dict) else getattr(out, "note", None)


def _error(out: Any) -> Any:
    return out.get("error") if isinstance(out, dict) else getattr(out, "error", None)


def _result(out: Any) -> Any:
    return out.get("result") if isinstance(out, dict) else getattr(out, "result", None)


def _seen_by_model(out: Any) -> str:
    """The text the model reads for this result."""
    if hasattr(out, "to_llm_content"):
        return "".join(block.get("text", "") for block in out.to_llm_content())
    return repr(out)


class _Actor:
    """A CodeActActor over ``fm`` whose ``execute_function`` tool is called directly."""

    def __init__(self, fm: FunctionManager):
        self.actor = CodeActActor(function_manager=fm, can_store=False)
        tool = self.actor.get_tools("act")["execute_function"]
        self.tool = getattr(tool, "fn", tool)

    async def call(self, function_name: str, **call_kwargs: Any) -> Any:
        return await self.tool(
            thought="Reusing a stored function.",
            function_name=function_name,
            call_kwargs=call_kwargs,
        )

    async def close(self) -> None:
        await self.actor.close()


@pytest.fixture
def note_on(monkeypatch):
    monkeypatch.setattr(SETTINGS, "UNIFY_PLACEHOLDER_NOTE", True)


async def _call(function_name: str, **call_kwargs: Any) -> Any:
    fm = FunctionManager(include_primitives=False)
    fm.add_functions(implementations=[LOGIN, COUNT])
    actor = _Actor(fm)
    try:
        return await actor.call(function_name, **call_kwargs)
    finally:
        await actor.close()


# --------------------------------------------------------------------------- #
#  The note                                                                    #
# --------------------------------------------------------------------------- #


@_handle_project
@pytest.mark.asyncio
async def test_a_placeholder_argument_gets_a_note_and_the_call_still_runs(note_on):
    """The AppWorld case: the function ran with the literal text and failed; the result says why."""
    out = await _call("fetch_profile", access_token="{{access_token}}")
    # the call ran as given: the function itself raised on the literal text
    assert "PermissionError: 401 Unauthorized: '{{access_token}}'" in _error(out)
    assert _note(out) == PLACEHOLDER_NOTE
    assert PLACEHOLDER_NOTE in _seen_by_model(out)


@_handle_project
@pytest.mark.asyncio
async def test_a_returning_call_with_a_stand_in_also_carries_the_note(note_on):
    out = await _call("count_playlists", password="", limit=3)
    assert (_result(out), _error(out)) == (3, None)
    assert _note(out) == (
        "`password` received an empty string. `call_kwargs` values are passed "
        "as written and session variables are not substituted; to pass a "
        "variable, call the function from `execute_code`."
    )


@_handle_project
@pytest.mark.asyncio
async def test_a_real_looking_value_gets_no_note_and_is_never_echoed(note_on):
    out = await _call("fetch_profile", access_token=REAL_TOKEN)
    assert (_result(out), _error(out), _note(out)) == ({"user": "ada"}, None, None)
    assert "note" not in _seen_by_model(out)
    # beside a stand-in, only the stand-in is named
    out = await _call("count_playlists", password=REAL_TOKEN, limit="x")
    assert _note(out) is None  # `limit` is not a credential
    note = placeholder_note.note(
        {"access_token": "{{access_token}}", "password": REAL_TOKEN},
    )
    assert note.startswith("`access_token` received the literal text")
    assert REAL_TOKEN not in note and "password" not in note


@_handle_project
@pytest.mark.asyncio
async def test_off_the_result_has_no_note(monkeypatch):
    monkeypatch.setattr(SETTINGS, "UNIFY_PLACEHOLDER_NOTE", False)
    out = await _call("fetch_profile", access_token="{{access_token}}")
    assert "PermissionError: 401 Unauthorized" in _error(out)
    assert _note(out) is None
    assert '"note"' not in _seen_by_model(out)


# --------------------------------------------------------------------------- #
#  The description                                                             #
# --------------------------------------------------------------------------- #


def _docs(monkeypatch, on: bool) -> dict:
    monkeypatch.setattr(SETTINGS, "UNIFY_PLACEHOLDER_NOTE", on)
    actor = CodeActActor(function_manager=FunctionManager(include_primitives=False))
    tools = actor.get_tools("act")
    docs = {}
    for name in ("execute_code", "execute_function"):
        tool = tools[name]
        fn = getattr(tool, "fn", tool)
        docs[name] = method_to_schema(fn, tool_name=name)["function"]["description"]
    return docs


@_handle_project
def test_on_the_description_says_values_are_literals(monkeypatch):
    off = _docs(monkeypatch, False)
    on = _docs(monkeypatch, True)
    sentence = " ".join(placeholder_note.DOC_SENTENCE.split())
    assert "session variables are not substituted" not in off["execute_function"]
    # one sentence, after call_kwargs's description, and nothing else changes
    anchor = "which fails type validation at the callee)."
    assert " ".join(on["execute_function"].split()) == " ".join(
        off["execute_function"].split(),
    ).replace(anchor, f"{anchor} {sentence}")
    assert sentence.startswith("Values are literals: session variables are not")
    assert on["execute_code"] == off["execute_code"]


# --------------------------------------------------------------------------- #
#  The detector                                                                #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("parameter", "value"),
    [
        ("access_token", "{{access_token}}"),
        ("access_token", "{{session.spotify_token}}"),
        ("access_token", "$spotify_token"),
        ("access_token", "<hidden-token>"),
        ("access_token", "access_token"),
        ("access_token", ""),
        ("password", " "),
        ("password", "unknown"),
        ("access_token", "x"),
        ("access_token", "token"),
        ("apiKey", "API_KEY"),
    ],
)
def test_stand_ins_seen_in_the_logs_are_recognised(parameter, value):
    assert placeholder_note.stand_in(parameter, value)


@pytest.mark.parametrize(
    ("parameter", "value"),
    [
        ("access_token", REAL_TOKEN),
        ("password", "hunter2pw"),
        ("password", None),
        ("sort_key", ""),
        ("keyword", "x"),
        ("author", ""),
        ("title", "{{name}}"),
    ],
)
def test_real_values_and_ordinary_parameters_are_not_stand_ins(parameter, value):
    assert not placeholder_note.stand_in(parameter, value)


def test_a_long_stand_in_is_shortened_when_echoed(monkeypatch):
    monkeypatch.setattr(SETTINGS, "UNIFY_PLACEHOLDER_NOTE", True)
    note = placeholder_note.note({"token": "{{" + "a" * 200 + "}}"})
    echoed = note.split("`")[3]
    assert len(echoed) == 60 and echoed.endswith("…")


def test_the_setting_parses_as_a_boolean():
    from unify.settings import ProductionSettings

    assert ProductionSettings().UNIFY_PLACEHOLDER_NOTE is False
    for value, expected in (("true", True), ("1", True), ("false", False), ("", False)):
        assert (
            ProductionSettings(UNIFY_PLACEHOLDER_NOTE=value).UNIFY_PLACEHOLDER_NOTE
            is expected
        )
