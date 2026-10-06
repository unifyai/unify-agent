"""Symbolic: ``UNIFY_EXECUTE_FUNCTION_HINT=neutral`` drops the ``execute_function`` preference.

In the AppWorld runs read for TOKEN-401 (264 paid cells), all 58 "401
Unauthorized" errors raised inside a stored function came from an
``execute_function`` call that passed a stand-in (``"{{access_token}}"``,
``"..."``) where the session's token belonged. ``execute_function`` takes
literal JSON, so it cannot name a session variable. The prompt told the
model to use it for any one exact call and ``execute_code`` "only when the
task genuinely requires multi-step composition" (upstream 0430b159e).

With the switch, each passage that says so states instead that either tool
can run a stored function, that ``execute_code`` can pass live session
values by name, and that ``execute_function`` takes literal values only;
the code tools' descriptions drop their preference as the lean profile's
do. Off, the prompt is as shipped (the switches-off golden pins the actor's
first request). The transport is scripted, so nothing leaves the process.
"""

from __future__ import annotations

import pytest

from tests import cache_discipline_helpers as h
from unify.actor import prompt_builders as pb
from unify.settings import ProductionSettings, SETTINGS

EITHER = (
    "Either `execute_code` or `execute_function` can run a stored function.\n"
    "`execute_code` can pass live session values by name, such as a variable\n"
    "holding a token; `execute_function` takes literal values only."
)

# The intended diff, written out: each shipped passage and what replaces it.
LIBRARY_CALL = (
    "function via `execute_function`, follow relevant guidance.",
    "function, follow relevant guidance.",
)
LIBRARY_PATH = (
    "choose the minimal correct execution path —\n"
    "if the request or discovery step already identifies one exact function\n"
    "or primitive call, use `execute_function`; use `execute_code` only\n"
    "when the task genuinely requires multi-step composition.",
    "choose an execution path.\n" + EITHER,
)
SEARCH_CALL = (
    "you find, call a relevant function via `execute_function` and follow\n",
    "you find, call a relevant function and follow\n",
)
SEARCH_PATH = (
    "via update/re-link. Choose the minimal correct execution path — if the\n"
    "request or a search result already identifies one exact function or\n"
    "primitive call, use `execute_function`; use `execute_code` only when\n"
    "the task genuinely requires multi-step composition.",
    "via update/re-link. " + EITHER,
)
GATE_STEP = (
    "2. Then choose the minimal correct execution path:\n"
    "   if one exact function or primitive call is enough, use execute_function;\n"
    "   use execute_code only when the task genuinely needs multi-step\n"
    "   composition, branching, iteration, or combining intermediate results.",
    "2. Then choose an execution path. Either `execute_code` or `execute_function`"
    " can run a stored function.\n"
    "   `execute_code` can pass live session values by name, such as a variable\n"
    "   holding a token; `execute_function` takes literal values only.",
)
TOOL_SELECTION = (
    "- One exact function or primitive call is\n"
    '  `execute_function(function_name="...", call_kwargs={...})`. Reach\n'
    "  for `execute_code` only for genuine multi-step composition\n"
    "  (branching, loops, combining intermediate results); a\n"
    "  `print()`, `await handle.result()`, or temporary variable around a\n"
    "  single call is boilerplate, not composition.",
    "- Either tool can run a stored function or primitive:\n"
    '  `execute_function(function_name="...", call_kwargs={...})` takes\n'
    "  literal values only; `execute_code` can also pass live session\n"
    "  values by name, such as a variable holding a token.",
)

PREFERENCES = (
    "use `execute_code` only",
    "use execute_code only",
    "Reach\n  for `execute_code` only",
    "minimal correct execution path",
)

CONFIGS = {
    # (lean profile, build_code_act_prompt kwargs, the passages it holds)
    "shipped-gate": (
        False,
        {"discovery_first_policy": True},
        (TOOL_SELECTION, LIBRARY_CALL, LIBRARY_PATH, GATE_STEP),
    ),
    "shipped-search-when-useful": (
        False,
        {"search_when_useful": True},
        (TOOL_SELECTION, SEARCH_CALL, SEARCH_PATH),
    ),
    # The AppWorld screen's shape: lean profile, no discovery gate.
    "lean-search-when-useful": (
        True,
        {"search_when_useful": True},
        (SEARCH_CALL, SEARCH_PATH),
    ),
    "lean-gate": (
        True,
        {"discovery_first_policy": True},
        (LIBRARY_CALL, LIBRARY_PATH, GATE_STEP),
    ),
}


def _prompt(monkeypatch, *, hint: str, lean: bool, **kwargs) -> str:
    from unify.actor.code_act_actor import CodeActActor

    monkeypatch.setattr(SETTINGS, "UNIFY_EXECUTE_FUNCTION_HINT", hint)
    monkeypatch.setattr(SETTINGS, "UNIFY_PROMPT_PROFILE", "lean" if lean else "")
    actor = CodeActActor()
    return pb.build_code_act_prompt(
        environments={},
        tools=dict(actor.get_tools("act")),
        can_store=True,
        **kwargs,
    )


def test_setting_defaults_off_and_rejects_unknown_values():
    assert ProductionSettings().UNIFY_EXECUTE_FUNCTION_HINT == ""
    assert (
        ProductionSettings(
            UNIFY_EXECUTE_FUNCTION_HINT=" Neutral ",
        ).UNIFY_EXECUTE_FUNCTION_HINT
        == "neutral"
    )
    with pytest.raises(ValueError):
        ProductionSettings(UNIFY_EXECUTE_FUNCTION_HINT="prefer_code")


@pytest.mark.parametrize("config", sorted(CONFIGS))
def test_on_the_prompt_differs_by_exactly_the_intended_passages(config, monkeypatch):
    lean, kwargs, passages = CONFIGS[config]
    off = _prompt(monkeypatch, hint="", lean=lean, **kwargs)
    on = _prompt(monkeypatch, hint="neutral", lean=lean, **kwargs)

    expected = off
    for old, new in passages:
        assert off.count(old) == 1, old
        expected = expected.replace(old, new)
    assert on == expected
    for phrase in PREFERENCES:
        assert phrase not in on, phrase
    flat = " ".join(on.split())
    assert flat.count("takes literal values only") == len(
        [p for p in passages if "literal values only" in " ".join(p[1].split())],
    )


@pytest.mark.parametrize("config", sorted(CONFIGS))
def test_off_the_prompt_keeps_every_shipped_passage(config, monkeypatch):
    lean, kwargs, passages = CONFIGS[config]
    off = _prompt(monkeypatch, hint="", lean=lean, **kwargs)
    for old, _new in passages:
        assert off.count(old) == 1, old
    assert "literal values only" not in off


def test_on_without_execute_code_the_prompt_is_unchanged(monkeypatch):
    """A function-only actor has nothing to choose between."""
    from unify.actor.code_act_actor import CodeActActor

    tools = {
        name: tool
        for name, tool in CodeActActor().get_tools("act").items()
        if name != "execute_code"
    }

    def prompt(hint: str) -> str:
        monkeypatch.setattr(SETTINGS, "UNIFY_EXECUTE_FUNCTION_HINT", hint)
        return pb.build_code_act_prompt(
            environments={},
            tools=tools,
            discovery_first_policy=True,
        )

    assert prompt("neutral") == prompt("")


def _tool_docs(monkeypatch, *, hint: str, lean: bool) -> dict[str, str]:
    from unify.actor.code_act_actor import CodeActActor

    monkeypatch.setattr(SETTINGS, "UNIFY_EXECUTE_FUNCTION_HINT", hint)
    monkeypatch.setattr(SETTINGS, "UNIFY_PROMPT_PROFILE", "lean" if lean else "")
    tools = CodeActActor().get_tools("act")
    docs = {}
    for name in ("execute_code", "execute_function"):
        tool = tools[name]
        docs[name] = (getattr(tool, "fn", None) or tool).__doc__
    return docs


SINGLE_CALL_RULE = "**IMPORTANT — single-call rule**"
PREFERRED_TOOL = "**This is the preferred tool for any task that maps to a single"


def test_on_the_code_tools_drop_their_preference_as_the_lean_profile_does(
    monkeypatch,
):
    off = _tool_docs(monkeypatch, hint="", lean=False)
    on = _tool_docs(monkeypatch, hint="neutral", lean=False)
    lean = _tool_docs(monkeypatch, hint="", lean=True)
    lean_on = _tool_docs(monkeypatch, hint="neutral", lean=True)

    from unify.actor import code_act_actor as caa

    assert SINGLE_CALL_RULE in off["execute_code"]
    assert PREFERRED_TOOL in off["execute_function"]
    assert SINGLE_CALL_RULE not in on["execute_code"]
    assert PREFERRED_TOOL not in on["execute_function"]
    # The lean profile's two rewrites of these passages, and nothing more.
    for name, doc in off.items():
        for pattern, replacement in caa._LEAN_TOOL_DOCS:
            doc = pattern.sub(replacement, doc)
        assert on[name] == doc
    # Under the lean profile, which already drops them, the switch adds nothing.
    assert lean_on == lean


@pytest.mark.asyncio
async def test_on_the_first_request_differs_only_in_those_passages(monkeypatch):
    """The actor's own first request, as a caller runs it."""
    from unify.actor.code_act_actor import CodeActActor

    async def first_request(hint: str) -> dict:
        monkeypatch.setattr(SETTINGS, "UNIFY_EXECUTE_FUNCTION_HINT", hint)
        actor = CodeActActor()
        try:
            with h.scripted(h.ACTOR_REPLIES) as provider:
                handle = await actor.act(
                    "List the files in the workspace.",
                    persist=False,
                )
                await handle.result()
        finally:
            await actor.close()
        return provider.requests[0]

    off = await first_request("")
    on = await first_request("neutral")

    system_off = off["messages"][0]["content"]
    system_on = on["messages"][0]["content"]
    held = [
        p
        for p in (
            TOOL_SELECTION,
            LIBRARY_CALL,
            LIBRARY_PATH,
            GATE_STEP,
            SEARCH_CALL,
            SEARCH_PATH,
        )
        if p[0] in system_off
    ]
    assert held, "the first request held no preference passage"
    expected = system_off
    for old, new in held:
        expected = expected.replace(old, new)
    assert system_on == expected

    def descriptions(request: dict) -> dict[str, str]:
        return {
            t["function"]["name"]: t["function"].get("description", "")
            for t in request["tools"] or []
        }

    d_off, d_on = descriptions(off), descriptions(on)
    assert d_off.keys() == d_on.keys()
    changed = {name for name in d_off if d_off[name] != d_on[name]}
    assert changed <= {"execute_code", "execute_function"}
    for name in changed:
        assert SINGLE_CALL_RULE not in d_on[name]
        assert PREFERRED_TOOL not in d_on[name]
