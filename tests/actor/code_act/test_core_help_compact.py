"""Symbolic and sandboxed: ``UNIFY_CORE_HELP_COMPACT``: a short, correct ``help()`` in core-surface cells.

As shipped ``help(functions.search)`` prints 3,448 characters: the full
contract, about 1.2k of them on the ranking formula, the text of three
parameters only the harness passes, and (until the include_dormant fix) a
parameter the method does not take; only ``run`` and ``install`` carry an
example, and ``functions.run`` defaults to ``"stateless"`` while every cell
runs in the session. With the switch on ``help(obj)`` prints a call with
parameter names and defaults, the first sentence and one example, and
``help(obj, full=True)`` the full contract without the harness-only text;
with UNIFY_STATEFUL_CELLS too, ``functions.run`` runs in the session unless
told otherwise.
"""

from __future__ import annotations

import ast
import asyncio
import inspect
import json

import pytest

from tests import cache_discipline_helpers as h
from tests.actor.code_act.core_world import (  # noqa: F401 (fixtures)
    core_world,
    new_actor as _actor,
    world,
)
from tests.actor.code_act.sandbox_world import needs_bwrap
from tests.helpers import _handle_project
from unify.actor import core_surface as cs
from unify.settings import ProductionSettings, SETTINGS

PRIVATE = ("_return_callable", "_namespace", "_also_return_metadata", "include_dormant")


@pytest.fixture
def compact(monkeypatch):
    monkeypatch.setattr(SETTINGS, "UNIFY_CORE_HELP_COMPACT", True)


def _objects():
    """``{label: object}`` for every harness object and method help() documents."""
    cs._document()
    functions = cs.FunctionLibrary.__new__(cs.FunctionLibrary)
    guidance = cs.GuidanceLibrary.__new__(cs.GuidanceLibrary)
    out = {"functions": functions, "guidance": guidance, "install": cs.install}
    out.update(cs._file_tools())
    for label, obj in (("functions", functions), ("guidance", guidance)):
        for attr in dir(obj):
            if not attr.startswith("_") and callable(getattr(obj, attr)):
                out[f"{label}.{attr}"] = getattr(obj, attr)
    return out


def test_off_by_default_and_help_is_as_shipped():
    assert ProductionSettings().UNIFY_CORE_HELP_COMPACT is False
    for label, obj in _objects().items():
        assert cs.help_text(obj, label) == cs._help_text(obj, label), label
    assert cs.run_state() == "stateless"
    assert '`run(name, state="stateless", **kwargs)`' in cs.PromptSurface().index()


def test_every_harness_object_has_one_example():
    labels = set(_objects()) | {"request_clarification"}
    assert labels - set(cs.HELP_EXAMPLES) == set()
    assert set(cs.HELP_EXAMPLES) - labels == set()


def test_each_example_calls_with_real_parameters():
    objects = _objects()
    for label, example in cs.HELP_EXAMPLES.items():
        obj = objects.get(label)
        tree = ast.parse(example.split("#")[0].strip())
        calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call)]
        target = ".".join(label.split(".")[1:]) if "." in label else label
        call = next((c for c in calls if ast.unparse(c.func).endswith(target)), None)
        if obj is None or not inspect.isroutine(obj):
            continue
        assert call is not None, label
        sig = inspect.signature(obj)
        bound = sig.bind(
            *[None] * len(call.args),
            **{k.arg: None for k in call.keywords if k.arg},
        )
        assert bound, label


def test_compact_help_is_short_and_says_how_to_see_more(compact):
    for label, obj in _objects().items():
        text = cs.help_text(obj, label)
        assert len(text) <= 450, (label, len(text), text)
        assert f"help({label}, full=True)" in text, label
        assert f"Example: {cs.HELP_EXAMPLES[label]}" in text, label
        for word in PRIVATE:
            assert word not in text, (label, word)
    search = cs.help_text(_objects()["functions.search"], "functions.search")
    assert search.splitlines()[0] == (
        "await functions.search(query='', n=5, include_implementations=True)"
    )


def test_full_help_has_no_harness_only_text(compact):
    objects = _objects()
    for label, obj in objects.items():
        full = cs.help_text(obj, label, full=True)
        for word in PRIVATE:
            assert word not in full, (label, word)
    search = cs.help_text(objects["functions.search"], "functions.search", full=True)
    shipped = cs._help_text(objects["functions.search"], "functions.search")
    assert len(search) < len(shipped)
    assert "- Up to ``n`` results, best match first." in search
    assert "Raises" not in search


def test_run_runs_in_the_session_only_with_stateful_cells(compact, monkeypatch):
    run = _objects()["functions.run"]
    assert cs.run_state() == "stateless"
    assert "state='stateless'" in cs.help_text(run, "functions.run")
    monkeypatch.setattr(SETTINGS, "UNIFY_STATEFUL_CELLS", True)
    assert cs.run_state() == "stateful"
    assert "state='stateful'" in cs.help_text(run, "functions.run")
    full = cs.help_text(run, "functions.run", full=True)
    assert "state: str = 'stateful'" in full
    assert '``"stateful"`` (the default)' in " ".join(full.split())
    index = cs.PromptSurface().index()
    assert '`run(name, state="stateful", **kwargs)`' in index
    assert "help(obj, full=True)" in index
    monkeypatch.setattr(SETTINGS, "UNIFY_CORE_HELP_COMPACT", False)
    assert cs.run_state() == "stateless"


# ── in the sandboxed worker ──────────────────────────────────────────────────

ADD_X = 'def add_x(y: int) -> int:\n    """Add the session\'s x."""\n    return x + y\n'


def _cell(code: str):
    return lambda: h.completion(
        calls=[("execute_code", {"thought": "Next step.", "code": code})],
    )


@needs_bwrap
@pytest.mark.asyncio
@pytest.mark.timeout(180)
@_handle_project
async def test_help_and_run_in_the_sandboxed_worker(core_world, monkeypatch):
    monkeypatch.setattr(SETTINGS, "UNIFY_CORE_HELP_COMPACT", True)
    monkeypatch.setattr(SETTINGS, "UNIFY_STATEFUL_CELLS", True)
    actor = _actor(can_store=False)
    actor.function_manager.add_functions(implementations=[ADD_X])
    replies = (
        _cell("help(functions.search)"),
        _cell("help(functions.search, full=True)"),
        _cell("help(functions.run)"),
        _cell("x = 3"),
        _cell("print('RUN', await functions.run('add_x', y=1))"),
        _cell(
            "print('ISOLATED', await functions.run('add_x', state='stateless', y=1))",
        ),
        lambda: h.completion(content="done"),
    )
    try:
        with h.scripted(replies) as provider:
            handle = await actor.act("Use the library.", persist=False)
            result = await asyncio.wait_for(handle.result(), 150)
    finally:
        await actor.close()
    assert result == "done"
    tool = [
        json.dumps(m["content"])
        for m in provider.requests[-1]["messages"]
        if m.get("role") == "tool"
    ]
    short, full, run = (json.loads(t)[-1]["text"] for t in tool[:3])
    assert short.startswith("await functions.search(query='', n=5")
    assert len(short) <= 450 and "full=True" in short
    assert len(full) > 1500 and "Parameters" in full
    assert "include_dormant" not in full and "_return_callable" not in full
    assert "state='stateful'" in run
    # The default runs in the session (sees x); "stateless" does not.
    assert "RUN 4" in tool[4], tool[4]
    assert "NameError" in tool[5] and "ISOLATED" not in tool[5], tool[5]
