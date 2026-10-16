"""``UNIFY_CELL_STATE=deployment``: the actor's top-level names are kept at each hand-back and come back in later
requests and after a restart (unify/actor/state_across_requests.py).

The save and restore sources run here in-process on a namespace built the way the worker builds it (the module
``__sandbox_worker__``, each cell's source under ``<cell N>`` in ``linecache``), and end to end in the real sandboxed
worker with a scripted model (tests/scripted_model.py): no key, no network.
"""

from __future__ import annotations

import ast
import asyncio
import json
import linecache
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest

from tests.actor.code_act.core_world import core_world  # noqa: F401 (fixture)
from tests.actor.code_act.sandbox_world import needs_bwrap, world  # noqa: F401 (fixture)
from tests.actor.code_act.test_in_code_tools_wiring import (  # noqa: F401 (fixture)
    _act,
    _assert_ended,
    _rows,
    _system_texts,
    _tool_results,
    _why,
    code,
    in_code,
)
from tests.scripted_model import ScriptedModel, reply
from unify.actor import drafts
from unify.actor import state_across_requests as kept
from unify.common import box_interpreter, tool_registry
from unify.settings import SETTINGS

ROOT = Path(__file__).resolve().parents[3]
SENTENCE = (
    "Top-level names (imports, functions, classes, and values that can be saved) are also kept each time you hand "
    "back an answer, and come back in later requests and after a restart; the first cell's output then names "
    "anything that could not be kept."
)


# --- an in-process namespace like the worker's ----------------------------------------------------------------------


class Box:
    """A namespace as the worker keeps it: the module ``__sandbox_worker__``, cells compiled under ``<cell N>``."""

    def __init__(self) -> None:
        for key in [k for k in linecache.cache if isinstance(k, str) and k.startswith(("<cell ", "<restored "))]:
            del linecache.cache[key]
        self.module = types.ModuleType("__sandbox_worker__")
        sys.modules["__sandbox_worker__"] = self.module
        self.ns = self.module.__dict__
        self.ns["tools"] = object()
        self.cells = 0

    def cell(self, source: str) -> None:
        self.cells += 1
        key = f"<cell {self.cells}>"
        linecache.cache[key] = (len(source), None, source.splitlines(True), key)
        exec(compile(source, key, "exec"), self.ns)

    def run(self, source: str, *, scratch: bool):
        ns = dict(self.ns) if scratch else self.ns
        exec(compile(source, "<string>", "exec"), ns)
        try:
            return asyncio.run(ns["__exec_wrapper"]())
        finally:
            ns.pop("__exec_wrapper", None)


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    yield tmp_path
    sys.modules.pop("__sandbox_worker__", None)


FIRST = (
    "import math\nfrom collections import Counter\nLIMIT = 7\n",
    "def area(r, k=LIMIT):\n    return math.pi * r * r + k\n\nclass Grid:\n    def __init__(self, rows):\n"
    "        self.rows = rows\n    def width(self):\n        return len(self.rows[0])\n",
    "g = Grid([[1, 2, 3]])\ncounts = Counter('abca')\nsq = lambda x: x * x\ngen = (i for i in range(3))\n"
    "alias = area\n_private = 1\ngone = 5\ndel gone\n",
)


def test_a_save_keeps_names_and_a_restore_in_a_new_namespace_brings_them_back(workspace):
    box = Box()
    for cell in FIRST:
        box.cell(cell)
    saved = box.run(kept.save_source(["tools"]), scratch=True)
    assert (saved["imports"], saved["defs"], saved["values"]) == (2, 2, 4), saved
    assert sorted(saved["not_kept"]) == [["gen", "generator"], ["sq", "function"]], saved
    assert "__exec_wrapper" not in box.ns and "gone" not in box.ns   # a scratch run changes nothing
    manifest = json.loads((workspace / kept.DIR / f"{saved['snapshot']}.json").read_text())
    assert "_private" not in manifest["values"] and "tools" not in manifest["values"], manifest

    later = Box()   # a later request: a new interpreter
    restored = later.run(kept.RESTORE, scratch=False)
    ns = later.ns
    assert (restored["imports"], restored["defs"], restored["values"], restored["failed"]) == (2, 2, 4, []), restored
    assert round(ns["area"](1), 3) == round(3.141592653589793 + 7, 3)   # a default that needs a value: the retry
    assert ns["g"].width() == 3 and dict(ns["counts"]) == {"a": 2, "b": 1, "c": 1} and ns["alias"] is ns["area"]
    assert not {"sq", "gen", "_private", "gone"} & set(ns), sorted(ns)
    assert kept.not_kept_note(restored, None) == "(not kept from earlier requests: sq (function), gen (generator))"
    # the drafts probe reads <cell N> sources only: a restored definition is never drafted again
    probed = later.run(drafts.PROBE, scratch=True)
    assert probed["defs"] == {}, probed


def test_a_restored_definition_is_saved_again_and_a_redefinition_replaces_it(workspace):
    box = Box()
    box.cell("def f():\n    return 1\n")
    box.run(kept.save_source(["tools"]), scratch=True)
    second = Box()
    second.run(kept.RESTORE, scratch=False)
    second.cell("def g():\n    return f() + 1\n")
    second.run(kept.save_source(["tools"]), scratch=True)
    third = Box()
    third.run(kept.RESTORE, scratch=False)
    assert third.ns["g"]() == 2
    third.cell("def f():\n    return 10\n")
    third.run(kept.save_source(["tools"]), scratch=True)
    fourth = Box()
    fourth.run(kept.RESTORE, scratch=False)
    assert fourth.ns["g"]() == 11
    # one snapshot is kept, beside the pointer
    names = sorted(p.name for p in (workspace / kept.DIR).iterdir())
    assert len(names) == 3 and "latest.json" in names, names


def test_no_snapshot_restores_nothing_and_a_broken_one_is_reported(workspace):
    box = Box()
    assert box.run(kept.RESTORE, scratch=False) == {"snapshot": None}
    assert kept.not_kept_note({"snapshot": None}, None) is None
    (workspace / kept.DIR).mkdir()
    (workspace / kept.DIR / "latest.json").write_text(json.dumps({"snapshot": "missing"}))
    with pytest.raises(FileNotFoundError):
        Box().run(kept.RESTORE, scratch=False)
    assert kept.not_kept_note(None, "FileNotFoundError") == (
        "(the names kept from earlier requests could not be restored: FileNotFoundError)"
    )


def test_the_host_never_unpickles():
    """pickle is used only inside the two sources, which run in the box: no host module of this switch imports it."""
    for rel in ("unify/actor/state_across_requests.py", "unify/actor/in_code_tools.py", "unify/common/box_interpreter.py"):
        tree = ast.parse((ROOT / rel).read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                assert all(a.name != "pickle" for a in node.names), rel
            elif isinstance(node, ast.ImportFrom):
                assert node.module != "pickle", rel
    assert "pickle" in kept.RESTORE and "pickle" in kept.save_source([])


def test_the_setting_is_refused_without_tools_in_code_and_counts_as_persistent(monkeypatch):
    on = SimpleNamespace(UNIFY_CELL_STATE="deployment", UNIFY_TOOLS_IN_CODE="on")
    off = SimpleNamespace(UNIFY_CELL_STATE="deployment", UNIFY_TOOLS_IN_CODE="off")
    assert kept.refusal(on) is None
    assert "UNIFY_TOOLS_IN_CODE must be on" in kept.refusal(off)
    assert kept.refusal(SimpleNamespace(UNIFY_CELL_STATE="persistent", UNIFY_TOOLS_IN_CODE="off")) is None
    monkeypatch.setattr(SETTINGS, "UNIFY_CELL_STATE", "deployment")
    assert box_interpreter.persistent() and box_interpreter.deployment()
    monkeypatch.setattr(SETTINGS, "UNIFY_TOOLS_IN_CODE", "off")
    from unify.cli import _profile_refusal

    assert "UNIFY_CELL_STATE=deployment" in (_profile_refusal() or "")


def test_only_the_actor_reads_the_deployment_paragraph():
    actor = tool_registry.sandbox_paragraph("actor", ("tools",), "deployment")
    mate = tool_registry.sandbox_paragraph("teammate", ("tools",), "deployment")
    assert SENTENCE in " ".join(actor.split()) and "variables are gone" not in actor
    assert SENTENCE not in " ".join(mate.split()) and mate.startswith(tool_registry.SANDBOX_PERSISTENT)
    assert tool_registry.sandbox_paragraph("actor", ("tools",), True).startswith(tool_registry.SANDBOX_PERSISTENT)


# --- end to end in the real worker ----------------------------------------------------------------------------------


@needs_bwrap
@pytest.mark.asyncio
@pytest.mark.timeout(600)
async def test_names_carry_over_to_the_next_request_and_what_could_not_be_kept_is_named(in_code, monkeypatch):
    monkeypatch.setattr(SETTINGS, "UNIFY_CELL_STATE", "deployment")
    define = (
        "def area(r):\n    return 3 * r * r\n\nx = 41\nz = 1\ndel z\ngen = (i for i in range(2))\nprint('DEFINED')\n"
    )
    first = ScriptedModel(actor=[reply(calls=[code(define)]), reply("done")])
    assert await _act(first, "Define things.") == "done", _why(first)
    assert SENTENCE in " ".join(_system_texts(first.calls[0])[0].split())
    use = "print('NEXT', x + 1, area(2))\ntry:\n    z\nexcept NameError:\n    print('NO Z')\n"
    second = ScriptedModel(actor=[reply(calls=[code(use)]), reply("done")])
    assert await _act(second, "Use them.") == "done", _why(second)
    shown = _tool_results(second.calls[1])
    assert "NEXT 42 12" in shown and "NO Z" in shown, shown
    assert "(not kept from earlier requests: gen (generator))" in shown, shown
    saved = [r for s in in_code["opened"][:1] for r in _rows(s) if r.get("type") == "state_saved"]
    restored = [r for s in in_code["opened"][1:] for r in _rows(s) if r.get("type") == "state_restored"]
    assert saved and saved[-1]["defs"] == 1 and saved[-1]["values"] == 1 and saved[-1]["not_kept"] == 1, saved
    assert restored and restored[0]["defs"] == 1 and restored[0]["values"] == 1, restored
    _assert_ended(in_code["opened"])


@needs_bwrap
@pytest.mark.asyncio
@pytest.mark.timeout(600)
async def test_a_restart_restores_the_names_of_the_last_hand_back(in_code, monkeypatch):
    monkeypatch.setattr(SETTINGS, "UNIFY_CELL_STATE", "deployment")
    first = ScriptedModel(actor=[reply(calls=[code("x = 41\nprint('SET')\n")]), reply("done")])
    assert await _act(first, "Set x.") == "done", _why(first)
    second = ScriptedModel(
        actor=[
            reply(calls=[code("y = 1\nprint('Y')\n")]),
            reply(calls=[code("import os\nos._exit(3)\n")]),
            reply(calls=[code("print('AFTER', x)\ntry:\n    y\nexcept NameError:\n    print('NO Y')\n")]),
            reply("done"),
        ],
    )
    assert await _act(second, "Crash once.") == "done", _why(second)
    after = _tool_results(second.calls[3])
    assert "AFTER 41" in after and "NO Y" in after, after
    crashed = _tool_results(second.calls[2])
    assert kept.LOST in crashed or kept.LOST in after, (crashed, after)
    _assert_ended(in_code["opened"])
