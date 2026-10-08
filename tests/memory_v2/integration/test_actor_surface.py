"""UNIFY_MEMORY_V2 in the actor: no review, no library objects, the index last in the system prompt (Task 19).

The model is the scripted transport (tests/scripted_model.py): a request of any kind the test did not
script (a storage review, its gate, ...) fails the test, so ``kinds() == ["actor"]`` proves none ran.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from tests.actor.code_act.core_world import (  # noqa: F401 (fixtures)
    core_world,
    new_actor,
    world,
)
from tests.actor.code_act.sandbox_world import needs_bwrap
from tests.helpers import _handle_project
from tests.memory_v2.integration.test_checkout import MOD, _seed
from tests.scripted_model import ScriptedModel, reply, scripted
from unify.memory_v2.gitio import Repo
from unify.memory_v2.integration import hooks, prompt
from unify.memory_v2.integration import request as request_mod
from unify.memory_v2.integration.checkout import export_checkout
from unify.memory_v2.integration.paths import Paths
from unify.settings import SETTINGS

DOUBLE = "def double(x: int) -> int:\n    return x * 2\n"


def _run_for(home, tmp_path) -> SimpleNamespace:
    """A stand-in for Task 25's RequestRun: the export of a seeded memory and its index."""
    paths = Paths.under(home)
    mem, sha = _seed(tmp_path)
    export_checkout(mem.git_dir, sha, paths.checkout)
    return SimpleNamespace(index=prompt.render_index(paths.checkout), paths=paths)


async def _first_request(request: str = "Say hi.", **script):
    """One act() on a fresh actor (``can_store`` at its default, True) with a stored function."""
    actor = new_actor()
    actor.function_manager.add_functions(implementations=[DOUBLE])
    model = ScriptedModel(**(script or {"actor": [reply("ok")]}))
    try:
        with scripted(model):
            handle = await actor.act(request, persist=False)
            result = await asyncio.wait_for(handle.result(), 120)
    finally:
        await actor.close()
    first = model.calls[0]
    return (
        result,
        model,
        {
            "system": first.messages[0]["content"],
            "tools": first.request["tools"],
            "user": next(m["content"] for m in first.messages if m["role"] == "user"),
        },
    )


# ── the hooks alone ──────────────────────────────────────────────────────────


def test_hooks_are_inert_while_the_switch_is_off(monkeypatch, tmp_path):
    monkeypatch.setattr(SETTINGS, "UNIFY_MEMORY_V2", "")
    monkeypatch.setattr(
        request_mod,
        "_CURRENT",
        SimpleNamespace(index="IDX", paths=Paths.under(tmp_path)),
    )
    objects = {"functions": 1, "guidance": 2, "install": 3}
    assert hooks.sandbox_objects(objects) is objects
    marker = object()
    assert hooks.can_store(marker) is marker
    text = "system"
    assert hooks.system_prompt(text) is text
    assert hooks.worker_paths() == [] and hooks.worker_mounts() == []


def test_hooks_under_the_switch(monkeypatch, tmp_path):
    monkeypatch.setattr(SETTINGS, "UNIFY_MEMORY_V2", "on")
    monkeypatch.setattr(request_mod, "_CURRENT", None)
    assert hooks.sandbox_objects({"functions": 1, "guidance": 2, "install": 3}) == {
        "install": 3,
    }
    assert hooks.can_store(True) is False
    assert hooks.system_prompt("system") == "system"  # no run: nothing appended
    from unify.actor.prompt_builders import _CORE_SANDBOX_SEARCH_PYTHON

    pointer = f"Intro. {_CORE_SANDBOX_SEARCH_PYTHON} - do not guess."
    assert "functions" not in hooks.system_prompt(pointer)
    assert hooks.worker_paths() == [] and hooks.worker_mounts() == []
    paths = Paths.under(tmp_path)
    paths.checkout.mkdir(parents=True)  # the export is mounted only once it exists
    monkeypatch.setattr(
        request_mod,
        "_CURRENT",
        SimpleNamespace(index="IDX\n", paths=paths),
    )
    assert hooks.system_prompt("system") == "system\n\nIDX\n"
    assert hooks.worker_paths() == [str(paths.checkout)]
    assert hooks.worker_mounts() == [paths.checkout]


# ── the index ────────────────────────────────────────────────────────────────


def test_index_is_a_pure_function_of_the_commit(tmp_path):
    mem, sha = _seed(tmp_path)
    a, b = tmp_path / "a", tmp_path / "b"
    export_checkout(mem.git_dir, sha, a)
    first = prompt.render_index(a)
    (a / "env/spotify/__init__.py").touch()  # mtimes never matter
    export_checkout(mem.git_dir, sha, a)
    assert prompt.render_index(a) == first
    assert "## env.spotify" in first and "`hello(apis, name)` — Say hi." in first
    assert first.endswith(prompt.export_line(a))
    export_checkout(mem.git_dir, sha, b)
    assert prompt.render_index(b) == first.replace(str(a), str(b))
    assert "suspect" in prompt.render_index(a, {"spotify"})


def test_an_empty_memory_adds_nothing(tmp_path):
    mem = Repo.init_bare(tmp_path / "memory")
    export_checkout(mem.git_dir, mem.head(), tmp_path / "co")
    assert prompt.render_index(tmp_path / "co") == ""


def test_an_index_over_budget_is_left_out(tmp_path, monkeypatch):
    mem, sha = _seed(tmp_path)
    export_checkout(mem.git_dir, sha, tmp_path / "co")
    monkeypatch.setattr(prompt, "INDEX_BUDGET_TOKENS", 10)
    warned: list[str] = []
    monkeypatch.setattr(
        prompt.logger,
        "warning",
        lambda msg, *a: warned.append(msg % a),
    )
    assert prompt.render_index(tmp_path / "co") == ""
    assert warned and "index left out" in warned[0]


# ── the actor ────────────────────────────────────────────────────────────────


@needs_bwrap
@pytest.mark.asyncio
@pytest.mark.timeout(240)
@_handle_project
async def test_switch_on_offers_no_library_and_no_review(
    core_world,
    monkeypatch,
    tmp_path,
):
    monkeypatch.setattr(SETTINGS, "UNIFY_MEMORY_V2", "on")
    run = _run_for(core_world["state"], tmp_path)
    monkeypatch.setattr(request_mod, "_CURRENT", run)
    result, model, first = await _first_request()
    assert result == "ok"
    assert model.kinds() == ["actor"]  # no review, no gate
    assert first["system"].endswith("\n\n" + run.index)
    assert "Say hi." in run.index
    assert "functions." not in first["system"] and "guidance." not in first["system"]
    assert "Library at task start" not in first["user"]
    assert [t["function"]["name"] for t in first["tools"]] == ["execute_code"]


@needs_bwrap
@pytest.mark.asyncio
@pytest.mark.timeout(240)
@_handle_project
async def test_two_requests_on_one_commit_send_the_same_prefix(
    core_world,
    monkeypatch,
    tmp_path,
):
    monkeypatch.setattr(SETTINGS, "UNIFY_MEMORY_V2", "on")
    monkeypatch.setattr(
        request_mod,
        "_CURRENT",
        _run_for(core_world["state"], tmp_path),
    )
    _, _, one = await _first_request("Pay Ada back 5 dollars.")
    _, _, two = await _first_request("List my playlists.")
    assert one["system"] == two["system"] and one["tools"] == two["tools"]
    assert "Ada" not in one["system"] and "playlists" not in two["system"]


@needs_bwrap
@pytest.mark.asyncio
@pytest.mark.timeout(240)
@_handle_project
async def test_switch_off_is_unchanged_even_with_a_run_set(
    core_world,
    monkeypatch,
    tmp_path,
):
    monkeypatch.setattr(SETTINGS, "UNIFY_MEMORY_V2", "")
    script = {"actor": [reply("ok")], "allow": ("review", "gate")}
    _, _, baseline = await _first_request(**script)
    monkeypatch.setattr(
        request_mod,
        "_CURRENT",
        _run_for(core_world["state"], tmp_path),
    )
    _, _, with_run = await _first_request(**script)
    assert with_run == baseline
    assert "functions." in baseline["system"]  # the library objects are offered
    assert MOD.splitlines()[0] not in baseline["system"]
