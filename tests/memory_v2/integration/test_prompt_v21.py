"""The v2.1 memory section (spec §4.5, §6): GUIDE, then the index view, last in the system prompt (cache rule)."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from tests.actor.code_act.core_world import (
    core_world,
    new_actor,
    world,
)  # noqa: F401 (fixtures)
from tests.actor.code_act.sandbox_world import needs_bwrap
from tests.helpers import _handle_project
from tests.memory_v2.integration.test_actor_surface import _first_request
from tests.memory_v2.test_layout import LIB, _tree
from tests.memory_v2.test_library_helper import _commit
from unify.memory_v2 import shape_rows
from unify.memory_v2.gitio import Repo
from unify.memory_v2.integration import hooks, prompt
from unify.memory_v2.integration import request as request_mod
from unify.memory_v2.integration.checkout import export_actor_v21
from unify.memory_v2.integration.paths import Paths
from unify.memory_v2.integration.state import State
from unify.memory_v2.integration.switch import v21_enabled
from unify.memory_v2.library_index import index_view
from unify.memory_v2.catalogue import body_digest
from unify.memory_v2.layout import function_bodies
from unify.settings import SETTINGS


def _export(tmp_path, files=None, message="seed"):
    mem = (
        Repo.init_bare(tmp_path / "memory")
        if not (tmp_path / "memory").exists()
        else Repo(tmp_path / "memory")
    )
    sha = _commit(mem, files if files is not None else LIB, message)
    export_actor_v21(mem.git_dir, sha, tmp_path / "co")
    return mem, sha, tmp_path / "co"


def test_the_section_is_guide_location_then_view_and_empty_adds_nothing(tmp_path):
    _, _, co = _export(tmp_path)
    text, shown = prompt.render_memory_v21(co)
    index = (co / "INDEX.md").read_text()
    assert text == prompt.GUIDE_V21 + prompt.location_line(co) + "\n" + index_view(
        index,
    )
    assert shown["renderer"] == "index_v21" and shown["channels"] == ["text", "web"]
    empty = tmp_path / "empty"
    mem = Repo.init_bare(empty / "memory")
    export_actor_v21(mem.git_dir, mem.head(), empty / "co")
    assert prompt.render_memory_v21(empty / "co")[0] == ""
    for word in ("apis", "venmo", "catalog()", "recorded for the next consolidation"):
        assert word not in prompt.GUIDE_V21
    # Amendment C (lead): read-only is enforced by the binds and modes, never told to the actor
    assert "read-only" not in prompt.GUIDE_V21 + prompt.location_line(co)


def test_the_section_is_stable_between_commits_and_changes_only_its_tail(tmp_path):
    mem, sha, co = _export(tmp_path)
    one = prompt.render_memory_v21(co)[0]
    export_actor_v21(mem.git_dir, sha, co)  # the next request on the same pin
    assert prompt.render_memory_v21(co)[0] == one
    new = _commit(
        mem,
        {"memory/text/extra.py": 'def more(x):\n    """More."""\n    return x\n'},
        "a pass lands",
    )
    export_actor_v21(mem.git_dir, new, co)
    two = prompt.render_memory_v21(co)[0]
    prefix = prompt.GUIDE_V21 + prompt.location_line(co) + "\n"
    assert one.startswith(prefix) and two.startswith(prefix) and one != two
    assert "memory.text.extra:more(x)" in two


def test_open_v21_exports_read_only_and_renders_the_section(tmp_path, monkeypatch):
    monkeypatch.setattr(SETTINGS, "UNIFY_MEMORY_V2", "on")
    paths = Paths.under(tmp_path / "home")
    mem = Repo.init_bare(paths.memory)
    _commit(mem, LIB, "seed")
    run = request_mod.RequestRun("r", paths)
    run.state = State.load(paths.state)
    run.pin = mem.head()
    run.v21 = True
    run._shape_rows = lambda consolidate, v21=False: {}  # no evidence store here
    run._open_v21(None)
    monkeypatch.setattr(request_mod, "_CURRENT", run)
    assert run.index == prompt.render_memory_v21(paths.checkout)[0] != ""
    assert hooks.system_prompt("SYS") == "SYS\n\n" + run.index
    assert run.item_ids == sorted(function_bodies(paths.checkout))
    assert "memory/__init__.py" in run.generated and "INDEX.md" in run.generated
    assert hooks.worker_mounts() == [] and hooks.worker_readonly_mounts() == [
        paths.checkout,
    ]
    assert run._memory_diff() == ""
    assert (
        not (paths.checkout / "memory.py").exists()
        and not (paths.checkout / "README.md").exists()
    )


def test_the_v21_switch_parses():
    assert v21_enabled(SimpleNamespace()) is False
    assert v21_enabled(SimpleNamespace(UNIFY_MEMORY_V21="off")) is False
    assert v21_enabled(SimpleNamespace(UNIFY_MEMORY_V21="on")) is True
    with pytest.raises(ValueError, match="UNIFY_MEMORY_V21"):
        v21_enabled(SimpleNamespace(UNIFY_MEMORY_V21="maybe"))


def test_shape_rows_key_v21_functions_by_body(tmp_path):
    root = _tree(tmp_path)
    got = shape_rows._functions(root, v21=True)
    bodies = function_bodies(root)
    assert got == {i: (body_digest(b), None) for i, b in bodies.items()}
    assert shape_rows._functions(root) == {}  # v2's ids come from env/ only


@needs_bwrap
@pytest.mark.asyncio
@pytest.mark.timeout(240)
@_handle_project
async def test_the_section_ends_the_system_prompt_after_the_clock(
    core_world,
    monkeypatch,
    tmp_path,
):  # noqa: F811
    monkeypatch.setattr(SETTINGS, "UNIFY_MEMORY_V2", "on")
    paths = Paths.under(core_world["state"])
    mem = Repo.init_bare(paths.memory)
    sha = _commit(mem, LIB, "seed")
    export_actor_v21(mem.git_dir, sha, paths.checkout)
    section = prompt.render_memory_v21(paths.checkout)[0]
    monkeypatch.setattr(
        request_mod,
        "_CURRENT",
        SimpleNamespace(index=section, paths=paths, v21=True),
    )
    _, _, first = await _first_request()
    system = first["system"]
    assert system.endswith("\n\n" + section)
    assert "### Current Time" in system and system.index(
        "### Current Time",
    ) < system.index(section)
