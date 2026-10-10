"""UNIFY_MEMORY_V2 in the actor: no review, no library objects, the memory section last in the system prompt
(Task 19; v2.1 ``catalogue``: a constant guide, the same bytes whatever the library holds; the channels are
what ``memory.catalog()`` prints in the cell).

The model is the scripted transport (tests/scripted_model.py): a request of any kind the test did not
script (a storage review, its gate, ...) fails the test, so ``kinds() == ["actor"]`` proves none ran.
"""

from __future__ import annotations

import asyncio
import importlib.util
import json
import re
from pathlib import Path
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
from unify.memory_v2.catalogue import estimate_tokens, write_generated
from unify.memory_v2.gitio import Repo
from unify.memory_v2.integration import hooks, prompt
from unify.memory_v2.integration import request as request_mod
from unify.memory_v2.integration.checkout import export_checkout
from unify.memory_v2.integration.paths import Paths
from unify.memory_v2.integration.state import State
from unify.memory_v2.integration.switch import SurfacingOptions
from unify.settings import SETTINGS

DOUBLE = "def double(x: int) -> int:\n    return x * 2\n"


def _run_for(home, tmp_path) -> SimpleNamespace:
    """A stand-in for Task 25's RequestRun: the export of a seeded memory and its memory section."""
    paths = Paths.under(home)
    mem, sha = _seed(tmp_path)
    export_checkout(mem.git_dir, sha, paths.checkout)
    write_generated(paths.checkout)
    return SimpleNamespace(
        index=prompt.render_memory_section(paths.checkout),
        paths=paths,
    )


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


# ── the memory section (UNIFY_MEMORY_V2_SURFACING=catalogue) ────────────────


def _helper(root: Path):
    """``import memory`` as a cell does it: the export's top-level ``memory.py``."""
    spec = importlib.util.spec_from_file_location(
        f"memory_under_test_{abs(hash(str(root)))}_{len(_LOADED)}",
        root / "memory.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    _LOADED.append(module)
    return module


_LOADED: list = []


def _land(mem: Repo, files: dict) -> str:
    """Commit *files* (relative path -> text, or None to delete) on memory ``main``; the new head."""
    base = mem.head()
    with mem.temp_checkout() as wt:
        for rel, text in files.items():
            target = wt / rel
            if text is None:
                target.unlink()
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(text)
        sha = mem.commit_all(wt, "land", {})
    mem.fast_forward("main", sha, expected_old=base)
    return sha


def _request(paths: Paths, monkeypatch) -> request_mod.RequestRun:
    """Open a request's catalogue surfacing as ``RequestRun._open`` does (export, generated catalogue,
    memory section) on the current memory ``main`` and the saved state; the run is the current one.
    """
    run = request_mod.RequestRun("r", paths)
    run.state = State.load(paths.state)
    run.pin = Repo(paths.memory).head()
    export_checkout(paths.memory, run.pin, paths.checkout)
    run.surfacing = SurfacingOptions(surfacing="catalogue")
    run._shape_rows = lambda consolidate: {}  # no evidence store here: no input shapes
    run._surface_catalogue(None)
    monkeypatch.setattr(request_mod, "_CURRENT", run)
    return run


FN = 'def {name}(apis):\n    """{doc}\n\n    Effect: read\n    """\n    return 1\n\n\n'


def test_the_prompt_never_changes_as_the_library_grows_or_drifts(tmp_path, monkeypatch):
    """The lead's rule: under ``catalogue`` the system prompt carries no library-dependent text. From the
    first request whose library lists anything it holds the constant guide, byte for byte, however the
    library grows, gains channels, turns suspect or even empties; before that it holds no memory text.
    """
    monkeypatch.setattr(SETTINGS, "UNIFY_MEMORY_V2", "on")
    paths = Paths.under(tmp_path / "home")
    mem = Repo.init_bare(paths.memory)

    # an empty library: no memory text at all, and nothing remembered
    run = _request(paths, monkeypatch)
    assert run.index == "" and hooks.system_prompt("SYS") == "SYS"
    assert run.state.guide is False and not paths.state.exists()
    assert _helper(paths.checkout).catalog().startswith("Memory library: empty")

    prompts: list[str] = []
    catalogs: list[str] = []
    stages = [
        ({"env/spotify/__init__.py": MOD}, set()),  # the first merge
        (  # the library grows
            {
                "env/spotify/__init__.py": MOD
                + "\n\n"
                + FN.format(name="list_playlists", doc="List the playlists.")
                + FN.format(name="play", doc="Play a track."),
            },
            set(),
        ),
        (  # a channel is added, with a note
            {
                "env/venmo/__init__.py": '"""Venmo payments."""\n\n'
                + FN.format(name="pay", doc="Pay a contact."),
                "env/venmo/NOTES.md": "# notes\n\n## Amounts\nDecimal strings.\n",
            },
            set(),
        ),
        ({}, {"spotify"}),  # a drift event: spotify turns suspect
    ]
    for files, suspect in stages:
        if files:
            _land(mem, files)
        if suspect:  # drift is recorded between requests (RequestRun._record_episode)
            state = State.load(paths.state)
            state.suspect |= suspect
            state.save()
        run = _request(paths, monkeypatch)
        prompts.append(hooks.system_prompt("SYS"))
        catalogs.append(_helper(paths.checkout).catalog())
    assert prompts == ["SYS\n\n" + prompt.GUIDE] * len(stages)
    assert State.load(paths.state).guide is True
    # what the prompt no longer says, the cell's memory.catalog() does, and it follows the library
    assert len(set(catalogs)) == len(stages)
    assert "- `env.spotify`: 1 function\n" in catalogs[0]
    assert "- `env.spotify`: 3 functions\n" in catalogs[1]
    assert "- `env.venmo`: 1 function, 1 note. Venmo payments.\n" in catalogs[2]
    assert (
        "- `env.spotify`: 3 functions (suspect: the environment changed since these were built; "
        "verify before use)\n"
    ) in catalogs[3]
    assert "suspect" not in catalogs[2]
    flags = json.loads((paths.checkout / ".memory/catalog.json").read_text())[
        "channels"
    ]
    assert [(c["channel"], c.get("suspect")) for c in flags] == [
        ("spotify", True),
        ("venmo", None),
    ]
    assert "Channel env.spotify is suspect" in _helper(paths.checkout).describe("hello")

    # the library empties (every item hidden): the guide stays, so the prefix still does not change
    _land(
        mem,
        {
            "env/spotify/__init__.py": None,
            "env/venmo/__init__.py": None,
            "env/venmo/NOTES.md": None,
        },
    )
    run = _request(paths, monkeypatch)
    assert hooks.system_prompt("SYS") == prompts[0]
    assert _helper(paths.checkout).catalog().startswith("Memory library: empty")


def test_the_catalogue_records_that_it_showed_the_guide_and_no_item(
    tmp_path,
    monkeypatch,
):
    """Use telemetry under ``catalogue``: the renderer records exactly what the prompt shows (the guide's
    digest and size, renderer ``catalogue``) and no channel or item, since the guide names none; per-item
    exposure comes from the cells. An empty library shows nothing.
    """
    from unify.memory_v2.analysis import use

    monkeypatch.setattr(SETTINGS, "UNIFY_MEMORY_V2", "on")
    paths = Paths.under(tmp_path / "home")
    mem = Repo.init_bare(paths.memory)
    run = _request(paths, monkeypatch)
    assert run.index == ""
    assert run.shown == use.record_shown("", channels=(), renderer="catalogue")
    assert run.shown["bytes"] == 0 and run.shown["channels"] == run.shown["items"] == []
    _land(mem, {"env/spotify/__init__.py": MOD})
    run = _request(paths, monkeypatch)
    assert run.index == prompt.GUIDE
    want = use.record_shown(prompt.GUIDE, channels=(), renderer="catalogue")
    assert run.shown == want
    assert want["renderer"] == "catalogue" and want["bytes"] == len(
        prompt.GUIDE.encode(),
    )
    assert want["channels"] == [] and want["items"] == []
    assert hooks.system_prompt("SYS").endswith(prompt.GUIDE)


def test_the_guide_is_constant_short_and_names_nothing_of_the_library():
    guide = prompt.GUIDE
    assert estimate_tokens(guide) <= 120
    assert (
        "{" not in guide and "}" not in guide
    )  # no template field: nothing is filled in per run
    assert not any(ch.isdigit() for ch in guide)  # no count
    for absent in ("env.", "spotify", "suspect", "Channels", "/", "README"):
        assert absent not in guide, absent
    for present in (
        "import memory; print(memory.catalog())",
        "memory.find(value)",
        "help(fn)",
        "memory.describe(name)",
        "MemoryInputError",
        "do the work directly",
        "candidates, not authority",
    ):
        assert present in guide, present


def test_the_memory_section_depends_only_on_whether_anything_is_listed(tmp_path):
    mem, sha = _seed(tmp_path)
    a, b = tmp_path / "a", tmp_path / "b"
    export_checkout(mem.git_dir, sha, a)
    export_checkout(mem.git_dir, sha, b)
    write_generated(a, suspect={"spotify"})
    assert (
        prompt.render_memory_section(a)
        == prompt.render_memory_section(b)
        == prompt.GUIDE
    )


def test_an_empty_memory_adds_nothing_until_the_guide_was_shown(tmp_path):
    mem = Repo.init_bare(tmp_path / "memory")
    export_checkout(mem.git_dir, mem.head(), tmp_path / "co")
    assert prompt.render_memory_section(tmp_path / "co") == ""
    assert (
        prompt.render_memory_section(tmp_path / "co", shown_before=True) == prompt.GUIDE
    )


def test_a_large_library_leaves_the_prompt_as_it_is(tmp_path):
    """No 4,000-token freeze and no growth: the section is the guide however many functions there are."""
    big = "".join(
        f"def f{i}(apis):\n    \"\"\"{'A long summary of what this does. ' * 8}\n\n    Effect: read\n    \"\"\"\n"
        f"    return {i}\n\n\n"
        for i in range(300)
    )
    mem, sha = _seed(tmp_path, {"env/spotify/__init__.py": big})
    export_checkout(mem.git_dir, sha, tmp_path / "co")
    assert prompt.render_memory_section(tmp_path / "co") == prompt.GUIDE
    write_generated(tmp_path / "co")
    assert "- `env.spotify`: 300 functions\n" in _helper(tmp_path / "co").catalog()


# ── a suspect channel's refusal (UNIFY_MEMORY_V2_SURFACING=catalogue) ───────

REFUSAL = (
    "Traceback (most recent call last):\n"
    '  File "<string>", line 2, in <module>\n'
    "env.spotify.MemoryInputError: expected a playlist id, got an empty string\n"
)


def test_a_suspect_channels_refusal_says_so_only_under_catalogue(monkeypatch, tmp_path):
    monkeypatch.setattr(SETTINGS, "UNIFY_MEMORY_V2", "on")

    def run(surfacing: str, suspect: set) -> SimpleNamespace:
        return SimpleNamespace(
            index="",
            paths=Paths.under(tmp_path),
            surfacing=SurfacingOptions(surfacing=surfacing),
            state=SimpleNamespace(suspect=suspect),
        )

    monkeypatch.setattr(request_mod, "_CURRENT", run("catalogue", {"spotify", "venmo"}))
    noted = hooks.cell_error(REFUSAL)
    assert noted.startswith(REFUSAL)
    assert noted[len(REFUSAL) :] == (
        "memory: env.spotify is suspect: the environment changed since its functions were built, so "
        "this refusal may come from that change; do the work directly.\n"
    )
    nested = REFUSAL.replace("env.spotify.", "env.spotify.parsers.")
    assert "env.spotify is suspect" in hooks.cell_error(nested)
    for text in (
        REFUSAL.replace("spotify", "slack"),  # a channel that is not suspect
        REFUSAL.replace("MemoryInputError", "KeyError"),  # not a refusal
        "ValueError: env.spotify.MemoryInputError mentioned mid-line\n",
    ):
        assert hooks.cell_error(text) == text
    monkeypatch.setattr(request_mod, "_CURRENT", run("index", {"spotify"}))
    assert hooks.cell_error(REFUSAL) == REFUSAL  # the v2 screen build's text
    monkeypatch.setattr(request_mod, "_CURRENT", None)
    assert hooks.cell_error(REFUSAL) == REFUSAL
    monkeypatch.setattr(SETTINGS, "UNIFY_MEMORY_V2", "")
    monkeypatch.setattr(request_mod, "_CURRENT", run("catalogue", {"spotify"}))
    assert hooks.cell_error(REFUSAL) == REFUSAL


# ── the v2 index (UNIFY_MEMORY_V2_SURFACING=index, the default) ─────────────


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
    # with the soft budget the request passes no cut, so the whole index is kept
    assert "`hello(apis, name)`" in prompt.render_index(
        tmp_path / "co",
        budget_tokens=10**9,
    )


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
    assert run.index == prompt.GUIDE  # constant: no channel, function or path
    # no use of the library objects (``functions.search``, ``guidance.get``); the guide's own prose may end a
    # sentence on "functions."
    assert not re.search(r"\b(?:functions|guidance)\.\w", first["system"])
    assert "### Function & Guidance Library" not in first["system"]
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
