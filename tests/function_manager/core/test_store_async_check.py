"""Symbolic: ``UNIFY_STORE_ASYNC_CHECK`` warns when stored code awaits a synchronous environment method.

In the AppWorld HIGH cell of the overhaul build with every lean switch on,
the storage review rewrote working ``apis.spotify.x(...)`` calls as
``await primitives.spotify.x(...)`` and stored 11 functions that raise
"object list can't be used in 'await' expression" when run from code. With
the switch on, ``add_functions`` and ``patch_function`` store the function as
given and return a warning that names each such line and every other stored
function with the same pattern; with it off, results are as shipped. A fake
synchronous environment and a fake asynchronous one stand in for real ones.
No model is called.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from tests.helpers import _handle_project
from unify.function_manager import store_async_check
from unify.function_manager.function_manager import FunctionManager
from unify.function_manager.primitives import (
    EnvironmentMethod,
    EnvironmentNamespace,
    EnvironmentSurface,
    register_environment,
)
from unify.function_manager.primitives.environment import (
    clear_environment_namespaces,
    namespace_object,
)
from unify.settings import ProductionSettings, SETTINGS


def _library(**kwargs):
    return [{"song_id": 1, **kwargs}]


async def _fetch(**kwargs):
    return {"page": 1, **kwargs}


def _reseed() -> None:
    from unify.function_manager import function_manager as fm_module

    fm_module._PRIMITIVES_SEEDED_FOR.clear()
    FunctionManager()


@pytest.fixture
def environments():
    """``primitives.music`` is synchronous (as AppWorld's apps are); ``primitives.web`` asynchronous."""
    clear_environment_namespaces()
    register_environment(
        EnvironmentSurface(
            namespaces=(
                EnvironmentNamespace(
                    name="music",
                    methods=(
                        EnvironmentMethod(
                            name="show_library",
                            call=_library,
                            effect="read",
                        ),
                        EnvironmentMethod(name="play", call=_library, effect="write"),
                    ),
                ),
                EnvironmentNamespace(
                    name="web",
                    methods=(
                        EnvironmentMethod(name="fetch", call=_fetch, effect="read"),
                    ),
                ),
            ),
        ),
        source="tests:async-check",
    )
    _reseed()
    yield
    clear_environment_namespaces()
    _reseed()


@pytest.fixture
def check(monkeypatch):
    monkeypatch.setattr(SETTINGS, "UNIFY_STORE_CHECK", "")
    monkeypatch.setattr(SETTINGS, "UNIFY_STORE_VERIFY", "")
    monkeypatch.setattr(SETTINGS, "UNIFY_STORE_INSTANCE_LINT", False)
    monkeypatch.setattr(SETTINGS, "UNIFY_FUNCTION_PATCH", True)

    def set_(on: bool) -> None:
        monkeypatch.setattr(SETTINGS, "UNIFY_STORE_ASYNC_CHECK", on)

    return set_


AWAITS_SYNC = (
    "async def list_library(page_limit: int = 20) -> list:\n"
    '    """Return the library."""\n'
    "    songs = await primitives.music.show_library(page_limit=page_limit)\n"
    "    return songs\n"
)
CALLS_SYNC = (
    "def list_library_plain(page_limit: int = 20) -> list:\n"
    '    """Return the library."""\n'
    "    return primitives.music.show_library(page_limit=page_limit)\n"
)
AWAITS_ASYNC = (
    "async def fetch_page(url: str) -> dict:\n"
    '    """Return one page."""\n'
    "    return await primitives.web.fetch(url=url)\n"
)


def _fm() -> FunctionManager:
    return FunctionManager(include_primitives=False)


# ---------------------------------------------------------------------------
# The static scan
# ---------------------------------------------------------------------------

SYNC = {"music": frozenset({"show_library", "play"}), "web": frozenset()}


def test_scan_finds_an_awaited_synchronous_method_with_its_line():
    assert store_async_check.awaited_sync_calls(AWAITS_SYNC, SYNC) == [
        (3, "songs = await primitives.music.show_library(page_limit=page_limit)"),
    ]


def test_scan_leaves_correct_code_alone():
    assert store_async_check.awaited_sync_calls(CALLS_SYNC, SYNC) == []
    assert store_async_check.awaited_sync_calls(AWAITS_ASYNC, SYNC) == []


def test_scan_follows_a_bound_namespace_and_nested_awaits():
    source = (
        "async def play_all(ids: list) -> list:\n"
        "    music = primitives.music\n"
        "    out = [await music.play(song_id=i) for i in ids]\n"
        "    out.append(await primitives.web.fetch(url='u'))\n"
        "    if ids:\n"
        "        out.append(\n"
        "            await primitives.music.show_library()\n"
        "        )\n"
        "    return out\n"
    )
    assert [line for line, _ in store_async_check.awaited_sync_calls(source, SYNC)] == [
        3,
        7,
    ]


def test_scan_skips_a_function_that_binds_primitives_itself():
    source = (
        "async def injected(primitives) -> list:\n"
        "    return await primitives.music.show_library()\n"
    )
    assert store_async_check.awaited_sync_calls(source, SYNC) == []


def test_scan_without_a_registered_environment_finds_nothing():
    assert store_async_check.awaited_sync_calls(AWAITS_SYNC, {}) == []


# ---------------------------------------------------------------------------
# add_functions and patch_function
# ---------------------------------------------------------------------------


@_handle_project
def test_on_an_awaited_synchronous_method_is_stored_with_a_warning(
    check,
    environments,
):
    check(True)
    fm = _fm()
    result = fm.add_functions(implementations=[AWAITS_SYNC])
    status = result["list_library"]
    assert status.startswith("added; warning: ")
    assert "'list_library' awaits a synchronous environment method" in status
    assert (
        "line 3: `songs = await primitives.music.show_library(page_limit=page_limit)`"
        in status
    )
    assert "Call it without `await`." in status
    assert "list_library" in fm.list_functions()


@_handle_project
def test_on_correct_code_is_stored_as_shipped(check, environments):
    check(True)
    fm = _fm()
    assert fm.add_functions(implementations=[CALLS_SYNC, AWAITS_ASYNC]) == {
        "list_library_plain": "added",
        "fetch_page": "added",
    }


@_handle_project
def test_on_the_warning_lists_every_stored_function_with_the_pattern(
    check,
    environments,
):
    fm = _fm()
    check(False)
    other = AWAITS_SYNC.replace("list_library", "list_library_twice")
    assert fm.add_functions(implementations=[AWAITS_SYNC, other]) == {
        "list_library": "added",
        "list_library_twice": "added",
    }
    check(True)
    third = AWAITS_SYNC.replace("list_library", "list_library_again")
    status = fm.add_functions(implementations=[third])["list_library_again"]
    assert (
        "Other stored functions with the same pattern: 'list_library' (line 3); "
        "'list_library_twice' (line 3)."
    ) in status
    # A clean write still lists the library's functions with the pattern, once.
    result = fm.add_functions(implementations=[CALLS_SYNC, AWAITS_ASYNC])
    assert result["fetch_page"] == "added"
    assert result["list_library_plain"].startswith(
        "added; warning: Stored functions that await a synchronous environment method",
    )
    assert "'list_library_again' (line 3)" in result["list_library_plain"]


@_handle_project
def test_on_a_patch_reports_what_it_leaves_and_falls_silent_when_fixed(
    check,
    environments,
):
    check(True)
    fm = _fm()
    fm.add_functions(implementations=[AWAITS_SYNC])
    patched = fm.patch_function(
        name="list_library",
        old="songs = await primitives.music.show_library(",
        new="songs = await primitives.music.play(",
        why="play instead",
    )
    assert patched["status"] == "patched"
    assert "line 3: `songs = await primitives.music.play(" in patched["warning"]
    fixed = fm.patch_function(
        name="list_library",
        old="await primitives.music.play(",
        new="primitives.music.play(",
        why="the environment is synchronous",
    )
    assert fixed["status"] == "patched"
    assert "warning" not in fixed


@_handle_project
def test_off_stores_as_shipped(check, environments):
    check(False)
    fm = _fm()
    assert fm.add_functions(implementations=[AWAITS_SYNC]) == {"list_library": "added"}
    patched = fm.patch_function(
        name="list_library",
        old="page_limit=page_limit)",
        new="page_limit=page_limit + 0)",
        why="noop",
    )
    assert patched["status"] == "patched"
    assert "warning" not in patched


@_handle_project
def test_the_warned_function_does_raise_when_run_from_code(check, environments):
    """What the warning says is what happens: the stored function, loaded as a search loads it, raises."""
    check(True)
    fm = _fm()
    fm.add_functions(implementations=[AWAITS_SYNC, CALLS_SYNC])
    namespace = {
        "primitives": SimpleNamespace(
            music=namespace_object("music"),
            web=namespace_object("web"),
        ),
    }
    fm.list_functions(_return_callable=True, _namespace=namespace)
    assert namespace["list_library_plain"]() == [{"song_id": 1, "page_limit": 20}]
    with pytest.raises(TypeError, match="can't be used in 'await' expression"):
        asyncio.run(namespace["list_library"]())


@pytest.mark.parametrize(
    "value, expected",
    [("1", True), ("true", True), ("", False), ("0", False)],
)
def test_the_setting_parses(value, expected):
    assert (
        ProductionSettings(UNIFY_STORE_ASYNC_CHECK=value).UNIFY_STORE_ASYNC_CHECK
        is expected
    )
