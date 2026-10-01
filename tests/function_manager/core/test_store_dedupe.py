"""Symbolic: ``UNIFY_STORE_DEDUPE=warn`` names the stored function a new one nearly copies.

As shipped, ``add_functions`` stores a new function beside a stored one that
does the same thing under another name; our audit found
rewind_spotify_until_artist and rewind_appworld_spotify_until_artist stored
side by side (normalised code similarity 0.96). With the switch on, a new
function whose normalised AST token Jaccard with a stored one is at least
``NEAR_DUPLICATE_JACCARD`` (0.9) is still stored, and its status carries a
warning naming the stored function with an excerpt and suggesting a patch.
Overwrites are never warned. Off, the statuses are exactly as shipped. No
model is called.
"""

from __future__ import annotations

import pytest

from tests.helpers import _handle_project
from unify.function_manager import near_duplicates
from unify.function_manager.function_manager import FunctionManager
from unify.function_manager.near_duplicates import (
    NEAR_DUPLICATE_JACCARD,
    code_tokens,
    similarity,
)
from unify.settings import ProductionSettings, SETTINGS

REWIND = (
    "def rewind_spotify_until_artist(artist: str, max_steps: int = 20) -> dict:\n"
    '    """Go back through the queue until a song by `artist` plays."""\n'
    "    steps = 0\n"
    "    song = primitives.actor.act(request='current song')\n"
    "    while song.get('artist') != artist and steps < max_steps:\n"
    "        song = primitives.actor.act(request='previous song')\n"
    "        steps += 1\n"
    "    return {'song': song, 'steps': steps}\n"
)
# The same code under another name, other local names, no docstring or hints.
REWIND_COPY = (
    "def rewind_appworld_spotify_until_artist(name, limit=20):\n"
    "    n = 0\n"
    "    current = primitives.actor.act(request='current song')\n"
    "    while current.get('artist') != name and n < limit:\n"
    "        current = primitives.actor.act(request='previous song')\n"
    "        n += 1\n"
    "    return {'song': current, 'steps': n}\n"
)
UNRELATED = (
    "def total_minor(rows: list) -> int:\n"
    "    return sum(int(row.get('amount', 0)) for row in rows)\n"
)


def _FM() -> FunctionManager:
    return FunctionManager(include_primitives=False)


@pytest.fixture
def warn_on(monkeypatch):
    monkeypatch.setattr(SETTINGS, "UNIFY_STORE_DEDUPE", "warn")


def test_the_switch_is_off_by_default_and_takes_only_warn():
    assert ProductionSettings.model_fields["UNIFY_STORE_DEDUPE"].default == ""
    assert ProductionSettings(UNIFY_STORE_DEDUPE=" Warn ").UNIFY_STORE_DEDUPE == "warn"
    with pytest.raises(ValueError, match="must be empty or 'warn'"):
        ProductionSettings(UNIFY_STORE_DEDUPE="refuse")


def test_similarity_ignores_names_docstrings_and_hints():
    assert similarity(REWIND, REWIND_COPY) == 1.0
    assert similarity(REWIND, UNRELATED) < 0.5
    changed = REWIND.replace("'previous song'", "'next song'").replace(
        "steps += 1",
        "steps += 2",
    )
    assert NEAR_DUPLICATE_JACCARD <= similarity(REWIND, changed) < 1.0
    # The own name and bound names are placeholders; free names are kept.
    tokens = code_tokens(REWIND)
    assert "FUNC" in tokens and "VAR0" in tokens and "primitives" in tokens
    assert "rewind_spotify_until_artist" not in tokens and "max_steps" not in tokens


@_handle_project
def test_a_near_copy_is_stored_with_a_warning_naming_the_original(
    warn_on,
    monkeypatch,
):
    monkeypatch.setattr(SETTINGS, "UNIFY_FUNCTION_PATCH", True)
    fm = _FM()
    assert fm.add_functions(implementations=REWIND) == {
        "rewind_spotify_until_artist": "added",
    }
    result = fm.add_functions(implementations=REWIND_COPY)
    status = result["rewind_appworld_spotify_until_artist"]
    assert status.startswith("added; warning: ")
    assert (
        "'rewind_appworld_spotify_until_artist' is nearly identical to the stored "
        "function 'rewind_spotify_until_artist' (code similarity 1.00" in status
    )
    # A short excerpt of the stored function, not its whole source.
    assert "def rewind_spotify_until_artist(artist: str" in status
    assert "return {'song': song" not in status
    assert "FunctionManager_patch_function" in status
    # Warn mode stores anyway.
    assert set(fm.list_functions()) == {
        "rewind_spotify_until_artist",
        "rewind_appworld_spotify_until_artist",
    }


@_handle_project
def test_without_patching_the_warning_suggests_an_overwrite(warn_on, monkeypatch):
    monkeypatch.setattr(SETTINGS, "UNIFY_FUNCTION_PATCH", False)
    fm = _FM()
    fm.add_functions(implementations=REWIND)
    status = fm.add_functions(implementations=REWIND_COPY)[
        "rewind_appworld_spotify_until_artist"
    ]
    assert "FunctionManager_add_functions (overwrite=True)" in status
    assert "patch_function" not in status


@_handle_project
def test_the_threshold_is_inclusive(warn_on, monkeypatch):
    fm = _FM()
    fm.add_functions(implementations=REWIND)
    variant = REWIND_COPY.replace("'previous song'", "'next song'")
    score = similarity(REWIND, variant)
    assert 0.9 < score < 1.0

    monkeypatch.setattr(near_duplicates, "NEAR_DUPLICATE_JACCARD", score)
    at = fm.add_functions(implementations=variant)
    assert at["rewind_appworld_spotify_until_artist"].startswith("added; warning: ")

    fm.delete_function(
        function_id=fm.list_functions()["rewind_appworld_spotify_until_artist"][
            "function_id"
        ],
    )
    monkeypatch.setattr(near_duplicates, "NEAR_DUPLICATE_JACCARD", score + 1e-9)
    below = fm.add_functions(implementations=variant)
    assert below == {"rewind_appworld_spotify_until_artist": "added"}


@_handle_project
def test_distinct_functions_and_overwrites_are_not_warned(warn_on):
    fm = _FM()
    fm.add_functions(implementations=REWIND)
    assert fm.add_functions(implementations=UNRELATED) == {"total_minor": "added"}
    changed = REWIND.replace("max_steps: int = 20", "max_steps: int = 30")
    assert fm.add_functions(implementations=changed, overwrite=True) == {
        "rewind_spotify_until_artist": "updated",
    }


@_handle_project
def test_off_the_statuses_are_as_shipped(monkeypatch):
    monkeypatch.setattr(SETTINGS, "UNIFY_STORE_DEDUPE", "")
    fm = _FM()
    fm.add_functions(implementations=REWIND)
    assert fm.add_functions(implementations=REWIND_COPY) == {
        "rewind_appworld_spotify_until_artist": "added",
    }
