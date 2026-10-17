"""Symbolic: the libraries store a task's identifiers as written.

A function whose name, code or docstring carries a task id or a value quoted
in a request, and a guidance entry that names one, are stored as written,
with no refusal and no warning: the harness never removes or flags
identifiers in what is stored. No model is called.
"""

from __future__ import annotations

import pytest

from tests.helpers import _handle_project
from unify.function_manager.function_manager import FunctionManager
from unify.guidance_manager.guidance_manager import GuidanceManager
from unify.settings import SETTINGS

NAMED = (
    "def solve_task_7a4cf12e(grid: list) -> list:\n"
    '    """Mirror each row, as task-7a4cf12e needs."""\n'
    "    return [row[::-1] for row in grid]\n"
)
LITERAL = (
    "def make_playlist(songs: list) -> dict:\n"
    '    return {"title": "R&B Recommendation", "order": 4417823, "songs": songs}\n'
)


@pytest.fixture
def no_verify(monkeypatch):
    monkeypatch.setattr(SETTINGS, "UNIFY_STORE_VERIFY", "")


@_handle_project
def test_a_function_naming_a_task_id_is_stored_as_written(no_verify):
    fm = FunctionManager()
    assert fm.add_functions(implementations=[NAMED]) == {"solve_task_7a4cf12e": "added"}
    assert fm.add_functions(implementations=[LITERAL]) == {"make_playlist": "added"}
    assert {"solve_task_7a4cf12e", "make_playlist"} <= set(fm.list_functions())
    patched = fm.patch_function(
        name="make_playlist",
        old='"songs": songs}',
        new='"songs": songs, "seen": "task-7a4cf12e"}',
        why="note where it was seen",
    )
    assert patched["status"] == "patched" and "warning" not in patched, patched


@_handle_project
def test_guidance_naming_a_task_id_is_stored_as_written(no_verify):
    gm = GuidanceManager()
    added = gm.add_guidance(
        title="Mirror for task-7a4cf12e",
        content='Name the playlist "R&B Recommendation" first.',
    )
    assert "warning" not in added, added
    gid = added["details"]["guidance_id"]
    updated = gm.update_guidance(guidance_id=gid, content="Seen on 7a4cf12e.")
    assert updated == {"outcome": "guidance updated", "details": {"guidance_id": gid}}
    patched = gm.patch_guidance(
        id_or_title=gid,
        old="Seen on",
        new="Learned on",
        why="wording",
    )
    assert patched["outcome"] == "guidance patched" and "warning" not in patched
