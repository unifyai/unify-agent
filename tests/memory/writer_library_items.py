"""A library item for the writer-pass tests, in rev 2's library layout (the older env/<channel> item went with it).

The writer writes these files from a cell (:func:`files_code`): the package ``memory/acct`` of
:mod:`tests.memory.library_example_world` (its module, README and package docstring) with a test file that calls the
function on literal values, and the manifest :data:`MANIFEST`. Rev 2's gate is labels only: a red test or an ungrounded
example is a label on the item, never a refusal. What still refuses is safety, confinement, credentials, paths and
syntax, so a scripted pass that needs a refused round writes :data:`MISPLACED` (a file the layout has no place for) or
a key-shaped string, and its repair removes it.
"""

from __future__ import annotations

import json

from tests.memory import library_example_world as lw

ITEM_ID = lw.ITEM_ID
MODULE = lw.MODULE
TEST_FILE = lw.TEST_FILE
MODULE_SOURCE = lw.MODULE_SOURCE
#: The function reads a key the response does not have: its test fails on the candidate (a G3 label, with KeyError).
MODULE_BAD = MODULE_SOURCE.replace('return response["user_id"]', 'return response["user"]')
TEST_SOURCE = '''import pytest

from memory.acct.users import MemoryInputError, user_id


def test_reads_the_recorded_user_ids():
    assert user_id({"user_id": "u-1", "name": "Ada"}) == "u-1"


def test_refuses_a_response_without_a_user_id():
    with pytest.raises(MemoryInputError):
        user_id({"name": "Ada"})
'''
FILES = {
    lw.PACKAGE_INIT: '"""Accounts."""\n',
    lw.README: "Reading account responses.\n",
    MODULE: MODULE_SOURCE,
    TEST_FILE: TEST_SOURCE,
}
ITEM = {"item": ITEM_ID, "kind": "function", "source_episodes": ["e1"], "tests": [TEST_FILE], "covers": []}
MANIFEST = {"items": [ITEM], "summary": "s"}
#: A file the library layout has no place for: the gate refuses it (G1) before anything runs.
MISPLACED = "memory/acct/scratch.txt"
MISPLACED_REASON = (
    f"G1: file {MISPLACED}: a module is memory/<package>/<module>.py, a lower-case identifier other than tests; the "
    "package's description is memory/<package>/README.md"
)
#: A key-shaped string (never a real key): the gate refuses a library that holds one (G6) after its runs.
KEY = "sk-or-v1-" + "cd" * 32  # pragma: allowlist secret


def write(path: str, text: str) -> str:
    """One line of cell code writing *text* at *path* (in the box)."""
    return (
        f"import pathlib; p=pathlib.Path({path!r}); "
        f"p.parent.mkdir(parents=True, exist_ok=True); p.write_text({text!r})"
    )


def remove(path: str) -> str:
    """One line of cell code removing the box file *path*."""
    return f"import os; os.remove({path!r})"


def files_code(files: dict | None = None, manifest: dict | None = None) -> str:
    """Cell code writing the library files (default :data:`FILES`) under /memory and the manifest."""
    lines = [write(f"/memory/{rel}", text) for rel, text in (FILES if files is None else files).items()]
    lines.append(write("/memory/.pass/manifest.json", json.dumps(MANIFEST if manifest is None else manifest)))
    return "\n".join(lines)


def paged(name: str, args: dict) -> str:
    """Cell code calling ``tools.<name>(**args)`` and printing what it returns, its head and tail only when it is
    longer than a cell shows (a tool's own page plus its markers is past the cell's output limit), so the tool's own
    markers stay in the output; a refusal prints as it was raised."""
    return (
        "try:\n"
        f"    _r = tools.{name}(**{args!r})\n"
        "except Exception as _exc:\n"
        "    print(_exc)\n"
        "else:\n"
        "    print(_r if len(_r) <= 7000 else _r[:3000] + '\\n[...]\\n' + _r[-3000:])\n"
    )


#: What a cell's output opens with after a call that stopped the writer's interpreter (a refused finish ends the round,
#: and with it the cell session): the next cell's output says so first.
STOPPED = "(the interpreter was stopped by the end of the last call; variables are gone, files remain)\n"
#: What a refused ``tools.finish`` cell prints before the refusal: finish ends the cell's interpreter whether or not it
#: is accepted, so the cell has no output of its own and the interpreter is restarted.
FINISH_CELL = (
    "(no output)\n(the interpreter exited during the cell; the interpreter was restarted: variables are gone, files "
    "remain)\n"
)
