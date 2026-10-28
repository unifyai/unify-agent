"""The actor's library guide (``unify.memory.library_prompts``) states what is true in a run: every ``memory`` call
and error it names is one the generated package defines, the statuses it explains are the statuses an item can have,
the files line names tools the actor and a teammate really get, the guide is the same bytes whatever the library
holds, and the one pointer to the package's own help is the line every memory text shares. The writer's brief has
its own file, ``test_writer_prompts``.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

from unify.actor import tool_handlers as H
from unify.common import tool_registry as R
from unify.memory import item_states
from unify.memory import library_export as LE
from unify.memory import library_helper as lh
from unify.memory import library_prompts as LP
from unify.memory.agent_profiles import _LIBRARY_LINE

GUIDE = LP.GUIDE_LIBRARIAN


def _package_names(tmp_path: Path) -> set[str]:
    """The public names of the generated ``memory`` package as a cell imports it (built in a child, never here)."""
    pkg = tmp_path / "memory"
    pkg.mkdir()
    (pkg / "__init__.py").write_bytes(LE.package_source(librarian=True))
    (pkg / "_contract.py").write_bytes(LE.CONTRACT_SOURCE.read_bytes())
    code = f"import sys, json\nsys.path.insert(0, {str(tmp_path)!r})\nimport memory\nprint(json.dumps(dir(memory)))\n"
    out = subprocess.run([sys.executable, "-I", "-c", code], cwd=tmp_path, capture_output=True, text=True, timeout=120)
    assert out.returncode == 0, out.stderr[-2000:]
    return set(json.loads(out.stdout.strip().splitlines()[-1]))


def test_every_call_and_error_the_guide_names_is_defined_by_the_package(tmp_path):
    names = _package_names(tmp_path)
    called = set(re.findall(r"\bmemory\.([a-z_]+)\b", GUIDE))
    assert called >= {"ask", "items", "get"}, called
    assert called <= names, sorted(called - names)
    errors = set(re.findall(r"`([A-Z][A-Za-z]+Error)`", GUIDE))
    assert errors == {"MemoryInputError", "AskSchemaError"}, errors
    assert "AskSchemaError" in names
    # a library package defines its own MemoryInputError; a call that raises one is the package's refusal, by its name
    own = type("MemoryInputError", (ValueError,), {})
    assert lh.REFUSAL == "MemoryInputError" and lh.call_outcome(own()) == "refused"
    assert lh.call_outcome(ValueError()) == "error"


def test_the_statuses_the_guide_explains_are_the_statuses_an_item_can_have():
    explained = set(re.findall(r"`([a-z]+(?:-[a-z]+)?)`", LP.STATUS_LINES_IN_CODE))
    assert explained == set(item_states.STATUSES), explained
    # the hidden status is the one the guide says failed uses lead to
    assert item_states.HIDDEN == {item_states.RETIRED}
    assert "`retired` after failed uses" in LP.STATUS_LINES_IN_CODE


def test_the_files_line_names_tools_the_actor_and_a_teammate_get():
    tools = set(re.findall(r"`tools\.([a-z_]+)`", LP.FILES_LINE_IN_CODE))
    assert tools == {"grep", "read"}
    for agent in ("actor", "teammate"):
        allow = set(R.allow_list(agent, H.actor_flags(agent, librarian=True)))
        assert tools | {"ask_library"} <= allow, (agent, sorted(allow))


def test_the_guide_is_constant_text_naming_no_item():
    # no format field, no placeholder and no id: the same bytes for every library, so the cached prefix never moves
    assert "{" not in GUIDE and "<<" not in GUIDE
    assert not re.search(r"capture:[0-9a-f]{16}|memory\.[a-z_]+\.[a-z_]+:[a-z_]+|notes/[a-z0-9_-]+/", GUIDE)
    assert GUIDE.startswith("### Memory Library\n")
    for part in (LP.LIBRARIAN_POINTER, LP.ASK_LINE_IN_CODE, LP.ASK_SCHEMA_LINE_IN_CODE, LP.FILES_LINE_IN_CODE,
                 LP.USE_LINE_IN_CODE, LP.STATUS_LINES_IN_CODE):
        assert GUIDE.count(part) == 1, part[:60]


def test_one_help_pointer_is_shared_by_the_guide_and_the_memory_agents_library_lines():
    assert LP.MEMORY_HELP_LINE == "`help(memory)` shows the library's calls and types."
    assert GUIDE.count(LP.MEMORY_HELP_LINE) == 1
    assert LP.MEMORY_HELP_LINE in _LIBRARY_LINE["writer"] and LP.MEMORY_HELP_LINE in _LIBRARY_LINE["librarian"]


def test_the_when_to_use_text_names_the_four_parts_the_gate_checks():
    from unify.memory import when_to_use

    for part in when_to_use.PARTS:
        assert f"`{part}:`" in LP.WHEN_TO_USE_TEXT, part
    assert f"`{when_to_use.SECTION}:`" in LP.WHEN_TO_USE_TEXT and f"`## {when_to_use.SECTION}`" in LP.WHEN_TO_USE_TEXT
    # missing parts are labelled, never refused: the text says so, as the gate policy decides
    assert "is labelled so" in LP.WHEN_TO_USE_TEXT and "refus" not in LP.WHEN_TO_USE_TEXT


def test_the_captures_text_says_a_capture_is_served_as_it_is_stored():
    text = LP.captures_input()
    assert "A capture is a library entry from the moment it is stored, served as it is" in text
    assert "The harness runs no capture" in text
    assert "`tools.land_capture(capture_id, package, name)`" in text
    assert "land_capture" in R.allow_list("writer", frozenset())
