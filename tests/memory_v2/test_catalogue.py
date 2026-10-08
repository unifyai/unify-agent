"""The export's generated catalogue and the ``memory`` helper (v2.1 surfacing, lane S1).

Memory is a library the working model works in: the harness writes ``README.md``, ``.memory/catalog.json``
and the ``memory`` helper beside every export. The helper is loaded here from the export exactly as a cell
imports it (a top-level module beside the catalogue), so what is tested is what the cell runs.
"""

import ast
import importlib.util
import json
import sys
from pathlib import Path

import pytest

from unify.memory_v2 import catalogue, memory_helper
from unify.memory_v2.analysis import shapes as shapes_module
from unify.memory_v2.evidence import EvidenceStore
from unify.memory_v2.gitio import Repo
from unify.memory_v2.integration.checkout import export_checkout
from unify.memory_v2.integration.prompt import render_memory_section
from unify.memory_v2.snapshot import item_bodies

WT_MOD = '''"""Readers for the workspace's files."""

import csv

__all__ = ["read_ledger", "read_any_ledger"]


class MemoryInputError(ValueError):
    """The input differs in shape from what this function was built from."""


def read_ledger(path: str) -> list:
    """Read the ledger into rows.

    Args:
        path: the ledger CSV file (a path).

    Returns:
        A list of dicts, one per row.

    Raises:
        MemoryInputError: when the file lacks the ledger's columns.

    Example:
        >>> read_ledger("env/worktree_workspace/tests/data/ledger.csv")[0]["vendor_id"]
        'V-17'

    Effect: read
    Input: path
    """
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


def read_any_ledger(path: str) -> list:
    """Read a ledger that may hold empty amounts.

    Effect: read
    Input: path
    """
    return read_ledger(path)


def hidden_reader(path):
    """Not listed.

    Effect: read
    """
    return path
'''
DLG_MOD = '''def parse_feedback(obs: dict) -> int:
    """Read the score from a feedback observation.

    Effect: read
    Input: observation
    """
    return obs["score"]
'''
NOTES = "# notes\n\n## Amounts\nAmounts are decimal strings.\n"

LEDGER = b"vendor_id,amount,due\nV-17,12.50,2026-10-14\nV-18,3.00,2026-10-20\n"
EMPTY_AMOUNTS = b"vendor_id,amount,due\nV-17,,2026-10-14\nV-18,,2026-10-20\n"
NEW_LEDGER = b"vendor_id,amount,due\nV-99,7.25,2026-11-01\n"
FEEDBACK = {"type": "Feedback", "score": 3, "notes": ["a"]}

READ_LEDGER = "env/worktree_workspace:read_ledger"
READ_ANY = "env/worktree_workspace:read_any_ledger"
PARSE_FEEDBACK = "env/dialogue_user:parse_feedback"


def _commit(tmp_path):
    mem = Repo.init_bare(tmp_path / "memory")
    base = mem.head()
    with mem.temp_checkout() as wt:
        for rel, text in {
            "env/worktree_workspace/__init__.py": WT_MOD,
            "env/worktree_workspace/NOTES.md": NOTES,
            "env/dialogue_user/__init__.py": DLG_MOD,
        }.items():
            (wt / rel).parent.mkdir(parents=True, exist_ok=True)
            (wt / rel).write_text(text)
        sha = mem.commit_all(wt, "seed", {})
    mem.fast_forward("main", sha, expected_old=base)
    return mem, sha


def _evidence(tmp_path, export):
    """An evidence store holding the shapes a merge would have recorded for these function bodies."""
    ev = EvidenceStore(tmp_path / "e.sqlite")
    bodies = item_bodies(export)
    for item, desc in (
        (READ_LEDGER, memory_helper.file_shape("ledger/2026/ledger.csv", LEDGER)),
        (READ_ANY, memory_helper.file_shape("old/ledger.csv", EMPTY_AMOUNTS)),
        (PARSE_FEEDBACK, memory_helper.value_shape(FEEDBACK)),
    ):
        ev.add_input_shapes(item, catalogue.body_digest(bodies[item][1]), [desc])
    return ev


@pytest.fixture
def library(tmp_path):
    mem, sha = _commit(tmp_path)
    export = tmp_path / "memory-checkout"
    export_checkout(mem.git_dir, sha, export)
    ev = _evidence(tmp_path, export)
    written = catalogue.write_generated(export, shapes=ev.input_shapes)
    return mem, sha, export, ev, written


def _helper(root: Path):
    """``import memory`` as a cell does it: the export's top-level ``memory.py``."""
    name = f"memory_under_test_{abs(hash(str(root)))}"
    spec = importlib.util.spec_from_file_location(name, root / "memory.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert not module.__package__  # a top-level module, as in the cell
    return module


# --- the generated files ---------------------------------------------------------------------------------


def test_the_catalogue_is_byte_stable_for_the_same_commit(library, tmp_path):
    mem, sha, export, ev, written = library
    assert set(written) == set(catalogue.GENERATED)
    for rel, data in written.items():
        assert (export / rel).read_bytes() == data
    # export the same commit again, and to another path: the same bytes
    export_checkout(mem.git_dir, sha, export)
    again = catalogue.write_generated(export, shapes=ev.input_shapes)
    other = tmp_path / "elsewhere"
    export_checkout(mem.git_dir, sha, other)
    elsewhere = catalogue.write_generated(other, shapes=ev.input_shapes)
    assert again == written == elsewhere
    # the helper and the shape functions are the harness's own files, byte for byte
    here = Path(memory_helper.__file__)
    assert written["memory.py"] == here.read_bytes()
    assert written[".memory/shapes.py"] == Path(shapes_module.__file__).read_bytes()


def test_the_readme_lists_channels_and_functions_but_not_unlisted_ones(library):
    _, _, export, _, written = library
    readme = written["README.md"].decode()
    assert readme.startswith("# Memory library\n")
    assert "never edit it" in readme
    assert (
        "\n## env.worktree_workspace\n\nReaders for the workspace's files.\n\n"
        "- `read_ledger(path: str) -> list`: Read the ledger into rows. (input: path)\n"
        "- `read_any_ledger(path: str) -> list`: Read a ledger that may hold empty amounts. "
        "(input: path)\n"
        "- note `NOTES.md` Amounts: Amounts are decimal strings.\n"
    ) in readme
    assert (
        "- `parse_feedback(obs: dict) -> int`: Read the score from a feedback observation. (input: observation)"
        in readme
    )
    assert "hidden_reader" not in readme and "MemoryInputError" not in readme
    assert readme.index("## env.dialogue_user") < readme.index(
        "## env.worktree_workspace",
    )


def test_the_catalog_holds_sections_input_forms_and_recorded_shapes(library):
    _, _, _, _, written = library
    data = json.loads(written[".memory/catalog.json"])
    assert data["version"] == catalogue.CATALOG_VERSION
    assert set(data["input_forms"]) == {"path", "text", "bytes", "observation", "env"}
    fns = {e["id"]: e for e in data["functions"]}
    assert set(fns) == {READ_LEDGER, READ_ANY, PARSE_FEEDBACK}
    e = fns[READ_LEDGER]
    assert (e["module"], e["name"], e["input"], e["effect"], e["tier"]) == (
        "env.worktree_workspace",
        "read_ledger",
        "path",
        "read",
        None,
    )
    assert e["signature"] == "read_ledger(path: str) -> list"
    assert e["summary"] == "Read the ledger into rows."
    assert e["doc"]["returns"] == "A list of dicts, one per row."
    assert e["doc"]["raises"].startswith("MemoryInputError:")
    assert e["doc"]["example"].startswith(">>> read_ledger(")
    assert e["doc"]["arg_list"] == [
        {"name": "path", "text": "the ledger CSV file (a path)."},
    ]
    (shape,) = e["input_shapes"]
    assert shape["kind"] == "file" and shape["ext"] == ".csv"
    assert shape["shape"]["columns"] == ["vendor_id", "amount", "due"]
    assert (
        "stats" not in shape["shape"]
    )  # row counts vary per file; never part of a shape
    assert fns[PARSE_FEEDBACK]["input_shapes"] == [
        {"kind": "value", "tree": {"notes": ["str"], "score": "int", "type": "str"}},
    ]
    assert [c["channel"] for c in data["channels"]] == [
        "dialogue_user",
        "worktree_workspace",
    ]


def test_a_function_without_recorded_shapes_omits_the_field(library, tmp_path):
    _, _, export, _, _ = library
    data = json.loads(catalogue.render_catalog(export, shapes=lambda item, body: None))
    assert all("input_shapes" not in e for e in data["functions"])
    data = json.loads(catalogue.render_catalog(export))  # no evidence at all
    assert all("input_shapes" not in e for e in data["functions"])


def test_a_changed_body_does_not_inherit_the_old_shapes(library):
    _, _, export, ev, _ = library
    mod = export / "env/worktree_workspace/__init__.py"
    mod.write_text(
        mod.read_text().replace("return list(csv.DictReader(fh))", "return []"),
    )
    data = json.loads(catalogue.render_catalog(export, shapes=ev.input_shapes))
    fns = {e["id"]: e for e in data["functions"]}
    assert "input_shapes" not in fns[READ_LEDGER]  # the gate never checked this body
    assert "input_shapes" in fns[READ_ANY]


def test_reserved_paths():
    for p in (
        "README.md",
        "memory.py",
        ".memory",
        ".memory/catalog.json",
        ".memory/x/y",
    ):
        assert catalogue.reserved(p)
    for p in (
        "env/x/README.md",
        "env/x/__init__.py",
        "memory",
        "unify_memory_testkit.py",
    ):
        assert not catalogue.reserved(p)


def test_the_helper_and_its_shape_functions_import_only_the_standard_library():
    optional = {"yaml", "openpyxl"}  # imported inside try blocks, never required
    for path in (Path(memory_helper.__file__), Path(shapes_module.__file__)):
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.Import):
                names = [a.name.split(".")[0] for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.level == 0:
                names = [(node.module or "").split(".")[0]]
            else:
                continue
            for name in names:
                assert name in sys.stdlib_module_names or name in optional, (path, name)


# --- the memory helper -----------------------------------------------------------------------------------


def test_find_by_shape_lists_the_reader_of_csv_bytes_and_paths(library, tmp_path):
    _, _, export, _, _ = library
    memory = _helper(export)
    found = memory.find(NEW_LEDGER)
    assert [(f.name, f.match) for f in found] == [
        ("env.worktree_workspace.read_ledger", "exact"),
        ("env.worktree_workspace.read_any_ledger", "structure"),
    ]
    assert (
        found[0].input == "path"
        and found[0].signature == "read_ledger(path: str) -> list"
    )
    assert found[0].summary == "Read the ledger into rows."
    path = tmp_path / "march.csv"
    path.write_bytes(NEW_LEDGER)
    assert memory.find(str(path)) == found
    assert memory.find(path) == found  # a PathLike
    assert memory.find(NEW_LEDGER.decode()) == found  # the file's text
    assert memory.find(NEW_LEDGER) == found  # deterministic


def test_find_by_shape_lists_the_parser_of_a_dict_observation(library):
    _, _, export, _, _ = library
    memory = _helper(export)
    assert [(f.name, f.match) for f in memory.find(FEEDBACK)] == [
        ("env.dialogue_user.parse_feedback", "exact"),
    ]
    other_values = {
        "type": "Feedback",
        "score": 9,
        "notes": [],
    }  # same keys, an empty list
    assert [(f.name, f.match) for f in memory.find(other_values)] == [
        ("env.dialogue_user.parse_feedback", "structure"),
    ]
    assert [f.name for f in memory.find(json.dumps(FEEDBACK))] == [
        "env.dialogue_user.parse_feedback",
    ]


@pytest.mark.parametrize(
    "value",
    [
        42,
        None,
        {"unrelated": 1},
        [1, 2],
        "just some words",
        b"\x00\x01\x02",
        b"a;b\n1;2\n",
    ],
    ids=["int", "none", "other-keys", "list", "text", "binary", "other-table"],
)
def test_find_lists_nothing_for_an_unrelated_value(library, value):
    _, _, export, _, _ = library
    assert _helper(export).find(value) == []


def test_find_never_matches_by_words(library):
    """A value naming the functions, their channel or their columns in prose has no shape: nothing."""
    _, _, export, _, _ = library
    prose = "read_ledger vendor_id amount due worktree_workspace ledger Feedback score"
    assert _helper(export).find(prose) == []


def test_describe_renders_the_docstring_sections_and_recorded_inputs(library):
    _, _, export, _, _ = library
    memory = _helper(export)
    text = memory.describe("read_ledger")
    assert text.startswith(
        "env.worktree_workspace.read_ledger(path: str) -> list\nRead the ledger into rows.\n",
    )
    for part in (
        "Input: path (a path to a work-tree file)",
        "Effect: read",
        "Args:\n    path: the ledger CSV file (a path).",
        "Returns:\n    A list of dicts, one per row.",
        "Raises:\n    MemoryInputError: when the file lacks the ledger's columns.",
        'Example:\n    >>> read_ledger("env/worktree_workspace/tests/data/ledger.csv")[0]["vendor_id"]',
        "Built and checked on inputs shaped as:\n    - a .csv table with columns vendor_id, amount, due",
        f"relative to the library root ({export})",
    ):
        assert part in text, (part, text)
    assert repr(text) == str(text)  # a cell's last expression shows the text itself
    assert memory.describe("env.worktree_workspace.read_ledger") == text
    assert memory.describe("worktree_workspace.read_ledger") == text
    assert memory.describe(READ_LEDGER) == text

    def read_ledger():  # a stand-in for the imported function
        pass

    read_ledger.__module__ = "env.worktree_workspace"
    assert memory.describe(read_ledger) == text
    feedback = memory.describe("parse_feedback")
    assert "- a value with keys notes, score, type" in feedback
    assert "Example paths" not in feedback  # it has no example section
    with pytest.raises(LookupError, match=r"memory\.catalog\(\) lists every function"):
        memory.describe("no_such_function")


def test_catalog_is_the_readme(library):
    _, _, export, _, written = library
    memory = _helper(export)
    assert memory.catalog() == written["README.md"].decode()


# --- the prompt's channel catalogue ------------------------------------------------------------------------


def test_the_prompt_section_shows_channels_not_functions_and_is_byte_stable(
    library,
    tmp_path,
):
    mem, sha, export, _, _ = library
    text = render_memory_section(export)
    assert text.endswith(
        "\nChannels:\n"
        "- `env.dialogue_user`: 1 function\n"
        "- `env.worktree_workspace`: 2 functions, 1 note. Readers for the workspace's files.\n",
    )
    for name in ("read_ledger", "parse_feedback", "Read the ledger", "hidden_reader"):
        assert name not in text
    for phrase in (
        "README.md",
        "memory.find(value)",
        "memory.describe",
        "help(",
        "example",
    ):
        assert phrase in text
    assert f"`{export}`" in text
    export_checkout(mem.git_dir, sha, export)
    catalogue.write_generated(export)
    assert render_memory_section(export) == text
    flagged = render_memory_section(export, {"dialogue_user"})
    assert "- `env.dialogue_user`: 1 function (suspect:" in flagged
    assert (
        "worktree_workspace`: 2 functions, 1 note. Readers for the workspace's files.\n"
        in flagged
    )
