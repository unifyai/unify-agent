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
from unify.memory_v2.shape_rows import lookup_from, shapes_at, snapshot_rows
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


def parse_status(obs: dict) -> str:
    """Read a status envelope.

    Effect: read
    Input: observation
    """
    return obs["status"]


def parse_submit(obs: dict) -> list:
    """Read the grid of a submission message.

    Effect: read
    Input: observation
    """
    return obs["grid"]


def read_pairs(path: str) -> list:
    """Read a headerless table of pairs.

    Effect: read
    Input: path
    """
    return []
'''
NOTES = "# notes\n\n## Amounts\nAmounts are decimal strings.\n"

LEDGER = b"vendor_id,amount,due\nV-17,12.50,2026-10-14\nV-18,3.00,2026-10-20\n"
EMPTY_AMOUNTS = b"vendor_id,amount,due\nV-17,,2026-10-14\nV-18,,2026-10-20\n"
NEW_LEDGER = b"vendor_id,amount,due\nV-99,7.25,2026-11-01\n"
FEEDBACK = {"type": "Feedback", "score": 3, "notes": ["a"]}
STATUS = {"status": "ok"}
SUBMIT = {"type": "SubmitFeedback", "attempt": 1, "grid": [[0, 1], [1, 0]]}  # ARC-like
PAIRS = b"1,2\n3,4\n"

READ_LEDGER = "env/worktree_workspace:read_ledger"
READ_ANY = "env/worktree_workspace:read_any_ledger"
PARSE_FEEDBACK = "env/dialogue_user:parse_feedback"
PARSE_STATUS = "env/dialogue_user:parse_status"
PARSE_SUBMIT = "env/dialogue_user:parse_submit"
READ_PAIRS = "env/dialogue_user:read_pairs"


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


def _descriptors():
    return {
        READ_LEDGER: memory_helper.file_shape("ledger/2026/ledger.csv", LEDGER),
        READ_ANY: memory_helper.file_shape("old/ledger.csv", EMPTY_AMOUNTS),
        PARSE_FEEDBACK: memory_helper.value_shape(FEEDBACK),
        PARSE_STATUS: memory_helper.value_shape(STATUS),
        PARSE_SUBMIT: memory_helper.value_shape(SUBMIT),
        READ_PAIRS: memory_helper.file_shape("pairs.csv", PAIRS),
    }


def _evidence(tmp_path, export, sha):
    """An evidence store holding the snapshot a merge of *sha* would have frozen for these bodies."""
    ev = EvidenceStore(tmp_path / "e.sqlite")
    bodies = item_bodies(export)
    rows = {
        item: {
            "body": catalogue.body_digest(bodies[item][1]),
            "shapes": [desc],
            "backfilled": False,
        }
        for item, desc in _descriptors().items()
        if desc is not None
    }
    assert ev.write_commit_shapes(sha, rows)
    return ev


def _shapes(ev, sha):
    return lookup_from(ev.commit_shapes(sha) or {})


@pytest.fixture
def library(tmp_path):
    mem, sha = _commit(tmp_path)
    export = tmp_path / "memory-checkout"
    export_checkout(mem.git_dir, sha, export)
    ev = _evidence(tmp_path, export, sha)
    rows = shapes_at(mem, ev, sha, export, freeze=True)  # its own frozen snapshot
    written = catalogue.write_generated(export, shapes=lookup_from(rows))
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
    again = catalogue.write_generated(export, shapes=_shapes(ev, sha))
    other = tmp_path / "elsewhere"
    export_checkout(mem.git_dir, sha, other)
    elsewhere = catalogue.write_generated(other, shapes=_shapes(ev, sha))
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
    assert set(fns) == {
        READ_LEDGER,
        READ_ANY,
        PARSE_FEEDBACK,
        PARSE_STATUS,
        PARSE_SUBMIT,
        READ_PAIRS,
    }
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
    (feedback,) = fns[PARSE_FEEDBACK]["input_shapes"]
    assert feedback == {
        "kind": "value",
        "tree": {"notes": ["str"], "score": "int", "type": "str"},
        "lengths": {"notes": "1"},
    }
    assert "input_shapes_backfilled" not in fns[PARSE_FEEDBACK]
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
    _, sha, export, ev, _ = library
    mod = export / "env/worktree_workspace/__init__.py"
    mod.write_text(
        mod.read_text().replace("return list(csv.DictReader(fh))", "return []"),
    )
    data = json.loads(catalogue.render_catalog(export, shapes=_shapes(ev, sha)))
    fns = {e["id"]: e for e in data["functions"]}
    assert "input_shapes" not in fns[READ_LEDGER]  # the gate never checked this body
    assert "input_shapes" in fns[READ_ANY]


def test_reexporting_an_older_commit_after_a_later_merge_gives_the_same_bytes(
    library,
    tmp_path,
):
    """I5: shapes are frozen per memory commit, so a later merge never changes an older commit's catalogue."""
    mem, sha, export, ev, written = library
    base = mem.head()
    with (
        mem.temp_checkout() as wt,
    ):  # a later commit: read_ledger unchanged, a note added
        (wt / "env/worktree_workspace/NOTES.md").write_text(
            NOTES + "\n## Dates\nISO dates.\n",
        )
        later = mem.commit_all(wt, "later", {})
    mem.fast_forward("main", later, expected_old=base)
    later_tree = tmp_path / "later"
    export_checkout(mem.git_dir, later, later_tree)
    digest = catalogue.body_digest(item_bodies(later_tree)[READ_LEDGER][1])
    extra = memory_helper.file_shape(
        "ledger/2027/ledger.tsv",
        LEDGER.replace(b",", b"\t"),
    )
    prev = shapes_at(mem, ev, sha, export)
    assert ev.write_commit_shapes(
        later,
        snapshot_rows(later_tree, prev, {READ_LEDGER: (digest, [extra])}),
    )
    assert not ev.write_commit_shapes(later, {})  # frozen: never rewritten
    # the older commit, exported again, renders the same bytes as before the later merge
    export_checkout(mem.git_dir, sha, export)
    again = catalogue.write_generated(
        export,
        shapes=lookup_from(shapes_at(mem, ev, sha, export, freeze=True)),
    )
    assert again == written
    # the later commit shows both of read_ledger's recorded shapes
    data = json.loads(
        catalogue.render_catalog(
            later_tree,
            shapes=lookup_from(shapes_at(mem, ev, later, later_tree, freeze=True)),
        ),
    )
    fns = {e["id"]: e for e in data["functions"]}
    assert len(fns[READ_LEDGER]["input_shapes"]) == 2
    assert len(fns[PARSE_FEEDBACK]["input_shapes"]) == 1  # carried over unchanged


def test_functions_without_recorded_shapes_are_backfilled_from_covers_and_frozen(
    tmp_path,
):
    from unify.memory_v2.blobs import BlobStore
    from unify.memory_v2.episodes import Action

    mem, sha = _commit(tmp_path)  # a commit from before shape snapshots: none recorded
    export = tmp_path / "co"
    export_checkout(mem.git_dir, sha, export)
    ev = EvidenceStore(tmp_path / "e.sqlite")
    blobs = BlobStore(tmp_path / "b")
    blob = blobs.put(LEDGER)
    ev.add_cover(READ_LEDGER, "e1", 0)
    ev.add_cover(PARSE_FEEDBACK, "e1", 1)
    recorded = {
        ("e1", 0): Action(
            0,
            "worktree:workspace",
            "read",
            ["ledger/2026/ledger.csv"],
            {},
            {"blob_before": blob},
            "ok",
            kind="worktree",
        ),
        ("e1", 1): Action(
            1,
            "dialogue:user",
            "say",
            ["hi"],
            {},
            FEEDBACK,
            "ok",
            kind="dialogue",
        ),
    }

    def lookup(eid, idx):
        return recorded.get((eid, idx))

    rows = shapes_at(mem, ev, sha, export, lookup=lookup, blobs=blobs, freeze=True)
    data = json.loads(catalogue.render_catalog(export, shapes=lookup_from(rows)))
    fns = {e["id"]: e for e in data["functions"]}
    assert fns[READ_LEDGER]["input_shapes"] == [
        memory_helper.file_shape("ledger/2026/ledger.csv", LEDGER),
    ]
    assert fns[READ_LEDGER]["input_shapes_backfilled"] is True
    assert fns[PARSE_FEEDBACK]["input_shapes"] == [memory_helper.value_shape(FEEDBACK)]
    assert "input_shapes" not in fns[READ_ANY]  # no covers recorded
    # frozen: a cover recorded later does not change this commit's rows
    ev.add_cover(READ_ANY, "e1", 0)
    assert (
        shapes_at(mem, ev, sha, export, lookup=lookup, blobs=blobs, freeze=True) == rows
    )


def test_reserved_paths():
    for p in (
        "README.md",
        "memory.py",
        ".memory",
        ".memory/catalog.json",
        ".memory/x/y",
    ):
        assert catalogue.reserved(p)
    # anything a cell's `import memory` or `import env` could resolve to, and root extension modules
    for p in (
        "memory",
        "memory.abi3.so",
        "memory.cpython-312-x86_64-linux-gnu.so",
        "memory.pyc",
        "memory/__init__.py",
        "memory/data.txt",
        "env.py",
        "env.so",
        "helper.so",
        "helper.pyd",
    ):
        assert catalogue.reserved(p), p
    for p in (
        "env/x/README.md",
        "env/x/__init__.py",
        "env/x/tests/data.json",
        "unify_memory_testkit.py",
        "memoryless.txt",
    ):
        assert not catalogue.reserved(p), p
    # the layout refuses them too, and an extension module anywhere
    from unify.memory_v2.manifest import layout_allowed

    for p in ("memory.abi3.so", "env.py", "memory/notes.md", "env/x/tests/fast.so"):
        assert not layout_allowed(p), p
    assert layout_allowed("env/x/tests/data.json") and layout_allowed(
        "unify_memory_testkit.py",
    )


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
    assert found[0].reason == "matched amount, due, vendor_id"
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
        True,
        {"unrelated": 1},
        [1, 2],
        [7],
        [],
        {},
        "just some words",
        b"\x00\x01\x02",
        b"a;b\n1;2\n",
        "5,6\n7,8\n",  # headerless, like the recorded pairs
        "2024-01-01,3\n2024-01-02,4\n",
        {"status": 404},  # the status envelope's key, with another leaf type
        [{"id": 1, "name": "a"}],  # records with other keys
    ],
    ids=[
        "int",
        "none",
        "bool",
        "other-keys",
        "list",
        "one-int-list",
        "empty-list",
        "empty-dict",
        "text",
        "binary",
        "other-table",
        "headerless",
        "headerless-dates",
        "status-int",
        "other-records",
    ],
)
def test_find_lists_nothing_for_an_unrelated_value(library, value):
    _, _, export, _, _ = library
    assert _helper(export).find(value) == []


def test_find_still_finds_real_arc_and_office_shapes(library, tmp_path):
    """The sharper signature keeps the real cases: an ARC message, a status envelope, an office ledger."""
    _, _, export, _, _ = library
    memory = _helper(export)
    other_arc = {
        "type": "SubmitFeedback",
        "attempt": 2,
        "grid": [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
    }
    found = memory.find(other_arc)
    assert [(f.name, f.match, f.reason) for f in found] == [
        ("env.dialogue_user.parse_submit", "exact", "matched attempt, grid, type"),
    ]
    big = {
        "type": "SubmitFeedback",
        "attempt": 2,
        "grid": [[0] * 12 for _ in range(12)],
    }
    assert [(f.name, f.match) for f in memory.find(big)] == [
        ("env.dialogue_user.parse_submit", "structure"),  # another length class
    ]
    assert [f.name for f in memory.find({"status": "fine"})] == [
        "env.dialogue_user.parse_status",
    ]
    assert [f.name for f in memory.find(NEW_LEDGER)][
        0
    ] == "env.worktree_workspace.read_ledger"


def test_find_returns_at_most_max_results(library):
    _, _, export, _, _ = library
    memory = _helper(export)
    assert memory.MAX_RESULTS == 5
    memory.MAX_RESULTS = 1
    assert [f.name for f in memory.find(NEW_LEDGER)] == [
        "env.worktree_workspace.read_ledger",
    ]


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
        "- `env.dialogue_user`: 4 functions\n"
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
    assert "- `env.dialogue_user`: 4 functions (suspect:" in flagged
    assert (
        "worktree_workspace`: 2 functions, 1 note. Readers for the workspace's files.\n"
        in flagged
    )
