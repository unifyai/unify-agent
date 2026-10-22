"""The gate's safety refusals on the library layout (rev 2), through the real gate on the shared library world.

What the import system could load, what lies outside the layout, a link or an executable file, a key-shaped string
in the library or the manifest, and a pass id already recorded are refused, and a refusal before extraction runs
nothing. A failing run's output reaches the reasons and the pass row with every key-shaped string redacted. These are
the library-layout forms of the older env/<channel> cases of test_gate.py and test_gate_layout.py (the safety rules
themselves, :func:`unify.memory.manifest.unsafe_path` and :func:`unify.memory.layout.misplaced`, are unchanged).
"""

from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path

import pytest

from tests.memory import library_example_world as lw
from tests.memory.library_example_world import library_example_world  # noqa: F401 - a fixture
from unify.memory.gate import Gate
from unify.memory.library_checks import LibraryConfig
from unify.memory.sandbox_run import PytestOutcome
from unify.memory.snapshot import listing, materialise

pytestmark = pytest.mark.skipif(shutil.which("bwrap") is None, reason="bubblewrap required")

KEY = "sk-or-v1-" + "ab" * 32  # pragma: allowlist secret
_MODULE_RULE = (
    "a module is memory/<package>/<module>.py, a lower-case identifier other than tests; the package's description "
    "is memory/<package>/README.md"
)
_ROOT_RULE = "the library holds only memory/ and notes/"
_NO_SUBPACKAGES = "packages have no subpackages; tests go in memory/<package>/tests/"

#: (extra files, the reasons the gate gives first, in order): each refused before anything is extracted or run.
EARLY = [
    (
        {"memory/acct/users.pyc": "x"},
        [
            f"G1: file memory/acct/users.pyc: {_MODULE_RULE}",
            "G6: bytecode or native code file memory/acct/users.pyc",
        ],
    ),
    (
        {"memory/acct/__pycache__/users.cpython-312.pyc": "x"},
        [
            f"G1: file memory/acct/__pycache__/users.cpython-312.pyc: {_NO_SUBPACKAGES}",
            "G6: bytecode cache path memory/acct/__pycache__/users.cpython-312.pyc",
        ],
    ),
    ({"x.pth": "x"}, ["G6: forbidden file x.pth", f"G1: file x.pth: {_ROOT_RULE}"]),
    ({"memory/acct/sitecustomize.py": "x = 1\n"}, ["G6: forbidden file memory/acct/sitecustomize.py"]),
    ({"notes.txt": "x"}, [f"G1: file notes.txt: {_ROOT_RULE}"]),
    (
        {"json/notes.txt": "x"},
        [
            f"G1: file json/notes.txt: {_ROOT_RULE}",
            "G6: root package json/ (json/notes.txt) shadows the json module",
        ],
    ),
    ({"helpers.py": "x = 1\n"}, [f"G1: file helpers.py: {_ROOT_RULE}"]),
    ({"memory/acct/util/__init__.py": "x = 1\n"}, [f"G1: file memory/acct/util/__init__.py: {_NO_SUBPACKAGES}"]),
]


@pytest.mark.parametrize("extra, first", EARLY, ids=[next(iter(e)) for e, _ in EARLY])
def test_importable_artefacts_and_paths_outside_the_layout_are_refused_before_any_run(
    library_example_world,
    extra,
    first,
):
    mem, ev, blobs, gate = library_example_world
    files, fixtures = lw.files_and_fixtures(blobs.get)
    res = gate().check(mem.head(), lw.candidate(mem, {**files, **extra}), lw.manifest(fixtures))
    assert not res.passed and res.reasons[: len(first)] == first, res.reasons
    # stopped before extraction: no later check was evaluated
    assert "G3: not evaluated" in res.reasons, res.reasons


def test_a_key_shaped_string_in_the_library_or_the_manifest_is_refused_and_never_quoted(library_example_world):
    mem, ev, blobs, gate = library_example_world
    files, fixtures = lw.files_and_fixtures(blobs.get)
    keyed = {**files, lw.MODULE: lw.MODULE_SOURCE + f"\nTOKEN = {KEY!r}\n"}
    res = gate().check(mem.head(), lw.candidate(mem, keyed), lw.manifest(fixtures))
    assert not res.passed and res.reasons[0] == f"G6: key-shaped string in {lw.MODULE}", res.reasons
    assert not any(KEY in r for r in res.reasons)
    res = gate().check(mem.head(), lw.candidate(mem, files), lw.manifest(fixtures, summary=KEY))
    assert not res.passed and res.reasons[0] == "G6: key-shaped string in the manifest", res.reasons
    assert not any(KEY in r for r in res.reasons)


def test_a_module_that_does_not_parse_is_refused(library_example_world):
    mem, ev, blobs, gate = library_example_world
    files, fixtures = lw.files_and_fixtures(blobs.get)
    broken = {**files, lw.MODULE: lw.MODULE_SOURCE + "\ndef broken(:\n"}
    res = gate().check(mem.head(), lw.candidate(mem, broken), lw.manifest(fixtures))
    assert not res.passed and res.reasons[0] == f"G1: {lw.ITEM_ID} (function) is not in the candidate", res.reasons


def test_a_failing_runs_output_is_kept_with_every_key_shaped_string_redacted(library_example_world):
    """Rev 2's gate is labels only: a red test is a label on its item, never a refusal; its output reaches the
    reasons and the stored pass row with the key-shaped string redacted."""
    mem, ev, blobs, _ = library_example_world
    files, fixtures = lw.files_and_fixtures(blobs.get)
    ran: list[str] = []

    def leaky(target, *, python, ro, rw, cwd, timeout_s=300.0, env=None, **kw):
        ran.append(target)
        return PytestOutcome(failed={"t::a"}, returncode=1, output=f"E   assert {KEY}")

    gate = Gate(
        mem,
        ev,
        blobs,
        action_lookup=lw.lookup,
        pytest_runner=leaky,
        library_mode=LibraryConfig(episodes=lw.EPISODES.get),
    )
    res = gate.merge(mem.head(), lw.candidate(mem, files), lw.manifest(fixtures), "p3", "write", None, "0")
    assert ran and any("<redacted:key-shaped>" in r for r in res.reasons), res.reasons
    assert not any(KEY in r for r in res.reasons)
    (stored,) = ev.db.execute("SELECT reasons FROM passes WHERE pass_id='p3'").fetchone()
    assert KEY not in stored and "<redacted:key-shaped>" in stored


def test_preview_refuses_links_and_executables_as_the_committed_listing_would(library_example_world, tmp_path):
    mem, ev, blobs, gate = library_example_world
    files, fixtures = lw.files_and_fixtures(blobs.get)
    tree = tmp_path / "tree"
    for rel, text in files.items():
        (tree / rel).parent.mkdir(parents=True, exist_ok=True)
        (tree / rel).write_text(text)
    (tree / "memory/acct/tests/data/link.json").symlink_to("/etc/hostname")
    os.chmod(tree / lw.MODULE, 0o755)
    assert gate().preview(mem.head(), tree, lw.manifest(fixtures)) == [
        f"G6: the candidate holds executable file {lw.MODULE}",
        "G6: the candidate holds symlink memory/acct/tests/data/link.json",
    ]


def test_the_gate_refuses_a_recorded_pass_id_and_keeps_the_earlier_attempt(library_example_world):
    mem, ev, blobs, gate = library_example_world
    files, fixtures = lw.files_and_fixtures(blobs.get)
    parent = mem.head()
    bad = gate().merge(parent, lw.candidate(mem, {**files, "notes.txt": "x"}), lw.manifest(fixtures), "dup", "write",
                       None, "0.5")  # fmt: skip
    assert not bad.passed
    res = gate().merge(parent, lw.candidate(mem, files), lw.manifest(fixtures), "dup", "write", None, "0.01")
    assert not res.passed and "pass dup is already recorded" in res.reasons, res.reasons
    assert mem.head() == parent
    row = ev.db.execute("SELECT passed, usd FROM passes WHERE pass_id='dup'").fetchone()
    assert row == (0, "0.5")  # the earlier failed attempt is preserved


def test_the_gate_inspects_exactly_the_committed_tree(library_example_world):
    """Trees are built from the committed blobs, so the inspected file set is the merged one."""
    mem, ev, blobs, gate = library_example_world
    files, fixtures = lw.files_and_fixtures(blobs.get)
    cand = lw.candidate(mem, files)
    committed, refused = listing(mem, cand)
    assert refused == [] and set(files) <= set(committed)
    with tempfile.TemporaryDirectory() as d:
        tree = materialise(mem, committed, Path(d) / "t")
        assert (tree / lw.MODULE).read_text() == files[lw.MODULE]
        assert {p.relative_to(tree).as_posix() for p in tree.rglob("*") if p.is_file()} == set(committed)


# --- G1: what the manifest names must exist ------------------------------------------------------------------------


def test_an_unknown_source_episode_and_an_item_not_in_the_candidate_are_refused(library_example_world):
    mem, ev, blobs, gate = library_example_world
    files, fixtures = lw.files_and_fixtures(blobs.get)
    cand = lw.candidate(mem, files)
    assert gate().check(mem.head(), cand, lw.manifest(fixtures)).passed
    res = gate().check(mem.head(), cand, lw.manifest(fixtures, lw.item(source_episodes=["e1", "nope"])))
    assert not res.passed and res.reasons[0] == f"G1: {lw.ITEM_ID} cites unknown episode nope", res.reasons
    ghost = "memory.acct.users:ghost"
    res = gate().check(mem.head(), cand, lw.manifest(fixtures, lw.item(item=ghost)))
    assert not res.passed and f"G1: {ghost} (function) is not in the candidate" in res.reasons, res.reasons


# --- preview: the writer's read-only check of an uncommitted tree ---------------------------------------------------


def test_preview_judges_an_uncommitted_tree_without_running_or_writing_anything(
    library_example_world,
    tmp_path,
    monkeypatch,
):
    mem, ev, blobs, _ = library_example_world
    files, fixtures = lw.files_and_fixtures(blobs.get)

    def boom(*a, **kw):
        raise AssertionError("preview must not run tests or held-out values")

    # nothing confined may run (the held-out plan of the older gate is gone; tests run through the runner)
    monkeypatch.setattr("unify.memory.sandbox_run.run_confined", boom)
    gate = Gate(
        mem,
        ev,
        blobs,
        action_lookup=lw.lookup,
        pytest_runner=boom,
        library_mode=LibraryConfig(episodes=lw.EPISODES.get),
    )
    tree = tmp_path / "tree"
    for rel, text in files.items():
        (tree / rel).parent.mkdir(parents=True, exist_ok=True)
        (tree / rel).write_text(text)
    objects = sorted(p.relative_to(mem.git_dir) for p in (mem.git_dir / "objects").rglob("*"))
    parent = mem.head()
    assert gate.preview(parent, tree, lw.manifest(fixtures)) == []
    assert gate.preview(parent, tree, {"items": 3}) == ["G1: malformed manifest: items must be a list"]
    # nothing is written: no git object, no pass row
    assert sorted(p.relative_to(mem.git_dir) for p in (mem.git_dir / "objects").rglob("*")) == objects
    assert mem.head() == parent and ev.db.execute("SELECT COUNT(*) FROM passes").fetchone() == (0,)


MARKER = "LIB-CONTENT-MARKER"


def test_many_refused_artefacts_are_listed_up_to_ten_and_never_their_contents(library_example_world):
    mem, ev, blobs, gate = library_example_world
    files, fixtures = lw.files_and_fixtures(blobs.get)
    bad = {f"memory/acct/tests/h{i:02d}.pyc": f"\x00{MARKER}\x00" for i in range(12)}
    res = gate().check(mem.head(), lw.candidate(mem, {**files, **bad}), lw.manifest(fixtures))
    g6 = [r for r in res.reasons if r.startswith("G6:")]
    assert g6[:10] == [f"G6: bytecode or native code file {p}" for p in sorted(bad)[:10]]
    assert g6[10:] == ["G6: and 2 more such paths"]
    assert not res.passed and "G3: not evaluated" in res.reasons and MARKER not in "\n".join(res.reasons)


def test_compiled_code_declared_as_support_is_refused_and_never_merged(library_example_world):
    mem, ev, blobs, gate = library_example_world
    files, fixtures = lw.files_and_fixtures(blobs.get)
    path = "memory/acct/tests/helper.pyc"
    parent = mem.head()
    cand = lw.candidate(mem, {**files, path: f"\x00{MARKER}\x00"})
    res = gate().merge(parent, cand, lw.manifest(fixtures, support=[path]), "pi4", "write", None, "0")
    first = (
        f"G1: malformed manifest: support file {path} is not a test helper under memory/<package>/tests/ or a module "
        "memory/<package>/<module>.py"
    )
    assert not res.passed and res.reasons[0] == first and "G3: not evaluated" in res.reasons, res.reasons
    assert mem.head() == parent
    passed, stored = ev.db.execute("SELECT passed, reasons FROM passes WHERE pass_id='pi4'").fetchone()
    assert passed == 0 and MARKER not in stored
