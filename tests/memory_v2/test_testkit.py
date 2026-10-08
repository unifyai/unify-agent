"""The library test kit (memory v2.1 stage 5): one versioned ``memlab`` wherever a library's tests run.

A stored library must never depend on a gate switch: its tests import the same kit in the gate (any switch
setting), in Sol's box and in the actor's export, and library code never uses it.
"""

import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

from unify.memory_v2 import testkit
from unify.memory_v2.blobs import BlobStore
from unify.memory_v2.qa import QAConfig
from unify.memory_v2.sol_pass import SolPass, _MEMLAB_FILES
from tests.memory_v2.test_episodes import _ep

USES_REPLAY = b"from memlab.replay import RecordedEnv\n\ndef test_x():\n    assert RecordedEnv([])\n"
PLAIN = b"from env.phone import current_datetime\n\ndef test_x():\n    assert current_datetime\n"


def _stage(root, store, refs):
    return testkit.stage(
        root,
        refs,
        has_blob=store.has,
        read_blob=store.get,
        blob_size=store.size,
    )


def test_the_kit_is_the_versioned_module_set_the_pin_and_the_named_blobs(
    tmp_path,
    monkeypatch,
):
    store = BlobStore(tmp_path / "store")
    a, b = store.put(b"recorded a"), store.put(b"recorded b")
    root = tmp_path / "kit"
    (root / "blobs").mkdir(parents=True)
    (root / "blobs" / b).write_bytes(b"already staged")  # a blob already there is kept
    assert _stage(root, store, [a, "0" * 64, b]) == []
    lab = root / "memlab"
    assert sorted(p.name for p in lab.iterdir()) == sorted(
        list(testkit.MODULES) + ["__init__.py", "gitio.py"],
    )
    assert f'KIT_VERSION = "{testkit.KIT_VERSION}"' in (lab / "__init__.py").read_text()
    assert "git is not available" in (lab / "gitio.py").read_text()
    pin = Path(testkit.__file__).with_name("pin.py").read_text()
    assert (root / "_memv2_pin.py").read_text() == pin
    assert (root / "blobs" / a).read_bytes() == b"recorded a"
    assert (root / "blobs" / b).read_bytes() == b"already staged"
    assert not (root / "blobs" / ("0" * 64)).exists()
    # Sol's toolkit before stage 5 is the kit without memlab.inputs: one module set everywhere
    assert set(testkit.MODULES) == set(_MEMLAB_FILES) | {"inputs.py"}
    monkeypatch.setattr(testkit, "MAX_REF_BLOBS", 1)
    (note,) = _stage(tmp_path / "kit2", store, [a, b])
    assert note.startswith("[qa:blobs] 1 referenced blob(s) not mounted")


def test_a_staged_kit_imports_alone_and_reads_blobs_beside_itself(tmp_path):
    store = BlobStore(tmp_path / "store")
    sha = store.put(b"recorded screen")
    root = tmp_path / "anywhere" / ".memlab"
    _stage(root, store, [sha])
    code = (
        "import sys; sys.path[:0] = [sys.argv[1]]\n"
        "import memlab, memlab.analysis.shapes, memlab.fingerprint, memlab.replay\n"
        "from memlab.inputs import blob, from_blob, inputs\n"
        "print(memlab.KIT_VERSION, blob(sys.argv[2]).decode(), len(inputs('x', [from_blob(sys.argv[2])])))\n"
    )
    r = subprocess.run(
        [sys.executable, "-I", "-c", code, str(root), sha],
        capture_output=True,
        text=True,
        env={"PATH": "/usr/bin:/bin"},
        timeout=60,
    )
    assert r.stdout.split() == [
        testkit.KIT_VERSION,
        "recorded",
        "screen",
        "1",
    ], r.stderr


def test_which_tests_use_the_kit():
    store_has = {"ab" * 32}.__contains__
    assert testkit.uses_kit([USES_REPLAY], store_has, [])
    assert testkit.uses_kit([b"import _memv2_pin\n"], store_has, [])
    assert testkit.uses_kit([PLAIN], store_has, ["ab" * 32])  # names a recorded blob
    assert not testkit.uses_kit([PLAIN], store_has, ["cd" * 32])  # a hex token, no blob
    assert not testkit.uses_kit([PLAIN, b"not python ("], store_has, [])
    assert testkit.is_test_side("env/phone/tests/test_x.py")
    assert testkit.is_test_side("unify_memory_testkit.py")
    assert not testkit.is_test_side("env/phone/__init__.py")


def test_library_code_using_the_kit_is_found_by_line():
    src = b"""import importlib

from memlab.replay import RecordedEnv


def f(apis):
    import _memv2_pin
    importlib.import_module("memlab.inputs")
    return open("/inputs/blobs/x").read()


def g(text):
    return "memlab is a word here, /inputsfoo is not a path"
"""
    assert testkit.library_uses(src) == [3, 7, 8, 9]
    assert testkit.library_uses(b"def f(x):\n    return x\n") == []
    assert testkit.names_inputs_path(src) == [9]


def test_a_tests_memlab_imports_resolve_against_the_current_kit():
    good = b"""import memlab
import memlab.analysis.shapes
from memlab import KIT_VERSION, inputs
from memlab.analysis import shapes
from memlab.episodes import Action
from memlab.inputs import blob, from_action, from_blob, inputs as cases
from memlab.replay import RecordedEnv
import pytest
"""
    assert testkit.unresolved(good) == []
    bad = b"""import memlab.nope
from memlab.replay import RecordedEnv, Nope
from memlab.analysis import nothing
from memlab import gone
"""
    assert testkit.unresolved(bad) == [
        (1, "memlab.nope"),
        (2, "memlab.replay.Nope"),
        (3, "memlab.analysis.nothing"),
        (4, "memlab.gone"),
    ]
    # the documented example of memlab.inputs resolves
    doc = (Path(testkit.__file__).with_name("inputs.py")).read_text().split("::", 1)[1]
    example = "\n".join(
        line[4:] for line in doc.splitlines() if line.startswith("    from memlab")
    )
    assert example and testkit.unresolved(example.encode()) == []


def test_a_tree_gets_the_kit_only_when_its_tests_use_it(tmp_path):
    store = BlobStore(tmp_path / "store")
    sha = store.put(b"recorded")
    kw = dict(has_blob=store.has, read_blob=store.get, blob_size=store.size)
    plain = tmp_path / "plain"
    (plain / "env/phone/tests").mkdir(parents=True)
    (plain / "env/phone/tests/test_x.py").write_bytes(PLAIN)
    assert not testkit.stage_for_tree(plain, tmp_path / "k1", **kw)
    assert not (tmp_path / "k1").exists()
    assert testkit.stage_for_tree(plain, tmp_path / "k2", force=True, **kw)
    assert (tmp_path / "k2/memlab/inputs.py").is_file()
    uses = tmp_path / "uses"
    (uses / "env/phone/tests").mkdir(parents=True)
    (uses / "env/phone/tests/test_x.py").write_text(f"SCREEN = {sha!r}\n")
    assert testkit.stage_for_tree(uses, tmp_path / "k3", **kw)
    assert (tmp_path / "k3/blobs" / sha).read_bytes() == b"recorded"


def _sol(tmp_path, qa, store):
    ep = _ep(episode_id="e9", actions=[])
    return SimpleNamespace(
        gate=SimpleNamespace(blobs=store, qa=qa),
        load=lambda eid: ep,
        _qa=qa,
    ), SimpleNamespace(kind="k", channel="c", episodes=["e9"], lift=False)


def test_sols_box_gets_the_kit_when_a_switch_is_on_or_the_library_uses_it(tmp_path):
    store = BlobStore(tmp_path / "store")
    sha = store.put(b"recorded")
    plain = tmp_path / "plain"
    (plain / "env/phone/tests").mkdir(parents=True)
    (plain / "env/phone/tests/test_x.py").write_bytes(PLAIN)
    uses = tmp_path / "uses"
    (uses / "env/phone/tests").mkdir(parents=True)
    (uses / "env/phone/tests/test_x.py").write_text(
        f"from memlab.inputs import blob\n\nB = {sha!r}\n",
    )
    cases = {
        "off-plain": (QAConfig(), plain, False),
        "off-uses": (QAConfig(), uses, True),
        "on-plain": (QAConfig(replay=True), plain, True),
    }
    for name, (qa, tree, kit) in cases.items():
        self, req = _sol(tmp_path, qa, store)
        inputs = tmp_path / name
        SolPass._stage_inputs(self, req, inputs, tree)
        lab = inputs / "memlab"
        assert (lab / "inputs.py").is_file() is kit, name
        assert (inputs / "_memv2_pin.py").is_file() is kit, name
        assert (inputs / "blobs" / sha).is_file() is (tree is uses), name
        assert (lab / "replay.py").is_file() and (
            lab / "analysis"
        ).is_dir()  # the toolkit always
        if (
            not kit
        ):  # every switch off and a library not using the kit: Sol's box as at the screen build
            assert sorted(p.name for p in lab.iterdir()) == sorted(
                list(_MEMLAB_FILES) + ["__init__.py", "gitio.py"],
            )
