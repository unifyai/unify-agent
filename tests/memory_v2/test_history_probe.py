import json
import shutil

import pytest

from unify.memory_v2 import history_probe as hp
from unify.memory_v2.evidence import EvidenceStore
from unify.memory_v2.gitio import Repo
from unify.memory_v2.sandbox_run import PytestOutcome
from tests.memory_v2.test_gate import _candidate, _merged

OPS = "memory/calc/ops.py"
TEST = "memory/calc/tests/test_ops.py"
FIXED = "def double(x):\n    return x * 2\n"
BUGGY = "def double(x):\n    return x * 3\n"
LATER_TEST = "from memory.calc.ops import double\n\n\ndef test_double():\n    assert double(4) == 8\n"
OLD_TEST = "from memory.calc.ops import double\n\n\ndef test_runs():\n    assert double(0) == 0\n"


@pytest.fixture
def repo(tmp_path):
    mem = Repo.init_bare(tmp_path / "mem.git")
    _merged(
        mem,
        {OPS: FIXED, TEST: LATER_TEST, "memory/calc/tests/data/x.json": "[1]\n"},
    )
    refused = _candidate(
        mem,
        {OPS: BUGGY, TEST: OLD_TEST, "memory/calc/tests/data/x.json": "[0]\n"},
    )
    return mem, refused


def test_overlay_lays_later_tests_and_data_over_the_version(repo, tmp_path):
    mem, refused = repo
    tree = hp.overlay(mem, refused, mem.head(), [TEST], tmp_path / "o")
    assert (tree / OPS).read_text() == BUGGY
    assert (tree / TEST).read_text() == LATER_TEST
    assert (tree / "memory/calc/tests/data/x.json").read_text() == "[1]\n"
    with pytest.raises(ValueError, match="holds no"):
        hp.overlay(
            mem,
            refused,
            mem.head(),
            ["memory/calc/tests/test_gone.py"],
            tmp_path / "o2",
        )


def test_probe_runs_each_test_file_confined(repo, tmp_path):
    mem, refused = repo
    seen = []

    def fake(target, *, python, ro, rw, cwd, timeout_s, env):
        seen.append((target, sorted(ro.values()), rw, cwd, env))
        return PytestOutcome(failed={"test_ops.py::test_double"}, returncode=1)

    rows = hp.probe(
        mem,
        [("refused:p9", refused), ("gone", "f" * 40)],
        mem.head(),
        [TEST],
        work=tmp_path / "w",
        pytest_runner=fake,
    )
    assert [(r.label, r.red) for r in rows] == [("refused:p9", True), ("gone", False)]
    assert rows[1].valid is False
    assert seen == [
        (
            TEST,
            ["/memory"],
            {},
            "/memory",
            {
                "PYTHONPATH": "/memory",
                "PYTEST_ADDOPTS": "-c /dev/null --import-mode=importlib",
            },
        ),
    ]


@pytest.mark.skipif(shutil.which("bwrap") is None, reason="bubblewrap required")
def test_later_tests_are_red_on_a_refused_version_and_green_on_main(repo, tmp_path):
    mem, refused = repo
    rows = hp.probe(
        mem,
        [("refused:p9", refused), ("main", mem.head())],
        mem.head(),
        [TEST],
        work=tmp_path / "w",
    )
    assert [(r.label, r.red, r.failed, r.passed) for r in rows] == [
        ("refused:p9", True, ["test_ops.py::test_double"], []),
        ("main", False, [], ["test_ops.py::test_double"]),
    ]


def test_refused_versions_reads_pass_rows(repo, tmp_path):
    mem, refused = repo
    ev = EvidenceStore(tmp_path / "e.sqlite")
    ev.record_pass(
        {
            "pass_id": "p9",
            "candidate": refused,
            "passed": 0,
            "reasons": "[]",
            "items_refused": json.dumps({"memory.calc.ops:double": ["G3"]}),
        },
    )
    ev.record_pass(
        {
            "pass_id": "p10",
            "candidate": "e" * 40,
            "passed": 0,
            "reasons": "[]",
            "items_refused": json.dumps({"memory.calc.ops:double": ["G3"]}),
        },
    )
    assert hp.refused_versions(ev, mem, "memory.calc.ops:double") == [
        ("refused:p9", refused),
    ]
    assert hp.refused_versions(ev, mem, "memory.calc.ops:other") == []
