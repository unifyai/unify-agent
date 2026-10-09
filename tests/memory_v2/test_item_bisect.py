"""Bisect and the rollback proposal (spec §10.2, §13.4): later tests on each version, confined, bounded, explicit."""

from __future__ import annotations

import pytest

from tests.memory_v2.test_library_helper import _commit
from unify.memory_v2 import item_bisect as ib
from unify.memory_v2.gitio import Repo
from unify.memory_v2.history_probe import ProbeRow

PD, WEEK = "memory.text.dates:parse_date", "memory.text.dates:week"
DATES = "memory/text/dates.py"
TEST = "memory/text/tests/test_dates.py"


def _dates(pd_body: str, week_body: str) -> str:
    return f"def parse_date(s):\n    return {pd_body}\n\n\ndef week(s):\n    return {week_body}\n"


class _Probe:
    """Stands in for history_probe.probe: red on the versions in *red*; records every call."""

    def __init__(self, red: set[str]):
        self.red, self.calls = red, []

    def __call__(self, mem, versions, later, tests, *, work):
        self.calls.append((list(versions), later, list(tests)))
        return [
            ProbeRow(
                label,
                sha,
                t,
                passed=[] if sha in self.red else ["t::x"],
                failed=["t::x"] if sha in self.red else [],
            )
            for label, sha in versions
            for t in tests
        ]


@pytest.fixture
def mem(tmp_path):
    repo = Repo.init_bare(tmp_path / "mem.git")
    shas = [
        _commit(repo, {DATES: _dates("s", "41")}, "add dates"),
        _commit(repo, {DATES: _dates("s", "42")}, "fix week"),
        _commit(
            repo,
            {"memory/text/parse.py": "def tokens(s):\n    return s.split()\n"},
            "add parse",
        ),
        _commit(repo, {DATES: _dates("s.strip()", "42")}, "strip dates"),
    ]
    return repo, shas


def test_versions_follow_the_function_body(mem):
    repo, (c1, c2, c3, c4) = mem
    assert ib.versions_of(repo, PD, repo.head()) == [c1, c4]
    assert ib.versions_of(repo, WEEK, repo.head()) == [c1, c2]


def test_bisect_names_the_version_that_introduced_the_behaviour_and_a_clean_rollback(
    mem,
    tmp_path,
):
    repo, (c1, c2, c3, c4) = mem
    probe = _Probe(red={c4})
    out = ib.bisect_item(
        repo,
        None,
        PD,
        repo.head(),
        [TEST],
        work=tmp_path / "w",
        probe=probe,
        refused=lambda ev, m, i: [],
    )
    assert (out["versions"], out["introduced"], out["changes"], out["red"]) == (
        [c1, c4],
        c4,
        [c4],
        [c4],
    )
    order = repo.log_shas()
    use = {
        c1: {"episodes": 2, "errors": 0, "negative_signals": 0},
        c3: {"episodes": 1, "errors": 0, "negative_signals": 0},
        c4: {"episodes": 2, "errors": 2, "negative_signals": 0},
    }
    assert ib.rollback_target(use, out["versions"], out["introduced"], order) == {
        "version": c1,
        "episodes": 3,
    }
    dirty = {**use, c2: {"episodes": 1, "errors": 1, "negative_signals": 0}}
    assert ib.rollback_target(dirty, out["versions"], out["introduced"], order) is None


def test_bisect_runs_only_through_the_probe_and_marks_the_version_cut(tmp_path):
    repo = Repo.init_bare(tmp_path / "mem.git")
    n = ib.BISECT_MAX_VERSIONS + 5
    shas = [_commit(repo, {DATES: _dates("s", str(k))}, f"week {k}") for k in range(n)]
    probe = _Probe(red=set())
    refused = [("refused:p9", shas[3])]
    out = ib.bisect_item(
        repo,
        None,
        WEEK,
        repo.head(),
        [TEST],
        work=tmp_path / "w",
        probe=probe,
        refused=lambda ev, m, i: refused,
    )
    (call,) = probe.calls
    assert [label for label, _ in call[0]] == [
        f"v{i}" for i in range(ib.BISECT_MAX_VERSIONS)
    ] + ["refused:p9"]
    assert out["versions"] == shas[5:] and out["versions_cut"] == 5
    assert out["introduced"] == shas[5] and out["changes"] == []
    assert out["refused"] == [{"label": "refused:p9", "version": shas[3], "red": False}]


def test_notes_and_untested_functions_are_not_bisected(mem, tmp_path):
    repo, _ = mem
    probe = _Probe(red=set())
    assert ib.bisect_item(
        repo,
        None,
        "notes/text/dates.md",
        repo.head(),
        [TEST],
        work=tmp_path,
        probe=probe,
    )["skipped"]
    assert ib.bisect_item(repo, None, PD, repo.head(), [], work=tmp_path, probe=probe)[
        "skipped"
    ]
    assert probe.calls == []
