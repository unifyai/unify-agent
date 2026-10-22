"""Items of any effect (``UNIFY_MEMORY_IMPURE_ITEMS=on``): labelled with their evidence, never filtered by purity.

A side-effect-only item lands with its stated effects and the replay tests it had; a quality finding the gate would
have refused it for is recorded on its label instead, while safety, paths and syntax still refuse; a note (a
checklist) lands and is served; the writer's brief says what an item may be and states no rule against side effects.
Off, the brief, the manifest and the gate are as built.
"""

from __future__ import annotations

import json
import shutil

import pytest

from tests.memory import library_example_world as w
from tests.memory.library_example_world import library_example_world  # noqa: F401 - the fixture
from unify.memory import item_label, manifest as mf
from unify.memory.episodes import Action, Episode
from unify.memory.gate import Gate, _Run, labelled
from unify.memory.library_checks import LibraryConfig

needs_bwrap = pytest.mark.skipif(shutil.which("bwrap") is None, reason="bubblewrap required")

ITEM = "memory.files.report:write_report"
MODULE = "memory/files/report.py"
TEST = "memory/files/tests/test_report.py"
REPORT_SOURCE = '''"""Reports."""


def write_report(state, env):
    """Write the totals report of the request's rows through the environment's files.

    Args:
        state: the episode's state; its first JSON value holds the rows.
        env: the environment's tools.
    Returns:
        None: the report is what it writes.
    """
    rows = state.data[0]["rows"]
    total = sum(sum(v) for v in rows.values())
    env.files.write("out/report.txt", f"total {total}")
'''
TEST_SOURCE = '''from memory.files.report import write_report


class _Files:
    def __init__(self, calls):
        self.calls = calls

    def write(self, path, text):
        self.calls.append(("files", "write", path, text))


class _Env:
    def __init__(self):
        self.calls = []
        self.files = _Files(self.calls)


class _State:
    data = [{"rows": {"a": [1, 2], "b": [3]}}]


def test_writes_the_report_once():
    env = _Env()
    assert write_report(_State(), env) is None
    assert env.calls == [("files", "write", "out/report.txt", "total 6")]
'''
STATED = {
    "effects": "writes",
    "situation": "writing a total to a report file",
    "inputs": ["state.data[0].rows"],
    "constants": [{"value": "out/report.txt", "source": "e1/action:0"}],
}


def _files(module=REPORT_SOURCE, test=TEST_SOURCE) -> dict[str, str]:
    return {
        "memory/files/__init__.py": '"""Files."""\n',
        "memory/files/README.md": "Writing reports through the environment's files.\n",
        MODULE: module,
        TEST: test,
    }


def _item(**over) -> dict:
    return {"item": ITEM, "kind": "function", "source_episodes": ["e1"], "tests": [TEST], "covers": [], **over}


def _gate(world, labels=True) -> Gate:
    mem, ev, blobs, _ = world
    return Gate(
        mem,
        ev,
        blobs,
        action_lookup=w.lookup,
        library_mode=LibraryConfig(episodes=w.EPISODES.get),
        )


def _merge(world, files, man, labels=True):
    mem = world[0]
    return _gate(world, labels).merge(mem.head(), w.candidate(mem, files), man, "p1", "write", None, "0.01")


def _write_episode() -> Episode:
    return Episode(
        episode_id="e1",
        started_at="2026-10-20T00:00:00Z",
        ended_at="2026-10-20T00:01:00Z",
        build="b",
        model="m",
        effort="low",
        regime="none",
        memory_main="0" * 40,
        worktree_before=None,
        worktree_after=None,
        request=['Report: {"rows": {"a": [1, 2], "b": [3]}}'],
        transcript=[],
        cells=[],
        actions=[Action(0, "files", "write", ["out/report.txt", "total 6"], {}, {"written": 7}, "ok", "write")],
    )


# --- the manifest's label fields ---------------------------------------------------------------------------------------


def test_an_item_states_its_label_fields_only_under_the_switch():
    raw = {"items": [_item(**STATED)], "support": []}
    # off, a label field is an unknown field (the first one the item states)
    with pytest.raises(mf.ManifestError, match="unknown field 'effects'"):
        mf.parse_manifest(raw)
    [it] = mf.parse_manifest(raw, labels=True).items
    assert it.stated == {**STATED, "constants": ["out/report.txt (source: e1/action:0)"]}
    for bad, words in (
        ({"effects": "sometimes"}, "effects is one of none, reads, writes, unknown"),
        ({"constants": [{"value": 3}]}, "a constant is"),
        ({"form": "checklist"}, "form is one of note, checklist, on a note"),
    ):
        with pytest.raises(mf.ManifestError, match=words):
            mf.parse_manifest({"items": [_item(**bad)]}, labels=True)
    note = {"item": "notes/files/checks.md", "kind": "note", "source_episodes": ["e1"], "form": "checklist"}
    [n] = mf.parse_manifest({"items": [note]}, labels=True).items
    assert n.stated == {"form": "checklist"}


# --- the gate's labels -----------------------------------------------------------------------------------------------------


def test_a_quality_finding_labels_its_item_and_a_hard_one_refuses(tmp_path):
    from unify.memory.gate import GateResult
    from unify.memory.manifest import Manifest

    def run():
        r = _Run(GateResult(True, {}), Manifest(), {}, "", "", tmp_path)
        r.labels_on = True
        return r

    r = run()
    r.fail("G2", f"{ITEM}: its recipe memory.files.report:apply does not give the expected output of case c", ITEM)
    r.fail("G2", f"{ITEM} held-out value refused: rows", ITEM)
    r.fail("G3", f"{TEST} is not green on the candidate", [ITEM])
    assert r.res.passed and not r.item_fail and len(r.label_findings[ITEM]) == 3
    assert all(n.startswith("note: label: ") for n in r.notes)
    for check, reason, owner in (
        ("G6", "a key-shaped string", ITEM),
        ("G1", f"undeclared change {MODULE}", None),
        ("G1", f"{ITEM} names no source episode", ITEM),
        ("G3", f"{TEST} does not parse", ITEM),
        ("G3", f"listed test {TEST} is not in the candidate", [ITEM]),
        ("G3", "the parent's tests no longer pass against the candidate's library: [x]", None),
    ):
        h = run()
        h.fail(check, reason, owner)
        assert not h.res.passed, (check, reason)
    # off: every finding refuses, as built
    off = _Run(GateResult(True, {}), Manifest(), {}, "", "", tmp_path)
    off.fail("G2", f"{ITEM} held-out value refused: rows", ITEM)
    assert not off.res.passed and off.item_fail == {ITEM: ["G2"]}
    assert labelled("G2", "x") and not labelled("G5", "x") and not labelled("G2", "ModuleNotFoundError: no x")


# --- 1. a side-effect-only item lands, labelled with its effects and its replay test -----------------------------------------


@needs_bwrap
def test_a_side_effect_only_item_lands_with_effects_writes_on_its_label(
    library_example_world, tmp_path, monkeypatch  # noqa: F811 - the fixture
):
    res = _merge(library_example_world, _files(), w.manifest({}, _item(**STATED)))
    assert res.passed, res.reasons
    assert res.items_merged == [ITEM] and res.merged
    v = res.verification[ITEM]
    assert v["stated"]["effects"] == "writes" and v["stated"]["situation"] == "writing a total to a report file"
    # rev 2 runs no replay test (the writer's tests never replay a recorded call): the label states the effects the
    # writer declared, with none seen in a test
    label = item_label.build(kind="function", record={"status": "on-trial", "provenance": {"episodes": ["e1"]}},
                             verification=v, tests=[])  # fmt: skip
    assert label["effects"]["declares"] == "writes" and label["effects"]["declared"] is True
    # rev 2.1: the episodes are its evidence (the harness's provenance); the writer states the situation it is for
    assert label["covers"] == {"episodes": ["e1"], "stated_by": "harness"}
    assert label["situation"] == "writing a total to a report file"
    assert label["status_basis"] == "no_outcome_signals"


# --- 6. former quality refusals are labels; safety still refuses ---------------------------------------------------------------


@needs_bwrap
def test_a_failing_test_is_a_label_and_a_key_shaped_string_still_refuses(library_example_world):  # noqa: F811
    failing = TEST_SOURCE.replace('"total 6"', '"total 7"')
    res = _merge(library_example_world, _files(test=failing), w.manifest({}, _item(**STATED)))
    assert res.passed, res.reasons
    findings = res.verification[ITEM]["label_findings"]
    # rev 2: the item is a state form (impure), so the gate runs neither its tests nor its example calls; each is a
    # "not run" label with why, never a refusal
    assert any(f.startswith("not run: G3 ") and "(a state form)" in f for f in findings), findings
    assert any(f.startswith("not run: G2 examples: ") and "(a state form)" in f for f in findings), findings
    mem = library_example_world[0]
    keyed = REPORT_SOURCE + '\n_KEY = "' + "sk-" + "a" * 40 + '"\n'
    bad = _gate(library_example_world).check(mem.head(), w.candidate(mem, _files(module=keyed)),
                                             w.manifest({}, _item(**STATED)))  # fmt: skip
    assert not bad.passed and "G6" in bad.refused


# --- 7. a note lands and is served ------------------------------------------------------------------------------------------


@needs_bwrap
def test_a_checklist_note_lands_and_is_served(library_example_world, tmp_path):  # noqa: F811
    from unify.memory.integration.checkout import export_actor_library_mode

    note = "notes/files/checks.md"
    files = {
        note: "---\ntitle: Before writing a report\ndescription: what to check first\nuses: []\n---\n"
        "- the rows are read from the latest message\n- the report path is given, never assumed\n",
    }
    res = _merge(
        library_example_world,
        files,
        w.manifest({}, {"item": note, "kind": "note", "source_episodes": ["e1"], "form": "checklist"}),
    )
    assert res.passed, res.reasons
    assert res.verification[note]["stated"] == {"form": "checklist"}
    mem = library_example_world[0]
    dest = tmp_path / "copy"
    export_actor_library_mode(mem.git_dir, res.merged, dest)
    items = json.loads((dest / ".memory" / "items.json").read_text())["items"]
    assert note in items
    label = item_label.build(kind="note", record={"status": "on-trial"}, verification=res.verification[note])
    assert label["form"] == "checklist" and label["effects"]["declares"] == "none"


# --- the label -------------------------------------------------------------------------------------------------------------


# --- the brief -------------------------------------------------------------------------------------------------------------


