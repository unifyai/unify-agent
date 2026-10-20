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
from unify.memory import writer_tests as wt
from unify.memory.episodes import Action, Episode
from unify.memory.gate import Gate, _Run, labelled
from unify.memory.library_checks import LibraryConfig
from unify.memory.library_prompts import WriterSwitches, write_brief_now

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
    "covers_episodes": ["e1"],
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
        library_mode=LibraryConfig(checks=True, episodes=w.EPISODES.get),
        labels=labels,
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
        mf.parse_manifest(raw, library_mode=True)
    [it] = mf.parse_manifest(raw, library_mode=True, labels=True).items
    assert it.stated == {**STATED, "constants": ["out/report.txt (source: e1/action:0)"]}
    for bad, words in (
        ({"effects": "sometimes"}, "effects is one of none, reads, writes, unknown"),
        ({"constants": [{"value": 3}]}, "a constant is"),
        ({"form": "checklist"}, "form is one of note, checklist, on a note"),
    ):
        with pytest.raises(mf.ManifestError, match=words):
            mf.parse_manifest({"items": [_item(**bad)]}, library_mode=True, labels=True)
    note = {"item": "notes/files/checks.md", "kind": "note", "source_episodes": ["e1"], "form": "checklist"}
    [n] = mf.parse_manifest({"items": [note]}, library_mode=True, labels=True).items
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
def test_a_side_effect_only_item_lands_with_effects_writes_and_a_replay_label(
    library_example_world, tmp_path, monkeypatch  # noqa: F811 - the fixture
):
    res = _merge(library_example_world, _files(), w.manifest({}, _item(**STATED)))
    assert res.passed, res.reasons
    assert res.items_merged == [ITEM] and res.merged
    v = res.verification[ITEM]
    assert v["stated"]["effects"] == "writes" and v["stated"]["covers_episodes"] == ["e1"]
    # the writer's replay test of the landed item, on the episode's own recorded write
    monkeypatch.setattr(wt, "REPLAY_REVIEWED", True)
    copy = tmp_path / "copy"
    copy.mkdir()
    mem = library_example_world[0]
    with mem.temp_checkout(mem.head()) as tree:
        shutil.copytree(tree / "memory", copy / "memory")
    state = tmp_path / "state"
    ctx = wt.TestContext("p1", state / "passes" / "p1" / wt.TESTS_FILE, copy, lambda e: _write_episode(),
                         frozenset({"value", "replay"}), work=tmp_path / "work")  # fmt: skip
    expect = {"kind": "issues_calls", "calls": [{"channel": "files", "method": "write",
                                                  "args": ["out/report.txt", "total 6"], "effect": "write"}]}  # fmt: skip
    got = wt.run_test(ctx, ITEM, "e1", expect, "replay")
    assert got.outcome == "as_stated", got.detail
    tests = item_label.tests_by_item(state)[ITEM]
    label = item_label.build(kind="function", record={"status": "on-trial", "provenance": {"episodes": ["e1"]}},
                             verification=v, tests=tests)  # fmt: skip
    assert label["effects"] == {"declares": "writes", "declared": True, "seen_in_tests": {"read": 0, "write": 1},
                                "contradictions": []}  # fmt: skip
    assert label["tests"]["as_stated"] == 1 and label["tests"]["by_mode"] == {"value": 0, "replay": 1}
    assert label["covers"] == {"episodes": ["e1"], "stated_by": "writer"}
    assert label["inputs_vs_constants"]["constants"] == ["out/report.txt (source: e1/action:0)"]
    assert label["status_basis"] == "no_outcome_signals"


# --- 6. former quality refusals are labels; safety still refuses ---------------------------------------------------------------


@needs_bwrap
def test_a_failing_test_is_a_label_and_a_key_shaped_string_still_refuses(library_example_world):  # noqa: F811
    failing = TEST_SOURCE.replace('"total 6"', '"total 7"')
    res = _merge(library_example_world, _files(test=failing), w.manifest({}, _item(**STATED)))
    assert res.passed, res.reasons
    findings = res.verification[ITEM]["label_findings"]
    # the failing test and the example check (no call on recorded fixture values) are both labels
    assert any("is not green on the candidate" in f for f in findings), findings
    assert any("[examples:grounded]" in f for f in findings), findings
    # the same candidate without the switch is refused, as built (at its first finding, the example check)
    mem = library_example_world[0]
    off = _gate(library_example_world, labels=False).check(
        mem.head(), w.candidate(mem, _files(test=failing)), w.manifest({}, _item())
    )
    assert not off.passed and any("[examples:grounded]" in r for r in off.reasons), off.reasons
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


def test_the_label_names_contradictions_tests_uses_and_failures():
    from unify.memory import trust

    seen = {"read": 2, "write": 1}
    assert item_label.contradictions("none", seen) == [
        "declared none, but a replay test saw 1 write",
        "declared none, but a replay test saw 2 reads",
    ]
    assert item_label.contradictions("reads", seen) == ["declared reads, but a replay test saw 1 write"]
    assert item_label.contradictions("writes", seen) == [] and item_label.contradictions(None, seen) == []
    uses = [trust.Use("e1", "sparse", "error"), trust.Use("e2", "none", "returned", answered=None)]
    label = item_label.build(
        kind="function",
        record={"status": "on-trial", "track": trust.track(uses)},
        verification={"stated": {"effects": "none"}, "label_findings": ["G2: x"]},
        tests=[{"mode": "replay", "outcome": "not_as_stated", "effects": {"write": 1}},
               {"mode": "value", "outcome": "as_stated", "effects": {}}],  # fmt: skip
        uses=uses,
        failures=[{"episode": "e1", "pin": "a" * 40, "why": "error", "ref": "episode:e1"}],
    )
    assert label["effects"]["contradictions"] == ["declared none, but a replay test saw 1 write"]
    assert label["tests"]["not_as_stated"] == 1 and label["tests"]["by_mode"] == {"value": 1, "replay": 1}
    assert label["uses"]["failed"] == 1 and label["uses"]["clean"] == 1
    assert label["uses"]["by_regime"] == {"sparse": 1, "none": 1}
    assert label["failure_record"][0]["why"] == "error" and label["findings"] == ["G2: x"]
    assert label["status_basis"] == "implicit"
    assert item_label.status_basis({"verdicts": {"accepted": 1, "rejected": 0}}, {}) == "verdicts"
    assert item_label.status_basis({"end_outcomes": {"positive": 0, "contrary": 1}}, {}) == "end_outcomes"


# --- the brief -------------------------------------------------------------------------------------------------------------


def test_the_brief_says_what_an_item_may_be_and_is_as_built_with_the_switches_off(monkeypatch):
    from unify.common import in_code_switch

    monkeypatch.setattr(in_code_switch, "tools_in_code", lambda: True)
    built = write_brief_now(staging=True, pool=True)
    assert write_brief_now(staging=True, pool=True, switches=WriterSwitches()) == built
    assert "the results of calls with side effects" in built and "every expected output" in built
    on = write_brief_now(staging=True, pool=True, switches=WriterSwitches(attention=True, impure=True, tests=True))
    for line in (
        "You are given entries that need your attention; decide each with `tools.decide`.",
        "Open entries listed beside them may share a rule with yours; look with `tools.open_entries` and "
        "`tools.find`, and `tools.take` those that do.",
        "An item is written for a kind of situation, with what varies as its inputs. It may take the current state "
        "and act through the environment's tools. When you land or change one, say which episodes it is meant to "
        "cover and which of its values are inputs and which are constants, with the source of each constant.",
        "Notes and checklists are items too.",
        "Running code is a test you state first: say what you expect, then run `tools.test`.",
        "your own code holds no key, opens no network and writes nothing outside /memory",
        '"effects": "<none|reads|writes|unknown>"',
    ):
        assert line in on, line
    for gone in (
        "the results of calls with side effects",
        "lands the procedure only if it gives every expected output",
        "literal equal to a value that varies across its recorded inputs",
        "cannot be verified, so it stays a draft",
    ):
        assert gone not in on, gone
    # each switch alone changes only its own part
    tests_only = write_brief_now(staging=True, pool=True, switches=WriterSwitches(tests=True))
    assert "tools.test" in tests_only and "tools.decide" not in tests_only
    assert "the results of calls with side effects" in tests_only
