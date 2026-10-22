"""The gate's examples checks on recorded calls, with the evidence, perturbation, view and shape parts replaced by
fakes: the reasons' exact texts, what reaches the verification record, the behaviour comparison on the parent's
calls, and the retention of a deleted function's fixtures.
"""

from types import SimpleNamespace

from unify.memory.example_calls import ArgRecord, CallKey, CallRecord, ExampleCall
from unify.memory.example_checks import ExampleChecks
from unify.memory.library_prompts import GROUNDED_FIX
from unify.memory.gate import CHECKS, GateResult, _Run

ITEM = "memory.grid.fill:flood_fill"
TEST = "memory/grid/tests/test_fill.py"
GRID = "memory/grid/tests/data/grid1.json"


def _arg(name, json, sha="s", path=None, cut=False):
    return ArgRecord(
        name=name,
        sha=sha,
        type=type(json).__name__,
        json=json,
        cut=cut,
        path=path,
    )


def _call(
    *args,
    item=ITEM,
    ordinal=0,
    outcome="returned",
    raised=None,
    result_sha="r",
    line=14,
    test="test_fills",
    indices=(),
    opened=(GRID,),
):
    return ExampleCall(
        item=item,
        key=CallKey(f"{TEST}::{test}", tuple(indices), ordinal),
        node=f"{TEST}::{test}",
        file=TEST,
        line=line,
        args=tuple(args),
        outcome=outcome,
        raised=raised,
        result_sha=result_sha if outcome == "returned" else None,
        result_type="list" if outcome == "returned" else None,
        result_json=None,
        opened=tuple(opened),
    )


def _record(calls, **over):
    rec = CallRecord(loaded=True, calls=list(calls), ended=True)
    for k, v in over.items():
        setattr(rec, k, v)
    return rec


class _Fakes:
    """The four parts and the fixture ledger, answering from what a test sets."""

    PERTURB_BUDGET_S = 10.0

    def __init__(self):
        self.kinds: dict[str, str] = {}  # param -> grounding kind
        self.verdict = SimpleNamespace(
            refused=[],
            special=[],
            derived={},
            notes=[],
            unjudged=[],
            runs=2,
            fields=3,
            judged=4,
            unstable=0,
            type_changes=[],
        )
        self.episodes = ["e1", "e2"]
        self.problems: list[str] = []
        self.descriptors: list[dict] = []
        self.inputs: list = []

    # fixtures
    def ledger_for(self, files, in_pass, stored, load, blob, episodes):
        return {p: {"kind": "copy"} for p in files}, []

    def value_views(self, path, data, entry, notes=None):
        return {}

    def source_action(self, entry, load):
        return None

    # perturbation
    def input_fixtures(
        self,
        item,
        calls,
        views,
        ledger,
        tree,
        *,
        load=None,
        notes=None,
    ):
        return list(self.inputs)

    def check(self, item, **kw):
        self.check_kw = kw
        return self.verdict

    # evidence
    def grounding(self, call, views, derived):
        return {
            a.name: SimpleNamespace(kind=self.kinds.get(a.name, "literal"))
            for a in call.args
        }

    def evidence_check(self, item, calls, ground, typed):
        return [], {"episodes": list(self.episodes), "fixtures": [GRID]}

    def groundings(self, calls, views, derived):
        return {c.key: self.grounding(c, views, derived.get(c.key, {})) for c in calls}

    def lint(self, item, source, calls, ground, *, extra=(), lint_min=3):
        return [], [
            f"G2 literal lint: {item} has fewer than 2 recorded inputs, so nothing varies yet",
        ]

    def rule_episodes(self, ground, calls=None):
        return set(self.episodes)

    def rule_problems(self, item, *, path, rule_line, unchecked, episodes):
        return [], []

    # view
    def pick_example(self, item, calls, grounded, named, tree):
        return (
            {
                "test": f"{TEST}::test_fills",
                "line": 14,
                "call": "flood_fill(grid, (0, 0), 3)",
            }
            if grounded
            else None
        )

    def docstring_example(self, doc):
        return None

    def docstring_problems(self, item, doc, tests, calls, grounded):
        return list(self.problems)

    # shapes
    def arg_descriptors(self, calls, tree, *, note=None):
        self.shape_note = note
        return list(self.descriptors)


class _Gate:
    def __init__(self, records):
        self.records = records  # test file -> CallRecord
        self.runs: list[tuple[str, list[str] | None]] = []
        self.library_mode = None
        self.ev = SimpleNamespace()
        self.lookup = lambda eid, idx: None

    def _pytest(self, tree, target, extra_env=None, qa_env=None, calls=None):
        self.runs.append((target, calls))
        return SimpleNamespace(
            calls=self.records.get(target),
            valid=True,
            passed=set(),
            failed=set(),
            skipped=set(),
            timed_out=False,
            returncode=0,
        )

    def _blob(self, sha):
        raise KeyError(sha)

    def _pool(self, run, covers):
        return [], False

    def _named_episodes(self, run):
        return []


def _run(tmp_path, items, *, changed=(), deleted=()):
    run = _Run(
        GateResult(True, {c: True for c in CHECKS}),
        SimpleNamespace(),
        {},
        "p",
        "c",
        tmp_path,
    )
    run.man = SimpleNamespace(items=items, deleted=list(deleted))
    run.c_tree = tmp_path / "c"
    (run.c_tree / "memory/grid").mkdir(parents=True, exist_ok=True)
    (run.c_tree / "memory/grid/fill.py").write_text(
        'def flood_fill(grid, start, colour):\n    """Fill."""\n',
    )
    run.c_files = {TEST: ("100644", "x"), GRID: ("100644", "y")}
    run.c_bodies = {it.item: ("def", "body", True) for it in items}
    run.p_bodies = {} if changed else dict(run.c_bodies)
    return run


def _item(item=ITEM, tests=(TEST,)):
    return SimpleNamespace(
        item=item,
        kind="function",
        tests=list(tests),
        source_episodes=["e1"],
        typed_covers=[],
        field_types={},
    )


def _checks(tmp_path, calls, *, changed=True, fakes=None, record=None, items=None):
    fakes = fakes or _Fakes()
    items = items or [_item()]
    run = _run(tmp_path, items, changed=changed)
    (run.c_tree / GRID).parent.mkdir(parents=True, exist_ok=True)
    (run.c_tree / GRID).write_text("[[0, 1]]")
    gate = _Gate({TEST: record if record is not None else _record(calls)})
    ex = ExampleChecks(
        gate,
        run,
        evidence=fakes,
        perturb=fakes,
        view=fakes,
        shapes=fakes,
        ledger=fakes,
    )
    return ex, run, gate, fakes


def _reasons(run, check="G2"):
    return [r[len(check) + 2 :] for r in run.res.reasons if r.startswith(f"{check}: ")]


def test_a_function_no_listed_test_calls_is_refused(tmp_path):
    ex, run, gate, _ = _checks(tmp_path, [])
    ex.judge()
    assert _reasons(run) == [
        f"[examples:none] {ITEM}: none of its listed tests calls it; its examples are the calls its tests make on "
        "fixtures made with fixture",
    ]
    assert gate.runs == [
        (TEST, [ITEM]),
    ]  # an unchanged listed test runs once more, recording the targets


def test_calls_on_literal_values_only_are_refused_with_their_places(tmp_path):
    calls = [
        _call(_arg("grid", [[0, 1]]), _arg("colour", 3), line=14),
        _call(_arg("grid", [[1]]), _arg("colour", 4), ordinal=1, line=15),
    ]
    ex, run, _, _ = _checks(tmp_path, calls)
    ex.judge()
    assert _reasons(run) == [
        f"[examples:grounded] {ITEM}: no call of it in its tests returns on fixture values (2 calls; none that "
        f"returns takes its structured arguments from fixtures: {TEST}:14, {TEST}:15); "
        + GROUNDED_FIX,
    ]


def test_scalar_literals_beside_a_grounded_argument_are_allowed_and_recorded(tmp_path):
    calls = [_call(_arg("grid", [[0, 1]]), _arg("start", [0, 0]), _arg("colour", 3))]
    fakes = _Fakes()
    fakes.kinds = {"grid": "copied"}
    ex, run, _, _ = _checks(tmp_path, calls, fakes=fakes)
    ex.judge()
    # start is a structured literal: the call is not grounded
    assert any(r.startswith("[examples:grounded]") for r in _reasons(run))
    fakes.kinds = {"grid": "copied", "start": "derived"}
    ex, run, _, _ = _checks(tmp_path / "2", calls, fakes=fakes)
    ex.judge()
    assert not _reasons(run) and run.res.passed
    examples = run.verification[ITEM]["examples"]
    assert examples["literal_args"] == ["colour"] and examples["grounded_calls"] == 1
    assert (
        examples["calls"] == 1
        and examples["accepted"] == 1
        and examples["episodes"] == ["e1", "e2"]
    )
    assert examples["fixtures"] == [GRID]
    assert examples["example"] == {
        "test": f"{TEST}::test_fills",
        "line": 14,
        "call": "flood_fill(grid, (0, 0), 3)",
    }
    assert examples["perturbation"] == {
        "runs": 2,
        "fields": 3,
        "judged": 4,
        "unstable": 0,
        "type_changes": [],
    }
    assert fakes.shape_note == run.note


def test_an_example_call_that_raises_is_refused(tmp_path):
    fakes = _Fakes()
    fakes.kinds = {"grid": "copied"}
    calls = [
        _call(_arg("grid", [[0, 1]])),
        _call(
            _arg("grid", [[0, 1]]),
            ordinal=1,
            outcome="raised",
            raised="KeyError",
            line=20,
        ),
        _call(_arg("grid", [[0, 1]]), ordinal=2, outcome="refused", line=21),
    ]
    ex, run, _, _ = _checks(tmp_path, calls, fakes=fakes)
    ex.judge()
    assert _reasons(run) == [
        f"[examples:raised] {ITEM} raises KeyError on its example call at {TEST}:20; a function returns or raises "
        "MemoryInputError",
    ]
    assert run.verification[ITEM]["examples"]["refused"] == 1


def test_a_function_the_plugin_cannot_observe_is_refused(tmp_path):
    rec = _record([], unresolved=[f"{ITEM} (partial)"])
    ex, run, _, _ = _checks(tmp_path, [], record=rec)
    ex.judge()
    assert _reasons(run) == [
        f"[examples:unobserved] {ITEM} cannot be observed when called (it is not a Python function: partial); "
        "define it with def",
    ]


def test_an_incomplete_record_judges_nothing_and_refuses_its_owners(tmp_path):
    rec = _record([_call(_arg("grid", [[0]]))], dropped=2)
    ex, run, _, _ = _checks(tmp_path, [], record=rec)
    ex.judge()
    assert _reasons(run) == []
    assert _reasons(run, "G3") == [
        f"[examples:record] the call record of {TEST} is incomplete (0 unreadable lines, 2 calls not recorded); its "
        "examples are not judged",
    ]
    assert run.item_fail == {ITEM: ["G3"]}
    rec = _record([], unobservable="no monitoring tool id")
    ex, run, _, _ = _checks(tmp_path / "2", [], record=rec)
    ex.judge()
    assert _reasons(run, "G3") == [
        f"[examples:record] the call record of {TEST} is incomplete (0 unreadable lines, 0 calls not recorded, "
        "unobservable: no monitoring tool id); its examples are not judged",
    ]


def test_new_or_changed_tests_use_g3s_candidate_run(tmp_path):
    calls = [_call(_arg("grid", [[0]]))]
    ex, run, gate, _ = _checks(tmp_path, calls)
    run.qa_first[TEST] = SimpleNamespace(calls=_record(calls))
    assert ex.record() == {ITEM: calls}
    assert gate.runs == []


def test_perturbation_verdicts_become_reasons_and_type_changes_reach_the_record(
    tmp_path,
):
    fakes = _Fakes()
    fakes.kinds = {"grid": "copied"}
    calls = [_call(_arg("grid", [[0, 1]]), line=14)]
    key = calls[0].key
    change = {
        "function": ITEM,
        "call": "k",
        "where": f"{TEST}:14",
        "fixture": GRID,
        "field": "[0][0]",
        "before": "list",
        "after": "NoneType",
    }
    fakes.verdict.refused = [(key, GRID, "[0][1]")]
    fakes.verdict.special = [(key, GRID, "[1][0]", "IndexError")]
    fakes.verdict.unjudged = ["budget"]
    fakes.verdict.type_changes = [change]
    ex, run, _, _ = _checks(tmp_path, calls, fakes=fakes)
    ex.judge()
    assert _reasons(run) == [
        f"[examples:perturbed-refused] {ITEM} refuses fixture values it accepted unchanged: with [0][1] of {GRID} "
        f"replaced by an unseen value of its type, its call at {TEST}:14 raises MemoryInputError; refuse inputs by "
        "their shape, not by values the recording happened to hold",
        f"[examples:special-cased] {ITEM} handles {GRID} only for its recorded values: with [1][0] replaced by an "
        f"unseen value of its type, its call at {TEST}:14 raises IndexError where it returned before",
        f"[examples:unjudged] {ITEM}: the check with unseen fixture values did not complete (budget); a function is "
        "admitted only once it is judged",
    ]
    examples = run.verification[ITEM]["examples"]
    assert (
        examples["perturbation"]["type_changes"] == [change]
        and examples["type_changes"] == 1
    )
    assert any(GRID in n and "recorded for review" in n for n in run.notes)
    assert not any("NoneType" in r for r in run.res.reasons + run.notes)
    assert fakes.check_kw["tests"] == [TEST] and fakes.check_kw["field_types"] == {}


def test_docstring_problems_of_a_changed_function_are_g1_reasons(tmp_path):
    fakes = _Fakes()
    fakes.kinds = {"grid": "copied"}
    fakes.problems = [f"{ITEM} docstring has no Example: line naming one of its tests"]
    ex, run, _, _ = _checks(tmp_path, [_call(_arg("grid", [[0]]))], fakes=fakes)
    ex.judge()
    assert _reasons(run, "G1") == fakes.problems
    ex, run, _, _ = _checks(
        tmp_path / "2",
        [_call(_arg("grid", [[0]]))],
        fakes=fakes,
        changed=False,
    )
    ex.judge()
    assert (
        _reasons(run, "G1") == []
    )  # an unchanged function's docstring is not judged again


def _behaviour(tmp_path, base_calls, reg_calls, scope=frozenset({ITEM})):
    ex, run, _, _ = _checks(tmp_path, [])
    ex.behaviour(
        {"memory/grid/tests": _record(base_calls)},
        {"memory/grid/tests": _record(reg_calls)},
        set(scope),
    )
    return ex, run


def test_a_changed_result_on_a_parent_tests_call_is_a_change_of_behaviour(tmp_path):
    secret = [[9, 9, 9]]
    base = [
        _call(_arg("grid", secret), result_sha="a"),
        _call(_arg("grid", secret), ordinal=1, result_sha="b"),
    ]
    same = [
        _call(_arg("grid", secret), result_sha="a"),
        _call(_arg("grid", secret), ordinal=1, result_sha="b"),
    ]
    ex, run = _behaviour(tmp_path, base, same)
    assert run.res.passed and run.same == {ITEM}
    differ = [
        _call(_arg("grid", secret), result_sha="a"),
        _call(_arg("grid", secret), ordinal=1, result_sha="c"),
    ]
    ex, run = _behaviour(tmp_path / "2", base, differ)
    assert _reasons(run, "G3") == [
        f"{ITEM} changes behaviour without a test: 1 of 2 calls of it in the parent's tests return, refuse or raise "
        f"differently on the candidate's library ({TEST}::test_fills#1); any change in what a stored function does "
        "needs a test, listed under it, that fails on the parent's library and passes on the candidate's",
    ]
    assert "9" not in run.res.reasons[0].split("#")[0]
    assert not run.same


def test_an_uncomparable_result_refuses_and_an_uncalled_function_is_named(tmp_path):
    base = [_call(_arg("grid", [[0]]), result_sha=None)]
    ex, run = _behaviour(tmp_path, base, base)
    assert len(_reasons(run, "G3")) == 1
    other = "memory.grid.fill:other"
    ex, run = _behaviour(tmp_path / "2", [], [], scope={other})
    assert ex.uncalled == {other}
    ex.no_comparison(other)
    assert _reasons(run, "G3") == [
        f"{other} is in a changed module but no test of the parent calls it and it has no stored cover; its "
        "behaviour cannot be compared, so change it only with a test listed under it that fails on the parent's "
        "library",
    ]


def test_a_deleted_functions_fixtures_must_stay_used(tmp_path):
    gone = "memory.grid.fill:old"
    parent = {
        gone: [
            _call(
                _arg("grid", [[0]]),
                item=gone,
                opened=(GRID, "memory/grid/tests/data/b.json"),
            ),
        ],
    }
    ex, run, _, _ = _checks(tmp_path, [])
    run.man.deleted = [gone]
    run.calls_cand = {
        "memory/grid/tests": _record([_call(_arg("grid", [[0]]), opened=(GRID,))]),
    }
    ex.retained(parent)
    assert _reasons(run, "G5") == [
        f"{gone} is deleted, but the fixtures its examples ran on (memory/grid/tests/data/b.json) are used by no "
        "remaining function's examples; a kept function's tests must take each over",
    ]
    run.calls_cand["memory/grid/tests"].calls.append(
        _call(
            _arg(
                "p",
                "memory/grid/tests/data/b.json",
                path="memory/grid/tests/data/b.json",
            ),
            opened=(),
        ),
    )
    run.res.reasons.clear()
    ex.retained(parent)
    assert not _reasons(run, "G5")


# --- the gate's library-mode flow ------------------------------------------------------------------------------

OTHER = "memory.grid.fill:other"


class _LibraryGate:
    """A gate with only what the library-mode flow reads: checks on, no stored covers."""

    def __new__(cls, covers=()):
        from unify.memory.gate import Gate

        g = Gate.__new__(Gate)
        g.library_mode = SimpleNamespace(
            checks=True,
            layout=True,
            role="write",
            episodes=lambda eid: None,
        )
        g.ev = SimpleNamespace(covers=lambda: set(covers))
        g.lookup = lambda eid, idx: None
        return g


def _flow_run(tmp_path, *, deleted=()):
    run = _run(tmp_path, [_item()], changed=True, deleted=deleted)
    run.p_tree = tmp_path / "p"
    (run.p_tree / "memory/grid").mkdir(parents=True, exist_ok=True)
    src = "def flood_fill(grid, start, colour):\n    return grid\n\n\ndef other(x):\n    return x\n"
    (run.p_tree / "memory/grid/fill.py").write_text(src)
    (run.c_tree / "memory/grid/fill.py").write_text(
        src.replace("return x", "return [x]"),
    )
    run.p_files = {"memory/grid/fill.py": ("100644", "a")}
    run.c_files["memory/grid/fill.py"] = ("100644", "b")
    run.changed = ["memory/grid/fill.py"]
    run.p_bodies = {ITEM: ("function", "b1", True), OTHER: ("function", "o1", True)}
    run.c_bodies = {ITEM: ("function", "b1", True), OTHER: ("function", "o2", True)}
    run.man.skeleton, run.man.unlisted = [], []
    return run


def _rev2(run):
    """*run* under rev 2's gate policy (:mod:`unify.memory.gate_policy`), as ``Gate.check`` sets it: both functions are
    pure, so the behaviour check runs them, and its findings are labels."""
    from unify.memory.gate_policy import Policy

    run.policy = Policy(
        replays=False,
        bodies={
            ITEM: "def flood_fill(grid, start, colour):\n    return grid\n",
            OTHER: "def other(x):\n    return [x]\n",
        },
    )
    return run


def _g3_labels(run):
    """The G3 findings recorded as labels (rev 2: none refuses): every item's, then the pass's (a finding no manifest
    item owns is a label on the pass, in the gate's notes)."""
    items = [f[len("G3: "):] for found in run.label_findings.values() for f in found if f.startswith("G3: ")]
    head = "note: label: the pass: G3: "
    return items + [n[len(head):] for n in run.notes if n.startswith(head)]


def test_library_functions_need_no_action_cover_in_g2(tmp_path):
    from unify.memory.gate import Gate

    run = _flow_run(tmp_path)
    Gate._g2(_LibraryGate(), run)
    assert run.res.passed and not run.res.reasons


def test_the_behaviour_check_compares_parent_calls_and_labels_an_uncalled_function(
    tmp_path,
):
    from unify.memory.gate import Gate

    run = _rev2(_flow_run(tmp_path))
    calls = [_call(_arg("grid", [[0]]), result_sha="a")]
    run.calls_base = {"memory/grid/tests": _record(calls)}
    run.calls_reg = {"memory/grid/tests": _record(calls)}
    Gate._behaviour(_LibraryGate(), run, set(), set(), set(), set())
    assert run.same == {ITEM}
    assert _reasons(run, "G3") == [] and run.res.passed
    assert _g3_labels(run) == [
        f"{OTHER} is in a changed module but no test of the parent calls it and it has no stored cover; its "
        "behaviour cannot be compared, so change it only with a test listed under it that fails on the parent's "
        "library",
    ]
    run = _rev2(_flow_run(tmp_path / "2"))
    run.calls_base = {"memory/grid/tests": _record(calls)}
    run.calls_reg = {
        "memory/grid/tests": _record([_call(_arg("grid", [[0]]), result_sha="b")]),
    }
    Gate._behaviour(_LibraryGate(), run, set(), {OTHER}, set(), set())
    assert _reasons(run, "G3") == [] and run.res.passed
    assert len(_g3_labels(run)) == 1 and "changes behaviour without a test" in _g3_labels(run)[0]


def test_a_stored_function_with_covers_and_an_input_line_takes_the_transition_path(
    tmp_path,
):
    from unify.memory.gate import Gate

    run = _rev2(_flow_run(tmp_path))
    (run.p_tree / "memory/grid/fill.py").write_text(
        'def flood_fill(grid, start, colour):\n    return grid\n\n\ndef other(x):\n    """Other.\n\n    Input: text\n'
        '    """\n    return x\n',
    )
    run.calls_base = {
        "memory/grid/tests": _record([_call(_arg("grid", [[0]]), result_sha="a")]),
    }
    run.calls_reg = dict(run.calls_base)
    g = _LibraryGate(covers={(OTHER, "e1", 0)})
    probed = []
    g._probe = lambda run, item, recorded, strict: probed.append(
        item,
    ) or SimpleNamespace(
        why=None,
        verdict=lambda: None,
        item=item,
    )
    g._run_probe = lambda run, pr, work, budget: True
    Gate._behaviour(g, run, set(), set(), set(), set())
    assert probed == [OTHER] and run.same == {ITEM, OTHER} and run.res.passed
    assert any(
        "has no example call in the parent's tests; compared on its stored covers" in n
        for n in run.notes
    )


def test_g5_in_library_mode_keeps_a_deleted_functions_fixtures_used(tmp_path):
    from unify.memory.gate import Gate

    run = _flow_run(tmp_path, deleted=[OTHER])
    run.calls_base = {
        "memory/grid/tests": _record([_call(_arg("x", 1), item=OTHER, opened=(GRID,))]),
    }
    run.calls_cand = {"memory/grid/tests": _record([])}
    Gate._g5(_LibraryGate(), run)
    assert _reasons(run, "G5") == [
        f"{OTHER} is deleted, but the fixtures its examples ran on ({GRID}) are used by no remaining function's "
        "examples; a kept function's tests must take each over",
    ]


def test_an_effect_comes_from_episode_covers_only():
    from unify.memory.library_checks import derive_effect
    from unify.memory.procedures import Cover

    episodes = {
        "w": SimpleNamespace(actions=[SimpleNamespace(kind="tool", effect="write")]),
    }
    assert (
        derive_effect([Cover("w", "episode", runner="tool")], episodes.get) == "write"
    )
    assert (
        derive_effect([Cover("w", "episode", runner="worktree")], episodes.get)
        == "write"
    )
    assert derive_effect([Cover("w", "cell", index=0)], episodes.get) is None
    assert derive_effect([], episodes.get) is None


def test_rendered_perturbation_reasons_are_filed_as_they_are_and_input_fixtures_kept_for_stage_5(
    tmp_path,
):
    fakes = _Fakes()
    fakes.kinds = {"grid": "copied"}
    fakes.inputs = [
        SimpleNamespace(path=GRID, data=b"[[0, 1]]", entry={"kind": "copy"}),
    ]
    fakes.verdict.reasons = ["[examples:unjudged] rendered by the perturbation check"]
    fakes.verdict.unjudged = ["budget"]
    ex, run, _, _ = _checks(tmp_path, [_call(_arg("grid", [[0, 1]]))], fakes=fakes)
    ex.judge()
    assert _reasons(run) == ["[examples:unjudged] rendered by the perturbation check"]
    assert run.input_fixtures == {ITEM: fakes.inputs}


def test_a_dialogue_procedure_reads():
    from unify.memory.library_checks import derive_effect
    from unify.memory.procedures import Cover

    assert (
        derive_effect(
            [Cover("d", "episode", runner="dialogue", action=0)],
            lambda eid: None,
        )
        == "read"
    )
