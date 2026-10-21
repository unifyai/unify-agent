import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

from unify.memory.episodes import Action
from unify.memory.example_calls import ArgRecord, CallKey, CallRecord, ExampleCall
from unify.memory.example_perturb import InputFixture
from unify.memory.fixtures import canonical, inputs_file
from unify.memory.qa import QAChecks, QAConfig, QAEnv, library_config
from unify.memory.sandbox_run import PytestOutcome

ITEM = "memory.calc.ops:half"
REL = "memory/calc/ops.py"
TEST = "memory/calc/tests/test_ops.py"
MOD = """class MemoryInputError(ValueError):
    pass


def half(n):
    if n % 2:
        raise MemoryInputError("an even number is needed")
    return n // 2
"""


class _Run(SimpleNamespace):
    def fail(self, check, reason, item=None):
        self.fails.append((check, reason, item))

    def note(self, reason):
        self.notes.append(reason)


def _world(tmp_path, pytest_fn, qa=None):
    tree = tmp_path / "cand"
    (tree / "memory/calc/tests/data").mkdir(parents=True)
    (tree / REL).write_text(MOD)
    (tree / TEST).write_text("# the tests (run by the fake pytest)\n")
    (tree / inputs_file(ITEM)).write_text(
        '{"input": 4, "source": {"episode": "e1", "source": "0", "slice": null}}\n',
    )
    (tmp_path / "inputs").mkdir()
    # a library-mode item: no declared input and no action cover; its recorded-inputs file line is the recorded
    # response of action 0 of e1, so a drawn input of its family is an observation (the fixture's source kind)
    it = SimpleNamespace(
        item=ITEM,
        kind="function",
        tests=[TEST],
        input=None,
        covers=[],
        typed_covers=[],
    )
    run = _Run(
        candidate="c" * 40,
        c_tree=tree,
        c_files={REL: 1, TEST: 1, inputs_file(ITEM): 1},
        changed=[REL, TEST],
        man=SimpleNamespace(items=[it]),
        qa_first={TEST: PytestOutcome(passed={f"{TEST}::test_half[row0]"})},
        qa_env=QAEnv(tmp_path / "inputs", {}),
        tmp=tmp_path,
        verification={},
        item_fail={},
        p_bodies={},
        c_bodies={ITEM: ("function", "b", True)},
        fails=[],
        notes=[],
        res=SimpleNamespace(passed=True),
        pass_wide=False,
    )
    gate = SimpleNamespace(
        qa=qa or library_config(QAConfig()),
        pytest=pytest_fn,
        python=Path("/usr/bin/python3"),
    )
    return QAChecks(gate, run), run, it


def _tree(ro):
    return next(p for p, d in ro.items() if d == "/memory")


def test_mutation_below_06_refuses_the_item(tmp_path):
    def fake(
        target,
        *,
        python,
        ro,
        rw,
        cwd,
        timeout_s,
        env,
        import_skips_fail=False,
        calls=None,
    ):
        text = (_tree(ro) / REL).read_text()
        killed = (
            "% 3" in text or "// 3" in text
        )  # only the two constant mutants are killed
        return (
            PytestOutcome(failed={"t::a"}, returncode=1)
            if killed
            else PytestOutcome(passed={"t::a"})
        )

    checks, run, it = _world(tmp_path, fake)
    checks._mutants(it, None, None)
    assert run.verification[ITEM]["mutation"] == {
        "killed": 2,
        "total": 5,
        "equivalent": 0,
    }
    assert [(c, i) for c, _, i in run.fails] == [
        ("G3", ITEM),
    ] and "threshold 0.6" in run.fails[0][1]


#: The form a drawn input reaches a library function's tests in, by the source kind of the input fixture it is drawn
#: for: what the tests load from such a fixture.
FORM_OF_SOURCE = {
    "tool response": "observation",
    "work-tree blob": "path",
    "dialogue": "observation",
    "request": "text",
    "cell": "text",
}


_MODULE: dict = {}
exec(
    MOD,
    _MODULE,
)  # the module under test, run by the fake pytest to make its call records


def _sha(v):
    return hashlib.sha256(canonical(v).encode()).hexdigest()


def _half_call(i, value):
    try:
        result, outcome, raised = _MODULE["half"](value), "returned", None
    except _MODULE["MemoryInputError"]:
        result, outcome, raised = None, "refused", None
    except Exception as exc:
        result, outcome, raised = None, "raised", type(exc).__name__
    ok = outcome == "returned"
    return ExampleCall(
        item=ITEM,
        key=CallKey(f"{TEST}::test_half", (("row", i),), 0),
        node=f"{TEST}::test_half[row{i}]",
        file=TEST,
        line=9,
        args=(ArgRecord("n", _sha(value), type(value).__name__, value, False, None),),
        outcome=outcome,
        raised=raised,
        result_sha=_sha(result) if ok else None,
        result_type=type(result).__name__ if ok else None,
        result_json=result if ok else None,
        opened=(inputs_file(ITEM),),
    )


def _record(ro, lines=None):
    """The calls a test parametrised over the recorded-inputs file makes, one per line."""
    rows = (_tree(ro) / inputs_file(ITEM)).read_text().splitlines()[:lines]
    calls = [_half_call(i, json.loads(line)["input"]) for i, line in enumerate(rows)]
    return CallRecord(loaded=True, calls=calls, ended=True)


def _lines(ro):
    return len((_tree(ro) / inputs_file(ITEM)).read_text().splitlines())


def test_appended_inputs_must_be_exercised(tmp_path):
    def reads_all(
        target,
        *,
        python,
        ro,
        rw,
        cwd,
        timeout_s,
        env,
        import_skips_fail=False,
        calls=None,
    ):
        assert calls == [ITEM]
        return PytestOutcome(
            passed={f"{TEST}::test_half[row{i}]" for i in range(_lines(ro))},
            calls=_record(ro),
        )

    checks, run, it = _world(tmp_path, reads_all)
    drawn = [
        {"id": 7, "role": "sample", "form": "observation", "action": {"response": 6}},
    ]
    checks._appended(it, drawn, tmp_path / "samples")
    assert run.fails == [] and run.verification[ITEM]["drawn_inputs_read"] == 1

    def reads_first(
        target,
        *,
        python,
        ro,
        rw,
        cwd,
        timeout_s,
        env,
        import_skips_fail=False,
        calls=None,
    ):
        return PytestOutcome(passed={f"{TEST}::test_half[row0]"}, calls=_record(ro, 1))

    checks, run, it = _world(tmp_path / "b", reads_first)
    checks._appended(it, drawn, tmp_path / "samples")
    assert run.verification[ITEM]["drawn_inputs_read"] == 0
    assert "every drawn input must be exercised" in run.fails[0][1]
    # the candidate's own inputs file is untouched: the drawn lines go into a copy
    assert len((run.c_tree / inputs_file(ITEM)).read_text().splitlines()) == 1


def test_no_inputs_file_refuses_when_inputs_were_drawn(tmp_path):
    checks, run, it = _world(tmp_path, None)
    run.c_files.pop(inputs_file(ITEM))
    checks._appended(
        it,
        [{"id": 7, "role": "sample", "form": "observation", "action": {"response": 6}}],
        tmp_path,
    )
    assert "has no recorded-inputs file" in run.fails[0][1]


def test_drawn_inputs_are_judged_on_the_calls_their_appended_lines_make(tmp_path):
    def run_all(
        target,
        *,
        python,
        ro,
        rw,
        cwd,
        timeout_s,
        env,
        import_skips_fail=False,
        calls=None,
    ):
        assert calls == [ITEM]
        return PytestOutcome(
            passed={f"{TEST}::test_half[row{i}]" for i in range(_lines(ro))},
            calls=_record(ro),
        )

    checks, run, it = _world(tmp_path, run_all)
    drawn = [
        {"id": i, "role": "sample", "form": "observation", "action": {"response": v}}
        for i, v in enumerate(
            [5, "x", 5.0],
        )  # refused like its covers, raising, refused on another type
    ]
    checks._appended(it, drawn, tmp_path / "samples")
    assert [r for _, r, _ in run.fails] == [
        f"[qa:fixtures] {ITEM} raises on 1 of 3 drawn recorded inputs of its family (TypeError (1)); it must "
        "return or raise MemoryInputError",
        f"[qa:fixtures] {ITEM} refuses 1 of 3 drawn recorded inputs shaped like its covers, which the "
        "environment accepted",
    ]
    assert run.notes == [
        f"[qa:fixtures] {ITEM} refuses 1 of 3 drawn inputs whose fields differ from its covers' (a precondition)",
    ]


def test_drawn_inputs_without_a_call_record_are_not_judged_and_refuse_when_strict(
    tmp_path,
):
    def unrecorded(
        target,
        *,
        python,
        ro,
        rw,
        cwd,
        timeout_s,
        env,
        import_skips_fail=False,
        calls=None,
    ):
        return PytestOutcome(
            passed={f"{TEST}::test_half[row{i}]" for i in range(_lines(ro))},
        )

    checks, run, it = _world(tmp_path, unrecorded)
    drawn = [
        {"id": 1, "role": "sample", "form": "observation", "action": {"response": 6}},
    ]
    checks._appended(it, drawn, tmp_path / "samples")
    assert [r for _, r, _ in run.fails] == [
        f"[qa:fixtures] {ITEM}: the call record of {TEST} with the drawn inputs appended is missing or "
        "incomplete; its drawn inputs are not judged",
    ]


def _fixture(name, entry, family=None):
    return InputFixture(f"memory/calc/tests/data/{name}", b"{}", entry, family)


def _copy(source, slice_=None):
    return {
        "kind": "copy",
        "episode": "e1",
        "source": source,
        "slice": slice_,
        "trust": 1,
        "sha256": "0" * 64,
    }


def test_drawn_inputs_come_from_the_families_of_the_input_fixtures(tmp_path):
    checks, run, it = _world(tmp_path, None)
    cover = Action(
        2,
        "crm",
        "get_user",
        [],
        {"id": "U1"},
        {"name": "Ann", "age": 30},
        "ok",
    )
    other = Action(
        5,
        "crm",
        "get_user",
        [],
        {"id": "U2"},
        {"name": "Bo", "age": 41},
        "ok",
    )
    checks.gate.lookup = lambda eid, idx: cover if (eid, idx) == ("e1", 2) else None
    checks.gate._pool = lambda run, covers: ([("e1", cover), ("e2", other)], False)
    checks.gate.blobs = SimpleNamespace(has=lambda sha: False, get=lambda sha: b"")
    run.input_fixtures = {
        ITEM: [
            _fixture("user.json", _copy("2"), ("tool", "crm", "get_user")),
            _fixture("part.json", _copy("2", [0, 4])),
        ],
    }
    rows = checks._draw_library(it)
    assert [(r.role, r.form) for r in rows if r.role != "negative"] == [
        ("cover", "observation"),
        ("sample", "observation"),
    ]
    assert [r.action for r in rows if r.role == "sample"] == [other]
    assert (
        f"[qa:fixtures] {ITEM}: 1 of its input fixtures are slices or messages; no recorded inputs of their "
        "family are drawn for them"
    ) in run.notes
    assert checks._form(it) is None  # no declared form in library mode


def test_sliced_or_message_fixtures_give_a_note_and_draw_nothing(tmp_path):
    checks, run, it = _world(tmp_path, None)
    checks.gate.lookup = lambda eid, idx: None
    run.input_fixtures = {
        ITEM: [
            _fixture("part.json", _copy("2", [0, 4])),
            _fixture("ask.json", _copy("request:0")),
        ],
    }
    assert checks._draw_library(it) == []
    assert run.notes == [
        f"[qa:fixtures] {ITEM}: its input fixtures are slices or messages; no recorded inputs of their family "
        "can be drawn",
    ]


def _mutant_run(digest):
    """A fake test run, green on every module text, whose one recorded call returns *digest(text)*."""

    def fake(
        target,
        *,
        python,
        ro,
        rw,
        cwd,
        timeout_s,
        env,
        import_skips_fail=False,
        calls=None,
    ):
        assert calls == [ITEM]
        text = (_tree(ro) / REL).read_text()
        made = []
        if digest is not None:
            call = _half_call(0, 4)
            made = [ExampleCall(**{**call.__dict__, "result_sha": digest(text)})]
        return PytestOutcome(
            passed={"t::a"},
            calls=CallRecord(loaded=True, calls=made, ended=True),
        )

    return fake


def test_mutant_equivalence_is_judged_on_the_tests_recorded_calls(tmp_path):
    # every survivor returns what the original returns on every call: all equivalent, nothing judged
    checks, run, it = _world(tmp_path, _mutant_run(lambda text: "same"))
    checks._mutants(it, None, None)
    assert run.fails == [] and run.verification[ITEM]["mutation"] is None
    assert any("no mutant changed its outputs" in n for n in run.notes)
    # every survivor's result differs: none equivalent, so none killed of all judged refuses the item
    checks, run, it = _world(tmp_path / "b", _mutant_run(lambda text: _sha(text)))
    checks._mutants(it, None, None)
    assert run.verification[ITEM]["mutation"]["equivalent"] == 0
    assert run.verification[ITEM]["mutation"]["killed"] == 0
    assert [i for _, _, i in run.fails] == [ITEM] and "threshold 0.6" in run.fails[0][1]


def test_survivors_without_a_recorded_call_to_compare_are_not_judged(tmp_path):
    checks, run, it = _world(tmp_path, _mutant_run(None))
    checks._mutants(it, None, None)
    (fail,) = run.fails
    assert fail[1].startswith(f"[qa:mutation] {ITEM}: mutation not judged: ")
    assert fail[1].endswith(
        "mutants survive and its tests made no recorded call of it to compare their outputs on",
    )
