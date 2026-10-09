from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace

from unify.memory_v2 import mutation
from unify.memory_v2.fixtures import inputs_file
from unify.memory_v2.qa import (
    MUTATION_MIN,
    QAChecks,
    QAConfig,
    QAEnv,
    drawn_value,
    v21_config,
)
from unify.memory_v2.sandbox_run import PytestOutcome

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
    it = SimpleNamespace(
        item=ITEM,
        kind="function",
        tests=[TEST],
        input="observation",
        covers=[],
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
        qa=qa or v21_config(QAConfig()),
        pytest=pytest_fn,
        python=Path("/usr/bin/python3"),
    )
    return QAChecks(gate, run), run, it


def _tree(ro):
    return next(p for p, d in ro.items() if d == "/memory")


def test_v21_config_gates_strict_drawn_inputs_and_mutation_at_06():
    cfg = v21_config(QAConfig())
    assert (
        cfg.fixtures == "strict"
        and cfg.mutation
        and cfg.min_kill == MUTATION_MIN == Decimal("0.6")
        and cfg.v21
    )
    assert QAConfig().v21 is False and QAConfig().guard_mutants == 8


def test_mutants_record_kill_share_and_guard_coverage(tmp_path):
    control = mutation.reprint(MOD)

    def fake(target, *, python, ro, rw, cwd, timeout_s, env, import_skips_fail=False):
        text = (_tree(ro) / REL).read_text()
        green = (
            text == control or "raise MemoryInputError" not in text
        )  # every mutant killed but a dropped guard
        return (
            PytestOutcome(passed={"t::a"})
            if green
            else PytestOutcome(failed={"t::a"}, returncode=1)
        )

    checks, run, it = _world(tmp_path, fake)
    checks._mutants(it, None, None)
    assert run.verification[ITEM]["mutation"] == {
        "killed": 4,
        "total": 5,
        "equivalent": 0,
    }
    assert run.verification[ITEM]["guard"] == {"killed": 0, "total": 1}
    assert run.fails == []  # 0.8 >= 0.6; guard coverage is measured, not gated


def test_mutation_below_06_refuses_the_item(tmp_path):
    def fake(target, *, python, ro, rw, cwd, timeout_s, env, import_skips_fail=False):
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


def test_drawn_value_forms(tmp_path):
    samples, data = tmp_path / "s", tmp_path / "d"
    (samples / "files/3").mkdir(parents=True)
    (samples / "files/3/pay.csv").write_bytes(b"a,1\n")
    assert drawn_value(
        {"id": 1, "form": "observation", "action": {"response": {"n": 1}}},
        samples,
        data,
    ) == {"n": 1}
    assert (
        drawn_value(
            {"id": 2, "form": "text", "text": "hi", "action": {}},
            samples,
            data,
        )
        == "hi"
    )
    ctx = [{"method": "get"}]
    assert (
        drawn_value(
            {"id": 4, "form": "env", "context": ctx, "action": {}},
            samples,
            data,
        )
        == ctx
    )
    got = drawn_value(
        {"id": 3, "form": "path", "file": "/qa/files/3/pay.csv", "action": {}},
        samples,
        data,
    )
    assert got == "_drawn/3/pay.csv" and (data / got).read_bytes() == b"a,1\n"


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
    ):
        return PytestOutcome(
            passed={f"{TEST}::test_half[row{i}]" for i in range(_lines(ro))},
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
    ):
        return PytestOutcome(passed={f"{TEST}::test_half[row0]"})

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
