"""The check with unseen values: a library function's tests rerun on its input fixtures with unseen values, and what
its calls do there recorded, never judged.

No sandbox: the test runner is a fake that plays a test module over the fixture files of the tree it is given and
records the calls as the gate's plugin would (digests of canonical JSON), so every run is deterministic.
"""

import hashlib
import json
import time
from pathlib import Path
from typing import Any, Callable

import pytest

from unify.memory import example_perturb
from unify.memory.episodes import Action
from unify.memory.example_calls import ArgRecord, CallKey, CallRecord, ExampleCall
from unify.memory.example_perturb import InputFixture, check, input_fixtures
from unify.memory.fixtures import canonical
from unify.memory.sandbox_run import PytestOutcome


ITEM = "memory.acct.users:pick"
DATA = "memory/acct/tests/data/"
TEST = "memory/acct/tests/test_users.py"


class MemoryInputError(Exception):
    pass


def _sha(v: Any) -> str:
    return hashlib.sha256(canonical(v).encode()).hexdigest()


def _call(
    fn: Callable,
    args: dict,
    *,
    opened: tuple = (),
    ordinal: int = 0,
    line: int = 7,
) -> ExampleCall:
    try:
        result, outcome, raised = fn(**args), "returned", None
    except MemoryInputError:
        result, outcome, raised = None, "refused", None
    except Exception as exc:  # the plugin's record: the builtin class name
        result, outcome, raised = None, "raised", type(exc).__name__
    ok = outcome == "returned"
    return ExampleCall(
        item=ITEM,
        key=CallKey(f"{TEST}::test_pick", (), ordinal),
        node=f"{TEST}::test_pick",
        file=TEST,
        line=line,
        args=tuple(
            ArgRecord(n, _sha(v), type(v).__name__, v, False, None)
            for n, v in args.items()
        ),
        outcome=outcome,
        raised=raised,
        result_sha=_sha(result) if ok else None,
        result_type=type(result).__name__ if ok else None,
        result_json=result if ok else None,
        opened=tuple(opened),
    )


def _load(tree: Path, name: str) -> Any:
    return json.loads((tree / DATA / name).read_text())


class _Runner:
    """A fake test run: *script(tree)* gives the calls; every run's fixture bytes are kept."""

    def __init__(
        self,
        script: Callable[[Path], list[ExampleCall]],
        failing: bool = False,
    ) -> None:
        self.script, self.failing, self.files = script, failing, []

    def __call__(self, tree: Path, files: list[str]) -> PytestOutcome:
        assert files == [TEST]
        self.files.append({p.name: p.read_bytes() for p in (tree / DATA).iterdir()})
        calls = self.script(tree)
        failed = (
            {"test_users.py::test_pick"}
            if self.failing and len(self.files) > 1
            else set()
        )
        return PytestOutcome(
            failed=failed,
            returncode=1 if failed else 0,
            calls=CallRecord(loaded=True, calls=calls, ended=True),
        )


def _entry(data: bytes, source: str = "3", episode: str = "e1") -> dict:
    return {
        "kind": "copy",
        "episode": episode,
        "source": source,
        "slice": None,
        "trust": 2 if source.startswith("args:") else 1,
        "sha256": hashlib.sha256(data).hexdigest(),
    }


def _tree(tmp_path: Path, files: dict[str, Any]) -> tuple[Path, list[InputFixture]]:
    tree = tmp_path / "tree"
    (tree / DATA).mkdir(parents=True)
    out = []
    for name, value in files.items():
        data = value if isinstance(value, bytes) else json.dumps(value).encode()
        (tree / DATA / name).write_bytes(data)
        out.append(InputFixture(DATA + name, data, _entry(data), None))
    return tree, out


def _check(tmp_path: Path, tree: Path, fixtures: list, runner: _Runner, **kw: Any):
    args = dict(
        calls=runner.script(tree),
        fixtures=fixtures,
        tree=tree,
        tests=[TEST],
        run_tests=runner,
        pool=[],
        pool_capped=False,
        seen=[],
        blob=_no_blob,
        field_types={},
        work=tmp_path / "work",
        deadline=time.monotonic() + 60,
    )
    args.update(kw)
    return check(ITEM, **args)


def _no_blob(sha: str) -> bytes:
    raise KeyError(sha)


def test_a_function_that_accepts_only_a_recorded_id_has_its_refusal_recorded_with_the_field_and_fixture(
    tmp_path,
):
    tree, fx = _tree(
        tmp_path,
        {"row.json": {"id": "E1243", "amount": 12, "note": "paid"}},
    )

    def pick(row):
        if row["id"] != "E1243":
            raise MemoryInputError("unknown row")
        return row["amount"] * 2

    runner = _Runner(
        lambda t: [
            _call(pick, {"row": _load(t, "row.json")}, opened=(DATA + "row.json",)),
        ],
    )
    v = _check(tmp_path, tree, fx, runner)
    assert [(f, name) for _, f, name in v.refused] == [(DATA + "row.json", "id")]
    assert v.special == [] and v.unjudged == []
    assert not hasattr(v, "reasons")  # what the calls did is recorded, never a refusal text
    assert v.runs <= example_perturb.PERTURB_MAX_RUNS and v.fields == 3
    assert "E1243" not in " ".join(v.notes)  # no recorded value in a note


def test_a_lookup_table_of_recorded_inputs_raises_and_the_raise_is_recorded(tmp_path):
    tree, fx = _tree(tmp_path, {"row.json": {"id": "E1243"}})

    def pick(row):
        return {"E1243": 5, "E1488": 9}[row["id"]]

    runner = _Runner(
        lambda t: [
            _call(pick, {"row": _load(t, "row.json")}, opened=(DATA + "row.json",)),
        ],
    )
    v = _check(tmp_path, tree, fx, runner)
    assert [(f, name, raised) for _, f, name, raised in v.special] == [
        (DATA + "row.json", "id", "KeyError"),
    ]
    assert not any("special-cased" in n for n in v.notes)


def test_a_join_of_two_fixtures_stays_joined_under_consistent_replacement(tmp_path):
    tree, fx = _tree(
        tmp_path,
        {
            "users.json": [{"user_id": "U-17", "name": "Ann"}],
            "orders.json": [{"order": "O-1", "user_id": "U-17", "total": 30}],
        },
    )

    def pick(users, orders):
        names = {u["user_id"]: u["name"] for u in users}
        return {names[o["user_id"]]: o["total"] for o in orders}

    opened = (DATA + "users.json", DATA + "orders.json")
    runner = _Runner(
        lambda t: [
            _call(
                pick,
                {"users": _load(t, "users.json"), "orders": _load(t, "orders.json")},
                opened=opened,
            ),
        ],
    )
    v = _check(tmp_path, tree, fx, runner)
    assert v.refused == [] and v.special == []
    group = runner.files[1]
    users, orders = json.loads(group["users.json"]), json.loads(group["orders.json"])
    assert users[0]["user_id"] != "U-17" and users[0]["user_id"] == orders[0]["user_id"]
    (key,) = v.derived
    assert v.derived[key]["users"] and v.derived[key]["orders"]
    assert v.judged >= 1


def test_an_exact_assertion_failing_on_changed_values_is_never_itself_a_refusal(
    tmp_path,
):
    tree, fx = _tree(tmp_path, {"row.json": {"id": "E1243", "amount": 12}})

    def pick(row):
        return {"id": row["id"], "double": row["amount"] * 2}

    runner = _Runner(
        lambda t: [
            _call(pick, {"row": _load(t, "row.json")}, opened=(DATA + "row.json",)),
        ],
        failing=True,
    )
    v = _check(tmp_path, tree, fx, runner)
    assert v.refused == [] and v.special == [] and v.type_changes == []
    assert v.judged == 1 and v.runs == 2
    assert v.derived[next(iter(v.derived))]["row"] == {DATA + "row.json"}


def test_a_nondeterministic_call_is_excluded_as_unstable(tmp_path):
    tree, fx = _tree(tmp_path, {"row.json": {"id": "E1243", "amount": 12}})
    counter = iter(range(1000))

    def stamp(row):
        return next(counter)

    def pick(row):
        return row["amount"]

    runner = _Runner(
        lambda t: [
            _call(
                stamp,
                {"row": _load(t, "row.json")},
                opened=(DATA + "row.json",),
                ordinal=0,
            ),
            _call(
                pick,
                {"row": _load(t, "row.json")},
                opened=(DATA + "row.json",),
                ordinal=1,
            ),
        ],
    )
    v = _check(tmp_path, tree, fx, runner)
    assert v.unstable == 1
    assert (
        f"1 calls of {ITEM} differ between two unperturbed runs; not judged" in v.notes
    )
    assert v.refused == [] and v.special == [] and v.judged == 1


def test_a_rejection_sourced_fixture_exempts_only_the_field_it_varied(tmp_path):
    request = {"args": [], "kwargs": {"amount": 999, "currency": "USD", "to": "ACC-1"}}
    data = json.dumps(request).encode()
    tree, _ = _tree(tmp_path, {"pay.json": data})
    rejected = Action(
        0,
        "bank",
        "pay",
        [],
        dict(request["kwargs"]),
        status="error",
        error="over the limit",
    )
    accepted = Action(
        1,
        "bank",
        "pay",
        [],
        {"amount": 10, "currency": "USD", "to": "ACC-1"},
        {"ok": True},
        "ok",
    )
    fx = [
        InputFixture(
            DATA + "pay.json",
            data,
            _entry(data, "args:0"),
            ("tool", "bank", "pay"),
            rejected,
        ),
    ]

    def pick(request):
        kw = request["kwargs"]
        if kw["amount"] > 1000 or kw["to"] != "ACC-1":
            raise MemoryInputError("not payable")
        return {"sent": kw["amount"]}

    runner = _Runner(
        lambda t: [
            _call(pick, {"request": _load(t, "pay.json")}, opened=(DATA + "pay.json",)),
        ],
    )
    v = _check(tmp_path, tree, fx, runner, seen=[accepted])
    assert [name for _, _, name in v.refused] == ["kwargs.to"]
    assert (
        f"refusal of kwargs.amount of {DATA}pay.json by {ITEM} allowed by a recorded rejection"
        in v.notes
    )


def test_a_declared_month_accepts_unseen_months_and_may_refuse_thirteen(tmp_path):
    rows = [{"month": 3, "amount": 5}, {"month": 11, "amount": 7}]
    tree, fx = _tree(tmp_path, {"rows.json": rows})

    def pick(rows):
        for r in rows:
            if not 1 <= r["month"] <= 12:
                raise MemoryInputError("month out of range")
        return sum(r["amount"] for r in rows)

    def script(t):
        return [
            _call(pick, {"rows": _load(t, "rows.json")}, opened=(DATA + "rows.json",)),
        ]

    undeclared = _check(tmp_path, tree, fx, _Runner(script))
    assert [name for _, _, name in undeclared.refused] == ["[].month"]

    runner = _Runner(script)
    declared = _check(
        tmp_path,
        tree,
        fx,
        runner,
        field_types={DATA + "rows.json#[].month": "month"},
    )
    assert declared.refused == []
    months = [json.loads(run["rows.json"])[0]["month"] for run in runner.files]
    assert (
        months[0] == 3 and 1 <= months[1] <= 12 and months[1] not in (3, 11)
    )  # control, then in-domain
    assert sorted(months[2:]) == [
        0,
        13,
    ]  # the out-of-domain runs: refused, which is allowed
    assert not any("accepts out-of-domain" in n for n in declared.notes)


def test_running_out_of_budget_leaves_the_function_unjudged(tmp_path, monkeypatch):
    tree, fx = _tree(tmp_path, {"row.json": {"id": "E1243", "amount": 12}})

    def pick(row):
        return row["amount"]

    def script(t):
        return [_call(pick, {"row": _load(t, "row.json")}, opened=(DATA + "row.json",))]

    late = _check(tmp_path, tree, fx, _Runner(script), deadline=time.monotonic() - 1)
    assert late.runs == 0
    assert late.unjudged == ["the budget ran out before the control run"]
    monkeypatch.setattr(example_perturb, "PERTURB_MAX_RUNS", 1)
    short = _check(tmp_path, tree, fx, _Runner(script))
    assert short.runs == 1
    assert short.unjudged == ["the budget ran out before the run with every field changed"]


def test_an_incomplete_call_record_of_the_control_leaves_the_function_unjudged(
    tmp_path,
):
    tree, fx = _tree(tmp_path, {"row.json": {"id": "E1243", "amount": 12}})

    def pick(row):
        return row["amount"]

    calls = [
        _call(
            pick,
            {"row": {"id": "E1243", "amount": 12}},
            opened=(DATA + "row.json",),
        ),
    ]

    def run(tree, files):
        return PytestOutcome(calls=CallRecord(loaded=True, calls=calls, ended=False))

    v = check(
        ITEM,
        calls=calls,
        fixtures=fx,
        tree=tree,
        tests=[TEST],
        run_tests=run,
        pool=[],
        pool_capped=False,
        seen=[],
        blob=_no_blob,
        field_types={},
        work=tmp_path / "work",
        deadline=time.monotonic() + 60,
    )
    assert v.unjudged == ["the call record of the control run is incomplete"]


def test_an_assembled_file_has_only_its_inputs_changed(tmp_path):
    lines = [
        {
            "input": {"id": "E1", "n": 2},
            "source": {"episode": "e1", "source": "4", "slice": None},
        },
        {
            "input": {"id": "E7", "n": 5},
            "source": {"episode": "e2", "source": "6", "slice": None},
        },
    ]
    data = "".join(json.dumps(x) + "\n" for x in lines).encode()
    tree, _ = _tree(tmp_path, {"cases.inputs.jsonl": data})
    entry = {
        "kind": "assembled",
        "lines": [
            {
                **x["source"],
                "trust": 1,
                "sha256": hashlib.sha256(canonical(x["input"]).encode()).hexdigest(),
            }
            for x in lines
        ],
    }
    fx = [InputFixture(DATA + "cases.inputs.jsonl", data, entry, None)]

    def pick(case):
        return case["n"] + 1

    def script(t):
        rows = [
            json.loads(x)
            for x in (t / DATA / "cases.inputs.jsonl").read_text().splitlines()
        ]
        opened = (DATA + "cases.inputs.jsonl",)
        return [
            _call(pick, {"case": r["input"]}, opened=opened, ordinal=i)
            for i, r in enumerate(rows)
        ]

    runner = _Runner(script)
    v = _check(tmp_path, tree, fx, runner)
    assert v.refused == [] and v.special == []
    changed = [
        json.loads(x)
        for x in runner.files[1]["cases.inputs.jsonl"].decode().splitlines()
    ]
    assert [c["source"] for c in changed] == [x["source"] for x in lines]
    assert changed[0]["input"] != lines[0]["input"]
    assert all(set(c) == {"input", "source"} for c in changed)


def test_a_lookup_returning_none_on_an_unseen_id_is_recorded_as_a_type_change_not_refused(
    tmp_path,
):
    tree, fx = _tree(tmp_path, {"row.json": {"id": "E1"}})

    def pick(row):
        return {"E1": 5}.get(row["id"])

    runner = _Runner(
        lambda t: [
            _call(pick, {"row": _load(t, "row.json")}, opened=(DATA + "row.json",)),
        ],
    )
    v = _check(tmp_path, tree, fx, runner)
    assert v.refused == [] and v.special == []
    (entry,) = v.type_changes
    assert (
        entry["item"] == ITEM
        and entry["fixture"] == DATA + "row.json"
        and entry["field"] == "id"
    )
    assert entry["call"] == {"test": f"{TEST}::test_pick", "indices": {}, "ordinal": 0}
    assert (entry["file"], entry["line"]) == (TEST, 7)
    assert entry["before"] == {"type": "int", "json": 5}
    assert entry["after"] == {"type": "NoneType", "sha": _sha(None)}
    assert entry["field_before"] == {"type": "str", "json": "E1"}
    assert (
        entry["field_after"]["type"] == "str" and entry["field_after"]["json"] != "E1"
    )
    # the type change is recorded for review, never shown as a finding
    assert not any(n.startswith("special-case suspected") for n in v.notes)


def test_a_large_result_in_a_type_change_is_kept_as_its_digest(tmp_path):
    tree, fx = _tree(tmp_path, {"row.json": {"id": "E1"}})

    def pick(row):
        return "x" * 2000 if row["id"] == "E1" else None

    runner = _Runner(
        lambda t: [
            _call(pick, {"row": _load(t, "row.json")}, opened=(DATA + "row.json",)),
        ],
    )
    v = _check(tmp_path, tree, fx, runner)
    (entry,) = v.type_changes
    assert entry["before"] == {"type": "str", "sha": _sha("x" * 2000)}


def test_input_fixtures_are_the_package_fixtures_the_calls_load(tmp_path):
    tree, _ = _tree(
        tmp_path,
        {"a.json": {"id": 1}, "b.json": {"id": 2}, "c.json": {"id": 3}},
    )
    (tree / "memory/other/tests/data").mkdir(parents=True)
    (tree / "memory/other/tests/data/d.json").write_text("{}")
    ledger = {
        DATA + n: _entry((tree / DATA / n).read_bytes())
        for n in ("a.json", "b.json", "c.json")
    } | {"memory/other/tests/data/d.json": _entry(b"{}")}
    a_sha = _sha({"id": 1})
    calls = [
        _call(
            lambda row: 1,
            {"row": {"id": 1}},
            opened=(DATA + "b.json", "memory/other/tests/data/d.json"),
        ),
    ]
    views = {DATA + "a.json": {a_sha: ["value"]}}
    action = Action(3, "files", "read", [], {}, {"ok": True}, "ok")
    notes: list[str] = []
    got = input_fixtures(
        ITEM,
        calls,
        views,
        ledger,
        tree,
        load=lambda eid: None,
        source_action=lambda row, load: ("e1", 3, action),
        notes=notes,
    )
    assert sorted(f.path for f in got) == [DATA + "a.json", DATA + "b.json"]
    assert all(
        f.family == ("tool", "files", "read") and f.action is action for f in got
    )
    assert got[0].data == (tree / got[0].path).read_bytes()


@pytest.mark.parametrize("n", [example_perturb.MAX_INPUT_FIXTURES + 2])
def test_input_fixtures_past_the_bound_are_noted(tmp_path, n):
    names = {f"f{i}.json": {"id": i} for i in range(n)}
    tree, _ = _tree(tmp_path, names)
    ledger = {DATA + k: _entry((tree / DATA / k).read_bytes()) for k in names}
    calls = [_call(lambda row: 1, {"row": {"id": 0}}, opened=tuple(ledger))]
    notes: list[str] = []
    got = input_fixtures(ITEM, calls, {}, ledger, tree, notes=notes)
    assert len(got) == example_perturb.MAX_INPUT_FIXTURES
    assert notes == [
        f"{ITEM}: 2 input fixtures past the first {example_perturb.MAX_INPUT_FIXTURES} (by a hash of item and "
        "path) are not perturbed",
    ]
