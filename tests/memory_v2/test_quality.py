from unify.memory_v2 import quality as tq
from unify.memory_v2.episodes import Action
from unify.memory_v2.fixtures import canonical
from tests.memory_v2.test_episodes import _ep

FN = b'''def total(rows):
    """Sum of the amounts."""
    return sum(r["amt"] for r in rows)
'''
FIXTURES = {
    "memory/office/tests/data/rows.json",
    "memory/office/tests/data/ledger.total.inputs.jsonl",
}
RECORDED = {canonical("E-1243"), canonical(42)}
HEAD = b"""import json
from pathlib import Path

import pytest

from memory.office.ledger import total

DATA = Path(__file__).parent / "data"
ROWS = json.loads((DATA / "rows.json").read_text())
INPUTS = [json.loads(x) for x in (DATA / "ledger.total.inputs.jsonl").read_text().splitlines() if x]
"""


def _exacts(body: bytes):
    return tq.exact_assertions(
        HEAD + body,
        FN,
        "total",
        FIXTURES,
        RECORDED,
        path="t.py",
    )


def test_fixture_input_and_recorded_literal_count():
    [e] = _exacts(b"\n\ndef test_recorded():\n    assert total(ROWS) == 42\n")
    assert (e.test, e.on_fixture, e.trusted, e.oracle) == (
        "t.py::test_recorded",
        True,
        True,
        False,
    )
    [e] = _exacts(
        b"\n\ndef test_via_name():\n    got = total(ROWS)\n    assert got == 42\n",
    )
    assert e.on_fixture and e.trusted


def test_parametrised_inputs_file_and_fixture_expected_value_count():
    body = (
        b"\n\n@pytest.mark.parametrize('row', INPUTS)\ndef test_each(row):\n"
        b"    assert total(row['input']) == row['input'][0]['amt'] + row['input'][1]['amt']\n"
    )
    [e] = _exacts(body)
    assert e.on_fixture and e.trusted and not e.oracle


def test_oracle_and_invented_literal_do_not_count():
    [e] = _exacts(b"\n\ndef test_invented():\n    assert total(ROWS) == 41\n")
    assert e.on_fixture and not e.trusted
    [e] = _exacts(
        b"\n\ndef test_oracle():\n    expected = sum(r['amt'] for r in ROWS)\n"
        b"    assert total(ROWS) == expected\n",
    )
    assert e.oracle
    [e] = _exacts(b"\n\ndef test_trivial():\n    assert total(ROWS) == 0\n")
    assert not e.trusted
    [e] = _exacts(
        b"\n\ndef test_no_fixture():\n    assert total([{'amt': 42}]) == 42\n",
    )
    assert not e.on_fixture
    assert (
        _exacts(
            b"\n\ndef test_metamorphic():\n    assert total(ROWS) == total(list(ROWS))\n",
        )
        == []
    )


def test_pytest_fixture_returning_a_fixture_value_taints_its_parameter():
    body = (
        b"\n\n@pytest.fixture\ndef rows():\n    return json.loads((DATA / 'rows.json').read_text())\n\n\n"
        b"def test_with_fixture(rows):\n    assert total(rows) == 42\n"
    )
    [e] = _exacts(body)
    assert e.on_fixture and e.trusted


def test_recorded_literals_cover_values_tokens_and_lines():
    ep = _ep(
        episode_id="e1",
        request=["Pay ENG-0042 now"],
        actions=[
            Action(
                0,
                "acct",
                "get",
                [],
                {"id": "E-1243"},
                {"amount": 42, "memo": "paid OPS-7 ok"},
                "ok",
            ),
        ],
    )
    lits = tq.recorded_literals([ep], lambda sha: b"")
    for v in (
        "E-1243",
        42,
        "ENG-0042",
        "OPS-7",
        {"amount": 42, "memo": "paid OPS-7 ok"},
        "Pay ENG-0042 now",
    ):
        assert canonical(v) in lits


def test_inventory_and_test_changes():
    before = tq.inventory(
        {
            "memory/o/tests/test_a.py": b"import pytest\n\ndef test_one():\n    assert f(1) == 2\n"
            b"    assert f(2) == 4\n\ndef test_two():\n    with pytest.raises(E):\n        f(0)\n\n"
            b"class TestThree:\n    def test_x(self):\n        assert f(3) == 6\n",
        },
    )
    assert before == {
        "memory/o/tests/test_a.py::test_one": tq.TestInfo(2, 2),
        "memory/o/tests/test_a.py::test_two": tq.TestInfo(1, 0),
        "memory/o/tests/test_a.py::TestThree::test_x": tq.TestInfo(1, 1),
    }
    after = tq.inventory(
        {
            "memory/o/tests/test_a.py": b"def test_one():\n    assert f(1) == 2\n    assert f(2)\n\n"
            b"class TestThree:\n    def test_x(self):\n        assert f(3) == 6\n",
        },
    )
    assert tq.test_changes(before, after) == (
        ["memory/o/tests/test_a.py::test_two"],
        ["memory/o/tests/test_a.py::test_one"],
    )
    broken = tq.inventory({"memory/o/tests/test_a.py": b"def (:\n"})
    assert tq.test_changes(before, broken)[0] == sorted(before)


def test_count_items():
    files = {
        "memory/__init__.py": b"def show(x):\n    pass\n",
        "memory/office/payroll.py": b"def export(p):\n    pass\n\ndef _helper():\n    pass\n\nasync def fetch(e):\n    pass\n",
        "memory/office/tests/test_payroll.py": b"def test_export():\n    pass\n",
        "notes/office/month-end.md": b"---\ntitle: x\n---\n",
        "INDEX.md": b"",
    }
    assert tq.count_items(files) == 3
    assert tq.TEST_FILE.match("memory/office/tests/test_payroll.py")
    assert not tq.TEST_FILE.match("memory/office/payroll.py")
