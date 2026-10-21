import pytest

from unify.memory import code_lint as cl


def test_varying_values_from_json_leaves_and_text_tokens():
    v = cl.varying_values(
        [
            {"id": "E-1243", "cur": "USD", "n": 12},
            {"id": "E-1488", "cur": "USD", "n": 15},
        ],
    )
    assert {"str:E-1243", "str:E-1488", "num:12", "num:15"} <= v and "str:USD" not in v
    t = cl.varying_values(["paid ENG-0042 today", "paid OPS-0043 today"])
    assert {"str:ENG-0042", "str:OPS-0043"} <= t and "str:paid" not in t
    assert cl.varying_values([{"id": "E-1243"}]) == set()


@pytest.mark.parametrize(
    "value,digits",
    [(10, 1), (12, 2), (0.5, 1), (3.14, 3), (1200, 2), (0, 1)],
)
def test_sig_digits(value, digits):
    assert cl.sig_digits(value) == digits


SOURCE = b'''TARGET = "E-1488"
UNUSED = "E-1243"


def pick(row, cur="E-1488"):
    """Pick E-1243 rows."""
    if row["id"] == "E-1243":
        return 12
    if row["id"] == TARGET:
        return 1.5
    if not row:
        raise ValueError("E-1243 is required")
    return 10
'''


def test_literal_problems_name_lines_never_values():
    varying = {"str:E-1243", "str:E-1488", "num:12", "num:10", "num:1.5"}
    got = cl.literal_problems(SOURCE, "pick", varying)
    assert got == [
        (1, "string of 6 characters"),
        (7, "string of 6 characters"),
        (8, "number"),
        (10, "number"),
    ]
    assert cl.literal_problems(SOURCE, "missing", varying) == []
    assert cl.literal_problems(SOURCE, "pick", varying, lint_min=7) == [
        (8, "number"),
        (10, "number"),
    ]


