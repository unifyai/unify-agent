"""The one comparison of an expected value with an actual one (:mod:`unify.memory.value_diff`): when two values are
the same, and the difference report that names the expected value, the actual one and where they differ. Pure.
"""

from __future__ import annotations

from unify.memory import value_diff as vd


def test_equal_values_have_no_difference_and_text_meets_json_halfway():
    assert vd.difference({"a": [1, 2]}, {"a": [1, 2]}) is None
    assert vd.same("42", 42) and vd.same(" [1, 2]\n", [1, 2]) and vd.same("A.L.\n", "A.L.")
    assert not vd.same("42", 43)


def test_a_json_difference_names_expected_actual_and_each_place():
    report = vd.difference({"table": [[1, 2], [3, 4]], "n": 2}, {"table": [[1, 2], [3, 5]], "extra": True})
    assert report is not None
    lines = report.splitlines()
    assert lines[0].startswith("expected: ") and lines[1].startswith("actual:   ")
    assert "table[1][1]: expected 4, got 5" in report
    assert "n: expected 2, missing" in report
    assert "extra: not expected, got true" in report
    short = vd.difference([1, 2, 3], [1, 2])
    assert "the value: expected 3 items, got 2" in short and "[2]: expected 3, missing" in short


def test_a_text_difference_is_a_unified_diff_and_types_are_named():
    report = vd.difference("line one\nline two\n", "line one\nline 2\n")
    assert "--- expected" in report and "+++ actual" in report
    assert "-line two" in report and "+line 2" in report
    mixed = vd.difference({"a": 1}, "not json")
    assert "expected an object" in mixed and "got text" in mixed


def test_a_long_report_is_cut_and_says_how_much_is_left_out():
    report = vd.difference(list(range(5000)), [x + 1 for x in range(5000)])
    assert len(report) <= vd.REPORT_CHARS + 80
    assert report.rstrip().endswith("more places not shown")
