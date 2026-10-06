"""Symbolic: the checks behind ``UNIFY_REPLY_RECEIPT``, as pure functions.

The offline gate (first-principles memo §E2, 6 Oct) ran smell-test checks over
2,126 logged final replies from office, AppWorld and Continual-ARC. Two passed
(precision >= 0.4, trigger <= 20%) and are built here: C2_nc, a degenerate
answer (0, NaN, None, empty, every item the same, or a JSON value identical to
one in the request), and C4_last, the last computing cell since the request
raised, or a cell caught and printed an error, while the reply mentions no
error. Pooled, the pair fired on 4.0% of replies with precision 0.80; all 7
failed "meal" office replies (answer 0.00) fired, the one correct one did not.

The rules are the offline ``checks.py`` (``receipt_checks``/``render``)
generalised: no benchmark is named, the request is parsed only as the
``request`` a cell reads under UNIFY_BIND_REQUEST (``json_values``), and the
single-colour rule the gate dropped is not here. A parity test pins the
offline outputs on synthetic cases, and re-runs the offline file when it is
on this machine.
"""

from __future__ import annotations

import hashlib
import importlib
import importlib.util
import json
import os
import pwd
import sys
from pathlib import Path

import pytest

from unify.common._async_tool import reply_receipt as rr

OFFICE_REQUEST = "What did the team spend on meals in March? Reply with the total."
GRID = [[1, 2, 0], [0, 1, 2], [2, 0, 1]]
TRACEBACK = json.dumps(
    {
        "error": 'Traceback (most recent call last):\n  File "<cell>", line 1\n'
        "ValueError: boom",
    },
)


def _cell(code: str, output: str | None, tool: str = "execute_code") -> dict:
    return {"tool": tool, "code": code, "output": output, "noop": rr.noop(code)}


def _lines(reply: str, request: str = "q", cells: list | None = None) -> list[str]:
    return rr.render(rr.checks(reply, request, cells or []))


# ── the switch ───────────────────────────────────────────────────────────


def test_the_switch_is_off_as_shipped_and_validated():
    from unify.settings import ProductionSettings

    assert ProductionSettings().UNIFY_REPLY_RECEIPT == ""
    assert ProductionSettings(UNIFY_REPLY_RECEIPT="off").UNIFY_REPLY_RECEIPT == ""
    assert ProductionSettings(UNIFY_REPLY_RECEIPT=" On ").UNIFY_REPLY_RECEIPT == "on"
    with pytest.raises(ValueError):
        ProductionSettings(UNIFY_REPLY_RECEIPT="all")


# ── C2_nc: a degenerate answer ───────────────────────────────────────────


@pytest.mark.parametrize(
    "reply, line",
    [
        # The office meal case: bold, bare, after "is", "total", "=" and ":".
        ("The total cost of the meals is **0.00**.", "The answer is zero (0.00)."),
        ("0", "The answer is zero (0)."),
        ("The team spent 0 on meals.", None),
        ("The amount is 0 dollars.", "The answer is zero (0)."),
        ("Meals total: 0.0", "The answer is zero (0.0)."),
        ("sum=0", "The answer is zero (0)."),
        ("The average is NaN.", "The answer is NaN."),
        ("The result is `None`.", "The answer is None."),
        ("{}", "The answer is an empty value ({})."),
        ("The overdue invoices: `[]`", "The answer is an empty value ([])."),
        ("```\n[]\n```", "The answer is an empty value ([])."),
        (
            "The ids are `[3, 3, 3]`.",
            "Every item of the answer is the same ([3, 3, 3]).",
        ),
        # A JSON object's primary answer: its field with the most scalars.
        ('{"total": 0}', "The answer is zero (0)."),
        ('{"total": null}', "The answer is null."),
        ('{"action": "submit", "rows": []}', None),
        ('{"rows": []}', "The answer is an empty value ([])."),
        ('{"ids": [4, 4, 4]}', "Every item of the answer is the same ([4, 4, 4])."),
        # Its other fields are not the answer: a small 0 or empty field
        # beside the answer is never checked (the gate never measured it).
        ('{"action": "submit", "x": 0, "rows": [[1, 2], [3, 4]]}', None),
        ('{"action": "answer", "total": 0}', None),
        ('{"action": "move", "note": "", "to": [1, 2]}', None),
        # A JSON array in prose is not the answer.
        ("The ids are [3, 3, 3].", None),
        # Not degenerate.
        ("The total cost of the meals is **42.50**.", None),
        ("Done.", None),
        ('{"action": "move", "to": [1, 2]}', None),
        ('{"ok": false}', None),
        # Rows that are all the same are not "every item the same": a value
        # of one repeated row is not called degenerate (no single-colour rule).
        ('{"action": "submit", "rows": [[0, 0], [0, 0]]}', None),
        ("`[[5, 5], [5, 5]]`", None),
        # A file name in backticks is not an answer.
        ("Wrote the totals to `out/0.csv`.", None),
        # The doubts after the answer are not the answer.
        ("The total is **12**.\n\n**Uncertainties**\nThe rate may be 0.", None),
    ],
)
def test_a_degenerate_answer_is_named(reply, line):
    assert _lines(reply, OFFICE_REQUEST) == ([line] if line else [])


IDENTICAL = "The answer is identical to a value in the request."
COPIED = json.dumps({"action": "submit", "rows": GRID})
OTHER = json.dumps({"action": "submit", "rows": [[0, 1, 2], [0, 1, 2], [2, 0, 1]]})


@pytest.mark.parametrize(
    "rows",
    [
        "1 2 0\n0 1 2\n2 0 1",
        "1, 2, 0\n0, 1, 2\n2, 0, 1",
        "120\n012\n201",
    ],
)
def test_an_answer_identical_to_a_number_table_in_the_request_text_is_named(rows):
    request = (
        f"Transform the input.\nInput (3x3):\n{rows}\nReply with an action object."
    )
    assert _lines(COPIED, request) == [IDENTICAL]
    assert _lines(OTHER, request) == []
    # A bare table, a fenced one and one in backticks are answers too.
    assert _lines(json.dumps(GRID), request) == [IDENTICAL]
    assert _lines("```json\n" + json.dumps(GRID) + "\n```", request) == [IDENTICAL]


def test_number_tables_are_rectangular_runs_of_integer_lines():
    text = (
        "Report for 2026\n"  # not a row: words
        "3 4\n"
        "5 6\n"
        "7 8 9\n"  # another length: the run ends
        "1 2 3\n"
        "\n"
        "42\n"  # one line of digits alone is no table
        "x\n"
        "-1, 0\n"
        "2, -3,\n"
    )
    assert rr.number_tables(text) == [
        [[3, 4], [5, 6]],
        [[7, 8, 9], [1, 2, 3]],
        [[-1, 0], [2, -3]],
    ]
    assert rr.number_tables("1.5 2\n3 4") == []


def test_an_answer_identical_to_a_json_table_in_the_request_is_named():
    request = "Input: " + json.dumps({"input": GRID})
    assert _lines(COPIED, request) == [IDENTICAL]
    # At any depth of the request's JSON.
    nested = json.dumps({"cases": [{"in": GRID, "out": [[0]]}]})
    assert _lines(COPIED, nested) == [IDENTICAL]
    assert _lines(OTHER, request) == []
    # Only a list of lists is compared: an option or a coordinate the request
    # offers is a choice, not a copied input.
    menu = 'Reply with one of {"action": "left"} or {"action": "move", "to": [1, 2]}.'
    assert _lines('{"action": "left"}', menu) == []
    assert _lines('{"action": "move", "to": [1, 2]}', menu) == []
    # Plain numbers in prose are not a table.
    assert _lines("The answer is 7.", "Compute 3 + 4 = 7.") == []


# ── C4_last: an error the reply does not mention ─────────────────────────


def test_the_last_cell_raising_is_named_unless_the_reply_mentions_an_error():
    cells = [_cell("print(1)", "1"), _cell("x = d['k']", rr.clean_output([TRACEBACK]))]
    line = (
        "Code cell 2 since the request raised `ValueError: boom`; the reply "
        "does not mention an error."
    )
    assert _lines("The answer is **7**.", cells=cells) == [line]
    for mention in (
        "I could not read the file, so the answer is **7**.",
        "The lookup failed; the answer is **7**.",
        "Unable to finish: **7** is a guess.",
        "There was an error in the last step; the answer is **7**.",
    ):
        assert _lines(mention, cells=cells) == [], mention


def test_an_error_caught_and_printed_is_named_wherever_it_was():
    cells = [
        _cell(
            "try:\n    f()\nexcept Exception as e:\n    print(f'Error: {e}')",
            "Error: no such file",
        ),
        _cell("print(7)", "7"),
    ]
    assert _lines("The answer is **7**.", cells=cells) == [
        "Code cell 1 since the request caught and printed `Error: no such file`; "
        "the reply does not mention an error.",
    ]


def test_an_error_fixed_by_a_later_cell_is_silent_and_no_op_cells_do_not_count():
    raised = _cell("x = d['k']", rr.clean_output([TRACEBACK]))
    assert _lines("The answer is **7**.", cells=[raised, _cell("print(7)", "7")]) == []
    # A cell that does nothing is not the last computing cell.
    assert _lines("The answer is **7**.", cells=[raised, _cell("pass", "")]) != []
    # execute_function results count as cells.
    call = _cell('lookup(**{"k": 1})', "KeyError: 'k'", tool="execute_function")
    assert _lines("The answer is **7**.", cells=[call]) == [
        "Code cell 1 since the request raised `KeyError: 'k'`; the reply does "
        "not mention an error.",
    ]


def test_quoted_code_and_errors_never_name_examples():
    error = json.dumps(
        {"error": "Traceback (most recent call last):\nKeyError: 'demo_pairs_example'"},
    )
    cells = [_cell("x = task['demo_pairs_example']", rr.clean_output([error]))]
    (line,) = _lines("The answer is **7**.", cells=cells)
    assert line == (
        "Code cell 1 since the request raised `KeyError: '…'`; the reply does "
        "not mention an error."
    )


def test_both_checks_render_in_order_and_hold_only_facts():
    cells = [_cell("x = d['k']", rr.clean_output([TRACEBACK]))]
    lines = _lines("The total is **0**.", OFFICE_REQUEST, cells)
    assert lines == [
        "The answer is zero (0).",
        "Code cell 1 since the request raised `ValueError: boom`; the reply "
        "does not mention an error.",
    ]
    text = "\n".join(lines).lower()
    for word in ("please", "should", "revise", "confirm", "check", "example", "must"):
        assert word not in text
    assert len(lines) <= 3


# ── the cells since the request, from the transcript ─────────────────────


def _call(call_id: str, name: str, args: dict) -> dict:
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": name, "arguments": json.dumps(args)},
    }


def test_the_cells_are_read_back_to_the_requesters_latest_message():
    header = json.dumps({"state_mode": "stateful", "session_id": 0, "duration_ms": 1})
    messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "first request"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [_call("a", "execute_code", {"code": "print(1)"})],
        },
        {"role": "tool", "tool_call_id": "a", "content": "1"},
        {"role": "assistant", "content": "1"},
        {"role": "user", "content": "second request", "_interjection": True},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                _call("b", "execute_code", {"code": "print(2)"}),
                _call("c", "search_functions", {"query": "x"}),
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "b",
            "content": [
                {"type": "text", "text": header},
                {"type": "text", "text": "\n--- stdout ---\n"},
                {"type": "text", "text": "2\n"},
            ],
        },
        {"role": "tool", "tool_call_id": "c", "content": "[]"},
        {"role": "user", "content": "[call b] finished.", "_loop_authored": True},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                _call(
                    "d",
                    "execute_function",
                    {"function_name": "f", "call_kwargs": {"x": 1}},
                ),
            ],
        },
        {"role": "tool", "tool_call_id": "d", "content": TRACEBACK},
        {"role": "assistant", "content": "The answer is 2."},
    ]
    cells = rr.cells_since_request(messages)
    assert [(c["tool"], c["code"], c["output"]) for c in cells] == [
        ("execute_code", "print(2)", "2\n"),
        ("execute_function", 'f(**{"x": 1})', rr.clean_output([TRACEBACK])),
    ]
    assert rr.request_text(messages) == "second request"


# ── parity with the offline checks ───────────────────────────────────────

# The test session may move HOME, so the account's own home is looked up.
OFFLINE = Path(pwd.getpwuid(os.getuid()).pw_dir) / (
    "continual-harness-research/artifacts/research-regression-diagnosis-20261001/"
    "overhaul-lanes/runtime-20261006/reply-receipt-offline-v1/checks.py"
)
OFFLINE_SHA256 = "f7823afce0dd1ac40b9925051f85578501f38cc45696ef98204151f8c0efc7fd"  # pragma: allowlist secret

# (reply, request, cells) -> the offline render(use=("C2", "C4_last")), run on
# 7 Oct 2026 with the file whose hash is above.
PARITY = {
    "office zero answer": (
        (
            "The total cost of the meals is **0.00**.",
            OFFICE_REQUEST,
            [("df.amount.sum()", "0.00")],
        ),
        ["The answer is zero (0.00)."],
    ),
    "office empty answer": (
        ("{}", "List the overdue invoices as JSON.", []),
        ["The answer is an empty value ({})."],
    ),
    "every item the same": (
        ("The ids are `[3, 3, 3]`.", "ids?", []),
        ["Every item of the answer is the same ([3, 3, 3])."],
    ),
    "a correct answer": (
        (
            "The total cost of the meals is **42.50**.",
            OFFICE_REQUEST,
            [("print(x)", "42.5")],
        ),
        [],
    ),
    "the last cell raised": (
        ("The answer is **7**.", "q", [("print(1)", "1"), ("x = d['k']", "TRACEBACK")]),
        [
            "Code cell 2 since the request raised `ValueError: boom`; the reply does not mention an error.",
        ],
    ),
    "an error the reply mentions": (
        (
            "I could not read the file, so the answer is **7**.",
            "q",
            [("x = d['k']", "TRACEBACK")],
        ),
        [],
    ),
    "an error fixed later": (
        ("The answer is **7**.", "q", [("x = d['k']", "TRACEBACK"), ("print(7)", "7")]),
        [],
    ),
}


def _parity_cells(spec: list) -> list[dict]:
    return [
        _cell(code, rr.clean_output([TRACEBACK]) if out == "TRACEBACK" else out)
        for code, out in spec
    ]


@pytest.mark.parametrize("case", sorted(PARITY))
def test_the_rules_give_the_offline_receipt(case):
    (reply, request, cells), expected = PARITY[case]
    assert _lines(reply, request, _parity_cells(cells)) == expected


# A request whose input is rows of digits, as the offline gate's largest
# catch read it ("Test input (HxW):" then the rows; 244 failed, 4 correct).
DIGIT_REQUEST = "Solve it.\nTest input (3x3):\n1 2 0\n0 1 2\n2 0 1\n"


def test_an_answer_equal_to_the_requests_input_fires_where_the_offline_rule_did():
    """Offline the rule read the input grid after "Test input (HxW):" and fired
    as "the grid is identical to the input grid"; here the same answer fires
    as identical to a table in the request, with no format named."""
    reply = json.dumps({"action": "submit", "grid": GRID})
    assert rr.checks(reply, DIGIT_REQUEST, [])["C2"] == {
        "why": "the answer is identical to a value in the request",
    }
    other = json.dumps({"action": "submit", "grid": [[0]]})
    assert rr.checks(other, DIGIT_REQUEST, [])["C2"] is None


@pytest.mark.skipif(
    not OFFLINE.exists(),
    reason="the offline checks are not on this machine",
)
def test_the_offline_file_still_gives_the_pinned_receipts():
    assert hashlib.sha256(OFFLINE.read_bytes()).hexdigest() == OFFLINE_SHA256
    spec = importlib.util.spec_from_file_location("offline_receipt_checks", OFFLINE)
    offline = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(offline)
    for case, ((reply, request, cells), expected) in PARITY.items():
        res = offline.receipt_checks(reply, request, _parity_cells(cells))
        assert offline.render(res, use=("C2", "C4_last")) == expected, case
    # The offline input rule, given the input as its extractor read it.
    sys.path.insert(0, str(OFFLINE.parent))
    try:
        arc = importlib.import_module("rr_arc")
    finally:
        sys.path.pop(0)
    test_input = arc.test_input(DIGIT_REQUEST)
    assert test_input == GRID
    for grid, fires in ((GRID, True), ([[0]], False)):
        res = offline.receipt_checks(
            json.dumps({"action": "submit", "grid": grid}),
            DIGIT_REQUEST,
            [],
            grid=grid,
            test_input=test_input,
        )
        assert (res["C2"] is not None) is fires
        ours = rr.checks(
            json.dumps({"action": "submit", "grid": grid}),
            DIGIT_REQUEST,
            [],
        )
        assert (ours["C2"] is not None) is fires


# ── run stats and the reply comparison ──────────────────────────────────


def test_run_stats_and_what_counts_as_a_revision():
    from types import SimpleNamespace

    state = SimpleNamespace(receipts_shown=2, receipts_revised=1)
    assert rr.run_stats(state) == {"receipts_shown": 2, "receipts_revised": 1}
    assert rr.run_stats(None) == {"receipts_shown": 0, "receipts_revised": 0}
    assert not rr.revised("The total is **0.00**.", "  The total is\n**0.00**. ")
    assert rr.revised("The total is **0.00**.", "The total is **31.20**.")
