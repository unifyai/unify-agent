"""Library mode: the request's use record lists the library's failures (``library_failures``) beside its counts.

Each failed library call the worker's audit hook saw (an item's or one of the package's helpers', refused or raised)
and each cell whose error was a missing ``memory`` name or a failed import of the package is one entry: its kind, its
target (an item id, or a helper call's argument shape: types and sizes only), the exception's type, the first line of
its message through the harness's redactors, and the cell's index in the transcript, where its full traceback is.
The list is capped (``MAX_LIBRARY_FAILURES``) and what is left out is counted. No worker, no model: the audit hook
is driven directly and the episode is built by hand.
"""

from __future__ import annotations

import json

from tests.memory.test_episodes import _ep
from unify.memory.integration import request as request_mod
from unify.memory.integration.adapters import audit as audit_mod
from unify.memory.integration.adapters.audit import CellAudit
from unify.memory.integration.request import (
    RequestRun,
    add_library_failures,
    library_failure_list,
    library_failures_empty,
    memory_use,
    settle_library_failures,
)
from unify.memory.redact import Redactor

PARSE_DATE = "memory.text.dates:parse_date"
# planted, readable test strings: an environment credential's value and a key-shaped string
PLANTED_SECRET = "planted-credential-value-for-the-failure-test"  # pragma: allowlist secret
KEY_SHAPED = "sk-" + "plantedkeyshapedvalueforthetest0123456789"  # pragma: allowlist secret


def _cell_lines(*calls: str) -> list[dict]:
    """A transcript with one ``execute_code`` cell per tool call id, each with its result."""
    lines: list[dict] = [{"type": "message", "message": {"role": "user", "content": "go"}}]
    for call in calls:
        lines.append(
            {
                "type": "message",
                "message": {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "id": call,
                            "type": "function",
                            "function": {"name": "execute_code", "arguments": json.dumps({"code": "pass"})},
                        },
                    ],
                },
            },
        )
        lines.append({"type": "message", "message": {"role": "tool", "tool_call_id": call, "content": "ok"}})
    return lines


# --- the audit hook ------------------------------------------------------------------------------------------


def test_the_hook_keeps_each_failure_first_line_masked_and_bounded():
    audit = CellAudit([])
    audit.begin()
    audit("memory.call", (PARSE_DATE, "ab12", "error"))
    audit("memory.call_failed", (PARSE_DATE, "AttributeError", f"no attribute\nsecond {KEY_SHAPED}", ""))
    audit("memory.call_failed", ("memory:get", "TypeError", f"bad {KEY_SHAPED}", "(dict[3])"))
    audit("memory.call_failed", ("memory:match", "LookupError", "x" * 1000, "(str[4])"))
    # malformed: an id that is neither an item's nor a helper's (a removed helper's among them), a non-identifier
    # type, a wrong arity
    audit("memory.call_failed", ("memory:nothing", "TypeError", "m", ""))
    audit("memory.call_failed", ("memory:find", "TypeError", "m", "(dict[3])"))
    audit("memory.call_failed", (PARSE_DATE, "not a type", "m", ""))
    audit("memory.call_failed", (PARSE_DATE, "TypeError", "m"))
    calls = audit.drain()["calls"]
    failures = calls["failures"]
    assert [(f["item"], f["exception"]) for f in failures] == [
        (PARSE_DATE, "AttributeError"),
        ("memory:get", "TypeError"),
        ("memory:match", "LookupError"),
    ]
    assert failures[0]["message"] == "no attribute" and failures[0]["shape"] == ""
    assert failures[1]["message"] == "bad <redacted:key-shaped>" and failures[1]["shape"] == "(dict[3])"
    assert len(failures[2]["message"]) <= audit_mod.MAX_FAILURE_TEXT + len(audit_mod.CLIP_MARK)
    assert failures[2]["message"].endswith(audit_mod.CLIP_MARK)
    assert KEY_SHAPED not in json.dumps(calls)
    assert calls["failures_dropped"] == 0 and calls["malformed"] == 4


def test_the_hook_lists_at_most_its_cap_per_cell_and_counts_the_rest(monkeypatch):
    monkeypatch.setattr(audit_mod, "MAX_CALL_FAILURES", 2)
    audit = CellAudit([])
    audit.begin()
    for _ in range(5):
        audit("memory.call_failed", (PARSE_DATE, "ValueError", "bad", ""))
    calls = audit.drain()["calls"]
    assert len(calls["failures"]) == 2 and calls["failures_dropped"] == 3
    # a cell without failures has no failure fields
    audit.begin()
    audit("memory.call", (PARSE_DATE, "ab12", "ok"))
    assert "failures" not in audit.drain()["calls"]


# --- the request's list --------------------------------------------------------------------------------------


def _failing_cell() -> dict:
    audit = CellAudit([])
    audit.begin()
    audit("memory.call", (PARSE_DATE, "ab12", "refused"))
    audit("memory.call_failed", (PARSE_DATE, "MemoryInputError", f"not a date: {PLANTED_SECRET}", ""))
    audit("memory.helper_call", ("memory:get", "ab12", "error"))
    audit("memory.call_failed", ("memory:get", "LookupError", "memory: no item 'x'", "(str[1])"))
    return audit.drain()


def test_the_list_is_capped_and_counts_what_it_left_out(monkeypatch):
    monkeypatch.setattr(request_mod, "MAX_LIBRARY_FAILURES", 3)
    acc = library_failures_empty()
    for n in range(3):
        add_library_failures(acc, _failing_cell())
        settle_library_failures(acc, f"call-{n}")
    listed, dropped = library_failure_list(acc, [])
    assert len(listed) == 3 and dropped == 3
    assert all(e["cell"] is None for e in listed)  # cells not in the transcript are unknown


def test_a_failure_whose_cell_never_reported_is_listed_without_a_cell():
    acc = library_failures_empty()
    add_library_failures(acc, _failing_cell())
    listed, _ = library_failure_list(acc, [{"index": 0, "call": "call-a"}])
    assert [e["cell"] for e in listed] == [None, None]


def test_the_run_settles_its_cells_and_the_use_record_lists_them():
    run = RequestRun("r", None)
    run.note_calls(_failing_cell())
    run.note_result("call-b", {"error": None, "session_id": 0})
    run.note_failure("call-c", RuntimeError("the tool raised"))
    ep = _ep(transcript=_cell_lines("call-a", "call-b", "call-c"), actions=[])
    rec = memory_use(
        ep,
        [PARSE_DATE],
        runtime_calls=run.runtime_calls,
        cell_errors=dict(run.cell_errors),
        failures=run.failures,
        redactor=Redactor({"SERVICE_TOKEN": PLANTED_SECRET}),
    )
    assert "error" not in rec, rec
    assert [(e["kind"], e["cell"]) for e in rec["library_failures"]] == [("item", 1), ("get", 1)]
    assert rec["library_failures_dropped"] == 0
    # beside the existing counts, which are unchanged
    assert rec["library_calls_failed"] == 2
    text = json.dumps(rec)
    assert PLANTED_SECRET not in text and "Traceback" not in text
