"""A captured procedure that ignores its input is labelled a record, not a procedure.

The label is read from the code's syntax alone (``capture_shape.record_kind``: no read of an input parameter, or every
return a literal), never from a word, a value or a task id, and nothing is run. It is stored when the capture lands in
the store, and a capture stored before the field existed is labelled from its code when it is exported. A record is
served only as what was done last time: its card, its catalogue card, its dispatch row, its search row and the
request-start block say so, the librarian's brief says it is never a fit for a different request, and the writer's
``land_capture`` reply labels it (it is never refused for it). Real capture store, export, package, librarian start
block and landing reply; only the episode is a fixture and no model is called.
"""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import pytest

from tests.memory.test_capture_examples import FIXED, RETIRED, TABLE, _cell
from tests.memory.test_capture_examples import _copy as _examples_copy
from tests.memory.test_capture_examples import _snapshot
from tests.memory.test_capture_trace import _ep, _staging, _write_action
from unify.memory import capture_shape as cs
from unify.memory import captures as cp
from unify.memory import librarian as L
from unify.memory import library_export as lx
from unify.memory import library_helper as lh
from unify.memory import library_search
from unify.memory.layout import discover

READS_STATE = '''"""Book what the latest message asks for."""


def run(state):
    """The latest message's date."""
    return {"date": state.data[-1]["date"]}
'''
FIXED_STATE = '''"""Hand back last time's answer."""


def run(state):
    """The answer of one attempt."""
    return [[1, 4], [2, 5]]
'''
LITERAL_RETURNS = '''"""Look at the request, then hand back a fixed value."""


def run(state):
    if state.last:
        return "yes"
    return "no"
'''
ENV_FIXED = '''"""Write the summary file."""


def run(state, env):
    """Write the report."""
    env.files.write("out/report.txt", "total 6")
    return None
'''
ENV_READS = '''"""Write the file the request names."""


def run(state, env):
    """Write the report where the request says."""
    env.files.write(state.data[0]["path"], "total 6")
    return "done"
'''
READER_IGNORES_REQUEST = '''"""A fixed value through a reader that reads nothing."""


def read_input(request):
    return [1, 2, 3]


def run(value):
    return sum(value)
'''
HELPER_READS = '''"""Hand the state to a helper."""


def _date(s):
    return s.data[-1]["date"]


def run(state):
    return _date(state)
'''


@pytest.mark.parametrize(
    "source, kind, why",
    [
        (TABLE, cs.PROCEDURE, ""),
        (READS_STATE, cs.PROCEDURE, ""),
        (HELPER_READS, cs.PROCEDURE, ""),
        (
            ENV_READS,
            cs.PROCEDURE,
            "",
        ),  # acts on what it was handed; its return is not its work
        (FIXED, cs.RECORD, cs.NEVER_READS),
        (FIXED_STATE, cs.RECORD, cs.NEVER_READS),
        (
            ENV_FIXED,
            cs.RECORD,
            cs.NEVER_READS,
        ),  # env is how it acts, never what it is handed
        (READER_IGNORES_REQUEST, cs.RECORD, cs.NEVER_READS),
        (LITERAL_RETURNS, cs.RECORD, cs.FIXED_RESULT),
        (RETIRED, cs.RECORD, cs.FIXED_RESULT),
    ],
)
def test_record_or_procedure_is_read_from_the_syntax_alone(source, kind, why):
    assert cs.record_kind(source) == {"kind": kind, "why": why}


def test_code_that_does_not_parse_or_has_no_run_is_not_labelled():
    assert cs.record_kind("def run(:\n") is None
    assert cs.record_kind("def main(x):\n    return x\n") is None
    assert cs.record_kind(None) is None


def test_the_label_does_not_depend_on_names_words_or_values():
    """The same code with other names, prose and constants gets the same label: nothing but the syntax is read."""
    renamed = FIXED_STATE.replace("last time's answer", "a total").replace(
        "[[1, 4], [2, 5]]",
        "'yes'",
    )
    assert (
        cs.record_kind(renamed)
        == cs.record_kind(FIXED_STATE)
        == {"kind": cs.RECORD, "why": cs.NEVER_READS}
    )
    other = READS_STATE.replace("state", "s").replace('"date"', '"anything"')
    assert cs.record_kind(other) == {"kind": cs.PROCEDURE, "why": ""}


@pytest.mark.parametrize(
    "source, kind",
    [(ENV_FIXED, cs.RECORD), (ENV_READS, cs.PROCEDURE)],
)
def test_the_capture_step_stores_the_label_when_the_capture_lands(
    tmp_path,
    source,
    kind,
):
    staging = _staging(
        tmp_path,
        "e1",
        {"procedure.py": source, "summary.md": "## Asked\nWrite the report.\n"},
    )
    store = cp.store_dir(tmp_path)
    row = cp.capture_trace_step(staging, _ep(actions=[_write_action()]), store)
    assert row["safety"] == "ok"
    rec = json.loads((store / row["capture_id"] / cp.RECORD_FILE).read_text())
    assert rec["kind"]["kind"] == kind
    exported = json.loads(
        cp.export_files(store)[f"{cp.EXPORT_DIR}/{row['capture_id']}/{cp.RECORD_FILE}"],
    )
    assert exported["kind"] == rec["kind"]
    dest = tmp_path / "inputs"
    assert cp.export_for_writer(store, [row["capture_id"]], dest) == [row["capture_id"]]
    assert (
        json.loads((dest / row["capture_id"] / cp.RECORD_FILE).read_text())["kind"]
        == rec["kind"]
    )


def test_a_capture_stored_before_the_label_is_labelled_from_its_code_when_exported(
    tmp_path,
):
    _, snap, ids = _snapshot(tmp_path)  # records written without a kind
    kinds = {
        k: json.loads(snap[f"{cp.EXPORT_DIR}/{cid}/{cp.RECORD_FILE}"]).get("kind")
        for k, cid in ids.items()
    }
    assert kinds == {
        "table": {"kind": cs.PROCEDURE, "why": ""},
        "grid": {"kind": cs.PROCEDURE, "why": ""},
        "fixed": {"kind": cs.RECORD, "why": cs.NEVER_READS},
        "retired": {"kind": cs.RECORD, "why": cs.FIXED_RESULT},
    }


def test_a_records_card_says_what_was_done_last_time_and_a_procedures_does_not(
    tmp_path,
):
    root, ids = _examples_copy(tmp_path)
    fixed, table = f"capture:{ids['fixed']}", f"capture:{ids['table']}"
    got = _cell(
        root,
        f"print(json.dumps({{'fixed': str(memory.get({fixed!r})), 'table': str(memory.get({table!r})),"
        f" 'call': memory.get({fixed!r})('x')}}))\n",
    )
    head = got["fixed"].split("\n\n", 1)[0]
    assert head.startswith(f"{fixed}  [record from 1 episode"), head
    assert lh.RECORD_CARD.format(why=cs.NEVER_READS) in head and lh.RECORD_CALL in head
    assert "re-runs" not in head and "A recorded procedure" not in head
    table_head = got["table"].split("\n\n", 1)[0]
    assert (
        table_head.startswith(f"{table}  [recorded procedure from 1 episode")
        and "A recorded procedure" in table_head
    )
    assert "A record of what was done last time" not in table_head
    # labelled, never refused: a call still gives what it gave last time
    assert got["call"] == [[1, 4], [2, 5], [3, 6]]


def test_the_catalogue_dispatch_and_search_rows_label_a_record(tmp_path):
    tree, snap, ids = _snapshot(tmp_path)
    text = lx.catalogue_text(tree, captures=snap)
    fixed = text.split(f"### capture:{ids['fixed']}\n", 1)[1].split("###", 1)[0]
    table = text.split(f"### capture:{ids['table']}\n", 1)[1].split("###", 1)[0]
    assert (
        lh.RECORD_CARD.format(why=cs.NEVER_READS) in fixed
        and "Evidence: record from 1 episode" in fixed
    )
    assert (
        "A record of what was done last time" not in table
        and "Evidence: recorded procedure" in table
    )
    rows = {
        r["id"]: r
        for r in lx.dispatch_data(discover(tree), captures=snap)["procedures"]
    }
    assert rows[f"capture:{ids['fixed']}"]["record"] == cs.NEVER_READS
    assert rows[f"capture:{ids['retired']}"]["record"] == cs.FIXED_RESULT
    assert rows[f"capture:{ids['table']}"]["record"] is None
    searched = {r["id"]: r["card"] for r in library_search.rows(tree, captures=snap)}
    assert (
        f"a record of what was done last time, not a procedure: {cs.NEVER_READS}"
        in searched[f"capture:{ids['fixed']}"]
    )
    assert (
        "a record of what was done last time" not in searched[f"capture:{ids['table']}"]
    )


def test_the_librarian_is_told_a_record_is_never_a_fit_for_a_different_request():
    for brief in (L.BRIEF, L.BRIEF_EVIDENCE):
        assert (
            "never as a fit for a different request" in brief
            and "A record (its card" in brief
        )


def test_the_request_start_block_names_a_record_as_what_was_done_last_time(
    tmp_path,
    monkeypatch,
):
    from unify.actor import tool_handlers as H
    from unify.memory.integration import request as R
    from unify.memory.writer_tools import Reply

    root, ids = _examples_copy(tmp_path)
    fixed, table = f"capture:{ids['fixed']}", f"capture:{ids['table']}"
    assert L.record_captures(root) == {
        fixed: cs.NEVER_READS,
        f"capture:{ids['retired']}": cs.FIXED_RESULT,
    }
    answer = {
        "fits": [
            {
                "id": fixed,
                "call": f"memory.get({fixed!r})()",
                "why": "the same request as before",
            },
            {
                "id": table,
                "call": f"memory.get({table!r})(memory.request())",
                "why": "it sums a table",
            },
        ],
    }

    async def ask_library(ctx, args):
        return Reply(json.dumps(answer))

    async def opener():
        raise AssertionError("the stub asks without a session")

    monkeypatch.setattr(H, "ask_library", ask_library)
    monkeypatch.setattr(R, "current", lambda: SimpleNamespace(copy_root=root))
    state = SimpleNamespace(
        request_text="Sum the table.",
        open_librarian=opener,
        librarian=None,
    )
    block = asyncio.run(L.request_start_block(state))
    lines = block.splitlines()
    assert lines[1].startswith(
        f"- {L.RECORD_LABEL.format(why=cs.NEVER_READS)}: {fixed}:",
    ), lines[1]
    assert L.BEST_FIT_LABEL not in lines[1]
    assert lines[3].startswith(f"- {L.FALLBACK_LABEL}: {table}:"), lines[3]
    # without the copy's records the block is as before
    assert (
        L.start_block(answer)
        .splitlines()[1]
        .startswith(f"- {L.BEST_FIT_LABEL}: {fixed}:")
    )


def test_the_landing_reply_labels_a_record_and_never_refuses_it():
    from unify.memory import land_capture as lc

    assert (
        lc._record_why({}, FIXED) == cs.NEVER_READS
    )  # a record stored before the label: read from its code
    assert (
        lc._record_why({"kind": {"kind": cs.PROCEDURE, "why": ""}}, FIXED) == ""
    )  # the stored label wins
    assert lc._record_why({}, TABLE) == ""
    line = lc.REPLY_RECORD.format(cid="a" * 16, why=cs.NEVER_READS)
    assert "is a record, not a procedure" in line and "what was done last time" in line
